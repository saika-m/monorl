from __future__ import annotations

import io
import json

import httpx
import pytest

from monorl.core.types import Sample, StepResult
from monorl.envs.protocol import PROTOCOL_VERSION, ProtocolError, encode_sample
from monorl.envs.remote import HttpTransport
from monorl.envs.server import (
    MEPServer,
    allowed_host_names,
    create_app,
    serve_stdio,
    start_http_server,
)
from monorl.envs.tool_calling import make_tool_calling_env


def _factory():
    return make_tool_calling_env({"tools": ["calculator"]}, max_turns=3)


def _sample(i: str = "s") -> dict:
    return encode_sample(Sample(id=i, messages=[{"role": "user", "content": "2+2?"}], reference="4"))


def test_handle_spec_health_and_episode_lifecycle():
    srv = MEPServer(_factory, name="tools", parallel_episodes=2)
    spec = srv.handle({"op": "spec"})
    assert spec["protocol"] == PROTOCOL_VERSION and spec["name"] == "tools"
    assert spec["max_turns"] == 3 and spec["parallel_episodes"] == 2
    assert spec["tools"][0]["function"]["name"] == "calculator"
    assert srv.handle({"op": "health"}) == {"ok": True}

    r = srv.handle({"op": "reset", "episode_id": "e1", "sample": _sample()})
    assert r["messages"][0]["role"] == "system" and r["info"]["episode_id"] == "e1"
    assert srv.active_episodes == 1

    s = srv.handle(
        {
            "op": "step",
            "episode_id": "e1",
            "action": {
                "text": '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>',
                "finish_reason": "stop",
            },
        }
    )
    assert s["done"] is False and s["messages"][0]["content"] == "4" and s["reward"] == 0.0
    s = srv.handle({"op": "step", "episode_id": "e1", "action": {"text": "4"}})
    assert s["done"] is True and s["messages"] == []
    assert srv.active_episodes == 0  # done episodes are closed server-side

    assert (
        srv.handle({"op": "step", "episode_id": "e1", "action": {"text": "x"}})["error"]["type"]
        == "unknown_episode"
    )
    srv.handle({"op": "reset", "episode_id": "e2", "sample": _sample()})
    assert srv.handle({"op": "close", "episode_id": "e2"}) == {}
    assert srv.active_episodes == 0
    assert srv.handle({"op": "close", "episode_id": "never"}) == {}


def test_handle_errors_and_capacity():
    srv = MEPServer(_factory, parallel_episodes=1)
    assert srv.handle({"op": "bogus"})["error"]["type"] == "protocol"
    assert srv.handle({"op": "reset"})["error"]["type"] == "protocol"
    assert srv.handle({"op": "reset", "episode_id": "a", "sample": {}})["error"]["type"] == "protocol"
    assert srv.handle({"op": "step", "episode_id": "a", "action": {}})["error"]["type"] == "protocol"
    srv.handle({"op": "reset", "episode_id": "a", "sample": _sample()})
    assert srv.handle({"op": "reset", "episode_id": "b", "sample": _sample()})["error"]["type"] == "capacity"
    # re-reset of an active episode replaces it without consuming capacity
    assert "messages" in srv.handle({"op": "reset", "episode_id": "a", "sample": _sample()})
    srv.close_all()
    assert srv.active_episodes == 0


def test_env_exception_becomes_error_envelope():
    class Boom:
        max_turns = 1

        def reset(self, sample):
            raise RuntimeError("kaboom")

        def step(self, action):
            raise AssertionError

        def close(self):
            pass

    srv = MEPServer(Boom)
    err = srv.handle({"op": "reset", "episode_id": "x", "sample": _sample()})["error"]
    assert err == {"type": "RuntimeError", "message": "kaboom"}


def test_serve_stdio_roundtrip():
    srv = MEPServer(_factory, parallel_episodes=4)
    requests = [
        {"op": "spec"},
        "not json",
        "",
        {"op": "reset", "episode_id": "e", "sample": _sample()},
        {"op": "step", "episode_id": "e", "action": {"text": "done"}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(r) if isinstance(r, dict) else r for r in requests) + "\n")
    stdout = io.StringIO()
    serve_stdio(srv, stdin=stdin, stdout=stdout)
    lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert len(lines) == 4  # blank line ignored
    assert lines[0]["protocol"] == PROTOCOL_VERSION
    assert lines[1]["error"]["type"] == "protocol"
    assert lines[2]["messages"][-1]["content"] == "2+2?"
    assert lines[3]["done"] is True


def test_http_app_routes_and_status_codes():
    from fastapi.testclient import TestClient

    app = create_app(MEPServer(_factory, parallel_episodes=1))
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert c.get("/spec").json()["protocol"] == PROTOCOL_VERSION
        assert c.get("/health").json() == {"ok": True}
        r = c.post("/reset", json={"episode_id": "e", "sample": _sample()})
        assert r.status_code == 200 and r.json()["messages"]
        assert c.post("/reset", json={"episode_id": "f", "sample": _sample()}).status_code == 503
        assert c.post("/step", json={"episode_id": "nope", "action": {"text": "x"}}).status_code == 400
        assert c.post("/bogus", json={}).status_code == 400
        assert (
            c.post("/step", content=b"{bad", headers={"content-type": "application/json"}).status_code == 400
        )
        assert c.post("/close", json={"episode_id": "e"}).json() == {}


def test_start_http_server_thread():
    handle = start_http_server(MEPServer(_factory), port=0)
    try:
        assert handle.port != 0
        assert httpx.get(f"{handle.url}/health").json() == {"ok": True}
    finally:
        handle.stop()


class _Recorder:
    """An environment that notes every call it gets."""

    max_turns = 2
    calls: list[str] = []

    def reset(self, sample):
        self.calls.append(f"reset {sample.id}")
        return [{"role": "user", "content": "go"}]

    def step(self, action):
        self.calls.append(f"step {action.text}")
        return StepResult([], reward=1.0, done=True)

    def close(self):
        pass


@pytest.fixture
def recorder():
    _Recorder.calls = []
    handle = start_http_server(MEPServer(_Recorder), port=0)
    try:
        yield handle
    finally:
        handle.stop()


RESET = json.dumps({"episode_id": "x", "sample": {"id": "1", "messages": []}})
STEP = json.dumps({"episode_id": "x", "action": {"text": "run this"}})


@pytest.mark.parametrize(
    ("headers", "status", "error"),
    [
        # what a page of another site can send to a server on localhost without a preflight
        ({"content-type": "text/plain", "origin": "http://evil.example"}, 403, "forbidden"),
        ({"content-type": "text/plain"}, 415, "unsupported_media_type"),
        ({"content-type": "application/x-www-form-urlencoded"}, 415, "unsupported_media_type"),
        ({}, 415, "unsupported_media_type"),
        # a script of another site whose preflight a proxy answered, or a same-site page
        ({"content-type": "application/json", "origin": "http://evil.example"}, 403, "forbidden"),
        ({"content-type": "application/json", "origin": "null"}, 403, "forbidden"),
        # a page whose own domain was pointed at 127.0.0.1 after it loaded
        ({"content-type": "application/json", "host": "evil.example"}, 403, "forbidden"),
    ],
)
def test_http_requests_of_a_web_page_never_reach_the_environment(recorder, headers, status, error):
    with httpx.Client(base_url=recorder.url) as client:
        for path, body in (("/reset", RESET), ("/step", STEP), ("/close", RESET)):
            response = client.post(path, content=body, headers=headers)
            assert response.status_code == status, path
            assert response.json()["error"]["type"] == error
    assert _Recorder.calls == [] and recorder.mep.active_episodes == 0

    # the same bodies from an environment client
    with httpx.Client(
        base_url=recorder.url, headers={"content-type": "application/json; charset=utf-8"}
    ) as c:
        assert c.post("/reset", content=RESET).status_code == 200
        assert c.post("/step", content=STEP).json()["reward"] == 1.0
    assert _Recorder.calls == ["reset 1", "step run this"]


def test_http_get_is_refused_for_a_foreign_host_or_an_origin(recorder):
    for path in ("/spec", "/health"):
        assert httpx.get(recorder.url + path).status_code == 200
        assert httpx.get(recorder.url + path, headers={"host": "evil.example"}).status_code == 403
        assert httpx.get(recorder.url + path, headers={"origin": "http://evil.example"}).status_code == 403


@pytest.mark.parametrize(
    ("allowed", "host", "status"),
    [
        ((), "localhost", 200),
        ((), "LOCALHOST:8080", 200),
        # an address is never the name of a page's site
        ((), "127.0.0.1:18080", 200),
        ((), "10.1.2.3:8080", 200),
        ((), "[::1]:8080", 200),
        ((), "[fd00::7]", 200),
        # names, and what only looks like an address
        ((), "env-1:8080", 403),
        ((), "localhost.evil.example", 403),
        ((), "127.0.0.1.evil.example:8080", 403),
        ((), "2130706433", 403),
        ((), "0x7f.0.0.1", 403),
        ((), "127.1", 403),
        ((), "localhost:8080@evil.example", 403),
        ((), "[::1].evil.example", 403),
        ((), "[evil.example]", 403),
        ((), "localhost.", 403),
        ((), "", 403),
        (("env-1", "env-2.internal:8080"), "env-1:8080", 200),
        (("env-1", "env-2.internal:8080"), "ENV-2.internal", 200),
        (("env-1", "env-2.internal:8080"), "env-3", 403),
        (("env_1",), "env_1:8080", 200),
        (("*",), "anything.example", 200),
    ],
)
def test_http_answers_only_to_its_own_host_names(allowed, host, status):
    from fastapi.testclient import TestClient

    app = create_app(MEPServer(_factory), allowed_hosts=allowed)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        assert c.get("/health", headers={"host": host}).status_code == status
        # no host name makes a page's request acceptable
        assert c.get("/health", headers={"host": host, "origin": "http://evil.example"}).status_code == 403


def test_an_allowed_host_that_is_not_a_host_name_is_an_error():
    assert allowed_host_names(["Env-1:8080", "*"]) == {"localhost", "env-1", "*"}
    for bad in ("http://env-1:8080", "env-1:port", "", "env 1", "env-1/path"):
        with pytest.raises(ValueError, match="not a host name"):
            allowed_host_names([bad])


def test_a_client_refused_for_its_host_name_is_told_how_to_allow_it(recorder):
    """A run that reaches the server under a name the server was not started with."""
    named = HttpTransport(recorder.url, headers={"host": "env-1.internal:8080"})
    try:
        with pytest.raises(ProtocolError, match="--allowed-host env-1.internal"):
            named.spec()
        with pytest.raises(ProtocolError, match="--allowed-host env-1.internal"):
            named.request({"op": "reset", "episode_id": "x", "sample": _sample()})
    finally:
        named.close()
    assert _Recorder.calls == []
