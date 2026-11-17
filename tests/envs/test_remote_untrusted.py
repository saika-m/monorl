"""What the remote client does with a server that sends rewards and bodies no environment
should: the servers here are scripted, since monorl's own server cannot be made to send them."""

from __future__ import annotations

import gzip
import json
import sys
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from monorl.core.errors import EnvironmentError_
from monorl.core.types import Completion, Sample
from monorl.envs.remote import RemoteEnv, StdioTransport, shutdown_pools, slot_source_for

SPEC = '{"name": "scripted-env", "protocol": "MEP/1", "max_turns": 3, "parallel_episodes": 4}'
RESET = '{"messages": [{"role": "user", "content": "hi"}], "info": {}}'


def _step(reward: str, padding: int = 0) -> str:
    pad = "x" * padding
    return f'{{"messages": [], "reward": {reward}, "done": false, "info": {{"pad": "{pad}"}}}}'


class ScriptedServer:
    """A MEP/1 http server on a free port whose ``/step`` answer is ``respond(handler)``."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.respond: Callable[[BaseHTTPRequestHandler], None] = lambda h: self.send(h, _step("0"))
        self.sent = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                outer.send(self, SPEC if self.path == "/spec" else '{"ok": true}')

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.paths.append(f"{self.path} {body['episode_id']}")
                if self.path == "/step":
                    outer.respond(self)
                else:
                    outer.send(self, RESET if self.path == "/reset" else "{}")

            def log_message(self, *args: object) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, args=(0.01,), daemon=True).start()

    @staticmethod
    def send(handler: BaseHTTPRequestHandler, text: str | bytes, **headers: str) -> None:
        body = text.encode() if isinstance(text, str) else text
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(body)))
        for name, value in headers.items():
            handler.send_header(name.replace("_", "-"), value)
        handler.end_headers()
        handler.wfile.write(body)

    def stream(self, handler: BaseHTTPRequestHandler, total: int) -> None:
        """A body of ``total`` bytes without a declared length, sent until the client hangs up."""
        handler.send_response(200)
        handler.send_header("Content-Type", "application/json")
        handler.end_headers()
        chunk = b" " * 65536
        try:
            handler.wfile.write(b'{"messages": [], "reward": 0, "done": false, "info": {}')
            while self.sent < total:
                handler.wfile.write(chunk)
                self.sent += len(chunk)
            handler.wfile.write(b"}")
        except OSError:
            pass

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[ScriptedServer]:
    shutdown_pools()
    scripted = ScriptedServer()
    try:
        yield scripted
    finally:
        shutdown_pools()
        scripted.close()


def _episode(params: dict) -> RemoteEnv:
    env = RemoteEnv.from_params(params)
    env.reset(Sample(id="s1", messages=[{"role": "user", "content": "hi"}]))
    return env


def _c(text: str = "x") -> Completion:
    return Completion(text=text, token_ids=[])


# -- rewards ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reward", ["NaN", "Infinity", "-Infinity", "1e999", "1" + "0" * 400, '"0.5"', "true", "null", "[1.0]"]
)
def test_a_reward_that_is_not_a_finite_number_fails_the_episode(server, reward):
    server.respond = lambda h: server.send(h, _step(reward))
    env = _episode({"transport": "http", "url": server.url})
    episode = env.episode_id
    with pytest.raises(EnvironmentError_) as failure:
        env.step(_c())
    message = str(failure.value)
    assert "'scripted-env'" in message and server.url in message and f"episode {episode}" in message
    assert "reward must be a" in message
    # the episode is over on both sides
    assert env.episode_id is None
    assert server.paths == [f"/reset {episode}", f"/step {episode}", f"/close {episode}"]


@pytest.mark.parametrize(("reward", "value"), [("1", 1.0), ("-0.25", -0.25), ("1e3", 1000.0), ("0", 0.0)])
def test_a_finite_reward_is_the_steps_reward(server, reward, value):
    server.respond = lambda h: server.send(h, _step(reward))
    env = _episode({"transport": "http", "url": server.url})
    result = env.step(_c())
    assert result.reward == value and isinstance(result.reward, float) and not result.done
    env.close()


def test_a_non_finite_reward_over_stdio_fails_the_episode(tmp_path: Path):
    script = tmp_path / "nan_env.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    op = json.loads(line)['op']\n"
        f"    answers = {{'spec': {SPEC!r}, 'reset': {RESET!r}, 'step': {_step('NaN')!r}}}\n"
        "    print(answers.get(op, '{}'), flush=True)\n"
    )
    env = _episode({"transport": "stdio", "command": [sys.executable, str(script)], "timeout_s": 30})
    episode = env.episode_id
    try:
        with pytest.raises(
            EnvironmentError_, match=f"'scripted-env'.*episode {episode}.*finite number, got nan"
        ):
            env.step(_c())
        assert env.episode_id is None
    finally:
        shutdown_pools()


def test_a_non_finite_reward_stops_the_run(server, make_config):
    """With the fake engine and trainer: the run ends at the first rollout, before any update."""
    from monorl.orchestrator.run import run

    server.respond = lambda h: server.send(h, _step("NaN"))
    config = make_config(env={"type": "remote", "params": {"transport": "http", "url": server.url}})
    result = run(config)
    assert result.state == "failed" and result.final_step == 0
    error = result.error.splitlines()[0]
    assert "environment 'scripted-env'" in error and "reward must be a finite number, got nan" in error
    assert not result.checkpoints


# -- response size ---------------------------------------------------------------------------


def test_http_response_larger_than_the_limit_is_refused(server):
    server.respond = lambda h: server.send(h, _step("0", padding=5000))
    env = _episode({"transport": "http", "url": server.url, "max_response_bytes": 2000})
    with pytest.raises(EnvironmentError_, match="larger than 2000 bytes.*env.params.max_response_bytes"):
        env.step(_c())
    assert env.episode_id is None

    roomy = _episode({"transport": "http", "url": server.url, "max_response_bytes": 10_000})
    assert len(roomy.step(_c()).info["pad"]) == 5000
    roomy.close()


def test_http_response_without_a_length_is_not_read_past_the_limit(server):
    total = 32 * 1024 * 1024
    server.respond = lambda h: server.stream(h, total)
    env = _episode({"transport": "http", "url": server.url, "max_response_bytes": 100_000})
    with pytest.raises(EnvironmentError_, match="larger than 100000 bytes"):
        env.step(_c())
    # the client hung up; what the server got rid of is what the sockets buffer
    assert server.sent < total


def test_http_compressed_response_is_refused(server):
    # 4 kB on the wire that unpack to 4 MB
    packed = gzip.compress(_step("0", padding=4 * 1024 * 1024).encode())
    server.respond = lambda h: server.send(h, packed, Content_Encoding="gzip")
    env = _episode({"transport": "http", "url": server.url})
    with pytest.raises(EnvironmentError_, match="compressed \\(gzip\\)"):
        env.step(_c())


def test_stdio_response_line_longer_than_the_limit_restarts_the_process(tmp_path: Path):
    script = tmp_path / "long_env.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    op = json.loads(line)['op']\n"
        f"    answers = {{'spec': {SPEC!r}, 'health': '{{\"ok\": true}}', 'reset': {_step('0', 50_000)!r}}}\n"
        "    print(answers.get(op, '{}'), flush=True)\n"
    )
    transport = StdioTransport([sys.executable, str(script)], timeout_s=30, max_response_bytes=10_000)
    try:
        assert transport.spec().name == "scripted-env"
        with pytest.raises(EnvironmentError_, match="longer than 10000 characters.*max_response_bytes"):
            transport.request({"op": "reset", "episode_id": "e", "sample": {"id": "s"}})
        assert transport.restarts == 1 and transport.health()
    finally:
        transport.close()


@pytest.mark.parametrize("value", [0, -1, "64MB", 1.5, True])
def test_max_response_bytes_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_response_bytes must be a positive integer"):
        slot_source_for({"transport": "http", "url": "http://127.0.0.1:9", "max_response_bytes": value})
