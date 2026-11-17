from __future__ import annotations

import asyncio
import json
import math
import multiprocessing
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import httpx
import pytest
import torch
from fastapi import FastAPI
from vllm_fakes import LoopbackServer

from monorl.adapters.vllm.client import VllmServerClient
from monorl.adapters.vllm.common import decode_weights
from monorl.adapters.vllm.server import attach_router
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, GenerationCancelled
from monorl.core.interfaces import RolloutEngine
from monorl.core.registry import registry
from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import chat_prompt_ids


class FakeServer:
    """Answers like a monorl vLLM server (vLLM v0.33.0 ``CompletionResponse`` shape): three
    tokens per completion, or the request's ``max_tokens`` of them (``finish_reason`` ``length``)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_paths: set[str] = set()
        self.finish_reasons = ["stop"]
        self.shuffle = True
        self.weights: dict[str, torch.Tensor] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.fail_paths:
            return httpx.Response(500, text="boom")
        if path == "/v1/completions":
            body = json.loads(request.content)
            n = body["n"]
            choices = []
            for p, ids in enumerate(body["prompt"]):
                for i in range(n):
                    toks = [p, i, 7][: body["max_tokens"]]
                    finish = self.finish_reasons[(p * n + i) % len(self.finish_reasons)]
                    choice: dict[str, Any] = {
                        "index": p * n + i,
                        "text": f"text-{p}-{i}",
                        "token_ids": toks,
                        "prompt_token_ids": list(ids),
                        "finish_reason": finish if len(toks) == 3 else "length",
                        "logprobs": None,
                    }
                    if body.get("logprobs") is not None:
                        choice["logprobs"] = {
                            "text_offset": list(range(len(toks))),
                            "token_logprobs": [-(k + 1) / 10 for k in range(len(toks))],
                            "tokens": [f"t{t}" for t in toks],
                            "top_logprobs": [None] * len(toks),
                        }
                    choices.append(choice)
            if self.shuffle:
                choices.reverse()
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
                    "object": "text_completion",
                    "model": body["model"],
                    "choices": choices,
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        if path == "/monorl/update_weights":
            assert int(request.headers["content-length"]) == len(request.content)
            weights, _ = decode_weights(bytearray(request.content))
            self.weights.update(weights)
            return httpx.Response(200, json={"tensors": len(weights), "loaded": [len(weights)]})
        return httpx.Response(200, json={"status": "ok", "path": path})

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def client(server: FakeServer, vllm_config: RunConfig):
    vllm_config.rollout.extra = {"mode": "server", "url": "http://vllm.test:8000/", "model_name": "policy"}
    eng = registry.get("rollout_engine", "vllm")(vllm_config, transport=httpx.MockTransport(server.handler))
    assert isinstance(eng, VllmServerClient) and isinstance(eng, RolloutEngine)
    yield eng
    eng.shutdown()


@pytest.fixture
def loopback(server: FakeServer, vllm_config: RunConfig):
    """A client talking to ``server`` over a real loopback connection."""
    web = LoopbackServer(server.handler)
    vllm_config.rollout.extra = {"mode": "server", "url": web.url, "model_name": "policy"}
    eng = VllmServerClient(vllm_config)
    yield web, eng
    eng.shutdown()
    web.close()


def test_server_mode_needs_url(vllm_config: RunConfig) -> None:
    vllm_config.rollout.extra = {"mode": "server"}
    with pytest.raises(ConfigError, match="rollout.extra.url"):
        registry.get("rollout_engine", "vllm")(vllm_config)


def test_urls_per_replica(vllm_config: RunConfig, server: FakeServer, tiny_model_dir) -> None:
    vllm_config.rollout.extra = {"mode": "server", "urls": ["http://a:1", "http://b:2/"]}
    transport = httpx.MockTransport(server.handler)
    second = VllmServerClient(vllm_config, replica=1, transport=transport)
    assert second.url == "http://b:2"
    second.shutdown()
    with pytest.raises(ConfigError, match="replica 2"):
        VllmServerClient(vllm_config, replica=2, transport=transport)


def test_completions_request_body_and_parsing(client: VllmServerClient, server: FakeServer) -> None:
    chat = [{"role": "user", "content": "hi"}]
    params = SamplingParams(
        n=2, max_new_tokens=9, temperature=0.5, top_k=-1, stop=["x"], seed=3, logprobs=True
    )
    out = client.generate([[4, 5], chat], params)
    req = server.requests[-1]
    assert req.url.path == "/v1/completions" and req.headers["content-type"].startswith("application/json")
    body = json.loads(req.content)
    chat_ids = chat_prompt_ids(client.tokenizer, chat)
    assert body["prompt"] == [[4, 5], chat_ids]
    assert body["model"] == "policy" and body["return_token_ids"] is True
    assert body["n"] == 2 and body["max_tokens"] == 9 and body["temperature"] == 0.5
    assert body["top_k"] == 0 and body["stop"] == ["x"] and body["seed"] == 3 and body["logprobs"] == 0
    assert body["include_stop_str_in_output"] is True  # the text must keep e.g. a closing </tool_call>
    assert len(out) == 2 and [len(g) for g in out] == [2, 2]
    assert [c.text for c in out[1]] == ["text-1-0", "text-1-1"]  # regrouped from shuffled choices
    assert out[1][1].token_ids == [1, 1, 7]
    assert out[0][0].logprobs == pytest.approx([-0.1, -0.2, -0.3])
    assert out[0][0].prompt_token_ids == [4, 5] and out[1][0].prompt_token_ids == chat_ids


class HeldCompletionServer(httpx.AsyncBaseTransport):
    """Answers the completion requests like ``server``, but none before ``expect`` of them have
    arrived, and the latest first."""

    def __init__(self, server: FakeServer, expect: int) -> None:
        self.server, self.expect = server, expect
        self.arrived = 0
        self.answered: list[int] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        if request.url.path != "/v1/completions":
            return self.server.handler(request)
        position, self.arrived = self.arrived, self.arrived + 1
        async with asyncio.timeout(10):  # a client sending one at a time ends here
            while self.arrived < self.expect:
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.02 * (self.expect - position))
        self.answered.append(position)
        return self.server.handler(request)


def test_prompts_are_one_completion_request_per_distinct_cap_in_flight_together(
    server: FakeServer, vllm_config: RunConfig
) -> None:
    """``/v1/completions`` has one ``max_tokens`` per request: the prompts of a call with caps of
    their own are grouped by cap, the groups are sent at once, and every prompt gets the
    completions of its own request back."""
    held = HeldCompletionServer(server, expect=3)
    vllm_config.rollout.extra = {"mode": "server", "url": "http://vllm.test:8000", "model_name": "policy"}
    client = VllmServerClient(vllm_config, transport=held)
    prompts = [[10], [11], [12], [13], [14]]
    params = SamplingParams(n=2, max_new_tokens=9, temperature=0.5)
    try:
        out = client.generate(prompts, params, max_new_tokens=[2, 9, 2, 1, 9])
        bodies = [json.loads(r.content) for r in server.requests if r.url.path == "/v1/completions"]
        assert held.answered == [2, 1, 0]  # all three were waiting; answered in another order
        assert sorted((b["max_tokens"], b["prompt"]) for b in bodies) == [
            (1, [[13]]),
            (2, [[10], [12]]),
            (9, [[11], [14]]),
        ]
        assert {(b["n"], b["temperature"], b["logprobs"]) for b in bodies} == {(2, 0.5, 0)}
        assert [group[0].prompt_token_ids for group in out] == prompts
        assert [[len(c.token_ids) for c in group] for group in out] == [
            [2, 2],
            [3, 3],
            [2, 2],
            [1, 1],
            [3, 3],
        ]
        assert [[c.finish_reason for c in group] for group in out] == [
            ["length"] * 2,
            ["stop"] * 2,
            ["length"] * 2,
            ["length"] * 2,
            ["stop"] * 2,
        ]
        assert [c.text for c in out[2]] == ["text-1-0", "text-1-1"]  # second prompt of its request
        assert out[4][1].token_ids == [1, 1, 7] and out[3][0].logprobs == pytest.approx([-0.1])

        # caps that are all the same are one request, as a call without caps is
        server.requests.clear()
        held.expect = 1
        client.generate(prompts, params, max_new_tokens=[4] * 5)
        (body,) = [json.loads(r.content) for r in server.requests]
        assert body["max_tokens"] == 4 and body["prompt"] == prompts
    finally:
        client.shutdown()


def test_cancel_disconnects_the_completion_request_of_every_cap(loopback) -> None:
    web, client = loopback
    web.stall = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(client.generate, [[1], [2]], SamplingParams(n=1), max_new_tokens=[3, 5])
        assert [web.stalled.get(timeout=5) for _ in range(2)] == ["/v1/completions"] * 2
        client.cancel()
        with pytest.raises(GenerationCancelled):
            running.result(timeout=5)
    assert [web.disconnects.get(timeout=5) for _ in range(2)] == ["/v1/completions"] * 2
    web.stall = False
    out = client.generate([[4, 5]], SamplingParams(n=2))
    assert [c.text for c in out[0]] == ["text-0-0", "text-0-1"]


def test_completions_without_logprobs_and_finish_reasons(
    client: VllmServerClient, server: FakeServer
) -> None:
    server.finish_reasons = ["length", "abort"]
    out = client.generate([[1]], SamplingParams(n=2, logprobs=False))
    assert json.loads(server.requests[-1].content)["logprobs"] is None
    assert [c.logprobs for c in out[0]] == [None, None]
    assert [c.finish_reason for c in out[0]] == ["length", "error"]


def test_choice_count_mismatch_is_an_error(client: VllmServerClient, server: FakeServer) -> None:
    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client._client = httpx.AsyncClient(base_url=client.url, transport=httpx.MockTransport(bad))
    with pytest.raises(ValueError, match="expected 2"):
        client.generate([[1]], SamplingParams(n=2))


def test_null_logprob_becomes_nan(client: VllmServerClient) -> None:
    def resp(request: httpx.Request) -> httpx.Response:
        choice = {
            "index": 0,
            "text": "",
            "token_ids": [1, 2],
            "finish_reason": "stop",
            "logprobs": {"token_logprobs": [None, -0.5], "tokens": ["a", "b"]},
        }
        return httpx.Response(200, json={"choices": [choice]})

    client._client = httpx.AsyncClient(base_url=client.url, transport=httpx.MockTransport(resp))
    lps = client.generate([[1]], SamplingParams(n=1))[0][0].logprobs
    assert math.isnan(lps[0]) and lps[1] == -0.5


class WeightsEngine:
    """The engine behind a monorl server, as far as weights go: every worker copies what an
    update carries into its parameters (``default_weight_loader``, vLLM ``weight_utils.py``)."""

    def __init__(self) -> None:
        self.params: dict[str, torch.Tensor] = {}
        self.updates: list[list[int]] = []
        """Per ``monorl_update_weights`` call, the byte sizes of the tensors it carried."""

    async def collective_rpc(
        self, method: str, timeout: float | None = None, args: tuple = (), kwargs: Any = None
    ):
        if method == "monorl_update_weights":
            (weights,) = args
            self.updates.append([t.numel() * t.element_size() for _, t in weights])
            for name, tensor in weights:
                self.params[name] = torch.empty_like(tensor).copy_(tensor)
            return [len(weights)]
        return [None]


@pytest.fixture
def served(vllm_config: RunConfig):
    """``client(budget)``: a client whose requests run the real server routes, and the engine
    those routes drive."""
    app = FastAPI()
    attach_router(app)
    app.post("/reset_prefix_cache")(lambda: {})
    engine = WeightsEngine()
    app.state.engine_client = engine
    clients: list[VllmServerClient] = []

    def client(weight_request_bytes: int | None = None) -> VllmServerClient:
        extra: dict[str, Any] = {"mode": "server", "url": "http://vllm.test:8000"}
        if weight_request_bytes is not None:
            extra["weight_request_bytes"] = weight_request_bytes
        vllm_config.rollout.extra = extra
        clients.append(VllmServerClient(vllm_config, transport=httpx.ASGITransport(app=app)))
        return clients[-1]

    yield client, engine
    for c in clients:
        c.shutdown()


def _mixed_weights() -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(3)
    return [
        ("embed.weight", torch.randn(40, 16, generator=g).to(torch.bfloat16)),  # 1280 B
        ("scale", torch.tensor(2.5)),  # 0-d
        ("l0.bias", torch.randn(7, generator=g).to(torch.float16)),
        ("empty", torch.empty(0, 3)),
        ("l0.weight", torch.randn(16, 16, generator=g, requires_grad=True)),  # 1024 B
        ("transposed", torch.randn(5, 9, generator=g, dtype=torch.float64).t()),  # not contiguous
        ("sliced", torch.arange(60, dtype=torch.int64)[10:50:3]),
        ("mask", torch.rand(3, 5, generator=g) > 0.5),
        ("steps", torch.tensor(7, dtype=torch.int32)),
        ("codes", torch.arange(200, dtype=torch.uint8).reshape(2, 4, 25)),
        ("l1.weight", torch.randn(3, 2, 4, generator=g)),
        ("head.weight", torch.randn(64, 16, generator=g).to(torch.bfloat16)),  # 2048 B
    ]


@pytest.mark.parametrize("budget", [1, 100, 512, 1500, None])
def test_update_weights_leaves_the_server_with_exactly_the_weights_sent(served, budget: int | None) -> None:
    make_client, engine = served
    weights = _mixed_weights()
    make_client(budget).update_weights(iter(weights))

    assert list(engine.params) == [name for name, _ in weights]
    for name, tensor in weights:
        got = engine.params[name]
        assert got.dtype == tensor.dtype and got.shape == tensor.shape, name
        assert torch.equal(got, tensor.detach()), name

    sizes = [t.numel() * t.element_size() for _, t in weights]
    if budget is None:  # the default budget holds all of a small model
        assert len(engine.updates) == 1
        return
    for carried in engine.updates:  # a request stays within the budget unless one tensor exceeds it
        assert sum(carried) <= budget or len(carried) == 1
    # Consecutive requests cannot both fit into one, so their number is bounded by the bytes sent.
    assert len(engine.updates) <= 2 * sum(sizes) // budget + 1
    if budget == 1500:  # nothing is oversized but the head: small tensors share requests
        assert [len(carried) for carried in engine.updates] == [4, 3, 4, 1]


def test_update_weights_bundles_small_tensors_around_large_ones(served) -> None:
    make_client, engine = served
    weights = [(f"norm{i}", torch.full((8,), float(i))) for i in range(200)]  # 32 B each
    weights.insert(50, ("big0", torch.ones(100, 10)))  # 4000 B
    weights.insert(120, ("big1", torch.zeros(100, 10)))
    make_client(1024).update_weights(weights)
    total = sum(t.numel() * t.element_size() for _, t in weights)
    oversized = 2
    assert len(engine.updates) <= -(-total // 1024) + 2 * oversized  # 202 tensors in 11 requests at most
    assert all(torch.equal(engine.params[name], tensor) for name, tensor in weights)


def test_update_weights_streams_large_tensors_over_a_socket(loopback, server: FakeServer) -> None:
    _, client = loopback
    g = torch.Generator().manual_seed(5)
    weights = [
        ("a.weight", torch.randn(700, 1000, generator=g).to(torch.bfloat16)),  # 1.4 MB: several pieces
        ("b", torch.tensor(2.5)),
        ("c.weight", torch.randn(300, 1000, generator=g)),
    ]
    client.update_weights(iter(weights))
    assert server.paths() == ["/monorl/update_weights", "/reset_prefix_cache"]
    assert server.requests[0].headers["content-type"] == "application/vnd.monorl.weights"
    for name, tensor in weights:
        assert server.weights[name].dtype == tensor.dtype and torch.equal(server.weights[name], tensor)


def test_update_weights_names_a_server_of_another_version(
    client: VllmServerClient, server: FakeServer
) -> None:
    def old_server(request: httpx.Request) -> httpx.Response:
        server.requests.append(request)
        return httpx.Response(404, json={"detail": "Not Found"})

    client._client = httpx.AsyncClient(base_url=client.url, transport=httpx.MockTransport(old_server))
    with pytest.raises(RuntimeError, match="same monorl version"):
        client.update_weights([("w", torch.zeros(2))])
    assert server.paths() == ["/monorl/update_weights"]  # no cache reset after a failed update
    with pytest.raises(ValueError, match="complex64"):  # a dtype the format lacks is refused before sending
        client.update_weights([("w", torch.zeros(2, dtype=torch.complex64))])


def test_load_weights_from_path(client: VllmServerClient, server: FakeServer, tmp_path) -> None:
    client.load_weights_from_path(tmp_path)
    req = server.requests[-2]
    assert req.url.path == "/load_weights_from_path" and json.loads(req.content) == {"path": str(tmp_path)}
    assert server.paths()[-1] == "/reset_prefix_cache"


def test_sleep_wake_requests(client: VllmServerClient, server: FakeServer) -> None:
    client.sleep()
    client.sleep()
    assert server.paths() == ["/sleep"] and server.requests[0].url.params["level"] == "1"
    client.update_weights([("w", torch.zeros(1))])  # wakes weights only
    wake = [r for r in server.requests if r.url.path == "/wake_up"]
    assert [r.url.params.get_list("tags") for r in wake] == [["weights"]]
    client.generate([[1]], SamplingParams())  # then the kv cache
    wake = [r for r in server.requests if r.url.path == "/wake_up"]
    assert [r.url.params.get_list("tags") for r in wake] == [["weights"], ["kv_cache"]]
    client.wake_up()
    assert len([r for r in server.requests if r.url.path == "/wake_up"]) == 2
    client.sleep()
    client.wake_up()
    assert server.requests[-1].url.params.get_list("tags") == ["weights", "kv_cache"]


def test_sleep_disabled_in_config(vllm_config: RunConfig, server: FakeServer) -> None:
    vllm_config.rollout.enable_sleep_mode = False
    vllm_config.rollout.extra = {"mode": "server", "url": "http://x"}
    c = VllmServerClient(vllm_config, transport=httpx.MockTransport(server.handler))
    c.sleep()
    c.wake_up()
    c.shutdown()
    assert server.paths() == []


def test_http_errors_surface(client: VllmServerClient, server: FakeServer) -> None:
    server.fail_paths = {"/v1/completions"}
    with pytest.raises(RuntimeError, match="/v1/completions -> 500: boom"):
        client.generate([[1]], SamplingParams())


# -- cancel ---------------------------------------------------------------------------------------


def test_cancel_disconnects_the_running_completion_request(loopback) -> None:
    web, client = loopback
    web.stall = True  # the server holds the request: a long generation
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(client.generate, [[1], [2]], SamplingParams(n=2))
        assert web.stalled.get(timeout=5) == "/v1/completions"
        client.cancel()
        client.cancel()  # repeated calls are harmless
        with pytest.raises(GenerationCancelled):
            running.result(timeout=5)
    assert web.disconnects.get(timeout=5) == "/v1/completions"  # the server saw the client leave
    web.stall = False
    out = client.generate([[4, 5]], SamplingParams(n=2))
    assert [c.text for c in out[0]] == ["text-0-0", "text-0-1"]


def test_cancel_while_idle_does_not_affect_the_next_generate(loopback) -> None:
    web, client = loopback
    client.cancel()
    out = client.generate([[4, 5]], SamplingParams(n=2))
    assert [c.prompt_token_ids for c in out[0]] == [[4, 5], [4, 5]]


# -- shutdown, and a fork of the process ----------------------------------------------------------


def _in_thread(fn: Callable[..., Any], *args: Any) -> Future[Any]:
    """``fn(*args)`` on a daemon thread: a call that never returns fails its test by the timeout
    of ``result`` and does not keep the test process alive."""
    outcome: Future[Any] = Future()

    def run() -> None:
        try:
            outcome.set_result(fn(*args))
        except BaseException as e:  # noqa: BLE001 - handed to the test
            outcome.set_exception(e)

    threading.Thread(target=run, daemon=True).start()
    return outcome


def _in_fork(fn: Callable[[], Any]) -> str:
    """What ``fn()`` does in a fork of this process: ``"returned"``, or the error it raises as
    ``"<type>: <message>"``. A child that has not answered after 5 s fails the test."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def report() -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - reported to the parent
            sender.send(f"{type(e).__name__}: {e}")
        else:
            sender.send("returned")

    child = context.Process(target=report, daemon=True)
    child.start()
    try:
        assert receiver.poll(5), "the call hangs in the forked process"
        return receiver.recv()
    finally:
        child.kill()
        child.join(5)


def test_shutdown_ends_the_generate_call_in_flight(loopback, server: FakeServer) -> None:
    web, client = loopback
    web.stall = True  # the server holds the request: a long generation
    running = _in_thread(client.generate, [[1], [2]], SamplingParams(n=2))
    assert web.stalled.get(timeout=5) == "/v1/completions"
    _in_thread(client.shutdown).result(timeout=5)
    with pytest.raises(GenerationCancelled):
        running.result(timeout=5)
    assert web.disconnects.get(timeout=5) == "/v1/completions"  # the server saw the client leave
    assert client._client.is_closed and not client._loop_thread.is_alive()

    # afterwards every call fails before it does anything, and shutting down again is harmless
    web.stall = False
    weights = iter([("w", torch.zeros(1))])
    for call in (
        lambda: client.generate([[4, 5]], SamplingParams(n=2)),
        lambda: client.update_weights(weights),
        lambda: client.load_weights_from_path("/ckpt"),
        client.sleep,
        client.wake_up,
    ):
        with pytest.raises(RuntimeError, match=f"client for {web.url} was shut down"):
            _in_thread(call).result(timeout=5)
    assert server.paths() == [] and next(weights)[0] == "w"
    _in_thread(client.shutdown).result(timeout=5)


def test_shutdown_ends_a_request_that_is_no_generation(loopback) -> None:
    web, client = loopback
    web.stall_paths = {"/load_weights_from_path"}
    loading = _in_thread(client.load_weights_from_path, "/ckpt")
    assert web.stalled.get(timeout=5) == "/load_weights_from_path"
    _in_thread(client.shutdown).result(timeout=5)
    with pytest.raises(RuntimeError, match="was shut down"):
        loading.result(timeout=5)


def test_shutdown_ends_a_generate_call_that_is_waking_the_server(loopback) -> None:
    web, client = loopback
    client.sleep()
    waking = "/wake_up?tags=weights&tags=kv_cache"
    web.stall_paths = {waking}
    running = _in_thread(client.generate, [[1]], SamplingParams(n=1))
    assert web.stalled.get(timeout=5) == waking
    _in_thread(client.shutdown).result(timeout=5)
    with pytest.raises(GenerationCancelled):
        running.result(timeout=5)


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="needs os.fork")
def test_a_client_in_a_forked_process_fails_at_once(loopback, server: FakeServer) -> None:
    """The thread that runs a client's requests exists in the process that built it only."""
    web, client = loopback
    generate = lambda: client.generate([[4, 5]], SamplingParams(n=2))  # noqa: E731
    assert len(generate()[0]) == 2
    for call in (generate, lambda: client.load_weights_from_path("/ckpt"), client.sleep):
        outcome = _in_fork(call)
        assert outcome.startswith("RuntimeError: ") and "fork" in outcome and "process" in outcome
    # a child that cleans up closes nothing of the parent's
    assert _in_fork(client.shutdown) == "returned"
    assert server.paths() == ["/v1/completions"]
    assert len(generate()[0]) == 2

    # A fork taken while a call is in flight copies the lock that call holds, and nothing in the
    # child releases it: the child is told before it waits for the lock.
    web.stall = True
    running = _in_thread(generate)
    assert web.stalled.get(timeout=5) == "/v1/completions"
    assert "fork" in _in_fork(generate)
    client.cancel()
    with pytest.raises(GenerationCancelled):
        running.result(timeout=5)
