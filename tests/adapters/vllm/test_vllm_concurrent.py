"""Real HTTP checks for concurrent vLLM batches, ordering, and sibling cancellation."""

from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import pytest
from vllm_fakes import LoopbackServer

from monorl.adapters.vllm.client import VllmServerClient
from monorl.core.errors import GenerationCancelled
from monorl.core.types import SamplingParams

_IMAGE = "data:image/png;base64,AAAA"


def _image(tag: str) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": _IMAGE}, {"type": "text", "text": tag}]
    return [{"role": "user", "content": content}]


def _tag(request: httpx.Request) -> str:
    if request.url.path == "/v1/completions":
        return "text"
    return json.loads(request.content)["messages"][0]["content"][-1]["text"]


def _reply(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if request.url.path == "/v1/completions":
        choices = [
            {
                "index": p * body["n"] + sample,
                "text": f"text-{p}-{sample}",
                "token_ids": [p, sample],
                "prompt_token_ids": ids,
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [-0.1, -0.2]},
            }
            for p, ids in enumerate(body["prompt"])
            for sample in range(body["n"])
        ]
        return httpx.Response(200, json={"choices": choices[::-1]})
    tag = _tag(request)
    choices = [
        {
            "index": sample,
            "message": {"role": "assistant", "content": f"{tag}-{sample}"},
            "token_ids": [ord(tag[-1]), sample],
            "finish_reason": "stop",
            "logprobs": {"content": [{"logprob": -0.5}, {"logprob": -0.25}]},
        }
        for sample in reversed(range(body["n"]))
    ]
    return httpx.Response(200, json={"prompt_token_ids": [ord(tag[-1])] * 3, "choices": choices})


@pytest.fixture
def make_client(vllm_config):
    opened = []

    def make(handler=_reply):
        server = LoopbackServer(handler)
        vllm_config.rollout.extra = {"mode": "server", "url": server.url}
        client = VllmServerClient(vllm_config)
        opened.append((server, client))
        return server, client

    yield make
    for server, client in opened:
        client.cancel()
        client.shutdown()
        server.close()


def test_mixed_batch_overlaps_and_restores_prompt_order(make_client) -> None:
    entered: queue.Queue[str] = queue.Queue()
    finished: queue.Queue[str] = queue.Queue()
    release = {tag: threading.Event() for tag in ("image-a", "text", "image-b")}

    def handler(request: httpx.Request) -> httpx.Response:
        tag = _tag(request)
        entered.put(tag)
        if not release[tag].wait(5):
            return httpx.Response(503, text="test barrier was not released")
        finished.put(tag)
        return _reply(request)

    _, client = make_client(handler)
    prompts = [_image("image-a"), [4, 5], _image("image-b")]
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(client.generate, prompts, SamplingParams(n=2))
        try:
            # No response can finish until every request is in flight: serialization deadlocks here.
            assert {entered.get(timeout=5) for _ in range(3)} == set(release)
            for tag in ("image-b", "text", "image-a"):
                release[tag].set()
                assert finished.get(timeout=5) == tag
            result = running.result(timeout=5)
        finally:
            for event in release.values():
                event.set()
            client.cancel()
    assert [[c.text for c in group] for group in result] == [
        ["image-a-0", "image-a-1"],
        ["text-0-0", "text-0-1"],
        ["image-b-0", "image-b-1"],
    ]
    assert result[0][1].token_ids == [ord("a"), 1]
    assert result[0][1].prompt_token_ids == [ord("a")] * 3
    assert result[0][1].logprobs == [-0.5, -0.25]
    assert result[1][0].prompt_token_ids == [4, 5]
    assert result[1][0].logprobs == [-0.1, -0.2]
    assert result[2][0].token_ids == [ord("b"), 0]


def test_cancel_disconnects_every_batch_request_and_allows_reuse(make_client) -> None:
    server, client = make_client()
    server.stall = True
    prompts = [_image("image-a"), [4, 5], _image("image-b")]
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(client.generate, prompts, SamplingParams())
        try:
            paths = [server.stalled.get(timeout=5) for _ in range(3)]
            client.cancel()
            client.cancel()
            with pytest.raises(GenerationCancelled):
                running.result(timeout=5)
            assert sorted(server.disconnects.get(timeout=5) for _ in range(3)) == sorted(paths)
        finally:
            client.cancel()
    server.stall = False
    result = client.generate(prompts, SamplingParams())
    assert [group[0].text for group in result] == ["image-a-0", "text-0-0", "image-b-0"]


def test_http_failure_disconnects_pending_siblings_and_allows_reuse(make_client) -> None:
    fail = True
    release_error = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        if fail and request.url.path == "/v1/completions":
            if not release_error.wait(5):
                return httpx.Response(503, text="test barrier was not released")
            return httpx.Response(500, text="failed batch")
        return _reply(request)

    server, client = make_client(handler)
    server.stall_paths = {"/v1/chat/completions"}
    prompts = [_image("image-a"), [4, 5], _image("image-b")]
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(client.generate, prompts, SamplingParams())
        try:
            assert [server.stalled.get(timeout=5) for _ in range(2)] == ["/v1/chat/completions"] * 2
            release_error.set()
            with pytest.raises(RuntimeError, match="/v1/completions -> 500: failed batch"):
                running.result(timeout=5)
            assert [server.disconnects.get(timeout=5) for _ in range(2)] == ["/v1/chat/completions"] * 2
        finally:
            release_error.set()
            client.cancel()
    fail = False
    server.stall_paths.clear()
    result = client.generate(prompts, SamplingParams())
    assert [group[0].text for group in result] == ["image-a-0", "text-0-0", "image-b-0"]
