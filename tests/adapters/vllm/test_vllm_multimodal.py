"""Multimodal prompts: ``TokensPrompt(multi_modal_data=...)`` in colocate mode and
``/v1/chat/completions`` with ``image_url`` parts in server mode (vLLM v0.33.0 shapes)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from PIL import Image
from vllm_fakes import FakeLLM, LoopbackServer

from monorl.adapters.vllm.client import VllmServerClient
from monorl.adapters.vllm.common import (
    ImageUrlCache,
    chat_completion_kwargs,
    chat_completion_messages,
    completions_from_chat_response,
    encode_prompt,
    limit_mm_per_prompt,
    tokens_prompt_kwargs,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, GenerationCancelled
from monorl.core.registry import registry
from monorl.core.types import SamplingParams

PLACEHOLDER_IDS = [101, 102, 103, 104]


class FakeProcessor:
    """``AutoProcessor`` stand-in: records the HF-shaped chat and returns fixed placeholder ids."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kw: Any) -> dict[str, torch.Tensor]:
        self.calls.append(messages)
        return {"input_ids": torch.tensor([PLACEHOLDER_IDS]), "pixel_values": torch.zeros(1, 3, 2, 2)}


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    p = tmp_path / "red.png"
    Image.new("RGB", (6, 4), (255, 0, 0)).save(p)
    return p


@pytest.fixture
def fake_processor(monkeypatch: pytest.MonkeyPatch) -> FakeProcessor:
    proc = FakeProcessor()
    import monorl.rollout.media as media

    monkeypatch.setattr(media, "load_processor", lambda config: proc)
    return proc


def _chat(image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": "colour?"}],
        }
    ]


# -- common helpers ---------------------------------------------------------------------------------


def test_encode_prompt_uses_processor_ids_and_images(
    image_path: Path, fake_processor: FakeProcessor, tiny_model_dir: Path
) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    ids, images = encode_prompt(tok, fake_processor, _chat(image_path))
    assert ids == PLACEHOLDER_IDS
    assert images is not None and len(images) == 1 and images[0].size == (6, 4)
    assert fake_processor.calls[0][0]["content"][0]["type"] == "image"
    text_ids, none = encode_prompt(tok, fake_processor, [{"role": "user", "content": "hi"}])
    assert none is None and text_ids and text_ids != PLACEHOLDER_IDS
    assert encode_prompt(tok, None, [1, 2, 3]) == ([1, 2, 3], None)
    with pytest.raises(ConfigError, match="AutoProcessor"):
        encode_prompt(tok, None, _chat(image_path))


def test_tokens_prompt_and_limit_mm() -> None:
    assert tokens_prompt_kwargs([1, 2], None) == {"prompt_token_ids": [1, 2]}
    assert tokens_prompt_kwargs([1, 2], ["img"]) == {
        "prompt_token_ids": [1, 2],
        "multi_modal_data": {"image": ["img"]},
    }
    assert limit_mm_per_prompt({}) is None
    assert limit_mm_per_prompt({"limit_mm_per_prompt": 3}) == {"image": 3}
    assert limit_mm_per_prompt({"limit_mm_per_prompt": {"image": 2, "video": 1}}) == {"image": 2, "video": 1}
    with pytest.raises(ConfigError):
        limit_mm_per_prompt({"limit_mm_per_prompt": "two"})


def test_chat_completion_messages_and_kwargs(image_path: Path) -> None:
    msgs = chat_completion_messages(
        [{"role": "system", "content": "be brief"}, *_chat(image_path)], ImageUrlCache()
    )
    assert msgs[0] == {"role": "system", "content": "be brief"}
    parts = msgs[1]["content"]
    assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"
    assert parts[1] == {"type": "text", "text": "colour?"}
    data_url = "data:image/png;base64,AAAA"
    passthrough = chat_completion_messages(
        [{"role": "user", "content": [{"type": "image", "image": data_url}]}], ImageUrlCache()
    )
    assert passthrough[0]["content"][0]["image_url"]["url"] == data_url
    with pytest.raises(ConfigError, match="audio"):
        chat_completion_messages(
            [{"role": "user", "content": [{"type": "audio", "audio": "a.wav"}]}], ImageUrlCache()
        )

    kw = chat_completion_kwargs(SamplingParams(n=3, max_new_tokens=5, top_k=-1, logprobs=True, seed=7))
    assert kw["n"] == 3 and kw["max_completion_tokens"] == 5 and kw["top_k"] == 0 and kw["seed"] == 7
    assert kw["logprobs"] is True and kw["top_logprobs"] == 0 and kw["return_token_ids"] is True
    assert kw["include_stop_str_in_output"] is True
    assert chat_completion_kwargs(SamplingParams(logprobs=False))["logprobs"] is False


def test_completions_from_chat_response_validation() -> None:
    resp = {
        "prompt_token_ids": [9, 8],
        "choices": [
            {
                "index": 1,
                "message": {"role": "assistant", "content": "b"},
                "token_ids": [2],
                "finish_reason": "length",
                "logprobs": {"content": [{"token": "b", "logprob": None}]},
            },
            {
                "index": 0,
                "message": {"role": "assistant", "content": "a"},
                "token_ids": [1, 3],
                "finish_reason": "stop",
                "logprobs": {"content": [{"token": "a", "logprob": -0.1}, {"token": "c", "logprob": -0.2}]},
            },
        ],
    }
    comps = completions_from_chat_response(resp, 2, want_logprobs=True)
    assert [c.text for c in comps] == ["a", "b"] and comps[0].logprobs == [-0.1, -0.2]
    assert math.isnan(comps[1].logprobs[0]) and comps[1].finish_reason == "length"
    assert comps[0].prompt_token_ids == [9, 8]
    with pytest.raises(ValueError, match="expected 1"):
        completions_from_chat_response(resp, 1, want_logprobs=False)
    with pytest.raises(ValueError, match="prompt_token_ids"):
        completions_from_chat_response({"choices": resp["choices"]}, 2, want_logprobs=False)
    no_lp = {
        "prompt_token_ids": [1],
        "choices": [{"index": 0, "message": {"content": "x"}, "token_ids": [1]}],
    }
    with pytest.raises(ValueError, match="logprobs.content"):
        completions_from_chat_response(no_lp, 1, want_logprobs=True)


# -- colocated engine ---------------------------------------------------------------------------


def test_engine_builds_multi_modal_tokens_prompt(
    fake_vllm, vllm_config: RunConfig, image_path: Path, fake_processor: FakeProcessor
) -> None:
    vllm_config.rollout.extra = {"limit_mm_per_prompt": {"image": 2}}
    eng = registry.get("rollout_engine", "vllm")(vllm_config)
    assert FakeLLM.instances[-1].kwargs["limit_mm_per_prompt"] == {"image": 2}
    out = eng.generate(
        [_chat(image_path), [{"role": "user", "content": "text only"}]], SamplingParams(n=1, logprobs=True)
    )
    prompts = FakeLLM.instances[-1].calls[-1][1]["prompts"]
    assert prompts[0]["prompt_token_ids"] == PLACEHOLDER_IDS
    assert len(prompts[0]["multi_modal_data"]["image"]) == 1 and prompts[0]["multi_modal_data"]["image"][
        0
    ].size == (6, 4)
    assert "multi_modal_data" not in prompts[1]
    assert out[0][0].prompt_token_ids == PLACEHOLDER_IDS and out[0][0].logprobs == [-0.1, -0.2, -0.3]
    assert len(fake_processor.calls) == 1  # text prompt never touched the processor


def test_engine_text_only_never_loads_processor(
    fake_vllm, vllm_config: RunConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monorl.rollout.media as media

    def boom(config: RunConfig) -> None:
        raise AssertionError("processor loaded for a text prompt")

    monkeypatch.setattr(media, "load_processor", boom)
    eng = registry.get("rollout_engine", "vllm")(vllm_config)
    assert eng.generate([[1, 2, 3]], SamplingParams(n=1))[0][0].prompt_token_ids == [1, 2, 3]


# -- server client ------------------------------------------------------------------------------


class ChatServer:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        if request.url.path == "/v1/chat/completions":
            n = body["n"]
            choices = []
            cap = body["max_completion_tokens"]  # two tokens per completion, or as many as it allows
            for i in reversed(range(n)):
                choice: dict[str, Any] = {
                    "index": i,
                    "message": {"role": "assistant", "content": f"chat-{i}"},
                    "token_ids": [i, 5][:cap],
                    "finish_reason": "stop" if cap >= 2 else "length",
                    "logprobs": None,
                }
                if body.get("logprobs"):
                    choice["logprobs"] = {
                        "content": [{"token": "a", "logprob": -0.5}, {"token": "b", "logprob": -0.25}][:cap]
                    }
                choices.append(choice)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl",
                    "model": body["model"],
                    "choices": choices,
                    "prompt_token_ids": [7, 7, 7],
                },
            )
        if request.url.path == "/v1/completions":
            choices = [
                {
                    "index": p * body["n"] + i,
                    "text": f"t-{p}-{i}",
                    "token_ids": [p, i],
                    "prompt_token_ids": ids,
                    "finish_reason": "stop",
                    "logprobs": {"token_logprobs": [-0.1, -0.2]}
                    if body.get("logprobs") is not None
                    else None,
                }
                for p, ids in enumerate(body["prompt"])
                for i in range(body["n"])
            ]
            return httpx.Response(200, json={"choices": choices, "model": body["model"]})
        return httpx.Response(200, json={"ok": True})


def test_client_routes_image_prompts_through_chat_completions(
    vllm_config: RunConfig, image_path: Path
) -> None:
    vllm_config.rollout.extra = {"mode": "server", "url": "http://vllm.test:8000", "model_name": "policy"}
    server = ChatServer()
    client = VllmServerClient(vllm_config, transport=httpx.MockTransport(server.handler))
    text = [{"role": "user", "content": "hello"}]
    out = client.generate(
        [_chat(image_path), text, _chat(image_path)], SamplingParams(n=2, logprobs=True, max_new_tokens=4)
    )
    client.shutdown()

    paths = [r.url.path for r in server.requests]
    assert paths == ["/v1/completions", "/v1/chat/completions", "/v1/chat/completions"]
    chat_body = json.loads(server.requests[1].content)
    assert chat_body["model"] == "policy" and chat_body["n"] == 2 and chat_body["max_completion_tokens"] == 4
    assert (
        chat_body["logprobs"] is True
        and chat_body["top_logprobs"] == 0
        and chat_body["return_token_ids"] is True
    )
    parts = chat_body["messages"][0]["content"]
    assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert parts[1] == {"type": "text", "text": "colour?"}

    assert len(out) == 3 and all(len(group) == 2 for group in out)
    assert [c.text for c in out[0]] == ["chat-0", "chat-1"]  # sorted by index
    assert out[0][0].prompt_token_ids == [7, 7, 7] and out[0][0].logprobs == [-0.5, -0.25]
    assert out[1][0].text == "t-0-0" and out[1][0].logprobs == [
        -0.1,
        -0.2,
    ]  # text prompt took /v1/completions
    assert out[2][1].token_ids == [1, 5]


def test_an_image_prompt_with_a_cap_of_its_own_asks_for_that_many_tokens(
    vllm_config: RunConfig, image_path: Path
) -> None:
    vllm_config.rollout.extra = {"mode": "server", "url": "http://vllm.test:8000", "model_name": "policy"}
    server = ChatServer()
    client = VllmServerClient(vllm_config, transport=httpx.MockTransport(server.handler))
    try:
        out = client.generate(
            [_chat(image_path), [1, 2, 3], _chat(image_path)],
            SamplingParams(n=2, logprobs=True, max_new_tokens=4),
            max_new_tokens=[1, 3, 4],
        )
    finally:
        client.shutdown()
    bodies = [(r.url.path, json.loads(r.content)) for r in server.requests]
    assert [b["max_completion_tokens"] for path, b in bodies if path == "/v1/chat/completions"] == [1, 4]
    assert [b["max_tokens"] for path, b in bodies if path == "/v1/completions"] == [3]
    assert [[c.token_ids for c in group] for group in out] == [[[0], [1]], [[0, 0], [0, 1]], [[0, 5], [1, 5]]]
    assert [c.finish_reason for c in out[0]] == ["length"] * 2 and out[0][0].logprobs == [-0.5]
    assert [c.finish_reason for c in out[2]] == ["stop"] * 2


# -- server mode: concurrent chat requests, images encoded once ----------------------------------


def _ask(image_path: Path, question: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [{"type": "image", "image": str(image_path)}, {"type": "text", "text": question}],
        }
    ]


def _sent_image(body: dict[str, Any], part: int = 0) -> Image.Image:
    url = body["messages"][0]["content"][part]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.partition(",")[2]))).convert("RGB")


class HeldChatServer(httpx.AsyncBaseTransport):
    """Answers a chat request only once ``expect`` of them have arrived, the latest first, with
    the question it was asked as the completion; ``fail`` questions get a 500 at once, and with
    ``hold`` the others are never answered."""

    def __init__(self, expect: int, *, fail: tuple[str, ...] = (), hold: bool = False) -> None:
        self.expect, self.fail, self.hold = expect, fail, hold
        self.arrived: list[str] = []
        self.answered: list[str] = []
        self.dropped: list[str] = []
        self.most_in_flight = self.in_flight = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(await request.aread())
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(200, json=json.loads(ChatServer().handler(request).content))
        question = body["messages"][-1]["content"][-1]["text"]
        position = len(self.arrived)
        self.arrived.append(question)
        if question in self.fail:
            return httpx.Response(500, text=f"boom {question}")
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)
        try:
            async with asyncio.timeout(10):  # a client sending one at a time ends here
                while len(self.arrived) < self.expect or self.hold:
                    await asyncio.sleep(0.005)
                await asyncio.sleep(0.02 * (self.expect - position))
        except asyncio.CancelledError:
            self.dropped.append(question)
            raise
        finally:
            self.in_flight -= 1
        self.answered.append(question)
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": question},
            "token_ids": [position],
            "finish_reason": "stop",
        }
        return httpx.Response(200, json={"choices": [choice], "prompt_token_ids": [7]})


def _server_client(
    vllm_config: RunConfig, transport: httpx.AsyncBaseTransport | None, url: str = "http://vllm.test:8000"
):
    vllm_config.rollout.extra = {"mode": "server", "url": url, "model_name": "policy"}
    return VllmServerClient(vllm_config, transport=transport)


def test_chat_requests_are_in_flight_together_and_results_keep_prompt_order(
    vllm_config: RunConfig, image_path: Path
) -> None:
    server = HeldChatServer(expect=4)
    client = _server_client(vllm_config, server)
    questions = ["q0", "q1", "q2", "q3"]
    prompts: list[Any] = [_ask(image_path, q) for q in questions]
    prompts.insert(2, [1, 2, 3])  # a text prompt in between keeps its place too
    try:
        out = client.generate(prompts, SamplingParams(n=1, logprobs=False, max_new_tokens=2))
    finally:
        client.shutdown()
    assert server.most_in_flight == 4
    assert server.answered == list(reversed(server.arrived))  # answered in another order than asked
    assert [group[0].text for i, group in enumerate(out) if i != 2] == questions
    assert len(out) == 5 and len(out[2]) == 1


def test_one_failing_chat_request_fails_the_call_with_its_error(
    vllm_config: RunConfig, image_path: Path
) -> None:
    server = HeldChatServer(expect=3, fail=("bad",), hold=True)
    client = _server_client(vllm_config, server)
    try:
        with pytest.raises(RuntimeError, match=r"vllm\.test:8000/v1/chat/completions -> 500: boom bad"):
            client.generate(
                [_ask(image_path, "slow0"), _ask(image_path, "bad"), _ask(image_path, "slow1")],
                SamplingParams(n=1, logprobs=False),
            )
        # The requests that were still running are dropped, not left generating on the server.
        assert sorted(server.dropped) == ["slow0", "slow1"] and server.in_flight == 0
        server.fail, server.hold, server.expect = (), False, 0
        out = client.generate([_ask(image_path, "again")], SamplingParams(n=1, logprobs=False))
        assert out[0][0].text == "again"
        # Two failures by the time the call fails: the earlier prompt's error is the one reported.
        server.fail = ("bad1", "bad0")
        with pytest.raises(RuntimeError, match="boom bad0"):
            client.generate(
                [_ask(image_path, "bad0"), _ask(image_path, "bad1")], SamplingParams(n=1, logprobs=False)
            )
    finally:
        client.shutdown()


def test_cancel_disconnects_every_chat_request_in_flight(vllm_config: RunConfig, image_path: Path) -> None:
    web = LoopbackServer(ChatServer().handler)
    client = _server_client(vllm_config, None, web.url)
    try:
        web.stall = True
        with ThreadPoolExecutor(max_workers=1) as pool:
            running = pool.submit(
                client.generate, [_ask(image_path, "a"), _ask(image_path, "b")], SamplingParams(n=1)
            )
            assert [web.stalled.get(timeout=5) for _ in range(2)] == ["/v1/chat/completions"] * 2
            client.cancel()
            with pytest.raises(GenerationCancelled):
                running.result(timeout=5)
        assert [web.disconnects.get(timeout=5) for _ in range(2)] == ["/v1/chat/completions"] * 2
        web.stall = False
        out = client.generate([_ask(image_path, "c")], SamplingParams(n=1, logprobs=True))
        assert out[0][0].text == "chat-0"
    finally:
        client.shutdown()
        web.close()


def test_a_conversation_encodes_each_image_once_and_never_sends_a_stale_one(
    vllm_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import monorl.rollout.media as media

    encodes: list[tuple[int, int]] = []
    encode = media.image_data_url

    def counting(image: Image.Image, **kw: Any) -> str:
        encodes.append(image.size)
        return encode(image, **kw)

    monkeypatch.setattr(media, "image_data_url", counting)
    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", (6, 4), (255, 0, 0)).save(red)
    Image.new("RGB", (6, 4), (0, 0, 255)).save(blue)
    conversation: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(red)},
                {"type": "image", "image": blue.as_uri()},
                {"type": "text", "text": "turn 0"},
            ],
        }
    ]
    server = ChatServer()
    client = _server_client(vllm_config, httpx.MockTransport(server.handler))
    params = SamplingParams(n=1, logprobs=False)
    try:
        for turn in range(1, 6):
            client.generate([conversation], params)
            conversation = [
                *conversation,
                {"role": "assistant", "content": f"answer {turn}"},
                {"role": "user", "content": [{"type": "text", "text": f"turn {turn}"}]},
            ]
        assert len(encodes) == 2  # not once per image and turn
        bodies = [json.loads(r.content) for r in server.requests]
        assert all(_sent_image(b, 0).getpixel((0, 0)) == (255, 0, 0) for b in bodies)
        assert all(_sent_image(b, 1).getpixel((0, 0)) == (0, 0, 255) for b in bodies)

        # The file behind a reference is replaced by a picture of the same size: the new one is sent.
        Image.new("RGB", (6, 4), (0, 255, 0)).save(red)
        client.generate([conversation], params)
        sent = json.loads(server.requests[-1].content)
        assert _sent_image(sent, 0).getpixel((0, 0)) == (0, 255, 0)
        assert _sent_image(sent, 1).getpixel((0, 0)) == (0, 0, 255)
        assert len(encodes) == 3
    finally:
        client.shutdown()


def test_image_url_cache_stays_within_its_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import monorl.rollout.media as media

    encodes: list[int] = []
    encode = media.image_data_url
    monkeypatch.setattr(media, "image_data_url", lambda image, **kw: encodes.append(1) or encode(image, **kw))
    paths = []
    for i, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        paths.append(str(tmp_path / f"{i}.png"))
        Image.new("RGB", (8, 8), colour).save(paths[-1])
    urls = [ImageUrlCache().data_url(p) for p in paths]
    assert len(set(urls)) == 3
    small, large = min(map(len, urls)), max(map(len, urls))

    def encodes_for(max_bytes: int, order: list[int]) -> int:
        cache = ImageUrlCache(max_bytes)
        encodes.clear()
        assert [cache.data_url(paths[i]) for i in order] == [urls[i] for i in order]
        return len(encodes)

    assert encodes_for(3 * large, [0, 1, 2, 0, 1, 2]) == 3
    assert encodes_for(2 * large, [0, 1, 0, 2, 0, 1]) == 4  # 2 pushes out 1, the least recently used
    assert encodes_for(2 * large + small // 2, [0, 1, 2, 0]) == 4  # room for two, not for three
    assert encodes_for(small - 1, [0, 0, 1]) == 3  # nothing fits: every use encodes, the URLs stay right
