from __future__ import annotations

import json
import struct
import warnings
from typing import Any

import httpx
import pytest
import torch
from fastapi import FastAPI

from monorl.adapters.vllm.common import WEIGHTS_WIRE_VERSION, WORKER_EXTENSION_CLS, encode_weights
from monorl.adapters.vllm.server import attach_router


class FakeEngineClient:
    """``AsyncLLM.collective_rpc(method, timeout, args, kwargs)`` (vllm/v1/engine/async_llm.py:1249)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def collective_rpc(
        self, method: str, timeout: float | None = None, args: tuple = (), kwargs: Any = None
    ):
        self.calls.append({"method": method, "args": args, "kwargs": kwargs})
        if method == "monorl_update_weights":
            return [len(args[0])]
        if method == "monorl_device_info":
            return [{"device": "cuda:0", "dtype": "bfloat16"}]
        return [None]


@pytest.fixture
def app_client() -> tuple[httpx.AsyncClient, FakeEngineClient]:
    app = FastAPI()
    attach_router(app)
    engine = FakeEngineClient()
    app.state.engine_client = engine
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://server")
    return client, engine


def _body(entries: list[tuple[str, torch.Tensor, bool]]) -> bytes:
    length, pieces = encode_weights(entries)
    body = b"".join(pieces)
    assert len(body) == length
    return body


async def test_update_weights_rebuilds_the_tensors_of_a_request(app_client) -> None:
    client, engine = app_client
    sent = {
        "lm_head.weight": torch.linspace(-1, 1, 6, dtype=torch.bfloat16).reshape(3, 2),
        "scalar": torch.tensor(2.5, dtype=torch.float64),
        "empty": torch.empty(0, 4, dtype=torch.float16),
        "mask": torch.tensor([True, False, True]),
        "ids": torch.arange(-3, 4, dtype=torch.int64),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # torch warns when a tensor aliases memory it may not write
        r = await client.post(
            "/monorl/update_weights", content=_body([(n, t, False) for n, t in sent.items()])
        )
    assert r.status_code == 200 and r.json() == {"tensors": 5, "loaded": [5]}
    call = engine.calls[-1]
    assert call["method"] == "monorl_update_weights" and len(call["args"]) == 1
    received = dict(call["args"][0])
    assert list(received) == list(sent)
    for name, tensor in sent.items():
        assert received[name].dtype == tensor.dtype and received[name].shape == tensor.shape
        assert torch.equal(received[name], tensor)
    # A worker may load in place: writing one tensor leaves its neighbours in the body alone.
    received["scalar"].fill_(-7.0)
    received["mask"].fill_(False)
    assert torch.equal(received["lm_head.weight"], sent["lm_head.weight"])
    assert torch.equal(received["ids"], sent["ids"]) and received["scalar"].item() == -7.0


async def test_update_weights_sends_mtp_entries_to_the_drafter_only(app_client) -> None:
    client, engine = app_client
    w, m = torch.ones(2, 2), torch.full((3,), 4.0)
    r = await client.post("/monorl/update_weights", content=_body([("a", w, False), ("mtp.0.proj", m, True)]))
    assert r.status_code == 200
    weights, mtp_weights = engine.calls[-1]["args"]
    assert [n for n, _ in weights] == ["a"] and torch.equal(weights[0][1], w)
    assert [n for n, _ in mtp_weights] == ["mtp.0.proj"] and torch.equal(mtp_weights[0][1], m)


async def test_update_weights_rejects_what_it_cannot_read_in_full(app_client) -> None:
    client, engine = app_client
    body = _body([("a", torch.arange(40, dtype=torch.float32), False), ("b", torch.ones(3), False)])

    async def detail(content: bytes) -> str:
        r = await client.post("/monorl/update_weights", content=content)
        assert r.status_code == 400
        return r.json()["detail"]

    assert "cut short" in await detail(body[:20]) and "fewer than" in await detail(body[:5])
    assert "expected" in await detail(body[:-64])  # a tensor is missing
    assert "expected" in await detail(body + bytes(64))
    assert "not a monorl weight payload" in await detail(b"XXXX" + body[4:])
    newer = struct.pack("<I", WEIGHTS_WIRE_VERSION + 1)
    message = await detail(body[:4] + newer + body[8:])
    assert f"wire version {WEIGHTS_WIRE_VERSION + 1}" in message and "same monorl version" in message
    header_len = struct.unpack_from("<I", body, 8)[0]
    header = json.loads(body[12 : 12 + header_len])
    header["tensors"][0]["dtype"] = "complex128"
    forged = json.dumps(header).encode()
    assert "complex128" in await detail(body[:8] + struct.pack("<I", len(forged)) + forged)
    assert "malformed header" in await detail(body[:12] + b"?" * header_len + body[12 + header_len :])
    assert engine.calls == []  # nothing of a rejected request reaches the workers
    # The per-tensor endpoint of earlier versions is gone: an old client fails, it does not load.
    r = await client.post("/update_named_param", params={"name": "a", "dtype": "float32"}, content=b"")
    assert r.status_code == 404


async def test_update_weights_reads_a_body_sent_without_content_length(app_client) -> None:
    client, engine = app_client
    t = torch.arange(1000, dtype=torch.int32)
    body = _body([("a", t, False)])

    async def chunks():
        for start in range(0, len(body), 300):
            yield body[start : start + 300]

    r = await client.post("/monorl/update_weights", content=chunks())
    assert r.status_code == 200 and "content-length" not in r.request.headers
    assert torch.equal(engine.calls[-1]["args"][0][0][1], t)
    r = await client.post(
        "/monorl/update_weights", content=chunks(), headers={"content-length": str(len(body) - 64)}
    )
    assert r.status_code == 400 and "Content-Length" in r.json()["detail"]


async def test_load_weights_from_path(app_client, tmp_path) -> None:
    client, engine = app_client
    r = await client.post("/load_weights_from_path", json={"path": str(tmp_path)})
    assert r.status_code == 200
    assert engine.calls[-1] == {
        "method": "monorl_load_weights_from_path",
        "args": (str(tmp_path),),
        "kwargs": None,
    }
    r = await client.post("/load_weights_from_path", json={"path": str(tmp_path / "missing")})
    assert r.status_code == 400
    r = await client.post("/load_weights_from_path", json={})
    assert r.status_code == 400


async def test_info(app_client) -> None:
    client, _ = app_client
    r = await client.get("/monorl/info")
    assert r.json() == {
        "workers": [{"device": "cuda:0", "dtype": "bfloat16"}],
        "worker_extension_cls": WORKER_EXTENSION_CLS,
    }


async def test_engine_not_ready() -> None:
    app = FastAPI()
    attach_router(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://s") as c:
        r = await c.get("/monorl/info")
    assert r.status_code == 503
