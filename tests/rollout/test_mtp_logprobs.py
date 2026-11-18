"""Speculative score extraction: numerical contracts and bounded accelerator-to-host copies."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

import monorl.rollout.hf_engine as hf_engine
from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import HFRolloutEngine


def _inputs(dtype: torch.dtype):
    rows, steps, vocab = 3, 7, 17
    position = torch.arange(steps + 2).reshape(-1, 1, 1)
    ranks = (torch.arange(vocab).reshape(1, 1, -1) + position) % vocab
    scale = 0.25 + position / 64 + torch.arange(rows).reshape(1, -1, 1) / 32
    raw = ((ranks - 8) * scale).to(dtype)
    generated = torch.arange(rows * steps).reshape(rows, steps) % vocab
    return raw, generated


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("temperature", "sampled", "top_k", "budget"),
    [(0.7, True, 0, 50), (0.7, False, 4, 250), (0.0, True, 4, 250), (0.7, True, 99, 1)],
)
def test_chunked_scores_match_reference_and_ignore_unreturned_logits(
    monkeypatch: pytest.MonkeyPatch, device, dtype, temperature, sampled, top_k, budget
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("a CUDA or ROCm PyTorch device is required")
    monkeypatch.setattr(hf_engine, "_LOGPROB_TRANSFER_BYTES", budget)
    raw, generated = _inputs(dtype)
    params = SamplingParams(temperature=temperature, logprobs=sampled, top_logprobs=top_k)
    logprobs, top = HFRolloutEngine._logprobs_from_logits(
        tuple(raw.to(device).unbind(0)), generated.to(device), params
    )
    scale = temperature if temperature > 0 else 1.0
    reference = torch.log_softmax(raw[: generated.shape[1]].float() / scale, dim=-1).transpose(0, 1)
    if sampled:
        assert logprobs.device.type == "cpu" and logprobs.dtype == torch.float32
        expected = reference.gather(-1, generated[:, :, None]).squeeze(-1)
        torch.testing.assert_close(logprobs, expected, rtol=1e-4, atol=1e-5)
    else:
        assert logprobs is None
    if top_k:
        expected_values, expected_ids = reference.topk(min(top_k, raw.shape[-1]))
        assert top[0].device.type == top[1].device.type == "cpu"
        assert top[0].dtype == torch.int32 and top[1].dtype == torch.float32
        assert torch.equal(top[0], expected_ids.to(torch.int32))
        torch.testing.assert_close(top[1], expected_values, rtol=1e-4, atol=1e-5)
    else:
        assert top is None


@pytest.mark.parametrize("sampled,top_k", [(False, 0), (True, 0), (False, 4), (True, 4)])
def test_empty_scores_preserve_shapes(sampled, top_k):
    logprobs, top = HFRolloutEngine._logprobs_from_logits(
        (), torch.empty((2, 0), dtype=torch.long), SamplingParams(logprobs=sampled, top_logprobs=top_k)
    )
    if sampled:
        assert logprobs.shape == (2, 0)
    else:
        assert logprobs is None
    if top_k:
        assert top[0].shape == top[1].shape == (2, 0, 0)
    else:
        assert top is None


def test_unrequested_scores_do_not_consume_logits():
    logprobs, top = HFRolloutEngine._logprobs_from_logits(
        None, torch.zeros((2, 3), dtype=torch.long), SamplingParams(logprobs=False, top_logprobs=0)
    )
    assert logprobs is None and top is None


class _HostCopies(TorchDispatchMode):
    """Trace logical device-to-host operations on symbolic tensors; no device timing is inferred."""

    def __init__(self):
        super().__init__()
        self.shapes = []
        self.payload_bytes = 0
        self.storage_bytes = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        source = None
        if func == torch.ops.aten.copy_.default:
            destination, candidate = args[:2]
            if destination.device.type == "cpu" and candidate.device.type != "cpu":
                source = candidate
        elif func == torch.ops.aten._to_copy.default:
            destination = kwargs.get("device")
            if destination is not None and torch.device(destination).type == "cpu":
                if args[0].device.type != "cpu":
                    source = args[0]
        if source is not None:
            self.shapes.append(tuple(source.shape))
            self.payload_bytes += source.numel() * source.element_size()
            key = (source.dtype, source.ndim)
            self.storage_bytes[key] = max(self.storage_bytes.get(key, 0), source.untyped_storage().nbytes())
        return func(*args, **kwargs)


@pytest.mark.parametrize(
    ("sampled", "top_k", "steps", "rows", "budget", "copies"),
    [
        (True, 0, 64, 3, 1 << 20, 1),
        (False, 4, 64, 3, 1 << 20, 2),
        (True, 4, 64, 3, 1 << 20, 3),
        (True, 4, 7, 3, 250, 12),
        (True, 99, 3, 2, 1, 9),
        (False, 0, 64, 3, 1 << 20, 0),
    ],
)
def test_accelerator_transfer_count_and_explicit_staging_budget(
    monkeypatch: pytest.MonkeyPatch, sampled, top_k, steps, rows, budget, copies
):
    monkeypatch.setattr(hf_engine, "_LOGPROB_TRANSFER_BYTES", budget)
    vocab = 17
    with FakeTensorMode():
        # Meta exercises the non-host branch without native CUDA indexing guards in a host-only build.
        # Numerical correctness is checked above with real tensors, including optional CUDA/ROCm cases.
        raw = torch.empty((steps, rows, vocab), device="meta")
        generated = torch.empty((rows, steps), dtype=torch.long, device="meta")
        trace = _HostCopies()
        with trace:
            logprobs, top = HFRolloutEngine._logprobs_from_logits(
                tuple(raw.unbind(0)), generated, SamplingParams(logprobs=sampled, top_logprobs=top_k)
            )
        assert len(trace.shapes) == copies
        step_bytes = rows * ((4 if sampled else 0) + 8 * min(top_k, vocab))
        assert trace.payload_bytes == step_bytes * steps
        assert sum(trace.storage_bytes.values()) <= max(budget, step_bytes)
        if sampled:
            assert logprobs.device.type == "cpu" and logprobs.shape == (rows, steps)
        if top_k:
            assert top[0].device.type == top[1].device.type == "cpu"
            assert top[0].shape == top[1].shape == (rows, steps, min(top_k, vocab))
