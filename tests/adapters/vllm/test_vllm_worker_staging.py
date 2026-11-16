"""Worker staging reuse with value-preserving transfer replay and optional real device copies."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
import torch
from vllm_fakes import FakeGPUWorker, is_mtp_completeness_check_enabled

from monorl.adapters.vllm.worker import MonorlWorkerExtension


def _weights(offset: float = 0.0, dtype: torch.dtype = torch.float32):
    policy = [
        ("model.layers.0.w", torch.full((2, 2), offset + 1, dtype=dtype)),
        ("lm_head.weight", torch.full((2, 2), offset + 2, dtype=dtype)),
        ("mtp.gate.weight", torch.full((2, 2), offset + 5, dtype=dtype)),
    ]
    draft = [
        ("mtp.fc.weight", torch.full((2, 2), offset + 3, dtype=dtype)),
        ("mtp.layers.0.mlp.experts.3.down_proj.weight", torch.full((2, 2), offset + 4, dtype=dtype)),
    ]
    return policy, draft


def _worker(quantization=None):
    return FakeGPUWorker(
        0, {"method": "mtp", "num_speculative_tokens": 1, "draft_lm_head_quantization": quantization}
    )


def _trace_transfers(monkeypatch: pytest.MonkeyPatch, weights):
    sources = {id(tensor): name for name, tensor in weights}
    moves = []
    original = torch.Tensor.to

    def transfer(tensor, *args, **kwargs):
        moved = original(tensor, *args, **kwargs)
        if id(tensor) in sources:
            # A real accelerator move already produces a new tensor. Without an accelerator,
            # materialize a real clone to replay nonresident-input staging, not device bandwidth.
            if moved is tensor:
                moved = moved.clone()
            moves.append((sources[id(tensor)], str(moved.device)))
        return moved

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    return moves


def _capture_loader(monkeypatch: pytest.MonkeyPatch, model):
    seen = {}
    original = model.load_weights

    def load(weights):
        def remember():
            for name, tensor in weights:
                seen[name] = tensor
                yield name, tensor

        return original(remember())

    monkeypatch.setattr(model, "load_weights", load)
    return seen


def _assert_values(loaded, expected):
    assert [name for name, _ in loaded] == list(expected)
    for name, tensor in loaded:
        assert torch.equal(tensor.detach().cpu(), expected[name])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("quantization", [None, "fp8"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_same_device_reuses_staged_values_without_stale_update_cache(
    fake_vllm, monkeypatch: pytest.MonkeyPatch, device: str, quantization: str | None, dtype: torch.dtype
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("a CUDA or ROCm PyTorch device is required")
    worker = _worker(quantization)
    target, drafter = worker.model_runner.model, worker.get_draft_model()
    target.to(device)
    drafter.to(device)
    policy, draft = _weights(dtype=dtype)
    moves = _trace_transfers(monkeypatch, policy + draft)
    target_inputs = _capture_loader(monkeypatch, target)
    draft_inputs = _capture_loader(monkeypatch, drafter)
    for update in range(2):
        if update:
            for _, tensor in policy + draft:
                tensor.add_(10)
        expected = {name: tensor.clone() for name, tensor in policy + draft}
        moves.clear()
        target_inputs.clear()
        draft_inputs.clear()
        assert MonorlWorkerExtension.monorl_update_weights(worker, policy, draft) == 5
        MonorlWorkerExtension.monorl_refresh_drafter(worker)
        assert Counter(name for name, _ in moves) == Counter({name: 1 for name in expected})
        for name, tensor in policy:
            assert target_inputs[name] is draft_inputs[name]
            assert target_inputs[name] is not tensor
            assert target_inputs[name].device == target.weight.device
        _assert_values(target.loaded[-len(policy) :], {name: expected[name] for name, _ in policy})
        _assert_values(drafter.loaded[-len(draft) :], {name: expected[name] for name, _ in draft})
        assert all(tensor.device == drafter.weight.device for _, tensor in drafter.loaded[-len(draft) :])
        assert all(torch.equal(tensor, expected[name]) for name, tensor in policy + draft)
        if quantization is not None:
            assert torch.equal(drafter.quantized_head.cpu(), expected["lm_head.weight"])
            assert not drafter.source_head_shared
        assert is_mtp_completeness_check_enabled()


def test_different_devices_get_independent_staging(fake_vllm, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _worker()
    target, drafter = worker.model_runner.model, worker.get_draft_model()
    drafter.to("meta")  # Exercise device routing without claiming another physical accelerator.
    policy, draft = _weights()
    moves = _trace_transfers(monkeypatch, policy + draft)
    target_inputs = _capture_loader(monkeypatch, target)
    draft_inputs = _capture_loader(monkeypatch, drafter)
    MonorlWorkerExtension.monorl_update_weights(worker, policy, draft)
    for name, tensor in policy:
        assert target_inputs[name].device.type == "cpu"
        assert draft_inputs[name].device.type == "meta"
        assert target_inputs[name] is not draft_inputs[name]
        assert [destination for moved, destination in moves if moved == name] == ["cpu", "meta"]
        assert torch.equal(target_inputs[name], tensor)
    for name, tensor in draft:
        assert draft_inputs[name].shape == tensor.shape and draft_inputs[name].dtype == tensor.dtype
        assert [destination for moved, destination in moves if moved == name] == ["meta"]


@pytest.mark.parametrize("failing_loader", ["target", "draft"])
def test_failed_load_restores_drafter_guards_and_allows_fresh_update(
    fake_vllm, monkeypatch: pytest.MonkeyPatch, failing_loader: str
) -> None:
    worker = _worker("fp8")
    target, drafter = worker.model_runner.model, worker.get_draft_model()
    policy, draft = _weights()

    def fail(weights: Any):
        next(iter(weights))
        assert is_mtp_completeness_check_enabled() == (failing_loader == "target")
        raise RuntimeError("rejected update")

    with monkeypatch.context() as patch:
        patch.setattr(target if failing_loader == "target" else drafter, "load_weights", fail)
        with pytest.raises(RuntimeError, match="rejected update"):
            MonorlWorkerExtension.monorl_update_weights(worker, policy, draft)
    assert is_mtp_completeness_check_enabled()
    assert not drafter.source_head_shared
    policy, draft = _weights(offset=10)
    assert MonorlWorkerExtension.monorl_update_weights(worker, policy, draft) == 5
    MonorlWorkerExtension.monorl_refresh_drafter(worker)
    _assert_values(target.loaded[-len(policy) :], dict(policy))
    _assert_values(drafter.loaded[-len(draft) :], dict(draft))
    assert torch.equal(drafter.quantized_head, dict(policy)["lm_head.weight"])


def test_worker_without_drafter_does_not_stage_mtp_weights(
    fake_vllm, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = FakeGPUWorker(0)
    policy, draft = _weights()
    moves = _trace_transfers(monkeypatch, policy + draft)
    assert MonorlWorkerExtension.monorl_update_weights(worker, policy, draft) == len(policy)
    assert [name for name, _ in moves] == [name for name, _ in policy]
    _assert_values(worker.model_runner.model.loaded, dict(policy))


def test_mtp_only_update_leaves_policy_untouched(fake_vllm, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _worker()
    _, draft = _weights()
    moves = _trace_transfers(monkeypatch, draft)
    assert MonorlWorkerExtension.monorl_update_weights(worker, [], draft) == len(draft)
    assert worker.model_runner.model.loaded == []
    assert [name for name, _ in moves] == [name for name, _ in draft]
    _assert_values(worker.get_draft_model().loaded, dict(draft))
