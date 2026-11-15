"""Bounded MTP drafting work, compared with the unmodified Transformers candidate loop."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import GenerationConfig
from transformers.generation.candidate_generator import MTPCandidateGenerator

import monorl.mtp as mtp_module
from monorl.mtp import MtpLayers
from tests.rollout.tiny_mtp import fit_to_greedy, two_layer_mtp


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def drafting_policy(request: pytest.FixtureRequest):
    # ROCm PyTorch uses the same device namespace and exercises these tests unchanged.
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("a CUDA or ROCm PyTorch device is required")
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        policy, drafter = two_layer_mtp(vocab_size=128)
        policy.to(device).requires_grad_(False)
        drafter.to(device).requires_grad_(False)
        layers = MtpLayers.of(policy, drafter)
        layers.install_drafting()
        yield policy, layers


def _generate(policy: Any, prompt: list[int], budget: int, *, sample: bool, mtp: bool = True):
    torch.manual_seed(19)
    config = GenerationConfig(
        max_new_tokens=budget,
        do_sample=sample,
        use_mtp=mtp,
        return_dict_in_generate=True,
        output_logits=True,
        pad_token_id=0,
        eos_token_id=[],
        **({"temperature": 0.7, "top_p": 0.8, "top_k": 0} if sample else {}),
    )
    return policy.generate(
        input_ids=torch.tensor([prompt], device=next(policy.parameters()).device),
        generation_config=config,
    )


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("budget", [1, 2])
def test_terminal_token_needs_no_drafter(drafting_policy, sample: bool, budget: int) -> None:
    policy, layers = drafting_policy
    prompt = [1, 5, 9, 17, 33, 21] * 24
    expected = _generate(policy, prompt, budget, sample=sample, mtp=False)
    actual = _generate(policy, prompt, budget, sample=sample)
    assert torch.equal(actual.sequences, expected.sequences)
    for got, wanted in zip(actual.logits, expected.logits, strict=True):
        torch.testing.assert_close(got, wanted, atol=1e-4, rtol=1e-4)
    stats = layers.take_stats()
    assert stats.emitted == budget and stats.drafted == 0


@pytest.mark.parametrize("sample", [False, True])
@pytest.mark.parametrize("warm", [False, True])
def test_bounded_history_preserves_long_sequence_verification(
    drafting_policy, monkeypatch: pytest.MonkeyPatch, sample: bool, warm: bool
) -> None:
    policy, layers = drafting_policy
    prompt = [1, 5, 9, 17, 33, 21, 64] * 40
    if warm:
        fit_to_greedy(layers, [prompt], max_new_tokens=24)
    held_factory = mtp_module._held_candidate_generator
    generators: list[Any] = []

    def upstream_factory(*args: Any, **kwargs: Any) -> MTPCandidateGenerator:
        # Reuse the real held weights/cache setup; run Transformers' original candidate methods.
        # Avoid its constructor's checkpoint reload (the fixture's two MTP layers live in memory).
        held = held_factory(*args, **kwargs)
        generator = MTPCandidateGenerator.__new__(MTPCandidateGenerator)
        generator.__dict__.update(vars(held))
        generators.append(generator)
        return generator

    with monkeypatch.context() as patch:
        patch.setattr(mtp_module, "_held_candidate_generator", upstream_factory)
        expected = _generate(policy, prompt, 24, sample=sample)
    upstream = generators.pop()

    def capture_factory(*args: Any, **kwargs: Any):
        generator = held_factory(*args, **kwargs)
        generators.append(generator)
        return generator

    with monkeypatch.context() as patch:
        patch.setattr(mtp_module, "_held_candidate_generator", capture_factory)
        actual = _generate(policy, prompt, 24, sample=sample)
    optimized = generators.pop()
    # Omitting the final unused draft changes only the terminal sampling RNG draws, not p(token).
    count = actual.sequences.shape[1] - int(sample)
    assert torch.equal(actual.sequences[:, :count], expected.sequences[:, :count])
    for got, wanted in zip(actual.logits[:-1], expected.logits[:-1], strict=True):
        torch.testing.assert_close(got, wanted, atol=1e-4, rtol=1e-4)
    with torch.no_grad():
        reference = policy(input_ids=actual.sequences).logits[:, len(prompt) - 1 : -1].float()
    captured = torch.stack(actual.logits, dim=1)
    temperature = 0.7 if sample else 1.0
    tokens = actual.sequences[:, len(prompt) :, None]
    torch.testing.assert_close(
        (captured / temperature).log_softmax(-1).gather(-1, tokens),
        (reference / temperature).log_softmax(-1).gather(-1, tokens),
        atol=1e-4,
        rtol=1e-4,
    )
    history = optimized.full_seq_last_hidden_states
    max_bytes = layers.num_layers * history.shape[-1] * history.element_size()
    assert history.untyped_storage().nbytes() <= max_bytes
    assert upstream.full_seq_last_hidden_states.untyped_storage().nbytes() > max_bytes
    stats = layers.take_stats()
    assert stats.emitted == 24 and stats.drafted > stats.accepted
    if warm:
        assert stats.accepted > 0
