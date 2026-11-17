"""Bounded MTP drafting work, compared with the unmodified Transformers candidate loop."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from transformers import GenerationConfig
from transformers.cache_utils import MtpCache
from transformers.generation.candidate_generator import MTPCandidateGenerator

import monorl.mtp as mtp_module
from monorl.mtp import MtpLayers
from tests.rollout.tiny_mtp import fit_to_greedy, init_parameters, mtp_layers_on, two_layer_mtp


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


def _generate(policy: Any, prompt: list[int], budget: int, *, sample: bool, mtp: bool = True, seed: int = 19):
    torch.manual_seed(seed)
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

    drafts = 0

    def capture_factory(*args: Any, **kwargs: Any):
        generator = held_factory(*args, **kwargs)
        get_candidates = generator.get_candidates

        def counting(input_ids: torch.Tensor, *positional: Any, **keyword: Any):
            nonlocal drafts
            candidates, logits = get_candidates(input_ids, *positional, **keyword)
            drafts += candidates.shape[1] > input_ids.shape[1]
            return candidates, logits

        generator.get_candidates = counting
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
    # every draft is one token per layer, and a step without a draft counts none
    assert stats.drafted == layers.num_layers * drafts
    if warm:
        assert stats.accepted > 0


SHORT_PROMPTS = [(2, 1, 0), (3, 1, 0), (3, 2, 0), (3, 1, 1), (3, 2, 1), (3, 3, 0)]
"""``(MTP layers, prompt tokens, drafts accepted of the first greedy draft)``. After a draft of
which fewer than ``layers - 1`` tokens were accepted, transformers corrects the drafter's cache
from ``layers`` positions before the verified ones. A prompt of fewer tokens has no such positions
at its first draft; one of exactly as many is the shortest that has."""


@pytest.fixture(params=SHORT_PROMPTS, ids=lambda case: "depth{}-prompt{}-accepted{}".format(*case))
def short_prompt(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """``(policy, layers, prompt, calls)`` of a case of :data:`SHORT_PROMPTS`; ``calls`` gets
    ``(n_last_matches, candidates, draft logits)`` of every ``get_candidates`` call. They come in
    this order: before the policy's first pass, the draft for the prompt, then one per verified
    draft, so the third call is the first that may have to correct the cache."""
    depth, length, accepted = request.param
    prompt = [1, 5, 9][:length]
    with torch.random.fork_rng():
        policy = two_layer_mtp(vocab_size=128)[0].requires_grad_(False)
        drafter = mtp_layers_on(policy, depth).requires_grad_(False)
        layers = MtpLayers.of(policy, drafter)
        layers.install_drafting()
        if accepted:
            # the first layers draft the policy's greedy tokens, the deeper ones at random again
            fit_to_greedy(layers, [prompt], max_new_tokens=12)
            init_parameters(drafter.layers[accepted:])
    held_factory = mtp_module._held_candidate_generator
    calls: list[tuple[int, torch.Tensor, torch.Tensor | None]] = []

    def capture_factory(*args: Any, **kwargs: Any):
        generator = held_factory(*args, **kwargs)
        get_candidates = generator.get_candidates

        def recording(input_ids: torch.Tensor, *positional: Any, **keyword: Any):
            candidates, logits = get_candidates(input_ids, *positional, **keyword)
            calls.append((keyword["n_last_matches"], candidates, logits))
            return candidates, logits

        generator.get_candidates = recording
        return generator

    monkeypatch.setattr(mtp_module, "_held_candidate_generator", capture_factory)
    return policy, layers, prompt, calls


def test_a_prompt_shorter_than_the_mtp_depth_is_drafted_for(short_prompt, request) -> None:
    """Greedy: the tokens are those of plain decoding, and after the rejected first draft the
    layers draft what they draft for the same sequence read from its first token."""
    policy, layers, prompt, calls = short_prompt
    depth, _, accepted = request.node.callspec.params["short_prompt"]
    budget = 12
    actual = _generate(policy, prompt, budget, sample=False)
    assert calls[2][0] == accepted < depth - 1
    expected = _generate(policy, prompt, budget, sample=False, mtp=False)
    assert torch.equal(actual.sequences, expected.sequences)
    for got, wanted in zip(actual.logits, expected.logits, strict=True):
        torch.testing.assert_close(got, wanted, atol=1e-4, rtol=1e-4)
    drafts = [(candidates, logits) for _, candidates, logits in calls if logits is not None]
    assert len(drafts) >= 3
    stats = layers.take_stats()
    # a pass of the policy per call, each emitting the drafts it accepted and a token of its own
    assert stats.steps == len(calls) and stats.emitted == stats.steps + stats.accepted == budget
    assert stats.drafted == depth * len(drafts)
    assert stats.accepted >= sum(matches for matches, _, _ in calls)
    for candidates, logits in drafts:
        # from nothing: no cache, the policy's hidden states of one pass over the sequence
        verified = candidates[:, :-depth]
        positions = torch.arange(verified.shape[1]).unsqueeze(0)
        with torch.no_grad():
            hidden = layers.last_hidden_states(policy(verified[:, :-1], **layers.last_hidden_request))
            fresh, fresh_logits, _ = layers.model(
                input_ids=verified[:, 1:],
                last_hidden_states=hidden,
                attention_mask=None,
                position_ids=positions[:, 1:],
                mtp_cache=MtpCache(config=policy.config.get_mtp_config()),
            )
        assert torch.equal(candidates[:, -depth:], fresh)
        torch.testing.assert_close(logits, fresh_logits, atol=1e-4, rtol=1e-4)


def test_a_short_prompt_is_sampled_from_the_policy(short_prompt, request) -> None:
    """Sampling: which drafts are rejected depends on the seed, so several are decoded. Every
    token's recorded logits are the policy's, and a seed gives the same tokens again."""
    policy, layers, prompt, calls = short_prompt
    depth = request.node.callspec.params["short_prompt"][0]
    budget = 12
    corrected = 0
    for seed in range(12):
        actual = _generate(policy, prompt, budget, sample=True, seed=seed)
        stats = layers.take_stats()
        corrected += calls[2][0] < depth - 1
        assert stats.steps == len(calls) and stats.emitted == stats.steps + stats.accepted == budget
        assert stats.drafted == depth * sum(logits is not None for _, _, logits in calls)
        with torch.no_grad():
            reference = policy(input_ids=actual.sequences).logits[:, len(prompt) - 1 : -1].float()
        tokens = actual.sequences[:, len(prompt) :, None]
        torch.testing.assert_close(
            (torch.stack(actual.logits, dim=1) / 0.7).log_softmax(-1).gather(-1, tokens),
            (reference / 0.7).log_softmax(-1).gather(-1, tokens),
            atol=1e-4,
            rtol=1e-4,
        )
        del calls[:]
        again = _generate(policy, prompt, budget, sample=True, seed=seed)
        assert torch.equal(again.sequences, actual.sequences) and layers.take_stats() == stats
        del calls[:]
    assert corrected  # the first draft of a seed was rejected
