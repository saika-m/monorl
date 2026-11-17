"""The output embedding is applied at the response positions only: a step and the log-probs are
those of the computation over every position, for the models that allow it; the others are
recognised at construction and keep the full computation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import LlamaForCausalLM

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.trainer.reference import ReferenceTrainer
from tests.trainer.conftest import make_batch, make_config


def _config(tiny_model_dir: Path, *, lora: bool, algorithm: dict[str, Any], **trainer: Any) -> RunConfig:
    cfg = make_config(tiny_model_dir, lora=lora, **trainer).to_dict()
    cfg["algorithm"].update(algorithm)
    return RunConfig.from_dict(cfg)


def _batch(tokenizer: Any, trainer: ReferenceTrainer, case: str) -> TrainBatch:
    """Six rows of different lengths with mixed-sign advantages and the sampling log-probs; a value
    position per row for ``ppo``, the sampler's top-4 tokens for ``centering``."""
    batch = make_batch(tokenizer, batch_size=6)
    generator = torch.Generator().manual_seed(1)
    batch.advantages = torch.randn(batch.advantages.shape, generator=generator) * batch.response_mask
    batch.old_logprobs = trainer.compute_logprobs(batch)
    if case == "ppo":
        rows = torch.arange(len(batch))
        state = batch.response_mask.argmax(dim=1) - 1
        batch.value_mask = torch.zeros_like(batch.response_mask)
        batch.value_mask[rows, state] = 1
        batch.returns = torch.zeros(batch.input_ids.shape)
        batch.returns[rows, state] = torch.linspace(-1.0, 2.0, len(batch))
        batch.values = torch.zeros(batch.input_ids.shape)
    if case == "centering":
        with torch.no_grad():
            logits = trainer.hf_model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        top = torch.log_softmax(logits[:, :-1] / trainer.temperature, dim=-1).topk(4, dim=-1)
        mask = batch.response_mask.bool()[..., None]
        ids = torch.nn.functional.pad(top.indices, (0, 0, 1, 0))
        logprobs = torch.nn.functional.pad(top.values, (0, 0, 1, 0), value=float("-inf"))
        batch.behavior_top_ids = ids * mask
        batch.behavior_top_logprobs = logprobs.masked_fill(~mask, float("-inf"))
    return batch


CASES = {
    "grpo": {"name": "grpo", "entropy_coef": 0.01},
    "ppo": {"name": "ppo"},
    "centering": {"name": "grpo"},
    "sft_smoothing": {"name": "sft", "sft_label_smoothing": 0.1},
    "sft_sequence": {"name": "sft", "sft_label_smoothing": 0.1, "loss_agg": "sequence"},
}


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
@pytest.mark.parametrize("case", list(CASES))
def test_a_step_is_that_of_the_logits_at_every_position(
    tiny_model_dir: Path, tokenizer, case: str, lora: bool
) -> None:
    results = []
    for narrow in (True, False):
        torch.manual_seed(0)  # the LoRA adapters are drawn at construction
        trainer = ReferenceTrainer(
            _config(tiny_model_dir, lora=lora, algorithm=CASES[case], gradient_checkpointing=lora)
        )
        assert trainer._narrow_head  # a transformers causal LM applies its head row by row
        trainer._narrow_head = narrow
        batch = _batch(tokenizer, trainer, case)
        metrics = [trainer.train_step(batch) for _ in range(2)]
        weights = torch.cat([w.detach().flatten() for _, w in trainer.export_weights()])
        logprobs = trainer.compute_logprobs(batch)
        reference = trainer.compute_logprobs(batch, reference=True)
        values = trainer.compute_values(batch) if case == "ppo" else None
        results.append((batch.old_logprobs, metrics, weights, logprobs, reference, values))
    (
        (old_a, metrics_a, weights_a, lp_a, ref_a, values_a),
        (old_b, metrics_b, weights_b, lp_b, ref_b, values_b),
    ) = results
    for step_a, step_b in zip(metrics_a, metrics_b, strict=True):
        assert step_a.keys() == step_b.keys()
        assert step_a["grad_norm"] > 0
        for key, value in step_a.items():
            assert value == pytest.approx(step_b[key], rel=1e-4, abs=1e-6), key
    torch.testing.assert_close(weights_a, weights_b, atol=1e-6, rtol=0)
    for a, b in ((old_a, old_b), (lp_a, lp_b), (ref_a, ref_b)):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
    torch.testing.assert_close(ref_a, old_a, atol=1e-5, rtol=0)  # the reference is the initial policy
    assert not torch.allclose(lp_a, old_a, atol=1e-3)  # and the two steps moved the policy
    if case == "ppo":
        assert values_a.abs().max() > 0
        torch.testing.assert_close(values_a, values_b, atol=1e-6, rtol=0)


def test_an_mtp_step_is_that_of_the_logits_at_every_position(tmp_path: Path) -> None:
    """The MTP layers read the last decoder layer's output and call the shared output head
    themselves, on their own hidden states: the narrowing of the policy's pass must leave both alone."""
    from tests.rollout.tiny_mtp import build_tiny_mtp
    from tests.trainer.test_mtp_training import NEW_TOKENS, PROMPTS
    from tests.trainer.test_mtp_training import _batch as mtp_batch

    mtp_dir = build_tiny_mtp(tmp_path / "tiny-mtp")
    generator = torch.Generator().manual_seed(0)
    rows = [
        (len(prompt), prompt + torch.randint(3, 200, (new,), generator=generator).tolist())
        for prompt, new in zip(PROMPTS, NEW_TOKENS, strict=True)
    ]
    batch = mtp_batch(rows, pad=0)
    cfg = make_config(mtp_dir).to_dict()
    cfg["model"]["mtp"] = {"enabled": True, "train": True, "detach_trunk": False, "loss_coef": 1.0}
    results = []
    for narrow in (True, False):
        trainer = ReferenceTrainer(RunConfig.from_dict(cfg))
        assert trainer._narrow_head
        trainer._narrow_head = narrow
        metrics = trainer.train_step(batch)
        results.append((metrics, torch.cat([w.detach().flatten() for _, w in trainer.export_weights()])))
    (metrics_a, weights_a), (metrics_b, weights_b) = results
    assert metrics_a["loss/mtp"] > 0 and metrics_a["grad_norm"] > 0
    for key, value in metrics_a.items():
        assert value == pytest.approx(metrics_b[key], rel=1e-4, abs=1e-6), key
    torch.testing.assert_close(weights_a, weights_b, atol=1e-6, rtol=0)


def test_the_output_embedding_sees_the_response_positions_only(tiny_model_dir: Path, tokenizer) -> None:
    """What the change is for: no ``[B, T, V]`` logits. The head computes one row per response token
    for the loss and the log-probs, and none for the values."""
    trainer = ReferenceTrainer(
        _config(tiny_model_dir, lora=False, algorithm={"name": "ppo"}, micro_batch_size=4)
    )
    batch = _batch(tokenizer, trainer, "ppo")
    vocab = trainer.hf_model.get_output_embeddings().weight.shape[0]
    shapes: list[tuple[int, ...]] = []
    handle = trainer.hf_model.get_output_embeddings().register_forward_hook(
        lambda _module, _args, output: shapes.append(tuple(output.shape))
    )
    per_micro = [int(batch.response_mask[start : start + 4].sum()) for start in (0, 4)]
    trainer.compute_logprobs(batch)
    assert shapes == [(n, vocab) for n in per_micro]
    shapes.clear()
    trainer.train_step(batch)
    assert shapes == [(n, vocab) for n in per_micro]
    shapes.clear()
    trainer.compute_values(batch)
    assert shapes == [(0, vocab), (0, vocab)]
    handle.remove()


class _CentredOverPositions(LlamaForCausalLM):
    """Logits that depend on the other positions: nothing fails on ``[N, V]``, the values differ."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = super().forward(*args, **kwargs)
        out.logits = out.logits - out.logits.mean(dim=1, keepdim=True)
        return out


class _SlicedVocabulary(LlamaForCausalLM):
    """Logits indexed as ``[B, T, V]`` after the head: fails on ``[N, V]``."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = super().forward(*args, **kwargs)
        out.logits = out.logits[:, :, : self.config.vocab_size - 1]
        return out


class _TwoHeadCalls(LlamaForCausalLM):
    """The output embedding is called a second time, on something that is not the batch."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        out = super().forward(*args, **kwargs)
        out.logits = out.logits + 0.5 * self.lm_head(self.model.embed_tokens.weight[:1]).squeeze(0)
        return out


@pytest.mark.parametrize("model_class", [_CentredOverPositions, _SlicedVocabulary, _TwoHeadCalls])
def test_a_model_that_shapes_its_logits_keeps_the_full_computation(
    tiny_model_dir: Path, tokenizer, model_class: type[LlamaForCausalLM]
) -> None:
    model = model_class.from_pretrained(tiny_model_dir)
    # one micro-batch: the logits of these models depend on what shares the batch with a row
    trainer = ReferenceTrainer(make_config(tiny_model_dir, micro_batch_size=6), model=model)
    assert not trainer._narrow_head
    assert model.training  # the check leaves the mode of the model alone
    batch = make_batch(tokenizer, batch_size=6)
    mask = batch.response_mask.bool()
    with torch.no_grad():
        model.eval()
        logits = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask).logits
        model.train()
    expected = torch.log_softmax(logits[:, :-1] / trainer.temperature, dim=-1)
    expected = expected.gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
    logprobs = trainer.compute_logprobs(batch)
    torch.testing.assert_close(logprobs[:, 1:][mask[:, 1:]], expected[mask[:, 1:]], atol=1e-5, rtol=0)
    assert not logprobs[~mask].any()
    batch.old_logprobs = logprobs
    losses = [trainer.train_step(batch)["loss/policy"] for _ in range(4)]
    assert losses[-1] < losses[0]
    assert trainer.compute_logprobs(batch)[mask].mean() > logprobs[mask].mean()


def test_construction_draws_no_random_numbers_for_the_check(tiny_model_dir: Path) -> None:
    """LoRA dropout is live in training mode: the check must not move the random stream a run's
    seed fixed, or every later draw of the process differs."""
    cfg = make_config(tiny_model_dir, lora=True).to_dict()
    cfg["model"]["lora"]["dropout"] = 0.5
    torch.manual_seed(0)
    trainer = ReferenceTrainer(RunConfig.from_dict(cfg))
    after = torch.rand(4)
    assert trainer._narrow_head
    torch.manual_seed(0)
    trainer._wrap_lora(trainer._load_base())  # the only draws of construction: the adapters' initialisation
    torch.testing.assert_close(after, torch.rand(4), atol=0, rtol=0)
