"""Sequences whose advantage is exactly zero at every response token (a prompt group whose rewards
are all equal) are not computed by the reference trainer when the update is the same without them:
the gradient, the optimizer's step and the metrics are those of the step that computes every
sequence, and where the loss has another term, or 0 times something is not 0, every sequence is
still computed."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.trainer.reference import ZERO_AT_ZERO_ADVANTAGE, ReferenceTrainer
from tests.trainer.conftest import (
    bitwise_equal,
    every_sequence_loss,
    forward_passes,
    make_batch,
    make_config,
    relative_difference,
    step_with_gradient,
    trainer_state,
)

LORA = pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
KEPT_ROWS = (1, 2, 5)
# Float32 gradients of the same sequences accumulated over other micro-batches (measured: 1.5e-7).
SAME_GRADIENT = 2e-5
# Warmup from 0 over three steps, then a linear decay; the decay of the weights is on.
SCHEDULE = {"warmup_steps": 3, "scheduler": "linear", "max_steps": 8, "weight_decay": 0.1}


def _config(model_dir: Path, *, lora: bool = False, algorithm: dict | None = None, **trainer: Any):
    cfg = make_config(model_dir, lora=lora, **trainer).to_dict()
    cfg["algorithm"].update(algorithm or {})
    return RunConfig.from_dict(cfg)


def _pair(config: RunConfig) -> tuple[ReferenceTrainer, ReferenceTrainer]:
    """Two trainers with the same weights: the one under test and one that computes every sequence."""
    trainers = []
    for loss_fn in (None, every_sequence_loss):
        torch.manual_seed(0)  # LoRA adapters are initialised at random
        trainers.append(ReferenceTrainer(config, loss_fn=loss_fn))
    return trainers[0], trainers[1]


def _mixed_batch(tiny_model_dir: Path, tokenizer: Any, *, loss_agg: str = "token") -> TrainBatch:
    """Six sequences of two lengths. Rows 0, 3 and 4 have advantage 0, the others advantages of
    both signs; the old log-probs are off the trainer's by up to a factor e^0.4 per token, so the
    ratio is not 1 and the clip is active on some tokens."""
    batch = make_batch(tokenizer, batch_size=6)
    mask = batch.response_mask.float()
    generator = torch.Generator().manual_seed(1)
    batch.advantages = torch.tensor([0.0, 1.5, -0.7, 0.0, 0.0, 0.4])[:, None] * mask
    logprobs = ReferenceTrainer(make_config(tiny_model_dir)).compute_logprobs(batch)
    batch.old_logprobs = logprobs + (torch.rand(logprobs.shape, generator=generator) - 0.5) * 0.8 * mask
    if loss_agg == "rollout":  # rows 0-1, 2-3, 4-5 are the sequences of three rollouts
        per_rollout = mask.sum(dim=1).reshape(3, 2).sum(dim=1).repeat_interleave(2)
        batch.loss_weights = mask / per_rollout[:, None]
    return batch


def _with_advantages(batch: TrainBatch, advantages: list[float]) -> TrainBatch:
    out = batch.slice(0, len(batch))
    out.advantages = torch.tensor(advantages)[:, None] * batch.response_mask.float()
    return out


def _same_metrics(ours: dict[str, float], theirs: dict[str, float]) -> bool:
    return ours.keys() == theirs.keys() and all(
        v == theirs[k] or (math.isnan(v) and math.isnan(theirs[k])) for k, v in ours.items()
    )


# -- the same gradient -----------------------------------------------------------------------------


@LORA
@pytest.mark.parametrize("loss_agg", ["token", "sequence", "rollout"])
def test_the_gradient_is_the_one_of_every_sequence(
    tiny_model_dir: Path, tokenizer, lora: bool, loss_agg: str
) -> None:
    batch = _mixed_batch(tiny_model_dir, tokenizer, loss_agg=loss_agg)
    config = _config(tiny_model_dir, lora=lora, algorithm={"loss_agg": loss_agg}, max_grad_norm=1e9)
    ours, every = _pair(config)
    with forward_passes(ours) as passes:
        metrics, gradient = step_with_gradient(ours, batch)
    with forward_passes(every) as every_passes:
        expected_metrics, expected = step_with_gradient(every, batch)

    # the three sequences with an advantage, in micro-batches of two; none of the other three
    kept = batch.select(torch.tensor(KEPT_ROWS))
    assert passes == [tuple(kept.slice(i, i + 2).trimmed().input_ids.shape) for i in (0, 2)]
    assert [rows for rows, _ in every_passes] == [2, 2, 2]
    assert metrics["tokens/sequences_skipped"] == 3 and expected_metrics["tokens/sequences_skipped"] == 0

    assert float(expected.nan_to_num().norm()) > 0
    assert relative_difference(gradient, expected) < SAME_GRADIENT
    assert metrics["grad_norm"] == pytest.approx(expected_metrics["grad_norm"], rel=SAME_GRADIENT)
    assert metrics["lr"] == expected_metrics["lr"]
    assert metrics["tokens/response_mean"] == expected_metrics["tokens/response_mean"] == 5.5

    # A sequence without advantage is 0 in these means, so they are still means over the batch ...
    assert metrics.keys() == expected_metrics.keys()
    for key in ZERO_AT_ZERO_ADVANTAGE:
        assert metrics[key] == pytest.approx(expected_metrics[key], rel=1e-5, abs=1e-9), key
    assert metrics["policy/clipfrac"] > 0 and metrics["loss/policy"] != 0
    # ... and these, which only a forward pass gives, are means over the sequences that were computed.
    ran = _pair(config)[1].train_step(kept)
    for key in ("loss/entropy", "policy/approx_kl", "policy/ratio_mean"):
        assert metrics[key] == pytest.approx(ran[key], rel=1e-5), key
    for key in ("policy/approx_kl", "policy/ratio_mean"):  # the entropy of a random model is flat
        assert metrics[key] != pytest.approx(expected_metrics[key], rel=1e-3), key


def test_sequences_without_response_tokens_are_left_out_with_them(tiny_model_dir: Path, tokenizer) -> None:
    """A sequence that lost its response to the cut at ``max_seq_len`` has no advantage either."""
    batch = _mixed_batch(tiny_model_dir, tokenizer, loss_agg="sequence")
    batch.response_mask[2] = 0
    batch.advantages[2] = 0
    batch.old_logprobs[2] = 0
    ours, every = _pair(_config(tiny_model_dir, algorithm={"loss_agg": "sequence"}, max_grad_norm=1e9))
    with forward_passes(ours) as passes:
        metrics, gradient = step_with_gradient(ours, batch)
    expected_metrics, expected = step_with_gradient(every, batch)
    assert [rows for rows, _ in passes] == [2] and metrics["tokens/sequences_skipped"] == 4
    assert relative_difference(gradient, expected) < SAME_GRADIENT
    assert metrics["loss/policy"] == pytest.approx(expected_metrics["loss/policy"], rel=1e-5)
    assert metrics["tokens/response_mean"] == expected_metrics["tokens/response_mean"]


# -- the same steps --------------------------------------------------------------------------------


@LORA
@pytest.mark.parametrize("optimizer", ["adamw", "adam", "sgd"])
def test_steps_without_any_advantage_are_bitwise_the_steps_of_every_sequence(
    tiny_model_dir: Path, tokenizer, lora: bool, optimizer: str
) -> None:
    """Two steps on advantages build up momentum, then four steps follow in which no sequence has
    one: momentum and weight decay keep moving the weights while the learning rate warms up and
    decays, and no pass is run for it."""
    batch = _mixed_batch(tiny_model_dir, tokenizer)
    ours, every = _pair(_config(tiny_model_dir, lora=lora, optimizer=optimizer, **SCHEDULE))
    with_advantages = _with_advantages(batch, [1.0, -1.0, 0.5, 2.0, -0.3, 0.8])
    without = _with_advantages(batch, [0.0] * 6)
    for step, item in enumerate([with_advantages] * 2 + [without] * 4, start=1):
        before = trainer_state(ours)
        with forward_passes(ours) as passes:
            metrics = ours.train_step(item)
        expected = every.train_step(item)
        state = trainer_state(ours)
        assert bitwise_equal(state, trainer_state(every)), step
        # the step was taken (the first one of the warmup has learning rate 0)
        assert step == 1 or not bitwise_equal(state, before), step
        if step <= 2:
            assert metrics == expected and len(passes) == 3
            continue
        assert passes == []
        assert metrics == {
            **dict.fromkeys(ZERO_AT_ZERO_ADVANTAGE, 0.0),
            "grad_norm": 0.0,
            "lr": expected["lr"],
            "tokens/response_mean": 5.5,
            "tokens/sequences_skipped": 6.0,
        }
        assert all(expected[key] == 0.0 for key in (*ZERO_AT_ZERO_ADVANTAGE, "grad_norm"))
    assert ours.step == every.step == 6


@LORA
def test_a_run_without_any_advantage_is_bitwise_the_run_of_every_sequence(
    tiny_model_dir: Path, tokenizer, lora: bool
) -> None:
    batch = _with_advantages(_mixed_batch(tiny_model_dir, tokenizer), [0.0] * 6)
    ours, every = _pair(_config(tiny_model_dir, lora=lora, **SCHEDULE))
    initial = trainer_state(ours)
    for step in range(1, 6):
        with forward_passes(ours) as passes:
            metrics = ours.train_step(batch)
        expected = every.train_step(batch)
        assert passes == [] and metrics["lr"] == expected["lr"] and metrics["grad_norm"] == 0.0
        assert bitwise_equal(trainer_state(ours), trainer_state(every)), step
    # AdamW decays the weights also on a gradient of zero
    assert not bitwise_equal(trainer_state(ours), initial)


@LORA
@pytest.mark.parametrize(
    ("optimizer", "same_step"),
    # Adam divides by the size of the gradient: where an entry is rounding noise, so is its sign.
    [("adamw", 1e-2), ("sgd", 1e-4)],
)
def test_steps_with_and_without_advantages_follow_the_steps_of_every_sequence(
    tiny_model_dir: Path, tokenizer, lora: bool, optimizer: str, same_step: float
) -> None:
    """Weight decay under a warmup and a decay of the learning rate, over batches in which some,
    none or all of the sequences have an advantage: after every step the weights are those of the
    trainer that computes every sequence, within ``same_step`` of the largest change of a weight,
    and Adam's first moments within the rounding of the gradients."""
    mixed = _mixed_batch(tiny_model_dir, tokenizer)
    none = _with_advantages(mixed, [0.0] * 6)
    everything = _with_advantages(mixed, [1.0, -1.0, 0.5, 2.0, -0.3, 0.8])
    ours, every = _pair(_config(tiny_model_dir, lora=lora, optimizer=optimizer, **SCHEDULE))
    initial = trainer_state(ours)
    for step, item in enumerate([mixed, none, none, mixed, none, everything, none], start=1):
        metrics, expected = ours.train_step(item), every.train_step(item)
        assert metrics["lr"] == expected["lr"]
        assert metrics["grad_norm"] == pytest.approx(expected["grad_norm"], rel=SAME_GRADIENT)
        state, expected_state = trainer_state(ours), trainer_state(every)
        assert state.keys() == expected_state.keys()
        weights = [name for name in state if name.startswith("weight.")]
        moved = max(float((expected_state[name] - initial[name]).abs().max()) for name in weights)
        assert step == 1 or moved > 0
        for name in weights:
            torch.testing.assert_close(
                state[name], expected_state[name], atol=same_step * moved, rtol=0, msg=f"{step} {name}"
            )
        for name in state:
            if name.endswith(".exp_avg"):
                scale = float(expected_state[name].abs().max())
                torch.testing.assert_close(
                    state[name], expected_state[name], atol=1e-4 * scale, rtol=0, msg=f"{step} {name}"
                )


def test_a_batch_without_a_response_token_takes_no_step_as_before(tiny_model_dir: Path, tokenizer) -> None:
    """Nothing was computed for such a batch before either: no parameter has a gradient, so the
    optimizer steps none, not even to decay it; only the schedule moves on."""
    batch = _mixed_batch(tiny_model_dir, tokenizer)
    batch.response_mask = torch.zeros_like(batch.response_mask)
    batch.advantages = torch.zeros_like(batch.advantages)
    batch.old_logprobs = torch.zeros_like(batch.old_logprobs)
    ours, every = _pair(_config(tiny_model_dir, **SCHEDULE))
    initial = trainer_state(ours)
    for _ in range(2):  # the first step of the warmup has learning rate 0
        metrics, expected = ours.train_step(batch), every.train_step(batch)
        assert metrics == {**expected, "tokens/sequences_skipped": 6.0} and "loss/policy" not in metrics
    assert bitwise_equal(trainer_state(ours), trainer_state(every))
    assert bitwise_equal(trainer_state(ours), initial)


# -- where every sequence is still computed --------------------------------------------------------


def _no_advantage(batch: TrainBatch) -> TrainBatch:
    return _with_advantages(batch, [0.0] * 6)


def _kl(batch: TrainBatch) -> TrainBatch:
    batch = _no_advantage(batch)
    batch.ref_logprobs = batch.old_logprobs - 0.3 * batch.response_mask
    return batch


def _nan_advantage(batch: TrainBatch) -> TrainBatch:
    batch.advantages[3, int(batch.response_mask[3].argmax())] = float("nan")
    return batch


def _old_logprob_far_below(batch: TrainBatch) -> TrainBatch:
    # the ratio of a token of a sequence without advantage overflows: 0 * inf
    batch.old_logprobs[0, int(batch.response_mask[0].argmax())] = -1000.0
    return batch


def _old_logprob_not_finite(batch: TrainBatch) -> TrainBatch:
    batch.old_logprobs[4, 0] = float("inf")  # a prompt token of a sequence without advantage
    return batch


def _weights_on_a_prompt_token(batch: TrainBatch) -> TrainBatch:
    batch.loss_weights[3, 0] = 0.5
    return batch


def _weights_under_the_floor(batch: TrainBatch) -> TrainBatch:
    batch.loss_weights = batch.loss_weights * 1e-9
    return batch


def _negative_weight(batch: TrainBatch) -> TrainBatch:
    batch.loss_weights[0, int(batch.response_mask[0].argmax())] = -0.25
    return batch


CASES: dict[str, tuple[dict[str, Any], Callable[[TrainBatch], TrainBatch], Callable[[dict], bool]]] = {
    # algorithm options, what is done to the mixed batch, what its metrics show
    "kl penalty": ({"kl_coef": 0.1}, _kl, lambda m: m["grad_norm"] > 0 and m["loss/kl"] > 0),
    "kl reported": ({}, _kl, lambda m: m["grad_norm"] == 0 and m["loss/kl"] > 0),
    "entropy bonus": ({"entropy_coef": 0.01}, _no_advantage, lambda m: m["grad_norm"] > 0),
    "tiny advantages": (
        {},
        lambda b: _with_advantages(b, [1e-12, -1e-12, 1e-12, 1e-12, -1e-12, 1e-12]),
        lambda m: 0 < m["grad_norm"] < 1e-9,
    ),
    "nan advantage": ({}, _nan_advantage, lambda m: math.isnan(m["grad_norm"])),
    "ratio overflow": ({}, _old_logprob_far_below, lambda m: math.isnan(m["grad_norm"])),
    "old log-prob not finite": ({}, _old_logprob_not_finite, lambda m: math.isnan(m["loss/policy"])),
    "weights off the response": ({"loss_agg": "rollout"}, _weights_on_a_prompt_token, lambda m: True),
    "weights under the floor": ({"loss_agg": "rollout"}, _weights_under_the_floor, lambda m: True),
    "negative weight": ({"loss_agg": "rollout"}, _negative_weight, lambda m: True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_batch_the_shortcut_is_not_exact_for_takes_the_step_of_every_sequence(
    tiny_model_dir: Path, tokenizer, case: str
) -> None:
    """Another loss term reads the sequences without advantage, or 0 times what the loss computes
    for them is not 0, or the loss mean is not linear in its parts: the step is the one that
    computes every sequence, bit for bit (the two trainers run the same passes)."""
    algorithm, change, shows = CASES[case]
    loss_agg = algorithm.get("loss_agg", "token")
    batch = change(_mixed_batch(tiny_model_dir, tokenizer, loss_agg=loss_agg))
    ours, every = _pair(_config(tiny_model_dir, algorithm=algorithm, **SCHEDULE))
    for _ in range(2):  # the first step of the warmup has learning rate 0
        metrics, expected = ours.train_step(batch), every.train_step(batch)
        assert shows(expected), expected
        assert _same_metrics(metrics, expected)
        assert bitwise_equal(trainer_state(ours), trainer_state(every))


def test_lora_dropout_keeps_every_sequence_and_its_masks(tiny_model_dir: Path, tokenizer) -> None:
    """A pass that draws dropout masks moves the generator for every pass after it, so none is
    left out, and finding that out at construction leaves the generator where it was."""
    cfg = _config(tiny_model_dir, lora=True, **SCHEDULE).to_dict()
    cfg["model"]["lora"]["dropout"] = 0.2
    ours, every = _pair(RunConfig.from_dict(cfg))
    batch = _mixed_batch(tiny_model_dir, tokenizer)
    for seed in (1, 2):
        states = []
        for trainer in (ours, every):
            torch.manual_seed(seed)
            with forward_passes(trainer) as passes:
                metrics = trainer.train_step(batch)
            assert [rows for rows, _ in passes] == [2, 2, 2] and metrics["tokens/sequences_skipped"] == 0
            states.append((trainer_state(trainer), torch.get_rng_state()))
        assert bitwise_equal(states[0][0], states[1][0])
        assert torch.equal(states[0][1], states[1][1])


def test_construction_leaves_the_generators_alone(tiny_model_dir: Path) -> None:
    """The pass that finds out whether the model draws random numbers puts back what it drew."""
    cfg = _config(tiny_model_dir, lora=True).to_dict()
    cfg["model"]["lora"]["dropout"] = 0.2
    draws = []
    for loss_fn in (None, every_sequence_loss):  # with the pass, and without it
        torch.manual_seed(3)
        ReferenceTrainer(RunConfig.from_dict(cfg), loss_fn=loss_fn)
        draws.append(torch.rand(4))
    assert torch.equal(draws[0], draws[1])


def test_an_injected_loss_sees_every_sequence(tiny_model_dir: Path, tokenizer) -> None:
    def likelihood(logprobs, old_logprobs, advantages, response_mask, *, algorithm, **_: Any):
        """Reads no advantage: the log-likelihood of every response token is what it maximises."""
        mask = response_mask.float()
        loss = -(logprobs * mask).sum() / mask.sum()
        return type("Out", (), {"loss": loss, "metrics": {"loss/policy": float(loss.detach())}})()

    batch = _no_advantage(_mixed_batch(tiny_model_dir, tokenizer))
    trainer = ReferenceTrainer(_config(tiny_model_dir), loss_fn=likelihood)
    with forward_passes(trainer) as passes:
        metrics = trainer.train_step(batch)
    assert [rows for rows, _ in passes] == [2, 2, 2]
    assert metrics["grad_norm"] > 0 and metrics["tokens/sequences_skipped"] == 0


def test_sft_trains_on_sequences_without_advantages(tiny_model_dir: Path, tokenizer) -> None:
    batch = _no_advantage(_mixed_batch(tiny_model_dir, tokenizer))
    trainer = ReferenceTrainer(_config(tiny_model_dir, algorithm={"name": "sft"}))
    with forward_passes(trainer) as passes:
        metrics = trainer.train_step(batch)
    assert [rows for rows, _ in passes] == [2, 2, 2]
    assert metrics["grad_norm"] > 0 and metrics["loss/sft"] > 0
    assert "tokens/sequences_skipped" not in metrics
