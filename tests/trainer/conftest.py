from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch


@dataclass
class SimpleLossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def reinforce_loss(
    logprobs, old_logprobs, advantages, response_mask, *, algorithm, ref_logprobs=None, entropy=None
):
    """Token-mean REINFORCE: ``-mean(adv * logprob)`` over response positions."""
    mask = response_mask.float()
    loss = -(logprobs * advantages * mask).sum() / mask.sum().clamp(min=1)
    return SimpleLossOutput(loss=loss, metrics={"loss/policy": float(loss.detach().item())})


def every_sequence_loss(*args: Any, **kwargs: Any) -> Any:
    """``policy_loss`` as an injected loss. The trainer cannot know the terms of a loss it is handed,
    so it computes every sequence of a batch, also those whose advantage is zero: the step that
    tests of leaving those out compare with."""
    from monorl.algorithms import policy_loss

    return policy_loss(*args, **kwargs)


@contextlib.contextmanager
def forward_passes(trainer: Any) -> Iterator[list[tuple[int, int]]]:
    """The ``(sequences, width)`` of every forward pass of the trainer's policy inside the block."""
    passes: list[tuple[int, int]] = []

    def record(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        passes.append(tuple(kwargs["input_ids"].shape))

    handle = trainer.hf_model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        yield passes
    finally:
        handle.remove()


def step_with_gradient(trainer: Any, batch: TrainBatch) -> tuple[dict[str, float], torch.Tensor]:
    """One ``train_step`` and the gradient its optimizer was handed (clipped), in one vector. A
    parameter without a gradient, which the optimizer does not step, is ``nan`` in it."""
    gradient: list[torch.Tensor] = []

    def keep(*_: Any) -> None:
        for p in trainer.trainable_params:
            grad = torch.full_like(p, float("nan")) if p.grad is None else p.grad.detach()
            gradient.append(grad.flatten().clone())

    handle = trainer.optimizer.register_step_pre_hook(keep)
    try:
        return trainer.train_step(batch), torch.cat(gradient)
    finally:
        handle.remove()


def relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """``|a - b| / |b|`` over all entries, in float64; ``nan`` entries have to be the same ones."""
    assert torch.equal(a.isnan(), b.isnan())
    a, b = a.double().nan_to_num(), b.double().nan_to_num()
    return float((a - b).norm() / b.norm())


def trainer_state(trainer: Any) -> dict[str, torch.Tensor]:
    """Every trainable tensor of ``trainer`` and what its optimizer keeps for it (moments, step)."""
    state = {f"weight.{i}": p.detach().clone() for i, p in enumerate(trainer.trainable_params)}
    for i, entry in enumerate(trainer.optimizer.state.values()):
        for key, value in entry.items():
            state[f"optimizer.{i}.{key}"] = torch.as_tensor(value).detach().clone()
    return state


def bitwise_equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    """The same tensors bit for bit: ``nan`` equals ``nan``, and 0.0 does not equal -0.0."""
    return a.keys() == b.keys() and all(
        a[k].dtype == b[k].dtype and a[k].numpy().tobytes() == b[k].numpy().tobytes() for k in a
    )


def make_config(tiny_model_dir: Path, *, lora: bool = False, **trainer: Any) -> RunConfig:
    trainer_cfg = {
        "backend": "reference",
        "device": "cpu",
        "precision": "fp32",
        "micro_batch_size": 2,
        "max_steps": 10,
        "learning_rate": 1e-2 if lora else 1e-3,
        "max_seq_len": 64,
    }
    trainer_cfg.update(trainer)
    return RunConfig.from_dict(
        {
            "seed": 0,
            "model": {
                "name_or_path": str(tiny_model_dir),
                "dtype": "float32",
                "lora": {"enabled": lora, "r": 4, "alpha": 8, "dropout": 0.0},
            },
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "max_new_tokens": 8, "device": "cpu"},
            "trainer": trainer_cfg,
            "rewards": [{"type": "exact_match"}],
        }
    )


SGD = {"optimizer": "sgd", "learning_rate": 0.5}
"""Trainer options for a test that compares the updates of two computations of one gradient: with
plain SGD the weights differ by as much as the gradients do. Adam divides by the gradient's own
size, which turns rounding noise in a nearly zero entry into a difference the size of the step."""


def exported(trainer: Any) -> torch.Tensor:
    """Every exported weight of ``trainer`` in one vector."""
    return torch.cat([w.detach().flatten().float() for _, w in trainer.export_weights()])


def assert_same_update(initial: torch.Tensor, a: torch.Tensor, b: torch.Tensor, rel: float = 1e-4) -> None:
    """``a`` and ``b`` are ``initial`` after the same update: they moved, and they differ by at most
    ``rel`` of the largest change of a weight (the rounding of two orders of the same sums)."""
    moved = float((a - initial).abs().max())
    assert moved > 0
    torch.testing.assert_close(a, b, atol=rel * moved, rtol=0)


def make_batch(
    tokenizer: Any, *, target: int = 1234, batch_size: int = 4, response_len: int = 5
) -> TrainBatch:
    """Right-padded synthetic batch: chat prompts of different lengths, responses of ``target``
    repeated, advantage +1 on every response token."""
    from monorl.rollout.hf_engine import chat_prompt_ids

    prompts = [
        chat_prompt_ids(tokenizer, [{"role": "user", "content": f"Question number {i}: " + "pad " * (i % 3)}])
        for i in range(batch_size)
    ]
    responses = [[target] * (response_len - (i % 2)) + [tokenizer.eos_token_id] for i in range(batch_size)]
    width = max(len(p) + len(r) for p, r in zip(prompts, responses, strict=True))
    pad = tokenizer.pad_token_id
    input_ids = torch.full((batch_size, width), pad, dtype=torch.long)
    attention = torch.zeros((batch_size, width), dtype=torch.long)
    response_mask = torch.zeros((batch_size, width), dtype=torch.long)
    for i, (p, r) in enumerate(zip(prompts, responses, strict=True)):
        seq = p + r
        input_ids[i, : len(seq)] = torch.tensor(seq)
        attention[i, : len(seq)] = 1
        response_mask[i, len(p) : len(seq)] = 1
    advantages = response_mask.float()
    return TrainBatch(
        input_ids=input_ids,
        attention_mask=attention,
        response_mask=response_mask,
        advantages=advantages,
        rewards=torch.ones(batch_size),
        group_ids=[f"g{i // 2}" for i in range(batch_size)],
    )


@pytest.fixture(scope="session")
def tokenizer(tiny_model_dir: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tiny_model_dir)


@pytest.fixture(scope="session")
def tiny_vlm_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.rollout.tiny_vlm import build_tiny_vlm

    return build_tiny_vlm(tmp_path_factory.mktemp("tiny-qwen2vl"))


@pytest.fixture(scope="session")
def test_images(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    from tests.rollout.tiny_vlm import write_test_images

    return write_test_images(tmp_path_factory.mktemp("images"))
