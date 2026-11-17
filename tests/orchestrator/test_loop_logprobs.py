"""The loop asks the engine for sampling log-probs exactly when the batch uses them."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fakes
import pytest
import torch

from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics
from monorl.core.types import Completion, SamplingParams
from monorl.orchestrator import run


class FlagEngine(fakes.FakeEngine):
    """Fake engine that, like a served one, computes log-probs only when asked."""

    def __init__(self, config: Any, replica: int = 0, **kwargs: Any) -> None:
        self.requests: list[tuple[float, bool]] = []  # (temperature, logprobs requested) per call
        self.logprobs_shipped = 0
        super().__init__(config, replica, **kwargs)

    def generate(
        self, prompts: Sequence[Any], params: SamplingParams, *, max_new_tokens: Sequence[int] | None = None
    ) -> list[list[Completion]]:
        out = super().generate(prompts, params, max_new_tokens=max_new_tokens)
        with self._lock:
            self.requests.append((params.temperature, params.logprobs))
            for completion in (c for group in out for c in group):
                if params.logprobs:
                    self.logprobs_shipped += len(completion.logprobs)
                else:
                    completion.logprobs = None
        return out


@pytest.fixture
def flag_engine(fake_components: Any) -> None:
    registry.register("rollout_engine", "fake_flag", FlagEngine, overwrite=True)


def _train_rows(run_dir: str) -> list[dict[str, Any]]:
    rows = [row for _, row in iter_metrics(Path(run_dir)) if row["phase"] == "train"]
    return [
        {k: v for k, v in row.items() if not k.startswith("time") and k != "weight_sync/seconds"}
        for row in rows
    ]


def _rollout(backend: str, **extra: Any) -> dict[str, Any]:
    return {"backend": backend, "extra": {"max_parallel_episodes": 1, **extra}}


def test_engine_logprobs_are_requested_and_used_without_truncation(
    make_config: Any, flag_engine: None
) -> None:
    result = run(make_config(rollout=_rollout("fake_flag"), trainer={"max_steps": 2}))
    assert result.state == "succeeded"
    engine, trainer = fakes.ENGINES[-1], fakes.TRAINERS[-1]
    assert engine.requests and all(asked for _, asked in engine.requests)
    assert trainer.logprob_calls == []  # nothing recomputed: the engine's log-probs are the policy
    for batch in trainer.batches:
        first = batch.old_logprobs[0][batch.response_mask[0] == 1]
        assert first[:2].tolist() == pytest.approx([-0.1, -0.11])


@pytest.mark.parametrize(
    "rollout",
    [
        {"top_k": 20},
        {"top_p": 0.9},
        {"extra": {"min_p": 0.05}},
        {"extra": {"repetition_penalty": 1.1}},
        {"extra": {"trust_engine_logprobs": False}},
    ],
)
def test_a_truncating_sampler_does_not_ask_for_logprobs_and_trains_the_same_batch(
    make_config: Any, flag_engine: None, rollout: dict[str, Any]
) -> None:
    def config(backend: str) -> Any:
        merged = _rollout(backend, **rollout.get("extra", {}))
        merged.update({k: v for k, v in rollout.items() if k != "extra"})
        return make_config(
            rollout=merged, trainer={"max_steps": 2}, algorithm={"kl_coef": 0.1}, run_name=f"run-{backend}"
        )

    # the plain fake engine returns log-probs whatever is asked: what the loop used to receive
    shipped = run(config("fake"))
    shipped_batches = fakes.TRAINERS[-1].batches
    withheld = run(config("fake_flag"))
    engine, trainer = fakes.ENGINES[-1], fakes.TRAINERS[-1]
    assert shipped.state == withheld.state == "succeeded"

    assert engine.requests and not any(asked for _, asked in engine.requests)
    assert engine.logprobs_shipped == 0
    assert trainer.logprob_calls == [False, True] * 2  # the trainer scores the batch instead
    assert len(trainer.batches) == len(shipped_batches) == 2
    for batch, reference in zip(trainer.batches, shipped_batches, strict=True):
        assert torch.all(batch.old_logprobs[batch.response_mask == 1] == -0.25)
        for f in dataclasses.fields(batch):
            got, want = getattr(batch, f.name), getattr(reference, f.name)
            if isinstance(want, torch.Tensor):
                assert torch.equal(got, want), f.name
            else:
                assert got == want, f.name
    rows = _train_rows(withheld.run_dir)
    assert rows == _train_rows(shipped.run_dir)
    assert all(row["tokens/old_logprobs_recomputed"] == 1.0 for row in rows)


def test_trusted_engine_logprobs_are_requested_despite_truncation(
    make_config: Any, flag_engine: None
) -> None:
    rollout = _rollout("fake_flag", trust_engine_logprobs=True)
    result = run(make_config(rollout={**rollout, "top_k": 20}, trainer={"max_steps": 1}))
    assert result.state == "succeeded"
    assert all(asked for _, asked in fakes.ENGINES[-1].requests)
    batch = fakes.TRAINERS[-1].batches[0]
    assert batch.old_logprobs[0][batch.response_mask[0] == 1][0].item() == pytest.approx(-0.1)


def test_decoupled_anchor_keeps_the_engine_logprobs_as_behaviour(make_config: Any, flag_engine: None) -> None:
    result = run(
        make_config(
            rollout=_rollout("fake_flag"),
            trainer={"max_steps": 1},
            algorithm={"off_policy": {"anchor": "trainer", "importance_weights": "token"}},
        )
    )
    assert result.state == "succeeded"
    assert all(asked for _, asked in fakes.ENGINES[-1].requests)
    batch = fakes.TRAINERS[-1].batches[0]
    mask = batch.response_mask == 1
    assert torch.all(batch.old_logprobs[mask] == -0.25)
    assert batch.behavior_logprobs[mask].max().item() == pytest.approx(-0.1)


def test_evaluation_never_asks_for_logprobs(make_config: Any, flag_engine: None, tmp_path: Path) -> None:
    eval_file = fakes.write_jsonl(tmp_path / "eval.jsonl", [{"prompt": "xyz", "answer": "XYZ"}])
    result = run(
        make_config(
            rollout=_rollout("fake_flag"),
            data={"eval_uri": str(eval_file)},
            trainer={"max_steps": 1, "eval_every": 1},
        )
    )
    assert result.state == "succeeded"
    by_phase = {temperature == 0.0: asked for temperature, asked in fakes.ENGINES[-1].requests}
    assert by_phase == {False: True, True: False}  # sampled rollouts yes, greedy evaluation no
