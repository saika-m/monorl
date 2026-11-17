"""Dynamic sampling rolls out only as many resample prompts as it has groups to replace."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fakes
import pytest
import torch

from monorl.core.registry import registry
from monorl.core.reporting import iter_metrics
from monorl.core.types import RewardResult, Trajectory
from monorl.orchestrator import run
from monorl.orchestrator.sampler import PromptSampler

PROMPTS_PER_STEP = 8
GROUP_SIZE = 2
NUM_SAMPLES = 40


class ParityReward:
    """Scores the members of a group 0, 1, 0, ... in the order they arrive, except the groups of
    the samples in ``uniform``, whose members all score 1 (no gradient: dynamic sampling drops them)."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.uniform = {str(i) for i in params.get("uniform", [])}

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        seen: dict[str, int] = {}
        out = []
        for t in trajectories:
            k = seen.get(t.group_id, 0)
            seen[t.group_id] = k + 1
            out.append(RewardResult(score=1.0 if t.sample.id in self.uniform else float(k % 2)))
        return out


class BrokenSampleEnv(fakes.FakeMultiTurnEnv):
    """Every episode of the samples in ``env.params.broken`` fails at its first step."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        self.broken = {str(i) for i in config.env.params.get("broken", ())}
        self.sample_id = ""

    def reset(self, sample: Any) -> list[Any]:
        self.sample_id = sample.id
        return super().reset(sample)

    def step(self, action: Any) -> Any:
        if self.sample_id in self.broken:
            raise RuntimeError("tool backend unreachable")
        return super().step(action)


@pytest.fixture
def make_dapo(make_config: Any, tmp_path: Path) -> Any:
    registry.register("reward", "test_parity", lambda params, **kw: ParityReward(params), overwrite=True)
    data = fakes.write_jsonl(
        tmp_path / "many.jsonl", [{"prompt": f"q{i:02d}", "answer": f"Q{i:02d}"} for i in range(NUM_SAMPLES)]
    )

    def _make(uniform: Sequence[int], env: dict[str, Any] | None = None, **overrides: Any) -> Any:
        return make_config(
            data={"uri": str(data)},
            env=env or {"type": "single_turn"},
            algorithm={"dynamic_sampling": True},
            rewards=[{"type": "test_parity", "params": {"uniform": list(uniform)}}],
            rollout={"group_size": GROUP_SIZE, "extra": {"fake_mode": "always_right"}},
            trainer={"prompts_per_step": PROMPTS_PER_STEP, "max_steps": 1, "save_every": 100, **overrides},
            seed=11,
        )

    return _make


def _resample_draw(step: int, round: int) -> list[int]:
    samples = [fakes.Sample(id=str(i), messages=[]) for i in range(NUM_SAMPLES)]
    sampler = PromptSampler(samples, prompts_per_step=PROMPTS_PER_STEP, seed=11)
    return [int(s.id) for s in sampler.batch(step, round=round)]


def _row(result: Any) -> dict[str, Any]:
    return [row for _, row in iter_metrics(Path(result.run_dir)) if row["phase"] == "train"][0]


def _groups(batch: Any) -> list[tuple[str, str]]:
    """``(group id, sample id)`` of every group of a batch, in batch order."""
    pairs = list(zip(batch.group_ids, batch.meta["sample_ids"], strict=True))
    return list(dict.fromkeys(pairs))


def _engine_prompts() -> int:
    return sum(call["num_prompts"] * call["n"] for call in fakes.ENGINES[-1].calls)


def test_without_a_dropped_group_the_step_is_its_own_prompts(make_dapo: Any) -> None:
    result = run(make_dapo(uniform=[]))
    assert result.state == "succeeded"
    row = _row(result)
    assert row["rollout/groups_dropped"] == 0.0 and row["rollout/resample_rounds"] == 0.0
    assert _groups(fakes.TRAINERS[-1].batches[0]) == [(f"s1-p{p}", str(p)) for p in range(PROMPTS_PER_STEP)]
    assert _engine_prompts() == PROMPTS_PER_STEP * GROUP_SIZE


def test_one_dropped_group_costs_one_resampled_prompt_not_a_whole_step(make_dapo: Any) -> None:
    draw = _resample_draw(step=1, round=1)
    assert draw[0] != 3
    result = run(make_dapo(uniform=[3]))
    assert result.state == "succeeded"
    row = _row(result)
    assert row["rollout/groups_dropped"] == 1.0 and row["rollout/resample_rounds"] == 1.0
    assert row["rollout/num_sequences"] == PROMPTS_PER_STEP * GROUP_SIZE
    batch = fakes.TRAINERS[-1].batches[0]
    kept = [(f"s1-p{p}", str(p)) for p in range(PROMPTS_PER_STEP) if p != 3]
    assert _groups(batch) == [*kept, ("s1-r1-p0", str(draw[0]))]  # the first prompt of the round's draw
    assert batch.rewards.tolist() == [0.0, 1.0] * PROMPTS_PER_STEP
    # 8 prompts of the step and 1 resampled, not 8 + 8
    assert _engine_prompts() == (PROMPTS_PER_STEP + 1) * GROUP_SIZE


def test_a_round_keeps_drawing_until_the_step_is_whole(make_dapo: Any) -> None:
    """The replacement is the first prompt of the round's draw whose group has a gradient, as when
    the whole draw was rolled out: a dropped candidate widens the next slice of the same draw."""
    draw = _resample_draw(step=1, round=1)
    assert not set(draw[:4]) & set(range(PROMPTS_PER_STEP))  # none of the step's own prompts
    result = run(make_dapo(uniform=[3, *draw[:3]]))
    assert result.state == "succeeded"
    row = _row(result)
    assert row["rollout/resample_rounds"] == 1.0
    assert row["rollout/groups_dropped"] == 1.0 + 3.0  # the step's own group and three candidates
    groups = _groups(fakes.TRAINERS[-1].batches[0])
    assert len(groups) == PROMPTS_PER_STEP
    assert groups[-1] == ("s1-r1-p3", str(draw[3]))
    # slices of 1, 2 and 4 candidates of the 8 drawn: the last candidate is never rolled out
    assert _engine_prompts() == (PROMPTS_PER_STEP + 1 + 2 + 4) * GROUP_SIZE


def test_a_draw_without_a_usable_group_leaves_the_step_short(make_dapo: Any) -> None:
    uniform = {3, *_resample_draw(1, 1), *_resample_draw(1, 2)}
    own_dropped = len(uniform & set(range(PROMPTS_PER_STEP)))
    config = make_dapo(uniform=sorted(uniform))
    config.rollout.extra["dynamic_sampling_max_rounds"] = 2
    result = run(config)
    row = _row(result)
    assert row["rollout/resample_rounds"] == 2.0
    assert row["rollout/num_sequences"] == (PROMPTS_PER_STEP - own_dropped) * GROUP_SIZE
    # every candidate of both draws was tried, as before, and counted as dropped
    assert row["rollout/groups_dropped"] == own_dropped + 2 * PROMPTS_PER_STEP
    assert _engine_prompts() == 3 * PROMPTS_PER_STEP * GROUP_SIZE


def test_failed_episodes_of_a_resample_slice_count_against_the_whole_step(make_dapo: Any) -> None:
    """Both episodes of the one resampled prompt fail: 2 of the 18 episodes of the step so far, not
    all of a rollout, so ``env.max_failed_frac`` tolerates them and the next slice fills the step."""
    registry.register("env", "broken_sample", BrokenSampleEnv, overwrite=True)
    draw = _resample_draw(step=1, round=1)
    env = {
        "type": "broken_sample",
        "max_turns": 1,
        "max_failed_frac": 0.2,
        "params": {"broken": [draw[0]], "done_after": 1},
    }
    result = run(make_dapo(uniform=[3], env=env))
    assert result.state == "succeeded", result.error
    row = _row(result)
    assert _groups(fakes.TRAINERS[-1].batches[0])[-1] == ("s1-r1-p1", str(draw[1]))
    assert row["rollout/failed_episodes"] == 2.0
    assert row["rollout/failed_frac"] == pytest.approx(2 / ((PROMPTS_PER_STEP + 1 + 2) * GROUP_SIZE))

    # the same two failures are too many for a threshold the step as a whole does not meet
    env["max_failed_frac"] = 0.1
    config = make_dapo(uniform=[3], env=env)
    config.run_name = "strict"
    result = run(config)
    assert result.state == "failed"
    assert "2 of 2 episodes failed (2 of 18 in the step so far)" in (result.error or "")


def test_two_runs_with_the_same_seed_are_identical(make_dapo: Any) -> None:
    uniform = [1, 3, 9, 12, 17, 20, 21, 22, *_resample_draw(2, 1)[:2], _resample_draw(3, 1)[0]]
    runs = []
    for name in ("first", "second"):
        config = make_dapo(uniform=uniform, max_steps=3)
        config.run_name = name
        result = run(config)
        assert result.state == "succeeded"
        rows = [
            {k: v for k, v in row.items() if not k.startswith("time") and k != "weight_sync/seconds"}
            for _, row in iter_metrics(Path(result.run_dir))
        ]
        batches = fakes.TRAINERS[-1].batches
        runs.append(
            (rows, [(_groups(b), b.rewards.tolist(), b.input_ids) for b in batches], _engine_prompts())
        )
    first, second = runs
    assert first[0] == second[0] and first[2] == second[2]
    assert sum(row["rollout/groups_dropped"] for row in first[0]) >= 3  # the runs did resample
    for (groups_a, rewards_a, ids_a), (groups_b, rewards_b, ids_b) in zip(first[1], second[1], strict=True):
        assert len(groups_a) == PROMPTS_PER_STEP
        assert groups_a == groups_b and rewards_a == rewards_b and torch.equal(ids_a, ids_b)
