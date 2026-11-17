"""The sampling stages of a pipeline (``rejection_sample``, ``eval``) send the rows of their
``metrics.jsonl`` to the metric sinks of ``logging.wandb_project`` and ``logging.tensorboard``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sink_support import FakeSinkPackages, install_fake_sink_packages, metric_points

from monorl.pipeline import run_pipeline

BOTH = {"wandb_project": "monorl-test", "tensorboard": True}


@pytest.fixture
def packages(monkeypatch: pytest.MonkeyPatch) -> FakeSinkPackages:
    return install_fake_sink_packages(monkeypatch)


def _logged(run_dir: Path) -> dict[tuple[str, int], float]:
    points = metric_points(run_dir)
    assert all(len(values) == 1 for values in points.values())
    return {key: values[0] for key, values in points.items()}


def _in_wandb(packages: FakeSinkPackages, run_dir: Path) -> dict[tuple[str, int], float]:
    run = packages.run_named(run_dir.name)  # a stage's run is named like its directory
    return {(name, step): value for step, row in run.history.items() for name, value in row.items()}


def _in_tensorboard(packages: FakeSinkPackages, run_dir: Path) -> dict[tuple[str, int], float]:
    points = packages.shown[str(run_dir / "tensorboard")]
    shown = {(tag, step): value for tag, step, value in points}
    assert len(shown) == len(points)
    return shown


def test_rejection_and_eval_stages_log_their_metrics_to_the_sinks(
    make_pipeline: Any, packages: FakeSinkPackages
) -> None:
    stages = [
        {"type": "rejection_sample", "name": "rs", "num_samples": 2, "min_reward": 1.0, "max_per_prompt": 1},
        {"type": "eval", "name": "eval", "max_prompts": 3},
    ]
    result = run_pipeline(make_pipeline(stages, logging=BOTH))
    assert result.state == "succeeded", result.error

    rs, evaluation = (Path(stage["run_dir"]) for stage in result.stages)
    logged = _logged(rs)
    assert {("rs/accepted", 1), ("reward/mean", 1), ("rs/rows", max(step for _, step in logged))} <= set(
        logged
    )
    assert _in_wandb(packages, rs) == logged and _in_tensorboard(packages, rs) == logged

    logged = _logged(evaluation)  # 3 prompts in chunks of 2: a row per chunk, then the totals
    assert {("eval/chunk_reward_mean", 1), ("eval/chunk_reward_mean", 2), ("eval/reward_mean", 2)} <= set(
        logged
    )
    assert _in_wandb(packages, evaluation) == logged and _in_tensorboard(packages, evaluation) == logged

    assert [run.exit_code for run in packages.runs] == [0, 0]
    assert len(packages.writers) == 2 and all(writer.closed for writer in packages.writers)


def test_a_failed_sampling_stage_closes_its_sinks(make_pipeline: Any, packages: FakeSinkPackages) -> None:
    stages = [{"type": "rejection_sample", "name": "rs", "num_samples": 1, "min_reward": 5.0}]
    result = run_pipeline(make_pipeline(stages, logging=BOTH))
    assert result.state == "failed" and "accepted 0" in result.stages[0]["error"]

    run_dir = Path(result.stages[0]["run_dir"])
    logged = _logged(run_dir)  # the chunks that were sampled before the stage gave up
    assert ("rs/accept_rate", 1) in logged
    assert _in_wandb(packages, run_dir) == logged and _in_tensorboard(packages, run_dir) == logged
    assert packages.run_named(run_dir.name).exit_code == 1
    assert all(writer.closed for writer in packages.writers)
