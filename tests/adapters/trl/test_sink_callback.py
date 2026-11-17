"""``ReporterCallback`` sends what it writes to ``metrics.jsonl`` to the metric sinks the run's
config asks for (``trainer.backend: trl`` compiles ``report_to: none``, so TRL itself logs to
neither)."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from transformers import TrainerControl, TrainerState
from trl_support import VowelReward

from monorl.adapters.trl.callbacks import ReporterCallback
from monorl.config.schema import RunConfig
from monorl.core.reporting import CONFIG_FILE, RunReporter
from monorl.orchestrator import run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "orchestrator"))

from sink_support import FakeSinkPackages, install_fake_sink_packages, metric_points  # noqa: E402

# What a GRPOTrainer logs for a step, for an evaluation and when it ends (keys of TRL 1.17).
TRAIN_LOGS = {"loss": 0.5, "reward": 0.25, "learning_rate": 5e-6, "epoch": 0.5}
EVAL_LOGS = {"eval_loss": 0.75, "eval_reward": 0.5, "eval_runtime": 1.5, "epoch": 0.5}
CLOSING_LOGS = {"train_runtime": 3.0, "train_loss": 0.5, "epoch": 1.0}


@pytest.fixture
def packages(monkeypatch: pytest.MonkeyPatch) -> FakeSinkPackages:
    return install_fake_sink_packages(monkeypatch)


@pytest.fixture
def callback(tmp_path: Path) -> ReporterCallback:
    """The callback of a run whose directory holds its config, as ``monorl.orchestrator.run``
    leaves it before it starts the backend."""
    config = RunConfig.from_dict(
        {
            "run_name": "trl-sinks",
            "model": {"name_or_path": str(tmp_path / "model")},
            "data": {"uri": str(tmp_path / "data.jsonl")},
            "trainer": {"backend": "trl"},
            "rewards": [{"type": "exact_match"}],
            "logging": {"run_root": str(tmp_path), "wandb_project": "monorl-test", "tensorboard": True},
        }
    )
    reporter = RunReporter(tmp_path / "run", echo=False)
    config.to_yaml(reporter.run_dir / CONFIG_FILE)
    return ReporterCallback(reporter)


def _train(callback: ReporterCallback, *, steps: int) -> TrainerState:
    state, control = TrainerState(max_steps=2), TrainerControl()
    callback.on_train_begin(None, state, control)
    for step in range(1, steps + 1):
        state.global_step = step
        callback.on_log(None, state, control, logs=dict(TRAIN_LOGS))
        callback.on_log(None, state, control, logs=dict(EVAL_LOGS))
    return state


def test_every_metrics_row_reaches_the_sinks_under_its_name(
    callback: ReporterCallback, packages: FakeSinkPackages
) -> None:
    state = _train(callback, steps=2)
    callback.on_log(None, state, TrainerControl(), logs=dict(CLOSING_LOGS))
    callback.on_train_end(None, state, TrainerControl())

    run_dir = callback.reporter.run_dir
    logged = metric_points(run_dir)
    assert {("loss/policy", 1), ("eval/reward/mean", 2), ("trl/train_loss", 2)} <= set(logged)
    # the evaluation and the closing row repeat the epoch of their step
    assert logged[("train/epoch", 2)] == [0.5, 0.5, 1.0]

    (wandb_run,) = packages.runs
    assert wandb_run.init["name"] == "trl-sinks" and wandb_run.exit_code == 0
    in_wandb = {(name, step): value for step, row in wandb_run.history.items() for name, value in row.items()}
    assert in_wandb == {key: values[-1] for key, values in logged.items()}  # a row per step: the last value

    in_tensorboard: dict[tuple[str, int], list[float]] = defaultdict(list)
    for tag, step, value in packages.shown[str(run_dir / "tensorboard")]:
        in_tensorboard[(tag, step)].append(value)
    assert in_tensorboard == logged
    assert all(writer.closed for writer in packages.writers)


def test_close_ends_the_sinks_of_a_run_that_did_not_reach_its_end(
    callback: ReporterCallback, packages: FakeSinkPackages
) -> None:
    _train(callback, steps=1)  # Trainer.train raised: no on_train_end
    (wandb_run,) = packages.runs
    assert wandb_run.exit_code is None
    callback.close()
    assert wandb_run.exit_code == 0 and list(wandb_run.history) == [1]
    assert all(writer.closed for writer in packages.writers)


# -- real GRPOTrainer runs on the tiny Llama, through monorl.orchestrator.run ----------------


def _points(history: dict[int, dict[str, float]]) -> dict[tuple[str, int], float]:
    return {(name, step): value for step, row in history.items() for name, value in row.items()}


@pytest.mark.slow
def test_a_trl_run_logs_every_metrics_row_to_the_sinks(
    trl_config: RunConfig, packages: FakeSinkPackages
) -> None:
    config = trl_config.model_copy(deep=True)
    config.logging.wandb_project, config.logging.tensorboard = "monorl-test", True
    result = run(config)
    assert result.state == "succeeded", result.error

    run_dir = Path(result.run_dir)
    logged = metric_points(run_dir)
    assert {("loss/policy", 1), ("reward/mean", 2), ("trl/train_loss", 2)} <= set(logged)
    (wandb_run,) = packages.runs
    assert wandb_run.init["name"] == "trl-test" and wandb_run.init["config"] == config.to_dict()
    assert _points(wandb_run.history) == {key: values[-1] for key, values in logged.items()}
    assert wandb_run.exit_code == 0
    shown = packages.shown[str(run_dir / "tensorboard")]
    assert sorted((tag, step) for tag, step, _ in shown) == sorted(
        key for key, values in logged.items() for _ in values
    )
    assert all(writer.closed for writer in packages.writers)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="needs callback.close() in the KeyboardInterrupt and Exception handlers of TrlBackend._train",
)
def test_a_failed_trl_run_closes_the_sinks_and_ends_the_wandb_run_as_failed(
    trl_config: RunConfig, packages: FakeSinkPackages, monkeypatch: pytest.MonkeyPatch
) -> None:
    score = VowelReward.__call__
    calls = []

    def fails_at_the_second_step(self: VowelReward, trajectories: Any) -> Any:
        calls.append(len(trajectories))
        if len(calls) == 2:
            raise RuntimeError("reward failed at step 2")
        return score(self, trajectories)

    monkeypatch.setattr(VowelReward, "__call__", fails_at_the_second_step)
    config = trl_config.model_copy(deep=True)
    config.logging.wandb_project, config.logging.tensorboard = "monorl-test", True
    result = run(config)
    assert result.state == "failed" and "reward failed at step 2" in (result.error or "")

    (wandb_run,) = packages.runs
    assert list(wandb_run.history) == [1]
    assert wandb_run.exit_code == 1
    assert all(writer.closed for writer in packages.writers)
