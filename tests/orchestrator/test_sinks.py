from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from sink_support import FakeSinkPackages, install_fake_sink_packages, metric_points, tensorboard_points

from monorl.core.reporting import RunReporter
from monorl.orchestrator import run
from monorl.orchestrator.sinks import MetricSinks


@pytest.fixture
def packages(monkeypatch: pytest.MonkeyPatch) -> FakeSinkPackages:
    return install_fake_sink_packages(monkeypatch)


def _last(points: dict[tuple[str, int], list[float]]) -> dict[tuple[str, int], float]:
    return {key: values[-1] for key, values in points.items()}


def _shown(points: list[tuple[str, int, float]]) -> dict[tuple[str, int], float]:
    shown = {(tag, step): value for tag, step, value in points}
    assert len(shown) == len(points), "a name has two points at one step"
    return shown


def _once(points: dict[tuple[str, int], list[float]]) -> dict[tuple[str, int], float]:
    assert all(len(values) == 1 for values in points.values()), "a name has two points at one step"
    return _last(points)


def test_sinks_are_inactive_when_not_configured(make_config: Any, tmp_path: Path) -> None:
    reporter = RunReporter(tmp_path / "r", echo=False)
    sinks = MetricSinks(make_config(), reporter.run_dir, reporter)
    assert not sinks.active
    sinks.log(1, {"reward/mean": 1.0})
    sinks.close()


def test_wandb_sink_logs_numeric_metrics(
    make_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged: list[tuple[dict[str, float], int]] = []
    inits: list[dict[str, Any]] = []

    class FakeRun:
        def log(self, data: dict[str, float], step: int) -> None:
            logged.append((data, step))

        def finish(self) -> None:
            inits.append({"finished": True})

    fake_wandb = types.ModuleType("wandb")

    def init(**kwargs: Any) -> FakeRun:
        inits.append(kwargs)
        return FakeRun()

    fake_wandb.init = init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = make_config(logging={"wandb_project": "monorl-test"})
    reporter = RunReporter(tmp_path / "r", echo=False)
    sinks = MetricSinks(cfg, reporter.run_dir, reporter)
    assert sinks.active
    assert inits[0]["project"] == "monorl-test" and inits[0]["name"] == "fake-run"
    sinks.log(3, {"reward/mean": 0.5, "note": "text"})
    assert logged == [({"reward/mean": 0.5}, 3)]
    sinks.close()
    assert inits[-1] == {"finished": True}


def test_tensorboard_tags_are_the_metric_names_in_every_phase(
    make_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scalars: list[tuple[str, float, int]] = []

    class FakeWriter:
        def __init__(self, log_dir: str, purge_step: int | None = None) -> None:
            self.log_dir = log_dir

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            scalars.append((tag, value, step))

        def close(self) -> None:
            pass

    fake_tensorboard = types.ModuleType("torch.utils.tensorboard")
    fake_tensorboard.SummaryWriter = FakeWriter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_tensorboard)

    reporter = RunReporter(tmp_path / "r", echo=False)
    sinks = MetricSinks(make_config(logging={"tensorboard": True}), reporter.run_dir, reporter)
    assert sinks.active
    sinks.log(2, {"reward/mean": 0.5})
    # evaluation rows carry their eval/ prefix already: the same chart name as in metrics.jsonl and W&B
    sinks.log(2, {"eval/reward_mean": 0.25}, phase="eval")
    sinks.close()
    assert scalars == [("reward/mean", 0.5, 2), ("eval/reward_mean", 0.25, 2)]


def test_missing_optional_packages_are_logged_and_skipped(
    make_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb" or name.startswith("torch.utils.tensorboard"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    cfg = make_config(logging={"wandb_project": "p", "tensorboard": True})
    reporter = RunReporter(tmp_path / "r", echo=False)
    sinks = MetricSinks(cfg, reporter.run_dir, reporter)
    assert not sinks.active
    log = (reporter.run_dir / "run.log").read_text()
    assert "wandb" in log and "not installed" in log
    assert "tensorboard" in log


def _run_three_steps_then_resume_to_five(make_config: Any, logging: dict[str, Any]) -> Path:
    """A run that ends at step 3 with its last checkpoint at step 2, resumed up to step 5: step 3
    is run and logged a second time."""

    def config(max_steps: int) -> Any:
        return make_config(trainer={"max_steps": max_steps, "save_every": 2}, logging=logging)

    first = run(config(3))
    resumed = run(config(5), resume=True)
    assert (first.state, resumed.state) == ("succeeded", "succeeded") and resumed.final_step == 5
    run_dir = Path(resumed.run_dir)
    logged = metric_points(run_dir)
    assert [len(logged[("reward/mean", step)]) for step in (1, 2, 3, 4, 5)] == [1, 1, 2, 1, 1]
    return run_dir


def test_tensorboard_shows_every_step_once_after_a_resume(
    make_config: Any, packages: FakeSinkPackages
) -> None:
    run_dir = _run_three_steps_then_resume_to_five(make_config, {"tensorboard": True})
    # each (name, step) once, with the value of the run that logged it last
    assert _shown(packages.shown[str(run_dir / "tensorboard")]) == _last(metric_points(run_dir))


def test_real_tensorboard_shows_every_step_once_after_a_resume(make_config: Any) -> None:
    pytest.importorskip("tensorboard")
    run_dir = _run_three_steps_then_resume_to_five(make_config, {"tensorboard": True})
    assert len(list((run_dir / "tensorboard").iterdir())) == 2  # one event file per start of the run
    assert _once(tensorboard_points(run_dir / "tensorboard")) == pytest.approx(
        _last(metric_points(run_dir)), rel=1e-6
    )
