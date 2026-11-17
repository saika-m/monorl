"""``MetricSinks``: what W&B and TensorBoard receive next to ``metrics.jsonl``. The tests with
``packages`` run against fakes of the two packages (``sink_support.py``); the tests at the end run
against the real ones where they are installed."""

from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import fakes
import numpy as np
import pytest
import torch
import yaml
from sink_support import (
    FakeSinkPackages,
    FakeWandbRun,
    install_fake_sink_packages,
    metric_points,
    read_offline_wandb_runs,
    tensorboard_points,
)

from monorl.config import loader
from monorl.config.schema import RunConfig
from monorl.core.reporting import RunReporter
from monorl.orchestrator import run
from monorl.orchestrator.sinks import MetricSinks

BOTH = {"wandb_project": "monorl-test", "tensorboard": True}


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


def test_every_number_of_a_metrics_row_reaches_both_sinks_under_its_name(
    make_config: Any, tmp_path: Path, packages: FakeSinkPackages
) -> None:
    cfg = make_config(logging=BOTH)
    reporter = RunReporter(tmp_path / "r", echo=False)
    sinks = MetricSinks(cfg, reporter.run_dir, reporter)
    assert sinks.active
    train = {
        "reward/mean": 0.5,
        "loss/policy": np.float32(0.25),  # metrics.jsonl records these three as numbers too
        "grad_norm": torch.tensor(2.0),
        "tokens/total": np.int64(3),
        "loss/kl": float("nan"),
        "note": "text",
        "nothing": None,
    }
    evaluation = {"eval/reward_mean": 0.25}
    reporter.log_metrics(3, train)
    sinks.log(3, train)
    reporter.log_metrics(3, evaluation, phase="eval")
    sinks.log(3, evaluation, phase="eval")
    sinks.close()

    expected = _last(metric_points(reporter.run_dir))
    assert set(expected) == {(name, 3) for name in (*train, *evaluation)} - {("note", 3), ("nothing", 3)}
    (wandb_run,) = packages.runs
    assert wandb_run.init["project"] == "monorl-test" and wandb_run.init["name"] == "fake-run"
    assert wandb_run.init["config"] == cfg.to_dict() and wandb_run.init["dir"] == str(reporter.run_dir)
    assert {(name, 3): value for name, value in wandb_run.history[3].items()} == pytest.approx(
        expected, nan_ok=True
    )
    assert list(wandb_run.history) == [3] and wandb_run.exit_code == 0
    assert _shown(packages.shown[str(reporter.run_dir / "tensorboard")]) == pytest.approx(
        expected, nan_ok=True
    )
    assert all(writer.closed for writer in packages.writers)
    log = (reporter.run_dir / "run.log").read_text()
    assert f"wandb: new run {wandb_run.id} in project monorl-test" in log
    assert f"tensorboard: events in {reporter.run_dir / 'tensorboard'}" in log


def test_wandb_gets_the_config_with_the_variable_of_a_secret_not_its_value(
    tmp_path: Path, fake_components: Any, packages: FakeSinkPackages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``api_key: ${JUDGE_API_KEY}`` in a config file: the W&B server stores the config of a run,
    so it is given the reference the file has. TensorBoard is given no config."""
    monkeypatch.setattr(loader, "_secrets", {})
    monkeypatch.setenv("JUDGE_API_KEY", "sk-sink-0123456789abcdef")
    monkeypatch.setenv("JUDGE_ENDPOINT", "https://judge.example/v1")
    params = {"api_key": "${JUDGE_API_KEY}", "url": "${JUDGE_ENDPOINT}", "auth": "Bearer ${JUDGE_API_KEY}"}
    data = fakes.fake_config_dict(tmp_path, rewards=[{"type": "test_exact", "params": params}], logging=BOTH)
    path = tmp_path / "user-config.yaml"
    path.write_text(yaml.safe_dump(data))
    cfg = RunConfig.from_yaml(path)
    assert cfg.rewards[0].params["auth"] == "Bearer sk-sink-0123456789abcdef"

    reporter = RunReporter(tmp_path / "r", echo=False)
    sinks = MetricSinks(cfg, reporter.run_dir, reporter)
    sinks.log(1, {"reward/mean": 0.5})
    sinks.close()

    (wandb_run,) = packages.runs
    assert "sk-sink-0123456789abcdef" not in repr(wandb_run.init)
    assert wandb_run.init["config"]["rewards"][0]["params"] == {
        "api_key": "${JUDGE_API_KEY}",
        "url": "https://judge.example/v1",
        "auth": "Bearer ${JUDGE_API_KEY}",
    }
    without_rewards = {k: v for k, v in wandb_run.init["config"].items() if k != "rewards"}
    assert without_rewards == {k: v for k, v in cfg.to_dict().items() if k != "rewards"}
    (writer,) = packages.writers
    assert vars(writer) == {"log_dir": str(reporter.run_dir / "tensorboard"), "closed": True}


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


def test_a_wandb_directory_on_the_path_is_not_taken_for_the_package(
    make_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if importlib.util.find_spec("wandb") is not None:
        pytest.skip("wandb is installed: a directory of that name does not shadow it")
    # wandb writes ./wandb by default; without the package Python imports that directory as an
    # empty namespace package, so `import wandb` succeeds.
    (tmp_path / "cwd" / "wandb").mkdir(parents=True)
    monkeypatch.syspath_prepend(str(tmp_path / "cwd"))
    reporter = RunReporter(tmp_path / "r", echo=False)
    try:
        sinks = MetricSinks(make_config(logging={"wandb_project": "p"}), reporter.run_dir, reporter)
    finally:
        sys.modules.pop("wandb", None)
    assert not sinks.active
    assert "`wandb` is not installed; skipping wandb" in (reporter.run_dir / "run.log").read_text()


def _run_three_steps_then_resume_to_five(make_config: Any, logging: dict[str, Any]) -> Path:
    """A run that ends at step 3 with its last checkpoint at step 2, resumed up to step 5: step 3
    is run a second time and the sinks are given it twice. ``metrics.jsonl`` has it once, since a
    resume takes the file back to the checkpoint."""

    def config(max_steps: int) -> Any:
        return make_config(trainer={"max_steps": max_steps, "save_every": 2}, logging=logging)

    first = run(config(3))
    resumed = run(config(5), resume=True)
    assert (first.state, resumed.state) == ("succeeded", "succeeded") and resumed.final_step == 5
    run_dir = Path(resumed.run_dir)
    logged = metric_points(run_dir)
    assert [len(logged[("reward/mean", step)]) for step in (1, 2, 3, 4, 5)] == [1, 1, 1, 1, 1]
    return run_dir


def test_tensorboard_shows_every_step_once_after_a_resume(
    make_config: Any, packages: FakeSinkPackages
) -> None:
    run_dir = _run_three_steps_then_resume_to_five(make_config, {"tensorboard": True})
    # each (name, step) once, with the value of the run that logged it last
    assert _shown(packages.shown[str(run_dir / "tensorboard")]) == _last(metric_points(run_dir))


def test_a_resumed_run_logs_which_wandb_run_it_writes_to(
    make_config: Any, packages: FakeSinkPackages
) -> None:
    run_dir = _run_three_steps_then_resume_to_five(make_config, {"wandb_project": "p"})
    first, resumed = packages.runs
    assert list(first.history) == [1, 2, 3] and list(resumed.history) == [3, 4, 5]
    log = (run_dir / "run.log").read_text()
    assert f"wandb: new run {first.id} in project p" in log
    assert f"wandb: new run {resumed.id} in project p" in log


def test_a_failed_run_ends_its_wandb_run_as_failed(make_config: Any, packages: FakeSinkPackages) -> None:
    failed = run(make_config(trainer={"max_steps": 3, "extra": {"fake_fail_at": 2}}, logging=BOTH))
    assert failed.state == "failed"
    (wandb_run,) = packages.runs
    assert list(wandb_run.history) == [1] and wandb_run.exit_code == 1
    assert all(writer.closed for writer in packages.writers)

    succeeded = run(make_config(run_name="fine", logging=BOTH))
    assert succeeded.state == "succeeded" and packages.run_named("fine").exit_code == 0


@pytest.mark.parametrize("broken", ["tensorboard", "wandb"])
def test_a_sink_that_fails_while_logging_is_dropped_and_the_run_goes_on(
    make_config: Any, packages: FakeSinkPackages, monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    def no_space(*args: Any, **kwargs: Any) -> None:
        raise OSError(28, "No space left on device")

    if broken == "tensorboard":
        monkeypatch.setattr(packages.SummaryWriter, "add_scalar", no_space)
    else:
        monkeypatch.setattr(FakeWandbRun, "log", no_space)
    result = run(make_config(trainer={"max_steps": 3}, logging=BOTH))

    assert result.state == "succeeded" and result.final_step == 3
    run_dir = Path(result.run_dir)
    lines = [line for line in (run_dir / "run.log").read_text().splitlines() if "logging failed" in line]
    assert len(lines) == 1
    assert f"{broken} logging failed (OSError: [Errno 28] No space left on device)" in lines[0]
    (wandb_run,) = packages.runs
    if broken == "tensorboard":  # the other sink has every step
        assert list(wandb_run.history) == [1, 2, 3] and wandb_run.exit_code == 0
        assert all(writer.closed for writer in packages.writers)
    else:
        assert wandb_run.exit_code == 1
        assert _shown(packages.shown[str(run_dir / "tensorboard")]) == _last(metric_points(run_dir))


# -- against the real packages --------------------------------------------------------------


def _config_with_evaluation(make_config: Any, tmp_path: Path, **overrides: Any) -> Any:
    eval_file = fakes.write_jsonl(tmp_path / "eval.jsonl", [{"prompt": "xyz", "answer": "XYZ"}])
    return make_config(data={"eval_uri": str(eval_file)}, **overrides)


def test_real_tensorboard_events_hold_every_metric_once(make_config: Any, tmp_path: Path) -> None:
    pytest.importorskip("tensorboard")
    cfg = _config_with_evaluation(
        make_config, tmp_path, trainer={"max_steps": 3, "eval_every": 2}, logging={"tensorboard": True}
    )
    run_dir = Path(run(cfg).run_dir)
    expected = _once(metric_points(run_dir))
    assert ("eval/reward_mean", 2) in expected
    # TensorBoard stores float32
    assert _once(tensorboard_points(run_dir / "tensorboard")) == pytest.approx(expected, rel=1e-6)


def test_real_tensorboard_shows_every_step_once_after_a_resume(make_config: Any) -> None:
    pytest.importorskip("tensorboard")
    run_dir = _run_three_steps_then_resume_to_five(make_config, {"tensorboard": True})
    assert len(list((run_dir / "tensorboard").iterdir())) == 2  # one event file per start of the run
    assert _once(tensorboard_points(run_dir / "tensorboard")) == pytest.approx(
        _last(metric_points(run_dir)), rel=1e-6
    )


@pytest.fixture
def offline_wandb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The real package, offline, with every directory it writes to below ``tmp_path``."""
    wandb = pytest.importorskip("wandb")
    for name in [name for name in os.environ if name.startswith("WANDB_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    for name in ("WANDB_DIR", "WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        monkeypatch.setenv(name, str(tmp_path / "wandb-home" / name.lower()))
    yield
    wandb.teardown()  # stops the service process that wandb.init started


def test_real_offline_wandb_run_holds_the_config_and_every_metric(
    make_config: Any, tmp_path: Path, offline_wandb: None
) -> None:
    cfg = _config_with_evaluation(
        make_config, tmp_path, trainer={"max_steps": 3, "eval_every": 2}, logging={"wandb_project": "p"}
    )
    run_dir = Path(run(cfg).run_dir)
    (offline,) = read_offline_wandb_runs(run_dir)
    assert (offline["project"], offline["name"], offline["exit_code"]) == ("p", "fake-run", 0)
    assert offline["config"] == yaml.safe_load((run_dir / "config.yaml").read_text())
    expected = metric_points(run_dir)
    assert ("eval/reward_mean", 2) in expected
    assert offline["history"] == pytest.approx(_last(expected))
    assert f"wandb: new run {offline['id']} in project p" in (run_dir / "run.log").read_text()


def test_real_offline_wandb_run_of_a_failed_run_ends_with_exit_code_1(
    make_config: Any, offline_wandb: None
) -> None:
    failed = run(
        make_config(trainer={"max_steps": 3, "extra": {"fake_fail_at": 2}}, logging={"wandb_project": "p"})
    )
    assert failed.state == "failed"
    (offline,) = read_offline_wandb_runs(Path(failed.run_dir))
    assert offline["exit_code"] == 1
    assert {step for _, step in offline["history"]} == {1}
