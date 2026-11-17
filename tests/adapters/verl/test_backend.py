"""VerlBackend: registry wiring, command line, env, run/cancel/fail against a stand-in entry module."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from monorl.adapters.verl import backend as backend_module
from monorl.adapters.verl.backend import (
    VerlBackend,
    checkpoint_dirs,
    final_model_path,
    latest_step,
    link_checkpoints,
    resume_step,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable
from monorl.core.registry import registry
from monorl.core.reporting import RunReporter, iter_metrics, list_checkpoints, read_status

_FAKE_ENTRY = """
import json, os, signal, sys, time
from pathlib import Path

run_dir = Path(os.environ["MONORL_RUN_DIR"])
mode = os.environ.get("FAKE_ENTRY_MODE", "ok")
overrides = dict(arg.split("=", 1) for arg in sys.argv[1:])
# verl/trainer/ppo/ray_trainer.py _load_checkpoint: "disable" loads nothing; "auto" continues after
# the step of latest_checkpointed_iteration.txt when its global_step_<n> exists, else from scratch.
tracker = run_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
start = 0
if overrides["trainer.resume_mode"] == "auto" and tracker.exists():
    start = int(tracker.read_text())
    if not (run_dir / "checkpoints" / f"global_step_{start}").exists():
        start = 0
print("fake verl entry argv:", sys.argv[1:], flush=True)
print("RAY_ADDRESS=", os.environ.get("RAY_ADDRESS"), flush=True)
if mode == "fail":
    print("boom", file=sys.stderr, flush=True)
    sys.exit(3)
with open(run_dir / "metrics.jsonl", "a") as f:
    for step in (start + 1, start + 2):
        f.write(json.dumps({"step": step, "time": 0.0, "phase": "train", "reward/mean": 0.25 * step, "actor/lr": 1e-6}) + "\\n")
        f.flush()
        if mode == "hang":
            time.sleep(0.2)
ckpt = run_dir / "checkpoints" / f"global_step_{start + 2}" / "actor" / "huggingface"
ckpt.mkdir(parents=True, exist_ok=True)
tracker.write_text(str(start + 2))
if mode == "hang":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    time.sleep(60)
print("done", flush=True)
"""


@pytest.fixture
def fake_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    mod_dir = tmp_path / "fake_entry_pkg"
    mod_dir.mkdir()
    (mod_dir / "fake_verl_entry.py").write_text(_FAKE_ENTRY)
    monkeypatch.setattr(backend_module, "ENTRY_MODULE", "fake_verl_entry")
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(mod_dir) + (os.pathsep + existing if existing else ""))
    return mod_dir


def _backend(cfg) -> VerlBackend:
    backend = VerlBackend(cfg, python=sys.executable)
    backend.poll_seconds = 0.05
    backend.grace_seconds = 5.0
    return backend


def test_registry_and_protocol(make_config):
    factory = registry.get("pipeline_backend", "verl")
    backend = factory(make_config())
    assert isinstance(backend, VerlBackend)
    assert backend.name == "verl"
    assert backend.python == sys.executable


def test_command_and_env(make_config, tmp_path: Path):
    cfg = make_config(
        cluster={"ray_address": "auto"},
        trainer={"backend": "verl", "extra": {"verl_python": "/opt/verl/bin/python"}},
    )
    backend = VerlBackend(cfg)
    cmd = backend.build_command(["a=1", "b.c=x"])
    assert cmd == ["/opt/verl/bin/python", "-m", "monorl.adapters.verl.entry", "a=1", "b.c=x"]
    env = backend.build_env(cfg, tmp_path)
    assert env["RAY_ADDRESS"] == "auto"
    assert env["MONORL_RUN_DIR"] == str(tmp_path.resolve())
    assert env["PYTHONUNBUFFERED"] == "1"
    assert "RAY_ADDRESS" not in VerlBackend(make_config()).build_env(make_config(), tmp_path)


def test_prepare_stages_run_dir(make_config, tmp_path: Path):
    cfg = make_config(env={"type": "tool_calling", "params": {"tools": ["calculator"]}})
    run_dir = tmp_path / "run"
    overrides = VerlBackend(cfg).prepare(cfg, run_dir)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "data" / "train.parquet").exists()
    assert (run_dir / "monorl_reward.py").exists()
    assert (run_dir / "verl_tools.yaml").exists()
    assert (run_dir / "verl_config.yaml").exists()
    assert (run_dir / "verl_overrides.txt").read_text().splitlines() == overrides
    assert "actor_rollout_ref.rollout.multi_turn.enable=true" in overrides


def test_prepare_leaves_a_secret_of_a_reward_in_no_file_of_the_run_directory(
    base_config: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``api_key: ${OPENAI_API_KEY}`` in the params of ``llm_judge``: no file that ``prepare`` stages
    holds the value, and the reward module names the variable for verl's workers to resolve."""
    from monorl.config import loader

    monkeypatch.setattr(loader, "_secrets", {})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-verl-0123456789abcdef")
    params = {"model": "judge-model", "api_key": "${OPENAI_API_KEY}"}
    base_config["rewards"] = [{"type": "llm_judge", "params": params}]
    source = tmp_path / "user-config.yaml"
    source.write_text(yaml.safe_dump(base_config))
    cfg = RunConfig.from_yaml(source)
    assert cfg.rewards[0].params["api_key"] == "sk-verl-0123456789abcdef"

    run_dir = tmp_path / "run"
    VerlBackend(cfg).prepare(cfg, run_dir)
    files = sorted(p for p in run_dir.rglob("*") if p.is_file())
    assert [p.name for p in files] == [
        "config.yaml",
        "train.parquet",
        "monorl_reward.py",
        "verl_config.yaml",
        "verl_overrides.txt",
    ]
    holding = [p.name for p in files if b"sk-verl-0123456789abcdef" in p.read_bytes()]
    assert holding == []
    assert "'api_key': '${OPENAI_API_KEY}'" in (run_dir / "monorl_reward.py").read_text()
    assert RunConfig.from_yaml(run_dir / "config.yaml") == cfg


def test_run_requires_verl_unless_custom_python(make_config, tmp_path: Path):
    if backend_module.is_available():
        pytest.skip("verl importable")
    with pytest.raises(BackendUnavailable):
        VerlBackend(make_config()).run(
            make_config(), tmp_path / "run", reporter=RunReporter(tmp_path / "run", echo=False)
        )


def test_run_succeeds_and_collects_checkpoints(make_config, tmp_path: Path, fake_entry: Path):
    cfg = make_config()
    run_dir = tmp_path / "run"
    reporter = RunReporter(run_dir, echo=False)
    result = _backend(cfg).run(cfg, run_dir, reporter=reporter)
    assert result.state == "succeeded", result.error
    assert result.final_step == 2
    assert result.checkpoints == [str(run_dir / "checkpoints" / "global_step_2")]
    assert result.final_model_path == str(run_dir / "checkpoints" / "global_step_2" / "actor" / "huggingface")
    assert result.metrics["reward/mean"] == 0.5
    assert read_status(run_dir)["state"] == "succeeded"
    assert read_status(run_dir)["step"] == 2
    log = (run_dir / "run.log").read_text()
    assert "fake verl entry argv:" in log and "data.train_files=" in log and "done" in log
    assert list_checkpoints(run_dir) == [str(run_dir / "checkpoints" / "step-2")]
    assert (run_dir / "checkpoints" / "step-2").resolve() == (
        run_dir / "checkpoints" / "global_step_2"
    ).resolve()


def _steps(run_dir: Path) -> list[int]:
    return [row["step"] for _, row in iter_metrics(run_dir)]


def _stopped_after_step_3(make_config, run_dir: Path):
    """A verl run with a checkpoint of step 2 that was stopped after it wrote the row of step 3."""
    cfg = make_config()
    assert _backend(cfg).run(cfg, run_dir, reporter=RunReporter(run_dir, echo=False)).state == "succeeded"
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"step": 3, "time": 0.0, "phase": "train", "reward/mean": 0.9}) + "\n")
    return cfg


def test_resume_continues_from_the_checkpoint_and_drops_the_rows_after_it(
    make_config, tmp_path: Path, fake_entry: Path
):
    run_dir = tmp_path / "run"
    cfg = _stopped_after_step_3(make_config, run_dir)
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write('{"step": 4, "time": 0.0, "pha')  # killed in the middle of the row of step 4
    assert resume_step(run_dir) == 2

    result = _backend(cfg).run(cfg, run_dir, reporter=RunReporter(run_dir, echo=False), resume=True)
    assert result.state == "succeeded", result.error
    assert result.final_step == 4
    log = (run_dir / "run.log").read_text()
    assert "'trainer.resume_mode=auto'" in log.rsplit("fake verl entry argv:", 1)[1]
    assert "continues after step 2" in log
    rows = [row for _, row in iter_metrics(run_dir)]
    assert [row["step"] for row in rows] == [1, 2, 3, 4]  # step 3 once, and no unreadable line
    assert rows[2]["reward/mean"] == 0.75  # the row of this run, not the one of the killed run
    assert len((run_dir / "metrics.jsonl").read_text().splitlines()) == 4
    assert [Path(p).name for p in result.checkpoints] == ["global_step_2", "global_step_4"]


def test_run_without_resume_starts_over_in_a_used_directory(make_config, tmp_path: Path, fake_entry: Path):
    run_dir = tmp_path / "run"
    cfg = _stopped_after_step_3(make_config, run_dir)
    rows_before = _steps(run_dir)

    result = _backend(cfg).run(cfg, run_dir, reporter=RunReporter(run_dir, echo=False))
    assert result.state == "succeeded", result.error
    log = (run_dir / "run.log").read_text()
    assert "'trainer.resume_mode=disable'" in log.rsplit("fake verl entry argv:", 1)[1]
    assert result.final_step == 2  # trained steps 1 and 2 again instead of continuing at step 3
    assert _steps(run_dir)[len(rows_before) :] == [1, 2]


def test_resume_step_follows_verls_marker(tmp_path: Path):
    root = tmp_path / "checkpoints"
    (root / "global_step_4").mkdir(parents=True)
    assert resume_step(tmp_path) == 0  # verl resumes from the marker, not from the directories
    (root / "latest_checkpointed_iteration.txt").write_text("6")
    assert resume_step(tmp_path) == 0  # global_step_6 is not there: verl trains from scratch
    (root / "latest_checkpointed_iteration.txt").write_text("4")
    assert resume_step(tmp_path) == 4
    (root / "latest_checkpointed_iteration.txt").write_text("four")
    assert resume_step(tmp_path) is None


def test_run_failure_reports_exit_code(
    make_config, tmp_path: Path, fake_entry: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FAKE_ENTRY_MODE", "fail")
    cfg = make_config()
    run_dir = tmp_path / "run"
    result = _backend(cfg).run(cfg, run_dir, reporter=RunReporter(run_dir, echo=False))
    assert result.state == "failed"
    assert "exited with code 3" in result.error and "boom" in result.error
    assert result.final_model_path is None
    assert read_status(run_dir)["state"] == "failed"


def test_cancel_propagates_sigterm(
    make_config, tmp_path: Path, fake_entry: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("FAKE_ENTRY_MODE", "hang")
    cfg = make_config()
    run_dir = tmp_path / "run"
    backend = _backend(cfg)

    def cancel_when_running():
        while True:
            status = read_status(run_dir)
            if status and status.get("step", 0) >= 2:
                backend.cancel()
                return

    thread = threading.Thread(target=cancel_when_running, daemon=True)
    thread.start()
    result = backend.run(cfg, run_dir, reporter=RunReporter(run_dir, echo=False))
    thread.join(timeout=5)
    assert result.state == "cancelled"
    assert result.error is None
    assert result.final_step == 2
    assert read_status(run_dir)["state"] == "cancelled"
    assert "cancel requested" in (run_dir / "run.log").read_text()


def test_checkpoint_helpers(tmp_path: Path):
    root = tmp_path / "checkpoints"
    for n in (10, 2):
        (root / f"global_step_{n}").mkdir(parents=True)
    (root / "junk").mkdir()
    assert [s for s, _ in checkpoint_dirs(tmp_path)] == [2, 10]
    assert latest_step(tmp_path) == 10
    assert final_model_path(tmp_path) == str(root / "global_step_10")
    assert link_checkpoints(tmp_path) == [str(root / "global_step_2"), str(root / "global_step_10")]
    assert list_checkpoints(tmp_path) == [str(root / "step-2"), str(root / "step-10")]
    (root / "latest_checkpointed_iteration.txt").write_text("7")
    assert latest_step(tmp_path) == 7
    assert json.loads(json.dumps(link_checkpoints(tmp_path))) == link_checkpoints(tmp_path)  # idempotent
