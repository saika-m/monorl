"""``MONORL_HOME`` moves everything monorl keeps for itself, the default run root included."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from monorl.config.pipeline import PipelineConfig
from monorl.config.schema import RunConfig
from monorl.core.home import monorl_home, run_root
from monorl.orchestrator.run import resolve_run_dir
from monorl.pipeline.runner import resolve_pipeline_run_dir
from monorl.server.jobs import JobManager


def _config(**logging: Any) -> RunConfig:
    return RunConfig.from_dict(
        {
            "run_name": "r",
            "model": {"name_or_path": "m"},
            "data": {"uri": "d.jsonl"},
            "rewards": [{"type": "exact_match"}],
            "logging": logging,
        }
    )


def test_the_default_run_root_is_under_monorl_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "elsewhere"
    monkeypatch.setenv("MONORL_HOME", str(home))
    config = _config()
    pipeline = PipelineConfig.from_dict(
        {"name": "p", "base": config.model_dump(), "stages": [{"name": "s", "type": "train"}]}
    )
    # the CLI's run, the CLI's pipeline and the server's jobs agree on where a run goes
    assert resolve_run_dir(config) == home / "runs" / "r"
    assert resolve_pipeline_run_dir(pipeline) == ("p", home / "runs" / "p")
    jobs = JobManager()
    assert jobs.home == home
    assert jobs._resolve_run_dir(config) == ("r", home / "runs" / "r")
    assert jobs._resolve_pipeline_dir(pipeline) == ("p", home / "runs" / "p")


def test_a_run_root_that_was_set_is_taken_as_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONORL_HOME", str(tmp_path / "elsewhere"))
    assert resolve_run_dir(_config(run_root=str(tmp_path / "mine"))) == tmp_path / "mine" / "r"
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    assert run_root("~/runs") == tmp_path / "user" / "runs"


def test_without_monorl_home_everything_is_under_the_users_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MONORL_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    assert monorl_home() == tmp_path / "user" / ".monorl"
    assert resolve_run_dir(_config()) == tmp_path / "user" / ".monorl" / "runs" / "r"
