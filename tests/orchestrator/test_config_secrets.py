"""A secret that a config reads from the environment (``api_key: ${JUDGE_API_KEY}``) reaches the
components and nothing else: no file of the run directory and no sink holds its value."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fakes
import pytest
import yaml
from sink_support import FakeSinkPackages, install_fake_sink_packages

from monorl.config import loader
from monorl.config.schema import RunConfig
from monorl.orchestrator import run

API_KEY = "sk-run-0123456789abcdef"


@pytest.fixture
def packages(monkeypatch: pytest.MonkeyPatch) -> FakeSinkPackages:
    return install_fake_sink_packages(monkeypatch)


@pytest.fixture
def load_config(tmp_path: Path, fake_components: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The fake config as a user has it: a file whose reward reads its key from the environment."""
    monkeypatch.setattr(loader, "_secrets", {})
    monkeypatch.setenv("JUDGE_API_KEY", API_KEY)

    def _load(reward: str) -> RunConfig:
        data = fakes.fake_config_dict(
            tmp_path,
            rewards=[{"type": reward, "params": {"api_key": "${JUDGE_API_KEY}"}}],
            logging={"wandb_project": "monorl-test", "tensorboard": True},
        )
        path = tmp_path / "user-config.yaml"
        path.write_text(yaml.safe_dump(data))
        return RunConfig.from_yaml(path)

    return _load


def _files_holding(run_dir: Path, value: str) -> list[str]:
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    assert files
    return sorted(str(p.relative_to(run_dir)) for p in files if value.encode() in p.read_bytes())


def test_no_file_of_the_run_directory_and_no_sink_holds_a_secret_of_the_config(
    load_config: Any, packages: FakeSinkPackages
) -> None:
    cfg = load_config("test_exact")
    assert cfg.rewards[0].params == {"api_key": API_KEY}  # what the reward is built with
    result = run(cfg)
    assert result.state == "succeeded"
    run_dir = Path(result.run_dir)

    assert _files_holding(run_dir, API_KEY) == []
    written = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert written["rewards"][0]["params"] == {"api_key": "${JUDGE_API_KEY}"}
    assert RunConfig.from_yaml(run_dir / "config.yaml") == cfg  # what a resume of the directory loads
    (wandb_run,) = packages.runs
    assert API_KEY not in repr(wandb_run.init)
    assert wandb_run.init["config"]["rewards"][0]["params"] == {"api_key": "${JUDGE_API_KEY}"}
    assert wandb_run.init["config"]["trainer"] == cfg.to_dict()["trainer"]
