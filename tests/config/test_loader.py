"""Reading a config file and applying ``--set`` overrides: what is refused, and how it is said."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from monorl.config import loader
from monorl.config.loader import apply_overrides, load_yaml
from monorl.config.pipeline import PipelineConfig
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError


def test_a_key_set_twice_is_refused_with_both_lines(tmp_path: Path) -> None:
    """PyYAML keeps the last of two equal keys: a second ``trainer:`` block would silently drop
    every setting of the first one."""
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model: {name_or_path: x}\n"
        "trainer:\n"
        "  device: cpu\n"
        "  precision: fp32\n"
        "data: {uri: d.jsonl}\n"
        "trainer:\n"
        "  max_steps: 3\n"
    )
    with pytest.raises(ConfigError, match=r"'trainer' is set twice, on lines 2 and 6") as raised:
        load_yaml(path)
    assert str(path) in str(raised.value)

    nested = tmp_path / "nested.yaml"
    nested.write_text("trainer:\n  max_steps: 3\n  device: cpu\n  max_steps: 5\n")
    with pytest.raises(ConfigError, match=r"'max_steps' is set twice, on lines 2 and 4"):
        load_yaml(nested)


def test_the_same_key_in_different_mappings_and_a_merge_key_are_fine(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "defaults: &defaults {device: cpu, max_steps: 3}\n"
        "rollout: {device: cpu}\n"
        "trainer:\n"
        "  <<: *defaults\n"
        "  max_steps: 5\n"
        "rewards:\n"
        "  - {type: exact_match}\n"
        "  - {type: length}\n"
    )
    data = load_yaml(path)
    assert data["trainer"] == {"device": "cpu", "max_steps": 5}
    assert [r["type"] for r in data["rewards"]] == ["exact_match", "length"]


def test_a_path_that_is_no_readable_text_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=re.escape(f"cannot read config file {tmp_path}: ")):
        load_yaml(tmp_path)
    binary = tmp_path / "cfg.yaml"
    binary.write_bytes(b"\xff\xfe\x00model")
    with pytest.raises(ConfigError, match=re.escape(f"config file {binary} is not UTF-8 text")):
        load_yaml(binary)


@pytest.mark.parametrize("override", ["=3", "trainer..max_steps=3", ".trainer=3", "trainer.=3", "trainer"])
def test_an_override_without_a_key_or_with_an_empty_part_is_refused(override: str) -> None:
    with pytest.raises(ConfigError, match="override must look like key.path=value") as raised:
        apply_overrides({"trainer": {"max_steps": 1}}, [override])
    assert repr(override) in str(raised.value)


def test_an_override_into_a_list_says_to_replace_the_list() -> None:
    data = {"rewards": [{"type": "exact_match", "weight": 1.0}]}
    with pytest.raises(ConfigError, match=r"rewards is a list.*rewards=\[\.\.\.\]"):
        apply_overrides(data, ["rewards.0.weight=2"])
    assert data == {"rewards": [{"type": "exact_match", "weight": 1.0}]}, "the list is left as it was"
    replaced = apply_overrides(data, ["rewards=[{type: length, weight: 2}]", "model.lora.r=8"])
    assert replaced["rewards"] == [{"type": "length", "weight": 2}]
    assert replaced["model"] == {"lora": {"r": 8}}


# -- secrets read from the environment ------------------------------------------------------------

API_KEY = "sk-test-0123456789abcdef"


@pytest.fixture
def forget_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader remembers secrets for the life of the process: each test starts with none."""
    monkeypatch.setattr(loader, "_secrets", {})


def _config_file(tmp_path: Path, params: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model: {name_or_path: org/model}\n"
        "data: {uri: d.jsonl}\n"
        "rewards:\n"
        "  - type: exact_match\n"
        f"    params: {params}\n"
    )
    return path


def test_a_written_config_names_the_variable_of_a_secret_and_loads_back_to_its_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forget_secrets: None
) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", API_KEY)
    params = '{api_key: "${JUDGE_API_KEY}", url: "https://judge.example/v1?key=${JUDGE_API_KEY}&n=1"}'
    cfg = RunConfig.from_yaml(_config_file(tmp_path, params))
    assert cfg.rewards[0].params == {
        "api_key": API_KEY,
        "url": f"https://judge.example/v1?key={API_KEY}&n=1",
    }

    written = tmp_path / "written.yaml"
    text = cfg.to_yaml(written)
    assert API_KEY not in text and API_KEY not in written.read_text()
    assert yaml.safe_load(text)["rewards"][0]["params"] == {
        "api_key": "${JUDGE_API_KEY}",
        "url": "https://judge.example/v1?key=${JUDGE_API_KEY}&n=1",
    }
    assert RunConfig.from_yaml(written) == cfg

    monkeypatch.delenv("JUDGE_API_KEY")
    with pytest.raises(ConfigError, match=r"environment variable JUDGE_API_KEY is not set"):
        RunConfig.from_yaml(written)


def test_a_secret_that_came_from_the_default_is_not_written_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forget_secrets: None
) -> None:
    monkeypatch.delenv("JUDGE_TOKEN", raising=False)
    cfg = RunConfig.from_yaml(_config_file(tmp_path, '{token: "${JUDGE_TOKEN:-fallback-token-0123}"}'))
    assert cfg.rewards[0].params == {"token": "fallback-token-0123"}
    text = cfg.to_yaml()
    assert "fallback-token-0123" not in text
    assert yaml.safe_load(text)["rewards"][0]["params"] == {"token": "${JUDGE_TOKEN}"}


def test_a_pipeline_config_is_written_without_the_secret_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forget_secrets: None
) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", API_KEY)
    base = load_yaml(_config_file(tmp_path, '{api_key: "${JUDGE_API_KEY}"}'))
    stage = {"type": "train", "name": "rl", "overrides": {"env": {"params": {"password": API_KEY}}}}
    cfg = PipelineConfig.from_dict({"base": base, "stages": [stage]})
    written = tmp_path / "pipeline.yaml"
    assert API_KEY not in cfg.to_yaml(written)
    assert PipelineConfig.from_yaml(written) == cfg


def test_a_variable_that_is_not_named_like_a_secret_and_a_short_value_are_written_as_they_are(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forget_secrets: None
) -> None:
    monkeypatch.setenv("JUDGE_ENDPOINT", "https://judge.example/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "short")
    params = '{url: "${JUDGE_ENDPOINT}", api_key: "${JUDGE_API_KEY}", mode: "${JUDGE_KEY_MODE:-strict}"}'
    cfg = RunConfig.from_yaml(_config_file(tmp_path, params))
    written = yaml.safe_load(cfg.to_yaml())
    assert written["rewards"][0]["params"] == {
        "url": "https://judge.example/v1",
        "api_key": "short",
        "mode": "strict",
    }
