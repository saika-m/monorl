"""Reading a config file and applying ``--set`` overrides: what is refused, and how it is said."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from monorl.config.loader import load_yaml
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
