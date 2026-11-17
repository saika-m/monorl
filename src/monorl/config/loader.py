"""YAML loading with ``${VAR}`` / ``${VAR:-default}`` environment interpolation and dotted
``key.path=value`` overrides (CLI ``--set``)."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from monorl.core.errors import ConfigError

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate_env(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable {name} is not set (referenced as ${{{name}}})")

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v) for v in value]
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a mapping with the same key twice. PyYAML keeps the last
    one, so a second ``trainer:`` block further down would drop every setting of the first."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        first_line: dict[Any, int] = {}
        for key_node, _ in node.value:
            if not isinstance(key_node, yaml.ScalarNode) or key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node)
            line = key_node.start_mark.line + 1
            if key in first_line:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"the key {key!r} is set twice, on lines {first_line[key]} and {line}; "
                    "merge the two into one",
                    key_node.start_mark,
                )
            first_line[key] = line
        return super().construct_mapping(node, deep=deep)


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"cannot read config file {p}: {e.strerror or e}") from e
    except UnicodeDecodeError as e:
        raise ConfigError(f"config file {p} is not UTF-8 text: {e}") from e
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {p}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"top level of {p} must be a mapping")
    return interpolate_env(data)


def _coerce(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """``apply_overrides(cfg, ["trainer.max_steps=10", "rewards=[{type: exact_match}]"])``."""
    if not overrides:
        return data
    out = dict(data)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        key, _, raw = item.partition("=")
        parts = key.strip().split(".")
        node: dict[str, Any] = out
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _coerce(raw)
    return out
