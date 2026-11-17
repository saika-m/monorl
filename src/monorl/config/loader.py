"""YAML loading with ``${VAR}`` / ``${VAR:-default}`` environment interpolation and dotted
``key.path=value`` overrides (CLI ``--set``).

A value read from a variable that is named like a secret is remembered for the life of the process,
and :func:`mask_secrets` puts ``${VAR}`` back wherever that value occurs: everything that writes a
config to a file or sends it somewhere passes it through that function."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from monorl.core.errors import ConfigError

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_SECRET_NAME_RE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
_MIN_SECRET_CHARS = 8
"""A shorter value is not remembered: replacing every ``1`` or ``true`` of a config would mangle it."""
_secrets: dict[str, str] = {}
"""Value to the name of the variable it was read from, longest value first."""


def _remember_secret(name: str, value: str) -> None:
    global _secrets
    if len(value) < _MIN_SECRET_CHARS or not _SECRET_NAME_RE.search(name) or value in _secrets:
        return
    # Longest first, so that a value that contains another one is replaced as a whole. Built aside
    # and swapped in: another thread may be iterating the mapping in mask_secrets.
    _secrets = dict(sorted({**_secrets, value: name}.items(), key=lambda item: -len(item[0])))


def mask_secrets(value: Any) -> Any:
    """``value`` with ``${NAME}`` in place of every value that :func:`interpolate_env` read from a
    variable named like a secret (``NAME`` contains KEY, TOKEN, SECRET, PASSWORD, PASSWD or
    CREDENTIAL, and the value has at least 8 characters), wherever it occurs inside a string.
    Mappings and lists are copied, their keys and items masked; anything else is returned as it is.

    The in-memory config keeps the real values. What is masked loads back to them through
    :func:`interpolate_env` where the variable is set."""
    secrets = _secrets
    if not secrets:
        return value
    if isinstance(value, str):
        for secret, name in secrets.items():
            if secret in value:
                value = value.replace(secret, f"${{{name}}}")
        return value
    if isinstance(value, dict):
        return {mask_secrets(k): mask_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [mask_secrets(v) for v in value]
    return value


def interpolate_env(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                resolved = os.environ[name]
            elif default is not None:
                resolved = default
            else:
                raise ConfigError(f"environment variable {name} is not set (referenced as ${{{name}}})")
            _remember_secret(name, resolved)
            return resolved

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
    """``apply_overrides(cfg, ["trainer.max_steps=10", "rewards=[{type: exact_match}]"])``.

    The dotted key names mappings only: one that leads into a list is a :class:`ConfigError`
    that says to replace the list, as is a key with an empty part (``=3``, ``trainer..x=1``)."""
    if not overrides:
        return data
    out = dict(data)
    for item in overrides:
        key, eq, raw = item.partition("=")
        parts = key.strip().split(".")
        if not eq or not all(parts):
            raise ConfigError(f"override must look like key.path=value, got {item!r}")
        node: dict[str, Any] = out
        for depth, part in enumerate(parts[:-1], 1):
            nxt = node.get(part)
            if isinstance(nxt, list):
                where = ".".join(parts[:depth])
                raise ConfigError(
                    f"override {item!r}: {where} is a list, and an override cannot address the items "
                    f"of a list; replace the whole list, {where}=[...]"
                )
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _coerce(raw)
    return out
