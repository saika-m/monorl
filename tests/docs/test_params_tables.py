"""The parameter table of every built-in reward and environment type lists the keys the type
accepts, no more and no fewer: a key that is documented is not refused, and a key that is
accepted is documented."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from monorl.adapters import PROVIDERS
from monorl.core.params import declared_params
from monorl.core.registry import registry

ROOT = Path(__file__).resolve().parents[2]

#: namespace, type -> the document, and the heading of the section that holds the table.
TABLES: dict[tuple[str, str], tuple[str, str]] = {
    **{
        ("reward", name): ("docs/REWARDS.md", f"### `{name}`")
        for name in (
            "exact_match",
            "math_verify",
            "format",
            "length",
            "code_exec",
            "llm_judge",
            "reward_model",
            "composite",
        )
    },
    ("env", "single_turn"): ("docs/ENVIRONMENTS.md", "## `single_turn`"),
    ("env", "tool_calling"): ("docs/ENVIRONMENTS.md", "## `tool_calling`"),
    ("env", "remote"): ("docs/ENVIRONMENTS.md", "## `remote`"),
    ("env", "openenv"): ("docs/ENVIRONMENTS.md", "### `openenv`"),
    ("env", "hermes"): ("docs/ENVIRONMENTS.md", "### `hermes`"),
    ("env", "harness"): ("docs/GATEWAY.md", "## 3. `env.type: harness`"),
}
HEADERS = ("Parameter", "`env.params` key", "Param")


def documented_params(document: str, heading: str) -> set[str]:
    """The names in backticks in the first column of the parameter tables of a section: from its
    heading to the next heading of the same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    names: set[str] = set()
    inside = in_table = False
    for line in (ROOT / document).read_text().splitlines():
        if line.startswith("#"):
            if inside and len(line) - len(line.lstrip("#")) <= level:
                break
            inside = inside or line.strip() == heading
            continue
        if not inside:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if not cells:
            in_table = False
        elif cells[0] in HEADERS:
            in_table = True
        elif in_table:
            names.update(re.findall(r"`([a-z_]+)`", cells[0]))
    return names


def test_every_built_in_type_has_a_table() -> None:
    built_in = {("reward", name) for name in registry.names("reward")}
    built_in |= {("env", name) for name in registry.names("env")}
    built_in |= {("env", name) for name in PROVIDERS if declared_params("env", name) is not None}
    declaring = {key for key in built_in if declared_params(*key) is not None}
    assert declaring == set(TABLES)


@pytest.mark.parametrize("key", sorted(TABLES), ids=lambda key: f"{key[0]}:{key[1]}")
def test_the_documented_parameters_are_the_accepted_ones(key: tuple[str, str]) -> None:
    document, heading = TABLES[key]
    documented = documented_params(document, heading)
    accepted = set(declared_params(*key) or ())
    assert documented - accepted == set(), f"{document} documents keys that {key[1]} refuses"
    assert accepted - documented == set(), f"{key[1]} accepts keys that {document} does not document"
