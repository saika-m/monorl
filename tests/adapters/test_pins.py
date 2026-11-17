"""`third_party/pins.toml`, the adapters' version gates and the extras of `pyproject.toml` agree.

A pin names the upstream release an adapter is written against. Three other places restate it: the
version constants of the adapter package, the `versions` specifier of each `Patch` (what the patch
engine checks against the installed distribution before applying), and the extra that installs the
distribution. A pin bumped without its gate makes the adapter skip its patches on exactly the
release it was ported to, so every restatement is checked against the pin here.

Constants and patch specifiers are read from the adapter sources with `ast`: importing the vllm
and sglang adapters imports torch, and the frameworks themselves are not installed where the tests
run.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import tomllib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

from monorl.adapters import PROVIDERS

REPO = Path(__file__).resolve().parents[2]
ADAPTERS = REPO / "src" / "monorl" / "adapters"
PINS: dict[str, dict[str, str]] = tomllib.loads((REPO / "third_party" / "pins.toml").read_text())
EXTRAS: dict[str, list[str]] = tomllib.loads((REPO / "pyproject.toml").read_text())["project"][
    "optional-dependencies"
]

#: Module-level names that restate a pin: `VLLM_VERSIONS`, `VERSIONS`, `PINNED_VERSION`, `VERL_TAG`, ...
PIN_CONSTANT = re.compile(r"(?:^|_)(?:VERSIONS?|TAG|COMMIT)$")


@dataclass(frozen=True)
class PinConstant:
    """A module-level string constant of an adapter package and how it relates to the pin.

    ``check`` is ``"range"`` (a specifier set that must contain the pinned version), ``"version"``
    (must equal the pinned version) or ``"tag"`` / ``"commit"`` (must equal that field of the pin).
    """

    pin: str
    file: str  # relative to src/monorl/adapters
    name: str
    check: str

    def __str__(self) -> str:
        return f"{self.file}:{self.name}"


PIN_CONSTANTS = [
    PinConstant("vllm", "vllm/common.py", "VLLM_VERSIONS", "range"),
    PinConstant("sglang", "sglang/common.py", "SGLANG_VERSIONS", "range"),
    PinConstant("verl", "verl/compiler.py", "VERL_VERSIONS", "range"),
    PinConstant("verl", "verl/compiler.py", "VERL_TAG", "tag"),
    PinConstant("verl", "verl/compiler.py", "VERL_COMMIT", "commit"),
    PinConstant("verl", "verl/patches.py", "VERSIONS", "range"),
    PinConstant("nemo_rl", "nemo_rl/__init__.py", "TARGET_TAG", "tag"),
    PinConstant("nemo_rl", "nemo_rl/__init__.py", "TARGET_COMMIT", "commit"),
    PinConstant("nemo_rl", "nemo_rl/patches.py", "VERSIONS", "range"),
    PinConstant("unsloth", "unsloth/__init__.py", "PINNED_VERSION", "version"),
    PinConstant("unsloth", "unsloth/patches.py", "VERSIONS", "range"),
]

#: Pins whose adapter checks nothing about the installed version: no constant above and no `Patch`.
UNGATED = {"openenv", "hermes-agent"}

#: The extra of `pyproject.toml` that installs each pinned distribution; `None` where there is none
#: (NeMo-RL is not on PyPI and is installed from git at the tag).
EXTRA_OF_PIN: dict[str, str | None] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "verl": "verl",
    "trl": "trl",
    "nemo_rl": None,
    "unsloth": "unsloth",
    "openenv": "openenv",
    "hermes-agent": "hermes",
}


def adapter_dir(pin: str) -> Path:
    return ADAPTERS / PINS[pin]["adapter"].removeprefix("monorl.adapters.")


@cache
def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def string_constants(path: Path) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` assignments of one source file."""
    found: dict[str, str] = {}
    for node in parse(path).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant) and isinstance(value.value, str):
            found[target.id] = value.value
    return found


def resolve_string(node: ast.expr, path: Path) -> str:
    """The string an expression of ``path`` denotes: a literal, or a constant of the adapter package."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        for source in [path, *sorted(path.parent.glob("*.py"))]:
            constants = string_constants(source)
            if node.id in constants:
                return constants[node.id]
    raise AssertionError(f"{path.relative_to(REPO)}:{node.lineno}: cannot resolve {ast.unparse(node)!r}")


def calls(pin: str, callee: str) -> list[tuple[Path, ast.Call]]:
    """Every ``callee(...)`` call in the adapter package of ``pin``."""
    return [
        (path, node)
        for path in sorted(adapter_dir(pin).rglob("*.py"))
        for node in ast.walk(parse(path))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee
    ]


def keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def patch_specifiers(pin: str) -> list[tuple[str, str]]:
    """``(location, versions)`` of every ``Patch(...)`` in the adapter package; ``""`` when not gated."""
    found = []
    for path, call in calls(pin, "Patch"):
        versions = keyword(call, "versions")
        spec = "" if versions is None else resolve_string(versions, path)
        found.append((f"{path.relative_to(REPO)}:{call.lineno}", spec))
    return found


def accepts(spec: str, version: str) -> bool:
    return Version(version) in SpecifierSet(spec, prereleases=True)


def gate_specifiers(pin: str) -> list[tuple[str, str]]:
    """``(location, specifier)`` of every version range the adapter of ``pin`` enforces or documents."""
    constants = [
        (str(c), string_constants(ADAPTERS / c.file)[c.name])
        for c in PIN_CONSTANTS
        if c.pin == pin and c.check == "range"
    ]
    return constants + [(where, spec) for where, spec in patch_specifiers(pin) if spec]


def bounds(spec: str | SpecifierSet) -> tuple[Version | None, Version | None]:
    """Inclusive lower and exclusive upper bound of a ``>=A,<B`` range; ``None`` for an open end."""
    lower: list[Version] = []
    upper: list[Version] = []
    for clause in SpecifierSet(str(spec)):
        assert clause.operator in (">=", "<"), f"{spec}: only >= and < clauses can be compared, got {clause}"
        (lower if clause.operator == ">=" else upper).append(Version(clause.version))
    return max(lower, default=None), min(upper, default=None)


def extra_requirement(pin: str) -> Requirement:
    """The requirement on the pinned distribution in the extra that installs it."""
    requirements = [Requirement(r) for r in EXTRAS[EXTRA_OF_PIN[pin]]]
    [requirement] = [r for r in requirements if canonicalize_name(r.name) == PINS[pin]["pypi"]]
    return requirement


def test_pins_and_providers_name_the_same_adapters() -> None:
    assert sorted(pin["adapter"] for pin in PINS.values()) == sorted(PROVIDERS.values())


@pytest.mark.parametrize("pin", PINS)
def test_pin_names_an_adapter_and_a_distribution(pin: str) -> None:
    entry = PINS[pin]
    assert importlib.util.find_spec(entry["adapter"]) is not None
    assert entry["pypi"] == canonicalize_name(entry["pypi"])
    Version(entry["version"])
    # The patch engine reads the installed version of `distribution` (the framework name when it is
    # omitted) to evaluate the gate: it has to be the distribution the pin is about.
    for path, call in calls(pin, "PatchSet"):
        named = keyword(call, "distribution") or keyword(call, "framework")
        assert named is not None, f"{path}:{call.lineno}"
        assert canonicalize_name(resolve_string(named, path)) == entry["pypi"]


def test_pinned_distributions_are_distinct() -> None:
    names = [pin["pypi"] for pin in PINS.values()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("constant", PIN_CONSTANTS, ids=str)
def test_pin_satisfies_adapter_constant(constant: PinConstant) -> None:
    value = string_constants(ADAPTERS / constant.file)[constant.name]
    entry = PINS[constant.pin]
    if constant.check == "range":
        assert accepts(value, entry["version"]), f"{constant} = {value!r} rejects pin {entry['version']}"
    elif constant.check == "version":
        assert Version(value) == Version(entry["version"])
    else:
        assert value == entry[constant.check]


@pytest.mark.parametrize("pin", PINS)
def test_pin_is_accepted_by_every_patch(pin: str) -> None:
    version = PINS[pin]["version"]
    rejecting = [(where, spec) for where, spec in patch_specifiers(pin) if not accepts(spec, version)]
    assert not rejecting, f"patches that would be skipped on the pinned {pin} {version}: {rejecting}"


def test_every_patch_is_version_gated() -> None:
    ungated = [where for pin in PINS for where, spec in patch_specifiers(pin) if not spec]
    assert not ungated


def test_every_pin_constant_of_the_adapters_is_listed() -> None:
    defined = {
        (path.relative_to(ADAPTERS).as_posix(), name)
        for path in ADAPTERS.rglob("*.py")
        for name in string_constants(path)
        if PIN_CONSTANT.search(name)
    }
    assert defined == {(c.file, c.name) for c in PIN_CONSTANTS}
    assert all(Path(c.file).parts[0] == adapter_dir(c.pin).name for c in PIN_CONSTANTS)


def test_ungated_adapters_are_exactly_the_listed_ones() -> None:
    gated = {c.pin for c in PIN_CONSTANTS if c.check in ("range", "version")}
    gated |= {pin for pin in PINS if patch_specifiers(pin)}
    assert set(PINS) - gated == UNGATED


def test_every_extra_of_a_pinned_distribution_is_listed() -> None:
    pin_of_distribution = {entry["pypi"]: pin for pin, entry in PINS.items()}
    installs = {
        (pin_of_distribution[name], extra)
        for extra, requirements in EXTRAS.items()
        for name in (canonicalize_name(Requirement(r).name) for r in requirements)
        if name in pin_of_distribution
    }
    assert set(EXTRA_OF_PIN) == set(PINS)
    assert installs == {(pin, extra) for pin, extra in EXTRA_OF_PIN.items() if extra is not None}


@pytest.mark.parametrize("pin", [pin for pin, extra in EXTRA_OF_PIN.items() if extra is not None])
def test_pin_satisfies_extra(pin: str) -> None:
    requirement = extra_requirement(pin)
    assert requirement.specifier.contains(PINS[pin]["version"], prereleases=True), str(requirement)


@pytest.mark.parametrize("pin", [pin for pin, extra in EXTRA_OF_PIN.items() if extra is not None])
def test_extra_allows_nothing_the_gate_refuses(pin: str) -> None:
    requirement = extra_requirement(pin)
    extra_lower, extra_upper = bounds(requirement.specifier)
    for where, spec in gate_specifiers(pin):
        gate_lower, gate_upper = bounds(spec)
        message = f"extra {requirement} installs releases that {where} ({spec}) refuses"
        if gate_lower is not None:
            assert extra_lower is not None and extra_lower >= gate_lower, message
        if gate_upper is not None:
            assert extra_upper is not None and extra_upper <= gate_upper, message
