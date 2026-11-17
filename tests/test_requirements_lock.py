"""`requirements.lock` is a resolution of what `pyproject.toml` and `third_party/pins.toml` ask for.

`make lock` writes the lock (`scripts/lock.py`); `make install`, CI and the CPU image install from
it, and nothing resolves again on the way. A dependency added to pyproject.toml, a raised lower
bound or a bumped pin that never reached the lock would therefore go unnoticed until something
imports the missing package, or would show on one platform only. These tests compare the files with
each other, for every platform and Python version the lock is written for. They do not look at
what is installed where they run.
"""

from __future__ import annotations

import importlib.util
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
PINS: dict[str, dict[str, str]] = tomllib.loads(
    (REPO / "third_party" / "pins.toml").read_text(encoding="utf-8")
)


def _lock_script():
    spec = importlib.util.spec_from_file_location("monorl_lock_script", REPO / "scripts" / "lock.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _lock_script()


def _environment(python: str, sys_platform: str, machine: str) -> dict[str, str]:
    """The values environment markers are evaluated against on one platform."""
    return {
        "implementation_name": "cpython",
        "implementation_version": python,
        "os_name": "nt" if sys_platform == "win32" else "posix",
        "platform_machine": machine,
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": {"darwin": "Darwin", "linux": "Linux", "win32": "Windows"}[sys_platform],
        "platform_version": "",
        "python_full_version": python,
        "python_version": python.rsplit(".", 1)[0],
        "sys_platform": sys_platform,
    }


#: Where the lock is installed (macOS arm64 workstations, Linux x86 in CI and in the image), and
#: the other platforms a universal lock claims to cover, on both Python versions of requires-python.
ENVIRONMENTS = {
    f"py{python.rsplit('.', 1)[0]}-{sys_platform}-{machine}": _environment(python, sys_platform, machine)
    for python in ("3.11.9", "3.12.7")
    for sys_platform, machine in (
        ("darwin", "arm64"),
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "AMD64"),
    )
}


@dataclass(frozen=True)
class Locked:
    """One pin of the lock: ``name==version``, the marker it applies under, its file hashes."""

    requirement: Requirement
    hashes: tuple[str, ...]

    @property
    def name(self) -> str:
        return canonicalize_name(self.requirement.name)

    @property
    def version(self) -> Version:
        (specifier,) = self.requirement.specifier
        assert specifier.operator == "==" and "*" not in specifier.version, self.requirement
        return Version(specifier.version)

    def applies(self, environment: dict[str, str]) -> bool:
        return self.requirement.marker is None or self.requirement.marker.evaluate(environment)


def read_lock() -> list[Locked]:
    """Every pin of the lock. A line that is not a comment, a pin or a hash fails to parse."""
    pins: list[tuple[Requirement, list[str]]] = []
    for raw in (REPO / SCRIPT.LOCK).read_text(encoding="utf-8").splitlines():
        line = raw.strip().removesuffix("\\").strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash="):
            pins[-1][1].append(line.removeprefix("--hash="))
        else:
            pins.append((Requirement(line), []))
    return [Locked(requirement, tuple(hashes)) for requirement, hashes in pins]


LOCKED = read_lock()


def locked(name: str, environment: dict[str, str]) -> list[Locked]:
    return [pin for pin in LOCKED if pin.name == canonicalize_name(name) and pin.applies(environment)]


def asked() -> list[tuple[str, Requirement]]:
    """What the lock is a resolution of, as ``(where it is declared, requirement)``."""
    project = PYPROJECT["project"]
    declared = [("dependencies", text) for text in project["dependencies"]]
    for extra in SCRIPT.EXTRAS:
        declared += [(f"the {extra} extra", text) for text in project["optional-dependencies"][extra]]
    declared += [("the build of monorl", text) for text in SCRIPT.build_requirements()]
    return [(where, Requirement(text)) for where, text in declared]


def test_every_pin_is_exact_and_carries_hashes():
    """`uv pip sync --require-hashes` installs nothing it cannot check against a hash."""
    assert len(LOCKED) > 100
    for pin in LOCKED:
        assert pin.version  # one `==`, no wildcard
        assert pin.hashes and all(value.startswith("sha256:") for value in pin.hashes), pin.requirement


@pytest.mark.parametrize("environment", ENVIRONMENTS.values(), ids=ENVIRONMENTS.keys())
def test_one_version_of_a_package_per_environment(environment):
    names = Counter(pin.name for pin in LOCKED if pin.applies(environment))
    assert {name: count for name, count in names.items() if count > 1} == {}


@pytest.mark.parametrize("environment", ENVIRONMENTS.values(), ids=ENVIRONMENTS.keys())
def test_what_pyproject_asks_for_is_locked_at_a_version_it_accepts(environment):
    """Dependencies, the extras of the locked environment and the build backend, on each platform."""
    stale = []
    for where, requirement in asked():
        if requirement.marker is not None and not requirement.marker.evaluate(environment):
            continue
        pins = locked(requirement.name, environment)
        if not pins:
            stale.append(f"{requirement} ({where}): not in the lock")
        elif not requirement.specifier.contains(pins[0].version, prereleases=True):
            stale.append(f"{requirement} ({where}): the lock has {pins[0].version}")
    assert stale == [], "run `make lock`:\n  " + "\n  ".join(stale)


def test_pinned_distributions_are_locked_at_their_pin():
    """The release an adapter is written against is the one the locked environment runs.

    A pin belongs in the lock when one of the locked extras installs its distribution. The GPU
    frameworks are pinned too, and stay out of the lock with their extras.
    """
    extras = PYPROJECT["project"]["optional-dependencies"]
    installed_by = {
        canonicalize_name(Requirement(text).name) for extra in SCRIPT.EXTRAS for text in extras[extra]
    }
    expected = {
        canonicalize_name(pin["pypi"]): Version(pin["version"])
        for pin in PINS.values()
        if canonicalize_name(pin["pypi"]) in installed_by
    }
    assert set(expected) == {"trl", "openenv-core", "hermes-agent"}
    found = {pin.name: pin.version for pin in LOCKED if pin.name in expected}
    assert found == expected, "run `make lock` after editing third_party/pins.toml"
    outside = {canonicalize_name(pin["pypi"]) for pin in PINS.values()} - set(expected)
    assert not outside & {pin.name for pin in LOCKED}


def test_torch_is_the_cpu_build():
    """What `--torch-backend cpu` gives: off macOS a `+cpu` wheel, and no CUDA runtime packages.

    A lock compiled without that setting installs on Linux as well, with a few gigabytes of nvidia
    wheels that neither CI nor the CPU image can use.
    """
    for name, environment in ENVIRONMENTS.items():
        (torch,) = locked("torch", environment)
        expected = None if environment["sys_platform"] == "darwin" else "cpu"
        assert torch.version.local == expected, name
    assert [pin.name for pin in LOCKED if pin.name.startswith("nvidia-")] == []
