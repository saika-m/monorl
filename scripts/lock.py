"""Write ``requirements.lock``: every package of the environment that ``make install``, CI and the
CPU image install, at one exact version, with the hashes of its files.

    python scripts/lock.py                 # after a change of pyproject.toml or third_party/pins.toml
    python scripts/lock.py --upgrade       # move everything to the newest versions that are allowed
    python scripts/lock.py -P transformers # move one package (repeatable; ``name==version`` picks one)

What is locked:

* monorl's dependencies with the extras in :data:`EXTRAS`: everything the tests can use on a CPU.
  The GPU frameworks are not part of it: each of them needs an environment of its own
  (``[tool.uv] conflicts`` in pyproject.toml, the images under ``docker/``).
* What builds monorl itself (``[build-system] requires``). An install passes the file as
  ``--build-constraint`` too, so the build backend is the locked one.
* For every platform and for Python :data:`PYTHON` and later: one file, with environment markers
  where a platform or a Python version needs another package or version. An install takes wheels
  only (``--no-build``); every pin has them for macOS on Apple silicon, Linux and Windows.
* torch from PyTorch's CPU builds, which is what CI and the CPU image run. Installing the file takes
  the same setting: ``uv pip sync --torch-backend cpu requirements.lock``.
* Every distribution of ``third_party/pins.toml`` at the version pinned there, so that the release
  an adapter is written against is the one the tests run on.

Without ``--upgrade`` uv keeps what the file already pins wherever that still satisfies
pyproject.toml and the pins: an edit of either changes only what it has to.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = "requirements.lock"
PINS = "third_party/pins.toml"
EXTRAS = ("dev", "trl", "distributed", "multimodal", "rewards", "openenv", "hermes")
PYTHON = "3.11"
"""The oldest Python of ``requires-python``; later versions are covered by markers."""
CONSTRAINTS = ".lock-pins.txt"
BUILD = ".lock-build.txt"
"""Where the pins (as constraints) and the build requirements are handed to uv while it runs. uv
names both files in the lock's ``# via`` comments, so the names are fixed and relative."""


def pinned() -> list[str]:
    """``name==version`` for every distribution of ``third_party/pins.toml``."""
    pins = tomllib.loads((ROOT / PINS).read_text(encoding="utf-8"))
    return sorted(f"{pin['pypi']}=={pin['version']}" for pin in pins.values())


def build_requirements() -> list[str]:
    """What an install of monorl builds with: ``[build-system] requires`` of pyproject.toml, and
    ``editables``, which hatchling asks for on top when the install is editable (``-e .``)."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return [*pyproject["build-system"]["requires"], "editables>=0.3,<1"]


def command(*, upgrade: bool, upgrade_packages: list[str]) -> list[str]:
    cmd = [
        "uv",
        "pip",
        "compile",
        "pyproject.toml",
        BUILD,
        "--universal",
        "--python-version",
        PYTHON,
        "--torch-backend",
        "cpu",
        "--generate-hashes",
        "--constraint",
        CONSTRAINTS,
        "--custom-compile-command",
        "make lock",
        "--output-file",
        LOCK,
        "--quiet",
    ]
    for extra in EXTRAS:
        cmd += ["--extra", extra]
    if upgrade:
        cmd.append("--upgrade")
    for package in upgrade_packages:
        cmd += ["--upgrade-package", package]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Write requirements.lock (see the module docstring).")
    parser.add_argument(
        "--upgrade", action="store_true", help="move every package to the newest allowed version"
    )
    parser.add_argument(
        "-P",
        "--upgrade-package",
        action="append",
        default=[],
        metavar="NAME",
        help="move one package to the newest allowed version, or to NAME==VERSION (repeatable)",
    )
    args = parser.parse_args()
    inputs = {ROOT / CONSTRAINTS: pinned(), ROOT / BUILD: build_requirements()}
    try:
        for path, lines in inputs.items():
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        done = subprocess.run(
            command(upgrade=args.upgrade, upgrade_packages=args.upgrade_package), cwd=ROOT, check=False
        )
    finally:
        for path in inputs:
            path.unlink(missing_ok=True)
    if done.returncode == 0:
        lines = (ROOT / LOCK).read_text(encoding="utf-8").splitlines()
        print(f"{LOCK}: {sum(1 for line in lines if re.match(r'[A-Za-z0-9][^ ]*==', line))} pins")
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
