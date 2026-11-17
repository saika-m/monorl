"""Recipe discovery: ``configs/recipes/*.yaml`` (or ``*.yml``) with a leading ``# `` description block.

The one definition of what a recipe is, for ``monorl recipes``, the server's ``GET /api/recipes``
and the tests that validate every shipped recipe. Torch-free.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monorl.config.loader import load_yaml
from monorl.config.pipeline import is_pipeline_payload
from monorl.core.errors import ConfigError

RECIPES_DIR_ENV = "MONORL_RECIPES_DIR"
RECIPE_SUFFIXES = (".yaml", ".yml")


def recipes_dir() -> Path:
    """``$MONORL_RECIPES_DIR`` or ``<repo>/configs/recipes`` next to the source tree."""
    override = os.environ.get(RECIPES_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[3] / "configs" / "recipes"


@dataclass(slots=True)
class Recipe:
    name: str
    path: Path
    description: str

    def load(self) -> dict[str, Any]:
        return load_yaml(self.path)

    @property
    def kind(self) -> str:
        """``"pipeline"`` for ``base:`` + ``stages:`` files, else ``"run"``."""
        try:
            return "pipeline" if is_pipeline_payload(self.load()) else "run"
        except ConfigError:
            return "run"


def read_description(path: Path) -> str:
    """The leading ``# ...`` comment block of a recipe, joined with spaces. Blank lines before it
    and a ``#!`` line are skipped; it ends at the first line that is not a comment."""
    lines: list[str] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("#!") or (not line and not lines):
                continue
            if not line.startswith("#"):
                break
            text = line.lstrip("#").strip()
            if text:
                lines.append(text)
    return " ".join(lines)


def recipe_files(directory: Path) -> list[Path]:
    """The recipe files of ``directory``: ``*.yaml`` and ``*.yml``, sorted by recipe name (the stem).

    A missing directory has no recipes. Two files with the same stem (``a.yaml`` next to ``a.yml``)
    would be one recipe name with two contents, so that is a :class:`ConfigError` naming both.
    """
    if not directory.is_dir():
        return []
    files = sorted(
        (p for p in directory.iterdir() if p.suffix in RECIPE_SUFFIXES and p.is_file()),
        key=lambda p: (p.stem, p.name),
    )
    for first, second in zip(files, files[1:], strict=False):
        if first.stem == second.stem:
            raise ConfigError(
                f"recipe {first.stem!r} is defined twice in {directory}: {first.name} and {second.name}; "
                "delete or rename one of them"
            )
    return files


def list_recipes(directory: Path | None = None) -> list[Recipe]:
    root = directory or recipes_dir()
    return [Recipe(name=p.stem, path=p, description=read_description(p)) for p in recipe_files(root)]


def get_recipe(name: str, directory: Path | None = None) -> Recipe:
    """The recipe called ``name``: its stem, or its file name (``name.yaml`` or ``name.yml``)."""
    root = directory or recipes_dir()
    files = recipe_files(root)
    for path in files:
        if name in (path.stem, path.name):
            return Recipe(name=path.stem, path=path, description=read_description(path))
    available = ", ".join(p.stem for p in files) or "(none)"
    raise ConfigError(f"no recipe named {name!r} in {root}; available: {available}")


__all__ = [
    "RECIPES_DIR_ENV",
    "RECIPE_SUFFIXES",
    "Recipe",
    "get_recipe",
    "list_recipes",
    "read_description",
    "recipe_files",
    "recipes_dir",
]
