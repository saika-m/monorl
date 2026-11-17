"""Where monorl keeps its own files: ``$MONORL_HOME``, and ``~/.monorl`` when that is not set.

The server's jobs file and token, the media cache, the Hermes workspaces and the runs that name no
directory of their own all live there, so that one variable moves them together.

Standard library only.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOME = "~/.monorl"
DEFAULT_RUN_ROOT = f"{DEFAULT_HOME}/runs"
"""The default of ``logging.run_root``: ``runs`` under the home (:func:`run_root`)."""


def monorl_home() -> Path:
    return Path(os.environ.get("MONORL_HOME") or DEFAULT_HOME).expanduser()


def run_root(configured: str | os.PathLike[str]) -> Path:
    """``logging.run_root`` as a directory. The default is ``runs`` under :func:`monorl_home`, so it
    follows ``MONORL_HOME``; any other value is the path that was written, with ``~`` expanded."""
    if os.fspath(configured) == DEFAULT_RUN_ROOT:
        return monorl_home() / "runs"
    return Path(configured).expanduser()
