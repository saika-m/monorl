"""``monorl env serve --transport http`` answers to the host names it was started with."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
from typer.testing import CliRunner

from monorl.cli.main import app

SERVE = [sys.executable, "-m", "monorl.cli", "env", "serve", "tool_calling", "--param", "tools=[calculator]"]


@contextmanager
def _served(*options: str) -> Iterator[str]:
    """The base URL of a server started on a port the system picked (uvicorn logs which)."""
    proc = subprocess.Popen(
        [*SERVE, "--transport", "http", "--port", "0", *options],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stderr is not None
        for line in proc.stderr:
            started = re.search(r"running on (http://127\.0\.0\.1:\d+)", line)
            if started:
                yield started.group(1)
                return
        pytest.fail(f"the server ended with status {proc.wait()} before it listened")
    finally:
        proc.kill()
        proc.wait()


def test_env_serve_answers_to_loopback_names_only_by_default() -> None:
    with _served() as url:
        assert httpx.get(f"{url}/health").json() == {"ok": True}
        refused = httpx.get(f"{url}/health", headers={"host": "env-1.internal:8080"})
        assert refused.status_code == 403
        assert "--allowed-host env-1.internal" in refused.json()["error"]["message"]


def test_env_serve_allowed_host_adds_names() -> None:
    with _served("--allowed-host", "env-1.internal", "--allowed-host", "env-2.internal") as url:
        for host in ("env-1.internal:8080", "env-2.internal", "localhost"):
            assert httpx.get(f"{url}/health", headers={"host": host}).status_code == 200
        assert httpx.get(f"{url}/health", headers={"host": "env-3.internal"}).status_code == 403


def test_env_serve_rejects_an_allowed_host_it_cannot_honour() -> None:
    runner = CliRunner()
    url_as_name = runner.invoke(
        app, ["env", "serve", "tool_calling", "--transport", "http", "--allowed-host", "http://env-1:8080"]
    )
    assert url_as_name.exit_code == 2 and "not a host name" in url_as_name.output
    over_stdio = runner.invoke(app, ["env", "serve", "tool_calling", "--allowed-host", "env-1"])
    assert over_stdio.exit_code == 2 and "--transport http" in over_stdio.output
