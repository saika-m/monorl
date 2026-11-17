"""A dataset file a web server serves (``http://``), against a server on the loopback interface."""

from __future__ import annotations

import functools
import json
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from monorl.cli.main import app
from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.data import open_source
from tests.data.conftest import QA_ROWS, write_csv, write_jsonl, write_parquet


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def served(tmp_path: Path) -> Iterator[str]:
    """Base URL of a server for ``tmp_path/served``, which holds the QA rows in three formats."""
    root = tmp_path / "served"
    write_jsonl(root / "x.jsonl", QA_ROWS)
    write_csv(root / "x.csv", QA_ROWS)
    write_parquet(root / "x.parquet", QA_ROWS)
    (root / "notes.txt").write_text("not a dataset\n")
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("name", ["x.jsonl", "x.csv", "x.parquet"])
def test_a_file_over_http_is_read_like_the_local_file(served: str, name: str) -> None:
    src = open_source(DataConfig(uri=f"{served}/{name}", shuffle=False))
    samples = list(src)
    assert len(samples) == len(QA_ROWS)
    assert samples[3].messages == [{"role": "user", "content": "What is 3+3?"}]
    assert str(samples[3].reference) == "6"
    assert samples[3].id == f"{served}/{name}#3"


def test_a_file_the_server_does_not_have_and_a_url_that_is_no_data_file(served: str) -> None:
    with pytest.raises(FileNotFoundError, match="dataset path not found: http://.*/missing.jsonl"):
        open_source(DataConfig(uri=f"{served}/missing.jsonl"))
    for uri in (f"{served}/notes.txt", f"{served}/"):
        with pytest.raises(DataError, match="an http\\(s\\) dataset is one file"):
            open_source(DataConfig(uri=uri))


def test_data_inspect_of_an_http_file(served: str) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["data", "inspect", f"{served}/x.jsonl", "--limit", "2", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["num_rows"] == len(QA_ROWS)
    assert [s["id"] for s in report["samples"]] == [f"{served}/x.jsonl#0", f"{served}/x.jsonl#1"]
    result = runner.invoke(app, ["data", "inspect", f"{served}/missing.jsonl"])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "dataset path not found" in result.output
