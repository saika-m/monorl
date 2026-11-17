"""What a run says when it fails: the message for a mistake of its user, the traceback otherwise."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from monorl.core.reporting import read_status
from monorl.orchestrator import loop, run


def _log(result: Any) -> str:
    return (Path(result.run_dir) / "run.log").read_text()


def test_an_unknown_component_fails_the_run_with_its_message_and_no_traceback(make_config: Any) -> None:
    result = run(make_config(trainer={"backend": "no_such_trainer"}))
    assert result.state == "failed"
    message = result.error or ""
    assert message.startswith("RegistryError: No trainer named 'no_such_trainer'. Available: [")
    assert "Traceback" not in message and "Traceback" not in _log(result)
    assert _log(result).splitlines()[-1].endswith(f"[failed] {message}")
    assert read_status(result.run_dir)["message"] == message


def test_a_missing_dataset_path_fails_the_run_with_its_message_and_no_traceback(
    make_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from monorl.data import open_source

    config = make_config(data={"uri": str(tmp_path / "no-such.jsonl")})
    monkeypatch.setattr(loop, "_resolve_open_source", lambda: open_source)
    result = run(config)
    assert result.state == "failed"
    assert result.error == f"DatasetNotFound: dataset path not found: {tmp_path / 'no-such.jsonl'}"
    assert "Traceback" not in _log(result)
    assert read_status(result.run_dir)["message"] == result.error


def test_any_other_failure_keeps_its_traceback_in_the_log_and_the_result(make_config: Any) -> None:
    result = run(make_config(trainer={"max_steps": 3, "extra": {"fake_fail_at": 2}}))
    assert result.state == "failed"
    head, _, traceback = (result.error or "").partition("\n")
    assert head == "RuntimeError: fake trainer failure at step 2"
    assert traceback.startswith("Traceback (most recent call last)") and "fakes.py" in traceback
    assert traceback in _log(result)
    assert read_status(result.run_dir)["message"] == head
