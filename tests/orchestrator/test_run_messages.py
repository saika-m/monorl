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


def test_a_local_model_path_that_does_not_exist_fails_the_run_before_anything_is_loaded(
    make_config: Any, tmp_path: Path, fake_components: Any
) -> None:
    missing = tmp_path / "no-such-model"
    result = run(make_config(model={"name_or_path": str(missing)}))
    assert result.state == "failed" and result.final_step == 0
    assert result.error == f"ConfigError: model.name_or_path: the local path {missing} does not exist"
    assert read_status(result.run_dir)["message"] == result.error
    assert "Traceback" not in _log(result)
    assert not fake_components.ENGINES and not fake_components.TRAINERS


def test_a_relative_model_path_is_reported_with_the_directory_it_was_looked_for_in(
    make_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "weights.bin").write_bytes(b"")
    for path, problem in [
        ("./nowhere/model", f"does not exist (looked for {Path.cwd() / 'nowhere' / 'model'})"),
        ("../no-such-model", f"does not exist (looked for {Path.cwd().parent / 'no-such-model'})"),
        ("./weights.bin", f"is a file, not a model directory (looked for {Path.cwd() / 'weights.bin'})"),
    ]:
        result = run(make_config(model={"name_or_path": path}))
        assert result.error == f"ConfigError: model.name_or_path: the local path {path} {problem}"


def test_a_model_path_with_a_tilde_is_not_expanded_because_the_loader_does_not_expand_it(
    make_config: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """transformers reads ``~/model`` as a Hub id, also when the directory is there."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "model").mkdir()
    for path, hint in [
        ("~/model", f"write {tmp_path / 'model'}"),
        ("~/no-such-model", f"{tmp_path / 'no-such-model'} does not exist either"),
    ]:
        result = run(make_config(model={"name_or_path": path}))
        assert result.state == "failed"
        assert result.error == (
            f"ConfigError: model.name_or_path: the local path {path} does not exist: "
            f"'~' is not expanded when a model is loaded ({hint})"
        )


def test_a_tokenizer_path_that_does_not_exist_fails_the_run_and_a_hub_id_is_not_looked_for(
    make_config: Any, tmp_path: Path
) -> None:
    missing = tmp_path / "no-such-tokenizer"
    result = run(make_config(model={"tokenizer_name_or_path": str(missing)}))
    assert result.error == (
        f"ConfigError: model.tokenizer_name_or_path: the local path {missing} does not exist"
    )
    result = run(make_config(model={"tokenizer_name_or_path": "org/tokenizer"}, trainer={"max_steps": 1}))
    assert result.state == "succeeded"
