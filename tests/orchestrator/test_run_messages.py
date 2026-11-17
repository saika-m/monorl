"""What a run says when it fails (the message for a mistake of its user, the traceback otherwise)
and where the warnings of its components go."""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

import fakes
import pytest

from monorl.core.reporting import read_status
from monorl.orchestrator import loop, run


def _log(result: Any) -> str:
    return (Path(result.run_dir) / "run.log").read_text()


def test_an_unknown_component_fails_the_run_with_its_message_and_no_traceback(make_config: Any) -> None:
    result = run(make_config(trainer={"backend": "no_such_trainer"}))
    assert result.state == "failed"
    message = result.error or ""
    assert message.startswith("RegistryError: No trainer named 'no_such_trainer'. Known: ")
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


WARNING = "WARNING monorl.rewards.test_warning: reward test_warning: 4 scored"


def _warning_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "reward test_warning" in line]


def test_warnings_logged_during_a_run_are_lines_of_its_log_until_it_ends(make_config: Any) -> None:
    log = logging.getLogger("monorl.rewards.test_warning")
    log.warning("reward test_warning: before the run")
    result = run(make_config(rewards=[{"type": "test_warning"}], trainer={"max_steps": 2}))
    log.warning("reward test_warning: after the run")
    assert result.state == "succeeded"
    lines = _log(result).splitlines()
    warned = [i for i, line in enumerate(lines) if "reward test_warning" in line]
    assert len(warned) == 2  # one per step, none of before or after
    for i in warned:
        assert re.fullmatch(rf"\d{{4}}-\d\d-\d\d \d\d:\d\d:\d\d {re.escape(WARNING)}", lines[i])
    steps = [i for i, line in enumerate(lines) if re.search(r"\[running\] step \d/2 ", line)]
    assert warned[0] < steps[0] < warned[1] < steps[1]  # where it happened, not at the end


def test_a_failed_run_keeps_its_warnings_and_stops_collecting_them(make_config: Any) -> None:
    result = run(
        make_config(
            rewards=[{"type": "test_warning"}], trainer={"max_steps": 3, "extra": {"fake_fail_at": 2}}
        )
    )
    assert result.state == "failed"
    logging.getLogger("monorl.rewards.test_warning").warning("reward test_warning: after the run")
    assert [line[20:] for line in _warning_lines(_log(result))] == [WARNING, WARNING]


def test_logging_below_warning_and_outside_monorl_is_not_in_the_log(
    make_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Chatty(fakes.ExactReward):
        def __call__(self, trajectories: Any) -> Any:
            logging.getLogger("monorl.rewards.chatty").info("chatty: an info")
            logging.getLogger("somelibrary").warning("chatty: not ours")
            logging.getLogger("monorl.rewards.chatty").error("chatty: an error")
            return super().__call__(trajectories)

    monkeypatch.setattr(fakes, "WarningReward", Chatty)
    monkeypatch.setattr(logging.getLogger("monorl"), "level", logging.DEBUG)
    result = run(make_config(rewards=[{"type": "test_warning"}], trainer={"max_steps": 1}))
    chatty = [line[20:] for line in _log(result).splitlines() if "chatty" in line]
    assert chatty == ["ERROR monorl.rewards.chatty: chatty: an error"]


def test_the_terminal_shows_a_warning_of_a_run_once(
    make_config: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no logging configured Python prints a warning to stderr itself; during a run its log
    line is echoed instead. A handler configured by the program shows the record: no echo then."""
    monkeypatch.delenv("MONORL_REPORTER_ECHO", raising=False)
    monorl_log = logging.getLogger("monorl")
    monkeypatch.setattr(monorl_log, "propagate", False)  # away from the handlers pytest installs
    config = make_config(rewards=[{"type": "test_warning"}], trainer={"max_steps": 1})

    result = run(config)
    assert [line[20:] for line in _warning_lines(capsys.readouterr().err)] == [WARNING]
    assert [line[20:] for line in _warning_lines(_log(result))] == [WARNING]

    stream = io.StringIO()
    own = logging.StreamHandler(stream)
    monorl_log.addHandler(own)
    try:
        result = run(config)
    finally:
        monorl_log.removeHandler(own)
    assert _warning_lines(capsys.readouterr().err) == []
    assert stream.getvalue() == "reward test_warning: 4 scored\n"
    assert [line[20:] for line in _warning_lines(_log(result))] == [WARNING, WARNING]  # appended to
