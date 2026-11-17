"""``web/mock/config_schema.json``, which the mock of the API serves to the web UI in development,
has the fields and the field descriptions of the config schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monorl.config.schema import RunConfig
from monorl.config.schema_json import REGENERATE_COMMAND, field_descriptions, main, stale_fields

MOCK_SCHEMA = Path(__file__).resolve().parents[2] / "web" / "mock" / "config_schema.json"


def test_the_checked_in_mock_schema_has_every_field_and_description() -> None:
    stale = stale_fields(MOCK_SCHEMA)
    assert not stale, f"{MOCK_SCHEMA} is out of date for {stale}; run: {REGENERATE_COMMAND}"
    # the comparison is over what the UI looks up: fields of the root and of the nested sections
    described = field_descriptions(json.loads(MOCK_SCHEMA.read_text(encoding="utf-8")))
    assert {"RunConfig.seed", "RolloutConfig.temperature", "LoraConfig.r"} <= set(described)
    assert described["RunConfig.seed"] == RunConfig.model_fields["seed"].description


def test_check_reports_a_stale_copy_and_write_repairs_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--check", str(MOCK_SCHEMA)]) == 0
    assert capsys.readouterr().err == ""

    schema = json.loads(MOCK_SCHEMA.read_text(encoding="utf-8"))
    rollout = schema["$defs"]["RolloutConfig"]["properties"]
    del rollout["temperature"]["description"]  # a description added in Python only
    del rollout["top_p"]  # a field added in Python only
    rollout["top_q"] = {"type": "number"}  # a field renamed or removed in Python
    schema["properties"]["seed"]["description"] = "The seed."  # a description reworded in Python
    copy = tmp_path / "config_schema.json"
    copy.write_text(json.dumps(schema), encoding="utf-8")
    assert stale_fields(copy) == [
        "RolloutConfig.temperature",
        "RolloutConfig.top_p",
        "RolloutConfig.top_q",
        "RunConfig.seed",
    ]
    assert main(["--check", str(copy)]) == 1
    err = capsys.readouterr().err
    assert REGENERATE_COMMAND in err and len(err.strip().splitlines()) == 1

    assert main(["--check", str(tmp_path / "absent.json")]) == 1
    capsys.readouterr()

    assert main(["--write", str(copy)]) == 0
    assert main(["--check", str(copy)]) == 0
    assert json.loads(copy.read_text(encoding="utf-8")) == RunConfig.json_schema()
