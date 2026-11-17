"""What an invalid config is told: the key, what is wrong with it, and what to write instead."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from monorl.config.pipeline import PipelineConfig
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError


def _base(**overrides: Any) -> dict[str, Any]:
    cfg = {"model": {"name_or_path": "x"}, "data": {"uri": "d.jsonl"}, "rewards": [{"type": "exact_match"}]}
    cfg.update(overrides)
    return cfg


def _problems(model: type[RunConfig] | type[PipelineConfig], data: dict[str, Any]) -> list[str]:
    with pytest.raises(ConfigError) as raised:
        model.from_dict(data)
    head, *lines = str(raised.value).splitlines()
    count = "1 problem" if len(lines) == 1 else f"{len(lines)} problems"
    assert head == f"invalid {model.__name__}, {count}:"
    return [line.strip() for line in lines]


@pytest.mark.parametrize(
    "data,problem",
    [
        (_base(trainr={"max_steps": 1}), "trainr: unknown key; did you mean trainer?"),
        (_base(trainer={"max_stps": 1}), "trainer.max_stps: unknown key; did you mean max_steps?"),
        (
            _base(model={"name_or_path": "x", "lora": {"rnk": 4}}),
            "model.lora.rnk: unknown key; a LoraConfig takes: enabled, r, alpha, dropout, target_modules",
        ),
        (
            _base(rewards=[{"type": "exact_match"}, {"type": "length", "wieght": 2}]),
            "rewards.1.wieght: unknown key; did you mean weight?",
        ),
    ],
)
def test_an_unknown_key_is_named_with_the_key_it_resembles_or_the_keys_there_are(
    data: dict[str, Any], problem: str
) -> None:
    assert _problems(RunConfig, data) == [problem]


def test_a_value_of_the_wrong_type_or_outside_the_allowed_ones_is_shown() -> None:
    problems = _problems(
        RunConfig,
        _base(trainer={"max_steps": "abc"}, algorithm={"name": "grpoo"}, data={"uri": "d", "limit": 0}),
    )
    assert problems == [
        "data.limit: Input should be greater than or equal to 1 (got 0)",
        "algorithm.name: Input should be 'grpo', 'dapo', 'rloo', 'reinforce', 'ppo' or 'sft' (got 'grpoo')",
        "trainer.max_steps: Input should be a valid integer, unable to parse string as an integer (got 'abc')",
    ]


def test_a_missing_section_and_a_rule_across_fields_keep_their_message() -> None:
    assert _problems(RunConfig, {"model": {"name_or_path": "x"}}) == ["data: Field required"]
    (problem,) = _problems(RunConfig, _base(algorithm={"name": "grpo"}, rollout={"group_size": 1}))
    assert problem.startswith("grpo needs rollout.group_size >= 2")


def test_a_pipeline_names_the_stage_and_the_keys_of_its_kind() -> None:
    stages = [
        {"type": "train", "name": "a", "overides": {}},
        {"type": "eval", "name": "e", "data": {"uri": "d.jsonl", "splt": "test"}},
    ]
    assert _problems(PipelineConfig, {"base": _base(), "stages": stages}) == [
        "stages.0.train.overides: unknown key; did you mean overrides?",
        "stages.1.eval.data.splt: unknown key; did you mean split?",
    ]
    # a RunConfig key at the top of a pipeline file: its place is under `base`
    assert _problems(
        PipelineConfig, {"base": _base(), "stages": [{"type": "train", "name": "a"}], "trainer": {}}
    ) == ["trainer: unknown key; a PipelineConfig takes: name, base, stages"]
    twice = [{"type": "train", "name": "a"}, {"type": "train", "name": "a"}]
    assert _problems(PipelineConfig, {"base": _base(), "stages": twice}) == [
        "stage names must be unique, got ['a', 'a']"
    ]


def test_the_pydantic_error_stays_the_cause() -> None:
    """The server reads the structured errors (``loc``, ``msg``, ``type``) from it."""
    with pytest.raises(ConfigError) as raised:
        RunConfig.from_dict(_base(trainer={"max_stps": 1}))
    cause = raised.value.__cause__
    assert isinstance(cause, ValidationError)
    assert [(e["loc"], e["type"]) for e in cause.errors()] == [(("trainer", "max_stps"), "extra_forbidden")]
