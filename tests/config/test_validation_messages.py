"""What an invalid config is told: the key, what is wrong with it, and what to write instead."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from monorl.config.pipeline import PipelineConfig
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError
from monorl.pipeline.stages import validate_stage_configs


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


EXACT_MATCH_TAKES = (
    "he exact_match reward takes: name, extract, normalize, numeric_tolerance, scope, reference_key"
)


@pytest.mark.parametrize(
    "data,problems",
    [
        (
            _base(rewards=[{"type": "exact_match", "params": {"extrct": "boxed"}}]),
            [f"rewards.0.params.extrct: unknown key; did you mean extract? T{EXACT_MATCH_TAKES}"],
        ),
        (
            _base(rewards=[{"type": "exact_match", "params": {"numeric_tolerence": 0.5, "api_key": "x"}}]),
            [
                "rewards.0.params.numeric_tolerence: unknown key; did you mean numeric_tolerance? "
                f"T{EXACT_MATCH_TAKES}",
                f"rewards.0.params.api_key: unknown key; t{EXACT_MATCH_TAKES}",
            ],
        ),
        (
            _base(
                rewards=[
                    {"type": "exact_match"},
                    {
                        "type": "composite",
                        "params": {"rewards": [{"type": "format", "params": {"patern": "x"}}]},
                    },
                ]
            ),
            [
                "rewards.1.params.rewards.0.params.patern: unknown key; did you mean pattern? The format "
                "reward takes: name, patterns, pattern, require_all, flags, fullmatch, scope"
            ],
        ),
        (
            _base(env={"type": "tool_calling", "params": {"max_turn": 3}}),
            [
                "env.params.max_turn: unknown key; did you mean max_turns? The tool_calling environment "
                "takes: tools, tool_modules, system_prompt, invalid_call_penalty, max_calls_per_turn, "
                "max_turns, python_timeout_s, python_memory_mb, max_tool_output_chars, corpus, "
                "lookup_max_results"
            ],
        ),
        (
            # the default type takes one parameter; a turn limit is not it
            _base(env={"params": {"max_turns": 3}}),
            ["env.params.max_turns: unknown key; the single_turn environment takes: system_prompt"],
        ),
        (
            _base(env={"type": "remote", "params": {"transport": "http", "uri": "http://e:1"}}),
            [
                "env.params.uri: unknown key; did you mean url? The remote environment takes: transport, "
                "command, cwd, env, url, urls, headers, parallel, timeout_s, max_response_bytes"
            ],
        ),
        (
            _base(env={"type": "harness", "params": {"command": "agent", "timeout_s": 5}}),
            [
                "env.params.timeout_s: unknown key; did you mean timeout? The harness environment takes: "
                "command, timeout, env, host_env, verifier, verifier_timeout, workdir, model, sandbox, "
                "gateway, fetch_images"
            ],
        ),
        (
            _base(env={"type": "openenv", "params": {"url": "http://e:1", "action_feild": "message"}}),
            [
                "env.params.action_feild: unknown key; did you mean action_field? The openenv environment "
                "takes: env_class, env_kwargs, action_class, url, image, env_vars, port, timeout_s, "
                "action_mode, action_field, observation_template, observation_to_text, system_prompt, "
                "invalid_action_penalty, reset_kwargs, seed"
            ],
        ),
        (
            _base(env={"type": "hermes", "params": {"skill": ["git"]}}),
            [
                "env.params.skill: unknown key; did you mean skills? The hermes environment takes: tools, "
                "tool_modules, skills, skills_dir, hermes_home, workspace_dir, sandbox, "
                "max_tool_output_chars, max_calls_per_turn, invalid_call_penalty, system_prompt, "
                "hermes_system_prompt, model_name"
            ],
        ),
    ],
)
def test_a_parameter_a_built_in_does_not_read_is_named_with_the_key_it_resembles_and_the_keys_there_are(
    data: dict[str, Any], problems: list[str]
) -> None:
    assert _problems(RunConfig, data) == problems


def test_a_parameter_mistake_in_a_pipeline_is_named_below_its_base_and_with_its_stage(tmp_path: Path) -> None:
    stage = {"type": "rejection_sample", "name": "pick"}
    base = _base(rewards=[{"type": "length", "params": {"max_token": 3}}])
    (problem,) = _problems(PipelineConfig, {"base": base, "stages": [stage]})
    assert problem.startswith("base.rewards.0.params.max_token: unknown key; did you mean max_tokens?")

    # the rewards a stage scores with replace the ones of the base: checked before any stage runs
    stage["reward_overrides"] = [{"type": "format", "params": {"patern": "x"}}]
    pipeline = PipelineConfig.from_dict({"base": _base(), "stages": [stage]})
    with pytest.raises(ConfigError) as raised:
        validate_stage_configs(pipeline, tmp_path)
    assert "stage 1/1 'pick' (rejection_sample)" in str(raised.value)
    assert "rewards.0.params.patern: unknown key; did you mean pattern?" in str(raised.value)


def test_the_types_of_a_provider_take_the_params_they_like() -> None:
    config = RunConfig.from_dict(
        _base(
            providers=["my_components"],
            rewards=[{"type": "keyword", "params": {"keyword": "the", "extrct": 1}}],
            env={"type": "guess", "params": {"high": 20, "max_turn": 3}},
        )
    )
    assert config.rewards[0].params == {"keyword": "the", "extrct": 1}
    assert config.env.params == {"high": 20, "max_turn": 3}
