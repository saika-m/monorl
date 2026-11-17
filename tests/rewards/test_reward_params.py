"""A built-in reward refuses the parameters it does not read and the values it cannot run with,
when it is built, as a mistake in the config."""

from __future__ import annotations

from typing import Any

import pytest

from monorl.core.errors import ConfigError
from monorl.core.registry import registry
from monorl.rewards import REWARD_NAMES, build_reward

#: What each built-in needs to be built at all.
REQUIRED: dict[str, dict[str, Any]] = {
    "exact_match": {},
    "format": {"pattern": "x"},
    "math_verify": {},
    "code_exec": {},
    "length": {},
    "llm_judge": {"base_url": "http://judge.test/v1", "model": "judge"},
    "reward_model": {"model": "/no/such/reward-model"},
    "composite": {"rewards": [{"type": "length"}]},
}


@pytest.mark.parametrize("name", sorted(REWARD_NAMES))
def test_every_built_in_reward_refuses_a_parameter_it_does_not_read(name: str) -> None:
    factory = registry.get("reward", name)
    with pytest.raises(ConfigError) as raised:
        factory({**REQUIRED[name], "no_such_key": 1})
    assert str(raised.value).startswith(f"params.no_such_key: unknown key; the {name} reward takes: name, ")


@pytest.mark.parametrize(
    ("name", "params", "meant"),
    [
        ("exact_match", {"extrct": "boxed"}, "extract"),
        ("exact_match", {"numeric_tolerence": 0.5}, "numeric_tolerance"),
        ("format", {"patern": "x"}, "pattern"),
        ("length", {"max_token": 3}, "max_tokens"),
        (
            "llm_judge",
            {"base_url": "http://judge.test/v1", "model": "judge", "rubic": "{response}"},
            "rubric",
        ),
    ],
)
def test_a_misspelt_parameter_is_refused_with_the_one_it_resembles(
    name: str, params: dict[str, Any], meant: str
) -> None:
    (typo,) = set(params) - set(REQUIRED[name])
    with pytest.raises(ConfigError, match=rf"params\.{typo}: unknown key; did you mean {meant}\?"):
        registry.get("reward", name)(params)


def test_build_reward_names_the_entry_of_a_parameter_mistake() -> None:
    specs = [{"type": "length"}, {"type": "format", "params": {"pattern": "x", "flag": ["DOTALL"]}}]
    with pytest.raises(ConfigError) as raised:
        build_reward(specs)
    assert "rewards[1]" in str(raised.value)
    assert "params.flag: unknown key; did you mean flags?" in str(raised.value)

    nested = {
        "type": "composite",
        "params": {"rewards": [{"type": "exact_match", "params": {"extract": "x"}}]},
    }
    with pytest.raises(ConfigError) as raised:
        build_reward([{"type": "length"}, nested])
    assert str(raised.value).startswith(
        "rewards[1] (composite): rewards[0] (exact_match): exact_match: params.extract must be one of "
    )


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("exact_match", {"scope": "lats"}, "params.scope must be 'last' or 'all', got 'lats'"),
        ("exact_match", {"normalize": True}, "unknown normalize options [True]"),
        ("format", {"pattern": "x", "flags": ["IGNORCASE"]}, "params.flags takes names of `re` flags"),
        ("format", {"pattern": "(unclosed"}, "params.patterns holds an invalid regex"),
        ("length", {"unit": "bytes"}, "params.unit must be 'tokens', 'chars' or 'words', got 'bytes'"),
        ("code_exec", {"mode": "any"}, "params.mode must be 'fraction' or 'all', got 'any'"),
        ("reward_model", {"model": "m", "normalize": "tanh"}, "params.normalize must be 'none' or 'sigmoid'"),
        ("reward_model", {"model": "m", "dtype": "int8"}, "params.dtype must be one of"),
        ("composite", {}, "composite reward needs params.rewards"),
        (
            "composite",
            {"rewards": [{"type": "length", "wieght": 2}]},
            "wieght: unknown key; did you mean weight?",
        ),
    ],
)
def test_a_value_a_built_in_cannot_run_with_is_a_config_error_when_it_is_built(
    name: str, params: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError) as raised:
        registry.get("reward", name)(params)
    assert message in str(raised.value)
