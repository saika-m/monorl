"""Reward functions (registry namespace ``reward``) and the composite builder.

Factories take ``(params: dict, **kw)``. :func:`build_reward` turns the ``rewards`` list of a
:class:`~monorl.config.schema.RunConfig` into one weighted :class:`CompositeReward`.

Every built-in declares the ``params`` keys it reads next to its class
(:func:`monorl.core.params.declare_params`) and refuses any other: the schema when a config is
validated, the factory when it is called. A reward of a provider takes the params it likes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from monorl.config.schema import RewardSpec, RunConfig, validation_message
from monorl.core.errors import ConfigError
from monorl.core.interfaces import RewardFunction
from monorl.core.params import declare_params
from monorl.core.registry import Registry, ensure_providers, registry
from monorl.rewards.base import reward_params
from monorl.rewards.code_exec import make_code_exec
from monorl.rewards.composite import CompositeReward, member_keys
from monorl.rewards.exact_match import make_exact_match
from monorl.rewards.format import make_format
from monorl.rewards.length import make_length
from monorl.rewards.llm_judge import make_llm_judge
from monorl.rewards.math_verify_reward import make_math_verify
from monorl.rewards.reward_model import make_reward_model

__all__ = ["CompositeReward", "REWARD_NAMES", "build_reward", "make_composite", "register"]


def _coerce_specs(raw: Sequence[RewardSpec | dict[str, Any]]) -> list[RewardSpec]:
    specs = []
    for i, spec in enumerate(raw):
        try:
            specs.append(spec if isinstance(spec, RewardSpec) else RewardSpec.model_validate(spec))
        except ValidationError as e:
            raise ConfigError(f"rewards[{i}]: {validation_message(e, RewardSpec)}") from e
    return specs


def build_reward(
    specs: Sequence[RewardSpec | dict[str, Any]], *, config: RunConfig | None = None
) -> RewardFunction:
    """Weighted composite of ``specs``; ``RewardResult.breakdown[spec.params["name"] or spec.type]``
    holds each member's raw score. Factories receive ``config=config`` when given. A
    :class:`ConfigError` of a factory (a parameter it does not take, a value it cannot run with)
    comes back naming the entry: ``rewards[i] (type): ...``."""
    specs = _coerce_specs(specs)
    kw: dict[str, Any] = {} if config is None else {"config": config}
    ensure_providers(config)
    members = []
    for i, (key, spec) in enumerate(zip(member_keys(specs), specs, strict=True)):
        factory = registry.get("reward", spec.type)
        try:
            member = factory(dict(spec.params), **kw)
        except ConfigError as e:
            where = f"rewards[{i}] ({spec.type}): "
            raise ConfigError("\n".join(where + line for line in str(e).splitlines())) from e
        members.append((key, float(spec.weight), member))
    return CompositeReward(members)


declare_params("reward", "composite", "name", "rewards")


def make_composite(params: Mapping[str, Any], **kw: Any) -> CompositeReward:
    params = reward_params("composite", params)
    specs = params.get("rewards")
    if not specs:
        raise ConfigError("composite reward needs params.rewards (list of {type, weight, params})")
    built = build_reward(specs, config=kw.get("config"))
    built.name = str(params.get("name") or "composite")
    return built


REWARD_NAMES: dict[str, Any] = {
    "exact_match": make_exact_match,
    "format": make_format,
    "math_verify": make_math_verify,
    "code_exec": make_code_exec,
    "length": make_length,
    "llm_judge": make_llm_judge,
    "reward_model": make_reward_model,
    "composite": make_composite,
}


def register(reg: Registry) -> None:
    for name, factory in REWARD_NAMES.items():
        reg.register("reward", name, factory)
