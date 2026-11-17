"""``composite``: weighted sum of member rewards with a per-member breakdown.

``RewardResult.score = Σ weight_i * raw_i``; ``breakdown[key_i] = raw_i`` where ``key_i`` is
``params["name"]`` of the member spec, else its ``type`` (duplicates get ``#2``, ``#3``, …).
A member that is itself a composite also contributes its own breakdown, each entry under
``key_i/<its key>`` (so ``outer/inner/leaf`` at depth two): metrics become
``reward/key_i/<its key>`` next to the member's total ``reward/key_i``.
Member ``info`` dicts are nested under ``info[key_i]``.

As a registered reward (``type: composite``), ``params["rewards"]`` holds the member specs
(``[{type, weight, params}, ...]``).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from monorl.config.schema import RewardSpec
from monorl.core.interfaces import RewardFunction
from monorl.core.types import RewardResult, Trajectory


def _unused(base: str, taken: Collection[str]) -> str:
    key, n = base, 1
    while key in taken:
        n += 1
        key = f"{base}#{n}"
    return key


def member_keys(specs: Sequence[RewardSpec]) -> list[str]:
    keys: list[str] = []
    for spec in specs:
        keys.append(_unused(str(spec.params.get("name") or spec.type), keys))
    return keys


class CompositeReward:
    def __init__(
        self, members: Sequence[tuple[str, float, RewardFunction]], *, name: str = "composite"
    ) -> None:
        if not members:
            raise ValueError("composite reward needs at least one member")
        self.name = name
        self.members = list(members)

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        per_member = [
            (key, weight, fn(trajectories), isinstance(fn, CompositeReward))
            for key, weight, fn in self.members
        ]
        # A member's own key always holds its total; a nested entry gives way to it.
        totals = {key for key, _, _ in self.members}
        out: list[RewardResult] = []
        for i in range(len(trajectories)):
            total = 0.0
            breakdown: dict[str, float] = {}
            info: dict[str, Any] = {}
            for key, weight, results, nested in per_member:
                if len(results) != len(trajectories):
                    raise ValueError(
                        f"reward {key!r} returned {len(results)} results for {len(trajectories)} trajectories"
                    )
                r = results[i]
                breakdown[key] = float(r.score)
                if nested:
                    for inner, score in r.breakdown.items():
                        breakdown[_unused(f"{key}/{inner}", totals | breakdown.keys())] = float(score)
                total += weight * float(r.score)
                if r.info:
                    info[key] = r.info
            out.append(RewardResult(score=total, breakdown=breakdown, info=info))
        return out
