"""``format``: regex-based structural check.

Params:
    patterns: list of regular expressions (or a single ``pattern`` string).
    require_all: ``True`` (default) → 1.0 only when every pattern matches, else 0.0;
        ``False`` → fraction of patterns that matched.
    flags: list of ``re`` flag names (default ``["DOTALL"]``); add ``"IGNORECASE"``, ``"MULTILINE"``.
    fullmatch: ``True`` to require the pattern to match the whole response (default ``search``).
    scope: ``last`` / ``all``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from monorl.core.errors import ConfigError
from monorl.core.params import declare_params
from monorl.core.types import RewardResult, Trajectory
from monorl.rewards.base import (
    check_trajectories,
    response_scope,
    response_text,
    reward_name,
    reward_params,
)

declare_params(
    "reward", "format", "name", "patterns", "pattern", "require_all", "flags", "fullmatch", "scope"
)


class FormatReward:
    def __init__(self, params: Mapping[str, Any]) -> None:
        params = reward_params("format", params)
        self.name = reward_name(params, "format")
        patterns = params.get("patterns")
        if patterns is None and "pattern" in params:
            patterns = [params["pattern"]]
        if not patterns:
            raise ConfigError("format reward needs params.patterns (list of regexes)")
        if isinstance(patterns, str):
            patterns = [patterns]
        flags = 0
        for flag_name in params.get("flags", ["DOTALL"]):
            flag = getattr(re, str(flag_name).upper(), None)
            if not isinstance(flag, re.RegexFlag):
                raise ConfigError(
                    f"format reward: params.flags takes names of `re` flags (DOTALL, IGNORECASE, "
                    f"MULTILINE, ...), got {flag_name!r}"
                )
            flags |= flag
        try:
            self.patterns = [re.compile(p, flags) for p in patterns]
        except (re.error, TypeError) as e:
            raise ConfigError(f"format reward: params.patterns holds an invalid regex: {e}") from e
        self.require_all = bool(params.get("require_all", True))
        self.fullmatch = bool(params.get("fullmatch", False))
        self.scope = response_scope(params, "format")

    def score_text(self, text: str) -> tuple[float, list[bool]]:
        hits = [bool(p.fullmatch(text) if self.fullmatch else p.search(text)) for p in self.patterns]
        if self.require_all:
            score = 1.0 if all(hits) else 0.0
        else:
            score = sum(hits) / len(hits)
        return score, hits

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        out = []
        for t in trajectories:
            score, hits = self.score_text(response_text(t, self.scope))
            out.append(RewardResult(score=score, breakdown={self.name: score}, info={"matched": hits}))
        return out


def make_format(params: Mapping[str, Any], **_: Any) -> FormatReward:
    return FormatReward(params)
