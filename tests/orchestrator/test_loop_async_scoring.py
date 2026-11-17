"""With ``rollout.async_steps_ahead`` the engines are free while a rollout's rewards are scored."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

import fakes

from monorl.core.registry import registry
from monorl.core.types import RewardResult, Trajectory
from monorl.orchestrator import run


class Probe:
    """Step 1 does not finish training before the rewards of step 2 are being scored, and that
    scoring does not return before the weights of step 1 reached the engine (or ``patience``
    seconds passed)."""

    def __init__(self, patience: float) -> None:
        self.patience = patience
        self.scoring = threading.Event()
        self.synced = threading.Event()
        self.synced_while_scoring: bool | None = None
        self.versions_scored: dict[str, float] = {}


def _install(probe: Probe) -> None:
    class Engine(fakes.FakeEngine):
        def update_weights(self, weights: Any) -> None:
            super().update_weights(weights)
            if probe.scoring.is_set():
                probe.synced.set()

    class Trainer(fakes.FakeTrainer):
        def train_step(self, batch: Any) -> dict[str, float]:
            if self.step == 0:
                assert probe.scoring.wait(20.0), "step 2 was never scored while step 1 trained"
            return super().train_step(batch)

    class Reward(fakes.ExactReward):
        def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
            step = trajectories[0].group_id.split("-")[0]
            if step == "s2" and not probe.scoring.is_set():
                probe.scoring.set()
                probe.synced_while_scoring = probe.synced.wait(probe.patience)
            probe.versions_scored[step] = fakes.ENGINES[-1].weight_version
            return super().__call__(trajectories)

    registry.register("rollout_engine", "fake_probe", Engine, overwrite=True)
    registry.register("trainer", "fake_probe", Trainer, overwrite=True)
    registry.register("reward", "test_probe", lambda params, **kw: Reward(params), overwrite=True)


def test_a_weight_sync_proceeds_while_rewards_are_scored(make_config: Any) -> None:
    probe = Probe(patience=20.0)
    _install(probe)
    result = run(
        make_config(
            rollout={"backend": "fake_probe", "async_steps_ahead": 1},
            trainer={"backend": "fake_probe", "max_steps": 3, "save_every": 100},
            rewards=[{"type": "test_probe"}],
        )
    )
    assert result.state == "succeeded" and result.final_step == 3
    assert probe.synced_while_scoring is True
    # the batch of step 2 was sampled before that sync and says so
    assert probe.versions_scored["s2"] == 1.0
    assert fakes.TRAINERS[-1].batches[1].meta["policy_version"] == 0
    assert fakes.ENGINES[-1].update_count == 3


def test_dynamic_sampling_scores_and_resamples_under_one_policy(make_config: Any) -> None:
    """Resampling needs the scores and must sample from the weights the step started with, so with
    dynamic sampling a weight sync still waits for the scoring."""
    probe = Probe(patience=0.5)
    _install(probe)
    result = run(
        make_config(
            rollout={"backend": "fake_probe", "async_steps_ahead": 1},
            trainer={"backend": "fake_probe", "max_steps": 2, "save_every": 100},
            rewards=[{"type": "test_probe"}],
            algorithm={"dynamic_sampling": True},
        )
    )
    assert result.state == "succeeded"
    assert probe.synced_while_scoring is False and probe.versions_scored["s2"] == 0.0
