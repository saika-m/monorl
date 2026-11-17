"""Real CPU runs of the ``trl`` pipeline backend on the tiny Llama (single- and multi-turn)."""

from __future__ import annotations

import json
import signal
from collections.abc import Sequence
from pathlib import Path

import pytest
from transformers import AutoModelForCausalLM
from trl_support import ENV_NAME, REWARD_NAME, TwoTurnEnv, VowelReward

from monorl.adapters.trl.rollout import EpisodeRollout, TrainerPolicyEngine
from monorl.config.schema import RunConfig
from monorl.core.registry import registry
from monorl.core.reporting import RunReporter, iter_metrics, iter_trajectories, list_checkpoints, read_status
from monorl.core.types import RewardResult, Trajectory
from monorl.orchestrator import run
from monorl.patching import is_applied

pytestmark = pytest.mark.slow


def _run(config: RunConfig, run_dir: Path, *, resume: bool = False):
    reporter = RunReporter(run_dir, echo=False)
    backend = registry.get("pipeline_backend", "trl")(config)
    assert backend.name == "trl"
    return backend.run(config, run_dir, reporter=reporter, resume=resume)


def test_single_turn_grpo_end_to_end(trl_config: RunConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = _run(trl_config, run_dir)

    assert result.state == "succeeded", result.error
    assert result.final_step == 2
    assert is_applied("trl.grpo.monorl_hooks", "trl")

    rows = [row for _, row in iter_metrics(run_dir)]
    train_rows = [r for r in rows if "loss/policy" in r]
    assert len(train_rows) == 2
    for r in train_rows:
        assert {
            "reward/mean",
            "reward/std",
            "loss/policy",
            "lr",
            "tokens/response_mean",
            "loss/entropy",
        } <= set(r)
        assert f"reward/{REWARD_NAME}" in r and r["phase"] == "train"
    assert [r["step"] for r in train_rows] == [1, 2]

    status = read_status(run_dir)
    assert status is not None and status["state"] == "succeeded" and status["step"] == 2

    assert (run_dir / "final" / "config.json").exists()
    assert result.final_model_path == str(run_dir / "final")
    AutoModelForCausalLM.from_pretrained(run_dir / "final")

    checkpoints = list_checkpoints(run_dir)
    assert [Path(p).name for p in checkpoints] == ["step-1", "step-2"]
    assert result.checkpoints == checkpoints
    assert (Path(checkpoints[0]) / "trainer_state.json").exists()

    trajs = [t for _, t in iter_trajectories(run_dir)]
    assert len(trajs) == 8  # 4 sequences/step x 2 steps, trajectories_per_step=4
    assert {t["step"] for t in trajs} == {1, 2}
    assert all(t["num_turns"] == 1 and t["num_response_tokens"] > 0 for t in trajs)
    assert all(REWARD_NAME in t["reward_breakdown"] for t in trajs)
    assert all(t["reference"] in {"4", "8", "6", "21"} for t in trajs)


def test_lora_and_resume_from_checkpoint(trl_config: RunConfig, tmp_path: Path) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.model.lora.enabled = True
    cfg.model.lora.r = 4
    cfg.model.lora.alpha = 8
    cfg.trainer.max_steps = 1
    run_dir = tmp_path / "run-lora"
    first = _run(cfg, run_dir)
    assert first.state == "succeeded", first.error
    final_cfg = json.loads((run_dir / "final" / "config.json").read_text())
    assert "peft_type" not in final_cfg and final_cfg["architectures"] == [
        "LlamaForCausalLM"
    ]  # merged export
    assert (run_dir / "checkpoints" / "step-1" / "adapter_config.json").exists()

    cfg.trainer.max_steps = 2
    second = _run(cfg, run_dir, resume=True)
    assert second.state == "succeeded", second.error
    assert second.final_step == 2
    log = (run_dir / "run.log").read_text()
    assert "resuming from" in log and "checkpoint-1" in log
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-1", "step-2"]


def _step_rows(run_dir: Path) -> list[dict]:
    """The rows TRL logs per optimizer step, without what is a wall-clock time."""
    return [
        {k: v for k, v in row.items() if "time" not in k}
        for _, row in iter_metrics(run_dir)
        if "loss/policy" in row
    ]


def _resume_config(trl_config: RunConfig, max_steps: int) -> RunConfig:
    """Eight prompts, two per step: four steps are one pass over the data."""
    cfg = trl_config.model_copy(deep=True)
    rows = [{"prompt": f"What is {i}+{i}?", "answer": str(2 * i)} for i in range(1, 9)]
    data = Path(cfg.data.uri).with_name("eight.jsonl")
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cfg.data.uri = str(data)
    cfg.trainer.max_steps = max_steps
    cfg.trainer.save_every = 2
    cfg.trainer.learning_rate = 1e-2  # the weights move at every step
    return cfg


def test_resume_of_a_killed_run_continues_as_the_uninterrupted_run(
    trl_config: RunConfig, tmp_path: Path
) -> None:
    from safetensors.torch import load_file

    full_dir, run_dir = tmp_path / "full", tmp_path / "split"
    assert _run(_resume_config(trl_config, 4), full_dir).state == "succeeded"
    rows, episodes = _step_rows(full_dir), [row for _, row in iter_trajectories(full_dir)]
    assert [row["step"] for row in rows] == [1, 2, 3, 4] and len(episodes) == 16

    # a run with a checkpoint of step 2, killed while it saved step 3 and wrote the rows of step 4
    assert _run(_resume_config(trl_config, 2), run_dir).final_step == 2
    with open(run_dir / "metrics.jsonl", "a") as f:
        f.write(json.dumps({"step": 3, "time": 0.0, "phase": "train", "loss/policy": 9.0}) + "\n")
        f.write('{"step": 4, "time": 0.0, "pha')
    with open(run_dir / "trajectories.jsonl", "a") as f:
        f.write(json.dumps({"step": 3, "phase": "train", "sample_id": "left over"}) + "\n")
    unfinished = run_dir / "checkpoints" / "checkpoint-3"
    unfinished.mkdir()
    (unfinished / "config.json").write_text("{}")  # no trainer_state.json: the save did not get there

    resumed = _run(_resume_config(trl_config, 4), run_dir, resume=True)
    assert resumed.state == "succeeded", resumed.error
    assert resumed.final_step == 4
    log = (run_dir / "run.log").read_text()
    assert f"resuming from {run_dir / 'checkpoints' / 'checkpoint-2'}" in log
    assert _step_rows(run_dir) == rows
    assert [row for _, row in iter_trajectories(run_dir)] == episodes
    weights, expected = (
        load_file(run_dir / "final" / "model.safetensors"),
        load_file(full_dir / "final" / "model.safetensors"),
    )
    assert weights.keys() == expected.keys()
    assert all((weights[name] == expected[name]).all() for name in expected)
    moved = load_file(run_dir / "checkpoints" / "checkpoint-2" / "model.safetensors")
    assert any((moved[name] != expected[name]).any() for name in expected)  # steps 3 and 4 did train


def test_first_signal_ends_the_run_after_its_step_and_resume_continues_from_it(
    trl_config: RunConfig, tmp_path: Path
) -> None:
    """SIGINT while step 2 of 4 scores its completions. The run ends after that step, cancelled,
    with TRL's checkpoint of it (no checkpoint was due before step 4), and ``--resume`` trains
    steps 3 and 4 from there: the rows are those of a run that was not interrupted."""
    full_dir, run_dir = tmp_path / "full", tmp_path / "cancelled"
    config = _resume_config(trl_config, 4)
    config.trainer.save_every = 4
    assert run(config, full_dir).state == "succeeded"

    class SignalReward(VowelReward):
        batches = 0

        def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
            SignalReward.batches += 1
            if SignalReward.batches == 2:
                signal.raise_signal(signal.SIGINT)
            return super().__call__(trajectories)

    registry.register("reward", REWARD_NAME, lambda params, **kw: SignalReward(), overwrite=True)
    try:
        result = run(config, run_dir)
        assert result.state == "cancelled", result.error
        assert result.final_step == 2 and result.final_model_path is None
        status = read_status(run_dir)
        assert status is not None and status["state"] == "cancelled" and status["step"] == 2
        assert [row["step"] for row in _step_rows(run_dir)] == [1, 2]
        assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-2"]
        assert not (run_dir / "final").exists()
        log = (run_dir / "run.log").read_text()
        assert "received SIGINT" in log and "stopping after step 2 (cancel requested)" in log

        resumed = run(config, run_dir, resume=True)
    finally:
        registry.register("reward", REWARD_NAME, lambda params, **kw: VowelReward(), overwrite=True)
    assert resumed.state == "succeeded", resumed.error
    assert resumed.final_step == 4 and (run_dir / "final" / "config.json").exists()
    assert f"resuming from {run_dir / 'checkpoints' / 'checkpoint-2'}" in (run_dir / "run.log").read_text()
    assert _step_rows(run_dir) == _step_rows(full_dir)


def test_run_without_resume_trains_from_step_0_next_to_old_checkpoints(
    trl_config: RunConfig, tmp_path: Path
) -> None:
    """A finished run in the directory: without ``--resume`` the next one trains its steps, with
    its own learning rate, instead of loading the last checkpoint and training nothing."""
    run_dir = tmp_path / "run"
    assert _run(_resume_config(trl_config, 2), run_dir).final_step == 2
    first = _step_rows(run_dir)

    cfg = _resume_config(trl_config, 2)
    cfg.trainer.learning_rate = 5e-3
    again = _run(cfg, run_dir)
    assert again.state == "succeeded", again.error
    assert again.final_step == 2
    assert "resuming from" not in (run_dir / "run.log").read_text()
    second = _step_rows(run_dir)[len(first) :]
    assert [(row["step"], row["lr"]) for row in second] == [(1, 5e-3), (2, 5e-3)]


def test_resume_without_a_checkpoint_starts_at_step_0(trl_config: RunConfig, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    RunReporter(run_dir, echo=False).log_metrics(7, {"loss/policy": 1.0})  # of a run that left no checkpoint
    result = _run(_resume_config(trl_config, 2), run_dir, resume=True)
    assert result.state == "succeeded", result.error
    assert [row["step"] for row in _step_rows(run_dir)] == [1, 2]
    assert (
        "resume requested but no complete checkpoint exists; starting from step 0"
        in (run_dir / "run.log").read_text()
    )


class _AlwaysStopEngine(TrainerPolicyEngine):
    """The random tiny model never emits EOS within 8 tokens; mark completions finished so the
    environment's second turn is actually exercised. Tokens, logprobs and prompts stay real."""

    def generate(self, prompts, params, *, max_new_tokens=None):
        out = super().generate(prompts, params, max_new_tokens=max_new_tokens)
        for group in out:
            for c in group:
                c.finish_reason = "stop"
        return out


def test_multi_turn_env_through_patch(
    trl_config: RunConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = trl_config.model_copy(deep=True)
    cfg.env.type = ENV_NAME
    cfg.env.max_turns = 2
    cfg.rewards = []  # the environment owns the reward
    cfg.trainer.save_every = 2
    monkeypatch.setattr(
        EpisodeRollout,
        "default_engine",
        staticmethod(
            lambda trainer: _AlwaysStopEngine(
                trainer.accelerator.unwrap_model(trainer.model),
                trainer.processing_class,
                device=trainer.accelerator.device,
                eos_token_ids=trainer.eos_token_ids,
            )
        ),
    )
    TwoTurnEnv.resets.clear()
    run_dir = tmp_path / "run-mt"
    result = _run(cfg, run_dir)

    assert result.state == "succeeded", result.error
    assert len(TwoTurnEnv.resets) == 8  # one episode per sequence: 4 per step x 2 steps
    train_rows = [r for _, r in iter_metrics(run_dir) if "loss/policy" in r]
    assert len(train_rows) == 2
    for r in train_rows:
        assert r["reward/mean"] == pytest.approx(1.5)  # 0.5 + 1.0 env reward per episode
        assert r["reward/env"] == pytest.approx(1.5)
        # Policy tokens only: up to 2 turns x 8; the ~15-token env follow-up must be masked out of
        # the count (a prefix-mismatched episode trains its last turn only, hence the lower bound).
        assert 8.0 <= r["tokens/response_mean"] <= 16.0 and r["trl/completions/max_length"] <= 16.0
    trajs = [t for _, t in iter_trajectories(run_dir)]
    assert trajs and all(t["num_turns"] == 2 for t in trajs)
    assert all(t["reward_breakdown"] == {"env": 1.5} for t in trajs)
    assert [Path(p).name for p in list_checkpoints(run_dir)] == ["step-2"]
    assert (run_dir / "final" / "config.json").exists()
