"""A run that is stopped after step N and resumed continues the run it resumes: from step N + 1 on
it logs the episodes and the metrics the uninterrupted run logs, timings apart. And a run that is
killed at any point and resumed leaves the record files of the run that was not killed."""

from __future__ import annotations

import os
import random
import signal
from pathlib import Path
from typing import Any

import fakes
import pytest
import torch

from monorl.config.schema import RunConfig
from monorl.core.registry import registry
from monorl.core.reporting import (
    METRICS_FILE,
    TRAJECTORIES_FILE,
    RunReporter,
    iter_metrics,
    iter_trajectories,
    list_checkpoints,
)
from monorl.orchestrator import run
from monorl.orchestrator.loop import LOOP_STATE_FILE


class SamplingEngine(fakes.FakeEngine):
    """Samples the way the ``hf`` engine does, from the global generators of the process, at every
    temperature: a completion is the right answer with probability 1/2 (torch's generator), else
    three letters from Python's generator followed by the weights it was sampled from."""

    def _respond(self, prompt_text: str) -> str:
        if torch.rand(()).item() < 0.5:
            return fakes.last_user_text(prompt_text).upper()
        return "".join(random.choices("xyz", k=3)) + f"@{self.weight_version:.6f}"


class DropoutTrainer(fakes.FakeTrainer):
    """Draws from torch's generator in every update and every forward pass, as dropout does. The
    weights it exports, and with them what :class:`SamplingEngine` answers, depend on the draws.
    ``tokenizer`` lets the SFT loop encode with it; ``trainer.extra.cancel_at`` sends SIGINT to
    the process during that step."""

    tokenizer = fakes.FakeTokenizer()

    def train_step(self, batch: Any) -> dict[str, float]:
        if self._step + 1 == self.config.trainer.extra.get("cancel_at"):
            os.kill(os.getpid(), signal.SIGINT)
        out = super().train_step(batch)
        noise = torch.rand(()).item()
        self.weight = self.weight + noise
        return {**out, "loss/noise": noise}

    def compute_logprobs(self, batch: Any, *, reference: bool = False) -> torch.Tensor:
        return super().compute_logprobs(batch, reference=reference) - torch.rand(())


@pytest.fixture
def sampling_components(fake_components: Any) -> None:
    registry.register("rollout_engine", "sampling", SamplingEngine, overwrite=True)
    registry.register("trainer", "dropout", DropoutTrainer, overwrite=True)


def _is_timing(key: str) -> bool:
    return key == "time" or key.startswith("time/") or key.endswith(("seconds", "/time"))


def _metric_rows(run_dir: Path, steps: tuple[int, ...]) -> list[dict[str, Any]]:
    """The rows of ``steps`` in ``metrics.jsonl``, of every phase, without the timings."""
    return [
        {k: v for k, v in row.items() if not _is_timing(k)}
        for _, row in iter_metrics(run_dir)
        if row["step"] in steps
    ]


def _episodes(run_dir: Path, steps: tuple[int, ...]) -> list[dict[str, Any]]:
    return [row for _, row in iter_trajectories(run_dir) if row["step"] in steps]


def _log(run_dir: Path) -> str:
    return (run_dir / "run.log").read_text()


def _rl_config(make_config: Any, run_name: str, max_steps: int, overrides: dict[str, Any]) -> RunConfig:
    return make_config(
        **{
            **overrides,
            "run_name": run_name,
            "rollout": {"backend": "sampling", **overrides.get("rollout", {})},
            "trainer": {
                "backend": "dropout",
                "max_steps": max_steps,
                "save_every": 2,
                **overrides.get("trainer", {}),
            },
            "logging": {"trajectories_per_step": 64},
        }
    )


def _eval_file(tmp_path: Path) -> str:
    return str(fakes.write_jsonl(tmp_path / "eval.jsonl", [{"prompt": "uvw", "answer": "UVW"}] * 3))


@pytest.mark.parametrize("case", ["plain", "evaluation", "dynamic_sampling", "multi_turn", "cancelled"])
def test_resumed_run_continues_the_run_it_resumes(
    case: str, make_config: Any, sampling_components: None, tmp_path: Path
) -> None:
    overrides: dict[str, Any] = {
        "plain": {},
        # evaluates after the checkpoint of step 2 is written, and this engine draws when it does
        "evaluation": {"data": {"eval_uri": _eval_file(tmp_path)}, "trainer": {"eval_every": 1}},
        "dynamic_sampling": {"algorithm": {"dynamic_sampling": True}},
        "multi_turn": {"env": {"type": "fake_multi", "max_turns": 3, "params": {"turn_reward": 0.25}}},
        # no periodic checkpoint: the one resumed from is written when the signal stops the run
        "cancelled": {"trainer": {"save_every": 100}},
    }[case]
    full = run(_rl_config(make_config, "full", 4, overrides))
    assert full.state == "succeeded" and full.final_step == 4
    full_dir = Path(full.run_dir)
    full_versions = [b.meta["policy_version"] for b in fakes.TRAINERS[-1].batches]

    if case == "cancelled":
        stopping = {**overrides, "trainer": {**overrides["trainer"], "extra": {"cancel_at": 2}}}
        stopped = run(_rl_config(make_config, "split", 4, stopping))
        assert stopped.state == "cancelled"
    else:
        stopped = run(_rl_config(make_config, "split", 2, overrides))
        assert stopped.state == "succeeded"
    assert stopped.final_step == 2
    split_dir = Path(stopped.run_dir)
    assert Path(list_checkpoints(split_dir)[-1]).name == "step-2"

    resumed = run(_rl_config(make_config, "split", 4, overrides), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4

    later = (3, 4)
    episodes = _episodes(full_dir, later)
    assert len(episodes) >= 2 * 2 * 2  # two steps of two prompts with two episodes each
    assert len({row["response"] for row in episodes}) > 2  # the engine did sample
    assert _episodes(split_dir, later) == episodes
    assert _metric_rows(split_dir, later) == _metric_rows(full_dir, later)
    assert [b.meta["policy_version"] for b in fakes.TRAINERS[-1].batches] == full_versions[2:]
    assert "will not continue exactly" not in _log(split_dir)


def test_checkpoint_without_loop_state_resumes_reseeded_and_says_so(
    make_config: Any, sampling_components: None
) -> None:
    """A checkpoint of before the loop saved its state: the weights and the step are restored, the
    generators are seeded with ``seed + step``, and the log says that this is another run."""
    full_dir = Path(run(_rl_config(make_config, "full", 4, {})).run_dir)
    later = (3, 4)

    resumes: list[list[dict[str, Any]]] = []
    for name in ("old-a", "old-b"):
        stopped = run(_rl_config(make_config, name, 2, {}))
        run_dir = Path(stopped.run_dir)
        (run_dir / "checkpoints" / "step-2" / LOOP_STATE_FILE).unlink()
        resumed = run(_rl_config(make_config, name, 4, {}), resume=True)
        assert resumed.state == "succeeded" and resumed.final_step == 4
        trainer = fakes.TRAINERS[-1]
        assert trainer.loaded_from == run_dir / "checkpoints" / "step-2" and trainer.step == 4
        assert [row["step"] for _, row in iter_metrics(run_dir)] == [1, 2, 3, 4]
        assert _episodes(run_dir, (1, 2)) == _episodes(full_dir, (1, 2))
        log = _log(run_dir)
        assert "will not continue exactly" in log and LOOP_STATE_FILE in log and "seed + step" in log
        resumes.append(_episodes(run_dir, later))

    assert resumes[0] == resumes[1]  # reseeded, not left to chance
    assert resumes[0] != _episodes(full_dir, later)


def test_resume_with_rollouts_ahead_says_it_is_not_exact(make_config: Any, sampling_components: None) -> None:
    overrides = {"rollout": {"async_steps_ahead": 1}}
    stopped = run(_rl_config(make_config, "ahead", 2, overrides))
    resumed = run(_rl_config(make_config, "ahead", 4, overrides), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4
    log = _log(Path(stopped.run_dir))
    assert "will not continue exactly" in log and "rollout.async_steps_ahead" in log


# -- SFT ----------------------------------------------------------------------------------------


def _sft_config(tmp_path: Path, run_name: str, max_steps: int, trainer: dict[str, Any]) -> RunConfig:
    rows = [{"prompt": p, "answer": p.upper()} for p in ("abc", "def", "ghi", "jkl", "mno", "pqr")]
    data_file = fakes.write_jsonl(tmp_path / "sft.jsonl", rows)
    cfg = fakes.fake_config_dict(tmp_path)
    return RunConfig.from_dict(
        {
            **cfg,
            "run_name": run_name,
            "algorithm": {"name": "sft"},
            "rewards": [],
            "data": {**cfg["data"], "uri": str(data_file), "eval_uri": str(data_file)},
            "trainer": {
                **cfg["trainer"],
                "backend": "dropout",
                "max_steps": max_steps,
                "save_every": 2,
                "eval_every": 1,
                **trainer,
            },
        }
    )


@pytest.mark.parametrize("stop", ["max_steps", "cancelled"])
def test_resumed_sft_run_continues_the_run_it_resumes(
    stop: str, sampling_components: None, tmp_path: Path
) -> None:
    """The SFT loop draws nothing itself; a trainer does, in its updates (dropout) and, this one,
    in the forward passes of the evaluation that follows the checkpoint."""
    save = {"save_every": 100} if stop == "cancelled" else {}
    full = run(_sft_config(tmp_path, "sft-full", 4, save))
    assert full.state == "succeeded" and full.final_step == 4
    full_dir = Path(full.run_dir)

    if stop == "cancelled":
        stopped = run(_sft_config(tmp_path, "sft-split", 4, {**save, "extra": {"cancel_at": 2}}))
        assert stopped.state == "cancelled"
    else:
        stopped = run(_sft_config(tmp_path, "sft-split", 2, save))
    assert stopped.final_step == 2
    split_dir = Path(stopped.run_dir)
    resumed = run(_sft_config(tmp_path, "sft-split", 4, save), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4

    later = (3, 4)
    rows = _metric_rows(full_dir, later)
    assert [(row["step"], row["phase"]) for row in rows] == [
        (3, "train"),
        (3, "eval"),
        (4, "train"),
        (4, "eval"),
    ]
    assert _metric_rows(split_dir, later) == rows
    assert "will not continue exactly" not in _log(split_dir)

    # the same checkpoint without the loop's state: resumes, reseeded, as another run
    old = run(_sft_config(tmp_path, "sft-old", 2, {}))
    old_dir = Path(old.run_dir)
    (old_dir / "checkpoints" / "step-2" / LOOP_STATE_FILE).unlink()
    resumed = run(_sft_config(tmp_path, "sft-old", 4, {}), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4
    assert "will not continue exactly" in _log(old_dir)
    assert [row["loss/noise"] for row in _metric_rows(old_dir, later) if row["phase"] == "train"] != [
        row["loss/noise"] for row in rows if row["phase"] == "train"
    ]


def test_resumed_sft_run_with_lora_dropout_on_the_reference_trainer(run_config_dict: dict[str, Any]) -> None:
    """The real trainer on the tiny Llama: the dropout masks of steps 3 and 4 are those of the
    uninterrupted run, so the losses and gradient norms are equal to the last digit."""

    def config(run_name: str, max_steps: int) -> RunConfig:
        lora = {"enabled": True, "r": 4, "alpha": 8, "dropout": 0.5}
        trainer = {"max_steps": max_steps, "save_every": 2, "learning_rate": 1e-2, "max_seq_len": 64}
        return RunConfig.from_dict(
            {
                **run_config_dict,
                "run_name": run_name,
                "algorithm": {"name": "sft"},
                "rewards": [],
                "model": {**run_config_dict["model"], "lora": lora},
                "trainer": {**run_config_dict["trainer"], **trainer},
            }
        )

    full = run(config("lora-full", 4))
    assert full.state == "succeeded", full.error
    assert run(config("lora-split", 2)).final_step == 2
    resumed = run(config("lora-split", 4), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4
    rows = _metric_rows(Path(full.run_dir), (3, 4))
    assert [row["step"] for row in rows] == [3, 4] and all(row["loss/sft"] > 0 for row in rows)
    assert _metric_rows(Path(resumed.run_dir), (3, 4)) == rows


@pytest.mark.slow
def test_resumed_run_on_the_hf_engine_and_the_reference_trainer(run_config_dict: dict[str, Any]) -> None:
    """No fakes: the tiny Llama sampled by the ``hf`` engine from torch's generator and trained on
    a reward that differs between the episodes of a group, so the weights move at every step."""

    def config(run_name: str, max_steps: int) -> RunConfig:
        length = {"unit": "chars", "max_tokens": 30, "penalty_per_token": 0.02}
        trainer = {"max_steps": max_steps, "save_every": 2, "learning_rate": 1e-2}
        return RunConfig.from_dict(
            {
                **run_config_dict,
                "run_name": run_name,
                "rewards": [{"type": "length", "params": length}],
                "trainer": {**run_config_dict["trainer"], **trainer},
            }
        )

    full = run(config("hf-full", 4))
    assert full.state == "succeeded", full.error
    assert run(config("hf-split", 2)).final_step == 2
    resumed = run(config("hf-split", 4), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4

    later = (3, 4)
    episodes = _episodes(Path(full.run_dir), later)
    assert len({row["response"] for row in episodes}) == len(episodes) == 8
    assert len({row["reward"] for row in episodes}) > 2
    assert _episodes(Path(resumed.run_dir), later) == episodes
    rows = _metric_rows(Path(full.run_dir), later)
    assert all(row["grad_norm"] > 0 for row in rows)
    assert _metric_rows(Path(resumed.run_dir), later) == rows


# -- killed runs --------------------------------------------------------------------------------


class _Killed(BaseException):
    """Stands in for SIGKILL: ``run()`` handles ``Exception`` and ``KeyboardInterrupt`` only, so
    nothing is written after it is raised; ``status.json`` stays at ``running``."""


def _run_until_killed(config: RunConfig, write: str, step: int, phase: str) -> Path:
    """Run ``config`` and kill it when the reporter is about to write ``write`` (``log_metrics``
    or ``log_trajectories``) of ``step`` and ``phase``. Returns the run directory."""
    original = getattr(RunReporter, write)

    def dying(self: RunReporter, at: int, *args: Any, **kwargs: Any) -> None:
        if at == step and kwargs.get("phase", "train") == phase:
            raise _Killed
        original(self, at, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch, pytest.raises(_Killed):
        patch.setattr(RunReporter, write, dying)
        run(config)
    return Path(config.logging.run_root) / str(config.run_name)


def _unfinished_lines(run_dir: Path, step: int, phase: str) -> None:
    """What a kill in the middle of a write leaves at the end of both files."""
    with open(run_dir / METRICS_FILE, "a") as f:
        f.write(f'{{"step": {step}, "time": 1.5, "phase": "{phase}", "reward/me')
    with open(run_dir / TRAJECTORIES_FILE, "a") as f:
        f.write(f'{{"step": {step}, "phase": "{phase}", "sample_id": "s1", "respo')


def _records(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every row of both files; the metrics without their timings."""
    return _metric_rows(run_dir, (1, 2, 3, 4)), _episodes(run_dir, (1, 2, 3, 4))


def _lines(run_dir: Path) -> int:
    files = (run_dir / METRICS_FILE, run_dir / TRAJECTORIES_FILE)
    return sum(len(f.read_text().splitlines()) for f in files)


# (the write the run dies in, its step, its phase, the checkpoint the resume starts from)
_KILLS = {
    # steps 3 and 4 ran after the checkpoint of step 2; the checkpoint of step 4 is step-4.tmp
    "after_later_steps": ("log_metrics", 4, "train", "step-2"),
    # the metrics row of step 2 is written, its episodes and the name of its checkpoint are not
    "in_the_rows_of_the_checkpoint_step": ("log_trajectories", 2, "train", None),
    # the checkpoint of step 2 is complete, the episodes of its evaluation are written, its row is not
    "in_the_evaluation_of_the_checkpoint_step": ("log_metrics", 2, "eval", "step-2"),
}


@pytest.mark.parametrize("kill", list(_KILLS))
def test_killed_and_resumed_run_leaves_the_records_of_the_uninterrupted_run(
    kill: str, make_config: Any, sampling_components: None, tmp_path: Path
) -> None:
    write, step, phase, checkpoint = _KILLS[kill]
    overrides = {"data": {"eval_uri": _eval_file(tmp_path)}, "trainer": {"eval_every": 1}}
    full = run(_rl_config(make_config, "full", 4, overrides))
    assert full.state == "succeeded"
    metrics, episodes = _records(Path(full.run_dir))
    assert [(row["step"], row["phase"]) for row in metrics] == [
        (s, p) for s in (1, 2, 3, 4) for p in ("train", "eval")
    ]
    assert {row["phase"] for row in episodes} == {"train", "eval"}

    run_dir = _run_until_killed(_rl_config(make_config, "killed", 4, overrides), write, step, phase)
    _unfinished_lines(run_dir, step, phase)
    # what a cancel leaves that is killed while it saves step 3; no later save replaces it
    (run_dir / "checkpoints" / "step-3.tmp").mkdir()
    complete = [Path(p).name for p in list_checkpoints(run_dir)]
    assert complete == ([checkpoint] if checkpoint else [])
    assert _records(run_dir) != (metrics, episodes)

    resumed = run(_rl_config(make_config, "killed", 4, overrides), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4
    assert fakes.TRAINERS[-1].loaded_from == (run_dir / "checkpoints" / checkpoint if checkpoint else None)
    assert _records(run_dir) == (metrics, episodes)
    assert _lines(run_dir) == _lines(Path(full.run_dir))  # nothing unreadable is left either
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["step-2", "step-4"]
    assert "will not continue exactly" not in _log(run_dir)


@pytest.mark.parametrize(
    ("write", "step", "phase"),
    [("log_metrics", 4, "train"), ("log_trajectories", 2, "sft"), ("log_metrics", 2, "eval")],
)
def test_killed_and_resumed_sft_run_leaves_the_records_of_the_uninterrupted_run(
    write: str, step: int, phase: str, sampling_components: None, tmp_path: Path
) -> None:
    full = run(_sft_config(tmp_path, "sft-full", 4, {}))
    assert full.state == "succeeded"
    metrics, episodes = _records(Path(full.run_dir))
    assert [(row["step"], row["phase"]) for row in metrics] == [
        (s, p) for s in (1, 2, 3, 4) for p in ("train", "eval")
    ]
    assert {row["step"] for row in episodes} == {1, 2, 3, 4}

    run_dir = _run_until_killed(_sft_config(tmp_path, "sft-killed", 4, {}), write, step, phase)
    _unfinished_lines(run_dir, step, phase)
    assert _records(run_dir) != (metrics, episodes)

    resumed = run(_sft_config(tmp_path, "sft-killed", 4, {}), resume=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4
    assert _records(run_dir) == (metrics, episodes)
    assert _lines(run_dir) == _lines(Path(full.run_dir))
    assert sorted(p.name for p in (run_dir / "checkpoints").iterdir()) == ["step-2", "step-4"]
