"""TRL → monorl run-directory reporting."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from monorl.config.schema import RunConfig
from monorl.core.reporting import CONFIG_FILE, RunReporter
from monorl.core.types import Trajectory
from monorl.orchestrator.sinks import MetricSinks

#: TRL log keys → contract metric names (``docs/ARCHITECTURE.md``).
METRIC_MAP: dict[str, str] = {
    "reward": "reward/mean",
    "reward_std": "reward/std",
    "kl": "loss/kl",
    "loss": "loss/policy",
    "entropy": "loss/entropy",
    "completions/mean_length": "tokens/response_mean",
    "clip_ratio/region_mean": "policy/clipfrac",
    "learning_rate": "lr",
    "num_tokens": "tokens/total",
    "completions/clipped_ratio": "tokens/truncated_frac",
    "frac_reward_zero_std": "reward/zero_std_frac",
    "grad_norm": "train/grad_norm",
    "epoch": "train/epoch",
}

#: ``SFTTrainer`` log keys (``trl/trainer/sft_trainer.py`` ``_metrics``) → contract names.
SFT_METRIC_MAP: dict[str, str] = {
    "loss": "loss/sft",
    "mean_token_accuracy": "sft/token_accuracy",
    "entropy": "loss/entropy",
    "num_tokens": "tokens/total",
    "learning_rate": "lr",
    "grad_norm": "train/grad_norm",
    "epoch": "train/epoch",
}
#: ``eval_<key>`` names that do not simply take the ``eval/`` prefix of the train mapping.
SFT_EVAL_METRIC_MAP: dict[str, str] = {"loss": "eval/loss", "mean_token_accuracy": "eval/sft/token_accuracy"}

#: TRL keys that duplicate ``reward/mean`` / ``reward/std`` for our single composite reward function.
_IGNORED_PREFIXES = ("rewards/monorl/",)
_IGNORED_KEYS = {"step", "total_flos", "train_runtime", "train_samples_per_second", "train_steps_per_second"}


def map_metrics(logs: dict[str, Any], *, sft: bool = False) -> dict[str, float]:
    """Rename TRL log keys to monorl names; unmapped numeric keys keep their TRL name under ``trl/``.

    ``eval_`` prefixed keys land under ``eval/`` (SFT: ``eval_loss`` → ``eval/loss``).
    ``rewards/<fn>/mean`` becomes ``reward/<fn>``; ``reward/<member>`` metrics emitted by the monorl
    reward wrapper pass through unchanged.
    """
    metric_map = SFT_METRIC_MAP if sft else METRIC_MAP
    out: dict[str, float] = {}
    for key, value in logs.items():
        if key in _IGNORED_KEYS or value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        prefix = ""
        if key.startswith("eval_"):
            prefix, key = "eval/", key[len("eval_") :]
            if sft and key in SFT_EVAL_METRIC_MAP:
                out[SFT_EVAL_METRIC_MAP[key]] = value
                continue
        if key.startswith(_IGNORED_PREFIXES):
            continue
        if key in metric_map:
            name = metric_map[key]
        elif key.startswith("rewards/") and key.endswith("/mean"):
            name = "reward/" + key[len("rewards/") : -len("/mean")]
        elif key.startswith("rewards/") and key.endswith("/std"):
            name = "reward/" + key[len("rewards/") : -len("/std")] + "_std"
        elif key.startswith(("reward/", "loss/", "policy/", "tokens/", "time/", "sft/")):
            name = key
        else:
            name = "trl/" + key
        out[prefix + name] = value
    return out


class ReporterCallback(TrainerCallback):
    """Writes ``metrics.jsonl``/``status.json``, sends every metrics row to the sinks of
    ``logging.wandb_project`` and ``logging.tensorboard`` as well, and mirrors TRL checkpoints as
    ``step-<n>`` entries. Once ``cancel`` is set (the first SIGINT or SIGTERM of the run), the
    step in flight is the last one: see :meth:`on_step_end`."""

    def __init__(
        self,
        reporter: RunReporter,
        *,
        sft: bool = False,
        label: str = "GRPOTrainer",
        cancel: threading.Event | None = None,
    ) -> None:
        self.reporter = reporter
        self.sft = sft
        self.label = label
        self.cancel = cancel
        self.last_metrics: dict[str, float] = {}
        self._sinks: MetricSinks | None = None

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw: Any
    ) -> None:
        self.reporter.set_status(
            "running", f"{self.label}: {state.max_steps} steps", step=int(state.global_step)
        )
        # The config of the run, as run() wrote it to the run directory before it started the backend.
        config_file = self.reporter.run_dir / CONFIG_FILE
        if self._sinks is None and config_file.exists():
            self._sinks = MetricSinks(RunConfig.from_yaml(config_file), self.reporter.run_dir, self.reporter)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kw: Any,
    ) -> None:
        if not logs:
            return
        phase = "eval" if any(k.startswith("eval_") for k in logs) else "train"
        metrics = map_metrics(logs, sft=self.sft)
        if phase == "train":
            self.last_metrics.update(metrics)
        self.reporter.log_metrics(int(state.global_step), metrics, phase=phase)
        if self._sinks is not None:
            self._sinks.log(int(state.global_step), metrics)  # under the names of metrics.jsonl

    def on_step_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw: Any
    ) -> None:
        """A cancelled run ends with the step that just finished, like the loops of the
        orchestrator: ``Trainer`` logs the step and saves its checkpoint whatever the logging
        and saving intervals are, so that a resume continues from it, and leaves its loop."""
        if self.cancel is not None and self.cancel.is_set() and state.global_step < state.max_steps:
            self.reporter.log(f"stopping after step {state.global_step} (cancel requested)")
            control.should_log = control.should_save = control.should_training_stop = True

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw: Any
    ) -> None:
        self.close()

    def close(self) -> None:
        """Closes the metric sinks. ``Trainer.train`` calls no callback when it raises, so a run
        that fails or is interrupted needs this call from whoever called ``train``."""
        if self._sinks is not None:
            self._sinks.close()
            self._sinks = None

    def on_save(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kw: Any
    ) -> None:
        step = int(state.global_step)
        link = link_checkpoint(Path(args.output_dir), step)
        if link is not None:
            self.reporter.log(f"checkpoint saved: {link}")
            self.reporter.set_status("running", step=step, last_checkpoint=str(link))


def link_checkpoint(output_dir: Path, step: int) -> Path | None:
    """Expose TRL's ``checkpoint-<n>`` as ``step-<n>`` (relative symlink) in the same directory."""
    source = output_dir / f"{PREFIX_CHECKPOINT_DIR}-{step}"
    if not source.is_dir():
        return None
    link = output_dir / f"step-{step}"
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == source.name:
            return link
        if link.is_symlink():
            link.unlink()
        else:
            return link
    os.symlink(source.name, link, target_is_directory=True)
    return link


def trajectories_with_advantages(trajectories: list[Trajectory], advantages: list[float]) -> list[Trajectory]:
    """Annotate the latest generation batch with the advantages TRL computed for it (same order)."""
    tail = advantages[-len(trajectories) :] if len(advantages) >= len(trajectories) else []
    for traj, adv in zip(trajectories, tail, strict=False):
        traj.info["advantage"] = float(adv)
    return trajectories
