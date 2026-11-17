"""Optional metric sinks (Weights & Biases, TensorBoard). Neither package is required: a missing
import is logged once and the sink is skipped; ``metrics.jsonl`` stays the source of truth."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from monorl.config.schema import RunConfig
    from monorl.core.reporting import RunReporter


class MetricSinks:
    def __init__(self, config: RunConfig, run_dir: Path, reporter: RunReporter) -> None:
        self._wandb: Any = None
        self._tb_dir = run_dir / "tensorboard"
        self._tb_writer: Any = None  # the SummaryWriter class while the TensorBoard sink is on
        self._tb: Any = None  # the writer, opened at the first row
        logging = config.logging
        if logging.wandb_project:
            try:
                import wandb

                # A `wandb` directory on sys.path (the package writes ./wandb by default) imports
                # as an empty namespace package when the package itself is not installed.
                if not hasattr(wandb, "init"):
                    raise ImportError("wandb")
                self._wandb = wandb.init(
                    project=logging.wandb_project,
                    name=config.run_name,
                    dir=str(run_dir),
                    config=config.to_dict(),
                    resume="allow",
                )
                reporter.log(
                    f"wandb: {'resumed' if self._wandb.resumed else 'new'} run {self._wandb.id} in project "
                    f"{logging.wandb_project}, files in {self._wandb.dir}"
                )
            except ImportError:
                reporter.log("logging.wandb_project is set but `wandb` is not installed; skipping wandb")
            except Exception as e:  # noqa: BLE001 - never let telemetry kill a run
                reporter.log(f"wandb init failed ({type(e).__name__}: {e}); skipping wandb")
        if logging.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter
                reporter.log(f"tensorboard: events in {self._tb_dir}")
            except ImportError:
                reporter.log("logging.tensorboard is set but `tensorboard` is not installed; skipping")
            except Exception as e:  # noqa: BLE001
                reporter.log(f"tensorboard init failed ({type(e).__name__}: {e}); skipping")

    @property
    def active(self) -> bool:
        return self._wandb is not None or self._tb_writer is not None

    def log(self, step: int, metrics: dict[str, float], *, phase: str = "train") -> None:
        """One ``metrics.jsonl`` row: every value the row holds as a number, under the same name in
        both sinks (a name of another phase than ``train`` gets the ``<phase>/`` prefix unless it
        has it)."""
        step = int(step)
        named: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                number = float(value)  # as RunReporter.log_metrics converts it
            except (TypeError, ValueError):
                continue
            named[key if phase == "train" or key.startswith(f"{phase}/") else f"{phase}/{key}"] = number
        if not named:
            return
        if self._wandb is not None:
            try:
                self._wandb.log(named, step=step)
            except Exception:  # noqa: BLE001
                pass
        if self._tb_writer is not None:
            if self._tb is None:
                # purge_step: TensorBoard hides what an earlier process wrote to the directory for
                # this step and later ones. A resumed run logs the steps after its checkpoint
                # again, and they would show twice.
                self._tb = self._tb_writer(log_dir=str(self._tb_dir), purge_step=step)
            for tag, number in named.items():
                self._tb.add_scalar(tag, number, step)

    def close(self) -> None:
        if self._tb is not None:
            try:
                self._tb.close()  # flushes
            except Exception:  # noqa: BLE001
                pass
        self._tb = self._tb_writer = None
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # noqa: BLE001
                pass
            self._wandb = None
