"""``pipeline_backend: verl`` — compile the RunConfig, stage run-dir artefacts, launch verl.

Launch is a subprocess ``python -m monorl.adapters.verl.entry <overrides>`` (cwd = run dir) so the
monorl process never imports verl/ray. Its stdout+stderr stream into ``run.log``; metrics arrive
through the ``Tracking.log`` patch (``metrics.jsonl``); ``status.json`` is kept current here.
Cancellation (``cancel()``, or the first SIGTERM/SIGINT in the main thread) forwards SIGTERM to
the verl driver, which is what ``ray job stop`` does, then SIGKILLs after
:attr:`VerlBackend.grace_seconds`. The signals are handled by
:func:`~monorl.core.signals.signal_guard`: the first sets an event that the loop waiting for the
driver acts on, a second one interrupts that loop and the driver is killed at once. SIGKILL goes
to the driver and to every process that descends from it (:func:`kill_tree`).

verl writes checkpoints to ``trainer.default_local_dir`` (= ``<run_dir>/checkpoints``) as
``global_step_<n>/`` plus ``latest_checkpointed_iteration.txt`` (``verl/trainer/ppo/v1/trainer_base.py``);
``step-<n>`` symlinks are added so :func:`monorl.core.reporting.list_checkpoints` sees them.
"""

from __future__ import annotations

import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import psutil
import yaml

from monorl.adapters.verl.compiler import compile_config, to_overrides
from monorl.adapters.verl.data import export_datasets, parquet_has_images
from monorl.adapters.verl.patches import RUN_DIR_ENV
from monorl.adapters.verl.reward import write_reward_module
from monorl.adapters.verl.tools import write_tool_config
from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable
from monorl.core.reporting import CHECKPOINT_DIR, CONFIG_FILE, LOG_FILE, RunReporter, iter_metrics
from monorl.core.signals import signal_guard
from monorl.core.types import RunResult

ENTRY_MODULE = "monorl.adapters.verl.entry"
COMPILED_CONFIG_FILE = "verl_config.yaml"
OVERRIDES_FILE = "verl_overrides.txt"
LATEST_ITERATION_FILE = "latest_checkpointed_iteration.txt"
GLOBAL_STEP_PREFIX = "global_step_"
HF_EXPORT_SUBDIR = Path("actor") / "huggingface"


def is_available() -> bool:
    return importlib.util.find_spec("verl") is not None


def checkpoint_dirs(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``(step, path)`` of verl ``global_step_<n>`` folders under ``<run_dir>/checkpoints``, ascending."""
    root = Path(run_dir) / CHECKPOINT_DIR
    if not root.is_dir():
        return []
    found = []
    for d in root.iterdir():
        if d.is_dir() and not d.is_symlink() and d.name.startswith(GLOBAL_STEP_PREFIX):
            try:
                found.append((int(d.name[len(GLOBAL_STEP_PREFIX) :]), d))
            except ValueError:
                continue
    return sorted(found)


def link_checkpoints(run_dir: str | Path) -> list[str]:
    """Expose verl checkpoints as ``step-<n>`` symlinks; returns the checkpoint paths."""
    out = []
    for step, path in checkpoint_dirs(run_dir):
        alias = path.parent / f"step-{step}"
        if not alias.exists() and not alias.is_symlink():
            try:
                alias.symlink_to(path.name, target_is_directory=True)
            except OSError:
                pass
        out.append(str(path))
    return out


def latest_step(run_dir: str | Path) -> int | None:
    marker = Path(run_dir) / CHECKPOINT_DIR / LATEST_ITERATION_FILE
    if marker.exists():
        try:
            return int(marker.read_text().strip())
        except ValueError:
            pass
    dirs = checkpoint_dirs(run_dir)
    return dirs[-1][0] if dirs else None


def resume_step(run_dir: str | Path) -> int | None:
    """The step verl continues after under ``trainer.resume_mode: auto``; 0 when it trains from
    scratch. ``find_latest_ckpt_path`` (``verl/utils/checkpoint/checkpoint_manager.py``) reads the
    step from ``latest_checkpointed_iteration.txt``, which is written once a checkpoint is
    complete, and takes ``global_step_<n>`` only if that directory exists. ``None`` for a marker
    that is not a number: verl stops on it."""
    marker = Path(run_dir) / CHECKPOINT_DIR / LATEST_ITERATION_FILE
    try:
        step = int(marker.read_text())
    except OSError:
        return 0
    except ValueError:
        return None
    return step if (marker.parent / f"{GLOBAL_STEP_PREFIX}{step}").exists() else 0


def final_model_path(run_dir: str | Path) -> str | None:
    dirs = checkpoint_dirs(run_dir)
    if not dirs:
        return None
    last = dirs[-1][1]
    for hf in (last / HF_EXPORT_SUBDIR, last / "huggingface"):  # PPO actor export / SFT ``hf_model`` export
        if hf.is_dir():
            return str(hf)
    return str(last)


def _descendants(pids: set[int]) -> list[psutil.Process]:
    """The processes that descend from one of ``pids``, from one look at the process table."""
    children: dict[int | None, list[psutil.Process]] = {}
    for process in psutil.process_iter(["ppid"]):
        children.setdefault(process.info["ppid"], []).append(process)
    found: dict[int, psutil.Process] = {}
    pending = list(pids)
    while pending:
        for child in children.get(pending.pop(), ()):
            if child.pid not in found and child.pid not in pids:
                found[child.pid] = child
                pending.append(child.pid)
    return list(found.values())


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the driver ``proc`` and the processes that descend from it, and reap the driver.

    The driver runs in the process group of monorl, where a signal to the group of whoever
    started monorl (the server's job control, a terminal) still reaches it; the price is that its
    group cannot be killed without killing monorl. So its tree is walked: the driver is killed
    first and starts nothing more, then what it had started. A process whose parent has died is
    no longer in anybody's tree, so the second look, for what the processes found have started
    since the first, comes before they are killed. Not reached: a process that left the tree
    earlier (its parent ended, or it was started detached), and one the driver starts between the
    first look and its own SIGKILL."""
    found = _descendants({proc.pid}) if proc.poll() is None else []
    proc.kill()
    found += _descendants({process.pid for process in found})
    for process in found:
        try:
            process.kill()
        except psutil.Error:  # gone already, or not ours to signal
            pass
    proc.wait()


class VerlBackend:
    name = "verl"
    grace_seconds = 30.0
    poll_seconds = 1.0

    def __init__(self, config: RunConfig | None = None, *, python: str | None = None) -> None:
        extra = dict(config.trainer.extra) if config is not None else {}
        self.python = python or extra.get("verl_python") or sys.executable
        self._custom_python = bool(python or extra.get("verl_python"))
        self._proc: subprocess.Popen[bytes] | None = None
        self._cancel = threading.Event()

    # -- staging ------------------------------------------------------------------------
    def prepare(self, config: RunConfig, run_dir: str | Path, *, resume: bool = False) -> list[str]:
        """Write parquet data, reward module, tool config and the compiled tree; return overrides."""
        run_dir = Path(run_dir).expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / CONFIG_FILE
        if not config_path.exists():
            config.to_yaml(config_path)
        paths = export_datasets(config, run_dir)
        multimodal = parquet_has_images(paths["train"])
        if not config.is_sft:
            write_reward_module(config, run_dir, config_path=config_path)
            if config.env.type == "tool_calling":
                write_tool_config(config, run_dir)
        tree = compile_config(config, run_dir, multimodal=multimodal, resume=resume)
        (run_dir / COMPILED_CONFIG_FILE).write_text(yaml.safe_dump(tree, sort_keys=False))
        overrides = to_overrides(tree)
        (run_dir / OVERRIDES_FILE).write_text("\n".join(overrides) + "\n")
        return overrides

    def build_command(self, overrides: list[str], *, sft: bool = False) -> list[str]:
        """``python -m monorl.adapters.verl.entry [--sft] <overrides>``."""
        return [self.python, "-m", ENTRY_MODULE, *(["--sft"] if sft else []), *overrides]

    def build_env(self, config: RunConfig, run_dir: str | Path) -> dict[str, str]:
        env = dict(os.environ)
        env[RUN_DIR_ENV] = str(Path(run_dir).expanduser().resolve())
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HYDRA_FULL_ERROR", "1")
        if config.cluster.ray_address:
            env["RAY_ADDRESS"] = config.cluster.ray_address
        return env

    # -- lifecycle ----------------------------------------------------------------------
    def cancel(self) -> None:
        """Request cancellation: ``run`` sends SIGTERM to the verl driver (``ray job stop``
        semantics) the next time it looks, at most :attr:`poll_seconds` later."""
        self._cancel.set()

    def run(
        self,
        config: RunConfig,
        run_dir: Path,
        *,
        reporter: RunReporter,
        resume: bool = False,
        cancel: threading.Event | None = None,
    ) -> RunResult:
        """``cancel`` is the event that the first SIGINT or SIGTERM sets. While the driver runs
        the backend handles the two signals with it (on the main thread; with an event of its
        own when it is given none): the first asks the driver to stop, a second one kills it
        (``docs/BACKENDS.md``, "Signals with a pipeline backend")."""
        run_dir = Path(run_dir).expanduser().resolve()
        if not self._custom_python and not is_available():
            raise BackendUnavailable("pipeline_backend 'verl' needs `pip install monorl[verl]` (verl>=0.9.1)")
        reporter.set_status("pending", "compiling verl config")
        overrides = self.prepare(config, run_dir, resume=resume)
        if resume and (step := resume_step(run_dir)) is not None:
            reporter.log(
                f"trainer.resume_mode=auto: verl continues after step {step}"
                if step
                else "resume requested but verl has no checkpoint to continue from; it starts from step 0"
            )
            reporter.drop_rows_after(step)
        cmd = self.build_command(overrides, sft=config.is_sft)
        reporter.log(
            "launching: " + " ".join(cmd[:3]) + f" (+{len(overrides)} overrides, see {OVERRIDES_FILE})"
        )

        stop = cancel if cancel is not None else threading.Event()
        interrupted = False
        tail: deque[str] = deque(maxlen=20)
        with (
            open(run_dir / LOG_FILE, "ab") as log_file,
            signal_guard(stop, reporter, "stopping the verl driver"),
        ):
            self._proc = subprocess.Popen(
                cmd,
                cwd=run_dir,
                env=self.build_env(config, run_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            try:
                reporter.set_status("running", "verl started", step=0, child_pid=self._proc.pid)
                self._wait(run_dir, reporter, stop)
            except BaseException as e:
                # The second signal, or a failure of the wait: the driver does not outlive the run.
                kill_tree(self._proc)
                if not isinstance(e, KeyboardInterrupt):
                    raise
                interrupted = True
        returncode = self._proc.returncode
        tail.extend(self._tail_log(run_dir))

        checkpoints = link_checkpoints(run_dir)
        step = latest_step(run_dir)
        last_metrics = self._last_metrics(run_dir)
        if step is None:
            step = int(last_metrics.get("step", 0))
        if interrupted or self._cancel.is_set() or stop.is_set():
            state, error = "cancelled", "interrupted" if interrupted else None
            reporter.set_status("cancelled", "verl run cancelled", step=step)
        elif returncode == 0:
            state, error = "succeeded", None
            reporter.set_status("succeeded", "verl run finished", step=step)
        else:
            state = "failed"
            error = f"verl exited with code {returncode}\n" + "\n".join(tail)
            reporter.set_status("failed", f"verl exited with code {returncode}", step=step)
        metrics = {
            k: float(v) for k, v in last_metrics.items() if isinstance(v, (int, float)) and k != "step"
        }
        return RunResult(
            state=state,
            final_step=step,
            run_dir=str(run_dir),
            checkpoints=checkpoints,
            final_model_path=final_model_path(run_dir) if state == "succeeded" else None,
            error=error,
            metrics=metrics,
        )

    # -- internals ----------------------------------------------------------------------
    def _wait(self, run_dir: Path, reporter: RunReporter, stop: threading.Event) -> None:
        assert self._proc is not None
        proc = self._proc
        next_line = 0
        last_step = 0
        deadline: float | None = None
        while True:
            if proc.poll() is not None:
                break
            for i, row in iter_metrics(run_dir, start_line=next_line):
                next_line = i + 1
                last_step = max(last_step, int(row.get("step", 0)))
            status = reporter.status
            if last_step > int(status.get("step", 0)):
                reporter.set_status("running", step=last_step)
            link_checkpoints(run_dir)
            if self._cancel.is_set() or stop.is_set():
                if deadline is None:
                    proc.send_signal(signal.SIGTERM)
                    reporter.log("cancel requested: sent SIGTERM to verl driver")
                    deadline = time.monotonic() + self.grace_seconds
                elif time.monotonic() > deadline:
                    reporter.log("verl driver did not exit in time: sending SIGKILL")
                    kill_tree(proc)
            time.sleep(self.poll_seconds)
        proc.wait()

    @staticmethod
    def _tail_log(run_dir: Path, n: int = 20) -> list[str]:
        path = run_dir / LOG_FILE
        if not path.exists():
            return []
        with open(path, "rb") as f:
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        return lines[-n:]

    @staticmethod
    def _last_metrics(run_dir: Path) -> dict[str, Any]:
        last: dict[str, Any] = {}
        for _, row in iter_metrics(run_dir):
            if row.get("phase", "train") == "train":
                last = row
        return last


def make_backend(config: RunConfig, **kwargs: Any) -> VerlBackend:
    return VerlBackend(config, python=kwargs.get("python"))
