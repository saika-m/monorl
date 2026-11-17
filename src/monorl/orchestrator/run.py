"""``monorl.orchestrator.run``: the single entrypoint every launcher (CLI, server job, tests) calls.

Owns the run directory (:class:`~monorl.core.reporting.RunReporter`), signal handling and the
dispatch between pipeline backends (trl/verl/nemo_rl own their loop, SFT included), the SFT loop
(``algorithm.name: sft``, no rollouts) and the component RL loop.
"""

from __future__ import annotations

import os
import re
import threading
import time
import traceback
from pathlib import Path

from monorl.config.schema import RunConfig
from monorl.core.errors import BackendUnavailable, ConfigError, DataError, RegistryError
from monorl.core.registry import ensure_providers, registry
from monorl.core.reporting import (
    CONFIG_FILE,
    METRICS_FILE,
    TRAJECTORIES_FILE,
    RunReporter,
    drop_rows,
    list_checkpoints,
)
from monorl.core.signals import signal_guard
from monorl.core.types import RunResult
from monorl.orchestrator.loop import ComponentLoop
from monorl.orchestrator.sft import SFTLoop

USER_ERRORS = (ConfigError, DataError, RegistryError, BackendUnavailable)
"""What a run fails with when its config, its data or its installation has to change. The message
says what; the traceback, which only shows where monorl noticed, is left out of the log."""


def default_run_name(config: RunConfig) -> str:
    model = (
        re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(config.model.name_or_path.rstrip("/")).name).strip("-")
        or "model"
    )
    return f"{model}-{config.algorithm.name}-{time.strftime('%Y%m%d-%H%M%S')}"


def resolve_run_dir(config: RunConfig) -> Path:
    """``<logging.run_root>/<run_name>`` (run_name defaults to ``<model>-<algorithm>-<timestamp>``)."""
    return Path(config.logging.run_root).expanduser() / (config.run_name or default_run_name(config))


def check_local_model(config: RunConfig) -> None:
    """Raise :class:`ConfigError` when ``model.name_or_path`` or ``model.tokenizer_name_or_path``
    is written as a path (it starts with ``/``, ``./``, ``../`` or ``~``) and is not a directory.

    The loaders (transformers, and the frameworks that load through it) read a string that is no
    directory as a Hub id, and say of a path only that it is no valid repository name. They take
    the string as it is written: ``~`` is not expanded, so neither is it here. With
    ``cluster.ray_address`` the model is loaded on the nodes of that cluster, which may hold a path
    this machine does not, and nothing is checked."""
    if config.cluster.mode == "ray" and config.cluster.ray_address:
        return
    for key in ("name_or_path", "tokenizer_name_or_path"):
        path = getattr(config.model, key)
        if not path or not path.startswith(("/", "./", "../", "~")) or os.path.isdir(path):
            continue
        problem = "is a file, not a model directory" if os.path.isfile(path) else "does not exist"
        if path.startswith("~"):
            home = os.path.expanduser(path)
            hint = f"write {home}" if os.path.isdir(home) else f"{home} does not exist either"
            problem = f"does not exist: '~' is not expanded when a model is loaded ({hint})"
        elif not os.path.isabs(path):
            problem += f" (looked for {os.path.abspath(path)})"
        raise ConfigError(f"model.{key}: the local path {path} {problem}")


def run(
    config: RunConfig,
    run_dir: Path | None = None,
    *,
    resume: bool = False,
    cancel: threading.Event | None = None,
) -> RunResult:
    """Run ``config`` in ``run_dir``. ``cancel`` is the event that the first SIGINT or SIGTERM sets
    while the run is in progress. A caller with more to do after this run (the pipeline, between
    its stages) passes its own: the result alone does not tell it that it was asked to stop, since
    a run that was in its last step when the signal came ends ``succeeded``."""
    if config.run_name is None:
        config = config.model_copy(update={"run_name": default_run_name(config)})
    target = Path(run_dir).expanduser() if run_dir is not None else resolve_run_dir(config)
    reporter = RunReporter(target)
    config.to_yaml(reporter.run_dir / CONFIG_FILE)
    if not resume:
        # This run appends to the records of a run that was here before. A line that one did not
        # finish goes first, or the first row of this run would share it and be unreadable. (A
        # resume does the same when it takes the files back to its checkpoint.)
        for name in (METRICS_FILE, TRAJECTORIES_FILE):
            if drop_rows(reporter.run_dir / name, lambda row: False):
                reporter.log(f"removed the unfinished last line an earlier run left in {name}")
    if cancel is None:
        cancel = threading.Event()
    backend = config.trainer.backend
    kind = "pipeline" if config.is_pipeline_backend else "sft" if config.is_sft else "component"
    with signal_guard(cancel, reporter, "finishing the current step, then stopping"):
        reporter.set_status("running", f"starting {kind} backend {backend!r} (resume={resume})", step=0)
        try:
            check_local_model(config)
            ensure_providers(config)
            if config.is_pipeline_backend:
                result = _run_pipeline(config, reporter, resume)
            elif config.is_sft:
                result = SFTLoop(config, reporter, cancel=cancel, resume=resume).run()
            else:
                result = ComponentLoop(config, reporter, cancel=cancel, resume=resume).run()
        except KeyboardInterrupt:
            result = RunResult(
                state="cancelled",
                final_step=int(reporter.status.get("step", 0)),
                run_dir=str(reporter.run_dir),
                checkpoints=list_checkpoints(reporter.run_dir),
                error="interrupted",
            )
        except Exception as e:  # noqa: BLE001 - every failure must land in status.json
            error = f"{type(e).__name__}: {e}"
            if not isinstance(e, USER_ERRORS):
                tb = traceback.format_exc()
                reporter.log(tb)
                error = f"{error}\n{tb}"
            result = RunResult(
                state="failed",
                final_step=int(reporter.status.get("step", 0)),
                run_dir=str(reporter.run_dir),
                checkpoints=list_checkpoints(reporter.run_dir),
                error=error,
            )
            reporter.set_status("failed", f"{type(e).__name__}: {e}", step=result.final_step)
            return result
    message = {
        "succeeded": f"finished {result.final_step} steps",
        "cancelled": f"cancelled after step {result.final_step}",
        "failed": result.error or "failed",
    }.get(result.state, result.state)
    reporter.set_status(result.state, message, step=result.final_step)
    return result


def _run_pipeline(config: RunConfig, reporter: RunReporter, resume: bool) -> RunResult:
    backend = registry.get("pipeline_backend", config.trainer.backend)(config)
    result = backend.run(config, reporter.run_dir, reporter=reporter, resume=resume)
    if not result.checkpoints:
        result.checkpoints = list_checkpoints(reporter.run_dir)
    return result
