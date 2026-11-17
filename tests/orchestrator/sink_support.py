"""For the tests of ``monorl.orchestrator.sinks``: fakes of the two optional packages that reproduce
the upstream behaviour ``MetricSinks`` relies on, and readers of what the real packages wrote.

Upstream behaviour the fakes reproduce:

* ``wandb``: ``Run.log(data, step=N)`` merges the calls of one step into one row and ignores a step
  below the newest one; ``Run.finish(exit_code)`` ends the run.
* ``torch.utils.tensorboard.SummaryWriter``: writers of one directory append to what the directory
  shows, and ``purge_step=N`` hides what earlier writers logged at step N and later.
"""

from __future__ import annotations

import importlib.machinery
import json
import struct
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest


class FakeWandbRun:
    resumed = False

    def __init__(self, number: int, **init: Any) -> None:
        self.id = f"fake{number:04d}"
        self.dir = str(Path(init["dir"]) / "wandb" / f"offline-run-{self.id}" / "files")
        self.init = init
        self.history: dict[int, dict[str, float]] = {}
        self.exit_code: int | None = None

    def log(self, data: dict[str, float], step: int) -> None:
        assert self.exit_code is None, "logged to a finished run"
        if self.history and step < max(self.history):
            return
        self.history.setdefault(step, {}).update(data)

    def finish(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code or 0


class FakeSinkPackages:
    """``runs``: every W&B run in the order of ``wandb.init``. ``shown``: per log directory the
    ``(tag, step, value)`` points TensorBoard shows. ``writers``: every ``SummaryWriter`` that was
    opened; ``SummaryWriter``: their class."""

    def __init__(self) -> None:
        self.runs: list[FakeWandbRun] = []
        self.shown: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
        self.writers: list[Any] = []
        self.SummaryWriter: Any = None

    def run_named(self, name: str) -> FakeWandbRun:
        (run,) = [r for r in self.runs if r.init["name"] == name]
        return run


def install_fake_sink_packages(monkeypatch: pytest.MonkeyPatch) -> FakeSinkPackages:
    packages = FakeSinkPackages()

    def init(**kwargs: Any) -> FakeWandbRun:
        run = FakeWandbRun(len(packages.runs), **kwargs)
        packages.runs.append(run)
        return run

    class FakeSummaryWriter:
        def __init__(self, log_dir: str, purge_step: int | None = None) -> None:
            self.log_dir = log_dir
            self.closed = False
            if purge_step is not None:
                packages.shown[log_dir] = [p for p in packages.shown[log_dir] if p[1] < purge_step]
            packages.writers.append(self)

        def add_scalar(self, tag: str, value: float, step: int) -> None:
            assert not self.closed, "wrote to a closed writer"
            packages.shown[self.log_dir].append((tag, step, value))

        def close(self) -> None:
            self.closed = True

    packages.SummaryWriter = FakeSummaryWriter
    fake_wandb = types.ModuleType("wandb")
    fake_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)  # importlib.util.find_spec needs one
    fake_wandb.init = init  # type: ignore[attr-defined]
    fake_tensorboard = types.ModuleType("torch.utils.tensorboard")
    fake_tensorboard.SummaryWriter = FakeSummaryWriter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_tensorboard)
    return packages


def metric_points(run_dir: Path) -> dict[tuple[str, int], list[float]]:
    """``(name, step) -> values`` of every number in ``metrics.jsonl``, in the order of its rows."""
    points: dict[tuple[str, int], list[float]] = defaultdict(list)
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        for name, value in row.items():
            if name not in ("step", "time", "phase") and isinstance(value, (int, float)):
                points[(name, row["step"])].append(float(value))
    return points


def tensorboard_points(log_dir: Path) -> dict[tuple[str, int], list[float]]:
    """What TensorBoard shows of an event directory written by the real package."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    points: dict[tuple[str, int], list[float]] = defaultdict(list)
    for tag in accumulator.Tags()["scalars"]:
        for event in accumulator.Scalars(tag):
            points[(tag, event.step)].append(event.value)
    return points


def read_offline_wandb_runs(run_dir: Path) -> list[dict[str, Any]]:
    """The offline W&B runs under ``<run_dir>/wandb``, oldest first, each as ``{"id", "project",
    "name", "config", "history": {(name, step): value}, "exit_code"}``.

    wandb ships no reader for its ``run-<id>.wandb`` file: it is a LevelDB log (a 7-byte header,
    then 32768-byte blocks of chunks ``crc32 (4) | length (2, little endian) | type (1)`` where
    type 1 is a whole record and 2, 3, 4 the first, middle and last part of one) of serialized
    ``wandb_internal_pb2.Record`` messages.
    """
    from wandb.proto import wandb_internal_pb2

    runs = []
    for path in sorted((run_dir / "wandb").glob("offline-run-*/run-*.wandb")):
        data = path.read_bytes()
        assert data[:4] == b":W&B", data[:7]
        run: dict[str, Any] = {"config": {}, "history": {}, "exit_code": None}
        position, pending = 7, b""
        while position + 7 <= len(data):
            left_in_block = 32768 - position % 32768
            if left_in_block < 7:
                position += left_in_block
                continue
            length, kind = struct.unpack("<HB", data[position + 4 : position + 7])
            pending += data[position + 7 : position + 7 + length]
            position += 7 + length
            if kind not in (1, 4):
                continue
            record = wandb_internal_pb2.Record()
            record.ParseFromString(pending)
            pending = b""
            which = record.WhichOneof("record_type")
            if which == "run":
                run.update(id=record.run.run_id, project=record.run.project, name=record.run.display_name)
                for item in record.run.config.update:
                    if item.key != "_wandb":
                        run["config"][item.key] = json.loads(item.value_json)
            elif which == "history":
                step = record.history.step.num
                for item in record.history.item:
                    name = item.key or ".".join(item.nested_key)
                    if not name.startswith("_"):
                        run["history"][(name, step)] = json.loads(item.value_json)
            elif which == "exit":
                run["exit_code"] = record.exit.exit_code
        runs.append(run)
    return runs
