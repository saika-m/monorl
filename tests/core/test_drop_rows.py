"""What a resume does to ``metrics.jsonl`` and ``trajectories.jsonl``: the rows of the steps it runs
again go, and so does a line that a killed process did not finish."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from monorl.core.reporting import (
    METRICS_FILE,
    TRAJECTORIES_FILE,
    RunReporter,
    drop_rows,
    iter_metrics,
    iter_trajectories,
)


def _lines(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def _steps(rows: Any) -> list[tuple[int, str]]:
    return [(row["step"], row["phase"]) for _, row in rows]


def _killed_run(run_dir: Path) -> RunReporter:
    """A run with a checkpoint of step 2 that was killed while it wrote the rows of step 4: every
    step has a train and an eval row, and the last line of both files stops in the middle."""
    reporter = RunReporter(run_dir, echo=False)
    for step in (1, 2, 3):
        reporter.log_metrics(step, {"reward/mean": step / 4})
        reporter.log_metrics(step, {"eval/reward_mean": step / 8}, phase="eval")
    with open(run_dir / METRICS_FILE, "a") as f:
        f.write('{"step": 4, "time": 17.5, "phase": "tr')
    rows = [
        {"step": step, "phase": phase, "sample_id": f"s{i}", "response": "x" * 50}
        for step in (1, 2, 3)
        for phase in ("train", "eval")
        for i in range(2)
    ]
    (run_dir / TRAJECTORIES_FILE).write_text(
        _lines(rows) + '{"step": 4, "phase": "train", "sample_id": "s0", "resp'
    )
    return reporter


def test_resume_drops_the_rows_after_the_checkpoint_and_the_unfinished_line(tmp_path: Path) -> None:
    reporter = _killed_run(tmp_path)
    kept_metrics = (tmp_path / METRICS_FILE).read_text().splitlines(keepends=True)[:4]
    kept_trajectories = (tmp_path / TRAJECTORIES_FILE).read_text().splitlines(keepends=True)[:8]

    reporter.drop_rows_after(2)

    # the rows of steps 1 and 2 are the bytes they were, evaluation rows of step 2 included
    assert (tmp_path / METRICS_FILE).read_text() == "".join(kept_metrics)
    assert (tmp_path / TRAJECTORIES_FILE).read_text() == "".join(kept_trajectories)
    assert _steps(iter_metrics(tmp_path)) == [(1, "train"), (1, "eval"), (2, "train"), (2, "eval")]
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")) == []
    log = (tmp_path / "run.log").read_text()
    assert "after step 2" in log and f"3 of {METRICS_FILE}" in log and f"5 of {TRAJECTORIES_FILE}" in log

    # the resumed run writes step 3 again: once in the files, and readable
    reporter.log_metrics(3, {"reward/mean": 0.75})
    assert _steps(iter_metrics(tmp_path))[-1] == (3, "train")
    assert [row["step"] for _, row in iter_metrics(tmp_path)].count(3) == 1


def test_the_row_after_an_unfinished_line_of_an_earlier_step_is_readable(tmp_path: Path) -> None:
    """The unfinished line goes whatever its step: the next row would share its line and be lost
    to every reader."""
    reporter = RunReporter(tmp_path, echo=False)
    reporter.log_metrics(1, {"reward/mean": 0.25})
    with open(tmp_path / METRICS_FILE, "a") as f:
        f.write('{"step": 1, "time": 3.0, "pha')
    reporter.drop_rows_after(1)
    reporter.log_metrics(2, {"reward/mean": 0.5})
    assert [row["step"] for _, row in iter_metrics(tmp_path)] == [1, 2]


def test_resume_without_a_checkpoint_starts_the_files_over(tmp_path: Path) -> None:
    reporter = _killed_run(tmp_path)
    reporter.drop_rows_after(0)
    assert (tmp_path / METRICS_FILE).read_text() == ""
    assert list(iter_trajectories(tmp_path)) == []


def test_a_file_that_loses_nothing_is_not_rewritten(tmp_path: Path) -> None:
    reporter = RunReporter(tmp_path, echo=False)
    for step in (1, 2):
        reporter.log_metrics(step, {"reward/mean": 0.5})
    before = os.stat(tmp_path / METRICS_FILE)
    reporter.drop_rows_after(2)
    reporter.drop_rows_after(7)
    after = os.stat(tmp_path / METRICS_FILE)
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_size,
    )
    assert not (tmp_path / "run.log").exists()  # nothing to say
    assert not (tmp_path / TRAJECTORIES_FILE).exists()  # a file that is not there stays away


def test_drop_rows_keeps_what_it_cannot_read_and_completes_a_whole_last_row(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    whole_last_row = json.dumps({"step": 2, "phase": "eval"})  # written, but not its newline
    path.write_text(
        _lines([{"step": 1}]) + "not json\n\n" + _lines([{"step": 5}, {"note": "no step"}]) + whole_last_row
    )
    removed = drop_rows(path, lambda row: row.get("step", 0) > 2)
    assert removed == 1
    assert path.read_text() == _lines([{"step": 1}]) + "not json\n\n" + _lines([{"note": "no step"}]) + (
        whole_last_row + "\n"
    )


def test_drop_rows_of_a_file_larger_than_its_copy_buffer(tmp_path: Path) -> None:
    """The part before the first change is copied in blocks of 1 MiB: 3 MiB of rows to keep, then
    rows to drop in between rows to keep."""
    path = tmp_path / "rows.jsonl"
    keep = [{"step": 1, "i": i, "text": "k" * 1000} for i in range(3200)]
    tail = [{"step": 9, "i": 0}, {"step": 2, "i": 1}, {"step": 9, "i": 2}, {"step": 2, "i": 3}]
    path.write_text(_lines(keep + tail))
    assert path.stat().st_size > 3 * (1 << 20)
    assert drop_rows(path, lambda row: row["step"] > 2) == 2
    assert path.read_text() == _lines(keep + [tail[1], tail[3]])
