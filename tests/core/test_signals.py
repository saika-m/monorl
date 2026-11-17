"""``signal_guard``: a signal that arrives while the run is writing must not hang the run."""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

import pytest

from monorl.core.reporting import LOG_FILE, RunReporter
from monorl.core.signals import signal_guard


@pytest.mark.timeout(30)
def test_a_signal_during_a_write_of_the_reporter_cancels_instead_of_hanging(tmp_path: Path) -> None:
    """The handler runs on the main thread in the middle of whatever it was doing. Here that is a
    write of the reporter, whose lock the thread holds: a handler that logged would wait for it
    for ever."""
    reporter = RunReporter(tmp_path, echo=False)
    cancel = threading.Event()
    with signal_guard(cancel, reporter, "finishing the current step, then stopping"):
        with reporter._lock:  # the main thread is inside a write
            os.kill(os.getpid(), signal.SIGTERM)
            assert cancel.wait(10)  # the handler ran and returned
    log = (tmp_path / LOG_FILE).read_text()
    assert "received SIGTERM: finishing the current step, then stopping" in log


def test_a_second_signal_interrupts_and_the_handlers_are_restored(tmp_path: Path) -> None:
    """The second signal is announced like the first, and sets the event that tells the threads
    working for the block that the main thread was interrupted."""
    reporter = RunReporter(tmp_path, echo=False)
    cancel, interrupt = threading.Event(), threading.Event()
    before = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt, match="SIGINT"):
        with signal_guard(cancel, reporter, "stopping", interrupt=interrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            assert cancel.wait(10) and not interrupt.is_set()
            os.kill(os.getpid(), signal.SIGINT)
            threading.Event().wait(10)  # the second signal interrupts this
    assert interrupt.is_set()
    assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == before
    log = (tmp_path / LOG_FILE).read_text().splitlines()
    assert [line.split(" ", 2)[2] for line in log] == [
        "received SIGTERM: stopping",
        "received SIGINT again: stopping now",
    ]
