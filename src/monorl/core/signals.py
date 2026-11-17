"""SIGINT and SIGTERM as a request to stop: the first one sets an event, a second one interrupts."""

from __future__ import annotations

import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from monorl.core.reporting import RunReporter

_AGAIN = 0x80
"""Set in the byte the handler passes on for a signal that is not the first (signal numbers are
below it)."""


@contextmanager
def signal_guard(
    cancel: threading.Event,
    reporter: RunReporter,
    then: str,
    *,
    interrupt: threading.Event | None = None,
) -> Iterator[None]:
    """While the block runs, the first SIGINT or SIGTERM sets ``cancel`` and is noted in the run
    log (``received <NAME>: <then>``, written before the block is left); a second one raises
    ``KeyboardInterrupt`` for an immediate stop and is noted too (``received <NAME> again:
    stopping now``). Only the main thread may install handlers; on any other thread the block
    simply runs.

    ``interrupt`` is set by the second signal before the exception is raised. The exception
    reaches the main thread only; the event is for what other threads do for the block, which the
    main thread would otherwise wait for on its way out (the episodes of a rollout, the engine
    call that serves them).

    The handler does not write the log line itself. It runs on the main thread, between two
    instructions of whatever that thread is doing; if that is a write of the reporter, the
    reporter's lock is held by the very thread that would now wait for it, and the run would
    hang until it is killed. The handler hands the signal to a thread through a pipe instead.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    wake_r, wake_w = os.pipe()

    def handler(signum: int, _frame: object) -> None:
        if cancel.is_set():
            if interrupt is not None:
                interrupt.set()
            os.write(wake_w, bytes([signum | _AGAIN]))
            raise KeyboardInterrupt(signal.Signals(signum).name)
        cancel.set()
        os.write(wake_w, bytes([signum]))

    def announce() -> None:
        while (received := os.read(wake_r, 1)) != b"\0":  # the block writes a zero when it ends
            name = signal.Signals(received[0] & ~_AGAIN).name
            reporter.log(
                f"received {name} again: stopping now" if received[0] & _AGAIN else f"received {name}: {then}"
            )

    announcer = threading.Thread(target=announce, name="monorl-signal-log", daemon=True)
    announcer.start()
    previous = {sig: signal.signal(sig, handler) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, prev in previous.items():
            signal.signal(sig, prev)
        os.write(wake_w, b"\0")
        announcer.join()
        os.close(wake_r)
        os.close(wake_w)


__all__ = ["signal_guard"]
