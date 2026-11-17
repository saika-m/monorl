"""The second SIGINT or SIGTERM stops a run at once, also while other threads work for it: the
generation in flight is aborted through the engine's ``cancel()`` instead of being waited for,
and the run ends ``cancelled`` with the files of the steps it finished.

The signals are sent from inside the call they interrupt, the second once the first was handled,
so nothing depends on timing. A run that waits for the generation ends too, when the fake engine
gives up waiting to be cancelled, and fails its test: the call is then in ``engine.calls``.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import fakes
import pytest

from monorl.core.errors import GenerationCancelled
from monorl.core.registry import registry
from monorl.core.reporting import LOG_FILE, iter_metrics, iter_trajectories, list_checkpoints, read_status
from monorl.core.types import Completion, SamplingParams
from monorl.orchestrator import loop, run
from monorl.rollout import BatchingEngine

RUN_THREADS = ("monorl-episode", "monorl-rollout-producer", "monorl-batching")
GENERATE_SECONDS = 30.0
"""How long a slow call of the fake engine waits to be cancelled before it completes."""


def two_signals(cancel: threading.Event) -> None:
    """Send the main thread SIGINT twice, the second time once its handler took the first."""
    main = threading.main_thread().ident
    assert main is not None
    signal.pthread_kill(main, signal.SIGINT)
    assert cancel.wait(30)
    signal.pthread_kill(main, signal.SIGINT)


class SlowEngine(fakes.FakeEngine):
    """A fake engine with ``cancel()``. The ``generate`` calls that ``slow`` picks take
    :data:`GENERATE_SECONDS` unless they are cancelled, like a real engine that is generating;
    with ``signals``, the first of them (of all engines) sends the two signals once it is in
    flight."""

    slow: ClassVar[Callable[[SamplingParams, int], bool]]
    """``slow(params, k)``: whether the engine's ``k``-th call (from 0), with ``params``, is slow."""
    signals: ClassVar[bool]
    cancel_event: ClassVar[threading.Event]
    """The event the run was started with, which the first signal sets."""
    class_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.in_flight = threading.Event()
        self.cancelled = threading.Event()
        self.cancel_calls = 0
        self.generate_calls = 0
        super().__init__(*args, **kwargs)

    def generate(
        self, prompts: Sequence[Any], params: SamplingParams, *, max_new_tokens: Sequence[int] | None = None
    ) -> list[list[Completion]]:
        cls = type(self)
        with cls.class_lock:
            slow = cls.slow(params, self.generate_calls)
            self.generate_calls += 1
            signals = slow and cls.signals
            if signals:
                cls.signals = False
        if slow:
            self.cancelled.clear()
            self.in_flight.set()
            if signals:
                two_signals(cls.cancel_event)
            if self.cancelled.wait(GENERATE_SECONDS):
                raise GenerationCancelled("fake generation cancelled")
        return super().generate(prompts, params, max_new_tokens=max_new_tokens)

    def cancel(self) -> None:
        self.cancel_calls += 1
        self.cancelled.set()


@pytest.fixture
def slow_engine(
    monkeypatch: pytest.MonkeyPatch, fake_components: Any
) -> Iterator[Callable[..., threading.Event]]:
    """``slow_engine(slow, signals=...)`` registers :class:`SlowEngine` as the ``fake`` engine,
    behind the real :class:`BatchingEngine` as in a run, and returns the event to start the run
    with."""
    monkeypatch.setattr(loop, "_resolve_batching_engine", lambda: BatchingEngine)

    def install(
        slow: Callable[[SamplingParams, int], bool] = lambda params, k: True, *, signals: bool = True
    ) -> threading.Event:
        SlowEngine.slow, SlowEngine.signals = staticmethod(slow), signals  # type: ignore[assignment]
        SlowEngine.cancel_event = threading.Event()
        registry.register("rollout_engine", "fake", SlowEngine, overwrite=True)
        return SlowEngine.cancel_event

    yield install
    registry.register("rollout_engine", "fake", fakes.FakeEngine, overwrite=True)


def _assert_stopped_at_once(result: Any, *, final_step: int) -> Path:
    """The run ended cancelled and said why, it cancelled the generation and left no thread
    behind, and its files are those of the ``final_step`` steps it finished."""
    run_dir = Path(result.run_dir)
    assert result.state == "cancelled" and result.final_step == final_step
    status = read_status(run_dir)
    assert status is not None and status["state"] == "cancelled" and status["step"] == final_step
    assert sum(engine.cancel_calls for engine in fakes.ENGINES) >= 1
    assert all(engine.shutdown_count >= 1 for engine in fakes.ENGINES)
    assert not [t.name for t in threading.enumerate() if t.name.startswith(RUN_THREADS)]
    steps = list(range(1, final_step + 1))
    assert [row["step"] for _, row in iter_metrics(run_dir)] == steps  # and no row of an evaluation
    assert sorted({row["step"] for _, row in iter_trajectories(run_dir)}) == steps
    assert [Path(p).name for p in list_checkpoints(run_dir)] == [f"step-{n}" for n in steps]
    assert not list((run_dir / "checkpoints").glob("*.tmp")) and not (run_dir / "final").exists()
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    assert "received SIGINT again: stopping now" in (run_dir / LOG_FILE).read_text()
    return run_dir


@pytest.mark.parametrize(
    "rollout",
    [
        {"turn_batching": "lockstep"},
        {"turn_batching": "arrival"},
        {"turn_batching": "lockstep", "data_parallel_size": 2},
    ],
    ids=["lockstep", "arrival", "two-engines"],
)
def test_second_signal_aborts_the_generation_of_a_rollout(
    rollout: dict[str, Any], make_config: Any, slow_engine: Any
) -> None:
    cancel = slow_engine()
    result = run(make_config(trainer={"max_steps": 3}, rollout=rollout), cancel=cancel)
    _assert_stopped_at_once(result, final_step=0)
    assert all(engine.calls == [] for engine in fakes.ENGINES)  # nothing was generated to the end


def test_second_signal_aborts_the_generation_of_an_evaluation(
    make_config: Any, slow_engine: Any, tmp_path: Path
) -> None:
    cancel = slow_engine(lambda params, k: params.temperature == 0.0)  # the greedy calls
    eval_file = fakes.write_jsonl(tmp_path / "eval.jsonl", fakes.DEFAULT_ROWS[:2])
    config = make_config(trainer={"max_steps": 3, "eval_every": 1}, data={"eval_uri": str(eval_file)})
    result = run(config, cancel=cancel)
    _assert_stopped_at_once(result, final_step=1)
    assert [call["temperature"] for call in fakes.ENGINES[-1].calls] == [1.0]  # the rollout of step 1


def test_second_signal_in_a_train_step_aborts_the_rollout_that_runs_ahead(
    make_config: Any, slow_engine: Any
) -> None:
    """Async rollouts: the signals reach the main thread in the train step of step 1, while the
    producer thread generates the rollout of step 2."""
    cancel = slow_engine(lambda params, k: k > 0, signals=False)

    class Trainer(fakes.FakeTrainer):
        def train_step(self, batch: Any) -> dict[str, float]:
            assert fakes.ENGINES[-1].in_flight.wait(30)
            two_signals(cancel)
            raise AssertionError("the second signal did not interrupt the train step")

    registry.register("trainer", "fake", Trainer, overwrite=True)
    try:
        result = run(make_config(trainer={"max_steps": 3}, rollout={"async_steps_ahead": 1}), cancel=cancel)
    finally:
        registry.register("trainer", "fake", fakes.FakeTrainer, overwrite=True)
    _assert_stopped_at_once(result, final_step=0)
    assert len(fakes.ENGINES[-1].calls) == 1  # the rollout of step 1


def test_second_signal_during_the_checkpoint_of_a_cancelled_run_leaves_no_checkpoint_of_that_step(
    make_config: Any,
) -> None:
    """The first signal ends the run after step 2 with a checkpoint of it; the second arrives
    while that checkpoint is being written. What was written of it is not a checkpoint, and a
    resume starts over and leaves the records of a run that was never interrupted."""
    cancel = threading.Event()
    sent: list[str] = []

    class Trainer(fakes.FakeTrainer):
        def train_step(self, batch: Any) -> dict[str, float]:
            if self._step == 1 and not sent:  # in step 2
                sent.append("first")
                signal.raise_signal(signal.SIGINT)
            return super().train_step(batch)

        def save_checkpoint(self, path: Path) -> None:
            if sent == ["first"]:
                sent.append("second")
                (Path(path) / "half-written").write_text("")
                signal.raise_signal(signal.SIGINT)
                raise AssertionError("the second signal did not interrupt the save")
            super().save_checkpoint(path)

    registry.register("trainer", "fake", Trainer, overwrite=True)
    try:
        config = make_config(trainer={"max_steps": 4, "save_every": 10})
        result = run(config, cancel=cancel)
        run_dir = Path(result.run_dir)
        assert result.state == "cancelled" and result.final_step == 2
        assert read_status(run_dir)["state"] == "cancelled"
        assert "received SIGINT again: stopping now" in (run_dir / LOG_FILE).read_text()
        assert list_checkpoints(run_dir) == [] and result.checkpoints == []
        assert [p.name for p in (run_dir / "checkpoints").iterdir()] == ["step-2.tmp"]
        resumed = run(config, resume=True)
    finally:
        registry.register("trainer", "fake", fakes.FakeTrainer, overwrite=True)
    assert resumed.state == "succeeded" and resumed.final_step == 4
    assert [row["step"] for _, row in iter_metrics(run_dir)] == [1, 2, 3, 4]
    assert not list((run_dir / "checkpoints").glob("*.tmp"))
