"""``rollout.turn_batching``: what reaches an engine, and in which order, when episodes run
concurrently. An engine's sampling depends on both (one random stream per call, one row per
prompt), so a run repeats only if neither depends on how long anything took."""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import fakes
import pytest

from monorl.core.types import Completion, Sample, SamplingParams, StepResult
from monorl.orchestrator.episodes import EpisodeFailure, EpisodeSpec, run_episodes
from monorl.rollout import BatchingEngine


class OrderSensitiveEngine:
    """What a completion says depends on the number of the call and on the prompt's row in it,
    like a sample drawn from one random stream per call. ``batches`` lists the prompts of every
    call, as the text of their last user turn."""

    def __init__(self, latency: float = 0.0) -> None:
        self.tokenizer = fakes.FakeTokenizer()
        self.latency = latency
        self.batches: list[list[str]] = []
        self._lock = threading.Lock()

    def generate(self, prompts: Any, params: SamplingParams) -> list[list[Completion]]:
        texts = [fakes.last_user_text(self.tokenizer.decode(list(p))) for p in prompts]
        with self._lock:
            call = len(self.batches)
            self.batches.append(texts)
        time.sleep(self.latency)
        out = []
        for row, prompt in enumerate(prompts):
            text = f"call{call} row{row}"
            ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            out.append(
                [
                    Completion(
                        text=text,
                        token_ids=ids,
                        logprobs=[-0.1] * len(ids),
                        finish_reason="stop",
                        prompt_token_ids=list(prompt),
                    )
                ]
            )
        return out

    def shutdown(self) -> None:
        pass


class JitteryEnv:
    """An episode of sample ``i`` lasts ``1 + i % 3`` turns; every ``reset`` and ``step`` takes a
    random time, so that nothing about the order in which episodes become ready is stable."""

    max_turns = 4

    def __init__(self, jitter: random.Random, fail_first_reset_of: str | None = None) -> None:
        self.jitter = jitter
        self.fail_first_reset_of = fail_first_reset_of
        self.turns = 0
        self.length = 0
        self.name = ""

    def _wait(self) -> None:
        time.sleep(self.jitter.uniform(0.0, 0.02))

    def reset(self, sample: Sample) -> list[dict[str, Any]]:
        self._wait()
        if sample.id == self.fail_first_reset_of and sample.id not in FAILED_ONCE:
            FAILED_ONCE.add(sample.id)
            raise RuntimeError("the environment was not up yet")
        self.name, self.length = sample.id, 1 + int(sample.id) % 3
        return list(sample.messages)

    def step(self, action: Completion) -> StepResult:
        self._wait()
        self.turns += 1
        done = self.turns >= self.length
        messages = [] if done else [{"role": "user", "content": f"e{self.name}t{self.turns}"}]
        return StepResult(messages=messages, reward=0.0, done=done, info={})

    def close(self) -> None:
        pass


FAILED_ONCE: set[str] = set()


def _specs(engines: list[Any], count: int) -> list[EpisodeSpec]:
    return [
        EpisodeSpec(
            sample=Sample(id=str(i), messages=[{"role": "user", "content": f"e{i}t0"}], reference=""),
            group_id=f"g{i}",
            engine=engines[i % len(engines)],
        )
        for i in range(count)
    ]


def _run(seed: int, *, turn_batching: str, engines: int = 1, max_parallel: int = 3, **env: Any) -> Any:
    FAILED_ONCE.clear()
    jitter = random.Random(seed)
    lock = threading.Lock()

    def factory() -> JitteryEnv:
        with lock:  # one stream of delays, drawn in whatever order the threads get here
            return JitteryEnv(random.Random(jitter.random()), **env)

    made = [OrderSensitiveEngine() for _ in range(engines)]
    failures: list[EpisodeFailure] = []
    trajectories = run_episodes(
        _specs(made, 8),
        env_factory=factory,
        tokenizer=made[0].tokenizer,
        params=SamplingParams(n=1),
        max_parallel=max_parallel,
        retries=1,
        failures=failures,
        turn_batching=turn_batching,  # type: ignore[arg-type]
    )
    assert not failures
    said = [[turn.completion.text for turn in t.turns] for t in trajectories]
    return [engine.batches for engine in made], said


def test_lockstep_calls_the_engine_with_the_same_batches_whatever_the_timing() -> None:
    batches, said = _run(1, turn_batching="lockstep")
    # three episodes at a time; a finished episode's place is taken before the next call
    assert batches[0][:3] == [
        ["e0t0", "e1t0", "e2t0"],
        ["e1t1", "e2t1", "e3t0"],
        ["e2t2", "e4t0", "e5t0"],
    ]
    assert [len(turns) for turns in said] == [1, 2, 3, 1, 2, 3, 1, 2]
    for seed in (2, 3, 4):
        assert _run(seed, turn_batching="lockstep") == (batches, said)


def test_lockstep_keeps_the_batches_of_an_engine_independent_of_the_other_engines() -> None:
    batches, said = _run(1, turn_batching="lockstep", engines=2, max_parallel=4)
    # every engine takes the turns of its own episodes together: two at a time each
    assert batches[0][:2] == [["e0t0", "e2t0"], ["e2t1", "e4t0"]]
    assert batches[1][:2] == [["e1t0", "e3t0"], ["e1t1", "e5t0"]]
    for seed in (2, 3):
        assert _run(seed, turn_batching="lockstep", engines=2, max_parallel=4) == (batches, said)


def test_lockstep_restarts_a_failed_episode_in_its_place() -> None:
    batches, said = _run(1, turn_batching="lockstep", fail_first_reset_of="1")
    assert batches[0][0] == ["e0t0", "e1t0", "e2t0"]  # the retry was ready before the first call
    assert [len(turns) for turns in said] == [1, 2, 3, 1, 2, 3, 1, 2]
    assert _run(2, turn_batching="lockstep", fail_first_reset_of="1") == (batches, said)


def test_arrival_gives_every_episode_its_completions_without_waiting_for_the_others() -> None:
    """One slow environment holds nobody back: the other episodes finish while it is in ``step``."""
    engine = OrderSensitiveEngine()
    proxy = BatchingEngine(engine, max_batch_size=8, max_wait_ms=1)
    in_step, release = threading.Event(), threading.Event()
    finished: list[str] = []

    class SlowFirst(JitteryEnv):
        def step(self, action: Completion) -> StepResult:
            if self.name == "0":
                in_step.set()
                assert release.wait(10)
            result = super().step(action)
            if result.done:
                finished.append(self.name)
                if len(finished) == 3:  # every episode but the slow one
                    release.set()
            return result

    try:
        trajectories = run_episodes(
            _specs([proxy], 4),
            env_factory=lambda: SlowFirst(random.Random(0)),
            tokenizer=engine.tokenizer,
            params=SamplingParams(n=1),
            max_parallel=4,
            turn_batching="arrival",
        )
    finally:
        proxy.shutdown()
    assert in_step.is_set() and finished[-1] == "0"
    assert [len(t.turns) for t in trajectories] == [1, 2, 3, 1]


@pytest.mark.parametrize("turn_batching", ["lockstep", "arrival"])
def test_both_ways_return_the_trajectories_in_spec_order(turn_batching: str) -> None:
    _, said = _run(5, turn_batching=turn_batching, max_parallel=8)
    assert [len(turns) for turns in said] == [1, 2, 3, 1, 2, 3, 1, 2]


@pytest.mark.timeout(60)
def test_lockstep_opens_no_more_episodes_than_the_environment_admits() -> None:
    """A server with two places makes the third ``reset`` wait for an episode to end. In lockstep
    that episode ends only after the turn the waiting one holds up, so the runner asks first."""
    from monorl.envs.remote import RemoteEnv
    from monorl.envs.server import MEPServer, start_http_server
    from monorl.envs.tool_calling import make_tool_calling_env

    server = MEPServer(
        lambda: make_tool_calling_env({"tools": ["calculator"]}, max_turns=2), parallel_episodes=2
    )
    handle = start_http_server(server, port=0)
    engine = OrderSensitiveEngine()
    open_at_once: list[int] = []

    class Counting(RemoteEnv):
        def step(self, action: Completion) -> StepResult:
            open_at_once.append(server.active_episodes)
            return super().step(action)

    try:
        trajectories = run_episodes(
            _specs([engine], 6),
            env_factory=lambda: Counting.from_params(
                {"transport": "http", "url": handle.url, "timeout_s": 30}
            ),
            tokenizer=engine.tokenizer,
            params=SamplingParams(n=1),
            max_parallel=6,
            turn_batching="lockstep",
        )
    finally:
        handle.stop()
    assert len(trajectories) == 6 and max(open_at_once) == 2
    assert [len(batch) for batch in engine.batches[:2]] == [2, 2]
    assert server.active_episodes == 0
