"""Parallel episode runner.

Each episode owns one environment instance. A stepped :class:`~monorl.core.interfaces.Environment`
follows the multi-turn loop from its docstring; an :class:`~monorl.core.interfaces.EpisodeDriver`
(an external agent harness behind the capture gateway) runs the whole episode itself.

Turn batching (``rollout.turn_batching``). ``lockstep``: the episodes of an engine take their
turns together. Environment work (``reset``, ``step``, rendering the next prompt) runs on a thread
pool; once every running episode of the engine has its next prompt, the prompts go to the engine
as one ``generate`` call per distinct :class:`SamplingParams`, in the order of the specs. Which
prompts share a call, and in which order, therefore never depends on timing, and neither does the
set of running episodes: a finished episode's place is taken by the next spec before the next
call. A run with a fixed seed repeats as long as its environments and engines are deterministic.
The price is that every episode waits for the slowest environment step of its round.

``arrival``: every episode runs on its own thread and asks the engine for one completion at a time,
as soon as its environment has answered; a :class:`monorl.rollout.BatchingEngine` coalesces the
requests that arrive together. Nothing waits for a slow environment, and nothing repeats. An
:class:`EpisodeDriver` always runs this way: its harness decides when it calls the policy.

Engines are independent: with several engines every engine batches the turns of its own episodes
and shares ``max_parallel`` evenly with the others (at least one episode each).

Text prompts are sent to the engine as token ids: turn 0 is rendered with :func:`chat_prompt_ids`
(with the environment's tool schemas, if any), every later turn is built by
:func:`monorl.rollout.render.continue_prompt_ids`. With ``rollout.turn_rendering: tokens`` the
previous prompt and the sampled completion stay in the context token for token and only the new
messages are rendered, so a turn's prompt is an exact extension of the tokens the policy was
sampled on and the batch builder trains the whole episode as one sequence. ``template`` re-renders
the conversation from its messages every turn instead.

Multimodal chats (image parts) are sent to the engine as chats on **every** turn: raw ids cannot
carry pixel tensors, so the engine re-runs the processor over the whole chat (image preprocessing
and placeholder expansion included) each turn and records the ids it conditioned on. This costs
one processor pass per turn and forgoes prefix caching. Tool schemas cannot be passed through
``RolloutEngine.generate``, so multimodal episodes against tool-exposing environments are
rejected.

Context budget (``max_context_tokens``): no prompt plus completion of a turn may exceed it. A
turn's generation is capped to the room that is left; an episode whose next prompt leaves no room
ends there with what it has (``Trajectory.info["context_exhausted"]``, ``done=False``); a first
prompt that does not fit is a failure of the episode. Multimodal prompts are tokenised by the
engine, so the budget is not applied to them.

Failures. An abort (:class:`EpisodeAborted`) and an engine failure
(:class:`~monorl.core.errors.GenerationFailed`) always end the batch. Anything else an episode
raises is a failure of its environment — a crashing ``reset``/``step``, an agent that dies, a
conversation the chat template rejects — and is retried ``retries`` times with a fresh
environment. What still fails is raised, or, when the caller passes a ``failures`` list, recorded
there and left out of the result.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Literal

from monorl.core.errors import DataError, EpisodeAborted, GenerationCancelled, GenerationFailed, MonorlError
from monorl.core.interfaces import Environment, EpisodeDriver, RolloutEngine
from monorl.core.types import (
    Completion,
    Message,
    PromptInput,
    Sample,
    SamplingParams,
    Trajectory,
    Turn,
    messages_have_media,
)
from monorl.rollout.batching import params_key
from monorl.rollout.render import chat_prompt_ids, continue_prompt_ids

log = logging.getLogger(__name__)

CANCEL_INTERVAL_S = 0.05
"""How often ``run_episodes`` re-issues ``engine.cancel()`` while an abort is being carried out."""

TurnBatching = Literal["lockstep", "arrival"]

_Request = tuple[PromptInput, SamplingParams]
"""What an episode asks the engine for: one prompt and the sampling parameters of that turn."""


@dataclass(slots=True)
class EpisodeSpec:
    sample: Sample
    group_id: str
    engine: RolloutEngine
    """Engine (normally a ``BatchingEngine``) this episode generates with."""


@dataclass(slots=True)
class EpisodeFailure:
    """An episode whose environment failed on every attempt."""

    spec: EpisodeSpec
    error: Exception
    attempts: int


def run_episode(
    spec: EpisodeSpec,
    *,
    env_factory: Callable[[], Environment | EpisodeDriver],
    tokenizer: Any,
    params: SamplingParams,
    abort: threading.Event | None = None,
    turn_rendering: str = "tokens",
    max_context_tokens: int | None = None,
    retries: int = 0,
) -> Trajectory:
    """One episode; an environment failure is retried ``retries`` times, each time with a fresh
    environment, and raised when the last attempt fails too (module docstring)."""
    return run_episodes(
        [spec],
        env_factory=env_factory,
        tokenizer=tokenizer,
        params=params,
        max_parallel=1,
        abort=abort,
        turn_rendering=turn_rendering,
        max_context_tokens=max_context_tokens,
        retries=retries,
    )[0]


def _ask(spec: EpisodeSpec, prompts: Sequence[PromptInput], params: SamplingParams) -> list[Completion]:
    """One completion per prompt from the spec's engine; ``spec`` names the call in errors."""
    try:
        outputs = spec.engine.generate(prompts, params)
    except GenerationCancelled as e:
        raise EpisodeAborted(f"episode {spec.group_id} aborted during generation") from e
    except Exception as e:
        raise GenerationFailed(f"episode {spec.group_id}: the rollout engine failed: {e}") from e
    if len(outputs) != len(prompts):
        raise GenerationFailed(
            f"episode {spec.group_id}: the rollout engine returned {len(outputs)} results for "
            f"{len(prompts)} prompts"
        )
    return [completions[0] for completions in outputs]


def _drive(turns: Generator[_Request, Completion, Trajectory], spec: EpisodeSpec) -> Trajectory:
    """Run an episode's turns to the end, asking the engine for one completion at a time."""
    try:
        prompt, params = next(turns)
        while True:
            prompt, params = turns.send(_ask(spec, [prompt], params)[0])
    except StopIteration as stop:
        return stop.value


def _turns(
    env: Environment,
    spec: EpisodeSpec,
    *,
    tokenizer: Any,
    params: SamplingParams,
    abort: threading.Event | None,
    turn_rendering: str,
    max_context_tokens: int | None,
) -> Generator[_Request, Completion, Trajectory]:
    """The turns of a stepped episode: yields what the next turn asks of the engine, is sent the
    completion, and returns the trajectory. Everything between two yields is environment work."""
    messages: list[Message] = list(env.reset(spec.sample))
    # ``[]`` (remote/OpenEnv environments without tools) means no tool schemas, like ``None``.
    tools = getattr(env, "tools", None) or None
    max_turns = max(1, int(getattr(env, "max_turns", 1)))
    turns: list[Turn] = []
    done = False
    context_exhausted = False
    answered_at = 0  # index of the previous turn's assistant message in ``messages``
    for _ in range(max_turns):
        if abort is not None and abort.is_set():
            raise EpisodeAborted(f"episode {spec.group_id} aborted")
        prompt: PromptInput
        turn_params = params
        multimodal = messages_have_media(messages)
        if multimodal and tools is not None:
            raise MonorlError(
                f"episode {spec.group_id}: multimodal chats cannot be combined with environment tool schemas"
            )
        if multimodal:
            prompt = list(messages)
        else:
            if not turns:
                prompt = chat_prompt_ids(tokenizer, messages, tools)
            else:
                prompt = continue_prompt_ids(
                    tokenizer,
                    previous_messages=messages[:answered_at],
                    previous_prompt_ids=turns[-1].prompt_token_ids,
                    completion=turns[-1].completion,
                    new_messages=messages[answered_at + 1 :],
                    tools=tools,
                    mode=turn_rendering,
                ).ids
            if max_context_tokens is not None:
                room = max_context_tokens - len(prompt)
                if room < 1:
                    if not turns:
                        raise DataError(
                            f"episode {spec.group_id}: the prompt has {len(prompt)} tokens but the context budget "
                            f"is {max_context_tokens} (rollout.max_context_tokens, by default "
                            "trainer.max_seq_len); shorten the prompt (data.max_prompt_tokens) or raise the budget"
                        )
                    context_exhausted = True
                    break
                if room < params.max_new_tokens:
                    turn_params = dataclasses.replace(params, max_new_tokens=room)
        completion = yield prompt, turn_params
        if completion.prompt_token_ids is not None:
            prompt_ids = list(completion.prompt_token_ids)
        elif multimodal:
            prompt_ids = chat_prompt_ids(tokenizer, messages, None)
        else:
            prompt_ids = list(prompt)  # type: ignore[arg-type]
        answered_at = len(messages)
        messages.append(_assistant_message(completion))
        result = env.step(completion)
        turns.append(
            Turn(
                prompt_token_ids=prompt_ids,
                completion=completion,
                reward=float(result.reward),
                info=dict(result.info),
                message_index=answered_at,
            )
        )
        messages.extend(result.messages)
        done = bool(result.done)
        if completion.finish_reason == "length" and turn_params is not params:
            context_exhausted = True  # cut short by the budget, not by rollout.max_new_tokens
        if done or completion.finish_reason in ("length", "error"):
            break
    info: dict[str, Any] = {"finish_reason": turns[-1].completion.finish_reason, "num_turns": len(turns)}
    if context_exhausted:
        info["context_exhausted"] = True
    if tools is not None:
        info["tools"] = tools
    return Trajectory(
        sample=spec.sample,
        group_id=spec.group_id,
        messages=messages,
        turns=turns,
        done=done and not context_exhausted,
        info=info,
    )


@dataclass(slots=True)
class _Live:
    """An episode that is running: ``request`` is set while it waits for its turn's completion."""

    index: int
    spec: EpisodeSpec
    attempt: int = 0
    env: Environment | EpisodeDriver | None = None
    turns: Generator[_Request, Completion, Trajectory] | None = None
    request: _Request | None = None


@dataclass(slots=True)
class _Outcome:
    """What the runs of one :func:`run_episodes` call share. ``fatal`` and ``failed`` are appended
    to by several threads and read once all of them have ended."""

    trajectories: list[Trajectory | None]
    fatal: list[tuple[int, BaseException]] = field(default_factory=list)
    """``(spec index, error)`` of what ends the whole call."""
    failed: list[tuple[int, EpisodeFailure]] = field(default_factory=list)
    halt: threading.Event = field(default_factory=threading.Event)
    """Set with the first fatal error: nothing new is started."""


class _Environments:
    """The environment factory of one :func:`run_episodes` call. :meth:`limit` has to make an
    environment to ask it; the first episode that needs one gets that one."""

    def __init__(self, factory: Callable[[], Environment | EpisodeDriver]) -> None:
        self._factory = factory
        self._made: list[Environment | EpisodeDriver] = []
        self._lock = threading.Lock()

    def limit(self) -> int | None:
        """``max_concurrent_episodes`` of the environments (``None``: they do not say, or the
        first one could not be made, which the episode that needs it will report)."""
        try:
            env = self._factory()
        except Exception:  # noqa: BLE001
            return None
        with self._lock:
            self._made.append(env)
        try:
            limit = getattr(env, "max_concurrent_episodes", None)
        except Exception:  # noqa: BLE001 - the same failure will be the episode's
            return None
        return max(1, int(limit)) if limit is not None else None

    def make(self) -> Environment | EpisodeDriver:
        with self._lock:
            if self._made:
                return self._made.pop()
        return self._factory()

    def __enter__(self) -> _Environments:
        return self

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            unused, self._made = self._made, []
        for env in unused:
            env.close()


class _Run:
    """Runs ``items`` (``(spec index, spec)``, in order), at most ``capacity`` at a time.

    The loop owns all the state. Environment work (starting an episode, its ``step``, the next
    prompt) runs on a pool and comes back as a trajectory, as the episode's next request, or as
    the error it raised. With ``lockstep`` the engine is called once nothing is on the pool, for
    every running episode; otherwise an episode asks the engine itself and stays on the pool until
    it has ended, like an :class:`EpisodeDriver` always does."""

    def __init__(
        self,
        items: Sequence[tuple[int, EpisodeSpec]],
        outcome: _Outcome,
        *,
        capacity: int,
        lockstep: bool,
        env_factory: Callable[[], Environment | EpisodeDriver],
        tokenizer: Any,
        params: SamplingParams,
        abort: threading.Event | None,
        turn_rendering: str,
        max_context_tokens: int | None,
        retries: int,
        tolerate: bool,
    ) -> None:
        self.queue = deque(items)
        self.outcome = outcome
        self.capacity = capacity
        self.lockstep = lockstep
        self.env_factory = env_factory
        self.tokenizer = tokenizer
        self.params = params
        self.abort = abort
        self.turn_rendering = turn_rendering
        self.max_context_tokens = max_context_tokens
        self.retries = retries
        self.tolerate = tolerate
        self.live: dict[int, _Live] = {}
        self.working: dict[Future[Trajectory | _Request | Exception], _Live] = {}
        self.pool: ThreadPoolExecutor | None = None

    def run(self) -> None:
        try:
            if self.capacity > 1:
                self.pool = ThreadPoolExecutor(self.capacity, thread_name_prefix="monorl-episode")
            while True:
                while self.queue and len(self.live) < self.capacity and not self.outcome.halt.is_set():
                    index, spec = self.queue.popleft()
                    self.live[index] = episode = _Live(index, spec)
                    self._work(episode, None)
                if self.working:
                    done, _ = wait(self.working, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda f: self.working[f].index):
                        self._settle(self.working.pop(future), future.result())
                elif self.live and not self.outcome.halt.is_set():
                    self._generate([self.live[index] for index in sorted(self.live)])
                else:
                    return
        except BaseException as e:
            self._fatal(-1, e)
        finally:
            for future in self.working:
                future.exception()  # a step in flight finishes before its environment is closed
            for episode in self.live.values():
                try:
                    self._close(episode)
                except Exception:  # noqa: BLE001 - the error that ended the run is the one to report
                    log.exception("closing the environment of episode %s failed", episode.spec.group_id)
            if self.pool is not None:
                self.pool.shutdown(wait=True)

    # -- environment work: on the pool, or inline when there is none ------------------------
    def _work(self, episode: _Live, completion: Completion | None) -> None:
        """Start ``episode`` (no ``completion``) or hand it the completion it asked for."""
        if self.pool is None:
            self._settle(episode, self._advance(episode, completion))
        else:
            self.working[self.pool.submit(self._advance, episode, completion)] = episode

    def _advance(self, episode: _Live, completion: Completion | None) -> Trajectory | _Request | Exception:
        """The episode's next request, its trajectory once it has ended (its environment is closed
        then), or what it raised (likewise)."""
        spec = episode.spec
        try:
            result: Trajectory | _Request
            if episode.turns is not None:
                result = self._next(episode.turns, completion)
            else:
                episode.env = env = self.env_factory()
                if hasattr(env, "run_episode"):
                    result = env.run_episode(
                        spec.sample,
                        group_id=spec.group_id,
                        engine=spec.engine,
                        tokenizer=self.tokenizer,
                        params=self.params,
                        abort=self.abort,
                        turn_rendering=self.turn_rendering,
                        max_context_tokens=self.max_context_tokens,
                    )
                else:
                    turns = _turns(
                        env,
                        spec,
                        tokenizer=self.tokenizer,
                        params=self.params,
                        abort=self.abort,
                        turn_rendering=self.turn_rendering,
                        max_context_tokens=self.max_context_tokens,
                    )
                    if self.lockstep:
                        episode.turns = turns
                        result = self._next(turns, None)
                    else:
                        result = _drive(turns, spec)
            if isinstance(result, Trajectory):
                self._close(episode)
            return result
        except Exception as e:
            try:
                self._close(episode)
            except Exception:  # noqa: BLE001 - the episode's own error is the one to report
                log.exception("closing the environment of episode %s failed", spec.group_id)
            return e

    @staticmethod
    def _next(
        turns: Generator[_Request, Completion, Trajectory], completion: Completion | None
    ) -> Trajectory | _Request:
        try:
            return next(turns) if completion is None else turns.send(completion)
        except StopIteration as stop:
            return stop.value

    @staticmethod
    def _close(episode: _Live) -> None:
        env, turns = episode.env, episode.turns
        episode.env = episode.turns = episode.request = None
        if turns is not None:
            turns.close()
        if env is not None:
            env.close()

    # -- the loop's side ----------------------------------------------------------------------
    def _settle(self, episode: _Live, result: Trajectory | _Request | Exception) -> None:
        spec = episode.spec
        if isinstance(result, Trajectory):
            self.outcome.trajectories[episode.index] = result
            del self.live[episode.index]
        elif not isinstance(result, Exception):
            episode.request = result
        elif isinstance(result, (EpisodeAborted, GenerationFailed)):
            self._fatal(episode.index, result)
        elif episode.attempt < self.retries and not self._stopping():
            episode.attempt += 1
            log.warning(
                "episode %s failed (%s: %s); retry %d of %d",
                spec.group_id,
                type(result).__name__,
                result,
                episode.attempt,
                self.retries,
            )
            self._work(episode, None)
        elif not self.tolerate:
            self._fatal(episode.index, result)
        else:
            failure = EpisodeFailure(spec=spec, error=result, attempts=self.retries + 1)
            self.outcome.failed.append((episode.index, failure))
            log.warning("episode %s failed for good (%s: %s)", spec.group_id, type(result).__name__, result)
            del self.live[episode.index]

    def _stopping(self) -> bool:
        return self.outcome.halt.is_set() or (self.abort is not None and self.abort.is_set())

    def _fatal(self, index: int, error: BaseException) -> None:
        self.outcome.fatal.append((index, error))
        self.outcome.halt.set()

    def _generate(self, waiting: Sequence[_Live]) -> None:
        """One engine call per distinct sampling parameters for the episodes that wait for their
        turn (in spec order); every episode goes back to its environment with its completion."""
        calls: dict[tuple[Any, ...], list[_Live]] = {}
        for episode in waiting:
            assert episode.request is not None
            calls.setdefault(params_key(episode.request[1]), []).append(episode)
        for episodes in calls.values():
            first = episodes[0]
            assert first.request is not None
            try:
                completions = _ask(
                    first.spec, [e.request[0] for e in episodes if e.request is not None], first.request[1]
                )
            except (EpisodeAborted, GenerationFailed) as e:
                self._fatal(first.index, e)
                return
            for episode, completion in zip(episodes, completions, strict=True):
                episode.request = None
                self._work(episode, completion)


def run_episodes(
    specs: Sequence[EpisodeSpec],
    *,
    env_factory: Callable[[], Environment | EpisodeDriver],
    tokenizer: Any,
    params: SamplingParams,
    max_parallel: int,
    abort: threading.Event | None = None,
    turn_rendering: str = "tokens",
    max_context_tokens: int | None = None,
    retries: int = 0,
    failures: list[EpisodeFailure] | None = None,
    turn_batching: TurnBatching = "lockstep",
) -> list[Trajectory]:
    """Run every spec to completion, ``max_parallel`` episodes at a time; order is preserved.
    ``turn_batching`` decides how their turns reach the engines (module docstring).

    Without ``failures`` an episode failure ends the call: it is re-raised once the environment
    work in flight has finished (the failure of the earliest spec, if several failed). With a
    ``failures`` list, an episode whose environment failed on all ``retries + 1`` attempts is
    appended to it (in spec order) and missing from the result; aborts and engine failures are
    still raised (module docstring).

    Once ``abort`` is set the episodes raise :class:`EpisodeAborted`: at their next turn boundary,
    or straight out of a ``generate`` call in flight when the spec's engine implements ``cancel()``
    (see :class:`~monorl.core.interfaces.RolloutEngine`). ``cancel()`` aborts every call in flight
    on that engine, so an engine must not be shared with work that is meant to outlive the abort.
    An ``env.step`` in flight finishes first. Either way this returns or raises only after every
    episode thread has ended and every environment is closed.
    """
    if not specs:
        return []
    workers = max(1, min(int(max_parallel), len(specs)))
    lockstep = turn_batching == "lockstep"
    parts: list[list[tuple[int, EpisodeSpec]]] = [list(enumerate(specs))]
    environments = _Environments(env_factory)
    if lockstep:
        by_engine: dict[int, list[tuple[int, EpisodeSpec]]] = {}
        for item in parts[0]:
            by_engine.setdefault(id(item[1].engine), []).append(item)
        parts = list(by_engine.values())
        # An episode that waited in ``reset`` for another one to end would hold up the turn that
        # lets it end. With fewer places than engines every engine runs one episode at a time,
        # which waits for a place while it holds none.
        workers = min(workers, environments.limit() or workers)
    outcome = _Outcome(trajectories=[None] * len(specs))
    runs = [
        _Run(
            part,
            outcome,
            capacity=min(max(1, workers // len(parts)), len(part)),
            lockstep=lockstep,
            env_factory=environments.make,
            tokenizer=tokenizer,
            params=params,
            abort=abort,
            turn_rendering=turn_rendering,
            max_context_tokens=max_context_tokens,
            retries=retries,
            tolerate=failures is not None,
        )
        for part in parts
    ]
    with _cancel_generation_on_abort(specs, abort), environments:
        if len(runs) == 1:
            runs[0].run()
        else:
            threads = [
                threading.Thread(target=run.run, name=f"monorl-episode-engine-{k}", daemon=True)
                for k, run in enumerate(runs)
            ]
            for thread in threads:
                thread.start()
            try:
                for thread in threads:
                    thread.join()
            except BaseException:
                outcome.halt.set()
                for thread in threads:
                    thread.join()
                raise
    if outcome.fatal:
        raise min(outcome.fatal, key=lambda entry: entry[0])[1]
    if failures is not None:
        failures.extend(failure for _, failure in sorted(outcome.failed, key=lambda entry: entry[0]))
    return [trajectory for trajectory in outcome.trajectories if trajectory is not None]


@contextlib.contextmanager
def _cancel_generation_on_abort(
    specs: Sequence[EpisodeSpec], abort: threading.Event | None
) -> Iterator[None]:
    """While the block runs, a watcher thread answers ``abort`` by calling ``cancel()`` on each
    distinct engine that has one — and keeps calling it every :data:`CANCEL_INTERVAL_S`, because an
    episode that was between two checks submits its next ``generate`` after the first cancel. The
    watcher has ended when the block exits, so no cancel can hit a later caller of the engines."""
    engines = list({id(s.engine): s.engine for s in specs if hasattr(s.engine, "cancel")}.values())
    if abort is None or not engines:
        yield
        return
    finished = threading.Event()

    def watch() -> None:
        while not finished.wait(CANCEL_INTERVAL_S):
            if not abort.is_set():
                continue
            for engine in list(engines):
                try:
                    engine.cancel()
                except Exception:  # noqa: BLE001 - the abort still takes effect at turn boundaries
                    log.exception("%s.cancel() failed; not cancelling it again", type(engine).__name__)
                    engines.remove(engine)

    watcher = threading.Thread(target=watch, name="monorl-episode-cancel", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        watcher.join()


def _assistant_message(completion: Completion) -> Message:
    return {"role": "assistant", "content": completion.text}


__all__ = [
    "EpisodeAborted",
    "EpisodeFailure",
    "EpisodeSpec",
    "chat_prompt_ids",
    "run_episode",
    "run_episodes",
]
