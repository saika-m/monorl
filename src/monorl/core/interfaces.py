"""Component protocols. Every backend/adapter implements one or more of these.

Two integration levels exist:

* **Component-level** (driven by :mod:`monorl.orchestrator`): :class:`RolloutEngine`,
  :class:`Trainer`, :class:`Environment`, :class:`RewardFunction`, :class:`DataSource`,
  :class:`WeightSync`. vLLM/SGLang/HF are rollout engines; the reference torch trainer and
  Unsloth-patched models are trainers.
* **Pipeline-level** (frameworks that own their whole RL loop: verl, NeMo-RL, TRL-native):
  :class:`PipelineBackend` receives the full :class:`~monorl.config.RunConfig` and must bridge
  monorl data/rewards/envs into the framework (typically via patches).

All protocols are ``runtime_checkable`` so the registry can validate factories cheaply.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from monorl.core.types import (
    Completion,
    Message,
    PromptInput,
    RewardResult,
    RunResult,
    Sample,
    SamplingParams,
    StepResult,
    Trajectory,
)

if TYPE_CHECKING:
    import torch

    from monorl.config.schema import RunConfig
    from monorl.core.batch import TrainBatch
    from monorl.core.reporting import RunReporter


@runtime_checkable
class DataSource(Protocol):
    """Iterable of :class:`Sample`. Implementations: local files, HF hub, S3 (fsspec)."""

    name: str

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Sample]: ...

    def schema(self) -> dict[str, str]:
        """Column name -> dtype string, for UI preview and field mapping."""
        ...


@runtime_checkable
class RolloutEngine(Protocol):
    """Text generation under the current policy weights.

    ``generate`` must return exactly ``params.n`` completions per prompt, preserving order.
    When a prompt is ``list[int]`` the engine conditions on those exact token ids; when it is a
    chat (``list[Message]``) the engine applies the tokenizer chat template with
    ``add_generation_prompt=True`` and records the resulting ids in ``Completion.prompt_token_ids``.
    A completion ended by one of ``params.stop`` keeps that stop string at the end of
    ``Completion.text`` (``finish_reason == "stop"``): environments parse closing tags such as
    ``</tool_call>`` from it, and a chat re-rendered from the text has to match the sampled tokens.

    ``max_new_tokens`` gives every prompt of the call a generation cap of its own: with it,
    ``max_new_tokens[i]`` (at least 1) takes the place of ``params.max_new_tokens`` for prompt
    ``i``, for each of its ``params.n`` completions. An engine never generates past a prompt's
    cap, and a completion that the cap ends has ``finish_reason == "length"``: the cap is what
    keeps a prompt and its completion inside a context budget when the prompts of a call have
    different lengths (the turns of multi-turn episodes). It is an argument of the call and not a
    field of :class:`SamplingParams` because it belongs to the prompts: the parameters stay one
    value that every prompt of the call shares, so callers batch the prompts that agree on them
    (:class:`~monorl.rollout.BatchingEngine`, the episode runner), and an engine that does not
    know the argument rejects the call instead of generating past the caps.
    :func:`~monorl.core.types.prompt_caps` resolves the cap of every prompt.

    Optional capability (duck-typed, not a protocol member): ``cancel() -> None`` aborts the
    ``generate`` calls that are in flight — they raise
    :class:`~monorl.core.errors.GenerationCancelled` instead of returning — without shutting the
    engine down. It is safe to call from any thread, a no-op when nothing is generating, and
    ``generate`` calls made afterwards run normally.

    Optional capability (duck-typed): ``speculation_stats() -> SpeculationStats | None`` returns
    what speculative decoding did since the previous call (reading restarts the counters), or
    ``None`` when the engine does not draft (``model.mtp.rollout``, ``docs/MTP.md``).

    Optional capability (duck-typed): ``accepts_partial_weights = True`` lets weight sync leave out
    of ``update_weights`` the tensors the engine was sent before and that did not change since. The
    engine promises that ``update_weights`` replaces the tensors it is given and keeps every other
    one, and that it still has its weights at the next update: an engine that drops them while it
    sleeps, or whose server may be restarted behind it, must not set it. Without the attribute
    every update carries all weights.
    """

    def generate(
        self,
        prompts: Sequence[PromptInput],
        params: SamplingParams,
        *,
        max_new_tokens: Sequence[int] | None = None,
    ) -> list[list[Completion]]: ...

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Hot-swap policy weights (HF parameter names, merged LoRA). Names that start with
        ``monorl.mtp.WIRE_PREFIX`` are weights of the model's MTP layers under their checkpoint
        keys: they go to the engine's drafter, or nowhere when it does not draft, and never into
        the policy."""
        ...

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        """Reload weights from an HF-format directory (checkpoint-based sync)."""
        ...

    def sleep(self) -> None:
        """Release accelerator memory for colocated training. No-op if unsupported."""
        ...

    def wake_up(self) -> None: ...

    def shutdown(self) -> None: ...

    @property
    def tokenizer(self) -> Any:
        """``transformers.PreTrainedTokenizerBase`` used for prompt formatting."""
        ...


@runtime_checkable
class Trainer(Protocol):
    """Owns the policy parameters and optimizer. Loss functions come from :mod:`monorl.algorithms`.

    Optional capability (duck-typed, not a protocol member): ``trainable_params``, the tensors the
    optimizer writes. With it weight sync sends an engine that takes partial updates
    (``RolloutEngine.accepts_partial_weights``) only what changed. The trainer promises that the
    list is complete, that an exported tensor it did not train and did not change is yielded on
    the same storage every time, and that any other write to such a tensor is an in-place operation
    of torch, which advances the tensor's version counter (``tensor.data`` and numpy views do not).
    Without the attribute every sync carries all weights.
    """

    @property
    def step(self) -> int: ...

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        """One optimizer step over ``batch`` (trainer handles micro-batching). Returns metrics."""
        ...

    def compute_logprobs(self, batch: TrainBatch, *, reference: bool = False) -> torch.Tensor:
        """``[B, T]`` logprobs of ``input_ids`` under the current policy (or the frozen reference
        when ``reference=True``). No gradient."""
        ...

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Merged, HF-named parameters for :meth:`RolloutEngine.update_weights`, followed by the
        MTP layers' weights (``monorl.mtp.MtpLayers.wire_weights``) when the trainer holds any."""
        ...

    def save_checkpoint(self, path: Path) -> None:
        """Resumable state (weights/adapters + optimizer + step)."""
        ...

    def load_checkpoint(self, path: Path) -> None: ...

    def save_pretrained(self, path: Path) -> None:
        """HF-format export (merged LoRA) loadable by any rollout engine."""
        ...


@runtime_checkable
class RewardFunction(Protocol):
    """Batched scoring. Return one :class:`RewardResult` per trajectory, in order."""

    name: str

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]: ...


@runtime_checkable
class Environment(Protocol):
    """Episode-level environment driven by the orchestrator::

        messages = env.reset(sample)
        for turn in range(env.max_turns):
            completion = engine.generate([messages], params)[0][0]
            messages.append({"role": "assistant", "content": completion.text})
            result = env.step(completion)
            messages.extend(result.messages)
            if result.done: break

    Environments are stateful per episode; the orchestrator creates one via the factory per
    episode (or calls ``reset`` again on a reusable instance).

    An environment that admits only so many episodes at a time, so that a further ``reset`` waits
    for one of them to end, says how many in an attribute ``max_concurrent_episodes``. The episode
    runner then keeps no more than that open: with ``rollout.turn_batching: lockstep`` an episode
    that waits in ``reset`` for another one to end would wait for ever, because the others take
    their next turn only once it is ready.
    """

    max_turns: int

    def reset(self, sample: Sample) -> list[Message]: ...

    def step(self, action: Completion) -> StepResult: ...

    def close(self) -> None: ...


@runtime_checkable
class EpisodeDriver(Protocol):
    """An environment that runs a whole episode itself instead of being stepped turn by turn: an
    external agent harness that calls the policy through the capture gateway
    (:mod:`monorl.gateway`). The environment factory may return either kind; the episode runner
    calls ``run_episode`` when the object has it and ``reset``/``step`` otherwise.

    ``run_episode`` returns the finished :class:`Trajectory` (unscored: rewards are applied by the
    caller, per-turn environment rewards go in ``Turn.reward``). It must honour ``abort`` by
    raising :class:`~monorl.core.errors.EpisodeAborted` and report an engine failure as
    :class:`~monorl.core.errors.GenerationFailed`; any other exception counts as a failure of the
    episode's environment, which a run may tolerate (``env.max_failed_frac``). ``turn_rendering``
    is ``rollout.turn_rendering`` (``"tokens"`` or ``"template"``) and ``max_context_tokens`` the
    episode's context budget in tokens (``None``: unlimited): no prompt plus completion may exceed
    it.
    """

    def run_episode(
        self,
        sample: Sample,
        *,
        group_id: str,
        engine: RolloutEngine,
        tokenizer: Any,
        params: SamplingParams,
        abort: threading.Event | None = None,
        turn_rendering: str = "tokens",
        max_context_tokens: int | None = None,
    ) -> Trajectory: ...

    def close(self) -> None: ...


@runtime_checkable
class WeightSync(Protocol):
    """Moves trainer weights into rollout engines."""

    name: str

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        """Returns metrics (e.g. ``bytes``, ``seconds``)."""
        ...


@runtime_checkable
class PipelineBackend(Protocol):
    """A framework that runs the entire RL loop itself (verl, NeMo-RL, TRL-native).

    ``resume`` is ``monorl train --resume``: continue the run of ``run_dir`` from its newest
    checkpoint, after ``reporter.drop_rows_after`` took the record files back to that step.
    Without it the run starts at step 0. A backend whose framework cannot do what was asked
    raises :class:`~monorl.core.errors.ConfigError`; it never does the other thing instead.

    ``cancel`` is the event that the first SIGINT or SIGTERM sets while ``run()`` of the
    orchestrator handles the signals of the process. A backend that trains in this process ends
    the run once the step in flight is done, as the loops do: the step is logged, the framework's
    checkpoint of it exists for a resume, nothing is exported, and the result is ``cancelled``
    unless that step was the last one. A backend that runs its framework in a child process
    cannot have the step finished: it does not start the child when the event is set by then,
    and while the child runs it handles the signals itself, for the same event, asks the child
    to stop at the first and kills it at a second one; ``docs/BACKENDS.md`` says what each
    backend does.
    """

    name: str

    def run(
        self,
        config: RunConfig,
        run_dir: Path,
        *,
        reporter: RunReporter,
        resume: bool = False,
        cancel: threading.Event | None = None,
    ) -> RunResult: ...


class RolloutEngineFactory(Protocol):
    def __call__(self, config: RunConfig, **kwargs: Any) -> RolloutEngine: ...


class TrainerFactory(Protocol):
    def __call__(self, config: RunConfig, **kwargs: Any) -> Trainer: ...


class EnvironmentFactory(Protocol):
    def __call__(self, config: RunConfig, **kwargs: Any) -> Environment: ...


class RewardFactory(Protocol):
    def __call__(self, params: dict[str, Any], **kwargs: Any) -> RewardFunction: ...
