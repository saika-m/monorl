"""Fake rollout engine / trainer used by the distributed tests.

Importable as a registry provider (``register(registry)``) so Ray actors can load it via
``cluster.extra.providers = ["distributed_fakes"]``. Weights are two small tensors; the engine's
``generate`` output encodes the weight sum so a weight sync is observable end to end.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import torch

from monorl.core.batch import TrainBatch
from monorl.core.types import Completion, PromptInput, SamplingParams, SpeculationStats, prompt_caps

SHAPES = {"model.embed.weight": (8, 4), "model.layers.0.proj.weight": (4, 4)}


def make_weights(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {name: torch.randn(shape, generator=g) for name, shape in SHAPES.items()}


def weight_sum(weights: dict[str, torch.Tensor]) -> float:
    return float(sum(t.double().sum().item() for t in weights.values()))


class FakeEngine:
    def __init__(self, seed: int = 0, *, replica: int = 0, **kw: Any) -> None:
        self.weights = make_weights(seed)
        self.replica = replica
        self.kwargs = kw
        self.slept = False
        self.shut_down = False
        self.updates = 0
        self.received: list[str] = []
        """Names of the tensors of the last update, in the order they came."""
        self.completions = 0

    def speculation_stats(self) -> SpeculationStats:
        """One pass and one accepted draft per completion since the previous call."""
        done, self.completions = self.completions, 0
        return SpeculationStats(steps=done, drafted=done, accepted=done, emitted=2 * done)

    def generate(
        self,
        prompts: Sequence[PromptInput],
        params: SamplingParams,
        *,
        max_new_tokens: Sequence[int] | None = None,
    ) -> list[list[Completion]]:
        tag = f"w={weight_sum(self.weights):.6f}"
        self.completions += len(prompts) * params.n
        return [
            [
                Completion(
                    text=f"{tag} p={i} n={k}",
                    token_ids=[i, k][:cap],
                    logprobs=[0.0, 0.0][:cap] if params.logprobs else None,
                    finish_reason="stop" if cap >= 2 else "length",
                )
                for k in range(params.n)
            ]
            for i, cap in enumerate(prompt_caps(params, len(prompts), max_new_tokens))
        ]

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.received = []
        for name, tensor in weights:
            self.weights[name].copy_(tensor.to(self.weights[name].device, copy=True))
            self.received.append(name)
        self.updates += 1

    def load_weights_from_path(self, path: str | Path) -> None:
        state = torch.load(Path(path) / "weights.pt")
        self.update_weights(state.items())

    def sleep(self) -> None:
        self.slept = True

    def wake_up(self) -> None:
        self.slept = False

    def shutdown(self) -> None:
        self.shut_down = True

    @property
    def tokenizer(self) -> Any:
        return {"name": "fake-tokenizer"}


class FakeTrainer:
    def __init__(self, seed: int = 1, *, rank: int = 0, world_size: int = 1, **kw: Any) -> None:
        self.weights = make_weights(seed)
        self.rank = rank
        self.world_size = world_size
        self.kwargs = kw
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        self._step += 1
        for t in self.weights.values():
            t.add_(0.5)
        return {"loss/policy": float(batch.advantages.sum().item()), "batch/size": float(len(batch))}

    def compute_logprobs(self, batch: TrainBatch, *, reference: bool = False) -> torch.Tensor:
        return torch.full(batch.input_ids.shape, -1.0 if reference else -0.5)

    def compute_values(self, batch: TrainBatch) -> torch.Tensor:
        return batch.input_ids.to(torch.float32) / 10

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield from self.weights.items()

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": self.weights, "step": self._step}, path / "ckpt.pt")

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path / "ckpt.pt")
        self.weights = state["weights"]
        self._step = int(state["step"])

    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.weights, path / "weights.pt")


class PatchingEngine(FakeEngine):
    """An engine that says it takes pushes without the tensors it already holds."""

    accepts_partial_weights = True


class FlakyEngine(PatchingEngine):
    """While ``failing`` is set or the file ``fail_flag`` exists (the switch for an engine in
    another process), an update takes its stream, fails, and leaves the engine with weights of its
    own, like an engine that broke down and came back."""

    failing = False

    def __init__(self, seed: int = 0, *, fail_flag: str | None = None, **kw: Any) -> None:
        super().__init__(seed, **kw)
        self.fail_flag = fail_flag

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        if not self.failing and not (self.fail_flag and Path(self.fail_flag).exists()):
            return super().update_weights(weights)
        self.received = [name for name, _ in weights]
        for tensor in self.weights.values():
            tensor.zero_()
        raise RuntimeError("engine broke down during the update")


class AdapterTrainer(FakeTrainer):
    """Trains the tensors named in ``trained`` and leaves the others frozen. Like the reference
    trainer it lists what its optimizer writes in ``trainable_params``, and like torch's fused
    optimizers it writes without advancing the tensors' version counters."""

    def __init__(
        self, seed: int = 1, *, trained: Sequence[str] = ("model.layers.0.proj.weight",), **kw: Any
    ) -> None:
        super().__init__(seed, **kw)
        self.trained = tuple(trained)

    @property
    def trainable_params(self) -> list[torch.Tensor]:
        return [self.weights[name] for name in self.trained]

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        self._step += 1
        for name in self.trained:
            self.weights[name].data.add_(0.5)
        return {"loss/policy": float(batch.advantages.sum().item()), "batch/size": float(len(batch))}


def make_batch() -> TrainBatch:
    ids = torch.zeros(2, 3, dtype=torch.long)
    return TrainBatch(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        response_mask=torch.ones_like(ids),
        advantages=torch.ones(2, 3),
        rewards=torch.ones(2),
        group_ids=["g", "g"],
    )


def register(registry: Any) -> None:
    registry.register("rollout_engine", "fake", lambda config, **kw: FakeEngine(0, **kw))
    registry.register("trainer", "fake", lambda config, **kw: FakeTrainer(1, **kw))
    registry.register(
        "rollout_engine",
        "flaky",
        lambda config, **kw: FlakyEngine(0, fail_flag=config.rollout.extra.get("fail_flag"), **kw),
    )
    registry.register("trainer", "adapter", lambda config, **kw: AdapterTrainer(1, **kw))


# -- helpers for multi-process weight-sync tests (importable by spawn) ------------------------


def make_broadcast_weights(seed: int) -> dict[str, torch.Tensor]:
    """Mixed dtypes and sizes so the stream needs several buckets."""
    g = torch.Generator().manual_seed(seed)
    return {
        "a.weight": torch.randn(16, 4, generator=g),
        "b.weight": torch.randn(4, 4, generator=g),
        "c.ids": torch.arange(10, dtype=torch.int64) * seed,
        "d.half": torch.randn(3, generator=g).to(torch.float16),
        "e.weight": torch.randn(32, 8, generator=g),
    }


class CollectorEngine(FakeEngine):
    """Records every (name, tensor) it is given, cloning so views into buckets stay valid."""

    def __init__(self) -> None:
        super().__init__(0)
        self.received: dict[str, torch.Tensor] = {}

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        for name, tensor in weights:
            self.received[name] = tensor.clone()
        self.updates += 1


def broadcast_worker(rank: int, spec: Any, seed: int, out_path: str) -> None:
    from monorl.distributed.weight_sync import BroadcastWeightSync

    sync = BroadcastWeightSync(spec, rank=rank)
    if rank == 0:
        metrics = sync.send(make_broadcast_weights(seed).items())
    else:
        engine = CollectorEngine()
        metrics = sync.receive(engine)
        torch.save(engine.received, out_path)
    sync.close()
    Path(out_path + f".rank{rank}.json").write_text(repr(metrics))


def ipc_consumer(handles: list[Any], out_queue: Any) -> None:
    from monorl.distributed.weight_sync import open_ipc_handles

    out_queue.put({name: float(t.double().sum().item()) for name, t in open_ipc_handles(handles)})


class RemoteBroadcastEngine:
    """Rank-0 side of the engine that :func:`broadcast_engine_process` runs: starts a receive there
    and, once it ended, has what that engine was given (``received``) and holds (``weights``)."""

    accepts_partial_weights = True

    def __init__(self, conn: Any) -> None:
        self.conn = conn
        self.failing = False
        self.received: list[str] = []
        self.weights: dict[str, torch.Tensor] = {}

    def begin_receive_broadcast(self, spec: Any, rank: int) -> RemoteBroadcastEngine:
        self.conn.send((spec, rank, self.failing))
        return self

    def result(self) -> dict[str, float]:
        error, self.received, weights = self.conn.recv()
        self.weights = {name: torch.tensor(values) for name, values in weights.items()}
        if error:
            raise RuntimeError(error)
        return {}


def broadcast_engine_process(conn: Any) -> None:
    """Receive one broadcast into a :class:`FlakyEngine` per ``(spec, rank, failing)`` message and
    answer ``(error, received, weights)``; ``None`` ends the process."""
    from monorl.distributed.weight_sync import BroadcastWeightSync

    engine = FlakyEngine(seed=0)
    sync: BroadcastWeightSync | None = None
    while (message := conn.recv()) is not None:
        spec, rank, engine.failing = message
        if sync is None:
            sync = BroadcastWeightSync(spec, rank=rank)
        error = ""
        try:
            sync.receive(engine)
        except RuntimeError as e:
            error = str(e)
        conn.send((error, engine.received, {name: t.tolist() for name, t in engine.weights.items()}))
    if sync is not None:
        sync.close()
