"""Ray runtime: engines and trainers as actors, placement groups per :class:`PlacementPlan`.

Topology
--------
* One placement group. With GPUs every bundle is one GPU (``{"GPU": 1}``), ``PACK`` for colocated
  (engines and trainer share the same bundles with ``num_gpus=0.5`` each) and ``SPREAD`` for
  disaggregated (engines own bundles ``[0, rollout_gpus)``, trainer ranks the next
  ``trainer_gpus``). Without GPUs each actor gets a ``{"CPU": cpus_per_actor}`` bundle.
* Engines are SPMD: replica ``i`` with tensor parallel ``tp`` is ``tp`` :class:`RolloutEngineActor`
  actors (one per GPU) that each build the registry engine with ``replica=i, tp_rank=r, tp_size=tp``
  after setting ``MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE/LOCAL_RANK`` for the replica's own
  process group (vLLM ``external_launcher`` style). ``generate`` fans out to all ranks and returns
  rank 0's completions.
* The trainer is ``trainer_world_size`` :class:`TrainerActor` actors built with
  ``rank=r, world_size=W`` and the same env vars; the proxy shards each :class:`TrainBatch`
  contiguously across ranks for ``train_step``/``compute_logprobs`` (a multi-rank trainer must
  all-reduce its gradients, e.g. DDP/FSDP); checkpoints and exports go through rank 0 (collective
  save/load calls reach every rank).
* Component factories run **inside** the actors so vLLM/SGLang live in their own processes. Extra
  provider modules (``cluster.extra.providers``, e.g. test fakes) are loaded there, and the
  driver's ``sys.path`` is replayed in the actor for same-filesystem clusters. ``cluster.extra.ray_init``
  is forwarded to ``ray.init``; ``cluster.extra.cpus_per_actor`` (default 1) sizes CPU bundles.

Weight sync (``plan.weight_sync``)
----------------------------------
* ``inprocess`` → :class:`RayObjectStoreWeightSync`: rank-0 trainer actor chunks its exported
  weights into the object store (``cluster.extra.sync_chunk_bytes``, default 64 MiB); every engine
  actor pulls and applies them. A tensor is stored as its raw bytes out of band
  (:func:`_store_chunk`), so a reader maps it instead of unpickling a copy (:func:`_stored_weights`).
* ``broadcast`` → :class:`RayBroadcastWeightSync`: rendezvous (node IP + free port) obtained from
  the rank-0 trainer actor; engine actors join as ranks ``1 + i*tp + r``; NCCL with GPUs, gloo without.
* ``cuda_ipc`` → :class:`RayCudaIpcWeightSync`: rank-0 trainer actor exports IPC handles (kept
  alive until every engine opened and copied them).
* ``checkpoint`` → :class:`~monorl.distributed.weight_sync.CheckpointWeightSync` via the registry
  (needs a filesystem shared by all actors; set ``cluster.extra.sync_dir``).

The first three leave out the tensors that the engine actors hold
(:class:`~monorl.distributed.weight_sync.SentWeights`). What the engines were sent is known where
the tensors are, in the rank-0 trainer actor (:meth:`TrainerActor._push`); whether an engine still
has it is known in its actor (:meth:`RolloutEngineActor.held_weights`). Every push has an id: the
strategy asks the engine actors which push they hold and the trainer actor sends the changes since
then only if that is its last one, and everything otherwise. So the first sync, the sync after one
that failed anywhere, and the sync after an engine was built anew or given weights in another way
carry every tensor.

``Runtime.close()`` shuts engines down, kills all actors and removes the placement group.
"""

from __future__ import annotations

import logging
import math
import os
import socket
import sys
import time
import uuid
import warnings
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from monorl.config.schema import RunConfig
from monorl.core.batch import TrainBatch
from monorl.core.errors import ConfigError, WeightSyncError
from monorl.core.registry import registry
from monorl.core.types import Completion, PromptInput, SamplingParams, SpeculationStats
from monorl.distributed.cluster import detect_cluster
from monorl.distributed.placement import PlacementPlan, plan_placement
from monorl.distributed.runtime import Runtime
from monorl.distributed.weight_sync import (
    BroadcastGroupSpec,
    BroadcastWeightSync,
    CudaIpcWeightSync,
    IpcHandle,
    SentWeights,
    accepts_partial_weights,
    open_ipc_handles,
)
from monorl.mtp import is_wire_name

if TYPE_CHECKING:
    from monorl.core.interfaces import RolloutEngine, Trainer, WeightSync

log = logging.getLogger(__name__)

DEFAULT_SYNC_CHUNK_BYTES = 64 << 20
PLACEMENT_GROUP_TIMEOUT_S = 300.0


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _node_ip() -> str:
    import ray.util

    return str(ray.util.get_node_ip_address())


def _prepare_process(sys_path: Sequence[str], providers: Sequence[str], env: dict[str, str]) -> None:
    for p in reversed(list(sys_path)):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.environ.update(env)
    for module_path in providers:
        registry.load_provider(module_path)


def _group_env(master_addr: str, master_port: int, rank: int, world_size: int) -> dict[str, str]:
    return {
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
        "RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_RANK": "0",
    }


# -- weights in the object store ------------------------------------------------------------------
#
# Ray pickles a torch tensor in band: its bytes are copied into the pickle stream on ``ray.put``
# and out of it, twice, on ``ray.get``. A numpy array travels out of band instead: ``ray.put``
# writes its bytes straight into the store and ``ray.get`` returns an array over the stored bytes.
# numpy has no bfloat16, so every tensor is stored as a flat uint8 view next to its dtype and shape.


def _store_chunk(chunk: list[tuple[str, torch.Tensor]]) -> Any:
    """``ray.put`` contiguous CPU tensors as ``(name, dtype, shape, bytes)`` records. The store
    holds its own copy when this returns: the tensors may change afterwards."""
    import ray

    return ray.put(
        [(name, t.dtype, tuple(t.shape), t.reshape(-1).view(torch.uint8).numpy()) for name, t in chunk]
    )


def _store_in_chunks(weights: Iterable[tuple[str, torch.Tensor]], chunk_bytes: int) -> tuple[list[Any], int]:
    """:func:`_store_chunk` ``weights`` in chunks of ``chunk_bytes``; returns ``(refs, total_bytes)``."""
    refs: list[Any] = []
    chunk: list[tuple[str, torch.Tensor]] = []
    pending = 0
    total = 0
    for name, tensor in weights:
        t = tensor.detach().cpu().contiguous().resolve_conj().resolve_neg()
        size = t.numel() * t.element_size()
        if chunk and pending + size > chunk_bytes:
            refs.append(_store_chunk(chunk))
            chunk, pending = [], 0
        chunk.append((name, t))
        pending += size
        total += size
    if chunk:
        refs.append(_store_chunk(chunk))
    return refs, total


def _stored_weights(ref: Any, *, private: bool) -> Iterator[tuple[str, torch.Tensor]]:
    """The tensors of one :func:`_store_chunk` object.

    They are views of the store's memory, which every reader of the object shares and which stays
    allocated for as long as one of them is alive: valid to read and to copy from, never to write
    to or to keep. ``private`` yields copies the consumer owns instead; so do the MTP layers'
    weights (:func:`monorl.mtp.is_wire_name`) in either case, because the loader that puts them
    into a drafter adopts the tensors it is given as parameters
    (:meth:`monorl.mtp.MtpLayers.load_checkpoint_weights`)."""
    import ray

    for name, dtype, shape, raw in ray.get(ref):
        with warnings.catch_warnings():
            # torch warns that it cannot mark the tensor read-only; the contract above does that.
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            tensor = torch.from_numpy(raw).view(dtype).view(shape)
        yield name, tensor.clone() if private or is_wire_name(name) else tensor


# -- actors (plain classes; wrapped with ray.remote lazily so importing this module is Ray-free) --


class RolloutEngineActor:
    def __init__(
        self,
        config_dict: dict[str, Any],
        *,
        replica: int,
        tp_rank: int,
        tp_size: int,
        providers: Sequence[str],
        sys_path: Sequence[str],
    ) -> None:
        self._config_dict = config_dict
        self.replica = replica
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._providers = list(providers)
        self._sys_path = list(sys_path)
        self.engine: RolloutEngine | None = None
        self._broadcast: BroadcastWeightSync | None = None
        self._held: str | None = None
        """The push of the trainer actor whose weights the engine holds (:meth:`_update`)."""

    # lifecycle
    def node_ip(self) -> str:
        return _node_ip()

    def free_port(self) -> int:
        return free_port()

    def build(self, env: dict[str, str]) -> None:
        _prepare_process(self._sys_path, self._providers, env)
        config = RunConfig.from_dict(self._config_dict)
        factory = registry.get("rollout_engine", config.rollout.backend)
        self._held = None
        self.engine = factory(config, replica=self.replica, tp_rank=self.tp_rank, tp_size=self.tp_size)

    def _engine(self) -> RolloutEngine:
        if self.engine is None:
            raise WeightSyncError("engine actor used before build()")
        return self.engine

    # RolloutEngine surface
    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        return self._engine().generate(prompts, params)

    def update_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        with self._update(None) as engine:
            engine.update_weights(weights)

    def load_weights_from_path(self, path: str) -> None:
        with self._update(None) as engine:
            engine.load_weights_from_path(path)

    def sleep(self) -> None:
        self._engine().sleep()

    def wake_up(self) -> None:
        self._engine().wake_up()

    def shutdown(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
        if self._broadcast is not None:
            self._broadcast.close()

    def tokenizer(self) -> Any:
        return self._engine().tokenizer

    def speculation_stats(self) -> SpeculationStats | None:
        read = getattr(self._engine(), "speculation_stats", None)
        return read() if callable(read) else None

    # weight sync entry points
    @contextmanager
    def _update(self, push: str | None) -> Iterator[RolloutEngine]:
        """An update of the engine's weights. When it ends the engine holds the trainer actor's
        ``push``; when it fails, or brings other weights than a push (``None``), the engine holds
        nothing that a later push could build on."""
        self._held = None
        yield self._engine()
        self._held = push

    def held_weights(self) -> str | None:
        """The trainer actor's push that the engine holds, for an engine that may be sent only what
        changed since (:func:`~monorl.distributed.weight_sync.accepts_partial_weights`); ``None``
        when it has to be sent everything."""
        return self._held if accepts_partial_weights(self._engine()) else None

    def pull_weights(self, refs: list[Any], push: str | None = None) -> int:
        counted = [0]

        def stream() -> Iterator[tuple[str, torch.Tensor]]:
            for ref in refs:
                for name, tensor in _stored_weights(ref, private=False):
                    counted[0] += tensor.numel() * tensor.element_size()
                    yield name, tensor

        with self._update(push) as engine:
            engine.update_weights(stream())
        return counted[0]

    def receive_broadcast(
        self, spec: BroadcastGroupSpec, rank: int, push: str | None = None
    ) -> dict[str, float]:
        with self._update(push) as engine:
            own = getattr(engine, "receive_broadcast", None)
            if callable(own):  # tensor-parallel adapters coordinate their own workers
                return own(spec, rank)
            if self._broadcast is None:
                self._broadcast = BroadcastWeightSync(spec, rank=rank)
            return self._broadcast.receive(engine)

    def update_weights_from_ipc(self, handles: Sequence[IpcHandle], push: str | None = None) -> None:
        with self._update(push) as engine:
            engine.update_weights(open_ipc_handles(handles))
            torch.cuda.synchronize()


class TrainerActor:
    def __init__(
        self,
        config_dict: dict[str, Any],
        *,
        rank: int,
        world_size: int,
        providers: Sequence[str],
        sys_path: Sequence[str],
    ) -> None:
        self._config_dict = config_dict
        self.rank = rank
        self.world_size = world_size
        self._providers = list(providers)
        self._sys_path = list(sys_path)
        self.trainer: Trainer | None = None
        self._broadcast: BroadcastWeightSync | None = None
        self._ipc: CudaIpcWeightSync | None = None
        self._ipc_keepalive: list[torch.Tensor] = []
        self._sent = SentWeights()
        self._pushed: str | None = None
        """The push that ``_sent`` describes: what an engine actor has to hold to be sent less."""

    def node_ip(self) -> str:
        return _node_ip()

    def free_port(self) -> int:
        return free_port()

    def build(self, env: dict[str, str]) -> None:
        _prepare_process(self._sys_path, self._providers, env)
        config = RunConfig.from_dict(self._config_dict)
        factory = registry.get("trainer", config.trainer.backend)
        self.trainer = factory(config, rank=self.rank, world_size=self.world_size)

    def _trainer(self) -> Trainer:
        if self.trainer is None:
            raise WeightSyncError("trainer actor used before build()")
        return self.trainer

    # Trainer surface
    def step(self) -> int:
        return self._trainer().step

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        return self._trainer().train_step(batch)

    def compute_logprobs(self, batch: TrainBatch, reference: bool) -> torch.Tensor:
        return self._trainer().compute_logprobs(batch, reference=reference).detach().cpu()

    def compute_values(self, batch: TrainBatch) -> torch.Tensor:
        compute = getattr(self._trainer(), "compute_values", None)
        if compute is None:
            raise ConfigError(
                f"trainer {type(self._trainer()).__name__} has no value head (compute_values), which "
                "algorithm.name='ppo' needs"
            )
        return compute(batch).detach().cpu()

    def save_checkpoint(self, path: str) -> None:
        self._trainer().save_checkpoint(Path(path))

    def load_checkpoint(self, path: str) -> None:
        self._trainer().load_checkpoint(Path(path))

    def save_pretrained(self, path: str) -> None:
        self._trainer().save_pretrained(Path(path))

    # weight export paths
    def export_weight_chunks(self, chunk_bytes: int) -> tuple[list[Any], int]:
        """Put all of the exported weights into the object store (:func:`_store_in_chunks`), for a
        reader that is not an engine in weight sync; returns ``(refs, total_bytes)``."""
        return _store_in_chunks(self._trainer().export_weights(), chunk_bytes)

    def export_weights_list(self) -> list[tuple[str, torch.Tensor]]:
        return [(n, t.detach().cpu()) for n, t in self._trainer().export_weights()]

    @contextmanager
    def _push(self, held: str | None, push: str) -> Iterator[Iterator[tuple[str, torch.Tensor]]]:
        """The weights of push ``push`` to engine actors that all hold push ``held`` (``None``:
        one of them may hold anything): the tensors that changed since ``held`` when that is this
        actor's last push, every tensor otherwise. ``push`` is the last one once the block ends;
        after a block that failed there is none."""
        if held is None or held != self._pushed:
            self._sent.forget()
        self._pushed = None
        with self._sent.push(self._trainer()) as stream:
            yield stream
        self._pushed = push

    def push_weight_chunks(self, chunk_bytes: int, held: str | None, push: str) -> tuple[list[Any], int]:
        """:meth:`_push` into the object store (:func:`_store_in_chunks`); returns
        ``(refs, total_bytes)``."""
        with self._push(held, push) as stream:
            return _store_in_chunks(stream, chunk_bytes)

    def broadcast_weights(self, spec: BroadcastGroupSpec, held: str | None, push: str) -> dict[str, float]:
        if self._broadcast is None:
            self._broadcast = BroadcastWeightSync(spec, rank=0)
        with self._push(held, push) as stream:
            return self._broadcast.send(stream)

    def export_ipc_handles(self, held: str | None, push: str) -> tuple[list[IpcHandle], int]:
        if self._ipc is None:
            self._ipc = CudaIpcWeightSync()
        with self._push(held, push) as stream:
            handles, keepalive = self._ipc.export_handles(stream)
        self._ipc_keepalive = keepalive
        return handles, sum(t.numel() * t.element_size() for t in keepalive)

    def release_ipc_exports(self) -> None:
        self._ipc_keepalive = []

    def shutdown(self) -> None:
        if self._broadcast is not None:
            self._broadcast.close()
        close = getattr(self.trainer, "shutdown", None)
        if callable(close):  # e.g. ReferenceTrainer leaves its torch.distributed group
            close()


# -- proxies ------------------------------------------------------------------------------------


class _RayHandle:
    def __init__(self, refs: list[Any]) -> None:
        self._refs = refs

    def result(self) -> Any:
        import ray

        results = ray.get(self._refs)
        return results[0] if results else {}


class RayRolloutEngine:
    """Driver-side :class:`RolloutEngine` over the ``tp`` actors of one replica."""

    def __init__(self, actors: list[Any], *, replica: int) -> None:
        self.actors = actors
        self.replica = replica
        self._tokenizer: Any = None

    def _all(self, method: str, *args: Any) -> list[Any]:
        return [getattr(a, method).remote(*args) for a in self.actors]

    def generate(self, prompts: Sequence[PromptInput], params: SamplingParams) -> list[list[Completion]]:
        import ray

        return ray.get(self._all("generate", list(prompts), params))[0]

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        import ray

        payload = [(n, t.detach().cpu()) for n, t in weights]
        ray.get(self._all("update_weights", payload))

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        import ray

        ray.get(self._all("load_weights_from_path", str(path)))

    def sleep(self) -> None:
        import ray

        ray.get(self._all("sleep"))

    def wake_up(self) -> None:
        import ray

        ray.get(self._all("wake_up"))

    def shutdown(self) -> None:
        import ray

        ray.get(self._all("shutdown"))

    @property
    def tokenizer(self) -> Any:
        import ray

        if self._tokenizer is None:
            self._tokenizer = ray.get(self.actors[0].tokenizer.remote())
        return self._tokenizer

    def speculation_stats(self) -> SpeculationStats | None:
        """Drafting counters of the replica since the previous call (its first actor serves the
        completions); ``None`` when the engine does not draft."""
        import ray

        return ray.get(self.actors[0].speculation_stats.remote())

    # sync hooks (non-blocking; return handles). ``push`` names the trainer actor's push that the
    # weights are (:meth:`TrainerActor._push`); without it the actors hold no push afterwards.
    def pull_weights_async(self, refs: list[Any], push: str | None = None) -> _RayHandle:
        return _RayHandle(self._all("pull_weights", refs, push))

    def begin_receive_broadcast(
        self, spec: BroadcastGroupSpec, rank: int, push: str | None = None
    ) -> _RayHandle:
        return _RayHandle(
            [a.receive_broadcast.remote(spec, rank + r, push) for r, a in enumerate(self.actors)]
        )

    def update_weights_from_ipc_async(
        self, handles: Sequence[IpcHandle], push: str | None = None
    ) -> _RayHandle:
        return _RayHandle(self._all("update_weights_from_ipc", list(handles), push))

    def update_weights_from_ipc(self, handles: Sequence[IpcHandle]) -> None:
        self.update_weights_from_ipc_async(handles).result()


class RayTrainer:
    """Driver-side :class:`Trainer` over ``world_size`` trainer actors (rank 0 first)."""

    def __init__(self, actors: list[Any]) -> None:
        self.actors = actors

    @property
    def world_size(self) -> int:
        return len(self.actors)

    def _shards(self, batch: TrainBatch) -> list[TrainBatch]:
        n = len(batch)
        per = math.ceil(n / self.world_size)
        shards = [batch.slice(r * per, min(n, (r + 1) * per)) for r in range(self.world_size)]
        if any(len(s) == 0 for s in shards):
            raise ConfigError(
                f"batch of {n} sequences cannot be sharded across {self.world_size} trainer ranks"
            )
        return shards

    @property
    def step(self) -> int:
        import ray

        return int(ray.get(self.actors[0].step.remote()))

    def train_step(self, batch: TrainBatch) -> dict[str, float]:
        import ray

        shards = self._shards(batch)
        results = ray.get([a.train_step.remote(s) for a, s in zip(self.actors, shards, strict=True)])
        keys = {k for r in results for k in r}
        return {
            k: sum(float(r[k]) for r in results if k in r) / sum(1 for r in results if k in r) for k in keys
        }

    def compute_logprobs(self, batch: TrainBatch, *, reference: bool = False) -> torch.Tensor:
        import ray

        shards = self._shards(batch)
        parts = ray.get(
            [a.compute_logprobs.remote(s, reference) for a, s in zip(self.actors, shards, strict=True)]
        )
        return torch.cat(parts, dim=0)

    def compute_values(self, batch: TrainBatch) -> torch.Tensor:
        """Value-head estimates (``algorithm.name: ppo``), sharded like :meth:`compute_logprobs`."""
        import ray

        shards = self._shards(batch)
        parts = ray.get([a.compute_values.remote(s) for a, s in zip(self.actors, shards, strict=True)])
        return torch.cat(parts, dim=0)

    def export_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        import ray

        refs, _ = ray.get(self.actors[0].export_weight_chunks.remote(DEFAULT_SYNC_CHUNK_BYTES))
        for ref in refs:
            yield from _stored_weights(ref, private=True)

    def save_checkpoint(self, path: Path) -> None:
        import ray

        ray.get([a.save_checkpoint.remote(str(path)) for a in self.actors])

    def load_checkpoint(self, path: Path) -> None:
        import ray

        ray.get([a.load_checkpoint.remote(str(path)) for a in self.actors])

    def save_pretrained(self, path: Path) -> None:
        import ray

        ray.get([a.save_pretrained.remote(str(path)) for a in self.actors])

    # sync hooks (rank 0); ``held`` and ``push`` as in :meth:`TrainerActor._push`
    def push_weight_refs(self, chunk_bytes: int, held: str | None, push: str) -> tuple[list[Any], int]:
        import ray

        return ray.get(self.actors[0].push_weight_chunks.remote(chunk_bytes, held, push))

    def node_ip(self) -> str:
        import ray

        return str(ray.get(self.actors[0].node_ip.remote()))

    def free_port(self) -> int:
        import ray

        return int(ray.get(self.actors[0].free_port.remote()))

    def broadcast_weights(self, spec: BroadcastGroupSpec, held: str | None, push: str) -> dict[str, float]:
        import ray

        return ray.get(self.actors[0].broadcast_weights.remote(spec, held, push))

    def export_ipc_handles(self, held: str | None, push: str) -> tuple[list[IpcHandle], int]:
        import ray

        return ray.get(self.actors[0].export_ipc_handles.remote(held, push))

    def release_ipc_exports(self) -> None:
        import ray

        ray.get(self.actors[0].release_ipc_exports.remote())

    def shutdown(self) -> None:
        import ray

        ray.get([a.shutdown.remote() for a in self.actors])


# -- weight sync strategies over the proxies -----------------------------------------------------


def _require(obj: Any, cls: type, role: str) -> None:
    if not isinstance(obj, cls):
        raise WeightSyncError(
            f"{role} must be a {cls.__name__} proxy from build_ray_runtime, got {type(obj).__name__}"
        )


def _proxies(trainer: Trainer, engines: Sequence[RolloutEngine]) -> tuple[RayTrainer, list[RayRolloutEngine]]:
    _require(trainer, RayTrainer, "trainer")
    for engine in engines:
        _require(engine, RayRolloutEngine, "engine")
    return trainer, list(engines)  # type: ignore[return-value]


def _held_by(engines: Sequence[RayRolloutEngine]) -> str | None:
    """The trainer actor's push that every actor of ``engines`` holds
    (:meth:`RolloutEngineActor.held_weights`); ``None`` unless it is one and the same."""
    import ray

    held = set(ray.get([actor.held_weights.remote() for engine in engines for actor in engine.actors]))
    return held.pop() if len(held) == 1 else None


class RayObjectStoreWeightSync:
    """Trainer rank 0 → ``ray.put`` chunks → every engine actor pulls. Ray's ``inprocess`` analogue."""

    name = "object_store"

    def __init__(self, chunk_bytes: int = DEFAULT_SYNC_CHUNK_BYTES) -> None:
        self.chunk_bytes = chunk_bytes

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        source, receivers = _proxies(trainer, engines)
        t0 = time.perf_counter()
        push = uuid.uuid4().hex
        refs, total = source.push_weight_refs(self.chunk_bytes, _held_by(receivers), push)
        t_exported = time.perf_counter()
        handles = [engine.pull_weights_async(refs, push) for engine in receivers]
        for h in handles:
            h.result()
        return {
            "weight_sync/bytes": float(total),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/export_seconds": t_exported - t0,
            "weight_sync/chunks": float(len(refs)),
        }


class RayBroadcastWeightSync:
    """Ray-coordinated :class:`BroadcastWeightSync`: rendezvous lives on the rank-0 trainer actor."""

    name = "broadcast"

    def __init__(
        self, plan: PlacementPlan, *, timeout_s: float = 600.0, bucket_bytes: int = 256 << 20
    ) -> None:
        self.plan = plan
        self.timeout_s = timeout_s
        self.bucket_bytes = bucket_bytes
        self.spec: BroadcastGroupSpec | None = None

    def _spec(self, trainer: RayTrainer) -> BroadcastGroupSpec:
        if self.spec is None:
            self.spec = BroadcastGroupSpec(
                master_addr=trainer.node_ip(),
                master_port=trainer.free_port(),
                world_size=1 + self.plan.engine_replicas * self.plan.engine_tensor_parallel,
                backend="nccl" if self.plan.total_gpus > 0 else "gloo",
                timeout_s=self.timeout_s,
                bucket_bytes=self.bucket_bytes,
            )
        return self.spec

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        source, receivers = _proxies(trainer, engines)
        if len(receivers) != self.plan.engine_replicas:
            raise WeightSyncError(f"expected {self.plan.engine_replicas} engines, got {len(receivers)}")
        spec = self._spec(source)
        t0 = time.perf_counter()
        tp = self.plan.engine_tensor_parallel
        held, push = _held_by(receivers), uuid.uuid4().hex
        handles = [
            engine.begin_receive_broadcast(spec, 1 + i * tp, push) for i, engine in enumerate(receivers)
        ]
        metrics = source.broadcast_weights(spec, held, push)
        for h in handles:
            h.result()
        metrics["weight_sync/seconds"] = time.perf_counter() - t0
        return metrics


class RayCudaIpcWeightSync:
    """Rank-0 trainer actor exports CUDA IPC handles; same-node engine actors open them."""

    name = "cuda_ipc"

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        source, receivers = _proxies(trainer, engines)
        t0 = time.perf_counter()
        push = uuid.uuid4().hex
        handles, total = source.export_ipc_handles(_held_by(receivers), push)
        try:
            pending = [engine.update_weights_from_ipc_async(handles, push) for engine in receivers]
            for h in pending:
                h.result()
        finally:
            source.release_ipc_exports()
        return {
            "weight_sync/bytes": float(total),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/tensors": float(len(handles)),
        }


# -- assembly -----------------------------------------------------------------------------------


def _bundles(plan: PlacementPlan, cpus_per_actor: float) -> tuple[list[dict[str, float]], str]:
    if plan.total_gpus > 0:
        bundles = [{"GPU": 1.0} for _ in range(plan.total_gpus)]
    else:
        bundles = [{"CPU": cpus_per_actor} for _ in range(plan.engine_replicas + plan.trainer_world_size)]
    return bundles, "PACK" if plan.placement == "colocated" else "SPREAD"


def _engine_bundle(plan: PlacementPlan, replica: int, tp_rank: int) -> int:
    return replica * plan.engine_tensor_parallel + tp_rank if plan.total_gpus > 0 else replica


def _trainer_bundle(plan: PlacementPlan, rank: int) -> int:
    if plan.total_gpus == 0:
        return plan.engine_replicas + rank
    return rank if plan.placement == "colocated" else plan.rollout_gpus + rank


def build_ray_runtime(config: RunConfig) -> Runtime:
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    extra = config.cluster.extra
    ray.init(
        address=config.cluster.ray_address or None,
        ignore_reinit_error=True,
        **dict(extra.get("ray_init", {})),
    )
    plan = plan_placement(config, detect_cluster())
    cpus_per_actor = float(extra.get("cpus_per_actor", 1))
    bundles, strategy = _bundles(plan, cpus_per_actor)
    pg = placement_group(bundles, strategy=strategy)
    try:
        ray.get(pg.ready(), timeout=float(extra.get("placement_group_timeout_s", PLACEMENT_GROUP_TIMEOUT_S)))
    except Exception as e:
        remove_placement_group(pg)
        raise ConfigError(f"Ray placement group {bundles} ({strategy}) not satisfiable: {e}") from e

    gpu_share = 0.5 if plan.placement == "colocated" else 1.0
    # the run's own providers, then the ones named for the actors only
    providers = list(dict.fromkeys([*config.providers, *extra.get("providers", [])]))
    sys_path = list(sys.path)
    config_dict = config.to_dict()

    def options(bundle: int) -> dict[str, Any]:
        res = {"num_gpus": gpu_share, "num_cpus": 0} if plan.total_gpus > 0 else {"num_cpus": cpus_per_actor}
        return {
            **res,
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle,
                placement_group_capture_child_tasks=True,
            ),
        }

    engine_cls = ray.remote(RolloutEngineActor)
    trainer_cls = ray.remote(TrainerActor)
    tp = plan.engine_tensor_parallel
    engine_actors: list[list[Any]] = [
        [
            engine_cls.options(**options(_engine_bundle(plan, i, r))).remote(
                config_dict, replica=i, tp_rank=r, tp_size=tp, providers=providers, sys_path=sys_path
            )
            for r in range(tp)
        ]
        for i in range(plan.engine_replicas)
    ]
    trainer_actors = [
        trainer_cls.options(**options(_trainer_bundle(plan, r))).remote(
            config_dict, rank=r, world_size=plan.trainer_world_size, providers=providers, sys_path=sys_path
        )
        for r in range(plan.trainer_world_size)
    ]
    all_actors = [a for group in engine_actors for a in group] + trainer_actors

    def kill_all() -> None:
        for a in all_actors:
            ray.kill(a, no_restart=True)
        remove_placement_group(pg)

    try:
        builds = []
        for group in engine_actors:
            addr, port = ray.get([group[0].node_ip.remote(), group[0].free_port.remote()])
            builds += [a.build.remote(_group_env(addr, port, r, tp)) for r, a in enumerate(group)]
        addr, port = ray.get([trainer_actors[0].node_ip.remote(), trainer_actors[0].free_port.remote()])
        builds += [
            a.build.remote(_group_env(addr, port, r, plan.trainer_world_size))
            for r, a in enumerate(trainer_actors)
        ]
        ray.get(builds)
    except Exception:
        kill_all()
        raise

    engines = [RayRolloutEngine(group, replica=i) for i, group in enumerate(engine_actors)]
    trainer = RayTrainer(trainer_actors)
    weight_sync: WeightSync
    if plan.weight_sync == "inprocess":
        weight_sync = RayObjectStoreWeightSync(int(extra.get("sync_chunk_bytes", DEFAULT_SYNC_CHUNK_BYTES)))
    elif plan.weight_sync == "broadcast":
        weight_sync = RayBroadcastWeightSync(plan)
    elif plan.weight_sync == "cuda_ipc":
        weight_sync = RayCudaIpcWeightSync()
    else:
        weight_sync = registry.get("weight_sync", plan.weight_sync)(config)
    log.info("ray runtime: %s", plan.to_dict())
    return Runtime(
        engines=engines,
        trainer=trainer,
        weight_sync=weight_sync,
        placement={
            **plan.to_dict(),
            "rollout_backend": config.rollout.backend,
            "trainer_backend": config.trainer.backend,
            "weight_sync": weight_sync.name,
            "placement_group": {"bundles": bundles, "strategy": strategy},
            "actors": {"engines": plan.engine_replicas * tp, "trainers": plan.trainer_world_size},
        },
        closers=[trainer.shutdown, kill_all],
    )
