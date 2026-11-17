"""Weight-sync strategies beyond the in-process hand-off (:class:`~monorl.distributed.runtime.InProcessWeightSync`).

* :class:`CheckpointWeightSync` — ``trainer.save_pretrained(dir)`` then
  ``engine.load_weights_from_path(dir)``; works everywhere a shared filesystem exists. Always the
  whole model: engines reload the directory.
* :class:`BroadcastWeightSync` — a dedicated ``torch.distributed`` process group (gloo on CPU, NCCL
  on CUDA) with the trainer at rank 0 and one rank per engine (``1 + replica * tp``). Weights are
  streamed as buckets: a pickled header (names/shapes/dtypes) precedes every flat bucket, and an
  empty header terminates the stream, so nothing is materialised twice. The group is built lazily
  on first use and kept for the run. Engines in other processes take part through
  ``begin_receive_broadcast(spec, rank)`` (see :class:`BroadcastCapableEngine`); the receiving side
  calls :meth:`BroadcastWeightSync.receive` with its local engine.
* :class:`CudaIpcWeightSync` — same-node, cross-process zero-copy hand-off via CUDA IPC handles
  (``torch.multiprocessing.reductions.reduce_tensor``). Exported tensors are kept alive until every
  engine has consumed them. Engines in other processes implement ``update_weights_from_ipc(handles)``
  (see :class:`IpcCapableEngine`); an engine living in the exporting process gets the tensors
  directly, because CUDA refuses to open an IPC handle in the process that created it.

The strategies that hand tensors over leave out the ones an engine already holds
(:class:`SentWeights`): with LoRA that is everything the adapter does not touch. An engine has to
say that it can take such a push (:func:`accepts_partial_weights`); any other engine, the first push
to an engine, and the push after one that failed carry every tensor.
"""

from __future__ import annotations

import logging
import pickle
import shutil
import time
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, runtime_checkable

import torch

from monorl.core.errors import WeightSyncError
from monorl.core.interfaces import RolloutEngine, Trainer
from monorl.mtp import is_wire_name

log = logging.getLogger(__name__)


def _tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


# -- what an engine already holds -------------------------------------------------------------


def accepts_partial_weights(engine: RolloutEngine) -> bool:
    """Whether ``engine`` declares, with ``accepts_partial_weights = True``, that a push may leave
    out the tensors it was sent before. That takes two things: ``update_weights`` patches the
    tensors it is given and leaves the others alone, and the engine still has its weights at the
    next push (one that drops them while it sleeps counts on getting all of them back). An engine
    that does not declare it is sent everything."""
    return bool(getattr(engine, "accepts_partial_weights", False))


class _Held(NamedTuple):
    """An exported tensor as it was when it was sent."""

    storage: weakref.ReferenceType[torch.UntypedStorage]
    layout: tuple[int, torch.Size, tuple[int, ...], torch.dtype]
    version: int


class SentWeights:
    """What one receiver of a trainer's export holds, so that a push carries only what changed
    since the last push that succeeded. The receiver is an engine, or the engines that read one
    export together.

    A tensor is left out when the trainer yields it as it was sent: on the same storage, which has
    been alive since, with the same offset, shape, strides and dtype, and with the same version
    counter, which torch advances on every in-place write. Nothing is read or hashed. Never left
    out are

    * a tensor on the memory of one of ``trainer.trainable_params``: an optimizer may write without
      advancing the counter (torch's fused implementations do), so what is trained goes every time.
      A trainer without that attribute names nothing, and all of its export goes every time;
    * a tensor that was made for the export (merged LoRA weights): its storage is new;
    * a tensor created in inference mode: it has no counter.

    MTP weights (:data:`monorl.mtp.WIRE_PREFIX`) go all together or not at all, after the policy's:
    an engine replaces its drafter's weights from a complete set only.

    What this cannot see is a write that neither advances the counter nor belongs to the optimizer
    of ``trainable_params``: through ``tensor.data``, a numpy view or another process.
    """

    def __init__(self) -> None:
        self._held: dict[str, _Held | None] = {}

    def forget(self) -> None:
        """The receiver may hold anything (it was rebuilt, it lost its weights): the next push
        carries every tensor."""
        self._held.clear()

    @contextmanager
    def push(self, trainer: Trainer) -> Iterator[Iterator[tuple[str, torch.Tensor]]]:
        """``with sent.push(trainer) as stream:`` deliver ``stream``, the part of
        ``trainer.export_weights()`` the receiver does not hold. Leaving the block normally says
        that what was taken from the stream has arrived; leaving it with an exception says that
        the receiver's state is unknown (:meth:`forget`)."""
        staged: dict[str, _Held | None] = {}
        try:
            yield self._changed(trainer, staged)
        except BaseException:
            self.forget()
            raise
        self._held.update(staged)

    def _changed(
        self, trainer: Trainer, staged: dict[str, _Held | None]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        trained = getattr(trainer, "trainable_params", None)
        if trained is None:
            self.forget()  # what was noted of another trainer's export says nothing after this one
            yield from trainer.export_weights()
            return
        written = {p.untyped_storage().data_ptr() for p in trained}
        mtp: list[tuple[str, torch.Tensor]] = []
        mtp_staged: dict[str, _Held | None] = {}
        for name, tensor in trainer.export_weights():
            storage = tensor.untyped_storage()
            now = None
            if not tensor.is_inference() and storage.data_ptr() not in written:
                layout = (tensor.storage_offset(), tensor.shape, tensor.stride(), tensor.dtype)
                now = _Held(weakref.ref(storage), layout, tensor._version)
            before = self._held.get(name)
            held = (
                now is not None
                and before is not None
                and before.storage() is storage
                and before.layout == now.layout
                and before.version == now.version
            )
            if is_wire_name(name):
                mtp.append((name, tensor))
                if not held:
                    mtp_staged[name] = now
            elif not held:
                staged[name] = now
                yield name, tensor
        if mtp_staged:
            staged.update(mtp_staged)
            yield from mtp


class _SharedSentWeights:
    """:class:`SentWeights` of a strategy whose engines all read one export: a tensor is left out
    only when every engine holds it, so the engines have to be the ones of the last push and
    every one has to take partial pushes (:func:`accepts_partial_weights`)."""

    def __init__(self) -> None:
        self._sent = SentWeights()
        self._engines: tuple[RolloutEngine, ...] = ()

    @contextmanager
    def push(
        self, trainer: Trainer, engines: Sequence[RolloutEngine]
    ) -> Iterator[Iterator[tuple[str, torch.Tensor]]]:
        same = len(engines) == len(self._engines) and all(
            a is b for a, b in zip(engines, self._engines, strict=True)
        )
        self._engines = tuple(engines)
        if not same or not all(accepts_partial_weights(engine) for engine in engines):
            self._sent.forget()
        with self._sent.push(trainer) as stream:
            yield stream


# -- checkpoint -------------------------------------------------------------------------------


class CheckpointWeightSync:
    """Export the trainer to ``<dir>/weights`` (atomically replaced) and reload it in every engine."""

    name = "checkpoint"

    def __init__(self, dir: str | Path) -> None:
        self.dir = Path(dir).expanduser()

    @property
    def weights_dir(self) -> Path:
        return self.dir / "weights"

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        t0 = time.perf_counter()
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f".weights-{trainer.step}.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        trainer.save_pretrained(tmp)
        if self.weights_dir.exists():
            shutil.rmtree(self.weights_dir)
        tmp.rename(self.weights_dir)
        total = sum(p.stat().st_size for p in self.weights_dir.rglob("*") if p.is_file())
        t_saved = time.perf_counter()
        for engine in engines:
            engine.load_weights_from_path(self.weights_dir)
        return {
            "weight_sync/bytes": float(total),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/save_seconds": t_saved - t0,
        }


# -- broadcast --------------------------------------------------------------------------------


@dataclass(frozen=True)
class BroadcastGroupSpec:
    """Rendezvous + transport parameters shared by every rank of a broadcast group."""

    master_addr: str
    master_port: int
    world_size: int
    backend: Literal["gloo", "nccl"] = "gloo"
    timeout_s: float = 600.0
    bucket_bytes: int = 256 << 20


class BroadcastReceiveHandle(Protocol):
    def result(self) -> dict[str, float]: ...


@runtime_checkable
class BroadcastCapableEngine(Protocol):
    """An engine living in another process that can join a broadcast group as ``rank``."""

    def begin_receive_broadcast(self, spec: BroadcastGroupSpec, rank: int) -> BroadcastReceiveHandle: ...


@dataclass(frozen=True)
class _BucketEntry:
    name: str
    shape: tuple[int, ...]
    dtype: str


def _dtype_from_str(s: str) -> torch.dtype:
    dt = getattr(torch, s.removeprefix("torch."), None)
    if not isinstance(dt, torch.dtype):
        raise WeightSyncError(f"unknown dtype in broadcast header: {s!r}")
    return dt


class BroadcastWeightSync:
    """Rank 0 (trainer side) sends; ranks ``>= 1`` (engine side) receive. See module docstring."""

    name = "broadcast"

    def __init__(
        self, spec: BroadcastGroupSpec, *, rank: int, device: str | torch.device | None = None
    ) -> None:
        if not 0 <= rank < spec.world_size:
            raise WeightSyncError(f"rank {rank} outside world_size {spec.world_size}")
        if spec.backend == "nccl" and not torch.cuda.is_available():
            raise WeightSyncError("broadcast backend 'nccl' needs CUDA")
        self.spec = spec
        self.rank = rank
        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if spec.backend == "nccl"
                else torch.device("cpu")
            )
        self.device = torch.device(device)
        self._store: Any = None
        self._group: Any = None
        self._sent = _SharedSentWeights()

    # -- group lifecycle ----------------------------------------------------------------
    def _ensure_group(self) -> Any:
        if self._group is not None:
            return self._group
        import torch.distributed as dist

        timeout = timedelta(seconds=self.spec.timeout_s)
        self._store = dist.TCPStore(
            self.spec.master_addr,
            self.spec.master_port,
            self.spec.world_size,
            is_master=self.rank == 0,
            timeout=timeout,
            wait_for_workers=False,
        )
        if self.spec.backend == "gloo":
            self._group = dist.ProcessGroupGloo(self._store, self.rank, self.spec.world_size, timeout)
        else:
            opts = dist.ProcessGroupNCCL.Options()
            opts.is_high_priority_stream = False
            opts._timeout = timeout
            self._group = dist.ProcessGroupNCCL(self._store, self.rank, self.spec.world_size, opts)
        log.info(
            "broadcast group ready: rank %d/%d via %s", self.rank, self.spec.world_size, self.spec.backend
        )
        return self._group

    def close(self) -> None:
        self._group = None
        self._store = None

    def _broadcast(self, tensor: torch.Tensor) -> None:
        import torch.distributed as dist

        opts = dist.BroadcastOptions()
        opts.rootRank = 0
        opts.rootTensor = 0
        self._ensure_group().broadcast([tensor], opts).wait()

    def _send_header(self, entries: list[_BucketEntry]) -> int:
        payload = pickle.dumps([(e.name, e.shape, e.dtype) for e in entries])
        self._broadcast(torch.tensor([len(payload)], dtype=torch.int64, device=self.device))
        if payload:
            self._broadcast(torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(self.device))
        return 8 + len(payload)

    def _recv_header(self) -> list[_BucketEntry]:
        n = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._broadcast(n)
        size = int(n.item())
        if size == 0:
            return []
        buf = torch.empty(size, dtype=torch.uint8, device=self.device)
        self._broadcast(buf)
        return [
            _BucketEntry(name, tuple(shape), dtype)
            for name, shape, dtype in pickle.loads(buf.cpu().numpy().tobytes())
        ]

    # -- sender -----------------------------------------------------------------------------
    def send(self, weights: Iterable[tuple[str, torch.Tensor]]) -> dict[str, float]:
        """Stream ``weights`` to every other rank. Blocks until all ranks consumed each bucket."""
        if self.rank != 0:
            raise WeightSyncError("only rank 0 sends")
        t0 = time.perf_counter()
        total = 0
        buckets = 0
        entries: list[_BucketEntry] = []
        parts: list[torch.Tensor] = []
        pending_bytes = 0

        def flush() -> None:
            nonlocal total, buckets, entries, parts, pending_bytes
            if not parts:
                return
            total += self._send_header(entries)
            flat = torch.cat(parts) if len(parts) > 1 else parts[0]
            self._broadcast(flat)
            total += _tensor_bytes(flat)
            buckets += 1
            entries, parts, pending_bytes = [], [], 0

        for name, tensor in weights:
            t = tensor.detach().to(self.device, non_blocking=False).contiguous().reshape(-1)
            if parts and (
                t.dtype != parts[0].dtype or pending_bytes + _tensor_bytes(t) > self.spec.bucket_bytes
            ):
                flush()
            entries.append(_BucketEntry(name, tuple(tensor.shape), str(tensor.dtype)))
            parts.append(t)
            pending_bytes += _tensor_bytes(t)
        flush()
        total += self._send_header([])
        return {
            "weight_sync/bytes": float(total),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/buckets": float(buckets),
        }

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        """Rank-0 entry point: start every engine's receiver, stream the trainer's weights (the
        ones the engines do not hold, see :class:`SentWeights`), join."""
        if self.rank != 0:
            raise WeightSyncError("BroadcastWeightSync.sync must run on rank 0 (the trainer side)")
        if not engines:
            raise WeightSyncError("BroadcastWeightSync.sync needs at least one engine")
        if (self.spec.world_size - 1) % len(engines):
            raise WeightSyncError(
                f"world_size {self.spec.world_size} is not 1 + k * {len(engines)} engines (k = ranks per engine)"
            )
        ranks_per_engine = (self.spec.world_size - 1) // len(engines)  # == tp when every TP worker joins
        handles: list[BroadcastReceiveHandle] = []
        for i, engine in enumerate(engines):
            if not isinstance(engine, BroadcastCapableEngine):
                raise WeightSyncError(
                    f"engine {type(engine).__name__} cannot receive broadcasts; it must live in another "
                    "process and implement begin_receive_broadcast(spec, rank)"
                )
            handles.append(engine.begin_receive_broadcast(self.spec, 1 + i * ranks_per_engine))
        with self._sent.push(trainer, engines) as stream:
            metrics = self.send(stream)
            for h in handles:
                h.result()
        return metrics

    # -- receiver ---------------------------------------------------------------------------
    def _recv_stream(self, counter: list[int]) -> Iterator[tuple[str, torch.Tensor]]:
        while True:
            entries = self._recv_header()
            if not entries:
                return
            dtype = _dtype_from_str(entries[0].dtype)
            numel = sum(int(torch.Size(e.shape).numel()) for e in entries)
            flat = torch.empty(numel, dtype=dtype, device=self.device)
            self._broadcast(flat)
            counter[0] += _tensor_bytes(flat)
            counter[1] += 1
            offset = 0
            for e in entries:
                n = int(torch.Size(e.shape).numel())
                yield e.name, flat[offset : offset + n].view(e.shape)
                offset += n

    def receive(self, engine: RolloutEngine) -> dict[str, float]:
        """Rank ``>= 1`` entry point: consume one weight stream into ``engine.update_weights``."""
        if self.rank == 0:
            raise WeightSyncError("rank 0 sends; call send()/sync()")
        t0 = time.perf_counter()
        counter = [0, 0]
        engine.update_weights(self._recv_stream(counter))
        return {
            "weight_sync/bytes": float(counter[0]),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/buckets": float(counter[1]),
        }


# -- cuda ipc ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class IpcHandle:
    """Picklable CUDA IPC descriptor of one exported tensor (``rebuild(*args)`` opens it)."""

    name: str
    rebuild: Callable[..., torch.Tensor]
    args: tuple[Any, ...]

    def open(self) -> torch.Tensor:
        return self.rebuild(*self.args)


@runtime_checkable
class IpcCapableEngine(Protocol):
    """An engine in another process on the same node that can open CUDA IPC handles."""

    def update_weights_from_ipc(self, handles: Sequence[IpcHandle]) -> None: ...


def open_ipc_handles(handles: Iterable[IpcHandle]) -> Iterator[tuple[str, torch.Tensor]]:
    """Consumer side: map IPC handles to live CUDA tensors (views of the exporter's memory)."""
    for h in handles:
        yield h.name, h.open()


class CudaIpcWeightSync:
    """Same-node zero-copy sync through CUDA IPC. See module docstring."""

    name = "cuda_ipc"

    def __init__(self, device: str | torch.device | None = None) -> None:
        if not torch.cuda.is_available():
            raise WeightSyncError(
                "weight_sync='cuda_ipc' needs CUDA; use 'inprocess', 'broadcast' or 'checkpoint'"
            )
        self.device = (
            torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
        )
        if self.device.type != "cuda":
            raise WeightSyncError(f"cuda_ipc device must be a CUDA device, got {self.device}")
        self._sent = _SharedSentWeights()

    def export_handles(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> tuple[list[IpcHandle], list[torch.Tensor]]:
        """Pin every tensor on ``self.device`` and produce IPC handles.

        Returns ``(handles, keepalive)``; the caller must hold ``keepalive`` until consumers finished.
        """
        from torch.multiprocessing.reductions import reduce_tensor

        handles: list[IpcHandle] = []
        keepalive: list[torch.Tensor] = []
        for name, tensor in weights:
            t = tensor.detach()
            if t.device != self.device or not t.is_contiguous():
                t = t.to(self.device).contiguous()
            rebuild, args = reduce_tensor(t)
            handles.append(IpcHandle(name=name, rebuild=rebuild, args=tuple(args)))
            keepalive.append(t)
        torch.cuda.synchronize(self.device)
        return handles, keepalive

    def sync(self, trainer: Trainer, engines: Sequence[RolloutEngine]) -> dict[str, float]:
        t0 = time.perf_counter()
        with self._sent.push(trainer, engines) as stream:
            handles, keepalive = self.export_handles(stream)
            total = sum(_tensor_bytes(t) for t in keepalive)
            try:
                for engine in engines:
                    if isinstance(engine, IpcCapableEngine):
                        engine.update_weights_from_ipc(handles)
                    else:
                        engine.update_weights((h.name, t) for h, t in zip(handles, keepalive, strict=True))
                torch.cuda.synchronize(self.device)
            finally:
                del keepalive
        return {
            "weight_sync/bytes": float(total),
            "weight_sync/seconds": time.perf_counter() - t0,
            "weight_sync/tensors": float(len(handles)),
        }
