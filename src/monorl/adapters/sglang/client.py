"""Server-mode SGLang rollout engine: an httpx client for ``python -m sglang.launch_server``.

Same :class:`~monorl.core.interfaces.RolloutEngine` protocol as the colocated engine, without
importing sglang. Endpoints (SGLang v0.5.23, ``python/sglang/srt/entrypoints/http_server.py``):

* ``POST /generate`` — body is ``GenerateReqInput`` (``srt/managers/io_struct.py`` /
  ``sglang/api/v1/api_types.py``): ``input_ids`` (list of token-id lists), ``sampling_params``
  (one dict per request, each with the ``max_new_tokens`` of its prompt: a ``generate`` call with
  a generation cap per prompt is still one request), ``return_logprob``, ``logprob_start_len``,
  ``top_logprobs_num`` (when ``SamplingParams.top_logprobs`` is set), ``image_data`` (one entry
  per request: a list of ``data:image/png;base64`` URLs, or ``None`` for text-only); returns the
  same flat list of ``{"text", "output_ids", "meta_info"}`` dicts as ``Engine.generate``.
* ``POST /update_weights_from_tensor`` — ``UpdateWeightsFromTensorReqInput``:
  ``serialized_named_tensors`` (one base64 ``MultiprocessingSerializer`` payload per TP rank),
  ``load_format``, ``flush_cache``; ``{"success", "message"}`` with 400 on failure. Tensors are
  shared through ``ForkingPickler`` (CUDA IPC / shm), so the server must run on the same host.
  The scheduler only accepts it inside a session: ``POST /begin_weight_update``
  ``{"selector": "all"}`` … ``POST /end_weight_update`` ``{}``, both ``{"success", "message"}``
  with 400 on failure. With selector ``"all"`` the target and the draft runner load the same
  tensors, each its own checkpoint keys, so MTP weights in the stream reach the drafter.
* ``POST /update_weights_from_disk`` — ``{"model_path", "load_format"}`` →
  ``{"success", "message", "num_paused_requests"}``.
* ``POST /release_memory_occupation`` / ``POST /resume_memory_occupation`` — ``{"tags"}``.
* ``POST /flush_cache`` — text response, 200 when the radix cache was flushed.
* ``GET /get_server_info`` — resolved ``ServerArgs`` (``tp_size`` and the ``speculative_*``
  settings are read from here).

The client does not start the server. With ``model.mtp.rollout`` it must have been launched with
``--speculative-algorithm EAGLE`` (or ``NEXTN``) and accept thresholds of 1.0; the first
``generate`` checks that against ``/get_server_info`` and raises ``ConfigError`` otherwise.

``cancel()`` relies on the server aborting a generation whose client went away: while a
non-streaming ``/generate`` waits, ``TokenizerManager._stream_one_response``
(``srt/managers/tokenizer_manager.py``) polls ``request.is_disconnected()`` — every
``SGLANG_REQUEST_STATE_WAIT_TIMEOUT`` seconds (4 by default) and on each intermediate scheduler
output — and calls ``abort_request(rid)`` for every request of the batch. The requests therefore
run as tasks of an ``httpx.AsyncClient`` on a private event-loop thread: cancelling a task closes
its connection, which a thread blocked in a synchronous ``httpx.Client`` call could not be made
to do. (``POST /abort_request`` exists too, but takes one request id per call.) ``shutdown()``
drops every request in flight the same way before it stops that thread, which exists only in the
process that built the client.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import logging
import os
import threading
from collections.abc import Coroutine, Iterable, Sequence
from typing import Any, TypeVar

import httpx
import torch

from monorl.adapters.sglang.common import (
    WEIGHT_UPDATE_SELECTOR,
    SpeculationCounter,
    bucket_bytes,
    bucket_weights,
    check_speculative_args,
    drafts_with_mtp,
    encode_prompt,
    expand_image_data,
    expand_requests,
    group_outputs,
    image_data_urls,
    is_media_prompt,
    logprob_kwargs,
    route_weights,
    serialized_payloads,
    server_url,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import GenerationCancelled, WeightSyncError
from monorl.core.types import Completion, PromptInput, SamplingParams, SpeculationStats, prompt_caps
from monorl.rollout.hf_engine import load_tokenizer

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600.0

_T = TypeVar("_T")


class _ClientClosed(RuntimeError):
    """A request was made after ``shutdown()``, or ``shutdown()`` dropped it."""

    def __init__(self, url: str) -> None:
        super().__init__(f"the sglang server client for {url} was shut down")


class _GenerateCall:
    """One ``generate`` call that has not returned yet; fields are guarded by the client's
    ``_calls_lock``."""

    def __init__(self) -> None:
        self.cancelled = False
        self.request: concurrent.futures.Future[httpx.Response] | None = None
        """The ``/generate`` request the call is blocked on, if any."""


class SGLangServerClient:
    """``rollout_engine: sglang`` with ``rollout.extra.mode='server'``.

    ``rollout.extra`` keys: ``url``/``urls``, ``timeout`` (seconds, default 600), ``bucket_bytes``,
    ``sleep_tags``. ``transport`` (an ``httpx.AsyncBaseTransport``) is for tests.

    ``cancel()`` disconnects the ``/generate`` requests in flight, which makes the server abort
    their generation; the interrupted ``generate`` calls raise :class:`GenerationCancelled`.
    ``shutdown()`` does the same and ends every other request in flight too; calls made after it
    raise ``RuntimeError``. So does every call in a process forked from the one that built the
    client, where the thread that runs the requests does not exist: a process builds its own
    client.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        replica: int = 0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.replica = replica
        self._url = server_url(config, replica)
        self._tokenizer = load_tokenizer(config)
        self._processor: Any | None = None  # AutoProcessor, loaded on the first multimodal prompt
        self._processor_loaded = False
        self._bucket_bytes = bucket_bytes(config)
        self._sleep_tags: list[str] | None = config.rollout.extra.get("sleep_tags")
        timeout = float(config.rollout.extra.get("timeout", DEFAULT_TIMEOUT))
        self._http = httpx.AsyncClient(base_url=self._url, timeout=timeout, transport=transport)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name=f"monorl-sglang-client-{replica}", daemon=True
        )
        self._loop_thread.start()
        self._pid = os.getpid()  # the only process the loop thread runs in
        self._lock = threading.Lock()
        self._calls: set[_GenerateCall] = set()
        self._calls_lock = threading.Lock()
        self._tp_size: int | None = None
        self._speculation = SpeculationCounter() if drafts_with_mtp(config) else None
        self._drafting_checked = False
        self._asleep = False
        self._closed = False  # guarded by _calls_lock: no request is scheduled once it is set

    # -- RolloutEngine --------------------------------------------------------------------
    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def url(self) -> str:
        return self._url

    def generate(
        self,
        prompts: Sequence[PromptInput],
        params: SamplingParams,
        *,
        max_new_tokens: Sequence[int] | None = None,
    ) -> list[list[Completion]]:
        if not prompts:
            return []
        self._check_usable()
        call = _GenerateCall()
        with self._calls_lock:
            self._calls.add(call)
        try:
            processor = self._load_processor() if any(is_media_prompt(p) for p in prompts) else None
            encoded = [encode_prompt(self._tokenizer, processor, p) for p in prompts]
            prompt_ids = [ids for ids, _ in encoded]
            caps = prompt_caps(params, len(prompts), max_new_tokens)
            input_ids, sampling = expand_requests(prompt_ids, params, caps)
            self._check_server_drafts()
            body: dict[str, Any] = {
                "input_ids": input_ids,
                "sampling_params": sampling,
                **logprob_kwargs(params),
            }
            image_data = image_data_urls(expand_image_data([images for _, images in encoded], params))
            if image_data is not None:
                body["image_data"] = image_data
            response = self._post("/generate", call=call, json=body)
        except _ClientClosed:
            if call.cancelled:  # shutdown() while this call was in flight
                raise GenerationCancelled("generation cancelled") from None
            raise
        finally:
            with self._calls_lock:
                self._calls.discard(call)
        if response.status_code >= 400:
            raise RuntimeError(f"sglang /generate failed (HTTP {response.status_code}): {response.text}")
        outputs = response.json()
        if isinstance(outputs, dict):
            outputs = [outputs]
        if self._speculation is not None:
            self._speculation.record(outputs)
        return group_outputs(outputs, prompt_ids, params)

    def speculation_stats(self) -> SpeculationStats | None:
        """Speculative-decoding counters of the requests finished since the previous call;
        ``None`` when rollouts do not draft."""
        return None if self._speculation is None else self._speculation.take()

    def cancel(self) -> None:
        """Abort the ``generate`` calls in flight (they raise :class:`GenerationCancelled`); a
        no-op when idle. The client stays usable: later calls run normally."""
        with self._calls_lock:
            for call in self._calls:
                call.cancelled = True
                if call.request is not None:
                    call.request.cancel()

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Push ``weights`` in one weight-update session covering the policy and the drafter, so
        both change together. MTP weights of the stream reach the drafter under their checkpoint
        keys; without drafting they are left out."""
        self._check_usable()
        with self._lock:
            updated = 0
            routed = route_weights(weights, drafting=self._speculation is not None)
            buckets = bucket_weights(routed, max_bytes=self._bucket_bytes)
            first = next(buckets, None)
            if first is None:
                return
            tp_size = self._server_tp_size()
            begin = self._post("/begin_weight_update", json={"selector": WEIGHT_UPDATE_SELECTOR})
            self._check(begin, "begin_weight_update")
            try:
                for bucket in itertools.chain([first], buckets):
                    body = {
                        "serialized_named_tensors": serialized_payloads(bucket, tp_size=tp_size),
                        "load_format": None,
                        "flush_cache": False,
                    }
                    self._check(
                        self._post("/update_weights_from_tensor", json=body), "update_weights_from_tensor"
                    )
                    updated += len(bucket)
            except BaseException:
                # A failed update leaves the session open and the next push could not begin one.
                try:
                    self._check(self._post("/end_weight_update", json={}), "end_weight_update")
                except Exception:
                    log.exception("sglang server %s: could not close the weight-update session", self._url)
                raise
            self._check(self._post("/end_weight_update", json={}), "end_weight_update")
            self._flush_cache()
        log.debug("sglang server %s: updated %d tensors", self._url, updated)

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        self._check_usable()
        with self._lock:
            body = {"model_path": os.fspath(path), "load_format": None}
            self._check(self._post("/update_weights_from_disk", json=body), "update_weights_from_disk")
            self._flush_cache()

    def sleep(self) -> None:
        if not self.config.rollout.enable_sleep_mode:
            return
        self._check_usable()
        with self._lock:
            if self._asleep:
                return
            self._post("/release_memory_occupation", json=self._tags_body()).raise_for_status()
            self._asleep = True

    def wake_up(self) -> None:
        if not self.config.rollout.enable_sleep_mode:
            return
        self._check_usable()
        with self._lock:
            if not self._asleep:
                return
            self._post("/resume_memory_occupation", json=self._tags_body()).raise_for_status()
            self._asleep = False

    def shutdown(self) -> None:
        """Close the connections and stop the event-loop thread. The requests in flight are
        dropped first: their ``generate`` calls raise :class:`GenerationCancelled` as after
        :meth:`cancel`, other calls ``RuntimeError``, as does every call made afterwards.
        Repeated calls are harmless. In a forked process there is nothing to close: connections
        and thread belong to the process that built the client."""
        if os.getpid() != self._pid:
            return
        with self._calls_lock:
            if self._closed:
                return
            self._closed = True
            for call in self._calls:
                call.cancelled = True
        asyncio.run_coroutine_threadsafe(self._close(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join()
        self._loop.close()

    # -- internals ----------------------------------------------------------------------
    def _check_usable(self) -> None:
        """Raise unless this process can run requests: the event-loop thread is not copied into
        a forked process, and :meth:`shutdown` stops it. A call waiting for a request there
        would wait forever. Public methods check before they take a lock, which a fork may have
        copied while another thread held it."""
        if os.getpid() != self._pid:
            raise RuntimeError(
                f"the sglang server client for {self._url} was built in process {self._pid} and is used in "
                f"process {os.getpid()}, a fork of it, where the thread that runs its requests does not "
                "exist; build the client in the process that uses it"
            )
        if self._closed:
            raise _ClientClosed(self._url)

    async def _close(self) -> None:
        """On the event loop: drop every request in flight, then close the connections."""
        requests = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in requests:
            task.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        await self._http.aclose()

    def _load_processor(self) -> Any | None:
        if not self._processor_loaded:
            from monorl.rollout.media import load_processor

            self._processor = load_processor(self.config)
            self._processor_loaded = True
        return self._processor

    def server_info(self) -> dict[str, Any]:
        with self._calls_lock:
            request = self._send(self._http.get("/get_server_info"))
        try:
            response = request.result()
        except concurrent.futures.CancelledError:  # shutdown() dropped the request
            raise _ClientClosed(self._url) from None
        response.raise_for_status()
        return response.json()

    def _server_tp_size(self) -> int:
        if self._tp_size is None:
            info = self.server_info()
            tp = info.get("tp_size")
            self._tp_size = int(tp) if tp is not None else int(self.config.rollout.tensor_parallel_size)
        return self._tp_size

    def _check_server_drafts(self) -> None:
        """With ``model.mtp.rollout``: the server must draft with the model's MTP layers, and
        losslessly. Checked once, against the ``ServerArgs`` the server resolved."""
        if self._speculation is None or self._drafting_checked:
            return
        check_speculative_args(self.config, self.server_info(), source=f"sglang server {self._url}")
        self._drafting_checked = True

    def _tags_body(self) -> dict[str, Any]:
        return {"tags": self._sleep_tags} if self._sleep_tags is not None else {}

    def _flush_cache(self) -> None:
        response = self._post("/flush_cache")
        if response.status_code >= 400:
            log.warning(
                "sglang /flush_cache did not run (%s); stale radix entries may be served", response.text
            )

    def _send(self, coro: Coroutine[Any, Any, _T]) -> concurrent.futures.Future[_T]:
        """Run an ``httpx.AsyncClient`` coroutine on the client's event-loop thread. The caller
        holds ``_calls_lock``, under which :meth:`shutdown` closes the client: a request is
        refused here, or it is on the loop before ``shutdown()`` drops the requests there."""
        try:
            self._check_usable()
        except RuntimeError:
            coro.close()
            raise
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _post(self, path: str, *, call: _GenerateCall | None = None, **kwargs: Any) -> httpx.Response:
        """POST and wait for the response. With ``call`` the request belongs to that ``generate``
        call and :meth:`cancel` can abandon it."""
        with self._calls_lock:
            if call is not None and call.cancelled:
                raise GenerationCancelled("generation cancelled")
            request = self._send(self._http.post(path, **kwargs))
            if call is not None:
                call.request = request
        try:
            return request.result()
        except concurrent.futures.CancelledError:
            if call is None:  # cancel() cancels the requests of generate calls, shutdown() all
                raise _ClientClosed(self._url) from None
            raise GenerationCancelled("generation cancelled") from None
        except BaseException:
            request.cancel()  # interrupted while waiting (KeyboardInterrupt): drop the request too
            raise

    @staticmethod
    def _check(response: httpx.Response, op: str) -> None:
        payload: Any
        try:
            payload = response.json()
        except ValueError:
            payload = {"success": response.status_code < 400, "message": response.text}
        if response.status_code >= 400 or not payload.get("success", True):
            raise WeightSyncError(
                f"sglang {op} failed (HTTP {response.status_code}): {payload.get('message', payload)}"
            )
