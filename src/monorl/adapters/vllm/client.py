"""Server-mode vLLM rollout engine: an HTTP client for :mod:`monorl.adapters.vllm.server`.

Implements the same :class:`~monorl.core.interfaces.RolloutEngine` protocol as the colocated
engine without importing vLLM, so the trainer process can live on another machine or in an
environment where vLLM is not installable.

Endpoints used (vLLM v0.33.0 unless noted):

* ``POST /v1/completions`` (``vllm/entrypoints/openai/completion/protocol.py``): ``prompt`` as a
  list of token-id lists, ``return_token_ids=true`` → ``choices[i].token_ids`` /
  ``prompt_token_ids``; ``logprobs=0`` → ``choices[i].logprobs.token_logprobs``. With
  ``SamplingParams.top_logprobs = k``: ``logprobs=k`` and ``return_tokens_as_token_ids=true`` →
  ``choices[i].logprobs.top_logprobs`` (``{"token_id:<id>": logprob}`` per position). The server
  must allow that many: ``--max-logprobs`` ≥ ``k`` (``k + 1`` for the chat endpoint; default 20)
  and ``--logprobs-mode processed_logprobs``. A request has one ``max_tokens`` for all its
  prompts (``CompletionRequest.max_tokens``), so the text prompts of a ``generate`` call go out
  as one request per distinct cap.
* ``POST /v1/chat/completions`` (``vllm/entrypoints/openai/chat_completion/protocol.py``) for
  prompts with image parts: ``messages`` with ``image_url`` data URLs, ``logprobs=true,
  top_logprobs=0`` and ``return_token_ids=true``. The chat endpoint's logprob entries only carry
  token *strings*, so token ids come from ``choices[i].token_ids`` and the prompt ids the server
  conditioned on (chat template + image placeholders rendered server-side) from the
  response-level ``prompt_token_ids``. One request per prompt (a chat request is one
  conversation). The completion and chat requests of a ``generate`` call are all in flight
  together; the responses are regrouped into the order of the prompts.
* ``POST /sleep?level=N``, ``POST /wake_up``, ``GET /is_sleeping``
  (``vllm/entrypoints/serve/dev/sleep/api_router.py``, enabled by ``VLLM_SERVER_DEV_MODE=1``).
* ``POST /reset_prefix_cache`` (``vllm/entrypoints/serve/dev/cache/api_router.py``).
* ``POST /monorl/update_weights``, ``POST /load_weights_from_path``,
  ``POST /monorl/finish_weight_update``, ``GET /monorl/info`` — monorl additions. A weight update
  goes out as requests of at most ``rollout.extra.weight_request_bytes`` of tensor data each (a
  larger tensor has a request to itself), streamed from the tensors' own memory.
* ``GET /metrics`` (``vllm/entrypoints/serve/instrumentator/metrics.py``): Prometheus text with
  the speculative-decoding counters (``vllm/v1/spec_decode/metrics.py``), absent when the server
  runs with ``--disable-log-stats`` or does not draft.

``cancel()`` relies on the server dropping a generation whose client went away: both completion
routes are wrapped in ``with_cancellation`` (``vllm/entrypoints/serve/utils/api_utils.py``), which
cancels the handler on ``http.disconnect``, and ``AsyncLLM.generate``
(``vllm/v1/engine/async_llm.py``) aborts its request in the engine when it is cancelled. The
requests therefore run as tasks of an ``httpx.AsyncClient`` on a private event-loop thread:
cancelling a task closes its connection, which a thread blocked in a synchronous ``httpx.Client``
call could not be made to do. ``shutdown()`` drops every request in flight the same way before it
stops that thread, which exists only in the process that built the client.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
from collections.abc import AsyncIterator, Coroutine, Iterable, Iterator, Sequence
from typing import Any, TypeVar

import httpx
import torch

from monorl.adapters.vllm.common import (
    DEFAULT_IMAGE_CACHE_BYTES,
    WEIGHTS_MEDIA_TYPE,
    ImageUrlCache,
    SpeculationCounters,
    chat_completion_kwargs,
    chat_completion_messages,
    chunk_weights,
    completions_from_chat_response,
    completions_from_choices,
    encode_weights,
    is_media_prompt,
    prometheus_samples,
    prompt_token_ids,
    sampling_kwargs,
    spec_decode_totals,
    speculative_config,
    speculative_serve_flag,
)
from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError, GenerationCancelled
from monorl.core.types import Completion, PromptInput, SamplingParams, SpeculationStats, prompt_caps
from monorl.mtp import checkpoint_key, is_wire_name

log = logging.getLogger(__name__)
_T = TypeVar("_T")
_ALL_TAGS: tuple[str, ...] = ("weights", "kv_cache")
_UPDATE_WEIGHTS_PATH = "/monorl/update_weights"
DEFAULT_WEIGHT_REQUEST_BYTES = 64 * 1024 * 1024


class _StatusError(RuntimeError):
    """The server answered a request with an error status."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class _ClientClosed(RuntimeError):
    """A request was made after ``shutdown()``, or ``shutdown()`` dropped it."""

    def __init__(self, url: str) -> None:
        super().__init__(f"the vLLM server client for {url} was shut down")


async def _stream(pieces: Iterator[bytes | memoryview]) -> AsyncIterator[bytes | memoryview]:
    for piece in pieces:
        yield piece


def server_url(config: RunConfig, replica: int) -> str:
    """``rollout.extra.url`` (one server) or ``rollout.extra.urls[replica]`` (one per replica)."""
    extra = config.rollout.extra
    urls = extra.get("urls")
    if urls:
        if replica >= len(urls):
            raise ConfigError(
                f"rollout.extra.urls has {len(urls)} entries but replica {replica} was requested"
            )
        return str(urls[replica]).rstrip("/")
    url = extra.get("url")
    if not url:
        raise ConfigError(
            "rollout.extra.mode='server' needs rollout.extra.url (or urls) of a monorl vLLM server"
        )
    return str(url).rstrip("/")


class _GenerateCall:
    """One ``generate`` call that has not returned yet; fields are guarded by the client's
    ``_calls_lock``."""

    def __init__(self) -> None:
        self.cancelled = False
        self.request: concurrent.futures.Future[Any] | None = None
        """The HTTP request, or the group of concurrent ones, the call is blocked on, if any."""


class VllmServerClient:
    """``rollout_engine: vllm`` with ``rollout.extra.mode='server'``.

    ``rollout.extra`` keys: ``url``/``urls``, ``model_name`` (served model name, defaults to
    ``model.name_or_path``), ``timeout`` seconds per request (default: none), ``sleep_level``,
    ``weight_request_bytes`` (tensor bytes per weight-update request, default 64 MiB: what the
    server holds of an update at a time, unless a single tensor is larger) and
    ``image_cache_bytes`` (bound on the image data URLs kept for re-sending, default 256 MiB).

    Text prompts are one completion request per distinct generation cap (one request, unless
    the call carries caps per prompt) and prompts with images one chat request each. All the
    requests of a call are sent together: a server-side error of any of them fails the call with
    that request's error and drops the others.

    ``cancel()`` disconnects the completion requests in flight, which makes the server abort
    their generation, and fails the ``generate`` calls waiting for the lock; all of them raise
    :class:`GenerationCancelled`. ``shutdown()`` does the same and ends every other request in
    flight too; calls made after it raise ``RuntimeError``. So does every call in a process
    forked from the one that built the client, where the thread that runs the requests does not
    exist: a process builds its own client.

    With ``model.mtp.rollout`` the server has to draft with the model's MTP layers: the client
    checks what ``/monorl/info`` reports against the config when it is created and raises
    :class:`ConfigError` naming the ``--speculative-config`` to start the server with. MTP weights
    of an update are then sent marked for the drafter; without ``model.mtp.rollout`` they are
    left out. ``speculation_stats()`` reports the drafting since its previous call, server-wide:
    replicas sharing one server each report it, which leaves the rates derived from the sum right.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        replica: int = 0,
        transport: httpx.AsyncBaseTransport | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        extra = config.rollout.extra
        speculative = speculative_config(config)
        self._drafting = speculative is not None
        self._config = config
        self._url = server_url(config, replica)
        self._model_name = str(extra.get("model_name") or config.model.name_or_path)
        self._sleep_level = int(extra.get("sleep_level", 1))
        self._sleep_enabled = bool(config.rollout.enable_sleep_mode)
        if tokenizer is None:
            from monorl.rollout.hf_engine import load_tokenizer

            tokenizer = load_tokenizer(config)
        self._tokenizer = tokenizer
        self._weight_request_bytes = int(extra.get("weight_request_bytes", DEFAULT_WEIGHT_REQUEST_BYTES))
        self._images = ImageUrlCache(int(extra.get("image_cache_bytes", DEFAULT_IMAGE_CACHE_BYTES)))
        timeout = extra.get("timeout")
        self._client = httpx.AsyncClient(
            base_url=self._url,
            # The timeout bounds the server's answer, not the wait for a free connection: the
            # chat requests of one generate call can outnumber the pool.
            timeout=httpx.Timeout(None if timeout is None else float(timeout), pool=None),
            transport=transport,
        )
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name=f"monorl-vllm-client-{replica}", daemon=True
        )
        self._loop_thread.start()
        self._pid = os.getpid()  # the only process the loop thread runs in
        self._lock = threading.RLock()
        self._calls: set[_GenerateCall] = set()
        self._calls_lock = threading.Lock()
        self._closed = False  # guarded by _calls_lock: no request is scheduled once it is set
        self._sleeping_tags: set[str] = set()  # vLLM memory tags currently released
        self._speculation: SpeculationCounters | None = None
        if speculative is not None:
            try:
                self._check_server_drafts(speculative)
                self._speculation = SpeculationCounters(self._spec_decode_totals())
            except BaseException:
                self.shutdown()
                raise

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
        caps = prompt_caps(params, len(prompts), max_new_tokens)
        text_idx: dict[int, list[int]] = {}  # generation cap -> its text prompts
        for i, p in enumerate(prompts):
            if not is_media_prompt(p):
                text_idx.setdefault(caps[i], []).append(i)
        media_idx = [i for i, p in enumerate(prompts) if is_media_prompt(p)]
        result: list[list[Completion] | None] = [None] * len(prompts)
        call = _GenerateCall()
        with self._calls_lock:
            self._calls.add(call)
        try:
            with self._lock:
                self._wake(_ALL_TAGS)
                ids = {
                    cap: [prompt_token_ids(self._tokenizer, prompts[i]) for i in indices]
                    for cap, indices in text_idx.items()
                }
                requests: list[tuple[str, dict[str, Any]]] = []
                for cap, cap_ids in ids.items():
                    body: dict[str, Any] = {
                        "model": self._model_name,
                        "prompt": cap_ids,
                        "return_token_ids": True,
                        **sampling_kwargs(params, cap),
                    }
                    if params.top_logprobs > 0:
                        body["return_tokens_as_token_ids"] = True
                    requests.append(("/v1/completions", body))
                for i in media_idx:
                    body = {
                        "model": self._model_name,
                        "messages": chat_completion_messages(prompts[i], self._images),  # type: ignore[arg-type]
                        **chat_completion_kwargs(params, caps[i]),
                    }
                    requests.append(("/v1/chat/completions", body))
                responses = self._run(self._post_each(requests), call)
                text, chat = responses[: len(text_idx)], responses[len(text_idx) :]
                for (cap, indices), resp in zip(text_idx.items(), text, strict=True):
                    grouped = completions_from_choices(
                        resp.json()["choices"],
                        ids[cap],
                        params.n,
                        want_logprobs=params.logprobs,
                        top_logprobs=params.top_logprobs,
                    )
                    for i, comps in zip(indices, grouped, strict=True):
                        result[i] = comps
                for i, resp in zip(media_idx, chat, strict=True):
                    result[i] = completions_from_chat_response(
                        resp.json(),
                        params.n,
                        want_logprobs=params.logprobs,
                        top_logprobs=params.top_logprobs,
                    )
        except _ClientClosed:
            if call.cancelled:  # shutdown() while this call was in flight
                raise GenerationCancelled("generation cancelled") from None
            raise
        finally:
            with self._calls_lock:
                self._calls.discard(call)
        return [r for r in result if r is not None]

    def cancel(self) -> None:
        """Abort the ``generate`` calls in flight (they raise :class:`GenerationCancelled`); a
        no-op when idle. The client stays usable: later calls run normally."""
        with self._calls_lock:
            for call in self._calls:
                call.cancelled = True
                if call.request is not None:
                    call.request.cancel()

    def update_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self._check_usable()
        if not self._drafting:
            # The server does not draft: the MTP layers the trainer holds are not its business.
            weights = ((name, tensor) for name, tensor in weights if not is_wire_name(name))
        with self._lock:
            self._wake(("weights",))
            for chunk in chunk_weights(weights, max_bytes=self._weight_request_bytes):
                # Tagged names are MTP weights: to the drafter, under their checkpoint keys.
                length, pieces = encode_weights(
                    [
                        (checkpoint_key(name), tensor, True) if is_wire_name(name) else (name, tensor, False)
                        for name, tensor in chunk
                    ]
                )
                try:
                    self._post(
                        _UPDATE_WEIGHTS_PATH,
                        content=_stream(pieces),
                        headers={"content-type": WEIGHTS_MEDIA_TYPE, "content-length": str(length)},
                    )
                except _StatusError as e:
                    if e.status != 404:
                        raise
                    raise RuntimeError(
                        f"the vLLM server {self._url} has no {_UPDATE_WEIGHTS_PATH}: it is not a monorl "
                        "server, or one of a monorl version with another weight transfer; run the same "
                        "monorl version on the trainer and on the vLLM server"
                    ) from None
            if self._drafting:
                self._post("/monorl/finish_weight_update")
            self._post("/reset_prefix_cache")

    def speculation_stats(self) -> SpeculationStats | None:
        """Drafting on the server since the previous call (reading resets it); ``None`` when
        ``model.mtp.rollout`` is off."""
        if self._speculation is None:
            return None
        return self._speculation.delta(self._spec_decode_totals())

    def load_weights_from_path(self, path: str | os.PathLike[str]) -> None:
        self._check_usable()
        with self._lock:
            self._wake(("weights",))
            self._post("/load_weights_from_path", json={"path": os.fspath(path)})
            self._post("/reset_prefix_cache")

    def sleep(self) -> None:
        if not self._sleep_enabled:
            return
        self._check_usable()
        with self._lock:
            if self._sleeping_tags:
                return
            self._post("/sleep", params={"level": self._sleep_level})
            self._sleeping_tags = set(_ALL_TAGS)

    def wake_up(self) -> None:
        if not self._sleep_enabled:
            return
        self._check_usable()
        with self._lock:
            self._wake(_ALL_TAGS)

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

    # -- internals ------------------------------------------------------------------------
    def _check_usable(self) -> None:
        """Raise unless this process can run requests: the event-loop thread is not copied into
        a forked process, and :meth:`shutdown` stops it. A call waiting for a request there
        would wait forever. Public methods check before they take a lock, which a fork may have
        copied while another thread held it."""
        if os.getpid() != self._pid:
            raise RuntimeError(
                f"the vLLM server client for {self._url} was built in process {self._pid} and is used in "
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
        await self._client.aclose()

    def _check_server_drafts(self, speculative: dict[str, Any]) -> None:
        """Raise :class:`ConfigError` unless every worker of the server drafts the way
        ``model.mtp.rollout`` compiles to (``speculative``)."""
        hint = f"start the server with {speculative_serve_flag(speculative)}"
        workers = self._request("GET", "/monorl/info").json().get("workers") or []
        for worker in workers or [{}]:
            served = worker.get("speculative")
            if not served:
                raise ConfigError(
                    f"model.mtp.rollout is set but the vLLM server {self._url} does not draft; {hint}"
                )
            method = str(served.get("method"))
            if not (method == "mtp" or method.endswith("_mtp")):
                raise ConfigError(
                    f"model.mtp.rollout is set but the vLLM server {self._url} drafts with method "
                    f"{method!r}, not with the model's MTP layers; {hint}"
                )
            if served.get("rejection_sample_method", "standard") != "standard":
                raise ConfigError(
                    f"the vLLM server {self._url} runs rejection_sample_method="
                    f"{served['rejection_sample_method']!r}, under which the recorded logprobs are not those "
                    f"of the distribution the tokens are sampled from; {hint}"
                )
            wanted = self._config.model.mtp.num_speculative_tokens
            if wanted is not None and served.get("num_speculative_tokens") != wanted:
                raise ConfigError(
                    f"model.mtp.num_speculative_tokens={wanted} but the vLLM server {self._url} drafts "
                    f"{served.get('num_speculative_tokens')} tokens per step; {hint}"
                )

    def _spec_decode_totals(self) -> tuple[int, int, int]:
        totals = spec_decode_totals(prometheus_samples(self._request("GET", "/metrics").text))
        if totals is None:
            raise ConfigError(
                f"the vLLM server {self._url} exposes no speculative-decoding counters on /metrics; "
                "start it without --disable-log-stats"
            )
        return totals

    def _wake(self, tags: tuple[str, ...]) -> None:
        """Re-map only the tags currently released (``/wake_up?tags=weights&tags=kv_cache``);
        weights must be mapped before loading into them, the KV cache only before generating."""
        pending = [t for t in tags if t in self._sleeping_tags]
        if not pending:
            return
        self._post("/wake_up", params=[("tags", t) for t in pending])
        self._sleeping_tags.difference_update(pending)

    def _post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self._request("POST", path, **kwargs)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send a request on the client's event loop and wait for the response."""
        return self._run(self._send(method, path, **kwargs), None)

    def _run(self, sending: Coroutine[Any, Any, _T], call: _GenerateCall | None) -> _T:
        """Run ``sending`` on the client's event loop and wait for its result. With ``call`` it
        belongs to that ``generate`` call and :meth:`cancel` can abandon it."""
        with self._calls_lock:
            if call is not None and call.cancelled:
                sending.close()
                raise GenerationCancelled("generation cancelled")
            try:
                self._check_usable()
            except RuntimeError:
                sending.close()
                raise
            request = asyncio.run_coroutine_threadsafe(sending, self._loop)
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

    async def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        resp = await self._client.request(method, path, **kwargs)
        if resp.status_code >= 400:
            raise _StatusError(
                f"vLLM server {self._url}{path} -> {resp.status_code}: {resp.text[:500]}", resp.status_code
            )
        return resp

    async def _post_each(self, requests: Sequence[tuple[str, dict[str, Any]]]) -> list[httpx.Response]:
        """POST every ``(path, body)`` at once; the responses in the order of ``requests``. The
        first request to fail fails the group with its error (the earliest in ``requests`` when
        several have failed by then), and the requests still in flight are dropped, as they are
        when the group is cancelled."""
        tasks = [asyncio.ensure_future(self._send("POST", path, json=body)) for path, body in requests]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in tasks:
                if task.done() and task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
            return [task.result() for task in tasks]
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
