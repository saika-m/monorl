"""monorl vLLM server: vLLM's OpenAI-compatible API server plus weight-update endpoints.

Run with the same arguments as ``vllm serve``::

    python -m monorl.adapters.vllm.server <model> --tensor-parallel-size 2 --enable-sleep-mode

What it adds on top of vLLM v0.33.0's launcher (``vllm/entrypoints/launchers/api_server/entry.py``):

* ``VLLM_SERVER_DEV_MODE=1`` so vLLM mounts its own ``/sleep``, ``/wake_up``, ``/is_sleeping``,
  ``/reset_prefix_cache``, ``/pause``, ``/resume`` (``vllm/entrypoints/serve/dev/*`` and
  ``vllm/entrypoints/rl/online/api_router.py``).
* ``--worker-extension-cls monorl.adapters.vllm.worker.MonorlWorkerExtension`` by default.
* The ``vllm.server.monorl_routes`` patch, which mounts :func:`router` on the app:

  - ``POST /monorl/update_weights`` with several tensors as body
    (:func:`monorl.adapters.vllm.common.encode_weights`, a versioned format) → one
    ``collective_rpc("monorl_update_weights")``; an entry marked ``mtp`` is a checkpoint key of an
    MTP layer, which goes to the drafter only. The tensors handed on are views of the received
    body, which is read straight into one buffer: the server holds each request once
  - ``POST /monorl/finish_weight_update`` after the last parameter of an update, when the server
    drafts → ``collective_rpc("monorl_refresh_drafter")``
  - ``POST /load_weights_from_path`` ``{"path": ...}`` → ``collective_rpc("monorl_load_weights_from_path")``
  - ``GET /monorl/info`` → per-worker device/dtype and the drafting the worker does

For ``model.mtp.rollout`` start the server with the drafter, e.g.
``--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'``; the client refuses a
server that does not draft with the MTP layers and exact rejection sampling, and reads the
acceptance counters from vLLM's own ``GET /metrics`` (so do not pass ``--disable-log-stats``).

:func:`router` / :func:`attach_router` import neither vLLM nor torch-heavy modules beyond
``torch`` itself, so the routes are unit-testable with a fake ``app.state.engine_client``
(``AsyncLLM.collective_rpc(method, timeout, args, kwargs)`` is awaited).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from monorl.adapters.vllm.common import WORKER_EXTENSION_CLS, decode_weights

router = APIRouter(tags=["monorl"])


def _engine_client(request: Request) -> Any:
    client = getattr(request.app.state, "engine_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="engine not initialised yet")
    return client


async def _read_body(request: Request) -> bytearray:
    """The request body in a buffer of its own. With a ``Content-Length`` the buffer has its
    final size from the start, so the body is never held twice (``Request.body()`` keeps the
    chunks and their concatenation, and a growing buffer is copied when it moves)."""
    declared = request.headers.get("content-length")
    if declared is None:
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
        return body
    try:
        body = bytearray(int(declared))
    except (ValueError, MemoryError) as e:
        raise HTTPException(status_code=400, detail=f"bad Content-Length {declared!r}: {e}") from e
    filled = 0
    async for chunk in request.stream():
        end = filled + len(chunk)
        if end > len(body):
            raise HTTPException(status_code=400, detail=f"body is longer than its Content-Length {declared}")
        body[filled:end] = chunk
        filled = end
    if filled != len(body):
        raise HTTPException(
            status_code=400, detail=f"body has {filled} bytes, its Content-Length says {declared}"
        )
    return body


@router.post("/monorl/update_weights")
async def update_weights(request: Request) -> JSONResponse:
    body = await _read_body(request)
    try:
        weights, mtp_weights = decode_weights(body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    args = (weights, mtp_weights) if mtp_weights else (weights,)
    results = await _engine_client(request).collective_rpc("monorl_update_weights", args=args)
    return JSONResponse({"tensors": len(weights) + len(mtp_weights), "loaded": [int(r) for r in results]})


@router.post("/monorl/finish_weight_update")
async def finish_weight_update(request: Request) -> JSONResponse:
    await _engine_client(request).collective_rpc("monorl_refresh_drafter")
    return JSONResponse({"status": "finished"})


@router.post("/load_weights_from_path")
async def load_weights_from_path(request: Request) -> JSONResponse:
    body = await request.json()
    path = body.get("path") if isinstance(body, dict) else None
    if not path:
        raise HTTPException(status_code=400, detail="body must be {'path': <hf model dir>}")
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"{path} is not a directory on the server host")
    await _engine_client(request).collective_rpc("monorl_load_weights_from_path", args=(str(path),))
    return JSONResponse({"path": str(path), "status": "loaded"})


@router.get("/monorl/info")
async def info(request: Request) -> JSONResponse:
    results = await _engine_client(request).collective_rpc("monorl_device_info")
    return JSONResponse({"workers": list(results), "worker_extension_cls": WORKER_EXTENSION_CLS})


def attach_router(app: FastAPI) -> None:
    app.include_router(router)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point: ``python -m monorl.adapters.vllm.server <vllm serve args>``."""
    os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    from monorl.adapters.vllm.engine import import_vllm

    import_vllm()
    import uvloop
    from vllm.entrypoints.common.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.common.startup import cli_env_setup
    from vllm.entrypoints.launchers.api_server.entry import run_server
    from vllm.logger import configure_logging_from_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    from monorl.patching import apply

    report = apply("vllm")
    if "vllm.server.monorl_routes" not in report.applied and not any(
        pid == "vllm.server.monorl_routes" and reason == "already applied" for pid, reason in report.skipped
    ):
        raise SystemExit(
            f"could not install monorl routes on the vLLM server: {report.failed or report.skipped}"
        )

    cli_env_setup()
    parser = FlexibleArgumentParser(description="monorl vLLM server (vllm serve + weight-update endpoints).")
    parser = make_arg_parser(parser)
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if not getattr(args, "worker_extension_cls", None):
        args.worker_extension_cls = WORKER_EXTENSION_CLS
    configure_logging_from_args(args)
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
