"""``create_app()`` — the FastAPI application behind ``monorl serve`` / ``python -m monorl.server``.

Everything the API returns about a run comes from files in the run directory (see
:mod:`monorl.core.reporting`); backends are never imported here.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from monorl.config.recipes import recipes_dir
from monorl.core.errors import ConfigError, MonorlError
from monorl.server.auth import AccessGuard, allowed_hosts_from_env, resolve_token, write_token
from monorl.server.jobs import JobActive, JobManager, JobNotFound, default_home

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ServerSettings:
    home: Path = field(default_factory=default_home)
    recipes_dir: Path = field(default_factory=recipes_dir)
    web_dist: Path = field(
        default_factory=lambda: Path(os.environ.get("MONORL_WEB_DIST") or REPO_ROOT / "web" / "dist")
    )
    envs_bin: Path = field(
        default_factory=lambda: Path(os.environ.get("MONORL_ENVS_BIN") or REPO_ROOT / "envs" / "bin")
    )
    envs_sdk: Path = field(
        default_factory=lambda: Path(os.environ.get("MONORL_ENVS_SDK") or REPO_ROOT / "envs" / "sdk")
    )
    # Origins other than the server's own that may call the API from a browser. None by default:
    # the UI is served by this server, and the Vite dev server proxies /api under the API's origin.
    cors_origins: list[str] = field(default_factory=list)
    # Access token (``MONORL_SERVER_TOKEN``, else generated) and the hosts accepted in ``Host``
    # besides localhost / 127.0.0.1 / [::1] (``MONORL_ALLOWED_HOSTS``); see monorl.server.auth.
    token: str = field(default_factory=resolve_token)
    allowed_hosts: list[str] = field(default_factory=allowed_hosts_from_env)


def validation_errors(exc: BaseException) -> list[dict[str, Any]]:
    """Flatten a pydantic/FastAPI validation error (or a :class:`ConfigError` wrapping one) into
    ``[{loc, msg, type}]``."""
    if isinstance(exc, ConfigError) and isinstance(exc.__cause__, ValidationError):
        exc = exc.__cause__
    if isinstance(exc, (ValidationError, RequestValidationError)):
        return [
            {
                "loc": [str(p) for p in e.get("loc", ())],
                "msg": str(e.get("msg", "")),
                "type": str(e.get("type", "")),
            }
            for e in exc.errors()
        ]
    return [{"loc": [], "msg": str(exc), "type": "value_error"}]


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    app.state.jobs.shutdown()


def create_app(settings: ServerSettings | None = None, *, jobs: JobManager | None = None) -> FastAPI:
    settings = settings or ServerSettings()
    app = FastAPI(
        title="monorl",
        version=_version(),
        # No Swagger UI / ReDoc pages: they run a script from a CDN in the origin whose session
        # cookie starts jobs. The schema itself stays available (with the token like every route).
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.jobs = jobs or JobManager(settings.home)
    write_token(settings.home, settings.token)

    # Added first, so it runs inside CORS: a preflight carries no credentials and is answered by
    # the CORS middleware; every request that reaches a route passes the guard.
    app.add_middleware(
        AccessGuard,
        token=settings.token,
        allowed_hosts=settings.allowed_hosts,
        allowed_origins=settings.cors_origins,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"ok": False, "errors": validation_errors(exc)}, status_code=422)

    @app.exception_handler(ConfigError)
    async def _config_error(_: Request, exc: ConfigError) -> JSONResponse:
        return JSONResponse({"ok": False, "errors": validation_errors(exc)}, status_code=422)

    @app.exception_handler(JobNotFound)
    async def _job_not_found(_: Request, exc: JobNotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(JobActive)
    async def _job_active(_: Request, exc: JobActive) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(MonorlError)
    async def _monorl_error(_: Request, exc: MonorlError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def _not_found(_: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    from monorl.server.routes_hub import router as hub_router
    from monorl.server.routes_jobs import router as jobs_router
    from monorl.server.routes_system import router as system_router

    app.include_router(system_router, prefix="/api")
    app.include_router(hub_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")

    _mount_static(app, settings.web_dist)
    return app


def _version() -> str:
    from monorl import __version__

    return __version__


def _mount_static(app: FastAPI, dist: Path) -> None:
    index = dist / "index.html"
    if not index.is_file():
        return
    dist = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    async def _spa(path: str) -> FileResponse:
        if path == "api" or path.startswith("api/"):
            return JSONResponse({"detail": "Not Found"}, status_code=404)  # type: ignore[return-value]
        candidate = (dist / path).resolve() if path else index
        if candidate.is_file() and candidate.is_relative_to(dist):
            return FileResponse(candidate)
        return FileResponse(index)
