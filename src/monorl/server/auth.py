"""Who may talk to ``monorl serve``: the access token, the ``Host`` and ``Origin`` checks.

A job's config names commands to run, so a request that reaches the API runs code as the user
of the server. Three checks stand in front of every route, in this order:

* ``Host`` has to name this machine (``localhost``, ``127.0.0.1``, ``[::1]``) or a host that was
  allowed explicitly. A page that reaches the server by DNS rebinding carries its own name.
* A request with an unsafe method and an ``Origin`` header has to come from the server's own
  origin or from the CORS allow-list.
* Everything except a ``GET`` of the UI's static files needs the token, as
  ``Authorization: Bearer <token>`` or as the session cookie. A ``GET`` with a valid ``token``
  query parameter sets that cookie and redirects to the same URL without the parameter.

The token is written to ``$MONORL_HOME/server_token`` (mode 0600) for local clients.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode

from monorl.core.errors import ConfigError
from monorl.core.home import monorl_home

TOKEN_ENV = "MONORL_SERVER_TOKEN"
ALLOWED_HOSTS_ENV = "MONORL_ALLOWED_HOSTS"
TOKEN_FILE = "server_token"
TOKEN_PARAM = "token"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
WILDCARD_HOSTS = ("0.0.0.0", "::", "")  # noqa: S104 - recognised, never bound here
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")

_TOKEN_RE = re.compile(r"[A-Za-z0-9._~-]{16,}")

Scope = dict[str, Any]


def resolve_token(token: str | None = None) -> str:
    """``token``, else ``$MONORL_SERVER_TOKEN``, else a new random one."""
    token = token or os.environ.get(TOKEN_ENV) or None
    if token is None:
        return secrets.token_urlsafe(32)
    if not _TOKEN_RE.fullmatch(token):
        raise ConfigError(
            f"the server token (--token / {TOKEN_ENV}) was refused: it needs at least 16 characters "
            "out of letters, digits and '.', '_', '~', '-'. Leave it unset to have one generated."
        )
    return token


def token_path(home: Path) -> Path:
    return Path(home) / TOKEN_FILE


def write_token(home: Path, token: str) -> Path:
    """Store the token for local clients, readable by the owner only."""
    path = token_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)  # a file that existed keeps its mode through O_CREAT
        os.write(fd, token.encode() + b"\n")
    finally:
        os.close(fd)
    return path


def read_token(home: Path | None = None) -> str:
    """The token of the server that last started with this ``MONORL_HOME``, for API clients."""
    path = token_path(home if home is not None else monorl_home())
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        raise ConfigError(
            f"no server token at {path}: start `monorl serve` with the same MONORL_HOME first"
        ) from None


def parse_allowed_hosts(value: str | None) -> list[str]:
    """Split the comma-separated ``$MONORL_ALLOWED_HOSTS``."""
    return [h.strip() for h in (value or "").split(",") if h.strip()]


def allowed_hosts_from_env() -> list[str]:
    return parse_allowed_hosts(os.environ.get(ALLOWED_HOSTS_ENV))


def configure(bind_host: str, token: str | None = None, allowed_hosts: Iterable[str] = ()) -> str:
    """Settle the token and the accepted hosts before the app is built, and leave them in the
    environment, where ``create_app`` reads them (also in the worker process of ``--reload``).
    An address the server is bound to is accepted as ``Host``. Returns the token."""
    token = resolve_token(token)
    hosts = [*allowed_hosts_from_env(), *allowed_hosts]
    if bind_host not in WILDCARD_HOSTS:
        hosts.append(bind_host)
    os.environ[TOKEN_ENV] = token
    os.environ[ALLOWED_HOSTS_ENV] = ",".join(dict.fromkeys(hosts))
    return token


def startup_lines(bind_host: str, port: int, token: str) -> list[str]:
    """What ``monorl serve`` prints: where to sign in and, for a wildcard bind, what is refused."""
    lines = ["monorl server: open this URL to sign in", signin_url(bind_host, port, token)]
    if bind_host in WILDCARD_HOSTS:
        lines.append(
            f"listening on every interface ({bind_host}): requests need the token, and a Host other "
            f"than localhost / 127.0.0.1 / [::1] is refused unless --allowed-host or "
            f"{ALLOWED_HOSTS_ENV} names it"
        )
    return lines


def _hostname(host_header: str) -> str:
    """``Host`` without its port, lower-cased; an IPv6 literal keeps its brackets."""
    host = host_header.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else ""
    return host.rsplit(":", 1)[0] if ":" in host else host


def _normalise_host(host: str) -> str:
    host = host.strip().lower()
    if ":" in host and not host.startswith("["):
        # A bare IPv6 address, as --host and --allowed-host take it.
        return f"[{host}]" if host.count(":") > 1 else host.rsplit(":", 1)[0]
    return _hostname(host)


def browser_host(bind_host: str) -> str:
    """The host to put in a URL for a server bound to ``bind_host``."""
    if bind_host in WILDCARD_HOSTS:
        return "127.0.0.1"
    return f"[{bind_host}]" if ":" in bind_host and not bind_host.startswith("[") else bind_host


def signin_url(bind_host: str, port: int, token: str) -> str:
    """The URL that signs a browser in: it sets the session cookie and redirects to the UI."""
    return f"http://{browser_host(bind_host)}:{port}/?{TOKEN_PARAM}={quote(token, safe='')}"


def cookie_name(host_header: str) -> str:
    """Cookies ignore ports, so two servers on one host need two names: the port the browser
    addresses (the one in ``Host``) is part of it."""
    port = host_header.strip().rsplit(":", 1)[-1]
    return f"monorl_session_{port if port.isdigit() else '80'}"


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key == name:
            return value.decode("latin-1")
    return None


def _cookie(scope: Scope, name: str) -> str | None:
    for part in (_header(scope, b"cookie") or "").split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key == name:
            return value
    return None


def _same(a: str, b: str) -> bool:
    return secrets.compare_digest(a.encode(), b.encode())


class AccessGuard:
    """ASGI middleware that applies the three checks of this module (see its docstring)."""

    def __init__(
        self,
        app: Any,
        *,
        token: str,
        allowed_hosts: Iterable[str] = (),
        allowed_origins: Iterable[str] = (),
    ) -> None:
        self.app = app
        self.token = token
        self.hosts = {*LOCAL_HOSTS, *(_normalise_host(h) for h in allowed_hosts)} - {""}
        self.origins = {o.rstrip("/").lower() for o in allowed_origins}

    async def __call__(self, scope: Scope, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":  # the server has no websocket routes
            await receive()
            await send({"type": "websocket.close", "code": 1008})
            return

        host = _header(scope, b"host") or ""
        if _hostname(host) not in self.hosts:
            await _refuse(
                send,
                403,
                f"Refused: this server does not answer for the host {host[:80]!r}. It accepts "
                "localhost, 127.0.0.1 and [::1]; start it with `--allowed-host NAME` (repeatable) "
                f"or {ALLOWED_HOSTS_ENV}=NAME[,NAME] to add the name you reach it by.",
            )
            return

        method = scope["method"]
        origin = _header(scope, b"origin")
        if method not in SAFE_METHODS and origin is not None:
            origin = origin.rstrip("/").lower()
            scheme, _, authority = origin.partition("://")
            own = scheme in ("http", "https") and authority == host.strip().lower()
            if not own and origin not in self.origins:
                await _refuse(
                    send,
                    403,
                    f"Refused: a {method} request from the origin {origin[:80]!r}, which is not "
                    "this server. Use the UI at the address the server is reached by.",
                )
                return

        query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        offered = [v for k, v in query if k == TOKEN_PARAM]
        if method == "GET" and offered:
            if not _same(offered[0], self.token):
                await _refuse(send, 401, _signin_help("The token in this URL is not valid."))
                return
            rest = urlencode([(k, v) for k, v in query if k != TOKEN_PARAM])
            # One leading slash: "//name" would send the browser to another host.
            location = "/" + quote(scope["path"].lstrip("/"), safe="/") + (f"?{rest}" if rest else "")
            cookie = f"{cookie_name(host)}={self.token}; HttpOnly; SameSite=Strict; Path=/"
            await _respond(
                send,
                303,
                b"",
                [
                    (b"location", location.encode()),
                    (b"set-cookie", cookie.encode()),
                    (b"cache-control", b"no-store"),
                    (b"referrer-policy", b"no-referrer"),
                ],
            )
            return

        path = scope["path"]
        public = method in ("GET", "HEAD") and not (path == "/api" or path.startswith("/api/"))
        if not public and not self._signed_in(scope, host):
            await _refuse(send, 401, _signin_help("Not signed in."))
            return
        await self.app(scope, receive, send)

    def _signed_in(self, scope: Scope, host: str) -> bool:
        authorization = _header(scope, b"authorization") or ""
        kind, _, value = authorization.partition(" ")
        if kind.lower() == "bearer" and _same(value.strip(), self.token):
            return True
        cookie = _cookie(scope, cookie_name(host))
        return cookie is not None and _same(cookie, self.token)


def _signin_help(what: str) -> str:
    return (
        f"{what} Open the sign-in URL that `monorl serve` printed when it started "
        f"(http://127.0.0.1:<port>/?{TOKEN_PARAM}=...). An API client sends "
        f"`Authorization: Bearer <token>` with the token in $MONORL_HOME/{TOKEN_FILE}."
    )


async def _refuse(send: Any, status: int, detail: str) -> None:
    headers = [(b"content-type", b"application/json"), (b"cache-control", b"no-store")]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await _respond(send, status, json.dumps({"detail": detail}).encode(), headers)


async def _respond(send: Any, status: int, body: bytes, headers: list[tuple[bytes, bytes]]) -> None:
    headers = [*headers, (b"content-length", str(len(body)).encode())]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
