"""Dataset URI parsing and filesystem resolution.

Supported shapes:

* ``hf://org/name[@revision][/path-in-repo]`` — Hugging Face Hub dataset repo (``hf://datasets/``
  prefix tolerated). An optional path narrows ``load_dataset`` to those ``data_files``.
* ``s3://bucket/prefix-or-file`` and any other fsspec scheme (``gs://``, ``az://``, ...) when the
  driver is installed. S3 credentials come **only** from the environment
  (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, ``AWS_SESSION_TOKEN``,
  ``AWS_DEFAULT_REGION``/``AWS_REGION``, ``AWS_ENDPOINT_URL``), never from config.
* ``http://host/file.jsonl`` or ``https://``: one file a web server serves, in the file formats
  of a local file. A web server lists nothing, so a directory or a glob is not read this way.
* local path or ``file://`` — file, directory or glob. A string without ``://`` is always a
  path: a Hub dataset needs the ``hf://`` prefix.
* ``<scheme>://anything`` for a scheme a provider module registered a ``data_source`` for
  (:mod:`monorl.data.rows`). The scheme has to be an identifier (letters, digits, ``_``).
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from monorl.core.errors import DataError

Kind = Literal["local", "hf", "remote"]

HTTP_SCHEMES: tuple[str, ...] = ("http", "https")

S3_ENV_VARS: tuple[str, ...] = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    "AWS_REGION",
    "AWS_ENDPOINT_URL",
)


@dataclass(frozen=True, slots=True)
class ParsedURI:
    raw: str
    kind: Kind
    scheme: str
    """``file``, ``hf``, or for ``remote`` what precedes ``://``: an fsspec protocol (``s3``,
    ``gs``...) or the scheme of a provider's data source."""
    path: str
    """Local: absolute path. Remote: what follows ``://`` without surrounding slashes
    (``bucket/key``); for ``http`` and ``https`` the whole URL, which is how fsspec's filesystem
    for them names a file. HF: ``org/name``."""
    revision: str | None = None
    """HF only: ``@revision``."""
    subpath: str | None = None
    """HF only: path inside the repo (used as ``data_files``)."""

    @property
    def repo_id(self) -> str:
        return self.path

    def with_path(self, path: str) -> str:
        """Render a URI for a path on the same filesystem (used for per-file ids and browse)."""
        if self.kind == "local" or self.scheme in HTTP_SCHEMES:
            return path
        return f"{self.scheme}://{path}"


def parse_uri(uri: str) -> ParsedURI:
    if not uri or not uri.strip():
        raise DataError("dataset uri is empty")
    uri = uri.strip()
    if uri.startswith("hf://"):
        return _parse_hf(uri)
    if uri.startswith("file://"):
        local = uri[len("file://") :]
        return ParsedURI(raw=uri, kind="local", scheme="file", path=str(Path(local).expanduser().absolute()))
    if "://" in uri:
        scheme, _, rest = uri.partition("://")
        if not scheme.isidentifier():
            raise DataError(f"unsupported dataset uri {uri!r}")
        if scheme in HTTP_SCHEMES:
            return ParsedURI(raw=uri, kind="remote", scheme=scheme, path=uri)
        return ParsedURI(raw=uri, kind="remote", scheme=scheme, path=rest.strip("/") if rest != "" else "")
    return ParsedURI(raw=uri, kind="local", scheme="file", path=str(Path(uri).expanduser().absolute()))


def _parse_hf(uri: str) -> ParsedURI:
    body = uri[len("hf://") :].strip("/")
    parts = [p for p in body.split("/") if p]
    if len(parts) >= 3 and parts[0] == "datasets":
        parts = parts[1:]
    if len(parts) < 2:
        raise DataError(f"hf uri must look like hf://org/name[@revision]; got {uri!r}")
    org, name = parts[0], parts[1]
    revision = None
    if "@" in name:
        name, _, revision = name.partition("@")
        revision = revision or None
    subpath = "/".join(parts[2:]) or None
    if not org or not name:
        raise DataError(f"hf uri must look like hf://org/name[@revision]; got {uri!r}")
    return ParsedURI(
        raw=uri, kind="hf", scheme="hf", path=f"{org}/{name}", revision=revision, subpath=subpath
    )


# -- credentials / storage options -----------------------------------------------------------


def s3_storage_options() -> dict[str, Any]:
    """s3fs constructor kwargs built from the AWS_* environment. Empty values are omitted so
    botocore's default chain (IAM role, config files) still applies when nothing is set."""
    env = os.environ
    opts: dict[str, Any] = {}
    if env.get("AWS_ACCESS_KEY_ID"):
        opts["key"] = env["AWS_ACCESS_KEY_ID"]
    if env.get("AWS_SECRET_ACCESS_KEY"):
        opts["secret"] = env["AWS_SECRET_ACCESS_KEY"]
    if env.get("AWS_SESSION_TOKEN"):
        opts["token"] = env["AWS_SESSION_TOKEN"]
    client_kwargs: dict[str, Any] = {}
    region = env.get("AWS_DEFAULT_REGION") or env.get("AWS_REGION")
    if region:
        client_kwargs["region_name"] = region
    if env.get("AWS_ENDPOINT_URL"):
        client_kwargs["endpoint_url"] = env["AWS_ENDPOINT_URL"]
    if client_kwargs:
        opts["client_kwargs"] = client_kwargs
    return opts


def s3_env_status() -> dict[str, Any]:
    """Non-secret summary of the S3 connector for the UI (``/api/connectors``)."""
    env = os.environ
    present = [v for v in S3_ENV_VARS if env.get(v)]
    return {
        "configured": bool(env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY")),
        "region": env.get("AWS_DEFAULT_REGION") or env.get("AWS_REGION"),
        "endpoint_url": env.get("AWS_ENDPOINT_URL"),
        "vars_present": present,
    }


def storage_options_for(scheme: str) -> dict[str, Any]:
    if scheme in ("s3", "s3a"):
        return s3_storage_options()
    return {}


def filesystem_for(parsed: ParsedURI):
    """Return an fsspec filesystem for a local or remote URI."""
    import fsspec

    if parsed.kind == "hf":
        raise DataError("hf:// datasets are loaded through `datasets`, not fsspec")
    scheme = "file" if parsed.kind == "local" else parsed.scheme
    try:
        cls = fsspec.get_filesystem_class(scheme)
        # A filesystem is opened from its scheme and the environment. One that takes more than
        # that (ftp and smb a host, github an org and a repo) cannot be, and is not constructed.
        needs = [
            name
            for name, param in list(inspect.signature(cls.__init__).parameters.items())[1:]
            if param.default is param.empty and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        ]
        if not needs:
            return cls(**storage_options_for(scheme))
    except (ValueError, ImportError) as e:  # unknown protocol or driver missing
        raise DataError(f"no fsspec driver for scheme {scheme!r} ({e}); install the matching package") from e
    raise DataError(
        f"{scheme}:// is not supported as a dataset location: its fsspec filesystem needs "
        f"{', '.join(needs)}, and monorl opens a filesystem from its scheme and the environment"
    )
