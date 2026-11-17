"""Raw row providers: the part of a data source that knows where bytes live.

A :class:`RowSource` yields ``(row_id, row)`` pairs. The built-in ones:

* :class:`FileRows` — local / fsspec files (one or many), either materialised into memory or
  streamed file-by-file. Row ids are ``<file-uri>#<index-in-file>`` and therefore stable across
  shuffles and seeds.
* :class:`HubRows` — ``datasets.load_dataset`` over a Hub repo, ``Dataset`` (materialised,
  indexable) or ``IterableDataset`` (streaming).
* :class:`~monorl.data.tasks.TaskRows` — a local directory of task directories (the Harbor task
  format), one row per task.

:func:`open_rows` finds the row source of a URI in the ``data_source`` namespace of the registry.
The name is the URI's scheme (:func:`source_name`), and what is registered is a factory::

    factory(uri: ParsedURI, *, split: str, config_name: str | None, streaming: bool,
            extra: dict[str, Any]) -> RowSource

``uri.raw`` is the URI as written and ``uri.path`` what follows ``scheme://``; ``split``,
``config_name``, ``streaming`` and ``extra`` are ``data.split`` (or the split the caller asked
for), ``data.hf_config_name``, ``data.streaming`` and a copy of ``data.extra``. A factory takes
``**_`` for the ones it has no use for. A provider module (``RunConfig.providers``) adds a source
for a scheme of its own with ``registry.register("data_source", "<scheme>", factory)``; the
built-in sources are registered the same way by :func:`monorl.data.register`.
"""

from __future__ import annotations

import abc
import itertools
import os
import posixpath
import random
from collections.abc import Iterator, Sequence
from typing import Any

from monorl.core.errors import DataError
from monorl.core.registry import Registry
from monorl.core.registry import registry as default_registry
from monorl.data import formats
from monorl.data.uri import HTTP_SCHEMES, ParsedURI, filesystem_for

SHUFFLE_BUFFER = 1000

RowRef = tuple[str, dict[str, Any]]


def buffered_shuffle(
    items: Iterator[RowRef], seed: int, buffer_size: int = SHUFFLE_BUFFER
) -> Iterator[RowRef]:
    """Reservoir-style approximate shuffle for streams (same idea as ``IterableDataset.shuffle``)."""
    rng = random.Random(seed)
    buffer: list[RowRef] = []
    for item in items:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        j = rng.randrange(buffer_size)
        yield buffer[j]
        buffer[j] = item
    rng.shuffle(buffer)
    yield from buffer


class RowSource(abc.ABC):
    """The rows behind a dataset URI: what a ``data_source`` factory returns, and all that
    :func:`monorl.data.open_source` and :mod:`monorl.data.browse` need from a source. Layout
    detection, field mapping, ``prompt_template``, ``system_prompt``, images, the
    ``max_prompt_tokens`` filter and ``limit`` are applied to the rows by
    :mod:`monorl.data.sources`, whichever source they come from.

    A subclass sets :attr:`uri` and implements :meth:`columns` and :meth:`iter_rows`; the other
    members have defaults."""

    uri: str
    """The URI the source was opened from (``DataSource.uri``, error messages)."""
    streaming: bool = False
    """Whether :meth:`iter_rows` reads lazily instead of holding every row in memory."""
    null_filled: bool = False
    """Whether rows come from a columnar store (parquet, arrow, Hub datasets), where a struct
    column holds ``null`` for every key a row does not have: such keys are then dropped from the
    row's tool schemas."""

    @abc.abstractmethod
    def columns(self) -> dict[str, str]:
        """Column name -> dtype string (``"string"``, ``"int64"``, ...): the columns the layout is
        detected from, and that ``data.*_field`` have to name."""

    def head(self, n: int) -> list[dict[str, Any]]:
        """The first ``n`` rows in source order, for layout detection and previews."""
        return [row for _, row in itertools.islice(self.iter_rows(), n)]

    @abc.abstractmethod
    def iter_rows(self, *, shuffle_seed: int | None = None) -> Iterator[RowRef]:
        """All rows as ``(row_id, row)``, from the start on every call. A row is a mapping of
        column name to value; ``row_id`` identifies it whatever the order (it is the sample's id
        unless the row has an ``id`` column). ``shuffle_seed=None`` is source order; a seed
        requests an order that is shuffled and the same for the same seed (``data.shuffle``)."""

    def known_len(self) -> int | None:
        """Row count when it is cheap to know (materialised, or file metadata), else ``None``."""
        return None

    def media_base(self, row_id: str) -> str | None:
        """Location relative image paths in the row ``row_id`` are resolved against (a local
        directory or a ``scheme://...`` prefix), or ``None`` when the source has no such place."""
        return None


# -- files ------------------------------------------------------------------------------------


def expand_files(fs, parsed: ParsedURI) -> list[str]:
    """Resolve a file / directory / glob into concrete data-file paths (sorted). A path that
    exists is that file or directory, whatever characters it holds (``runs[v1]/train.jsonl``);
    only one that does not is read as a glob."""
    path = parsed.path
    if parsed.scheme in HTTP_SCHEMES:  # one file: a web server has no listing to expand
        if formats.format_of(path) is None:
            raise DataError(
                f"unsupported file type: {parsed.raw} (an http(s) dataset is one file, expected "
                f"{sorted(formats.FORMAT_EXTENSIONS)})"
            )
        if not fs.exists(path):
            raise FileNotFoundError(
                f"dataset path not found: {parsed.raw} (the server did not answer or does not serve it)"
            )
        return [path]
    exists = fs.exists(path)
    if not exists and any(ch in path for ch in "*?["):
        fs.invalidate_cache()
        matches = [p for p in fs.glob(path) if fs.isfile(p)]
        files = formats.select_files(matches)
        if not files:
            raise DataError(f"glob {parsed.raw!r} matched no supported data files")
        return files
    if not exists:
        raise FileNotFoundError(f"dataset path not found: {parsed.raw}")
    if fs.isdir(path):
        fs.invalidate_cache(path)
        files = formats.select_files(fs.find(path))
        if not files:
            raise DataError(
                f"directory {parsed.raw!r} contains no supported data files ({sorted(formats.FORMAT_EXTENSIONS)})"
            )
        return files
    if formats.format_of(path) is None:
        raise DataError(f"unsupported file type: {parsed.raw} (expected {sorted(formats.FORMAT_EXTENSIONS)})")
    return [path]


class FileRows(RowSource):
    def __init__(self, parsed: ParsedURI, *, streaming: bool = False) -> None:
        self.parsed = parsed
        self.uri = parsed.raw
        self.streaming = streaming
        self.fs = filesystem_for(parsed)
        self.files = expand_files(self.fs, parsed)
        self.format = formats.format_of(self.files[0])
        self.null_filled = self.format not in ("jsonl", "json")
        self._rows: list[RowRef] | None = None
        self._columns: dict[str, str] | None = None

    # -- helpers
    def file_uri(self, path: str) -> str:
        return self.parsed.with_path(path)

    def _iter_file(self, path: str) -> Iterator[RowRef]:
        uri = self.file_uri(path)
        for i, row in enumerate(formats.iter_file_rows(self.fs, path, self.format)):
            yield f"{uri}#{i}", row

    def _iter_files(self, order: Sequence[str]) -> Iterator[RowRef]:
        for path in order:
            yield from self._iter_file(path)

    def _materialise(self) -> list[RowRef]:
        if self._rows is None:
            self._rows = list(self._iter_files(self.files))
        return self._rows

    # -- RowSource
    def columns(self) -> dict[str, str]:
        if self._columns is None:
            cols = formats.arrow_schema_of(self.fs, self.files[0], self.format)
            if cols is None:
                cols = formats.infer_columns(self.head(formats.SNIFF_ROWS))
            self._columns = cols
        return self._columns

    def head(self, n: int) -> list[dict[str, Any]]:
        if self._rows is not None:
            return [row for _, row in self._rows[:n]]
        return [row for _, row in itertools.islice(self._iter_files(self.files), n)]

    def iter_rows(self, *, shuffle_seed: int | None = None) -> Iterator[RowRef]:
        if not self.streaming:
            rows = self._materialise()
            if shuffle_seed is None:
                return iter(rows)
            order = list(range(len(rows)))
            random.Random(shuffle_seed).shuffle(order)
            return (rows[i] for i in order)
        if shuffle_seed is None:
            return self._iter_files(self.files)
        files = list(self.files)
        random.Random(shuffle_seed).shuffle(files)
        return buffered_shuffle(self._iter_files(files), shuffle_seed)

    def known_len(self) -> int | None:
        if self._rows is not None:
            return len(self._rows)
        if self.format == "parquet":
            return sum(formats.parquet_num_rows(self.fs, p) for p in self.files)
        if not self.streaming:
            return len(self._materialise())
        return None

    def media_base(self, row_id: str) -> str | None:
        file_uri = row_id.rsplit("#", 1)[0]
        if self.parsed.kind == "local":
            return os.path.dirname(file_uri)
        return posixpath.dirname(file_uri)


# -- hugging face hub -------------------------------------------------------------------------


class HubRows(RowSource):
    null_filled = True

    def __init__(
        self,
        parsed: ParsedURI,
        *,
        split: str,
        config_name: str | None = None,
        streaming: bool = False,
        load_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from datasets import load_dataset

        self.parsed = parsed
        self.uri = parsed.raw
        self.split = split
        self.streaming = streaming
        kwargs: dict[str, Any] = dict(load_kwargs or {})
        if parsed.revision:
            kwargs.setdefault("revision", parsed.revision)
        if parsed.subpath:
            kwargs.setdefault("data_files", parsed.subpath)
        try:
            self.dataset = load_dataset(
                parsed.repo_id, name=config_name, split=split, streaming=streaming, **kwargs
            )
        except (ValueError, OSError) as e:  # no such repo, split or config; the Hub cannot be reached
            raise DataError(
                f"cannot load {parsed.raw!r} (split={split!r}, config={config_name!r}): {e}"
            ) from e
        self._columns: dict[str, str] | None = None

    def _row_id(self, index: int) -> str:
        return f"{self.uri}#{self.split}#{index}"

    def columns(self) -> dict[str, str]:
        if self._columns is None:
            features = getattr(self.dataset, "features", None)
            if features:
                self._columns = formats.features_to_columns(features)
            else:
                self._columns = formats.infer_columns(self.head(formats.SNIFF_ROWS))
        return self._columns

    def head(self, n: int) -> list[dict[str, Any]]:
        if self.streaming:
            return list(self.dataset.take(n))
        return [self.dataset[i] for i in range(min(n, len(self.dataset)))]

    def iter_rows(self, *, shuffle_seed: int | None = None) -> Iterator[RowRef]:
        if self.streaming:
            ds = (
                self.dataset
                if shuffle_seed is None
                else self.dataset.shuffle(seed=shuffle_seed, buffer_size=SHUFFLE_BUFFER)
            )
            return ((self._row_id(i), row) for i, row in enumerate(ds))
        n = len(self.dataset)
        order = list(range(n))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(order)
        return ((self._row_id(i), self.dataset[i]) for i in order)

    def known_len(self) -> int | None:
        if not self.streaming:
            return len(self.dataset)
        info = getattr(self.dataset, "info", None)
        splits = getattr(info, "splits", None) or {}
        split_info = splits.get(self.split) if hasattr(splits, "get") else None
        n = getattr(split_info, "num_examples", None)
        return int(n) if n else None

    def media_base(self, row_id: str) -> str | None:
        rev = f"@{self.parsed.revision}" if self.parsed.revision else ""
        return f"hf://datasets/{self.parsed.repo_id}{rev}"


# -- registry ---------------------------------------------------------------------------------

LOCAL_SOURCE = "local"
FSSPEC_SOURCE = "s3"


def open_local_rows(uri: ParsedURI, *, streaming: bool, extra: dict[str, Any], **_: Any) -> RowSource:
    """``data_source`` ``local``: a file, directory or glob, or task directories."""
    from monorl.data.tasks import TaskRows, find_task_dirs  # imports this module

    task_dirs = find_task_dirs(uri.path)
    if task_dirs:  # a directory of task directories (Harbor format), one row per task
        return TaskRows(uri, task_dirs, skip_unsupported=bool(extra.get("skip_unsupported")))
    return FileRows(uri, streaming=streaming)


def open_hub_rows(
    uri: ParsedURI, *, split: str, config_name: str | None, streaming: bool, extra: dict[str, Any]
) -> RowSource:
    """``data_source`` ``hf``: a Hub dataset repo; ``extra`` goes to ``load_dataset``."""
    return HubRows(uri, split=split, config_name=config_name, streaming=streaming, load_kwargs=extra)


def open_fsspec_rows(uri: ParsedURI, *, streaming: bool, **_: Any) -> RowSource:
    """``data_source`` ``s3``: files on S3 or any other fsspec filesystem."""
    return FileRows(uri, streaming=streaming)


def source_name(parsed: ParsedURI, registry: Registry | None = None) -> str:
    """Name in the ``data_source`` namespace of the factory that opens ``parsed``.

    * a path or ``file://`` -> ``local``
    * ``scheme://`` with a factory registered as ``scheme`` (``hf``, ``s3``, a provider's) -> it
    * another scheme fsspec has a protocol for (``gs``, ``az``, ...) -> ``s3``, which reads files
      from any fsspec filesystem
    * anything else is a :class:`DataError` that lists the registered names
    """
    registry = registry or default_registry
    if parsed.kind == "local":
        return LOCAL_SOURCE
    if registry.contains("data_source", parsed.scheme):
        return parsed.scheme
    import fsspec

    if parsed.scheme in fsspec.available_protocols():
        return FSSPEC_SOURCE
    raise DataError(
        f"no data source for the scheme {parsed.scheme!r} of {parsed.raw!r}. Registered: "
        f"{registry.names('data_source')} ({FSSPEC_SOURCE!r} also reads the other fsspec protocols, "
        f"{LOCAL_SOURCE!r} paths). A module in the config's `providers` adds one with "
        f'registry.register("data_source", "{parsed.scheme}", factory).'
    )


def open_rows(
    parsed: ParsedURI,
    *,
    split: str,
    config_name: str | None,
    streaming: bool,
    extra: dict[str, Any] | None = None,
    registry: Registry | None = None,
) -> RowSource:
    """The :class:`RowSource` of ``parsed``, from the factory :func:`source_name` names."""
    registry = registry or default_registry
    name = source_name(parsed, registry)
    rows = registry.get("data_source", name)(
        parsed, split=split, config_name=config_name, streaming=streaming, extra=dict(extra or {})
    )
    if not isinstance(rows, RowSource):
        raise DataError(
            f"data_source {name!r} returned {type(rows).__name__} for {parsed.raw!r}; a factory returns "
            "a monorl.data.rows.RowSource"
        )
    return rows
