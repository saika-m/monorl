"""Row sources are found through the ``data_source`` registry, by URI scheme."""

from __future__ import annotations

import json
import random
import sys
import textwrap
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.registry import Registry
from monorl.data import open_source
from monorl.data.browse import preview_uri
from monorl.data.rows import source_name
from monorl.data.uri import parse_uri
from tests.data.conftest import QA_ROWS

PROVIDER = """
    import random

    from monorl.data.rows import RowSource


    class Countdown(RowSource):
        def __init__(self, uri, split, streaming, extra):
            self.uri = uri.raw
            mode = "stream" if streaming else "memory"
            self.items = [
                (f"{uri.raw}#{i}", {"q": f"{split} {i}", "a": f"#### {i * extra['scale']}", "mode": mode})
                for i in range(int(uri.path))
            ]

        def columns(self):
            return {"q": "string", "a": "string", "mode": "string"}

        def iter_rows(self, *, shuffle_seed=None):
            items = list(self.items)
            if shuffle_seed is not None:
                random.Random(shuffle_seed).shuffle(items)
            return iter(items)


    def open_countdown(uri, *, split, streaming, extra, **_):
        return Countdown(uri, split, streaming, extra)


    def register(registry):
        registry.register("data_source", "countdown", open_countdown)
        registry.register("data_source", "rowlist", lambda uri, **_: [{"q": "not a RowSource"}])
"""


@pytest.fixture
def provider_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Registry]:
    """A registry of its own that loaded a provider module from the working directory."""
    name = f"data_provider_{uuid.uuid4().hex[:10]}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(PROVIDER))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", ".", str(tmp_path))])
    registry = Registry()
    registry.load_providers([name])
    yield registry
    sys.modules.pop(name, None)


def test_a_provider_scheme_goes_through_mapping_shuffle_and_limit(provider_registry: Registry) -> None:
    cfg = DataConfig(
        uri="countdown://20",
        prompt_field="q",
        answer_field="a",
        prompt_template="Row {prompt}, read from {mode}.",
        system_prompt="Be brief.",
        limit=5,
        extra={"scale": 3},
    )
    source = open_source(cfg, seed=7, registry=provider_registry)
    order = list(range(20))
    random.Random(7).shuffle(order)
    assert source.name == "countdown" and source.uri == "countdown://20"
    assert [s.id for s in source] == [f"countdown://20#{i}" for i in order[:5]]
    assert [s.id for s in open_source(cfg, seed=8, registry=provider_registry)] != [s.id for s in source]
    first = source[0]
    assert first.messages == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": f"Row train {order[0]}, read from memory."},
    ]
    assert first.reference == str(3 * order[0])
    assert first.metadata["row"] == {"a": f"#### {3 * order[0]}", "mode": "memory"}

    # the split the caller asks for, data.streaming and shuffle: false reach the source as well
    lazy = cfg.model_copy(update={"streaming": True, "shuffle": False, "limit": 3})
    stream = open_source(lazy, split="test", registry=provider_registry)
    assert [s.messages[1]["content"] for s in stream] == [
        f"Row test {i}, read from stream." for i in range(3)
    ]


def test_an_unknown_scheme_names_it_and_lists_the_registered_ones(provider_registry: Registry) -> None:
    with pytest.raises(DataError) as builtin:
        open_source(DataConfig(uri="nosuch://bucket/x.jsonl"), registry=Registry())
    assert "'nosuch'" in str(builtin.value) and "['hf', 'local', 's3']" in str(builtin.value)
    with pytest.raises(DataError, match=r"'nosuch'.*\['countdown', 'hf', 'local', 'rowlist', 's3'\]"):
        open_source(DataConfig(uri="nosuch://bucket/x.jsonl"), registry=provider_registry)
    with pytest.raises(DataError, match="no data source for the scheme 'nosuch'"):
        preview_uri("nosuch://bucket/x.jsonl")
    with pytest.raises(DataError, match="data_source 'rowlist' returned list"):
        open_source(DataConfig(uri="rowlist://x"), registry=provider_registry)


URI_FORMS = [
    # uri, kind, scheme, path (None: a local path, checked separately), data_source
    ("hf://openai/gsm8k", "hf", "hf", "openai/gsm8k", "hf"),
    ("hf://datasets/openai/gsm8k@main/main/train.parquet", "hf", "hf", "openai/gsm8k", "hf"),
    ("s3://bucket/data/a.jsonl", "remote", "s3", "bucket/data/a.jsonl", "s3"),
    ("gs://bucket/data/", "remote", "gs", "bucket/data", "s3"),
    ("memory://bucket/a.jsonl", "remote", "memory", "bucket/a.jsonl", "s3"),
    ("data/qa.jsonl", "local", "file", None, "local"),
    ("file://data/qa.jsonl", "local", "file", None, "local"),
    ("data/*.parquet", "local", "file", None, "local"),
    # without hf:// a repo id is a relative path
    ("openai/gsm8k", "local", "file", None, "local"),
    ("countdown://5", "remote", "countdown", "5", "countdown"),
]


@pytest.mark.parametrize(("uri", "kind", "scheme", "path", "name"), URI_FORMS)
def test_uri_forms_and_the_source_that_opens_them(
    provider_registry: Registry, uri: str, kind: str, scheme: str, path: str | None, name: str
) -> None:
    parsed = parse_uri(uri)
    if path is None:
        path = str(Path.cwd() / uri.removeprefix("file://"))
    assert (parsed.kind, parsed.scheme, parsed.path) == (kind, scheme, path)
    assert source_name(parsed, provider_registry) == name


def test_hub_revision_and_path_in_repo() -> None:
    parsed = parse_uri("hf://datasets/openai/gsm8k@main/main/train.parquet")
    assert (parsed.repo_id, parsed.revision, parsed.subpath) == ("openai/gsm8k", "main", "main/train.parquet")


def test_builtin_sources_open_through_a_fresh_registry(qa_jsonl: Path) -> None:
    import fsspec

    local = open_source(DataConfig(uri=f"file://{qa_jsonl}", shuffle=False), registry=Registry())
    assert local.name == "local" and len(local) == 12
    assert local[0].id == f"{qa_jsonl}#0" and local[0].reference == "0"

    # an fsspec protocol nothing is registered for is read by the fsspec file source
    bucket = f"monorl-{uuid.uuid4().hex[:10]}"
    fs = fsspec.filesystem("memory")
    fs.pipe(f"/{bucket}/qa.jsonl", "".join(json.dumps(r) + "\n" for r in QA_ROWS).encode())
    try:
        remote = open_source(
            DataConfig(uri=f"memory://{bucket}/qa.jsonl", shuffle=False), registry=Registry()
        )
        assert remote.name == "s3" and len(remote) == 12
        assert remote[3].messages == [{"role": "user", "content": "What is 3+3?"}]
    finally:
        fs.rm(f"/{bucket}", recursive=True)
