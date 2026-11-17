"""Data sources: local files, ``hf://`` Hub datasets, ``s3://`` / fsspec objects and the schemes
of provider modules → :class:`Sample`.

Public surface::

    open_source(data_config, *, split=None, tokenizer=None, seed=0) -> DataSource
    sniff_format(columns, sample_rows) -> {format, prompt_field, answer_field, messages_field, image_fields}
    browse.browse_uri(uri) / browse.preview_uri(uri, split=, hf_config_name=, limit=)
    register(registry)   # data_source: local | hf | s3, row source factories (monorl.data.rows)
"""

from __future__ import annotations

from typing import Any

from monorl.config.schema import DataConfig
from monorl.core.errors import DataError
from monorl.core.interfaces import DataSource
from monorl.core.registry import Registry
from monorl.data.formats import SNIFF_ROWS, sniff_format
from monorl.data.rows import open_fsspec_rows, open_hub_rows, open_local_rows, open_rows, source_name
from monorl.data.sources import Layout, MaterializedSource, SampleMapper, StreamingSource
from monorl.data.uri import parse_uri

__all__ = ["open_source", "sniff_format", "register"]

_FORBIDDEN_EXTRA = ("token", "storage_options", "key", "secret", "aws_access_key_id", "aws_secret_access_key")


def open_source(
    data_config: DataConfig,
    *,
    split: str | None = None,
    tokenizer: Any = None,
    seed: int = 0,
    registry: Registry | None = None,
) -> DataSource:
    """Resolve ``data_config.uri`` into a :class:`DataSource` of :class:`Sample` objects.

    * scheme → the row source registered for it as a ``data_source`` (``registry``, default the
      global one): local / hf / s3 (fsspec), or a provider module's; ``streaming`` keeps
      hf/s3/local lazy. Everything below applies to the rows of any of them
    * layout from ``format``/field overrides, else :func:`sniff_format` on the first rows
    * ``prompt_template`` (``str.format_map`` over the row + ``{prompt}``), ``system_prompt``
    * ``image_field`` column(s) → ``{"type": "image", "image": <path|uri>}`` parts on the user
      turn (``'<image>'`` in ``prompt_template`` positions them; default before the text); chat
      rows with OpenAI-style parts are normalised the same way (:mod:`monorl.data.media`)
    * ``tools_field`` (default: a chat dataset's ``tools`` column holding schemas) →
      ``metadata["tools"]``, the tool schemas the row's prompts are rendered with
    * ``#### answer`` extraction into ``Sample.reference`` (raw under ``metadata["answer_raw"]``);
      the SFT target under ``metadata["response"]`` (``response_field`` → ``answer_field`` →
      trailing assistant turn, else ``None``)
    * ``max_prompt_tokens`` filter when ``tokenizer`` is given (chat template applied to the text
      parts only; media placeholder tokens are not counted)
    * deterministic ``shuffle(seed)`` then ``limit``
    """
    parsed = parse_uri(data_config.uri)
    for key in _FORBIDDEN_EXTRA:
        if key in data_config.extra:
            raise DataError(f"data.extra[{key!r}] is not allowed: credentials come from the environment only")
    rows = open_rows(
        parsed,
        split=split or data_config.split,
        config_name=data_config.hf_config_name,
        streaming=data_config.streaming,
        extra=data_config.extra,
        registry=registry,
    )
    layout = Layout.resolve(data_config, rows.columns(), rows.head(SNIFF_ROWS))
    mapper = SampleMapper.from_config(data_config, layout, tokenizer, rows=rows)
    cls = StreamingSource if data_config.streaming else MaterializedSource
    return cls(
        source_name(parsed, registry),
        rows,
        mapper,
        seed=seed,
        shuffle=data_config.shuffle,
        limit=data_config.limit,
    )


def register(registry: Registry) -> None:
    registry.register("data_source", "local", open_local_rows)
    registry.register("data_source", "hf", open_hub_rows)
    registry.register("data_source", "s3", open_fsspec_rows)
