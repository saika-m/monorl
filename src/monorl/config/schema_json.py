"""The JSON schemas of the run config and of the pipeline config as checked-in files, for a
consumer that cannot ask Python.

``web/mock/config_schema.json`` holds what ``monorl serve`` answers on ``GET /api/config/schema``
(``RunConfig.json_schema()``) and ``web/mock/pipeline_schema.json`` what it answers on
``GET /api/config/pipeline_schema`` (``PipelineConfig.json_schema()``, with ``--pipeline``); the
Bun mock of the API (``web/mock/server.ts``) serves the files. Torch-free::

    python -m monorl.config.schema_json                                      # print
    python -m monorl.config.schema_json --write web/mock/config_schema.json  # regenerate
    python -m monorl.config.schema_json --check web/mock/config_schema.json  # exit 1 when stale
    python -m monorl.config.schema_json --pipeline --write web/mock/pipeline_schema.json

``--check`` compares the fields of every model and their descriptions, which is what the web UI
reads from the schema, and not the bytes: how pydantic spells a type or a default may differ
between the version that wrote the file and the one that checks it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from monorl.config.pipeline import PipelineConfig
from monorl.config.schema import RunConfig

REGENERATE_COMMAND = "python -m monorl.config.schema_json --write web/mock/config_schema.json"
REGENERATE_PIPELINE_COMMAND = (
    "python -m monorl.config.schema_json --pipeline --write web/mock/pipeline_schema.json"
)


def schema(*, pipeline: bool = False) -> dict[str, Any]:
    """What the server answers: the run config's schema, or the pipeline config's."""
    return PipelineConfig.json_schema() if pipeline else RunConfig.json_schema()


def render_schema(*, pipeline: bool = False) -> str:
    return json.dumps(schema(pipeline=pipeline), indent=2, ensure_ascii=False) + "\n"


def field_descriptions(schema: dict[str, Any]) -> dict[str, str | None]:
    """``{"RolloutConfig.turn_batching": "...", ...}``: every field of the root model and of the
    models under ``$defs``, with its description (``None`` without one)."""
    models = {str(schema.get("title")): schema, **schema.get("$defs", {})}
    return {
        f"{model}.{name}": prop.get("description")
        for model, body in models.items()
        for name, prop in body.get("properties", {}).items()
    }


def stale_fields(path: Path, *, pipeline: bool = False) -> list[str]:
    """Fields the file at ``path`` lacks, has in excess or describes differently from the schema;
    all of them when it is missing or not JSON."""
    current = field_descriptions(schema(pipeline=pipeline))
    try:
        written = field_descriptions(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        written = {}
    return sorted({key for key, _ in current.items() ^ written.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monorl.config.schema_json",
        description="Write the JSON schema of the run config, or of the pipeline config.",
    )
    parser.add_argument("--pipeline", action="store_true", help="the pipeline config's schema")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", metavar="PATH", help="write the schema to PATH")
    group.add_argument("--check", metavar="PATH", help="exit 1 when PATH lacks a field or a description")
    args = parser.parse_args(argv)
    if args.write:
        Path(args.write).write_text(render_schema(pipeline=args.pipeline), encoding="utf-8")
    elif args.check:
        stale = stale_fields(Path(args.check), pipeline=args.pipeline)
        if stale:
            command = REGENERATE_PIPELINE_COMMAND if args.pipeline else REGENERATE_COMMAND
            print(
                f"{args.check} is out of date with the config schema ({len(stale)} fields, first: "
                f"{stale[0]}); run: {command}",
                file=sys.stderr,
            )
            return 1
    else:
        sys.stdout.write(render_schema(pipeline=args.pipeline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
