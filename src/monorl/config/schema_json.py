"""The JSON schema of the run config as a checked-in file, for a consumer that cannot ask Python.

``web/mock/config_schema.json`` holds what ``monorl serve`` answers on ``GET /api/config/schema``
(``RunConfig.json_schema()``); the Bun mock of the API (``web/mock/server.ts``) serves the file.
Torch-free::

    python -m monorl.config.schema_json                                      # print
    python -m monorl.config.schema_json --write web/mock/config_schema.json  # regenerate
    python -m monorl.config.schema_json --check web/mock/config_schema.json  # exit 1 when stale

``--check`` compares the fields of every model and their descriptions, which is what the web UI
reads from the schema, and not the bytes: pydantic is not pinned, and how it spells a type or a
default may differ between the version that wrote the file and the one that checks it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from monorl.config.schema import RunConfig

REGENERATE_COMMAND = "python -m monorl.config.schema_json --write web/mock/config_schema.json"


def render_schema() -> str:
    return json.dumps(RunConfig.json_schema(), indent=2, ensure_ascii=False) + "\n"


def field_descriptions(schema: dict[str, Any]) -> dict[str, str | None]:
    """``{"RolloutConfig.turn_batching": "...", ...}``: every field of the root model and of the
    models under ``$defs``, with its description (``None`` without one)."""
    models = {str(schema.get("title")): schema, **schema.get("$defs", {})}
    return {
        f"{model}.{name}": prop.get("description")
        for model, body in models.items()
        for name, prop in body.get("properties", {}).items()
    }


def stale_fields(path: Path) -> list[str]:
    """Fields the file at ``path`` lacks, has in excess or describes differently from the schema;
    all of them when it is missing or not JSON."""
    current = field_descriptions(RunConfig.json_schema())
    try:
        written = field_descriptions(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        written = {}
    return sorted({key for key, _ in current.items() ^ written.items()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m monorl.config.schema_json", description="Write the JSON schema of the run config."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--write", metavar="PATH", help="write the schema to PATH")
    group.add_argument("--check", metavar="PATH", help="exit 1 when PATH lacks a field or a description")
    args = parser.parse_args(argv)
    if args.write:
        Path(args.write).write_text(render_schema(), encoding="utf-8")
    elif args.check:
        stale = stale_fields(Path(args.check))
        if stale:
            print(
                f"{args.check} is out of date with the config schema ({len(stale)} fields, first: "
                f"{stale[0]}); run: {REGENERATE_COMMAND}",
                file=sys.stderr,
            )
            return 1
    else:
        sys.stdout.write(render_schema())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
