# Data

Datasets are described by `data:` in the `RunConfig` and resolved by `monorl.data.open_source`
into a stream of `Sample(id, messages, reference, metadata)`. The web UI's dataset picker and
`monorl data inspect` use the same code (`monorl.data.browse.preview_uri`), so what you preview
is what training sees. Every field, with its default, is in [CONFIG.md](CONFIG.md#data).
Example files live in `$DEMO` (`/tmp/monorl-docs-data` when written).

## URI schemes

`open_source` parses `data.uri` and finds the source of its rows in the registry namespace
`data_source`, by scheme (`source_name` in `src/monorl/data/rows.py`).

| `data.uri` | Source | Notes |
|---|---|---|
| `/path/to/file.jsonl`, `./dir/`, `data/*.parquet`, `file://...` | `local`: the local filesystem | A file, a directory (every data file below it, at any depth) or a glob, in the [file formats](#file-formats) below; `~` is expanded. Anything without `://` is a path. |
| a task directory, a directory of them, a glob of either | `local`: one row per task | Harbor task directories, read as the columns `id`, `instruction` and `task` for `env.type: harness`: [SANDBOX.md](SANDBOX.md#3-task-directories). |
| `hf://org/name` | `hf`: Hugging Face Hub dataset | `data.split` (default `train`), `data.hf_config_name` for subsets (GSM8K: `main`). `hf://org/name@revision` pins a revision; `hf://org/name/path/in/repo.jsonl` narrows to specific files. `hf://datasets/org/name` is tolerated. Gated/private datasets use your `HF_TOKEN` / `huggingface-cli login`. |
| `org/name` | `local` | Without `hf://` a repository id is a relative path: the run fails with `dataset path not found: org/name` unless that path exists. |
| `s3://bucket/key.jsonl`, `s3://bucket/prefix/` | `s3`: S3 or S3-compatible object store via `s3fs` | Credentials from the environment only (below). A prefix reads every supported file under it, in the same file formats. |
| `http://host/train.jsonl`, `https://...` | `s3`: fsspec's HTTP filesystem (`aiohttp`) | One file, in the same file formats, told by the extension the URL ends with; no directories, no globs. A server without range requests works: the file is read whole. |
| `gs://`, `az://`, ... | `s3`: any other fsspec filesystem | Works when the driver (`gcsfs`, `adlfs`, ...) is installed, credentials per that driver; without it: `no fsspec driver for scheme`. |
| `myscheme://anything` | the source a provider module registered as `myscheme` | See [Writing a data source](#writing-a-data-source). A scheme that is neither registered nor an fsspec protocol is an error that lists the registered names. |

Every row has an id that does not depend on the order rows are read in: `<file>#<index in file>`
for files, `<uri>#<split>#<index>` for Hub datasets, the directory for a task. `data.extra`
holds loader options: with `hf://` they are keyword arguments of `datasets.load_dataset`, with
task directories `skip_unsupported: true` leaves out tasks that cannot be run, a provider's
source gets a copy, and file sources read none.

Run here: every `local` form, `hf://openai/gsm8k`, `s3://` against a local moto server,
`http://` against a server on the loopback interface (JSONL, CSV, Parquet), `memory://` as
another fsspec protocol, a provider's scheme. Not run here: AWS, MinIO, R2, `https://`, `gs://`
and `az://` (no driver installed), gated Hub datasets.

## S3 credentials and endpoints

Nothing secret ever goes in a config file or the UI; `data.extra` rejects `key`/`secret`/`token`/
`storage_options`. Set the standard AWS variables in the environment that runs `monorl serve` or
`monorl train`:

| Variable | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | static credentials (omit both to use the botocore default chain: IAM role, `~/.aws/credentials`, SSO) |
| `AWS_SESSION_TOKEN` | temporary credentials (STS / SSO) |
| `AWS_DEFAULT_REGION` or `AWS_REGION` | bucket region |
| `AWS_ENDPOINT_URL` | custom endpoint for **MinIO** (`http://localhost:9000`), **Cloudflare R2** (`https://<account>.r2.cloudflarestorage.com`), Ceph RGW, LocalStack... |
| `AWS_PROFILE` | pick a profile from `~/.aws/config` (botocore reads it) |

`GET /api/connectors` (and the UI's S3 panel) reports `configured`, `region`, `endpoint_url` and
which variable names are present, never values. Quick check from the shell (run here as printed,
with a moto server on port 9000 holding that object):

```
AWS_ENDPOINT_URL=http://localhost:9000 AWS_ACCESS_KEY_ID=minio \
  AWS_SECRET_ACCESS_KEY=minio123 \
  monorl data inspect s3://my-bucket/data/train.jsonl --limit 3
```

See [`configs/recipes/s3_dataset_example.yaml`](../configs/recipes/s3_dataset_example.yaml).

## File formats

`.jsonl`/`.ndjson`, `.json` (array of objects, or an object wrapping one), `.csv`/`.tsv`,
`.parquet`, `.arrow` (`datasets.save_to_disk` directories work). The text formats may be
compressed (`.gz`, `.bz2`, `.xz`, and `.zst` with the `zstandard` package, which was not
installed here). Directories and globs skip files whose name starts with `.` or `_`, and when
they mix formats read only the highest-priority one (parquet > arrow > jsonl > csv > json).
Readers stream, so previews and `data.streaming: true` never load whole files; the exception is
`.json`, which is parsed whole.

## Layouts and field mapping

`data.format` selects how rows become chats; `auto` (default) sniffs the column names, whatever
their case, and the first 50 rows (`sniff_format`):

| format | columns | result |
|---|---|---|
| `chat` | a messages column (`messages`, `conversations`, `conversation`; ShareGPT `from`/`value` is normalised) | `Sample.messages` = the list; the last assistant turn, if any, becomes the reference when no answer column exists |
| `prompt_answer` | prompt column (`prompt`, `question`, `instruction`, `input`, `problem`, `query`) + answer column (`answer`, `solution`, `response`, `output`, `completion`, `target`, `label`, `final_answer`) | `messages = [user: prompt]`, `reference = answer` |
| `text` | a lone prompt/text column (`text`, else the first string column) | `messages = [user: text]`, `reference = None` |

Override the guess with `messages_field`, `prompt_field`, `answer_field`. With `format: auto`,
`messages_field` alone means `chat`, and `prompt_field` alone means `prompt_answer` when an
answer column is named or found, else `text`. A named column the dataset does not have is an
error that lists the columns.

### Agent conversations: tool results, context turns, tool schemas

Chat rows may hold whole agent conversations; `algorithm.sft_turns: all` trains every assistant
turn of them ([`PIPELINES.md`](PIPELINES.md), which also describes the rows rejection sampling
writes in this layout).

- Messages keep `tool_calls` (assistant), `name` and `tool_call_id` (`tool` results) and `weight`.
  An assistant message with `"weight": 0` is context that SFT does not train (the OpenAI
  fine-tuning convention).
- Tool schemas come from the column named by `data.tools_field`; without it, from a chat
  dataset's `tools` column when that holds a list of schema objects. The value is a list of
  OpenAI-style function schemas or its JSON string (CSV, many Hub datasets). It becomes
  `metadata["tools"]`, which SFT passes to the chat template and `max_prompt_tokens` counts. A
  column named in `tools_field` that holds something else is an error; a chat dataset's own
  `tools` column that does not hold schemas is left in `metadata["row"]`.
- jsonl, parquet, arrow and Hub datasets read the same. Parquet, arrow and Hub datasets give all
  rows one struct type and fill the keys a row lacks with nulls: null message keys are ignored and
  null keys are removed from tool schemas at every depth. Keep schemas as a JSON string column
  if a `null` inside one matters.

Processing per row:

1. `prompt_template`: Python format string over the row's columns plus `{prompt}`:
   `"{prompt}\nThink step by step and end with '#### <answer>'."` It is applied to the user turn
   that ends the prompt; a `{name}` that is not a column is an error.
2. `system_prompt`: the first message, as a `system` message. It replaces a `system` message the
   row starts with.
3. Reference extraction: GSM8K-style `... #### 42` answers are reduced to `42`, without
   thousands separators (the raw text stays under `metadata["answer_raw"]`). Rewards may extract
   further (`exact_match` `extract: last_number|boxed|after_hash|none`, `math_verify` parses
   LaTeX). A chat row's trailing assistant turn is taken out of the prompt.
4. `image_field`: the values of these columns become `{"type": "image", "image": <path or URL>}`
   parts of the user turn, in front of its text or where `prompt_template` has `<image>`. A
   value is an HF `Image`, bytes, base64, a `data:` URL, a path (relative to the dataset file) or
   a URI, or a list of those; all but local files and `http(s)` URLs are stored once under
   `$MONORL_HOME/cache/media`. Engines: [BACKENDS.md](BACKENDS.md#multimodal-image-prompts).
5. `max_prompt_tokens`: samples whose chat-templated prompt is longer are dropped (needs the
   tokenizer; the orchestrator passes the policy tokenizer). Text tokens only: image
   placeholders are not counted. The count is kept as `metadata["prompt_tokens"]`.
6. `shuffle` (deterministic, `seed`) then `limit`: the first `limit` samples that passed the
   filter.

With `streaming: true` rows are read lazily, a Hub dataset without downloading it, and the
shuffle works within a buffer of 1000 rows (and over the order of the files). The loops still
collect every sample before the first step, so set `limit` with it: reading stops there.

The result, with `$DEMO/qa.jsonl` holding these rows (a `.venv/bin/python` session; comments
show output):

```json
{"question": "What is 2 + 2?", "answer": "Add them. #### 4", "level": "easy"}
{"question": "What is 30 * 40?", "answer": "3 * 4 = 12. #### 1,200", "level": "easy"}
{"question": "What is 7 squared?", "answer": "7 * 7 #### 49", "level": "hard"}
```

```python
from monorl.config.schema import DataConfig
from monorl.data import open_source

template = "{prompt} ({level})"
cfg = DataConfig(uri="/tmp/monorl-docs-data/qa.jsonl", prompt_template=template, limit=2)
source = open_source(cfg, seed=1)
print(source.name, len(source), source.layout.format)  # local 2 prompt_answer
sample = source[0]
print(sample.id, repr(sample.reference))  # /tmp/monorl-docs-data/qa.jsonl#1 '1200'
print(sample.messages)  # [{'role': 'user', 'content': 'What is 30 * 40? (easy)'}]
print(sample.metadata["row"])  # {'answer': '3 * 4 = 12. #### 1,200', 'level': 'easy'}
```

| `Sample` field | Value |
|---|---|
| `id` | the row's `id` column when it has one, else the row id |
| `messages` | the prompt: `role`/`content` messages; `content` is a list of parts when the turn has images |
| `reference` | the answer after extraction, or the text of a chat row's trailing assistant turn, or `None` |
| `metadata["row"]` | the row's other columns: all but the prompt or messages column, the image columns and the tools column |
| `metadata["response"]` | the SFT target (below), or `None` |
| `metadata["answer_raw"]`, `["assistant_turn"]`, `["tools"]`, `["prompt_tokens"]` | present when steps 3 and 5 or the tool schemas above produced them |

## Evaluation data and the SFT target

`data.eval_uri` names the evaluation set. It is opened with the same `data` section, where `uri`
is replaced by it, `split` by `data.eval_split` and `shuffle` is off: the mapping, `limit` and the
token filter above apply unchanged, and its scheme may differ from that of `data.uri`. When it is
evaluated and what is reported is in [RUNS.md](RUNS.md#evaluation). In a pipeline, an `eval` or
`rejection_sample` stage can have a `data` section of its own ([PIPELINES.md](PIPELINES.md#config)).

SFT trains on `metadata["response"]`: the `data.response_field` column when it is set and the row
has a value, else the raw answer column (`prompt_answer` and `text` rows), else the text of the
trailing assistant turn. A sample with none of them, and no string reference, fails the run. What
the loop does with the target is in [ALGORITHMS.md](ALGORITHMS.md#sft).

## Inspecting

```
monorl data inspect hf://openai/gsm8k --config-name main --limit 3
monorl data inspect ./data/train.parquet --json
```

prints dtype per column, the inferred layout, the first rows and the `Sample`s monorl would
build. The UI's dataset page shows the same preview plus a column mapper that writes
`prompt_field`/`answer_field`/`messages_field` back into the config. Both ran here, the second
in `$DEMO` on a parquet copy of `qa.jsonl`. The command has no tokenizer and reads no config: it
applies no `max_prompt_tokens` filter. `--provider <module>` (repeatable) imports a provider
module first, so that `monorl data inspect sums://4 --provider my_data` shows the rows of the
source written [below](#writing-a-data-source); the UI's preview knows no provider's scheme.

## Writing a data source

A data source is a `RowSource` (`src/monorl/data/rows.py`): it yields rows, and everything in
[Layouts and field mapping](#layouts-and-field-mapping) is applied to them as to the rows of a
file. A subclass provides:

| Member | Contract |
|---|---|
| `uri` | attribute: the URI the source was opened from |
| `columns()` | a dict of column name to dtype string (`string`, `int64`, ...); the layout and the `data.*_field` options go by these names |
| `iter_rows(*, shuffle_seed=None)` | an iterator of `(row_id, row)`, from the start on every call. `row` is a dict of column values, `row_id` a string that identifies it in any order. `None` is source order; a seed asks for a shuffled order that is the same for the same seed |
| `head(n)`, `known_len()`, `media_base(row_id)`, `streaming`, `null_filled` | optional. Defaults: the first `n` of `iter_rows()`; `None` (unknown: a streaming source then has no `len()`); `None` (no base for relative image paths); `False`; `False` (`True` drops null keys from tool schemas) |

What gets registered, in the namespace `data_source` under the scheme, is a factory
`factory(uri, *, split, config_name, streaming, extra)` that returns the `RowSource`. `uri` is
the parsed URI: `uri.raw` as written, `uri.path` what follows `://` without the slashes around
it. The keywords are `data.split` (`data.eval_split` for evaluation data), `data.hf_config_name`,
`data.streaming` and a copy of `data.extra`; take `**_` for the ones you do not use. A scheme
consists of letters, digits and `_`. `$DEMO/my_data.py`:

```python
import random

from monorl.data.rows import RowSource


class Sums(RowSource):
    def __init__(self, uri):
        self.uri = uri.raw
        self.rows = [
            (f"{uri.raw}#{i}", {"question": f"{i} + {i} = ?", "answer": f"#### {i + i}"})
            for i in range(int(uri.path))
        ]

    def columns(self):
        return {"question": "string", "answer": "string"}

    def iter_rows(self, *, shuffle_seed=None):
        rows = list(self.rows)
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(rows)
        return iter(rows)


def register(registry):
    registry.register("data_source", "sums", lambda uri, **_: Sums(uri))
```

Name the module in the config's `providers` (in a pipeline config: under `base`) and use the
scheme like a built-in one:

```yaml
providers: [my_data]
data:
  uri: sums://20
  eval_uri: sums://6
```

How and where `providers` are imported is in
[REWARDS.md](REWARDS.md#writing-a-reward-function): start the run where the module is. Run here
from `$DEMO`: one CPU step of `grpo_gsm8k_smollm_cpu.yaml` with these keys as `--set` overrides
(`--set 'providers=[my_data]'`, `--set 'data.uri=sums://20'`, `--set 'data.eval_uri=sums://6'`),
`trainer.max_steps=1` and `trainer.eval_every=1` logged `data: 20 samples from sums://20` and
`eval data: 6 samples from sums://6`, trained on the prompt `19 + 19 = ?` and succeeded. A
pipeline with that `base` and an `eval` stage with
`data: {uri: "sums://4", prompt_field: question, answer_field: answer}` evaluated those four
rows. Without the `providers` override the run fails before step 1 and `monorl train` ends with:

```text
monorl.core.errors.DataError: no data source for the scheme 'sums' of
'sums://20'. Registered: ['hf', 'local', 's3'] ('s3' also reads the other fsspec
protocols, 'local' paths). A module in the config's `providers` adds one with
registry.register("data_source", "sums", factory).
```

A source that belongs in monorl itself is registered in `register` of
`src/monorl/data/__init__.py`; its tests go in `tests/data/` (see `test_source_registry.py`).
