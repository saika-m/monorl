# Rewards

This document covers the `rewards:` section of a run config: how entries combine into an episode's
total reward, every built-in reward function with its parameters and run-time behaviour, and how
to add one. How the total becomes an advantage is in [ALGORITHMS.md](ALGORITHMS.md).

## Configuring `rewards`

`rewards` is a list of entries (`RewardSpec` in `src/monorl/config/schema.py`) with three fields:
`type` (required, a name in the registry namespace `reward`), `weight` (default `1.0`) and `params`
(default `{}`, the dict passed to the reward's factory).

```yaml
rewards:
  - {type: exact_match, params: {extract: last_number}}
  - {type: format, weight: 0.1, params: {name: phrasing, patterns: ["answer is"]}}
```

`monorl.rewards.build_reward` turns the list into one `CompositeReward` whose score is the sum of
`weight * score` over the entries; weights are not normalised. Each entry's unweighted score is
kept in a breakdown under `params.name`, else the `type` (a repeated key gets `#2`, `#3`, ...). An
unknown `type` passes config validation and raises `RegistryError` when the loop is built.

A reward function receives whole `Trajectory` objects. Every built-in accepts `params.name`. All
except `reward_model` and `composite` accept `scope`: `last` (default) scores the text of the
final policy turn only, `all` the texts of all policy turns joined in order with no separator;
environment messages such as tool results are in neither. `reward_model` always scores the prompt
messages plus the final turn. The reference is `sample.reference` (see [DATA.md](DATA.md)), or
with `reference_key: k` (`exact_match`, `math_verify`, `code_exec`, `llm_judge`) `reference[k]` of
a dict reference holding `k`, else `sample.metadata[k]` when present, else the whole reference.

`rewards: []` is rejected at validation with `env.type: single_turn` and valid otherwise: the
environment then supplies the whole reward. Per-turn rewards returned by an environment
(`Turn.reward`, see [ENVIRONMENTS.md](ENVIRONMENTS.md)) are added in `ComponentLoop._score`
(`src/monorl/orchestrator/loop.py`): the total is the weighted sum, plus the sum over turns `k`
(from 0) of `algorithm.gamma ** k * turn.reward`, minus `algorithm.overlong_penalty` when the
final turn's finish reason is `length`. The discounted environment sum is the breakdown key `env`,
metric `reward/env`, present whenever a turn reward is non-zero; the penalty is reported as
`reward/overlong_penalty`. Checked here with `_score`: turn rewards 1, 2, 0, `gamma: 0.5` and a
matching `exact_match` give total `3.0` with `env` `2.0`. For `algorithm.credit_assignment: turn`
see [CREDIT_ASSIGNMENT.md](CREDIT_ASSIGNMENT.md).

## Built-in rewards

The examples form one `.venv/bin/python` session that starts with this helper; comments show output.

```python
from monorl.core.registry import registry
from monorl.core.types import Completion, RewardResult, Sample, Trajectory, Turn

def traj(*responses, reference=None):
    msgs = [{"role": "user", "content": "Solve it."}]
    done = [Completion(text=r, token_ids=list(range(len(r.split())))) for r in responses]
    turns = [Turn(prompt_token_ids=[1], completion=c) for c in done]
    sample = Sample(id="s0", messages=msgs, reference=reference)
    return Trajectory(sample=sample, group_id="g0", messages=list(msgs), turns=turns)
```

### `exact_match`

Returns 1.0 when the answer extracted from the response equals the reference, else 0.0. The
reference goes through the same extractor (raw when nothing is extracted) and may be a list of
acceptable answers. Two numbers (after removing `,`, `$`, a trailing `%`) are compared numerically.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `extract` | `last_number` | `last_number`, `boxed` (last `\boxed{}`), `after_hash`, `none` |
| `normalize` | `[whitespace, case]` | For string comparison: collapse whitespace, lower-case |
| `numeric_tolerance` | `1e-6` | Absolute tolerance of the numeric comparison |

```yaml
rewards:
  - {type: exact_match, params: {extract: boxed, normalize: [whitespace]}}
```

```python
fn = registry.get("reward", "exact_match")({"extract": "last_number"})
r = fn([traj("3 + 4 = 7. The answer is 7.", reference="#### 7")])[0]
print(r.score, r.breakdown, r.info)  # 1.0 {'exact_match': 1.0} {'extracted': '7'}
print(fn([traj("I think 7.", "No, it is 9.", reference="7")])[0].score)  # 0.0 (two turns)
```

### `math_verify`

Returns 1.0 when the response's final answer is mathematically equivalent to the reference per
`math-verify`, else 0.0. Without the `rewards` extra the factory raises `BackendUnavailable`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `timeout_s` | `5` | Parse/verify timeout, applied on the main thread only (`signal.alarm`) |
| `float_rounding`, `numeric_precision`, `strict` | `6`, `15`, `True` | To `math_verify.verify` |

```yaml
rewards:
  - type: math_verify
```

```python
fn = registry.get("reward", "math_verify")({})
r = fn([traj("So the result is $\\boxed{\\frac{1}{2}}$.", reference="0.5")])[0]
print(r.score, r.info)  # 1.0 {'extracted': ['1/2', '\\frac{1}{2}']}
```

### `format`

Regex check of the response: 1.0 or 0.0, or a fraction in [0, 1] with `require_all: false`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `patterns` | required | List of regexes; one string, or the key `pattern`, also works |
| `require_all` | `True` | `True`: 1.0 only if every pattern matches. `False`: fraction matched |
| `flags`, `fullmatch` | `[DOTALL]`, `False` | `re` flag names; `re.fullmatch`, not `re.search` |

```yaml
rewards:
  - {type: format, weight: 0.2, params: {patterns: ["<think>.*</think>"]}}
```

```python
params = {"patterns": ["<think>.*</think>", "<answer>.*</answer>"], "require_all": False}
r = registry.get("reward", "format")(params)([traj("<think>2+2</think> 4")])[0]
print(r.score, r.info)  # 0.5 {'matched': [True, False]}
```

### `length`

Returns `base` when the response length is inside `[min_tokens, max_tokens]` and
`max(min_reward, base - penalty_per_token * excess)` outside it: the range is `[min_reward, base]`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `min_tokens`, `max_tokens` | `0`, unset | Band in `unit`; no upper bound when unset |
| `penalty_per_token`, `base`, `min_reward` | `0.01`, `1.0`, `0.0` | See the formula above |
| `unit` | `tokens` | `tokens` (completion token ids), `chars`, or `words` (whitespace split) |

```yaml
rewards:
  - {type: length, weight: 0.1, params: {unit: words, min_tokens: 2, max_tokens: 4}}
```

```python
params = {"unit": "words", "min_tokens": 2, "max_tokens": 4, "penalty_per_token": 0.25}
r = registry.get("reward", "length")(params)([traj("one two three four five six")])[0]
print(r.score, r.info)  # 0.5 {'length': 6}
```

### `code_exec`

Runs the last fenced code block tagged `language` (or untagged) of the response against the test
cases in the reference. Returns the fraction of cases passed, or with `mode: all` 1.0 only if all
pass; 0.0 without a code block; without test cases, 1.0 if the code exits with status 0. The
reference is test code appended to the solution (a string), a list of such strings or of
`{"input": ..., "output": ...}` dicts (stdin, expected stdout), or one of the other layouts
accepted by `normalize_tests` in `src/monorl/rewards/code_exec.py`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `timeout_s`, `memory_mb` | `5.0`, `512` | Per test case: wall-clock limit, memory cap (`0`: off) |
| `language`, `python` | `python`, current interpreter | Fence tag; interpreter used |
| `mode`, `max_workers` | `fraction`, `min(8, cpu_count)` | `fraction` or `all`; thread count |

```yaml
rewards:
  - {type: code_exec, params: {mode: all, timeout_s: 10, memory_mb: 256}}
```

```python
fence = "`" * 3
code = f"{fence}python\ndef add(a, b):\n    return a + b\n{fence}"
fn = registry.get("reward", "code_exec")({"timeout_s": 5})
r = fn([traj(code, reference=["assert add(1, 2) == 3", "assert add(2, 2) == 5"])])[0]
print(r.score, r.info["cases"])  # 0.5 ['passed', 'error']
```

### `llm_judge`

Asks an OpenAI-compatible chat-completions endpoint for a grade and parses the last
`<score>N</score>` of the reply, clamped to `[0, max_score]`. The score is `N / max_score`, or `N`
with `normalize: false`. A request that fails for a passing reason is sent again; a judgment that
fails for good scores 0.0 and is reported, see [Operational behaviour](#operational-behaviour).
Run here against `httpx.MockTransport` and a local stand-in endpoint, not against a judge model;
the entry below was validated by building the reward from it.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `base_url`, `model` | required | Endpoint root such as `http://host:8000/v1`; judge model name |
| `rubric` | built-in | `str.format` template, see below |
| `api_key_env`, `system_prompt` | `OPENAI_API_KEY`, unset | Bearer token variable; system message |
| `max_score`, `normalize` | `10`, `True` | Scale of `N`, above 0; whether to divide by it |
| `temperature`, `max_tokens`, `extra_body` | `0.0`, `512`, `{}` | Request fields; extra fields |
| `timeout_s`, `max_workers` | `60`, `8` | `httpx` timeout of a request (s); requests at a time |
| `max_attempts` | `4` | Requests per judgment at most; `1` turns retrying off |
| `retry_backoff_s` | `1.0` | Longest wait before the first retry, in seconds; doubles per retry |
| `retry_max_wait_s` | `60` | Seconds a judgment may wait between its attempts in total |
| `on_error` | `zero` | A judgment that failed for good: `zero` (score 0.0) or `raise` |

The rubric may use `{prompt}`, `{response}`, `{reference}` and `{max_score}` and has to ask for
the `<score>N</score>` line; a literal brace in it is written `{{` or `}}`. A rubric that does not
format, a `max_score` of 0 and a `max_attempts` below 1 raise `ValueError` when the reward is
built.

```yaml
rewards:
  - type: llm_judge
    params: {base_url: "http://127.0.0.1:8000/v1", model: judge, api_key_env: JUDGE_KEY}
```

The factory takes two keyword arguments for tests: `client`, an `httpx.Client` to send through,
and `wait(stop, seconds)`, called in place of each wait between attempts. Below, an endpoint that
is busy once and then grades, one that stays down, and one whose reply has no score:

```python
import httpx

def judge(*replies, **params):  # replies of the endpoint, the last one repeated
    left = list(replies)
    answer = lambda request: left.pop(0) if left[1:] else left[0]
    client = httpx.Client(transport=httpx.MockTransport(answer))
    params = {"base_url": "http://judge.test/v1", "model": "judge", **params}
    wait = lambda stop, seconds: print("wait", seconds)
    return registry.get("reward", "llm_judge")(params, client=client, wait=wait)

def reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

seven = traj("It is 7.", reference="7")
busy = httpx.Response(429, headers={"Retry-After": "2"})
r = judge(busy, reply("Right. <score>8</score>"))([seven])[0]
# wait 2.0
print(r.score, r.info)
# 0.8 {'raw_score': 8.0, 'judge_output': 'Right. <score>8</score>', 'attempts': 2}

r = judge(httpx.Response(503), max_attempts=3, retry_backoff_s=0)([seven])[0]
# wait 0.0
# wait 0.0
# on stderr: reward llm_judge: 1 of 1 judgments failed and scored 0 (1x HTTPStatusError:
#   503 Service Unavailable)
print(r.score, r.info)
# 0.0 {'error': 'HTTPStatusError: 503 Service Unavailable', 'attempts': 3}

fn = judge(reply("Right."), reply("Right. <score>8</score>"), max_workers=1)
r = fn([seven, seven])[0]
# on stderr: reward llm_judge: 1 of 2 judgments failed and scored 0 (1x no_score)
print(r.score, r.info)
# 0.0 {'error': 'no_score', 'judge_output': 'Right.'}
```

### `reward_model`

Scores the conversation with an `AutoModelForSequenceClassification` checkpoint: logit
`label_index`, unbounded, or its sigmoid in (0, 1) with `normalize: sigmoid`. Not run here (no
checkpoint was loaded); the entry below was validated only.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model`, `trust_remote_code` | required, `False` | Hugging Face id or path; to `from_pretrained` |
| `device`, `dtype` | `cpu`, `float32` | Torch device; `float32`, `bfloat16` or `float16` |
| `batch_size`, `max_length` | `8`, `4096` | Texts per forward pass; truncation length in tokens |
| `label_index`, `normalize` | `0`, `none` | Logit used as the score; `none` or `sigmoid` |

```yaml
rewards:
  - {type: reward_model, params: {model: /models/my-reward-model, normalize: sigmoid}}
```

### `composite`

A weighted sum of member rewards as one entry: `params.rewards` (required) is a list shaped like
top-level `rewards`. As a top-level entry its total is in the breakdown under its `name` and each
member's score under `<name>/<member key>` (metric `reward/<name>/<member key>`).

```yaml
rewards:
  - type: composite
    params: {rewards: [{type: length}, {type: format, params: {pattern: "7"}}]}
```

```python
members = [{"type": "length"}, {"type": "format", "params": {"pattern": "7"}}]
r = registry.get("reward", "composite")({"rewards": members})([traj("It is 7.")])[0]
print(r.score, r.breakdown)  # 2.0 {'length': 1.0, 'format': 1.0}
```

## Operational behaviour

`code_exec` runs every test case in a new `python -I` subprocess in its own session and process
group, with a fresh temporary directory (deleted afterwards) as working directory, `HOME` and
`TMPDIR`, and an environment of only those two, `PATH` and three `PYTHON*` settings. On timeout or
over the memory cap the process group gets `SIGKILL` (case status `timeout` or `memory`); a
non-zero exit is `error`. The cap is set in the child with `resource.setrlimit` (which only
Linux enforces) and enforced by the parent, which polls the resident set size of the process tree
every 20 ms. This is no security
boundary: same user, filesystem and network access, no container, no CPU or process-count limit.

`llm_judge` sends one `POST {base_url}/chat/completions` per trajectory, up to `max_workers` at a
time. The only environment variable read is the one named by `api_key_env`, at construction; if
unset, no `Authorization` header is sent.

A request is sent again after a connection error, a timeout or one of the statuses 408, 429, 500,
502, 503 and 504, until the judgment has made `max_attempts` requests. The wait before retry `k`
(from 1) is drawn between half of and all of `retry_backoff_s * 2 ** (k - 1)` seconds, so that the
workers of a call do not return together, or is what the response's `Retry-After` header asks for
(seconds, or a date) when that is longer. A wait that would take the waits of the judgment past
`retry_max_wait_s` is not begun: the judgment ends with the error it has. A judgment whose
requests each end within `timeout_s` therefore takes at most
`max_attempts * timeout_s + retry_max_wait_s` seconds. Every other failure is final at once:
another status (400, 401, 404, ...), a body that is not a chat completion, a reply without a
score tag or without text.

A judgment that failed for good scores 0.0 with `on_error: zero`; with `raise` the error
propagates and ends the run. The 0.0 is the same number as a grade of 0 in `reward/mean` and
`reward/<name>`, and it is trained on. What tells them apart:

- One warning per call of the reward, which the loop makes once with all trajectories of a step,
  through Python's `logging` under `monorl.rewards.llm_judge`:
  `reward <name>: N of M judgments failed and scored 0 (<count>x <error>, ...)`, followed by the
  body of the first error response. It goes to stderr and carries no time: that is the terminal
  of `monorl train`, and `run.log` for a job started by `monorl serve`, whose stderr is that file
  (from `src/monorl/server/jobs.py`; run here with the same redirection, not through the server).
- In `trajectories.jsonl`, `info.reward_info.<name>` of a row holds `error` (`no_score`, or the
  exception's name and message, for a status `HTTPStatusError: 503 Service Unavailable`), `body`
  (the first 300 characters of an error response), `judge_output` for a reply without a score,
  and `attempts` when the judgment made more than one request, successful or not.
- `time/reward` grows by the waits.

No metric counts failed judgments. Run here, one step of four episodes on a tiny random model
against a local endpoint that always answers 503 with `Retry-After: 3`: the run succeeded with
`reward/mean` 0.0 and `time/reward` 9.07, stderr had
`reward llm_judge: 4 of 4 judgments failed and scored 0 (4x HTTPStatusError: 503 Service
Unavailable)` with the response body after it, `run.log` did not (it did with stderr appended to
it), and the four rows had `attempts: 4`. Against an endpoint that answered every fourth request
with 429, all four rewards were the judge's 0.6 and nothing was logged.

A judgment that waits keeps its worker thread and nothing else; the other workers go on. When a
call ends early (an error with `on_error: raise`, a `KeyboardInterrupt`), the waiting judgments
give up at once. The first Ctrl-C or SIGTERM of a run ends it after the current step, and that
step finishes its judgments, retries included; a second one stops at once. Run here against the
endpoint above: a first SIGINT two seconds into the judgments of step 1 was followed by the rest
of their nine seconds of waits and `cancelled after step 1`; a second one a second later gave
`cancelled after step 0` within that second.

`reward_model` loads tokenizer and model once, when the loop is built, in the loop's process. Each
conversation is rendered with the tokenizer's chat template when it has one, else as
`role: content` lines, then scored in batches of `batch_size`, truncated to `max_length`.

The loop calls the composite once per step with all of that step's trajectories and stores each
result's `info` in `Trajectory.info["reward_info"]`. It does not catch exceptions from rewards: run
here with a reward that raises, the run ended `failed` at step 0 with the error in `status.json`.

## Metrics

`reward/mean`, `reward/std`, `reward/<name>` per breakdown key (including `reward/env` and
`reward/overlong_penalty`), `reward/success_rate`, `reward/pass_at_k`, `time/reward`,
`eval/reward_mean`, `eval/reward_std`, `eval/reward/<name>`: see [RUNS.md](RUNS.md#metrics).

## Pipeline backends

From the adapters' code, not run here. With `trainer.backend: trl` the composite is wrapped as one
TRL reward function (`src/monorl/adapters/trl/rewards.py`) that computes the total as above,
including `env` and `overlong_penalty`. With `verl` (`src/monorl/adapters/verl/reward.py`) at
least one entry is required: a generated `<run_dir>/monorl_reward.py` scores one response at a
time, and the breakdown goes to verl's `reward_extra_info`. With `nemo_rl` (`env_bridge.py` of
that adapter) the score is added to the final turn's reward. Under `verl` and `nemo_rl` a reward
sees the decoded text of a response and no token ids, so `length` with `unit: tokens` (its
default) is rejected at validation: use `chars` or `words`. See [BACKENDS.md](BACKENDS.md).

## Writing a reward function

The interface is the `RewardFunction` protocol in `src/monorl/core/interfaces.py`: an object with
a `name` attribute and `__call__(trajectories) -> list[RewardResult]`, one result per trajectory,
in order. What gets registered (`src/monorl/core/registry.py`) is a factory
`factory(params: dict, **kw)`; `build_reward` passes it `config=<RunConfig>` as a keyword. A
module a run can load exposes `register(registry)`. `my_rewards.py`:

```python
from monorl.core.types import RewardResult


class KeywordReward:
    def __init__(self, params):
        self.name = str(params.get("name") or "keyword")
        self.keyword = str(params["keyword"]).lower()

    def __call__(self, trajs):
        scores = [float(self.keyword in t.last_response_text.lower()) for t in trajs]
        return [RewardResult(score=s, breakdown={self.name: s}) for s in scores]


def make_keyword(params, **_):
    return KeywordReward(params)


def register(registry):
    registry.register("reward", "keyword", make_keyword)
```

Name the module in the config's `providers` (in a pipeline config: under `base`) and use the
reward like a built-in one:

```yaml
providers: [my_rewards]
rewards:
  - {type: keyword, params: {keyword: the}}
```

The modules in `providers` are imported before a run builds its components, in every process
that builds them: the run itself, Ray actors and, from the code (not run here), the worker
processes in which `verl` and `nemo_rl` score responses, where the module has to be on the
Python path. Otherwise they are imported from the working directory or the Python path: start
the run where the module is. Jobs launched from the web UI also find modules in the directory
`monorl serve` was started in. Run here from the directory that holds `my_rewards.py`: one CPU
step of `grpo_gsm8k_smollm_cpu.yaml` with `--set 'providers=[my_rewards]'` and
`--set 'rewards=[{type: keyword, params: {keyword: the}}]'` reported `reward/keyword`, and
`--set 'providers=[no_such_module]'` failed the run with
`provider no_such_module failed to import: No module named 'no_such_module'`.

Continuing the session above, with `my_rewards.py` in the working directory:

```python
import my_rewards

my_rewards.register(registry)
print(registry.get("reward", "keyword")({"keyword": "seven"})([traj("Seven.")])[0])
# RewardResult(score=1.0, breakdown={'keyword': 1.0}, info={})
```

A reward that belongs in monorl itself is added to `REWARD_NAMES` in
`src/monorl/rewards/__init__.py`; its tests go in `tests/rewards/`, whose `conftest.py` has a
`traj` fixture that builds a trajectory from a response and a reference.
