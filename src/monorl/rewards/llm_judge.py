"""``llm_judge``: score responses with an OpenAI-compatible chat-completions endpoint.

Params:
    base_url: e.g. ``https://api.openai.com/v1`` or a local vLLM/SGLang server (required).
    model: judge model name (required).
    api_key_env: environment variable holding the bearer token (default ``OPENAI_API_KEY``;
        missing variable → no ``Authorization`` header, which local servers accept).
    rubric: ``str.format`` template with ``{prompt}``, ``{response}``, ``{reference}`` and
        ``{max_score}`` placeholders (a literal brace is ``{{`` or ``}}``); the judge must
        answer with ``<score>N</score>``.
    system_prompt: optional system message.
    max_score: scale of ``N`` (default 10; above 0).
    normalize: divide the score by ``max_score`` (default ``True`` → reward in ``[0, 1]``).
    temperature (0.0), max_tokens (512), timeout_s (60), max_workers (8),
    extra_body: dict merged into the request body.
    max_attempts: requests per judgment (default 4; 1 turns retrying off). A request is repeated
        after a connection error, a timeout or one of the statuses in :data:`RETRY_STATUS`;
        any other failure is final at once.
    retry_backoff_s: longest wait before the first retry (default 1.0). The bound doubles with
        every retry and the wait is drawn from its upper half, so that the workers of a call do
        not come back together; a ``Retry-After`` that asks for more is waited out instead.
    retry_max_wait_s: seconds one judgment may spend waiting between its attempts (default 60).
        A wait that would go past it ends the judgment with the error it has.
    on_error: what a judgment that failed for good becomes: ``zero`` (default; score 0 with
        ``info["error"]``) or ``raise``.

A 0 for a judgment that failed is not a grade. The breakdown of every judgment therefore has
``failed`` (1.0 or 0.0) next to the score, which a run reports as ``reward/<name>/failed``, the
share of failed judgments of a step; and a call that scored failures as 0 says so in one warning
of this module's logger (``reward <name>: N of M judgments failed and scored 0 (...)``).

Factory kwargs: ``client=httpx.Client(...)`` injects a transport (tests use ``MockTransport``);
``wait=fn(stop, seconds)`` replaces the wait between attempts (``threading.Event.wait``) and
returns whether to give up.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC
from email.utils import parsedate_to_datetime
from functools import partial
from typing import Any

import httpx

from monorl.core.types import RewardResult, Trajectory
from monorl.rewards.base import check_trajectories, prompt_text, reference_value, response_text, reward_name

DEFAULT_RUBRIC = """You are a strict grader. Evaluate the response to the task below.

[Task]
{prompt}

[Reference answer]
{reference}

[Response]
{response}

Judge correctness and quality against the reference. Reply with a brief justification followed by
a final line of the form <score>N</score> where N is an integer from 0 to {max_score}."""

_SCORE_RE = re.compile(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", re.IGNORECASE)

log = logging.getLogger(__name__)

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
"""Statuses that say "not now": the same request may succeed a moment later."""
_RETRY_ERRORS = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)
_MAX_BODY_CHARS = 300
"""How much of an error response is kept in ``info["body"]``."""


def parse_score(text: str) -> float | None:
    matches = _SCORE_RE.findall(text)
    return float(matches[-1]) if matches else None


def retry_after_seconds(response: httpx.Response) -> float:
    """What ``Retry-After`` asks for, as a number of seconds or an HTTP date; 0.0 without one."""
    value = response.headers.get("Retry-After", "")
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except ValueError:
            return 0.0
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = when.timestamp() - time.time()
    return seconds if 0.0 < seconds < math.inf else 0.0


def _describe(error: Exception) -> dict[str, Any]:
    """``info`` of a failed request: ``error`` names the kind, ``body`` holds what the server said."""
    if not isinstance(error, httpx.HTTPStatusError):
        return {"error": f"{type(error).__name__}: {error}"}
    response = error.response
    info = {"error": f"HTTPStatusError: {response.status_code} {response.reason_phrase}"}
    if response.text:
        info["body"] = response.text[:_MAX_BODY_CHARS]
    return info


class LLMJudgeReward:
    def __init__(
        self,
        params: dict[str, Any],
        *,
        client: httpx.Client | None = None,
        wait: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        self.name = reward_name(params, "llm_judge")
        base_url = params.get("base_url")
        model = params.get("model")
        if not base_url or not model:
            raise ValueError("llm_judge needs params.base_url and params.model")
        self.model = str(model)
        self.rubric = str(params.get("rubric", DEFAULT_RUBRIC))
        try:
            self.rubric.format(prompt="", response="", reference="", max_score=0)
        except (KeyError, IndexError, ValueError, AttributeError) as e:
            raise ValueError(
                "llm_judge: rubric is a str.format template that may use {prompt}, {response}, "
                f"{{reference}} and {{max_score}}; write a literal brace as {{{{ or }}}} ({type(e).__name__}: {e})"
            ) from None
        self.system_prompt = params.get("system_prompt")
        self.max_score = float(params.get("max_score", 10))
        if not 0 < self.max_score < math.inf:
            raise ValueError("llm_judge: max_score must be a number above 0")
        self.normalize = bool(params.get("normalize", True))
        self.temperature = float(params.get("temperature", 0.0))
        self.max_tokens = int(params.get("max_tokens", 512))
        self.max_workers = int(params.get("max_workers", 8))
        self.extra_body = dict(params.get("extra_body", {}))
        self.on_error = str(params.get("on_error", "zero"))
        if self.on_error not in ("zero", "raise"):
            raise ValueError("llm_judge: on_error must be 'zero' or 'raise'")
        self.max_attempts = int(params.get("max_attempts", 4))
        if self.max_attempts < 1:
            raise ValueError("llm_judge: max_attempts must be at least 1 (1 turns retrying off)")
        self.retry_backoff_s = float(params.get("retry_backoff_s", 1.0))
        self.retry_max_wait_s = float(params.get("retry_max_wait_s", 60.0))
        if not (self.retry_backoff_s >= 0 and self.retry_max_wait_s >= 0):
            raise ValueError("llm_judge: retry_backoff_s and retry_max_wait_s must not be negative")
        self._wait = wait or threading.Event.wait
        # A generator of its own: the process-wide one is seeded by the run and saved with its
        # checkpoints, and a retry must not move it.
        self._jitter = random.Random()
        self.scope = str(params.get("scope", "last"))
        self.reference_key = params.get("reference_key")

        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(str(params.get("api_key_env", "OPENAI_API_KEY")))
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if client is None:
            client = httpx.Client(
                base_url=str(base_url).rstrip("/"), timeout=float(params.get("timeout_s", 60))
            )
        else:
            client.base_url = httpx.URL(str(base_url).rstrip("/"))
        client.headers.update(headers)
        self.client = client

    def build_messages(self, trajectory: Trajectory) -> list[dict[str, str]]:
        reference = reference_value(trajectory, self.reference_key)
        user = self.rubric.format(
            prompt=prompt_text(trajectory),
            response=response_text(trajectory, self.scope),
            reference="" if reference is None else str(reference),
            max_score=int(self.max_score) if self.max_score.is_integer() else self.max_score,
        )
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": str(self.system_prompt)})
        messages.append({"role": "user", "content": user})
        return messages

    def _retry_delay(self, error: Exception, attempt: int) -> float | None:
        """Seconds to wait before the request that failed with ``error`` is sent again, ``None``
        when it is not: the failure is not transient, or ``attempt`` was the last one allowed."""
        asked = 0.0
        if isinstance(error, httpx.HTTPStatusError):
            if error.response.status_code not in RETRY_STATUS:
                return None
            asked = retry_after_seconds(error.response)
        elif not isinstance(error, _RETRY_ERRORS):
            return None
        if attempt >= self.max_attempts:
            return None
        return max(math.ldexp(self.retry_backoff_s, attempt - 1) * self._jitter.uniform(0.5, 1.0), asked)

    def _result(self, score: float, attempts: int, info: dict[str, Any]) -> RewardResult:
        if attempts > 1:
            info["attempts"] = attempts
        breakdown = {self.name: score}
        # On every judgment, so that its mean is the share that failed; a judge named "failed"
        # keeps its score there.
        breakdown.setdefault("failed", float("error" in info))
        return RewardResult(score=score, breakdown=breakdown, info=info)

    def judge(self, trajectory: Trajectory, stop: threading.Event | None = None) -> RewardResult:
        """Score one trajectory. Setting ``stop`` ends a wait between attempts: the judgment then
        fails with the error of its last attempt instead of trying again."""
        stop = stop or threading.Event()
        body = {
            "model": self.model,
            "messages": self.build_messages(trajectory),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        attempts = 0
        waited = 0.0
        while True:
            attempts += 1
            try:
                # The reply is read in full and its connection is back in the pool before any wait.
                resp = self.client.post("/chat/completions", json=body)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                break
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as e:
                delay = self._retry_delay(e, attempts)
                if delay is not None and waited + delay <= self.retry_max_wait_s:
                    if not self._wait(stop, delay):
                        waited += delay
                        continue
                if self.on_error == "raise":
                    raise
                return self._result(0.0, attempts, _describe(e))
        # ``content`` is null when the judge spent its tokens on reasoning, refused, or called a tool.
        raw = parse_score(content) if isinstance(content, str) else None
        if raw is None:
            if self.on_error == "raise":
                raise ValueError(f"llm_judge: no <score> tag in judge output: {str(content)[:200]!r}")
            return self._result(0.0, attempts, {"error": "no_score", "judge_output": content})
        raw = min(max(raw, 0.0), self.max_score)
        score = raw / self.max_score if self.normalize else raw
        return self._result(score, attempts, {"raw_score": raw, "judge_output": content})

    def __call__(self, trajectories: Sequence[Trajectory]) -> list[RewardResult]:
        check_trajectories(trajectories)
        if len(trajectories) <= 1 or self.max_workers <= 1:
            results = [self.judge(t) for t in trajectories]
        else:
            stop = threading.Event()
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                try:
                    results = list(pool.map(partial(self.judge, stop=stop), trajectories))
                finally:
                    # Left early (an error with on_error: raise, an interrupt): the judgments that
                    # wait for a retry give up at once, so the pool is back after the requests
                    # that are on the wire and not after their backoff.
                    stop.set()
        failed = [r.info for r in results if "error" in r.info]
        if failed:
            kinds = Counter(str(info["error"]) for info in failed)
            said = next((f"; the endpoint said: {info['body']!r}" for info in failed if "body" in info), "")
            log.warning(
                "reward %s: %d of %d judgments failed and scored 0 (%s)%s",
                self.name,
                len(failed),
                len(results),
                ", ".join(f"{n}x {kind}" for kind, n in kinds.most_common()),
                said,
            )
        return results


def make_llm_judge(
    params: dict[str, Any],
    *,
    client: httpx.Client | None = None,
    wait: Callable[[threading.Event, float], bool] | None = None,
    **_: Any,
) -> LLMJudgeReward:
    return LLMJudgeReward(params, client=client, wait=wait)
