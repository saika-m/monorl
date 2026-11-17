from __future__ import annotations

import json
import logging
import random
import threading

import httpx
import pytest

from monorl.core.registry import registry
from monorl.orchestrator.metrics import reward_metrics
from monorl.rewards import CompositeReward
from monorl.rewards.llm_judge import parse_score, retry_after_seconds


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://placeholder")


def _reply(content) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def _judge(handler, waits: list[float] | None = None, **params):
    """An ``llm_judge`` on ``handler`` that records its waits between attempts in ``waits``
    instead of sleeping; without ``waits`` it must not wait at all."""

    def wait(stop: threading.Event, seconds: float) -> bool:
        assert waits is not None, f"waited {seconds}s"
        waits.append(seconds)
        return stop.is_set()

    return registry.get("reward", "llm_judge")(
        {"base_url": "http://j", "model": "m", **params}, client=_mock_client(handler), wait=wait
    )


def _failing_then(first: httpx.Response | Exception, calls: list[int]):
    """A handler whose first answer is ``first`` and every later one a 7 out of 10."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        if len(calls) > 1:
            return _reply("<score>7</score>")
        if isinstance(first, Exception):
            raise first
        return first

    return handler


def test_parse_score():
    assert parse_score("Reasoning...\n<score>7</score>") == 7.0
    assert parse_score("<SCORE> 3.5 </SCORE>") == 3.5
    assert parse_score("<score>1</score> then <score>9</score>") == 9.0
    assert parse_score("no tag") is None


def test_llm_judge_scores_via_mock_endpoint(traj, monkeypatch):
    monkeypatch.setenv("JUDGE_KEY", "sk-test")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        content = body["messages"][-1]["content"]
        score = 10 if "correct answer" in content else 2
        return httpx.Response(
            200, json={"choices": [{"message": {"content": f"Looks fine.\n<score>{score}</score>"}}]}
        )

    fn = registry.get("reward", "llm_judge")(
        {
            "base_url": "http://judge.local/v1/",
            "model": "judge-1",
            "api_key_env": "JUDGE_KEY",
            "rubric": "Q: {prompt}\nRef: {reference}\nA: {response}\nScore 0-{max_score}.",
            "system_prompt": "Be strict.",
            "max_workers": 2,
        },
        client=_mock_client(handler),
    )
    results = fn([traj("the correct answer", "42", prompt="What?"), traj("garbage", "42", prompt="What?")])
    assert [r.score for r in results] == [1.0, 0.2]
    assert results[0].breakdown == {"llm_judge": 1.0, "failed": 0.0}
    assert results[0].info["raw_score"] == 10.0
    assert len(seen) == 2
    req = seen[0]
    assert str(req.url) == "http://judge.local/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == "judge-1" and body["temperature"] == 0.0
    assert body["messages"][0] == {"role": "system", "content": "Be strict."}
    assert "Q: What?\nRef: 42\nA: " in body["messages"][1]["content"]
    assert "Score 0-10." in body["messages"][1]["content"]


def test_llm_judge_raw_scores_and_clamping(traj):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "<score>12</score>"}}]})

    fn = registry.get("reward", "llm_judge")(
        {"base_url": "http://j", "model": "m", "normalize": False, "max_score": 5},
        client=_mock_client(handler),
    )
    assert fn([traj("x")])[0].score == 5.0


def test_llm_judge_error_handling(traj):
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        if len(calls) == 1:
            return httpx.Response(404, text="no such model")
        return _reply("no tag here")

    results = _judge(handler, max_workers=1)([traj("a"), traj("b")])
    assert [r.score for r in results] == [0.0, 0.0]
    assert results[0].info == {"error": "HTTPStatusError: 404 Not Found", "body": "no such model"}
    assert results[1].info == {"error": "no_score", "judge_output": "no tag here"}

    waits: list[float] = []
    strict = _judge(lambda r: httpx.Response(503), waits, on_error="raise")
    with pytest.raises(httpx.HTTPStatusError):
        strict([traj("a")])
    assert len(waits) == 3  # the default is four attempts


def test_a_reply_without_text_is_a_judgment_without_a_score(traj):
    result = _judge(lambda request: _reply(None))([traj("a")])[0]
    assert result.score == 0.0 and result.info == {"error": "no_score", "judge_output": None}
    with pytest.raises(ValueError, match="no <score> tag"):
        _judge(lambda request: _reply(None), on_error="raise")([traj("a")])


def test_a_rate_limit_is_waited_out_and_the_answer_scored(traj, caplog):
    calls: list[int] = []
    waits: list[float] = []
    first = httpx.Response(429, headers={"Retry-After": "2"}, text="slow down")
    with caplog.at_level(logging.WARNING):
        result = _judge(_failing_then(first, calls), waits)([traj("a")])[0]
    assert result.score == 0.7 and result.breakdown == {"llm_judge": 0.7, "failed": 0.0}
    assert result.info == {"raw_score": 7.0, "judge_output": "<score>7</score>", "attempts": 2}
    assert len(calls) == 2
    assert waits == [2.0]  # what the server asked for: more than the first backoff of at most 1s
    assert not caplog.records  # nothing failed


@pytest.mark.parametrize(
    "first",
    [
        *(httpx.Response(status) for status in (408, 429, 500, 502, 503, 504)),
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("timed out"),
        httpx.RemoteProtocolError("server disconnected without sending a response"),
    ],
    ids=lambda first: str(first.status_code) if isinstance(first, httpx.Response) else type(first).__name__,
)
def test_transient_failures_are_retried(traj, first):
    calls: list[int] = []
    waits: list[float] = []
    result = _judge(_failing_then(first, calls), waits, retry_backoff_s=4)([traj("a")])[0]
    assert result.score == 0.7 and result.info["attempts"] == 2
    assert len(waits) == 1 and 2.0 <= waits[0] <= 4.0


@pytest.mark.parametrize(
    ("first", "error"),
    [
        (httpx.Response(400, text="context too long"), "400 Bad Request"),
        (httpx.Response(401), "401 Unauthorized"),
        (httpx.Response(404), "404 Not Found"),
        (httpx.Response(422), "422 Unprocessable Entity"),
        (httpx.Response(501), "501 Not Implemented"),
        (httpx.Response(200, json={"error": "overloaded"}), "KeyError: 'choices'"),
        (_reply("It is fine."), "no_score"),
    ],
    ids=["400", "401", "404", "422", "501", "no choices", "no score tag"],
)
def test_other_failures_are_final_at_once(traj, first, error):
    calls: list[int] = []
    result = _judge(_failing_then(first, calls))([traj("a")])[0]
    assert result.score == 0.0 and error in result.info["error"]
    assert "attempts" not in result.info and len(calls) == 1

    calls.clear()
    with pytest.raises((httpx.HTTPStatusError, KeyError, ValueError)):
        _judge(_failing_then(first, calls), on_error="raise")([traj("a")])
    assert len(calls) == 1


def test_a_persistent_outage_ends_after_max_attempts_with_doubling_waits(traj, caplog):
    calls: list[int] = []
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        return httpx.Response(503, text="upstream down")

    fn = _judge(handler, waits, max_attempts=4, retry_backoff_s=2, max_workers=1)
    with caplog.at_level(logging.WARNING):
        results = fn([traj("a"), traj("b")])
    assert [r.score for r in results] == [0.0, 0.0]
    assert results[0].info == {
        "error": "HTTPStatusError: 503 Service Unavailable",
        "body": "upstream down",
        "attempts": 4,
    }
    assert len(calls) == 8 and len(waits) == 6
    for first, second, third in (waits[:3], waits[3:]):
        assert 1.0 <= first <= 2.0 and 2.0 <= second <= 4.0 and 4.0 <= third <= 8.0
    assert [r.getMessage() for r in caplog.records] == [
        "reward llm_judge: 2 of 2 judgments failed and scored 0 "
        "(2x HTTPStatusError: 503 Service Unavailable); the endpoint said: 'upstream down'"
    ]


def test_failed_judgments_of_a_call_are_counted_by_kind_in_one_warning(traj, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        response = json.loads(request.content)["messages"][-1]["content"]
        if "refused" in response:
            return httpx.Response(403, json={"error": "key has no access to judge-1"})
        return _reply("<score>4</score>" if "graded" in response else "I cannot grade this.")

    fn = _judge(handler, name="quality", rubric="{response}", max_workers=3)
    texts = ["graded", "refused", "ungradable", "refused", "graded"]
    with caplog.at_level(logging.WARNING):
        results = fn([traj(text) for text in texts])
    assert [r.score for r in results] == [0.4, 0.0, 0.0, 0.0, 0.4]
    assert [r.getMessage() for r in caplog.records] == [
        "reward quality: 3 of 5 judgments failed and scored 0 (2x HTTPStatusError: 403 Forbidden, "
        """1x no_score); the endpoint said: '{"error":"key has no access to judge-1"}'"""
    ]


def test_retries_leave_the_process_wide_random_generator_alone(traj):
    calls: list[int] = []
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        return httpx.Response(502) if len(calls) % 2 else _reply("<score>7</score>")

    # a run seeds that generator and saves its state with a checkpoint; the loop and the judge's
    # workers would otherwise draw from one stream
    state = random.getstate()
    results = _judge(handler, waits, max_workers=2)([traj("a"), traj("b"), traj("c")])
    assert [r.score for r in results] == [0.7] * 3 and len(waits) == 3
    assert random.getstate() == state


def test_the_share_of_failed_judgments_of_a_step_is_a_metric(traj):
    def handler(request: httpx.Request) -> httpx.Response:
        response = json.loads(request.content)["messages"][-1]["content"]
        if response == "unreachable":
            return httpx.Response(503)
        return _reply("no grade" if response == "ungradable" else f"<score>{response}</score>")

    waits: list[float] = []
    judge = _judge(handler, waits, name="quality", rubric="{response}", max_attempts=2)
    fn = CompositeReward([("quality", 2.0, judge)])
    trajectories = [traj(text) for text in ("10", "0", "unreachable", "ungradable", "5")]
    results = fn(trajectories)
    # a grade of 0 and a failure are the same reward, and not the same row
    assert [r.score for r in results] == [2.0, 0.0, 0.0, 0.0, 1.0]
    assert [r.breakdown for r in results] == [
        {"quality": 1.0, "quality/failed": 0.0},
        {"quality": 0.0, "quality/failed": 0.0},
        {"quality": 0.0, "quality/failed": 1.0},
        {"quality": 0.0, "quality/failed": 1.0},
        {"quality": 0.5, "quality/failed": 0.0},
    ]
    for t, res in zip(trajectories, results, strict=True):
        t.reward, t.reward_breakdown = res.score, res.breakdown
    metrics = reward_metrics(trajectories)
    assert metrics["reward/quality"] == 0.3 and metrics["reward/quality/failed"] == 0.4

    # a judge that somebody called "failed" keeps its score under that name
    [result] = _judge(handler, name="failed", rubric="{response}")([traj("5")])
    assert result.breakdown == {"failed": 0.5}


def test_the_waits_of_a_judgment_stay_within_retry_max_wait_s(traj):
    calls: list[int] = []
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls))
        return httpx.Response(429, headers={"Retry-After": "40"})

    result = _judge(handler, waits, max_attempts=10, retry_max_wait_s=100)([traj("a")])[0]
    # 40s and 40s fit into 100s, a third wait of 40s does not
    assert waits == [40.0, 40.0] and len(calls) == 3
    assert result.score == 0.0 and result.info["attempts"] == 3

    calls.clear()
    far = httpx.Response(503, headers={"Retry-After": "Fri, 31 Dec 2100 23:59:59 GMT"})
    result = _judge(_failing_then(far, calls), max_attempts=10)([traj("a")])[0]
    assert "503 Service Unavailable" in result.info["error"] and len(calls) == 1


def test_retry_after_seconds():
    def asked(value: str | None) -> float:
        return retry_after_seconds(httpx.Response(429, headers={"Retry-After": value} if value else {}))

    assert asked("3") == 3.0 and asked("0.5") == 0.5
    assert asked("Fri, 31 Dec 2100 23:59:59 GMT") > 365 * 86400
    # nothing to honour: absent, in the past, or not a delay at all
    for value in (None, "0", "-5", "nan", "inf", "soon", "Wed, 21 Oct 2015 07:28:00 GMT"):
        assert asked(value) == 0.0, value


def test_the_default_wait_is_used_between_attempts(traj):
    calls: list[int] = []
    fn = registry.get("reward", "llm_judge")(
        {"base_url": "http://j", "model": "m", "retry_backoff_s": 0},
        client=_mock_client(_failing_then(httpx.Response(502), calls)),
    )
    assert fn([traj("a")])[0].score == 0.7 and len(calls) == 2


def test_a_judgment_waiting_for_its_retry_does_not_hold_up_the_others(traj):
    order: list[str] = []
    others_done = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        response = json.loads(request.content)["messages"][-1]["content"]
        order.append(response)
        if order.count("limited") == 1 and response == "limited":
            return httpx.Response(429)
        if order.count("fine") == 3:
            others_done.set()
        return _reply("<score>7</score>")

    def wait(stop: threading.Event, seconds: float) -> bool:
        # the retry comes when the three other judgments have been answered, however long that takes
        assert others_done.wait(30), "the other judgments did not run during the wait"
        return False

    fn = registry.get("reward", "llm_judge")(
        {"base_url": "http://j", "model": "m", "rubric": "{response}", "max_workers": 2},
        client=_mock_client(handler),
        wait=wait,
    )
    results = fn([traj("limited"), traj("fine"), traj("fine"), traj("fine")])
    assert [r.score for r in results] == [0.7] * 4
    assert sorted(order[:4]) == ["fine", "fine", "fine", "limited"] and order[4:] == ["limited"]


def test_an_error_that_ends_the_call_ends_the_waits_of_the_other_judgments(traj):
    seen: list[str] = []
    waiting = threading.Event()
    gave_up: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = json.loads(request.content)["messages"][-1]["content"]
        seen.append(response)
        if response == "limited":
            return httpx.Response(429, headers={"Retry-After": "50"})
        assert waiting.wait(30), "the other judgment never waited"
        return httpx.Response(401)

    def wait(stop: threading.Event, seconds: float) -> bool:
        waiting.set()
        gave_up.append(stop.wait(30))  # as threading.Event.wait(stop, 50) would, with less patience
        return True

    fn = registry.get("reward", "llm_judge")(
        {"base_url": "http://j", "model": "m", "rubric": "{response}", "on_error": "raise"},
        client=_mock_client(handler),
        wait=wait,
    )
    with pytest.raises(httpx.HTTPStatusError, match="401"):
        fn([traj("rejected"), traj("limited")])
    assert gave_up == [True]
    assert seen.count("limited") == 1


def test_llm_judge_requires_endpoint_params():
    with pytest.raises(ValueError):
        registry.get("reward", "llm_judge")({"model": "m"})


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"rubric": 'Reply as JSON: {"score": N}\n{response}'}, "literal brace"),
        ({"rubric": "{response} {}"}, "literal brace"),
        ({"max_score": 0}, "max_score"),
        ({"max_attempts": 0}, "max_attempts"),
        ({"retry_backoff_s": -1}, "retry_backoff_s"),
        ({"retry_max_wait_s": -1}, "retry_max_wait_s"),
    ],
)
def test_llm_judge_rejects_params_it_cannot_run_with(params, message):
    with pytest.raises(ValueError, match=message):
        registry.get("reward", "llm_judge")({"base_url": "http://j", "model": "m", **params})


def test_a_rubric_may_hold_escaped_braces(traj):
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["messages"][-1]["content"])
        return _reply("<score>7</score>")

    fn = _judge(handler, rubric='Grade {response} like {{"score": N}}, then <score>N</score>.')
    assert fn([traj("this")])[0].score == 0.7
    assert seen == ['Grade this like {"score": N}, then <score>N</score>.']
