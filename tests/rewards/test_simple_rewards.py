from __future__ import annotations

import pytest

from monorl.config.schema import RewardSpec
from monorl.core.errors import ConfigError
from monorl.core.interfaces import RewardFunction
from monorl.core.registry import registry
from monorl.core.types import RewardResult
from monorl.orchestrator.metrics import reward_metrics
from monorl.rewards import CompositeReward, build_reward
from monorl.rewards.base import extract_boxed, extract_code_block, extract_last_number

# -- extraction helpers --------------------------------------------------------------------


def test_extractors():
    assert extract_last_number("costs $1,250.50 so total 3") == "3"
    assert extract_last_number("answer is -4.5e2") == "-4.5e2"
    assert extract_last_number("no digits") is None
    assert extract_boxed(r"so \boxed{\frac{1}{2}} and \boxed{ 42 }") == " 42 "
    assert extract_boxed(r"\boxed{unbalanced") is None
    assert extract_code_block("```python\nprint(1)\n```") == "print(1)\n"
    assert extract_code_block("```\nx=1\n```\n```python\nx=2\n```") == "x=2\n"
    assert extract_code_block("```go\nfmt.Println()\n```") is None


# -- exact_match --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params,response,reference,expected",
    [
        ({}, "The answer is 42.", "42", 1.0),
        ({}, "The answer is 42.", "43", 0.0),
        ({}, "Total: 1,000 dollars", "1000", 1.0),
        ({"extract": "boxed"}, r"so \boxed{Paris}", "paris", 1.0),
        ({"extract": "boxed", "normalize": ["whitespace"]}, r"\boxed{Paris}", "paris", 0.0),
        ({"extract": "after_hash"}, "reasoning...\n#### 17", "The sum is 10 + 7\n#### 17", 1.0),
        ({"extract": "none"}, "  Hello   World ", "hello world", 1.0),
        ({"extract": "none"}, "0.3333", "1/3", 0.0),
        ({"numeric_tolerance": 1e-3}, "pi is 3.1416", "3.14159", 1.0),
        ({}, "answers 5", ["4", "5"], 1.0),
        ({}, "answers 5", {"answer": "5"}, 1.0),
        ({}, "nothing numeric", "5", 0.0),
    ],
)
def test_exact_match(traj, params, response, reference, expected):
    fn = registry.get("reward", "exact_match")(params)
    assert isinstance(fn, RewardFunction)
    [res] = fn([traj(response, reference)])
    assert res.score == expected
    assert res.breakdown == {"exact_match": expected}


def test_exact_match_scope_all_and_reference_key(traj):
    fn = registry.get("reward", "exact_match")({"scope": "all", "reference_key": "final"})
    t = traj("done", {"final": "7", "other": "9"}, earlier_turns=["the value is 7"])
    assert fn([t])[0].score == 1.0
    assert fn([traj("the value is 7", {"final": "7"})])[0].score == 1.0
    assert fn([traj("the value is 9", {"final": "7"})])[0].score == 0.0


def test_exact_match_rejects_bad_params():
    with pytest.raises(ConfigError):
        registry.get("reward", "exact_match")({"extract": "magic"})
    with pytest.raises(ConfigError):
        registry.get("reward", "exact_match")({"normalize": ["unicode"]})


# -- format -------------------------------------------------------------------------------


def test_format_require_all_and_fraction(traj):
    patterns = [r"<think>.*</think>", r"<answer>.*</answer>"]
    strict = registry.get("reward", "format")({"patterns": patterns})
    partial = registry.get("reward", "format")({"patterns": patterns, "require_all": False, "name": "fmt"})
    good = traj("<think>\nhmm\n</think><answer>4</answer>")
    half = traj("<think>hmm</think> 4")
    results = strict([good, half])
    assert [r.score for r in results] == [1.0, 0.0]
    assert results[0].info["matched"] == [True, True]
    results = partial([good, half])
    assert [r.score for r in results] == [1.0, 0.5]
    assert results[1].breakdown == {"fmt": 0.5}


def test_format_flags_and_fullmatch(traj):
    fn = registry.get("reward", "format")({"pattern": r"yes", "flags": ["IGNORECASE"], "fullmatch": True})
    assert fn([traj("YES")])[0].score == 1.0
    assert fn([traj("yes!")])[0].score == 0.0
    with pytest.raises(ConfigError):
        registry.get("reward", "format")({})


# -- length -------------------------------------------------------------------------------


def test_length_reward_band_and_penalty(traj):
    fn = registry.get("reward", "length")({"min_tokens": 2, "max_tokens": 4, "penalty_per_token": 0.25})
    scores = [
        r.score for r in fn([traj("a b c"), traj("a b c d e f"), traj("a"), traj("a b c d e f g h i j")])
    ]
    assert scores == [1.0, 0.5, 0.75, 0.0]
    chars = registry.get("reward", "length")(
        {"max_tokens": 3, "unit": "chars", "penalty_per_token": 0.1, "base": 0.0, "min_reward": -1.0}
    )
    assert chars([traj("abcdef")])[0].score == pytest.approx(-0.3)
    assert chars([traj("abcdef")])[0].info["length"] == 6
    with pytest.raises(ConfigError):
        registry.get("reward", "length")({"min_tokens": 5, "max_tokens": 2})


# -- math_verify --------------------------------------------------------------------------


def test_math_verify(traj):
    pytest.importorskip("math_verify")
    fn = registry.get("reward", "math_verify")({})
    cases = [
        (r"The answer is $\boxed{\frac{1}{2}}$", "0.5", 1.0),
        (r"so $x = 2$", r"$\boxed{2}$", 1.0),
        (r"Final answer: 42", "42", 1.0),
        (r"$\boxed{3}$", "4", 0.0),
        ("I have no idea", "4", 0.0),
        (r"$\boxed{(1, 2)}$", ["(2, 1)", "(1, 2)"], 1.0),
    ]
    results = fn([traj(resp, ref) for resp, ref, _ in cases])
    assert [r.score for r in results] == [c[2] for c in cases]


def test_math_verify_runs_off_main_thread(traj):
    pytest.importorskip("math_verify")
    from concurrent.futures import ThreadPoolExecutor

    fn = registry.get("reward", "math_verify")({})
    with ThreadPoolExecutor(1) as pool:
        [res] = pool.submit(fn, [traj(r"$\boxed{10}$", "10")]).result()
    assert res.score == 1.0


# -- composite / build_reward -------------------------------------------------------------


def test_build_reward_weighted_sum_with_breakdown(traj):
    specs = [
        RewardSpec(type="exact_match", weight=2.0),
        RewardSpec(type="format", weight=0.5, params={"patterns": [r"\d"], "name": "has_digit"}),
        RewardSpec(type="format", weight=0.25, params={"patterns": [r"^#"]}),
        RewardSpec(type="format", weight=0.25, params={"patterns": [r"!$"]}),
    ]
    fn = build_reward(specs)
    assert isinstance(fn, RewardFunction)
    results = fn([traj("# answer 4!", "4"), traj("nope", "4")])
    assert results[0].score == pytest.approx(2.0 + 0.5 + 0.25 + 0.25)
    assert results[0].breakdown == {"exact_match": 1.0, "has_digit": 1.0, "format": 1.0, "format#2": 1.0}
    assert results[1].score == 0.0
    assert results[1].breakdown == {"exact_match": 0.0, "has_digit": 0.0, "format": 0.0, "format#2": 0.0}
    assert results[0].info["exact_match"]["extracted"] == "4"


def test_build_reward_accepts_dicts_and_registered_composite(traj):
    fn = build_reward(
        [{"type": "exact_match"}, {"type": "length", "weight": 0.1, "params": {"max_tokens": 1}}]
    )
    [res] = fn([traj("4 4", "4")])
    assert res.score == pytest.approx(1.0 + 0.1 * 0.99)
    nested = registry.get("reward", "composite")(
        {"name": "outer", "rewards": [{"type": "exact_match", "weight": 3.0}]}
    )
    assert nested.name == "outer"
    assert nested([traj("4", "4")])[0].score == 3.0
    with pytest.raises(ValueError):
        build_reward([])


def test_nested_composite_keeps_its_total_and_shows_its_members(traj):
    style = {
        "type": "composite",
        "weight": 0.5,
        "params": {
            "name": "style",
            "rewards": [
                {"type": "format", "params": {"patterns": [r"^#"]}},
                {"type": "format", "weight": 2.0, "params": {"patterns": [r"!$"], "name": "bang"}},
            ],
        },
    }
    fn = build_reward([{"type": "exact_match"}, style])
    trajectories = [traj("# answer 4!", "4"), traj("# nope", "4")]
    good, bad = fn(trajectories)
    assert good.score == pytest.approx(1.0 + 0.5 * 3.0)
    assert good.breakdown == {"exact_match": 1.0, "style": 3.0, "style/format": 1.0, "style/bang": 1.0}
    assert bad.score == pytest.approx(0.5 * 1.0)
    assert bad.breakdown == {"exact_match": 0.0, "style": 1.0, "style/format": 1.0, "style/bang": 0.0}

    for t, res in zip(trajectories, (good, bad), strict=True):
        t.reward, t.reward_breakdown = res.score, res.breakdown
    metrics = reward_metrics(trajectories)
    assert metrics["reward/style"] == 2.0
    assert metrics["reward/style/format"] == 1.0 and metrics["reward/style/bang"] == 0.5


def test_nested_composite_members_at_depth_two_and_next_to_a_colliding_name(traj):
    inner = {"type": "composite", "params": {"name": "inner", "rewards": [{"type": "exact_match"}]}}
    outer = {
        "type": "composite",
        "params": {"name": "outer", "rewards": [inner, {"type": "format", "params": {"patterns": ["4"]}}]},
    }
    # A top-level reward may already be called what a nested member would be reported as: its
    # score stays under that name wherever it is listed, and the nested member gets a suffix.
    taken = {"type": "format", "weight": 0.0, "params": {"patterns": ["never"], "name": "outer/format"}}
    expected = {
        "outer": 2.0,
        "outer/inner": 1.0,
        "outer/inner/exact_match": 1.0,
        "outer/format#2": 1.0,
        "outer/format": 0.0,
    }
    for specs in ([outer, taken], [taken, outer]):
        [res] = build_reward(specs)([traj("4", "4")])
        assert res.score == 2.0
        assert res.breakdown == expected


class Checked:
    """A reward function that reports more than its score: whether it could check the response."""

    def __init__(self, name: str = "checked") -> None:
        self.name = name

    def __call__(self, trajectories):
        out = []
        for t in trajectories:
            score = float("4" in t.last_response_text)
            breakdown = {self.name: score}
            if "?" in t.last_response_text:  # reported for some responses only
                breakdown["unsure"] = 1.0
            out.append(RewardResult(score=score, breakdown=breakdown))
        return out


def test_what_a_reward_function_reports_besides_its_score_is_listed_under_its_key(traj):
    members = [("exact_match", 1.0, registry.get("reward", "exact_match")({})), ("check", 2.0, Checked())]
    trajectories = [traj("4", "4"), traj("4?", "4"), traj("5?", "4"), traj("5", "4")]
    results = CompositeReward(members)(trajectories)
    # the totals are what they are without the extra entry, and the score under the function's own
    # name is not listed a second time
    assert [r.score for r in results] == [3.0, 3.0, 0.0, 0.0]
    assert [r.breakdown for r in results] == [
        {"exact_match": 1.0, "check": 1.0},
        {"exact_match": 1.0, "check": 1.0, "check/unsure": 1.0},
        {"exact_match": 0.0, "check": 0.0, "check/unsure": 1.0},
        {"exact_match": 0.0, "check": 0.0},
    ]

    for t, res in zip(trajectories, results, strict=True):
        t.reward, t.reward_breakdown = res.score, res.breakdown
    metrics = reward_metrics(trajectories)
    assert metrics["reward/mean"] == 1.5 and metrics["reward/check"] == 0.5
    assert metrics["reward/check/unsure"] == 0.5  # two of four, though two rows do not have it


def test_an_extra_entry_gives_way_to_a_member_of_that_name_and_nests_like_a_member(traj):
    # ``check/unsure`` is taken by a member, at the top and inside a composite, listed before or
    # after the function that reports ``unsure``
    taken = ("check/unsure", 0.0, registry.get("reward", "format")({"patterns": ["never"]}))
    for members in ([("check", 1.0, Checked()), taken], [taken, ("check", 1.0, Checked())]):
        expected = {"check": 1.0, "check/unsure#2": 1.0, "check/unsure": 0.0}
        [res] = CompositeReward(members)([traj("4?")])
        assert res.score == 1.0 and res.breakdown == expected
        [res] = CompositeReward([("outer", 1.0, CompositeReward(members))])([traj("4?")])
        assert res.score == 1.0
        assert res.breakdown == {"outer": 1.0, **{f"outer/{k}": v for k, v in expected.items()}}

    # a composite called like the function and a member called like the entry: two members never
    # share a key, so neither do their entries
    inner = CompositeReward([("unsure", 1.0, registry.get("reward", "format")({"patterns": ["never"]}))])
    specs = [("check", 1.0, inner), ("check#2", 1.0, Checked())]
    [res] = CompositeReward(specs)([traj("4?")])
    assert res.breakdown == {"check": 0.0, "check/unsure": 0.0, "check#2": 1.0, "check#2/unsure": 1.0}


def test_a_callable_without_a_name_is_still_a_member(traj):
    def plain(trajectories):
        return [RewardResult(score=1.0, breakdown={"part": 0.5}) for _ in trajectories]

    [res] = CompositeReward([("plain", 2.0, plain)])([traj("4")])
    assert res.score == 2.0 and res.breakdown == {"plain": 1.0, "plain/part": 0.5}


def test_rewards_reject_non_trajectories():
    fn = registry.get("reward", "exact_match")({})
    with pytest.raises(TypeError):
        fn(["just a string"])  # type: ignore[list-item]
