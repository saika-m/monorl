"""The keys a built-in declares are the keys it may read, and the keys a config may give it."""

from __future__ import annotations

import pytest

from monorl.core.errors import ConfigError
from monorl.core.params import checked_params, declared_params, unknown_params


def test_a_built_in_cannot_read_a_key_it_does_not_declare() -> None:
    params = checked_params("reward", "exact_match", {"extract": "boxed"})
    assert params.get("extract") == "boxed" and params.get("scope", "last") == "last"
    assert dict(params) == {"extract": "boxed"}
    # a key that is read but not declared would be refused in every config that sets it
    for read in (lambda: params.get("extract_regex"), lambda: params["extract_regex"]):
        with pytest.raises(RuntimeError, match=r"exact_match reward reads params\.extract_regex"):
            read()
    with pytest.raises(RuntimeError, match="does not declare"):
        "extract_regex" in params  # noqa: B015


def test_unknown_keys_are_reported_one_per_line_with_the_key_they_resemble() -> None:
    with pytest.raises(ConfigError) as raised:
        checked_params("env", "remote", {"comand": ["x"], "zzz": 1, "cwd": None}, where="env.params")
    takes = (
        "he remote environment takes: transport, command, cwd, env, url, urls, headers, parallel, "
        "timeout_s, max_response_bytes"
    )
    assert str(raised.value).splitlines() == [
        f"env.params.comand: unknown key; did you mean command? T{takes}",
        f"env.params.zzz: unknown key; t{takes}",
    ]


def test_a_type_that_is_not_built_in_declares_nothing_and_takes_anything() -> None:
    assert declared_params("reward", "my_keyword_reward") is None
    assert unknown_params("reward", "my_keyword_reward", {"keyword": "the"}) == []
    assert declared_params("env", "my_guess_env") is None
    assert unknown_params("env", "my_guess_env", {"high": 20}) == []
