import pytest

from monorl.config.schema import RunConfig
from monorl.core.errors import ConfigError


def _base(**overrides):
    cfg = {"model": {"name_or_path": "x"}, "data": {"uri": "d.jsonl"}, "rewards": [{"type": "exact_match"}]}
    cfg.update(overrides)
    return cfg


def test_ppo_is_accepted_wherever_a_critic_exists() -> None:
    # the reference trainer (and the Unsloth wrapper around it) has a value head, verl its own critic
    for backend in ("reference", "unsloth", "verl"):
        cfg = RunConfig.from_dict(_base(algorithm={"name": "ppo"}, trainer={"backend": backend}))
        assert cfg.algorithm.name == "ppo"
    with pytest.raises(ConfigError, match="credit_assignment"):
        RunConfig.from_dict(_base(algorithm={"credit_assignment": "token"}))
    with pytest.raises(ConfigError, match="value_clip"):
        RunConfig.from_dict(_base(algorithm={"name": "ppo", "value_clip": 0.0}))


def test_group_algorithms_need_groups() -> None:
    for name in ("grpo", "dapo", "rloo"):
        with pytest.raises(ConfigError, match=f"{name} needs rollout.group_size >= 2"):
            RunConfig.from_dict(_base(algorithm={"name": name}, rollout={"group_size": 1}))


@pytest.mark.parametrize("backend", ["reference", "verl"])
def test_one_rollout_per_prompt_is_enough_where_no_group_statistic_is_used(backend: str) -> None:
    """reinforce and ppo have no group baseline: advantage_norm (default 'group') is not theirs to read."""
    ppo = RunConfig.from_dict(
        _base(algorithm={"name": "ppo"}, rollout={"group_size": 1}, trainer={"backend": backend})
    )
    assert ppo.algorithm.advantage_norm == "group"
    for norm in ("batch", "none"):
        algorithm = {"name": "reinforce", "advantage_norm": norm}
        RunConfig.from_dict(
            _base(algorithm=algorithm, rollout={"group_size": 1}, trainer={"backend": backend})
        )
    for other in ("reference", "nemo_rl"):
        RunConfig.from_dict(
            _base(algorithm={"name": "reinforce"}, rollout={"group_size": 1}, trainer={"backend": other})
        )


def test_verl_reinforce_with_a_group_baseline_needs_a_group() -> None:
    """verl compiles reinforce + advantage_norm 'group' to REINFORCE++ with the group mean as baseline:
    with one rollout per prompt every advantage would be zero."""
    algorithm = {"name": "reinforce"}
    with pytest.raises(ConfigError, match="verl.*advantage_norm.*'batch' or 'none'"):
        RunConfig.from_dict(
            _base(algorithm=algorithm, rollout={"group_size": 1}, trainer={"backend": "verl"})
        )
    RunConfig.from_dict(_base(algorithm=algorithm, rollout={"group_size": 2}, trainer={"backend": "verl"}))


def _off_policy(off_policy: dict, **rollout):
    return RunConfig.from_dict(_base(algorithm={"off_policy": off_policy}, rollout=rollout))


def test_off_policy_corrections_imply_the_trainer_anchor() -> None:
    assert not RunConfig.from_dict(_base()).algorithm.off_policy.decoupled
    assert _off_policy({"importance_weights": "token"}).algorithm.off_policy.decoupled
    assert _off_policy({"ratio_mask": "sequence_mean"}).algorithm.off_policy.decoupled
    assert not _off_policy({"anchor": "behavior"}).algorithm.off_policy.decoupled


def test_off_policy_corrections_need_two_distinct_policies() -> None:
    for correction in ({"importance_weights": "sequence"}, {"ratio_mask": "token"}):
        with pytest.raises(ConfigError, match="anchor"):
            _off_policy({"anchor": "behavior", **correction})


def test_off_policy_bounds_must_keep_on_policy_data_untouched() -> None:
    for bounds in ({"importance_cap": 0.9}, {"ratio_mask_low": 1.5}, {"ratio_mask_high": 0.9}):
        with pytest.raises(ConfigError):
            _off_policy({"importance_weights": "token", "ratio_mask": "token", **bounds})


def test_score_centering_needs_the_behaviour_anchor_and_untruncated_sampling() -> None:
    assert _off_policy({"score_centering": True}).algorithm.off_policy.score_centering
    for conflict in ({"anchor": "trainer"}, {"importance_weights": "token"}, {"ratio_mask": "sequence"}):
        with pytest.raises(ConfigError, match="score_centering"):
            _off_policy({"score_centering": True, **conflict})
    for truncation in (
        {"temperature": 0.0},
        {"top_k": 50},
        {"top_p": 0.9},
        {"extra": {"min_p": 0.05}},
        {"extra": {"repetition_penalty": 1.1}},
    ):
        with pytest.raises(ConfigError, match="sampling distribution"):
            _off_policy({"score_centering": True}, **truncation)
