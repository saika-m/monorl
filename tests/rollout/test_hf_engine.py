from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from monorl.config.schema import RunConfig
from monorl.core.errors import GenerationCancelled
from monorl.core.interfaces import RolloutEngine
from monorl.core.types import SamplingParams
from monorl.rollout.hf_engine import HFRolloutEngine, _FinishRecorder, chat_prompt_ids

CHATS = [
    [{"role": "user", "content": "Say hello."}],
    [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2+2? Explain briefly."},
    ],
]


def test_registry_and_protocol(engine: HFRolloutEngine) -> None:
    from monorl.core.registry import registry

    assert registry.get("rollout_engine", "hf").__name__ == "make_hf_engine"
    assert isinstance(engine, RolloutEngine)
    assert engine.tokenizer.pad_token_id is not None


def test_n_completions_order_and_prompt_ids(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=3, max_new_tokens=6, temperature=1.0, seed=1)
    out = engine.generate(CHATS, params)
    assert len(out) == 2
    for chat, group in zip(CHATS, out, strict=True):
        assert len(group) == 3
        expected = chat_prompt_ids(engine.tokenizer, chat)
        for c in group:
            assert c.prompt_token_ids == expected
            assert 1 <= len(c.token_ids) <= 6
            assert c.logprobs is not None and len(c.logprobs) == len(c.token_ids)
            assert c.finish_reason in ("stop", "length")
            assert all(lp <= 0.0 for lp in c.logprobs)
    # Different prompts condition on different ids, so groups are not interchangeable.
    assert out[0][0].prompt_token_ids != out[1][0].prompt_token_ids
    # Sampling at T=1 with a 49k vocab yields distinct completions within a group.
    assert len({tuple(c.token_ids) for c in out[0]}) > 1


def test_seed_is_deterministic(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=2, max_new_tokens=5, temperature=1.0, seed=123)
    a = engine.generate(CHATS, params)
    b = engine.generate(CHATS, params)
    assert [[c.token_ids for c in g] for g in a] == [[c.token_ids for c in g] for g in b]
    assert [[c.logprobs for c in g] for g in a] == [[c.logprobs for c in g] for g in b]


def test_logprobs_match_forward_pass_and_exclude_padding(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=2, max_new_tokens=7, temperature=1.0, seed=7, logprobs=True)
    out = engine.generate(CHATS, params)
    lengths = {len(chat_prompt_ids(engine.tokenizer, c)) for c in CHATS}
    assert len(lengths) == 2, "prompts must differ in length so the batch is left-padded"
    width = max(lengths)
    for chat, group in zip(CHATS, out, strict=True):
        for c in group:
            # Unpadded prompt ids are recorded even though the batch was left-padded to `width`.
            assert c.prompt_token_ids == chat_prompt_ids(engine.tokenizer, chat)
            assert len(c.prompt_token_ids) <= width
            assert len(c.logprobs) == len(c.token_ids)
            ids = torch.tensor([c.prompt_token_ids + c.token_ids])
            with torch.no_grad():
                logits = engine.model(input_ids=ids).logits[0].float()
            start = len(c.prompt_token_ids)
            ref = torch.log_softmax(logits[start - 1 : -1], dim=-1).gather(1, ids[0, start:, None]).squeeze(1)
            torch.testing.assert_close(torch.tensor(c.logprobs), ref, atol=1e-4, rtol=0.0)


def test_logprobs_are_temperature_scaled(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=1, max_new_tokens=4, temperature=0.5, seed=3)
    (c,) = engine.generate(CHATS[:1], params)[0]
    ids = torch.tensor([c.prompt_token_ids + c.token_ids])
    with torch.no_grad():
        logits = engine.model(input_ids=ids).logits[0].float()
    start = len(c.prompt_token_ids)
    ref = torch.log_softmax(logits[start - 1 : -1] / 0.5, dim=-1).gather(1, ids[0, start:, None]).squeeze(1)
    torch.testing.assert_close(torch.tensor(c.logprobs), ref, atol=1e-4, rtol=0.0)


def test_logprobs_disabled(engine: HFRolloutEngine) -> None:
    out = engine.generate(CHATS[:1], SamplingParams(n=1, max_new_tokens=3, logprobs=False, seed=0))
    assert out[0][0].logprobs is None


def test_raw_token_prompt_continues_exactly(engine: HFRolloutEngine) -> None:
    greedy = SamplingParams(n=1, max_new_tokens=8, temperature=0.0)
    (first,) = engine.generate(CHATS[1:], greedy)[0]
    assert len(first.token_ids) >= 4
    prefix = first.prompt_token_ids + first.token_ids[:3]
    (cont,) = engine.generate(
        [prefix], SamplingParams(n=1, max_new_tokens=len(first.token_ids) - 3, temperature=0.0)
    )[0]
    assert cont.prompt_token_ids == prefix
    assert cont.token_ids == first.token_ids[3:]


def test_greedy_replicates_n(engine: HFRolloutEngine) -> None:
    out = engine.generate(CHATS[:1], SamplingParams(n=3, max_new_tokens=4, temperature=0.0))
    assert len(out[0]) == 3
    assert len({tuple(c.token_ids) for c in out[0]}) == 1


def test_stop_string_sets_finish_reason(engine: HFRolloutEngine) -> None:
    sampled = SamplingParams(n=1, max_new_tokens=8, temperature=1.0, seed=11)
    (full,) = engine.generate(CHATS[:1], sampled)[0]
    assert full.finish_reason == "length" and len(full.token_ids) == 8
    # Stop on the text of the first few generated tokens: the same seed must now end early with "stop".
    stop_text = next(t for k in range(1, 7) if (t := engine.tokenizer.decode(full.token_ids[:k])).strip())
    (stopped,) = engine.generate(
        CHATS[:1], SamplingParams(n=1, max_new_tokens=8, temperature=1.0, seed=11, stop=[stop_text])
    )[0]
    assert stopped.finish_reason == "stop"
    assert len(stopped.token_ids) < 8
    assert stopped.token_ids == full.token_ids[: len(stopped.token_ids)]
    assert len(stopped.logprobs) == len(stopped.token_ids)
    assert stopped.text.endswith(stop_text)  # the stop string stays in the text (RolloutEngine contract)


def test_finish_recorder_counts_the_tokens_up_to_eos() -> None:
    rec = _FinishRecorder([9] * 3, eos_ids=[2], stop=None)
    ids = torch.tensor([[5, 2], [5, 6], [2, 2]])
    assert rec(ids[:, :1], None).tolist() == [False, False, True]
    assert rec(ids, None).tolist() == [True, False, True]
    # the padding generate writes after a row's EOS is not counted; the middle row never finished
    assert rec.num_tokens == [2, None, 1]
    assert not rec.cut


def test_finish_recorder_ends_the_call_for_the_rows_that_go_on() -> None:
    rec = _FinishRecorder([9] * 3, eos_ids=[2], stop=None, slack=0)
    ids = torch.tensor([[5, 7, 2], [5, 2, 2], [6, 6, 2]])
    assert rec(ids[:, :1], None).tolist() == [False, False, False]
    assert rec(ids[:, :2], None).tolist() == [True, True, True]  # one row finished, two go on
    assert rec.cut and rec.num_tokens == [None, 2, None]
    # On MPS generate runs one more step before it stops, and takes it back: the padding it wrote
    # for the two rows (EOS here) is not their end.
    assert rec(ids, None).tolist() == [True, True, True]
    assert rec.num_tokens == [None, 2, None]


def test_finish_recorder_lets_finished_rows_cost_what_ending_the_call_costs() -> None:
    """With a slack of 4 forward passes of a row, a call of 4 rows goes on for 4 steps after one row
    has finished, or for 2 after two have, and is then ended for the others."""
    for finished, steps_after in ((1, 4), (2, 2)):
        rec = _FinishRecorder([9] * 4, eos_ids=[2], stop=None, slack=4)
        ids = torch.tensor([[2]] * finished + [[5]] * (4 - finished))
        verdicts = [rec(ids, None).tolist()]
        while not rec.cut:
            ids = torch.cat([ids, torch.full((4, 1), 5)], dim=1)
            verdicts.append(rec(ids, None).tolist())
        assert len(verdicts) - 1 == steps_after
        assert verdicts[0] == [True] * finished + [False] * (4 - finished)
        assert verdicts[-1] == [True] * 4
        assert rec.num_tokens == [1] * finished + [None] * (4 - finished)


def test_finish_recorder_without_slack_runs_to_the_last_row() -> None:
    rec = _FinishRecorder([9] * 2, eos_ids=[2], stop=None)
    ids = torch.tensor([[2, 2, 2], [5, 5, 2]])
    assert [rec(ids[:, :k], None).tolist() for k in (1, 2, 3)] == [[True, False], [True, False], [True, True]]
    assert not rec.cut and rec.num_tokens == [1, 3]


def test_logprobs_stay_those_of_the_raw_distribution_under_every_sampling_knob(
    engine: HFRolloutEngine,
) -> None:
    """Repetition penalty and the truncations change what is sampled from, not what is reported:
    the trainer's log-softmax of the same logits has to give an importance ratio of 1."""
    params = SamplingParams(
        n=2,
        max_new_tokens=6,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        min_p=0.05,
        repetition_penalty=1.3,
        seed=11,
        top_logprobs=3,
    )
    for group in engine.generate(CHATS, params):
        for c in group:
            ids = torch.tensor([c.prompt_token_ids + c.token_ids])
            with torch.no_grad():
                logits = engine.model(input_ids=ids).logits[0].float()
            start = len(c.prompt_token_ids)
            reference = torch.log_softmax(logits[start - 1 : -1] / 0.8, dim=-1)
            sampled = reference.gather(1, ids[0, start:, None]).squeeze(1)
            torch.testing.assert_close(torch.tensor(c.logprobs), sampled, atol=1e-4, rtol=0.0)
            top = torch.tensor(list(c.top_logprobs.logprobs)).reshape(len(c.token_ids), 3)
            torch.testing.assert_close(top, reference.topk(3).values, atol=1e-4, rtol=0.0)


def test_sampling_knobs_accepted(engine: HFRolloutEngine) -> None:
    params = SamplingParams(
        n=2,
        max_new_tokens=4,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        min_p=0.05,
        repetition_penalty=1.2,
        seed=5,
    )
    out = engine.generate(CHATS, params)
    assert all(len(g) == 2 for g in out)
    assert all(len(c.logprobs) == len(c.token_ids) for g in out for c in g)


def test_logprobs_of_a_peft_wrapped_policy(tiny_config: RunConfig) -> None:
    """The TRL adapter samples from the trainer's LoRA-wrapped model. ``generate`` of the wrapper
    runs the transformers model inside it, never the wrapper's own forward pass."""
    from peft import LoraConfig, get_peft_model

    eng = HFRolloutEngine(tiny_config)
    try:
        # init_lora_weights=False: the adapter changes the logits, as it does after a training step
        lora = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
        eng.model = get_peft_model(eng.model, lora).eval()
        params = SamplingParams(n=2, max_new_tokens=5, temperature=1.0, seed=3, top_logprobs=2)
        for group in eng.generate(CHATS, params):
            for c in group:
                start = len(c.prompt_token_ids)
                reference = torch.log_softmax(_logits(eng, c.prompt_token_ids + c.token_ids)[0], dim=-1)
                sampled = reference[start - 1 : -1].gather(1, torch.tensor(c.token_ids)[:, None]).squeeze(1)
                torch.testing.assert_close(torch.tensor(c.logprobs), sampled, rtol=1e-4, atol=1e-5)
                top = torch.tensor(list(c.top_logprobs.logprobs)).reshape(len(c.token_ids), 2)
                torch.testing.assert_close(
                    top, reference[start - 1 : -1].topk(2).values, rtol=1e-4, atol=1e-5
                )
    finally:
        eng.shutdown()


def _logits(engine: HFRolloutEngine, ids: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return engine.model(input_ids=torch.tensor([ids])).logits.float().clone()


def test_update_weights_and_load_from_path(
    tiny_config: RunConfig, tiny_model_dir: Path, tmp_path: Path
) -> None:
    eng = HFRolloutEngine(tiny_config)
    probe = chat_prompt_ids(eng.tokenizer, CHATS[0])
    before = _logits(eng, probe)
    name = "model.layers.0.mlp.down_proj.weight"
    original = eng.model.state_dict()[name].clone()
    eng.update_weights([(name, original + 0.5)])
    assert not torch.allclose(_logits(eng, probe), before)
    with pytest.raises(KeyError):
        eng.update_weights([("model.layers.0.nope.weight", original)])
    with pytest.raises(ValueError):
        eng.update_weights([(name, original[:1])])
    eng.load_weights_from_path(tiny_model_dir)
    torch.testing.assert_close(_logits(eng, probe), before)
    # Weights exported by another model instance (HF names) load too.
    other = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
    eng.update_weights(other.state_dict().items())
    torch.testing.assert_close(_logits(eng, probe), before)
    eng.sleep()
    eng.wake_up()
    torch.testing.assert_close(_logits(eng, probe), before)
    eng.shutdown()


@pytest.mark.slow
def test_smollm2_generates_sensible_text(smol_model_id: str) -> None:
    config = RunConfig.from_dict(
        {
            "model": {"name_or_path": smol_model_id, "dtype": "float32"},
            "data": {"uri": "unused.jsonl"},
            "rollout": {"backend": "hf", "group_size": 2, "device": "cpu"},
            "rewards": [{"type": "exact_match"}],
        }
    )
    eng = HFRolloutEngine(config)
    try:
        out = eng.generate(
            [[{"role": "user", "content": "What is the capital of France? Answer in one word."}]],
            SamplingParams(n=1, max_new_tokens=24, temperature=0.0),
        )
    finally:
        eng.shutdown()
    (c,) = out[0]
    assert "paris" in c.text.lower()
    assert c.finish_reason == "stop"
    assert c.token_ids[-1] in (eng.tokenizer.eos_token_id,) or c.finish_reason == "stop"


def test_top_k_minus_one_disables_truncation(engine: HFRolloutEngine) -> None:
    """transformers backfills ``top_k=None`` with 50; the engine must pass 0 so sampling is
    full-vocabulary and matches the full-softmax logprobs it returns."""
    gc = engine._generation_config(
        SamplingParams(n=1, max_new_tokens=1, temperature=1.0, top_k=-1, min_p=0.0), 1, output_logits=False
    )
    effective, _ = engine.model._prepare_generation_config(gc)
    assert effective.top_k == 0
    assert effective.min_p == 0.0
    gc = engine._generation_config(
        SamplingParams(n=1, max_new_tokens=1, temperature=1.0, top_k=7), 1, output_logits=False
    )
    effective, _ = engine.model._prepare_generation_config(gc)
    assert effective.top_k == 7


def test_cancel_interrupts_generation_and_engine_stays_usable(
    engine: HFRolloutEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = [[{"role": "user", "content": "hi there"}]]
    short = SamplingParams(n=2, max_new_tokens=6, temperature=1.0, seed=11, logprobs=True)
    before = engine.generate(prompts, short)
    engine.cancel()  # idle: must not leak into the next call

    decoding = threading.Event()
    model_generate = engine.model.generate

    def spy(*args, **kwargs):
        decoding.set()
        return model_generate(*args, **kwargs)

    monkeypatch.setattr(engine.model, "generate", spy)
    # 2000 steps take seconds on this model. EOS cannot end the call early: with this seed the
    # random-init policy (uniform-ish over a 49k vocab) emits none, and both rows would have to.
    long = SamplingParams(n=2, max_new_tokens=2000, temperature=1.0, seed=3, logprobs=True)
    rng_state = torch.random.get_rng_state()
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["result"] = engine.generate(prompts, long)
        except BaseException as e:  # noqa: BLE001 - recorded for the assertions
            outcome["error"] = e

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert decoding.wait(30)
    time.sleep(0.1)  # a few decoding steps in
    t0 = time.monotonic()
    engine.cancel()
    thread.join(30)
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), GenerationCancelled), outcome
    assert time.monotonic() - t0 < 3.0
    assert torch.equal(torch.random.get_rng_state(), rng_state)  # the seeded RNG fork was unwound
    assert not engine.model.training

    after = engine.generate(prompts, short)
    assert len(after) == 1 and len(after[0]) == short.n
    for c in after[0]:
        assert c.logprobs is not None and len(c.logprobs) == len(c.token_ids) >= 1
    assert [(c.token_ids, c.logprobs) for c in after[0]] == [(c.token_ids, c.logprobs) for c in before[0]]


@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("short_row", [False, True])
def test_shared_prefill_preserves_samples_and_policy_logprobs(
    engine: HFRolloutEngine, monkeypatch: pytest.MonkeyPatch, n: int, short_row: bool
) -> None:
    # Duplicates need not be adjacent; unequal lengths exercise left-padding and cache row order.
    first, second = [11, 12, 13, 14] * 40, [17, 18, 19] * 45
    prompts = [first, [21] if short_row else second, first]
    params = SamplingParams(
        n=n,
        max_new_tokens=6,
        temperature=0.8,
        top_p=0.9,
        top_k=31,
        repetition_penalty=1.1,
        seed=41,
        top_logprobs=5,
    )
    with monkeypatch.context() as patch:
        patch.setattr(engine, "_shared_prompt_cache", lambda *args: None)
        expected = engine.generate(prompts, params)
    actual = engine.generate(prompts, params)
    repeated = engine.generate(prompts, params)
    for ids, expected_group, actual_group, repeated_group in zip(
        prompts, expected, actual, repeated, strict=True
    ):
        for before, after, again in zip(expected_group, actual_group, repeated_group, strict=True):
            assert after.prompt_token_ids == ids
            assert after.token_ids == before.token_ids == again.token_ids
            assert after.finish_reason == before.finish_reason
            torch.testing.assert_close(
                torch.tensor(after.logprobs), torch.tensor(before.logprobs), atol=1e-4, rtol=0.0
            )
            assert list(after.top_logprobs.token_ids) == list(before.top_logprobs.token_ids)
            torch.testing.assert_close(
                torch.tensor(after.top_logprobs.logprobs),
                torch.tensor(before.top_logprobs.logprobs),
                atol=1e-4,
                rtol=0.0,
            )


def test_shared_prefill_does_not_reuse_cache_after_weight_updates(
    engine: HFRolloutEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    prompts = [[11, 12, 13, 14] * 40] * 2
    params = SamplingParams(n=2, max_new_tokens=4, temperature=0.8, seed=7)
    before = engine.generate(prompts, params)
    name = "model.layers.0.mlp.down_proj.weight"
    original = engine.model.state_dict()[name].clone()
    try:
        engine.update_weights([(name, original + 0.5)])
        actual = engine.generate(prompts, params)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_shared_prompt_cache", lambda *args: None)
            expected = engine.generate(prompts, params)
        for actual_group, expected_group in zip(actual, expected, strict=True):
            for after, reference in zip(actual_group, expected_group, strict=True):
                assert after.token_ids == reference.token_ids
                torch.testing.assert_close(
                    torch.tensor(after.logprobs), torch.tensor(reference.logprobs), atol=1e-4, rtol=0.0
                )
        assert any(
            old.logprobs != new.logprobs
            for old_group, new_group in zip(before, actual, strict=True)
            for old, new in zip(old_group, new_group, strict=True)
        )
    finally:
        engine.update_weights([(name, original)])
