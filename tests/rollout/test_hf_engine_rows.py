"""Batches of the HF engine whose rows end at different steps: a finished row is not computed
again, and what the engine promises about the others (log-probs of the raw distribution, finish
reasons, token counts, stop strings, seeds, the distribution that is sampled from) holds across the
``generate`` calls such a batch is decoded with."""

from __future__ import annotations

import contextlib
import dataclasses
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import pytest
import torch

from monorl.adapters.trl.rollout import TrainerPolicyEngine
from monorl.config.schema import RunConfig
from monorl.core.types import Completion, SamplingParams
from monorl.rollout.hf_engine import HFRolloutEngine, chat_prompt_ids
from monorl.rollout.media import vlm_encode
from tests.conftest import TINY_TOKENIZER
from tests.rollout.tiny_vlm import image_chat

CHATS = [
    [{"role": "user", "content": "Say hello."}],
    [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "What is 2+2? Explain briefly."},
    ],
]
EVERY = 8


def _ending(engine: HFRolloutEngine, eos_ids: Iterable[int] | None = None) -> HFRolloutEngine:
    """An engine on the model of ``engine`` whose rows end at any of ``eos_ids``, by default at
    every 8th token of the vocabulary: the tiny models are random, so a row then ends after 8
    tokens on average, and the rows of a batch at steps of their own."""
    if eos_ids is None:
        eos_ids = range(0, len(engine.tokenizer), EVERY)
    return TrainerPolicyEngine(
        engine.model, engine.processor or engine.tokenizer, device=engine.device, eos_token_ids=eos_ids
    )


@contextlib.contextmanager
def _forward_rows(model: torch.nn.Module) -> Iterator[list[int]]:
    """The number of rows of every forward pass of ``model`` inside the block."""
    rows: list[int] = []
    hook = model.register_forward_hook(lambda module, args, output: rows.append(output.logits.shape[0]))
    try:
        yield rows
    finally:
        hook.remove()


def _needed_rows(completions: list[Completion]) -> list[int]:
    """The rows a decoding of ``completions`` has to compute per forward pass: pass ``t`` gives the
    logits token ``t + 1`` is chosen from, which only a row with such a token needs."""
    lengths = [len(c.token_ids) for c in completions]
    return [sum(length > t for length in lengths) for t in range(max(lengths))]


def _teacher_forced(model: torch.nn.Module, c: Completion, temperature: float) -> torch.Tensor:
    """``log_softmax(logits / temperature)`` ``[tokens, vocab]`` the tokens of ``c`` were chosen
    from, by one forward pass over its unpadded prompt and completion."""
    ids = torch.tensor([c.prompt_token_ids + c.token_ids])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    return torch.log_softmax(logits[len(c.prompt_token_ids) - 1 : -1] / temperature, dim=-1)


def _assert_row(c: Completion, reference: torch.Tensor, params: SamplingParams, eos: Iterable[int]) -> None:
    """``c`` ends where its row did, and carries the log-probs of ``reference`` (:func:`_teacher_forced`)."""
    sampled = reference.gather(1, torch.tensor(c.token_ids)[:, None]).squeeze(1)
    torch.testing.assert_close(torch.tensor(c.logprobs), sampled, rtol=1e-4, atol=1e-5)
    if params.top_logprobs:
        top = torch.tensor(list(c.top_logprobs.logprobs)).reshape(len(c.token_ids), params.top_logprobs)
        torch.testing.assert_close(top, reference.topk(params.top_logprobs).values, rtol=1e-4, atol=1e-5)
    ends = [token in eos for token in c.token_ids]
    assert not any(ends[:-1])
    assert c.finish_reason == ("stop" if ends[-1] else "length")
    assert ends[-1] or len(c.token_ids) == params.max_new_tokens


def test_a_finished_row_is_in_no_later_forward_pass(engine: HFRolloutEngine) -> None:
    params = SamplingParams(n=4, max_new_tokens=24, temperature=1.0, seed=4)
    with _forward_rows(engine.model) as rows:
        out = _ending(engine).generate(CHATS, params)
    completions = [c for group in out for c in group]
    assert len({len(c.token_ids) for c in completions}) > 2  # the rows end at different steps
    assert rows == _needed_rows(completions)


@pytest.mark.parametrize(
    "knobs",
    [
        {"temperature": 1.0},
        {"temperature": 0.8, "top_k": 50, "top_p": 0.9, "min_p": 1e-5, "repetition_penalty": 1.3},
    ],
    ids=["plain", "truncated"],
)
def test_rows_that_outlive_others_keep_logprobs_reasons_and_counts(
    engine: HFRolloutEngine, knobs: dict[str, float]
) -> None:
    """Rows of two prompt lengths (one is left-padded), decoded by as many ``generate`` calls as
    there are steps at which a row ends: every completion is the one of its own prompt."""
    params = SamplingParams(n=4, max_new_tokens=24, seed=7, top_logprobs=3, **knobs)
    ending = _ending(engine)
    out = ending.generate(CHATS, params)
    assert len({len(c.token_ids) for group in out for c in group}) > 2
    eos = set(range(0, len(engine.tokenizer), EVERY))
    for chat, group in zip(CHATS, out, strict=True):
        assert len(group) == 4
        for c in group:
            assert c.prompt_token_ids == chat_prompt_ids(engine.tokenizer, chat)
            assert c.text == engine.tokenizer.decode(c.token_ids, skip_special_tokens=True)
            _assert_row(c, _teacher_forced(engine.model, c, params.temperature), params, eos)
    again = ending.generate(CHATS, params)  # the seed fixes the rows, whatever the calls they take
    assert [[(c.token_ids, c.logprobs) for c in g] for g in again] == [
        [(c.token_ids, c.logprobs) for c in g] for g in out
    ]


def test_rows_with_caps_of_their_own_end_there_and_leave_the_batch(engine: HFRolloutEngine) -> None:
    """One call, a generation cap per prompt (the turns of episodes with different room left in
    their context budget). The random model never ends a row itself here, so every row ends at the
    cap of its prompt, and from then on is in no forward pass."""
    caps = [3, 20, 9, 1]
    prompts = [CHATS[0], CHATS[1], CHATS[1], CHATS[0]]
    params = SamplingParams(n=2, max_new_tokens=20, temperature=1.0, seed=6, top_logprobs=3)
    with _forward_rows(engine.model) as rows:
        out = engine.generate(prompts, params, max_new_tokens=caps)
    assert [[len(c.token_ids) for c in group] for group in out] == [[cap] * 2 for cap in caps]
    assert rows == [8] + [6] * 2 + [4] * 6 + [2] * 11  # pass t: the rows with a token t + 1
    for chat, cap, group in zip(prompts, caps, out, strict=True):
        for c in group:
            assert c.finish_reason == "length"
            assert c.prompt_token_ids == chat_prompt_ids(engine.tokenizer, chat)
            assert c.text == engine.tokenizer.decode(c.token_ids, skip_special_tokens=True)
            assert len(c.top_logprobs) == cap
            capped = dataclasses.replace(params, max_new_tokens=cap)
            _assert_row(c, _teacher_forced(engine.model, c, 1.0), capped, engine._eos_ids)
    # the caps alone say where a row ends: they are those of the parameters when all are equal
    uniform = engine.generate(prompts[:2], dataclasses.replace(params, max_new_tokens=5))
    assert engine.generate(prompts[:2], params, max_new_tokens=[5, 5]) == uniform
    with pytest.raises(ValueError, match="2 max_new_tokens for 4 prompts"):
        engine.generate(prompts, params, max_new_tokens=[3, 4])
    with pytest.raises(ValueError, match=">= 1"):
        engine.generate(prompts[:2], params, max_new_tokens=[3, 0])


@pytest.mark.parametrize("greedy", [False, True], ids=["sampled", "greedy"])
def test_rows_end_at_their_cap_or_before_it(engine: HFRolloutEngine, greedy: bool) -> None:
    """Rows that end with EOS next to rows that end at caps of their own: each ends at whichever
    comes first, with the reason for it, and is computed no further."""
    caps = [2, 24, 6, 12]
    prompts = [CHATS[0], CHATS[1], CHATS[1], CHATS[0]]
    params = SamplingParams(n=1 if greedy else 3, max_new_tokens=24, seed=9, top_logprobs=2)
    eos: Iterable[int] = set(range(0, len(engine.tokenizer), EVERY))
    whole: list[Completion] = []
    if greedy:
        # The random model repeats a token for some steps. The first other token of the second row
        # ends a row: that one long before its cap, the third (the same prompt) at its cap.
        params.temperature, params.repetition_penalty = 0.0, 1.5
        whole = [group[0] for group in engine.generate(prompts, params)]
        ending_token = next(token for token in whole[1].token_ids if token != whole[1].token_ids[0])
        eos, caps[2] = {ending_token}, whole[1].token_ids.index(ending_token) + 1
    ending = _ending(engine, eos)
    with _forward_rows(engine.model) as rows:
        out = ending.generate(prompts, params, max_new_tokens=caps)
    completions = [c for group in out for c in group]
    assert rows == _needed_rows(completions)
    reasons = set()
    for cap, group in zip(caps, out, strict=True):
        for c in group:
            assert len(c.token_ids) <= cap
            capped = dataclasses.replace(params, max_new_tokens=cap)
            _assert_row(c, _teacher_forced(engine.model, c, params.temperature or 1.0), capped, eos)
            reasons.add((c.finish_reason, len(c.token_ids) == cap))
    assert ("length", True) in reasons and ("stop", False) in reasons
    if greedy:  # the tokens of the batch in which no row ends, as far as the row goes
        assert [c.token_ids for c in completions] == [
            c.token_ids[: len(row.token_ids)] for c, row in zip(whole, completions, strict=True)
        ]
        assert (completions[2].finish_reason, len(completions[2].token_ids)) == ("stop", caps[2])


def test_greedy_rows_are_those_of_a_batch_in_which_no_row_ends(engine: HFRolloutEngine) -> None:
    """Greedy decoding has no randomness to share: a row that goes on after others have ended gets
    the tokens it gets when none ends, up to its own end. The repetition penalty (without it the
    random model repeats one token) is applied to the whole sequence in every ``generate`` call."""
    prompts = [
        *CHATS,
        [{"role": "user", "content": "Name three colours."}],
        [{"role": "user", "content": "Hi."}],
    ]
    greedy = SamplingParams(n=1, max_new_tokens=12, temperature=0.0, repetition_penalty=1.5)
    whole = [group[0] for group in engine.generate(prompts, greedy)]
    assert all(len(c.token_ids) == 12 for c in whole)

    def until(token: int) -> list[list[int]]:
        """The rows of ``whole`` when ``token`` ends a row."""
        return [
            c.token_ids[: next((i + 1 for i, t in enumerate(c.token_ids) if t == token), 12)] for c in whole
        ]

    # a token that ends the rows at different steps, as the one a row turns to away from the others
    eos = next(
        token for c in whole for token in c.token_ids[1:] if len({len(row) for row in until(token)}) > 1
    )
    expected = until(eos)
    with _forward_rows(engine.model) as rows:
        out = [group[0] for group in _ending(engine, [eos]).generate(prompts, greedy)]
    assert [c.token_ids for c in out] == expected
    assert [c.finish_reason for c in out] == ["stop" if row[-1] == eos else "length" for row in expected]
    assert rows == _needed_rows(out)
    for c, before in zip(out, whole, strict=True):
        torch.testing.assert_close(
            torch.tensor(c.logprobs), torch.tensor(before.logprobs[: len(c.logprobs)]), rtol=1e-4, atol=1e-5
        )


def test_stop_string_ends_a_row_while_the_others_go_on(engine: HFRolloutEngine) -> None:
    sampled = SamplingParams(n=1, max_new_tokens=10, temperature=1.0, seed=21)
    prompts = [*CHATS, [{"role": "user", "content": "Name three colours."}]]
    whole = [group[0] for group in engine.generate(prompts, sampled)]
    assert all(c.finish_reason == "length" for c in whole)
    decode = engine.tokenizer.decode
    # the text of one token of the middle row, which no other row and no earlier token of it has
    at, stop_text = next(
        (k, text)
        for k in range(1, 9)
        if len((text := decode(whole[1].token_ids[k : k + 1])).strip()) >= 3
        and text not in decode(whole[1].token_ids[:k])
        and all(text not in c.text for c in (whole[0], whole[2]))
    )
    params = SamplingParams(n=1, max_new_tokens=10, temperature=1.0, seed=21, stop=[stop_text])
    with _forward_rows(engine.model) as rows:
        out = [group[0] for group in engine.generate(prompts, params)]
    assert out[1].finish_reason == "stop" and out[1].token_ids == whole[1].token_ids[: at + 1]
    assert out[1].text.endswith(stop_text)  # the stop string stays in the text
    assert rows == _needed_rows(out)
    for c, before in zip(out, whole, strict=True):
        # the same random numbers up to the step at which the batch lost a row, others after it
        assert c.token_ids[: at + 1] == before.token_ids[: at + 1]
        reference = _teacher_forced(engine.model, c, 1.0)
        sampled_logprobs = reference.gather(1, torch.tensor(c.token_ids)[:, None]).squeeze(1)
        torch.testing.assert_close(torch.tensor(c.logprobs), sampled_logprobs, rtol=1e-4, atol=1e-5)
    for c in (out[0], out[2]):
        assert c.finish_reason == ("stop" if stop_text in c.text else "length")
        assert c.finish_reason == "stop" or len(c.token_ids) == 10


def test_tokens_keep_their_distribution_when_rows_leave_the_batch(engine: HFRolloutEngine) -> None:
    """Top-2 sampling of three tokens after a two-token prompt has at most five outcomes, whose
    probabilities a few forward passes give exactly. The likelier first token ends the row, so
    about half of the rows leave the batch after one step, and the second and third token of the
    others are drawn in other ``generate`` calls than the first, next to the rows of another prompt."""
    model, temperature, n = engine.model, 1.0, 120
    short, other = [11, 23], [7, 31, 5]

    def choices(prefix: list[int]) -> dict[int, float]:
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([prefix])).logits[0, -1].float()
        probabilities, tokens = torch.softmax(logits / temperature, dim=-1).topk(2)
        return dict(zip(tokens.tolist(), (probabilities / probabilities.sum()).tolist(), strict=True))

    first = choices(short)
    eos = max(first, key=first.__getitem__)
    expected: dict[tuple[int, ...], float] = {}

    def walk(tokens: tuple[int, ...], probability: float) -> None:
        if len(tokens) == 3 or tokens[-1:] == (eos,):
            expected[tokens] = probability
            return
        for token, p in choices(short + list(tokens)).items():
            walk((*tokens, token), probability * p)

    walk((), 1.0)
    params = SamplingParams(n=n, max_new_tokens=3, temperature=temperature, top_k=2, seed=12)
    with _forward_rows(model) as rows:
        out = _ending(engine, [eos]).generate([short, other], params)
    assert rows[0] == 2 * n and rows[1] < rows[0]  # the rows that ended are not in the second pass
    counts = Counter(tuple(c.token_ids) for c in out[0])
    assert set(counts) <= set(expected)
    chi2 = sum((counts[outcome] - n * p) ** 2 / (n * p) for outcome, p in expected.items())
    assert chi2 < 30.0  # 4 degrees of freedom at most: one draw in 100 000 is beyond 28.5


def test_a_sliding_window_cache_gives_up_rows(tmp_path: Path) -> None:
    """Mistral and Gemma keep the last ``sliding_window`` positions per row in their cache: rows
    are taken out of it as out of the plain one."""
    from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(TINY_TOKENIZER)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(0)
    model = MistralForCausalLM(
        MistralConfig(
            vocab_size=len(tokenizer),
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            sliding_window=6,  # shorter than the prompts: every step is past the window
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    ).eval()
    eos = range(0, len(tokenizer), EVERY)
    windowed = TrainerPolicyEngine(model, tokenizer, device=torch.device("cpu"), eos_token_ids=eos)
    params = SamplingParams(n=4, max_new_tokens=24, temperature=1.0, seed=4, top_logprobs=3)
    with _forward_rows(model) as rows:
        out = windowed.generate(CHATS, params)
    completions = [c for group in out for c in group]
    assert len({len(c.token_ids) for c in completions}) > 2
    assert rows == _needed_rows(completions)
    for c in completions:
        _assert_row(c, _teacher_forced(model, c, 1.0), params, set(eos))


def test_rows_of_a_peft_wrapped_policy_leave_the_batch(tiny_config: RunConfig) -> None:
    """The TRL adapter samples from the trainer's LoRA-wrapped model, whose ``generate`` hands the
    cache of the call before on to the transformers model inside it."""
    from peft import LoraConfig, get_peft_model

    base = HFRolloutEngine(tiny_config)  # a model of its own: the adapter is put into its layers
    try:
        lora = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
        wrapped = get_peft_model(base.model, lora).eval()
        eos = range(0, len(base.tokenizer), EVERY)
        policy = TrainerPolicyEngine(wrapped, base.tokenizer, device=base.device, eos_token_ids=eos)
        params = SamplingParams(n=4, max_new_tokens=24, temperature=1.0, seed=4, top_logprobs=3)
        with _forward_rows(base.model) as rows:
            out = policy.generate(CHATS, params)
        completions = [c for group in out for c in group]
        assert len({len(c.token_ids) for c in completions}) > 2
        assert rows == _needed_rows(completions)
        for c in completions:
            _assert_row(c, _teacher_forced(wrapped, c, 1.0), params, set(eos))
    finally:
        base.shutdown()


def test_a_cache_that_cannot_give_up_rows_is_decoded_in_one_call(
    engine: HFRolloutEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint's ``generation_config.json`` can ask for a static cache, whose rows have fixed
    places: the rows that end early are computed to the end, and the completions are right."""
    monkeypatch.setattr(engine.model.generation_config, "cache_implementation", "static")
    params = SamplingParams(n=4, max_new_tokens=24, temperature=1.0, seed=4, top_logprobs=3)
    out = _ending(engine).generate(CHATS, params)
    completions = [c for group in out for c in group]
    assert len({len(c.token_ids) for c in completions}) > 2
    eos = set(range(0, len(engine.tokenizer), EVERY))
    for c in completions:
        _assert_row(c, _teacher_forced(engine.model, c, 1.0), params, eos)


def test_rows_of_a_vlm_end_at_different_steps(
    vlm_engine: HFRolloutEngine, test_images: dict[str, Path]
) -> None:
    """Qwen2-VL keeps the offsets of its position ids per row from one decoding step to the next,
    so its batches are decoded in one ``generate`` call. Image and text rows that end early leave
    the others their log-probs."""
    prompts = [
        image_chat(test_images["red"]),
        image_chat(test_images["blue"], "Describe it."),
        [{"role": "user", "content": "Say hello."}],
    ]
    params = SamplingParams(n=3, max_new_tokens=16, temperature=1.0, seed=2)
    ending = _ending(vlm_engine)
    assert ending.processor is not None
    out = ending.generate(prompts, params)
    assert len({len(c.token_ids) for group in out for c in group}) > 2
    eos = set(range(0, len(vlm_engine.tokenizer), EVERY))
    for chat, group in zip(prompts, out, strict=True):
        for c in group:
            _assert_row(c, _vlm_teacher_forced(vlm_engine, chat, c), params, eos)
    # ... and so do rows that end at a cap of their own, which stay in the batch like the others
    caps = [2, 16, 5]
    out = vlm_engine.generate(prompts, params, max_new_tokens=caps)
    for chat, cap, group in zip(prompts, caps, out, strict=True):
        for c in group:
            assert len(c.token_ids) == cap and c.finish_reason == "length"
            capped = dataclasses.replace(params, max_new_tokens=cap)
            _assert_row(c, _vlm_teacher_forced(vlm_engine, chat, c), capped, vlm_engine._eos_ids)


def _vlm_teacher_forced(engine: HFRolloutEngine, chat: list[dict], c: Completion) -> torch.Tensor:
    """:func:`_teacher_forced` for a chat that may hold images (temperature 1)."""
    if isinstance(chat[0]["content"], str):
        return _teacher_forced(engine.model, c, 1.0)
    enc = vlm_encode(engine.processor, chat)
    completion = torch.tensor([c.token_ids])
    full = torch.cat([enc["input_ids"], completion], dim=1)
    media = {k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask", "mm_token_type_ids")}
    with torch.no_grad():
        logits = (
            engine.model(
                input_ids=full,
                attention_mask=torch.ones_like(full),
                mm_token_type_ids=torch.cat([enc["mm_token_type_ids"], torch.zeros_like(completion)], dim=1),
                **media,
            )
            .logits[0]
            .float()
        )
    return torch.log_softmax(logits[enc["input_ids"].shape[1] - 1 : -1], dim=-1)
