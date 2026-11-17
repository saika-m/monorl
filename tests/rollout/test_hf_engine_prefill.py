"""Batches of the HF engine in which rows share a prompt (the ``n`` samples of a prompt, the equal
prompts of episodes that were batched together): the prompt is computed once, and every row comes
back as it does when each row computes its prompt itself."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import pytest
import torch

from monorl.adapters.trl.rollout import TrainerPolicyEngine
from monorl.config.schema import RunConfig
from monorl.core.errors import GenerationCancelled
from monorl.core.types import Completion, SamplingParams
from monorl.rollout.hf_engine import HFRolloutEngine
from tests.conftest import TINY_TOKENIZER

LONG = [11, 12, 13, 14] * 40  # 160 tokens
OTHER = [17, 18, 19] * 45  # 135 tokens: left-padded next to LONG
EVERY = 8


@contextlib.contextmanager
def _forward_shapes(model: torch.nn.Module) -> Iterator[list[tuple[int, int]]]:
    """``(rows, tokens)`` of every forward pass of ``model`` inside the block."""
    shapes: list[tuple[int, int]] = []

    def note(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        given = kwargs.get("input_ids")
        if given is None:  # an adapter of virtual tokens hands on embeddings
            given = kwargs["inputs_embeds"] if kwargs.get("inputs_embeds") is not None else args[0]
        shapes.append((given.shape[0], given.shape[1]))

    hook = model.register_forward_pre_hook(note, with_kwargs=True)
    try:
        yield shapes
    finally:
        hook.remove()


def _unshared(
    engine: HFRolloutEngine, prompts: list[list[int]], params: SamplingParams
) -> list[list[Completion]]:
    """What ``engine`` generates when every row computes its prompt itself."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(engine, "_shared_prompt_cache", lambda *args: None)
        return engine.generate(prompts, params)


def _teacher_forced(model: torch.nn.Module, c: Completion, temperature: float) -> torch.Tensor:
    """``log_softmax(logits / temperature)`` ``[tokens, vocab]`` the tokens of ``c`` were chosen
    from, by one forward pass over its unpadded prompt and completion (counted from the end: an
    adapter of virtual tokens puts positions in front)."""
    ids = torch.tensor([c.prompt_token_ids + c.token_ids])
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()
    return torch.log_softmax(logits[-len(c.token_ids) - 1 : -1] / temperature, dim=-1)


def _assert_logprobs(c: Completion, reference: torch.Tensor, top: int) -> None:
    sampled = reference.gather(1, torch.tensor(c.token_ids)[:, None]).squeeze(1)
    torch.testing.assert_close(torch.tensor(c.logprobs), sampled, rtol=1e-4, atol=1e-5)
    if top:
        logprobs = torch.tensor(list(c.top_logprobs.logprobs)).reshape(len(c.token_ids), top)
        torch.testing.assert_close(logprobs, reference.topk(top).values, rtol=1e-4, atol=1e-5)


def _assert_same_rows(shared: list[list[Completion]], plain: list[list[Completion]]) -> None:
    for shared_group, plain_group in zip(shared, plain, strict=True):
        for a, b in zip(shared_group, plain_group, strict=True):
            assert (a.token_ids, a.finish_reason, a.text) == (b.token_ids, b.finish_reason, b.text)
            torch.testing.assert_close(
                torch.tensor(a.logprobs), torch.tensor(b.logprobs), rtol=1e-4, atol=1e-5
            )
            if a.top_logprobs is not None:
                assert list(a.top_logprobs.token_ids) == list(b.top_logprobs.token_ids)
                torch.testing.assert_close(
                    torch.tensor(list(a.top_logprobs.logprobs)),
                    torch.tensor(list(b.top_logprobs.logprobs)),
                    rtol=1e-4,
                    atol=1e-5,
                )


def test_a_prompt_that_rows_share_is_computed_once(engine: HFRolloutEngine) -> None:
    """Three samples each of three prompts, two of them equal: the first forward pass is over the
    two distinct prompts without their last token, the second over that token for all nine rows.
    Without the sharing the nine rows compute 160 positions each."""
    prompts = [LONG, OTHER, LONG]
    params = SamplingParams(n=3, max_new_tokens=4, temperature=0.8, seed=41)
    with _forward_shapes(engine.model) as shapes:
        engine.generate(prompts, params)
    assert shapes == [(2, 159), (9, 1), (9, 1), (9, 1), (9, 1)]
    with _forward_shapes(engine.model) as plain:
        _unshared(engine, prompts, params)
    assert plain == [(9, 160), (9, 1), (9, 1), (9, 1)]


@pytest.mark.parametrize(
    ("prompts", "distinct", "knobs"),
    [
        ([LONG, OTHER], 2, {"n": 3, "temperature": 0.8}),
        ([LONG, OTHER, LONG, LONG, OTHER], 2, {"n": 1, "temperature": 1.0}),
        ([LONG, [21], LONG], 2, {"n": 3, "temperature": 0.8}),
        (
            [LONG, OTHER, LONG],
            2,
            {"n": 2, "temperature": 0.8, "top_p": 0.9, "top_k": 31, "min_p": 1e-5, "repetition_penalty": 1.1},
        ),
        ([OTHER, LONG, OTHER, LONG, OTHER], 2, {"n": 4, "temperature": 0.0}),
    ],
    ids=["samples", "episodes", "one-token prompt", "truncated", "greedy"],
)
def test_rows_that_share_a_prompt_are_those_of_rows_that_compute_it_themselves(
    engine: HFRolloutEngine, prompts: list[list[int]], distinct: int, knobs: dict[str, Any]
) -> None:
    """The rows of a prompt need not be adjacent, the prompts have different lengths (the shorter
    are left-padded), and one has nothing before its last token."""
    params = SamplingParams(max_new_tokens=6, seed=41, top_logprobs=5, **knobs)
    with _forward_shapes(engine.model) as shapes:
        shared = engine.generate(prompts, params)
    assert shapes[0] == (distinct, 159)  # the batch was shared
    _assert_same_rows(shared, _unshared(engine, prompts, params))
    temperature = params.temperature or 1.0
    for ids, group in zip(prompts, shared, strict=True):
        assert len(group) == params.n
        for c in group:
            assert c.prompt_token_ids == ids
            _assert_logprobs(c, _teacher_forced(engine.model, c, temperature), params.top_logprobs)
    assert [[c.token_ids for c in g] for g in engine.generate(prompts, params)] == [
        [c.token_ids for c in g] for g in shared
    ]


def test_rows_that_share_a_prompt_leave_the_batch_when_they_end(engine: HFRolloutEngine) -> None:
    """The cache of the shared prompts is one whose rows can be taken out, so the first call
    already ends when a row does: after the prompts, a forward pass has the rows that still need
    a token. The engine is the TRL adapter's, built around a model without a run config."""
    eos = set(
        range(0, len(engine.tokenizer), EVERY)
    )  # the model is random: a row ends after 8 tokens on average
    ending = TrainerPolicyEngine(engine.model, engine.tokenizer, device=engine.device, eos_token_ids=eos)
    params = SamplingParams(n=4, max_new_tokens=24, temperature=1.0, seed=4, top_logprobs=3)
    with _forward_shapes(engine.model) as shapes:
        out = ending.generate([LONG, OTHER], params)
    completions = [c for group in out for c in group]
    lengths = [len(c.token_ids) for c in completions]
    assert len(set(lengths)) > 2  # the rows end at different steps
    assert shapes[0] == (2, 159)
    assert [rows for rows, _ in shapes[1:]] == [
        sum(length > t for length in lengths) for t in range(max(lengths))
    ]
    for c in completions:
        _assert_logprobs(c, _teacher_forced(engine.model, c, 1.0), 3)
        ends = [token in eos for token in c.token_ids]
        assert not any(ends[:-1]) and c.finish_reason == ("stop" if ends[-1] else "length")


def test_a_shared_prompt_is_computed_with_the_weights_of_its_call(engine: HFRolloutEngine) -> None:
    prompts = [LONG, LONG]
    params = SamplingParams(n=2, max_new_tokens=4, temperature=0.8, seed=7)
    before = engine.generate(prompts, params)
    name = "model.layers.0.mlp.down_proj.weight"
    original = engine.model.state_dict()[name].clone()
    try:
        engine.update_weights([(name, original + 0.5)])
        with _forward_shapes(engine.model) as shapes:
            after = engine.generate(prompts, params)
        assert shapes[0] == (1, 159)
        _assert_same_rows(after, _unshared(engine, prompts, params))
        assert [c.logprobs for g in after for c in g] != [c.logprobs for g in before for c in g]
    finally:
        engine.update_weights([(name, original)])


def test_cancel_while_the_shared_prompts_are_computed(engine: HFRolloutEngine) -> None:
    """The forward pass over the prompts cannot be interrupted; the call ends when it returns."""
    prompts = [LONG, LONG]
    params = SamplingParams(n=2, max_new_tokens=4, temperature=0.8, seed=7)
    expected = engine.generate(prompts, params)
    hook = engine.model.register_forward_pre_hook(lambda module, args: engine.cancel())
    try:
        with _forward_shapes(engine.model) as shapes, pytest.raises(GenerationCancelled):
            engine.generate(prompts, params)
    finally:
        hook.remove()
    assert shapes == [(1, 159)]  # nothing was generated
    assert [c.token_ids for g in engine.generate(prompts, params) for c in g] == [
        c.token_ids for g in expected for c in g
    ]


@pytest.mark.parametrize(
    "case",
    ["no repeats", "little to save", "training mode", "static cache", "no cache object", "remote code"],
)
def test_batches_that_every_row_computes_itself(
    engine: HFRolloutEngine, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """No prompt occurs twice; the copies would save fewer than ``shared_prefill_min_positions``
    positions (3 rows of 39); dropout would be drawn once per prompt; the checkpoint asks for a
    cache with fixed places for its rows; the model class takes no cache object from ``generate``;
    the model's code came with the checkpoint (``trust_remote_code``)."""
    prompts, n = [LONG, OTHER, LONG], 2
    if case == "no repeats":
        prompts, n = [LONG, OTHER], 1
    elif case == "little to save":
        prompts = [LONG[:40]] * 2
    elif case == "training mode":
        monkeypatch.setattr(engine.model, "training", True)
    elif case == "static cache":
        monkeypatch.setattr(engine.model.generation_config, "cache_implementation", "static")
    elif case == "no cache object":
        monkeypatch.setattr(
            type(engine.model), "_supports_default_dynamic_cache", classmethod(lambda cls: False)
        )
    else:
        monkeypatch.setattr(type(engine.model), "__module__", "transformers_modules.org.repo.modeling")
    with _forward_shapes(engine.model) as shapes:
        out = engine.generate(prompts, SamplingParams(n=n, max_new_tokens=3, temperature=1.0, seed=3))
    assert shapes[0] == (len(prompts) * n, max(len(p) for p in prompts))
    assert [len(group) for group in out] == [n] * len(prompts)


def _tiny_model(kind: str, tokenizer: Any) -> torch.nn.Module:
    """A random model of two to four layers, in evaluation mode, over the tiny tokenizer's vocabulary."""
    import transformers
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    tokens = {
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    llama = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
    }
    torch.manual_seed(0)
    if kind == "bloom":  # ALiBi: forward takes no position_ids
        config = transformers.BloomConfig(hidden_size=64, n_layer=2, n_head=4, **tokens)
        model = transformers.BloomForCausalLM(config)
    elif kind == "gpt2":  # learned absolute positions
        config = transformers.GPT2Config(n_embd=64, n_layer=2, n_head=4, n_positions=512, **tokens)
        model = transformers.GPT2LMHeadModel(config)
    elif kind == "mamba":  # a state per row instead of keys and values
        config = transformers.MambaConfig(hidden_size=64, num_hidden_layers=2, state_size=8, **tokens)
        model = transformers.MambaForCausalLM(config)
    elif kind == "sliding window":  # a cache of the last positions only
        config = transformers.MistralConfig(sliding_window=6, **llama, **tokens)
        model = transformers.MistralForCausalLM(config)
    elif kind == "dynamic rope":  # frequencies that depend on the longest position of a forward pass
        rope = {"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000.0}
        model = transformers.LlamaForCausalLM(
            transformers.LlamaConfig(rope_parameters=rope, **llama, **tokens)
        )
    elif kind == "no logits_to_keep":  # a forward pass that cannot be asked for the last logits only

        class EveryLogit(transformers.LlamaForCausalLM):
            def forward(  # type: ignore[override]
                self,
                input_ids: Any = None,
                attention_mask: Any = None,
                position_ids: Any = None,
                past_key_values: Any = None,
                use_cache: Any = None,
                return_dict: Any = None,
            ) -> Any:
                return super().forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=return_dict,
                )

        model = EveryLogit(transformers.LlamaConfig(**llama, **tokens))
    elif kind == "prompt tuning":  # virtual tokens in front of every prompt
        from peft import PromptTuningConfig, TaskType, get_peft_model

        base = transformers.LlamaForCausalLM(transformers.LlamaConfig(**llama, **tokens))
        model = get_peft_model(base, PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=4))
    else:
        # Qwen3.5 mixes linear attention, which keeps a convolution state and a recurrent state
        # per row, with full attention, and names its layers in the text config of a composite one.
        text = {
            **llama,
            "num_hidden_layers": 4,
            "head_dim": 16,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            **tokens,
        }
        vision = {
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
        }
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision))
        assert getattr(model.config, "layer_types", None) is None  # they are the text config's
        assert "linear_attention" in model.config.get_text_config(decoder=True).layer_types
    return model.eval()


@pytest.mark.parametrize(
    ("kind", "shared"),
    [
        ("bloom", True),
        ("gpt2", True),
        ("mamba", False),
        ("sliding window", False),
        ("dynamic rope", False),
        ("no logits_to_keep", False),
        ("prompt tuning", False),
        ("qwen3.5", False),
    ],
)
def test_models_that_do_and_do_not_share_prompts(kind: str, shared: bool) -> None:
    """Prompts are shared where a forward pass over a prompt leaves keys and values that
    ``generate`` can go on from; every other model decodes its batch as before. Either way the
    log-probs are those of one forward pass over the row's own prompt and completion."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TINY_TOKENIZER)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _tiny_model(kind, tokenizer)
    policy = TrainerPolicyEngine(
        model, tokenizer, device=torch.device("cpu"), eos_token_ids=[tokenizer.eos_token_id]
    )
    params = SamplingParams(n=3, max_new_tokens=4, temperature=0.8, seed=41, top_logprobs=3)
    with _forward_shapes(policy._generating_model()) as shapes:
        out = policy.generate([LONG, OTHER, [21]], params)
    if shared:
        assert shapes[:2] == [(3, 159), (9, 1)]
    else:
        assert shapes[0][0] == 9  # every row computes its prompt
    assert [len(group) for group in out] == [3, 3, 3]
    # The virtual tokens of an adapter come before the padding of a shorter row, which moves its
    # tokens away from them: only the longest prompt is the one a forward pass of its own sees.
    for group in out[:1] if kind == "prompt tuning" else out:
        for c in group:
            _assert_logprobs(c, _teacher_forced(model, c, 0.8), 3)


def test_a_checkpoint_with_a_processor_computes_every_row(vlm_engine: HFRolloutEngine) -> None:
    """Qwen2-VL numbers its positions itself, from offsets it keeps per row: text prompts of such
    a checkpoint are decoded as before."""
    assert vlm_engine.processor is not None
    with _forward_shapes(vlm_engine.model) as shapes:
        out = vlm_engine.generate(
            [LONG, OTHER], SamplingParams(n=3, max_new_tokens=3, temperature=1.0, seed=3)
        )
    assert shapes[0] == (6, 160)
    assert [len(group) for group in out] == [3, 3]


def test_a_peft_wrapped_policy_shares_prompts(tiny_config: RunConfig) -> None:
    """The TRL adapter samples from the trainer's LoRA-wrapped model: the prompts are computed
    through the wrapper, with the adapter."""
    from peft import LoraConfig, get_peft_model

    base = HFRolloutEngine(tiny_config)  # a model of its own: the adapter is put into its layers
    try:
        lora = LoraConfig(r=4, target_modules=["q_proj", "v_proj"], init_lora_weights=False)
        wrapped = get_peft_model(base.model, lora).eval()
        policy = TrainerPolicyEngine(
            wrapped, base.tokenizer, device=base.device, eos_token_ids=[base.tokenizer.eos_token_id]
        )
        params = SamplingParams(n=3, max_new_tokens=5, temperature=1.0, seed=3, top_logprobs=2)
        with _forward_shapes(base.model) as shapes:
            out = policy.generate([LONG, OTHER], params)
        assert shapes[0] == (2, 159)
        for group in out:
            for c in group:
                _assert_logprobs(c, _teacher_forced(wrapped, c, 1.0), 2)
    finally:
        base.shutdown()
