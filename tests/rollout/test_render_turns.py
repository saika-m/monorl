"""Whole conversations through ``continue_prompt_ids``, one call per turn as the episode runner and
the gateway make them, with the prompt of every turn checked against what the module promises: the
previous prompt and the completion token for token, then the template's own text for what follows
(which is the full re-render whenever the sampled text round-trips), or the full re-render where
the template has to be followed.

The same conversations are then run from several threads on shared tokenizers: what the module
remembers per tokenizer must not leak between conversations, templates or tokenizers.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from transformers import AutoTokenizer

from monorl.core.types import Completion
from monorl.rollout.render import Continuation, chat_prompt_ids, continue_prompt_ids
from tests.orchestrator.fakes import FakeTokenizer
from tests.rollout.test_render import CALL_ONLY, REASONING_DROPPING, TOOL_CALLING

END = "<|im_end|>"
USER = [{"role": "user", "content": "What is 2+2?"}]

# Turns end with another special token than ChatML's.
OTHER_END = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|endoftext|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)

# Turns end with plain text: there is no end-of-turn token a completion could have produced.
NO_END_TOKEN = (
    "{%- for m in messages -%}{{- m['role'] + ': ' + m['content'] + '\n' -}}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- 'assistant: ' -}}{%- endif -%}"
)

# Every turn is closed with text of its own (its position in the chat).
NUMBERED_END = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + (loop.index0 | string) + '\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)

REJECTS_BOOM = (
    "{%- for m in messages -%}"
    "{%- if m['content'] == 'boom' -%}{{- raise_exception('no boom in this template') -}}{%- endif -%}"
    "{{- '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)


@dataclass
class Step:
    """One assistant turn and what follows it."""

    text: str
    """The completion's text; its ids are the tokenizer's encoding of it unless ``sampled`` is given."""
    new: list[dict[str, Any]]
    message: dict[str, Any] | None = None
    """The turn as a chat message, when that is not plain ``{"role": "assistant", "content": text}``."""
    finish: str = "stop"
    """``"stop"``: the completion ends with the end-of-turn token; ``"length"``: it was cut before it."""
    end: str = END


@dataclass
class Prompt:
    """What one call returned, with what it was called on."""

    out: Continuation
    context: list[int]
    """The previous prompt followed by the completion as sampled."""
    messages: list[dict[str, Any]]
    """The chat the next turn is prompted with."""
    step: Step


def _load(tiny_model_dir: Path, template: str | None = None) -> Any:
    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    if template is not None:
        tok.chat_template = template
    return tok


@pytest.fixture
def tok(tiny_model_dir: Path) -> Any:
    return _load(tiny_model_dir)


def _ids(tok: Any, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=False)["input_ids"])


def _drive(tok: Any, steps: list[Step], *, mode: str = "tokens") -> list[Prompt]:
    """A conversation, turn by turn: each call gets the prompt the call before it returned."""
    messages = list(USER)
    prompt = chat_prompt_ids(tok, messages)
    prompts = []
    for step in steps:
        sampled = _ids(tok, step.text)
        if step.finish == "stop":
            sampled += _ids(tok, step.end)
        out = continue_prompt_ids(
            tok,
            previous_messages=messages,
            previous_prompt_ids=prompt,
            completion=Completion(text=step.text, token_ids=sampled, finish_reason=step.finish),  # type: ignore[arg-type]
            new_messages=step.new,
            assistant_message=step.message,
            mode=mode,
        )
        messages = [*messages, step.message or {"role": "assistant", "content": step.text}, *step.new]
        prompts.append(Prompt(out=out, context=[*prompt, *sampled], messages=messages, step=step))
        prompt = out.ids
    return prompts


def _chatml(messages: list[dict[str, Any]]) -> str:
    """What the ChatML-shaped templates of these tests render for messages after an assistant turn."""
    return (
        "".join(f"<|im_start|>{m['role']}\n{m['content']}{END}\n" for m in messages)
        + "<|im_start|>assistant\n"
    )


def assert_keeps_the_sampled_tokens(tok: Any, prompt: Prompt) -> None:
    """The token-stream identity: previous prompt, completion as sampled, the end of the turn unless
    the completion produced it, the new messages, the generation prompt."""
    assert prompt.out.bridged
    assert prompt.out.ids[: len(prompt.context)] == prompt.context
    closing = "\n" if prompt.step.finish == "stop" else END + "\n"
    assert tok.decode(prompt.out.ids[len(prompt.context) :]) == closing + _chatml(prompt.step.new)


def assert_is_the_full_render(tok: Any, prompt: Prompt) -> None:
    assert prompt.out.ids == chat_prompt_ids(tok, prompt.messages)


def _user(text: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": text}]


def _tool(text: str) -> list[dict[str, Any]]:
    return [{"role": "tool", "content": text}]


def _call(expression: str) -> dict[str, Any]:
    """A turn that is one structured tool call and nothing else, as the gateway hands it back."""
    function = {"name": "calc", "arguments": {"expression": expression}}
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": function}]}


def _sampled_call(expression: str) -> str:
    """The call as the policy wrote it: compact JSON, which no template re-serialises this way."""
    return '{"name":"calc","arguments":{"expression":"' + expression + '"}}'


def plain_steps(salt: int = 0) -> list[Step]:
    """Answers that round-trip through text, every other one cut off by the length limit."""
    return [
        Step(
            f"The answer to question {salt}.{i} is {i * 7 + salt}.",
            _user(f"And {i}+{salt}?"),
            finish="stop" if i % 2 == 0 else "length",
        )
        for i in range(6)
    ]


def tool_steps(salt: int = 0) -> list[Step]:
    """A tool loop under ``TOOL_CALLING``: structured calls, tool results, answers in between."""
    steps = []
    for i in range(3):
        call = f"{i}+{salt}"
        text = "<tool_call>\n" + _sampled_call(call) + "\n</tool_call>"
        steps.append(Step(text, _tool(str(i + salt)), message=_call(call)))
        steps.append(
            Step(f"That makes {i + salt}.", _user(f"Now add {i}."), finish="stop" if i else "length")
        )
    return steps


def test_plain_turns_continue_into_exactly_what_the_template_renders(tok: Any) -> None:
    prompts = _drive(tok, plain_steps())
    assert {p.step.finish for p in prompts} == {"stop", "length"}
    for prompt in prompts:
        assert_keeps_the_sampled_tokens(tok, prompt)
        assert_is_the_full_render(tok, prompt)


def test_a_tool_loop_keeps_every_call_as_it_was_sampled(tok: Any) -> None:
    tok.chat_template = TOOL_CALLING
    prompts = _drive(tok, tool_steps())
    for prompt in prompts:
        assert_keeps_the_sampled_tokens(tok, prompt)
    # ... which re-rendering would not: the template serialises the calls with other spacing
    final = prompts[-1]
    assert chat_prompt_ids(tok, final.messages) != final.out.ids
    assert _sampled_call("2+0") in tok.decode(final.out.ids)
    assert _sampled_call("2+0") not in tok.decode(chat_prompt_ids(tok, final.messages))


def test_turns_rendered_from_their_tool_calls_alone(tok: Any) -> None:
    """Calls without other content continue the stream; a call with text next to it, which this
    template drops, is followed with a full render, and the turns after it continue from that."""
    tok.chat_template = CALL_ONLY
    wordy = {**_call("3+3"), "content": "Let me compute."}
    steps = [
        Step(_sampled_call("1+1"), _tool("2"), message=_call("1+1")),
        Step(_sampled_call("2+2"), _tool("4"), message=_call("2+2")),
        Step("Let me compute.", _tool("6"), message=wordy),
        Step(_sampled_call("4+4"), _tool("8"), message=_call("4+4")),
        Step("All done: 2, 4, 6 and 8.", _user("Thanks. And 5+5?"), finish="length"),
    ]
    prompts = _drive(tok, steps)
    assert [p.out.bridged for p in prompts] == [True, True, False, True, True]
    for prompt in prompts:
        if prompt.out.bridged:
            assert_keeps_the_sampled_tokens(tok, prompt)
        else:
            assert_is_the_full_render(tok, prompt)
    assert "Let me compute." not in tok.decode(prompts[-1].out.ids)


def test_a_template_that_rewrites_history_is_followed_turn_by_turn(tok: Any) -> None:
    """Reasoning is dropped once the user speaks again, whether it is this turn's or, after a tool
    loop that kept it, an earlier turn's: those prompts are full renders, and the conversation
    continues token for token from each of them."""
    tok.chat_template = REASONING_DROPPING
    steps = [
        Step("<think>\n2+2 is 4\n</think>\n\nThe answer is 4.", _user("And 3+3? Use the tool.")),
        Step("<think>\nuse the tool\n</think>\n\ncalc(3+3)", _tool("6")),
        Step("<think>\nthe tool says 6\n</think>\n\nThe answer is 6.", _user("And 4+4?")),
        Step("The answer is 8.", _user("And 5+5?")),
        Step("The answer is 10.", _user("Thanks."), finish="length"),
    ]
    prompts = _drive(tok, steps)
    assert [p.out.bridged for p in prompts] == [False, True, False, True, True]
    assert_is_the_full_render(tok, prompts[0])
    assert "2+2 is 4" not in tok.decode(prompts[0].out.ids)
    assert_keeps_the_sampled_tokens(tok, prompts[1])
    assert "use the tool" in tok.decode(prompts[1].out.ids)
    assert_is_the_full_render(tok, prompts[2])
    assert "use the tool" not in tok.decode(prompts[2].out.ids)
    for prompt in prompts[3:]:
        assert_keeps_the_sampled_tokens(tok, prompt)
        assert_is_the_full_render(tok, prompt)


def test_template_mode_renders_every_turn_again(tok: Any) -> None:
    tok.chat_template = TOOL_CALLING
    for prompt in _drive(tok, [*tool_steps(), *plain_steps()], mode="template"):
        assert not prompt.out.bridged
        assert_is_the_full_render(tok, prompt)


def test_the_end_of_turn_follows_the_template_in_use(tok: Any) -> None:
    """One tokenizer object under three templates in turn: what closes a turn, and whether the
    completion already produced it, is decided by the template the turn is rendered with."""
    other = "<|endoftext|>"
    default = tok.chat_template
    answer = "The answer is 4."
    follow_up = _user("And 3+3?")
    rendered = [*USER, {"role": "assistant", "content": answer}, *follow_up]

    (chatml,) = _drive(tok, [Step(answer, follow_up)])
    assert chatml.out.bridged and chatml.out.ids == chat_prompt_ids(tok, rendered)

    tok.chat_template = OTHER_END
    (ended,) = _drive(tok, [Step(answer, follow_up, end=other)])
    assert ended.out.bridged and ended.out.ids == chat_prompt_ids(tok, rendered)
    # ChatML's end token closes nothing here: the turn still gets this template's end
    (foreign,) = _drive(tok, [Step(answer, follow_up)])
    assert foreign.out.bridged and foreign.out.ids[: len(foreign.context)] == foreign.context
    assert tok.decode(foreign.out.ids[len(foreign.context) :]).startswith(other + "\n<|im_start|>user\n")

    tok.chat_template = NO_END_TOKEN
    (texty,) = _drive(tok, [Step(answer, follow_up, finish="length")])
    assert texty.out.bridged and texty.out.ids[: len(texty.context)] == texty.context
    assert tok.decode(texty.out.ids[len(texty.context) :]) == "\nuser: And 3+3?\nassistant: "

    tok.chat_template = default
    (again,) = _drive(tok, [Step(answer, follow_up)])
    assert again.out.ids == chatml.out.ids


class _CallsOtherTokensSpecial:
    """The same vocabulary and template, but only ``<|endoftext|>`` counts as a special token."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.eos_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def __call__(self, text: str, **kwargs: Any) -> Any:
        return self._tokenizer(text, **kwargs)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return self._tokenizer.apply_chat_template(messages, **kwargs)


def test_the_end_of_turn_token_is_the_tokenizers_own(tok: Any) -> None:
    """Two tokenizer objects whose templates close a turn with the same text: only the one that
    calls ``<|im_end|>`` special takes a completion ending in it as a closed turn."""
    steps = plain_steps()[:3]
    for prompt in _drive(tok, steps):
        assert_is_the_full_render(tok, prompt)
    other = _CallsOtherTokensSpecial(tok)
    for prompt in _drive(other, steps):
        assert prompt.out.bridged and prompt.out.ids[: len(prompt.context)] == prompt.context
        assert other.decode(prompt.out.ids[len(prompt.context) :]) == END + "\n" + _chatml(prompt.step.new)
    for prompt in _drive(tok, steps):
        assert_is_the_full_render(tok, prompt)


def test_closings_that_differ_from_turn_to_turn(tok: Any) -> None:
    """More distinct closing texts on one tokenizer than a chat template normally has."""
    tok.chat_template = NUMBERED_END
    steps = [Step(f"Answer number {i}.", _user(f"Question {i}?")) for i in range(12)]
    for prompt in _drive(tok, steps):
        assert prompt.out.bridged
        assert prompt.out.ids[: len(prompt.context)] == prompt.context
        assert_is_the_full_render(tok, prompt)


class _Slotted:
    """A tokenizer object that can neither take attributes nor be referenced weakly."""

    __slots__ = ("_tokenizer", "eos_token_id")

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, text: str, **kwargs: Any) -> Any:
        return self._tokenizer(text, **kwargs)

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        return self._tokenizer.apply_chat_template(messages, **kwargs)


def test_other_tokenizer_objects_are_continued_the_same_way(tok: Any) -> None:
    for tokenizer in (FakeTokenizer(), _Slotted(tok)):
        steps = [
            Step(
                s.text, s.new, finish=s.finish, end="<|end|>" if isinstance(tokenizer, FakeTokenizer) else END
            )
            for s in plain_steps()
        ]
        for prompt in _drive(tokenizer, steps):
            assert prompt.out.bridged
            assert prompt.out.ids[: len(prompt.context)] == prompt.context
            assert_is_the_full_render(tokenizer, prompt)


def test_ids_come_back_as_python_ints_in_a_list_of_their_own(tok: Any) -> None:
    """Prompts and completions also arrive as arrays; what comes back is plain ints, and a list
    the caller may change without touching the turn it was built from."""
    answer = "The answer is 4."
    previous = chat_prompt_ids(tok, USER)
    sampled = [*_ids(tok, answer), *_ids(tok, END)]

    def call(prompt_ids: Any, token_ids: Any, mode: str = "tokens") -> list[int]:
        return continue_prompt_ids(
            tok,
            previous_messages=USER,
            previous_prompt_ids=prompt_ids,
            completion=Completion(text=answer, token_ids=token_ids),
            new_messages=_user("And 3+3?"),
            mode=mode,
        ).ids

    expected = call(previous, sampled)
    for mode in ("tokens", "template"):
        for ids in (
            call(np.asarray(previous), np.asarray(sampled), mode),
            call([np.int64(t) for t in previous], [np.int32(t) for t in sampled], mode),
            call(tuple(previous), tuple(sampled), mode),
        ):
            assert ids == expected and {type(t) for t in ids} == {int}

    kept_prompt, kept_sampled = list(previous), list(sampled)
    expected.clear()
    assert previous == kept_prompt and sampled == kept_sampled
    assert (
        chat_prompt_ids(tok, USER) == kept_prompt and call(previous, sampled)[: len(previous)] == kept_prompt
    )


def test_an_error_of_the_template_on_the_real_conversation_is_raised(tok: Any) -> None:
    tok.chat_template = REJECTS_BOOM
    steps = [Step("The answer is 4.", _user("And 3+3?")), Step("The answer is 6.", _tool("boom"))]
    with pytest.raises(Exception, match="no boom in this template"):
        _drive(tok, steps)


def test_conversations_running_in_threads_get_the_prompts_they_get_alone(tiny_model_dir: Path) -> None:
    """Sixteen conversations at once on four shared tokenizers (three templates on the real one,
    and the char-level fake): every prompt equals the one the same conversation gets on its own."""
    chatml = _load(tiny_model_dir)
    tools = _load(tiny_model_dir, TOOL_CALLING)
    numbered = _load(tiny_model_dir, NUMBERED_END)
    fake = FakeTokenizer()
    jobs: list[tuple[Any, list[Step], str]] = []
    for salt in range(4):
        fake_steps = [Step(s.text, s.new, finish=s.finish, end="<|end|>") for s in plain_steps(salt)]
        numbered_steps = [Step(f"Answer {salt}.{i}.", _user(f"Question {salt}.{i}?")) for i in range(10)]
        jobs += [
            (chatml, plain_steps(salt), "tokens"),
            (tools, tool_steps(salt), "tokens" if salt % 2 else "template"),
            (numbered, numbered_steps, "tokens"),
            (fake, fake_steps, "tokens"),
        ]

    def run(job: tuple[Any, list[Step], str]) -> list[Prompt]:
        tokenizer, steps, mode = job
        return _drive(tokenizer, steps, mode=mode)

    start = threading.Barrier(len(jobs))

    def together(job: tuple[Any, list[Step], str]) -> list[list[Prompt]]:
        start.wait(timeout=60)
        return [run(job) for _ in range(3)]

    # the threads go first, on tokenizers that have not continued a turn yet
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        concurrent = list(pool.map(together, jobs))

    for job, runs in zip(jobs, concurrent, strict=True):
        tokenizer, _, mode = job
        alone = [(p.out.ids, p.out.bridged) for p in run(job)]
        for prompts in runs:
            assert [(p.out.ids, p.out.bridged) for p in prompts] == alone
        # ... and what a conversation gets alone is what the module promises
        for prompt in runs[0]:
            if tokenizer is tools and mode == "tokens":
                assert_keeps_the_sampled_tokens(tokenizer, prompt)
            else:
                assert prompt.out.bridged == (mode == "tokens")
                assert_is_the_full_render(tokenizer, prompt)
