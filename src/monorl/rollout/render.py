"""Chat rendering for multi-turn prompts: full template renders and token-level continuation.

:func:`chat_prompt_ids` renders a whole conversation with the tokenizer's chat template. Doing that
again for turn ``k+1`` re-encodes what the policy already produced through text, and that round
trip is not exact: special tokens are dropped by decoding, whitespace is normalised, and templates
may drop earlier reasoning or re-serialise tool calls. The new prompt then no longer starts with
the tokens the policy was sampled on, which are the tokens it is trained on.

:func:`continue_prompt_ids` keeps the sampled tokens instead ("token in, token out"): the next
prompt is the previous prompt, the completion exactly as sampled, the template's end-of-turn text
when the completion did not close the turn itself, and the rendering of only the new messages plus
the generation prompt. Both template pieces come from rendering the conversation with a sentinel
in place of the assistant turn, so nothing depends on how the template would have re-tokenised the
sampled text.

This only fixes tokens, never what the context says: a turn is continued this way only when the
template itself would carry the assistant's content over unchanged (surrounding whitespace
aside). A template that rewrites history — dropping the reasoning of earlier turns, rendering a
turn differently once more messages follow it — is followed instead: the conversation is rendered
in full, as with ``mode="template"``, and the result says so (``bridged=False``). The batch
builder then trains such a turn as a sequence of its own.

A continuation is proven with renders of the whole conversation, on every turn, so the work around
them is kept small: the probes are rendered in the order that lets a failed check skip the others,
the render with the real assistant turn doubles as the full render of a turn that cannot be
continued, and the ids of a closing's end-of-turn marker are worked out once per tokenizer object.
That last part takes a tokenizer's vocabulary and special tokens as fixed once it has continued a
turn.

Torch-free; shared by the episode runner and the capture gateway.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from monorl.core.types import Completion, Message

SENTINEL = "MONORLTURNSENTINEL"
"""Stand-in assistant content used to locate the end of a turn in rendered chat text."""

_REMEMBERED_CLOSINGS = 8
"""Closing texts per tokenizer whose end-of-turn marker is remembered; a chat template has one or two."""
_REMEMBERED_CLOSING_CHARS = 256
"""A longer closing is not template boilerplate and is encoded every time."""


def render_chat(
    tokenizer: Any, messages: Sequence[Message], tools: Any = None, *, add_generation_prompt: bool = True
) -> str:
    """``messages`` as chat-template text (``tools`` is only passed when given)."""
    kwargs: dict[str, Any] = {"add_generation_prompt": add_generation_prompt, "tokenize": False}
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template(list(messages), **kwargs)


def encode_text(tokenizer: Any, text: str) -> list[int]:
    """Token ids of already-templated text: no special tokens are added, the template owns them."""
    if not text:
        return []
    return _ints(tokenizer(text, add_special_tokens=False)["input_ids"])


def chat_prompt_ids(tokenizer: Any, messages: Sequence[Message], tools: Any = None) -> list[int]:
    """Token ids of ``messages`` rendered with the chat template and a generation prompt.

    Rendered as text first and tokenised without special tokens, so the template alone decides
    about BOS/EOS; rollout engines use the same recipe for chat prompts.
    """
    return encode_text(tokenizer, render_chat(tokenizer, messages, tools))


def special_token_ids(tokenizer: Any) -> frozenset[int]:
    """Ids the tokenizer treats as special: its named special tokens, added tokens flagged special
    (end-of-turn markers are often only that) and the EOS id(s)."""
    ids = {int(t) for t in (getattr(tokenizer, "all_special_ids", None) or ())}
    for token_id, token in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items():
        if getattr(token, "special", False):
            ids.add(int(token_id))
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    elif eos is not None:
        ids.update(int(t) for t in eos)
    return frozenset(ids)


@dataclass(slots=True)
class Continuation:
    ids: list[int]
    """Prompt token ids for the next turn."""
    bridged: bool
    """``True``: ``ids`` extend the previous prompt and its sampled completion token for token.
    ``False``: the conversation was rendered again from its messages."""


def continue_prompt_ids(
    tokenizer: Any,
    *,
    previous_messages: Sequence[Message],
    previous_prompt_ids: Sequence[int],
    completion: Completion,
    new_messages: Sequence[Message],
    tools: Any = None,
    assistant_message: Message | None = None,
    mode: str = "tokens",
) -> Continuation:
    """Prompt ids for the turn after ``completion``.

    ``previous_messages`` is the chat the previous turn was prompted with, ``previous_prompt_ids``
    the ids it conditioned on, ``new_messages`` what follows the assistant turn (tool results, the
    next user message). ``assistant_message`` is the completion as a chat message when it differs
    from plain ``{"role": "assistant", "content": completion.text}`` (structured ``tool_calls``).

    ``mode="tokens"`` continues the token stream (module docstring); ``mode="template"`` always
    re-renders ``previous_messages + [assistant_message] + new_messages``.
    """
    if mode not in ("tokens", "template"):
        raise ValueError(f"unknown turn rendering mode {mode!r}; expected 'tokens' or 'template'")
    assistant: Message = (
        dict(assistant_message)  # type: ignore[assignment]
        if assistant_message is not None
        else {"role": "assistant", "content": completion.text}
    )
    rendered: str | None = None
    if mode == "tokens":
        pieces, rendered = _turn_pieces(tokenizer, previous_messages, assistant, new_messages, tools)
        if pieces is not None:
            closing, tail = pieces
            sampled = _ints(completion.token_ids)
            ids = _ints(previous_prompt_ids)
            ids += sampled
            ids += _after_emitted_end(tokenizer, closing, tail, sampled[-1] if sampled else None)
            return Continuation(ids=ids, bridged=True)
    if rendered is None:
        rendered = render_chat(tokenizer, [*previous_messages, assistant, *new_messages], tools)
    return Continuation(ids=encode_text(tokenizer, rendered), bridged=False)


def _ints(ids: Iterable[Any]) -> list[int]:
    """``ids`` as a new list of Python ints. A list that holds nothing else, which is what prompts
    and completions arrive as, is copied without converting each of its thousands of elements."""
    if type(ids) is list and set(map(type, ids)) <= {int}:
        return ids.copy()
    return [int(t) for t in ids]


class _ProbeRejected(Exception):
    """The chat template raised on a probe conversation."""


def _turn_pieces(
    tokenizer: Any,
    previous_messages: Sequence[Message],
    assistant: Message,
    new_messages: Sequence[Message],
    tools: Any,
) -> tuple[tuple[str, str] | None, str | None]:
    """``(closing, tail)``: the text the template puts after an assistant turn's content, and its
    rendering of ``new_messages`` plus the generation prompt. ``None`` when the template cannot be
    taken apart like that, renders the earlier messages differently now that more follow, or
    would not carry the assistant's content over as it is.

    The second value is the conversation rendered with the real turn and the generation prompt,
    when a check needed it: the text a caller that cannot continue would render next."""
    content = assistant.get("content") or ""
    if not isinstance(content, str):
        return None, None

    def probe(*turn_and_after: Message, add_generation_prompt: bool = True) -> str:
        messages = [*previous_messages, *turn_and_after]
        try:
            return render_chat(tokenizer, messages, tools, add_generation_prompt=add_generation_prompt)
        except Exception as e:  # noqa: BLE001 - a template may reject the probe; the full render reports real errors
            raise _ProbeRejected from e

    plain: Message = {"role": "assistant", "content": SENTINEL}
    # keeps tool_calls: templates may need them to render the tool messages that follow
    marked: Message = {**assistant, "content": SENTINEL}
    try:
        # The assistant turn must follow the previous prompt as the template renders it today ...
        prompt = probe()
        extended = probe(marked, *new_messages)
        if not extended.startswith(prompt):
            return None, None
        at = extended.find(SENTINEL, len(prompt))
        if at < 0:
            # A template that renders a turn from its tool_calls alone (Llama 3.1, Mistral) leaves the
            # content out; that only holds the sampled turn when there was no other content.
            if content.strip() or not assistant.get("tool_calls"):
                return None, None
        elif extended[len(prompt) : at].strip():
            return None, None
        closed_plain = probe(plain, add_generation_prompt=False)
        plain_at = closed_plain.rfind(SENTINEL)
        if plain_at < 0:
            return None, None
        closing = closed_plain[plain_at + len(SENTINEL) :]
        closed = closed_plain if marked == plain else probe(marked, add_generation_prompt=False)
        if at < 0:
            if not (closed.startswith(prompt) and closed.endswith(closing) and extended.startswith(closed)):
                return None, None
            return (closing, extended[len(closed) :]), None
        head, rest = extended[:at], extended[at + len(SENTINEL) :]
        # ... end the way a finished turn ends (after the content: tool calls if any, then the closing) ...
        closed_at = closed.rfind(SENTINEL)
        if closed_at < 0:
            return None, None
        turn_end = closed[closed_at + len(SENTINEL) :]
        if not rest.startswith(turn_end) or not turn_end.endswith(closing):
            return None, None
        # ... and hold the real content as it is.
        actual = probe(assistant, *new_messages) if content.strip() else None
    except _ProbeRejected:
        return None, None
    if actual is not None:
        if len(actual) < len(head) + len(rest) or not (actual.startswith(head) and actual.endswith(rest)):
            return None, actual
        if actual[len(head) : len(actual) - len(rest)].strip() != content.strip():
            return None, actual
    return (closing, rest[len(turn_end) :]), actual


def _after_emitted_end(tokenizer: Any, closing: str, tail: str, last_sampled: int | None) -> list[int]:
    """Ids of ``closing + tail`` without what the policy already produced: when its last sampled
    token is the closing's end-of-turn marker (the first special token of ``closing``), the ids
    resume after that marker. A completion cut off by the length limit or a stop string gets the
    whole closing."""
    rest = encode_text(tokenizer, closing + tail)
    if last_sampled is None:
        return rest
    marker = _end_of_turn(tokenizer, closing)
    if marker is None or marker[-1] != last_sampled or tuple(rest[: len(marker)]) != marker:
        return rest
    return rest[len(marker) :]


_END_OF_TURN: weakref.WeakKeyDictionary[Any, dict[str, tuple[int, ...] | None]] = weakref.WeakKeyDictionary()
"""Per tokenizer object, and dropped with it: closing text -> :func:`_end_of_turn`."""
_END_OF_TURN_LOCK = threading.Lock()
_UNKNOWN: Any = object()


def _end_of_turn(tokenizer: Any, closing: str) -> tuple[int, ...] | None:
    """Ids of ``closing`` up to and including its first special token, the end-of-turn marker a
    completion may already end with; ``None`` when ``closing`` has no special token."""
    known: dict[str, tuple[int, ...] | None] | None = None
    marker = _UNKNOWN
    if len(closing) <= _REMEMBERED_CLOSING_CHARS:
        try:
            with _END_OF_TURN_LOCK:
                known = _END_OF_TURN.setdefault(tokenizer, {})
                marker = known.get(closing, _UNKNOWN)
        except TypeError:  # not hashable, or no weak references to it: worked out every time
            pass
    if marker is _UNKNOWN:
        closing_ids = encode_text(tokenizer, closing)
        specials = special_token_ids(tokenizer)
        end = next((i for i, token_id in enumerate(closing_ids) if token_id in specials), None)
        marker = None if end is None else tuple(closing_ids[: end + 1])
        if known is not None:
            with _END_OF_TURN_LOCK:
                while len(known) >= _REMEMBERED_CLOSINGS and closing not in known:
                    del known[next(iter(known))]
                known[closing] = marker
    return marker


__all__ = [
    "Continuation",
    "chat_prompt_ids",
    "continue_prompt_ids",
    "encode_text",
    "render_chat",
    "special_token_ids",
]
