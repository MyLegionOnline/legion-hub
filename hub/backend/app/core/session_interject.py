"""Messages a user sends into a dialogue turn that is still running.

A direct room has no task while the agent works — the task is recorded at the
end of the turn (internal design notes) — so there is nothing to
interject into (task_interject.py).  Before this module the second message
started a *parallel* turn on the same session: "stop" was answered by a second
agent loop while the first kept calling tools.

The message is parked here and the running loop picks it up before its next
model request, exactly like a task interjection.  Whatever the loop did not
reach before the turn ended is replayed as a follow-up turn by
``process_user_turn``.

In-process on purpose, like ``_ACTIVE_TURNS``: the Hub runs one worker, and a
queued message only matters for as long as the turn that will read it.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

_PENDING: dict[UUID, list[str]] = {}
# The chat rows behind the queued texts, so the UI can mark each bubble
# "queued" and later "delivered" / "not delivered" (app/core/dialogue_queue.py).
# Kept apart from ``_PENDING`` so the text API stays a plain list of strings.
_PENDING_IDS: dict[UUID, list[UUID]] = {}
_DELIVERED_IDS: dict[UUID, list[UUID]] = {}
_DROPPED_IDS: dict[UUID, list[UUID]] = {}

# What the model reads in place of the bare message — it must understand that
# this arrived mid-work and may redirect or cancel it.
NOTE_PREFIX = (
    "[The user sent a new message while you were working. It may change, narrow "
    "or cancel the task — act on it before taking another step.]\n"
)


def queue(session_id: UUID, text: str, message_id: UUID | None = None) -> None:
    if text and text.strip():
        _PENDING.setdefault(session_id, []).append(text)
        if message_id is not None:
            _PENDING_IDS.setdefault(session_id, []).append(message_id)


def drain(session_id: UUID) -> list[str]:
    ids = _PENDING_IDS.pop(session_id, [])
    if ids:
        _DELIVERED_IDS.setdefault(session_id, []).extend(ids)
    _FLUSH_REQUESTED.discard(session_id)
    return _PENDING.pop(session_id, [])


def has_pending(session_id: UUID) -> bool:
    return bool(_PENDING.get(session_id))


def _drop_pending(session_id: UUID) -> None:
    _PENDING.pop(session_id, None)
    ids = _PENDING_IDS.pop(session_id, [])
    if ids:
        _DROPPED_IDS.setdefault(session_id, []).extend(ids)


def clear(session_id: UUID) -> None:
    """End of the busy window: whatever is still queued will not be read."""
    _drop_pending(session_id)
    _STOP_REQUESTED.discard(session_id)
    _FLUSH_REQUESTED.discard(session_id)


def take_marks(session_id: UUID) -> tuple[list[UUID], list[UUID]]:
    """``(delivered, dropped)`` chat rows not yet reported to the UI."""
    return _DELIVERED_IDS.pop(session_id, []), _DROPPED_IDS.pop(session_id, [])


# The Stop button.  It ends the running turn, not the conversation: the session
# stays open (unlike a task Cancel there is no task row to retire), the work
# done so far is kept and gated, and the next message continues the dialogue.
_STOP_REQUESTED: set[UUID] = set()


def request_stop(session_id: UUID) -> None:
    _STOP_REQUESTED.add(session_id)
    _drop_pending(session_id)
    # A model request in flight is not a boundary worth waiting for: in the
    # incident of 2026-11-09 one request hung for 14 minutes.
    _abort_llm_call(session_id, ABORT_STOP)


def stop_requested(session_id: UUID) -> bool:
    return session_id in _STOP_REQUESTED


def consume_stop(session_id: UUID) -> bool:
    """True once for a stop that was requested; the turn handles it."""
    if session_id in _STOP_REQUESTED:
        _STOP_REQUESTED.discard(session_id)
        return True
    return False


# ---------------------------------------------------------------------------
# The model request in flight
# ---------------------------------------------------------------------------
#
# A queued message does NOT interrupt the request on its own: the model may be
# in the middle of a long, legitimate generation (a many-page report), and
# cutting it would throw that work away.  The message waits for the next step.
# Two explicit user actions do cut it: Stop, and "Send now" on a queued message
# (``request_flush``).  The loop awaits its request together with this event.

ABORT_STOP = "stop"
ABORT_FLUSH = "flush"

_LLM_CALLS: dict[UUID, asyncio.Event] = {}
_ABORT_REASON: dict[UUID, str] = {}
_FLUSH_REQUESTED: set[UUID] = set()


def begin_llm_call(session_id: UUID) -> asyncio.Event:
    """Register the request the loop is about to await; returns its abort event."""
    event = asyncio.Event()
    _LLM_CALLS[session_id] = event
    _ABORT_REASON.pop(session_id, None)
    if session_id in _STOP_REQUESTED:
        _ABORT_REASON[session_id] = ABORT_STOP
        event.set()
    elif session_id in _FLUSH_REQUESTED and has_pending(session_id):
        # "Send now" pressed between two requests is still meant for this one.
        _ABORT_REASON[session_id] = ABORT_FLUSH
        event.set()
    return event


def end_llm_call(session_id: UUID, event: asyncio.Event) -> None:
    if _LLM_CALLS.get(session_id) is event:
        _LLM_CALLS.pop(session_id, None)


def in_llm_call(session_id: UUID) -> bool:
    return session_id in _LLM_CALLS


def abort_reason(session_id: UUID) -> str | None:
    return _ABORT_REASON.get(session_id)


def _abort_llm_call(session_id: UUID, reason: str) -> bool:
    event = _LLM_CALLS.get(session_id)
    if event is None:
        return False
    # Stop wins over "send now" if both arrive for the same request.
    if _ABORT_REASON.get(session_id) != ABORT_STOP:
        _ABORT_REASON[session_id] = reason
    event.set()
    return True


FLUSH_INTERRUPTED = "interrupted"
FLUSH_NEXT_STEP = "next_step"
FLUSH_NOTHING_QUEUED = "nothing_queued"


def request_flush(session_id: UUID) -> str:
    """The user pressed "Send now" on a queued message.

    ``interrupted`` — the request in flight was dropped; the loop asks the
    model again at once, with the queued messages.  ``next_step`` — the loop
    is running a tool; the messages go in before its next model request, which
    follows the tool directly.  ``nothing_queued`` — already delivered.
    """
    if not has_pending(session_id):
        return FLUSH_NOTHING_QUEUED
    _FLUSH_REQUESTED.add(session_id)
    if _abort_llm_call(session_id, ABORT_FLUSH):
        return FLUSH_INTERRUPTED
    return FLUSH_NEXT_STEP


def format_note(texts: list[str]) -> str:
    return NOTE_PREFIX + "\n\n".join(texts)


def append_user_note(messages: list[dict[str, Any]], note: str) -> None:
    """Add a user turn to a provider transcript without breaking role order.

    After a tool round the transcript already ends on a user message (the
    Anthropic ``tool_result`` blocks); a second user message is merged into it
    as a text block so strict endpoints never see two user turns in a row.
    """
    if messages and messages[-1].get("role") == "user" and isinstance(
        messages[-1].get("content"), list
    ):
        messages[-1] = {
            **messages[-1],
            "content": [*messages[-1]["content"], {"type": "text", "text": note}],
        }
        return
    messages.append({"role": "user", "content": note})
