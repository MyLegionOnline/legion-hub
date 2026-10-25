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

from typing import Any
from uuid import UUID

_PENDING: dict[UUID, list[str]] = {}

# What the model reads in place of the bare message — it must understand that
# this arrived mid-work and may redirect or cancel it.
NOTE_PREFIX = (
    "[The user sent a new message while you were working. It may change, narrow "
    "or cancel the task — act on it before taking another step.]\n"
)


def queue(session_id: UUID, text: str) -> None:
    if text and text.strip():
        _PENDING.setdefault(session_id, []).append(text)


def drain(session_id: UUID) -> list[str]:
    return _PENDING.pop(session_id, [])


def has_pending(session_id: UUID) -> bool:
    return bool(_PENDING.get(session_id))


def clear(session_id: UUID) -> None:
    _PENDING.pop(session_id, None)
    _STOP_REQUESTED.discard(session_id)


# The Stop button.  It ends the running turn, not the conversation: the session
# stays open (unlike a task Cancel there is no task row to retire), the work
# done so far is kept and gated, and the next message continues the dialogue.
_STOP_REQUESTED: set[UUID] = set()


def request_stop(session_id: UUID) -> None:
    _STOP_REQUESTED.add(session_id)
    _PENDING.pop(session_id, None)


def stop_requested(session_id: UUID) -> bool:
    return session_id in _STOP_REQUESTED


def consume_stop(session_id: UUID) -> bool:
    """True once for a stop that was requested; the turn handles it."""
    if session_id in _STOP_REQUESTED:
        _STOP_REQUESTED.discard(session_id)
        return True
    return False


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
