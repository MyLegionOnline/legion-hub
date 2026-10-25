"""In-memory registry for agent ask_user pending questions.

When the agent calls ask_user, the tool creates a PendingQuestion entry here,
emits an AGENT_QUESTION event to the room chat, and awaits the asyncio.Event.
The REST endpoint POST /rooms/{room_id}/questions/{qid}/answer resolves the
event with the user's chosen options.

Restart behaviour: the registry is in-memory, so it is always empty after a
backend restart.  The existing startup_orphan_cleanup in watchdog.py closes
all open proxy sessions, including any that were waiting for an answer —
orphaned questions simply expire with the session cleanup.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

# session_id → True  (one active question per session at a time)
# The watchdog checks this to avoid killing a session that is intentionally idle.
_WAITING_SESSIONS: set[UUID] = set()

# question_id (str) → PendingQuestion
_PENDING: dict[str, "PendingQuestion"] = {}


@dataclass
class PendingQuestion:
    question_id: str
    session_id: UUID
    room_id: UUID
    data: dict[str, Any]                     # {"questions": [...]}
    event: asyncio.Event = field(default_factory=asyncio.Event)
    answer: dict[str, Any] | None = None     # set by resolve() before .set()


def register(
    question_id: str,
    session_id: UUID,
    room_id: UUID,
    data: dict[str, Any],
) -> PendingQuestion:
    """Register a new pending question; returns the PendingQuestion to await."""
    pq = PendingQuestion(
        question_id=question_id,
        session_id=session_id,
        room_id=room_id,
        data=data,
    )
    _PENDING[question_id] = pq
    _WAITING_SESSIONS.add(session_id)
    logger.debug(
        "ask_user_registry: registered question_id=%s session=%s", question_id, session_id
    )
    return pq


def resolve(question_id: str, answer: dict[str, Any]) -> bool:
    """Resolve a pending question with the user's answer.

    Returns True if the question existed and was resolved, False if not found
    (already timed-out, duplicate call, or wrong question_id).
    """
    pq = _PENDING.pop(question_id, None)
    if pq is None:
        return False
    _WAITING_SESSIONS.discard(pq.session_id)
    pq.answer = answer
    pq.event.set()
    logger.debug(
        "ask_user_registry: resolved question_id=%s session=%s", question_id, pq.session_id
    )
    return True


def expire(question_id: str) -> None:
    """Remove a timed-out question from the registry (called by the tool after wait_for)."""
    pq = _PENDING.pop(question_id, None)
    if pq:
        _WAITING_SESSIONS.discard(pq.session_id)
        logger.debug(
            "ask_user_registry: expired question_id=%s session=%s",
            question_id, pq.session_id,
        )


def is_session_waiting(session_id: UUID) -> bool:
    """True when session is paused waiting for a user answer — skip watchdog kill."""
    return session_id in _WAITING_SESSIONS
