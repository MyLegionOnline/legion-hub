"""Chat marks for messages queued behind a running dialogue turn.

A message the user sends while the agent's turn is still running is parked for
the loop (``session_interject``) and read before the agent's next model
request.  Until then the chat showed nothing: in the 2026-11-09 incident six
messages sat behind one model request for 14 minutes with no sign that any of
them had been received.

Each queued message's bubble now carries a mark in ``Message.event_metadata``
(so it survives a reload) and a live ``dialogue_queue`` WebSocket frame:

* ``dialogue_queued_at`` — waiting for the agent's next step; the UI offers
  "Send now" (``session_interject.request_flush``);
* ``dialogue_delivered_at`` — the loop handed it to the model;
* ``dialogue_dropped_at`` — the turn ended (Stop, close, restart) before the
  loop read it.

Marking never breaks the turn: every failure is logged and swallowed.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

QUEUED = "queued"
DELIVERED = "delivered"
DROPPED = "dropped"

_FIELD = {
    QUEUED: "dialogue_queued_at",
    DELIVERED: "dialogue_delivered_at",
    DROPPED: "dialogue_dropped_at",
}

# The row is written by the chat handler just before the turn is called, so it
# is always recent; the window only bounds the lookup.
_LOOKUP_WINDOW = timedelta(minutes=30)
_LOOKUP_LIMIT = 20


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _factory():
    from app.db.session import engine

    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _broadcast(room_id: UUID, message_id: UUID, state: str, session_id: Any) -> None:
    from app.websockets.manager import manager

    try:
        await manager.broadcast(
            {
                "type": "dialogue_queue",
                "state": state,
                "message_id": str(message_id),
                "session_id": str(session_id) if session_id else None,
                "ts": _now().isoformat(),
            },
            room_id,
        )
    except Exception as exc:
        logger.debug("dialogue_queue: broadcast failed: %s", exc)


async def mark_queued(session: Any, text: str) -> UUID | None:
    """Find the chat row behind ``text`` and mark it queued; its id or ``None``."""
    room_id = getattr(session, "room_id", None)
    user_id = getattr(session, "user_id", None)
    if room_id is None or user_id is None or not (text or "").strip():
        return None
    from app.models.message import Message

    try:
        async with _factory()() as db:
            query = (
                select(Message)
                .where(
                    Message.room_id == room_id,
                    Message.event_type == "MESSAGE",
                    Message.sender_user_id == user_id,
                    Message.content == text,
                    # ``timestamp`` is a naive UTC column (datetime.utcnow).
                    Message.timestamp >= (_now() - _LOOKUP_WINDOW).replace(tzinfo=None),
                )
                .order_by(Message.timestamp.desc())
                .limit(_LOOKUP_LIMIT)
            )
            rows = (await db.execute(query)).scalars().all()
            # "Where is it?" sent twice: the newer copy is the one being queued.
            msg = next(
                (m for m in rows if not (m.event_metadata or {}).get(_FIELD[QUEUED])),
                None,
            )
            if msg is None:
                return None
            msg.event_metadata = {
                **(msg.event_metadata or {}),
                _FIELD[QUEUED]: _now().isoformat(),
                "dialogue_session_id": str(session.id),
            }
            db.add(msg)
            await db.commit()
            message_id = msg.id
    except Exception as exc:
        logger.warning("dialogue_queue: could not mark a queued message: %s", exc)
        return None
    await _broadcast(room_id, message_id, QUEUED, session.id)
    return message_id


async def _mark(room_id: UUID, ids: Iterable[UUID], state: str, session_id: Any) -> None:
    ids = [i for i in ids if i is not None]
    if not ids:
        return
    from app.models.message import Message

    field = _FIELD[state]
    when = _now().isoformat()
    try:
        async with _factory()() as db:
            rows = (
                await db.execute(select(Message).where(Message.id.in_(ids)))
            ).scalars().all()
            for msg in rows:
                meta = msg.event_metadata or {}
                if meta.get(_FIELD[DELIVERED]) or meta.get(_FIELD[DROPPED]):
                    continue
                # A new dict, not an in-place change: the column is plain JSON.
                msg.event_metadata = {**meta, field: when}
                db.add(msg)
            await db.commit()
    except Exception as exc:
        logger.warning("dialogue_queue: could not mark %s messages %s: %s", state, ids, exc)
        return
    for message_id in ids:
        await _broadcast(room_id, message_id, state, session_id)


async def publish_marks(session_id: UUID, room_id: UUID | None) -> None:
    """Report rows the loop delivered or dropped since the last call."""
    from app.core import session_interject

    delivered, dropped = session_interject.take_marks(session_id)
    if room_id is None:
        return
    await _mark(room_id, delivered, DELIVERED, session_id)
    await _mark(room_id, dropped, DROPPED, session_id)


_STALE_CHECKED: set[UUID] = set()


async def drop_stale(room_id: UUID, user_id: UUID | None) -> None:
    """A new turn starts with an empty queue: older "queued" marks are stale.

    The queue lives in process memory, so a backend restart mid-turn loses it;
    without this the bubbles of those messages would say "queued" forever.
    Within one process every queued row is resolved by ``publish_marks``, so
    one look per room after start-up is enough.
    """
    if user_id is None or room_id in _STALE_CHECKED:
        return
    _STALE_CHECKED.add(room_id)
    from app.models.message import Message

    try:
        async with _factory()() as db:
            query = (
                select(Message)
                .where(
                    Message.room_id == room_id,
                    Message.event_type == "MESSAGE",
                    Message.sender_user_id == user_id,
                )
                .order_by(Message.timestamp.desc())
                .limit(_LOOKUP_LIMIT)
            )
            rows = (await db.execute(query)).scalars().all()
            stale = [
                m for m in rows
                if (m.event_metadata or {}).get(_FIELD[QUEUED])
                and not (m.event_metadata or {}).get(_FIELD[DELIVERED])
                and not (m.event_metadata or {}).get(_FIELD[DROPPED])
            ]
            if not stale:
                return
            when = _now().isoformat()
            for msg in stale:
                msg.event_metadata = {**(msg.event_metadata or {}), _FIELD[DROPPED]: when}
                db.add(msg)
            await db.commit()
            ids = [m.id for m in stale]
    except Exception as exc:
        logger.warning("dialogue_queue: could not clear stale queued marks: %s", exc)
        return
    for message_id in ids:
        await _broadcast(room_id, message_id, DROPPED, None)
