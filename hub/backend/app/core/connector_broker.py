"""Durable delivery for the connector WS transport.

``ConnectionManager.send_to_client`` is best-effort: it writes into whatever
socket happens to be live right now and silently drops the frame otherwise.
That is fine for a chat tab that reloads; it is fatal for a connector on
someone's VPS that drops off the network mid-task, because the dropped
``dispatch``/``proxy_message``/``answer``/``stop`` frame leaves the task
wedged with nothing left to redeliver it.

Every downstream frame is written to :class:`ConnectorOutbox` first (this
module owns the per-connector ``seq`` counter), then handed to the live
socket if one is open. It only stops being "must redeliver" once the
connector's ``ack`` names its ``seq``. ``resume`` replays exactly the rows a
reconnecting connector has not yet acked for one session — see
``hub/docs/connector_protocol.md`` §5 for the wire contract this implements.

Fencing (``accept_inbound_frame``) reuses ``core/task_runtime.is_run_current``
rather than inventing a second notion of "current generation" — a frame from
a superseded run must never mutate task/session state, exactly like a local
proxy-loop callback from a cancelled generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.connector import ConnectorOutbox

logger = logging.getLogger(__name__)

# Frame `type`s that carry a `run_id` and are subject to generation fencing —
# hub/docs/connector_protocol.md §2. Connection-level frames (hello/ping/pong/
# ack/resume) are exempt.
FENCED_FRAME_TYPES = frozenset(
    {
        "dispatch",
        "proxy_message",
        "answer",
        "stop",
        "status",
        "reasoning",
        "ask_cm",
        "ask_user",
        "result",
        "error",
    }
)

SendFrame = Callable[[Dict[str, Any]], Awaitable[None]]


def _frame(row: ConnectorOutbox) -> Dict[str, Any]:
    return {
        "type": row.kind,
        "seq": row.seq,
        "run_id": row.run_id,
        "session_id": str(row.session_id) if row.session_id else None,
        "payload": row.payload,
    }


async def enqueue(
    db: AsyncSession,
    *,
    connector_id: UUID,
    kind: str,
    payload: Dict[str, Any],
    session_id: Optional[UUID] = None,
    run_id: Optional[str] = None,
) -> ConnectorOutbox:
    """Persist one downstream frame and assign it the next per-connector seq.

    The caller commits/flushes; this only adds to the session so it composes
    with a surrounding transaction (same convention as connector_pairing.py).
    """
    next_seq = (
        await db.execute(
            select(func.coalesce(func.max(ConnectorOutbox.seq), 0) + 1).where(
                ConnectorOutbox.connector_id == connector_id
            )
        )
    ).scalar_one()
    row = ConnectorOutbox(
        connector_id=connector_id,
        session_id=session_id,
        run_id=run_id,
        seq=next_seq,
        kind=kind,
        payload=payload,
    )
    db.add(row)
    await db.flush()
    return row


async def deliver(
    db: AsyncSession,
    *,
    connector_id: UUID,
    send: SendFrame,
) -> int:
    """Push every unacked frame for this connector to a live socket, in order.

    Called right after a connector's WS registers — this is what makes a
    frame enqueued while the connector was offline arrive automatically on
    reconnect, without requiring the connector to know it missed anything.
    Delivery failure (socket already gone again) is left for the next
    ``deliver``/``resume`` call; frames only leave the "pending" set via
    :func:`ack`.
    """
    rows = (
        await db.execute(
            select(ConnectorOutbox)
            .where(
                ConnectorOutbox.connector_id == connector_id,
                ConnectorOutbox.acked_at.is_(None),
            )
            .order_by(ConnectorOutbox.seq)
        )
    ).scalars().all()
    delivered = 0
    for row in rows:
        try:
            await send(_frame(row))
        except Exception:
            logger.info(
                "connector_broker.deliver: send failed for connector=%s seq=%s, "
                "leaving pending for next attempt",
                connector_id, row.seq,
            )
            break
        delivered += 1
    return delivered


async def resume(
    db: AsyncSession,
    *,
    connector_id: UUID,
    session_id: UUID,
    last_seq: int,
    send: SendFrame,
) -> int:
    """Replay frames for one session the connector has not yet acked past.

    Unlike :func:`deliver` (all pending frames, any session), this answers an
    explicit ``{"type": "resume", "payload": {"session_id", "last_seq"}}``
    frame scoped to the one session the connector is asking about.
    """
    rows = (
        await db.execute(
            select(ConnectorOutbox)
            .where(
                ConnectorOutbox.connector_id == connector_id,
                ConnectorOutbox.session_id == session_id,
                ConnectorOutbox.seq > last_seq,
            )
            .order_by(ConnectorOutbox.seq)
        )
    ).scalars().all()
    delivered = 0
    for row in rows:
        try:
            await send(_frame(row))
        except Exception:
            logger.info(
                "connector_broker.resume: send failed for connector=%s seq=%s",
                connector_id, row.seq,
            )
            break
        delivered += 1
    return delivered


async def ack(db: AsyncSession, *, connector_id: UUID, seq: int) -> int:
    """Mark every frame up to and including ``seq`` as delivered (cumulative).

    Cumulative, not single-row: a connector acking seq=42 has necessarily
    already processed everything before it (frames are delivered in order),
    so there is no reason to keep retrying seq 1..41 after a crash.
    """
    from datetime import datetime, timezone

    rows = (
        await db.execute(
            select(ConnectorOutbox).where(
                ConnectorOutbox.connector_id == connector_id,
                ConnectorOutbox.seq <= seq,
                ConnectorOutbox.acked_at.is_(None),
            )
        )
    ).scalars().all()
    now = datetime.now(timezone.utc)
    for row in rows:
        row.acked_at = now
        db.add(row)
    if rows:
        await db.flush()
    return len(rows)


@dataclass(frozen=True)
class FrameDecision:
    accept: bool
    task_id: Optional[UUID] = None


async def accept_inbound_frame(
    db: AsyncSession,
    *,
    frame_type: str,
    session_id: Optional[UUID],
    run_id: Optional[str],
) -> FrameDecision:
    """Whether an upstream frame belongs to the session's *current* run.

    Reuses ``task_runtime.is_run_current`` — the same fence that protects a
    local proxy loop from a cancelled/replaced generation — rather than a
    second, transport-specific notion of staleness. A connection-level frame
    (no ``session_id``, e.g. ``hello``/``ping``/``pong``/``ack``) is always
    accepted; it never carries a ``run_id`` to fence against.
    """
    if frame_type not in FENCED_FRAME_TYPES or session_id is None:
        return FrameDecision(accept=True)

    from app.core.task_runtime import is_run_current
    from app.models.session.cm_agent_proxy import CmAgentProxySession

    proxy = (
        await db.execute(
            select(CmAgentProxySession.task_id).where(CmAgentProxySession.id == session_id)
        )
    ).scalar_one_or_none()
    if proxy is None:
        logger.warning(
            "connector_broker.accept_inbound_frame: unknown session_id=%s for frame_type=%s — dropping",
            session_id, frame_type,
        )
        return FrameDecision(accept=False)

    task_id = proxy
    current = await is_run_current(task_id, run_id, db)
    if not current:
        logger.info(
            "connector_broker.accept_inbound_frame: stale run_id=%s for task=%s "
            "frame_type=%s — dropping",
            run_id, task_id, frame_type,
        )
        return FrameDecision(accept=False, task_id=task_id)
    return FrameDecision(accept=True, task_id=task_id)


__all__ = [
    "FENCED_FRAME_TYPES",
    "FrameDecision",
    "enqueue",
    "deliver",
    "resume",
    "ack",
    "accept_inbound_frame",
]
