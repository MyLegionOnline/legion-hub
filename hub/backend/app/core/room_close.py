"""Close a room: its agents, its container, its meter and every open view.

A room open in the browser costs resources even with no agent working, so it
is billed (owner's rule, core/billing_interval.py) — and the Stop button of
the workspace does not end that: the room stays open, so it is still used.
The only way the meter stops while the user is around is to close the room
(owner's decision, 2026-10-02):

* the button next to the room's dot in the sidebar (``POST /rooms/{id}/close``);
* the hour hard stop (the cloud plugin's reconciler step): out of hours, the room closes.

Closing pauses the room's tasks, stops its container, closes the billing
interval at the moment of closing and drops the room from every open view.
``workspace.closed_at`` makes the last part reliable across tabs and
replicas: a view opened before it is not presence any more, and its socket is
shut on its next heartbeat (billing_interval.presence_beat). Opening the room
again later is an ordinary open, billed from then on.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

#: ``room_closed.reason`` / ``workspace_usage.stop_reason`` of the button.
REASON_USER = "user_close"
_RUNNING_WORKSPACE_STATES = ("starting", "running", "draining")


def _live_task_states() -> list:
    # PAUSING is already on its way to a stop; BLOCKED is paused along with
    # these by pause_workspace_tasks but is not by itself work in progress.
    from app.models.task import TaskState

    return [TaskState.IN_PROGRESS, TaskState.RESUMING]


async def close_room(
    room_id: UUID,
    db: AsyncSession,
    *,
    reason: str = REASON_USER,
    pause_reason: str = "user",
    notice: Optional[str] = None,
    notice_event: Optional[str] = None,
    close_view: bool = True,
) -> dict[str, Any]:
    """Stop all work of ``room_id``; with ``close_view`` also its meter and views.

    ``close_view=False`` stops the work only (agents, container) and leaves
    an open view billed as usual — for an expired plan, whose rooms stay
    readable (internal design notes TR4). ``notice`` is posted to the room's
    history as ``notice_event`` when anything was actually stopped.
    """
    from app.core.billing_interval import close_usage
    from app.core.workspace_lifecycle import (
        WorkspaceStartError,
        pause_workspace_tasks,
        publish_room_closed,
        stop_workspace,
    )
    from app.models.task import Task
    from app.models.workspace import Workspace, WorkspaceUsage

    ws = (
        await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ).scalar_one_or_none()
    if ws is None:
        return {"paused_task_ids": [], "container_stopped": False, "billing_closed": False}

    paused: list[UUID] = []
    live_task = (
        await db.execute(
            select(Task.id).where(
                Task.room_id == room_id, Task.state.in_(_live_task_states())
            ).limit(1)
        )
    ).first()
    if live_task is not None:
        paused = await pause_workspace_tasks(room_id, db, pause_reason=pause_reason)

    container_stopped = False
    if ws.status in _RUNNING_WORKSPACE_STATES:
        try:
            await stop_workspace(room_id, db, reason=reason, force=True)
            container_stopped = True
        except WorkspaceStartError as exc:
            logger.warning(
                "room_close: could not stop the container of room %s: %s",
                room_id, exc.message,
            )

    billing_closed = False
    if close_view:
        now = datetime.now(timezone.utc)
        # The stop path keeps the interval open while the room is open in a
        # browser; closing the room is exactly what ends that.
        for usage in (
            await db.execute(
                select(WorkspaceUsage).where(
                    WorkspaceUsage.workspace_id == ws.id,
                    WorkspaceUsage.stopped_at.is_(None),
                )
            )
        ).scalars().all():
            await close_usage(usage, ws, db, stop_reason=reason, now=now)
            billing_closed = True
        ws.closed_at = now
        ws.presence_at = None
        db.add(ws)
    await db.commit()

    if close_view:
        await publish_room_closed(room_id, reason, notice)
    if notice and notice_event and (paused or container_stopped or billing_closed):
        from app.core.events import emit_event

        await emit_event(
            room_id, notice_event, notice, db,
            metadata={
                "reason": reason,
                "paused_task_ids": [str(task_id) for task_id in paused],
                "container_stopped": container_stopped,
                "room_closed": close_view,
            },
        )
    logger.info(
        "room_close: room %s closed (reason=%s, view=%s, tasks paused=%d, "
        "container stopped=%s, billing closed=%s)",
        room_id, reason, close_view, len(paused), container_stopped, billing_closed,
    )
    return {
        "paused_task_ids": paused,
        "container_stopped": container_stopped,
        "billing_closed": billing_closed,
    }


def closed_since(closed_at: Optional[datetime], since: Optional[datetime]) -> bool:
    """The room was closed after a view that opened at ``since``."""
    if closed_at is None or since is None:
        return False
    if closed_at.tzinfo is None:
        closed_at = closed_at.replace(tzinfo=timezone.utc)
    return closed_at >= since


async def publish_read_only(room_id: UUID, message: str, *, reason: str = "plan_expired") -> None:
    """Switch an open view of a room to read-only: an expired plan's room (TR4),
    or one frozen after the plan shrank.

    Sent every tick the room is open: idempotent on the client, and it
    reaches a tab that opened the room before the plan expired.
    """
    from app.websockets.manager import manager

    try:
        await manager.broadcast(
            {"type": "room_read_only", "room_id": str(room_id),
             "reason": reason, "message": message},
            room_id,
        )
    except Exception as exc:
        logger.warning("room_close: read-only notice failed for room %s: %s", room_id, exc)
