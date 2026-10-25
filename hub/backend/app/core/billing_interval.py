"""Billing intervals follow the room's work, not its container.

The unit the tariff is sold in (internal design notes §1, T2/T4) is "the room
is being used": from the moment a member opens the room, through the whole
conversation with its agents — including the minutes a person spends thinking
over an answer — and on after the browser is closed for as long as agents keep
working on the project. Whether the Hub happens to run the room's container
during that time is our own optimisation and plays no part in it.

The interval used to be born only in the container start path, so work that
never needed a container (ChatMaster reading files and git, which go straight
to the volume) was not billed at all. This module owns the other ways in:

* :func:`touch_presence` — the room socket's heartbeat, "the room is open";
* :func:`open_intervals_for_working_rooms` — the reconciler pass that opens
  an interval for any room doing work without one, back-dated to the start of
  that work;
* :func:`ensure_usage_open` — the shared primitive, also used by the container
  start path.

An open interval is reused only while it was watched without a break. One
that outlived a window nobody watched (backend or Docker down, a sleeping
workstation) is split at that gap: see :func:`close_if_unobserved_gap`.

What ends an interval is unchanged: ``room_has_live_work`` turning false
(workspace_lifecycle), swept by the reconciler. Running out of hours
(legion_cloud hours_guard) stops the agents, not the meter: a room open in the
browser is still used, and still billed.

Inside an interval we also record how long it had a container
(``container_seconds``). That does not change the price; it is the number the
cost model (stage 0.6) and the storage choice (stage 5: RWX vs RWO vs S3) need.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

#: How often an open room socket refreshes ``workspace.presence_at``.
PRESENCE_BEAT_SEC = 60
#: How long after the last beat the room still counts as open. Above one beat
#: plus one reconciler tick, so a live socket never flickers the interval shut;
#: it also bounds the overcount after a tab is closed or a replica dies.
PRESENCE_FRESH_SEC = 150
#: How far back the sweep may date a new interval. Genuinely new work is at
#: most one reconciler tick old when the sweep sees it; the cap keeps a session
#: row left open for months by some old bug from being billed retroactively.
MAX_BACKDATE_SEC = 600
#: Session kinds that are never a reason to bill on their own. cm_self_plan is
#: the machine decomposition turn after a plan is confirmed — no user in it,
#: median 75 s on the live stack — and it runs while the room is open anyway
#: (the user just pressed confirm) and hands off to tasks, which are billed.
#: It is also the kind that leaked: 42 rows left open for up to two months.
NON_BILLING_SESSION_KINDS = ("cm_self_plan",)
#: How far the last evidence of use (plus its billable tail) may lag behind
#: `now` before an open interval counts as having outlived an unobserved gap.
#: Well above a reconciler tick and a presence beat, so a watched room never
#: splits.
GAP_SLACK_SEC = 300

#: "Take it from the workspace row" for the evidence overrides below.
_FROM_WS: Any = object()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------- The sidebar dot follows the interval ----------------------------
#
# Intervals open and close in half a dozen places (container start, presence,
# the reconciler's two sweeps, the stop path, admin stop). Rather than a
# publish call next to each, every site marks the change on its DB session and
# one after_commit hook tells the room's members — so the dot never shows a
# change that was rolled back, and a new site cannot forget it.

_INFO_KEY = "billing_changes"


def mark_billing_change(db: Any, room_id: Optional[UUID], active: bool) -> None:
    """Record that ``room_id``'s interval opened/closed in this transaction."""
    if room_id is None:
        return
    sync_session = getattr(db, "sync_session", None)
    info = getattr(sync_session, "info", None)
    if isinstance(info, dict):
        info.setdefault(_INFO_KEY, {})[room_id] = active


def _publish_after_commit(session: Any) -> None:
    changes = session.info.pop(_INFO_KEY, None)
    if not changes:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    from app.core.workspace_lifecycle import publish_room_billing

    for room_id, active in changes.items():
        loop.create_task(publish_room_billing(room_id, active))


def _drop_after_rollback(session: Any) -> None:
    session.info.pop(_INFO_KEY, None)


def _register_listeners() -> None:
    from sqlalchemy import event
    from sqlalchemy.orm import Session as SyncSession

    if not event.contains(SyncSession, "after_commit", _publish_after_commit):
        event.listen(SyncSession, "after_commit", _publish_after_commit)
        event.listen(SyncSession, "after_rollback", _drop_after_rollback)


_register_listeners()


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    # Stragglers (older rows, SQLite in tests, Task.updated_at) come back
    # naive; they are UTC by construction.
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def presence_is_fresh(presence_at: Optional[datetime], now: Optional[datetime] = None) -> bool:
    presence_at = _aware(presence_at)
    if presence_at is None:
        return False
    return (now or _now()) - presence_at < timedelta(seconds=PRESENCE_FRESH_SEC)


def fold_container_time(usage: Any, now: Optional[datetime] = None) -> None:
    """Move the running container stretch of ``usage`` into ``container_seconds``.

    Called whenever the container goes away and when the interval closes.
    Idempotent: a second call finds ``container_started_at`` empty.
    """
    since = _aware(getattr(usage, "container_started_at", None))
    if since is None:
        return
    elapsed = ((now or _now()) - since).total_seconds()
    usage.container_seconds = (usage.container_seconds or 0.0) + max(0.0, elapsed)
    usage.container_started_at = None


# ---------- Where a closing interval really ends ---------------------------
#
# Every close path runs when the Hub *notices* the room has stopped working:
# the reconciler seeing a vanished container, the work_ended sweep, the idle
# stop. That is within a tick of the real end only while the control plane is
# watching. After a window in which nobody watched — the backend or Docker
# down, a sleeping workstation, a deploy, auto-stop failing — closing at `now`
# billed the whole unobserved gap: 40.2 h on the live stack for a room last
# used 5 minutes after its interval opened (economics.md §2). By §1 "Unit
# of accounting" that time is not the client's, so the close is dated by the last
# evidence of use plus the tail the rule itself allows for that evidence.
#
# Deliberately not evidence: ``workspace.last_heartbeat_at``. The reconciler
# refreshes it on every successful status probe, so it tracks the container,
# not the room's use — counting it would bill an idle container for exactly
# as long as auto-stop failed to reap it, which is one of the gaps this fixes.
# Real heartbeats from the room (api/v1/workspace, proxy_session) stamp
# ``last_activity_at`` alongside and are covered by it.


def billed_stop_at(
    started_at: Optional[datetime],
    now: datetime,
    *,
    activity: tuple[Optional[datetime], ...] = (),
    presence_at: Optional[datetime] = None,
    idle_sec: float,
) -> datetime:
    """When a closing interval stops being billable.

    ``activity`` stamps (room activity, session and task activity) keep the
    room billed for the idle window after them — the same window
    ``room_has_live_work`` gives a quiet session. ``presence_at`` keeps it
    billed for the presence freshness, as a live room socket would. Never
    later than ``now``, never before ``started_at``; with no evidence at all
    the interval is worth one idle window from its start.
    """
    start = _aware(started_at)
    ends = [
        t + timedelta(seconds=idle_sec)
        for t in (_aware(a) for a in activity if isinstance(a, datetime))
    ]
    if isinstance(presence_at, datetime):
        ends.append(_aware(presence_at) + timedelta(seconds=PRESENCE_FRESH_SEC))
    if not ends and start is not None:
        ends.append(start + timedelta(seconds=idle_sec))
    stop = min(now, max(ends)) if ends else now
    if start is not None and stop < start:
        stop = min(start, now)
    return stop


async def _last_work_activity(ws: Any, db: AsyncSession) -> tuple[Optional[datetime], ...]:
    """Latest session and task activity in the room, in one round trip."""
    from app.models.session.base import Session
    from app.models.task import Task

    room_id = getattr(ws, "room_id", None)
    if room_id is None:
        return ()
    last_session = (
        select(func.max(func.coalesce(Session.last_activity_at, Session.started_at)))
        .where(
            Session.room_id == room_id,
            Session.kind.not_in(NON_BILLING_SESSION_KINDS),
        )
        .scalar_subquery()
    )
    last_task = (
        select(func.max(Task.updated_at)).where(Task.room_id == room_id).scalar_subquery()
    )
    row = (await db.execute(select(last_session, last_task))).first()
    # Tolerant of partial rows (and of test doubles): only datetimes count.
    return tuple(v for v in (tuple(row) if row is not None else ()) if isinstance(v, datetime))


async def _evidence_stop(
    usage: Any,
    ws: Any,
    db: AsyncSession,
    now: datetime,
    last_activity_at: Any = _FROM_WS,
) -> datetime:
    """:func:`billed_stop_at` for ``usage`` from everything the Hub knows."""
    from app.core.workspace_lifecycle import WORKSPACE_IDLE_TIMEOUT_SEC

    if last_activity_at is _FROM_WS:
        last_activity_at = getattr(ws, "last_activity_at", None)
    activity = (last_activity_at,) + await _last_work_activity(ws, db)
    return billed_stop_at(
        usage.started_at,
        now,
        activity=activity,
        presence_at=getattr(ws, "presence_at", None),
        idle_sec=WORKSPACE_IDLE_TIMEOUT_SEC,
    )


async def close_usage(
    usage: Any,
    ws: Any,
    db: AsyncSession,
    *,
    stop_reason: str,
    now: Optional[datetime] = None,
    stop_at: Optional[datetime] = None,
) -> float:
    """Close ``usage`` at the last billable moment and count the hours.

    The single closer for every path (stop, reconciler, sweeps), so the row,
    the ``workspace_session_hours_total`` counter and ``container_seconds``
    all describe the same interval. ``stop_at`` is that moment when the
    caller has already worked it out. Returns the billed hours. Stages
    changes only — the caller commits.
    """
    from app.core.metrics import record_workspace_session_hours

    now = now or _now()
    stop = stop_at if stop_at is not None else await _evidence_stop(usage, ws, db, now)
    started = _aware(usage.started_at)
    billed_sec = max(0.0, (stop - started).total_seconds()) if started else 0.0

    # The container stretch ends with the interval: container time outside
    # the billed interval would push the container share of §2 above 100 %.
    fold_container_time(usage, stop)
    if started is not None and (usage.container_seconds or 0.0) > billed_sec:
        usage.container_seconds = billed_sec

    usage.stopped_at = stop
    usage.stop_reason = stop_reason
    db.add(usage)
    mark_billing_change(db, getattr(ws, "room_id", None), False)
    if started is not None:
        record_workspace_session_hours(billed_sec / 3600.0)
    if (now - stop).total_seconds() > 60:
        logger.info(
            "billing: closed interval of room %s at %s, %.1f min before it was "
            "noticed (reason=%s) — the gap had no evidence of use",
            getattr(ws, "room_id", None), stop.isoformat(),
            (now - stop).total_seconds() / 60.0, stop_reason,
        )
    return billed_sec / 3600.0


async def _open_row(ws: Any, db: AsyncSession) -> Any:
    from app.models.workspace import WorkspaceUsage

    result = await db.execute(
        select(WorkspaceUsage).where(
            WorkspaceUsage.workspace_id == ws.id,
            WorkspaceUsage.stopped_at.is_(None),
        )
    )
    return result.scalars().first()


async def _last_stopped_at(ws: Any, db: AsyncSession) -> Optional[datetime]:
    from app.models.workspace import WorkspaceUsage

    result = await db.execute(
        select(func.max(WorkspaceUsage.stopped_at)).where(
            WorkspaceUsage.workspace_id == ws.id
        )
    )
    return _aware(result.scalar())


async def close_if_unobserved_gap(
    usage: Any,
    ws: Any,
    db: AsyncSession,
    *,
    now: Optional[datetime] = None,
    last_activity_at: Any = _FROM_WS,
) -> Optional[datetime]:
    """Close ``usage`` if it outlived a window nobody watched; return its end.

    While the Hub watches, an open interval is closed within a tick of the
    room's work ending, so its evidence of use is never far behind. If the
    last evidence plus the tail the rule gives it ended well before `now`,
    nobody watched in between (the backend or Docker down, the workstation
    asleep), and reusing the row would bill that gap the moment the room is
    used again: 20.97 h on the live stack for half an hour one evening and
    half an hour the next afternoon. The row is closed where the evidence
    ends, as :func:`close_usage` dates any close, with reason ``gap``.

    Evidence is read as it was before the caller stamps new evidence:
    ``ws.presence_at`` as it is now, and ``last_activity_at`` instead of the
    row's own when the caller has already overwritten it. Returns None when
    the interval is still current.
    """
    now = now or _now()
    stop = await _evidence_stop(usage, ws, db, now, last_activity_at)
    if (now - stop).total_seconds() <= GAP_SLACK_SEC:
        return None
    await close_usage(usage, ws, db, stop_reason="gap", now=now, stop_at=stop)
    return stop


async def close_unobserved_intervals(db: AsyncSession) -> int:
    """Split every open interval at an unobserved gap; run after a blind window.

    The reconciler calls this on its first tick after start and after a tick
    that came far too late (a suspended host): rooms nobody touches again
    would otherwise keep the gap until the work-ended sweep, and rooms whose
    work carries on through the restart would keep it for good. A room that
    still works gets a new interval from `now`; the others stay closed until
    the open sweep or presence sees new work.
    """
    from app.core.workspace_lifecycle import room_has_live_work
    from app.models.workspace import Workspace, WorkspaceUsage

    rows = (
        await db.execute(
            select(WorkspaceUsage, Workspace)
            .join(Workspace, Workspace.id == WorkspaceUsage.workspace_id)
            .where(WorkspaceUsage.stopped_at.is_(None))
        )
    ).all()
    now = _now()
    split = 0
    for usage, ws in rows:
        had_container = usage.container_started_at is not None
        stop = await close_if_unobserved_gap(usage, ws, db, now=now)
        if stop is None:
            continue
        split += 1
        if await room_has_live_work(ws.room_id, db):
            _open_new_row(ws, db, start=now, container_at=now if had_container else None)
    if split:
        await db.commit()
        logger.info("billing: split %d interval(s) at an unobserved gap", split)
    return split


def _open_new_row(
    ws: Any, db: AsyncSession, *, start: datetime, container_at: Optional[datetime]
) -> Any:
    from app.models.workspace import WorkspaceUsage

    usage = WorkspaceUsage(
        workspace_id=ws.id,
        mem_limit_mb=ws.mem_limit_mb or 1024,
        started_at=start,
    )
    usage.container_started_at = container_at
    db.add(usage)
    mark_billing_change(db, ws.room_id, True)
    logger.info(
        "billing: opened interval for room %s (container=%s, started_at=%s)",
        ws.room_id, container_at is not None, start.isoformat(),
    )
    return usage


async def ensure_usage_open(
    ws: Any,
    db: AsyncSession,
    *,
    started_at: Optional[datetime] = None,
    container: bool = False,
    last_activity_at: Any = _FROM_WS,
) -> Any:
    """Return the room's open interval, opening one if there is none.

    ``started_at`` back-dates a new interval to when the work really began; it
    is clamped so it never overlaps the previous interval (no double billing)
    and never lies in the future. ``container=True`` marks the container as
    running from now on.

    An open interval that outlived an unobserved gap is closed there and a new
    one opened (:func:`close_if_unobserved_gap`). So call this before stamping
    fresh presence on ``ws``; a caller that has already stamped
    ``ws.last_activity_at`` passes the previous value. Stages changes only —
    the caller commits.
    """
    now = _now()
    usage = await _open_row(ws, db)
    if usage is not None:
        # A container the Hub still believes runs carries on into the new
        # interval; its stretch before the gap is folded into the old one.
        had_container = usage.container_started_at is not None
        gap_stop = await close_if_unobserved_gap(
            usage, ws, db, now=now, last_activity_at=last_activity_at
        )
        if gap_stop is not None:
            return _open_new_row(
                ws, db, start=now,
                container_at=now if (container or had_container) else None,
            )
    if usage is None:
        start = min(_aware(started_at) or now, now)
        last_stop = await _last_stopped_at(ws, db)
        if last_stop is not None and start < last_stop:
            start = last_stop
        return _open_new_row(ws, db, start=start, container_at=now if container else None)
    if container and usage.container_started_at is None:
        usage.container_started_at = now
        db.add(usage)
    return usage


async def touch_presence(room_id: UUID, since: Optional[datetime] = None) -> bool:
    """Record that a member has the room open, and make sure it is billed.

    ``since`` is when the view opened. A view older than the room's closing
    (core/room_close.py: the stop button, the hour hard stop) is not
    presence any more: nothing is stamped or billed, and False tells the
    caller to shut it.
    """
    from app.core.room_close import closed_since
    from app.core.workspace_lifecycle import _get_or_create_workspace
    from app.db.session import engine
    from sqlalchemy.ext.asyncio import async_sessionmaker

    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db:
        ws = await _get_or_create_workspace(room_id, db)
        if closed_since(getattr(ws, "closed_at", None), since):
            await db.rollback()
            return False
        # Before the stamp: the presence this beat replaces says when the room
        # was last watched, and a gap since then is not billed.
        await ensure_usage_open(ws, db)
        ws.presence_at = _now()
        db.add(ws)
        await db.commit()
    return True


async def presence_beat(
    room_id: UUID,
    websocket: Any = None,
    since: Optional[datetime] = None,
) -> None:
    """Heartbeat for one open room socket; cancelled when the socket closes.

    Stored in Postgres rather than in the socket manager so every replica and
    the reconciler see it, and so a crashed process stops beating on its own.
    When the room is closed after the socket opened, the beat tells the tab
    and shuts the socket: a view left open must not keep the meter running.
    """
    while True:
        try:
            if not await touch_presence(room_id, since):
                await _shut_view(room_id, websocket)
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("billing: presence beat failed for room %s: %s", room_id, exc)
        await asyncio.sleep(PRESENCE_BEAT_SEC)


async def _shut_view(room_id: UUID, websocket: Any) -> None:
    if websocket is None:
        return
    from app.core.workspace_lifecycle import room_closed_frame

    try:
        await asyncio.wait_for(
            websocket.send_json(room_closed_frame(room_id, "closed", None)), timeout=5
        )
    except Exception:
        pass
    try:
        await asyncio.wait_for(websocket.close(code=4409), timeout=1)
    except Exception:
        pass


async def open_intervals_for_working_rooms(db: AsyncSession) -> int:
    """Open an interval for every room that works without one.

    Catches work that started anywhere — a task dispatched by the router, a
    ChatMaster session, an agent continuing after the browser closed — without
    hooking each of those paths: the interval is back-dated to the start of
    the work, so the one-tick delay costs nothing in accuracy.

    Only work that began after the room's previous interval counts as a reason
    to reopen. The previous interval ended either because work stopped (then
    anything live now is new) or by the 24 h stale guard on a stuck session —
    and reopening on that same stuck row would defeat the guard.
    """
    from app.core.workspace_lifecycle import (
        WORKSPACE_IDLE_TIMEOUT_SEC,
        _get_or_create_workspace,
    )
    from app.models.session.base import Session
    from app.models.task import Task, TaskState
    from app.models.workspace import Workspace, WorkspaceUsage

    now = _now()
    fresh_after = now - timedelta(seconds=PRESENCE_FRESH_SEC)

    open_ws = select(Workspace.room_id).join(
        WorkspaceUsage, WorkspaceUsage.workspace_id == Workspace.id
    ).where(WorkspaceUsage.stopped_at.is_(None))

    session_rows = (
        await db.execute(
            select(Session.room_id, func.min(Session.started_at))
            .where(
                Session.closed_at.is_(None),
                Session.kind.not_in(NON_BILLING_SESSION_KINDS),
                # Same notion of a working session as room_has_live_work: an
                # open row with no recent activity is a leak, not work.
                func.coalesce(Session.last_activity_at, Session.started_at)
                >= now - timedelta(seconds=WORKSPACE_IDLE_TIMEOUT_SEC),
                Session.room_id.not_in(open_ws),
            )
            .group_by(Session.room_id)
        )
    ).all()
    task_rows = (
        await db.execute(
            select(Task.room_id, func.min(Task.updated_at))
            .where(
                Task.state.in_([TaskState.IN_PROGRESS, TaskState.PAUSING, TaskState.RESUMING]),
                Task.room_id.not_in(open_ws),
            )
            .group_by(Task.room_id)
        )
    ).all()
    present_rows = (
        await db.execute(
            select(Workspace.room_id).where(
                Workspace.presence_at >= fresh_after,
                Workspace.room_id.not_in(open_ws),
            )
        )
    ).all()

    # room_id -> (earliest session start, earliest task activity, present)
    evidence: dict[UUID, list] = {}
    for room_id, started in session_rows:
        evidence.setdefault(room_id, [None, None, False])[0] = _aware(started)
    for room_id, updated in task_rows:
        evidence.setdefault(room_id, [None, None, False])[1] = _aware(updated)
    for (room_id,) in present_rows:
        evidence.setdefault(room_id, [None, None, False])[2] = True

    opened = 0
    for room_id, (session_start, task_touch, present) in evidence.items():
        if room_id is None:
            continue
        ws = await _get_or_create_workspace(room_id, db)
        if await _open_row(ws, db) is not None:
            continue
        last_stop = await _last_stopped_at(ws, db)
        candidates = [
            t for t in (session_start, task_touch)
            if t is not None and (last_stop is None or t > last_stop)
        ]
        if present:
            candidates.append(now)
        if not candidates:
            continue
        start = max(min(candidates), now - timedelta(seconds=MAX_BACKDATE_SEC))
        await ensure_usage_open(ws, db, started_at=start)
        opened += 1
    if opened:
        await db.commit()
    return opened
