"""Disk watchdog of the rooms' data volume (internal design notes T0.16,
internal design notes §5.2, T10).

Users' quotas add up to ~6× the volume (§6.3), which is normal while
someone watches the actual fill. Every reconciler tick (60 s):

* ``workspaced`` reports the filesystem holding the rooms
  (``GET /host/data-volume``); the size and free bytes go to Prometheus,
  where ``alerting_rules.yml`` alerts at 60 % and 80 % full (grow the
  volume) and at < 15 % free;
* below :data:`STOP_FREE_PERCENT` free the watchdog acts itself: it stops
  the room whose files grew most over the last hour — its tasks pause with
  :data:`PAUSE_REASON`, its container stops, a notice goes to the room. The
  open view stays (the room is readable; the Stop button does the same).
  ``workspace.disk_stopped_at`` marks it: until the volume is back above
  :data:`STOP_FREE_PERCENT` free the room does not start, takes no new agent
  work and no writes (:func:`disk_stop_refusal`); deleting files and "Clean up
  dependencies" stay. The first pass that finds the volume above the threshold
  lifts every mark.

Stopping a room frees nothing, it only stops the growth, so the next stop
waits :data:`STOP_COOLDOWN_SEC` and takes only a room that grew at least
:data:`MIN_GROWTH_BYTES` in the hour: rooms writing a few megabytes are not
the problem, and stopping them one by one would only stop everyone.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

ALERT_FREE_PERCENT = 15.0
STOP_FREE_PERCENT = 5.0
#: The growth that picks the room to stop is measured over this window.
GROWTH_WINDOW_SEC = 3600
#: Below the stop threshold, rooms measured longer ago than this are
#: re-measured first — the regular cadence is 10 minutes.
FRESH_SEC = 120
#: At most this many rooms re-measured per tick (a walk can take seconds).
EMERGENCY_MEASURE_MAX = 20
MIN_GROWTH_BYTES = int(os.getenv("DISK_WATCH_MIN_GROWTH_MB", "64")) * 1024 * 1024
STOP_COOLDOWN_SEC = 300
#: ``task.pause_reason`` / ``workspace_usage.stop_reason`` of the stop.
PAUSE_REASON = "disk_emergency"
#: Event type of the room notice; rendered as a system line in the chat.
DISK_EMERGENCY_EVENT = "DISK_EMERGENCY"

_GIB = 1024 ** 3

_last_level: Optional[str] = None
_last_stop_at: Optional[datetime] = None


@dataclass(frozen=True)
class VolumeState:
    size_bytes: int
    free_bytes: int

    @property
    def free_percent(self) -> float:
        if self.size_bytes <= 0:
            return 0.0
        return self.free_bytes * 100.0 / self.size_bytes

    @property
    def level(self) -> str:
        """ok | low (< 15 % free, alert) | emergency (< 5 % free, stop a room)."""
        pct = self.free_percent
        if pct < STOP_FREE_PERCENT:
            return "emergency"
        if pct < ALERT_FREE_PERCENT:
            return "low"
        return "ok"


def growth_by_room(
    samples: Iterable[tuple[UUID, datetime, int]],
) -> dict[UUID, int]:
    """Bytes each room grew between its first and last sample."""
    first: dict[UUID, tuple[datetime, int]] = {}
    last: dict[UUID, tuple[datetime, int]] = {}
    for room_id, at, n_bytes in samples:
        if room_id not in first or at < first[room_id][0]:
            first[room_id] = (at, n_bytes)
        if room_id not in last or at >= last[room_id][0]:
            last[room_id] = (at, n_bytes)
    return {room_id: last[room_id][1] - first[room_id][1] for room_id in last}


def pick_room_to_stop(
    growth: dict[UUID, int], *, min_growth: int = MIN_GROWTH_BYTES,
) -> Optional[tuple[UUID, int]]:
    """The fastest-growing room, when it grew at least ``min_growth``."""
    if not growth:
        return None
    room_id, grew = max(growth.items(), key=lambda item: item[1])
    return (room_id, grew) if grew >= min_growth else None


def refusal_text() -> str:
    return (
        "The room was stopped by the disk watchdog: the server disk is almost full and its "
        "files grew faster than any other's. Launching and agent work will return when "
        "space is freed on the server. You can already delete unneeded files or \"Clean up "
        "dependencies\"."
    )


async def disk_stop_refusal(room_id: UUID, db: AsyncSession) -> Optional[str]:
    """Why ``room_id`` may not start or take new work, or None when it may.

    Fails open: a failed check lets the room through.
    """
    from app.models.workspace import Workspace

    try:
        stopped_at = (
            await db.execute(
                select(Workspace.disk_stopped_at).where(Workspace.room_id == room_id)
            )
        ).scalar_one_or_none()
    except Exception as exc:
        logger.warning(
            "disk_watch: stop-mark check failed for room %s (%s: %s); allowing",
            room_id, type(exc).__name__, exc,
        )
        return None
    return refusal_text() if isinstance(stopped_at, datetime) else None


async def _lift_stop_marks(db: AsyncSession) -> int:
    from sqlalchemy import update

    from app.models.workspace import Workspace

    result = await db.execute(
        update(Workspace)
        .where(Workspace.disk_stopped_at.is_not(None))
        .values(disk_stopped_at=None)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    lifted = int(result.rowcount or 0)
    if lifted:
        logger.info("disk_watch: volume has room again; %d stopped room(s) may start", lifted)
    return lifted


def _gib(n_bytes: float) -> str:
    return f"{n_bytes / _GIB:.2f}".rstrip("0").rstrip(".") + " GiB"


def notice_text(state: VolumeState, grew: int) -> str:
    return (
        f"The server disk is almost full ({state.free_percent:.1f} % free). "
        f"The room was stopped: its files grew by {_gib(grew)} over the last hour — "
        "faster than any other on the server. Tasks are paused. Before continuing, delete "
        "unneeded files or press \"Clean up dependencies\" in \"Settings → Plan and limits\"."
    )


def _aware(at: Optional[datetime]) -> Optional[datetime]:
    if at is not None and at.tzinfo is None:
        return at.replace(tzinfo=timezone.utc)
    return at


def _publish(state: VolumeState, now: datetime) -> None:
    from app.core.metrics import (
        data_volume_checked_timestamp_seconds,
        data_volume_free_bytes,
        data_volume_size_bytes,
    )

    data_volume_size_bytes.set(state.size_bytes)
    data_volume_free_bytes.set(state.free_bytes)
    data_volume_checked_timestamp_seconds.set(now.timestamp())


def _log_level(state: VolumeState) -> None:
    global _last_level
    level = state.level
    if level == _last_level:
        return
    message = "disk_watch: data volume %s — %.1f %% free (%s of %s)"
    args = (level, state.free_percent, _gib(state.free_bytes), _gib(state.size_bytes))
    if level == "ok":
        logger.info(message, *args)
    else:
        logger.warning(message, *args)
    _last_level = level


async def _candidate_rooms(db: AsyncSession) -> list[tuple[UUID, Optional[datetime]]]:
    """Rooms that can be writing: a running container or a task at work."""
    from app.core.room_close import _live_task_states
    from app.models.room import Room
    from app.models.task import Task
    from app.models.workspace import Workspace

    live_rooms = select(Task.room_id).where(Task.state.in_(_live_task_states()))
    rows = (
        await db.execute(
            select(Room.id, Workspace.disk_measured_at)
            .outerjoin(Workspace, Workspace.room_id == Room.id)
            .where(
                Room.deletion_started_at.is_(None),
                (Workspace.status == "running") | Room.id.in_(live_rooms),
            )
        )
    ).all()
    return [(room_id, _aware(measured_at)) for room_id, measured_at in rows]


async def _stop_fastest_room(
    state: VolumeState, db: AsyncSession, now: datetime,
) -> Optional[dict[str, Any]]:
    from app.core.metrics import disk_watch_room_stops_total
    from app.core.room_close import close_room
    from app.core.room_storage import measure_room
    from app.models.workspace import WorkspaceDiskSample

    global _last_stop_at
    if _last_stop_at is not None and now - _last_stop_at < timedelta(seconds=STOP_COOLDOWN_SEC):
        return None

    rooms = await _candidate_rooms(db)
    if not rooms:
        logger.warning("disk_watch: volume below %.0f %% free and no room is working", STOP_FREE_PERCENT)
        return None
    # The regular cadence is 10 minutes; a room filling the disk now may not
    # be in the history yet. Least recently measured first.
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    stale = sorted(
        (r for r in rooms if r[1] is None or now - r[1] >= timedelta(seconds=FRESH_SEC)),
        key=lambda r: r[1] or epoch,
    )
    for room_id, _ in stale[:EMERGENCY_MEASURE_MAX]:
        await measure_room(room_id, db)

    room_ids = [room_id for room_id, _ in rooms]
    samples = (
        await db.execute(
            select(
                WorkspaceDiskSample.room_id,
                WorkspaceDiskSample.measured_at,
                WorkspaceDiskSample.disk_bytes,
            ).where(
                WorkspaceDiskSample.room_id.in_(room_ids),
                WorkspaceDiskSample.measured_at >= now - timedelta(seconds=GROWTH_WINDOW_SEC),
            )
        )
    ).all()
    growth = growth_by_room(
        (room_id, _aware(at), int(n_bytes or 0)) for room_id, at, n_bytes in samples
    )
    picked = pick_room_to_stop(growth)
    if picked is None:
        logger.warning(
            "disk_watch: volume below %.0f %% free, but no working room grew %s in the "
            "last hour — nothing stopped; grow the volume",
            STOP_FREE_PERCENT, _gib(MIN_GROWTH_BYTES),
        )
        return None

    from sqlalchemy import update

    from app.models.workspace import Workspace

    room_id, grew = picked
    # Marked first, so a start racing the stop is already refused.
    await db.execute(
        update(Workspace)
        .where(Workspace.room_id == room_id)
        .values(disk_stopped_at=now)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    result = await close_room(
        room_id, db,
        reason=PAUSE_REASON,
        pause_reason=PAUSE_REASON,
        notice=notice_text(state, grew),
        notice_event=DISK_EMERGENCY_EVENT,
        close_view=False,
    )
    _last_stop_at = now
    disk_watch_room_stops_total.inc()
    logger.warning(
        "disk_watch: stopped room %s — grew %s in the last hour, volume %.1f %% free",
        room_id, _gib(grew), state.free_percent,
    )
    return {"room_id": room_id, "grew_bytes": grew, **result}


async def watch_data_volume(
    db: AsyncSession, *, now: Optional[datetime] = None,
) -> dict[str, Any]:
    """One watchdog pass; returns the level and the room stopped, if any."""
    from app.core import workspaced_client as wsd
    from app.db.release import release_db

    now = now or datetime.now(timezone.utc)
    await release_db(db, reason="disk_watch/probe")
    try:
        raw = await wsd.get_data_volume()
        state = VolumeState(int(raw["size_bytes"]), int(raw["free_bytes"]))
    except Exception as exc:
        # The checked-timestamp gauge goes stale and DataVolumeWatchStale fires.
        logger.warning("disk_watch: data volume probe failed: %s", exc)
        return {"level": "unknown", "stopped": None}

    _publish(state, now)
    _log_level(state)
    stopped = None
    if state.level == "emergency":
        stopped = await _stop_fastest_room(state, db, now)
    else:
        await _lift_stop_marks(db)
    return {"level": state.level, "free_percent": state.free_percent, "stopped": stopped}
