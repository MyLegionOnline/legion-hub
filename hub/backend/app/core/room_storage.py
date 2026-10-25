"""Room files quota, level 1: accounting in the app (internal design notes §5, T3).

The quota ``room_storage_quota_mb`` covers every room directory of the owner
on the volume — sources, ``.git``, dependencies, uploads (§5.1). It is
measured by ``workspaced`` (``GET /workspaces/{id}/disk-usage``) and stored
on ``workspace.disk_bytes``:

* when a room stops (:func:`schedule_room_measure` from the stop path);
* every :data:`MEASURE_INTERVAL_SEC` while it runs (reconciler,
  :func:`enforce_storage_quota`);
* after a write through the file API and after "Clean up dependencies".

Thresholds (§5.3): 80 % and 95 % warn (``/me/usage``, the banner in the
room, the settings bar). At 100 % writes through the file API are refused —
deletion and the clean button stay —, a stopped room does not start, new
agent work is not routed or dispatched, and the tasks running in the
owner's rooms are paused with :data:`PAUSE_REASON` and a notice to the room.
Once the owner is back under the quota — "Clean up dependencies", files
deleted, a bigger plan — those tasks resume by themselves
(:func:`resume_owner_storage_work`). At 80 % the owner also gets a letter,
once per hour window; Market sends it (:func:`notify_storage_warning`).

Between two measurements a process can still write gigabytes; the hard
limit is the kernel's (XFS project quota, T9). Quotas are the room owner's
(TR6: a room has one user).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

#: Event type of the room notice; rendered as a system line in the chat.
STORAGE_FULL_EVENT = "STORAGE_FULL"
#: The notice when space was freed and the paused tasks went on.
STORAGE_FREED_EVENT = "STORAGE_FREED"
#: Kind of the 80 % letter Market sends (``/internal/billing/hub/notice``).
WARN_NOTICE_KIND = "storage_warning"
#: ``task.pause_reason`` of tasks paused for a full disk, and the error code
#: of the refusals.
PAUSE_REASON = "storage_full"
#: How often a running room is re-measured (§5.2: every 10 minutes).
MEASURE_INTERVAL_SEC = 600
#: Never-measured rooms measured per reconciler tick.
BACKFILL_PER_TICK = 10
WARN_PERCENT = 80
CRITICAL_PERCENT = 95

_MIB = 1024 * 1024


@dataclass(frozen=True)
class RoomDisk:
    room_id: UUID
    name: str
    bytes: int
    measured_at: Optional[datetime]


@dataclass(frozen=True)
class StorageStatus:
    tier: str
    used_bytes: int
    quota_mb: Optional[int]          # None — no cap (selfhosted)
    rooms: list[RoomDisk] = field(default_factory=list)

    @property
    def quota_bytes(self) -> Optional[int]:
        return None if self.quota_mb is None else self.quota_mb * _MIB

    @property
    def percent(self) -> Optional[float]:
        if self.quota_bytes is None:
            return None
        if self.quota_bytes <= 0:
            return 100.0 if self.used_bytes > 0 else 0.0
        return self.used_bytes * 100.0 / self.quota_bytes

    @property
    def level(self) -> str:
        """ok | warn (≥ 80 %) | critical (≥ 95 %) | full (≥ 100 %)."""
        pct = self.percent
        if pct is None:
            return "ok"
        if self.full:
            return "full"
        if pct >= CRITICAL_PERCENT:
            return "critical"
        if pct >= WARN_PERCENT:
            return "warn"
        return "ok"

    @property
    def full(self) -> bool:
        quota = self.quota_bytes
        # A zero quota (expired plan) is not "full": its rooms are read-only
        # for other reasons (TR4) and must not be paused for the disk.
        return quota is not None and quota > 0 and self.used_bytes >= quota


@dataclass
class StorageReservation:
    """A short-lived byte reservation held while a file is being written.

    The measured disk value is intentionally still the source of truth for
    reporting.  Reservations close the race between two uploads that both
    observe the same last measurement.  They live only for the write request;
    the follow-up measurement records the actual volume size.
    """

    owner_id: Optional[UUID] = None
    bytes: int = 0
    _released: bool = False

    def release(self) -> None:
        if self._released or self.owner_id is None or self.bytes <= 0:
            return
        pending = _pending_reservations.get(self.owner_id, 0) - self.bytes
        if pending > 0:
            _pending_reservations[self.owner_id] = pending
        else:
            _pending_reservations.pop(self.owner_id, None)
        self._released = True


# The reservation is process-local because the upload itself is handled by a
# single backend worker.  The existing room measurement remains authoritative
# across workers/restarts; this guard covers the important in-process race and
# keeps the API additive without a schema migration in Stage 0.
_RESERVATION_LOCKS: dict[UUID, asyncio.Lock] = {}
_pending_reservations: dict[UUID, int] = {}


def _reservation_lock(owner_id: UUID) -> asyncio.Lock:
    lock = _RESERVATION_LOCKS.get(owner_id)
    if lock is None:
        lock = asyncio.Lock()
        _RESERVATION_LOCKS[owner_id] = lock
    return lock


def pending_storage_reservation(owner_id: UUID) -> int:
    """Return bytes tentatively held by uploads for this room owner."""
    return max(0, int(_pending_reservations.get(owner_id, 0)))


def _quota_mb(raw: Any) -> Optional[int]:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    return int(raw)


async def storage_status(owner_id: UUID, db: AsyncSession) -> StorageStatus:
    """The owner's room files against the plan's ``room_storage_quota_mb``."""
    from app.core.workspace_quota import get_effective_limits
    from app.models.room import Room
    from app.models.workspace import Workspace

    limits = await get_effective_limits(owner_id, db)
    rows = (
        await db.execute(
            select(Room.id, Room.name, Workspace.disk_bytes, Workspace.disk_measured_at)
            .outerjoin(Workspace, Workspace.room_id == Room.id)
            .where(Room.creator_id == owner_id, Room.deletion_started_at.is_(None))
        )
    ).all()
    rooms = [
        RoomDisk(room_id, name or "", int(disk or 0), measured)
        for room_id, name, disk, measured in rows
    ]
    rooms.sort(key=lambda r: r.bytes, reverse=True)
    return StorageStatus(
        tier=str(limits.get("plan_tier") or ""),
        used_bytes=sum(r.bytes for r in rooms),
        quota_mb=_quota_mb(limits.get("room_storage_quota_mb")),
        rooms=rooms,
    )


def _gib(n_bytes: float) -> str:
    return f"{n_bytes / (1024 * _MIB):.2f}".rstrip("0").rstrip(".") + " GiB"


def refusal_text(status: StorageStatus) -> str:
    return (
        f"The room files quota is exhausted: {_gib(status.used_bytes)} of "
        f"{_gib(status.quota_bytes or 0)} used on the '{status.tier}' plan. Free up "
        "space: \"Clean up dependencies\" in \"Settings → Plan and limits\" or "
        "delete unneeded files."
    )


def usage_payload(status: StorageStatus) -> dict[str, Any]:
    """The "Room files" block of ``/me/usage``."""
    return {
        "storage_used_mb": round(status.used_bytes / _MIB, 1),
        "storage_quota_mb": status.quota_mb,
        "storage_percent": None if status.percent is None else round(status.percent, 1),
        "storage_level": status.level,
        "storage_full": status.full,
        "storage_rooms": [
            {
                "room_id": str(r.room_id),
                "name": r.name,
                "used_mb": round(r.bytes / _MIB, 1),
                "measured_at": r.measured_at.isoformat() if r.measured_at else None,
            }
            for r in status.rooms
        ],
    }


async def _room_owner(room_id: UUID, db: AsyncSession) -> Optional[UUID]:
    from app.models.room import Room

    owner = (
        await db.execute(select(Room.creator_id).where(Room.id == room_id))
    ).scalar_one_or_none()
    return owner if isinstance(owner, UUID) else None


async def room_storage_refusal(room_id: UUID, db: AsyncSession) -> Optional[str]:
    """Why ``room_id`` may not take new writes or work, or None when it may.

    Fails open, like the hour guard: the reconciler re-checks within a minute.
    A room the disk watchdog stopped is refused too, until the volume has
    room again (core/disk_watch.py, T0.16).
    """
    from app.core.disk_watch import disk_stop_refusal

    stopped = await disk_stop_refusal(room_id, db)
    if stopped is not None:
        return stopped
    try:
        owner = await _room_owner(room_id, db)
        if owner is None:
            return None
        status = await storage_status(owner, db)
        pending = pending_storage_reservation(owner)
        if pending:
            status = StorageStatus(
                tier=status.tier,
                used_bytes=status.used_bytes + pending,
                quota_mb=status.quota_mb,
                rooms=status.rooms,
            )
    except Exception as exc:
        logger.warning(
            "room_storage: quota check failed for room %s (%s: %s); allowing",
            room_id, type(exc).__name__, exc,
        )
        return None
    return refusal_text(status) if status.full else None


async def require_room_storage(room_id: UUID, db: AsyncSession) -> None:
    """File API gate: 402 ``storage_full`` when the owner's quota is used up."""
    text = await room_storage_refusal(room_id, db)
    if text is not None:
        raise HTTPException(
            status_code=402, detail={"error_code": PAUSE_REASON, "message": text}
        )


async def reserve_room_storage(
    room_id: UUID,
    db: AsyncSession,
    bytes_to_reserve: int,
    *,
    owner_id: Optional[UUID] = None,
) -> StorageReservation:
    """Reserve upload bytes before the volume write begins.

    ``disk_bytes`` is deliberately a sampled value.  The owner lock and the
    pending counter make concurrent uploads consume the remaining measured
    quota together, while the reservation is released after the write and the
    normal asynchronous measurement records the real usage.

    A missing owner is treated as a no-op only for compatibility with the
    standalone workspaced/file-router tests; a real room always has a UUID
    creator.  Negative values are rejected instead of silently weakening the
    quota check.
    """
    amount = int(bytes_to_reserve)
    if amount < 0:
        raise ValueError("bytes_to_reserve must be non-negative")
    if amount == 0:
        return StorageReservation()

    resolved_owner = owner_id or await _room_owner(room_id, db)
    if not isinstance(resolved_owner, UUID):
        return StorageReservation()

    async with _reservation_lock(resolved_owner):
        status = await storage_status(resolved_owner, db)
        pending = pending_storage_reservation(resolved_owner)
        quota = status.quota_bytes
        projected = status.used_bytes + pending + amount
        # A zero quota is a real cap for an expired plan; self-hosted has None.
        if quota is not None and (quota <= 0 or projected > quota):
            projected_status = StorageStatus(
                tier=status.tier,
                used_bytes=projected,
                quota_mb=status.quota_mb,
                rooms=status.rooms,
            )
            raise HTTPException(
                status_code=402,
                detail={"error_code": PAUSE_REASON, "message": refusal_text(projected_status)},
            )
        _pending_reservations[resolved_owner] = pending + amount
        return StorageReservation(owner_id=resolved_owner, bytes=amount)


# ---------- Measuring -----------------------------------------------------------


#: How long measurements are kept for the disk watchdog's growth rate.
SAMPLE_KEEP_SEC = 2 * 3600


async def record_room_disk(room_id: UUID, n_bytes: int, db: AsyncSession) -> None:
    from sqlalchemy import delete

    from app.core.workspace_lifecycle import _get_or_create_workspace
    from app.models.workspace import WorkspaceDiskSample

    now = datetime.now(timezone.utc)
    ws = await _get_or_create_workspace(room_id, db)
    ws.disk_bytes = max(0, int(n_bytes))
    ws.disk_measured_at = now
    db.add(ws)
    # The history the disk watchdog reads (core/disk_watch.py, T0.16).
    db.add(WorkspaceDiskSample(room_id=room_id, measured_at=now, disk_bytes=ws.disk_bytes))
    await db.execute(
        delete(WorkspaceDiskSample)
        .where(
            WorkspaceDiskSample.room_id == room_id,
            WorkspaceDiskSample.measured_at < now - timedelta(seconds=SAMPLE_KEEP_SEC),
        )
        .execution_options(synchronize_session=False)
    )
    await db.commit()


async def measure_room(room_id: UUID, db: AsyncSession) -> Optional[int]:
    """Measure one room now and store it; None when workspaced is unavailable."""
    from app.core import workspaced_client as wsd
    from app.db.release import release_db

    # A big tree takes seconds to walk: no pool connection held meanwhile.
    await release_db(db, reason="room_storage/measure")
    try:
        result = await wsd.measure_workspace_disk(str(room_id))
    except Exception as exc:
        logger.warning("room_storage: measuring room %s failed: %s", room_id, exc)
        return None
    if not result.get("exists", True):
        n_bytes = 0
    else:
        n_bytes = int(result.get("bytes") or 0)
    await record_room_disk(room_id, n_bytes, db)
    return n_bytes


_pending_measures: set[UUID] = set()
_measure_tasks: set[asyncio.Task] = set()


def schedule_room_measure(room_id: UUID) -> None:
    """Measure ``room_id`` in the background with its own session.

    One measurement per room at a time; a request made while one runs is
    covered by it closely enough (the next stop or tick re-measures).
    """
    if room_id in _pending_measures:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _pending_measures.add(room_id)

    async def _run() -> None:
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker

            from app.db.session import engine

            factory = async_sessionmaker(engine, expire_on_commit=False)
            async with factory() as db:
                await measure_room(room_id, db)
        except Exception as exc:
            logger.warning("room_storage: background measure of %s failed: %s", room_id, exc)
        finally:
            _pending_measures.discard(room_id)

    task = loop.create_task(_run())
    _measure_tasks.add(task)
    task.add_done_callback(_measure_tasks.discard)


async def clean_room_dependencies(room_id: UUID, db: AsyncSession) -> dict[str, Any]:
    """The "Clean up dependencies" button for one room."""
    from app.core import workspaced_client as wsd
    from app.db.release import release_db

    await release_db(db, reason="room_storage/clean_dependencies")
    result = await wsd.clean_workspace_dependencies(str(room_id))
    await record_room_disk(room_id, int(result.get("bytes") or 0), db)
    owner = await _room_owner(room_id, db)
    if owner is not None:
        schedule_owner_resume(owner)
    return result


# ---------- Resuming after space was freed --------------------------------------


async def _rooms_with_paused_work(db: AsyncSession, owner_id: Optional[UUID] = None) -> list[tuple[UUID, UUID]]:
    """(room, owner) pairs with tasks paused for a full disk."""
    from app.models.room import Room
    from app.models.task import Task, TaskState

    stmt = (
        select(Task.room_id, Room.creator_id)
        .join(Room, Room.id == Task.room_id)
        .where(Task.state == TaskState.PAUSED, Task.pause_reason == PAUSE_REASON)
        .distinct()
    )
    if owner_id is not None:
        stmt = stmt.where(Room.creator_id == owner_id)
    return [(room, owner) for room, owner in (await db.execute(stmt)).all() if owner is not None]


async def resume_owner_storage_work(owner_id: UUID, db: AsyncSession) -> int:
    """Resume the owner's tasks paused for the disk, if the quota allows now.

    Each task goes through the dispatch preflight, so one still refused (out
    of hours, a room the disk watchdog stopped) stays paused. Returns rooms
    where anything resumed.
    """
    from app.core.events import emit_event
    from app.core.workspace_lifecycle import resume_paused_tasks

    status = await storage_status(owner_id, db)
    if status.full:
        return 0
    acted = 0
    for room_id, _owner in await _rooms_with_paused_work(db, owner_id):
        try:
            resumed = await resume_paused_tasks(
                room_id, db, pause_reason=PAUSE_REASON, source="storage_freed",
            )
        except Exception as exc:
            await db.rollback()
            logger.warning("room_storage: resuming tasks of room %s failed: %s", room_id, exc)
            continue
        if resumed:
            acted += 1
            await emit_event(
                room_id, STORAGE_FREED_EVENT,
                "Space for files was freed — tasks stopped because of the quota were resumed.",
                db,
                metadata={"resumed_task_ids": [str(task_id) for task_id in resumed]},
            )
    return acted


_resume_tasks: set[asyncio.Task] = set()


def schedule_owner_resume(owner_id: UUID) -> None:
    """:func:`resume_owner_storage_work` in the background with its own
    session: the clean button answers without waiting for the dispatches."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run() -> None:
        try:
            from sqlalchemy.ext.asyncio import async_sessionmaker

            from app.db.session import engine

            factory = async_sessionmaker(engine, expire_on_commit=False)
            async with factory() as db:
                await resume_owner_storage_work(owner_id, db)
        except Exception as exc:
            logger.warning("room_storage: background resume for %s failed: %s", owner_id, exc)

    task = loop.create_task(_run())
    _resume_tasks.add(task)
    task.add_done_callback(_resume_tasks.discard)


# ---------- The 80 % letter -------------------------------------------------------

#: (owner, window start) already handed to Market by this process — saves a
#: call per tick; Market's notice key is what makes the letter go once.
_warned: set[tuple[UUID, str]] = set()

#: Plugin event: ``async handler(account_id, kind, ref, params) -> bool`` mails
#: the account a letter once per ``ref`` and says whether it was taken.
USER_NOTICE_EVENT = "user_notice"


async def notify_storage_warning(owner_id: UUID, status: StorageStatus, db: AsyncSession) -> bool:
    """§5.3: at 80 % a letter, once per hour window. Hub has no mail of its
    own; the cloud plugin's ``user_notice`` handler asks Market to send it to
    the owner's Legion account (Resend there). False when nothing was asked
    of Market (below 80 %, no account, no plugin = self-hosted)."""
    from app.core.workspace_quota import get_hours_window
    from app.models.user import User
    from app.plugins import get_registry

    handlers = get_registry().handlers_for(USER_NOTICE_EVENT)
    # No cap (self-hosted) or a zero one (expired plan, read-only anyway).
    if not status.quota_bytes or status.level == "ok" or not handlers:
        return False
    window_start, _ = await get_hours_window(owner_id, db)
    ref = window_start.strftime("%Y-%m-%d")
    if (owner_id, ref) in _warned:
        return False
    account_id = (
        await db.execute(select(User.legion_account_id).where(User.id == owner_id))
    ).scalar_one_or_none()
    if account_id is None:
        _warned.add((owner_id, ref))
        return False
    params = {
        "used_mb": round(status.used_bytes / _MIB, 1),
        "quota_mb": status.quota_mb,
        "percent": round(status.percent or 0, 1),
        "tier": status.tier,
    }
    delivered = False
    for handler in handlers:
        delivered = bool(await handler(account_id, WARN_NOTICE_KIND, ref, params)) or delivered
    if delivered:
        _warned.add((owner_id, ref))
    return delivered


# ---------- The reconciler pass -------------------------------------------------


def _stale(measured_at: Optional[datetime], now: datetime) -> bool:
    if measured_at is None:
        return True
    if measured_at.tzinfo is None:
        measured_at = measured_at.replace(tzinfo=timezone.utc)
    return now - measured_at >= timedelta(seconds=MEASURE_INTERVAL_SEC)


async def _pause_owner_work(owner_id: UUID, text: str, db: AsyncSession) -> int:
    """Pause the live tasks in the owner's rooms; a notice where anything paused."""
    from app.core.events import emit_event
    from app.core.room_close import _live_task_states
    from app.core.workspace_lifecycle import pause_workspace_tasks
    from app.models.room import Room
    from app.models.task import Task

    rooms = (
        await db.execute(
            select(Task.room_id)
            .join(Room, Room.id == Task.room_id)
            .where(Room.creator_id == owner_id, Task.state.in_(_live_task_states()))
            .distinct()
        )
    ).scalars().all()
    acted = 0
    for room_id in rooms:
        try:
            paused = await pause_workspace_tasks(room_id, db, pause_reason=PAUSE_REASON)
            await db.commit()
        except Exception as exc:
            await db.rollback()
            logger.warning("room_storage: pausing tasks of room %s failed: %s", room_id, exc)
            continue
        if paused:
            acted += 1
            await emit_event(
                room_id, STORAGE_FULL_EVENT, text, db,
                metadata={
                    "reason": PAUSE_REASON,
                    "paused_task_ids": [str(task_id) for task_id in paused],
                },
            )
    return acted


async def enforce_storage_quota(db: AsyncSession, *, now: Optional[datetime] = None) -> int:
    """Re-measure running rooms due for it, then pause the work of every
    owner whose room files are at or over the quota. Returns rooms paused."""
    from app.models.room import Room
    from app.models.workspace import Workspace

    from app.core.room_close import _live_task_states
    from app.models.task import Task

    now = now or datetime.now(timezone.utc)
    # Rooms that can be writing: a running container, or a task at work
    # (the file tools need no container).
    live_rooms = select(Task.room_id).where(Task.state.in_(_live_task_states()))
    candidates = (
        await db.execute(
            select(Room.id, Room.creator_id, Workspace.disk_measured_at)
            .outerjoin(Workspace, Workspace.room_id == Room.id)
            .where((Workspace.status == "running") | Room.id.in_(live_rooms))
        )
    ).all()
    owners: set[UUID] = set()
    for room_id, owner, measured_at in candidates:
        if owner is not None:
            owners.add(owner)
        if _stale(measured_at, now):
            await measure_room(room_id, db)

    # Rooms never measured (created before T3, or whose stop-time measure
    # was lost) — a few per tick, so the totals become complete.
    unmeasured = (
        await db.execute(
            select(Room.id, Room.creator_id)
            .outerjoin(Workspace, Workspace.room_id == Room.id)
            .where(Room.deletion_started_at.is_(None), Workspace.disk_measured_at.is_(None))
            .limit(BACKFILL_PER_TICK)
        )
    ).all()
    for room_id, owner in unmeasured:
        if await measure_room(room_id, db) is not None and owner is not None:
            owners.add(owner)

    # Owners whose disk-paused tasks may go on: space freed by deleting files
    # or a bigger plan (the clean button resumes at once on its own).
    waiting = {owner for _room, owner in await _rooms_with_paused_work(db)}

    acted = 0
    for owner in owners | waiting:
        status = await storage_status(owner, db)
        try:
            await notify_storage_warning(owner, status, db)
        except Exception as exc:
            logger.warning("room_storage: 80 %% letter for %s failed: %s", owner, exc)
        if status.full:
            if owner in owners:
                acted += await _pause_owner_work(owner, refusal_text(status), db)
        elif owner in waiting:
            await resume_owner_storage_work(owner, db)
    if acted:
        logger.info("room_storage: paused work in %d room(s) for a full disk quota", acted)
    return acted


# ---------- Kernel-level ENOSPC (T9, §5.3) --------------------------------------

#: What a tool prints when the XFS project quota (or a full tmpfs) refuses a write.
KERNEL_ENOSPC_MARKERS = ("No space left on device", "Disk quota exceeded")

KERNEL_ENOSPC_HINT = (
    "[storage: the write was refused with 'no space left'. If the path is under "
    "/workspace, the room files quota is exhausted — free space (delete files, "
    "remove node_modules/venv/caches; the owner can use \"Clean up dependencies\") "
    "before retrying. The room is being re-measured; if the quota is confirmed "
    "full, running tasks are paused until space is freed. /tmp is RAM-backed and "
    "separate.]"
)


def kernel_enospc_hint(room_id: UUID, exit_code: Any, *outputs: str) -> str:
    """The note for the agent when a command failed with ENOSPC, else ``""``.

    The kernel limit sits 10 % above the plan quota, so the app measurement
    that pauses the work (:func:`enforce_storage_quota`) usually lags behind.
    Re-measure the room right away instead of waiting for the next
    ten-minute window; the reconciler then pauses the owner's tasks.
    """
    if exit_code in (0, None):
        return ""
    if not any(marker in (text or "") for text in outputs for marker in KERNEL_ENOSPC_MARKERS):
        return ""
    schedule_room_measure(room_id)
    return KERNEL_ENOSPC_HINT
