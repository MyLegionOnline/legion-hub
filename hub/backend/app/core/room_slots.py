"""Parallel-task slots of a room: how many tasks may run at once.

internal design notes, stage E1. Every running
task of a room shares the room's one workspace container, so five tasks that
each start a dev server and a build exhaust its memory (a dev room, 2026-11:
five Next.js tasks, "the shared .next is mid-churn from parallel sessions").

Owner's decisions:

* the room's ``parallel_tasks_base`` (default 2) tasks always run side by side;
* a task beyond the base starts only while the room's memory is below
  ``ROOM_EXTRA_SLOT_MEM_PCT`` (60 %) and the kernel is not stalling on memory
  (PSI ``some avg10`` below ``ROOM_EXTRA_SLOT_PSI_SOME``); an unknown reading
  never admits an extra task;
* a task beyond the base also waits ``ROOM_EXTRA_SLOT_SETTLE_SEC`` (180 s)
  after the room's previous start: right after a start the reading does not
  yet show that task's dev server and build (E4: a cold room let four tasks in
  at 30 % and two parallel ``next build`` then filled it);
* never more than ``ROOM_PARALLEL_TASKS_CAP`` (4);
* only a task that is really running holds a slot (working state *and* a
  claimed dispatch) — one waiting for review, a checkpoint or the user does not;
* a slot is never taken back from a running task.

A task that gets no slot stays READY with ``queued_at`` set and the way it was
dispatched saved in ``queued_dispatch``; :func:`dispatch_queued` replays it
when a slot frees (a run finished, the base was raised) and the watchdog sweeps
the queue every cycle, which also picks it up after a restart. Time in the
queue is not run time: no session exists yet, so no tool timeout or run
deadline is ticking.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

TASK_QUEUED = "TASK_QUEUED"

DEFAULT_BASE = 2
PRESSURE_TIMEOUT_SEC = 8.0
SCHEDULE_DELAY_SEC = 1.0


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def slots_cap() -> int:
    return _env_int("ROOM_PARALLEL_TASKS_CAP", 4, 1, 16)


def extra_slot_mem_pct() -> float:
    return _env_float("ROOM_EXTRA_SLOT_MEM_PCT", 60.0)


def extra_slot_settle_sec() -> float:
    return _env_float("ROOM_EXTRA_SLOT_SETTLE_SEC", 180.0)


def extra_slot_psi_some() -> float:
    return _env_float("ROOM_EXTRA_SLOT_PSI_SOME", 10.0)


@dataclass(frozen=True)
class Admission:
    ok: bool
    busy: int
    base: int
    cap: int
    # base | memory | cap | memory_busy | memory_unknown | settling
    reason: str
    pct: Optional[float] = None


def decide(
    busy: int,
    base: int,
    cap: int,
    pressure: Optional[dict[str, Any]],
    since_last_start: Optional[float] = None,
) -> Admission:
    """Pure admission rule; ``pressure`` is only consulted beyond the base.

    ``since_last_start``: seconds since a task of the room last got a slot
    (``None``: none recorded).
    """
    base = max(1, min(base, cap))
    if busy < base:
        return Admission(True, busy, base, cap, "base")
    if busy >= cap:
        return Admission(False, busy, base, cap, "cap")
    if since_last_start is not None and since_last_start < extra_slot_settle_sec():
        return Admission(False, busy, base, cap, "settling")
    pct = (pressure or {}).get("pct")
    psi = (pressure or {}).get("psi_some_avg10")
    if pct is None:
        return Admission(False, busy, base, cap, "memory_unknown")
    if pct >= extra_slot_mem_pct() or (psi is not None and psi >= extra_slot_psi_some()):
        return Admission(False, busy, base, cap, "memory_busy", pct)
    return Admission(True, busy, base, cap, "memory", pct)


async def room_pressure(room_id: Any) -> Optional[dict[str, Any]]:
    """The room's memory pressure, or ``None`` when it cannot be read."""
    from app.core import workspaced_client as wsd

    try:
        payload = await asyncio.wait_for(
            wsd.get_room_pressure(str(room_id)), PRESSURE_TIMEOUT_SEC
        )
    except Exception as exc:  # unreachable workspaced, timeout, 4xx/5xx
        logger.info("room_slots: pressure of room %s unavailable: %s", room_id, exc)
        return None
    if not isinstance(payload, dict) or payload.get("status") != "running":
        return None
    return payload


def connector_agent_ids() -> Any:
    """Agents whose tasks run on an external connector, not in the workspace.

    Mirrors ``connector_session.resolve_executor``: executor "connector" *and*
    a linked connector.
    """
    from app.models.agent import Agent

    return select(Agent.id).where(
        Agent.executor == "connector", Agent.connector_id.is_not(None)
    )


def busy_clause(room_id: Any) -> Any:
    """A task holding one of the room's slots.

    A connector agent's task works on the user's machine, not in the room's
    container, so it neither holds nor needs a slot (E1 known limitation,
    closed in E4).
    """
    from app.core.task_runtime import task_is_working_clause
    from app.models.task import DispatchStatus, Task

    return (
        (Task.room_id == room_id)
        & task_is_working_clause()
        & (Task.dispatch_status == DispatchStatus.DISPATCHED)
        & ((Task.assigned_agent_id.is_(None)) | Task.assigned_agent_id.not_in(connector_agent_ids()))
    )


async def needs_slot(task: Any, db: AsyncSession) -> bool:
    """False for a task that runs on a connector (see :func:`busy_clause`)."""
    from app.core.connector_session import resolve_executor
    from app.models.agent import Agent

    agent_id = getattr(task, "assigned_agent_id", None)
    if agent_id is None:
        return True
    agent = (await db.execute(select(Agent).where(Agent.id == agent_id))).scalars().first()
    return await resolve_executor(agent, db) != "connector"


async def _lock_room_admission(room_id: Any, db: AsyncSession) -> None:
    """Serialise admission within one room until the caller's transaction ends.

    A transaction-scoped advisory lock rather than the room row: admission may
    wait a few seconds on the pressure reading and must not hold up ordinary
    room updates meanwhile.
    """
    bind = db.get_bind()
    if getattr(getattr(bind, "dialect", None), "name", "") != "postgresql":
        return
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": f"room_slots:{room_id}"},
    )


async def admit(room_id: Any, db: AsyncSession) -> Admission:
    """Decide whether one more task of the room may start now.

    Takes the room's admission lock inside the caller's transaction; the
    caller claims the task (or queues it) and commits, which releases it.
    """
    from app.models.room import Room
    from app.models.task import Task

    await _lock_room_admission(room_id, db)
    row = (
        await db.execute(
            select(Room.parallel_tasks_base, Room.last_slot_admit_at).where(Room.id == room_id)
        )
    ).first()
    base = int((row[0] if row else None) or DEFAULT_BASE)
    last = row[1] if row else None
    busy = int(
        (await db.execute(select(func.count(Task.id)).where(busy_clause(room_id)))).scalar() or 0
    )
    cap = slots_cap()
    now = datetime.now(timezone.utc)
    since = None
    if isinstance(last, datetime):
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        since = (now - last).total_seconds()
    pressure = None
    if base <= busy < cap and (since is None or since >= extra_slot_settle_sec()):
        pressure = await room_pressure(room_id)
    admission = decide(busy, base, cap, pressure, since)
    if admission.ok:
        # Still under the admission lock: the caller's claim commits it.
        await db.execute(
            update(Room)
            .where(Room.id == room_id)
            .values(last_slot_admit_at=now)
            .execution_options(synchronize_session=False)
        )
    return admission


def queue_reason_text(admission: Admission) -> str:
    if admission.reason == "cap":
        return f"{admission.busy} of {admission.cap} tasks are already running in this room."
    if admission.reason == "memory_busy" and admission.pct is not None:
        return (
            f"{admission.busy} tasks are running and the room's memory is "
            f"{admission.pct:.0f}% full."
        )
    if admission.reason == "settling":
        return (
            f"{admission.busy} tasks are running and one started moments ago; "
            "waiting to see how much memory it takes."
        )
    if admission.reason == "memory_busy":
        return f"{admission.busy} tasks are running and the room's memory is under pressure."
    return (
        f"{admission.busy} tasks are running and the room's memory cannot be checked "
        "right now."
    )


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.isoformat()


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


async def queue_task(
    task: Any,
    admission: Admission,
    db: AsyncSession,
    *,
    resume: bool,
    rework_feedback: Optional[str],
    auto_retry_started_at: Optional[datetime],
) -> None:
    """Park a READY task until a slot frees, and tell the room once.

    ``updated_at`` is written back unchanged: an automatic retry is replayed
    only while the task still carries the ``updated_at`` it was scheduled with.
    """
    from app.core.events import emit_event
    from app.models.task import DispatchStatus, Task

    first_time = task.queued_at is None
    queued_at = task.queued_at or datetime.now(timezone.utc)
    intent = {
        "resume": bool(resume),
        "rework_feedback": rework_feedback,
        "auto_retry_started_at": _iso(auto_retry_started_at),
    }
    result = await db.execute(
        update(Task)
        .where(Task.id == task.id, Task.dispatch_status == DispatchStatus.READY)
        .values(queued_at=queued_at, queued_dispatch=intent, updated_at=Task.updated_at)
        .execution_options(synchronize_session=False)
    )
    if getattr(result, "rowcount", 1) == 0:
        await db.rollback()
        return
    if first_time:
        await emit_event(
            room_id=task.room_id,
            event_type=TASK_QUEUED,
            content=f"Task \"{task.title}\" is waiting for a free slot: {queue_reason_text(admission)}",
            db=db,
            task_id=task.id,
            metadata={
                "task_id": str(task.id),
                "busy": admission.busy,
                "base": admission.base,
                "cap": admission.cap,
                "reason": admission.reason,
                "pct": admission.pct,
            },
        )
    await db.commit()
    logger.info(
        "room_slots: task %s queued in room %s (busy=%d base=%d cap=%d reason=%s pct=%s)",
        task.id, task.room_id, admission.busy, admission.base, admission.cap,
        admission.reason, admission.pct,
    )


# States a queued task may be replayed from; any other state means the user
# (or the system) moved it on while it waited, and its queue entry is dropped.
def _replayable_states() -> frozenset:
    from app.models.task import TaskState

    return frozenset({
        TaskState.APPROVED, TaskState.ASSIGNED, TaskState.IN_PROGRESS, TaskState.RESUMING,
    })


async def dispatch_queued(room_id: Any, db: AsyncSession) -> int:
    """Replay the room's oldest queued tasks while slots may be free.

    Each replay goes through the ordinary dispatcher, which admits it again
    under the room lock — a task that still gets no slot simply stays queued.
    Returns how many dispatches were requested.
    """
    from app.core.proxy_session import TaskDispatchPreflightError, request_task_dispatch
    from app.models.task import DispatchStatus, Task

    busy = int(
        (await db.execute(select(func.count(Task.id)).where(busy_clause(room_id)))).scalar() or 0
    )
    free = slots_cap() - busy
    if free <= 0:
        return 0
    queued = (
        await db.execute(
            select(Task)
            .where(Task.room_id == room_id, Task.queued_at.is_not(None))
            .order_by(Task.queued_at.asc(), Task.id.asc())
        )
    ).scalars().all()
    replayable = _replayable_states()
    started = 0
    for task in queued:
        if task.state not in replayable or task.dispatch_status != DispatchStatus.READY:
            await clear_queue_entry(task.id, db)
            continue
        if started >= free:
            break
        intent = task.queued_dispatch or {}
        try:
            outcome = await request_task_dispatch(
                task,
                task.room_id,
                db,
                resume=bool(intent.get("resume")),
                rework_feedback=intent.get("rework_feedback"),
                source="room_slots",
                auto_retry_started_at=_parse_iso(intent.get("auto_retry_started_at")),
            )
        except TaskDispatchPreflightError as exc:
            logger.info("room_slots: queued task %s cannot start (%s) — dropped", task.id, exc)
            await clear_queue_entry(task.id, db)
            continue
        if outcome == "started":
            started += 1
    await db.commit()
    return started


async def clear_queue_entry(task_id: Any, db: AsyncSession) -> None:
    from app.models.task import Task

    await db.execute(
        update(Task)
        .where(Task.id == task_id)
        .values(queued_at=None, queued_dispatch=None, updated_at=Task.updated_at)
        .execution_options(synchronize_session=False)
    )


async def sweep_queues(db: AsyncSession) -> int:
    """Watchdog pass: replay every room that has queued tasks."""
    from app.models.task import Task

    room_ids = (
        await db.execute(
            select(Task.room_id).where(Task.queued_at.is_not(None)).distinct()
        )
    ).scalars().all()
    total = 0
    for room_id in room_ids:
        try:
            total += await dispatch_queued(room_id, db)
        except Exception:
            await db.rollback()
            logger.exception("room_slots: sweeping the queue of room %s failed", room_id)
    return total


_PENDING: dict[str, asyncio.Task] = {}


def schedule_queued_dispatch(room_id: Any, delay: float = SCHEDULE_DELAY_SEC) -> None:
    """Replay the room's queue shortly, in its own DB session.

    Called where a slot may have freed — inside a transaction that has not
    committed yet — hence the short delay. Debounced per room; a no-op outside
    a running event loop.
    """
    key = str(room_id)
    pending = _PENDING.get(key)
    if pending is not None and not pending.done():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run() -> None:
        await asyncio.sleep(delay)
        from sqlalchemy.orm import sessionmaker

        from app.db.session import engine

        factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with factory() as db:
                if await _has_queue(room_id, db):
                    await dispatch_queued(UUID(key), db)
        except Exception:
            logger.exception("room_slots: replaying the queue of room %s failed", key)
        finally:
            _PENDING.pop(key, None)

    _PENDING[key] = loop.create_task(_run(), name=f"room_slots_{key}")


async def _has_queue(room_id: Any, db: AsyncSession) -> bool:
    from app.models.task import Task

    row = (
        await db.execute(
            select(Task.id).where(Task.room_id == room_id, Task.queued_at.is_not(None)).limit(1)
        )
    ).first()
    return row is not None
