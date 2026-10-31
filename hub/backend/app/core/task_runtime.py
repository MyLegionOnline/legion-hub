"""Durable and local control for one task's proxy runtime.

The database fields provide a cross-worker cancellation fence.  The in-memory
registry adds prompt cancellation when the API request and the worker happen
to live in the same process; correctness never depends on that optimisation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession


RUNTIME_IDLE = "idle"
RUNTIME_RUNNING = "running"
RUNTIME_STOP_REQUESTED = "stop_requested"
RUNTIME_PAUSED = "paused"
RUNTIME_STOPPED = "stopped"
RUNTIME_FAILED = "failed"

_LOCAL_PROXY_TASKS: dict[str, asyncio.Task[Any]] = {}


def task_is_working_clause() -> Any:
    """SQL condition: the task is work in progress.

    IN_PROGRESS and RESUMING always are. PAUSING is only while its run is
    still alive to acknowledge the pause; once the runtime has settled, a task
    left PAUSING is a wedge, not work, and must not keep the room's container
    or billing interval running (the watchdog settles it to PAUSED).
    """
    from app.models.task import Task, TaskState

    return or_(
        Task.state.in_([TaskState.IN_PROGRESS, TaskState.RESUMING]),
        and_(
            Task.state == TaskState.PAUSING,
            Task.runtime_state.in_([RUNTIME_RUNNING, RUNTIME_STOP_REQUESTED]),
        ),
    )


def register_local_proxy_task(task_id: Any, task: asyncio.Task[Any]) -> None:
    """Track the current local dispatcher without making it the source of truth."""
    key = str(task_id)
    _LOCAL_PROXY_TASKS[key] = task

    def _discard(done: asyncio.Task[Any]) -> None:
        if _LOCAL_PROXY_TASKS.get(key) is done:
            _LOCAL_PROXY_TASKS.pop(key, None)

    task.add_done_callback(_discard)


def cancel_local_proxy_task(task_id: Any) -> bool:
    """Best-effort immediate cancellation for a run hosted in this worker."""
    task = _LOCAL_PROXY_TASKS.get(str(task_id))
    if task is None or task.done():
        return False
    task.cancel()
    return True


async def fence_task_runtime(
    task: Any,
    reason: str,
    db: AsyncSession,
    *,
    cancel_local: bool = False,
    now: datetime | None = None,
) -> bool:
    """Persist the cross-worker stop fence for an already loaded task.

    The caller owns the surrounding transaction.  This is intentionally a
    small shared primitive: manual cancellation and confirmed replacement
    must write the same durable runtime fields before either path closes proxy
    sessions or emits user-visible events.
    """

    if task is None:
        return False
    was_running = task.runtime_state in {RUNTIME_RUNNING, RUNTIME_STOP_REQUESTED}
    task.runtime_state = RUNTIME_STOP_REQUESTED
    task.stop_requested_at = now or datetime.now(timezone.utc)
    task.stop_reason = reason
    db.add(task)
    await db.flush()
    if cancel_local:
        cancel_local_proxy_task(task.id)
    return was_running


async def request_task_stop(
    task_id: Any,
    reason: str,
    db: AsyncSession,
    *,
    cancel_local: bool = False,
) -> bool:
    """Fence a proxy run so it cannot execute another model/tool boundary."""
    from app.models.task import Task

    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(Task)
        .where(Task.id == task_id)
        .execution_options(populate_existing=True)
    )
    task = result.scalars().first()
    if task is None:
        return False
    if task.runtime_state == RUNTIME_PAUSED:
        if cancel_local:
            cancel_local_proxy_task(task_id)
        return False
    return await fence_task_runtime(
        task,
        reason,
        db,
        cancel_local=cancel_local,
        now=now,
    )


async def is_run_current(
    task_id: Any,
    run_id: str | None,
    db: AsyncSession,
) -> bool:
    """Return whether this coroutine still owns a runnable task generation."""
    if not run_id:
        # Keeps the lower-level helpers compatible with focused unit tests that
        # exercise a proxy loop without going through dispatcher setup.
        return True
    from app.models.task import Task

    row = (
        await db.execute(
            select(Task.run_id, Task.runtime_state).where(Task.id == task_id)
        )
    ).first()
    if row is None:
        return False
    current_run, state = row[0], row[1]
    # Real SQLAlchemy rows contain two scalar strings.  Keeping lightweight
    # mock rows permissive lets focused proxy-loop tests replace only the
    # provider boundary rather than recreating the runtime table.
    if not isinstance(current_run, str) or not isinstance(state, str):
        return True
    return current_run == run_id and state == RUNTIME_RUNNING


async def touch_run_activity(
    session_id: Any,
    task_id: Any,
    run_id: str | None,
    db: AsyncSession,
) -> bool:
    """Persist activity at every tool/model boundary without identity-map staleness.

    Rows are locked task → session, the order the workspace pause path uses
    (it flushes every task of the room, then ``checkpoint_workspace`` closes
    the sessions); the reverse order deadlocked against it on every ladder
    step teardown of the server load run.  The transaction is committed on
    both sides of the boundary: before, so a session row a tool already wrote
    (read ledger, todo state) is not held ahead of the task lock; after, so
    neither lock is held through the next model call.
    """
    await db.commit()
    if not await is_run_current(task_id, run_id, db):
        return False
    from app.models.session.base import Session
    from app.models.task import Task

    now = datetime.now(timezone.utc)
    await db.execute(
        update(Task)
        .where(Task.id == task_id, Task.run_id == run_id)
        .values(updated_at=now.replace(tzinfo=None))
    )
    await db.execute(
        update(Session)
        .where(Session.id == session_id, Session.closed_at.is_(None))
        .values(last_activity_at=now)
    )
    await db.commit()
    return True


async def finalize_run(
    task_id: Any,
    run_id: str | None,
    db: AsyncSession,
    *,
    failed: bool = False,
    task: Any | None = None,
) -> None:
    """Settle only the generation that is finishing; never overwrite a retry."""
    if not run_id:
        return
    from app.models.task import DispatchStatus, Task, TaskState

    if task is None:
        result = await db.execute(
            select(Task)
            .where(Task.id == task_id)
            .execution_options(populate_existing=True)
        )
        task = result.scalars().first()
        if task is None:
            return
        current_run, state, runtime = task.run_id, task.state, task.runtime_state
        dispatch = getattr(task, "dispatch_status", None)
    else:
        # The caller's object is the run loop's own copy, loaded when the run
        # began. The pause that is finishing it was written by another session
        # (the close/pause request), so that copy still reads IN_PROGRESS /
        # running: deciding from it settled the runtime as "stopped" and left
        # the task PAUSING for good — counted as live work, it kept the room's
        # container and billing interval running for 19.7 h on the live stack.
        row = (
            await db.execute(
                select(Task.run_id, Task.state, Task.runtime_state, Task.dispatch_status)
                .where(Task.id == task_id)
            )
        ).first()
        if row is None:
            return
        current_run, state, runtime, dispatch = tuple(row)
    if current_run != run_id:
        return
    pausing = runtime == RUNTIME_STOP_REQUESTED and state == TaskState.PAUSING
    next_runtime = (
        RUNTIME_PAUSED if pausing
        else RUNTIME_FAILED if failed
        else RUNTIME_STOPPED
    )
    values: dict[str, Any] = {
        "runtime_state": next_runtime,
        "updated_at": datetime.utcnow(),
    }
    if pausing:
        values["state"] = TaskState.PAUSED
        # A paused generation is over; its DISPATCHED marker is stale. Hand it
        # back to READY so Start/Resume (and any auto-resume) can redispatch —
        # otherwise request_task_dispatch short-circuits on "already active"
        # and the task is wedged IN_PROGRESS with no running loop.
        if dispatch == DispatchStatus.DISPATCHED:
            values["dispatch_status"] = DispatchStatus.READY
    # The conditional update is the cross-worker fence: a retry's new run_id
    # makes rowcount zero.
    update_result = await db.execute(
        update(Task)
        .where(Task.id == task_id, Task.run_id == run_id)
        .values(**values)
    )
    if getattr(update_result, "rowcount", 1) == 0:
        return
    if pausing:
        task.state = TaskState.PAUSED
        if dispatch == DispatchStatus.DISPATCHED:
            task.dispatch_status = DispatchStatus.READY
    task.runtime_state = next_runtime
    task.updated_at = datetime.utcnow()
    db.add(task)
    await db.flush()
    if task.runtime_state == RUNTIME_PAUSED:
        from app.core.events import emit_event

        await emit_event(
            room_id=task.room_id,
            event_type="TASK_RUNTIME_PAUSED",
            content=f"Execution of task '{task.title}' has really stopped.",
            db=db,
            task_id=task.id,
            metadata={
                "task_id": str(task.id),
                "run_id": run_id,
                "pause_reason": task.pause_reason,
            },
        )
