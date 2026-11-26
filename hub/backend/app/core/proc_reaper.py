"""Stop the processes of agents that are finished, and remove their scratch files.

internal design notes (E2). Every ``bash``
command carries ``LEGION_PROC_OWNER`` (see :func:`owner_tag_for`), inherited by
everything it spawns. Once the owner is over — the task generation ended,
paused, was retried or cancelled; the direct-chat session closed — whatever it
left running (a background ``next dev``, a build that outlived its tool timeout)
only eats the room's memory, so it is stopped here.

Two triggers, one rule:

* :func:`schedule_room_reap` right after a generation settles
  (``task_runtime.finalize_run``) or task sessions are closed;
* :func:`reap_running_rooms` from the watchdog loop — covers every other path
  (Retry/Cancel fencing a run without ``finalize_run``, a server restart, an
  idle-closed direct chat).

Room services are started by workspaced itself, carry no tag and are never
touched. A failure to reach workspaced is logged and retried on the next pass.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

logger = logging.getLogger(__name__)

OWNER_ENV = "LEGION_PROC_OWNER"
NO_RUN = "-"
SCHEDULE_DELAY_SEC = 2.0

_TERMINAL_STATES = {"completed", "cancelled", "continued", "rejected"}
_LIVE_RUNTIME = {"running", "stop_requested"}


TMP_ROOT = "/tmp/legion"
TMPDIR_ENV = "TMPDIR"
# Prepended to every command: the scratch directory must exist before mktemp,
# tempfile or os.tmpdir() reach for it.
TMPDIR_PREFIX = 'mkdir -p "$TMPDIR" 2>/dev/null; '


def tmp_dir_for(ctx: Any) -> str:
    """The scratch directory of a tool call's owner (``TMPDIR`` of its commands).

    One per task, not per run: a paused or resumed task keeps its files. It is
    removed with the task's processes once the task is terminal, or once the
    direct-chat session closes (workspaced ``room_procs``). /tmp is tmpfs and
    counts against the room's memory, so leftovers of finished tasks are not
    harmless (E4: 246 of 256 MB).
    """
    task_id = getattr(ctx, "task_id", None)
    if task_id is not None:
        return f"{TMP_ROOT}/task-{task_id}"
    return f"{TMP_ROOT}/session-{ctx.session_id}"


def owner_tag_for(ctx: Any) -> str:
    """The owner tag of a tool call: a task generation or a direct-chat session."""
    task_id = getattr(ctx, "task_id", None)
    if task_id is not None:
        return f"task:{task_id}:{getattr(ctx, 'run_id', None) or NO_RUN}"
    return f"session:{ctx.session_id}"


def parse_owner(tag: str) -> tuple[str, UUID, str | None] | None:
    """``("task", task_id, run_id)`` or ``("session", session_id, None)``; None if foreign."""
    parts = (tag or "").split(":")
    try:
        if parts[0] == "task" and len(parts) == 3:
            return "task", UUID(parts[1]), parts[2]
        if parts[0] == "session" and len(parts) == 2:
            return "session", UUID(parts[1]), None
    except ValueError:
        return None
    return None


def _value(v: Any) -> str:
    return str(getattr(v, "value", v) or "").lower()


async def finished_owners(tags: Iterable[str], db: AsyncSession) -> list[str]:
    """The tags whose owner is over and whose processes may be stopped.

    A task generation is alive while it is the task's current run and its
    runtime is running (or acknowledging a stop). A tag without a run id —
    a tool call outside a generation — is kept until the task is terminal:
    with no generation to compare, only a finished task is certainly done.
    A direct-chat session is alive until it is closed. Unknown tag shapes are
    never reaped.
    """
    from app.models.session import Session
    from app.models.task import Task

    parsed = {t: p for t in tags if (p := parse_owner(t)) is not None}
    task_ids = {p[1] for p in parsed.values() if p[0] == "task"}
    session_ids = {p[1] for p in parsed.values() if p[0] == "session"}

    tasks: dict[UUID, tuple[Any, Any, Any]] = {}
    if task_ids:
        rows = await db.execute(
            select(Task.id, Task.run_id, Task.runtime_state, Task.state).where(Task.id.in_(task_ids))
        )
        tasks = {row[0]: (row[1], row[2], row[3]) for row in rows.all()}
    open_sessions: set[UUID] = set()
    if session_ids:
        rows = await db.execute(
            select(Session.id).where(Session.id.in_(session_ids), Session.closed_at.is_(None))
        )
        open_sessions = set(rows.scalars().all())

    dead: list[str] = []
    for tag, (kind, ident, run) in parsed.items():
        if kind == "session":
            if ident not in open_sessions:
                dead.append(tag)
            continue
        row = tasks.get(ident)
        if row is None:
            dead.append(tag)
            continue
        current_run, runtime, state = row
        if run == NO_RUN:
            if _value(state) in _TERMINAL_STATES:
                dead.append(tag)
        elif str(current_run or "") != run or _value(runtime) not in _LIVE_RUNTIME:
            dead.append(tag)
    return sorted(dead)


async def reap_room(room_id: Any, db: AsyncSession) -> list[int]:
    """Stop the processes of finished owners in one room; return the stopped pids."""
    from app.core import workspaced_client as wsd
    from app.models.workspace import Workspace, WorkspaceBgProc

    rid = str(room_id)
    try:
        listing = await wsd.list_proc_owners(rid)
    except Exception as exc:
        logger.warning("proc_reaper: listing processes of room %s failed: %s", rid, exc)
        return []
    owners = listing.get("owners") or {}
    if not owners:
        return []
    dead = await finished_owners(owners.keys(), db)
    if not dead:
        return []
    try:
        result = await wsd.reap_proc_owners(rid, dead)
    except Exception as exc:
        logger.warning("proc_reaper: stopping processes in room %s failed: %s", rid, exc)
        return []
    pids = [int(p) for p in result.get("pids") or []]
    removed = list(result.get("removed_tmp") or [])
    if removed:
        logger.info("proc_reaper: room %s — removed scratch dir(s) %s", rid, ", ".join(removed))
    if not pids:
        return []
    logger.info(
        "proc_reaper: room %s — stopped %d process(es) of %s",
        rid, len(pids), ", ".join(dead),
    )
    ws_id = (
        await db.execute(select(Workspace.id).where(Workspace.room_id == UUID(rid)))
    ).scalar_one_or_none()
    if ws_id is not None:
        await db.execute(
            update(WorkspaceBgProc)
            .where(
                WorkspaceBgProc.workspace_id == ws_id,
                WorkspaceBgProc.status == "running",
                WorkspaceBgProc.pid.in_(pids),
            )
            .values(status="killed")
            .execution_options(synchronize_session=False)
        )
        await db.commit()
    return pids


async def reap_running_rooms(db: AsyncSession) -> int:
    """Watchdog pass over every running workspace; returns the stopped process count."""
    from app.models.workspace import Workspace

    room_ids = (
        await db.execute(select(Workspace.room_id).where(Workspace.status == "running"))
    ).scalars().all()
    total = 0
    for room_id in room_ids:
        try:
            total += len(await reap_room(room_id, db))
        except Exception:
            await db.rollback()
            logger.exception("proc_reaper: room %s failed", room_id)
    # Worktrees of tasks that are over (internal design notes E1).
    try:
        from app.core.task_trees import sweep_task_trees

        await sweep_task_trees(db)
    except Exception:
        await db.rollback()
        logger.exception("proc_reaper: sweeping task worktrees failed")
    return total


_PENDING: dict[str, asyncio.Task] = {}


def schedule_room_reap(room_id: Any, delay: float = SCHEDULE_DELAY_SEC) -> None:
    """Reap the room shortly, in its own DB session.

    Called inside a transaction that has not committed yet — hence the delay.
    Debounced per room; a no-op outside a running event loop.
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
                await reap_room(key, db)
        except Exception:
            logger.exception("proc_reaper: reaping room %s failed", key)
        finally:
            _PENDING.pop(key, None)

    _PENDING[key] = loop.create_task(_run(), name=f"proc_reaper_{key}")
