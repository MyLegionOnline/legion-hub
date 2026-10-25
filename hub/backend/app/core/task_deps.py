"""
Task dependency graph utilities.

Public API:
    TaskDependencyCycleError        — raised when adding an edge would create a cycle
    validate_no_cycle               — check proposed edges against the live graph
    create_tasks_with_deps          — batch-create tasks, resolve ref-names, write deps
    refresh_blocked_dispatch_status — level-triggered unblock for one task
    blocking_predecessors           — the predecessors that are still unfinished
    try_unblock_tasks               — unblock tasks whose last predecessor just finished
    get_blocked_dependents          — return BLOCKED tasks that depend on a given task
    emit_checkpoint_request         — publish the interactive CHECKPOINT_REQUEST message
    publish_unblocked_tasks         — TASK_UNBLOCKED/dispatch for regular tasks,
                                      CHECKPOINT_REQUEST for checkpoint tasks
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from app.models.task import Task

logger = logging.getLogger(__name__)


class TaskDependencyCycleError(ValueError):
    """Raised when proposed dependency edges would create a cycle."""

    def __init__(self, task_id: UUID, depends_on_id: UUID) -> None:
        self.task_id = task_id
        self.depends_on_id = depends_on_id
        super().__init__(
            f"Adding {task_id} → {depends_on_id} would create a cycle in task dependencies"
        )


async def validate_no_cycle(
    new_edges: list[tuple[UUID, UUID]],
    db: AsyncSession,
) -> None:
    """Raise TaskDependencyCycleError if any edge in new_edges would create a cycle.

    Must be called AFTER the new edges have been flushed to task_dependency so that
    the recursive CTE sees both existing and within-batch edges in one pass.

    new_edges: list of (task_id, depends_on_id) — same as what was just inserted.
    """
    for task_id, depends_on_id in new_edges:
        result = await db.execute(
            text("""
                WITH RECURSIVE reach AS (
                    SELECT depends_on_id AS id
                    FROM task_dependency
                    WHERE task_id = :start
                    UNION
                    SELECT d.depends_on_id
                    FROM task_dependency d
                    JOIN reach r ON d.task_id = r.id
                )
                SELECT 1 FROM reach WHERE id = :target LIMIT 1
            """),
            {"start": str(depends_on_id), "target": str(task_id)},
        )
        if result.first() is not None:
            raise TaskDependencyCycleError(task_id, depends_on_id)


async def create_tasks_with_deps(
    task_triples: list[tuple[str | None, list[str], "Task"]],
    db: AsyncSession,
) -> list["Task"]:
    """Batch-create tasks with dependency tracking.

    Each triple is (ref, depends_on, task_orm_object) where:
        ref         — optional short name for referencing this task in depends_on lists
        depends_on  — list of ref-names (same batch) or UUID strings (existing tasks)
        task_orm_object — already constructed and db.add()-ed Task, not yet flushed

    Steps:
        1. Flush to assign UUIDs to all new tasks.
        2. Build ref_map: {ref → task.id} ∪ {str(task.id) → task.id}.
        3. Resolve depends_on entries; unresolvable refs are logged and skipped.
        4. Write TaskDependency rows and flush.
        5. Set dispatch_status: 'blocked' while a predecessor is unfinished,
           'ready' otherwise — including a task whose predecessors are ALREADY
           finished, which must not be blocked on an event that cannot repeat.
        6. Call validate_no_cycle (now all edges are in DB — catches within-batch cycles).
        7. Final flush and return tasks.

    Raises TaskDependencyCycleError if dependencies form a cycle.
    """
    from app.models.task import DispatchStatus, TaskDependency

    if not task_triples:
        return []

    # 1. Flush to get IDs
    await db.flush()

    # 2. Build ref_map
    ref_map: dict[str, UUID] = {}
    tasks_out: list["Task"] = []
    for ref, _deps, task in task_triples:
        tasks_out.append(task)
        ref_map[str(task.id)] = task.id  # allow UUID-string self-ref
        if ref:
            if ref in ref_map and ref_map[ref] != task.id:
                logger.warning("task_deps: duplicate ref %r — later task wins", ref)
            ref_map[ref] = task.id

    # 3–5. Resolve deps, write rows, set dispatch_status
    new_edges: list[tuple[UUID, UUID]] = []
    resolved_by_task: dict[UUID, list[UUID]] = {}
    for ref, depends_on, task in task_triples:
        resolved: list[UUID] = []
        for dep_ref in depends_on:
            dep_id = ref_map.get(dep_ref)
            if dep_id is None:
                try:
                    dep_id = UUID(dep_ref)
                except ValueError:
                    logger.warning(
                        "task_deps: unresolvable depends_on ref %r for task %s — skipping",
                        dep_ref,
                        task.id,
                    )
                    continue
            if dep_id == task.id:
                logger.warning("task_deps: task %s lists itself in depends_on — skipping", task.id)
                continue
            resolved.append(dep_id)
            new_edges.append((task.id, dep_id))
            db.add(TaskDependency(task_id=task.id, depends_on_id=dep_id))

        resolved_by_task[task.id] = resolved

    # A predecessor that is already finished can never transition again, and
    # try_unblock_tasks only ever runs on a transition. Blocking on one leaves
    # the task waiting forever — not dispatchable, and refused even by a manual
    # Start. So readiness is decided by the predecessors' states, not by the
    # mere presence of an edge.
    batch_ids = {task.id for _ref, _deps, task in task_triples}
    existing_dep_ids = {
        dep_id
        for deps in resolved_by_task.values()
        for dep_id in deps
        if dep_id not in batch_ids
    }
    finished_dep_ids = await _finished_task_ids(existing_dep_ids, db)

    for _ref, _depends_on, task in task_triples:
        resolved = resolved_by_task[task.id]
        pending = [dep_id for dep_id in resolved if dep_id not in finished_dep_ids]
        task.dispatch_status = DispatchStatus.BLOCKED if pending else DispatchStatus.READY
        db.add(task)
        if pending:
            from app.core.metrics import record_task_blocked
            record_task_blocked()
        elif resolved:
            logger.info(
                "task_deps: task %s depends only on already finished tasks — ready at creation",
                task.id,
            )

    # 4. Flush deps + dispatch_status before cycle check
    await db.flush()

    # 6. Validate — CTE now sees all edges including within-batch ones
    if new_edges:
        await validate_no_cycle(new_edges, db)

    return tasks_out


async def _finished_task_ids(
    task_ids: set[UUID],
    db: AsyncSession,
) -> set[UUID]:
    """Return the subset of *task_ids* already in a terminal state."""
    from app.models.task import TASK_TERMINAL_STATES, Task

    if not task_ids:
        return set()
    rows = await db.execute(
        select(Task.id, Task.state).where(Task.id.in_(task_ids))
    )
    return {row[0] for row in rows.all() if row[1] in TASK_TERMINAL_STATES}


async def blocking_predecessors(
    task_id: UUID,
    db: AsyncSession,
) -> list["Task"]:
    """Return the predecessors of *task_id* that are not finished yet.

    Used to explain a refused Start: "waiting for a predecessor" reads very
    differently when that predecessor was cancelled and will never finish.
    """
    from app.models.task import TASK_TERMINAL_STATES, Task, TaskDependency

    rows = await db.execute(
        select(Task)
        .join(TaskDependency, TaskDependency.depends_on_id == Task.id)
        .where(TaskDependency.task_id == task_id)
    )
    return [t for t in rows.scalars().all() if t.state not in TASK_TERMINAL_STATES]


async def refresh_blocked_dispatch_status(
    task: "Task",
    db: AsyncSession,
) -> bool:
    """Promote a BLOCKED task to READY when every predecessor is already finished.

    try_unblock_tasks is edge-triggered: it only fires while a predecessor
    transitions. A task that was created (or re-linked) after its predecessors
    had already finished never sees that transition, so this level-triggered
    check is what keeps such a task from waiting forever. Returns True when the
    task was promoted.
    """
    from app.models.task import DispatchStatus, TaskDependency

    if task.dispatch_status != DispatchStatus.BLOCKED:
        return False
    rows = await db.execute(
        select(TaskDependency.depends_on_id).where(TaskDependency.task_id == task.id)
    )
    dep_ids = {row[0] for row in rows.all()}
    if dep_ids - await _finished_task_ids(dep_ids, db):
        return False

    task.dispatch_status = DispatchStatus.READY
    db.add(task)
    await db.flush()
    logger.info(
        "task_deps: task %s unblocked on demand — all %d predecessor(s) already finished",
        task.id, len(dep_ids),
    )
    return True


async def try_unblock_tasks(
    completed_task_id: UUID,
    db: AsyncSession,
) -> list["Task"]:
    """Unblock tasks whose last unfinished predecessor just completed.

    Call this after a task transitions to a terminal state (approved / completed).
    Returns the list of tasks that were just promoted to dispatch_status='ready'.
    The caller is responsible for dispatching them.
    """
    from app.models.task import DispatchStatus, Task, TASK_TERMINAL_STATES

    # task.state is a native PostgreSQL ENUM whose labels match the Python enum *names*
    # (uppercase: 'APPROVED', 'COMPLETED'), not the string *values* ('approved', 'completed').
    terminal_clause = ", ".join(f"'{s.name}'" for s in TASK_TERMINAL_STATES)

    result = await db.execute(
        text(f"""
            SELECT DISTINCT td.task_id
            FROM task_dependency td
            JOIN task t ON t.id = td.task_id
            WHERE td.depends_on_id = :completed_id
              AND t.dispatch_status = 'blocked'
              AND NOT EXISTS (
                SELECT 1
                FROM task_dependency td2
                JOIN task dep ON dep.id = td2.depends_on_id
                WHERE td2.task_id = td.task_id
                  AND dep.state NOT IN ({terminal_clause})
              )
        """),
        {"completed_id": str(completed_task_id)},
    )
    ready_ids = [UUID(str(row[0])) for row in result.fetchall()]

    if not ready_ids:
        return []

    _t0 = time.monotonic()
    await db.execute(
        update(Task)
        .where(Task.id.in_(ready_ids))
        .values(dispatch_status=DispatchStatus.READY)
    )
    await db.flush()

    rows = await db.execute(select(Task).where(Task.id.in_(ready_ids)))
    unblocked = list(rows.scalars().all())

    latency = time.monotonic() - _t0
    from app.core.metrics import record_task_unblocked
    for _ in unblocked:
        record_task_unblocked(latency)

    logger.info(
        "task_deps: unblocked %d task(s) after predecessor %s: %s (latency=%.3fs)",
        len(unblocked),
        completed_task_id,
        [t.id for t in unblocked],
        latency,
    )

    return unblocked


async def get_blocked_dependents(
    failed_task_id: UUID,
    db: AsyncSession,
) -> list["Task"]:
    """Return all BLOCKED tasks that directly depend on failed_task_id.

    Used to emit TASK_DEPENDENCY_FAILED when a predecessor is rejected or cancelled.
    """
    from app.models.task import Task

    result = await db.execute(
        text("""
            SELECT DISTINCT td.task_id
            FROM task_dependency td
            JOIN task t ON t.id = td.task_id
            WHERE td.depends_on_id = :failed_id
              AND t.dispatch_status = 'blocked'
        """),
        {"failed_id": str(failed_task_id)},
    )
    dep_ids = [UUID(str(row[0])) for row in result.fetchall()]

    if not dep_ids:
        return []

    rows = await db.execute(select(Task).where(Task.id.in_(dep_ids)))
    return list(rows.scalars().all())


async def emit_checkpoint_request(
    room_id: UUID,
    task: "Task",
    db: AsyncSession,
    *,
    unblocked_by: UUID | None = None,
    session_id: UUID | None = None,
) -> None:
    """Publish the interactive CHECKPOINT_REQUEST message for a checkpoint task.

    Generation checkpoint plan, stage 3: the card is rendered by the frontend
    from ``metadata.checkpoint_payload`` (stage 4); the task itself stays in
    WAITING_APPROVAL until POST .../checkpoint/resolve moves it to
    APPROVED/REJECTED. Also used at creation time for a checkpoint born
    READY (no dependencies), so it never waits for an unblock that will
    never come.
    """
    from app.core.events import emit_event
    from app.models.message import CHECKPOINT_REQUEST

    payload = task.checkpoint_payload or {}
    prompt = payload.get("prompt") or task.title
    metadata: dict[str, object] = {
        "task_id": str(task.id),
        "task_title": task.title,
        "checkpoint_kind": task.checkpoint_kind,
        "checkpoint_payload": payload,
    }
    if unblocked_by is not None:
        metadata["unblocked_by"] = str(unblocked_by)
    if session_id is not None:
        metadata["session_id"] = str(session_id)
    await emit_event(
        room_id=room_id,
        event_type=CHECKPOINT_REQUEST,
        content=f"Checkpoint '{task.title}': {prompt}",
        db=db,
        task_id=task.id,
        metadata=metadata,
    )


async def publish_unblocked_tasks(
    unblocked: list["Task"],
    room_id: UUID,
    db: AsyncSession,
    *,
    source: str,
    unblocked_by: UUID | None = None,
    predecessor_title: str | None = None,
    session_id: UUID | None = None,
) -> list["Task"]:
    """Publish tasks that try_unblock_tasks just promoted to READY.

    Generation checkpoint plan, stage 3 (P5): this is the single interception
    point between "dependencies satisfied" and "dispatched to an agent".

    * A task with ``checkpoint_kind`` set is a human-decision gate: it is
      NEVER dispatched — instead a CHECKPOINT_REQUEST message is published
      and the task stays in WAITING_APPROVAL until the human resolves it.
    * Any other task gets the usual TASK_UNBLOCKED event and, when it has an
      assigned agent, goes through request_task_dispatch.

    Returns the regular (non-checkpoint) tasks that were published.
    """
    from app.core.events import emit_event
    from app.models.message import TASK_UNBLOCKED

    regular: list["Task"] = []
    for task in unblocked:
        if task.checkpoint_kind:
            await emit_checkpoint_request(
                room_id, task, db,
                unblocked_by=unblocked_by, session_id=session_id,
            )
            logger.info(
                "task_deps: unblocked checkpoint task %s published as "
                "CHECKPOINT_REQUEST (kind=%s) — not dispatched",
                task.id, task.checkpoint_kind,
            )
            continue

        regular.append(task)
        content = (
            f"Task '{task.title}' unblocked — "
            f"predecessor '{predecessor_title}' completed."
            if predecessor_title
            else f"Task '{task.title}' is now unblocked."
        )
        metadata: dict[str, object] = {
            "task_id": str(task.id),
            "unblocked_by": str(unblocked_by) if unblocked_by else None,
        }
        if session_id is not None:
            metadata["session_id"] = str(session_id)
        await emit_event(
            room_id=room_id,
            event_type=TASK_UNBLOCKED,
            content=content,
            db=db,
            task_id=task.id,
            metadata=metadata,
        )
        if task.assigned_agent_id:
            from app.core.proxy_session import (
                TaskDispatchPreflightError,
                request_task_dispatch,
            )

            try:
                outcome = await request_task_dispatch(
                    task, room_id, db, source=source,
                )
            except TaskDispatchPreflightError as exc:
                logger.info(
                    "task_deps: unblocked task %s not dispatched: %s",
                    task.id, exc.code,
                )
            else:
                logger.info(
                    "task_deps: unblocked task %s dispatch outcome=%s",
                    task.id, outcome,
                )
    return regular
