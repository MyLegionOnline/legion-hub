"""Atomic application of confirmed task-replacement proposals.

The proposal itself is produced and stored by the ChatMaster flow.  This
module is deliberately independent from that flow: applying a proposal never
calls an LLM and never interprets free-form user text.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectin_polymorphic

from app.core.events import emit_event
from app.core.proxy_session import (
    TaskDispatchPreflightError,
    request_task_dispatch,
)
from app.core.session_close import resource_lock
from app.core.task_deps import create_tasks_with_deps, validate_no_cycle
from app.core.task_runtime import cancel_local_proxy_task, fence_task_runtime
from app.core.watchdog import close_open_task_sessions
from app.models.goal import GoalStatus, RoomGoal
from app.models.message import PersistedEventType
from app.models.room import RoomAgentLink
from app.models.session import SESSION_SUBTYPES, Session
from app.models.task import (
    DispatchStatus,
    Task,
    TaskDependency,
    TaskState,
    TASK_TERMINAL_STATES,
)
from app.schemas.task_change import (
    TaskChangeProposal,
    TaskChangeValidationContext,
    TaskChangeValidationError,
    validate_task_change_proposal,
)

logger = logging.getLogger(__name__)

_LOCK_TIMEOUT_MS = 5_000
_STATEMENT_TIMEOUT_MS = 30_000


class TaskChangeConflictError(RuntimeError):
    """The board no longer matches the snapshot shown to the user."""

    code = "task_change_conflict"

    def __init__(
        self,
        message: str,
        *,
        task_id: UUID | None = None,
        expected_updated_at: datetime | None = None,
        actual_updated_at: datetime | None = None,
        expected_state: TaskState | None = None,
        actual_state: TaskState | None = None,
    ) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.expected_updated_at = expected_updated_at
        self.actual_updated_at = actual_updated_at
        self.expected_state = expected_state
        self.actual_state = actual_state


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _same_timestamp(left: datetime | None, right: datetime | None) -> bool:
    return _as_utc(left) == _as_utc(right)


def _task_state(value: Any) -> TaskState:
    return value if isinstance(value, TaskState) else TaskState(value)


def _json_state(value: Any) -> str:
    return _task_state(value).value


def _json_dispatch(value: Any) -> str:
    return value.value if isinstance(value, DispatchStatus) else str(value)


async def _set_transaction_limits(db: AsyncSession) -> None:
    """Keep a replacement from waiting indefinitely on a board lock."""

    await db.execute(
        # PostgreSQL is the supported runtime database.  Keeping these as
        # separate statements mirrors the existing close-actions service and
        # makes the timeout apply only to this transaction.
        text(f"SET LOCAL lock_timeout = {_LOCK_TIMEOUT_MS}")
    )
    await db.execute(
        text(f"SET LOCAL statement_timeout = {_STATEMENT_TIMEOUT_MS}")
    )


async def _load_locked_session(
    db: AsyncSession,
    session_id: UUID,
) -> Session | None:
    """Lock the base session row, then load its concrete subtype."""

    await db.execute(
        select(Session.id).where(Session.id == session_id).with_for_update()
    )
    result = await db.execute(
        select(Session)
        .where(Session.id == session_id)
        .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
    )
    return result.scalars().first()


async def _load_board(
    db: AsyncSession,
    room_id: UUID,
) -> tuple[list[Task], set[UUID], set[tuple[UUID, UUID]]]:
    """Lock the room task graph and return its immutable validation snapshot."""

    tasks = list(
        (
            await db.execute(
                select(Task).where(Task.room_id == room_id).with_for_update()
            )
        )
        .scalars()
        .all()
    )
    agent_ids = set(
        (
            await db.execute(
                select(RoomAgentLink.agent_id).where(RoomAgentLink.room_id == room_id)
            )
        )
        .scalars()
        .all()
    )
    edge_rows = (
        await db.execute(
            select(TaskDependency.task_id, TaskDependency.depends_on_id)
            .join(Task, Task.id == TaskDependency.task_id)
            .where(Task.room_id == room_id)
        )
    ).all()
    edges = {(row[0], row[1]) for row in edge_rows}
    return tasks, agent_ids, edges


def _validate_optimistic_lock(
    proposal: TaskChangeProposal,
    tasks_by_id: dict[UUID, Task],
) -> None:
    for source in proposal.sources:
        task = tasks_by_id.get(source.task_id)
        # Membership and terminal-state errors are reported by the stage-0
        # validator.  A missing row here is still a conflict from the user's
        # perspective because the board changed after the card was rendered.
        if task is None:
            raise TaskChangeConflictError(
                f"source task {source.task_id} disappeared from the board",
                task_id=source.task_id,
                expected_updated_at=source.expected_updated_at,
                expected_state=source.expected_state,
            )

        actual_state = _task_state(task.state)
        if not _same_timestamp(task.updated_at, source.expected_updated_at) or (
            actual_state != source.expected_state
        ):
            raise TaskChangeConflictError(
                f"source task {source.task_id} changed while the proposal was pending",
                task_id=source.task_id,
                expected_updated_at=source.expected_updated_at,
                actual_updated_at=task.updated_at,
                expected_state=source.expected_state,
                actual_state=actual_state,
            )


def _resolve_dependency(value: str, replacements_by_ref: dict[str, Task]) -> UUID:
    task = replacements_by_ref.get(value)
    if task is not None:
        return task.id
    return UUID(value)


async def _rewrite_dependencies(
    db: AsyncSession,
    proposal: TaskChangeProposal,
    replacements_by_ref: dict[str, Task],
    edges: set[tuple[UUID, UUID]],
) -> dict[UUID, dict[str, list[UUID]]]:
    """Apply downstream rewrites and return their concrete before/after IDs."""

    result: dict[UUID, dict[str, list[UUID]]] = {}
    new_edges: list[tuple[UUID, UUID]] = []

    for rewrite in proposal.dependency_rewrites:
        target = rewrite.task_id
        removed = list(rewrite.remove_depends_on)
        added = [_resolve_dependency(value, replacements_by_ref) for value in rewrite.add_depends_on]

        for dependency_id in removed:
            edges.discard((target, dependency_id))
        await db.execute(
            delete(TaskDependency).where(
                TaskDependency.task_id == target,
                TaskDependency.depends_on_id.in_(removed),
            )
        )

        concrete_added: list[UUID] = []
        for dependency_id in added:
            if (target, dependency_id) not in edges:
                db.add(TaskDependency(task_id=target, depends_on_id=dependency_id))
                edges.add((target, dependency_id))
                new_edges.append((target, dependency_id))
            concrete_added.append(dependency_id)

        result[target] = {
            "removed_dependency_ids": removed,
            "added_dependency_ids": concrete_added,
        }

    await db.flush()
    if new_edges:
        await validate_no_cycle(new_edges, db)
    return result


async def _refresh_dispatch_statuses(
    db: AsyncSession,
    affected_tasks: list[Task],
    edges: set[tuple[UUID, UUID]],
    tasks_by_id: dict[UUID, Task],
    now: datetime,
) -> None:
    """Recompute readiness after a graph rewrite, including completed deps."""

    for task in affected_tasks:
        if _task_state(task.state) in {
            TaskState.COMPLETED,
            TaskState.CANCELLED,
            TaskState.CONTINUED,
        }:
            continue
        # A running/dispatched task is already owned by the runtime; changing
        # its dependency metadata must not make it appear newly dispatchable.
        if task.dispatch_status == DispatchStatus.DISPATCHED or task.runtime_state == "running":
            continue
        predecessors = [tasks_by_id[dependency_id] for task_id, dependency_id in edges if task_id == task.id]
        task.dispatch_status = (
            DispatchStatus.READY
            if all(_task_state(dep.state) in TASK_TERMINAL_STATES for dep in predecessors)
            else DispatchStatus.BLOCKED
        )
        task.updated_at = now.replace(tzinfo=None)
        db.add(task)
    await db.flush()


def _build_applied_result(
    proposal: TaskChangeProposal,
    sources: list[dict[str, Any]],
    created: list[Task],
    dependency_rewrites: dict[UUID, dict[str, list[UUID]]],
    edges: set[tuple[UUID, UUID]],
    reopened_goals: list[dict[str, str]],
    ready_task_ids: list[UUID],
    now: datetime,
) -> dict[str, Any]:
    created_payload = []
    for spec, task in zip(proposal.create_tasks, created, strict=True):
        depends_on = [
            str(dependency_id)
            for task_id, dependency_id in edges
            if task_id == task.id
        ]
        created_payload.append(
            {
                "task_id": str(task.id),
                "ref": spec.ref,
                "title": task.title,
                "description": task.description,
                "goal_id": str(task.goal_id) if task.goal_id else None,
                "assigned_agent_id": str(task.assigned_agent_id),
                "state": _json_state(task.state),
                "dispatch_status": _json_dispatch(task.dispatch_status),
                "depends_on": sorted(depends_on),
            }
        )

    rewrite_payload = []
    for task_id, rewrite in dependency_rewrites.items():
        rewrite_payload.append(
            {
                "task_id": str(task_id),
                "remove_depends_on": [str(value) for value in rewrite["removed_dependency_ids"]],
                "add_depends_on": [str(value) for value in rewrite["added_dependency_ids"]],
            }
        )

    return {
        "version": 1,
        "proposal_id": str(proposal.proposal_id),
        "applied_at": now.isoformat(),
        "rationale": proposal.rationale,
        "cancelled_tasks": sources,
        "created_tasks": created_payload,
        "dependency_rewrites": rewrite_payload,
        "reopened_goals": reopened_goals,
        "ready_task_ids": [str(task_id) for task_id in ready_task_ids],
        "running_sources_cancelled": sum(1 for source in sources if source["was_running"]),
        "dispatch": {
            str(task_id): {"status": "pending"}
            for task_id in ready_task_ids
        },
    }


async def _persist_dispatch_outcomes(
    db: AsyncSession,
    session_id: UUID,
    proposal_id: UUID,
    dispatch: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Persist post-commit dispatch outcomes without changing board state."""

    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalars().first()
    if session is None:
        return None
    snapshot = dict(session.context_snapshot or {})
    applied = snapshot.get("last_applied_task_change")
    if not isinstance(applied, dict) or str(applied.get("proposal_id")) != str(proposal_id):
        return None
    applied = dict(applied)
    applied["dispatch"] = dispatch
    snapshot["last_applied_task_change"] = applied
    session.context_snapshot = snapshot
    db.add(session)
    await db.commit()
    return applied


async def apply_task_change_proposal(
    session: Session,
    proposal: TaskChangeProposal,
    db: AsyncSession,
) -> dict[str, Any]:
    """Atomically cancel sources, create replacements, and rewrite dependents.

    The board mutation and the optimistic-lock snapshot are committed together.
    Runtime session closure, durable events, and dispatch happen only after that
    commit, so a failed side effect cannot leave a partially replaced graph.
    Re-applying a proposal with the same ``proposal_id`` returns the saved
    result and performs no task or dispatch mutation.
    """

    if not isinstance(proposal, TaskChangeProposal):
        proposal = TaskChangeProposal.model_validate(proposal)

    room_id = session.room_id
    session_id = session.id
    if getattr(session, "kind", "user_cm_p1") not in {None, "user_cm_p1"}:
        raise TaskChangeValidationError("task replacement is only supported for user_cm_p1 sessions")

    confirm_started_at = time.monotonic()
    try:
        await _set_transaction_limits(db)
        await resource_lock(db, room_id)

        fresh_session = await _load_locked_session(db, session_id)
        if fresh_session is None:
            raise TaskChangeValidationError(f"session {session_id} was not found")

        snapshot = dict(fresh_session.context_snapshot or {})
        previous_result = snapshot.get("last_applied_task_change")
        if (
            isinstance(previous_result, dict)
            and str(previous_result.get("proposal_id")) == str(proposal.proposal_id)
        ):
            from app.core.metrics import (
                log_task_change_event,
                record_task_change_idempotent_retry,
            )

            record_task_change_idempotent_retry()
            log_task_change_event(
                "task_change_confirmed_idempotent",
                room_id=room_id,
                session_id=session_id,
                proposal_id=proposal.proposal_id,
            )
            await db.commit()
            return previous_result

        goal_id = getattr(fresh_session, "goal_id", None) or getattr(session, "goal_id", None)
        if goal_id is None:
            raise TaskChangeValidationError("user_cm_p1 session has no goal")
        goal = (
            await db.execute(
                select(RoomGoal).where(RoomGoal.id == goal_id, RoomGoal.room_id == room_id)
            )
        ).scalars().first()
        if goal is None:
            raise TaskChangeValidationError(
                f"session goal {goal_id} does not belong to room {room_id}"
            )

        tasks, room_agent_ids, edges = await _load_board(db, room_id)
        context = TaskChangeValidationContext(
            room_id=room_id,
            goal_id=goal_id,
            tasks=tasks,
            room_agent_ids=room_agent_ids,
            dependency_edges=list(edges),
        )
        tasks_by_id = {task.id: task for task in tasks}
        # Check the optimistic lock before the semantic validator so a source
        # that became terminal is reported as a stale confirmation snapshot,
        # not as a malformed proposal.
        _validate_optimistic_lock(proposal, tasks_by_id)
        validate_task_change_proposal(proposal, context)

        now = datetime.now(timezone.utc)
        source_payload: list[dict[str, Any]] = []
        for source in proposal.sources:
            task = tasks_by_id[source.task_id]
            source_payload.append(
                {
                    "task_id": str(task.id),
                    "title": task.title,
                    "previous_state": _json_state(task.state),
                    "assigned_agent_id": str(task.assigned_agent_id)
                    if task.assigned_agent_id
                    else None,
                    "was_running": task.runtime_state in {"running", "stop_requested"},
                }
            )

        replacement_tasks = [
            Task(
                room_id=room_id,
                goal_id=goal_id,
                title=spec.title,
                description=spec.description,
                assigned_agent_id=spec.assigned_agent_id,
                # Generation checkpoint plan, stage 2: a checkpoint replacement
                # is a human-decision gate — start it in WAITING_APPROVAL so the
                # dispatcher skips it until the human resolves it (stage 3).
                state=(
                    TaskState.WAITING_APPROVAL
                    if spec.checkpoint_kind is not None
                    else TaskState.ASSIGNED
                ),
                checkpoint_kind=spec.checkpoint_kind,
                checkpoint_payload=spec.checkpoint_payload,
            )
            for spec in proposal.create_tasks
        ]
        triples = [
            (spec.ref, list(spec.depends_on), task)
            for spec, task in zip(proposal.create_tasks, replacement_tasks, strict=True)
        ]
        for task in replacement_tasks:
            db.add(task)
        created_tasks = await create_tasks_with_deps(triples, db)
        tasks_by_id.update({task.id: task for task in created_tasks})
        replacements_by_ref = {
            spec.ref: task
            for spec, task in zip(proposal.create_tasks, created_tasks, strict=True)
        }

        rewrite_result = await _rewrite_dependencies(
            db, proposal, replacements_by_ref, edges
        )

        affected = [tasks_by_id[rewrite.task_id] for rewrite in proposal.dependency_rewrites]
        await _refresh_dispatch_statuses(db, [*created_tasks, *affected], edges, tasks_by_id, now)

        if goal.status == GoalStatus.COMPLETED:
            goal.status = GoalStatus.ACTIVE
            goal.completed_at = None
            goal.updated_at = now.replace(tzinfo=None)
            db.add(goal)
            reopened_goals = [{"goal_id": str(goal.id), "title": goal.title}]
        else:
            reopened_goals = []

        # The fence is written before cancellation and before the transaction
        # commits.  A stale proxy therefore cannot win a result write after the
        # replacement becomes visible to other workers.
        for source in proposal.sources:
            task = tasks_by_id[source.task_id]
            await fence_task_runtime(task, "task_replaced", db, cancel_local=False)
            task.state = TaskState.CANCELLED
            task.updated_at = now.replace(tzinfo=None)
            db.add(task)

        await db.flush()

        ready_task_ids = [
            task.id
            for task in created_tasks
            if task.dispatch_status == DispatchStatus.READY
        ]
        applied_result = _build_applied_result(
            proposal,
            source_payload,
            created_tasks,
            rewrite_result,
            edges,
            reopened_goals,
            ready_task_ids,
            now,
        )
        snapshot.pop("pending_task_change", None)
        snapshot["last_applied_task_change"] = applied_result
        fresh_session.context_snapshot = snapshot
        db.add(fresh_session)
        if session is not fresh_session:
            session.context_snapshot = snapshot
        await db.commit()
    except Exception as exc:
        if isinstance(exc, TaskChangeValidationError) or "cycle" in str(exc).lower():
            from app.core.metrics import (
                log_task_change_event,
                record_task_change_validation_error,
            )

            message = str(exc).lower()
            reason = (
                "graph_cycle" if "cycle" in message
                else "dependency" if "depend" in message or "rewir" in message
                else "semantic"
            )
            record_task_change_validation_error(reason)
            log_task_change_event(
                "task_change_validation_failed",
                stage="apply",
                reason=reason,
                error_type=type(exc).__name__,
                proposal_id=proposal.proposal_id,
            )
        await db.rollback()
        raise

    from app.core.metrics import (
        log_task_change_event,
        record_task_change_confirm_to_dispatch,
        record_task_change_running_cancellation,
    )

    running_sources_cancelled = int(applied_result.get("running_sources_cancelled", 0))
    record_task_change_running_cancellation(running_sources_cancelled)
    log_task_change_event(
        "task_change_applied",
        room_id=room_id,
        session_id=session_id,
        proposal_id=proposal.proposal_id,
        source_count=len(proposal.sources),
        replacement_count=len(created_tasks),
        running_sources_cancelled=running_sources_cancelled,
        ready_replacement_count=len(ready_task_ids),
    )

    # Everything below this point is deliberately post-commit.  It is safe to
    # retry these side effects because the durable task replacement is already
    # represented by last_applied_task_change.
    for source in proposal.sources:
        task = next(item for item in tasks if item.id == source.task_id)
        try:
            await close_open_task_sessions(
                task.id, "cancelled", db, cancel_local=True
            )
        except Exception:
            await db.rollback()
            logger.exception(
                "task_change: failed to close proxy sessions for replaced task %s",
                task.id,
            )
        cancel_local_proxy_task(task.id)

        try:
            await emit_event(
                room_id=room_id,
                event_type=PersistedEventType.TASK_STATE_CHANGED,
                content=f'Task "{task.title}": {source.expected_state.value} → cancelled',
                db=db,
                task_id=task.id,
                metadata={
                    "task_id": str(task.id),
                    "task_title": task.title,
                    "prev_state": source.expected_state.value,
                    "task_state": TaskState.CANCELLED.value,
                    "proposal_id": str(proposal.proposal_id),
                    "stop_reason": "task_replaced",
                },
            )
        except Exception:
            await db.rollback()
            logger.exception("task_change: failed to emit source state event for %s", task.id)

    first_dispatch_recorded = False
    for task in created_tasks:
        task_dependencies = [
            str(dependency_id)
            for task_id, dependency_id in edges
            if task_id == task.id
        ]
        try:
            await emit_event(
                room_id=room_id,
                event_type=PersistedEventType.TASK_CREATED,
                content=f"Task '{task.title}' created as a replacement",
                db=db,
                task_id=task.id,
                metadata={
                    "task_id": str(task.id),
                    "task_title": task.title,
                    "goal_id": str(task.goal_id) if task.goal_id else None,
                    "assigned_agent_id": str(task.assigned_agent_id)
                    if task.assigned_agent_id
                    else None,
                    "dispatch_status": _json_dispatch(task.dispatch_status),
                    "depends_on": sorted(task_dependencies),
                    "proposal_id": str(proposal.proposal_id),
                    "replacement": True,
                },
            )
        except Exception:
            await db.rollback()
            logger.exception("task_change: failed to emit created event for %s", task.id)

    dispatch_outcomes: dict[str, dict[str, Any]] = {}
    for task in created_tasks:
        if task.id not in ready_task_ids:
            dispatch_outcomes[str(task.id)] = {"status": "blocked"}
            continue
        if task.checkpoint_kind:
            # Generation checkpoint plan, stage 3: a READY checkpoint
            # replacement is published as CHECKPOINT_REQUEST, never dispatched
            # (WAITING_APPROVAL would fail dispatch preflight anyway).
            from app.core.task_deps import emit_checkpoint_request

            try:
                await emit_checkpoint_request(room_id, task, db)
            except Exception:
                await db.rollback()
                logger.exception(
                    "task_change: failed to emit CHECKPOINT_REQUEST for %s", task.id
                )
            dispatch_outcomes[str(task.id)] = {"status": "checkpoint_waiting"}
            continue
        try:
            outcome = await request_task_dispatch(
                task,
                room_id,
                db,
                source="task_change_apply",
            )
        except TaskDispatchPreflightError as exc:
            logger.info(
                "task_change: replacement task %s was not dispatched: %s",
                task.id,
                exc.code,
            )
            dispatch_outcomes[str(task.id)] = {
                "status": "not_scheduled",
                "error": exc.code,
            }
        except Exception as exc:
            logger.exception("task_change: dispatch failed for replacement %s", task.id)
            dispatch_outcomes[str(task.id)] = {
                "status": "not_scheduled",
                "error": type(exc).__name__,
            }
        else:
            dispatch_outcomes[str(task.id)] = {
                "status": "scheduled",
                "outcome": outcome,
            }
            if not first_dispatch_recorded:
                record_task_change_confirm_to_dispatch(
                    time.monotonic() - confirm_started_at
                )
                first_dispatch_recorded = True
                log_task_change_event(
                    "task_change_first_dispatch",
                    room_id=room_id,
                    session_id=session_id,
                    proposal_id=proposal.proposal_id,
                    task_id=task.id,
                    elapsed_seconds=time.monotonic() - confirm_started_at,
                )

    try:
        applied_result = await _persist_dispatch_outcomes(
            db, session_id, proposal.proposal_id, dispatch_outcomes
        ) or applied_result
    except Exception:
        await db.rollback()
        logger.exception("task_change: failed to persist dispatch outcomes")
        applied_result = dict(applied_result)
        applied_result["dispatch"] = dispatch_outcomes

    try:
        await emit_event(
            room_id=room_id,
            event_type=PersistedEventType.TASK_CHANGE_APPLIED,
            content="The proposed task replacement was applied.",
            db=db,
            metadata={
                "proposal_id": str(proposal.proposal_id),
                "source_task_ids": [str(source.task_id) for source in proposal.sources],
                "created_task_ids": [str(task.id) for task in created_tasks],
                "dependency_rewrites": applied_result.get("dependency_rewrites", []),
                "dispatch": applied_result.get("dispatch", dispatch_outcomes),
            },
        )
    except Exception:
        await db.rollback()
        logger.exception("task_change: failed to emit TASK_CHANGE_APPLIED")
    return applied_result


__all__ = [
    "TaskChangeConflictError",
    "apply_task_change_proposal",
]
