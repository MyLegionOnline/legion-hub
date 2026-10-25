"""Goal auto-completion.

A RoomGoal is completed automatically when every task attached to it has
reached a terminal state (completed / cancelled) and at least one task
actually completed. Called after any code path that moves a task into a
terminal state (CM approval in proxy_session, tasks_update/tasks_cancel in
apply_close_actions).
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import emit_event
from app.models.goal import GoalStatus, RoomGoal
from app.models.task import Task, TaskState

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {TaskState.COMPLETED, TaskState.CANCELLED}


async def maybe_complete_goal(goal_id: UUID, room_id: UUID, db: AsyncSession) -> bool:
    """Mark *goal_id* completed if all its tasks are terminal. Returns True if completed."""
    goal = (
        await db.execute(select(RoomGoal).where(RoomGoal.id == goal_id))
    ).scalars().first()
    if goal is None or goal.status != GoalStatus.ACTIVE:
        return False

    rows = (
        await db.execute(
            select(Task.state, Task.checkpoint_kind).where(Task.goal_id == goal_id)
        )
    ).all()
    # A checkpoint approved before approvals landed in COMPLETED sits in
    # APPROVED forever — it is a resolved human gate, so it counts as done.
    states = [
        TaskState.COMPLETED
        if checkpoint_kind and state == TaskState.APPROVED
        else state
        for state, checkpoint_kind in rows
    ]
    if not states:
        return False
    if any(s not in _TERMINAL_STATES for s in states):
        return False
    if not any(s == TaskState.COMPLETED for s in states):
        # Everything was cancelled — the goal was abandoned, not achieved.
        return False

    # internal design notes Stage 5: a goal developed on its own branch does not
    # complete straight away — its branch first goes through a human gate
    # (goal/<id> → dev). maybe_open_goal_review opens that gate and returns
    # truthy; the goal then finishes only when a human approves the gate
    # (resolve_goal_review → finalize_goal_completion). A goal with no isolated
    # branch (feature off, or its tasks never branched) falls straight through.
    from app.core.room_git_flow import maybe_open_goal_review

    if await maybe_open_goal_review(goal, room_id, db):
        return False

    return await finalize_goal_completion(goal, room_id, db)


async def finalize_goal_completion(goal: RoomGoal, room_id: UUID, db: AsyncSession) -> bool:
    """Mark *goal* completed, emit ``GOAL_COMPLETED`` and refresh Room memory.

    The tail of ``maybe_complete_goal``, split out so ``resolve_goal_review``
    (internal design notes Stage 5) can reuse it once the goal branch is merged
    into ``dev``. Idempotent-ish: a no-op guard on an already-completed goal.
    """
    if goal.status == GoalStatus.COMPLETED:
        return False

    now = datetime.utcnow()
    goal.status = GoalStatus.COMPLETED
    goal.completed_at = now
    goal.updated_at = now
    db.add(goal)
    await db.flush()

    await emit_event(
        room_id=room_id,
        event_type="GOAL_COMPLETED",
        content=f"Goal '{goal.title}' completed — all tasks done.",
        db=db,
        metadata={"goal_id": str(goal.id), "goal_title": goal.title},
    )
    logger.info("goal_progress: goal %s ('%s') completed", goal.id, goal.title)

    # internal design notes §4.2: a goal closing is one of
    # the two Room memory refresh triggers.
    from app.core.session_close import maybe_refresh_room_memory

    await maybe_refresh_room_memory(room_id, db, trigger="goal_completed")

    return True
