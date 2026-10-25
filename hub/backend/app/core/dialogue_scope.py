"""One dialogue produces one kind of work — quick tasks OR a goal, never both.

Incident (2026-09-18): while still gathering requirements in a ``user_cm_p2``
dialogue, ChatMaster created two goal-less tasks with ``task_create`` and
dispatched agents on them. It then proposed a plan from the *same* dialogue and
the user accepted it, so a goal with its own branch was formed out of a spec
that knew nothing about the work already running. The two lines of work edited
the same files from different bases and the task branches diverged.

The invariant this module enforces is the rule the user stated: a dialogue that
has already spawned quick goal-less tasks cannot also form a goal, and a
dialogue that has already formed (or proposed) a goal cannot spawn quick tasks
beside it. Detection is by ``Task.origin``/``origin_session_id`` — written by
the ``task_create`` tool — and by ``RoomGoal.origin_session_id``.
"""
from __future__ import annotations

from typing import Any, List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.goal import RoomGoal
from app.models.task import TASK_ORIGIN_CM_QUICK, Task, TaskState

# A quick task in one of these no longer holds the dialogue back: the work it
# represents is over, so a goal formed afterwards cannot diverge from it.
_SETTLED_STATES = frozenset({
    TaskState.COMPLETED,
    TaskState.CANCELLED,
    TaskState.REJECTED,
})


async def quick_tasks_for_session(
    session_id: UUID | None, db: AsyncSession
) -> List[Task]:
    """Unfinished goal-less tasks this dialogue created with ``task_create``."""
    if session_id is None:
        return []
    rows = (
        await db.execute(
            select(Task).where(
                Task.origin_session_id == session_id,
                Task.origin == TASK_ORIGIN_CM_QUICK,
            )
        )
    ).scalars().all()
    return [task for task in rows if task.state not in _SETTLED_STATES]


async def goal_for_session(session_id: UUID | None, db: AsyncSession) -> RoomGoal | None:
    """The goal this dialogue already formed, if any."""
    if session_id is None:
        return None
    return (
        await db.execute(
            select(RoomGoal).where(RoomGoal.origin_session_id == session_id).limit(1)
        )
    ).scalars().first()


def describe_quick_tasks(tasks: List[Task]) -> str:
    """One-line-per-task list for a message shown to the user or the model."""
    return "\n".join(f"• “{task.title}” ({task.state.value})" for task in tasks)


def quick_tasks_block_plan_message(tasks: List[Task]) -> str:
    """User-facing explanation of why no goal is formed from this dialogue."""
    return (
        "This discussion already has quick tasks running without a goal:\n"
        f"{describe_quick_tasks(tasks)}\n\n"
        "Until they finish, a goal is not formed from this same dialogue — "
        "otherwise the goal's plan would not know about this work, and the task branches would diverge. "
        "Wait for (or stop) those tasks and start the goal discussion again, "
        "or we will continue here without a goal."
    )


def has_pending_plan(session: Any) -> bool:
    """True when this dialogue is already holding a proposed plan."""
    return bool((getattr(session, "context_snapshot", None) or {}).get("pending_plan"))
