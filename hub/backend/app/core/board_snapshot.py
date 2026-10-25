"""Board snapshot — the Task Board as a memory-layer brick.

internal design notes stage 3.  Until now ``user_cm_p2``
received Room memory + Repo snapshot + Agents list and *nothing about the
project's work*: no goals, no plan, no tasks (defect M1).  ``user_cm_p1`` saw
ten tasks ordered by ``updated_at`` as a flat list, with no goal grouping, no
state columns and no blocking information (defect M2).  Because of that the
only way to give ChatMaster context was to keep replaying the previous
conversation — which is exactly what made one session live for eight days.

This module is the single server-side projector of the board.  It reproduces
the grouping the user sees on the "Board" tab (``TaskBoardTab.jsx``: goals in
board order → rows per task state → assigned agent), so the agent and the user
are looking at the same thing.

Deliberately compact: no descriptions, no acceptance criteria, no timestamps.
Those are fetched point-wise through Task Data.  The rendered block is capped
at ``BOARD_SNAPSHOT_TOKEN_BUDGET`` tokens; over budget it degrades in fixed
steps (plan steps → fewer tasks per column → hard truncation) and always says
how many tasks it hid.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.token_counter import TokenCounter
from app.models.agent import Agent
from app.models.goal import RoomGoal
from app.models.goal_plan import GoalPlan, GoalPlanStep
from app.models.task import (
    TASK_TERMINAL_STATES,
    DispatchStatus,
    Task,
    TaskDependency,
)

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})

# Board rows, in the order TaskBoardTab.jsx renders them.
BOARD_STATE_ORDER: tuple[str, ...] = (
    "draft",
    "waiting_approval",
    "approved",
    "assigned",
    "in_progress",
    "pausing",
    "paused",
    "resuming",
    "waiting_review",
    "blocked",
    "rejected",
    "completed",
    "cancelled",
    "continued",
)

# Finished states: no agent-driven transition leads out of them (the only way
# out is a person cancelling a continued chain by hand), so they are collapsed
# into a counter instead of listed task by task.
COLLAPSED_STATES: frozenset[str] = frozenset({"completed", "cancelled", "continued"})

_TITLE_MAX_CHARS = 90
_DEFAULT_TASKS_PER_COLUMN = 6
_MAX_PLAN_STEPS = 10
_MAX_BLOCKED_SUMMARY = 10
# Upper bound on rows pulled from the task table.  A room with more open work
# than this is already unreadable as a board; the snapshot says so explicitly
# rather than silently paging.
_MAX_TASKS_SCANNED = 500

BOARD_SNAPSHOT_TOKEN_BUDGET = 1_200

_counter = TokenCounter()


def board_snapshot_enabled() -> bool:
    """Rollback switch for stage 3 (``MEMORY_BOARD_SNAPSHOT=false``)."""
    raw = os.getenv("MEMORY_BOARD_SNAPSHOT", "true")
    return str(raw).strip().casefold() in _TRUTHY


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _enum_value(value: Any) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _clip_title(title: str | None) -> str:
    text = (title or "").strip() or "(untitled)"
    if len(text) <= _TITLE_MAX_CHARS:
        return text
    return text[: _TITLE_MAX_CHARS - 1] + "…"


def _blocked_reason(task: Task) -> Optional[str]:
    """Why this task is not moving, derived from Task columns only.

    Mirrors the signals the board shows the user (``_compute_task_runtime`` in
    api/v1/tasks.py) without its per-task event queries: a memory brick may not
    cost one round trip per card.
    """
    state = _enum_value(task.state)
    if state == "blocked":
        # runtime_status=needs_retry — the card shows a Retry button.
        return "Not started — Retry"
    if task.dispatch_status == DispatchStatus.BLOCKED:
        return "Waiting for predecessor"
    if state in ("paused", "pausing", "resuming"):
        reason = (task.pause_reason or "").strip()
        return f"Paused: {reason}" if reason else "Paused"
    if state == "rejected":
        return "Rejected — needs a re-run"
    if state == "waiting_approval":
        return "Waiting for user confirmation"
    return None


def _compact_task(
    task: Task,
    *,
    agent_names: dict[UUID, str],
    waiting_for: list[str],
) -> dict[str, Any]:
    return {
        "id": str(task.id),
        "title": _clip_title(task.title),
        "state": _enum_value(task.state),
        "agent": agent_names.get(task.assigned_agent_id) if task.assigned_agent_id else None,
        "blocked_reason": _blocked_reason(task),
        "waiting_for": waiting_for,
        "retry_count": int(task.retry_count or 0),
    }


def _group_into_columns(
    tasks: Iterable[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Split compact tasks into state columns plus counters for collapsed states."""
    columns: dict[str, list[dict[str, Any]]] = {}
    collapsed: dict[str, int] = {}
    by_state: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_state.setdefault(task["state"], []).append(task)

    for state in BOARD_STATE_ORDER:
        bucket = by_state.pop(state, [])
        if not bucket:
            continue
        if state in COLLAPSED_STATES:
            collapsed[state] = len(bucket)
        else:
            columns[state] = bucket
    # Any state the UI does not know about still has to be visible.
    for state, bucket in by_state.items():
        columns[state] = bucket
    return columns, collapsed


async def build_board_snapshot(room_id: UUID, db: AsyncSession) -> dict[str, Any]:
    """Return the room's board: goals → state columns → tasks, plus blockers.

    The shape is stable and JSON-safe so it can travel in the SESSION_STARTED
    payload for external agents as well as into the system prompt.
    """
    goals = list(
        (
            await db.execute(
                sa.select(RoomGoal)
                .where(RoomGoal.room_id == room_id)
                .order_by(RoomGoal.position, RoomGoal.created_at)
            )
        )
        .scalars()
        .all()
    )

    tasks = list(
        (
            await db.execute(
                sa.select(Task)
                .where(Task.room_id == room_id)
                .order_by(Task.updated_at.desc())
                .limit(_MAX_TASKS_SCANNED)
            )
        )
        .scalars()
        .all()
    )

    agent_names: dict[UUID, str] = {}
    dependencies: dict[UUID, list[UUID]] = {}
    if tasks:
        agent_ids = {t.assigned_agent_id for t in tasks if t.assigned_agent_id}
        if agent_ids:
            rows = await db.execute(
                sa.select(Agent.id, Agent.nickname).where(Agent.id.in_(agent_ids))
            )
            agent_names = {row[0]: row[1] for row in rows}

        task_ids = [t.id for t in tasks]
        dep_rows = await db.execute(
            sa.select(TaskDependency.task_id, TaskDependency.depends_on_id).where(
                TaskDependency.task_id.in_(task_ids)
            )
        )
        for task_id, depends_on_id in dep_rows:
            dependencies.setdefault(task_id, []).append(depends_on_id)

    plans_by_goal: dict[UUID, dict[str, Any]] = {}
    if goals:
        plan_rows = list(
            (
                await db.execute(
                    sa.select(GoalPlan).where(GoalPlan.goal_id.in_([g.id for g in goals]))
                )
            )
            .scalars()
            .all()
        )
        if plan_rows:
            step_rows = list(
                (
                    await db.execute(
                        sa.select(GoalPlanStep)
                        .where(GoalPlanStep.plan_id.in_([p.id for p in plan_rows]))
                        .order_by(GoalPlanStep.position)
                    )
                )
                .scalars()
                .all()
            )
            steps_by_plan: dict[UUID, list[GoalPlanStep]] = {}
            for step in step_rows:
                steps_by_plan.setdefault(step.plan_id, []).append(step)
            for plan in plan_rows:
                steps = steps_by_plan.get(plan.id, [])
                done = sum(1 for s in steps if _enum_value(s.status) == "completed")
                plans_by_goal[plan.goal_id] = {
                    "id": str(plan.id),
                    "status": _enum_value(plan.status),
                    "steps_total": len(steps),
                    "steps_done": done,
                    "steps": [
                        {
                            "position": s.position,
                            "title": _clip_title(s.title),
                            "status": _enum_value(s.status),
                            "agent": (
                                agent_names.get(s.assigned_agent_id)
                                if s.assigned_agent_id
                                else None
                            ),
                        }
                        for s in steps[:_MAX_PLAN_STEPS]
                    ],
                    "steps_truncated": len(steps) > _MAX_PLAN_STEPS,
                }

    # Board order: tasks inside a column follow the state order of the board and
    # then their own update order (newest first), same as the UI cells.
    task_titles = {t.id: _clip_title(t.title) for t in tasks}
    task_states = {t.id: _enum_value(t.state) for t in tasks}
    # A predecessor stops blocking once it reaches a terminal state
    # (TASK_TERMINAL_STATES); anything else — including cancelled — is still in
    # the way and worth naming.  An unknown id (older than the scan window) is
    # reported too rather than silently dropped.
    done_states = {_enum_value(s) for s in TASK_TERMINAL_STATES}

    compact_by_goal: dict[Optional[UUID], list[dict[str, Any]]] = {}
    for task in tasks:
        waiting_for = [
            task_titles.get(dep_id) or str(dep_id)[:8]
            for dep_id in dependencies.get(task.id, [])
            if task_states.get(dep_id) not in done_states
        ]
        compact_by_goal.setdefault(task.goal_id, []).append(
            _compact_task(task, agent_names=agent_names, waiting_for=waiting_for)
        )

    goal_blocks: list[dict[str, Any]] = []
    for goal in goals:
        goal_tasks = compact_by_goal.get(goal.id, [])
        columns, collapsed = _group_into_columns(goal_tasks)
        goal_blocks.append(
            {
                "id": str(goal.id),
                "title": _clip_title(goal.title),
                "status": _enum_value(goal.status),
                "position": goal.position,
                "plan": plans_by_goal.get(goal.id),
                "columns": columns,
                "collapsed": collapsed,
                "total": len(goal_tasks),
            }
        )

    ungrouped = compact_by_goal.get(None, [])
    ungrouped_columns, ungrouped_collapsed = _group_into_columns(ungrouped)

    blocked_summary = [
        {
            "task_id": t["id"],
            "title": t["title"],
            "reason": t["blocked_reason"],
            "agent": t["agent"],
        }
        for bucket in list(compact_by_goal.values())
        for t in bucket
        if t["blocked_reason"]
    ][:_MAX_BLOCKED_SUMMARY]

    completed = sum(1 for t in tasks if _enum_value(t.state) == "completed")
    finished = sum(1 for t in tasks if _enum_value(t.state) in COLLAPSED_STATES)
    blocked = sum(
        1
        for t in tasks
        if _enum_value(t.state) == "blocked" or t.dispatch_status == DispatchStatus.BLOCKED
    )
    return {
        "goals": goal_blocks,
        "without_goal": {
            "columns": ungrouped_columns,
            "collapsed": ungrouped_collapsed,
            "total": len(ungrouped),
        },
        "blocked_summary": blocked_summary,
        "counters": {
            "total": len(tasks),
            "active": max(0, len(tasks) - finished - blocked),
            "blocked": blocked,
            "completed": completed,
        },
        "scan_truncated": len(tasks) >= _MAX_TASKS_SCANNED,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STATE_LABELS: dict[str, str] = {
    "draft": "Draft",
    "waiting_approval": "Needs Approval",
    "approved": "Approved",
    "assigned": "Assigned",
    "in_progress": "In Progress",
    "pausing": "Pausing",
    "paused": "Paused",
    "resuming": "Resuming",
    "waiting_review": "Waiting Review",
    "blocked": "Blocked",
    "rejected": "Rejected",
    "completed": "Completed",
    "cancelled": "Cancelled",
    "continued": "Continued",
}


def _render_task_line(task: dict[str, Any]) -> str:
    parts = [f"  - **{task['title']}** (`{task['id']}`)"]
    if task.get("agent"):
        parts.append(f" — @{task['agent']}")
    extras: list[str] = []
    if task.get("blocked_reason"):
        extras.append(str(task["blocked_reason"]))
    if task.get("waiting_for"):
        extras.append("waiting: " + ", ".join(str(x) for x in task["waiting_for"][:3]))
    if task.get("retry_count"):
        extras.append(f"retry {task['retry_count']}")
    if extras:
        parts.append(" · " + " · ".join(extras))
    return "".join(parts)


def _render_columns(
    columns: dict[str, list[dict[str, Any]]],
    collapsed: dict[str, int],
    *,
    max_per_column: int,
) -> list[str]:
    lines: list[str] = []
    ordered_states = [s for s in BOARD_STATE_ORDER if s in columns]
    ordered_states += [s for s in columns if s not in BOARD_STATE_ORDER]
    for state in ordered_states:
        bucket = columns[state]
        label = _STATE_LABELS.get(state, state)
        lines.append(f"- {label} ({len(bucket)}):")
        for task in bucket[:max_per_column]:
            lines.append(_render_task_line(task))
        hidden = len(bucket) - max_per_column
        if hidden > 0:
            lines.append(f"  - *(showing {max_per_column} of {len(bucket)})*")
    if collapsed:
        summary = ", ".join(
            f"{_STATE_LABELS.get(state, state)}: {count}"
            for state, count in collapsed.items()
        )
        lines.append(f"- Completed: {summary}")
    return lines


def _render_once(
    snapshot: dict[str, Any],
    *,
    max_per_column: int,
    include_plan: bool,
) -> str:
    counters = snapshot.get("counters") or {}
    lines: list[str] = [
        "## Task Board",
        "",
        "This is the same Board the user sees: goals → states → tasks. "
        "Task descriptions are not included — read them selectively by id.",
        "",
        (
            f"Total tasks: {counters.get('total', 0)} · "
            f"in progress: {counters.get('active', 0)} · "
            f"blocked: {counters.get('blocked', 0)} · "
            f"completed: {counters.get('completed', 0)}"
        ),
    ]

    for goal in snapshot.get("goals", []):
        lines.append("")
        lines.append(
            f"### Goal: {goal['title']} ({goal['status']}, `{goal['id']}`) — "
            f"tasks: {goal.get('total', 0)}"
        )
        plan = goal.get("plan")
        if plan and include_plan:
            lines.append(
                f"Plan: {plan['status']} — steps {plan['steps_total']}, "
                f"done {plan['steps_done']}"
            )
            for step in plan.get("steps", []):
                agent = f" — @{step['agent']}" if step.get("agent") else ""
                lines.append(
                    f"  {step['position']}. [{str(step['status']).upper()}] "
                    f"{step['title']}{agent}"
                )
            if plan.get("steps_truncated"):
                lines.append("  *(first plan steps shown)*")
        elif plan:
            lines.append(
                f"Plan: {plan['status']} — steps {plan['steps_total']}, "
                f"done {plan['steps_done']}"
            )
        if not goal.get("columns") and not goal.get("collapsed"):
            lines.append("- *(no tasks)*")
        else:
            lines.extend(
                _render_columns(
                    goal.get("columns", {}),
                    goal.get("collapsed", {}),
                    max_per_column=max_per_column,
                )
            )

    without_goal = snapshot.get("without_goal") or {}
    if without_goal.get("columns") or without_goal.get("collapsed"):
        lines.append("")
        lines.append(f"### No goal — tasks: {without_goal.get('total', 0)}")
        lines.extend(
            _render_columns(
                without_goal.get("columns", {}),
                without_goal.get("collapsed", {}),
                max_per_column=max_per_column,
            )
        )

    blocked = snapshot.get("blocked_summary") or []
    if blocked:
        lines.append("")
        lines.append("### Blockers")
        for item in blocked:
            agent = f" (@{item['agent']})" if item.get("agent") else ""
            lines.append(
                f"- **{item['title']}** (`{item['task_id']}`){agent}: {item['reason']}"
            )

    if snapshot.get("scan_truncated"):
        lines.append("")
        lines.append("*(the room has more tasks than fit in the snapshot)*")

    return "\n".join(lines)


def render_board_snapshot(
    snapshot: dict[str, Any] | None,
    *,
    token_budget: int = BOARD_SNAPSHOT_TOKEN_BUDGET,
) -> str:
    """Render the snapshot as a markdown block that fits *token_budget*.

    Degrades in fixed steps rather than by trimming text at random, so the same
    board always renders the same way: plan steps go first (the goal's own plan
    is available point-wise), then tasks per column, then a hard cut.
    """
    if not snapshot:
        return "## Task Board\n\n*(no goals or tasks)*"

    for max_per_column, include_plan in ((_DEFAULT_TASKS_PER_COLUMN, True), (6, False), (3, False), (1, False)):
        rendered = _render_once(
            snapshot, max_per_column=max_per_column, include_plan=include_plan
        )
        if _counter.count(rendered) <= token_budget:
            return rendered

    return _counter.truncate(rendered, max(1, token_budget - 8)) + "\n…[board truncated]"
