"""task_create — create a new task mid-session (CM only).

Agents do not create tasks; that is a ChatMaster / user decision.
Reuses the same dep-resolution logic as close_actions (create_tasks_with_deps).
"""
from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from app.core.tools.registry import REGISTRY, Tool, ToolContext

logger = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Task title."},
        "description": {"type": "string", "description": "Task description and acceptance criteria."},
        "goal_id": {
            "type": "string",
            "format": "uuid",
            "description": "UUID of the goal this task belongs to.",
        },
        "assigned_agent_id": {
            "type": "string",
            "format": "uuid",
            "description": "UUID of the agent to assign the task to.",
        },
        "depends_on": {
            "type": "array",
            "description": "UUIDs of tasks that must complete before this one starts.",
            "items": {"type": "string", "format": "uuid"},
            "default": [],
        },
    },
    "required": ["title"],
}


class TaskCreateTool(Tool):
    name = "task_create"
    description = (
        "Create a new task in this room immediately (mid-session). "
        "Provide a title, optional description, goal_id, assigned_agent_id, and depends_on. "
        "A TASK_CREATED event is emitted and the task is visible to the team right away. "
        "For a direct task handled by one executor, assign that agent and omit goal_id; "
        "it is dispatched automatically when it has no unfinished dependencies."
    )
    input_schema = _SCHEMA
    roles = frozenset({"cm"})   # agents do not create tasks
    requires_container = False
    mutating = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from sqlalchemy import select

        from app.core.events import emit_event
        from app.core.session_close import resource_lock
        from app.core.task_deps import TaskDependencyCycleError, create_tasks_with_deps
        from app.models.goal import RoomGoal
        from app.models.task import (
            TASK_ORIGIN_CM_QUICK, DispatchStatus, Task, TaskState,
        )

        title = (tool_input.get("title") or "").strip()
        if not title:
            return "Error: title is required."

        description = str(tool_input.get("description") or "")
        goal_id_raw = tool_input.get("goal_id")
        assignee_raw = tool_input.get("assigned_agent_id")
        depends_on_raw = tool_input.get("depends_on") or []

        goal_id: UUID | None = None
        if goal_id_raw:
            try:
                goal_id = UUID(str(goal_id_raw).strip())
            except ValueError:
                return f"Error: invalid goal_id {goal_id_raw!r}"

        assigned_agent_id: UUID | None = None
        if assignee_raw:
            try:
                assigned_agent_id = UUID(str(assignee_raw).strip())
            except ValueError:
                return f"Error: invalid assigned_agent_id {assignee_raw!r}"

        dep_strs: list[str] = []
        for dep in depends_on_raw:
            try:
                dep_strs.append(str(UUID(str(dep).strip())))
            except ValueError:
                return f"Error: invalid depends_on entry {dep!r}"

        await resource_lock(ctx.db, ctx.room_id)

        # One dialogue, one kind of work (core/dialogue_scope). A goal-less
        # quick task started beside a goal this same dialogue formed is the
        # mirror image of the divergence incident, so it is refused here rather
        # than left to produce a second branch off a different base.
        if goal_id is None and ctx.session_id is not None:
            from app.core.dialogue_scope import goal_for_session

            existing_goal = await goal_for_session(ctx.session_id, ctx.db)
            if existing_goal is not None:
                return (
                    "Error: this dialogue already formed the goal "
                    f"'{existing_goal.title}' ({existing_goal.id}). Create the task "
                    "inside that goal by passing its goal_id, or discuss the new "
                    "work in a separate dialogue — a goal-less task started beside "
                    "a goal from the same conversation makes the branches diverge."
                )

        # Validate goal belongs to this room
        if goal_id is not None:
            valid = (
                await ctx.db.execute(
                    select(RoomGoal.id).where(
                        RoomGoal.id == goal_id, RoomGoal.room_id == ctx.room_id
                    )
                )
            ).first()
            if not valid:
                logger.warning(
                    "task_create: goal_id %s not in room %s — ignoring", goal_id, ctx.room_id
                )
                goal_id = None

        task = Task(
            room_id=ctx.room_id,
            goal_id=goal_id,
            title=title,
            description=description,
            assigned_agent_id=assigned_agent_id,
            state=TaskState.ASSIGNED,
            # Stamp the quick, goal-less variant with the dialogue that spawned
            # it: that is what later stops the same dialogue from forming a goal
            # on top of already-running work (core/dialogue_scope).
            origin=TASK_ORIGIN_CM_QUICK if goal_id is None else None,
            origin_session_id=ctx.session_id if goal_id is None else None,
        )
        ctx.db.add(task)

        try:
            created_list = await create_tasks_with_deps([(None, dep_strs, task)], ctx.db)
            created = created_list[0]
        except TaskDependencyCycleError as exc:
            return f"Error: dependency cycle detected: {exc}"

        try:
            await emit_event(
                room_id=ctx.room_id,
                event_type="TASK_CREATED",
                content=f"Task '{title}' created",
                db=ctx.db,
                task_id=created.id,
                metadata={
                    "task_id": str(created.id),
                    "task_title": title,
                    "goal_id": str(goal_id) if goal_id else None,
                    "assigned_agent_id": str(assigned_agent_id) if assigned_agent_id else None,
                    "session_id": str(ctx.session_id),
                },
            )
        except Exception as exc:
            logger.warning("task_create: emit TASK_CREATED failed (session=%s): %s", ctx.session_id, exc)

        # internal design notes: the "Discussion context" an executor
        # reads is `.legion/plans/task-{id}.md` (or its goal's archive). A task
        # created here — goal-less quick work above all — has no goal archive to
        # fall back on, so without this the agent got only the one-line
        # description. Written BEFORE dispatch: the executor's system prompt
        # links the archive only if the file already exists when it starts.
        await _archive_dialogue_for_task(ctx, created, goal_id)

        # A task created directly by ChatMaster with one assigned executor is
        # ready to run now. Goal/plan are optional context for this path; tasks
        # held by dependencies remain assigned until their dispatcher is ready.
        dispatch_scheduled = False
        dispatch_error: str | None = None
        if (
            created.assigned_agent_id is not None
            and created.dispatch_status == DispatchStatus.READY
        ):
            from app.core.proxy_session import (
                TaskDispatchPreflightError,
                request_task_dispatch,
            )

            try:
                await request_task_dispatch(
                    created, ctx.room_id, ctx.db, source="task_create_tool",
                )
            except TaskDispatchPreflightError as exc:
                dispatch_error = exc.code
                logger.info(
                    "task_create: direct task %s not started: %s", created.id, exc.code
                )
            else:
                dispatch_scheduled = True

        return json.dumps(
            {
                "task_id": str(created.id),
                "title": created.title,
                "state": created.state.value,
                "created": True,
                "dispatch": "scheduled" if dispatch_scheduled else "not_scheduled",
                "dispatch_error": dispatch_error,
            },
            ensure_ascii=False,
        )


_DIALOGUE_KINDS = frozenset({"user_cm_p1", "user_cm_p2", "user_agent_p1", "user_agent_p2"})


async def _archive_dialogue_for_task(ctx: ToolContext, task: Any, goal_id: UUID | None) -> None:
    """Best-effort: snapshot the user dialogue that is creating *task*.

    Only a direct user dialogue has a transcript worth archiving — ``task_create``
    called from the ChatMaster side of a proxy session would archive the
    CM↔agent exchange instead, which the executor already has.
    """
    if ctx.session_id is None:
        return
    try:
        from app.core import plan_dialogue
        from app.core.room_coordinator import resolve_planner_llm_config
        from app.models.room import Room
        from app.models.session.base import Session

        session = await ctx.db.get(Session, ctx.session_id)
        if session is None or getattr(session, "kind", None) not in _DIALOGUE_KINDS:
            return
        llm_config = None
        room = await ctx.db.get(Room, ctx.room_id)
        if room is not None:
            try:
                _agent, llm_config = await resolve_planner_llm_config(room, ctx.db)
            except Exception:
                llm_config = None
        await plan_dialogue.archive_for_task(
            ctx.room_id,
            task.id,
            session,
            ctx.db,
            goal_id=goal_id,
            task_title=task.title or "",
            llm_config=llm_config or None,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("task_create: plan-dialogue archive failed for task %s: %s", task.id, exc)


REGISTRY.register(TaskCreateTool())
