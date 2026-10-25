"""task_update — update a task's state, title, or description mid-session.

Available to both agent and CM roles. Acquires an advisory lock on the room
(same as close_actions) so concurrent writes are serialized. Last write wins.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select

from app.core.tools.registry import REGISTRY, Tool, ToolContext

_BLOCKED_NEEDS_VISION = "needs_vision"

logger = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_id": {
            "type": "string",
            "format": "uuid",
            "description": "UUID of the task to update (get it from task_list).",
        },
        "state": {
            "type": "string",
            "description": (
                "New task state. Must follow allowed transitions: "
                "assigned→in_progress, in_progress→waiting_review/blocked/cancelled, "
                "waiting_review→completed/in_progress."
            ),
        },
        "blocked_reason": {
            "type": "string",
            "enum": ["needs_vision"],
            "description": (
                "With state=blocked: why the task cannot proceed. 'needs_vision' — the task "
                "requires looking at an image/screenshot and you cannot see images. The task "
                "is closed as blocked and your session ends."
            ),
        },
        "title": {"type": "string", "description": "New title (optional)."},
        "description": {"type": "string", "description": "New description (optional)."},
    },
    "required": ["task_id"],
}


class TaskUpdateTool(Tool):
    name = "task_update"
    description = (
        "Update a task's state, title, or description immediately (mid-session). "
        "Mark a task 'in_progress' when you start it, 'waiting_review' when done, "
        "'completed' on approval. "
        "Call task_list first to confirm the task_id and its current state."
    )
    input_schema = _SCHEMA
    roles = frozenset({"agent", "cm"})
    requires_container = False
    mutating = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core.events import emit_event
        from app.core.session_close import resource_lock
        from app.models.task import DispatchStatus, Task, TaskState, TASK_TRANSITIONS

        raw_id = tool_input.get("task_id", "")
        try:
            task_id = UUID(str(raw_id).strip())
        except (ValueError, AttributeError):
            return f"Error: invalid task_id: {raw_id!r}"

        new_state_raw = tool_input.get("state")
        new_title = tool_input.get("title")
        new_description = tool_input.get("description")

        if not new_state_raw and new_title is None and new_description is None:
            return "Error: at least one of state, title, description must be provided."

        # Normalize state value (tolerates "TaskState.IN_PROGRESS" or "IN_PROGRESS")
        new_state: TaskState | None = None
        if new_state_raw:
            raw = str(new_state_raw).strip()
            if "." in raw:
                raw = raw.rsplit(".", 1)[-1]
            try:
                new_state = TaskState(raw.lower())
            except ValueError:
                valid = [s.value for s in TaskState]
                return f"Error: unknown state {new_state_raw!r}. Valid values: {valid}"

        # Advisory lock on room — same namespace as close_actions (last write wins)
        await resource_lock(ctx.db, ctx.room_id)

        task = (
            await ctx.db.execute(select(Task).where(Task.id == task_id))
        ).scalars().first()
        if task is None:
            return f"Error: task {task_id} not found."
        if task.room_id != ctx.room_id:
            return "Error: task does not belong to this room."

        should_dispatch = False

        # Validate state transition.  A READY task must pass the same
        # dispatcher preflight used by the background runner before it can be
        # marked in_progress.  This prevents a visible but session-less task.
        if new_state is not None and new_state != task.state:
            allowed = TASK_TRANSITIONS.get(task.state, [])
            if new_state not in allowed:
                return (
                    f"Error: invalid state transition {task.state.value!r} → {new_state.value!r}. "
                    f"Allowed from {task.state.value!r}: {[s.value for s in allowed]}"
                )
            blocked_reason = str(tool_input.get("blocked_reason") or "").strip()
            if new_state == TaskState.BLOCKED and blocked_reason:
                if blocked_reason != _BLOCKED_NEEDS_VISION:
                    return f"Error: unknown blocked_reason {blocked_reason!r}. Valid: ['needs_vision']"
                # Same stop path as a stalled task: the runtime fence ends this
                # agent's run, the pair's sessions are closed, and the event
                # carries the machine-readable reason.
                from app.core.watchdog import block_stalled_task, close_open_task_sessions

                await block_stalled_task(
                    task.id,
                    blocked_reason,
                    f"Task '{task.title}' is blocked: it needs an agent that can view images "
                    "(vision override or mcp-vision).",
                    ctx.db,
                    extra_meta={"blocked_reason": blocked_reason, "capability_error_code": blocked_reason},
                )
                await close_open_task_sessions(task.id, blocked_reason, ctx.db)
                return json.dumps(
                    {"task_id": str(task.id), "state": "blocked", "blocked_reason": blocked_reason},
                    ensure_ascii=False,
                )
            if new_state == TaskState.IN_PROGRESS:
                if task.dispatch_status == DispatchStatus.READY:
                    from app.core.proxy_session import (
                        TaskDispatchPreflightError,
                        prepare_task_dispatch,
                    )

                    try:
                        # Same as the REST endpoint: validate the state the
                        # task is moving INTO — task.state is still the old
                        # (possibly non-dispatchable, e.g. waiting_review) one.
                        await prepare_task_dispatch(
                            task, ctx.db, target_state=TaskState.IN_PROGRESS,
                        )
                    except TaskDispatchPreflightError as exc:
                        return (
                            "Error: task was not moved to in_progress because dispatch "
                            f"is unavailable ({exc.code}: {exc})."
                        )
                    should_dispatch = True
                else:
                    return (
                        "Error: task was not moved to in_progress because dispatch "
                        "was already claimed. Wait for its session to appear or use Retry "
                        "if it has no live session."
                    )
            task.state = new_state

        if new_title is not None:
            task.title = new_title
        if new_description is not None:
            task.description = new_description
        task.updated_at = datetime.utcnow()
        ctx.db.add(task)
        await ctx.db.flush()

        changes: dict[str, Any] = {}
        if new_state is not None:
            changes["state"] = new_state.value
        if new_title is not None:
            changes["title"] = new_title
        if new_description is not None:
            changes["description"] = new_description

        try:
            change_summary = ", ".join(
                f"{k}={v!r}" if k != "description" else "description updated"
                for k, v in changes.items()
            )
            await emit_event(
                room_id=ctx.room_id,
                event_type="TASK_UPDATED",
                content=f"Task '{task.title}' updated: {change_summary}",
                db=ctx.db,
                task_id=task.id,
                metadata={
                    "task_id": str(task.id),
                    "task_title": task.title,
                    "changes": changes,
                    "session_id": str(ctx.session_id),
                },
            )
        except Exception as exc:
            logger.warning("task_update: emit TASK_UPDATED failed (session=%s): %s", ctx.session_id, exc)

        if should_dispatch:
            from app.core.proxy_session import request_task_dispatch

            outcome = await request_task_dispatch(
                task, ctx.room_id, ctx.db, source="task_update_tool",
            )
            logger.info(
                "task_update: dispatch for task %s outcome=%s (session=%s)",
                task.id, outcome, ctx.session_id,
            )

        return json.dumps(
            {"task_id": str(task.id), "title": task.title, "state": task.state.value, "updated": True},
            ensure_ascii=False,
        )


REGISTRY.register(TaskUpdateTool())
