"""task_list — list all tasks in the current room.

Available to both agent and CM roles.
Returns id, title, state, assigned_agent_id, and depends_on for each task.
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from app.core.tools.registry import REGISTRY, Tool, ToolContext


class TaskListTool(Tool):
    name = "task_list"
    description = (
        "Return all tasks in this room with their id, title, state, "
        "assigned_agent_id, and dependencies. "
        "Use this before task_update to confirm task ids and current states."
    )
    input_schema = {"type": "object", "properties": {}}
    roles = frozenset({"agent", "cm"})
    requires_container = False
    mutating = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.models.task import Task, TaskDependency

        tasks = (
            await ctx.db.execute(
                select(Task)
                .where(Task.room_id == ctx.room_id)
                .order_by(Task.created_at)
            )
        ).scalars().all()

        if not tasks:
            return "No tasks found in this room."

        all_ids = [t.id for t in tasks]
        dep_rows = (
            await ctx.db.execute(
                select(TaskDependency).where(TaskDependency.task_id.in_(all_ids))
            )
        ).scalars().all()
        deps_map: dict = {}
        for d in dep_rows:
            deps_map.setdefault(d.task_id, []).append(str(d.depends_on_id))

        result = [
            {
                "id": str(t.id),
                "title": t.title,
                "state": t.state.value,
                "assigned_agent_id": str(t.assigned_agent_id) if t.assigned_agent_id else None,
                "depends_on": deps_map.get(t.id, []),
            }
            for t in tasks
        ]
        return json.dumps(result, ensure_ascii=False, indent=2)


REGISTRY.register(TaskListTool())
