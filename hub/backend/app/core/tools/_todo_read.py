"""Read the current persistent todo list for the agent session."""
from __future__ import annotations

import json
from typing import Any

from app.core.context_observations import build_task_todo, load_context_state
from app.core.tools.registry import REGISTRY, Tool, ToolContext


class TodoReadTool(Tool):
    name = "todo_read"
    description = "Read the current persistent todo list for this task."
    input_schema = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    roles = frozenset({"agent"})
    requires_container = False
    mutating = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        state = await load_context_state(ctx.db, ctx.session_id)
        todo = state.get("todo")
        if not isinstance(todo, list) and ctx.task_id is not None:
            todo = await build_task_todo(ctx.db, task_id=ctx.task_id)
        if not isinstance(todo, list):
            todo = []
        return json.dumps(todo, ensure_ascii=False)


REGISTRY.register(TodoReadTool())
