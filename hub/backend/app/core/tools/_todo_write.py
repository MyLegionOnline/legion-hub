"""Persist the complete todo list for the current agent session."""
from __future__ import annotations

import json
from typing import Any

from app.core.context_observations import load_context_state, save_context_state
from app.core.tools.registry import REGISTRY, Tool, ToolContext


_STATUSES = frozenset({"pending", "in_progress", "done"})


def _validate_items(raw_items: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    if not isinstance(raw_items, list):
        return None, "Error: items must be an array."

    items: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict):
            return None, f"Error: items[{index}] must be an object."

        item_id = raw_item.get("id")
        title = raw_item.get("title")
        status = raw_item.get("status")
        if not isinstance(item_id, str) or not item_id.strip():
            return None, f"Error: items[{index}].id must be a non-empty string."
        if not isinstance(title, str) or not title.strip():
            return None, f"Error: items[{index}].title must be a non-empty string."
        if not isinstance(status, str) or status not in _STATUSES:
            return None, (
                f"Error: items[{index}].status must be one of: "
                "pending, in_progress, done."
            )

        normalized_id = item_id.strip()
        if normalized_id in seen_ids:
            return None, f"Error: duplicate todo id: {normalized_id!r}."
        seen_ids.add(normalized_id)
        items.append({
            "id": normalized_id,
            "title": title.strip(),
            "status": status,
        })
    return items, None


class TodoWriteTool(Tool):
    name = "todo_write"
    description = (
        "Replace the current persistent todo list for this task. "
        "Use short stable ids and set each status to pending, in_progress, or done."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
                    },
                    "required": ["id", "title", "status"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    roles = frozenset({"agent"})
    requires_container = False
    mutating = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        items, error = _validate_items(tool_input.get("items"))
        if error:
            return error

        state = await load_context_state(ctx.db, ctx.session_id)
        state["todo"] = items
        await save_context_state(ctx.db, ctx.session_id, state)
        return json.dumps({"updated": True, "items": items}, ensure_ascii=False)


REGISTRY.register(TodoWriteTool())
