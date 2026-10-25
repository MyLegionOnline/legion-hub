"""flag_requirement_conflict — internal design notes Stage 8.

ChatMaster-only tool for the proxy mid-task dialogue: when the agent's
question is not a wording bug in the task text (that is fixed with
task_update, see the ANSWERING prompt) but a genuine contradiction between
requirements — or a requirement that names something absent from the
repository — this records that as a first-class event instead of letting it
dissolve into a one-off reply. No separate CR-NNN mechanism; this labels the
same mid-task turn the dialogue already has.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import select

from app.core.tools.registry import REGISTRY, Tool, ToolContext

logger = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "codes": {
            "type": "array",
            "description": (
                "Requirement/non-goal codes in conflict (e.g. [\"R-003\", \"R-005\"]). "
                "Use the codes from the task's \"## Requirements\" section. Leave empty "
                "only when the conflict cannot be pinned to a specific code."
            ),
            "items": {"type": "string"},
            "default": [],
        },
        "note": {
            "type": "string",
            "description": (
                "2-4 sentences: what contradicts what, or what named asset/path is "
                "missing, and what you told the agent to do about it."
            ),
        },
    },
    "required": ["note"],
}


class FlagRequirementConflictTool(Tool):
    name = "flag_requirement_conflict"
    description = (
        "Record that a mid-task question from the agent is a real requirement "
        "conflict, not a task-wording bug — two requirements cannot both be "
        "satisfied, or a requirement points at a path/asset/decision that does not "
        "exist. Marks the named requirement codes status=conflict and raises a card "
        "for the user. Call this in the SAME turn you answer the agent; it does not "
        "replace your reply, and it does not replace task_update when the task text "
        "itself needs correcting."
    )
    input_schema = _SCHEMA
    roles = frozenset({"cm"})
    requires_container = False
    mutating = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.models.task import Task

        note = str(tool_input.get("note") or "").strip()
        if not note:
            return "Error: note is required."

        codes_raw = tool_input.get("codes") or []
        codes = [str(c).strip() for c in codes_raw if str(c or "").strip()]

        if ctx.task_id is None:
            return "Error: no active task in this context — this tool only applies to a proxy task session."

        task = (
            await ctx.db.execute(select(Task).where(Task.id == ctx.task_id))
        ).scalars().first()
        if task is None:
            return f"Error: task {ctx.task_id} not found."
        if task.room_id != ctx.room_id:
            return "Error: task does not belong to this room."

        from app.core.requirements import flag_requirement_conflict

        updated = await flag_requirement_conflict(
            task, codes, note, ctx.db, agent_label=ctx.actor_label,
        )

        return json.dumps(
            {"task_id": str(task.id), "updated_codes": updated, "flagged": True},
            ensure_ascii=False,
        )


REGISTRY.register(FlagRequirementConflictTool())
