"""reply_to_user — answer a message the user sent into the running task.

internal design notes Stage 2.  The user's message arrives in the request as
a "[MESSAGE FROM THE USER DURING THE TASK …]" note; this tool posts the
answer as the agent's chat bubble and returns, so the turn goes on.  With no
delivered message waiting it refuses: it is an answer channel, not a way to
chat with the user past ChatMaster.
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext


class ReplyToUserTool(Tool):
    name = "reply_to_user"
    description = (
        "Answer a message the user sent you while you are working on the task "
        "(it appears as [MESSAGE FROM THE USER DURING THE TASK]). The answer is "
        "shown to the user in the chat; your turn does not end — keep working "
        "after it. Only for answering such a message."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Short answer in the user's language: what you are doing, what is left, or the answer to the question.",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }
    roles = frozenset({"agent"})
    requires_container = False
    mutating = False
    always_offered = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core.task_interject import record_reply

        text = str(tool_input.get("text") or "").strip()
        if not text:
            return "Error: text is empty."
        row = await record_reply(
            room_id=ctx.room_id,
            task_id=ctx.task_id,
            agent_id=ctx.actor_agent_id,
            actor_label=ctx.actor_label,
            text=text,
            db=ctx.db,
        )
        if row is None:
            return (
                "Error: there is no user message waiting for an answer. "
                "Continue the task; talk to ChatMaster with a normal message."
            )
        return "Delivered to the user. Continue the task."


REGISTRY.register(ReplyToUserTool())
