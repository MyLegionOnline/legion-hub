"""ask_user — pause the agent loop and ask the user a clarifying question.

The tool emits a persistent AGENT_QUESTION event (visible on page reload),
then awaits the user's answer via asyncio.Event with a configurable timeout.
On timeout it returns a "user is away" message so the agent continues on its
own without intervention.

Role availability: "agent" only.  CM sessions ARE the user-facing party and
have no need to ask the user — they can simply reply in their next turn.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

from app.core.tools.registry import REGISTRY, Tool, ToolContext

logger = logging.getLogger(__name__)

_TIMEOUT = int(os.getenv("ASK_USER_TIMEOUT_SECONDS", "600"))

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "description": "The user may also answer in free text.",
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Full question ending with '?'.",
                    },
                    "header": {
                        "type": "string",
                        "description": "Chip label.",
                        "maxLength": 12,
                    },
                    "options": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "1-5 words.",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "What it means.",
                                },
                            },
                            "required": ["label", "description"],
                        },
                    },
                    "multi_select": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["question", "header", "options"],
            },
        }
    },
    "required": ["questions"],
}


class AskUserTool(Tool):
    name = "ask_user"
    description = (
        "Pause and ask the user a question with 2-4 concrete options — ONLY for a "
        "decision you cannot infer from context. You resume when they answer; after "
        f"{_TIMEOUT // 60} minutes you resume with 'user is away' — then use your best judgement."
    )
    input_schema = _SCHEMA
    roles = frozenset({"agent"})
    requires_container = False
    mutating = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import ask_user_registry as _reg
        from app.core.events import emit_event

        questions = tool_input.get("questions") or []
        if not questions:
            return "Error: ask_user requires at least one question."

        question_id = str(uuid4())
        first_q = questions[0].get("question", "The agent is asking a question")

        # Persist to chat so the card is visible after page reload.
        try:
            await emit_event(
                room_id=ctx.room_id,
                event_type="AGENT_QUESTION",
                content=first_q,
                db=ctx.db,
                task_id=ctx.task_id,
                metadata={
                    "question_id": question_id,
                    "session_id": str(ctx.session_id),
                    "questions": questions,
                    "timeout_seconds": _TIMEOUT,
                },
            )
        except Exception as exc:
            logger.error(
                "ask_user: failed to emit AGENT_QUESTION (session=%s): %s", ctx.session_id, exc
            )
            return f"Error: could not send question to user: {exc}"

        pq = _reg.register(question_id, ctx.session_id, ctx.room_id, {"questions": questions})

        logger.info(
            "ask_user: session=%s question_id=%s — waiting up to %ds for user answer",
            ctx.session_id, question_id, _TIMEOUT,
        )

        try:
            await asyncio.wait_for(pq.event.wait(), timeout=float(_TIMEOUT))
            _reg.expire(question_id)
            if pq.answer:
                return json.dumps(pq.answer, ensure_ascii=False)
            return "The user resolved the question but no answer data was captured. Use your best judgement."
        except asyncio.TimeoutError:
            _reg.expire(question_id)
            logger.info(
                "ask_user: timeout for session=%s question_id=%s — continuing automatically",
                ctx.session_id, question_id,
            )
            return (
                "The user did not respond within the allotted time. "
                "Use your best judgement and continue without waiting."
            )


REGISTRY.register(AskUserTool())
