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
            "description": (
                "List of 1-2 questions to present to the user. "
                "Each question offers 2-4 predefined options; the user may also "
                "type a free-text answer ('Other')."
            ),
            "minItems": 1,
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Complete question text, ending with a question mark.",
                    },
                    "header": {
                        "type": "string",
                        "description": "Short chip label shown above the options (max 12 chars).",
                        "maxLength": 12,
                    },
                    "options": {
                        "type": "array",
                        "description": "2-4 mutually exclusive answer choices.",
                        "minItems": 2,
                        "maxItems": 4,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "description": "Concise option name (1-5 words).",
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Brief explanation of what this option means.",
                                },
                            },
                            "required": ["label", "description"],
                        },
                    },
                    "multi_select": {
                        "type": "boolean",
                        "default": False,
                        "description": "Allow selecting multiple options simultaneously.",
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
        "Pause the current task session and ask the user a clarifying question. "
        "The session resumes automatically when the user clicks an option or types "
        f"a free-text answer. If the user does not respond within "
        f"{_TIMEOUT // 60} minutes the session resumes on its own with a "
        "'user is away' message — always use your best judgement in that case. "
        "Use this ONLY when a decision truly requires user input that cannot be "
        "inferred from context. Provide 2-4 concrete options per question."
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
