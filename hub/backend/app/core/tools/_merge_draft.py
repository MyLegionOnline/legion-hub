"""merge_draft — attach an unfinished task-formation dialogue to this session.

internal design notes Stage 5 (L1). When the router cold-closes a
user↔ChatMaster / user↔agent session, its turns are saved as a draft. If the
user's next message continues that same dialogue, the memory-layer block
"Unfinished dialogue" lists the candidates and this tool prepends the carried
turns to the current session (history + journal) so the context is not lost.

Only for the direct dialogue kinds (user_cm_p1/p2, user_agent_p1/p2). Not
offered to proxy sessions, decomposition, or run_subtask children.
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext

_ALLOWED_KINDS = frozenset(
    {"user_cm_p1", "user_cm_p2", "user_agent_p1", "user_agent_p2"}
)


class MergeDraftTool(Tool):
    name = "merge_draft"
    description = (
        "Attach an unfinished task-formation dialogue (listed under "
        '"Unfinished dialogue") to the current conversation. Call it ONLY when '
        "the user's latest message clearly continues that earlier dialogue — a "
        "new topic or a consultation is NOT a continuation. Pass the "
        "session_id exactly as shown in the candidates list."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "draft_session_id": {
                "type": "string",
                "description": "session_id of the draft to attach (from the candidates list).",
            }
        },
        "required": ["draft_session_id"],
    }
    roles = frozenset({"agent", "cm"})
    requires_container = False
    mutating = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.models.session.base import Session

        draft_id = str(tool_input.get("draft_session_id") or "").strip()
        if not draft_id:
            return "Error: draft_session_id is required."

        try:
            session = await ctx.db.get(Session, ctx.session_id)
        except Exception:
            session = None
        if session is None:
            return "Error: current session not found."
        if getattr(session, "kind", None) not in _ALLOWED_KINDS:
            return "Error: merge_draft is only available in a direct user dialogue."

        try:
            from app.core.plan_dialogue import merge_draft_into_session

            ok = await merge_draft_into_session(session, draft_id, ctx.db)
        except Exception as exc:  # pragma: no cover - defensive
            return f"Error: could not merge the draft: {exc}"
        if not ok:
            return (
                "Draft not found or empty — continue without it, "
                "nothing is lost."
            )
        # The merge lands in the session history for the NEXT turns; this turn's
        # request was built before it, so the turns have to travel in the tool
        # result or the model answers the current message without them.
        transcript = await _draft_transcript(draft_id, ctx.db)
        return (
            f"Unfinished dialogue {draft_id} was merged into the current session and will be in "
            "the history of the next turns. Its content:\n\n" + transcript
        )


# Tail-biased: the end of an interrupted dialogue (the last open question, the
# latest decisions) matters more than its opening.
_TRANSCRIPT_MAX_CHARS = 16_000


async def _draft_transcript(draft_id: str, db: Any) -> str:
    from app.core.plan_dialogue import _ROLE_LABEL, journal_read
    from app.models.session.base import Session

    try:
        draft = await db.get(Session, draft_id)
        turns = await journal_read(draft, db) if draft is not None else []
    except Exception:
        turns = []
    if not turns:
        return "(draft turns unavailable)"
    parts = [
        f"[{_ROLE_LABEL.get(t['role'], t['role'])}]: {t['content']}" for t in turns
    ]
    text = "\n\n".join(parts)
    if len(text) > _TRANSCRIPT_MAX_CHARS:
        text = "…(beginning omitted)…\n\n" + text[-_TRANSCRIPT_MAX_CHARS:]
    return text


REGISTRY.register(MergeDraftTool())
