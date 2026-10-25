"""plan_dialogue — read the user↔ChatMaster task-formation dialogue (read-only).

The dialogue that produced this task/goal is archived under ``.legion/plans/``
(``internal design notes``). Those files are deliberately absent from
``repo_snapshot`` / ``list_files`` and the grep fallback cannot see them without
a live container, so this tool is the supported way to consult the user's own
words — pasted configs, reference links, the exact wording of a requirement —
instead of ChatMaster's lossy paraphrase in ``task_data``.

Scope is resolved from the session on the backend: the agent gets its own
task archive and its own goal archive and nothing else — there is no path or
id argument to point somewhere else.
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext


class PlanDialogueTool(Tool):
    name = "plan_dialogue"
    description = (
        "Read the archived user↔ChatMaster dialogue that defined this task/goal. "
        "Use it whenever task_data is ambiguous, underspecified, or seems to "
        "contradict the repository — the archive holds the user's own words, "
        "pasted configs and reference links. These files are NOT in the "
        "repository file list. "
        'mode="search" (default): pass query, get matching turns with [#N] '
        'markers. mode="digest": the agreed-requirements summary, attachments '
        'index and links. mode="turns": pass turn_ids to get whole turns.'
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for (substring or regex). Required for mode=search.",
            },
            "mode": {
                "type": "string",
                "enum": ["search", "digest", "turns"],
                "default": "search",
            },
            "turn_ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Turn numbers to return in full. Required for mode=turns.",
            },
            "max_chars": {
                "type": "integer",
                "default": 6000,
                "description": "Output budget; large archives are trimmed to fit.",
            },
        },
    }
    roles = frozenset({"agent", "cm"})
    requires_container = False
    mutating = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core.plan_dialogue import read_dialogue_for_session

        mode = str(tool_input.get("mode") or "search")
        if mode not in ("search", "digest", "turns"):
            return 'Error: mode must be one of "search", "digest", "turns".'
        raw_ids = tool_input.get("turn_ids") or []
        turn_ids: list[int] = []
        for item in raw_ids if isinstance(raw_ids, list) else []:
            try:
                turn_ids.append(int(item))
            except (TypeError, ValueError):
                continue
        try:
            return await read_dialogue_for_session(
                ctx,
                mode=mode,
                query=str(tool_input.get("query") or ""),
                turn_ids=turn_ids,
                max_chars=tool_input.get("max_chars", 6000),
            )
        except Exception as exc:  # pragma: no cover - defensive
            return f"Error: could not read the plan dialogue archive: {exc}"


REGISTRY.register(PlanDialogueTool())
