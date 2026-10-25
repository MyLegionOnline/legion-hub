"""Small, durable event contract for isolated ``run_subtask`` activity.

The child loop deliberately reuses its parent's session id, so a session id
alone cannot distinguish the parent tool call from the child tool calls.  This
module supplies the correlation id and the intentionally compact metadata the
chat UI needs to render that activity as one collapsible sub-step.
"""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4


def new_subtask_activity(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any] | None:
    """Create a correlation payload only for an accepted ``run_subtask`` call."""
    if tool_name != "run_subtask":
        return None
    title = tool_input.get("step_title")
    if not isinstance(title, str) or not title.strip():
        title = "Sub-step"
    raw_budget = tool_input.get("max_tool_calls")
    budget = raw_budget if isinstance(raw_budget, int) and not isinstance(raw_budget, bool) else None
    return {
        "id": str(uuid4()),
        "title": title.strip()[:240],
        "max_tool_calls": budget,
    }


def activity_metadata(activity: dict[str, Any] | None, *, phase: str) -> dict[str, Any] | None:
    """Return a JSON-safe activity object, or ``None`` for ordinary calls."""
    if not isinstance(activity, dict):
        return None
    activity_id = activity.get("id")
    title = activity.get("title")
    if not isinstance(activity_id, str) or not activity_id:
        return None
    if not isinstance(title, str) or not title.strip():
        title = "Sub-step"
    result: dict[str, Any] = {"id": activity_id, "title": title.strip()[:240], "phase": phase}
    budget = activity.get("max_tool_calls")
    if isinstance(budget, int) and not isinstance(budget, bool) and budget > 0:
        result["max_tool_calls"] = budget
    return result


def completion_metadata(activity: dict[str, Any] | None, result_text: Any) -> dict[str, Any] | None:
    """Build a safe lifecycle payload from the compact subtask report.

    A malformed report must not hide the failed lifecycle event.  It is shown
    as a failure without copying arbitrary tool output into persistent chat.
    """
    result = activity_metadata(activity, phase="finished")
    if result is None:
        return None

    report: dict[str, Any] = {}
    if isinstance(result_text, str):
        try:
            parsed = json.loads(result_text)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            report = parsed

    status = report.get("status")
    result["status"] = "completed" if status == "completed" else "failed"
    for field in ("summary", "error"):
        value = report.get(field)
        if isinstance(value, str) and value.strip():
            result[field] = value.strip()[:1_000]
    for field in ("tokens_in", "tokens_out", "tokens_used", "tool_calls_used"):
        value = report.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            result[field] = value
    paths = report.get("file_writes")
    if isinstance(paths, list):
        result["file_writes"] = [path.strip() for path in paths if isinstance(path, str) and path.strip()][:20]
    return result
