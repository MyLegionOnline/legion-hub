"""request_file_read tool — CM-only: asks user before reading a file."""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext


class RequestFileReadTool(Tool):
    name = "request_file_read"
    description = (
        "Request user permission to read a specific file from the workspace. "
        "Call this INSTEAD of read_file whenever you want to look at a file. "
        "The session pauses and shows the user a confirmation card — they can "
        "approve (you receive the file content), reject (you get a refusal), or "
        "redirect you to a different file. "
        "Always use this tool; never call read_file directly."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path (legacy /workspace/... is also accepted)",
            },
            "reason": {
                "type": "string",
                "description": "One-sentence explanation of why you need this file",
            },
        },
        "required": ["path"],
    }
    # Not exposed to LLM schemas yet; handled programmatically when the
    # Stage 4 user-approval flow is implemented (ask_user pause/resume).
    roles: frozenset[str] = frozenset()
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        # Intercepted before execute() in _call_with_ws_loop — never actually runs.
        return "[request_file_read: intercepted — pending user approval]"


REGISTRY.register(RequestFileReadTool())
