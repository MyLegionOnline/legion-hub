"""git_diff tool — available to both agents and ChatMaster."""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Show the unified diff of changes made in the current task branch vs main."
    role_descriptions: dict[str, str] = {
        "agent": "Show the unified diff of changes made in the current task branch vs main.",
        "cm": "Show the unified diff of changes made by the agent on the task branch vs main.",
    }
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Optional: restrict diff to this file or directory path. "
                    "Without it, a compact --stat summary of all changed files "
                    "is returned; call again with a path for the full diff."
                ),
            },
        },
    }
    roles: frozenset[str] = frozenset({"agent", "cm"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import workspaced_client as wsd

        path = (tool_input.get("path") or "").strip()
        if not path:
            data = await wsd.git_diff_workspace(
                str(ctx.room_id), from_ref="main", stat=True, merge_base=True
            )
            stat = (data.get("diff") or "").strip()
            if not stat:
                return "No changes vs main."
            return _truncate(
                "Changed files vs main (diff --stat):\n"
                f"{stat}\n\n"
                "Call git_diff with `path` (a file or directory from the list "
                "above, exactly as printed) to see the full diff."
            )

        data = await wsd.git_diff_workspace(
            str(ctx.room_id), from_ref="main", path=path, merge_base=True
        )
        diff = data.get("diff", "")
        if diff:
            return _truncate(diff)

        # The repo may live in a subfolder of the workspace (e.g. workspace/…),
        # so an exact path filter silently matches nothing. Retry as a glob
        # pathspec that matches the path anywhere in the tree.
        glob = f"*{path.strip('/')}*"
        data = await wsd.git_diff_workspace(
            str(ctx.room_id), from_ref="main", path=glob, merge_base=True
        )
        diff = data.get("diff", "")
        if diff:
            return _truncate(
                f"(No exact match for `{path}`; showing changes matching `{glob}`.)\n{diff}"
            )
        return (
            f"No changes vs main under `{path}`. "
            "Call git_diff without `path` to list ALL changed files before "
            "concluding nothing was changed."
        )


REGISTRY.register(GitDiffTool())
