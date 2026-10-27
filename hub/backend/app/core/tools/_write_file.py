"""write_file tool — agent only, auto-commits to the task branch."""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext


def main_protected_error(path: str, task_id: Any) -> str:
    """Tool result for a write that ``workspaced`` would not commit on ``main``.

    The file is already on disk; only the auto-commit was refused.  The agent
    can recover by itself once it knows which branch to go back to.
    """
    if task_id:
        from app.core.room_git_flow import task_branch_name

        switch = f"run `git checkout {task_branch_name(task_id)}` (the change carries over)"
    else:
        switch = "check out your working branch (`dev` or your task branch; the change carries over)"
    return (
        f"Error: {path} was written to disk but not committed: the workspace is on "
        f"`main`, and agents never commit to main. To recover, {switch}, then "
        "repeat this call so it is committed there. Do not commit to main with git either."
    )


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write content to a file in the workspace (creates or overwrites). "
        "The current directory is the workspace root; use workspace-relative paths. "
        "Legacy /workspace/... paths are accepted and normalized once. "
        "Auto-commits to the current task branch."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Destination path relative to the workspace root (legacy /workspace/... accepted)",
            },
            "content": {
                "type": "string",
                "description": "UTF-8 text content to write",
            },
        },
        "required": ["path", "content"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        import logging

        from app.core import workspace_fs as wfs
        from app.core import workspaced_client as wsd

        raw_path = str(tool_input.get("path") or "").strip()
        content = tool_input.get("content", "")
        if not raw_path:
            return "Error: path is required"
        try:
            path = wfs.normalize_workspace_path(raw_path)
        except ValueError as exc:
            return f"Error: invalid path: {exc}"
        boundary_path = wfs.normalize_ws_path(path)
        # internal design notes §5.3 (T3): no new files at the files quota.
        from app.core.room_storage import room_storage_refusal

        if ctx.room_id is not None and ctx.db is not None:
            refusal = await room_storage_refusal(ctx.room_id, ctx.db)
            if refusal is not None:
                return f"Error: {refusal}"
        task_ref = f" (task/{ctx.task_id})" if ctx.task_id else ""
        commit_msg = f"session:{ctx.session_id}: write {boundary_path}{task_ref}"
        try:
            await wsd.write_workspace_file(
                str(ctx.room_id), boundary_path, content, commit_message=commit_msg
            )
        except wsd.WorkspaceMainProtectedError:
            return main_protected_error(path, ctx.task_id)
        try:
            from app.core.context_observations import invalidate_path
            await invalidate_path(ctx.db, ctx.room_id, path)
        except Exception:
            # Artifact invalidation must not hide a successful workspace write.
            pass
        try:
            from app.core.tools._repo_map import invalidate_repo_map_path
            invalidate_repo_map_path(str(ctx.room_id), path)
        except Exception:
            pass
        # Stage 6: no repofile metadata sync — the volume is the single
        # source of truth and the snapshot reflects this write on the next read.
        return f"Written: {path} ({len(content)} chars)"


REGISTRY.register(WriteFileTool())
