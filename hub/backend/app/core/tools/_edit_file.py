"""edit_file tool — exact string replacement (agent only).

Rules (mirror Claude Code Edit semantics):
- The file must have been read with read_file in this session.
- old_string must appear exactly once, OR replace_all=true must be set.
- Prefer this over write_file when changing a small part of a large file.
"""
from __future__ import annotations

from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Replace an exact string in a workspace file (read it with read_file first). "
        "old_string must match once unless replace_all=true."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to replace.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every match.",
                "default": False,
            },
            "expected_fingerprint": {
                "type": "string",
                "description": "Fingerprint from read_file: edit without a fresh read, reject stale content.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import workspaced_client as wsd
        from app.core.tools._session_state import mark_read, read_fingerprint, was_read
        from app.core.context_observations import fingerprint

        raw_path = str(tool_input.get("path") or "").strip()
        old_string = tool_input.get("old_string", "")
        new_string = tool_input.get("new_string", "")
        replace_all = bool(tool_input.get("replace_all", False))

        if not raw_path:
            return "Error: path is required"
        from app.core.path_contract import WorkspacePathError, normalize_workspace_path
        try:
            path = normalize_workspace_path(raw_path)
        except WorkspacePathError as exc:
            return f"Error: invalid path: {exc}"
        from app.core.workspace_fs import normalize_ws_path
        boundary_path = normalize_ws_path(path)
        if not old_string:
            return "Error: old_string is required and cannot be empty"
        if old_string == new_string:
            return "Error: old_string and new_string are identical — nothing to change"

        expected_fingerprint = str(tool_input.get("expected_fingerprint") or "").strip() or None
        prior_fingerprint = read_fingerprint(str(ctx.session_id), path)
        if not was_read(str(ctx.session_id), path) and not expected_fingerprint:
            try:
                from app.core.metrics import record_workspace_read_before_edit

                record_workspace_read_before_edit()
            except Exception:
                pass
            return (
                f"Error: {path!r} has not been read in this session. "
                "Call read_file first — this ensures you are editing the current content."
            )

        if ctx.task_id:
            # Read and commit on this task's own branch even if a gate merge
            # moved the shared checkout elsewhere meanwhile.
            from app.core.room_git_flow import pin_task_head

            await pin_task_head(ctx.room_id, ctx.task_id, ctx.db)
        content = await wsd.read_workspace_file(str(ctx.room_id), path)
        if content is None:
            try:
                from app.core.metrics import (
                    record_workspace_file_not_found,
                    record_workspace_path_mismatch,
                )

                record_workspace_file_not_found("edit_file")
                if raw_path != path:
                    record_workspace_path_mismatch("edit_file")
            except Exception:
                pass
            return f"File not found: {path!r}"
        current_fingerprint = fingerprint(content)
        if expected_fingerprint and expected_fingerprint != current_fingerprint:
            return (
                f"Error: {path!r} changed since the supplied expected_fingerprint. "
                "Call read_file again and retry with the fresh fingerprint."
            )
        if prior_fingerprint and prior_fingerprint != current_fingerprint:
            return (
                f"Error: {path!r} changed since it was read. "
                "Call read_file again before editing."
            )

        count = content.count(old_string)
        if count == 0:
            return (
                f"Error: old_string not found in {path!r}. "
                "Use read_file to verify the exact content (whitespace and newlines matter)."
            )
        if not replace_all and count > 1:
            return (
                f"Error: old_string found {count} times in {path!r} — it must be unique. "
                "Expand old_string to include more surrounding context, "
                "or set replace_all=true to replace all occurrences."
            )

        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        task_ref = f" (task/{ctx.task_id})" if ctx.task_id else ""
        commit_msg = f"session:{ctx.session_id}: edit {path}{task_ref}"
        try:
            write_result = await wsd.write_workspace_file(
                str(ctx.room_id), boundary_path, new_content, commit_message=commit_msg
            )
        except wsd.WorkspaceMainProtectedError:
            from app.core.room_git_flow import room_working_branch
            from app.core.tools._write_file import main_protected_error

            # The edit is on disk; only the commit was refused.
            mark_read(str(ctx.session_id), path, fingerprint(new_content))
            return main_protected_error(
                path, ctx.task_id, await room_working_branch(ctx.room_id, ctx.db)
            )
        mark_read(str(ctx.session_id), path, fingerprint(new_content))
        try:
            from app.core.context_observations import invalidate_path
            await invalidate_path(ctx.db, ctx.room_id, path)
        except Exception:
            # Artifact invalidation is best-effort; the edit itself succeeded.
            pass
        try:
            from app.core.tools._repo_map import invalidate_repo_map_path
            invalidate_repo_map_path(str(ctx.room_id), path)
        except Exception:
            pass
        # Stage 6: no repofile metadata sync — the volume is the single
        # source of truth and the snapshot reflects this edit on the next read.

        n = count if replace_all else 1
        return f"Edited: {path} ({n} replacement{'s' if n != 1 else ''})"


REGISTRY.register(EditFileTool())
