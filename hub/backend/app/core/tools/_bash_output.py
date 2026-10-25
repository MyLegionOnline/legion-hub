"""bash_output tool — read accumulated log of a background bash process."""
from __future__ import annotations

from typing import Any

from app.core.context_observations import fingerprint, record_observation
from app.core.secrets.room_masking import bg_secrets, mask_room_secrets
from app.core.tools.registry import REGISTRY, Tool, ToolContext


class BashOutputTool(Tool):
    name = "bash_output"
    description = (
        "Read the accumulated log output of a background process started with bash(background=true). "
        "Log is stored at /workspace/.legion/bg/<bg_id>.log and read via file API (no exec needed). "
        "Check status field: 'running' means still alive, 'killed'/'failed' means stopped."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "bg_id": {
                "type": "string",
                "description": "Background process ID returned by bash(background=true)",
            },
            "tail": {
                "type": "integer",
                "default": 100,
                "description": "Number of lines to return from the end of the log (1–1000)",
            },
            "cursor": {
                "type": "integer",
                "default": 0,
                "description": "Read from this 0-based line cursor; returns next_cursor for paging",
            },
            "limit": {
                "type": "integer",
                "default": 100,
                "description": "Number of lines after cursor (1–1000)",
            },
        },
        "required": ["bg_id"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from sqlalchemy.future import select

        from app.core import workspaced_client as wsd
        from app.models.workspace import WorkspaceBgProc

        bg_id = str(tool_input.get("bg_id") or "").strip()
        try:
            tail = max(1, min(int(tool_input.get("tail") or 100), 1000))
            cursor_raw = tool_input.get("cursor")
            cursor = max(0, int(cursor_raw)) if cursor_raw is not None else None
            limit = max(1, min(int(tool_input.get("limit") or 100), 1000))
        except (TypeError, ValueError):
            return "Error: tail, cursor, and limit must be integers"

        if not bg_id:
            return "Error: bg_id is required"

        res = await ctx.db.execute(
            select(WorkspaceBgProc).where(WorkspaceBgProc.bg_id == bg_id)
        )
        proc = res.scalar_one_or_none()
        if proc is None:
            return f"Error: no background process with bg_id={bg_id!r}"

        # Read log via file API — no exec, no lock contention
        from app.core.path_contract import normalize_workspace_path
        try:
            log_path = normalize_workspace_path(proc.log_path)
        except ValueError as exc:
            return f"Error: invalid background log path: {exc}"
        content = await wsd.read_workspace_file(str(ctx.room_id), log_path)

        if content is None:
            return "\n".join([
                f"bg_id: {bg_id}",
                f"status: {proc.status}",
                f"pid: {proc.pid}",
                "[log file not yet created — process may still be starting up]",
            ])

        lines = content.splitlines()
        total = len(lines)
        if cursor is not None:
            start = min(cursor, total)
            page = lines[start : start + limit]
            next_cursor = start + len(page)
            window_label = f"lines {start + 1}-{next_cursor} of {total}"
            body = "\n".join(page)
            if body:
                body += "\n"
        else:
            start = max(0, total - tail)
            body = "\n".join(lines[start:])
            window_label = f"last {min(tail, total)} of {total} lines"

        observation_ref = ""
        try:
            observation = await record_observation(
                ctx.db,
                room_id=ctx.room_id,
                session_id=ctx.session_id,
                task_id=ctx.task_id,
                tool_name=self.name,
                canonical_args={
                    "bg_id": bg_id,
                    "cursor": start if cursor is not None else None,
                    "limit": limit if cursor is not None else tail,
                },
                payload="\n".join(lines),
                source_fingerprint=fingerprint("\n".join(lines)),
                metadata={"kind": "bash_log", "bg_id": bg_id, "log_path": log_path},
            )
            observation_ref = f"\nobservation_ref: {observation.ref}"
            if cursor is not None and observation.seen_in_session:
                return "\n".join(filter(None, [
                    f"bg_id: {bg_id}",
                    f"status: {proc.status}",
                    f"log_lines_total: {total}",
                    f"next_cursor: {next_cursor}",
                    observation.receipt(),
                ]))
        except Exception:
            pass

        result = "\n".join(filter(None, [
            f"bg_id: {bg_id}",
            f"status: {proc.status}",
            f"pid: {proc.pid}",
            f"log_lines_total: {total}",
            f"cmd: {proc.cmd}",
            f"\nlog ({window_label}):",
            mask_room_secrets(body, bg_secrets(bg_id)),
            f"next_cursor: {next_cursor}" if cursor is not None else "",
            observation_ref,
        ]))
        return result


REGISTRY.register(BashOutputTool())
