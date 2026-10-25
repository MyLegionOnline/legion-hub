"""bash_kill tool — terminate a background bash process by bg_id."""
from __future__ import annotations

import asyncio
from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext
from app.core.tools._bash import _ws_lock


class BashKillTool(Tool):
    name = "bash_kill"
    description = (
        "Terminate a background process started with bash(background=true). "
        "Sends SIGTERM to the process group, waits briefly, then SIGKILL if still running."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "bg_id": {
                "type": "string",
                "description": "Background process ID returned by bash(background=true)",
            },
        },
        "required": ["bg_id"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = True
    mutating: bool = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from sqlalchemy.future import select

        from app.core import workspaced_client as wsd
        from app.models.workspace import Workspace, WorkspaceBgProc

        bg_id = str(tool_input.get("bg_id") or "").strip()
        if not bg_id:
            return "Error: bg_id is required"

        res = await ctx.db.execute(
            select(WorkspaceBgProc).where(WorkspaceBgProc.bg_id == bg_id)
        )
        proc = res.scalar_one_or_none()
        if proc is None:
            return f"Error: no background process with bg_id={bg_id!r}"

        if proc.status != "running":
            return f"bg_id: {bg_id}\nstatus: {proc.status} (already stopped — nothing to kill)"

        ws_res = await ctx.db.execute(
            select(Workspace).where(Workspace.room_id == ctx.room_id)
        )
        ws = ws_res.scalar_one_or_none()
        if ws is None or ws.status != "running":
            # Container gone — just mark as failed
            proc.status = "failed"
            ctx.db.add(proc)
            await ctx.db.flush()
            return f"bg_id: {bg_id}\nstatus: failed\n[workspace not running — process already dead]"

        rid = str(ctx.room_id)
        pid = proc.pid
        pgid = proc.pgid or pid

        if pid:
            # Kill by process group (covers child processes like node spawned by npm)
            kill_script = (
                f"kill -TERM -{pgid} 2>/dev/null || kill -TERM {pid} 2>/dev/null; "
                f"sleep 0.5; "
                f"kill -9 -{pgid} 2>/dev/null || kill -9 {pid} 2>/dev/null; "
                f"true"
            )
            async with _ws_lock(rid):
                try:
                    await asyncio.wait_for(
                        wsd.exec_in_workspace(rid, ["bash", "-c", kill_script], timeout_sec=5),
                        timeout=10,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass  # best-effort; always mark killed below

        proc.status = "killed"
        ctx.db.add(proc)
        await ctx.db.flush()

        return "\n".join([
            f"bg_id: {bg_id}",
            f"status: killed",
            f"pid: {pid}",
            f"pgid: {pgid}",
        ])


REGISTRY.register(BashKillTool())
