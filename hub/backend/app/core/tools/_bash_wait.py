"""bash_wait tool — block until a background bash process exits (or a timeout).

Polling ``bash_output`` costs one LLM step per poll; a package install or a long
build then eats the tool-call budget while nothing useful happens. One
``bash_wait`` call waits server-side instead. The wait is not counted against the
turn's tool-call budget and is credited back to the proxy run's wall clock
(``app.core.run_clock``).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from app.core import run_clock
from app.core.secrets.room_masking import bg_secrets, mask_room_secrets
from app.core.tools.registry import REGISTRY, Tool, ToolContext, release_tool_db

logger = logging.getLogger(__name__)

# Tool name the loops exempt from their tool-call budget.
TOOL_NAME = "bash_wait"

MAX_WAIT_SECONDS = int(os.getenv("BASH_WAIT_MAX_SECONDS", "600"))
_POLL_SECONDS = 3.0
_LIVENESS_EVERY_SECONDS = 30.0
_TOUCH_EVERY_SECONDS = 60.0
_FENCE_EVERY_SECONDS = 10.0
_TAIL_LINES = 40


class BashWaitTool(Tool):
    name = TOOL_NAME
    description = (
        "Wait for a background process started with bash(background=true) to finish, "
        f"for up to max_wait_sec (1–{MAX_WAIT_SECONDS}). Returns as soon as the process exits, with its "
        "exit code and the tail of its log; if it is still running at the deadline the status is "
        "'running' — call bash_wait again. Use this instead of polling bash_output in a loop: it "
        "does not use up your tool-call budget and the waiting time does not count against the "
        "task time limit. Run long installs/builds with bash(background=true) and then bash_wait."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "bg_id": {
                "type": "string",
                "description": "Background process ID returned by bash(background=true)",
            },
            "max_wait_sec": {
                "type": "integer",
                "default": 120,
                "description": f"Longest time to wait in this call (1–{MAX_WAIT_SECONDS} seconds)",
            },
        },
        "required": ["bg_id"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from sqlalchemy import update
        from sqlalchemy.future import select

        from app.core import workspaced_client as wsd
        from app.models.session import Session
        from app.models.workspace import WorkspaceBgProc

        bg_id = str(tool_input.get("bg_id") or "").strip()
        if not bg_id:
            return "Error: bg_id is required"
        try:
            max_wait = max(1, min(int(tool_input.get("max_wait_sec") or 120), MAX_WAIT_SECONDS))
        except (TypeError, ValueError):
            return "Error: max_wait_sec must be an integer"

        proc = (
            await ctx.db.execute(select(WorkspaceBgProc).where(WorkspaceBgProc.bg_id == bg_id))
        ).scalar_one_or_none()
        if proc is None:
            return f"Error: no background process with bg_id={bg_id!r}"

        rid = str(ctx.room_id)
        log_path = proc.log_path
        exit_path = f"{log_path}.exit"
        pid = proc.pid
        status = proc.status
        exit_code: int | None = None
        started = time.monotonic()
        last_alive_check = started
        last_touch = started

        # Never hold a pool connection across the wait (hub_scale_out_plan §8.2).
        await release_tool_db(ctx, reason="bash_wait")
        stopped = False
        last_fence_check = started
        # Credit the wait to the run's deadline as it happens, not afterwards.
        async with run_clock.accruing(ctx.task_id):
            while status == "running":
                exit_code = await _read_exit_code(wsd, rid, exit_path)
                if exit_code is not None:
                    break
                now = time.monotonic()
                if now - started >= max_wait:
                    break
                if ctx.run_id and now - last_fence_check >= _FENCE_EVERY_SECONDS:
                    # A Stop/Cancel/Pause must not wait out a 10-minute block.
                    last_fence_check = now
                    from app.core.task_runtime import is_run_current

                    try:
                        if not await is_run_current(ctx.task_id, ctx.run_id, ctx.db):
                            stopped = True
                            break
                    except Exception:
                        logger.debug("bash_wait: run fence check failed", exc_info=True)
                    finally:
                        await release_tool_db(ctx, reason="bash_wait")
                if now - last_touch >= _TOUCH_EVERY_SECONDS:
                    # Keep the stall watchdog quiet: the session is alive, just waiting.
                    last_touch = now
                    try:
                        await ctx.db.execute(
                            update(Session)
                            .where(Session.id == ctx.session_id)
                            .values(last_activity_at=datetime.now(timezone.utc))
                        )
                        await ctx.db.commit()
                    except Exception:
                        logger.debug("bash_wait: activity touch failed", exc_info=True)
                if pid and now - last_alive_check >= _LIVENESS_EVERY_SECONDS:
                    last_alive_check = now
                    if not await _pid_alive(wsd, rid, pid):
                        # Died without writing an exit file (killed, or started before
                        # exit files existed); one more look in case it just finished.
                        exit_code = await _read_exit_code(wsd, rid, exit_path)
                        status = "failed" if exit_code is None else status
                        break
                await asyncio.sleep(min(_POLL_SECONDS, max(0.1, max_wait - (now - started))))

        if stopped:
            return (
                f"bg_id: {bg_id}\nstatus: {status}\n"
                "[the task was stopped or paused while waiting — the process keeps "
                "running; stop here]"
            )
        if exit_code is not None:
            status = "done" if exit_code == 0 else "failed"
        if status in ("done", "failed") and proc.status == "running":
            try:
                refreshed = (
                    await ctx.db.execute(
                        select(WorkspaceBgProc).where(WorkspaceBgProc.bg_id == bg_id)
                    )
                ).scalar_one_or_none()
                if refreshed is not None and refreshed.status == "running":
                    refreshed.status = status
                    ctx.db.add(refreshed)
                    await ctx.db.flush()
            except Exception:
                logger.debug("bash_wait: status update failed", exc_info=True)

        content = await wsd.read_workspace_file(rid, log_path)
        lines = (content or "").splitlines()
        tail = mask_room_secrets("\n".join(lines[-_TAIL_LINES:]), bg_secrets(bg_id))
        return "\n".join(filter(None, [
            f"bg_id: {bg_id}",
            f"status: {status}",
            f"exit_code: {exit_code}" if exit_code is not None else "",
            f"waited_sec: {int(time.monotonic() - started)}",
            f"log_lines_total: {len(lines)}",
            f"\nlog (last {min(_TAIL_LINES, len(lines))} lines):",
            tail,
            "" if status != "running" else
            "\n[still running — call bash_wait again, or bash_kill(bg_id) to stop it]",
        ]))


async def _read_exit_code(wsd: Any, rid: str, exit_path: str) -> int | None:
    try:
        raw = await wsd.read_workspace_file(rid, exit_path)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


async def _pid_alive(wsd: Any, rid: str, pid: int) -> bool:
    """True unless the process is demonstrably gone (errors count as alive)."""
    try:
        result = await asyncio.wait_for(
            wsd.exec_in_workspace(rid, ["bash", "-c", f"kill -0 {int(pid)}"], timeout_sec=5),
            timeout=10,
        )
    except Exception:
        return True
    return result.get("exit_code", 0) == 0


REGISTRY.register(BashWaitTool())
