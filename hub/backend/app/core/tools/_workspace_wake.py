"""Start the room's workspace on demand for tools that exec in the container.

A workspace can stop under a running task — idle stop, a quota switch to
another room, later hibernation (internal design notes stage 7).  File and git
tools keep working through the volume, but ``bash`` needs the container, and
the agent has no tool of its own to start it.  So a tool that needs exec wakes
the workspace itself, through the same quota-checked path the room UI uses
(``ensure_workspace_started``), and tells the agent in plain words when the
account limits do not allow it.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.tools.registry import ToolContext, release_tool_db

logger = logging.getLogger(__name__)

# How often to re-read a workspace another caller is already starting.
_POLL_INTERVAL_SEC = 1.0

_FILE_TOOLS_HINT = (
    "File and git tools (read_file, write_file, edit_file, grep, git_diff) still work on the "
    "room files without the container; only commands need it."
)


def start_error_message(exc: Any) -> str:
    """Agent-facing text for a :class:`WorkspaceStartError`."""
    code = getattr(exc, "error_code", None) or "start_failed"
    detail = getattr(exc, "message", None) or str(exc)
    if code in {"quota_exceeded", "monthly_hours_exceeded"}:
        return (
            f"Error: the workspace is stopped and could not be started ({code}): {detail} "
            "This is an account limit you cannot change — do not retry the command. "
            f"{_FILE_TOOLS_HINT} If the command is essential, tell the user the workspace "
            "needs a free slot (stop another room's workspace or raise the plan limit)."
        )
    if code == "server_busy":
        return (
            f"Error: the workspace is stopped and could not be started ({code}): {detail} "
            "The whole server is full, not this account; it frees up as other users stop. "
            f"Do not retry in a loop. {_FILE_TOOLS_HINT} If the command is essential, tell "
            "the user to try again in a few minutes."
        )
    return (
        f"Error: the workspace is stopped and could not be started ({code}): {detail} "
        f"{_FILE_TOOLS_HINT}"
    )


async def _read_workspace(ctx: ToolContext):
    from sqlalchemy.future import select

    from app.models.workspace import Workspace

    result = await ctx.db.execute(
        select(Workspace)
        .where(Workspace.room_id == ctx.room_id)
        .execution_options(populate_existing=True)
    )
    return result.scalar_one_or_none()


async def ensure_workspace_running(
    ctx: ToolContext, *, reason: str = "tool"
) -> tuple[Any | None, str | None]:
    """Return ``(workspace, None)`` once it runs, or ``(None, error_text)``.

    A running workspace costs one SELECT.  Otherwise the loop's transaction is
    committed before the start (stage 1.2: no connection held across the
    Docker call), the start runs synchronously, and a start already in flight
    from another caller is waited for up to ``start_timeout_sec``.
    """
    from app.core.workspace_lifecycle import (
        CONFIG,
        WorkspaceStartError,
        ensure_workspace_started,
    )

    ws = await _read_workspace(ctx)
    if ws is not None and ws.status == "running":
        return ws, None

    previous = ws.status if ws is not None else None
    logger.info(
        "workspace wake: room %s status=%s — starting for %s", ctx.room_id, previous, reason
    )
    await release_tool_db(ctx, reason=f"{reason}/workspace_start")
    try:
        await ensure_workspace_started(ctx.room_id, ctx.db, background=False)
    except WorkspaceStartError as exc:
        return None, start_error_message(exc)
    except Exception as exc:  # workspaced unreachable, Docker failure, ...
        logger.exception("workspace wake failed for room %s", ctx.room_id)
        return None, (
            f"Error: the workspace is stopped and could not be started: {exc}. "
            f"{_FILE_TOOLS_HINT}"
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + CONFIG.start_timeout_sec
    while True:
        ws = await _read_workspace(ctx)
        if ws is None:
            return None, "Error: workspace not found for this room"
        if ws.status == "running":
            return ws, None
        if ws.status != "starting" or loop.time() >= deadline:
            detail = f": {ws.last_error}" if getattr(ws, "last_error", None) else ""
            return None, (
                f"Error: the workspace did not start (status={ws.status!r}){detail}. "
                f"{_FILE_TOOLS_HINT}"
            )
        # Another caller (the room UI, a parallel tool) owns the start.
        await release_tool_db(ctx, reason=f"{reason}/workspace_wait")
        await asyncio.sleep(_POLL_INTERVAL_SEC)
