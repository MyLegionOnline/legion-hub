"""
Backward-compatibility shim — new code should import from app.core.tools directly.

Imports app.core.tools to populate REGISTRY, then re-exports the same public
API that proxy_session and other callers previously depended on.
"""
from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

# Importing this package populates REGISTRY with all workspace tools.
from app.core.tools import REGISTRY, ToolContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema lists — generated from the registry (same content as before)
# ---------------------------------------------------------------------------

ANTHROPIC_AGENT_TOOLS: list[dict[str, Any]] = REGISTRY.ant_schemas_for("agent")
OPENAI_AGENT_TOOLS: list[dict[str, Any]] = REGISTRY.oai_schemas_for("agent")
ANTHROPIC_CM_TOOLS: list[dict[str, Any]] = REGISTRY.ant_schemas_for("cm")
OPENAI_CM_TOOLS: list[dict[str, Any]] = REGISTRY.oai_schemas_for("cm")

_WORKSPACE_TOOL_NAMES: frozenset[str] = frozenset(
    t.name for t in REGISTRY.tools_for("agent")
) | frozenset(t.name for t in REGISTRY.tools_for("cm"))


def is_workspace_tool(name: str) -> bool:
    return REGISTRY.is_registered(name)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


async def workspace_tools_available(room_id: UUID, db: AsyncSession) -> bool:
    """True if a workspace record exists for this room.

    file/git tools (read_file, write_file, list_files, git_diff) work even when
    the container is stopped; bash requires the container to be running (the tool
    executor returns an error in that case).
    """
    from app.models.workspace import Workspace

    res = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    return res.scalar_one_or_none() is not None


# ---------------------------------------------------------------------------
# Executor (delegates to the registry)
# ---------------------------------------------------------------------------


async def execute_workspace_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    room_id: UUID,
    session_id: UUID,
    task_id: UUID | None,
    db: AsyncSession,
) -> str:
    """Execute one workspace tool call and return the string result."""
    from app.core.tools._session_state import (
        MAX_IDENTICAL_TOOL_ERRORS,
        record_tool_result,
        repeated_tool_error,
        restore_tool_failure,
    )
    from app.models.session import Session

    tool = REGISTRY.get(tool_name)
    if tool is None:
        return f"Error: unknown workspace tool {tool_name!r}"
    sid = str(session_id)
    session = await db.get(Session, session_id)
    context_state = dict(getattr(session, "context_state", None) or {})
    if context_state.get("tool_error_retry"):
        restore_tool_failure(sid, context_state["tool_error_retry"])
    blocked = repeated_tool_error(sid, tool_name, tool_input)
    if blocked is not None:
        return (
            "Error: identical tool call blocked after "
            f"{MAX_IDENTICAL_TOOL_ERRORS} failures "
            f"(category={blocked.get('category', 'tool_error')}, "
            f"signature={blocked.get('signature')}). "
            "Change the arguments or resolve the reported cause before retrying. "
            f"Last error: {blocked.get('last_error', '')}"
        )
    ctx = ToolContext(room_id=room_id, session_id=session_id, task_id=task_id, db=db)
    try:
        result = await tool.execute(ctx, tool_input)
    except Exception as exc:
        logger.warning(
            "workspace_tools: error executing %s for session %s: %s",
            tool_name,
            session_id,
            exc,
        )
        result = f"Error: {exc}"
    retry_state = record_tool_result(sid, tool_name, tool_input, result)
    if session is not None:
        context_state = dict(getattr(session, "context_state", None) or {})
        if retry_state:
            context_state["tool_error_retry"] = retry_state
        else:
            context_state.pop("tool_error_retry", None)
        session.context_state = context_state
        db.add(session)
        await db.flush()
    return result
