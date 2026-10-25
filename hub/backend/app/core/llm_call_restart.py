"""Abort the request and retry (internal design notes Stage 3).

A hung provider keeps one model request open until ``LLM_ATTEMPT_TIMEOUT_S``
(600 s).  The user can cut it short: the request in flight is dropped and the
same request is sent again at once, without burning a retry attempt and
without losing the turn — tool results the turn already produced stay.

Two delivery paths, one effect:

- the loop runs in this process → ``llm_call_state.request_restart`` sets the
  attempt's event right away;
- it does not (scale-out, another worker) → ``session.llm_call_restart_requested_at``
  is stamped on the task's open proxy sessions; the loop's heartbeat consumes
  the stamp within ``LLM_HEARTBEAT_SECONDS`` (``_apply_db_restart_request``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import llm_call_state
from app.models.session.base import Session

DELIVERED_NOW = "now"
DELIVERED_QUEUED = "queued"

_REFUSALS = {
    llm_call_state.RESTART_NOT_IN_LLM: (
        "not_in_llm",
        "There is no model request right now — nothing to abort.",
    ),
    llm_call_state.RESTART_LIMIT_REACHED: (
        "limit",
        f"This request has already been restarted {llm_call_state.RESTART_LIMIT} times — "
        "wait for the automatic timeout or pause the task.",
    ),
}


class RestartRefused(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def request(task: Any, db: AsyncSession) -> str:
    """``DELIVERED_NOW`` or ``DELIVERED_QUEUED``; ``RestartRefused`` otherwise."""
    from app.models.session.cm_agent_proxy import CmAgentProxySession

    if getattr(task, "awaiting_user_kind", None) or (getattr(task, "runtime_state", None) or "idle") != "running":
        raise RestartRefused("not_running", "The task is not running right now.")

    run = llm_call_state.run_snapshot(task.id)
    if run is not None:
        side = run.get("side")
        session_id = run["cm_session_id"] if side == llm_call_state.SIDE_CHAT_MASTER else run["agent_session_id"]
        outcome = llm_call_state.request_restart(session_id)
        if outcome == llm_call_state.RESTART_OK:
            return DELIVERED_NOW
        code, message = _REFUSALS.get(outcome, ("not_in_llm", "Could not abort the request."))
        raise RestartRefused(code, message)

    # The loop is not in this process: leave the request for its heartbeat.
    # Both sides are stamped — only the one in a model request acts on it, and
    # for the other the stamp predates its next attempt (stale, ignored).
    # The stamp lives on the base ``session`` table (joined inheritance), so
    # update it there by id rather than through the subclass.
    open_ids = (
        select(CmAgentProxySession.id)
        .where(
            CmAgentProxySession.task_id == task.id,
            CmAgentProxySession.closed_at.is_(None),
        )
    )
    result = await db.execute(
        update(Session)
        .where(Session.id.in_(open_ids))
        .values(llm_call_restart_requested_at=datetime.now(timezone.utc))
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    if not result.rowcount:
        raise RestartRefused("no_session", "The task has no open executor session.")
    return DELIVERED_QUEUED
