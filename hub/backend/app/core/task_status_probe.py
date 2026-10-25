"""Executor status — what a running task is doing, answered by the Hub.

internal design notes Stage 1.  While a model request is in flight the agent
reads nothing, so "check the status" cannot be answered by the agent when the
provider is the thing that hangs.  This module answers it from runtime data
only, no LLM: the task row, its open proxy sessions, and the in-process
``llm_call_state`` (turn, side, attempt, tool in flight).

``verdict`` is a pure function so the thresholds are unit-tested; the async
``collect`` is the thin DB layer around it.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import llm_call_state

LEVEL_OK = "ok"
LEVEL_SLOW = "slow"
LEVEL_STUCK = "stuck"
LEVEL_IDLE = "not_running"

# One model request past SLOW is worth a look; past STUCK it is most likely a
# hung upstream (the request itself is cut at LLM_ATTEMPT_TIMEOUT_S = 600).
# SLOW matches SLOW_ATTEMPT_S in frontend/src/hooks/useAgentStatuses.js.
SLOW_ATTEMPT_S = int(os.getenv("TASK_STATUS_SLOW_ATTEMPT_S", "180"))
STUCK_ATTEMPT_S = int(os.getenv("TASK_STATUS_STUCK_ATTEMPT_S", "360"))
SLOW_TOOL_S = int(os.getenv("TASK_STATUS_SLOW_TOOL_S", "300"))
# No loop in this process (connector executor, just dispatched, or another
# worker): judge by the sessions' last activity only.
STUCK_IDLE_S = int(os.getenv("TASK_STATUS_STUCK_IDLE_S", "600"))
DEADLINE_WARN_S = 300

_SIDE_LABEL = {
    llm_call_state.SIDE_AGENT: "the executor",
    llm_call_state.SIDE_CHAT_MASTER: "ChatMaster",
}

_NOT_RUNNING_TEXT = {
    "idle": "The task is not running right now.",
    "paused": "The task is paused.",
    "stop_requested": "The task is stopping.",
    "stopped": "The task is stopped.",
    "failed": "The task execution ended with an error.",
}


def _fmt_s(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} s"
    minutes, sec = divmod(seconds, 60)
    return f"{minutes} min {sec:02d} s" if minutes < 10 else f"{minutes} min"


def verdict(
    *,
    runtime_state: str,
    awaiting_user_kind: str | None,
    run: dict[str, Any] | None,
    call: dict[str, Any] | None,
    idle_s: int | None,
) -> dict[str, str]:
    """``{"level", "text"}`` for the card header.

    ``call`` is the ``llm_call_state.snapshot`` of the side whose turn it is;
    ``idle_s`` is seconds since the freshest session activity.
    """
    if awaiting_user_kind:
        return {"level": LEVEL_IDLE, "text": "The task is waiting for your reply."}
    if runtime_state != "running":
        return {
            "level": LEVEL_IDLE,
            "text": _NOT_RUNNING_TEXT.get(runtime_state, f"State: {runtime_state}."),
        }

    if run is None or not call:
        if idle_s is not None and idle_s >= STUCK_IDLE_S:
            return {
                "level": LEVEL_STUCK,
                "text": f"No activity for {_fmt_s(idle_s)}.",
            }
        return {"level": LEVEL_OK, "text": "Working."}

    who = _SIDE_LABEL.get(run.get("side") or "", "the executor")
    phase = call.get("phase")
    level, text = LEVEL_OK, f"{who.capitalize()} is working."

    if phase == llm_call_state.PHASE_LLM:
        waited = int(call.get("attempt_elapsed_s") or 0)
        attempt = int(call.get("attempt") or 1)
        retry = (
            f" Attempt {attempt} of {call.get('max_attempts')}"
            + (f", previous error: {call['last_error']}." if call.get("last_error") else ".")
            if attempt > 1 else ""
        )
        if waited >= STUCK_ATTEMPT_S:
            level = LEVEL_STUCK
            text = (
                f"The model has not responded for {_fmt_s(waited)} — the provider seems to have hung. "
                "The request will break off and repeat automatically, or you can abort it manually."
            )
        elif waited >= SLOW_ATTEMPT_S or attempt > 1:
            level = LEVEL_SLOW
            text = f"{who.capitalize()} has been waiting for the model for a long time: {_fmt_s(waited)}."
        else:
            text = f"{who.capitalize()} is waiting for the model: {_fmt_s(waited)}."
        text += retry
    elif phase == llm_call_state.PHASE_TOOL:
        ran = int(call.get("tool_elapsed_s") or 0)
        tool = call.get("tool") or "a tool"
        if ran >= SLOW_TOOL_S:
            level = LEVEL_SLOW
            text = f"Tool {tool} has been running for {_fmt_s(ran)}."
        else:
            text = f"{who.capitalize()} is running {tool}."

    deadline = run.get("deadline_in_s")
    if deadline is not None and deadline <= DEADLINE_WARN_S:
        text += f" {_fmt_s(max(0, deadline))} left until the task time limit."
        if level == LEVEL_OK:
            level = LEVEL_SLOW
    return {"level": level, "text": text}


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def collect(task: Any, db: AsyncSession) -> dict[str, Any]:
    """The status card payload for ``task``."""
    from app.models.agent import Agent
    from app.models.session.cm_agent_proxy import CmAgentProxySession

    now = datetime.now(timezone.utc)
    rows = (
        await db.execute(
            select(CmAgentProxySession, Agent.nickname)
            .join(Agent, Agent.id == CmAgentProxySession.agent_id, isouter=True)
            .where(
                CmAgentProxySession.task_id == task.id,
                CmAgentProxySession.closed_at.is_(None),
            )
            .order_by(CmAgentProxySession.started_at.asc())
        )
    ).all()

    run = llm_call_state.run_snapshot(task.id)
    sessions: list[dict[str, Any]] = []
    freshest: datetime | None = None
    for session, nickname in rows:
        last = _aware(session.last_activity_at) or _aware(session.started_at)
        if last and (freshest is None or last > freshest):
            freshest = last
        sessions.append({
            "session_id": str(session.id),
            "side": session.side,
            "agent_id": str(session.agent_id) if session.agent_id else None,
            "agent_nickname": nickname,
            "started_at": _aware(session.started_at),
            "last_activity_at": last,
            "idle_s": int((now - last).total_seconds()) if last else None,
            "call": llm_call_state.snapshot(session.id),
        })

    active_call: dict[str, Any] | None = None
    if run is not None and run.get("side"):
        wanted = run["agent_session_id"] if run["side"] == llm_call_state.SIDE_AGENT else run["cm_session_id"]
        active_call = llm_call_state.snapshot(wanted) or None

    from sqlalchemy import func

    from app.models.task_interjection import TaskInterjection

    pending, unanswered = (
        await db.execute(
            select(
                func.count().filter(TaskInterjection.delivered_at.is_(None)),
                func.count().filter(
                    TaskInterjection.delivered_at.is_not(None),
                    TaskInterjection.answered_at.is_(None),
                ),
            ).where(TaskInterjection.task_id == task.id)
        )
    ).one()

    idle_s = int((now - freshest).total_seconds()) if freshest else None
    runtime_state = getattr(task, "runtime_state", None) or "idle"
    state = getattr(task, "state", None)
    return {
        "task_id": str(task.id),
        "task_state": getattr(state, "value", state),
        "runtime_state": runtime_state,
        "awaiting_user_kind": getattr(task, "awaiting_user_kind", None),
        "run": run,
        "active_call": active_call,
        "sessions": sessions,
        "idle_s": idle_s,
        # internal design notes Stage 2: user messages still waiting for the
        # next model-request boundary / delivered but not answered yet.
        "interjections": {"pending": int(pending or 0), "unanswered": int(unanswered or 0)},
        "checked_at": now,
        "verdict": verdict(
            runtime_state=runtime_state,
            awaiting_user_kind=getattr(task, "awaiting_user_kind", None),
            run=run,
            call=active_call,
            idle_s=idle_s,
        ),
    }
