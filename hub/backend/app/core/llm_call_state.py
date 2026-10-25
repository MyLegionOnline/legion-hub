"""What a session is waiting on right now — for honest "thinking…" statuses.

internal design notes Stage 0.  ``_call_with_heartbeat`` wraps a whole agent
turn — every model request *and* every tool the tool loop runs — so its
``elapsed_s`` is the length of the turn, not of the request in flight.  A
780-second "thinking…" could be a hung provider or fifty healthy read_file
calls; nothing told them apart, and retries were visible only in the log.

This registry is written at the boundaries that already exist (one model
attempt in ``_llm_request_with_retry``, one tool call via the
``tool_call_*`` statuses) and read by the heartbeat, which puts the snapshot
into its ``extra``.  In-process on purpose: the Hub runs one uvicorn worker
and the heartbeat reads it every few seconds, so a DB mirror would only add
writes.  A cross-worker reader (the status card, Stage 1) gets one then.

Stage 3 adds the attempt's restart event: "Abort the request and retry" sets
it (``request_restart``) and ``_llm_request_with_retry`` drops the request
and sends it again without burning a retry attempt.

Nothing here may break a caller: every function is total and cheap.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

PHASE_LLM = "llm"
PHASE_TOOL = "tool"
PHASE_BETWEEN = "between"

# Errors are shown in a status pill; keep them one short line.
_ERROR_TEXT_LIMIT = 160

# Stage 3: "Abort the request and retry" per model call (one
# ``_llm_request_with_retry``).  A manual restart does not burn a retry
# attempt, so without a cap a user could keep a call alive forever.
RESTART_LIMIT = 3


@dataclass
class _CallState:
    phase: str = PHASE_BETWEEN
    label: str | None = None
    provider: str | None = None
    attempt: int = 0
    max_attempts: int = 0
    attempt_started: float | None = None
    # Wall clock of the same moment: a restart requested through the DB flag
    # (another worker) carries a timestamp, and a request older than the
    # attempt in flight was meant for an attempt that is already over.
    attempt_started_wall: float | None = None
    restarts: int = 0
    restart_event: asyncio.Event | None = None
    tool: str | None = None
    tool_started: float | None = None
    last_error: str | None = None
    last_ok: float | None = None
    llm_calls: int = 0
    touched: float = field(default_factory=time.monotonic)


_STATES: dict[str, _CallState] = {}
# A session whose loop died without a clean end leaves its entry behind;
# entries older than this are dropped on the next write.
_STALE_AFTER_S = 6 * 3600


def _key(session_id: Any) -> str | None:
    return str(session_id) if session_id is not None else None


def _state(session_id: Any) -> _CallState | None:
    key = _key(session_id)
    if key is None:
        return None
    state = _STATES.get(key)
    if state is None:
        _prune()
        state = _STATES[key] = _CallState()
    state.touched = time.monotonic()
    return state


def _prune() -> None:
    cutoff = time.monotonic() - _STALE_AFTER_S
    for key in [k for k, s in _STATES.items() if s.touched < cutoff]:
        _STATES.pop(key, None)


def describe_error(exc: BaseException) -> str:
    """``ReadTimeout``, ``HTTP 502``, ``LLMAttemptTimeout: no answer in 600s``."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    text = f"{name} (HTTP {status})" if isinstance(status, int) else name
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if message and message not in text:
        text = f"{text}: {message}"
    return text[:_ERROR_TEXT_LIMIT]


def begin_attempt(
    session_id: Any,
    *,
    attempt: int,
    max_attempts: int,
    label: str | None = None,
    provider: str | None = None,
    restarts: int = 0,
) -> asyncio.Event | None:
    """``attempt`` is 1-based; ``restarts`` — manual restarts of this call so far.

    Returns the event a manual restart sets (``None`` without a session).
    """
    state = _state(session_id)
    if state is None:
        return None
    state.phase = PHASE_LLM
    state.label = label
    state.provider = provider
    state.attempt = attempt
    state.max_attempts = max_attempts
    state.attempt_started = time.monotonic()
    state.attempt_started_wall = time.time()
    state.restarts = restarts
    state.restart_event = asyncio.Event()
    state.tool = None
    state.tool_started = None
    return state.restart_event


def end_attempt(session_id: Any, *, error: BaseException | None = None) -> None:
    state = _state(session_id)
    if state is None:
        return
    state.phase = PHASE_BETWEEN
    state.attempt_started = None
    state.attempt_started_wall = None
    state.restart_event = None
    if error is None:
        state.last_ok = time.monotonic()
        state.last_error = None
        state.llm_calls += 1
        state.attempt = 0
        state.restarts = 0
    else:
        state.last_error = describe_error(error)


RESTART_OK = "ok"
RESTART_NOT_IN_LLM = "not_in_llm"
RESTART_LIMIT_REACHED = "limit"
RESTART_STALE = "stale"


def request_restart(session_id: Any, *, requested_at_wall: float | None = None) -> str:
    """Abort the model request in flight so it is sent again (Stage 3).

    ``requested_at_wall`` (``time.time()``) comes with a request that waited in
    the DB: one made before the current attempt began is ``RESTART_STALE`` —
    that attempt already ended on its own.  Never creates state.
    """
    key = _key(session_id)
    state = _STATES.get(key) if key is not None else None
    if state is None or state.phase != PHASE_LLM or state.restart_event is None:
        return RESTART_NOT_IN_LLM
    if (
        requested_at_wall is not None
        and state.attempt_started_wall is not None
        and requested_at_wall < state.attempt_started_wall
    ):
        return RESTART_STALE
    if state.restarts >= RESTART_LIMIT:
        return RESTART_LIMIT_REACHED
    state.restart_event.set()
    return RESTART_OK


def begin_tool(session_id: Any, tool: str | None) -> None:
    state = _state(session_id)
    if state is None:
        return
    state.phase = PHASE_TOOL
    state.tool = tool
    state.tool_started = time.monotonic()


def end_tool(session_id: Any) -> None:
    state = _state(session_id)
    if state is None or state.phase != PHASE_TOOL:
        return
    state.phase = PHASE_BETWEEN
    state.tool_started = None


def clear(session_id: Any) -> None:
    key = _key(session_id)
    if key is not None:
        _STATES.pop(key, None)


# ---------------------------------------------------------------------------
# Run registry (Stage 1): what a task's proxy loop is on — turn N of M, whose
# turn it is, how long until the run deadline.  Those live only in the loop's
# locals, and the status card needs them.  Keyed by task, fenced by run_id so
# a finishing run never wipes its successor's entry.
# ---------------------------------------------------------------------------


@dataclass
class _RunState:
    run_id: str | None
    agent_session_id: str | None
    cm_session_id: str | None
    max_turns: int
    started: float
    deadline: float | None
    turn: int = 0
    side: str | None = None


_RUNS: dict[str, _RunState] = {}

SIDE_AGENT = "agent"
SIDE_CHAT_MASTER = "chat_master"


def begin_run(
    task_id: Any,
    *,
    run_id: str | None,
    agent_session_id: Any,
    cm_session_id: Any,
    max_turns: int,
    deadline_monotonic: float | None,
) -> None:
    key = _key(task_id)
    if key is None:
        return
    _RUNS[key] = _RunState(
        run_id=run_id,
        agent_session_id=_key(agent_session_id),
        cm_session_id=_key(cm_session_id),
        max_turns=max_turns,
        started=time.monotonic(),
        deadline=deadline_monotonic,
    )


def set_turn(task_id: Any, turn: int, side: str) -> None:
    run = _RUNS.get(_key(task_id) or "")
    if run is not None:
        run.turn = turn
        run.side = side


def end_run(task_id: Any, run_id: str | None) -> None:
    key = _key(task_id)
    run = _RUNS.get(key or "")
    if run is None or run.run_id != run_id:
        return
    _RUNS.pop(key, None)
    clear(run.agent_session_id)
    clear(run.cm_session_id)


def run_snapshot(task_id: Any) -> dict[str, Any] | None:
    """``None`` when no loop for this task runs in this process."""
    run = _RUNS.get(_key(task_id) or "")
    if run is None:
        return None
    now = time.monotonic()
    return {
        "run_id": run.run_id,
        "agent_session_id": run.agent_session_id,
        "cm_session_id": run.cm_session_id,
        # 1-based for people; the loop counts from 0.
        "turn": run.turn + 1,
        "max_turns": run.max_turns,
        "side": run.side,
        "elapsed_s": int(now - run.started),
        "deadline_in_s": int(run.deadline - now) if run.deadline is not None else None,
    }


def snapshot(session_id: Any) -> dict[str, Any]:
    """Heartbeat ``extra`` fields; ``{}`` when nothing was recorded."""
    key = _key(session_id)
    state = _STATES.get(key) if key is not None else None
    if state is None:
        return {}
    now = time.monotonic()
    out: dict[str, Any] = {"phase": state.phase, "llm_calls": state.llm_calls}
    if state.phase == PHASE_LLM and state.attempt_started is not None:
        out["attempt"] = state.attempt
        out["max_attempts"] = state.max_attempts
        out["attempt_elapsed_s"] = int(now - state.attempt_started)
        out["restarts"] = state.restarts
        out["restart_limit"] = RESTART_LIMIT
        if state.provider:
            out["provider"] = state.provider
    if state.phase == PHASE_TOOL and state.tool_started is not None:
        out["tool"] = state.tool
        out["tool_elapsed_s"] = int(now - state.tool_started)
    if state.last_error:
        out["last_error"] = state.last_error
    if state.last_ok is not None:
        out["last_ok_ago_s"] = int(now - state.last_ok)
    return out
