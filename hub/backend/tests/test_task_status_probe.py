"""internal design notes Stage 1 — "Executor status" verdict and run registry.

Pure — no database required.
"""

from __future__ import annotations

import time
import uuid

from app.core import llm_call_state as lcs
from app.core.task_status_probe import (
    LEVEL_IDLE,
    LEVEL_OK,
    LEVEL_SLOW,
    LEVEL_STUCK,
    SLOW_ATTEMPT_S,
    STUCK_ATTEMPT_S,
    STUCK_IDLE_S,
    verdict,
)

_RUN = {"side": lcs.SIDE_AGENT, "deadline_in_s": 1800}


def _llm(waited: int, attempt: int = 1, last_error: str | None = None) -> dict:
    call = {
        "phase": lcs.PHASE_LLM, "attempt": attempt, "max_attempts": 4,
        "attempt_elapsed_s": waited,
    }
    if last_error:
        call["last_error"] = last_error
    return call


def _v(**kw):
    base = {"runtime_state": "running", "awaiting_user_kind": None, "run": _RUN, "call": None, "idle_s": 5}
    base.update(kw)
    return verdict(**base)


def test_not_running_states():
    assert _v(runtime_state="paused")["level"] == LEVEL_IDLE
    assert "paus" in _v(runtime_state="paused")["text"]
    assert _v(awaiting_user_kind="question")["text"] == "The task is waiting for your reply."


def test_fresh_model_request_is_ok():
    out = _v(call=_llm(20))
    assert out["level"] == LEVEL_OK
    assert "waiting for the model" in out["text"]


def test_long_request_is_slow_then_stuck():
    assert _v(call=_llm(SLOW_ATTEMPT_S))["level"] == LEVEL_SLOW
    stuck = _v(call=_llm(STUCK_ATTEMPT_S))
    assert stuck["level"] == LEVEL_STUCK
    assert "provider seems to have hung" in stuck["text"]


def test_retry_is_slow_and_names_previous_error():
    out = _v(call=_llm(10, attempt=2, last_error="ReadTimeout"))
    assert out["level"] == LEVEL_SLOW
    assert "Attempt 2 of 4" in out["text"]
    assert "ReadTimeout" in out["text"]


def test_tool_phase():
    out = _v(call={"phase": lcs.PHASE_TOOL, "tool": "read_file", "tool_elapsed_s": 3})
    assert out["level"] == LEVEL_OK and "read_file" in out["text"]
    assert _v(call={"phase": lcs.PHASE_TOOL, "tool": "bash", "tool_elapsed_s": 900})["level"] == LEVEL_SLOW


def test_no_loop_in_process_falls_back_to_activity():
    assert _v(run=None, idle_s=30)["level"] == LEVEL_OK
    assert _v(run=None, idle_s=STUCK_IDLE_S)["level"] == LEVEL_STUCK


def test_near_deadline_warns():
    out = _v(run={"side": lcs.SIDE_AGENT, "deadline_in_s": 120}, call=_llm(5))
    assert out["level"] == LEVEL_SLOW
    assert "time limit" in out["text"]


def test_run_registry_is_fenced_by_run_id():
    task_id, agent_sid, cm_sid = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    lcs.begin_run(
        task_id, run_id="r1", agent_session_id=agent_sid, cm_session_id=cm_sid,
        max_turns=40, deadline_monotonic=time.monotonic() + 100,
    )
    lcs.set_turn(task_id, 2, lcs.SIDE_CHAT_MASTER)
    lcs.begin_tool(agent_sid, "grep")
    snap = lcs.run_snapshot(task_id)
    assert snap["turn"] == 3 and snap["max_turns"] == 40
    assert snap["side"] == lcs.SIDE_CHAT_MASTER
    assert 0 < snap["deadline_in_s"] <= 100

    lcs.end_run(task_id, "stale-run")  # an older generation finishing late
    assert lcs.run_snapshot(task_id) is not None
    lcs.end_run(task_id, "r1")
    assert lcs.run_snapshot(task_id) is None
    assert lcs.snapshot(agent_sid) == {}  # session call state dropped with the run
