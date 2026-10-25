"""internal design notes Stage 3 — "Abort the request and retry".

- a manual restart drops the request in flight and sends it again at once,
  without burning a retry attempt; at most RESTART_LIMIT per call;
- an answer that already arrived is never thrown away;
- a DB-stamped request older than the attempt in flight is stale;
- the endpoint core signals the in-process loop of the side whose turn it is.

Pure/mocked — no database required.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core import llm_call_restart, llm_call_state
from app.core.proxy_session import (
    LLMCallRestarted,
    _apply_db_restart_request,
    _await_attempt,
    _call_with_heartbeat,
    _llm_request_with_retry,
)


async def _wait_for_phase(sid, phase, timeout=1.0):
    t0 = time.monotonic()
    while llm_call_state.snapshot(sid).get("phase") != phase:
        if time.monotonic() - t0 > timeout:
            raise AssertionError(f"phase {phase} not reached")
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_restart_resends_without_burning_an_attempt():
    sid = uuid.uuid4()
    calls = {"n": 0}
    seen: dict = {}

    async def hangs_once():
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.Event().wait()  # hung upstream
        seen.update(llm_call_state.snapshot(sid))
        return "ok"

    call = asyncio.ensure_future(_llm_request_with_retry(hangs_once, session_id=sid, label="t"))
    await _wait_for_phase(sid, llm_call_state.PHASE_LLM)
    await asyncio.sleep(0.01)
    assert llm_call_state.request_restart(sid) == llm_call_state.RESTART_OK

    assert await asyncio.wait_for(call, 1.0) == "ok"
    assert calls["n"] == 2
    assert seen["attempt"] == 1  # the retry budget is untouched
    assert seen["restarts"] == 1
    assert seen["last_error"].startswith("LLMCallRestarted")
    snap = llm_call_state.snapshot(sid)
    assert snap["llm_calls"] == 1 and "last_error" not in snap
    llm_call_state.clear(sid)


@pytest.mark.asyncio
async def test_restart_limit_per_call():
    sid = uuid.uuid4()
    gate = asyncio.Event()

    async def hangs_until_gate():
        await gate.wait()
        return "ok"

    call = asyncio.ensure_future(_llm_request_with_retry(hangs_until_gate, session_id=sid, label="t"))
    for _ in range(llm_call_state.RESTART_LIMIT):
        await _wait_for_phase(sid, llm_call_state.PHASE_LLM)
        assert llm_call_state.request_restart(sid) == llm_call_state.RESTART_OK
        # wait for the restarted attempt to begin
        t0 = time.monotonic()
        while not (
            llm_call_state.snapshot(sid).get("phase") == llm_call_state.PHASE_LLM
            and llm_call_state.snapshot(sid).get("restarts") == _ + 1
        ):
            assert time.monotonic() - t0 < 1.0
            await asyncio.sleep(0.005)

    assert llm_call_state.request_restart(sid) == llm_call_state.RESTART_LIMIT_REACHED
    gate.set()
    assert await asyncio.wait_for(call, 1.0) == "ok"
    llm_call_state.clear(sid)


@pytest.mark.asyncio
async def test_answer_already_in_wins_over_restart():
    event = asyncio.Event()

    async def answers():
        event.set()  # the restart races the answer
        return "answer"

    assert await _await_attempt(answers, event) == "answer"


@pytest.mark.asyncio
async def test_await_attempt_raises_restart():
    event = asyncio.Event()

    async def hangs():
        await asyncio.Event().wait()

    task = asyncio.ensure_future(_await_attempt(hangs, event))
    await asyncio.sleep(0.01)
    event.set()
    with pytest.raises(LLMCallRestarted):
        await asyncio.wait_for(task, 1.0)


def test_request_restart_refusals():
    sid = uuid.uuid4()
    assert llm_call_state.request_restart(sid) == llm_call_state.RESTART_NOT_IN_LLM
    assert llm_call_state.snapshot(sid) == {}  # refusal creates no state

    llm_call_state.begin_tool(sid, "read_file")
    assert llm_call_state.request_restart(sid) == llm_call_state.RESTART_NOT_IN_LLM

    llm_call_state.begin_attempt(sid, attempt=1, max_attempts=4)
    assert (
        llm_call_state.request_restart(sid, requested_at_wall=time.time() - 60)
        == llm_call_state.RESTART_STALE
    )
    llm_call_state.begin_attempt(sid, attempt=1, max_attempts=4, restarts=llm_call_state.RESTART_LIMIT)
    assert llm_call_state.request_restart(sid) == llm_call_state.RESTART_LIMIT_REACHED
    llm_call_state.clear(sid)


def test_snapshot_exposes_restart_budget():
    sid = uuid.uuid4()
    llm_call_state.begin_attempt(sid, attempt=2, max_attempts=4, restarts=1)
    snap = llm_call_state.snapshot(sid)
    assert snap["restarts"] == 1
    assert snap["restart_limit"] == llm_call_state.RESTART_LIMIT
    llm_call_state.end_attempt(sid)
    assert "restarts" not in llm_call_state.snapshot(sid)
    llm_call_state.clear(sid)


@pytest.mark.asyncio
async def test_heartbeat_checks_the_db_flag_each_tick():
    sid = uuid.uuid4()

    async def slow_turn():
        await asyncio.sleep(0.12)
        return "done"

    with patch("app.core.proxy_session.emit_status", new_callable=AsyncMock), \
         patch("app.core.proxy_session._apply_db_restart_request", new_callable=AsyncMock) as check:
        await _call_with_heartbeat(
            slow_turn(), room_id=uuid.uuid4(), agent_id=None,
            session_id=sid, task_id=None, interval=0.05,
        )
    assert check.await_count >= 1
    assert check.await_args.args == (sid,)


@pytest.mark.asyncio
async def test_db_flag_check_skips_the_db_outside_a_model_request():
    sid = uuid.uuid4()
    llm_call_state.begin_tool(sid, "grep")
    with patch("sqlalchemy.ext.asyncio.async_sessionmaker", side_effect=AssertionError("no DB")):
        await _apply_db_restart_request(sid)  # must not touch the DB
    llm_call_state.clear(sid)


def _task(**kw):
    base = {"id": uuid.uuid4(), "runtime_state": "running", "awaiting_user_kind": None}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_endpoint_core_signals_the_side_in_turn():
    task = _task()
    agent_sid, cm_sid = uuid.uuid4(), uuid.uuid4()
    llm_call_state.begin_run(
        task.id, run_id="r1", agent_session_id=agent_sid, cm_session_id=cm_sid,
        max_turns=80, deadline_monotonic=None,
    )
    llm_call_state.set_turn(task.id, 3, llm_call_state.SIDE_CHAT_MASTER)
    event = llm_call_state.begin_attempt(cm_sid, attempt=1, max_attempts=4)

    assert await llm_call_restart.request(task, db=None) == llm_call_restart.DELIVERED_NOW
    assert event.is_set()

    llm_call_state.end_attempt(cm_sid)
    with pytest.raises(llm_call_restart.RestartRefused) as exc:
        await llm_call_restart.request(task, db=None)
    assert exc.value.code == "not_in_llm"
    llm_call_state.end_run(task.id, "r1")


@pytest.mark.asyncio
async def test_endpoint_core_refuses_when_not_running():
    for task in (_task(runtime_state="paused"), _task(awaiting_user_kind="question")):
        with pytest.raises(llm_call_restart.RestartRefused) as exc:
            await llm_call_restart.request(task, db=None)
        assert exc.value.code == "not_running"
