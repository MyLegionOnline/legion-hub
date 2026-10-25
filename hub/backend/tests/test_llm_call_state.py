"""internal design notes Stage 0 — what the "thinking…" heartbeat reports.

- one model request is capped (``LLMAttemptTimeout``) and retried, instead of
  hanging on keep-alives until the 40-minute proxy deadline;
- ``llm_call_state`` records attempt N of M, the previous error and the tool
  in flight, and the heartbeat carries that snapshot.

Pure/mocked — no database required.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core import llm_call_state
from app.core.proxy_session import (
    _LLM_RETRY_BACKOFF_S,
    LLMAttemptTimeout,
    _call_with_heartbeat,
    _is_retryable_llm_error,
    _llm_request_with_retry,
)


def test_attempt_timeout_is_retryable():
    assert _is_retryable_llm_error(LLMAttemptTimeout(600))


@pytest.mark.asyncio
async def test_hung_attempt_is_cut_and_retried():
    calls = {"n": 0}

    async def hangs_once():
        calls["n"] += 1
        if calls["n"] == 1:
            await asyncio.Event().wait()  # upstream never answers
        return {"ok": True}

    sid = uuid.uuid4()
    with patch("app.core.proxy_session._LLM_ATTEMPT_TIMEOUT_S", 0.05), \
         patch("app.core.proxy_session.asyncio.sleep", new_callable=AsyncMock):
        result = await _llm_request_with_retry(hangs_once, session_id=sid, label="t")

    assert result == {"ok": True}
    assert calls["n"] == 2
    snap = llm_call_state.snapshot(sid)
    assert snap["phase"] == llm_call_state.PHASE_BETWEEN
    assert snap["llm_calls"] == 1
    assert "last_error" not in snap  # cleared by the successful attempt
    llm_call_state.clear(sid)


@pytest.mark.asyncio
async def test_always_hung_gives_up_with_attempt_timeout():
    async def hangs():
        await asyncio.Event().wait()

    sid = uuid.uuid4()
    with patch("app.core.proxy_session._LLM_ATTEMPT_TIMEOUT_S", 0.02), \
         patch("app.core.proxy_session.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMAttemptTimeout):
            await _llm_request_with_retry(hangs, session_id=sid, label="t")
    assert "LLMAttemptTimeout" in llm_call_state.snapshot(sid)["last_error"]
    llm_call_state.clear(sid)


@pytest.mark.asyncio
async def test_retry_shows_attempt_and_previous_error_while_in_flight():
    sid = uuid.uuid4()
    seen: dict = {}
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("read timed out")
        seen.update(llm_call_state.snapshot(sid))
        return "ok"

    with patch("app.core.proxy_session.asyncio.sleep", new_callable=AsyncMock):
        await _llm_request_with_retry(flaky, session_id=sid, label="t", provider="openrouter")

    assert seen["phase"] == llm_call_state.PHASE_LLM
    assert seen["attempt"] == 2
    assert seen["max_attempts"] == len(_LLM_RETRY_BACKOFF_S) + 1
    assert seen["provider"] == "openrouter"
    assert seen["last_error"].startswith("ReadTimeout")
    llm_call_state.clear(sid)


def test_tool_phase_and_describe_error():
    sid = uuid.uuid4()
    llm_call_state.begin_tool(sid, "read_file")
    snap = llm_call_state.snapshot(sid)
    assert snap["phase"] == llm_call_state.PHASE_TOOL
    assert snap["tool"] == "read_file"
    llm_call_state.end_tool(sid)
    assert llm_call_state.snapshot(sid)["phase"] == llm_call_state.PHASE_BETWEEN
    llm_call_state.clear(sid)
    assert llm_call_state.snapshot(sid) == {}

    req = httpx.Request("POST", "https://x")
    exc = httpx.HTTPStatusError("bad gateway", request=req, response=httpx.Response(502, request=req))
    assert llm_call_state.describe_error(exc).startswith("HTTPStatusError (HTTP 502)")


def test_none_session_is_ignored():
    llm_call_state.begin_attempt(None, attempt=1, max_attempts=4)
    llm_call_state.begin_tool(None, "x")
    assert llm_call_state.snapshot(None) == {}


@pytest.mark.asyncio
async def test_heartbeat_carries_the_snapshot():
    sid = uuid.uuid4()

    async def slow_turn():
        llm_call_state.begin_tool(sid, "grep")
        await asyncio.sleep(0.12)
        return "done"

    with patch("app.core.proxy_session.emit_status", new_callable=AsyncMock) as status:
        await _call_with_heartbeat(
            slow_turn(), room_id=uuid.uuid4(), agent_id=None,
            session_id=sid, task_id=None, interval=0.05,
        )
    extra = status.await_args_list[0].kwargs["extra"]
    assert "elapsed_s" in extra
    assert extra["phase"] == llm_call_state.PHASE_TOOL
    assert extra["tool"] == "grep"
    llm_call_state.clear(sid)
