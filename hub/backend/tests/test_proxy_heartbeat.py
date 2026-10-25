"""Unit tests for the LLM heartbeat wrapper (chat upgrade stage 2).

Covers:
- fast coroutines return their result with no heartbeat emitted
- slow coroutines produce llm_heartbeat statuses with growing elapsed_s
- exceptions from the wrapped coroutine propagate unchanged

All tests are pure/mocked — no database required.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.proxy_session import _call_with_heartbeat


@pytest.mark.asyncio
async def test_fast_call_returns_result_without_heartbeat():
    room_id = uuid.uuid4()
    session_id = uuid.uuid4()

    async def fast():
        return ("text", None, 10, 20, "model-x")

    with patch("app.core.proxy_session.emit_status", new_callable=AsyncMock) as status:
        result = await _call_with_heartbeat(
            fast(),
            room_id=room_id,
            agent_id=None,
            session_id=session_id,
            task_id=None,
            interval=0.5,
        )

    assert result == ("text", None, 10, 20, "model-x")
    status.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_call_emits_heartbeats():
    room_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    session_id = uuid.uuid4()
    task_id = uuid.uuid4()

    async def slow():
        await asyncio.sleep(0.12)
        return "done"

    with patch("app.core.proxy_session.emit_status", new_callable=AsyncMock) as status:
        result = await _call_with_heartbeat(
            slow(),
            room_id=room_id,
            agent_id=agent_id,
            session_id=session_id,
            task_id=task_id,
            interval=0.05,
        )

    assert result == "done"
    assert status.await_count >= 1
    args, kwargs = status.await_args_list[0]
    assert args[0] == room_id
    assert kwargs["agent_id"] == agent_id
    assert kwargs["session_id"] == session_id
    assert kwargs["task_id"] == task_id
    assert "elapsed_s" in kwargs["extra"]


@pytest.mark.asyncio
async def test_exception_propagates():
    async def boom():
        raise ValueError("llm down")

    with patch("app.core.proxy_session.emit_status", new_callable=AsyncMock):
        with pytest.raises(ValueError, match="llm down"):
            await _call_with_heartbeat(
                boom(),
                room_id=uuid.uuid4(),
                agent_id=None,
                session_id=uuid.uuid4(),
                task_id=None,
                interval=0.5,
            )
