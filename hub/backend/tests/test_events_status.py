"""Unit tests for the ephemeral status channel (chat upgrade stage 1).

Covers:
- emit_status broadcasts over WS without any DB write
- payload shape: type=agent_status, status string, ids stringified, extra merged
- emit_status never raises even when the broadcast fails

All tests are pure/mocked — no database required.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.events import AgentStatus, emit_status


@pytest.mark.asyncio
async def test_emit_status_broadcasts_payload_without_db():
    room_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    session_id = uuid.uuid4()
    task_id = uuid.uuid4()

    with patch("app.core.events.manager.broadcast", new_callable=AsyncMock) as broadcast:
        await emit_status(
            room_id,
            AgentStatus.LLM_REQUEST_SENT,
            agent_id=agent_id,
            session_id=session_id,
            task_id=task_id,
            detail="turn 1",
            extra={"elapsed_s": 5},
        )

    broadcast.assert_awaited_once()
    payload, sent_room = broadcast.await_args.args
    assert sent_room == room_id
    assert payload["type"] == "agent_status"
    assert payload["status"] == "llm_request_sent"
    assert payload["room_id"] == str(room_id)
    assert payload["agent_id"] == str(agent_id)
    assert payload["session_id"] == str(session_id)
    assert payload["task_id"] == str(task_id)
    assert payload["detail"] == "turn 1"
    assert payload["elapsed_s"] == 5
    assert "ts" in payload


@pytest.mark.asyncio
async def test_emit_status_accepts_plain_string_and_omits_optional_ids():
    room_id = uuid.uuid4()

    with patch("app.core.events.manager.broadcast", new_callable=AsyncMock) as broadcast:
        await emit_status(room_id, "llm_heartbeat")

    payload, _ = broadcast.await_args.args
    assert payload["status"] == "llm_heartbeat"
    assert payload["agent_id"] is None
    assert payload["session_id"] is None
    assert payload["task_id"] is None
    assert payload["detail"] is None


@pytest.mark.asyncio
async def test_emit_status_swallows_broadcast_errors():
    room_id = uuid.uuid4()

    with patch(
        "app.core.events.manager.broadcast",
        new_callable=AsyncMock,
        side_effect=RuntimeError("ws down"),
    ):
        # Must not raise — the status channel never breaks the caller.
        await emit_status(room_id, AgentStatus.LLM_ERROR, detail="RuntimeError")
