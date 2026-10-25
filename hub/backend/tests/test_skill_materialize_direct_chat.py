"""Skill materialization must also run for direct @agent chat, not only the
task/ChatMaster proxy loop (internal design notes Stage 3 follow-up).

Mirrors the setup in test_mcp_direct_chat.py: process_user_turn is exercised
directly with everything below it mocked — no DB, no workspace volume.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.user_sessions import _LLMTurnResult, process_user_turn
from app.models.agent import Agent


def _make_session(kind: str = "user_agent_p2", agent_id: uuid.UUID | None = None) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.context_snapshot = None
    s.agent_id = agent_id or uuid.uuid4()
    s.chat_master_agent_id = None
    s.goal_id = uuid.uuid4()
    s.task_id = None
    s.summary = None
    s.pending_actions = None
    s.started_at = datetime.now(timezone.utc)
    s.last_context_window_tokens = 0
    return s


def _make_db(agent_row: MagicMock) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    db.rollback = AsyncMock()
    db.get = AsyncMock(return_value=agent_row)
    return db


@pytest.mark.asyncio
async def test_process_user_turn_materializes_skills_for_user_agent_session():
    """A plain @agent message (no task) must still materialize its bundled skills."""
    session = _make_session("user_agent_p2")
    agent_row = MagicMock()
    agent_row.id = session.agent_id
    agent_row.owner_id = uuid.uuid4()
    db = _make_db(agent_row)

    llm_result = _LLMTurnResult(text="ok", tool_input=None, tokens_in=1, tokens_out=1, model="m")

    with (
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
        patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
        patch("app.core.mcp_gateway.room_mcp_tools", new_callable=AsyncMock, return_value=([], [], {})),
        patch("app.core.user_sessions._call_with_ws_loop", new_callable=AsyncMock, return_value=llm_result),
        patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock),
        patch(
            "app.core.skill_materialize.materialize_agent_skills", new_callable=AsyncMock
        ) as mock_materialize,
    ):
        await process_user_turn(session, "hello", db)

    mock_materialize.assert_awaited_once_with(session.room_id, agent_row, db)


@pytest.mark.asyncio
async def test_process_user_turn_equips_chat_master_sessions_like_agent_ones():
    """user ↔ ChatMaster sessions resolve the ChatMaster's own Agent row: its
    skills are materialized and room MCP tools are discovered under its owner,
    so ChatMaster sees the same capabilities it coordinates for other agents."""
    session = _make_session("user_cm_p2")
    session.chat_master_agent_id = uuid.uuid4()
    session.agent_id = None
    cm_row = MagicMock()
    cm_row.id = session.chat_master_agent_id
    cm_row.owner_id = uuid.uuid4()
    db = _make_db(cm_row)

    llm_result = _LLMTurnResult(text="ok", tool_input=None, tokens_in=1, tokens_out=1, model="m")

    with (
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=False),
        patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
        patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
        patch(
            "app.core.mcp_gateway.room_mcp_tools", new_callable=AsyncMock, return_value=([], [], {})
        ) as mock_discover,
        patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=llm_result),
        patch("app.core.user_sessions._preflight_direct_user_request", new_callable=AsyncMock) as mock_preflight,
        patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock),
        patch(
            "app.core.skill_materialize.materialize_agent_skills", new_callable=AsyncMock
        ) as mock_materialize,
    ):
        mock_preflight.return_value = MagicMock(rejected_reason=None, messages=[{"role": "user", "content": "hi"}])
        await process_user_turn(session, "hi", db)

    db.get.assert_any_await(Agent, session.chat_master_agent_id)
    mock_materialize.assert_awaited_once_with(session.room_id, cm_row, db)
    mock_discover.assert_awaited_once_with(session.room_id, cm_row.owner_id, db)


@pytest.mark.asyncio
async def test_materialization_failure_does_not_break_the_turn():
    """Materialization is best-effort — a failure must not abort the chat turn."""
    session = _make_session("user_agent_p2")
    agent_row = MagicMock()
    agent_row.id = session.agent_id
    agent_row.owner_id = uuid.uuid4()
    db = _make_db(agent_row)

    llm_result = _LLMTurnResult(text="ok", tool_input=None, tokens_in=1, tokens_out=1, model="m")

    with (
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
        patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
        patch("app.core.mcp_gateway.room_mcp_tools", new_callable=AsyncMock, return_value=([], [], {})),
        patch("app.core.user_sessions._call_with_ws_loop", new_callable=AsyncMock, return_value=llm_result),
        patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock),
        patch(
            "app.core.skill_materialize.materialize_agent_skills",
            new_callable=AsyncMock,
            side_effect=RuntimeError("workspaced unreachable"),
        ),
    ):
        still_open = await process_user_turn(session, "hello", db)

    assert still_open is True
