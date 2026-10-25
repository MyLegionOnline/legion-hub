"""Direct-chat (@agent) MCP tool discovery + dispatch (internal design notes §9.3).

Before this fix, room MCP servers were only discovered in the task/proxy
loop (proxy_session.py). An ordinary @agent message — no task, no plan —
never saw `room_mcp_server` tools at all. These tests guard the direct-chat
path specifically: process_user_turn must call room_mcp_tools() for a plain
user_agent_* session, and the ws loop must route a resulting `mcp__*` tool
call through the Gateway client (execute_mcp_tool), not the builtin registry.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.mcp_gateway import McpToolBinding
from app.core.user_sessions import _LLMTurnResult, _call_with_ws_loop, process_user_turn


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
    s.task_id = None  # ordinary @agent message: no task, no plan
    s.summary = None
    s.pending_actions = None
    s.started_at = datetime.now(timezone.utc)
    s.last_context_window_tokens = 0
    return s


def _make_db(agent_owner_id: uuid.UUID) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    db.rollback = AsyncMock()
    agent_row = MagicMock()
    agent_row.owner_id = agent_owner_id
    db.get = AsyncMock(return_value=agent_row)
    return db


@pytest.mark.asyncio
async def test_process_user_turn_discovers_room_mcp_tools_without_a_task():
    """A plain @agent message (no task_id) must still trigger room MCP discovery."""
    owner_id = uuid.uuid4()
    session = _make_session("user_agent_p2")
    db = _make_db(owner_id)

    llm_result = _LLMTurnResult(
        text="search done", tool_input=None, tokens_in=10, tokens_out=5,
        model="claude-haiku-4-5-20251001",
    )

    with (
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=False),
        patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
        patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
        patch(
            "app.core.mcp_gateway.room_mcp_tools",
            new_callable=AsyncMock,
            return_value=(
                [{"name": "mcp__mcp_searxng__searxng_web_search", "description": "d", "input_schema": {}}],
                [{"type": "function", "function": {"name": "mcp__mcp_searxng__searxng_web_search"}}],
                {"mcp__mcp_searxng__searxng_web_search": MagicMock()},
            ),
        ) as mock_discover,
        patch("app.core.user_sessions._call_with_ws_loop", new_callable=AsyncMock, return_value=llm_result) as mock_loop,
        patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock),
    ):
        still_open = await process_user_turn(session, "find news about Legion", db)

    assert still_open is True
    # Discovery ran for THIS session's room/owner, not gated on a task/plan.
    mock_discover.assert_awaited_once_with(session.room_id, owner_id, db)
    # The discovered bindings/schemas must reach the ws loop even though
    # workspace tools are unavailable — MCP alone is enough to enter it.
    assert mock_loop.call_args.kwargs["mcp_user_id"] == owner_id
    assert "mcp__mcp_searxng__searxng_web_search" in mock_loop.call_args.kwargs["mcp_bindings"]
    assert mock_loop.call_args.kwargs["mcp_ant_tools"]


@pytest.mark.asyncio
async def test_process_user_turn_discovers_room_mcp_tools_for_chat_master_sessions():
    """A user ↔ ChatMaster dialogue receives the coordinator's room MCP tools."""
    owner_id = uuid.uuid4()
    session = _make_session("user_cm_p2")
    session.chat_master_agent_id = uuid.uuid4()
    session.agent_id = None
    db = _make_db(owner_id)

    llm_result = _LLMTurnResult(text="ok", tool_input=None, tokens_in=1, tokens_out=1, model="m")

    with (
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=False),
        patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
        patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
        patch(
            "app.core.mcp_gateway.room_mcp_tools",
            new_callable=AsyncMock,
            return_value=(
                [{"name": "mcp__mcp-vision__describe_image", "description": "d", "input_schema": {}}],
                [{"type": "function", "function": {"name": "mcp__mcp-vision__describe_image"}}],
                {"mcp__mcp-vision__describe_image": MagicMock()},
            ),
        ) as mock_discover,
        patch("app.core.user_sessions._call_with_ws_loop", new_callable=AsyncMock, return_value=llm_result) as mock_loop,
        patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock),
    ):
        still_open = await process_user_turn(session, "hi", db)

    assert still_open is True
    mock_discover.assert_awaited_once_with(session.room_id, owner_id, db)
    assert mock_loop.call_args.kwargs["mcp_user_id"] == owner_id
    assert "mcp__mcp-vision__describe_image" in mock_loop.call_args.kwargs["mcp_bindings"]


@pytest.mark.asyncio
async def test_ws_loop_routes_mcp_tool_call_through_gateway_not_registry():
    """An mcp__* tool_use block must dispatch via execute_mcp_tool, and its
    result must feed back into the next turn (proving a real round trip, not
    a stub)."""
    room_id = uuid.uuid4()
    session_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    user_id = uuid.uuid4()
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)

    binding = McpToolBinding(
        installed_item_id=uuid.uuid4(),
        market_item_id=uuid.uuid4(),
        market_slug="mcp-searxng",
        tool_name="searxng_web_search",
        custom_config=None,
        isolation="stateless",
    )
    tool_name = "mcp__mcp_searxng__searxng_web_search"

    tool_use_response = MagicMock()
    tool_use_response.raise_for_status = MagicMock()
    tool_use_response.json.return_value = {
        "usage": {"input_tokens": 20, "output_tokens": 5},
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "id": "t1", "name": tool_name, "input": {"query": "legion mcp"}},
        ],
    }
    final_response = MagicMock()
    final_response.raise_for_status = MagicMock()
    final_response.json.return_value = {
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Found 3 results about Legion."}],
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=[tool_use_response, final_response])
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.core.user_sessions.httpx.AsyncClient", return_value=mock_context),
        patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch(
            "app.core.mcp_gateway.execute_mcp_tool", new_callable=AsyncMock,
            return_value='{"result": "3 results"}',
        ) as mock_exec,
    ):
        result = await _call_with_ws_loop(
            {"provider": "anthropic", "model": "m", "api_key": "k"},
            "SYS",
            [{"role": "user", "content": "search legion"}],
            "user_agent_p2",
            room_id, session_id, agent_id, db,
            ant_ws_tools=[],
            oai_ws_tools=[],
            mcp_ant_tools=[{"name": tool_name, "description": "search", "input_schema": {"type": "object"}}],
            mcp_oai_tools=[],
            mcp_bindings={tool_name: binding},
            mcp_user_id=user_id,
        )

    assert result.text == "Found 3 results about Legion."
    mock_exec.assert_awaited_once_with(
        binding,
        user_id,
        room_id,
        {"query": "legion mcp"},
        db=db,
        request_id=session_id,
    )
    # The Gateway result (not a builtin-registry placeholder) must have been
    # sent back to the model as the tool_result content.
    second_call_messages = mock_client.post.call_args_list[1].kwargs["json"]["messages"]
    tool_result_block = next(
        block
        for message in second_call_messages
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_result"
    )
    assert tool_result_block["content"] == '{"result": "3 results"}'


@pytest.mark.asyncio
async def test_ws_loop_stops_before_second_tool_after_external_close():
    """Pause/Stop must fence a direct agent loop between tool calls."""
    room_id, session_id, agent_id, user_id = (uuid.uuid4() for _ in range(4))
    tool_name = "mcp__mcp_searxng__searxng_web_search"
    binding = McpToolBinding(
        installed_item_id=uuid.uuid4(),
        market_item_id=uuid.uuid4(),
        market_slug="mcp-searxng",
        tool_name="searxng_web_search",
        custom_config=None,
        isolation="stateless",
    )
    db = _make_db(user_id)

    tool_use_response = MagicMock()
    tool_use_response.raise_for_status = MagicMock()
    tool_use_response.json.return_value = {
        "usage": {"input_tokens": 20, "output_tokens": 5},
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "id": "t1", "name": tool_name, "input": {"query": "first"}},
            {"type": "tool_use", "id": "t2", "name": tool_name, "input": {"query": "second"}},
        ],
    }
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=tool_use_response)
    mock_context = MagicMock()
    mock_context.__aenter__ = AsyncMock(return_value=mock_client)
    mock_context.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.core.user_sessions.httpx.AsyncClient", return_value=mock_context),
        patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch(
            "app.core.user_sessions._session_was_closed_externally",
            new_callable=AsyncMock,
            side_effect=[False, False, True],
        ),
        patch(
            "app.core.mcp_gateway.execute_mcp_tool",
            new_callable=AsyncMock,
            return_value='{"result": "first result"}',
        ) as mock_exec,
    ):
        result = await _call_with_ws_loop(
            {"provider": "anthropic", "model": "m", "api_key": "k"},
            "SYS",
            [{"role": "user", "content": "search twice"}],
            "user_agent_p2",
            room_id, session_id, agent_id, db,
            ant_ws_tools=[],
            oai_ws_tools=[],
            mcp_ant_tools=[{"name": tool_name, "description": "search", "input_schema": {"type": "object"}}],
            mcp_oai_tools=[],
            mcp_bindings={tool_name: binding},
            mcp_user_id=user_id,
        )

    assert result.externally_stopped is True
    assert mock_client.post.await_count == 1
    mock_exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_user_turn_never_publishes_a_result_after_external_stop():
    session = _make_session("user_agent_p2")
    db = _make_db(uuid.uuid4())
    stopped = _LLMTurnResult(
        text=None,
        tool_input=None,
        tokens_in=0,
        tokens_out=0,
        model="m",
        externally_stopped=True,
    )

    with (
        patch("app.core.user_sessions._session_was_closed_externally", new_callable=AsyncMock, return_value=False),
        patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=False),
        patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
        patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
        patch(
            "app.core.mcp_gateway.room_mcp_tools",
            new_callable=AsyncMock,
            return_value=(
                [{"name": "mcp__fixture__wait", "description": "fixture", "input_schema": {}}],
                [],
                {"mcp__fixture__wait": MagicMock()},
            ),
        ),
        patch("app.core.user_sessions._call_with_ws_loop", new_callable=AsyncMock, return_value=stopped),
        patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock) as broadcast,
    ):
        still_open = await process_user_turn(session, "stop now", db)

    assert still_open is False
    # Only the "turn accepted" status may go out; no message or result.
    sent = [call.args[0] for call in broadcast.await_args_list]
    assert all(event.get("type") == "agent_status" and event.get("status") == "turn_accepted" for event in sent), sent
