"""
Tests for in-loop token-budget enforcement and tool-note persistence.

Covers the fixes for ChatMaster on large projects:
  1. _call_with_ws_loop stops (budget_exhausted) when session tokens hit TOKEN_LIMIT
     instead of looping past it (459k-token sessions).
  2. process_user_turn persists the tool exchange into history so the next turn
     does not re-read the same files.
  3. Token-limit close saves memory via a final close_session call instead of
     silently discarding the whole conversation.
  4. proxy_session._call_llm_with_tools honours budget_used mid-loop.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.token_budget import TOKEN_LIMIT
from app.core.user_sessions import (
    _LLMTurnResult,
    _call_with_ws_loop,
    _handle_token_limit_close,
    process_user_turn,
)


def _make_session(kind: str = "user_cm_p2", tokens_in: int = 0, tokens_out: int = 0) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.side = None
    s.tokens_in = tokens_in
    s.tokens_out = tokens_out
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.context_snapshot = None
    s.agent_id = None
    s.chat_master_agent_id = uuid.uuid4()
    s.goal_id = None
    s.task_id = None
    s.summary = None
    s.pending_actions = None
    s.started_at = datetime.now(timezone.utc)
    s.last_context_window_tokens = 0
    return s


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    db.rollback = AsyncMock()
    return db


class TestWsLoopBudgetGuard:
    @pytest.mark.asyncio
    async def test_exhausted_budget_stops_before_any_llm_call(self):
        """budget_used >= TOKEN_LIMIT → budget_exhausted=True, no HTTP request."""
        db = _make_db()
        with (
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=True),
            patch("app.core.user_sessions.httpx.AsyncClient") as mock_client,
        ):
            result = await _call_with_ws_loop(
                {"provider": "anthropic", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                "user_cm_p2",
                uuid.uuid4(),
                uuid.uuid4(),
                None,
                db,
                ant_ws_tools=[],
                oai_ws_tools=[],
                budget_used=TOKEN_LIMIT,
            )
        assert result.budget_exhausted is True
        assert result.tool_input is None
        assert result.text is None
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_proxy_tool_loop_raises_on_exhausted_budget(self):
        """proxy _call_llm_with_tools with budget_used >= TOKEN_LIMIT and
        on_exhaust='raise' pauses via TokenBudgetLimitReached without an HTTP call."""
        from app.core.proxy_session import TokenBudgetLimitReached, _call_llm_with_tools

        db = _make_db()
        with (
            patch("app.core.proxy_session.uses_cumulative_limit", return_value=True),
            patch("app.core.proxy_session.uses_window_limit", return_value=False),
            patch("app.core.context_assembler.uses_cumulative_limit", return_value=True),
            patch("app.core.proxy_session.httpx.AsyncClient") as mock_client,
        ):
            with pytest.raises(TokenBudgetLimitReached) as caught:
                await _call_llm_with_tools(
                    {"provider": "anthropic", "model": "m", "api_key": "k"},
                    "SYS",
                    [{"role": "user", "content": "hi"}],
                    room_id=uuid.uuid4(),
                    session_id=uuid.uuid4(),
                    task_id=None,
                    db=db,
                    workspace_ant_tools=[{"name": "read_file"}],
                    on_exhaust="raise",
                    budget_used=TOKEN_LIMIT,
                )
        assert caught.value.reason == "token_budget"
        assert caught.value.tokens_used == TOKEN_LIMIT
        assert caught.value.tool_calls_used == 0
        mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_proxy_reports_tool_call_limit_separately(self):
        from app.core.proxy_session import ToolLimitReached, _call_llm_with_tools

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "usage": {"input_tokens": 20, "output_tokens": 5},
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {"path": "a.py"}},
            ],
        }
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=response)
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        with patch("app.core.proxy_session.httpx.AsyncClient", return_value=mock_context):
            with pytest.raises(ToolLimitReached) as caught:
                await _call_llm_with_tools(
                    {"provider": "anthropic", "model": "m", "api_key": "k"},
                    "SYS",
                    [{"role": "user", "content": "hi"}],
                    room_id=uuid.uuid4(),
                    session_id=uuid.uuid4(),
                    task_id=None,
                    db=_make_db(),
                    workspace_ant_tools=[{"name": "read_file"}],
                    max_tool_calls=0,
                    on_exhaust="raise",
                )

        assert caught.value.reason == "tool_calls"
        assert caught.value.tool_calls_used == 0
        assert caught.value.tool_calls_limit == 0
        assert caught.value.tokens_used == 25

    @pytest.mark.asyncio
    async def test_repeated_identical_tool_error_carries_tool_and_safe_text(self):
        """internal design notes, part 4, stage 1: the loop mechanics that
        populate trigger_tool/repeat_count/safe_error on
        RepeatedToolErrorLimitReached — as opposed to just the exception's
        constructor accepting them (covered separately in test_proxy_session)."""
        from app.core.proxy_session import RepeatedToolErrorLimitReached, _call_llm_with_tools

        # Same tool_use block every turn — the model keeps retrying the same
        # call with the same (bad) argument.
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "usage": {"input_tokens": 20, "output_tokens": 5},
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use", "id": "tool-1", "name": "workspace_service",
                    "input": {"cwd": "/nonexistent"},
                },
            ],
        }
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=response)
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_client)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        fake_tool = MagicMock()
        # A fake secret in the raw error text — must not survive into safe_error.
        fake_tool.execute = AsyncMock(
            return_value="Error: invalid cwd '/nonexistent'; token: abcDEF123456xyz not found"
        )
        fake_registry = MagicMock()
        fake_registry.is_registered.return_value = True
        fake_registry.get.return_value = fake_tool

        with (
            patch("app.core.proxy_session.httpx.AsyncClient", return_value=mock_context),
            patch("app.core.tools.REGISTRY", fake_registry),
            patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
        ):
            with pytest.raises(RepeatedToolErrorLimitReached) as caught:
                await _call_llm_with_tools(
                    {"provider": "anthropic", "model": "m", "api_key": "k"},
                    "SYS",
                    [{"role": "user", "content": "hi"}],
                    room_id=uuid.uuid4(),
                    session_id=uuid.uuid4(),
                    task_id=None,
                    db=_make_db(),
                    workspace_ant_tools=[{"name": "workspace_service"}],
                    on_exhaust="raise",
                )

        assert caught.value.reason == "repeated_tool_error"
        assert caught.value.trigger_tool == "workspace_service"
        # max_identical_tool_errors defaults to 2: warning at repeat 2, fatal at repeat 3.
        assert caught.value.repeat_count == 3
        assert caught.value.safe_error is not None
        assert "invalid cwd" in caught.value.safe_error
        assert "abcDEF123456xyz" not in caught.value.safe_error
        assert "token=[REDACTED]" in caught.value.safe_error


class TestToolNotePersistence:
    @pytest.mark.asyncio
    async def test_tool_notes_saved_into_history(self):
        """Tool exchange from the ws loop must survive into session history."""
        session = _make_session("user_cm_p2")
        db = _make_db()

        notes = [
            {"role": "assistant", "content": "[tool call] read_file({\"path\": \"A.md\"})"},
            {"role": "user", "content": "[tool results]\n\n### read_file\ncontent of A"},
        ]
        llm_result = _LLMTurnResult(
            text="Studied the documentation.",
            tool_input=None,
            tokens_in=100,
            tokens_out=50,
            model="claude-haiku-4-5-20251001",
            tool_notes=notes,
        )

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._call_with_ws_loop", new_callable=AsyncMock, return_value=llm_result) as mock_loop,
            patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock),
        ):
            still_open = await process_user_turn(session, "study the docs", db)

        assert still_open is True
        saved = session.context_snapshot["messages"]
        assert notes[0] in saved
        assert notes[1] in saved
        assert saved[-1] == {"role": "assistant", "content": "Studied the documentation."}
        # budget_used must be forwarded so the loop can enforce the limit
        assert mock_loop.call_args.kwargs["budget_used"] == 0


class TestTokenLimitGracefulClose:
    @pytest.mark.asyncio
    async def test_budget_exhausted_mid_loop_triggers_close(self):
        """budget_exhausted result → session closes via _handle_token_limit_close."""
        session = _make_session("user_cm_p2", tokens_in=90_000, tokens_out=9_000)
        db = _make_db()

        llm_result = _LLMTurnResult(
            text=None, tool_input=None, tokens_in=5_000, tokens_out=500,
            model="claude-haiku-4-5-20251001", budget_exhausted=True,
            tool_notes=[{"role": "assistant", "content": "[tool call] x"}],
        )

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._call_with_ws_loop", new_callable=AsyncMock, return_value=llm_result),
            patch("app.core.user_sessions._handle_token_limit_close", new_callable=AsyncMock) as mock_close,
        ):
            still_open = await process_user_turn(session, "continue", db)

        assert still_open is False
        mock_close.assert_awaited_once()
        # loop usage must be tracked before closing
        assert session.tokens_in == 95_000
        assert session.tokens_out == 9_500

    @pytest.mark.asyncio
    async def test_close_saves_memory_from_final_llm_call(self):
        """_handle_token_limit_close asks the LLM for close_session and applies it."""
        session = _make_session("user_cm_p2")
        db = _make_db()
        history: list[dict[str, Any]] = [
            {"role": "user", "content": "we are doing the project"},
            {"role": "assistant", "content": "clarifying: SQLite or Postgres?"},
            {"role": "user", "content": "SQLite, the priority is the public part"},
        ]

        final = _LLMTurnResult(
            text=None,
            tool_input={
                "agent_memory_patch": {"content": "Decisions: SQLite; priority — the public part"},
                "summary": "Requirements gathered, the plan is not finished.",
            },
            tokens_in=1_000,
            tokens_out=200,
            model="claude-haiku-4-5-20251001",
            tool_name="close_session",
        )

        with (
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=final) as mock_llm,
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock) as mock_bc,
        ):
            await _handle_token_limit_close(session, {}, "SYS", history, db)

        mock_llm.assert_awaited_once()
        # user decisions must reach apply_close_actions via agent_memory_patch
        actions = mock_apply.call_args[0][1]
        assert actions.agent_memory_patch is not None
        assert "SQLite" in actions.agent_memory_patch.content
        assert actions.summary == "Requirements gathered, the plan is not finished."
        # user gets a visible notice with the summary
        mock_bc.assert_awaited_once()
        notice = mock_bc.call_args[0][0]["content"]
        assert "token limit" in notice
        assert "Requirements gathered" in notice
        # history ends with the close marker and is persisted
        assert history[-1] == {"role": "assistant", "content": "[session closed: token limit]"}

    @pytest.mark.asyncio
    async def test_close_without_assistant_turns_skips_llm(self):
        """Nothing to save when the agent never replied — no extra LLM call."""
        session = _make_session("user_cm_p2")
        db = _make_db()
        history = [{"role": "user", "content": "hi"}]

        with (
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock) as mock_llm,
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager.broadcast", new_callable=AsyncMock),
        ):
            await _handle_token_limit_close(session, {}, "SYS", history, db)

        mock_llm.assert_not_called()
