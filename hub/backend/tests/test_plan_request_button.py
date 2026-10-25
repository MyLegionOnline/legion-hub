"""The composer's "Form a plan" button (router-free plan request).

The button exists because a ChatMaster dialogue about a spec has no reliable
ending: every further message is one more thing for the model to read as
"keep clarifying". These tests pin the forced propose_plan turn behind it.

All LLM calls are stubbed; no DB or network is required.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.user_sessions import (
    FORCE_PLAN_USER_MESSAGE,
    OFFER_PLAN_TOOL_NAME,
    _LLMTurnResult,
    _openai_tool_choice,
    process_user_turn,
)


@pytest.fixture(autouse=True)
def _no_workspace():
    """Default to a room without a workspace — one test opts back in."""
    with patch(
        "app.core.workspace_tools.workspace_tools_available",
        new_callable=AsyncMock,
        return_value=False,
    ):
        yield


def _make_session(kind: str = "user_cm_p2") -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.tokens_in = 0
    s.tokens_out = 0
    s.reasoning_tokens = 0
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.agent_id = None
    s.chat_master_agent_id = uuid.uuid4()
    s.goal_id = None
    s.task_id = None
    s.summary = None
    s.pending_actions = None
    s.started_at = datetime.now(timezone.utc)
    s.last_context_window_tokens = 0
    # A dialogue that has already been talking: the button is only reachable
    # from an open conversation.
    s.context_snapshot = {
        "messages": [
            {"role": "user", "content": "I want to re-layout the hero from the reference"},
            {"role": "assistant", "content": "Which background and which logos?"},
        ]
    }
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


def _plan_tool_result() -> _LLMTurnResult:
    return _LLMTurnResult(
        text=None,
        tool_input={
            "goal_title": "Hero in the style of the reference",
            "goal_description": "Header with logos only, hero on bg_1.png, LEGION #9DC8E4.",
            "plan_summary": "Re-layout the header and hero.",
        },
        tokens_in=900,
        tokens_out=120,
        model="claude-haiku-4-5-20251001",
        tool_name="propose_plan",
    )


async def _run(
    session: MagicMock,
    db: AsyncMock,
    call_llm: AsyncMock,
    *,
    ws_loop: AsyncMock | None = None,
    message: str = FORCE_PLAN_USER_MESSAGE,
    **kwargs: Any,
) -> bool:
    with (
        patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
        patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
        patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
        patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
        patch("app.core.user_sessions._call_llm_with_tools", call_llm),
        patch("app.core.user_sessions._call_with_ws_loop", ws_loop or AsyncMock()),
        patch("app.core.user_sessions.track_llm_usage"),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch("app.core.user_sessions._maybe_warn_growth", new_callable=AsyncMock),
        patch(
            "app.core.user_sessions._maybe_rotate_user_session",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "app.core.user_sessions._session_was_closed_externally",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        return await process_user_turn(session, message, db, **kwargs)


class TestForcedPlanTurn:
    @pytest.mark.asyncio
    async def test_button_forces_propose_plan_and_stores_the_plan(self):
        session = _make_session()
        db = _make_db()
        call_llm = AsyncMock(return_value=_plan_tool_result())

        still_open = await _run(session, db, call_llm, force_plan=True)

        assert still_open is True
        assert call_llm.await_args.kwargs["force_tool"] == "propose_plan"
        pending = session.context_snapshot["pending_plan"]
        assert pending["goal_title"] == "Hero in the style of the reference"
        # The click itself is part of the transcript, so an Edit round can see
        # where requirement gathering was cut off.
        assert session.context_snapshot["messages"][-2]["content"] == FORCE_PLAN_USER_MESSAGE

    @pytest.mark.asyncio
    async def test_forced_turn_never_reads_files_first(self):
        """A room with a workspace still answers with the plan, not a tool loop."""
        session = _make_session()
        db = _make_db()
        call_llm = AsyncMock(return_value=_plan_tool_result())
        ws_loop = AsyncMock(return_value=_plan_tool_result())

        with (
            patch(
                "app.core.workspace_tools.workspace_tools_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "app.core.room_git_flow.ensure_session_branch",
                new_callable=AsyncMock,
                return_value="dev",
            ),
        ):
            still_open = await _run(session, db, call_llm, ws_loop=ws_loop, force_plan=True)

        assert still_open is True
        ws_loop.assert_not_awaited()
        assert call_llm.await_args.kwargs["force_tool"] == "propose_plan"

    @pytest.mark.asyncio
    async def test_ordinary_turn_is_left_alone(self):
        session = _make_session()
        db = _make_db()
        call_llm = AsyncMock(
            return_value=_LLMTurnResult(
                text="Please clarify the background",
                tool_input=None,
                tokens_in=10,
                tokens_out=5,
                model="claude-haiku-4-5-20251001",
            )
        )

        with patch("app.core.user_sessions.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await _run(session, db, call_llm)

        assert call_llm.await_args.kwargs["force_tool"] is None

    @pytest.mark.asyncio
    async def test_force_plan_is_ignored_where_the_tool_does_not_exist(self):
        """user_cm_p1 offers propose_task_change, never propose_plan."""
        session = _make_session("user_cm_p1")
        db = _make_db()
        call_llm = AsyncMock(
            return_value=_LLMTurnResult(
                text="Task updated",
                tool_input=None,
                tokens_in=10,
                tokens_out=5,
                model="claude-haiku-4-5-20251001",
            )
        )

        with patch("app.core.user_sessions.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await _run(session, db, call_llm, force_plan=True)

        assert call_llm.await_args.kwargs["force_tool"] is None

    @pytest.mark.asyncio
    async def test_close_session_answer_is_salvaged_as_a_text_plan(self):
        """An endpoint that dropped tool_choice must not cost the dialogue."""
        session = _make_session()
        db = _make_db()
        xml = (
            "<propose_plan><goal_title>Hero</goal_title>"
            "<goal_description>Header and hero per the reference.</goal_description>"
            "<plan_summary>Layout</plan_summary></propose_plan>"
        )
        call_llm = AsyncMock(
            side_effect=[
                _LLMTurnResult(
                    text=None,
                    tool_input={"summary": "done"},
                    tokens_in=10,
                    tokens_out=5,
                    model="m",
                    tool_name="close_session",
                ),
                _LLMTurnResult(text=xml, tool_input=None, tokens_in=10, tokens_out=5, model="m"),
            ]
        )

        with patch("app.core.user_sessions._run_close_flow", new_callable=AsyncMock) as close_flow:
            still_open = await _run(session, db, call_llm, force_plan=True)

        assert still_open is True
        close_flow.assert_not_awaited()
        assert session.context_snapshot["pending_plan"]["goal_title"] == "Hero"
        assert call_llm.await_args.kwargs["force_text"] is True

    @pytest.mark.asyncio
    async def test_proposing_the_plan_takes_the_button_down(self):
        session = _make_session()
        session.context_snapshot["plan_offered"] = True
        db = _make_db()
        call_llm = AsyncMock(return_value=_plan_tool_result())

        await _run(session, db, call_llm, force_plan=True)

        assert "pending_plan" in session.context_snapshot
        assert "plan_offered" not in session.context_snapshot


class TestOfferPlanTool:
    """ChatMaster raises the "Form a plan" button itself (offer_plan)."""

    @staticmethod
    def _offer(message: str | None) -> _LLMTurnResult:
        return _LLMTurnResult(
            text=None,
            tool_input={"message": message} if message is not None else {},
            tokens_in=10,
            tokens_out=5,
            model="m",
            tool_name=OFFER_PLAN_TOOL_NAME,
        )

    @pytest.mark.asyncio
    async def test_offer_is_a_reply_that_raises_the_button(self):
        session = _make_session()
        db = _make_db()
        call_llm = AsyncMock(return_value=self._offer("Got it: header + hero on bg_1."))

        with patch("app.core.user_sessions.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            still_open = await _run(session, db, call_llm, message="Logos only, bg_1")

        assert still_open is True
        assert session.context_snapshot["plan_offered"] is True
        assert session.context_snapshot["messages"][-1] == {
            "role": "assistant",
            "content": "Got it: header + hero on bg_1.",
        }
        frame = mock_manager.broadcast.await_args.args[0]
        assert frame["content"] == "Got it: header + hero on bg_1."
        assert frame["plan_offered"] is True

    @pytest.mark.asyncio
    async def test_first_turn_offer_is_not_mistaken_for_a_close(self):
        session = _make_session()
        session.context_snapshot = {"messages": []}
        db = _make_db()
        call_llm = AsyncMock(return_value=self._offer(""))

        with patch("app.core.user_sessions.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await _run(session, db, call_llm, message="Rename the button to Save")

        # One call: the guard that retries a premature close never fired.
        assert call_llm.await_count == 1
        assert session.context_snapshot["plan_offered"] is True
        assert mock_manager.broadcast.await_args.args[0]["content"]

    def test_tool_is_offered_only_outside_a_forced_plan(self):
        from app.core.user_sessions import OFFER_PLAN_TOOL_OPENAI

        assert OFFER_PLAN_TOOL_OPENAI["function"]["parameters"]["required"] == ["message"]


class TestPlanRequestState:
    """What the composer button is allowed to do, and why not when it is not."""

    @staticmethod
    def _session(kind: str = "user_cm_p2", snapshot: dict[str, Any] | None = None) -> MagicMock:
        s = MagicMock()
        s.id = uuid.uuid4()
        s.kind = kind
        s.context_snapshot = snapshot
        return s

    @staticmethod
    async def _state(
        session: MagicMock | None,
        *,
        busy: bool = False,
        quick_tasks: list[Any] | None = None,
    ) -> dict[str, Any]:
        from app.api.v1 import chat as chat_api

        repo = MagicMock()
        repo.find_open_user_session = AsyncMock(return_value=session)
        with (
            patch("app.repositories.session.SessionRepository", return_value=repo),
            patch.object(chat_api, "turn_in_progress", return_value=busy),
            # The quick-task check runs a real query; _make_db gives it a result
            # object instead of a bare AsyncMock that returns coroutines.
            patch(
                "app.core.dialogue_scope.quick_tasks_for_session",
                new_callable=AsyncMock,
                return_value=quick_tasks or [],
            ),
        ):
            return await chat_api._plan_request_state(uuid.uuid4(), MagicMock(), _make_db())

    @pytest.mark.asyncio
    async def test_an_offered_plan_can_be_confirmed(self):
        state = await self._state(self._session(snapshot={"plan_offered": True}))
        assert state["available"] is True

    @pytest.mark.asyncio
    async def test_no_button_until_chatmaster_offers_it(self):
        state = await self._state(self._session(), busy=True)
        assert (state["available"], state["reason"]) == (False, "not_offered")

    @pytest.mark.asyncio
    async def test_without_a_dialogue_there_is_nothing_to_plan(self):
        state = await self._state(None)
        assert (state["available"], state["reason"]) == (False, "no_session")

    @pytest.mark.asyncio
    async def test_another_kind_of_dialogue_is_refused(self):
        state = await self._state(self._session("user_agent_p2"))
        assert (state["available"], state["reason"]) == (False, "wrong_kind")

    @pytest.mark.asyncio
    async def test_a_proposed_plan_waits_for_its_own_buttons(self):
        state = await self._state(self._session(snapshot={"pending_plan": {"goal_title": "x"}}))
        assert (state["available"], state["reason"]) == (False, "plan_pending")

    @pytest.mark.asyncio
    async def test_a_running_turn_is_never_raced(self):
        state = await self._state(self._session(snapshot={"plan_offered": True}), busy=True)
        assert (state["available"], state["reason"]) == (False, "busy")


class TestToolChoiceWire:
    def test_named_tool_is_requested_by_name(self):
        assert _openai_tool_choice(False, "propose_plan") == {
            "type": "function",
            "function": {"name": "propose_plan"},
        }

    def test_text_wins_over_a_named_tool(self):
        assert _openai_tool_choice(True, "propose_plan") == "none"

    def test_unconstrained_by_default(self):
        assert _openai_tool_choice(False, None) == "auto"
