"""
Integration tests for user_sessions module (Step 7 / §3).

Four scenarios, each from start_user_session → process_user_turn → close:
  1. user_agent_p1 — task-focused agent session, close via tool call
  2. user_agent_p2 — open agent conversation, multiple turns then close
  3. user_cm_p1    — ChatMaster with existing goal, close via tool call
  4. user_cm_p2    — ChatMaster new topic, close via tool call

All LLM calls are stubbed; no real DB or network is required.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.user_sessions import (
    _LLMTurnResult,
    _compute_cost,
    _get_history,
    _save_history,
    process_user_turn,
    start_user_session,
)
from app.core.router import RouteDecision


@pytest.fixture(autouse=True)
def _no_workspace(monkeypatch):
    """Disable workspace loop in all user_sessions tests — avoids real HTTP calls."""
    with patch("app.core.workspace_tools.workspace_tools_available", new_callable=AsyncMock, return_value=False):
        yield


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_room(room_id: uuid.UUID | None = None) -> MagicMock:
    room = MagicMock()
    room.id = room_id or uuid.uuid4()
    return room


def _make_user(user_id: uuid.UUID | None = None) -> MagicMock:
    user = MagicMock()
    user.id = user_id or uuid.uuid4()
    return user


def _make_session(
    kind: str = "user_agent_p1",
    tokens_in: int = 0,
    tokens_out: int = 0,
    closed: bool = False,
    agent_id: uuid.UUID | None = None,
    chat_master_agent_id: uuid.UUID | None = None,
) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.tokens_in = tokens_in
    s.tokens_out = tokens_out
    s.cost_usd = Decimal("0")
    s.closed_at = MagicMock() if closed else None
    s.close_reason = None
    s.is_open = not closed
    s.context_snapshot = None
    s.agent_id = agent_id or (uuid.uuid4() if "agent" in kind else None)
    s.chat_master_agent_id = chat_master_agent_id or (uuid.uuid4() if "cm" in kind else None)
    s.goal_id = uuid.uuid4()
    s.task_id = uuid.uuid4() if kind == "user_agent_p1" else None
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
    # db.execute is awaited; the resulting Result object is synchronous.
    # Default every lookup to "no rows" so post-close pipelines skip gracefully.
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    db.rollback = AsyncMock()
    return db


def _route(
    route: str,
    agent_ids: list[uuid.UUID] | None = None,
    task_id: uuid.UUID | None = None,
    goal_id: uuid.UUID | None = None,
) -> RouteDecision:
    return RouteDecision(
        route=route,
        mentioned_agent_ids=agent_ids or [uuid.uuid4()],
        matched_task_id=task_id,
        matched_goal_id=goal_id,
        confidence=0.9,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# Scenario 1: user_agent_p1 — start + close via tool call
# ---------------------------------------------------------------------------


class TestUserAgentP1:
    @pytest.mark.asyncio
    async def test_start_creates_user_agent_p1_session(self):
        room = _make_room()
        user = _make_user()
        agent_id = uuid.uuid4()
        task_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        rd = _route("single_agent_p1", [agent_id], task_id=task_id, goal_id=goal_id)

        fake_session = _make_session("user_agent_p1", agent_id=agent_id)
        fake_session.task_id = task_id
        fake_session.goal_id = goal_id

        db = _make_db()

        with (
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=fake_session)
            MockRepo.return_value = repo_instance

            session = await start_user_session(room, user, rd, db)

        assert session.kind == "user_agent_p1"
        create_call = repo_instance.create.call_args[0][0]
        assert create_call.kind == "user_agent_p1"
        assert create_call.user_id == user.id
        assert create_call.agent_id == agent_id
        assert create_call.task_id == task_id
        assert create_call.goal_id == goal_id

    @pytest.mark.asyncio
    async def test_p1_missing_goal_id_backfilled_from_task(self):
        """Router names the task but leaves matched_goal_id null — the goal is
        loaded from the task row so the p1 session is still created."""
        room = _make_room()
        user = _make_user()
        agent_id = uuid.uuid4()
        task_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        rd = _route("single_agent_p1", [agent_id], task_id=task_id, goal_id=None)

        db = _make_db()
        task_row = MagicMock()
        task_row.goal_id = goal_id
        db.execute.return_value.scalars.return_value.first.return_value = task_row

        with (
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(
                return_value=_make_session("user_agent_p1", agent_id=agent_id)
            )
            MockRepo.return_value = repo_instance

            await start_user_session(room, user, rd, db)

        create_call = repo_instance.create.call_args[0][0]
        assert create_call.kind == "user_agent_p1"
        assert create_call.task_id == task_id
        assert create_call.goal_id == goal_id

    @pytest.mark.asyncio
    async def test_p1_without_resolvable_goal_falls_back_to_p2(self):
        """No goal on the decision and no task row to backfill from → p2,
        not a ValidationError that the caller would only swallow."""
        room = _make_room()
        user = _make_user()
        agent_id = uuid.uuid4()
        rd = _route(
            "single_agent_p1", [agent_id], task_id=uuid.uuid4(), goal_id=None
        )

        db = _make_db()  # every lookup returns no rows

        with (
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(
                return_value=_make_session("user_agent_p2", agent_id=agent_id)
            )
            MockRepo.return_value = repo_instance

            await start_user_session(room, user, rd, db)

        create_call = repo_instance.create.call_args[0][0]
        assert create_call.kind == "user_agent_p2"
        # p2 does not put the task in its prompt, but the row keeps the link:
        # that is what lets Cancel/Block on the task close this dialogue instead
        # of leaving it working on a task the user already stopped.
        assert create_call.task_id == rd.matched_task_id
        assert create_call.goal_id is None

    @pytest.mark.asyncio
    async def test_process_turn_closes_on_tool_call(self):
        """LLM calls close_session → session marked closed, return False."""
        session = _make_session("user_agent_p1")
        db = _make_db()

        tool_input = {"agent_memory_patch": {"content": "done"}, "summary": "task finished"}
        llm_result = _LLMTurnResult(
            text=None, tool_input=tool_input, tokens_in=100, tokens_out=50, model="claude-haiku-4-5-20251001"
        )

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=llm_result),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "done!", db)

        assert open_ is False
        assert session.close_reason == "task_done"
        assert session.closed_at is not None

    @pytest.mark.asyncio
    async def test_process_turn_stays_open_on_text_reply(self):
        """LLM returns text → session stays open, message published, return True."""
        session = _make_session("user_agent_p1")
        db = _make_db()

        llm_result = _LLMTurnResult(
            text="Here is my answer.", tool_input=None, tokens_in=80, tokens_out=30,
            model="claude-haiku-4-5-20251001"
        )

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=llm_result),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "how do I do X?", db)

        assert open_ is True
        assert session.close_reason is None
        # Conversation history saved
        history = _get_history(session)
        assert history[0] == {"role": "user", "content": "how do I do X?"}
        assert history[1] == {"role": "assistant", "content": "Here is my answer."}


# ---------------------------------------------------------------------------
# Scenario 2: user_agent_p2 — multi-turn then close
# ---------------------------------------------------------------------------


class TestUserAgentP2:
    @pytest.mark.asyncio
    async def test_start_creates_user_agent_p2_session(self):
        room = _make_room()
        user = _make_user()
        agent_id = uuid.uuid4()
        rd = _route("single_agent_p2", [agent_id])

        fake_session = _make_session("user_agent_p2", agent_id=agent_id)
        db = _make_db()

        with (
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=fake_session)
            MockRepo.return_value = repo_instance

            session = await start_user_session(room, user, rd, db)

        assert session.kind == "user_agent_p2"
        create_call = repo_instance.create.call_args[0][0]
        assert create_call.kind == "user_agent_p2"
        assert create_call.task_id is None  # p2 has no task

    @pytest.mark.asyncio
    async def test_multi_turn_accumulates_history(self):
        """Three turns of text replies accumulate in context_snapshot."""
        session = _make_session("user_agent_p2")
        db = _make_db()

        def _make_text_result(text: str) -> _LLMTurnResult:
            return _LLMTurnResult(text=text, tool_input=None, tokens_in=10, tokens_out=10,
                                  model="claude-haiku-4-5-20251001")

        replies = ["First reply.", "Second reply.", "Third reply."]
        call_count = 0

        async def _fake_call(*args: Any, **kwargs: Any) -> _LLMTurnResult:
            nonlocal call_count
            r = _make_text_result(replies[call_count])
            call_count += 1
            return r

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", side_effect=_fake_call),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            for msg in ["turn 1", "turn 2", "turn 3"]:
                open_ = await process_user_turn(session, msg, db)
                assert open_ is True

        history = _get_history(session)
        assert len(history) == 6  # 3 user + 3 assistant
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "First reply."

    @pytest.mark.asyncio
    async def test_closes_on_token_limit(self):
        """Token budget exhausted → process_user_turn returns False immediately."""
        session = _make_session("user_agent_p2")
        db = _make_db()

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=True),
            # Budget check returns False (limit hit)
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=False),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock) as mock_llm,
        ):
            open_ = await process_user_turn(session, "hello", db)

        assert open_ is False
        mock_llm.assert_not_called()  # LLM must not be called after budget exceeded

    @pytest.mark.asyncio
    async def test_window_mode_allows_compact_turn_after_cumulative_usage(self):
        """Cumulative accounting above 100k is not a user-turn admission guard."""
        session = _make_session("user_agent_p2", tokens_in=110_000, tokens_out=5_000)
        db = _make_db()
        assembly = MagicMock(
            messages=[{"role": "user", "content": "next compact prompt"}],
            rejected_reason=None,
        )
        result = _LLMTurnResult("Still open.", None, 20, 10, "test-model")

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.uses_cumulative_limit", return_value=False),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock) as budget_guard,
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._preflight_direct_user_request", new_callable=AsyncMock, return_value=assembly),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=result) as mock_llm,
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "next compact prompt", db)

        assert open_ is True
        budget_guard.assert_not_awaited()
        mock_llm.assert_awaited_once()


# ---------------------------------------------------------------------------
# Scenario 3: user_cm_p1 — ChatMaster with existing goal
# ---------------------------------------------------------------------------


class TestUserCmP1:
    @pytest.mark.asyncio
    async def test_start_creates_user_cm_p1_session(self):
        room = _make_room()
        user = _make_user()
        cm_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        rd = _route("chatmaster_p1", [], goal_id=goal_id)

        fake_session = _make_session("user_cm_p1", chat_master_agent_id=cm_id)
        db = _make_db()

        with (
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=fake_session)
            MockRepo.return_value = repo_instance

            session = await start_user_session(room, user, rd, db, cm_agent_id=cm_id)

        assert session.kind == "user_cm_p1"
        create_call = repo_instance.create.call_args[0][0]
        assert create_call.kind == "user_cm_p1"
        assert create_call.chat_master_agent_id == cm_id
        assert create_call.goal_id == goal_id

    @pytest.mark.asyncio
    async def test_requires_cm_agent_id(self):
        room = _make_room()
        user = _make_user()
        rd = _route("chatmaster_p1", [])
        db = _make_db()

        with pytest.raises(ValueError, match="cm_agent_id is required"):
            await start_user_session(room, user, rd, db)

    @pytest.mark.asyncio
    async def test_close_applies_plan_update_and_tasks(self):
        """close_session with plan_update + tasks_create must be accepted for user_cm_p1."""
        session = _make_session("user_cm_p1")
        db = _make_db()

        tool_input = {
            "plan_update": {"plan_id": str(uuid.uuid4()), "steps": []},
            "tasks_create": [{"title": "New task"}],
            "summary": "plan updated",
        }
        llm_result = _LLMTurnResult(
            text=None, tool_input=tool_input, tokens_in=200, tokens_out=100,
            model="claude-haiku-4-5-20251001"
        )

        applied_actions: list[Any] = []

        async def _fake_apply(s: Any, actions: Any, db: Any) -> None:
            applied_actions.append(actions)

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=llm_result),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", side_effect=_fake_apply),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "update the plan", db)

        assert open_ is False
        assert len(applied_actions) == 1
        actions = applied_actions[0]
        assert actions.plan_update is not None
        assert len(actions.tasks_create) == 1
        assert actions.summary == "plan updated"


# ---------------------------------------------------------------------------
# Scenario 4: user_cm_p2 — ChatMaster new topic
# ---------------------------------------------------------------------------


class TestUserCmP2:
    @pytest.mark.asyncio
    async def test_start_creates_user_cm_p2_session(self):
        room = _make_room()
        user = _make_user()
        cm_id = uuid.uuid4()
        rd = _route("chatmaster_p2", [])

        fake_session = _make_session("user_cm_p2", chat_master_agent_id=cm_id)
        db = _make_db()

        with (
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=fake_session)
            MockRepo.return_value = repo_instance

            session = await start_user_session(room, user, rd, db, cm_agent_id=cm_id)

        assert session.kind == "user_cm_p2"
        create_call = repo_instance.create.call_args[0][0]
        assert create_call.kind == "user_cm_p2"
        assert create_call.goal_id is None  # p2 has no goal yet

    @pytest.mark.asyncio
    async def test_close_with_goal_creation(self):
        """close_session with goals_create is accepted for user_cm_p2."""
        session = _make_session("user_cm_p2")
        db = _make_db()

        tool_input = {
            "goals_create": [{"title": "New landing page", "description": "Next.js site"}],
            "summary": "goal created",
        }
        llm_result = _LLMTurnResult(
            text=None, tool_input=tool_input, tokens_in=150, tokens_out=80,
            model="claude-haiku-4-5-20251001"
        )

        applied_actions: list[Any] = []

        async def _fake_apply(s: Any, actions: Any, db: Any) -> None:
            applied_actions.append(actions)

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=llm_result),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", side_effect=_fake_apply),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "I want a landing page", db)

        assert open_ is False
        assert len(applied_actions) == 1
        assert len(applied_actions[0].goals_create) == 1
        assert applied_actions[0].goals_create[0].title == "New landing page"

    @pytest.mark.asyncio
    async def test_premature_close_mid_gathering_is_suppressed(self):
        """project_cm_p2_close_midgathering: the CM read a commit, formulated
        clarifying questions and called close_session(task_done). The session
        must stay open (questions live only here) and the summary is published
        as an ordinary reply."""
        session = _make_session("user_cm_p2")
        _save_history(session, [
            {"role": "user", "content": "write an article for commit 3a78030"},
            {"role": "assistant", "content": "Got it. Is the diff of this commit available to the agent?"},
        ])
        db = _make_db()

        close_result = _LLMTurnResult(
            text=None,
            tool_input={
                "summary": (
                    "Analyzed commit 3a78030. Asked clarifying questions: "
                    "the level of code detail and the way of publishing. "
                    "The plan is not formed yet."
                )
            },
            tokens_in=120, tokens_out=60, model="claude-haiku-4-5-20251001",
            tool_name="close_session",
        )
        empty_forced = _LLMTurnResult(
            text="", tool_input=None, tokens_in=10, tokens_out=1,
            model="claude-haiku-4-5-20251001",
        )
        _seq = iter([close_result, empty_forced])

        async def _fake_call(*a: Any, **k: Any) -> _LLMTurnResult:
            try:
                return next(_seq)
            except StopIteration:
                return empty_forced

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", side_effect=_fake_call),
            patch("app.core.user_sessions._finalize_session_work", new_callable=AsyncMock),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "go ahead, details as we go", db)

        assert open_ is True
        assert session.close_reason is None
        history = _get_history(session)
        assert history[-1]["role"] == "assistant"
        assert "clarifying questions" in history[-1]["content"]

    @pytest.mark.asyncio
    async def test_close_allowed_when_user_drops_the_topic(self):
        """A close right after the user says "no need, forget it" is legitimate."""
        session = _make_session("user_cm_p2")
        _save_history(session, [
            {"role": "user", "content": "write an article for commit 3a78030"},
            {"role": "assistant", "content": "What level of code detail is needed?"},
        ])
        db = _make_db()

        close_result = _LLMTurnResult(
            text=None,
            tool_input={"summary": "The user cancelled the task."},
            tokens_in=90, tokens_out=20, model="claude-haiku-4-5-20251001",
            tool_name="close_session",
        )

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=close_result),
            patch("app.core.user_sessions._finalize_session_work", new_callable=AsyncMock),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "no need, forget about it", db)

        assert open_ is False
        assert session.close_reason == "task_done"


# ---------------------------------------------------------------------------
# Edge cases shared across all kinds
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_already_closed_session_returns_false(self):
        session = _make_session("user_agent_p1", closed=True)
        db = _make_db()

        with patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock) as mock_llm:
            result = await process_user_turn(session, "anything", db)

        assert result is False
        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_close_actions_salvages_valid_fields(self):
        """One malformed field is pruned; the rest of the payload is applied."""
        session = _make_session("user_agent_p1")
        db = _make_db()

        llm_result = _LLMTurnResult(
            text=None,
            # tasks_cancel must contain UUIDs — a non-UUID string fails validation
            tool_input={"agent_memory_patch": {"content": "m"}, "tasks_cancel": ["not-a-uuid"]},
            tokens_in=50, tokens_out=20, model="claude-haiku-4-5-20251001"
        )

        applied: list[Any] = []

        async def _fake_apply(s: Any, actions: Any, db: Any) -> None:
            applied.append(actions)

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=llm_result),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", side_effect=_fake_apply),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        ):
            open_ = await process_user_turn(session, "done", db)

        # Session still closes (graceful degradation)
        assert open_ is False
        # The invalid tasks_cancel is dropped, but agent_memory_patch survives
        assert len(applied) == 1
        assert applied[0].agent_memory_patch is not None
        assert applied[0].agent_memory_patch.content == "m"
        assert applied[0].tasks_cancel == []

    @pytest.mark.asyncio
    async def test_session_started_event_emitted(self):
        """start_user_session must emit SESSION_STARTED."""
        room = _make_room()
        user = _make_user()
        agent_id = uuid.uuid4()
        rd = _route("single_agent_p2", [agent_id])

        fake_session = _make_session("user_agent_p2", agent_id=agent_id)
        db = _make_db()
        emitted: list[str] = []

        async def _fake_emit(**kwargs: Any) -> MagicMock:
            emitted.append(kwargs["event_type"])
            return MagicMock()

        with (
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.emit_event", side_effect=_fake_emit),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=fake_session)
            MockRepo.return_value = repo_instance
            await start_user_session(room, user, rd, db)

        assert "SESSION_STARTED" in emitted

    def test_compute_cost_known_model(self):
        cost = _compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0)
        assert cost == pytest.approx(0.8, rel=1e-6)

    def test_compute_cost_unknown_model_is_not_priced(self):
        # No guessed default: an unpriced model must not look like Claude Sonnet.
        assert _compute_cost("unknown-model-xyz", 1_000_000, 0) == 0

    def test_history_roundtrip(self):
        session = _make_session()
        history = [{"role": "user", "content": "hi"}]
        _save_history(session, history)
        loaded = _get_history(session)
        assert loaded == history

    def test_save_history_updates_message_and_turn_counts(self):
        """Stage 5.3: message_count/turn_count are recomputed on every save."""
        session = _make_session()
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "and then?"},
        ]
        _save_history(session, history)
        assert session.message_count == 3
        assert session.turn_count == 2  # two role='user' entries

        _save_history(session, history + [{"role": "assistant", "content": "..."}])
        assert session.message_count == 4
        assert session.turn_count == 2

    def test_save_history_does_not_alias_the_caller_list(self):
        """incident 2026-09-08#3: storing the caller's live list into the JSONB
        snapshot let later `history.append(...)` mutate the value SQLAlchemy
        holds as committed state, so the flush-time comparison saw "unchanged"
        and the turn's reply never reached the DB. The stored messages must be a
        detached copy."""
        session = _make_session()
        history = [{"role": "user", "content": "hi"}]
        _save_history(session, history)

        history.append({"role": "assistant", "content": "the reply"})
        history[0]["content"] = "mutated"

        stored = session.context_snapshot["messages"]
        assert stored == [{"role": "user", "content": "hi"}]
        assert stored is not history


# ---------------------------------------------------------------------------
# Stage 3: retry + SESSION_CLOSE_ACTIONS_DROPPED + summary publish
# ---------------------------------------------------------------------------


class TestCloseFlowRetry:
    """Dropped-field retry logic, SESSION_CLOSE_ACTIONS_DROPPED event, and summary broadcast."""

    @pytest.mark.asyncio
    async def test_retry_on_dropped_fields_succeeds(self):
        """LLM first returns file_writes (dropped for user_cm_p1), then on retry
        returns tasks_create → tasks_create applied, no DROPPED event emitted."""
        session = _make_session("user_cm_p1")
        db = _make_db()

        first_tool_input = {"file_writes": [{"path": "x.txt", "content": "hi"}]}
        second_tool_input = {
            "tasks_create": [{"title": "Do the work"}],
            "summary": "work delegated",
        }

        first_result = _LLMTurnResult(
            text=None, tool_input=first_tool_input,
            tokens_in=100, tokens_out=50, model="claude-haiku-4-5-20251001",
        )
        second_result = _LLMTurnResult(
            text=None, tool_input=second_tool_input,
            tokens_in=80, tokens_out=40, model="claude-haiku-4-5-20251001",
        )

        llm_call_count = 0
        llm_results = [first_result, second_result]

        async def _fake_llm(*args: Any, **kwargs: Any) -> _LLMTurnResult:
            nonlocal llm_call_count
            r = llm_results[llm_call_count]
            llm_call_count += 1
            return r

        applied_actions: list[Any] = []

        async def _fake_apply(s: Any, actions: Any, db: Any) -> None:
            applied_actions.append(actions)

        emitted_types: list[str] = []

        async def _fake_emit(**kwargs: Any) -> MagicMock:
            emitted_types.append(kwargs["event_type"])
            return MagicMock()

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", side_effect=_fake_llm),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", side_effect=_fake_apply),
            patch("app.core.user_sessions.emit_event", side_effect=_fake_emit),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "do the work", db)

        assert open_ is False
        assert llm_call_count == 2, "LLM should have been called twice (main + retry)"
        assert "SESSION_CLOSE_ACTIONS_DROPPED" not in emitted_types
        assert len(applied_actions) == 1
        assert len(applied_actions[0].tasks_create) == 1
        assert applied_actions[0].tasks_create[0].title == "Do the work"

    @pytest.mark.asyncio
    async def test_dropped_event_when_retry_still_disallowed(self):
        """Both LLM calls return file_writes → SESSION_CLOSE_ACTIONS_DROPPED emitted,
        dropped field names included in event metadata."""
        session = _make_session("user_cm_p1")
        db = _make_db()

        bad_result = _LLMTurnResult(
            text=None,
            tool_input={"file_writes": [{"path": "x.txt", "content": "hi"}]},
            tokens_in=100, tokens_out=50, model="claude-haiku-4-5-20251001",
        )

        emitted_events: list[dict[str, Any]] = []

        async def _fake_emit(**kwargs: Any) -> MagicMock:
            emitted_events.append(kwargs)
            return MagicMock()

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=bad_result),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_event", side_effect=_fake_emit),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "do the work", db)

        assert open_ is False
        dropped_events = [e for e in emitted_events if e["event_type"] == "SESSION_CLOSE_ACTIONS_DROPPED"]
        assert len(dropped_events) == 1
        assert "file_writes" in dropped_events[0]["metadata"]["dropped"]

    @pytest.mark.asyncio
    async def test_dropped_event_when_retry_returns_text(self):
        """First LLM call returns file_writes, retry returns plain text →
        SESSION_CLOSE_ACTIONS_DROPPED emitted, session still closes."""
        session = _make_session("user_cm_p1")
        db = _make_db()

        bad_result = _LLMTurnResult(
            text=None,
            tool_input={"file_writes": [{"path": "x.txt", "content": "hi"}]},
            tokens_in=100, tokens_out=50, model="claude-haiku-4-5-20251001",
        )
        text_result = _LLMTurnResult(
            text="I cannot do that, sorry.", tool_input=None,
            tokens_in=60, tokens_out=20, model="claude-haiku-4-5-20251001",
        )

        llm_call_count = 0
        llm_results = [bad_result, text_result]

        async def _fake_llm(*args: Any, **kwargs: Any) -> _LLMTurnResult:
            nonlocal llm_call_count
            r = llm_results[llm_call_count]
            llm_call_count += 1
            return r

        emitted_types: list[str] = []

        async def _fake_emit(**kwargs: Any) -> MagicMock:
            emitted_types.append(kwargs["event_type"])
            return MagicMock()

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", side_effect=_fake_llm),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_event", side_effect=_fake_emit),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            open_ = await process_user_turn(session, "do the work", db)

        assert open_ is False
        assert "SESSION_CLOSE_ACTIONS_DROPPED" in emitted_types
        assert "SESSION_CLOSED" in emitted_types

    @pytest.mark.asyncio
    async def test_summary_published_as_visible_message(self):
        """Non-empty summary from close_session is broadcast as a MESSAGE to the room."""
        session = _make_session("user_cm_p1")
        db = _make_db()

        tool_input = {
            "tasks_create": [{"title": "Do X"}],
            "summary": "Work done: fixed the bug",
        }
        llm_result = _LLMTurnResult(
            text=None, tool_input=tool_input,
            tokens_in=100, tokens_out=50, model="claude-haiku-4-5-20251001",
        )

        broadcast_calls: list[dict[str, Any]] = []

        async def _capture_broadcast(data: dict[str, Any], room_id: Any) -> None:
            broadcast_calls.append(data)

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=llm_result),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = _capture_broadcast
            open_ = await process_user_turn(session, "done", db)

        assert open_ is False
        message_broadcasts = [c for c in broadcast_calls if c.get("event_type") == "MESSAGE"]
        assert len(message_broadcasts) == 1
        assert message_broadcasts[0]["content"] == "Work done: fixed the bug"

    @pytest.mark.asyncio
    async def test_no_summary_broadcast_when_summary_empty(self):
        """If summary is absent, no extra MESSAGE broadcast is sent."""
        session = _make_session("user_cm_p1")
        db = _make_db()

        tool_input = {"tasks_create": [{"title": "Do X"}]}
        llm_result = _LLMTurnResult(
            text=None, tool_input=tool_input,
            tokens_in=100, tokens_out=50, model="claude-haiku-4-5-20251001",
        )

        broadcast_calls: list[dict[str, Any]] = []

        async def _capture_broadcast(data: dict[str, Any], room_id: Any) -> None:
            broadcast_calls.append(data)

        with (
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.build_system_prompt", return_value="SYS"),
            patch("app.core.user_sessions.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.user_sessions._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions._call_llm_with_tools", new_callable=AsyncMock, return_value=llm_result),
            patch("app.core.user_sessions.track_llm_usage"),
            patch("app.core.user_sessions.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = _capture_broadcast
            open_ = await process_user_turn(session, "done", db)

        assert open_ is False
        assert broadcast_calls == []
