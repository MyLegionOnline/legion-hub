"""
Integration tests for Step 8: PLAN_CREATED → CmSelfPlanSession → tasks.

Scenarios:
  1. PLAN_CREATED → session created, LLM called, tasks created, TASK_CREATED emitted
  2. Correct agent_id assigned on tasks from close_session payload
  3. Room without master_agent_id is silently skipped
  4. LLM error → session closed with reason=error, no tasks
  5. LLM answers in prose → nudge-retry, then fall back to decomposing the
     plan steps directly (+ PLAN_DECOMPOSITION_INCOMPLETE warning)
  6. render_plan_text formats plan steps correctly
  7. _call_llm_single_turn forces close_session and a 16k output budget

All LLM and DB calls are stubbed.
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.core.plan_session as plan_session
from app.core.plan_session import (
    _PLAN_MAX_TOKENS,
    _call_llm_single_turn,
    _compute_cost,
    _run_close_flow,
    dispatch_plan_event,
    handle_plan_created,
    render_plan_text,
)
from app.core.secrets.crypto import SecretValue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()

    # execute returns a result with scalars().first()
    result_mock = MagicMock()
    result_mock.scalars.return_value.first.return_value = None
    result_mock.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result_mock)

    return db


def _empty_result() -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.first.return_value = None
    r.scalars.return_value.all.return_value = []
    r.first.return_value = None
    return r


def _queue(*results: Any):
    """db.execute results in order, then empty results for any later lookup
    (the requirements-gap goal lookup, the failure-close UPDATE, ...)."""
    yield from results
    while True:
        yield _empty_result()


def _make_room(
    room_id: uuid.UUID | None = None,
    has_coordinator: bool = True,
    master_agent_id: uuid.UUID | None = None,
) -> MagicMock:
    room = MagicMock()
    room.id = room_id or uuid.uuid4()
    room.has_coordinator = has_coordinator
    _mid = master_agent_id or uuid.uuid4()
    room.coordinator_config = {"master_agent_id": str(_mid)} if has_coordinator else {}
    return room


def _make_session(
    kind: str = "cm_self_plan",
    tokens_in: int = 0,
    tokens_out: int = 0,
    closed: bool = False,
) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.tokens_in = tokens_in
    s.tokens_out = tokens_out
    s.cost_usd = Decimal("0")
    s.closed_at = datetime.now(timezone.utc) if closed else None
    s.close_reason = None
    s.is_open = not closed
    s.context_snapshot = None
    s.chat_master_agent_id = uuid.uuid4()
    s.goal_id = uuid.uuid4()
    s.plan_id = uuid.uuid4()
    s.started_at = datetime.now(timezone.utc)
    s.summary = None
    s.pending_actions = None
    return s


def _make_task(title: str, assigned_agent_id: uuid.UUID | None = None) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.title = title
    t.goal_id = uuid.uuid4()
    t.assigned_agent_id = assigned_agent_id or uuid.uuid4()
    t.created_at = datetime.now(timezone.utc)
    return t


def _make_goal_plan() -> MagicMock:
    p = MagicMock()
    p.id = uuid.uuid4()
    return p


def _make_goal_plan_step(title: str, description: str = "", agent_id: uuid.UUID | None = None) -> MagicMock:
    s = MagicMock()
    s.title = title
    s.description = description
    s.assigned_agent_id = agent_id
    return s


# ---------------------------------------------------------------------------
# render_plan_text
# ---------------------------------------------------------------------------


class TestRenderPlanText:
    def test_renders_numbered_steps(self):
        plan = _make_goal_plan()
        steps = [
            _make_goal_plan_step("Frontend setup", "Create React app"),
            _make_goal_plan_step("Backend API", "FastAPI routes"),
        ]
        text = render_plan_text(plan, steps)
        assert "1." in text
        assert "Frontend setup" in text
        assert "2." in text
        assert "Backend API" in text

    def test_empty_steps(self):
        plan = _make_goal_plan()
        text = render_plan_text(plan, [])
        assert "no steps" in text

    def test_includes_agent_note_when_assigned(self):
        agent_id = uuid.uuid4()
        plan = _make_goal_plan()
        steps = [_make_goal_plan_step("Task A", agent_id=agent_id)]
        text = render_plan_text(plan, steps)
        assert str(agent_id) in text

    def test_renders_nickname_when_agent_names_provided(self):
        agent_id = uuid.uuid4()
        plan = _make_goal_plan()
        steps = [_make_goal_plan_step("Task A", agent_id=agent_id)]
        text = render_plan_text(plan, steps, agent_names={agent_id: "Writer"})
        assert "— agent: @Writer" in text
        assert str(agent_id) not in text

    def test_no_agent_note_when_unassigned(self):
        plan = _make_goal_plan()
        steps = [_make_goal_plan_step("Task A", agent_id=None)]
        text = render_plan_text(plan, steps)
        assert "agent" not in text.lower() or "— agent:" not in text


# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_known_model(self):
        cost = _compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0)
        assert cost == pytest.approx(0.8, rel=1e-5)

    def test_unknown_model_is_not_priced(self):
        assert _compute_cost("unknown-xyz", 1_000_000, 0) == 0


# ---------------------------------------------------------------------------
# Scenario 1: happy path — PLAN_CREATED → session → tasks → events
# ---------------------------------------------------------------------------


class TestHandlePlanCreated:
    @pytest.fixture(autouse=True)
    def _planner_llm_config(self):
        """§8.1: handle_plan_created resolves the planner model via
        resolve_planner_llm_config (room.get(Agent) + planner_model override).
        Default it to a normal ChatMaster model; tests that exercise the
        missing-model path patch it themselves.
        """
        agent = MagicMock()
        agent.nickname = "cm"
        agent.owner_id = uuid.uuid4()
        agent.id = uuid.uuid4()
        with patch(
            "app.core.plan_session.resolve_planner_llm_config",
            new_callable=AsyncMock,
            return_value=(agent, {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}),
        ) as m:
            yield m

    @pytest.mark.asyncio
    async def test_happy_path_creates_session_and_tasks(self):
        """PLAN_CREATED → CmSelfPlanSession created, LLM called, close actions applied.

        TASK_CREATED emission and proxy dispatch happen inside apply_close_actions
        (mocked here) — plan_session must NOT duplicate them.
        """
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        plan_text = "# Plan\n1. **Build UI**\n2. **Write API**"
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id = room_id
        session.goal_id = goal_id
        session.plan_id = plan_id
        session.chat_master_agent_id = cm_agent_id

        task1 = _make_task("Build UI", assigned_agent_id=agent_id)
        task2 = _make_task("Write API", assigned_agent_id=agent_id)

        db = _make_db()

        # Sequence of db.execute return values:
        # 1. Room lookup
        # 2. Agent llm_config lookup (in _get_agent_llm_config)
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room

        agent_result = MagicMock()
        agent_mock = MagicMock()
        agent_mock.llm_config = {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
        agent_result.scalars.return_value.first.return_value = agent_mock

        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        tool_input = {
            "tasks_create": [
                {"title": "Build UI", "assigned_agent_id": str(agent_id)},
                {"title": "Write API", "assigned_agent_id": str(agent_id)},
            ],
            "summary": "Plan decomposed into 2 tasks",
        }

        emitted_events: list[str] = []

        async def _fake_emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
            emitted_events.append(event_type)
            return MagicMock()

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=(None, tool_input, 200, 100, "claude-haiku-4-5-20251001")),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance

            await handle_plan_created(room_id, goal_id, plan_id, plan_text, db)

        assert "SESSION_STARTED" in emitted_events
        mock_apply.assert_awaited_once()
        # plan_session must not emit TASK_CREATED itself — apply_close_actions owns it
        assert "TASK_CREATED" not in emitted_events
        assert "SESSION_CLOSED" in emitted_events
        assert session.close_reason == "task_done"
        assert session.closed_at is not None

    @pytest.mark.asyncio
    async def test_tasks_get_correct_agent_id(self):
        """Tasks in close_session payload must carry the correct assigned_agent_id."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id = room_id
        session.goal_id = goal_id

        db = _make_db()

        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_mock = MagicMock()
        agent_mock.llm_config = {}
        agent_result.scalars.return_value.first.return_value = agent_mock
        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        tool_input = {
            "tasks_create": [
                {"title": "Implement feature", "assigned_agent_id": str(agent_id)},
            ],
        }

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=(None, tool_input, 100, 50, "claude-haiku-4-5-20251001")),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance

            await handle_plan_created(room_id, goal_id, plan_id, "plan text", db)

        mock_apply.assert_awaited_once()
        actions = mock_apply.await_args.args[1]
        assert len(actions.tasks_create) == 1
        assert str(actions.tasks_create[0].assigned_agent_id) == str(agent_id)

    # ---------------------------------------------------------------------------
    # Scenario 3: no master_agent_id → skipped silently
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_skips_when_no_master_agent_id(self):
        """Rooms without master_agent_id must not create a session."""
        room_id = uuid.uuid4()
        room = _make_room(room_id=room_id, has_coordinator=False)
        db = _make_db()

        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(return_value=room_result)

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            await handle_plan_created(room_id, uuid.uuid4(), uuid.uuid4(), "plan", db)

        MockRepo.return_value.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_room_not_found(self):
        """Missing room must not crash — log and return."""
        db = _make_db()

        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = None
        db.execute = AsyncMock(return_value=room_result)

        # Should not raise
        await handle_plan_created(uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), "plan", db)

    # ---------------------------------------------------------------------------
    # §8.1: planner model resolution
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_missing_planner_model_emits_event_and_deterministic_fallback(
        self, _planner_llm_config
    ):
        """ChatMaster with no model → PLANNER_MODEL_MISSING + plan-step
        decomposition, zero LLM calls (no silent Haiku)."""
        _planner_llm_config.return_value = (MagicMock(nickname="cm"), {"provider": "anthropic"})

        room_id, goal_id, plan_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        room = _make_room(room_id=room_id)
        session = _make_session()
        session.room_id, session.goal_id = room_id, goal_id
        db = _make_db()
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(return_value=room_result)

        emitted: list[str] = []

        async def _fake_emit(room_id, event_type, **kw):
            emitted.append(event_type)
            return MagicMock()

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock) as mock_llm,
            patch("app.core.plan_session._decompose_from_plan_steps", new_callable=AsyncMock) as mock_fb,
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance
            await handle_plan_created(room_id, goal_id, plan_id, "plan text", db)

        assert "PLANNER_MODEL_MISSING" in emitted
        mock_llm.assert_not_awaited()
        mock_fb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_planner_model_override_reaches_the_llm_call(self, _planner_llm_config):
        """coordinator_config['planner_model'] override flows into the turn."""
        _planner_llm_config.return_value = (
            MagicMock(nickname="cm"),
            {"provider": "anthropic", "model": "claude-opus-4-8"},
        )
        room_id, goal_id = uuid.uuid4(), uuid.uuid4()
        room = _make_room(room_id=room_id)
        session = _make_session()
        session.room_id, session.goal_id = room_id, goal_id
        db = _make_db()
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(return_value=room_result)

        tool_input = {"tasks_create": [{"title": "T", "description": "d"}]}
        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._preflight_plan_request", new_callable=AsyncMock, return_value=None),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=(None, tool_input, 5, 3, "claude-opus-4-8")) as mock_llm,
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance
            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "plan text", db)

        assert mock_llm.await_args.args[0]["model"] == "claude-opus-4-8"

    # ---------------------------------------------------------------------------
    # Scenario 4: LLM error → session closed with reason=error
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_llm_error_closes_session_with_error_reason(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id = room_id
        session.goal_id = goal_id
        db = _make_db()

        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_mock = MagicMock()
        agent_mock.llm_config = {}
        agent_result.scalars.return_value.first.return_value = agent_mock
        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        emitted: list[str] = []

        async def _fake_emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
            emitted.append(event_type)
            return MagicMock()

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn",
                  new_callable=AsyncMock, side_effect=RuntimeError("network timeout")),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance

            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "plan text", db)

        assert session.close_reason == "error"
        assert "SESSION_CLOSED" in emitted
        assert "SESSION_STARTED" in emitted
        assert "TASK_CREATED" not in emitted

    # ---------------------------------------------------------------------------
    # Scenario 5: LLM keeps answering in prose → nudge-retry, then fall back to
    # decomposing the plan steps directly so the goal is not stranded.
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_prose_reply_is_retried_with_a_nudge(self):
        """First prose reply → retry with the nudge appended; a tool call on the
        second attempt is honoured normally."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id = room_id
        session.goal_id = goal_id
        db = _make_db()

        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_mock = MagicMock()
        agent_mock.llm_config = {}
        agent_result.scalars.return_value.first.return_value = agent_mock
        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        turn_messages: list[str] = []

        async def _fake_turn(cfg, sys_prompt, user_msg, **kw):
            turn_messages.append(user_msg)
            if len(turn_messages) == 1:
                return ("Sure, here is what I'd do...", None, 100, 50, "m")
            return (
                None,
                {"tasks_create": [{"title": "Build", "assigned_agent_id": str(agent_id)}]},
                80, 40, "m",
            )

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn", side_effect=_fake_turn),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance

            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "plan text", db)

        assert len(turn_messages) == 2
        assert turn_messages[0] == "plan text"
        assert turn_messages[1].startswith("plan text")
        assert "close_session" in turn_messages[1]   # nudge text present
        mock_apply.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persistent_prose_falls_back_to_plan_step_decomposition(self):
        """Still prose after the retry → one task per plan step + a visible
        PLAN_DECOMPOSITION_INCOMPLETE warning, never a silent zero-task close."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        step_agent_id = uuid.uuid4()

        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id = room_id
        session.goal_id = goal_id
        session.plan_id = plan_id
        db = _make_db()

        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_mock = MagicMock()
        agent_mock.llm_config = {}
        agent_result.scalars.return_value.first.return_value = agent_mock
        step1 = _make_goal_plan_step("Set up Vite pipeline", "package.json + vite.config", step_agent_id)
        step2 = _make_goal_plan_step("Glitch component", "cyclical glitch on subtitle")
        steps_result = MagicMock()
        steps_result.scalars.return_value.all.return_value = [step1, step2]
        db.execute = AsyncMock(side_effect=_queue(room_result, steps_result))

        emitted: list[str] = []

        async def _fake_emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
            emitted.append(event_type)
            return MagicMock()

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=("Here are the tasks: ...", None, 100, 50, "claude-haiku-4-5-20251001")),
            patch("app.core.plan_session._call_llm_json_mode", new_callable=AsyncMock,
                  return_value=("not json either", 10, 5)),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance

            await handle_plan_created(room_id, goal_id, plan_id, "plan text", db)

        assert session.close_reason == "task_done"
        assert session.closed_at is not None
        assert "PLAN_DECOMPOSITION_INCOMPLETE" in emitted
        mock_apply.assert_awaited_once()
        actions = mock_apply.await_args.args[1]
        assert [t.title for t in actions.tasks_create] == [
            "Set up Vite pipeline", "Glitch component",
        ]
        assert str(actions.tasks_create[0].assigned_agent_id) == str(step_agent_id)

    @pytest.mark.asyncio
    async def test_close_session_with_empty_tasks_create_is_nudged(self):
        """The weak-model failure seen on qwen3-27b: close_session called, plan
        described in `summary`, tasks_create empty. Nudge, then honour a filled
        retry."""
        room_id, goal_id = uuid.uuid4(), uuid.uuid4()
        cm_agent_id, agent_id = uuid.uuid4(), uuid.uuid4()
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id, session.goal_id = room_id, goal_id
        db = _make_db()
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_result.scalars.return_value.first.return_value = MagicMock(llm_config={})
        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        turns: list[str] = []

        async def _fake_turn(cfg, sysp, msg, **kw):
            turns.append(msg)
            if len(turns) == 1:
                return (None, {"summary": "1. do X  2. do Y"}, 90, 40, "m")
            return (None, {"tasks_create": [{"title": "X", "assigned_agent_id": str(agent_id)},
                                            {"title": "Y", "assignee": "@Coder"}]}, 70, 30, "m")

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn", side_effect=_fake_turn),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance
            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "plan text", db)

        assert len(turns) == 2
        assert "tasks_create field is EMPTY" in turns[1]
        mock_apply.assert_awaited_once()
        assert [t.title for t in mock_apply.await_args.args[1].tasks_create] == ["X", "Y"]

    @pytest.mark.asyncio
    async def test_persistent_empty_tasks_folds_model_summary_into_fallback(self):
        """Still only a summary after every retry → plan-step fallback, with the
        model's own breakdown folded into the task description."""
        room_id, goal_id, plan_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id, session.goal_id, session.plan_id = room_id, goal_id, plan_id
        db = _make_db()
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_result.scalars.return_value.first.return_value = MagicMock(llm_config={})
        step = _make_goal_plan_step("Glitch + Vite + CI", "the whole thing")
        steps_result = MagicMock()
        steps_result.scalars.return_value.all.return_value = [step]
        db.execute = AsyncMock(side_effect=_queue(room_result, steps_result))

        emitted: list[str] = []

        async def _fake_emit(room_id, event_type, **kw):
            emitted.append(event_type)
            return MagicMock()

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=(None, {"summary": "1) Frontender: Vite  2) Backender: CI"}, 90, 40, "m")),
            patch("app.core.plan_session._call_llm_json_mode", new_callable=AsyncMock,
                  return_value=('{"summary": "1) Frontender: Vite  2) Backender: CI"}', 10, 5)),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance
            await handle_plan_created(room_id, goal_id, plan_id, "plan text", db)

        assert "PLAN_DECOMPOSITION_INCOMPLETE" in emitted
        mock_apply.assert_awaited_once()
        tc = mock_apply.await_args.args[1].tasks_create
        assert len(tc) == 1
        assert "Frontender: Vite" in tc[0].description

    @pytest.mark.asyncio
    async def test_recovers_via_json_mode_when_tool_use_returns_empty(self):
        """The qwen3-27b case: forced tool_choice returns `{}` on every attempt,
        but response_format=json_object yields a full decomposition → real tasks,
        no fallback warning."""
        room_id, goal_id = uuid.uuid4(), uuid.uuid4()
        cm_agent_id, a1, a2 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id, session.goal_id = room_id, goal_id
        db = _make_db()
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_result.scalars.return_value.first.return_value = MagicMock(llm_config={})
        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        emitted: list[str] = []

        async def _fake_emit(room_id, event_type, **kw):
            emitted.append(event_type)
            return MagicMock()

        json_reply = (
            '```json\n{"tasks_create": ['
            f'{{"ref": "a", "title": "Vite pipeline", "description": "d1", "assigned_agent_id": "{a1}"}},'
            f'{{"ref": "b", "title": "CI step", "description": "d2", "assigned_agent_id": "{a2}", "depends_on": ["a"]}}'
            '], "summary": "2 tasks"}\n```'
        )

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=(None, {}, 50, 20, "qwen/qwen3.8-27b")),
            patch("app.core.plan_session._call_llm_json_mode", new_callable=AsyncMock,
                  return_value=(json_reply, 60, 400)) as mock_json,
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance
            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "plan text", db)

        mock_json.assert_awaited_once()
        assert "PLAN_DECOMPOSITION_INCOMPLETE" not in emitted
        mock_apply.assert_awaited_once()
        actions = mock_apply.await_args.args[1]
        assert [t.title for t in actions.tasks_create] == ["Vite pipeline", "CI step"]
        assert str(actions.tasks_create[0].assigned_agent_id) == str(a1)

    @pytest.mark.asyncio
    async def test_dialogue_digest_is_prepended_to_decomposition_turn(self):
        """Stage 4: the task-formation dialogue digest lands in the LLM turn
        message ahead of the plan text."""
        room_id, goal_id = uuid.uuid4(), uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id, session.goal_id = room_id, goal_id
        db = _make_db()
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_result.scalars.return_value.first.return_value = MagicMock(llm_config={})
        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        digest = "# Spec: NGINX\n\n## Digest\n- rate limit 100 r/s [#3]"
        tool_input = {"tasks_create": [{"title": "T", "description": "d"}]}

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._preflight_plan_request", new_callable=AsyncMock, return_value=None),
            patch("app.core.plan_dialogue.digest_for_decomposition", new_callable=AsyncMock, return_value=digest),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=(None, tool_input, 10, 5, "claude-haiku-4-5-20251001")) as mock_llm,
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance
            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "PLAN BODY", db)

        turn_message = mock_llm.await_args.args[2]
        assert "spec-agreement dialogue archive" in turn_message
        assert "rate limit 100 r/s [#3]" in turn_message
        assert turn_message.index("Digest") < turn_message.index("PLAN BODY")

    # ---------------------------------------------------------------------------
    # Scenario 6: token budget reached → session closed immediately
    # ---------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_token_budget_reached_stops_before_llm(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id = room_id
        db = _make_db()

        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(return_value=room_result)

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.uses_cumulative_limit", return_value=True),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=False),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock) as mock_llm,
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance

            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "big plan text", db)

        mock_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_window_mode_does_not_block_planner_on_cumulative_usage(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session(tokens_in=110_000, tokens_out=5_000)
        session.room_id = room_id
        session.goal_id = goal_id
        db = _make_db()

        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent = MagicMock()
        agent.llm_config = {"provider": "anthropic"}
        agent_result.scalars.return_value.first.return_value = agent
        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.uses_cumulative_limit", return_value=False),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock) as budget_guard,
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=(None, {"tasks_create": [{"title": "X", "assignee": "@a"}], "summary": "ok"}, 20, 10, "test-model")) as mock_llm,
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance

            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "compact plan", db)

        budget_guard.assert_not_awaited()
        mock_llm.assert_awaited_once()


class TestPlanSessionAlwaysCloses:
    """Every exit of the decomposition turn closes the session, and every LLM
    step stamps last_activity_at (the watchdog stale-plan sweep reads it)."""

    @pytest.fixture(autouse=True)
    def _planner_llm_config(self):
        agent = MagicMock()
        agent.nickname = "cm"
        agent.owner_id = uuid.uuid4()
        agent.id = uuid.uuid4()
        with patch(
            "app.core.plan_session.resolve_planner_llm_config",
            new_callable=AsyncMock,
            return_value=(agent, {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}),
        ):
            yield

    def _setup(self, *, update_matches: bool = True):
        room_id = uuid.uuid4()
        room = _make_room(room_id=room_id)
        session = _make_session()
        session.room_id = room_id
        session.last_activity_at = None
        db = _make_db()
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        update_result = _empty_result()
        update_result.first.return_value = (session.id,) if update_matches else None

        def _execute(stmt, *args, **kwargs):
            # Room lookup first; the failure-close UPDATE ... RETURNING gets
            # update_result; any other lookup finds nothing.
            if _execute.calls == 0:
                _execute.calls += 1
                return room_result
            _execute.calls += 1
            if getattr(stmt, "is_update", False):
                return update_result
            return _empty_result()

        _execute.calls = 0
        db.execute = AsyncMock(side_effect=_execute)
        return room_id, session, db

    @contextlib.contextmanager
    def _patched(self, session, emitted, *, fail_first_close_emit=False, **llm):
        async def _fake_emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
            emitted.append((event_type, kwargs.get("metadata") or {}))
            if (
                fail_first_close_emit
                and event_type == "SESSION_CLOSED"
                and len([e for e in emitted if e[0] == "SESSION_CLOSED"]) == 1
            ):
                raise RuntimeError("broadcast failed")
            return MagicMock()

        repo = MagicMock()
        repo.return_value.create = AsyncMock(return_value=session)
        with (
            patch("app.core.plan_session.SessionRepository", repo),
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session._augment_plan_text_with_dialogue",
                  new_callable=AsyncMock, side_effect=lambda text, *_a: text),
            patch("app.core.plan_session._preflight_plan_request", new_callable=AsyncMock, return_value=None),
            patch("app.core.plan_session.resolve_agent_secret", new_callable=AsyncMock, return_value=None),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock, **llm),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
        ):
            yield

    _TOOL_INPUT = {"tasks_create": [
        {"title": "T", "description": "d", "assigned_agent_id": str(uuid.uuid4())},
    ]}
    _MODEL = "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_close_flow_exception_closes_session_as_error(self):
        room_id, session, db = self._setup()
        emitted: list = []
        with self._patched(
            session, emitted, return_value=(None, self._TOOL_INPUT, 1, 1, self._MODEL)
        ), patch(
            "app.core.plan_session.apply_close_actions",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db exploded"),
        ):
            with pytest.raises(RuntimeError, match="db exploded"):
                await handle_plan_created(room_id, uuid.uuid4(), uuid.uuid4(), "plan", db)

        db.rollback.assert_awaited()
        assert session.close_reason == "error"
        assert session.closed_at is not None
        closed = [m for t, m in emitted if t == "SESSION_CLOSED"]
        assert [m["close_reason"] for m in closed] == ["error"]

    @pytest.mark.asyncio
    async def test_cancellation_closes_session_as_cancelled(self):
        room_id, session, db = self._setup()
        emitted: list = []
        with self._patched(session, emitted, side_effect=asyncio.CancelledError()):
            with pytest.raises(asyncio.CancelledError):
                await handle_plan_created(room_id, uuid.uuid4(), uuid.uuid4(), "plan", db)

        assert session.close_reason == "cancelled"
        assert [m["close_reason"] for t, m in emitted if t == "SESSION_CLOSED"] == ["cancelled"]

    @pytest.mark.asyncio
    async def test_failure_after_own_close_keeps_its_reason(self):
        """The turn's own SESSION_CLOSED emit committed the close and then
        failed (broadcast): the conditional UPDATE matches nothing, so the
        reason stays task_done and no second SESSION_CLOSED is emitted."""
        room_id, session, db = self._setup(update_matches=False)
        emitted: list = []
        with self._patched(
            session,
            emitted,
            fail_first_close_emit=True,
            return_value=(None, self._TOOL_INPUT, 1, 1, self._MODEL),
        ), patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="broadcast failed"):
                await handle_plan_created(room_id, uuid.uuid4(), uuid.uuid4(), "plan", db)

        assert session.close_reason == "task_done"
        assert [t for t, _m in emitted].count("SESSION_CLOSED") == 1

    @pytest.mark.asyncio
    async def test_each_llm_step_stamps_last_activity(self):
        room_id, session, db = self._setup()
        emitted: list = []
        with self._patched(
            session,
            emitted,
            side_effect=[
                ("prose", None, 1, 1, self._MODEL),
                (None, self._TOOL_INPUT, 1, 1, self._MODEL),
            ],
        ), patch(
            "app.core.plan_session.apply_close_actions", new_callable=AsyncMock
        ), patch(
            "app.core.plan_session._touch_activity",
            new_callable=AsyncMock,
            wraps=plan_session._touch_activity,
        ) as touch:
            await handle_plan_created(room_id, uuid.uuid4(), uuid.uuid4(), "plan", db)

        assert touch.await_count == 2
        assert session.last_activity_at is not None
        assert session.close_reason == "task_done"

    @pytest.mark.asyncio
    async def test_return_without_close_is_closed_as_error(self):
        room_id, session, db = self._setup()
        emitted: list = []
        with self._patched(session, emitted, return_value=(None, None, 0, 0, self._MODEL)), patch(
            "app.core.plan_session._run_plan_turn", new_callable=AsyncMock
        ):
            await handle_plan_created(room_id, uuid.uuid4(), uuid.uuid4(), "plan", db)

        assert session.close_reason == "error"
        assert session.last_activity_at is not None  # stamped at start


# ---------------------------------------------------------------------------
# _run_close_flow unit test
# ---------------------------------------------------------------------------


class TestRunCloseFlow:
    @pytest.mark.asyncio
    async def test_delegates_task_creation_to_apply_close_actions(self):
        """TASK_CREATED emission and dispatch belong to apply_close_actions —
        _run_close_flow must not duplicate them, only close the session."""
        session = _make_session()
        agent_id = uuid.uuid4()

        db = _make_db()

        tool_input = {
            "tasks_create": [
                {"title": "Task A", "assigned_agent_id": str(agent_id)},
                {"title": "Task B", "assigned_agent_id": str(agent_id)},
            ],
            "summary": "Two tasks",
        }

        emitted: list[str] = []
        closed_metadata: list[dict] = []

        async def _fake_emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
            emitted.append(event_type)
            if event_type == "SESSION_CLOSED":
                closed_metadata.append(kwargs.get("metadata", {}))
            return MagicMock()

        with (
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
        ):
            await _run_close_flow(session, tool_input, db)

        mock_apply.assert_awaited_once()
        assert "TASK_CREATED" not in emitted
        assert "SESSION_CLOSED" in emitted
        assert closed_metadata[0]["tasks_created"] == 2
        assert session.close_reason == "task_done"
        assert session.closed_at is not None

    @pytest.mark.asyncio
    async def test_invalid_tool_input_falls_back_to_empty_actions(self):
        session = _make_session()

        db = _make_db()

        emitted: list[str] = []

        async def _fake_emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
            emitted.append(event_type)
            return MagicMock()

        with (
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.plan_session.emit_event", side_effect=_fake_emit),
            patch(
                "app.core.plan_session.extract_close_actions_from_tool_use",
                side_effect=ValueError("bad input"),
            ),
        ):
            await _run_close_flow(session, {"bad": "data"}, db)

        assert "SESSION_CLOSE_ACTIONS_INVALID" in emitted
        assert "SESSION_CLOSED" in emitted
        assert session.close_reason == "task_done"


# ---------------------------------------------------------------------------
# _call_llm_single_turn — forced close_session + 16k budget (messages wire)
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code: int, payload: dict, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (str(payload) if status_code >= 400 else "")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeHttpxClient:
    def __init__(self, responses: list[_FakeResp]):
        self._responses = list(responses)
        self.posts: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, url: str, headers: dict | None = None, json: dict | None = None):
        self.posts.append({"url": url, "headers": headers or {}, "json": json or {}})
        return self._responses.pop(0)


_TOOL_USE_PAYLOAD = {
    "stop_reason": "tool_use",
    "content": [{"type": "tool_use", "name": "close_session", "input": {"tasks_create": []}}],
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class TestCallLlmSingleTurnForcedTool:
    @pytest.mark.asyncio
    async def test_messages_wire_forces_close_tool_and_16k_budget(self, monkeypatch):
        client = _FakeHttpxClient([_FakeResp(200, _TOOL_USE_PAYLOAD)])
        monkeypatch.setattr(plan_session.httpx, "AsyncClient", lambda *a, **kw: client)

        _text, tool_input, _ti, _to, _model = await _call_llm_single_turn(
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            "SYS",
            "plan text",
            secret_value=SecretValue("sk-test"),
        )

        assert tool_input == {"tasks_create": []}
        body = client.posts[0]["json"]
        assert body["max_tokens"] == _PLAN_MAX_TOKENS == 16384
        assert body["tool_choice"] == {"type": "tool", "name": "close_session"}

    @pytest.mark.asyncio
    async def test_messages_wire_retries_without_tool_choice_on_400(self, monkeypatch):
        client = _FakeHttpxClient([
            _FakeResp(400, {}, text='{"error":"tool_choice not supported"}'),
            _FakeResp(200, _TOOL_USE_PAYLOAD),
        ])
        monkeypatch.setattr(plan_session.httpx, "AsyncClient", lambda *a, **kw: client)

        _text, tool_input, *_ = await _call_llm_single_turn(
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
            "SYS",
            "plan text",
            secret_value=SecretValue("sk-test"),
        )

        assert tool_input == {"tasks_create": []}
        assert len(client.posts) == 2
        assert "tool_choice" not in client.posts[1]["json"]


# ---------------------------------------------------------------------------
# dispatch_plan_event
# ---------------------------------------------------------------------------


class TestDispatchPlanEvent:
    def test_schedules_asyncio_task(self):
        """dispatch_plan_event must call asyncio.create_task without raising."""
        with patch("app.core.plan_session.asyncio.create_task") as mock_create_task:
            dispatch_plan_event(
                room_id=uuid.uuid4(),
                goal_id=uuid.uuid4(),
                plan_id=uuid.uuid4(),
                plan_text="some plan text",
            )
        mock_create_task.assert_called_once()
        task_name = mock_create_task.call_args.kwargs.get("name", "")
        assert "plan_session_" in task_name


class TestUnassignedTaskNudge:
    """A task the planner leaves without an agent never starts: nudge for it."""

    @pytest.fixture(autouse=True)
    def _planner_llm_config(self):
        agent = MagicMock(nickname="cm", owner_id=uuid.uuid4(), id=uuid.uuid4())
        with patch(
            "app.core.plan_session.resolve_planner_llm_config",
            new_callable=AsyncMock,
            return_value=(agent, {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}),
        ), patch(
            "app.core.plan_session.resolve_agent_secret", new_callable=AsyncMock, return_value=None,
        ):
            yield

    async def _run(self, replies):
        room_id, goal_id, cm_agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        session = _make_session()
        session.room_id, session.goal_id = room_id, goal_id
        db = _make_db()
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        agent_result = MagicMock()
        agent_result.scalars.return_value.first.return_value = MagicMock(llm_config={})
        db.execute = AsyncMock(side_effect=_queue(room_result, agent_result))

        turns: list[str] = []

        async def _fake_turn(cfg, sysp, msg, **kw):
            turns.append(msg)
            return (None, replies[min(len(turns), len(replies)) - 1], 10, 5, "m")

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="SYS"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.plan_session._call_llm_single_turn", side_effect=_fake_turn),
            patch("app.core.plan_session.track_llm_usage"),
            patch("app.core.plan_session.apply_close_actions", new_callable=AsyncMock) as mock_apply,
            patch("app.core.plan_session.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(return_value=session)
            MockRepo.return_value = repo_instance
            await handle_plan_created(room_id, goal_id, uuid.uuid4(), "plan text", db)
        mock_apply.assert_awaited_once()
        return turns, mock_apply.await_args.args[1].tasks_create

    @pytest.mark.asyncio
    async def test_missing_assignee_is_nudged(self):
        agent_id = str(uuid.uuid4())
        turns, tasks = await self._run([
            {"tasks_create": [{"title": "Build CI"},
                              {"title": "Review", "assigned_agent_id": agent_id}]},
            {"tasks_create": [{"title": "Build CI", "assigned_agent_id": agent_id},
                              {"title": "Review", "assigned_agent_id": agent_id}]},
        ])
        assert len(turns) == 2
        assert "NO assigned_agent_id" in turns[1]
        assert "- Build CI" in turns[1]
        assert all(str(t.assigned_agent_id) == agent_id for t in tasks)

    @pytest.mark.asyncio
    async def test_persistent_gap_keeps_the_best_attempt(self):
        agent_id = str(uuid.uuid4())
        turns, tasks = await self._run([
            {"tasks_create": [{"title": "A"}, {"title": "B"}]},
            {"tasks_create": [{"title": "A", "assigned_agent_id": agent_id}, {"title": "B"}]},
            {"summary": "nothing"},
        ])
        assert len(turns) == 3
        assert [str(t.assigned_agent_id) if t.assigned_agent_id else None for t in tasks] == [
            agent_id, None,
        ]

    @pytest.mark.asyncio
    async def test_nickname_and_checkpoint_are_not_nudged(self):
        turns, tasks = await self._run([
            {"tasks_create": [{"title": "Build", "assigned_agent_id": "@Coder"},
                              {"title": "Approve", "checkpoint_kind": "confirm"}]},
        ])
        assert len(turns) == 1
        assert tasks[0].assignee_hint == "@Coder"
