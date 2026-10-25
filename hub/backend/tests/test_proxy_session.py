"""
Integration tests for Step 9: TASK_CREATED → CmAgentProxySession pair → approve/reject.

Scenarios:
  1. Happy path (Variant A): TASK_CREATED → two sessions, agent turn → CM approve → TASK_APPROVED
  2. CM rejects (TASK_REJECTED sentinel in summary) → TASK_REJECTED emitted, task state set
  3. Agent calls close_session first → CM gets notification → CM approves
  4. Agent token limit → CM independently closed → loop ends
  5. CM token limit → agent side stays open (independent budgets)
  6. Task has no assigned agent → silently skipped
  7. Room has no master_agent_id → silently skipped
  8. LLM error on agent side → both sessions closed with reason=error
  9. Max turns exceeded → both sessions force-closed with reason=cancelled
  10. render_task_data formats task correctly
  11. dispatch_task_event schedules an asyncio task

All LLM and DB calls are stubbed.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.core.proxy_session import (
    _MAX_PROXY_TURNS,
    _REJECT_SENTINEL,
    _compute_cost,
    _save_history,
    dispatch_task_event,
    handle_task_created,
    render_task_data,
)
from app.models.task import DispatchStatus, TaskState


class TestProxySaveHistoryCounts:
    """Stage 5.3 of internal design notes."""

    def test_updates_message_and_turn_counts(self):
        session = SimpleNamespace(context_snapshot=None, message_count=0, turn_count=0)
        history = [
            {"role": "user", "content": "kick off"},
            {"role": "assistant", "content": "on it"},
            {"role": "tool", "content": "read_file(...) -> ..."},
        ]
        _save_history(session, history)
        assert session.message_count == 3
        assert session.turn_count == 1  # one role='user' entry


class TestOpenAIEmptyChoices:
    @pytest.mark.asyncio
    async def test_empty_choices_are_retried_before_indexing(self):
        """OpenRouter can intermittently return HTTP 200 with no choices."""
        from app.core import proxy_session

        responses = [
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace()]),
        ]

        async def request():
            return responses.pop(0)

        with patch.object(proxy_session, "_LLM_RETRY_BACKOFF_S", (0.0,)):
            response = await proxy_session._llm_request_with_retry(
                request,
                session_id="test-session",
                label="test/openai",
                validate_response=proxy_session._require_openai_choices,
            )

        assert len(response.choices) == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()

    result_mock = MagicMock()
    result_mock.scalars.return_value.first.return_value = None
    result_mock.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result_mock)
    return db


def _make_task(
    assigned_agent_id: uuid.UUID | None = None,
    goal_id: uuid.UUID | None = None,
) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "Implement login page"
    t.description = "Build the login page with email/password fields."
    t.state = TaskState.IN_PROGRESS
    t.assigned_agent_id = assigned_agent_id or uuid.uuid4()
    t.goal_id = goal_id or uuid.uuid4()
    t.created_at = datetime.now(timezone.utc)
    t.updated_at = datetime.now(timezone.utc)
    t.dispatch_status = DispatchStatus.READY
    return t


def _claim_result() -> MagicMock:
    """Result for the atomic READY -> DISPATCHED claim UPDATE."""
    r = MagicMock()
    r.rowcount = 1
    return r


def _agent_result() -> MagicMock:
    """Result for the media-routing Agent lookup in handle_task_created."""
    r = MagicMock()
    r.scalars.return_value.first.return_value = MagicMock(llm_config={})
    return r


def _empty_result() -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.first.return_value = None
    r.scalars.return_value.all.return_value = []
    r.first.return_value = None
    r.all.return_value = []
    return r


def _db_results(*results: MagicMock):
    """Yield the given results in order, then empty results forever.

    Follow-up queries (task deps unblock, goal progress, etc.) get a harmless
    empty result instead of exhausting the side_effect list.
    """
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
    _mid = master_agent_id or uuid.uuid4()
    room.coordinator_config = {"master_agent_id": str(_mid)} if has_coordinator else {}
    return room


def _make_session(
    kind: str = "cm_agent_proxy",
    side: str = "agent",
    closed: bool = False,
) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.side = side
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.closed_at = datetime.now(timezone.utc) if closed else None
    s.close_reason = None
    s.is_open = not closed
    s.context_snapshot = None
    s.agent_id = uuid.uuid4()
    s.task_id = uuid.uuid4()
    s.goal_id = uuid.uuid4()
    s.plan_id = uuid.uuid4()
    s.peer_session_id = None
    s.started_at = datetime.now(timezone.utc)
    s.summary = None
    s.pending_actions = None
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = 0
    return s


def _fake_emit_factory(captured: list[str]):
    async def _emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
        captured.append(event_type)
        return MagicMock()
    return _emit


# ---------------------------------------------------------------------------
# render_task_data
# ---------------------------------------------------------------------------


class TestRenderTaskData:
    def test_includes_title_and_description(self):
        task = _make_task()
        text = render_task_data(task)
        assert task.title in text
        assert task.description in text

    def test_includes_goal_id(self):
        goal_id = uuid.uuid4()
        task = _make_task(goal_id=goal_id)
        text = render_task_data(task)
        assert str(goal_id) in text

    def test_includes_assigned_agent(self):
        agent_id = uuid.uuid4()
        task = _make_task(assigned_agent_id=agent_id)
        text = render_task_data(task)
        assert str(agent_id) in text

    def test_includes_state(self):
        task = _make_task()
        text = render_task_data(task)
        assert "in_progress" in text.lower()


# ---------------------------------------------------------------------------
# _compute_cost
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_known_model(self):
        cost = _compute_cost("claude-haiku-4-5-20251001", 1_000_000, 0)
        assert cost == pytest.approx(0.8, rel=1e-5)

    def test_unknown_model_is_not_priced(self):
        assert _compute_cost("unknown-xyz", 0, 1_000_000) == 0


# ---------------------------------------------------------------------------
# Scenario 1: happy path — agent replies, CM approves
# ---------------------------------------------------------------------------


class TestHappyPathApprove:
    @pytest.mark.asyncio
    async def test_agent_replies_cm_approves(self):
        """Full Variant A: agent submits → CM calls close_session (approve) → TASK_APPROVED."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id

        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()

        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room

        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        agent_llm_result = MagicMock()
        agent_llm_result.scalars.return_value.first.return_value = MagicMock(llm_config={})
        cm_llm_result = MagicMock()
        cm_llm_result.scalars.return_value.first.return_value = MagicMock(llm_config={})

        emitted: list[str] = []
        cm_tool_input = {"summary": "Work looks good.", "plan_update": None}

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=[
                # Agent turn 1: close_session — submitting work is what makes the
                # next CM turn a review. A plain text turn is a mid-task question,
                # and a verdict on one is converted into an answer instead.
                (None, {"summary": "Feature implemented."}, 100, 80, "claude-haiku-4-5-20251001", 100, 100),
                # CM turn 1: close_session (approve)
                (None, cm_tool_input, 120, 90, "claude-haiku-4-5-20251001", 120, 120),
            ]),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory(emitted)),
            patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock) as mock_complete,
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance

            broker_instance = AsyncMock()
            MockBroker.return_value = broker_instance

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        assert "SESSION_STARTED" in emitted
        assert emitted.count("SESSION_STARTED") == 2
        assert "SESSION_CLOSED" in emitted
        assert "TASK_APPROVED" in emitted
        assert "TASK_REJECTED" not in emitted
        assert task.state == TaskState.COMPLETED
        # Approving the last task should trigger the goal-completion check
        mock_complete.assert_awaited_once_with(goal_id, room_id, db)


# ---------------------------------------------------------------------------
# Scenario 2: CM rejects
# ---------------------------------------------------------------------------


class TestCmRejects:
    @pytest.mark.asyncio
    async def test_cm_calls_close_session_with_reject_sentinel(self):
        """CM calls close_session with TASK_REJECTED in summary → TASK_REJECTED emitted."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id

        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        emitted: list[str] = []
        reject_tool_input = {"summary": f"Output is incomplete. {_REJECT_SENTINEL}"}

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=[
                (None, {"summary": "Agent work done."}, 100, 80, "claude-haiku-4-5-20251001", 100, 100),
                (None, reject_tool_input, 120, 90, "claude-haiku-4-5-20251001", 120, 120),
            ]),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory(emitted)),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        assert "TASK_REJECTED" in emitted
        assert "TASK_APPROVED" not in emitted
        assert task.state == TaskState.REJECTED


# ---------------------------------------------------------------------------
# Scenario 3: agent calls close_session first → CM gets notified → CM approves
# ---------------------------------------------------------------------------


class TestAgentClosesFirst:
    @pytest.mark.asyncio
    async def test_agent_close_then_cm_approves(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id

        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        emitted: list[str] = []
        agent_tool_input = {"summary": "All done.", "tasks_update": []}
        cm_tool_input = {"summary": "Looks great!", "plan_update": None}

        llm_calls = [
            # Agent calls close_session first
            (None, agent_tool_input, 100, 80, "claude-haiku-4-5-20251001", 100, 100),
            # CM approves
            (None, cm_tool_input, 120, 90, "claude-haiku-4-5-20251001", 120, 120),
        ]

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=llm_calls),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory(emitted)),
            patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        # Agent session should be closed at task_done, CM should also close
        assert "TASK_APPROVED" in emitted
        assert emitted.count("SESSION_CLOSED") >= 2


class TestStallIntervention:
    """loop_guard: a non-productive agent loop is handed to the ChatMaster."""

    @pytest.mark.asyncio
    async def test_stall_signal_triggers_cm_intervention_and_bumps_streak(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        task.stall_streak = 0
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        events: list[tuple[str, dict]] = []

        async def _capture_emit(room_id, event_type, **kwargs):
            events.append((event_type, kwargs.get("metadata") or {}))
            return MagicMock()

        model = "claude-haiku-4-5-20251001"

        async def _fake_llm(*args, **kwargs):
            calls = _fake_llm.calls
            _fake_llm.calls += 1
            if calls == 0:  # agent turn 1 — stall
                sig = kwargs.get("loop_signals")
                assert sig is not None, "agent call must receive loop_signals"
                sig["stall"] = {
                    "reason": "dup_call",
                    "tool_name": "grep",
                    "repeat_count": 4,
                    "summary": "40 tool calls, 3 distinct, 0 productive",
                }
                sig["productive_calls"] = 0
                return ("[AUTOMATIC STALL REPORT ...]", None, 50, 20, model, 50, 50)
            if calls == 1:  # CM turn 1 — text steer, not a verdict
                return ("Stop grepping. Read blog/templatetags/blog_extras.py directly.", None, 30, 10, model, 30, 30)
            if calls == 2:  # agent turn 2 — finishes
                return (None, {"summary": "Fixed per steer.", "tasks_update": []}, 40, 15, model, 40, 40)
            return (None, {"summary": "Approved.", "plan_update": None}, 20, 8, model, 20, 20)

        _fake_llm.calls = 0

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", side_effect=_fake_llm),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_capture_emit),
            patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        types = [e[0] for e in events]
        assert "AGENT_STALL_INTERVENTION" in types
        meta = next(m for t, m in events if t == "AGENT_STALL_INTERVENTION")
        assert meta["reason"] == "dup_call"
        assert meta["tool_name"] == "grep"
        assert meta["stall_streak"] == 1  # bumped when the intervention fired
        # The agent then finished the task — a clean close is progress, so the
        # streak is back to 0 by the end.
        assert task.stall_streak == 0
        assert "TASK_APPROVED" in types

    @pytest.mark.asyncio
    async def test_productive_turn_clears_existing_streak(self):
        room_id, goal_id, plan_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        task_id, agent_id, cm_agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        task.stall_streak = 3  # carried in from earlier dead-end restarts
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        model = "claude-haiku-4-5-20251001"

        async def _fake_llm(*args, **kwargs):
            n = _fake_llm.calls
            _fake_llm.calls += 1
            if n == 0:  # agent made file changes this turn, then asks a question
                sig = kwargs.get("loop_signals")
                sig["productive_calls"] = 2
                return ("Wrote the fix, one question about naming.", None, 50, 20, model, 50, 50)
            if n == 1:  # CM answers
                return ("Use snake_case.", None, 20, 8, model, 20, 20)
            if n == 2:  # agent closes
                return (None, {"summary": "Done.", "tasks_update": []}, 30, 10, model, 30, 30)
            return (None, {"summary": "Approved.", "plan_update": None}, 15, 5, model, 15, 15)

        _fake_llm.calls = 0
        emitted: list[str] = []

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", side_effect=_fake_llm),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory(emitted)),
            patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        assert task.stall_streak == 0
        assert "AGENT_STALL_INTERVENTION" not in emitted


# ---------------------------------------------------------------------------
# Scenario 4: agent token limit → CM also closed (token_limit)
# ---------------------------------------------------------------------------


class TestAgentTokenLimit:
    @pytest.mark.asyncio
    async def test_agent_token_limit_closes_cm_too(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id

        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        emitted: list[str] = []
        # Simulate budget exhaustion inside _call_llm_with_tools (new architecture:
        # budget check happens via context assembler inside the LLM function).
        from app.core.proxy_session import TokenBudgetLimitReached
        budget_exc = TokenBudgetLimitReached([], 0, 0, "claude-haiku-4-5-20251001")
        llm_mock = AsyncMock(side_effect=budget_exc)

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", llm_mock),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory(emitted)),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        llm_mock.assert_called()
        assert "SESSION_CLOSED" in emitted

    @pytest.mark.asyncio
    async def test_context_overflow_closes_with_context_reason_and_payload(self):
        """A real window failure is never presented as a depleted token budget."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(
            task_result, _claim_result(), _agent_result(), room_result,
        ))

        events: list[dict[str, Any]] = []

        async def capture_event(**kwargs: Any) -> MagicMock:
            events.append(kwargs)
            return MagicMock()

        from app.core.proxy_session import ContextWindowLimitReached
        overflow = ContextWindowLimitReached(
            [], 0, 0, "test-model", context_window_limit=100_000,
        )

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=overflow),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=capture_event),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        assert agent_session.close_reason == "context_limit"
        assert cm_session.close_reason == "context_limit"
        limit_event = next(event for event in events if event.get("event_type") == "TOOL_LIMIT_REACHED")
        assert limit_event["metadata"]["reason"] == "context_window"
        assert "token budget" not in limit_event["content"]


class TestRepeatedToolErrorInterruption:
    """internal design notes, part 4, stage 1: trigger_tool/repeat_count/
    safe_error must reach the persisted TOOL_LIMIT_REACHED event, not just
    exist as unused fields on the exception."""

    @pytest.mark.asyncio
    async def test_metadata_reaches_persisted_event(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(
            task_result, _claim_result(), _agent_result(), room_result,
        ))

        events: list[dict[str, Any]] = []

        async def capture_event(**kwargs: Any) -> MagicMock:
            events.append(kwargs)
            return MagicMock()

        from app.core.proxy_session import RepeatedToolErrorLimitReached
        # Same shape as the real incident this stage fixes: workspace_service
        # repeated an invalid-cwd error twice after a warning turn.
        repeated = RepeatedToolErrorLimitReached(
            [], 32, 0, "test-model",
            tool_calls_used=32, tool_calls_limit=200,
            trigger_tool="workspace_service",
            repeat_count=3,
            safe_error="Error: invalid cwd '/nonexistent' for django dev server",
        )

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=repeated),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=capture_event),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        limit_event = next(event for event in events if event.get("event_type") == "TOOL_LIMIT_REACHED")
        meta = limit_event["metadata"]
        assert meta["reason"] == "repeated_tool_error"
        assert meta["trigger_tool"] == "workspace_service"
        assert meta["repeat_count"] == 3
        assert meta["safe_error"] == "Error: invalid cwd '/nonexistent' for django dev server"
        assert meta["recommended_action"] == "resume_with_correction"
        # 32/200 must never be presented as "the limit was reached" — the
        # bug this stage fixes.
        assert "32" not in limit_event["content"]

    @pytest.mark.asyncio
    async def test_other_reasons_leave_trigger_tool_fields_none(self):
        """Only RepeatedToolErrorLimitReached sets these — no cross-talk from
        a plain ToolLimitReached (real tool_calls exhaustion)."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(
            task_result, _claim_result(), _agent_result(), room_result,
        ))

        events: list[dict[str, Any]] = []

        async def capture_event(**kwargs: Any) -> MagicMock:
            events.append(kwargs)
            return MagicMock()

        from app.core.proxy_session import ToolLimitReached
        exhausted = ToolLimitReached(
            [], 100, 0, "test-model", tool_calls_used=200, tool_calls_limit=200,
        )

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=exhausted),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=capture_event),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        limit_event = next(event for event in events if event.get("event_type") == "TOOL_LIMIT_REACHED")
        meta = limit_event["metadata"]
        assert meta["reason"] == "tool_calls"
        assert meta["trigger_tool"] is None
        assert meta["repeat_count"] is None
        assert meta["safe_error"] is None
        assert meta["recommended_action"] == "resume"


# ---------------------------------------------------------------------------
# Scenario 5: task has no assigned agent → skip
# ---------------------------------------------------------------------------


class TestNoAssignedAgent:
    @pytest.mark.asyncio
    async def test_skips_when_no_assigned_agent(self):
        room_id = uuid.uuid4()
        task_id = uuid.uuid4()

        task = _make_task()
        task.assigned_agent_id = None
        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        db.execute = AsyncMock(return_value=task_result)

        with patch("app.core.proxy_session.SessionRepository") as MockRepo:
            await handle_task_created(task_id, room_id, uuid.uuid4(), uuid.uuid4(), db)

        MockRepo.return_value.create.assert_not_called()


# ---------------------------------------------------------------------------
# Scenario 6: room has no master_agent_id → skip
# ---------------------------------------------------------------------------


class TestNoMasterAgent:
    @pytest.mark.asyncio
    async def test_skips_when_no_master_agent_id(self):
        room_id = uuid.uuid4()
        task_id = uuid.uuid4()

        task = _make_task()
        room = _make_room(room_id=room_id, has_coordinator=False)
        db = _make_db()

        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        with patch("app.core.proxy_session.SessionRepository") as MockRepo:
            await handle_task_created(task_id, room_id, uuid.uuid4(), uuid.uuid4(), db)

        MockRepo.return_value.create.assert_not_called()


# ---------------------------------------------------------------------------
# Scenario 7: LLM error on agent side → both sessions closed with reason=error
# ---------------------------------------------------------------------------


class TestAgentLlmError:
    @pytest.mark.asyncio
    async def test_llm_error_closes_both_with_error(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id

        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        emitted: list[str] = []

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock,
                  side_effect=RuntimeError("network error")),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory(emitted)),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        assert agent_session.close_reason == "error"
        assert cm_session.close_reason == "error"
        assert "SESSION_CLOSED" in emitted


class TestAgentLlmFailureIsPersisted:
    """internal design notes, part 4, stage 4: a provider failure that
    survives the retry budget must leave a persisted, Retry-able TASK_STALLED
    (reason provider_error/timeout) instead of only the ephemeral LLM_ERROR."""

    @staticmethod
    async def _run(exc: BaseException) -> AsyncMock:
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()
        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(
            task_result, _claim_result(), _agent_result(), room_result,
        ))

        block = AsyncMock()
        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=exc),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory([])),
            patch("app.core.proxy_session._block_proxy_for_runtime_failure", block),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()
            await handle_task_created(task_id, room_id, goal_id, uuid.uuid4(), db)
        return block

    @pytest.mark.asyncio
    async def test_http_status_failure_becomes_provider_error(self):
        import httpx

        request = httpx.Request("POST", "https://llm.example.com/v1/messages?key=SECRET123")
        exc = httpx.HTTPStatusError(
            "boom", request=request, response=httpx.Response(401, request=request)
        )
        block = await self._run(exc)

        block.assert_awaited_once()
        kwargs = block.await_args.kwargs
        assert kwargs["reason"] == "provider_error"
        meta = kwargs["metadata"]
        assert meta["http_status"] == 401
        assert meta["safe_error"] == "HTTP 401 Unauthorized"
        assert meta["recommended_action"] == "retry"
        assert meta["origin_side"] == "agent"
        assert "API key" in kwargs["content"]
        # The request URL (and the key in its query string) never reaches the chat.
        assert "SECRET123" not in kwargs["content"]
        assert "SECRET123" not in str(meta)

    @pytest.mark.asyncio
    async def test_attempt_timeout_becomes_timeout(self):
        from app.core.proxy_session import LLMAttemptTimeout

        block = await self._run(LLMAttemptTimeout(600))

        kwargs = block.await_args.kwargs
        assert kwargs["reason"] == "timeout"
        assert kwargs["metadata"]["recommended_action"] == "retry"
        assert kwargs["metadata"]["http_status"] is None


class TestClassifyLlmFailure:
    def test_transport_error_is_unreachable_provider(self):
        import httpx

        from app.core.proxy_session import _classify_llm_failure

        reason, safe, status = _classify_llm_failure(httpx.ConnectError("dns failure"))
        assert reason == "provider_error"
        assert safe == "the model provider is unreachable"
        assert status is None

    def test_httpx_timeout_is_timeout(self):
        import httpx

        from app.core.proxy_session import _classify_llm_failure

        reason, _safe, status = _classify_llm_failure(httpx.ReadTimeout("slow"))
        assert reason == "timeout"
        assert status is None

    def test_unknown_error_redacts_secrets_and_keeps_first_line(self):
        from app.core.proxy_session import _classify_llm_failure

        reason, safe, _status = _classify_llm_failure(
            RuntimeError("bad payload api_key=sk-live-abc123\nTraceback ...")
        )
        assert reason == "provider_error"
        assert safe.startswith("RuntimeError: bad payload")
        assert "sk-live-abc123" not in safe
        assert "Traceback" not in safe


# ---------------------------------------------------------------------------
# Scenario 8: max turns exceeded → force-closed as cancelled
# ---------------------------------------------------------------------------


class TestMaxTurnsExceeded:
    @pytest.mark.asyncio
    async def test_force_close_on_max_turns(self):
        """After _MAX_PROXY_TURNS text replies without close_session, loop forcibly stops."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)

        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id

        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _make_db()
        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        room_result = MagicMock()
        room_result.scalars.return_value.first.return_value = room
        db.execute = AsyncMock(side_effect=_db_results(task_result, _claim_result(), _agent_result(), room_result))

        emitted: list[str] = []
        # Both sides only ever return text — no close_session, no budget limit.
        text_reply = ("Some reply.", None, 50, 50, "claude-haiku-4-5-20251001", 50, 50)
        many_text_replies = [text_reply] * (_MAX_PROXY_TURNS * 2 + 5)

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            # The run_id/runtime_state fence is exercised in test_proxy_stale_loop.py;
            # these mocks predate it and would otherwise report the pair as superseded.
            patch("app.core.proxy_session._pair_closed_externally", new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock,
                  side_effect=many_text_replies),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=_fake_emit_factory(emitted)),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        # Both sides should be closed (either still open when loop ends or already closed during loop)
        assert "SESSION_CLOSED" in emitted
        assert "SESSION_STARTED" in emitted


# ---------------------------------------------------------------------------
# dispatch_task_event
# ---------------------------------------------------------------------------


class TestDispatchTaskEvent:
    def test_schedules_asyncio_task(self):
        with patch("app.core.proxy_session.asyncio.create_task") as mock_create:
            dispatch_task_event(
                task_id=uuid.uuid4(),
                room_id=uuid.uuid4(),
                goal_id=uuid.uuid4(),
                plan_id=uuid.uuid4(),
            )
        mock_create.assert_called_once()
        name = mock_create.call_args.kwargs.get("name", "")
        assert "proxy_session_" in name
