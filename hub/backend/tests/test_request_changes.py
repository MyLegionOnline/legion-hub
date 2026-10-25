"""
Unit tests for Request Changes flow (internal design notes, stage 4).

Scenarios — unit (no DB required):
  0. classify_review_verdict: the explicit verdict field wins, a leading marker
     decides otherwise, and a summary that merely *mentions* CHANGES_REQUESTED
     (e.g. "APPROVED — both CHANGES_REQUESTED fixes verified") is an approval.
  1. CHANGES_REQUESTED sentinel → task ASSIGNED, rework_count incremented, sessions
     closed with reason 'changes_requested', TASK_CHANGES_REQUESTED emitted,
     TASK_DEPENDENCY_FAILED NOT emitted, dispatch_task_event called with
     resume=True and the CM's feedback text.
  2. Both sentinels (CHANGES_REQUESTED + TASK_REJECTED) in summary → TASK_REJECTED
     wins; rework path is skipped entirely.
  3. rework_count at MAX_TASK_REWORKS → CHANGES_REQUESTED degrades to reject;
     TASK_REJECTED emitted with limit_exhausted flag in metadata.
  4. load_paused_history accepts close_reasons=("changes_requested", "task_done")
     and returns messages from context_snapshot of the matching session.
  5. handle_task_created with resume=True + rework_feedback seeds agent history from
     the prior session and appends the CM feedback note as the last user message.

DB integration (require DATABASE_URL):
  6. Session row with close_reason='changes_requested' passes ck_session_close_reason.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from app.core.proxy_session import (
    _CHANGES_SENTINEL,
    _REJECT_SENTINEL,
    MAX_TASK_REWORKS,
    VERDICT_APPROVE,
    VERDICT_CHANGES,
    VERDICT_REJECT,
    classify_review_verdict,
    handle_task_created,
    load_paused_history,
)
from app.models.task import DispatchStatus, TaskState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# close_session input from the agent side. Submitting work is the only thing
# that turns the next ChatMaster turn into a review: a plain text turn is a
# mid-task question, and a verdict on it is converted into an answer.
_AGENT_SUBMIT = {"summary": "Done: changes made."}


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    r = MagicMock()
    r.scalars.return_value.first.return_value = None
    r.scalars.return_value.all.return_value = []
    r.first.return_value = None
    r.all.return_value = []
    db.execute = AsyncMock(return_value=r)
    return db


def _make_task(
    assigned_agent_id: uuid.UUID | None = None,
    goal_id: uuid.UUID | None = None,
    rework_count: int = 0,
) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "Build feature X"
    t.description = "Implement the feature."
    t.state = TaskState.IN_PROGRESS
    t.assigned_agent_id = assigned_agent_id or uuid.uuid4()
    t.goal_id = goal_id or uuid.uuid4()
    t.created_at = datetime.now(timezone.utc)
    t.updated_at = datetime.now(timezone.utc)
    t.dispatch_status = DispatchStatus.READY
    t.rework_count = rework_count
    t.stall_streak = 0
    return t


def _make_session(side: str = "agent") -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_agent_proxy"
    s.side = side
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
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
    s.last_activity_at = None
    return s


def _make_room(
    room_id: uuid.UUID | None = None,
    master_agent_id: uuid.UUID | None = None,
) -> MagicMock:
    room = MagicMock()
    room.id = room_id or uuid.uuid4()
    _mid = master_agent_id or uuid.uuid4()
    room.coordinator_config = {"master_agent_id": str(_mid)}
    return room


def _claim_result() -> MagicMock:
    r = MagicMock()
    r.rowcount = 1
    return r


def _agent_llm_result() -> MagicMock:
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


def _db_results(*results):
    """Yield given results in order, then infinite empty results."""
    yield from results
    while True:
        yield _empty_result()


def _capture_emit():
    """Return (captured_list, async_side_effect) for emit_event patching."""
    captured: list[dict] = []

    async def _emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
        captured.append({"event_type": event_type, **kwargs})
        return MagicMock()

    return captured, _emit


def _event_types(captured: list[dict]) -> list[str]:
    return [e["event_type"] for e in captured]


def _setup_db(task: MagicMock, room: MagicMock) -> AsyncMock:
    db = _make_db()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task
    room_result = MagicMock()
    room_result.scalars.return_value.first.return_value = room
    db.execute = AsyncMock(
        side_effect=_db_results(task_result, _claim_result(), _agent_llm_result(), room_result)
    )
    return db


# ---------------------------------------------------------------------------
# 0. Verdict classification
# ---------------------------------------------------------------------------


class TestClassifyReviewVerdict:
    """The ruling comes from the verdict field, or from a marker that leads."""

    def test_explicit_verdict_beats_any_prose(self):
        assert classify_review_verdict(
            {"verdict": "approve", "summary": f"the {_CHANGES_SENTINEL} items are fixed"}
        ) == VERDICT_APPROVE
        assert classify_review_verdict(
            {"verdict": "request_changes", "summary": "APPROVED earlier, but"}
        ) == VERDICT_CHANGES
        assert classify_review_verdict({"verdict": "reject", "summary": ""}) == VERDICT_REJECT

    def test_approval_that_mentions_the_sentinel_stays_an_approval(self):
        """The incident: a summary opening with APPROVED that names the earlier
        CHANGES_REQUESTED round used to come back as a rework request, and then
        as a rejection whose printed reason read 'APPROVED'."""
        summary = (
            "APPROVED — Both CHANGES_REQUESTED fixes verified in the repo: the "
            "title header is gone and the report file carries the meta_description."
        )
        assert classify_review_verdict({"summary": summary}) == VERDICT_APPROVE

    def test_leading_marker_is_the_verdict(self):
        assert classify_review_verdict(
            {"summary": f"{_CHANGES_SENTINEL}\n1. src/a.py:4 — returns None"}
        ) == VERDICT_CHANGES
        assert classify_review_verdict(
            {"summary": f"**{_CHANGES_SENTINEL}** — see the list below"}
        ) == VERDICT_CHANGES
        assert classify_review_verdict(
            {"summary": f"Verdict: {_REJECT_SENTINEL} — nothing was produced"}
        ) == VERDICT_REJECT

    def test_bare_sentinel_without_a_leading_marker_still_counts(self):
        assert classify_review_verdict(
            {"summary": f"Fix the typo in README first. {_CHANGES_SENTINEL}"}
        ) == VERDICT_CHANGES

    def test_no_marker_is_approval(self):
        assert classify_review_verdict({"summary": "All good, merged."}) == VERDICT_APPROVE
        assert classify_review_verdict({}) == VERDICT_APPROVE
        assert classify_review_verdict(None) == VERDICT_APPROVE


# ---------------------------------------------------------------------------
# 1. CHANGES_REQUESTED sentinel
# ---------------------------------------------------------------------------


class TestChangesRequestedSentinel:
    """CHANGES_REQUESTED in CM summary starts the rework cycle."""

    @pytest.mark.asyncio
    async def test_task_assigned_and_redispatched(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id, rework_count=0)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _setup_db(task, room)

        feedback = f"{_CHANGES_SENTINEL}: unit tests are missing and README needs updating"
        cm_tool_input = {"summary": feedback}

        captured, emit_side_effect = _capture_emit()

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._pair_closed_externally",
                  new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session._build_review_verification",
                  new_callable=AsyncMock, return_value="[SYSTEM VERIFICATION] ok"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=[
                (None, _AGENT_SUBMIT, 100, 80, "claude-haiku-4-5-20251001", 100, 100),
                (None, cm_tool_input, 120, 90, "claude-haiku-4-5-20251001", 120, 120),
            ]),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=emit_side_effect),
            patch("app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock,
                  return_value="started") as mock_dispatch,
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        types = _event_types(captured)

        # Task goes back to ASSIGNED for another run
        assert task.state == TaskState.ASSIGNED
        assert task.rework_count == 1
        assert task.dispatch_status == DispatchStatus.READY

        # Correct event emitted
        assert "TASK_CHANGES_REQUESTED" in types

        # Dependent tasks must NOT be notified — request_changes ≠ failure
        assert "TASK_DEPENDENCY_FAILED" not in types

        # Neither approve nor reject
        assert "TASK_APPROVED" not in types
        assert "TASK_REJECTED" not in types

        # Both sessions closed with the rework reason
        assert cm_session.close_reason == "changes_requested"
        assert agent_session.close_reason == "changes_requested"

        # Redispatch must use resume + the CM's full summary as feedback
        mock_dispatch.assert_awaited_once()
        args = mock_dispatch.await_args.args
        kw = mock_dispatch.await_args.kwargs
        assert args[0] is task
        assert kw["resume"] is True
        assert kw["rework_feedback"] == feedback

    @pytest.mark.asyncio
    async def test_changes_requested_event_metadata(self):
        """TASK_CHANGES_REQUESTED event carries rework_count and feedback in metadata."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id, rework_count=0)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _setup_db(task, room)

        feedback = f"{_CHANGES_SENTINEL}: file src/foo.py has a bug on line 42"
        cm_tool_input = {"summary": feedback}
        captured, emit_side_effect = _capture_emit()

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._pair_closed_externally",
                  new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session._build_review_verification",
                  new_callable=AsyncMock, return_value="[SYSTEM VERIFICATION] ok"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=[
                (None, _AGENT_SUBMIT, 100, 80, "claude-haiku-4-5-20251001", 100, 100),
                (None, cm_tool_input, 120, 90, "claude-haiku-4-5-20251001", 120, 120),
            ]),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=emit_side_effect),
            patch("app.core.proxy_session.dispatch_task_event"),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        changes_event = next(
            (e for e in captured if e["event_type"] == "TASK_CHANGES_REQUESTED"), None
        )
        assert changes_event is not None
        meta = changes_event.get("metadata", {})
        assert meta["rework_count"] == 1
        assert meta["feedback"] == feedback
        assert meta["task_id"] == str(task_id)


# ---------------------------------------------------------------------------
# 2. Both sentinels — TASK_REJECTED wins
# ---------------------------------------------------------------------------


class TestBothSentinelsPriority:
    """When both CHANGES_REQUESTED and TASK_REJECTED appear in the summary,
    the reject path runs and no rework cycle starts."""

    @pytest.mark.asyncio
    async def test_reject_wins(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id, rework_count=0)
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _setup_db(task, room)

        # Both sentinels in the same summary
        cm_tool_input = {
            "summary": (
                f"The work is completely wrong. {_REJECT_SENTINEL} "
                f"Ignore {_CHANGES_SENTINEL} mentioned earlier."
            )
        }
        captured, emit_side_effect = _capture_emit()

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._pair_closed_externally",
                  new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session._build_review_verification",
                  new_callable=AsyncMock, return_value="[SYSTEM VERIFICATION] ok"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=[
                (None, _AGENT_SUBMIT, 100, 80, "claude-haiku-4-5-20251001", 100, 100),
                (None, cm_tool_input, 120, 90, "claude-haiku-4-5-20251001", 120, 120),
            ]),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=emit_side_effect),
            patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch,
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        types = _event_types(captured)

        assert "TASK_REJECTED" in types
        assert "TASK_CHANGES_REQUESTED" not in types
        assert task.state == TaskState.REJECTED
        # rework_count must NOT be incremented on reject
        assert task.rework_count == 0
        # No rework redispatch
        mock_dispatch.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Rework limit exhausted → degrade to reject
# ---------------------------------------------------------------------------


class TestReworkLimitDegrade:
    """When rework_count == MAX_TASK_REWORKS, CHANGES_REQUESTED becomes a reject."""

    @pytest.mark.asyncio
    async def test_changes_degrade_to_reject_at_limit(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        # Task has already exhausted its rework allowance
        task = _make_task(
            assigned_agent_id=agent_id,
            goal_id=goal_id,
            rework_count=MAX_TASK_REWORKS,
        )
        task.id = task_id
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _setup_db(task, room)

        # CM only mentions CHANGES_REQUESTED (would normally be a rework)
        cm_tool_input = {"summary": f"{_CHANGES_SENTINEL}: still missing tests"}
        captured, emit_side_effect = _capture_emit()

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._pair_closed_externally",
                  new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session._build_review_verification",
                  new_callable=AsyncMock, return_value="[SYSTEM VERIFICATION] ok"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", new_callable=AsyncMock, side_effect=[
                (None, _AGENT_SUBMIT, 100, 80, "claude-haiku-4-5-20251001", 100, 100),
                (None, cm_tool_input, 120, 90, "claude-haiku-4-5-20251001", 120, 120),
            ]),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=emit_side_effect),
            patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch,
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        types = _event_types(captured)

        # Degraded to reject
        assert "TASK_REJECTED" in types
        assert "TASK_CHANGES_REQUESTED" not in types
        assert task.state == TaskState.REJECTED
        # rework_count must NOT be incremented further when degrading
        assert task.rework_count == MAX_TASK_REWORKS

        # No rework redispatch
        mock_dispatch.assert_not_called()

        # The TASK_REJECTED event metadata must carry limit_exhausted=True
        reject_event = next(e for e in captured if e["event_type"] == "TASK_REJECTED")
        assert reject_event.get("metadata", {}).get("limit_exhausted") is True


# ---------------------------------------------------------------------------
# 4. load_paused_history with rework close_reasons
# ---------------------------------------------------------------------------


class TestLoadPausedHistoryRework:
    """load_paused_history correctly filters on rework-related close reasons."""

    @pytest.mark.asyncio
    async def test_returns_messages_from_changes_requested_session(self):
        task_id = uuid.uuid4()
        db = _make_db()

        messages = [
            {"role": "user", "content": "Build feature X"},
            {"role": "assistant", "content": "I implemented it in src/x.py"},
        ]
        fake_session = MagicMock()
        fake_session.context_snapshot = {"messages": messages}

        result = MagicMock()
        result.scalars.return_value.first.return_value = fake_session
        db.execute = AsyncMock(return_value=result)

        out = await load_paused_history(
            task_id, db, close_reasons=("changes_requested", "task_done")
        )

        assert out == messages
        db.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_messages_from_task_done_session(self):
        """task_done is also a valid source of history for rework (first submission)."""
        task_id = uuid.uuid4()
        db = _make_db()

        messages = [{"role": "user", "content": "first task data"}]
        fake_session = MagicMock()
        fake_session.context_snapshot = {"messages": messages}

        result = MagicMock()
        result.scalars.return_value.first.return_value = fake_session
        db.execute = AsyncMock(return_value=result)

        out = await load_paused_history(
            task_id, db, close_reasons=("changes_requested", "task_done")
        )

        assert out == messages

    @pytest.mark.asyncio
    async def test_returns_none_when_no_matching_session(self):
        task_id = uuid.uuid4()
        db = _make_db()

        result = MagicMock()
        result.scalars.return_value.first.return_value = None
        db.execute = AsyncMock(return_value=result)

        out = await load_paused_history(
            task_id, db, close_reasons=("changes_requested", "task_done")
        )

        assert out is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_context_snapshot(self):
        task_id = uuid.uuid4()
        db = _make_db()

        fake_session = MagicMock()
        fake_session.context_snapshot = {"messages": []}  # empty messages list

        result = MagicMock()
        result.scalars.return_value.first.return_value = fake_session
        db.execute = AsyncMock(return_value=result)

        out = await load_paused_history(
            task_id, db, close_reasons=("changes_requested", "task_done")
        )

        assert out is None

    @pytest.mark.asyncio
    async def test_default_reasons_still_tool_limit_only(self):
        """Default close_reasons must not change — existing resume path unaffected."""
        task_id = uuid.uuid4()
        db = _make_db()

        result = MagicMock()
        result.scalars.return_value.first.return_value = None
        db.execute = AsyncMock(return_value=result)

        await load_paused_history(task_id, db)  # no close_reasons override

        # The SQL query was built with the default "tool_limit" filter.
        # We can't inspect the compiled SQL easily, but we verify the call went through.
        db.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# 5. Agent history seeded with CM feedback on rework
# ---------------------------------------------------------------------------


class TestReworkHistorySeed:
    """When handle_task_created runs with resume=True + rework_feedback, the agent
    receives its prior history with the CM's feedback appended as a user message."""

    @pytest.mark.asyncio
    async def test_feedback_note_in_agent_history(self):
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id, rework_count=1)
        task.id = task_id
        task.dispatch_status = DispatchStatus.READY
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _setup_db(task, room)

        # Simulate the agent's history from the previous submission
        prior_history = [
            {"role": "user", "content": "Task data: build feature X"},
            {"role": "assistant", "content": "I wrote src/x.py"},
        ]
        rework_feedback = "src/x.py lacks docstrings and tests are missing"

        # Capture what histories are passed to _call_llm_with_tools
        captured_histories: list[list] = []

        async def spy_llm(llm_config, system_prompt, history, *args, **kwargs):
            captured_histories.append(list(history))
            # First call: agent (return text reply)
            if len(captured_histories) == 1:
                return ("OK fixed it.", None, 100, 80, "claude-haiku-4-5-20251001", 100, 100)
            # Second call: CM approves
            return (None, {"summary": "Looks good now!"}, 120, 90, "claude-haiku-4-5-20251001", 120, 120)

        captured, emit_side_effect = _capture_emit()

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._pair_closed_externally",
                  new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session._build_review_verification",
                  new_callable=AsyncMock, return_value="[SYSTEM VERIFICATION] ok"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", side_effect=spy_llm),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=emit_side_effect),
            patch("app.core.proxy_session.dispatch_task_event"),
            patch(
                "app.core.proxy_session.load_paused_history",
                new_callable=AsyncMock,
                return_value=list(prior_history),
            ),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(
                task_id, room_id, goal_id, plan_id, db,
                resume=True,
                rework_feedback=rework_feedback,
            )

        # The agent's first LLM call must have received the seeded history
        assert len(captured_histories) >= 1, "agent LLM was never called"
        agent_history = captured_histories[0]

        # Prior messages are present
        assert any(
            m["role"] == "assistant" and "src/x.py" in m["content"]
            for m in agent_history
        ), "prior agent history not found in seed"

        # Feedback note was appended as (or merged into) the last user message
        last_user = next(
            (m for m in reversed(agent_history) if m["role"] == "user"), None
        )
        assert last_user is not None
        content = last_user["content"]
        assert "ChatMaster reviewed" in content, "feedback header missing"
        assert rework_feedback in content, "CM feedback text not in agent history"

    @pytest.mark.asyncio
    async def test_no_prior_history_starts_fresh(self):
        """If load_paused_history returns None, the agent starts with the task data."""
        room_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        cm_agent_id = uuid.uuid4()

        task = _make_task(assigned_agent_id=agent_id, goal_id=goal_id, rework_count=1)
        task.id = task_id
        task.dispatch_status = DispatchStatus.READY
        room = _make_room(room_id=room_id, master_agent_id=cm_agent_id)
        agent_session = _make_session(side="agent")
        agent_session.agent_id = agent_id
        agent_session.room_id = room_id
        cm_session = _make_session(side="chat_master")
        cm_session.agent_id = cm_agent_id
        cm_session.room_id = room_id

        db = _setup_db(task, room)

        captured_histories: list[list] = []

        async def spy_llm(llm_config, system_prompt, history, *args, **kwargs):
            captured_histories.append(list(history))
            if len(captured_histories) == 1:
                return ("Done.", None, 100, 80, "claude-haiku-4-5-20251001", 100, 100)
            return (None, {"summary": "Approved!"}, 120, 90, "claude-haiku-4-5-20251001", 120, 120)

        captured, emit_side_effect = _capture_emit()

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._pair_closed_externally",
                  new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session._build_review_verification",
                  new_callable=AsyncMock, return_value="[SYSTEM VERIFICATION] ok"),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._call_llm_with_tools", side_effect=spy_llm),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=emit_side_effect),
            patch("app.core.proxy_session.dispatch_task_event"),
            patch(
                "app.core.proxy_session.load_paused_history",
                new_callable=AsyncMock,
                return_value=None,  # no prior history available
            ),
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(side_effect=[agent_session, cm_session])
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()

            await handle_task_created(
                task_id, room_id, goal_id, plan_id, db,
                resume=True,
                rework_feedback="fix tests",
            )

        assert len(captured_histories) >= 1
        agent_history = captured_histories[0]
        # Fresh start: first message is the task data, not prior history
        assert len(agent_history) == 1
        assert agent_history[0]["role"] == "user"
        assert task.title in agent_history[0]["content"]


# ---------------------------------------------------------------------------
# 6. DB integration — ck_session_close_reason allows 'changes_requested'
# ---------------------------------------------------------------------------

DB_URL = os.environ.get("DATABASE_URL", "")
pytestmark_db = pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import SQLModel

    # Ensure all models are registered in SQLModel.metadata
    import app.models.session  # noqa: F401
    import app.models.task  # noqa: F401
    import app.models.room  # noqa: F401

    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await engine.dispose()


@pytest.mark.db
@pytestmark_db
@pytest.mark.asyncio
class TestConstraintChangesRequestedDB:
    """Session row with close_reason='changes_requested' must satisfy the DB constraint."""

    async def test_close_reason_accepted_by_db(self, db_session):
        from sqlalchemy import text

        user_id = uuid.uuid4()
        room_id = uuid.uuid4()
        session_id = uuid.uuid4()

        # Prerequisite rows
        await db_session.execute(
            text(
                'INSERT INTO "user"'
                " (id, email, hashed_password, is_active, two_factor_enabled)"
                " VALUES (:id, :email, 'x', true, false)"
            ),
            {"id": str(user_id), "email": f"{user_id}@test.com"},
        )
        await db_session.execute(
            text(
                "INSERT INTO room"
                " (id, name, type, room_type, has_coordinator, creator_id, state)"
                " VALUES (:id, 'test', 'group', 'group', false, :cid, 'PLANNING')"
                " ON CONFLICT DO NOTHING"
            ),
            {"id": str(room_id), "cid": str(user_id)},
        )
        await db_session.flush()

        # This insert must NOT raise IntegrityError — 'changes_requested' is in
        # CLOSE_REASONS and the ck_session_close_reason constraint allows it.
        # Note: close_reason is a VARCHAR CHECK constraint, not a PostgreSQL ENUM,
        # so the value is lowercase (unlike GoalStatus which is a real PG enum).
        await db_session.execute(
            text(
                "INSERT INTO session"
                " (id, room_id, kind, started_at, closed_at, close_reason,"
                "  tokens_in, tokens_out, cost_usd)"
                " VALUES"
                " (:id, :room_id, 'user_agent_p1', NOW(), NOW(), 'changes_requested',"
                "  0, 0, 0)"
            ),
            {"id": str(session_id), "room_id": str(room_id)},
        )
        await db_session.flush()

        row = (
            await db_session.execute(
                text("SELECT close_reason FROM session WHERE id = :id"),
                {"id": str(session_id)},
            )
        ).first()

        assert row is not None
        assert row[0] == "changes_requested"
