"""One dialogue, one kind of work — and a continued task the user can stop.

Two incidents from 2026-09-18, reported together:

1. While still gathering requirements, ChatMaster created quick goal-less tasks
   with ``task_create`` and dispatched agents on them, then proposed a plan from
   the same dialogue. The accepted goal planned work that was already running
   from another base and the task branches diverged.
2. One of those tasks was a duplicate that had been handed off, so its card sat
   in CONTINUED — a state with no outgoing transition and no buttons — while the
   successor kept the agents working on it.
"""
from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.dialogue_scope import (
    _SETTLED_STATES,
    describe_quick_tasks,
    quick_tasks_block_plan_message,
)
from app.models.task import (
    TASK_ORIGIN_CM_QUICK,
    TASK_TRANSITIONS,
    TaskState,
)


def _quick_task(title: str, state: TaskState = TaskState.IN_PROGRESS) -> MagicMock:
    task = MagicMock()
    task.id = uuid.uuid4()
    task.title = title
    task.state = state
    task.origin = TASK_ORIGIN_CM_QUICK
    return task


# ---------------------------------------------------------------------------
# 1. A dialogue that already dispatched quick tasks forms no goal
# ---------------------------------------------------------------------------


class TestQuickTasksBlockThePlan:
    def test_only_unfinished_quick_tasks_hold_a_dialogue_back(self):
        # A finished quick task cannot diverge from anything, so it must not
        # keep blocking the goal forever.
        assert TaskState.COMPLETED in _SETTLED_STATES
        assert TaskState.CANCELLED in _SETTLED_STATES
        assert TaskState.IN_PROGRESS not in _SETTLED_STATES
        assert TaskState.WAITING_REVIEW not in _SETTLED_STATES

    def test_the_refusal_names_the_running_tasks(self):
        message = quick_tasks_block_plan_message(
            [_quick_task("Fix the compilation of internal/server")]
        )
        assert "Fix the compilation of internal/server" in message
        assert "a goal is not formed from this same dialogue" in message

    def test_the_listing_carries_state_per_task(self):
        listing = describe_quick_tasks(
            [_quick_task("A"), _quick_task("B", TaskState.ASSIGNED)]
        )
        assert listing.splitlines() == ["• “A” (in_progress)", "• “B” (assigned)"]

    @pytest.mark.asyncio
    async def test_the_plan_button_refuses_while_quick_tasks_run(self):
        from app.api.v1 import chat as chat_api

        session = MagicMock()
        session.id = uuid.uuid4()
        session.kind = "user_cm_p2"
        session.context_snapshot = None
        repo = MagicMock()
        repo.find_open_user_session = AsyncMock(return_value=session)

        with (
            patch("app.repositories.session.SessionRepository", return_value=repo),
            patch.object(chat_api, "turn_in_progress", return_value=False),
            patch(
                "app.core.dialogue_scope.quick_tasks_for_session",
                new_callable=AsyncMock,
                return_value=[_quick_task("Fix the compilation of api_kanban_test.go")],
            ),
        ):
            state = await chat_api._plan_request_state(
                uuid.uuid4(), MagicMock(), AsyncMock()
            )

        assert (state["available"], state["reason"]) == (False, "quick_tasks")

    @pytest.mark.asyncio
    async def test_a_proposed_plan_becomes_a_plain_refusal(self):
        """The propose_plan tool call never reaches the confirmation card."""
        import app.core.user_sessions as us
        from tests.test_plan_request_button import (
            _make_db,
            _make_session,
            _plan_tool_result,
            _run,
        )

        session = _make_session()
        db = _make_db()
        call_llm = AsyncMock(return_value=_plan_tool_result())
        quick = [_quick_task("Fix the rune literal in web/static_test.go")]

        with (
            patch.object(
                us,
                "quick_tasks_for_session",
                new_callable=AsyncMock,
                return_value=quick,
            ),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            still_open = await _run(session, db, call_llm, force_plan=True)

        # The dialogue stays open, but no plan card is pending and the user is
        # told why: the tasks already running are named in the reply.
        assert still_open is True
        assert "pending_plan" not in session.context_snapshot
        last_reply = session.context_snapshot["messages"][-1]
        assert last_reply["role"] == "assistant"
        assert "static_test.go" in last_reply["content"]

    @pytest.mark.asyncio
    async def test_a_plan_written_as_text_is_refused_too(self):
        """The XML fallback is the same door — it must carry the same lock."""
        import app.core.user_sessions as us
        from app.core.user_sessions import _LLMTurnResult
        from tests.test_plan_request_button import _make_db, _make_session, _run

        session = _make_session()
        db = _make_db()
        xml = (
            "<propose_plan><goal_title>Kanban</goal_title>"
            "<goal_description>Fix the test compilation.</goal_description>"
            "<plan_summary>CI</plan_summary></propose_plan>"
        )
        call_llm = AsyncMock(
            return_value=_LLMTurnResult(
                text=xml, tool_input=None, tokens_in=10, tokens_out=5, model="m"
            )
        )
        quick = [_quick_task("Fix the compilation of api_kanban_test.go")]

        with (
            patch.object(
                us,
                "quick_tasks_for_session",
                new_callable=AsyncMock,
                return_value=quick,
            ),
            patch("app.core.user_sessions.manager") as mock_manager,
        ):
            mock_manager.broadcast = AsyncMock()
            # force_plan keeps the turn on the single-call path (the workspace
            # tool loop is mocked out wholesale in this harness).
            await _run(session, db, call_llm, force_plan=True)

        assert "pending_plan" not in session.context_snapshot
        assert "api_kanban_test.go" in session.context_snapshot["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# 2. A continued task the user can actually stop
# ---------------------------------------------------------------------------


class TestContinuedTaskIsStoppable:
    def test_continued_can_be_cancelled(self):
        assert TASK_TRANSITIONS[TaskState.CONTINUED] == [TaskState.CANCELLED]

    def test_continued_can_be_marked_done_by_a_person(self):
        from app.api.v1.tasks import _MANUAL_COMPLETE_STATES

        assert TaskState.CONTINUED in _MANUAL_COMPLETE_STATES

    def test_completed_and_cancelled_stay_terminal(self):
        assert TASK_TRANSITIONS[TaskState.COMPLETED] == []
        assert TASK_TRANSITIONS[TaskState.CANCELLED] == []

    @pytest.mark.asyncio
    async def test_the_whole_continuation_chain_is_walked(self):
        """Cancelling the head must reach the successor of the successor."""
        from app.api.v1.tasks import _continuation_chain

        head_id = uuid.uuid4()
        child = _quick_task("child")
        child.continuation_of_task_id = head_id
        grandchild = _quick_task("grandchild")
        grandchild.continuation_of_task_id = child.id

        batches: list[list[Any]] = [[child], [grandchild], []]

        def _next_batch(*_args: Any, **_kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.scalars.return_value.all.return_value = batches.pop(0)
            return result

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=_next_batch)

        chain = await _continuation_chain(head_id, db)

        assert [task.title for task in chain] == ["child", "grandchild"]

    @pytest.mark.asyncio
    async def test_a_cycle_cannot_loop_forever(self):
        from app.api.v1.tasks import _continuation_chain

        head_id = uuid.uuid4()
        looping = _quick_task("looping")
        looping.continuation_of_task_id = head_id

        def _always_same(*_args: Any, **_kwargs: Any) -> MagicMock:
            result = MagicMock()
            result.scalars.return_value.all.return_value = [looping]
            return result

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=_always_same)

        chain = await _continuation_chain(head_id, db)

        assert [task.title for task in chain] == ["looping"]
