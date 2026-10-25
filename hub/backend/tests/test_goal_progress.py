"""Unit tests for goal_progress.maybe_complete_goal (no DB)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.goal_progress import maybe_complete_goal
from app.models.goal import GoalStatus, RoomGoal
from app.models.task import TaskState

_GOAL_ID = uuid.uuid4()
_ROOM_ID = uuid.uuid4()


def _make_goal(status: GoalStatus = GoalStatus.ACTIVE) -> RoomGoal:
    return RoomGoal(id=_GOAL_ID, room_id=_ROOM_ID, title="G", status=status)


def _make_db(goal, task_states) -> AsyncMock:
    db = AsyncMock()

    goal_result = MagicMock()
    goal_result.scalars.return_value.first.return_value = goal
    # Rows are (state, checkpoint_kind); a bare TaskState is a regular task.
    states_result = MagicMock()
    states_result.all.return_value = [
        s if isinstance(s, tuple) else (s, None) for s in task_states
    ]

    db.execute = AsyncMock(side_effect=[goal_result, states_result])
    return db


class TestMaybeCompleteGoal:
    @pytest.mark.asyncio
    async def test_all_tasks_completed_marks_goal_completed(self):
        goal = _make_goal()
        db = _make_db(goal, [TaskState.COMPLETED, TaskState.COMPLETED])

        with patch("app.core.goal_progress.emit_event", new_callable=AsyncMock) as mock_emit:
            done = await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db)

        assert done is True
        assert goal.status == GoalStatus.COMPLETED
        assert goal.completed_at is not None
        mock_emit.assert_awaited_once()
        assert mock_emit.call_args.kwargs["event_type"] == "GOAL_COMPLETED"

    @pytest.mark.asyncio
    async def test_mixed_terminal_states_complete(self):
        goal = _make_goal()
        db = _make_db(goal, [TaskState.COMPLETED, TaskState.CANCELLED])

        with patch("app.core.goal_progress.emit_event", new_callable=AsyncMock):
            assert await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db) is True

    @pytest.mark.asyncio
    async def test_approved_checkpoint_counts_as_done(self):
        """A checkpoint left in APPROVED (pre-fix data) must not wedge its goal."""
        goal = _make_goal()
        db = _make_db(goal, [TaskState.COMPLETED, (TaskState.APPROVED, "confirm")])

        with (
            patch("app.core.goal_progress.emit_event", new_callable=AsyncMock),
            patch(
                "app.core.room_git_flow.maybe_open_goal_review",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            assert await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db) is True
        assert goal.status == GoalStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_approved_regular_task_keeps_goal_active(self):
        goal = _make_goal()
        db = _make_db(goal, [TaskState.COMPLETED, TaskState.APPROVED])

        with patch("app.core.goal_progress.emit_event", new_callable=AsyncMock):
            assert await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db) is False
        assert goal.status == GoalStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_open_task_keeps_goal_active(self):
        goal = _make_goal()
        db = _make_db(goal, [TaskState.COMPLETED, TaskState.IN_PROGRESS])

        with patch("app.core.goal_progress.emit_event", new_callable=AsyncMock) as mock_emit:
            assert await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db) is False

        assert goal.status == GoalStatus.ACTIVE
        mock_emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_cancelled_is_not_completion(self):
        goal = _make_goal()
        db = _make_db(goal, [TaskState.CANCELLED, TaskState.CANCELLED])

        with patch("app.core.goal_progress.emit_event", new_callable=AsyncMock):
            assert await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db) is False
        assert goal.status == GoalStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_goal_without_tasks_stays_active(self):
        goal = _make_goal()
        db = _make_db(goal, [])

        with patch("app.core.goal_progress.emit_event", new_callable=AsyncMock):
            assert await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db) is False

    @pytest.mark.asyncio
    async def test_already_completed_goal_is_noop(self):
        goal = _make_goal(status=GoalStatus.COMPLETED)
        db = _make_db(goal, [TaskState.COMPLETED])

        with patch("app.core.goal_progress.emit_event", new_callable=AsyncMock) as mock_emit:
            assert await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db) is False
        mock_emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_missing_goal_is_noop(self):
        db = _make_db(None, [])

        with patch("app.core.goal_progress.emit_event", new_callable=AsyncMock):
            assert await maybe_complete_goal(_GOAL_ID, _ROOM_ID, db) is False
