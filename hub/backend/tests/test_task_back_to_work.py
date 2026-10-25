"""Back to Work — returning a task from review to work.

`TASK_TRANSITIONS` allows WAITING_REVIEW → IN_PROGRESS, and both surfaces
(the Board and the task panel) show the button. But the endpoint ran the pre-flight
check `prepare_task_dispatch` against the *current* state of the task, and
WAITING_REVIEW is in `_NON_DISPATCHABLE_STATES` — that is, a transition
whose only purpose is to leave that state refused itself
with a 422 `invalid_task_state`. The second half of the bug: even if it passed the check,
a plain Start would not have launched anything — after work submitted for review
`dispatch_status` stays DISPATCHED, and `request_task_dispatch` would have returned
"already_active", leaving the task IN_PROGRESS without a session.

This checks both: the check goes by the target state, and the return itself
is an explicit restart (like Resume), passing the agent its previous
conversation and a note that the work was not accepted.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.tasks import _REVIEW_REOPEN_FEEDBACK, change_task_state
from app.core.proxy_session import (
    TaskDispatchContext,
    TaskDispatchPreflightError,
    prepare_task_dispatch,
)
from app.models.task import DispatchStatus, TASK_TRANSITIONS, TaskState
from app.schemas.task import TaskStateChange
from tests.test_task_dispatch_contract import _endpoint_db, _make_task


def _patched_endpoint():
    """Side effects around the task row — everything except what is under test."""
    return (
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock),
        patch("app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock,
              return_value=1),
        patch("app.api.v1.tasks.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=TaskDispatchContext(None, None, MagicMock(), None, False)),
    )


class TestPreflightTargetState:
    """`prepare_task_dispatch(target_state=...)` — check the state the task
    is moving into, not the one it is leaving."""

    @staticmethod
    def _db_with_media_agent() -> AsyncMock:
        # a media agent is the shortest exit from prepare_task_dispatch:
        # ChatMaster is not needed for it, one db.execute is enough.
        agent = MagicMock()
        agent.llm_config = {"media": {"capability": "image_generation"}}
        result = MagicMock()
        result.scalars.return_value.first.return_value = agent
        db = AsyncMock()
        db.execute = AsyncMock(return_value=result)
        return db

    @pytest.mark.asyncio
    async def test_waiting_review_is_refused_without_target_state(self):
        task = _make_task(state=TaskState.WAITING_REVIEW)
        with pytest.raises(TaskDispatchPreflightError) as exc:
            await prepare_task_dispatch(task, self._db_with_media_agent())
        assert exc.value.code == "invalid_task_state"

    @pytest.mark.asyncio
    async def test_waiting_review_passes_when_moving_to_in_progress(self):
        task = _make_task(state=TaskState.WAITING_REVIEW)
        ctx = await prepare_task_dispatch(
            task, self._db_with_media_agent(),
            require_ready=False, target_state=TaskState.IN_PROGRESS,
        )
        assert ctx.is_media_task is True

    @pytest.mark.asyncio
    async def test_non_dispatchable_target_is_still_refused(self):
        """The target state does not weaken the check — it only shifts it."""
        task = _make_task(state=TaskState.ASSIGNED)
        with pytest.raises(TaskDispatchPreflightError) as exc:
            await prepare_task_dispatch(
                task, self._db_with_media_agent(), target_state=TaskState.COMPLETED,
            )
        assert exc.value.code == "invalid_task_state"


class TestBackToWork:
    @pytest.mark.asyncio
    async def test_review_to_in_progress_restarts_the_agent(self):
        """The task goes to work, the submitted run is closed, and the agent
        is brought up again on its own conversation with a note about the unaccepted
        work — rather than "already active" on a stale DISPATCHED."""
        assert TaskState.IN_PROGRESS in TASK_TRANSITIONS[TaskState.WAITING_REVIEW]
        task = _make_task(
            state=TaskState.WAITING_REVIEW,
            dispatch_status=DispatchStatus.DISPATCHED,
            run_id="run-review",
            runtime_state="stopped",
            stop_requested_at=object(),
            stop_reason="task_done",
        )
        db = _endpoint_db(task)
        emit, close_sessions, _preflight = _patched_endpoint()

        with emit, close_sessions as mock_close, _preflight, patch(
            "app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
            return_value="started",
        ) as mock_dispatch:
            out = await change_task_state(
                task.room_id, task.id,
                TaskStateChange(state=TaskState.IN_PROGRESS), db, MagicMock(),
            )

        assert out.state == TaskState.IN_PROGRESS
        # an explicit restart: the old session is closed, the dispatch request is reset
        mock_close.assert_awaited_once_with(task.id, "retry", db)
        assert task.dispatch_status == DispatchStatus.READY
        assert task.run_id is None
        assert task.runtime_state == "idle"
        assert task.stop_requested_at is None
        assert task.stop_reason is None
        # and the agent respawns on its previous conversation, knowing the work was not accepted
        kwargs = mock_dispatch.await_args.kwargs
        assert kwargs["resume"] is True
        assert kwargs["rework_feedback"] == _REVIEW_REOPEN_FEEDBACK
        assert kwargs["source"] == "manual_rework"

    @pytest.mark.asyncio
    async def test_preflight_target_state_is_the_one_being_entered(self):
        """Regression for the 422 itself: the endpoint must ask about IN_PROGRESS."""
        task = _make_task(state=TaskState.WAITING_REVIEW,
                          dispatch_status=DispatchStatus.DISPATCHED, run_id="run-review")
        db = _endpoint_db(task)
        emit, close_sessions, preflight = _patched_endpoint()

        with emit, close_sessions, preflight as mock_preflight, patch(
            "app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
            return_value="started",
        ):
            await change_task_state(
                task.room_id, task.id,
                TaskStateChange(state=TaskState.IN_PROGRESS), db, MagicMock(),
            )

        assert mock_preflight.await_args.kwargs["target_state"] == TaskState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_ordinary_start_is_not_treated_as_rework(self):
        """A plain Start from ASSIGNED must neither close sessions nor
        slip review feedback to the agent."""
        task = _make_task(state=TaskState.ASSIGNED, dispatch_status=DispatchStatus.READY)
        db = _endpoint_db(task)
        emit, close_sessions, preflight = _patched_endpoint()

        with emit, close_sessions as mock_close, preflight, patch(
            "app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
            return_value="started",
        ) as mock_dispatch:
            await change_task_state(
                task.room_id, task.id,
                TaskStateChange(state=TaskState.IN_PROGRESS), db, MagicMock(),
            )

        mock_close.assert_not_awaited()
        kwargs = mock_dispatch.await_args.kwargs
        assert kwargs["resume"] is False
        assert kwargs["rework_feedback"] is None
        assert kwargs["source"] == "manual_start"
