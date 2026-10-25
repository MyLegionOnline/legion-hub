"""Pause / Resume through PATCH /tasks/{id}/state.

The Resume button issues state=in_progress (not a dedicated /resume call). Two
regressions this covers:

* A workspace/user pause leaves dispatch_status=DISPATCHED. request_task_dispatch
  then reports "already active" and Resume silently does nothing — the task is
  wedged IN_PROGRESS with no running loop. Resume from a paused state must do
  the explicit close+reset+redispatch that /retry does, but resume=True so the
  agent continues from the saved transcript.
* There was no clean Pause: in_progress → pausing now fences the runtime and
  closes the sessions with a resumable reason.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.tasks import change_task_state
from app.models.task import DispatchStatus, TaskState
from app.schemas.task import TaskStateChange
from tests.test_task_dispatch_contract import _endpoint_db, _make_task


def _patched(*, was_running: bool = True, sessions_closed: int = 2):
    return (
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock),
        patch("app.api.v1.tasks.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=MagicMock()),
        patch("app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock,
              return_value=sessions_closed),
        patch("app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
              return_value="started"),
        patch("app.api.v1.tasks.request_task_stop", new_callable=AsyncMock,
              return_value=was_running),
    )


class TestResumeFromPaused:
    @pytest.mark.asyncio
    async def test_resume_does_explicit_restart_with_resume_true(self):
        task = _make_task(state=TaskState.PAUSED, dispatch_status=DispatchStatus.DISPATCHED,
                          run_id="run-old", runtime_state="paused", pause_reason="image_switch",
                          paused_at=None, stop_requested_at=None, stop_reason=None)
        db = _endpoint_db(task)
        emit, prep, close, dispatch, stop = _patched()

        with emit, prep, close as mock_close, dispatch as mock_dispatch, stop:
            await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.IN_PROGRESS),
                db, MagicMock(),
            )

        mock_close.assert_awaited_once()
        assert mock_close.await_args.args[1] == "retry"
        assert task.dispatch_status == DispatchStatus.READY
        assert task.run_id is None
        assert task.pause_reason is None
        _, kwargs = mock_dispatch.call_args
        assert kwargs["resume"] is True
        assert kwargs["source"] == "manual_start"

    @pytest.mark.asyncio
    async def test_wedged_in_progress_is_unwedged_and_resumed(self):
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED,
                          run_id="run-dead", runtime_state="paused")
        db = _endpoint_db(task)
        # the extra execute the unwedge branch runs after _get_task: the
        # open-session lookup (returns none -> nothing left running).
        seq = list(db.execute.side_effect)
        db.execute.side_effect = seq[:2] + [_no_open_session()] + seq[2:]
        emit, prep, close, dispatch, stop = _patched()

        with emit, prep, close as mock_close, dispatch as mock_dispatch, stop:
            await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.IN_PROGRESS),
                db, MagicMock(),
            )

        mock_close.assert_awaited_once()
        assert task.dispatch_status == DispatchStatus.READY
        assert task.run_id is None
        _, kwargs = mock_dispatch.call_args
        assert kwargs["resume"] is True
        assert kwargs["source"] == "manual_start_unwedge"


class TestPauseRunningTask:
    @pytest.mark.asyncio
    async def test_pause_fences_runtime_and_closes_sessions(self):
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED,
                          run_id="run-live", runtime_state="running", pause_reason=None,
                          paused_at=None)
        db = _endpoint_db(task)
        emit, prep, close, dispatch, stop = _patched(was_running=True)

        with emit as mock_emit, prep, close as mock_close, dispatch, stop as mock_stop:
            await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.PAUSING),
                db, MagicMock(),
            )

        mock_stop.assert_awaited_once()
        mock_close.assert_awaited_once()
        assert mock_close.await_args.args[1] == "workspace_paused"
        assert task.state == TaskState.PAUSING
        assert task.pause_reason == "user"
        emitted = [c.kwargs["event_type"] for c in mock_emit.await_args_list]
        assert "TASK_RUNTIME_STOP_REQUESTED" in emitted

    @pytest.mark.asyncio
    async def test_pause_of_idle_task_settles_straight_to_paused(self):
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED,
                          run_id="run-x", runtime_state="stopped", pause_reason=None,
                          paused_at=None)
        db = _endpoint_db(task)
        emit, prep, close, dispatch, stop = _patched(was_running=False, sessions_closed=0)

        with emit, prep, close, dispatch, stop:
            await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.PAUSING),
                db, MagicMock(),
            )

        assert task.state == TaskState.PAUSED
        assert task.runtime_state == "paused"
        assert task.dispatch_status == DispatchStatus.READY


def _no_open_session() -> MagicMock:
    r = MagicMock()
    r.first.return_value = None
    return r
