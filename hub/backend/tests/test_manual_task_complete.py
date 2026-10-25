"""Manual "mark as done" — the human-only override in change_task_state.

TASK_TRANSITIONS routes completion through WAITING_REVIEW so that an agent can
never declare its own work finished. A person can: a task may sit BLOCKED (or
stuck IN_PROGRESS) while the work is in fact done — the case that motivated
this, where a reviewing agent simply could not read the files the task had
produced. Before the override the only exits were Cancel (which marks the goal
abandoned and fails dependants) or closing the parent goal by hand and leaving
the task hanging.

Covered here:
- the override is accepted from the states the board offers it from, and is
  refused everywhere else (the transition table still rules);
- it fences a *live* runtime, so a still-executing agent cannot keep writing
  into a session whose task is already terminal — while an idle task is not
  left wearing a "Stopping…" fence nobody will ever clear;
- it unblocks dependants and closes the goal when it was the last open task;
- the agent-facing table itself is unchanged, so `task_update` still cannot
  reach COMPLETED from BLOCKED.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.tasks import _MANUAL_COMPLETE_STATES, change_task_state
from app.models.task import DispatchStatus, TASK_TRANSITIONS, TaskState
from app.schemas.task import TaskStateChange
from tests.test_task_dispatch_contract import _endpoint_db, _make_task


def _patched_endpoint(*, sessions_closed: int = 0, was_running: bool = False):
    """Patch every side effect change_task_state performs around the DB row."""
    return (
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock),
        patch("app.api.v1.tasks.fence_task_runtime", new_callable=AsyncMock,
              return_value=was_running),
        patch("app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock,
              return_value=sessions_closed),
        patch("app.api.v1.tasks.try_unblock_tasks", new_callable=AsyncMock, return_value=[]),
        patch("app.api.v1.tasks.publish_unblocked_tasks", new_callable=AsyncMock),
        patch("app.core.goal_progress.maybe_complete_goal", new_callable=AsyncMock),
        # internal design notes Stage 4 — COMPLETED now runs the promote_policy
        # hook; a no-op under the default `manual` policy but it still touches
        # the DB, so it is stubbed alongside the other side effects.
        patch("app.core.room_git_flow.maybe_open_promotion", new_callable=AsyncMock),
    )


class TestManualComplete:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", sorted(_MANUAL_COMPLETE_STATES, key=lambda s: s.value))
    async def test_accepted_from_every_offered_state(self, state):
        """The board/side panel offer Mark Done exactly for these states —
        the endpoint must accept all of them, none of which TASK_TRANSITIONS
        allows to reach COMPLETED on its own."""
        assert TaskState.COMPLETED not in TASK_TRANSITIONS[state]
        task = _make_task(state=state, dispatch_status=DispatchStatus.DISPATCHED,
                          run_id="run-abc", runtime_state="running")
        db = _endpoint_db(task)
        emit, fence, close, unblock, publish, goal, promo = _patched_endpoint()

        with emit, fence as mock_fence, close as mock_close, unblock, publish, goal, promo:
            out = await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.COMPLETED),
                db, MagicMock(),
            )

        assert task.state == TaskState.COMPLETED
        assert out.state == TaskState.COMPLETED
        # Runtime is fenced and sessions are closed as "done", not as a failure.
        mock_fence.assert_awaited_once()
        assert mock_fence.await_args.kwargs["cancel_local"] is True
        assert mock_close.await_args.args[1] == "task_done"

    @pytest.mark.asyncio
    async def test_refused_from_a_state_the_board_never_offers_it_from(self):
        """The override widens the transition table by exactly one edge set —
        a DRAFT task still cannot skip straight to done."""
        task = _make_task(state=TaskState.DRAFT, dispatch_status=DispatchStatus.READY,
                          runtime_state="idle")
        db = _endpoint_db(task)
        emit, fence, close, unblock, publish, goal, promo = _patched_endpoint()

        with emit, fence, close, unblock, publish, goal, promo:
            with pytest.raises(HTTPException) as exc:
                await change_task_state(
                    task.room_id, task.id, TaskStateChange(state=TaskState.COMPLETED),
                    db, MagicMock(),
                )

        assert exc.value.status_code == 422
        assert task.state == TaskState.DRAFT

    @pytest.mark.asyncio
    async def test_unblocks_dependants_and_closes_the_goal(self):
        """The reason the user had to close the goal by hand: this endpoint
        never ran the goal check that every agent-driven completion runs."""
        task = _make_task(state=TaskState.BLOCKED, dispatch_status=DispatchStatus.READY,
                          runtime_state="idle", goal_id=_make_task().id)
        db = _endpoint_db(task)
        emit, fence, close, unblock, publish, goal, promo = _patched_endpoint()

        with emit, fence, close, unblock as mock_unblock, publish, goal as mock_goal, promo:
            await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.COMPLETED),
                db, MagicMock(),
            )

        mock_unblock.assert_awaited_once()
        mock_goal.assert_awaited_once()
        assert mock_goal.await_args.args[0] == task.goal_id

    @pytest.mark.asyncio
    async def test_idle_task_records_no_stop_event(self):
        """A blocked task has nothing left running — emitting a "stop
        requested" event for it would be pure noise in the room feed."""
        task = _make_task(state=TaskState.BLOCKED, dispatch_status=DispatchStatus.READY,
                          runtime_state="idle")
        db = _endpoint_db(task)
        emit, fence, close, unblock, publish, goal, promo = _patched_endpoint()

        with emit as mock_emit, fence as mock_fence, close, unblock, publish, goal, promo:
            await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.COMPLETED),
                db, MagicMock(),
            )

        emitted = [c.kwargs["event_type"] for c in mock_emit.await_args_list]
        assert emitted == ["TASK_STATE_CHANGED"]
        assert mock_emit.await_args_list[0].kwargs["metadata"]["manual_complete"] is True
        # …and no stop fence is written on a task that was not executing.
        mock_fence.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_running_task_records_the_stop(self):
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED,
                          run_id="run-abc", runtime_state="running")
        db = _endpoint_db(task)
        emit, fence, close, unblock, publish, goal, promo = _patched_endpoint(
            sessions_closed=2, was_running=True,
        )

        with emit as mock_emit, fence, close, unblock, publish, goal, promo:
            await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.COMPLETED),
                db, MagicMock(),
            )

        emitted = [c.kwargs["event_type"] for c in mock_emit.await_args_list]
        assert emitted == ["TASK_STATE_CHANGED", "TASK_RUNTIME_STOP_REQUESTED"]

    @pytest.mark.asyncio
    async def test_send_for_review_stops_the_running_agent(self):
        """"Send for Review" on an executing task used to repaint the card only:
        the agent kept thinking and ChatMaster kept waiting on it. It must stop
        the run like Mark Done does and save loose work to the task branch."""
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED,
                          run_id="run-abc", runtime_state="running")
        db = _endpoint_db(task)
        emit, fence, close, unblock, publish, goal, promo = _patched_endpoint(
            sessions_closed=2, was_running=True,
        )

        with emit as mock_emit, fence as mock_fence, close as mock_close, unblock, publish, \
                goal, promo, patch("app.core.room_git_flow.commit_working_tree",
                                   new_callable=AsyncMock) as mock_commit:
            await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.WAITING_REVIEW),
                db, MagicMock(),
            )

        assert task.state == TaskState.WAITING_REVIEW
        mock_fence.assert_awaited_once()
        assert mock_fence.await_args.kwargs["cancel_local"] is True
        assert mock_close.await_args.args[1] == "task_done"
        assert mock_close.await_args.kwargs["include_dialogue"] is True
        mock_commit.assert_awaited_once()
        emitted = [c.kwargs["event_type"] for c in mock_emit.await_args_list]
        assert emitted == ["TASK_STATE_CHANGED", "TASK_RUNTIME_STOP_REQUESTED"]

    def test_agent_tool_still_cannot_self_complete(self):
        """The override lives in the REST endpoint only. `task_update`
        validates against TASK_TRANSITIONS, which must keep review as the one
        road to COMPLETED for agents."""
        for state, allowed in TASK_TRANSITIONS.items():
            if state is TaskState.WAITING_REVIEW:
                assert TaskState.COMPLETED in allowed
            else:
                assert TaskState.COMPLETED not in allowed
