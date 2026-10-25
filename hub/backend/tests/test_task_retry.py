"""Unit tests for chat upgrade stage 5: task retry endpoint.

Covers:
- close_open_task_sessions: closes both sides (incl. peer found via the pair
  link) with the given reason and emits SESSION_CLOSED per side; noop when
  nothing is open
- retry_task: BLOCKED task → sessions closed, state ASSIGNED, dispatch READY,
  retry_count incremented, persistent TASK_RETRIED emitted, redispatch called
- retry_task: non-retryable state → 422
- retry_task: retry limit reached → 422; ?force=true bypasses the limit
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.v1.tasks import MAX_TASK_RETRIES, change_task_state, retry_task
from app.core.watchdog import close_open_task_sessions
from app.models.task import DispatchStatus, TaskState
from app.schemas.task import TaskRetry, TaskStateChange

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(all_rows=None, first_row=None, row=None) -> MagicMock:
    m = MagicMock()
    m.scalars.return_value.all.return_value = all_rows or []
    m.scalars.return_value.first.return_value = first_row
    m.first.return_value = row
    return m


def _make_db(results: list) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock(side_effect=results)
    return db


def _make_proxy_session(task_id=None, peer_id=None) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_agent_proxy"
    s.side = "agent"
    s.task_id = task_id or uuid.uuid4()
    s.peer_session_id = peer_id
    s.closed_at = None
    s.close_reason = None
    return s


def _make_task(state=TaskState.BLOCKED, retry_count=0) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        room_id=uuid.uuid4(),
        goal_id=uuid.uuid4(),
        title="Test task",
        description="",
        state=state,
        dispatch_status=DispatchStatus.DISPATCHED,
        retry_count=retry_count,
        assigned_agent_id=uuid.uuid4(),
        created_by_agent_id=None,
        created_at=NOW,
        updated_at=NOW,
        run_id=None,
        pause_reason=None,
        paused_at=None,
    )


# ---------------------------------------------------------------------------
# close_open_task_sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_open_task_sessions_closes_both_sides_and_peer():
    task_id = uuid.uuid4()
    peer = _make_proxy_session()          # CM side, not matched by task_id filter
    agent_side = _make_proxy_session(task_id=task_id, peer_id=peer.id)

    db = _make_db([
        _result(all_rows=[agent_side]),   # task_id filter hits only the agent side
        _result(first_row=peer),          # peer looked up via pair link
    ])

    with patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event:
        closed = await close_open_task_sessions(task_id, "retry", db, now=NOW)

    assert closed == 2
    for s in (agent_side, peer):
        assert s.closed_at == NOW
        assert s.close_reason == "retry"

    event_types = [c.kwargs["event_type"] for c in mock_event.await_args_list]
    assert event_types.count("SESSION_CLOSED") == 2
    reasons = {c.kwargs["metadata"]["close_reason"] for c in mock_event.await_args_list}
    assert reasons == {"retry"}


@pytest.mark.asyncio
async def test_close_open_task_sessions_noop_when_nothing_open():
    db = _make_db([_result(all_rows=[])])

    with patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event:
        closed = await close_open_task_sessions(uuid.uuid4(), "retry", db, now=NOW)

    assert closed == 0
    mock_event.assert_not_awaited()
    db.add.assert_not_called()


# ---------------------------------------------------------------------------
# retry_task endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_rejects_before_changing_state_when_dispatch_is_unavailable():
    """The Start button must not recreate an in-progress task with no session."""
    from app.core.proxy_session import TaskDispatchPreflightError

    task = _make_task(TaskState.ASSIGNED)
    db = _make_db([
        _result(first_row=object()),  # _verify_membership
        _result(first_row=task),      # _get_task
    ])

    with patch(
        "app.api.v1.tasks.prepare_task_dispatch",
        new_callable=AsyncMock,
        side_effect=TaskDispatchPreflightError("no_chat_master", "coordinator not configured"),
    ):
        with pytest.raises(HTTPException) as exc:
            await change_task_state(
                task.room_id,
                task.id,
                TaskStateChange(state=TaskState.IN_PROGRESS),
                db,
                MagicMock(),
            )

    assert exc.value.status_code == 422
    assert "no_chat_master" in exc.value.detail
    assert task.state == TaskState.ASSIGNED
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_dispatches_direct_task_without_goal_or_plan():
    task = _make_task(TaskState.ASSIGNED)
    task.goal_id = None
    db = _endpoint_db(task)

    with (
        patch("app.api.v1.tasks.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=_dispatch_context(task)),
        patch("app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock,
              return_value=0),
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock),
        patch("app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
              return_value="started") as mock_dispatch,
    ):
        out = await change_task_state(
            task.room_id,
            task.id,
            TaskStateChange(state=TaskState.IN_PROGRESS),
            db,
            MagicMock(),
        )

    assert task.state == TaskState.IN_PROGRESS
    assert out.goal_id is None
    mock_dispatch.assert_awaited_once_with(
        task, task.room_id, db, resume=False, rework_feedback=None, source="manual_start",
    )


def _endpoint_db(task, plan_id=None):
    """db.execute sequence for the endpoint happy path."""
    agent = MagicMock()
    agent.nickname = "worker"
    return _make_db([
        _result(first_row=object()),          # _verify_membership
        _result(first_row=task),              # _get_task
        _result(first_row=agent),             # _enrich_task: agent
        _result(all_rows=[]),                 # _enrich_task: sessions
        _result(row=(1,)),                    # _enrich_task: assignee still in the room
        _result(row=None),                    # _compute_task_runtime: last dispatch failure
        _result(row=None),                    # _compute_task_runtime: last interruption
    ])


def _dispatch_context(task, plan_id=None):
    from app.core.proxy_session import TaskDispatchContext

    return TaskDispatchContext(
        task.goal_id, plan_id, MagicMock(), uuid.uuid4(), False
    )


@pytest.mark.asyncio
async def test_retry_blocked_task_resets_state_and_redispatches():
    task = _make_task(TaskState.BLOCKED, retry_count=1)
    plan_id = uuid.uuid4()
    db = _endpoint_db(task, plan_id=plan_id)

    with (
        patch("app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock, return_value=2) as mock_close,
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock) as mock_event,
        patch("app.api.v1.tasks.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=_dispatch_context(task, plan_id)),
        patch("app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
              return_value="started") as mock_dispatch,
    ):
        out = await retry_task(task.room_id, task.id, TaskRetry(), False, db, MagicMock())

    mock_close.assert_awaited_once_with(task.id, "retry", db)
    assert task.state == TaskState.ASSIGNED
    assert task.dispatch_status == DispatchStatus.READY
    assert task.retry_count == 2
    assert out.retry_count == 2

    mock_event.assert_awaited_once()
    kwargs = mock_event.await_args.kwargs
    assert kwargs["event_type"] == "TASK_RETRIED"
    assert kwargs["metadata"]["prev_state"] == "blocked"
    assert kwargs["metadata"]["retry_count"] == 2
    assert kwargs["metadata"]["sessions_closed"] == 2

    mock_dispatch.assert_awaited_once_with(
        task, task.room_id, db, resume=False, source="retry",
    )


@pytest.mark.asyncio
async def test_retry_assigned_task_after_proxy_failure_redispatches():
    """A failed proxy can leave the task assigned after closing both sessions."""
    task = _make_task(TaskState.ASSIGNED, retry_count=1)
    db = _endpoint_db(task, plan_id=uuid.uuid4())

    with (
        patch("app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock),
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock),
        patch("app.api.v1.tasks.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=_dispatch_context(task, task.goal_id)),
        patch("app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
              return_value="started"),
    ):
        out = await retry_task(task.room_id, task.id, TaskRetry(), False, db, MagicMock())

    assert task.state == TaskState.ASSIGNED
    assert out.retry_count == 2


@pytest.mark.asyncio
async def test_retry_pausing_task_restarts_and_clears_pause_metadata():
    """A task stuck while stopping remains recoverable from either UI surface."""
    task = _make_task(TaskState.PAUSING)
    task.pause_reason = "workspace_switch"
    task.paused_at = NOW
    db = _endpoint_db(task, plan_id=uuid.uuid4())

    with (
        patch("app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock),
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock) as mock_event,
        patch("app.api.v1.tasks.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=_dispatch_context(task, task.goal_id)),
        patch("app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
              return_value="started") as mock_dispatch,
    ):
        out = await retry_task(task.room_id, task.id, TaskRetry(), False, db, MagicMock())

    assert task.state == TaskState.ASSIGNED
    assert task.pause_reason is None
    assert task.paused_at is None
    assert out.retry_count == 1
    assert mock_event.await_args.kwargs["metadata"]["prev_state"] == "pausing"
    mock_dispatch.assert_awaited_once_with(
        task, task.room_id, db, resume=False, source="retry",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [TaskState.COMPLETED, TaskState.DRAFT, TaskState.CANCELLED])
async def test_retry_refused_for_non_retryable_states(state):
    task = _make_task(state)
    db = _make_db([
        _result(first_row=object()),
        _result(first_row=task),
    ])

    with pytest.raises(HTTPException) as exc:
        await retry_task(task.room_id, task.id, TaskRetry(), False, db, MagicMock())
    assert exc.value.status_code == 422
    assert task.state == state


@pytest.mark.asyncio
async def test_retry_limit_reached_refused_without_force():
    task = _make_task(TaskState.BLOCKED, retry_count=MAX_TASK_RETRIES)
    db = _make_db([
        _result(first_row=object()),
        _result(first_row=task),
    ])

    with pytest.raises(HTTPException) as exc:
        await retry_task(task.room_id, task.id, TaskRetry(), False, db, MagicMock())
    assert exc.value.status_code == 422
    assert "force" in exc.value.detail
    assert task.retry_count == MAX_TASK_RETRIES


@pytest.mark.asyncio
async def test_retry_limit_bypassed_with_force():
    task = _make_task(TaskState.REJECTED, retry_count=MAX_TASK_RETRIES)
    db = _endpoint_db(task, plan_id=uuid.uuid4())

    with (
        patch("app.api.v1.tasks.close_open_task_sessions", new_callable=AsyncMock, return_value=0),
        patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock),
        patch("app.api.v1.tasks.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=_dispatch_context(task, task.goal_id)),
        patch("app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
              return_value="started"),
    ):
        out = await retry_task(task.room_id, task.id, TaskRetry(), True, db, MagicMock())

    assert task.state == TaskState.ASSIGNED
    assert out.retry_count == MAX_TASK_RETRIES + 1


# ---------------------------------------------------------------------------
# _run_proxy_bg crash safety net
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_proxy_bg_crash_closes_sessions_and_blocks_task():
    """Unhandled crash in handle_task_created → rollback, then the safety net
    closes the task's open sessions (reason='error'), blocks the task and
    commits — nothing is left hanging for the watchdog to miss."""
    from app.core import proxy_session as ps

    task_id, room_id, goal_id, plan_id = (uuid.uuid4() for _ in range(4))
    db = _make_db([])

    factory = MagicMock()
    factory.return_value.__aenter__ = AsyncMock(return_value=db)
    factory.return_value.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(ps, "sessionmaker", return_value=factory),
        patch.object(ps, "handle_task_created", new_callable=AsyncMock,
                     side_effect=RuntimeError("boom")),
        patch("app.core.watchdog.close_open_task_sessions",
              new_callable=AsyncMock, return_value=2) as mock_close,
        patch("app.core.watchdog.block_stalled_task",
              new_callable=AsyncMock) as mock_block,
    ):
        await ps._run_proxy_bg(task_id, room_id, goal_id, plan_id)

    db.rollback.assert_awaited()
    mock_close.assert_awaited_once_with(task_id, "error", db)
    mock_block.assert_awaited_once()
    assert mock_block.await_args.args[0] == task_id
    assert mock_block.await_args.args[1] == "proxy_error"
    db.commit.assert_awaited_once()
