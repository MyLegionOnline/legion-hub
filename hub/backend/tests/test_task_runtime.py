"""Regression coverage for durable proxy-run ownership and pause acknowledgement."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.task_runtime import (
    RUNTIME_PAUSED,
    RUNTIME_RUNNING,
    RUNTIME_STOP_REQUESTED,
    fence_task_runtime,
    finalize_run,
    is_run_current,
    touch_run_activity,
)
from app.models.task import TaskState


def _result(*, first=None):
    result = MagicMock()
    result.first.return_value = first
    result.scalars.return_value.first.return_value = first
    return result


@pytest.mark.asyncio
async def test_run_ownership_rejects_stopped_or_replaced_generation():
    task_id = uuid.uuid4()
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _result(first=("run-current", RUNTIME_RUNNING)),
        _result(first=("run-current", RUNTIME_STOP_REQUESTED)),
        _result(first=("run-new", RUNTIME_RUNNING)),
    ])

    assert await is_run_current(task_id, "run-current", db) is True
    assert await is_run_current(task_id, "run-current", db) is False


@pytest.mark.asyncio
async def test_fence_task_runtime_writes_shared_stop_fields():
    task = SimpleNamespace(
        id=uuid.uuid4(),
        runtime_state=RUNTIME_RUNNING,
        stop_requested_at=None,
        stop_reason=None,
    )
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    assert await fence_task_runtime(task, "task_replaced", db) is True

    assert task.runtime_state == RUNTIME_STOP_REQUESTED
    assert task.stop_reason == "task_replaced"
    assert task.stop_requested_at is not None
    db.add.assert_called_once_with(task)
    db.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_pause_is_confirmed_only_when_owning_run_finishes():
    task = SimpleNamespace(
        id=uuid.uuid4(),
        room_id=uuid.uuid4(),
        title="pause me",
        run_id="run-current",
        runtime_state=RUNTIME_STOP_REQUESTED,
        state=TaskState.PAUSING,
        pause_reason="user",
        updated_at=datetime.now(timezone.utc),
    )
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_result(first=task))
    db.add = MagicMock()
    db.flush = AsyncMock()

    with patch("app.core.events.emit_event", new_callable=AsyncMock) as event:
        await finalize_run(task.id, "run-current", db)

    assert task.state == TaskState.PAUSED
    assert task.runtime_state == RUNTIME_PAUSED
    event.assert_awaited_once()


@pytest.mark.asyncio
async def test_pause_hands_dispatch_status_back_to_ready():
    """A settled pause must clear the stale DISPATCHED marker so Resume/Start
    can redispatch instead of hitting request_task_dispatch's already-active
    short-circuit (the wedge behind 'Resume does nothing')."""
    from app.models.task import DispatchStatus

    task = SimpleNamespace(
        id=uuid.uuid4(),
        room_id=uuid.uuid4(),
        title="pause me",
        run_id="run-current",
        runtime_state=RUNTIME_STOP_REQUESTED,
        state=TaskState.PAUSING,
        dispatch_status=DispatchStatus.DISPATCHED,
        pause_reason="user",
        updated_at=datetime.now(timezone.utc),
    )
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_result(first=task))
    db.add = MagicMock()
    db.flush = AsyncMock()

    with patch("app.core.events.emit_event", new_callable=AsyncMock):
        await finalize_run(task.id, "run-current", db)

    assert task.state == TaskState.PAUSED
    assert task.dispatch_status == DispatchStatus.READY


@pytest.mark.asyncio
async def test_finishing_old_generation_cannot_overwrite_a_retry():
    task = SimpleNamespace(
        id=uuid.uuid4(),
        room_id=uuid.uuid4(),
        title="retry me",
        run_id="run-new",
        runtime_state=RUNTIME_RUNNING,
        state=TaskState.IN_PROGRESS,
        updated_at=datetime.now(timezone.utc),
    )
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_result(first=task))
    db.add = MagicMock()
    db.flush = AsyncMock()

    await finalize_run(task.id, "run-old", db)

    assert task.run_id == "run-new"
    assert task.runtime_state == RUNTIME_RUNNING
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_touch_locks_task_before_session_and_commits_around_the_boundary():
    """Server load run 2026-09-28: session → task deadlocked against the
    workspace pause path, which flushes tasks and then closes sessions."""
    calls: list[str] = []
    db = AsyncMock()

    async def _execute(stmt, *args, **kwargs):
        text = str(stmt).lstrip().upper()
        if text.startswith("SELECT"):
            calls.append("select")
            return _result(first=("run-1", RUNTIME_RUNNING))
        calls.append("update " + text.split()[1].strip('"').lower())
        return MagicMock()

    async def _commit():
        calls.append("commit")

    db.execute = AsyncMock(side_effect=_execute)
    db.commit = AsyncMock(side_effect=_commit)

    assert await touch_run_activity(uuid.uuid4(), uuid.uuid4(), "run-1", db) is True
    assert calls == ["commit", "select", "update task", "update session", "commit"]


@pytest.mark.asyncio
async def test_touch_on_a_stopped_run_writes_nothing():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_result(first=("run-2", RUNTIME_RUNNING)))

    assert await touch_run_activity(uuid.uuid4(), uuid.uuid4(), "run-1", db) is False
    assert db.execute.await_count == 1


@pytest.mark.asyncio
async def test_pause_settles_even_when_the_loop_passes_a_stale_task():
    """The run loop hands finalize_run its own copy of the task, loaded when
    the run began. A pause written meanwhile by another session must still
    settle to PAUSED — deciding from the stale copy left the task PAUSING for
    good, and the room billed for 19.7 h."""
    from app.models.task import DispatchStatus

    stale = SimpleNamespace(
        id=uuid.uuid4(),
        room_id=uuid.uuid4(),
        title="pause me",
        run_id="run-current",
        runtime_state=RUNTIME_RUNNING,
        state=TaskState.IN_PROGRESS,
        dispatch_status=DispatchStatus.DISPATCHED,
        pause_reason="user",
        updated_at=datetime.now(timezone.utc),
    )
    fresh_row = (
        "run-current", TaskState.PAUSING, RUNTIME_STOP_REQUESTED, DispatchStatus.DISPATCHED,
    )
    update_result = MagicMock(rowcount=1)
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[_result(first=fresh_row), update_result])
    db.add = MagicMock()
    db.flush = AsyncMock()

    with patch("app.core.events.emit_event", new_callable=AsyncMock):
        await finalize_run(stale.id, "run-current", db, task=stale)

    update_values = db.execute.await_args_list[1].args[0].compile().params
    assert update_values["state"] == TaskState.PAUSED
    assert update_values["runtime_state"] == RUNTIME_PAUSED
    assert update_values["dispatch_status"] == DispatchStatus.READY
    assert stale.state == TaskState.PAUSED
    assert stale.runtime_state == RUNTIME_PAUSED


@pytest.mark.asyncio
async def test_stale_task_from_a_replaced_generation_writes_nothing():
    stale = SimpleNamespace(id=uuid.uuid4(), run_id="run-old", state=TaskState.IN_PROGRESS)
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_result(
        first=("run-new", TaskState.IN_PROGRESS, RUNTIME_RUNNING, None)
    ))
    db.add = MagicMock()

    await finalize_run(stale.id, "run-old", db, task=stale)

    assert db.execute.await_count == 1
    db.add.assert_not_called()


def test_wedged_pausing_task_is_not_work():
    from app.core.task_runtime import task_is_working_clause

    sql = str(task_is_working_clause().compile(compile_kwargs={"literal_binds": True}))
    assert "PAUSING" in sql
    assert "'running'" in sql and "'stop_requested'" in sql
    assert "'stopped'" not in sql
