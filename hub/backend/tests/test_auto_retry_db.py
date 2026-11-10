"""Integration coverage for durable auto-retry scheduling and row ownership."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.api.v1.rooms import update_room
from app.api.v1.tasks import (
    cancel_task_auto_retry,
    list_tasks,
    resume_task,
    retry_task,
    run_task_auto_retry_now,
)
from app.core import task_restart, work_gates
from app.core.auto_retry import run_due_auto_retries
from app.core.loop_guard import STALL_RESTART_THRESHOLD
from app.core.room_close import close_room
from app.core.watchdog import block_stalled_task
from app.models import (
    Agent,
    CmAgentProxySession,
    Message,
    Room,
    RoomAgentLink,
    RoomUserLink,
    Task,
    TaskState,
    User,
    Workspace,
    WorkspaceUsage,
)
from app.schemas.room import RoomUpdate
from app.schemas.task import TaskRetry

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def sessions():
    url = os.environ.get("DATABASE_URL")
    if not url or not (make_url(url).database or "").endswith("_test"):
        pytest.skip("DATABASE_URL must point to a PostgreSQL database ending in _test")

    schema = f"auto_retry_{uuid4().hex}"
    admin = create_async_engine(url)
    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def _interrupted_task(sessions, monkeypatch, *, reason="provider_error", history=None):
    monkeypatch.setattr("app.core.watchdog.maybe_refresh_room_memory", AsyncMock())
    user = User(email="owner@example.test", hashed_password="test")
    room = Room(name="Recovery test", creator_id=user.id)
    workspace = Workspace(room_id=room.id)
    usage = WorkspaceUsage(workspace_id=workspace.id, mem_limit_mb=1024)
    agent = Agent(owner_id=user.id, nickname="worker", agent_uid="worker", token="test")
    task = Task(
        room_id=room.id, title="Recover the task", state=TaskState.IN_PROGRESS,
        assigned_agent_id=agent.id,
    )
    async with sessions() as db:
        for row in (
            user, room, workspace, usage, agent,
            RoomAgentLink(room_id=room.id, agent_id=agent.id),
            RoomUserLink(room_id=room.id, user_id=user.id), task,
        ):
            db.add(row)
            await db.flush()
        if history is not None:
            db.add(
                CmAgentProxySession(
                    room_id=room.id, task_id=task.id, agent_id=agent.id, side="agent",
                    closed_at=datetime.now(timezone.utc), close_reason="context_limit",
                    context_snapshot={"messages": history},
                )
            )
        await block_stalled_task(
            task.id, reason, "The task was interrupted.", db,
            extra_meta={"http_status": 503} if reason == "provider_error" else None,
        )
        assert task.state == TaskState.BLOCKED
        assert task.auto_retry_at is not None
        assert task.auto_retry_reason == reason
    return user, task, usage.id


async def test_two_workers_cannot_claim_the_same_attempt(sessions, monkeypatch):
    _, task, _ = await _interrupted_task(sessions, monkeypatch)
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def hold_claim(*_args, **_kwargs):
        entered.set()
        await finish.wait()

    monkeypatch.setattr(task_restart, "prepare_task_dispatch", hold_claim)
    dispatch = AsyncMock(return_value="started")
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)
    now = task.auto_retry_at + timedelta(seconds=1)
    async with sessions() as first, sessions() as second:
        first_worker = asyncio.create_task(run_due_auto_retries(first, now=now))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            await asyncio.wait_for(run_due_auto_retries(second, now=now), timeout=5)
            dispatch.assert_not_awaited()
        finally:
            finish.set()
            await asyncio.wait_for(first_worker, timeout=5)

    dispatch.assert_awaited_once()
    async with sessions() as db:
        current = await db.get(Task, task.id)
        assert current.state == TaskState.ASSIGNED
        assert current.auto_retry_at is None
        assert current.auto_retry_count == 1
        assert current.retry_count == 0
        events = (
            await db.scalars(select(Message).where(Message.event_type == "TASK_AUTO_RETRIED"))
        ).all()
        assert [event.event_metadata for event in events] == [
            {"attempt": 1, "max_attempts": 3, "reason": "provider_error", "action": "retry"},
        ]
        assert "Auto-retry 1/3" in events[0].content


@pytest.mark.parametrize("stop", ["billing", "frozen", "close", "disable", "cancel"])
async def test_auto_retry_is_withdrawn_when_work_is_stopped(sessions, monkeypatch, stop):
    user, task, usage_id = await _interrupted_task(sessions, monkeypatch)
    dispatch = AsyncMock()
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)
    monkeypatch.setattr("app.core.workspace_checkpoint.checkpoint_workspace", AsyncMock())
    async with sessions() as db:
        if stop == "billing":
            usage = await db.get(WorkspaceUsage, usage_id)
            usage.stopped_at = datetime.now(timezone.utc)
        elif stop == "frozen":
            room = await db.get(Room, task.room_id)
            room.frozen_at = datetime.now(timezone.utc)
        elif stop == "close":
            await close_room(task.room_id, db)
        elif stop == "disable":
            await update_room(task.room_id, RoomUpdate(auto_retry_enabled=False), db, user)
        else:
            current = await db.get(Task, task.id)
            current.state = TaskState.CANCELLED
        await db.commit()
        if stop in {"close", "disable", "cancel"}:
            assert (await db.get(Task, task.id)).auto_retry_at is None

    async with sessions() as db:
        await run_due_auto_retries(db, now=task.auto_retry_at + timedelta(seconds=1))
        current = await db.get(Task, task.id)
        assert current.auto_retry_at is None
        assert current.auto_retry_count == 0
        dispatch.assert_not_awaited()


@pytest.mark.parametrize("guard_schedule", [False, True])
async def test_manual_retry_wins_against_a_due_automatic_attempt(sessions, monkeypatch, guard_schedule):
    user, task, _ = await _interrupted_task(sessions, monkeypatch)
    async with sessions() as db:
        current = await db.get(Task, task.id)
        current.auto_retry_count = 2
        await db.commit()

    entered = asyncio.Event()
    finish = asyncio.Event()

    async def hold_claim(*_args, **_kwargs):
        entered.set()
        await finish.wait()

    monkeypatch.setattr(task_restart, "prepare_task_dispatch", hold_claim)
    dispatch = AsyncMock(return_value="started")
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)
    async with sessions() as manual, sessions() as watchdog:
        request = asyncio.create_task(
            retry_task(
                task.room_id, task.id, TaskRetry(), False, manual, user,
                auto_retry_at=task.auto_retry_at if guard_schedule else None,
            ),
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            await asyncio.wait_for(
                run_due_auto_retries(watchdog, now=task.auto_retry_at + timedelta(seconds=1)),
                timeout=5,
            )
            dispatch.assert_not_awaited()
        finally:
            finish.set()
            result = await asyncio.wait_for(request, timeout=5)

    assert result.auto_retry_count == 0
    assert result.auto_retry_at is None
    assert result.retry_count == 1
    dispatch.assert_awaited_once()
    assert dispatch.await_args.kwargs["source"] == "retry"


async def test_cancel_scheduled_auto_retry_preserves_the_blocked_task(sessions, monkeypatch):
    user, task, _ = await _interrupted_task(sessions, monkeypatch)
    dispatch = AsyncMock()
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)

    async with sessions() as db:
        current = await db.get(Task, task.id)
        current.auto_retry_count = 1
        current.retry_count = 2
        await db.commit()
        pending = await list_tasks(task.room_id, db, user, auto_retry_pending=True)
        assert [item.id for item in pending] == [task.id]
        for _ in range(2):
            response = await cancel_task_auto_retry(task.room_id, task.id, db, user, task.auto_retry_at)
            assert response.status_code == 204

    async with sessions() as db:
        await run_due_auto_retries(db, now=task.auto_retry_at + timedelta(seconds=1))
        current = await db.get(Task, task.id)
        assert current.state == TaskState.BLOCKED
        assert current.auto_retry_at is None
        assert current.auto_retry_reason is None
        assert current.auto_retry_count == 1
        assert current.retry_count == 2
        assert await list_tasks(task.room_id, db, user, auto_retry_pending=True) == []
        events = (
            await db.scalars(select(Message).where(Message.event_type == "TASK_AUTO_RETRY_CANCELLED"))
        ).all()
        assert [event.task_id for event in events] == [task.id]
        with pytest.raises(HTTPException) as conflict:
            await retry_task(task.room_id, task.id, TaskRetry(), False, db, user, task.auto_retry_at)
        assert conflict.value.status_code == 409
        dispatch.assert_not_awaited()


async def test_stale_auto_retry_actions_leave_a_new_schedule_intact(sessions, monkeypatch):
    user, task, _ = await _interrupted_task(sessions, monkeypatch)
    previous_at = task.auto_retry_at - timedelta(minutes=1)
    async with sessions() as db:
        with pytest.raises(HTTPException) as cancel_conflict:
            await cancel_task_auto_retry(task.room_id, task.id, db, user, previous_at)
        assert cancel_conflict.value.status_code == 409
        with pytest.raises(HTTPException) as retry_conflict:
            await retry_task(task.room_id, task.id, TaskRetry(), False, db, user, previous_at)
        assert retry_conflict.value.status_code == 409
        assert (await db.get(Task, task.id)).auto_retry_at == task.auto_retry_at


async def test_due_context_retry_resumes_saved_history(sessions, monkeypatch):
    _, task, _ = await _interrupted_task(
        sessions, monkeypatch, reason="context_window",
        history=[{"role": "user", "content": "Continue the task"}],
    )
    monkeypatch.setattr(task_restart, "prepare_task_dispatch", AsyncMock())
    dispatch = AsyncMock(return_value="started")
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)
    async with sessions() as db:
        await run_due_auto_retries(db, now=task.auto_retry_at + timedelta(seconds=1))

    dispatch.assert_awaited_once()
    assert dispatch.await_args.kwargs["resume"] is True


async def test_due_attempt_uses_shared_hard_quota_gate(sessions, monkeypatch):
    _, task, _ = await _interrupted_task(sessions, monkeypatch)
    gate = AsyncMock(
        return_value=work_gates.WorkRefusal(
            "Monthly workspace hours exhausted.", "ROOM_QUOTA_EXCEEDED", "quota_exhausted",
        ),
    )
    monkeypatch.setattr(work_gates, "work_refusal", gate)
    dispatch = AsyncMock()
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)

    async with sessions() as db:
        await run_due_auto_retries(db, now=task.auto_retry_at + timedelta(seconds=1))
        current = await db.get(Task, task.id)
        assert current.state == TaskState.BLOCKED
        assert current.auto_retry_at is None
        assert current.auto_retry_count == 0
        gate.assert_awaited_once_with(task.room_id, db)
        dispatch.assert_not_awaited()
        skipped = (
            await db.scalars(select(Message).where(Message.event_type == "TASK_AUTO_RETRY_SKIPPED"))
        ).all()
        assert [event.event_metadata["skip_reason"] for event in skipped] == ["dispatch_refused"]
        assert "Monthly workspace hours exhausted." in skipped[0].content


async def _task_events(db, task_id):
    return (
        await db.scalars(
            select(Message.event_type)
            .where(Message.task_id == task_id)
            .order_by(Message.timestamp, Message.id)
        )
    ).all()


async def test_interruption_is_announced_before_its_schedule(sessions, monkeypatch):
    _, task, _ = await _interrupted_task(sessions, monkeypatch)
    async with sessions() as db:
        events = await _task_events(db, task.id)
    assert events.index("TASK_STALLED") < events.index("TASK_AUTO_RETRY_SCHEDULED")


async def test_retry_now_runs_the_scheduled_resume(sessions, monkeypatch):
    user, task, _ = await _interrupted_task(
        sessions, monkeypatch, reason="context_window",
        history=[{"role": "user", "content": "Continue the task"}],
    )
    monkeypatch.setattr(task_restart, "prepare_task_dispatch", AsyncMock())
    dispatch = AsyncMock(return_value="started")
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)

    async with sessions() as db:
        result = await run_task_auto_retry_now(task.room_id, task.id, db, user, task.auto_retry_at)

    dispatch.assert_awaited_once()
    assert dispatch.await_args.kwargs["resume"] is True
    assert dispatch.await_args.kwargs["source"] == "auto_retry"
    assert result.auto_retry_count == 1
    assert result.retry_count == 0
    async with sessions() as db:
        with pytest.raises(HTTPException) as conflict:
            await run_task_auto_retry_now(task.room_id, task.id, db, user, task.auto_retry_at)
        assert conflict.value.status_code == 409


@pytest.mark.parametrize("stop", ["billing", "frozen"])
async def test_due_attempt_in_an_inactive_room_is_reported(sessions, monkeypatch, stop):
    _, task, usage_id = await _interrupted_task(sessions, monkeypatch)
    dispatch = AsyncMock()
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)
    async with sessions() as db:
        if stop == "billing":
            (await db.get(WorkspaceUsage, usage_id)).stopped_at = datetime.now(timezone.utc)
        else:
            (await db.get(Room, task.room_id)).frozen_at = datetime.now(timezone.utc)
        await db.commit()

    async with sessions() as db:
        await run_due_auto_retries(db, now=task.auto_retry_at + timedelta(seconds=1))
        skipped = (
            await db.scalars(select(Message).where(Message.event_type == "TASK_AUTO_RETRY_SKIPPED"))
        ).all()
    assert [event.event_metadata["skip_reason"] for event in skipped] == ["room_inactive"]
    dispatch.assert_not_awaited()


async def test_due_attempt_waits_while_the_room_row_is_updated(sessions, monkeypatch):
    _, task, _ = await _interrupted_task(sessions, monkeypatch)
    monkeypatch.setattr(task_restart, "prepare_task_dispatch", AsyncMock())
    dispatch = AsyncMock(return_value="started")
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)
    now = task.auto_retry_at + timedelta(seconds=1)

    async with sessions() as holder:
        await holder.execute(select(Room).where(Room.id == task.room_id).with_for_update())
        async with sessions() as db:
            await asyncio.wait_for(run_due_auto_retries(db, now=now), timeout=5)
            current = await db.get(Task, task.id)
            assert current.auto_retry_at is not None
            assert current.auto_retry_count == 0
        dispatch.assert_not_awaited()
        await holder.rollback()

    async with sessions() as db:
        await run_due_auto_retries(db, now=now)
    dispatch.assert_awaited_once()


async def test_stall_streak_skip_is_reported(sessions, monkeypatch):
    _, task, _ = await _interrupted_task(sessions, monkeypatch)
    async with sessions() as db:
        (await db.get(Task, task.id)).stall_streak = STALL_RESTART_THRESHOLD
        await db.commit()
        await run_due_auto_retries(db, now=task.auto_retry_at + timedelta(seconds=1))
        events = await _task_events(db, task.id)
    assert events[-1] == "TASK_AUTO_RETRY_SKIPPED"


@pytest.mark.parametrize("limit_reason", ["prompt_cap", "model_context_limit"])
async def test_resume_is_refused_after_a_prompt_cap_stop(sessions, monkeypatch, limit_reason):
    user, task, _ = await _interrupted_task(
        sessions, monkeypatch, history=[{"role": "user", "content": "Continue the task"}],
    )
    dispatch = AsyncMock()
    monkeypatch.setattr(task_restart, "request_task_dispatch", dispatch)
    async with sessions() as db:
        db.add(
            Message(
                room_id=task.room_id, task_id=task.id, event_type="TOOL_LIMIT_REACHED",
                content="The prompt is over the cap.", event_metadata={"reason": limit_reason},
            )
        )
        await db.commit()
        with pytest.raises(HTTPException) as refused:
            await resume_task(task.room_id, task.id, db, user)
    assert refused.value.status_code == 422
    assert "cannot be resumed" in refused.value.detail
    dispatch.assert_not_awaited()
