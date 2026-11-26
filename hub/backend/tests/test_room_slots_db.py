"""Parallel-task slots against PostgreSQL (internal design notes, E1)."""

import asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from app.core import room_slots
from app.core.proxy_session import handle_task_created
from app.core.watchdog import block_stalled_legacy_tasks
from app.models import Agent, Message, Room, Task, TaskState, User
from app.models.task import DispatchStatus

pytestmark = [pytest.mark.db, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def sessions():
    url = os.environ.get("DATABASE_URL")
    if not url or not (make_url(url).database or "").endswith("_test"):
        pytest.skip("DATABASE_URL must point to a PostgreSQL database ending in _test")

    schema = f"room_slots_{uuid4().hex}"
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


@pytest.fixture(autouse=True)
def _no_background_replay(monkeypatch):
    monkeypatch.setattr(room_slots, "schedule_queued_dispatch", lambda *a, **k: None)


async def _room(sessions, *, running: int, base: int = 2):
    """A room with ``running`` claimed tasks and one more task ready to start."""
    user = User(email=f"{uuid4().hex}@example.test", hashed_password="test")
    room = Room(name="Slots", creator_id=user.id, parallel_tasks_base=base)
    agent = Agent(owner_id=user.id, nickname="worker", agent_uid=uuid4().hex, token="t")
    tasks = [
        Task(
            room_id=room.id, title=f"running {i}", state=TaskState.IN_PROGRESS,
            assigned_agent_id=agent.id, dispatch_status=DispatchStatus.DISPATCHED,
            run_id=uuid4().hex, runtime_state="running",
        )
        for i in range(running)
    ]
    ready = Task(
        room_id=room.id, title="next", state=TaskState.ASSIGNED,
        assigned_agent_id=agent.id, dispatch_status=DispatchStatus.READY,
    )
    async with sessions() as db:
        for row in (user, room, agent, *tasks, ready):
            db.add(row)
            await db.flush()
        await db.commit()
    return room, tasks, ready


def _pressure(monkeypatch, payload):
    mock = AsyncMock(return_value=payload)
    monkeypatch.setattr(room_slots, "room_pressure", mock)
    return mock


async def test_admission_counts_only_claimed_working_tasks(sessions, monkeypatch):
    room, tasks, _ = await _room(sessions, running=2)
    probe = _pressure(monkeypatch, {"pct": 40.0, "psi_some_avg10": 0.0})
    async with sessions() as db:
        # A task waiting for review holds no slot.
        tasks[0].state = TaskState.WAITING_REVIEW
        db.add(tasks[0])
        await db.commit()
        a = await room_slots.admit(room.id, db)
        await db.rollback()
    assert a.ok and a.reason == "base" and a.busy == 1
    probe.assert_not_awaited()


async def test_third_task_is_queued_with_its_dispatch_intent(sessions, monkeypatch):
    room, _, ready = await _room(sessions, running=2)
    _pressure(monkeypatch, {"pct": 75.0, "psi_some_avg10": 0.0})

    async with sessions() as db:
        await handle_task_created(ready.id, room.id, None, None, db, resume=True,
                                  rework_feedback="fix the form")
    async with sessions() as db:
        await handle_task_created(ready.id, room.id, None, None, db, resume=True,
                                  rework_feedback="fix the form")

    async with sessions() as db:
        current = await db.get(Task, ready.id)
        assert current.dispatch_status == DispatchStatus.READY
        assert current.run_id is None and current.queued_at is not None
        assert current.queued_dispatch == {
            "resume": True, "rework_feedback": "fix the form", "auto_retry_started_at": None,
        }
        events = (await db.scalars(
            select(Message).where(Message.event_type == room_slots.TASK_QUEUED)
        )).all()
    assert len(events) == 1  # told once, not on every replay
    assert events[0].event_metadata["reason"] == "memory_busy"


async def test_queue_keeps_updated_at_for_automatic_retries(sessions, monkeypatch):
    room, _, ready = await _room(sessions, running=2)
    _pressure(monkeypatch, None)
    async with sessions() as db:
        task = await db.get(Task, ready.id)
        before = task.updated_at
        admission = await room_slots.admit(room.id, db)
        assert not admission.ok
        await room_slots.queue_task(
            task, admission, db, resume=False, rework_feedback=None,
            auto_retry_started_at=before,
        )
    async with sessions() as db:
        current = await db.get(Task, ready.id)
    assert current.updated_at == before
    assert current.queued_dispatch["auto_retry_started_at"] == before.isoformat()


async def test_cap_queues_without_reading_memory(sessions, monkeypatch):
    room, _, ready = await _room(sessions, running=4)
    probe = _pressure(monkeypatch, {"pct": 1.0})
    async with sessions() as db:
        await handle_task_created(ready.id, room.id, None, None, db)
    async with sessions() as db:
        assert (await db.get(Task, ready.id)).queued_at is not None
    probe.assert_not_awaited()


async def test_freed_slot_replays_the_queue_in_order(sessions, monkeypatch):
    room, tasks, ready = await _room(sessions, running=2)
    _pressure(monkeypatch, None)
    async with sessions() as db:
        await handle_task_created(ready.id, room.id, None, None, db, resume=True)
    dispatch = AsyncMock(return_value="started")
    monkeypatch.setattr("app.core.proxy_session.request_task_dispatch", dispatch)

    async with sessions() as db:
        tasks[0].state = TaskState.COMPLETED
        db.add(tasks[0])
        await db.commit()
        started = await room_slots.dispatch_queued(room.id, db)

    assert started >= 1
    kwargs = dispatch.await_args.kwargs
    assert dispatch.await_args.args[0].id == ready.id
    assert kwargs["resume"] is True and kwargs["source"] == "room_slots"


async def test_a_task_moved_on_while_queued_leaves_the_queue(sessions, monkeypatch):
    room, _, ready = await _room(sessions, running=2)
    _pressure(monkeypatch, None)
    async with sessions() as db:
        await handle_task_created(ready.id, room.id, None, None, db)
        current = await db.get(Task, ready.id)
        current.state = TaskState.PAUSED
        db.add(current)
        await db.commit()
    dispatch = AsyncMock(return_value="started")
    monkeypatch.setattr("app.core.proxy_session.request_task_dispatch", dispatch)
    async with sessions() as db:
        await room_slots.sweep_queues(db)
    async with sessions() as db:
        current = await db.get(Task, ready.id)
    assert current.queued_at is None and current.queued_dispatch is None
    dispatch.assert_not_awaited()


async def test_queued_resume_is_not_blocked_as_a_stalled_legacy_run(sessions, monkeypatch):
    room, _, ready = await _room(sessions, running=2)
    _pressure(monkeypatch, None)
    async with sessions() as db:
        current = await db.get(Task, ready.id)
        current.state = TaskState.IN_PROGRESS  # Resume moved it back to work
        db.add(current)
        await db.commit()
        await handle_task_created(ready.id, room.id, None, None, db, resume=True)
    async with sessions() as db:
        await db.execute(
            Task.__table__.update().where(Task.id == ready.id)
            .values(updated_at=datetime.utcnow() - timedelta(hours=2))
        )
        await db.commit()
        assert await block_stalled_legacy_tasks(db) == 0
        assert (await db.get(Task, ready.id)).state == TaskState.IN_PROGRESS


async def test_admission_is_serialised_within_a_room(sessions, monkeypatch):
    room, _, _ = await _room(sessions, running=0)
    _pressure(monkeypatch, None)
    async with sessions() as first, sessions() as second:
        await room_slots.admit(room.id, first)  # holds the room's admission lock
        waiter = asyncio.create_task(room_slots.admit(room.id, second))
        await asyncio.sleep(0.3)
        assert not waiter.done()
        await first.commit()
        assert (await asyncio.wait_for(waiter, timeout=5)).ok
        await second.rollback()


async def test_connector_tasks_hold_no_slot_and_need_none(sessions, monkeypatch):
    """A connector agent's task runs on the user's machine, not in the room."""
    from app.models.connector import Connector

    room, tasks, ready = await _room(sessions, running=2)
    probe = _pressure(monkeypatch, {"pct": 99.0, "psi_some_avg10": 0.0})
    async with sessions() as db:
        owner = (await db.execute(select(Room.creator_id).where(Room.id == room.id))).scalar()
        connector = Connector(owner_id=owner, label="laptop")
        db.add(connector)
        await db.flush()
        remote = Agent(
            owner_id=owner, nickname="remote", agent_uid=uuid4().hex, token="t2",
            executor="connector", connector_id=connector.id,
        )
        db.add(remote)
        await db.flush()
        remote_id = remote.id
        # One of the two running tasks runs on the connector.
        t0 = await db.get(Task, tasks[0].id)
        t0.assigned_agent_id = remote.id
        await db.commit()

        a = await room_slots.admit(room.id, db)
        await db.rollback()
        assert a.ok and a.reason == "base" and a.busy == 1
        probe.assert_not_awaited()

        r = await db.get(Task, ready.id, populate_existing=True)
        assert await room_slots.needs_slot(r, db) is True
        r.assigned_agent_id = remote_id
        assert await room_slots.needs_slot(r, db) is False
        await db.rollback()
