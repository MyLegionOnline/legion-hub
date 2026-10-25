"""Tests for the dispatcher branching of connector-executed tasks.

Covers ``app/core/connector_session.py`` and the branch point it adds to
``proxy_session.handle_task_created``:
  * a task assigned to a granted connector agent claims READY→DISPATCHED and
    gets a run_id through the exact same path a local task does, then the
    session pair is marked executor="connector"/billing="external" and a
    ``dispatch`` frame lands in the durable outbox;
  * Stop (cancelling the coroutine, as the real Stop/Pause endpoints do via
    ``task_runtime.cancel_local_proxy_task``) makes the loop notify the
    connector with a ``stop`` frame and propagate the cancellation instead of
    swallowing it;
  * a connector that never acks the dispatch frame within
    CONNECTOR_DISPATCH_TIMEOUT_MINUTES fails the generation the same
    retryable way ``connector_not_granted``/``room_not_found`` do.

Gated on INIT_DB_TEST_URL, same as tests/test_connector_pairing.py.
"""

import asyncio
import os
from uuid import uuid4

import pytest

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()
    return engine


async def _make_user_and_room(room_id, owner_id, *, cm_agent_id):
    from app.db.session import get_session
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    async for session in get_session():
        session.add(
            User(id=owner_id, email=f"conn-{owner_id}@example.com", hashed_password="x")
        )
        room = Room(id=room_id, name="conn-room", creator_id=owner_id)
        room.coordinator_config = {"master_agent_id": str(cm_agent_id)}
        session.add(room)
        await session.flush()
        session.add(RoomUserLink(room_id=room_id, user_id=owner_id))
        await session.commit()
        break


async def _make_granted_connector_setup(owner_id, room_id):
    """A connector agent with an active Room grant, a ChatMaster agent, and a
    READY/ASSIGNED task pointed at the connector agent."""
    from app.core import connector_pairing
    from app.db.session import get_session
    from app.models.agent import Agent
    from app.models.task import DispatchStatus, Task, TaskState

    cm_agent_id = uuid4()
    task_id = uuid4()

    await _make_user_and_room(room_id, owner_id, cm_agent_id=cm_agent_id)

    async for db in get_session():
        cm_agent = Agent(
            id=cm_agent_id,
            owner_id=owner_id,
            nickname="chat-master",
            agent_uid=f"cm-{cm_agent_id.hex[:8]}",
            token="tok-" + uuid4().hex,
        )
        db.add(cm_agent)
        await db.commit()
        break

    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps-4")
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await connector_pairing.grant_room(
            db, connector_id=paired.connector_id, room_id=room_id, granted_by=owner_id
        )
        agent = Agent(
            id=uuid4(),
            owner_id=owner_id,
            nickname="remote-op",
            agent_uid=f"remote-op-{uuid4().hex[:8]}",
            token="tok-" + uuid4().hex,
            executor="connector",
            connector_id=paired.connector_id,
        )
        db.add(agent)
        await db.flush()
        task = Task(
            id=task_id,
            room_id=room_id,
            title="deploy main to prod",
            assigned_agent_id=agent.id,
            state=TaskState.ASSIGNED,
            dispatch_status=DispatchStatus.READY,
        )
        db.add(task)
        await db.commit()
        break

    return task_id, agent.id, paired.connector_id


def _set_timeouts(*, timeout_minutes: str, poll_seconds: str):
    os.environ["CONNECTOR_DISPATCH_TIMEOUT_MINUTES"] = timeout_minutes
    os.environ["CONNECTOR_DISPATCH_POLL_SECONDS"] = poll_seconds


def _clear_timeouts():
    os.environ.pop("CONNECTOR_DISPATCH_TIMEOUT_MINUTES", None)
    os.environ.pop("CONNECTOR_DISPATCH_POLL_SECONDS", None)


@pytest.mark.asyncio
async def test_resolve_executor_connector_vs_local():
    from app.core.connector_session import resolve_executor
    from app.models.agent import Agent

    await _fresh_db()
    owner_id = uuid4()

    local_agent = Agent(
        id=uuid4(), owner_id=owner_id, nickname="a", agent_uid="a-uid", token="t1",
    )
    connector_agent_no_link = Agent(
        id=uuid4(), owner_id=owner_id, nickname="b", agent_uid="b-uid", token="t2",
        executor="connector",
    )
    connector_agent = Agent(
        id=uuid4(), owner_id=owner_id, nickname="c", agent_uid="c-uid", token="t3",
        executor="connector", connector_id=uuid4(),
    )

    assert await resolve_executor(local_agent, None) == "local"
    # executor="connector" but no linked connector must not try to dispatch
    # remotely — nothing to dispatch to.
    assert await resolve_executor(connector_agent_no_link, None) == "local"
    assert await resolve_executor(connector_agent, None) == "connector"
    assert await resolve_executor(None, None) == "local"


@pytest.mark.asyncio
async def test_dispatch_claims_task_and_enqueues_frame_with_connector_billing():
    from app.core.proxy_session import handle_task_created
    from app.core.task_runtime import RUNTIME_RUNNING
    from app.db.session import get_session
    from app.models.connector import ConnectorOutbox
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.task import DispatchStatus, Task
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id = uuid4()
        room_id = uuid4()
        task_id, agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)

        bg_db = _one_db()
        runtime_task = asyncio.create_task(
            handle_task_created(task_id, room_id, None, None, bg_db)
        )
        try:
            # One poll tick is enough for the dispatch frame to land; the loop
            # then blocks on is_run_current()/ack polling well past here.
            await asyncio.sleep(0.4)

            async for db in get_session():
                task = await db.get(Task, task_id)
                assert task.dispatch_status == DispatchStatus.DISPATCHED
                assert task.run_id is not None
                assert task.runtime_state == RUNTIME_RUNNING
                run_id = task.run_id

                sessions = (
                    await db.execute(
                        select(CmAgentProxySession).where(
                            CmAgentProxySession.task_id == task_id
                        )
                    )
                ).scalars().all()
                assert len(sessions) == 2
                for session in sessions:
                    assert session.executor == "connector"
                    assert session.connector_id == connector_id
                    assert session.billing == "external"

                agent_side = next(s for s in sessions if s.side == "agent")
                frames = (
                    await db.execute(
                        select(ConnectorOutbox).where(
                            ConnectorOutbox.connector_id == connector_id,
                            ConnectorOutbox.session_id == agent_side.id,
                        )
                    )
                ).scalars().all()
                assert len(frames) == 1
                assert frames[0].kind == "dispatch"
                assert frames[0].run_id == run_id
                assert frames[0].payload["task_id"] == str(task_id)
                assert frames[0].payload["mode"] == "operator"
                break
        finally:
            runtime_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runtime_task
            await bg_db.close()
    finally:
        _clear_timeouts()


@pytest.mark.asyncio
async def test_cancellation_sends_stop_frame_and_propagates():
    from app.core.proxy_session import handle_task_created
    from app.db.session import get_session
    from app.models.connector import ConnectorOutbox
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id = uuid4()
        room_id = uuid4()
        task_id, agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)

        bg_db = _one_db()
        runtime_task = asyncio.create_task(
            handle_task_created(task_id, room_id, None, None, bg_db)
        )
        await asyncio.sleep(0.4)  # past the dispatch enqueue, into the wait loop

        runtime_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runtime_task
        await bg_db.close()

        async for db in get_session():
            sessions = (
                await db.execute(
                    select(CmAgentProxySession).where(
                        CmAgentProxySession.task_id == task_id
                    )
                )
            ).scalars().all()
            agent_side = next(s for s in sessions if s.side == "agent")
            stop_frames = (
                await db.execute(
                    select(ConnectorOutbox).where(
                        ConnectorOutbox.connector_id == connector_id,
                        ConnectorOutbox.session_id == agent_side.id,
                        ConnectorOutbox.kind == "stop",
                    )
                )
            ).scalars().all()
            assert len(stop_frames) == 1
            break
    finally:
        _clear_timeouts()


@pytest.mark.asyncio
async def test_pause_fence_without_cancellation_sends_stop_frame():
    """Pause fences the run but does not cancel the coroutine (the local loop
    stops at its next boundary). The remote agent has no such boundary, so the
    loop must tell the connector to stop when it sees the fence."""
    from app.core.proxy_session import handle_task_created
    from app.core.task_runtime import request_task_stop
    from app.db.session import get_session
    from app.models.connector import ConnectorOutbox
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id = uuid4()
        room_id = uuid4()
        task_id, agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)

        bg_db = _one_db()
        runtime_task = asyncio.create_task(
            handle_task_created(task_id, room_id, None, None, bg_db)
        )
        await asyncio.sleep(0.4)  # past the dispatch enqueue, into the wait loop

        async for db in get_session():
            # What PATCH .../state {"state": "pausing"} does before the fence.
            from app.models.task import Task, TaskState

            row = await db.get(Task, task_id)
            row.state = TaskState.PAUSING
            db.add(row)
            await db.commit()
            await request_task_stop(task_id, "user", db)
            await db.commit()
            break

        # The loop returns on its own once it observes the fence — no cancel().
        await asyncio.wait_for(runtime_task, timeout=5)
        await bg_db.close()

        async for db in get_session():
            sessions = (
                await db.execute(
                    select(CmAgentProxySession).where(
                        CmAgentProxySession.task_id == task_id
                    )
                )
            ).scalars().all()
            agent_side = next(s for s in sessions if s.side == "agent")
            stop_frames = (
                await db.execute(
                    select(ConnectorOutbox).where(
                        ConnectorOutbox.connector_id == connector_id,
                        ConnectorOutbox.session_id == agent_side.id,
                        ConnectorOutbox.kind == "stop",
                    )
                )
            ).scalars().all()
            assert len(stop_frames) == 1
            assert stop_frames[0].payload == {"reason": "user"}

            # The paused run settles: PAUSING must become PAUSED, and the
            # DISPATCHED marker go back to READY so Resume can redispatch.
            from app.models.task import DispatchStatus, Task, TaskState

            task = (
                await db.execute(
                    select(Task)
                    .where(Task.id == task_id)
                    .execution_options(populate_existing=True)
                )
            ).scalars().one()
            assert task.state == TaskState.PAUSED
            assert task.dispatch_status == DispatchStatus.READY
            break
    finally:
        _clear_timeouts()


@pytest.mark.asyncio
async def test_connector_offline_timeout_fails_dispatch_and_resets_to_ready():
    from app.core.proxy_session import handle_task_created
    from app.db.session import get_session
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.task import DispatchStatus, Task, TaskState
    from sqlalchemy import select

    await _fresh_db()
    # Timeout fires on the very first poll tick — the connector "never shows up".
    _set_timeouts(timeout_minutes="0.0005", poll_seconds="0.02")
    try:
        owner_id = uuid4()
        room_id = uuid4()
        task_id, agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)

        async for db in get_session():
            await handle_task_created(task_id, room_id, None, None, db)
            break

        async for db in get_session():
            task = await db.get(Task, task_id)
            assert task.dispatch_status == DispatchStatus.READY
            assert task.state == TaskState.BLOCKED

            sessions = (
                await db.execute(
                    select(CmAgentProxySession).where(
                        CmAgentProxySession.task_id == task_id
                    )
                )
            ).scalars().all()
            assert len(sessions) == 2
            for session in sessions:
                assert session.closed_at is not None
                assert session.close_reason == "connector_offline"
            break
    finally:
        _clear_timeouts()


def _one_db():
    """One live AsyncSession for a background handle_task_created task.

    handle_task_created holds this session open for the duration of the
    coroutine (exactly like _run_proxy_bg's own sessionmaker use). Built the
    same way ``get_session()`` builds its session, but as a plain object
    instead of through an async generator: abandoning a generator mid-``async
    for`` (instead of exhausting or explicitly ``aclose()``-ing it) leaves its
    ``async with`` cleanup to run whenever Python happens to garbage-collect
    it — possibly after pytest-asyncio has already torn down the event loop
    the underlying asyncpg connection belongs to, corrupting the connection
    pool for every test that runs afterward. Callers must ``await
    db.close()`` explicitly once the background task is done with it.
    """
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.db.session import engine

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return async_session()
