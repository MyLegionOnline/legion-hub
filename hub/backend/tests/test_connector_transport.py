"""Tests for the connector transport: durable outbox, ack, resume, fencing.

Covers ``app/core/connector_broker.py`` (hub/docs/connector_protocol.md §5):
  * a frame enqueued while no socket is open is delivered once a `send`
    callable becomes available, and is never redelivered after `ack`;
  * `resume(session_id, last_seq)` replays only the unacked tail for that
    session, not frames from other sessions or already-acked ones;
  * `accept_inbound_frame` drops a frame carrying a stale `run_id` (the same
    fence `task_runtime.is_run_current` gives local proxy loops) without
    touching task state.

Gated on INIT_DB_TEST_URL, same as tests/test_connector_pairing.py.
"""

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


async def _make_user_and_room(room_id, owner_id):
    from app.db.session import get_session
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    async for session in get_session():
        session.add(
            User(id=owner_id, email=f"conn-{owner_id}@example.com", hashed_password="x")
        )
        session.add(Room(id=room_id, name="conn-room", creator_id=owner_id))
        await session.flush()
        session.add(RoomUserLink(room_id=room_id, user_id=owner_id))
        await session.commit()
        break


async def _make_connector(owner_id, room_id, *, label="vps-transport"):
    from app.core import connector_pairing
    from app.db.session import get_session

    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label=label)
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await connector_pairing.grant_room(
            db, connector_id=paired.connector_id, room_id=room_id, granted_by=owner_id
        )
        await db.commit()
        return paired.connector_id


async def _make_agent_task_session(owner_id, room_id, *, run_id="run-A"):
    """Enough of a dispatched task + cm_agent_proxy session pair to fence against."""
    from app.db.session import get_session
    from app.models.agent import Agent
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.task import DispatchStatus, Task, TaskState

    task_id = uuid4()
    session_id = uuid4()
    agent_id = uuid4()
    async for db in get_session():
        agent = Agent(
            id=agent_id,
            owner_id=owner_id,
            nickname="remote-op",
            agent_uid=f"remote-op-{agent_id.hex[:8]}",
            token="tok-" + uuid4().hex,
            executor="connector",
        )
        db.add(agent)
        task = Task(
            id=task_id,
            room_id=room_id,
            title="deploy",
            assigned_agent_id=agent_id,
            state=TaskState.IN_PROGRESS,
            dispatch_status=DispatchStatus.DISPATCHED,
            run_id=run_id,
            runtime_state="running",
        )
        db.add(task)
        await db.flush()
        proxy_session = CmAgentProxySession(
            id=session_id,
            room_id=room_id,
            kind="cm_agent_proxy",
            side="agent",
            task_id=task_id,
            agent_id=agent_id,
            executor="connector",
        )
        db.add(proxy_session)
        await db.commit()
        break
    return task_id, session_id


@pytest.mark.asyncio
async def test_enqueue_deliver_ack_no_redelivery():
    from app.core import connector_broker
    from app.db.session import get_session

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)
    connector_id = await _make_connector(owner_id, room_id)

    async for db in get_session():
        row = await connector_broker.enqueue(
            db, connector_id=connector_id, kind="ping", payload={}
        )
        await db.commit()
        assert row.seq == 1
        break

    delivered = []

    async def _send(frame):
        delivered.append(frame)

    # Frame enqueued while "offline" (no send attempted yet) is delivered on
    # the first opportunity — this is what makes reconnect self-healing.
    async for db in get_session():
        count = await connector_broker.deliver(db, connector_id=connector_id, send=_send)
        assert count == 1
        assert delivered[0]["type"] == "ping"
        assert delivered[0]["seq"] == 1
        break

    async for db in get_session():
        await connector_broker.ack(db, connector_id=connector_id, seq=1)
        await db.commit()
        break

    # Acked frames are never redelivered.
    delivered.clear()
    async for db in get_session():
        count = await connector_broker.deliver(db, connector_id=connector_id, send=_send)
        assert count == 0
        assert delivered == []
        break

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_deliver_stops_on_send_failure_and_retries_later():
    """A drop mid-delivery leaves the frame pending, not lost."""
    from app.core import connector_broker
    from app.db.session import get_session

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)
    connector_id = await _make_connector(owner_id, room_id)

    async for db in get_session():
        await connector_broker.enqueue(db, connector_id=connector_id, kind="ping", payload={})
        await connector_broker.enqueue(db, connector_id=connector_id, kind="ping", payload={})
        await db.commit()
        break

    async def _broken_send(frame):
        raise ConnectionError("socket gone")

    async for db in get_session():
        count = await connector_broker.deliver(db, connector_id=connector_id, send=_broken_send)
        assert count == 0
        break

    delivered = []

    async def _send(frame):
        delivered.append(frame)

    async for db in get_session():
        count = await connector_broker.deliver(db, connector_id=connector_id, send=_send)
        assert count == 2
        assert [f["seq"] for f in delivered] == [1, 2]
        break

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_resume_replays_only_unacked_tail_for_one_session():
    from app.core import connector_broker
    from app.db.session import get_session

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)
    connector_id = await _make_connector(owner_id, room_id)
    # connector_outbox.session_id FKs into session.id — needs two real
    # cm_agent_proxy sessions, not just uuid4()s, to exercise the FK path.
    _, session_a = await _make_agent_task_session(owner_id, room_id, run_id="run-A")
    _, session_b = await _make_agent_task_session(owner_id, room_id, run_id="run-B")

    async for db in get_session():
        await connector_broker.enqueue(
            db, connector_id=connector_id, session_id=session_a, run_id="run-A",
            kind="status", payload={"status": "TOOL_CALL_STARTED"},
        )
        await connector_broker.enqueue(
            db, connector_id=connector_id, session_id=session_a, run_id="run-A",
            kind="status", payload={"status": "TOOL_CALL_FINISHED"},
        )
        # A frame for a different session must not leak into session_a's resume.
        await connector_broker.enqueue(
            db, connector_id=connector_id, session_id=session_b, run_id="run-B",
            kind="status", payload={"status": "LLM_REQUEST_SENT"},
        )
        await db.commit()
        break

    replayed = []

    async def _send(frame):
        replayed.append(frame)

    # Connector says: "for session_a, I only ever saw seq=1" — reconnect
    # after a drop mid-task must hand back exactly the missed tail.
    async for db in get_session():
        count = await connector_broker.resume(
            db, connector_id=connector_id, session_id=session_a, last_seq=1, send=_send
        )
        assert count == 1
        assert replayed[0]["seq"] == 2
        assert replayed[0]["session_id"] == str(session_a)
        break

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_accept_inbound_frame_drops_stale_run_id_without_state_change():
    from app.core import connector_broker
    from app.db.session import get_session
    from app.models.task import Task

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)
    task_id, session_id = await _make_agent_task_session(owner_id, room_id, run_id="run-A")

    # A `result` frame from a superseded generation (run-B, task is on run-A)
    # must be dropped — never processed as the real result of the current run.
    async for db in get_session():
        decision = await connector_broker.accept_inbound_frame(
            db, frame_type="result", session_id=session_id, run_id="run-B",
        )
        assert decision.accept is False
        break

    async for db in get_session():
        task = await db.get(Task, task_id)
        # Untouched: no finalize/fence ran, state stays exactly as dispatched.
        assert task.run_id == "run-A"
        assert task.runtime_state == "running"
        break

    # The current generation's own frame is accepted.
    async for db in get_session():
        decision = await connector_broker.accept_inbound_frame(
            db, frame_type="result", session_id=session_id, run_id="run-A",
        )
        assert decision.accept is True
        assert decision.task_id == task_id
        break

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_accept_inbound_frame_passes_connection_level_frames():
    """hello/ping/pong/ack/resume never carry run_id — always accepted."""
    from app.core import connector_broker
    from app.db.session import get_session

    await _fresh_db()

    async for db in get_session():
        decision = await connector_broker.accept_inbound_frame(
            db, frame_type="hello", session_id=None, run_id=None,
        )
        assert decision.accept is True
        break

    from app.db.session import engine
    await engine.dispose()
