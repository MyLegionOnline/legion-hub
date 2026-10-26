"""Direct room with a connector agent (core/connector_direct.py).

  * a message becomes a task dispatched to the connector, with no ChatMaster
    side and the earlier chat inside ``task_data_text``;
  * the connector's ``result`` comes back as the agent's chat message and
    completes the task without a review;
  * ``ask_cm`` without a ChatMaster becomes a question to the human, and a
    typed message answers it;
  * a message while the agent is busy does not start a second task;
  * a direct room created by the agent's owner grants the connector access;
  * ``GET /connectors/{id}/grants`` and ``connector_granted`` in members.

Gated on INIT_DB_TEST_URL, same as tests/test_connector_dispatch.py.
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


def _one_db():
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.db.session import engine

    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)()


async def _setup(*, grant: bool = True):
    """Owner, connector agent, direct room with it. Returns ids."""
    from app.core import connector_pairing
    from app.models.agent import Agent
    from app.models.room import Room, RoomAgentLink, RoomType, RoomUserLink
    from app.models.user import User

    owner_id, room_id, agent_id = uuid4(), uuid4(), uuid4()
    async with _one_db() as db:
        db.add(User(id=owner_id, email=f"direct-{owner_id}@example.com", hashed_password="x"))
        await db.flush()
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps")
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        db.add(
            Agent(
                id=agent_id,
                owner_id=owner_id,
                nickname="ext-prod",
                agent_uid=f"ext-{agent_id.hex[:8]}",
                token="tok-" + uuid4().hex,
                executor="connector",
                connector_id=paired.connector_id,
            )
        )
        db.add(Room(id=room_id, name="direct", creator_id=owner_id, room_type=RoomType.direct))
        await db.flush()
        db.add(RoomUserLink(room_id=room_id, user_id=owner_id))
        db.add(RoomAgentLink(room_id=room_id, agent_id=agent_id))
        if grant:
            await connector_pairing.grant_room(
                db, connector_id=paired.connector_id, room_id=room_id, granted_by=owner_id
            )
        await db.commit()
    return owner_id, room_id, agent_id, paired.connector_id


async def _say(room_id, agent_id, content, *, from_user_id=None):
    """Persist a chat message (as chat.py does) and route it like _bg_direct_agent."""
    from app.core import connector_direct
    from app.models.agent import Agent
    from app.models.message import Message
    from app.models.room import Room

    async with _one_db() as db:
        db.add(Message(room_id=room_id, content=content, sender_user_id=from_user_id))
        await db.commit()
        room = await db.get(Room, room_id)
        agent = await db.get(Agent, agent_id)
        return await connector_direct.handle_direct_message(content, room, agent, db)


async def _wait_for(predicate, timeout=10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        value = await predicate()
        if value:
            return value
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.05)


async def _dispatch_frame(connector_id):
    from sqlalchemy import select

    from app.models.connector import ConnectorOutbox

    async with _one_db() as db:
        return (
            await db.execute(
                select(ConnectorOutbox).where(
                    ConnectorOutbox.connector_id == connector_id,
                    ConnectorOutbox.kind == "dispatch",
                ).order_by(ConnectorOutbox.seq.desc())
            )
        ).scalars().first()


async def _tasks(room_id):
    from sqlalchemy import select

    from app.models.task import Task

    async with _one_db() as db:
        return (
            await db.execute(select(Task).where(Task.room_id == room_id).order_by(Task.created_at))
        ).scalars().all()


async def _events(room_id, event_type):
    from sqlalchemy import select

    from app.models.message import Message

    async with _one_db() as db:
        return (
            await db.execute(
                select(Message).where(Message.room_id == room_id, Message.event_type == event_type)
            )
        ).scalars().all()


async def _frame(connector_id, frame_type, payload, *, session_id, run_id, task_id):
    from app.core import connector_reverse

    async with _one_db() as db:
        await connector_reverse.handle_inbound_frame(
            db,
            frame_type=frame_type,
            payload=payload,
            session_id=session_id,
            run_id=run_id,
            task_id=task_id,
            connector_id=connector_id,
        )


def _runtime_task(task_id):
    name = f"proxy_session_{task_id}"
    return next((t for t in asyncio.all_tasks() if t.get_name() == name), None)


async def _stop_runtime(task_id):
    runtime = _runtime_task(task_id)
    if runtime is not None and not runtime.done():
        runtime.cancel()
        try:
            await runtime
        except (asyncio.CancelledError, Exception):
            pass


@pytest.fixture(autouse=True)
def _fast_poll():
    os.environ["CONNECTOR_DISPATCH_POLL_SECONDS"] = "0.05"
    os.environ["CONNECTOR_ASK_USER_TIMEOUT_SECONDS"] = "5"
    yield
    os.environ.pop("CONNECTOR_DISPATCH_POLL_SECONDS", None)
    os.environ.pop("CONNECTOR_ASK_USER_TIMEOUT_SECONDS", None)


@pytest.mark.asyncio
async def test_message_dispatches_and_result_comes_back_as_agent_reply():
    from sqlalchemy import select

    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.task import TASK_ORIGIN_DIRECT_CONNECTOR, TaskState

    await _fresh_db()
    owner_id, room_id, agent_id, connector_id = await _setup()

    # Earlier chat the remote agent should see.
    from app.models.message import Message

    async with _one_db() as db:
        db.add(Message(room_id=room_id, content="what is the uptime?", sender_user_id=owner_id))
        db.add(Message(room_id=room_id, content="up 3 days", sender_agent_id=agent_id))
        await db.commit()

    outcome = await _say(room_id, agent_id, "restart nginx", from_user_id=owner_id)
    assert outcome == "dispatched"

    frame = await _wait_for(lambda: _dispatch_frame(connector_id))
    (task,) = await _tasks(room_id)
    try:
        assert task.origin == TASK_ORIGIN_DIRECT_CONNECTOR
        assert task.title == "restart nginx"
        text = frame.payload["task_data_text"]
        assert "direct chat" in text
        assert "User: what is the uptime?" in text
        assert "You: up 3 days" in text
        assert text.rstrip().endswith("restart nginx")
        assert text.count("restart nginx") == 1

        async with _one_db() as db:
            sessions = (
                await db.execute(
                    select(CmAgentProxySession).where(CmAgentProxySession.task_id == task.id)
                )
            ).scalars().all()
        assert [s.side for s in sessions] == ["agent"]
        assert sessions[0].peer_session_id is None
        assert sessions[0].executor == "connector"
        assert not await _events(room_id, "TASK_DISPATCH_FAILED")

        await _frame(
            connector_id, "result", {"summary": "nginx restarted, 200 OK", "success": True},
            session_id=sessions[0].id, run_id=frame.run_id, task_id=task.id,
        )

        replies = [
            m for m in await _events(room_id, "MESSAGE")
            if m.sender_agent_id == agent_id and m.task_id == task.id
        ]
        assert [m.content for m in replies] == ["nginx restarted, 200 OK"]

        async def _settled():
            (current,) = await _tasks(room_id)
            return current if current.runtime_state != "running" else None

        settled = await _wait_for(_settled)
        assert settled.state == TaskState.COMPLETED
        assert not await _events(room_id, "TASK_APPROVED")

        # The answered message no longer counts as "busy".
        assert await _say(room_id, agent_id, "and now php-fpm", from_user_id=owner_id) == "dispatched"
        assert len(await _tasks(room_id)) == 2
    finally:
        for t in await _tasks(room_id):
            await _stop_runtime(t.id)

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_ask_cm_goes_to_the_human_and_a_typed_message_answers_it():
    from sqlalchemy import select

    from app.core import ask_user_registry
    from app.models.connector import ConnectorOutbox
    from app.models.session.cm_agent_proxy import CmAgentProxySession

    await _fresh_db()
    owner_id, room_id, agent_id, connector_id = await _setup()
    assert await _say(room_id, agent_id, "deploy v2", from_user_id=owner_id) == "dispatched"
    frame = await _wait_for(lambda: _dispatch_frame(connector_id))
    (task,) = await _tasks(room_id)
    try:
        async with _one_db() as db:
            agent_session = (
                await db.execute(
                    select(CmAgentProxySession).where(CmAgentProxySession.task_id == task.id)
                )
            ).scalars().one()

        await _frame(
            connector_id, "ask_cm", {"content": "Which port?"},
            session_id=agent_session.id, run_id=frame.run_id, task_id=task.id,
        )
        questions = await _events(room_id, "AGENT_QUESTION")
        assert [q.content for q in questions] == ["Which port?"]
        assert ask_user_registry.pending_question_for_session(agent_session.id) is not None

        assert await _say(room_id, agent_id, "8080", from_user_id=owner_id) == "answered"
        assert len(await _tasks(room_id)) == 1
        assert await _events(room_id, "AGENT_QUESTION_ANSWERED")

        async def _answer_frame():
            async with _one_db() as db:
                return (
                    await db.execute(
                        select(ConnectorOutbox).where(
                            ConnectorOutbox.connector_id == connector_id,
                            ConnectorOutbox.kind == "answer",
                        )
                    )
                ).scalars().first()

        answer = await _wait_for(_answer_frame)
        assert answer.payload["content"] == "8080"
    finally:
        await _stop_runtime(task.id)

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_message_while_busy_does_not_start_a_second_task():
    await _fresh_db()
    owner_id, room_id, agent_id, connector_id = await _setup()
    assert await _say(room_id, agent_id, "long job", from_user_id=owner_id) == "dispatched"
    await _wait_for(lambda: _dispatch_frame(connector_id))
    (task,) = await _tasks(room_id)
    try:
        assert await _say(room_id, agent_id, "are you done?", from_user_id=owner_id) == "busy"
        assert len(await _tasks(room_id)) == 1
        (busy,) = await _events(room_id, "AGENT_BUSY")
        assert "@ext-prod" in busy.content
    finally:
        await _stop_runtime(task.id)

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_without_grant_the_failure_names_the_connector():
    await _fresh_db()
    owner_id, room_id, agent_id, connector_id = await _setup(grant=False)
    assert await _say(room_id, agent_id, "hello", from_user_id=owner_id) == "dispatched"

    failures = await _wait_for(lambda: _events(room_id, "TASK_DISPATCH_FAILED"))
    meta = failures[0].event_metadata
    assert meta["reason"] == "connector_not_granted"
    assert meta["connector_id"] == str(connector_id)
    assert meta["agent_id"] == str(agent_id)
    (task,) = await _tasks(room_id)
    await _stop_runtime(task.id)

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_direct_room_by_owner_grants_and_grants_are_listed():
    from app.api.v1.connectors import list_connector_grants
    from app.api.v1.rooms import create_room, get_room_members
    from app.core.connector_pairing import has_active_grant
    from app.models.room import RoomType
    from app.models.user import User
    from app.schemas.room import RoomCreate

    await _fresh_db()
    owner_id, _room_id, agent_id, connector_id = await _setup(grant=False)

    async with _one_db() as db:
        owner = await db.get(User, owner_id)
        room = await create_room(
            RoomCreate(name="dm", room_type=RoomType.direct, agent_ids=[agent_id]),
            db=db,
            current_user=owner,
        )
        assert await has_active_grant(db, connector_id=connector_id, room_id=room.id)

        grants = await list_connector_grants(connector_id, db=db, current_user=owner)
        assert [g.room_id for g in grants] == [room.id]

        members = await get_room_members(room.id, db=db, current_user=owner)
        (agent_entry,) = members["agents"]
        assert agent_entry["executor"] == "connector"
        assert agent_entry["connector_id"] == str(connector_id)
        assert agent_entry["connector_granted"] is True
        assert agent_entry["owned"] is True

        # The room from _setup has no grant.
        members = await get_room_members(_room_id, db=db, current_user=owner)
        assert members["agents"][0]["connector_granted"] is False

    from app.db.session import engine
    await engine.dispose()
