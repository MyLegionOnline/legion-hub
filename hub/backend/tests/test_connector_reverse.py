"""Tests for the connector reverse channel.

Covers ``app/core/connector_reverse.py`` (called from
``app/api/v1/connector_ws.py`` once a fenced inbound frame is accepted) and
the completion bridge it shares with ``core/connector_session.py``'s
``_run_connector_loop``:

  * ``status``/``reasoning`` frames reach the room over the same ephemeral/
    persistent channels a local proxy turn uses (``emit_status``/
    ``emit_reasoning_event``), with no new event types.
  * ``ask_cm`` is published as a visible PROXY_MESSAGE and gets a real CM
    reply relayed back down as a ``proxy_message`` frame (the CM turn itself
    is mocked out — this stage's contract is the wiring, not the LLM call).
  * ``ask_user`` raises the same AGENT_QUESTION card and ``ask_user_registry``
    entry a local ``ask_user`` tool call would, and resolving it relays an
    ``answer`` frame down through the durable outbox.
  * ``result`` closes both proxy sessions, applies the CM's verdict (approve/
    reject/changes, same sentinel contract as the local loop), and resolves
    the future ``_run_connector_loop`` is blocked on so the task finalizes.
  * ``error`` fails the dispatch retryably, mirroring the existing
    ``connector_offline`` path.

Gated on INIT_DB_TEST_URL, same as tests/test_connector_dispatch.py.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch
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


def _one_db():
    """See test_connector_dispatch.py's identical helper for the rationale:
    a plain AsyncSession the caller closes explicitly, not an abandoned
    ``get_session()`` generator."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    from app.db.session import engine

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return async_session()


async def _dispatch_and_wait(task_id, room_id):
    """Start handle_task_created in the background and wait past the dispatch
    enqueue, into connector_session._run_connector_loop's wait-on-future state.

    Returns (runtime_task, bg_db, agent_session, cm_session, run_id) — the
    caller feeds inbound frames to connector_reverse directly (bypassing the
    WS transport, exactly like test_connector_dispatch.py bypasses HTTP).
    """
    from app.core.proxy_session import handle_task_created
    from app.db.session import get_session
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.task import Task
    from sqlalchemy import select

    bg_db = _one_db()
    runtime_task = asyncio.create_task(
        handle_task_created(task_id, room_id, None, None, bg_db)
    )
    await asyncio.sleep(0.4)

    async for db in get_session():
        task = await db.get(Task, task_id)
        run_id = task.run_id
        sessions = (
            await db.execute(
                select(CmAgentProxySession).where(CmAgentProxySession.task_id == task_id)
            )
        ).scalars().all()
        agent_session = next(s for s in sessions if s.side == "agent")
        cm_session = next(s for s in sessions if s.side == "chat_master")
        break

    return runtime_task, bg_db, agent_session.id, cm_session.id, run_id


@pytest.mark.asyncio
async def test_status_and_reasoning_frames_broadcast_without_new_event_types():
    from app.core import connector_reverse
    from app.db.session import get_session
    from app.websockets.manager import manager

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)
        runtime_task, bg_db, agent_session_id, _cm_id, run_id = await _dispatch_and_wait(
            task_id, room_id
        )
        try:
            broadcasts = []
            with patch.object(manager, "broadcast", AsyncMock(side_effect=lambda msg, rid: broadcasts.append(msg))):
                async for db in get_session():
                    await connector_reverse.handle_inbound_frame(
                        db,
                        frame_type="status",
                        payload={"status": "tool_call_started", "detail": {"tool": "bash"}},
                        session_id=agent_session_id,
                        run_id=run_id,
                        task_id=task_id,
                        connector_id=connector_id,
                    )
                    await connector_reverse.handle_inbound_frame(
                        db,
                        frame_type="reasoning",
                        payload={"content": "thinking about the deploy plan"},
                        session_id=agent_session_id,
                        run_id=run_id,
                        task_id=task_id,
                        connector_id=connector_id,
                    )
                    break

            status_msgs = [m for m in broadcasts if m.get("type") == "agent_status"]
            assert any(m["status"] == "tool_call_started" for m in status_msgs)

            reasoning_msgs = [m for m in broadcasts if m.get("event_type") == "AGENT_REASONING"]
            assert any("deploy plan" in m["content"] for m in reasoning_msgs)
        finally:
            runtime_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runtime_task
            await bg_db.close()
    finally:
        _clear_timeouts()

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_ask_cm_publishes_proxy_message_and_relays_cm_reply_down():
    from app.core import connector_reverse
    from app.db.session import get_session
    from app.models.connector import ConnectorOutbox
    from app.models.message import Message, PROXY_MESSAGE
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)
        runtime_task, bg_db, agent_session_id, _cm_id, run_id = await _dispatch_and_wait(
            task_id, room_id
        )
        try:
            fake_cm_turn = AsyncMock(
                return_value=("Use blue-green, drain connections first.", None, 0, 0, "mock", 0, 0)
            )
            with patch("app.core.connector_reverse._cm_turn", fake_cm_turn):
                async for db in get_session():
                    await connector_reverse.handle_inbound_frame(
                        db,
                        frame_type="ask_cm",
                        payload={"content": "Rolling or blue-green deploy?"},
                        session_id=agent_session_id,
                        run_id=run_id,
                        task_id=task_id,
                        connector_id=connector_id,
                    )
                    break

            async for db in get_session():
                messages = (
                    await db.execute(
                        select(Message).where(
                            Message.task_id == task_id, Message.event_type == PROXY_MESSAGE
                        )
                    )
                ).scalars().all()
                kinds = {m.event_metadata.get("kind") for m in messages}
                assert "question" in kinds
                assert "answer" in kinds

                down_frames = (
                    await db.execute(
                        select(ConnectorOutbox).where(
                            ConnectorOutbox.connector_id == connector_id,
                            ConnectorOutbox.kind == "proxy_message",
                        )
                    )
                ).scalars().all()
                assert len(down_frames) == 1
                assert "blue-green" in down_frames[0].payload["content"]
                break
        finally:
            runtime_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runtime_task
            await bg_db.close()
    finally:
        _clear_timeouts()

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_ask_user_round_trip_relays_answer_frame():
    from app.core import ask_user_registry, connector_reverse
    from app.db.session import get_session
    from app.models.connector import ConnectorOutbox
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    os.environ["CONNECTOR_ASK_USER_TIMEOUT_SECONDS"] = "5"
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)
        runtime_task, bg_db, agent_session_id, _cm_id, run_id = await _dispatch_and_wait(
            task_id, room_id
        )
        try:
            async for db in get_session():
                await connector_reverse.handle_inbound_frame(
                    db,
                    frame_type="ask_user",
                    payload={
                        "question_id": "q1",
                        "content": "Overwrite the existing NGINX vhost?",
                        "options": ["Yes", "No"],
                    },
                    session_id=agent_session_id,
                    run_id=run_id,
                    task_id=task_id,
                    connector_id=connector_id,
                )
                break

            assert ask_user_registry.is_session_waiting(agent_session_id)

            resolved = ask_user_registry.resolve("q1", {"selected_options": ["Yes"], "free_text": None})
            assert resolved

            # The relay is a fire-and-forget background task woken by the same
            # asyncio.Event resolve() just set — give the loop one tick.
            for _ in range(20):
                await asyncio.sleep(0.05)
                async for db in get_session():
                    frames = (
                        await db.execute(
                            select(ConnectorOutbox).where(
                                ConnectorOutbox.connector_id == connector_id,
                                ConnectorOutbox.kind == "answer",
                            )
                        )
                    ).scalars().all()
                    break
                if frames:
                    break
            assert len(frames) == 1
            assert "Yes" in frames[0].payload["content"]
        finally:
            runtime_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await runtime_task
            await bg_db.close()
    finally:
        _clear_timeouts()
        os.environ.pop("CONNECTOR_ASK_USER_TIMEOUT_SECONDS", None)

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_result_approve_closes_sessions_and_completes_task():
    from app.core import connector_reverse
    from app.db.session import get_session
    from app.models.session.cm_agent_proxy import CmAgentProxySession
    from app.models.task import Task, TaskState
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)
        runtime_task, bg_db, agent_session_id, cm_session_id, run_id = await _dispatch_and_wait(
            task_id, room_id
        )

        fake_cm_turn = AsyncMock(
            return_value=(None, {"summary": "Deployed and healthy."}, 0, 0, "mock", 0, 0)
        )
        with patch("app.core.connector_reverse._cm_turn", fake_cm_turn):
            async for db in get_session():
                await connector_reverse.handle_inbound_frame(
                    db,
                    frame_type="result",
                    payload={
                        "summary": "Pulled main, restarted nginx, healthcheck OK.",
                        "success": True,
                        "artifacts": {"healthcheck": "200 OK"},
                    },
                    session_id=agent_session_id,
                    run_id=run_id,
                    task_id=task_id,
                    connector_id=connector_id,
                )
                break

        # The result frame resolved connector_reverse's run future — the
        # background _run_connector_loop should now return on its own,
        # letting handle_task_created finalize without needing a cancel.
        await asyncio.wait_for(runtime_task, timeout=5)
        await bg_db.close()

        async for db in get_session():
            task = await db.get(Task, task_id)
            assert task.state == TaskState.COMPLETED

            agent_session = await db.get(CmAgentProxySession, agent_session_id)
            cm_session = await db.get(CmAgentProxySession, cm_session_id)
            assert agent_session.closed_at is not None
            assert cm_session.closed_at is not None
            assert cm_session.close_reason == "task_done"
            break
    finally:
        _clear_timeouts()

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_error_frame_fails_dispatch_retryably():
    from app.core import connector_reverse
    from app.db.session import get_session
    from app.models.task import DispatchStatus, Task, TaskState
    from sqlalchemy import select

    await _fresh_db()
    _set_timeouts(timeout_minutes="5", poll_seconds="0.05")
    try:
        owner_id, room_id = uuid4(), uuid4()
        task_id, _agent_id, connector_id = await _make_granted_connector_setup(owner_id, room_id)
        runtime_task, bg_db, agent_session_id, _cm_id, run_id = await _dispatch_and_wait(
            task_id, room_id
        )

        async for db in get_session():
            await connector_reverse.handle_inbound_frame(
                db,
                frame_type="error",
                payload={"code": "driver_crash", "message": "opencode serve exited"},
                session_id=agent_session_id,
                run_id=run_id,
                task_id=task_id,
                connector_id=connector_id,
            )
            break

        await asyncio.wait_for(runtime_task, timeout=5)
        await bg_db.close()

        async for db in get_session():
            task = await db.get(Task, task_id)
            assert task.dispatch_status == DispatchStatus.READY
            break
    finally:
        _clear_timeouts()

    from app.db.session import engine
    await engine.dispose()
