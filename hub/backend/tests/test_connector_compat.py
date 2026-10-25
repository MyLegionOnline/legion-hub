"""Hub ↔ connector compatibility (protocol version range and agent mode).

Covers ``app/core/connector_compat.py``, the way ``connector_ws.py`` stores
``protocol_version`` from ``hello`` and withholds task frames from an
incompatible connector, the ``compatibility`` field of ``ConnectorOut``, and
(DB-gated, like the other connector dispatch tests) the dispatch preflight in
``proxy_session.handle_task_created``.
"""

import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core import connector_compat as cc

# ── Pure rules ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [("2", 2), (2, 2), ("2.1", 2), (" 3 ", 3), (None, 1), ("", 1), ("x", 1), (True, 1), ([], 1)],
)
def test_parse_protocol_version(raw, expected):
    assert cc.parse_protocol_version(raw) == expected


def test_version_range():
    assert cc.compatibility("0.1.0", {"protocol_version": str(cc.CONNECTOR_PROTOCOL)}) == "ok"
    assert cc.compatibility("0.1.0", {"protocol_version": str(cc.MIN_CONNECTOR_PROTOCOL)}) == "ok"
    assert cc.compatibility("0.1.0", {"protocol_version": str(cc.MIN_CONNECTOR_PROTOCOL - 1)}) == "protocol_too_old"
    assert cc.compatibility("0.1.0", {"protocol_version": str(cc.CONNECTOR_PROTOCOL + 1)}) == "protocol_too_new"


def test_connector_that_never_sent_protocol_version_is_old_not_accepted():
    # Said hello (version known) but without the field: the pre-check generation.
    assert cc.compatibility("0.0.9", {"modes": ["operator"]}) == "protocol_too_old"
    assert cc.compatibility("0.0.9", {"protocol_version": "garbage"}) == "protocol_too_old"


def test_never_connected_is_unknown_and_not_blocked():
    assert cc.compatibility(None, {}) == "unknown"
    assert cc.compatibility(None, None) == "unknown"
    assert cc.dispatch_block_reason(None, {}, "operator") is None
    assert cc.dispatch_block_reason(None, {}, "worker") is None


def test_incompatible_protocol_blocks_dispatch():
    assert cc.dispatch_block_reason("0.1.0", {"protocol_version": "1"}, "operator") == "connector_incompatible"
    assert cc.dispatch_block_reason("9.0.0", {"protocol_version": "99"}, "operator") == "connector_incompatible"


def test_unsupported_mode_blocks_dispatch():
    caps = {"protocol_version": "2", "modes": ["operator"]}
    assert cc.dispatch_block_reason("0.1.0", caps, "operator") is None
    assert cc.dispatch_block_reason("0.1.0", caps, "worker") == "connector_mode_unsupported"
    both = {"protocol_version": "2", "modes": ["operator", "worker"]}
    assert cc.dispatch_block_reason("0.1.0", both, "worker") is None


def test_missing_modes_means_operator_only():
    caps = {"protocol_version": "2"}
    assert cc.dispatch_block_reason("0.1.0", caps, "operator") is None
    assert cc.dispatch_block_reason("0.1.0", caps, "worker") == "connector_mode_unsupported"


def test_protocol_problem_wins_over_mode_problem():
    caps = {"protocol_version": "1", "modes": ["operator"]}
    assert cc.dispatch_block_reason("0.0.9", caps, "worker") == "connector_incompatible"


def test_agent_mode_defaults_to_operator():
    assert cc.agent_mode(None) == "operator"
    assert cc.agent_mode({}) == "operator"
    assert cc.agent_mode({"connector_mode": "worker"}) == "worker"


# ── ConnectorOut ─────────────────────────────────────────────────────────────


def _out(version, capabilities):
    from app.schemas.connector import ConnectorOut

    return ConnectorOut(
        id=uuid.uuid4(),
        owner_id=uuid.uuid4(),
        label="vps",
        status="idle",
        version=version,
        capabilities=capabilities,
        created_at=datetime.now(timezone.utc),
    ).model_dump(mode="json")


def test_connector_out_exposes_compatibility_and_protocol_version():
    ok = _out("0.1.0", {"protocol_version": "2", "modes": ["operator"]})
    assert ok["compatibility"] == "ok"
    assert ok["protocol_version"] == "2"

    old = _out("0.0.9", {"protocol_version": "1"})
    assert old["compatibility"] == "protocol_too_old"

    new = _out("3.0.0", {"protocol_version": "7"})
    assert new["compatibility"] == "protocol_too_new"

    fresh = _out(None, {})
    assert fresh["compatibility"] == "unknown"
    assert fresh["protocol_version"] is None


# ── WebSocket: hello is stored, an incompatible connector gets no task frames ──


class _Row:
    """Stand-in for the Connector row ``connector_ws`` mutates."""

    def __init__(self, version=None, capabilities=None):
        self.id = uuid.uuid4()
        self.version = version
        self.capabilities = capabilities if capabilities is not None else {}
        self.status = "offline"
        self.last_seen_at = None
        self.revoked_at = None


def _ws_app(db):
    from app.api.v1 import connector_ws
    from app.db.session import get_session

    app = FastAPI()
    app.include_router(connector_ws.router, prefix="/api/v1/connectors")

    async def _override():
        yield db

    app.dependency_overrides[get_session] = _override
    return app


def _run_ws(row, frames):
    """Connect as ``row``, send ``frames``, return (deliver mock, resume mock)."""
    from app.api.v1 import connector_ws

    db = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    deliver = AsyncMock(return_value=0)
    resume = AsyncMock(return_value=0)
    with patch.object(connector_ws, "_authenticate", AsyncMock(return_value=row)), \
         patch.object(connector_ws, "_still_active", AsyncMock(return_value=True)), \
         patch.object(connector_ws.connector_broker, "deliver", deliver), \
         patch.object(connector_ws.connector_broker, "resume", resume):
        with TestClient(_ws_app(db)).websocket_connect("/api/v1/connectors/ws?token=t") as ws:
            for frame in frames:
                ws.send_json(frame)
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}
    return deliver, resume


def _hello(**payload):
    return {"type": "hello", "payload": {"connector_id": str(uuid.uuid4()), "version": "0.1.0", **payload}}


def test_hello_stores_protocol_version_and_keeps_capabilities():
    row = _Row()
    deliver, _ = _run_ws(
        row, [_hello(protocol_version="2", capabilities={"agent_cli": "opencode", "modes": ["operator"]})]
    )
    assert row.capabilities["protocol_version"] == "2"
    assert row.capabilities["agent_cli"] == "opencode"
    assert row.capabilities["modes"] == ["operator"]
    assert cc.compatibility(row.version, row.capabilities) == "ok"
    deliver.assert_awaited()  # compatible: pending frames are replayed


def test_hello_without_protocol_version_is_recorded_as_legacy():
    row = _Row()
    _run_ws(row, [_hello(capabilities={"modes": ["operator"]})])
    assert row.capabilities["protocol_version"] == "1"
    assert cc.compatibility(row.version, row.capabilities) == "protocol_too_old"


def test_incompatible_connector_stays_connected_but_gets_no_replay():
    row = _Row(version="0.0.9", capabilities={"protocol_version": "1"})
    frames = [
        _hello(protocol_version="1", capabilities={}),
        {"type": "resume", "payload": {"session_id": str(uuid.uuid4()), "last_seq": 0}},
    ]
    deliver, resume = _run_ws(row, frames)
    # The socket stayed open (ping/pong round-tripped inside _run_ws), but no
    # task frame was replayed, neither on connect nor on resume.
    deliver.assert_not_awaited()
    resume.assert_not_awaited()


def test_connector_upgraded_after_being_incompatible_gets_the_replay():
    row = _Row(version="0.0.9", capabilities={"protocol_version": "1"})
    deliver, _ = _run_ws(row, [_hello(protocol_version="2", capabilities={"modes": ["operator"]})])
    assert row.capabilities["protocol_version"] == "2"
    deliver.assert_awaited_once()  # skipped on connect, done once hello proved it compatible


def test_compatible_connector_gets_replay_on_connect_and_resume_works():
    row = _Row(version="0.1.0", capabilities={"protocol_version": "2"})
    session_id = str(uuid.uuid4())
    deliver, resume = _run_ws(
        row,
        [
            _hello(protocol_version="2", capabilities={"modes": ["operator"]}),
            {"type": "resume", "payload": {"session_id": session_id, "last_seq": 3}},
        ],
    )
    deliver.assert_awaited_once()  # on connect only; hello does not replay twice
    resume.assert_awaited_once()


# ── Dispatch preflight (needs a disposable Postgres, like test_connector_dispatch) ──

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")
needs_db = pytest.mark.skipif(
    not TEST_DB_URL, reason="set INIT_DB_TEST_URL to a disposable Postgres database to run"
)


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()


async def _setup(*, version, capabilities, llm_config=None):
    """A granted connector agent + READY task; the connector's hello state is set explicitly."""
    from app.core import connector_pairing
    from app.db.session import get_session
    from app.models.agent import Agent
    from app.models.connector import Connector
    from app.models.room import Room, RoomUserLink
    from app.models.task import DispatchStatus, Task, TaskState
    from app.models.user import User

    owner_id, room_id, task_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    async for db in get_session():
        db.add(User(id=owner_id, email=f"compat-{owner_id}@example.com", hashed_password="x"))
        db.add(Room(id=room_id, name="compat-room", creator_id=owner_id))
        await db.flush()
        db.add(RoomUserLink(room_id=room_id, user_id=owner_id))
        await db.commit()
        break
    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps")
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await connector_pairing.grant_room(
            db, connector_id=paired.connector_id, room_id=room_id, granted_by=owner_id
        )
        connector = await db.get(Connector, paired.connector_id)
        connector.version = version
        connector.capabilities = capabilities
        db.add(connector)
        agent = Agent(
            id=uuid.uuid4(),
            owner_id=owner_id,
            nickname="remote-op",
            agent_uid=f"remote-op-{uuid.uuid4().hex[:8]}",
            token="tok-" + uuid.uuid4().hex,
            executor="connector",
            connector_id=paired.connector_id,
            llm_config=llm_config or {},
        )
        db.add(agent)
        await db.flush()
        db.add(
            Task(
                id=task_id,
                room_id=room_id,
                title="deploy",
                assigned_agent_id=agent.id,
                state=TaskState.ASSIGNED,
                dispatch_status=DispatchStatus.READY,
            )
        )
        await db.commit()
        break
    return task_id, room_id, paired.connector_id


async def _dispatch_and_inspect(task_id, room_id, connector_id):
    from sqlalchemy import select

    from app.core.proxy_session import handle_task_created
    from app.db.session import engine, get_session
    from app.models.connector import ConnectorOutbox
    from app.models.message import Message, TASK_DISPATCH_FAILED
    from app.models.task import DispatchStatus, Task

    async for db in get_session():
        await handle_task_created(task_id, room_id, None, None, db)
        break
    async for db in get_session():
        task = await db.get(Task, task_id)
        outbox = (
            await db.execute(select(ConnectorOutbox).where(ConnectorOutbox.connector_id == connector_id))
        ).scalars().all()
        failed = (
            await db.execute(
                select(Message).where(Message.task_id == task_id, Message.event_type == TASK_DISPATCH_FAILED)
            )
        ).scalars().all()
        result = (task.dispatch_status == DispatchStatus.READY, list(outbox), list(failed))
        break
    await engine.dispose()
    return result


@needs_db
@pytest.mark.asyncio
async def test_incompatible_connector_fails_dispatch_at_once_without_a_frame():
    await _fresh_db()
    task_id, room_id, connector_id = await _setup(
        version="0.0.9", capabilities={"protocol_version": "1", "modes": ["operator"]}
    )
    back_to_ready, outbox, failed = await _dispatch_and_inspect(task_id, room_id, connector_id)
    assert back_to_ready
    assert outbox == []  # nothing queued, so nothing to wait 15 minutes for
    assert len(failed) == 1 and "connector_incompatible" in failed[0].content


@needs_db
@pytest.mark.asyncio
async def test_unsupported_agent_mode_fails_dispatch_at_once():
    await _fresh_db()
    task_id, room_id, connector_id = await _setup(
        version="0.1.0",
        capabilities={"protocol_version": "2", "modes": ["operator"]},
        llm_config={"connector_mode": "worker"},
    )
    back_to_ready, outbox, failed = await _dispatch_and_inspect(task_id, room_id, connector_id)
    assert back_to_ready
    assert outbox == []
    assert len(failed) == 1 and "connector_mode_unsupported" in failed[0].content
