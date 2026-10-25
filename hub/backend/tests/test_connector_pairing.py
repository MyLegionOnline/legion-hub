"""Tests for connector pairing, refresh, and Room grants.

Covers ``app/core/connector_pairing.py``:
  * create → pair (one-time code, TTL, repeat-exchange → 409-shaped error);
  * pair → issue_access_token (revoked connector → 401-shaped error);
  * grant_room / revoke_room_grant / has_active_grant, including the case a
    grant exists but the connector itself was revoked.

Gated on INIT_DB_TEST_URL (a disposable Postgres), exactly like
tests/test_room_secrets.py.

    docker run -d --rm --name pg_connector_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_connector_pairing.py -v
"""

import os
from datetime import datetime, timedelta, timezone
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


@pytest.mark.asyncio
async def test_pair_roundtrip_and_repeat_exchange_conflicts():
    from app.core import connector_pairing
    from app.db.session import get_session

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)

    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps-1")
        await db.commit()
        assert issued.pairing_code
        assert issued.expires_at > datetime.now(timezone.utc)
        break

    async for db in get_session():
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await db.commit()
        assert paired.connector_id == issued.connector_id
        assert paired.refresh_token
        break

    # Repeat exchange of the same (now-consumed) code must not succeed again.
    async for db in get_session():
        with pytest.raises(connector_pairing.ConnectorPairingAlreadyUsed):
            await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await db.rollback()
        break

    async for db in get_session():
        token = await connector_pairing.issue_access_token(db, refresh_token=paired.refresh_token)
        await db.commit()
        break
    connector_id = connector_pairing.decode_access_token(token)
    assert connector_id == issued.connector_id

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_pair_with_expired_code_fails():
    from app.core import connector_pairing
    from app.db.session import get_session
    from app.models.connector import Connector

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)

    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps-2")
        await db.commit()
        break

    # Force the TTL into the past instead of sleeping 15 minutes.
    async for db in get_session():
        connector = await db.get(Connector, issued.connector_id)
        connector.pairing_code_expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.add(connector)
        await db.commit()
        break

    async for db in get_session():
        with pytest.raises(connector_pairing.ConnectorPairingExpired):
            await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await db.rollback()
        break

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_revoked_connector_cannot_refresh_or_get_active_grant():
    from app.core import connector_pairing
    from app.db.session import get_session
    from app.models.connector import Connector

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)

    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps-3")
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await connector_pairing.grant_room(
            db, connector_id=paired.connector_id, room_id=room_id, granted_by=owner_id
        )
        await db.commit()
        break

    async for db in get_session():
        assert await connector_pairing.has_active_grant(
            db, connector_id=paired.connector_id, room_id=room_id
        )
        break

    async for db in get_session():
        connector = await db.get(Connector, paired.connector_id)
        await connector_pairing.revoke_connector(db, connector=connector)
        await db.commit()
        break

    # Revocation is immediate: no new access token, and the grant check that
    # dispatch relies on (proxy_session.py handle_task_created) goes false
    # even though the ConnectorRoomGrant row itself was never revoked.
    async for db in get_session():
        with pytest.raises(connector_pairing.ConnectorPairingAlreadyUsed):
            await connector_pairing.issue_access_token(db, refresh_token=paired.refresh_token)
        await db.rollback()
        break

    async for db in get_session():
        assert not await connector_pairing.has_active_grant(
            db, connector_id=paired.connector_id, room_id=room_id
        )
        break

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_grant_and_revoke_room_grant():
    from app.core import connector_pairing
    from app.db.session import get_session

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)

    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps-4")
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        await db.commit()
        break

    async for db in get_session():
        assert not await connector_pairing.has_active_grant(
            db, connector_id=paired.connector_id, room_id=room_id
        )
        await connector_pairing.grant_room(
            db, connector_id=paired.connector_id, room_id=room_id, granted_by=owner_id
        )
        await db.commit()
        break

    async for db in get_session():
        assert await connector_pairing.has_active_grant(
            db, connector_id=paired.connector_id, room_id=room_id
        )
        # Granting again is idempotent — no duplicate active row.
        await connector_pairing.grant_room(
            db, connector_id=paired.connector_id, room_id=room_id, granted_by=owner_id
        )
        await db.commit()
        break

    async for db in get_session():
        await connector_pairing.revoke_room_grant(
            db, connector_id=paired.connector_id, room_id=room_id
        )
        await db.commit()
        break

    async for db in get_session():
        assert not await connector_pairing.has_active_grant(
            db, connector_id=paired.connector_id, room_id=room_id
        )
        break

    from app.db.session import engine
    await engine.dispose()


@pytest.mark.asyncio
async def test_dispatch_fails_without_grant():
    """An agent on `executor="connector"` with no
    active grant into the task's Room must not dispatch — same failure shape
    (`TASK_DISPATCH_FAILED` + back to READY) as `room_not_found`/`no_chat_master`.
    """
    from app.core import connector_pairing
    from app.core.proxy_session import handle_task_created
    from app.db.session import get_session
    from app.models.agent import Agent
    from app.models.task import DispatchStatus, Task, TaskState

    await _fresh_db()
    owner_id = uuid4()
    room_id = uuid4()
    await _make_user_and_room(room_id, owner_id)

    task_id = uuid4()
    async for db in get_session():
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps-5")
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        agent = Agent(
            id=uuid4(),
            owner_id=owner_id,
            nickname="remote-op",
            agent_uid="remote-op-uid",
            token="tok-" + uuid4().hex,
            executor="connector",
            connector_id=paired.connector_id,
        )
        db.add(agent)
        await db.flush()
        task = Task(
            id=task_id,
            room_id=room_id,
            title="deploy",
            assigned_agent_id=agent.id,
            state=TaskState.ASSIGNED,
            dispatch_status=DispatchStatus.READY,
        )
        db.add(task)
        await db.commit()
        break

    # No ConnectorRoomGrant exists for (connector, room) — dispatch must fail
    # closed and hand the task back to READY rather than hang or crash.
    # handle_task_created claims READY→DISPATCHED itself before the connector
    # check runs, so it starts the task in READY like a real dispatch would.
    async for db in get_session():
        await handle_task_created(task_id, room_id, None, None, db)
        break

    async for db in get_session():
        task = await db.get(Task, task_id)
        assert task.dispatch_status == DispatchStatus.READY
        break

    from app.db.session import engine
    await engine.dispose()
