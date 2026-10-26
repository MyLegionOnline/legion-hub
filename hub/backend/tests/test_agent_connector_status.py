"""An external agent's status in the agents list is its connector's.

Nothing writes ``agent.status`` for an agent linked to a connector, so the
list used to show such an agent "Offline" while its connector was connected.

The serialization test is gated on INIT_DB_TEST_URL (a disposable Postgres),
like tests/test_connector_pairing.py; the mapping tests need no database.
"""

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

# Before any app import: app.db.session binds its engine to DATABASE_URL on
# import, and the DB test below drops every table of that database.
if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL

from app.api.v1.agents import agent_status_from_connector  # noqa: E402


@pytest.mark.parametrize(
    ("connector_status", "expected"),
    [("idle", "online"), ("busy", "busy"), ("offline", "offline"), ("unknown", "offline")],
)
def test_connector_status_maps_to_agent_status(connector_status, expected):
    assert agent_status_from_connector(connector_status, None) == expected


def test_revoked_connector_reads_offline():
    assert agent_status_from_connector("idle", datetime.now(timezone.utc)) == "offline"


@pytest.mark.db
@pytest.mark.asyncio
@pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)
async def test_agents_list_takes_status_from_connector():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.api.v1.agents import _with_skill_names
    from app.core import connector_pairing
    from app.db.session import engine, get_session, init_db
    from app.models.agent import Agent
    from app.models.connector import Connector
    from app.models.user import User

    from sqlalchemy.engine import make_url

    expected = make_url(TEST_DB_URL)
    if (engine.url.host, engine.url.database) != (expected.host, expected.database):
        pytest.fail(f"engine is bound to {engine.url.host}/{engine.url.database}, not the test database")

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()

    owner_id = uuid4()
    async for db in get_session():
        db.add(User(id=owner_id, email=f"status-{owner_id}@example.com", hashed_password="x"))
        await db.flush()
        issued = await connector_pairing.create_connector(db, owner_id=owner_id, label="vps")
        paired = await connector_pairing.pair(db, pairing_code=issued.pairing_code)
        external = Agent(
            owner_id=owner_id,
            nickname="remote",
            agent_uid="remote-uid",
            token="tok-" + uuid4().hex,
            executor="connector",
            connector_id=paired.connector_id,
        )
        internal = Agent(
            owner_id=owner_id,
            nickname="local",
            agent_uid="local-uid",
            token="tok-" + uuid4().hex,
            agent_type="internal",
            status="online",
        )
        db.add_all([external, internal])
        await db.commit()

        connector = await db.get(Connector, paired.connector_id)
        connector.status = "idle"
        await db.commit()
        out = {a.nickname: a.status for a in await _with_skill_names(db, [external, internal], owner_id)}
        assert out == {"remote": "online", "local": "online"}

        connector.status = "offline"
        await db.commit()
        out = {a.nickname: a.status for a in await _with_skill_names(db, [external], owner_id)}
        assert out == {"remote": "offline"}
        break
