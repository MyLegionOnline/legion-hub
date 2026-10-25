"""
Stage 1 regression test for internal design notes — model + migration.

Covers the three acceptance criteria of stage 1 against a disposable
PostgreSQL:
  * the migration applies cleanly on a fresh DB and is idempotent on a
    second run (R-M1/R-M2/R-M3);
  * a newly inserted agent gets `has_avatar=false` from the DB default
    without any write from the Python side;
  * deleting an agent cascades to its `agentavatar` row (ON DELETE CASCADE
    declared in the stage-1 CREATE TABLE, not via SQLModel — see
    app/models/agent_avatar.py).

Gated on INIT_DB_TEST_URL, exactly like tests/test_init_db_fresh.py — the
project has no in-process Postgres. Run locally:

    docker run -d --rm --name pg_avatar_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_agent_avatar_migration.py -v
"""
import os
from uuid import uuid4

import pytest

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="set INIT_DB_TEST_URL to a disposable Postgres database to run",
)

# init_db uses the module-level engine built from DATABASE_URL, so it must be
# overridden before app.db.session is imported.
if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL


async def _existing_tables(engine):
    from sqlalchemy import text

    async with engine.connect() as conn:
        result = await conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        return {row[0] for row in result}


@pytest.mark.asyncio
async def test_avatar_migration_applies_and_is_idempotent():
    """Stage-1 columns/table exist after init_db and survive a second run."""
    from sqlmodel import SQLModel

    import app.models  # noqa: F401 — registers AgentAvatar in metadata
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

    await init_db()
    existing = await _existing_tables(engine)
    assert "agentavatar" in existing, "agentavatar table missing after init_db"

    # Second startup against the now-populated DB must not raise (R-M3).
    await init_db()
    existing = await _existing_tables(engine)
    assert "agentavatar" in existing

    await engine.dispose()


@pytest.mark.asyncio
async def test_new_agent_has_avatar_false_from_db_default():
    """An inserted agent picks up has_avatar=false without an explicit set."""
    from sqlmodel import SQLModel
    from sqlalchemy import text

    from app.models.agent import Agent
    from app.models.user import User
    from app.db.session import engine, init_db, get_session

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()

    owner_id = uuid4()
    agent_id = uuid4()
    async for session in get_session():
        session.add(
            User(
                id=owner_id,
                email=f"avatar-{owner_id}@example.com",
                hashed_password="x",
            )
        )
        session.add(
            Agent(
                id=agent_id,
                owner_id=owner_id,
                nickname="avatar-test",
                agent_uid=f"agent-{agent_id}",
                token=f"tok-{agent_id}",
            )
        )
        await session.commit()

    # Read the raw DB column to make sure the DEFAULT propagated server-side
    # and that we are not just reading the Python model default.
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT has_avatar, avatar_updated_at "
                    "FROM agent WHERE id = :id"
                ),
                {"id": agent_id},
            )
        ).one()
    assert row.has_avatar is False
    assert row.avatar_updated_at is None

    await engine.dispose()


@pytest.mark.asyncio
async def test_agentavatar_cascades_on_agent_delete():
    """ON DELETE CASCADE on agentavatar.agent_id actually fires in PostgreSQL."""
    from sqlmodel import SQLModel
    from sqlalchemy import text

    from app.models.agent import Agent
    from app.models.user import User
    from app.db.session import engine, init_db, get_session

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()

    owner_id = uuid4()
    agent_id = uuid4()
    async for session in get_session():
        session.add(
            User(
                id=owner_id,
                email=f"cascade-{owner_id}@example.com",
                hashed_password="x",
            )
        )
        session.add(
            Agent(
                id=agent_id,
                owner_id=owner_id,
                nickname="cascade-test",
                agent_uid=f"agent-{agent_id}",
                token=f"tok-{agent_id}",
            )
        )
        await session.commit()
        # Insert the avatar row directly via SQL so we don't depend on stage 3.
        await session.execute(
            text(
                "INSERT INTO agentavatar (agent_id, mime, content, size_bytes) "
                "VALUES (:id, 'image/png', 'aGVsbG8=', 5)"
            ),
            {"id": agent_id},
        )
        await session.commit()

    async with engine.connect() as conn:
        before = (
            await conn.execute(
                text("SELECT count(*) FROM agentavatar WHERE agent_id = :id"),
                {"id": agent_id},
            )
        ).scalar()
    assert before == 1

    async for session in get_session():
        agent = await session.get(Agent, agent_id)
        await session.delete(agent)
        await session.commit()

    async with engine.connect() as conn:
        after = (
            await conn.execute(
                text("SELECT count(*) FROM agentavatar WHERE agent_id = :id"),
                {"id": agent_id},
            )
        ).scalar()
    assert after == 0, "agentavatar row was not cascaded on agent delete"

    await engine.dispose()
