"""Stage 0 regression test for internal design notes — model + migrations.

Covers the Stage 0 acceptance criteria against a disposable PostgreSQL:
  * the migration applies cleanly on a fresh DB and is idempotent on a
    second run (the ``room_variable`` table, its indexes, and the rebuilt
    secret CHECK lists all survive re-running ``init_db``);
  * ``room_variable`` enforces the name / value-length / environment CHECKs
    and the per-environment uniqueness of ``(room_id, name)``;
  * the widened ``ck_user_secret_kind`` / ``ck_secret_binding_purpose`` lists
    accept the new Room kinds and purposes;
  * deleting a Room cascades to its ``room_variable`` rows.

Gated on INIT_DB_TEST_URL exactly like tests/test_init_db_fresh.py — the
project has no in-process Postgres. Run locally:

    docker run -d --rm --name pg_roomcfg_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_room_config_migration.py -v
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


async def _existing(engine, sql, params=None):
    from sqlalchemy import text

    async with engine.connect() as conn:
        return (await conn.execute(text(sql), params or {})).all()


async def _make_room(get_session, room_id, owner_id):
    from app.models.room import Room
    from app.models.user import User

    async for session in get_session():
        session.add(User(id=owner_id, email=f"rc-{owner_id}@example.com",
                         hashed_password="x"))
        session.add(Room(id=room_id, name="rc", creator_id=owner_id))
        await session.commit()


@pytest.mark.asyncio
async def test_migration_applies_and_is_idempotent():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

    await init_db()
    tables = {r[0] for r in await _existing(
        engine, "SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
    assert "room_variable" in tables

    # Second startup against the populated DB must not raise.
    await init_db()
    idx = {r[0] for r in await _existing(
        engine,
        "SELECT indexname FROM pg_indexes WHERE tablename = 'room_variable'")}
    assert "uq_room_variable_name_env" in idx
    assert "ix_room_variable_room" in idx

    await engine.dispose()


@pytest.mark.asyncio
async def test_room_variable_checks_and_uniqueness():
    from sqlmodel import SQLModel
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError, DBAPIError

    from app.db.session import engine, init_db, get_session

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()

    room_id, owner_id = uuid4(), uuid4()
    await _make_room(get_session, room_id, owner_id)

    async def _insert(name, value="v", environment=None):
        async for session in get_session():
            await session.execute(
                text("INSERT INTO room_variable (room_id, name, value, environment) "
                     "VALUES (:r, :n, :v, :e)"),
                {"r": room_id, "n": name, "v": value, "e": environment},
            )
            await session.commit()

    # bad name
    with pytest.raises((IntegrityError, DBAPIError)):
        await _insert("lowercase")
    # oversized value
    with pytest.raises((IntegrityError, DBAPIError)):
        await _insert("BIG", value="x" * 4097)
    # bad environment
    with pytest.raises((IntegrityError, DBAPIError)):
        await _insert("OK", environment="Staging!")

    # same name, different environments — allowed
    await _insert("API_URL", value="shared")            # environment NULL
    await _insert("API_URL", value="stg", environment="staging")
    # duplicate within the same (NULL) environment — rejected
    with pytest.raises((IntegrityError, DBAPIError)):
        await _insert("API_URL", value="again")

    await engine.dispose()


@pytest.mark.asyncio
async def test_widened_secret_checks_accept_room_values():
    from sqlmodel import SQLModel
    from sqlalchemy import text

    from app.db.session import engine, init_db, get_session

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()

    owner_id = uuid4()
    secret_id = uuid4()
    async for session in get_session():
        await session.execute(
            text('INSERT INTO "user" (id, email, hashed_password) '
                 "VALUES (:id, :email, 'x')"),
            {"id": owner_id, "email": f"sec-{owner_id}@example.com"},
        )
        await session.execute(
            text("INSERT INTO user_secret (id, owner_id, kind, label, ciphertext, key_id) "
                 "VALUES (:id, :o, 'ssh_private_key', 'deploy key', decode('00', 'hex'), 'k1')"),
            {"id": secret_id, "o": owner_id},
        )
        await session.execute(
            text("INSERT INTO secret_binding "
                 "(secret_id, owner_id, resource_type, resource_id, purpose, injection_config) "
                 "VALUES (:s, :o, 'room', :rid, 'deploy_ssh_key', "
                 "CAST(:cfg AS JSONB))"),
            {"s": secret_id, "o": owner_id, "rid": uuid4(),
             "cfg": '{"env_name": "SSH_DEPLOY_KEY", "scope": ["deploy"]}'},
        )
        await session.commit()

    await engine.dispose()


@pytest.mark.asyncio
async def test_room_variable_cascades_on_room_delete():
    from sqlmodel import SQLModel
    from sqlalchemy import text

    from app.db.session import engine, init_db, get_session

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()

    room_id, owner_id = uuid4(), uuid4()
    await _make_room(get_session, room_id, owner_id)

    async for session in get_session():
        await session.execute(
            text("INSERT INTO room_variable (room_id, name, value) "
                 "VALUES (:r, 'APP_ENV', 'staging')"),
            {"r": room_id},
        )
        await session.execute(text("DELETE FROM room WHERE id = :r"), {"r": room_id})
        await session.commit()

    rows = await _existing(
        engine, "SELECT count(*) FROM room_variable WHERE room_id = :r", {"r": room_id})
    assert rows[0][0] == 0

    await engine.dispose()
