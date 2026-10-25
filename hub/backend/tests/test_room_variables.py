"""Stage 1 tests for internal design notes — Room variables (backend).

Covers ``app/core/room_config.py``:
  * CRUD + name / value / environment validation;
  * ``(room, name)`` uniqueness is per-environment;
  * per-environment count limit and the room-wide value-byte budget;
  * the resolve rule: an exact-environment value overrides the shared
    (``NULL``) one — "exact environment → NULL".

Gated on INIT_DB_TEST_URL exactly like tests/test_room_config_migration.py.

    docker run -d --rm --name pg_roomvar_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_room_variables.py -v
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


async def _make_room(room_id, owner_id):
    from app.db.session import get_session
    from app.models.room import Room
    from app.models.user import User

    async for session in get_session():
        session.add(
            User(id=owner_id, email=f"rv-{owner_id}@example.com", hashed_password="x")
        )
        session.add(Room(id=room_id, name="rv", creator_id=owner_id))
        await session.commit()


@pytest.mark.asyncio
async def test_crud_and_validation():
    from fastapi import HTTPException

    from app.core import room_config
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        v = await room_config.create_variable(
            db, room_id, name="APP_ENV", value="staging", created_by=owner_id
        )
        assert v.name == "APP_ENV" and v.environment is None

        with pytest.raises(HTTPException):  # bad name
            await room_config.create_variable(db, room_id, name="bad-name", value="x")
        with pytest.raises(HTTPException):  # oversized value
            await room_config.create_variable(
                db, room_id, name="BIG", value="x" * 4097
            )
        with pytest.raises(HTTPException):  # bad environment
            await room_config.create_variable(
                db, room_id, name="OK", value="x", environment="Staging!"
            )

        # empty environment string normalises to the shared scope
        with pytest.raises(HTTPException):  # duplicate in shared scope
            await room_config.create_variable(
                db, room_id, name="APP_ENV", value="again", environment=""
            )

        updated = await room_config.update_variable(
            db, v, fields={"value": "prod", "description": "env name"}
        )
        assert updated.value == "prod" and updated.description == "env name"

        rows = await room_config.list_variables(db, room_id)
        assert [r.name for r in rows] == ["APP_ENV"]

        await room_config.delete_variable(db, updated)
        assert await room_config.list_variables(db, room_id) == []
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_uniqueness_is_per_environment_and_resolve_priority():
    from app.core import room_config
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        await room_config.create_variable(
            db, room_id, name="API_URL", value="https://shared", created_by=owner_id
        )
        await room_config.create_variable(
            db, room_id, name="API_URL", value="https://stg",
            environment="staging", created_by=owner_id,
        )
        await room_config.create_variable(
            db, room_id, name="ONLY_PROD", value="p",
            environment="production", created_by=owner_id,
        )

        # filtering
        shared = await room_config.list_variables(db, room_id, environment="")
        assert {r.name for r in shared} == {"API_URL"}
        stg = await room_config.list_variables(db, room_id, environment="staging")
        assert {r.name for r in stg} == {"API_URL"}

        # resolve: exact environment wins over the shared value
        eff_stg = await room_config.effective_variables(db, room_id, "staging")
        assert eff_stg == {"API_URL": "https://stg"}

        eff_prod = await room_config.effective_variables(db, room_id, "production")
        assert eff_prod == {"API_URL": "https://shared", "ONLY_PROD": "p"}

        eff_none = await room_config.effective_variables(db, room_id, None)
        assert eff_none == {"API_URL": "https://shared"}

        # materialize_to_env=False drops from the resolved env set
        await room_config.create_variable(
            db, room_id, name="UI_ONLY", value="x",
            materialize_to_env=False, created_by=owner_id,
        )
        assert "UI_ONLY" not in await room_config.effective_variables(db, room_id, None)
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_limits():
    from fastapi import HTTPException

    from app.core import room_config
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        # room-wide value byte budget
        big = "y" * 4000
        for i in range(16):  # 16 * 4000 = 64000 bytes, still under 64 KiB
            await room_config.create_variable(
                db, room_id, name=f"V{i}", value=big, created_by=owner_id
            )
        with pytest.raises(HTTPException):
            await room_config.create_variable(
                db, room_id, name="OVERFLOW", value=big, created_by=owner_id
            )
        break

    await engine.dispose()
