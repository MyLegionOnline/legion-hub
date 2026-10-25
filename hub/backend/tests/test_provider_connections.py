"""Settings → Providers stored per account (provider_connection).

Gated on INIT_DB_TEST_URL exactly like tests/test_room_variables.py:

    docker run -d --rm --name pg_provconn_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_provider_connections.py -v
"""

import os
from types import SimpleNamespace
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


async def _make_user(**secrets):
    """A user plus named llm_api_key secrets; returns (user, {name: secret_id})."""
    from app.db.session import engine, get_session
    from app.models.secret import UserSecret
    from app.models.user import User

    user_id = uuid4()
    ids = {}
    async for db in get_session():
        db.add(User(id=user_id, email=f"pc-{user_id}@example.com", hashed_password="x"))
        await db.flush()
        for name, kind in secrets.items():
            secret = UserSecret(
                owner_id=user_id, kind=kind, label=name, ciphertext=b"x", key_id="k1",
            )
            db.add(secret)
            await db.flush()
            ids[name] = secret.id
        await db.commit()
    return SimpleNamespace(id=user_id), ids


@pytest.mark.asyncio
async def test_crud_default_and_ownership():
    from fastapi import HTTPException

    from app.api.v1 import provider_connections as pc
    from app.db.session import engine, get_session

    await _fresh_db()
    alice, keys = await _make_user(vllm="llm_api_key", git="git_access_token")
    bob, _ = await _make_user()

    async for db in get_session():
        a = await pc.create_connection(
            pc.ConnectionCreate(provider="openai_compatible", name=" Local vLLM ",
                                base_url="http://vllm:8000/v1", secret_id=keys["vllm"]),
            db=db, current_user=alice,
        )
        b = await pc.create_connection(
            pc.ConnectionCreate(provider="openai_compatible", name="Work gateway", is_default=True),
            db=db, current_user=alice,
        )
        assert a.name == "Local vLLM" and a.status == "idle" and a.models == []

        # Only one default per account.
        await pc.update_connection(a.id, pc.ConnectionUpdate(is_default=True), db=db, current_user=alice)
        rows = await pc.list_connections(db=db, current_user=alice)
        assert [(r.id, r.is_default) for r in rows] == [(a.id, True), (b.id, False)]

        # Validation result is persisted.
        updated = await pc.update_connection(
            b.id,
            pc.ConnectionUpdate(status="ok", models=[{"id": "gw-model"}], selected_model="gw-model"),
            db=db, current_user=alice,
        )
        assert updated.models == [{"id": "gw-model"}] and updated.status == "ok"

        # A non-LLM secret and someone else's row are refused.
        with pytest.raises(HTTPException) as err:
            await pc.update_connection(b.id, pc.ConnectionUpdate(secret_id=keys["git"]), db=db, current_user=alice)
        assert err.value.status_code == 422
        with pytest.raises(HTTPException) as err:
            await pc.update_connection(a.id, pc.ConnectionUpdate(name="x"), db=db, current_user=bob)
        assert err.value.status_code == 404
        assert await pc.list_connections(db=db, current_user=bob) == []

        await pc.delete_connection(a.id, db=db, current_user=alice)
        assert [r.id for r in await pc.list_connections(db=db, current_user=alice)] == [b.id]
    await engine.dispose()


def test_rejects_unknown_provider_and_bad_url():
    from pydantic import ValidationError

    from app.api.v1 import provider_connections as pc

    with pytest.raises(ValidationError):
        pc.ConnectionCreate(provider="nope")
    with pytest.raises(ValidationError):
        pc.ConnectionCreate(provider="openai_compatible", base_url="file:///etc/passwd")


@pytest.mark.asyncio
async def test_import_runs_once_and_keeps_order():
    from app.api.v1 import provider_connections as pc
    from app.db.session import engine, get_session

    await _fresh_db()
    alice, keys = await _make_user(gw="llm_api_key")
    other, other_keys = await _make_user(theirs="llm_api_key")

    payload = pc.ConnectionImport(connections=[
        pc.ConnectionCreate(provider="openrouter", status="ok", models=[{"id": "m"}], is_default=True),
        pc.ConnectionCreate(provider="openai_compatible", name="gw", secret_id=keys["gw"], is_default=True),
        # A key from another account is dropped, not trusted.
        pc.ConnectionCreate(provider="openai_compatible", name="stolen", secret_id=other_keys["theirs"]),
    ])
    async for db in get_session():
        rows = await pc.import_connections(payload, db=db, current_user=alice)
        assert [r.provider for r in rows] == ["openrouter", "openai_compatible", "openai_compatible"]
        assert [r.is_default for r in rows] == [True, False, False]
        assert rows[1].secret_id == keys["gw"] and rows[2].secret_id is None

        # The same list again (a second tab) adds nothing; another browser's
        # extra connection is merged in, and cannot steal the default.
        again = await pc.import_connections(payload, db=db, current_user=alice)
        assert [r.id for r in again] == [r.id for r in rows]
        office = await pc.import_connections(
            pc.ConnectionImport(connections=[
                pc.ConnectionCreate(provider="openai_compatible", name="gw", secret_id=keys["gw"]),
                pc.ConnectionCreate(provider="groq", is_default=True),
            ]),
            db=db, current_user=alice,
        )
        assert [r.provider for r in office][-1] == "groq" and len(office) == 4
        assert [r.is_default for r in office] == [True, False, False, False]
    await engine.dispose()
