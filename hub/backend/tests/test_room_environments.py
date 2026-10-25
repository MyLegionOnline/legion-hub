"""Stage 12 tests for internal design notes — environments (backend).

Covers ``app/core/secrets/room_environment_gate.py`` and the gate wired into
``room_secrets.resolve_room_secret``:
  * a protected environment without an active unlock window rejects resolve
    with ``EnvironmentLocked`` (``environment_locked``);
  * granting an unlock lets resolve through, and it self-expires;
  * the window is scoped to exactly one ``(room, environment)`` pair — it does
    not leak to a neighbouring environment in the same room, nor to the same
    environment name in a different room;
  * the shared (``NULL``) bucket is never protected;
  * lifting ``protected`` removes the gate immediately, without needing a
    window at all.

Gated on INIT_DB_TEST_URL exactly like test_room_secrets.py.
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

pytest.importorskip("nacl")


def _keyring():
    from app.core.secret_keyring import SecretKeyring

    return SecretKeyring(active_key_id="t", keys={"t": os.urandom(32)}, path="test")


def _service(db):
    from app.core.secrets.service import SecretService
    from app.core.secrets.store import PostgresEncryptedSecretStore

    return SecretService(db, store=PostgresEncryptedSecretStore(db, keyring=_keyring()))


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()
    return engine


async def _make_room(room_id, creator_id, *members, reuse_creator=False):
    """``reuse_creator=True`` skips inserting ``creator_id`` as a new user —
    for a second room owned by the same already-created user."""
    from app.db.session import get_session
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    async for session in get_session():
        if not reuse_creator:
            session.add(
                User(id=creator_id, email=f"re-{creator_id}@example.com", hashed_password="x")
            )
        for m in members:
            session.add(
                User(id=m, email=f"re-{m}@example.com", hashed_password="x")
            )
        session.add(Room(id=room_id, name="re", creator_id=creator_id))
        await session.flush()
        session.add(RoomUserLink(room_id=room_id, user_id=creator_id))
        for m in members:
            session.add(RoomUserLink(room_id=room_id, user_id=m))
        await session.commit()
        break


@pytest.mark.asyncio
async def test_protected_environment_locked_without_unlock():
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", value="prod-token-value",
            environment="production",
        )
        await db.commit()

        # not protected yet — resolves fine
        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", environment="production",
        ) == "prod-token-value"
        await db.commit()

        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True
        )

        with pytest.raises(room_secrets.EnvironmentLocked) as excinfo:
            await room_secrets.resolve_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DEPLOY_TOKEN", environment="production",
            )
        assert excinfo.value.code == "environment_locked"
        assert excinfo.value.http_status == 423
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_unlock_grants_a_window_that_expires():
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", value="prod-token-value",
            environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True
        )
        await db.commit()

        with pytest.raises(room_secrets.EnvironmentLocked):
            await room_secrets.resolve_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DEPLOY_TOKEN", environment="production",
            )

        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="production", actor_user_id=owner_id,
        )

        # the window is open: resolve succeeds without re-confirming
        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", environment="production",
        ) == "prod-token-value"
        await db.commit()

        # the grant itself is audited (Stage 12 — separate from a resolve row)
        from sqlalchemy import text

        audited = (
            await db.execute(
                text(
                    "SELECT count(*) FROM secret_audit WHERE action = 'unlock_environment' "
                    "AND resource_id = :r AND env_name = 'production'"
                ),
                {"r": room_id},
            )
        ).scalar_one()
        assert audited == 1

        # an already-expired window does not count as active
        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="staging", actor_user_id=owner_id,
        )
        rows = (
            await db.execute(
                text(
                    "SELECT id FROM room_environment_unlock WHERE room_id = :r "
                    "AND environment = 'staging'"
                ),
                {"r": room_id},
            )
        ).all()
        assert len(rows) == 1
        await db.execute(
            text(
                "UPDATE room_environment_unlock SET expires_at = :ts "
                "WHERE room_id = :r AND environment = 'staging'"
            ),
            {"ts": datetime.now(timezone.utc) - timedelta(minutes=1), "r": room_id},
        )
        await db.commit()
        assert not await room_environment_gate.has_active_unlock(db, room_id, "staging")
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_unlock_does_not_leak_to_other_environment():
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        for env in ("staging", "production"):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DEPLOY_TOKEN", value=f"{env}-token-value",
                environment=env,
            )
        for env in ("staging", "production"):
            await room_environment_gate.set_protected(
                db, room_id=room_id, environment=env, protected=True
            )
        await db.commit()

        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="staging", actor_user_id=owner_id,
        )

        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", environment="staging",
        ) == "staging-token-value"
        await db.commit()

        # production has its own, still-closed window
        with pytest.raises(room_secrets.EnvironmentLocked):
            await room_secrets.resolve_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DEPLOY_TOKEN", environment="production",
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_unlock_does_not_leak_to_other_room():
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_a, room_b, owner_id = uuid4(), uuid4(), uuid4()
    await _make_room(room_a, owner_id)
    await _make_room(room_b, owner_id, reuse_creator=True)

    async for db in get_session():
        svc = _service(db)
        for room_id in (room_a, room_b):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DEPLOY_TOKEN", value="prod-token-value",
                environment="production",
            )
            await room_environment_gate.set_protected(
                db, room_id=room_id, environment="production", protected=True
            )
        await db.commit()

        await room_environment_gate.grant_unlock(
            db, room_id=room_a, environment="production", actor_user_id=owner_id,
        )

        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_a, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", environment="production",
        ) == "prod-token-value"
        await db.commit()

        # room_b's identically-named environment is untouched by room_a's grant
        with pytest.raises(room_secrets.EnvironmentLocked):
            await room_secrets.resolve_room_secret(
                db, service=svc, room_id=room_b, actor_user_id=owner_id,
                env_name="DEPLOY_TOKEN", environment="production",
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_shared_environment_is_never_protected():
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", value="shared-token-value",
        )
        await db.commit()

        assert await room_environment_gate.is_protected(db, room_id, None) is False

        # even a bogus attempt to "protect" the shared scope has no effect on
        # resolve, since resolve only ever consults the gate for a non-NULL
        # environment (plan §2.4 — NULL is always the fallback bucket).
        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN",
        ) == "shared-token-value"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_unprotecting_removes_the_gate_immediately():
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", value="prod-token-value",
            environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True
        )
        await db.commit()

        with pytest.raises(room_secrets.EnvironmentLocked):
            await room_secrets.resolve_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DEPLOY_TOKEN", environment="production",
            )

        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=False
        )
        await db.commit()

        # no window needed once the flag itself is lifted
        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", environment="production",
        ) == "prod-token-value"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_protected_environment_blocks_mcp_env_lease_bindings():
    """internal design notes Stage 0 — the gate now covers the managed-stdio MCP
    path too: ``room_mcp_env_bindings_for_lease`` must raise the same
    ``EnvironmentLocked`` as ``resolve_room_secret``, before the binding rows
    (and thus the encrypted value behind them) are even queried."""
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session
    from app.models.market import UserInstalledItem

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        item = UserInstalledItem(user_id=owner_id, type="mcp", name="test-mcp")
        db.add(item)
        await db.commit()
        await db.refresh(item)

        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="SHARED_KEY", value="shared-secret-value", mcp_item_id=item.id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_ENV, environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True
        )
        await db.commit()

        with pytest.raises(room_secrets.EnvironmentLocked):
            await room_secrets.room_mcp_env_bindings_for_lease(
                db, room_id=room_id, mcp_item_id=item.id, environment="production",
            )

        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="production", actor_user_id=owner_id,
        )
        rows = await room_secrets.room_mcp_env_bindings_for_lease(
            db, room_id=room_id, mcp_item_id=item.id, environment="production",
        )
        assert len(rows) == 1
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_protected_environment_blocks_room_mcp_auth_header():
    """Same gate, the HTTP-link MCP path (``resolve_room_mcp_auth_header``)."""
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session
    from app.models.market import UserInstalledItem

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        item = UserInstalledItem(user_id=owner_id, type="mcp", name="test-mcp")
        db.add(item)
        await db.commit()
        await db.refresh(item)

        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_MCP_TOKEN", value="bearer-token-value", mcp_item_id=item.id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_AUTH_HEADER, environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True
        )
        await db.commit()

        with pytest.raises(room_secrets.EnvironmentLocked):
            await room_secrets.resolve_room_mcp_auth_header(
                db, service=svc, room_id=room_id, mcp_item_id=item.id, actor_user_id=owner_id,
                environment="production",
            )

        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="production", actor_user_id=owner_id,
        )
        got = await room_secrets.resolve_room_mcp_auth_header(
            db, service=svc, room_id=room_id, mcp_item_id=item.id, actor_user_id=owner_id,
            environment="production",
        )
        assert got == ("DEPLOY_MCP_TOKEN", "bearer-token-value")
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_list_configs_and_active_unlocks():
    from app.core.secrets import room_environment_gate
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="staging", protected=False
        )
        await db.commit()

        configs = await room_environment_gate.list_configs(db, room_id)
        assert configs == {"production": True, "staging": False}

        assert await room_environment_gate.list_active_unlocks(db, room_id) == {}
        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="production", actor_user_id=owner_id,
        )
        unlocks = await room_environment_gate.list_active_unlocks(db, room_id)
        assert set(unlocks) == {"production"}
        assert unlocks["production"] > datetime.now(timezone.utc)
        break

    await engine.dispose()
