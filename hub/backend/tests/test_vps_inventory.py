"""internal design notes Stage 3 tests — server inventory + connection resolver.

Gated on INIT_DB_TEST_URL and PyNaCl, exactly like tests/test_room_secrets.py.

    docker run -d --rm --name pg_vps_test -p 55433:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=vpstest postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55433/vpstest" \
        pytest tests/test_vps_inventory.py -v
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


async def _make_room(db, room_id, creator_id, *members):
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    db.add(User(id=creator_id, email=f"vi-{creator_id}@example.com", hashed_password="x"))
    for m in members:
        db.add(User(id=m, email=f"vi-{m}@example.com", hashed_password="x"))
    db.add(Room(id=room_id, name="vi", creator_id=creator_id))
    await db.flush()
    db.add(RoomUserLink(room_id=room_id, user_id=creator_id))
    for m in members:
        db.add(RoomUserLink(room_id=room_id, user_id=m))
    await db.commit()


def _generate_test_ssh_key() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.OpenSSH,
        serialization.NoEncryption(),
    ).decode()


# A real unencrypted ed25519 key — room_secrets.bind_room_secret parses it with
# cryptography's OpenSSH loader, so it must actually decode. Generated per run so
# no private key is committed.
_TEST_SSH_KEY = _generate_test_ssh_key()


@pytest.mark.asyncio
async def test_add_server_and_list_never_exposes_values():
    from app.core import vps_inventory
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)

        result = await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", root_password="root-password-value",
        )
        await db.commit()
        assert result["name"] == "PROD"
        assert result["host"] == "203.0.113.10"
        assert result["port"] == 22
        assert result["user"] == "root"
        assert result["has_root_password"] is True
        assert result["has_root_key"] is False
        assert result["has_key"] is False
        assert result["has_password"] is False
        assert "root-password-value" not in str(result)

        rows = await vps_inventory.list_servers(db, room_id)
        assert [r["name"] for r in rows] == ["PROD"]
        assert "root-password-value" not in str(rows)
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_add_server_requires_exactly_one_root_credential():
    from app.core import vps_inventory
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)

        with pytest.raises(vps_inventory.ServerInvalid):
            await vps_inventory.add_server(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                name="prod", host="203.0.113.10",
            )
        with pytest.raises(vps_inventory.ServerInvalid):
            await vps_inventory.add_server(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                name="prod", host="203.0.113.10",
                root_password="pw", root_key=_TEST_SSH_KEY,
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_connection_prefers_key_over_password():
    from app.core import vps_inventory
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session
    from app.models.secret import SecretKind, SecretPurpose

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", port=2222, user="legion",
            root_password="root-password-value",
        )
        await db.commit()

        # everyday credentials, bound directly (Stage 8 will do this via the
        # bootstrap wizard; here we simulate its end state).
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_PASSWORD", value="everyday-password-value",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_KEY", value=_TEST_SSH_KEY,
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.SSH_PRIVATE_KEY,
        )
        await db.commit()

        conn = await vps_inventory.resolve_connection(
            db, service=svc, room_id=room_id, actor_user_id=owner_id, server="prod",
        )
        assert conn.legion_conn["host"] == "203.0.113.10"
        assert conn.legion_conn["port"] == 2222
        assert conn.legion_conn["user"] == "legion"
        assert conn.legion_conn["auth"]["kind"] == "key"
        assert conn.legion_conn["auth"]["private_key"] == _TEST_SSH_KEY
        assert "ROOT" not in "".join(conn.mask_values.keys())
        assert conn.legion_conn["egress_ticket"]
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_connection_server_not_found():
    from app.core import vps_inventory
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        with pytest.raises(vps_inventory.ServerNotFound):
            await vps_inventory.resolve_connection(
                db, service=svc, room_id=room_id, actor_user_id=owner_id, server="missing",
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_connection_missing_credentials():
    """A server with a host but no everyday key/password — root creds alone
    (plan §3.4) must never satisfy the resolver."""
    from app.core import vps_inventory
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", root_password="root-password-value",
        )
        await db.commit()

        with pytest.raises(vps_inventory.ServerCredentialsMissing):
            await vps_inventory.resolve_connection(
                db, service=svc, room_id=room_id, actor_user_id=owner_id, server="prod",
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_connection_gated_by_protected_environment():
    from app.core import vps_inventory
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session
    from app.models.secret import SecretKind, SecretPurpose

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", root_password="root-password-value",
            environment="production",
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_PASSWORD", value="everyday-password-value",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
            environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True,
        )
        await db.commit()

        with pytest.raises(vps_inventory.EnvironmentLocked):
            await vps_inventory.resolve_connection(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                server="prod", environment="production",
            )

        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="production", actor_user_id=owner_id,
        )
        await db.commit()

        conn = await vps_inventory.resolve_connection(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            server="prod", environment="production",
        )
        assert conn.legion_conn["auth"]["kind"] == "password"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_connection_never_reads_root_credentials():
    """Even if the everyday KEY/PASSWORD is absent, the resolver must not
    fall back to ROOT_KEY/ROOT_PASSWORD (plan §3.4)."""
    from app.core import vps_inventory
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", root_key=_TEST_SSH_KEY,
        )
        await db.commit()

        with pytest.raises(vps_inventory.ServerCredentialsMissing):
            await vps_inventory.resolve_connection(
                db, service=svc, room_id=room_id, actor_user_id=owner_id, server="prod",
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_add_server_rejects_a_duplicate_name_in_the_same_environment_scope():
    """8b.3 — a repeat name must 409 as `ServerExists`, not fall through to
    `room_config.create_variable`'s raw `HTTPException` on the HOST variable.
    """
    from app.core import vps_inventory
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", root_password="root-password-value",
        )
        await db.commit()

        with pytest.raises(vps_inventory.ServerExists):
            await vps_inventory.add_server(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                name="prod", host="203.0.113.99", root_password="other-password-value",
            )

        # The same name in a *different* environment scope is unaffected —
        # uniqueness is per (room, name, environment), same as any other
        # Room variable/secret.
        result = await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.50", root_password="staging-password-value",
            environment="staging",
        )
        assert result["host"] == "203.0.113.50"
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_add_server_rolls_back_partial_variables_on_credential_bind_failure():
    """8b.2 — a failure partway through `add_server` (here: the root secret
    name colliding with one bound directly, bypassing `add_server`'s own
    existence check) must not leave a HOST/PORT/USER-only, credential-less,
    undeletable server behind.
    """
    from app.core import vps_inventory
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session
    from app.models.secret import SecretKind, SecretPurpose

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)

        # Pre-bind the ROOT_PASSWORD name `add_server` will try to create,
        # forcing `bind_room_secret`'s own uniqueness check to fail *after*
        # HOST/PORT/USER have already been committed.
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="VPS_PROD_ROOT_PASSWORD", value="pre-existing-value",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
        )
        await db.commit()

        with pytest.raises(room_secrets.RoomSecretExists):
            await vps_inventory.add_server(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                name="prod", host="203.0.113.10", root_password="root-password-value",
            )
        await db.rollback()

        # No orphaned HOST/PORT/USER rows: the server does not exist at all,
        # so a corrected retry (a different name, or after removing the
        # colliding secret) is possible rather than being permanently
        # blocked by a half-created "prod".
        rows = await vps_inventory.list_servers(db, room_id)
        assert rows == []
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_forget_root_credentials_is_idempotent_and_scoped():
    from app.core import vps_inventory
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        svc = _service(db)
        await vps_inventory.add_server(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            name="prod", host="203.0.113.10", root_password="root-password-value",
        )
        await db.commit()

        await vps_inventory.forget_root_credentials(
            db, service=svc, room_id=room_id, actor_user_id=owner_id, name="prod",
        )
        await db.commit()
        rows = await vps_inventory.list_servers(db, room_id)
        assert rows[0]["has_root_password"] is False
        assert rows[0]["host"] == "203.0.113.10"  # host/port/user untouched

        # calling again is a no-op, not an error
        await vps_inventory.forget_root_credentials(
            db, service=svc, room_id=room_id, actor_user_id=owner_id, name="prod",
        )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_cross_room_vps_inventory_isolation():
    """8b.9 — two Rooms with a same-named server never see or touch each
    other's inventory, mirroring
    ``test_room_mcp_secrets.py::test_environment_lock_scoped_per_room_for_mcp_auth_header``
    for the VPS resolver's three operations (list/resolve/forget-root).
    """
    from app.core import vps_inventory
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session
    from app.models.secret import SecretKind, SecretPurpose

    await _fresh_db()
    room_a, owner_a = uuid4(), uuid4()
    room_b, owner_b = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_a, owner_a)
        await _make_room(db, room_b, owner_b)
        svc = _service(db)

        await vps_inventory.add_server(
            db, service=svc, room_id=room_a, actor_user_id=owner_a,
            name="prod", host="203.0.113.10", root_password="room-a-root-password",
        )
        await db.commit()
        await vps_inventory.add_server(
            db, service=svc, room_id=room_b, actor_user_id=owner_b,
            name="prod", host="198.51.100.20", root_password="room-b-root-password",
        )
        await db.commit()

        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_a, actor_user_id=owner_a,
            env_name="VPS_PROD_PASSWORD", value="room-a-everyday-password",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_b, actor_user_id=owner_b,
            env_name="VPS_PROD_PASSWORD", value="room-b-everyday-password",
            purpose=SecretPurpose.VPS_ACCESS, kind_override=SecretKind.ROOM_ENV,
        )
        await db.commit()

        # list_servers: each Room sees only its own "PROD", with its own host.
        rows_a = await vps_inventory.list_servers(db, room_a)
        rows_b = await vps_inventory.list_servers(db, room_b)
        assert [r["name"] for r in rows_a] == ["PROD"]
        assert [r["name"] for r in rows_b] == ["PROD"]
        assert rows_a[0]["host"] == "203.0.113.10"
        assert rows_b[0]["host"] == "198.51.100.20"

        # resolve_connection: each Room's actor resolving "prod" gets only
        # its own host/credentials, never the other Room's.
        conn_a = await vps_inventory.resolve_connection(
            db, service=svc, room_id=room_a, actor_user_id=owner_a, server="prod",
        )
        conn_b = await vps_inventory.resolve_connection(
            db, service=svc, room_id=room_b, actor_user_id=owner_b, server="prod",
        )
        assert conn_a.legion_conn["host"] == "203.0.113.10"
        assert conn_a.legion_conn["auth"]["password"] == "room-a-everyday-password"
        assert conn_b.legion_conn["host"] == "198.51.100.20"
        assert conn_b.legion_conn["auth"]["password"] == "room-b-everyday-password"

        # forget-root: forgetting Room A's root credential must not touch
        # Room B's, and Room B's server must still be resolvable afterwards.
        await vps_inventory.forget_root_credentials(
            db, service=svc, room_id=room_a, actor_user_id=owner_a, name="prod",
        )
        await db.commit()
        rows_a_after = await vps_inventory.list_servers(db, room_a)
        rows_b_after = await vps_inventory.list_servers(db, room_b)
        assert rows_a_after[0]["has_root_password"] is False
        assert rows_b_after[0]["has_root_password"] is True
        break

    await engine.dispose()