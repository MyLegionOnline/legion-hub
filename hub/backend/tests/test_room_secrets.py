"""Stage 2 tests for internal design notes — Room secrets (backend).

Covers ``app/core/secrets/room_secrets.py``:
  * bind → list (metadata only, never a value) → resolve → unbind;
  * §2.3 — a member may *use* another member's secret but not replace its
    value or unbind it, and the value resolves under the binding owner;
  * §2.4 — an exact-environment binding overrides the shared (NULL) one;
  * scope filtering and the value-free ``secret_not_available`` failure;
  * the room-delete path detaches bindings but leaves ``user_secret`` intact.

Gated on INIT_DB_TEST_URL (a disposable Postgres) and PyNaCl, exactly like
tests/test_room_variables.py and tests/test_secret_operations.py.

    docker run -d --rm --name pg_roomsec_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_room_secrets.py -v
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


async def _make_room(room_id, creator_id, *members):
    from app.db.session import get_session
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    async for session in get_session():
        session.add(
            User(id=creator_id, email=f"rs-{creator_id}@example.com", hashed_password="x")
        )
        for m in members:
            session.add(
                User(id=m, email=f"rs-{m}@example.com", hashed_password="x")
            )
        session.add(Room(id=room_id, name="rs", creator_id=creator_id))
        await session.flush()
        session.add(RoomUserLink(room_id=room_id, user_id=creator_id))
        for m in members:
            session.add(RoomUserLink(room_id=room_id, user_id=m))
        await session.commit()
        break


@pytest.mark.asyncio
async def test_bind_list_resolve_unbind_roundtrip():
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", value="s3cr3t-token-value", scope=["deploy"],
        )
        await db.commit()

        rows = await room_secrets.list_room_secrets(
            db, room_id=room_id, actor_user_id=owner_id
        )
        assert len(rows) == 1
        assert rows[0]["env_name"] == "DEPLOY_TOKEN"
        assert rows[0]["scope"] == ["deploy"]
        assert rows[0]["is_own"] is True
        assert rows[0]["last_used_at"] is None
        # the metadata dict never carries a value
        assert "value" not in rows[0]

        value = await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", scope="deploy",
        )
        assert value == "s3cr3t-token-value"
        await db.commit()

        rows = await room_secrets.list_room_secrets(
            db, room_id=room_id, actor_user_id=owner_id
        )
        assert rows[0]["last_used_at"] is not None

        # wrong scope → value-free not-available
        with pytest.raises(room_secrets.RoomSecretNotAvailable):
            await room_secrets.resolve_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DEPLOY_TOKEN", scope="services",
            )
        # unknown name → same error class
        with pytest.raises(room_secrets.RoomSecretNotAvailable):
            await room_secrets.resolve_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="NOPE",
            )

        binding_id = rows[0]["binding_id"]
        await room_secrets.unbind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            binding_id=binding_id,
        )
        await db.commit()
        assert await room_secrets.list_room_secrets(
            db, room_id=room_id, actor_user_id=owner_id
        ) == []
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_member_uses_but_cannot_change_anothers_secret():
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id, other_id = uuid4(), uuid4(), uuid4()
    await _make_room(room_id, owner_id, other_id)

    async for db in get_session():
        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="API_KEY", value="another-members-value",
        )
        await db.commit()

        # the other member can resolve it (resolves under the owner)
        value = await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=other_id,
            env_name="API_KEY",
        )
        assert value == "another-members-value"
        await db.commit()

        rows = await room_secrets.list_room_secrets(
            db, room_id=room_id, actor_user_id=other_id
        )
        assert rows[0]["is_own"] is False
        bid = rows[0]["binding_id"]

        with pytest.raises(room_secrets.RoomSecretAccessDenied):
            await room_secrets.replace_room_secret_value(
                db, service=svc, room_id=room_id, actor_user_id=other_id,
                binding_id=bid, value="hijacked-value",
            )
        with pytest.raises(room_secrets.RoomSecretAccessDenied):
            await room_secrets.unbind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=other_id,
                binding_id=bid,
            )

        # a non-member gets nothing
        with pytest.raises(room_secrets.RoomSecretAccessDenied):
            await room_secrets.resolve_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=uuid4(),
                env_name="API_KEY",
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_environment_exact_match_beats_shared():
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DATABASE_URL", value="shared-connection-string",
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DATABASE_URL", value="staging-connection-string",
            environment="staging",
        )
        await db.commit()

        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DATABASE_URL", environment="staging",
        ) == "staging-connection-string"
        await db.commit()

        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DATABASE_URL", environment="production",
        ) == "shared-connection-string"
        await db.commit()

        assert await room_secrets.resolve_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DATABASE_URL",
        ) == "shared-connection-string"
        await db.commit()

        # same env_name + environment twice → conflict
        with pytest.raises(room_secrets.RoomSecretExists):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DATABASE_URL", value="another-staging-value",
                environment="staging",
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_short_value_rejected_and_batch_resolve():
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        with pytest.raises(room_secrets.RoomSecretInvalid):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="TINY", value="short",
            )
        for name in ("HOST_A", "HOST_B"):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name=name, value=f"{name}-value-long-enough",
            )
        await db.commit()

        got = await room_secrets.resolve_room_env(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            names=["HOST_A", "HOST_B"],
        )
        assert got == {
            "HOST_A": "HOST_A-value-long-enough",
            "HOST_B": "HOST_B-value-long-enough",
        }
        await db.commit()

        with pytest.raises(room_secrets.RoomSecretNotAvailable):
            await room_secrets.resolve_room_env(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                names=["HOST_A", "MISSING"],
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_room_delete_detaches_bindings_keeps_secret():
    from sqlalchemy import text

    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)

    async for db in get_session():
        svc = _service(db)
        binding = await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", value="s3cr3t-token-value",
        )
        secret_id = binding.secret_id
        await db.commit()

        await room_secrets.unbind_all_for_room(db, room_id)
        await db.commit()

        bindings = (
            await db.execute(
                text("SELECT count(*) FROM secret_binding WHERE resource_id = :r"),
                {"r": room_id},
            )
        ).scalar_one()
        assert bindings == 0
        secrets = (
            await db.execute(
                text("SELECT count(*) FROM user_secret WHERE id = :s"),
                {"s": secret_id},
            )
        ).scalar_one()
        assert secrets == 1
        break


def _generate_ed25519_openssh_private_key() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


@pytest.mark.asyncio
async def test_deploy_ssh_key_fingerprint_scope_and_mixed_resolve():
    """internal design notes Stage 5 — deploy keys."""
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)
    private_key_pem = _generate_ed25519_openssh_private_key()

    async for db in get_session():
        svc = _service(db)

        # rejects anything that isn't a real OpenSSH private key
        with pytest.raises(room_secrets.RoomSecretInvalid):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="BAD_KEY", value="not-an-ssh-key-but-long-enough",
                purpose=room_secrets.SecretPurpose.DEPLOY_SSH_KEY,
            )

        binding = await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="SSH_DEPLOY_KEY", value=private_key_pem,
            purpose=room_secrets.SecretPurpose.DEPLOY_SSH_KEY,
        )
        await db.commit()

        rows = await room_secrets.list_room_secrets(
            db, room_id=room_id, actor_user_id=owner_id
        )
        row = next(r for r in rows if r["binding_id"] == binding.id)
        # scope defaults to ["deploy"] without the caller asking for it
        assert row["scope"] == ["deploy"]
        assert row["public_key"].startswith("ssh-ed25519 ")
        assert row["fingerprint"].startswith("SHA256:")
        # never the private key itself
        assert "BEGIN OPENSSH PRIVATE KEY" not in str(row)

        env_values, ssh_values = await room_secrets.resolve_room_secrets_mixed(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            names=["SSH_DEPLOY_KEY"],
        )
        assert env_values == {}
        assert ssh_values == {"SSH_DEPLOY_KEY": private_key_pem}
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_room_secrets_mixed_covers_plain_and_ssh_names():
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()
    await _make_room(room_id, owner_id)
    private_key_pem = _generate_ed25519_openssh_private_key()

    async for db in get_session():
        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_TOKEN", value="s3cr3t-token-value",
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="SSH_DEPLOY_KEY", value=private_key_pem,
            purpose=room_secrets.SecretPurpose.DEPLOY_SSH_KEY,
        )
        await db.commit()

        env_values, ssh_values = await room_secrets.resolve_room_secrets_mixed(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            names=["DEPLOY_TOKEN", "SSH_DEPLOY_KEY"],
        )
        assert env_values == {"DEPLOY_TOKEN": "s3cr3t-token-value"}
        assert ssh_values == {"SSH_DEPLOY_KEY": private_key_pem}

        with pytest.raises(room_secrets.RoomSecretNotAvailable):
            await room_secrets.resolve_room_secrets_mixed(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                names=["DEPLOY_TOKEN", "MISSING"],
            )
        break

    await engine.dispose()
