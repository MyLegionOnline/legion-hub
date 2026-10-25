"""Stage 7 tests for internal design notes — Room secrets × MCP.

Covers:
  * §3.4 / ``authorize_target`` — binding ``room_mcp_env`` / ``room_mcp_auth_header``
    requires ``mcp_item_id`` to name an MCP item the *binder* owns, not just
    Room membership;
  * ``bind_room_secret`` requiring/forbidding ``mcp_item_id`` per purpose;
  * ``lease.py``'s ``issue(room_id=...)`` folding ``(room, room_mcp_env)``
    bindings into a managed-stdio lease alongside the item owner's own
    ``mcp_env`` bindings, each resolved under its own binding owner (§2.3),
    and silently skipping Room secrets when the item isn't attached to that
    Room via ``RoomMcpServer``;
  * the lease staying one-time-use and rejecting an expired token — same
    invariants as the Stage 3 personal-only lease, now exercised through the
    Stage 7 code paths;
  * ``resolve_room_mcp_auth_header`` for the HTTP-link route.

Gated on INIT_DB_TEST_URL (a disposable Postgres) and PyNaCl, exactly like
tests/test_room_secrets.py.

    docker run -d --rm --name pg_roommcp_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_room_mcp_secrets.py -v
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


async def _make_room(db, room_id, creator_id, *members):
    from app.models.room import Room, RoomUserLink
    from app.models.user import User

    db.add(User(id=creator_id, email=f"rms-{creator_id}@example.com", hashed_password="x"))
    for m in members:
        db.add(User(id=m, email=f"rms-{m}@example.com", hashed_password="x"))
    db.add(Room(id=room_id, name="rms", creator_id=creator_id))
    await db.flush()
    db.add(RoomUserLink(room_id=room_id, user_id=creator_id))
    for m in members:
        db.add(RoomUserLink(room_id=room_id, user_id=m))
    await db.commit()


async def _installed_item(db, *, user_id):
    from app.models.market import UserInstalledItem

    item = UserInstalledItem(user_id=user_id, type="mcp", name="test-mcp")
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item.id


async def _runtime(db, *, owner_id, item_id, env_names):
    from app.models.personal_mcp import PersonalMcpRuntime

    runtime = PersonalMcpRuntime(
        installed_item_id=item_id,
        owner_id=owner_id,
        preview_id=uuid4(),
        candidate_id=uuid4(),
        launch_spec={"package": "test-pkg", "command_args": ["mcp"], "env_names": env_names},
    )
    db.add(runtime)
    await db.commit()
    return runtime


async def _bind_personal_mcp_env(db, service, *, owner_id, item_id, env_name, value):
    from app.core.secrets.crypto import SecretValue
    from app.core.secrets.store import BindingTarget
    from app.models.secret import SecretKind, SecretPurpose, SecretResourceType

    meta = await service.create(
        owner_id=owner_id, kind=SecretKind.MCP_ENV, label=env_name, provider=None,
        value=SecretValue(value),
    )
    await service.bind(
        owner_id=owner_id,
        secret_id=meta.id,
        target=BindingTarget(
            resource_type=SecretResourceType.INSTALLED_MCP,
            resource_id=item_id,
            purpose=SecretPurpose.MCP_ENV,
            injection_config={"name": env_name},
        ),
    )
    await db.commit()


@pytest.mark.asyncio
async def test_authorize_target_and_bind_require_owned_mcp_item():
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id, other_id = uuid4(), uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, other_id)
        own_item_id = await _installed_item(db, user_id=owner_id)
        foreign_item_id = await _installed_item(db, user_id=other_id)
        svc = _service(db)

        # mcp_item_id required for the two MCP-scoped purposes.
        with pytest.raises(room_secrets.RoomSecretInvalid):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="MISSING_ITEM", value="value-long-enough",
                purpose=room_secrets.SecretPurpose.ROOM_MCP_ENV,
            )
        # ...and forbidden for a purpose that isn't MCP-scoped.
        with pytest.raises(room_secrets.RoomSecretInvalid):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="PLAIN", value="value-long-enough", mcp_item_id=own_item_id,
            )
        # someone else's installed_item_id does not pass, even for a member
        # binding into their own Room.
        with pytest.raises(room_secrets.RoomSecretAccessDenied):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="STOLEN", value="value-long-enough", mcp_item_id=foreign_item_id,
                purpose=room_secrets.SecretPurpose.ROOM_MCP_ENV,
            )

        binding = await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_ENV_TOKEN", value="value-long-enough", mcp_item_id=own_item_id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_ENV,
        )
        await db.commit()
        rows = await room_secrets.list_room_secrets(db, room_id=room_id, actor_user_id=owner_id)
        row = next(r for r in rows if r["binding_id"] == binding.id)
        assert row["mcp_item_id"] == str(own_item_id)
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_resolve_room_mcp_auth_header_roundtrip():
    from app.core.secrets import room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id = uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id)
        item_id = await _installed_item(db, user_id=owner_id)
        other_item_id = await _installed_item(db, user_id=owner_id)
        svc = _service(db)

        # no binding yet → silent no-op, not an error.
        assert await room_secrets.resolve_room_mcp_auth_header(
            db, service=svc, room_id=room_id, mcp_item_id=item_id, actor_user_id=owner_id,
        ) is None

        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="DEPLOY_MCP_TOKEN", value="bearer-token-value", mcp_item_id=item_id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_AUTH_HEADER,
        )
        await db.commit()

        got = await room_secrets.resolve_room_mcp_auth_header(
            db, service=svc, room_id=room_id, mcp_item_id=item_id, actor_user_id=owner_id,
        )
        assert got == ("DEPLOY_MCP_TOKEN", "bearer-token-value")

        # a different MCP item in the same Room still gets nothing.
        assert await room_secrets.resolve_room_mcp_auth_header(
            db, service=svc, room_id=room_id, mcp_item_id=other_item_id, actor_user_id=owner_id,
        ) is None
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_lease_issue_merges_room_env_under_its_own_owner():
    """Stage 7: ``authorize_target`` requires the *binder* to own the MCP
    item (§3.4), so a ``room_mcp_env`` binding's owner is always the item
    owner too — the point of the merge is drawing one lease from two
    different stores (the item's own ``mcp_env`` slot plus a Room-shared
    one), not cross-member sharing of someone else's item.
    """
    from app.core.secrets import room_secrets
    from app.core.secrets.lease import ManagedMcpSecretLeaseService
    from app.db.session import engine, get_session
    from app.models.market import RoomMcpServer

    await _fresh_db()
    room_id, owner_id, other_id = uuid4(), uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, other_id)
        item_id = await _installed_item(db, user_id=owner_id)
        db.add(RoomMcpServer(room_id=room_id, user_installed_item_id=item_id, added_by=owner_id))
        await db.commit()
        await _runtime(db, owner_id=owner_id, item_id=item_id, env_names=["OWN_KEY", "SHARED_KEY"])

        svc = _service(db)
        await _bind_personal_mcp_env(
            db, svc, owner_id=owner_id, item_id=item_id, env_name="OWN_KEY", value="own-secret-value",
        )
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="SHARED_KEY", value="shared-secret-value", mcp_item_id=item_id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_ENV,
        )
        await db.commit()

        lease_service = ManagedMcpSecretLeaseService(db, store=svc.store)
        lease = await lease_service.issue(
            owner_id=owner_id, resource_id=item_id, runtime_revision=1, room_id=room_id,
        )
        await db.commit()
        assert set(lease.env_names) == {"OWN_KEY", "SHARED_KEY"}

        consumed = await lease_service.consume(
            owner_id=owner_id, resource_id=item_id, lease_token=lease.lease_token,
        )
        await db.commit()
        values = {name: value.reveal() for name, value in consumed.values.items()}
        assert values == {"OWN_KEY": "own-secret-value", "SHARED_KEY": "shared-secret-value"}
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_lease_issue_skips_room_env_when_item_not_attached():
    from app.core.secrets import room_secrets
    from app.core.secrets.lease import ManagedMcpSecretLeaseService
    from app.db.session import engine, get_session

    await _fresh_db()
    room_id, owner_id, other_id = uuid4(), uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, other_id)
        item_id = await _installed_item(db, user_id=owner_id)
        # deliberately no RoomMcpServer attachment
        await _runtime(db, owner_id=owner_id, item_id=item_id, env_names=["SHARED_KEY"])

        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="SHARED_KEY", value="shared-secret-value", mcp_item_id=item_id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_ENV,
        )
        await db.commit()

        lease_service = ManagedMcpSecretLeaseService(db, store=svc.store)
        lease = await lease_service.issue(
            owner_id=owner_id, resource_id=item_id, runtime_revision=1, room_id=room_id,
        )
        await db.commit()
        assert lease.env_names == ()
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_lease_issue_raises_environment_locked_when_room_env_protected():
    """internal design notes Stage 0 — ``issue(room_id=..., environment=...)`` must
    raise the same ``EnvironmentLocked`` ``resolve_room_secret`` does, and
    must not leave a ``SecretLease`` row behind when it does."""
    from sqlalchemy import select as sa_select

    from app.core.secrets import room_environment_gate, room_secrets
    from app.core.secrets.lease import ManagedMcpSecretLeaseService
    from app.db.session import engine, get_session
    from app.models.market import RoomMcpServer
    from app.models.secret import SecretLease

    await _fresh_db()
    room_id, owner_id, other_id = uuid4(), uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, other_id)
        item_id = await _installed_item(db, user_id=owner_id)
        db.add(RoomMcpServer(room_id=room_id, user_installed_item_id=item_id, added_by=owner_id))
        await db.commit()
        await _runtime(db, owner_id=owner_id, item_id=item_id, env_names=["SHARED_KEY"])

        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="SHARED_KEY", value="shared-secret-value", mcp_item_id=item_id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_ENV, environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True
        )
        await db.commit()

        lease_service = ManagedMcpSecretLeaseService(db, store=svc.store)
        with pytest.raises(room_secrets.EnvironmentLocked):
            await lease_service.issue(
                owner_id=owner_id, resource_id=item_id, runtime_revision=1,
                room_id=room_id, environment="production",
            )

        leases = (await db.execute(sa_select(SecretLease))).scalars().all()
        assert leases == []
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_lease_issue_succeeds_after_unlock_for_protected_room_env():
    from app.core.secrets import room_environment_gate, room_secrets
    from app.core.secrets.lease import ManagedMcpSecretLeaseService
    from app.db.session import engine, get_session
    from app.models.market import RoomMcpServer

    await _fresh_db()
    room_id, owner_id, other_id = uuid4(), uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_id, owner_id, other_id)
        item_id = await _installed_item(db, user_id=owner_id)
        db.add(RoomMcpServer(room_id=room_id, user_installed_item_id=item_id, added_by=owner_id))
        await db.commit()
        await _runtime(db, owner_id=owner_id, item_id=item_id, env_names=["SHARED_KEY"])

        svc = _service(db)
        await room_secrets.bind_room_secret(
            db, service=svc, room_id=room_id, actor_user_id=owner_id,
            env_name="SHARED_KEY", value="shared-secret-value", mcp_item_id=item_id,
            purpose=room_secrets.SecretPurpose.ROOM_MCP_ENV, environment="production",
        )
        await room_environment_gate.set_protected(
            db, room_id=room_id, environment="production", protected=True
        )
        await room_environment_gate.grant_unlock(
            db, room_id=room_id, environment="production", actor_user_id=owner_id,
        )
        await db.commit()

        lease_service = ManagedMcpSecretLeaseService(db, store=svc.store)
        lease = await lease_service.issue(
            owner_id=owner_id, resource_id=item_id, runtime_revision=1,
            room_id=room_id, environment="production",
        )
        await db.commit()
        assert lease.env_names == ("SHARED_KEY",)
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_environment_lock_scoped_per_room_for_mcp_auth_header():
    """An unlock granted for (room A, production) must not unlock (room B,
    production), even for the same MCP item's ``room_mcp_auth_header``."""
    from app.core.secrets import room_environment_gate, room_secrets
    from app.db.session import engine, get_session

    await _fresh_db()
    room_a, room_b, owner_id = uuid4(), uuid4(), uuid4()

    async for db in get_session():
        await _make_room(db, room_a, owner_id)
        from app.models.room import Room, RoomUserLink

        db.add(Room(id=room_b, name="rms-b", creator_id=owner_id))
        await db.flush()
        db.add(RoomUserLink(room_id=room_b, user_id=owner_id))
        await db.commit()

        item_id = await _installed_item(db, user_id=owner_id)
        svc = _service(db)
        for room_id in (room_a, room_b):
            await room_secrets.bind_room_secret(
                db, service=svc, room_id=room_id, actor_user_id=owner_id,
                env_name="DEPLOY_MCP_TOKEN", value="bearer-token-value", mcp_item_id=item_id,
                purpose=room_secrets.SecretPurpose.ROOM_MCP_AUTH_HEADER, environment="production",
            )
            await room_environment_gate.set_protected(
                db, room_id=room_id, environment="production", protected=True
            )
        await db.commit()

        await room_environment_gate.grant_unlock(
            db, room_id=room_a, environment="production", actor_user_id=owner_id,
        )

        got = await room_secrets.resolve_room_mcp_auth_header(
            db, service=svc, room_id=room_a, mcp_item_id=item_id, actor_user_id=owner_id,
            environment="production",
        )
        assert got == ("DEPLOY_MCP_TOKEN", "bearer-token-value")

        with pytest.raises(room_secrets.EnvironmentLocked):
            await room_secrets.resolve_room_mcp_auth_header(
                db, service=svc, room_id=room_b, mcp_item_id=item_id, actor_user_id=owner_id,
                environment="production",
            )
        break

    await engine.dispose()


@pytest.mark.asyncio
async def test_lease_is_one_time_use_and_rejects_expired():
    from sqlalchemy import select

    from app.core.secrets.lease import (
        ManagedMcpSecretLeaseService,
        SecretLeaseDenied,
        SecretLeaseExpired,
    )
    from app.db.session import engine, get_session
    from app.models.secret import SecretLease

    await _fresh_db()
    owner_id = uuid4()

    async for db in get_session():
        from app.models.user import User

        db.add(User(id=owner_id, email=f"rms-{owner_id}@example.com", hashed_password="x"))
        await db.commit()
        item_id = await _installed_item(db, user_id=owner_id)
        await _runtime(db, owner_id=owner_id, item_id=item_id, env_names=[])

        svc = _service(db)
        lease_service = ManagedMcpSecretLeaseService(db, store=svc.store)
        lease = await lease_service.issue(owner_id=owner_id, resource_id=item_id, runtime_revision=1)
        await db.commit()

        await lease_service.consume(owner_id=owner_id, resource_id=item_id, lease_token=lease.lease_token)
        await db.commit()
        with pytest.raises(SecretLeaseDenied):
            await lease_service.consume(
                owner_id=owner_id, resource_id=item_id, lease_token=lease.lease_token,
            )

        lease2 = await lease_service.issue(owner_id=owner_id, resource_id=item_id, runtime_revision=1)
        await db.commit()
        row = (
            await db.execute(select(SecretLease).where(SecretLease.id == lease2.lease_id))
        ).scalar_one()
        row.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        db.add(row)
        await db.commit()
        with pytest.raises(SecretLeaseExpired):
            await lease_service.consume(
                owner_id=owner_id, resource_id=item_id, lease_token=lease2.lease_token,
            )
        break

    await engine.dispose()
