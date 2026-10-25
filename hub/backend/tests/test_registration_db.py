"""Registration, owner and invitations against a real Postgres (internal design notes stage 4.3).

Gated on INIT_DB_TEST_URL like the other database tests:

    docker run -d --rm --name pg_registration_test -p 55432:5432 \
        -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=somemaru postgres:15-alpine
    INIT_DB_TEST_URL="postgresql+asyncpg://postgres:postgres@localhost:55432/somemaru" \
        pytest tests/test_registration_db.py -v
"""

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

TEST_DB_URL = os.getenv("INIT_DB_TEST_URL")

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not TEST_DB_URL, reason="set INIT_DB_TEST_URL to a disposable Postgres database to run"),
]

if TEST_DB_URL:
    os.environ["DATABASE_URL"] = TEST_DB_URL


@pytest.fixture(autouse=True)
def _mode(monkeypatch):
    from app.core import registration

    monkeypatch.delenv("LEGION_CONFIG", raising=False)
    monkeypatch.delenv("LEGION_REGISTRATION", raising=False)
    registration.reset_mode()
    yield registration
    registration.reset_mode()


@pytest_asyncio.fixture(autouse=True)
async def _release_connections():
    """Each test runs in its own event loop; pooled asyncpg connections cannot cross loops."""
    yield
    from app.db.session import engine

    await engine.dispose()


async def _fresh_db():
    from sqlmodel import SQLModel

    import app.models  # noqa: F401
    from app.db.session import engine, init_db

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
    await init_db()


def _create(email: str, **extra):
    from app.schemas.user import UserCreate

    return UserCreate(email=email, password="correct horse battery", **extra)


async def _register(email: str, **extra):
    from app.api.v1 import auth
    from app.db.session import get_session

    async for db in get_session():
        return await auth.register(_create(email, **extra), db=db)


async def _config():
    from app.api.v1 import auth
    from app.db.session import get_session

    async for db in get_session():
        return (await auth.auth_config(db=db))["registration"]


@pytest.mark.asyncio
async def test_first_user_owns_the_hub_and_the_door_closes():
    from fastapi import HTTPException

    await _fresh_db()
    assert await _config() == "first_user"

    owner = await _register("owner@example.com")
    assert owner.is_owner is True
    assert await _config() == "closed"

    with pytest.raises(HTTPException) as exc:
        await _register("guest@example.com")
    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == "registration_closed"


@pytest.mark.asyncio
async def test_open_mode_admits_guests_who_are_not_owners(monkeypatch):
    await _fresh_db()
    monkeypatch.setenv("LEGION_REGISTRATION", "open")
    await _register("owner@example.com")
    guest = await _register("guest@example.com")
    assert guest.is_owner is False
    assert await _config() == "open"


@pytest.mark.asyncio
async def test_two_first_registrations_make_one_owner():
    from sqlalchemy import func, select

    from app.db.session import get_session
    from app.models.user import User

    await _fresh_db()
    results = await asyncio.gather(
        _register("a@example.com"), _register("b@example.com"), return_exceptions=True,
    )
    owners = [r for r in results if not isinstance(r, Exception) and r.is_owner]
    refused = [r for r in results if isinstance(r, Exception)]
    assert len(owners) == 1 and len(refused) == 1
    async for db in get_session():
        count = (await db.execute(select(func.count()).select_from(User).where(User.is_owner.is_(True)))).scalar()
    assert count == 1


@pytest.mark.asyncio
async def test_the_database_allows_only_one_owner():
    from sqlalchemy.exc import IntegrityError

    from app.core import security
    from app.db.session import get_session
    from app.models.user import User

    await _fresh_db()
    await _register("owner@example.com")
    async for db in get_session():
        db.add(User(email="usurper@example.com", hashed_password=security.get_password_hash("x"), is_owner=True))
        with pytest.raises(IntegrityError):
            await db.commit()


@pytest.mark.asyncio
async def test_invitations_end_to_end(monkeypatch):
    from fastapi import HTTPException

    from app.api.v1 import registration as invites
    from app.db.session import get_session

    await _fresh_db()
    monkeypatch.setenv("LEGION_REGISTRATION", "invite")
    owner = await _register("owner@example.com")

    # Without a token, with a made-up one: refused.
    with pytest.raises(HTTPException) as exc:
        await _register("a@example.com")
    assert exc.value.detail["code"] == "invite_required"
    with pytest.raises(HTTPException) as exc:
        await _register("a@example.com", invite_token="lgi_nope")
    assert exc.value.detail["code"] == "invite_invalid"

    async for db in get_session():
        made = await invites.create_invite(invites.InviteCreate(), db=db, owner=owner)
        for_bob = await invites.create_invite(invites.InviteCreate(email="bob@example.com"), db=db, owner=owner)
    token = made["token"]

    # The token is the only thing that works, once.
    user = await _register("a@example.com", invite_token=token)
    assert user.is_owner is False
    with pytest.raises(HTTPException) as exc:
        await _register("b@example.com", invite_token=token)
    assert exc.value.detail["code"] == "invite_invalid"

    # An invitation for one address is useless to another.
    with pytest.raises(HTTPException):
        await _register("mallory@example.com", invite_token=for_bob["token"])
    assert (await _register("Bob@Example.com", invite_token=for_bob["token"])).is_owner is False

    async for db in get_session():
        listed = await invites.list_invites(db=db, owner=owner)
    assert sorted(i["status"] for i in listed) == ["used", "used"]
    assert all("token" not in i for i in listed)


@pytest.mark.asyncio
async def test_a_taken_address_does_not_burn_the_invitation(monkeypatch):
    from fastapi import HTTPException

    from app.api.v1 import registration as invites
    from app.db.session import get_session

    await _fresh_db()
    monkeypatch.setenv("LEGION_REGISTRATION", "invite")
    owner = await _register("owner@example.com")
    async for db in get_session():
        made = await invites.create_invite(invites.InviteCreate(), db=db, owner=owner)
    with pytest.raises(HTTPException) as exc:
        await _register("owner@example.com", invite_token=made["token"])
    assert exc.value.status_code == 400
    assert (await _register("fresh@example.com", invite_token=made["token"])).email == "fresh@example.com"


@pytest.mark.asyncio
async def test_revoked_and_expired_invitations_do_not_work(monkeypatch):
    from fastapi import HTTPException
    from sqlalchemy import update

    from app.api.v1 import registration as invites
    from app.db.session import get_session
    from app.models.registration_invite import RegistrationInvite

    await _fresh_db()
    monkeypatch.setenv("LEGION_REGISTRATION", "invite")
    owner = await _register("owner@example.com")
    async for db in get_session():
        revoked = await invites.create_invite(invites.InviteCreate(), db=db, owner=owner)
        expired = await invites.create_invite(invites.InviteCreate(), db=db, owner=owner)
        await invites.revoke_invite(revoked["id"], db=db, owner=owner)
        await db.execute(update(RegistrationInvite).where(RegistrationInvite.id == expired["id"])
                         .values(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1)))
        await db.commit()
        with pytest.raises(HTTPException) as exc:  # a revoked one cannot be revoked again
            await invites.revoke_invite(revoked["id"], db=db, owner=owner)
        assert exc.value.status_code == 409
    for token in (revoked["token"], expired["token"]):
        with pytest.raises(HTTPException) as exc:
            await _register("late@example.com", invite_token=token)
        assert exc.value.detail["code"] == "invite_invalid"


@pytest.mark.asyncio
async def test_invitations_are_off_unless_the_mode_is_invite():
    from fastapi import HTTPException

    from app.api.v1 import registration as invites
    from app.db.session import get_session

    await _fresh_db()
    owner = await _register("owner@example.com")
    async for db in get_session():
        with pytest.raises(HTTPException) as exc:
            await invites.create_invite(invites.InviteCreate(), db=db, owner=owner)
    assert exc.value.status_code == 409 and exc.value.detail["code"] == "invites_off"


@pytest.mark.asyncio
async def test_only_the_owner_manages_invitations():
    from fastapi import HTTPException

    from app.api import deps

    await _fresh_db()
    owner = await _register("owner@example.com")
    assert await deps.get_current_owner(owner) is owner
    owner.is_owner = False
    with pytest.raises(HTTPException) as exc:
        await deps.get_current_owner(owner)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_owner_from_the_environment_for_a_hub_that_predates_owners(monkeypatch):
    from app.core import registration
    from app.db.session import get_session
    from app.models.user import User

    await _fresh_db()
    monkeypatch.setenv("LEGION_REGISTRATION", "open")
    first = await _register("first@example.com")
    second = await _register("second@example.com")
    async for db in get_session():  # a Hub from before owners: nobody has the flag
        for u in (first, second):
            row = await db.get(User, u.id)
            row.is_owner = False
        await db.commit()
        assert await registration.claim_owner_from_env(db, {}) is None
        assert await registration.claim_owner_from_env(db, {"LEGION_OWNER_EMAIL": "nobody@example.com"}) is None
        claimed = await registration.claim_owner_from_env(db, {"LEGION_OWNER_EMAIL": "Second@Example.com"})
        assert claimed.email == "second@example.com"
        # Already has an owner: the variable changes nothing.
        assert await registration.claim_owner_from_env(db, {"LEGION_OWNER_EMAIL": "first@example.com"}) is None


# --- legion.toml [[personal_mcp]] (stage 4.6) ---------------------------------

def _declare(monkeypatch, tmp_path, body: str):
    from app.core import personal_mcp_config

    path = tmp_path / "legion.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("LEGION_CONFIG", str(path))
    personal_mcp_config.reset_declared()
    return personal_mcp_config


async def _mcp_names(user_id):
    from sqlalchemy import select

    from app.db.session import get_session
    from app.models.market import UserInstalledItem

    async for db in get_session():
        rows = (await db.execute(select(UserInstalledItem).where(
            UserInstalledItem.user_id == user_id, UserInstalledItem.type == "mcp",
            UserInstalledItem.is_custom.is_(True)))).scalars().all()
        return {r.name: r.custom_config for r in rows}


@pytest.mark.asyncio
async def test_declared_mcp_is_installed_for_the_owner_when_they_register(monkeypatch, tmp_path):
    pmc = _declare(monkeypatch, tmp_path, '''
[[personal_mcp]]
name = "docs"
url = "http://docs-mcp:8080/mcp"
[[personal_mcp]]
name = "notes"
image = "ghcr.io/owner/notes-mcp:1"
''')
    try:
        await _fresh_db()
        monkeypatch.setenv("LEGION_REGISTRATION", "open")
        owner = await _register("owner@example.com")
        guest = await _register("guest@example.com")
        assert await _mcp_names(owner.id) == {
            "docs": {"url": "http://docs-mcp:8080/mcp"},
            "notes": {"mode": "hosted", "image": "ghcr.io/owner/notes-mcp:1", "port": 8000},
        }
        assert await _mcp_names(guest.id) == {}  # only the owner gets them
    finally:
        pmc.reset_declared()


@pytest.mark.asyncio
async def test_startup_installs_missing_servers_once_and_leaves_edits_alone(monkeypatch, tmp_path):
    from app.db.session import get_session
    from app.models.market import UserInstalledItem

    pmc = _declare(monkeypatch, tmp_path, '[[personal_mcp]]\nname = "docs"\nurl = "http://docs-mcp:8080/mcp"\n')
    try:
        await _fresh_db()
        # A Hub that already has an owner but was started without the file.
        owner = await _register("owner@example.com")
        async for db in get_session():
            for item in (await _mcp_rows(db, owner.id)):
                await db.delete(item)
            await db.commit()
        assert await _mcp_names(owner.id) == {}

        async for db in get_session():
            assert await pmc.apply_for_owner(db) == ["docs"]
            assert await pmc.apply_for_owner(db) == []  # second start: nothing to do
        # The owner edits the server in Settings; the next start does not undo it.
        async for db in get_session():
            item = (await _mcp_rows(db, owner.id))[0]
            item.custom_config = {"url": "http://elsewhere:9000/mcp"}
            await db.commit()
            assert await pmc.apply_for_owner(db) == []
        assert (await _mcp_names(owner.id))["docs"] == {"url": "http://elsewhere:9000/mcp"}
    finally:
        pmc.reset_declared()


async def _mcp_rows(db, user_id):
    from sqlalchemy import select

    from app.models.market import UserInstalledItem

    return (await db.execute(select(UserInstalledItem).where(
        UserInstalledItem.user_id == user_id, UserInstalledItem.type == "mcp",
        UserInstalledItem.is_custom.is_(True)))).scalars().all()
