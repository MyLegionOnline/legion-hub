"""A self-hosted Hub user links a Legion account with an API key (internal design notes stage 6).

Unit part: the key format, what Market's answers mean, the API switch by
AUTH_MODE. DB part (PostgreSQL, marked `db`): the key is stored encrypted and
shows the account's email; a key Market stops accepting turns ``rejected``;
Market down keeps the last answer; replace and unlink. They create users, so
run them against a scratch database only:

    DATABASE_URL=postgresql+asyncpg://.../legion_link_test pytest tests/test_legion_link.py

They skip unless the database name contains "test".
"""

from __future__ import annotations

import os
from typing import AsyncGenerator
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from app.core import legion_auth, legion_link
from app.core.legion_link import LegionLinkError
from app.core.secret_keyring import SecretKeyring

MARKET = "https://market.legion.test"
KEY = "lgn_" + "abcdefghijklmnopqrstuvwxyz234567abcdefgh"
OTHER_KEY = "lgn_" + "zyxwvutsrqponmlkjihgfedcba765432zyxwvuts"
# The Hub-side test client; Market fixtures replace httpx.AsyncClient.
_AsyncClient = httpx.AsyncClient
KEYRING = SecretKeyring(active_key_id="t", keys={"t": os.urandom(32)}, path="test")


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "local")
    monkeypatch.setattr(legion_auth, "LEGION_INTERNAL_URL", MARKET)
    monkeypatch.setattr(legion_auth, "LEGION_ISSUER", MARKET)
    monkeypatch.setattr(legion_link, "_keyring", lambda: KEYRING)


class Market:
    """Market's GET /api/account/me as the Hub sees it: a key -> account map."""

    def __init__(self, monkeypatch):
        self.accounts: dict[str, dict] = {}
        self.down = False
        self.seen: list[str] = []
        def client(**kwargs):
            return _AsyncClient(transport=httpx.MockTransport(self.handle), **kwargs)

        monkeypatch.setattr(legion_link.httpx, "AsyncClient", client)

    def issue(self, key: str, email: str, account_id: UUID | None = None) -> UUID:
        account_id = account_id or uuid4()
        self.accounts[key] = {"id": str(account_id), "email": email, "display_name": "Ada", "status": "active"}
        return account_id

    def handle(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("down", request=request)
        if request.url == f"{MARKET}/api/wallet":
            # The settings card also shows the Own Hubs wallet; not counted in ``seen``.
            account = self.accounts.get(request.headers.get("authorization", "").partition(" ")[2])
            if account is None:
                return httpx.Response(401, json={"detail": "API key is invalid or revoked"})
            return httpx.Response(200, json={"account": account, "wallet": {"balance": 0, "ledger": []}})
        assert request.url == f"{MARKET}/api/account/me"
        scheme, _, key = request.headers.get("authorization", "").partition(" ")
        assert scheme == "Bearer"
        self.seen.append(key)
        account = self.accounts.get(key)
        if account is None:
            return httpx.Response(401, json={"detail": "API key is invalid or revoked"})
        return httpx.Response(200, json={"account": account, "auth": "api_key"})


@pytest.fixture
def market(monkeypatch) -> Market:
    return Market(monkeypatch)


# ---------- unit ------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", "abc", "lgn_short", "sk-or-v1-" + "a" * 40, "lgn_" + "A" * 40, "lgn_" + "a" * 20 + " x"])
def test_only_legion_keys_are_accepted(value):
    with pytest.raises(LegionLinkError) as err:
        legion_link.normalize_key(value)
    assert err.value.code == "bad_key_format"


def test_a_pasted_key_is_trimmed():
    assert legion_link.normalize_key(f"  {KEY}\n") == KEY


@pytest.mark.asyncio
async def test_market_answers_map_to_the_account_or_a_refusal(market):
    account_id = market.issue(KEY, "ada@example.com")
    account = await legion_link.fetch_account(KEY)
    assert account["id"] == str(account_id) and account["email"] == "ada@example.com"

    with pytest.raises(LegionLinkError) as err:
        await legion_link.fetch_account(OTHER_KEY)
    assert (err.value.code, err.value.status) == ("key_rejected", 400)

    market.down = True
    with pytest.raises(LegionLinkError) as err:
        await legion_link.fetch_account(KEY)
    assert (err.value.code, err.value.status) == ("unreachable", 503)


@pytest.mark.asyncio
async def test_an_answer_that_is_not_an_api_key_one_is_refused(monkeypatch):
    body = {"account": {"id": str(uuid4()), "email": "a@b.test"}, "auth": "session"}
    monkeypatch.setattr(legion_link.httpx, "AsyncClient", lambda **kw: _AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)), **kw))
    with pytest.raises(LegionLinkError) as err:
        await legion_link.fetch_account(KEY)
    assert err.value.code == "bad_answer"


def test_the_key_is_bound_to_its_user():
    pytest.importorskip("nacl")
    from app.models.legion_link import LegionLink

    user_id = uuid4()
    ciphertext, key_id, aad_version = legion_link._encrypt(user_id, KEY)
    assert KEY.encode() not in ciphertext
    row = LegionLink(user_id=user_id, ciphertext=ciphertext, key_id=key_id, aad_version=aad_version,
                     key_prefix=KEY[:12], account_id=uuid4(), email="a@b.test")
    assert legion_link.stored_key(row).reveal() == KEY
    row.user_id = uuid4()  # the same ciphertext moved to another user's row
    with pytest.raises(LegionLinkError) as err:
        legion_link.stored_key(row)
    assert err.value.code == "storage_failed"


def _app(session_factory=None, user_id: UUID | None = None) -> FastAPI:
    from app.api import deps
    from app.api.v1 import legion_link as legion_link_api
    from app.db.session import get_session
    from app.models.user import User

    app = FastAPI()
    app.include_router(legion_link_api.router, prefix="/api/v1/legion-link")
    user = User(id=user_id or uuid4(), email="local@example.com", hashed_password="x")
    app.dependency_overrides[deps.get_current_user] = lambda: user

    async def _session():
        if session_factory is None:
            yield None
            return
        async with session_factory() as db:
            yield db

    app.dependency_overrides[get_session] = _session
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return _AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hub.test")


def test_the_link_state_points_to_the_documents_of_market():
    from app.api.v1 import legion_link as legion_link_api

    state = legion_link_api._state(None)
    assert state["legal_urls"] == {
        "terms": f"{MARKET}/terms",
        "privacy": f"{MARKET}/privacy",
        "acceptable_use": f"{MARKET}/acceptable-use",
    }


@pytest.mark.asyncio
async def test_a_hub_signing_in_with_legion_has_nothing_to_link(monkeypatch):
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "legion")
    async with _client(_app()) as client:
        state = (await client.get("/api/v1/legion-link?refresh=true")).json()
        assert state["available"] is False and state["link"] is None
        response = await client.put("/api/v1/legion-link", json={"key": KEY})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "legion_sign_in"


@pytest.mark.asyncio
async def test_a_bad_key_is_refused_before_anything_is_stored(market):
    async with _client(_app()) as client:
        response = await client.put("/api/v1/legion-link", json={"key": "not-a-key"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "bad_key_format"
    assert market.seen == []


# ---------- DB integration (requires PostgreSQL) ----------------------------------

DB_URL = os.environ.get("DATABASE_URL", "")


def _is_scratch_db(url: str) -> bool:
    from sqlalchemy.engine import make_url

    try:
        return "test" in (make_url(url).database or "")
    except Exception:
        return False


needs_db = pytest.mark.skipif(not _is_scratch_db(DB_URL), reason="needs DATABASE_URL pointing at a *test* database")


@pytest_asyncio.fixture
async def sessions() -> AsyncGenerator:
    pytest.importorskip("nacl")
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import SQLModel

    import app.models  # noqa: F401

    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


async def _user(sessions) -> UUID:
    from app.models.user import User

    async with sessions() as db:
        user = User(email=f"{uuid4().hex[:12]}@example.com", hashed_password="x")
        db.add(user)
        await db.commit()
        return user.id


async def _row(sessions, user_id: UUID):
    from app.models.legion_link import LegionLink

    async with sessions() as db:
        return await db.get(LegionLink, user_id)


@pytest.mark.db
@needs_db
@pytest.mark.asyncio
async def test_a_key_links_the_account_and_is_kept_encrypted(sessions, market):
    user_id = await _user(sessions)
    account_id = market.issue(KEY, "ada@example.com")
    async with _client(_app(sessions, user_id)) as client:
        assert (await client.get("/api/v1/legion-link")).json()["link"] is None
        response = await client.put("/api/v1/legion-link", json={"key": f" {KEY} "})
        assert response.status_code == 200, response.text
        link = response.json()["link"]
        assert link["email"] == "ada@example.com"
        assert link["account_id"] == str(account_id)
        assert link["status"] == "linked"
        assert link["key_prefix"] == KEY[:12]
        assert KEY not in response.text

        state = (await client.get("/api/v1/legion-link?refresh=true")).json()
    assert state["link"]["email"] == "ada@example.com" and state["check_error"] is None
    assert market.seen == [KEY, KEY]
    row = await _row(sessions, user_id)
    assert KEY.encode() not in bytes(row.ciphertext)
    assert legion_link.stored_key(row).reveal() == KEY


@pytest.mark.db
@needs_db
@pytest.mark.asyncio
async def test_a_revoked_key_turns_rejected_and_a_rejected_key_does_not_link(sessions, market):
    user_id = await _user(sessions)
    market.issue(KEY, "ada@example.com")
    async with _client(_app(sessions, user_id)) as client:
        await client.put("/api/v1/legion-link", json={"key": KEY})
        del market.accounts[KEY]  # revoked in the Legion profile
        state = (await client.get("/api/v1/legion-link?refresh=true")).json()
        assert state["link"]["status"] == "rejected"
        assert state["link"]["email"] == "ada@example.com"

        response = await client.put("/api/v1/legion-link", json={"key": OTHER_KEY})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "key_rejected"
    assert (await _row(sessions, user_id)).key_prefix == KEY[:12]


@pytest.mark.db
@needs_db
@pytest.mark.asyncio
async def test_with_market_down_the_last_answer_stays_and_a_new_link_waits(sessions, market):
    user_id = await _user(sessions)
    market.issue(KEY, "ada@example.com")
    async with _client(_app(sessions, user_id)) as client:
        await client.put("/api/v1/legion-link", json={"key": KEY})
        market.down = True
        state = (await client.get("/api/v1/legion-link?refresh=true")).json()
        assert state["check_error"] == "unreachable"
        assert state["link"]["status"] == "linked"
        response = await client.put("/api/v1/legion-link", json={"key": OTHER_KEY})
    assert response.status_code == 503


@pytest.mark.db
@needs_db
@pytest.mark.asyncio
async def test_another_key_replaces_the_link_and_unlink_forgets_it(sessions, market):
    user_id = await _user(sessions)
    market.issue(KEY, "ada@example.com")
    other_account = market.issue(OTHER_KEY, "bob@example.com")
    async with _client(_app(sessions, user_id)) as client:
        await client.put("/api/v1/legion-link", json={"key": KEY})
        link = (await client.put("/api/v1/legion-link", json={"key": OTHER_KEY})).json()["link"]
        assert (link["email"], link["account_id"]) == ("bob@example.com", str(other_account))
        assert legion_link.stored_key(await _row(sessions, user_id)).reveal() == OTHER_KEY

        removed = (await client.delete("/api/v1/legion-link")).json()
        assert removed["removed"] is True and removed["link"] is None
        assert (await client.delete("/api/v1/legion-link")).json()["removed"] is False
    assert await _row(sessions, user_id) is None
