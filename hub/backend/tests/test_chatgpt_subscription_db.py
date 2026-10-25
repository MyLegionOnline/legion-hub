"""PostgreSQL SIWC lifecycle and cross-session isolation. Disposable DB only."""
import asyncio
import os
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlmodel import SQLModel

TEST_URL = os.getenv("SIWC_TEST_DB_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="SIWC_TEST_DB_URL must point to a disposable PostgreSQL DB")

@pytest_asyncio.fixture
async def environment(monkeypatch):
    import app.models
    from app.api.v1 import chatgpt_auth as api
    from app.core import chatgpt_oauth as oauth
    from app.core.secret_keyring import SecretKeyring
    from app.core.secrets import runtime
    from app.db import session
    from app.models.chatgpt_auth import ChatGPTAuthorization, ChatGPTHost
    from app.models.provider_connection import ProviderConnection
    from app.models.secret import UserSecret, SecretBinding, SecretAudit
    from app.models.user import User
    engine = create_async_engine(TEST_URL)
    tables = [User.__table__, UserSecret.__table__, SecretBinding.__table__, SecretAudit.__table__,
              ProviderConnection.__table__, ChatGPTHost.__table__, ChatGPTAuthorization.__table__]
    async with engine.begin() as connection:
        await connection.run_sync(lambda bind: SQLModel.metadata.drop_all(bind, tables=tables))
        await connection.run_sync(lambda bind: SQLModel.metadata.create_all(bind, tables=tables))
    monkeypatch.setattr(session, "engine", engine)
    monkeypatch.setattr(runtime, "_RUNTIME_KEYRING", SecretKeyring(active_key_id="test", keys={"test": b"x" * 32}, path="<test>"))
    alice = User(id=uuid4(), email=f"{uuid4()}@example.com", hashed_password="x")
    bob = User(id=uuid4(), email=f"{uuid4()}@example.com", hashed_password="x")
    provider = ProviderConnection(user_id=alice.id, provider="openai_chatgpt")
    async with AsyncSession(engine, expire_on_commit=False) as db:
        db.add_all([alice, bob])
        await db.flush()
        db.add(provider)
        await db.commit()
    tokens = {"access_token": "test-access", "refresh_token": "test-refresh", "id_token": "test-id",
              "token_type": "Bearer", "expires_in": 3600, "scope": oauth.SCOPES}
    monkeypatch.setattr(oauth, "token_request", AsyncMock(return_value=tokens))
    monkeypatch.setattr(oauth, "verify_identity", AsyncMock(return_value={"sub": "openai-user", "email": "person@example.com"}))
    yield SimpleNamespace(engine=engine, api=api, oauth=oauth, alice=alice, bob=bob, provider=provider)
    await engine.dispose()

async def connect(env):
    api = env.api
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        setup = await api.setup(env.provider.id, db=db, user=env.alice)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        started = await api.helper_start(api.Start(ticket=setup["ticket"], port=12345), db=db)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        await api.helper_complete(api.Complete(ticket=setup["ticket"], state=started["state"],
                                  code="one-use-code", client_id="oaiapp_test"), db=db)
    return setup, started

@pytest.mark.asyncio
async def test_oauth_persists_between_browser_sessions_and_rejects_other_owner(environment):
    env = environment
    setup, started = await connect(env)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        status = await env.api.status(env.provider.id, db=db, user=env.alice)
        assert status == {"pending": False, "phase": "connected", "connected": True, "email": "person@example.com", "error": None}
        with pytest.raises(HTTPException) as caught:
            await env.api.status(env.provider.id, db=db, user=env.bob)
        assert caught.value.status_code == 404
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        with pytest.raises(HTTPException) as caught:
            await env.api.helper_complete(env.api.Complete(ticket=setup["ticket"], state=started["state"],
                                           code="replayed-code", client_id="oaiapp_test"), db=db)
        assert caught.value.status_code == 410
    exchange = env.oauth.token_request.call_args.args[0]
    assert exchange["client_id"] == "oaiapp_test"
    assert exchange["redirect_uri"] == "http://127.0.0.1:12345/auth/callback"
    assert "client_secret" not in exchange

@pytest.mark.asyncio
async def test_refresh_is_serialized_and_keeps_latest_rotating_token(environment, monkeypatch):
    import json
    from sqlalchemy import select
    from app.core.chatgpt_credentials import runtime_reference
    from app.core.secrets.crypto import SecretValue
    from app.core.secrets.runtime import service_for
    from app.models.provider_connection import ProviderConnection
    from app.models.secret import UserSecret, SecretState
    from app.core.secrets.store import SecretUnavailable
    env = environment
    await connect(env)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        connection = await db.get(ProviderConnection, env.provider.id)
        secret = await db.get(UserSecret, connection.secret_id)
        bundle = env.oauth.read_bundle(db, secret)
        bundle["expires_at"] = time.time() - 1
        await service_for(db).store.replace_value(owner_id=env.alice.id, secret_id=secret.id,
                                                  value=SecretValue(json.dumps(bundle)))
        await db.commit()
        reference = runtime_reference(env.alice.id, secret.id).reveal()
        secret_id = secret.id
    async def renew(fields):
        assert fields["client_id"] == "oaiapp_test"
        assert fields["refresh_token"] == "test-refresh"
        await asyncio.sleep(0.03)
        return {"access_token": "renewed", "refresh_token": "rotated", "expires_in": 3600, "scope": env.oauth.SCOPES}
    mock = AsyncMock(side_effect=renew)
    monkeypatch.setattr(env.oauth, "token_request", mock)
    assert await asyncio.gather(*(env.oauth.access_token(reference) for _ in range(8))) == ["renewed"] * 8
    assert mock.await_count == 1
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        row = await db.get(UserSecret, secret_id)
        assert env.oauth.read_bundle(db, row)["refresh_token"] == "rotated"
        row.state = SecretState.DISABLED
        await db.commit()
    with pytest.raises(SecretUnavailable):
        await env.oauth.access_token(reference)

@pytest.mark.asyncio
async def test_disconnect_clears_tokens_and_reuses_client_and_host_on_reauthorization(environment, monkeypatch):
    from urllib.parse import parse_qs, urlsplit
    from app.models.provider_connection import ProviderConnection
    from app.models.secret import UserSecret, SecretState
    env = environment
    _, first = await connect(env)
    monkeypatch.setattr(env.oauth, "revoke", AsyncMock(return_value=False))
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        outcome = await env.api.disconnect(env.provider.id, db=db, user=env.alice)
        assert not outcome["remote_revocation_confirmed"]
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        connection = await db.get(ProviderConnection, env.provider.id)
        secret = await db.get(UserSecret, connection.secret_id)
        mapping = env.api.registration(db, secret)
        assert secret.state == SecretState.DISABLED
        assert set(mapping) == {"client_id", "sub", "email"}
        setup = await env.api.setup(env.provider.id, db=db, user=env.alice)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        started = await env.api.helper_start(env.api.Start(ticket=setup["ticket"], port=23456), db=db)
    before = parse_qs(urlsplit(first["authorize_url"]).query)
    after = parse_qs(urlsplit(started["authorize_url"]).query)
    assert after["client_id"] == ["oaiapp_test"]
    assert before["ext_agent_host_id"] == after["ext_agent_host_id"]
    assert "agent_name_hint" not in after

@pytest.mark.asyncio
async def test_terminal_refresh_clears_credentials_but_retains_registration(environment, monkeypatch):
    import json
    from app.core.chatgpt_credentials import runtime_reference
    from app.core.secrets.crypto import SecretValue
    from app.core.secrets.runtime import service_for
    from app.core.chatgpt_oauth import ChatGPTError
    from app.models.provider_connection import ProviderConnection
    from app.models.secret import UserSecret, SecretState
    env = environment
    await connect(env)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        connection = await db.get(ProviderConnection, env.provider.id)
        secret = await db.get(UserSecret, connection.secret_id)
        bundle = env.oauth.read_bundle(db, secret)
        bundle["expires_at"] = time.time() - 1
        await service_for(db).store.replace_value(owner_id=env.alice.id, secret_id=secret.id,
                                                  value=SecretValue(json.dumps(bundle)))
        reference = runtime_reference(env.alice.id, secret.id).reveal()
        secret_id = secret.id
        await db.commit()
    monkeypatch.setattr(env.oauth, "token_request", AsyncMock(side_effect=ChatGPTError("Revoked", code="invalid_grant")))
    with pytest.raises(ChatGPTError):
        await env.oauth.access_token(reference)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        secret = await db.get(UserSecret, secret_id)
        assert secret.state == SecretState.DISABLED
        assert env.api.registration(db, secret) == {"client_id": "oaiapp_test", "sub": "openai-user", "email": "person@example.com"}

@pytest.mark.asyncio
async def test_declined_plan_scope_keeps_identity_and_requests_explicit_reconsent(environment, monkeypatch):
    from urllib.parse import parse_qs, urlsplit
    env = environment
    response = {**env.oauth.token_request.return_value, "scope": "openid profile email"}
    monkeypatch.setattr(env.oauth, "token_request", AsyncMock(return_value=response))
    await connect(env)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        status = await env.api.status(env.provider.id, db=db, user=env.alice)
        assert not status["connected"]
        assert status["email"] == "person@example.com"
        assert "not granted" in status["error"]
        setup = await env.api.setup(env.provider.id, db=db, user=env.alice)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        started = await env.api.helper_start(env.api.Start(ticket=setup["ticket"], port=23456), db=db)
    query = parse_qs(urlsplit(started["authorize_url"]).query)
    assert query["client_id"] == ["oaiapp_test"]
    assert query["prompt"] == ["consent"]

@pytest.mark.asyncio
async def test_vault_resolution_returns_reference_and_generic_replace_is_denied(environment):
    from app.api.v1 import secrets
    from app.core.chatgpt_credentials import parse_reference
    from app.core.secrets.runtime import service_for
    from app.models.provider_connection import ProviderConnection
    from app.schemas.secrets import SecretValueReplace
    env = environment
    await connect(env)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        connection = await db.get(ProviderConnection, env.provider.id)
        value = await service_for(db).store.resolve_owned_llm_for_validation(
            owner_id=env.alice.id, secret_id=connection.secret_id, request_id=uuid4())
        assert parse_reference(value.reveal()) == (env.alice.id, connection.secret_id)
        assert "test-refresh" not in value.reveal()
        with pytest.raises(HTTPException) as caught:
            await secrets.replace_secret_value(connection.secret_id, SecretValueReplace(value="bad"),
                                                request=None, db=db, current_user=env.alice)
        assert caught.value.status_code == 422

@pytest.mark.asyncio
async def test_status_distinguishes_downloaded_helper_from_started_authorization(environment):
    env = environment
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        setup = await env.api.setup(env.provider.id, db=db, user=env.alice)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        status = await env.api.status(env.provider.id, db=db, user=env.alice)
        assert status["pending"] and status["phase"] == "awaiting_helper"
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        await env.api.helper_start(env.api.Start(ticket=setup["ticket"], port=12345), db=db)
    async with AsyncSession(env.engine, expire_on_commit=False) as db:
        status = await env.api.status(env.provider.id, db=db, user=env.alice)
        assert status["pending"] and status["phase"] == "authorizing"
