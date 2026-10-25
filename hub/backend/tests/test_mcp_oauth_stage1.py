"""internal design notes Stage 1 — metadata cache and idempotent DCR registration."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.exc import IntegrityError

from app.core import mcp_oauth
from app.models.mcp_oauth import McpOAuthClient

ISSUER = "https://mcp.cloudflare.com"
REDIRECT_URI = "https://legion.example/api/v1/mcp/oauth/callback"


class _Db:
    def __init__(self, *, scalars=None, commit_exc=None):
        self._scalars = list(scalars or [])
        self.added: list = []
        self.commit_exc = commit_exc
        self.commit_calls = 0
        self.rollback_calls = 0

    async def scalar(self, *_args, **_kwargs):
        return self._scalars.pop(0) if self._scalars else None

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commit_calls += 1
        if self.commit_exc is not None and self.commit_calls == 1:
            raise self.commit_exc

    async def rollback(self):
        self.rollback_calls += 1

    async def refresh(self, _value):
        return None


def _existing_client() -> McpOAuthClient:
    return McpOAuthClient(
        issuer=ISSUER,
        redirect_uri=REDIRECT_URI,
        client_id="already-registered",
        registration_source="dcr",
    )


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    mcp_oauth._metadata_cache.clear()
    yield
    mcp_oauth._metadata_cache.clear()


@pytest.mark.asyncio
async def test_register_client_reuses_existing_row_without_a_second_registration():
    db = _Db(scalars=[_existing_client()])

    with patch.object(mcp_oauth, "_gateway_oauth_post", AsyncMock()) as gateway_post:
        result = await mcp_oauth.register_client(db, issuer=ISSUER, redirect_uri=REDIRECT_URI)

    assert result.created is False
    assert result.client.client_id == "already-registered"
    gateway_post.assert_not_awaited()
    assert db.added == []


@pytest.mark.asyncio
async def test_register_client_performs_dcr_when_not_yet_registered():
    db = _Db(scalars=[None])
    asm = {
        "issuer": ISSUER,
        "registration_endpoint": f"{ISSUER}/register",
        "authorization_endpoint": f"{ISSUER}/authorize",
        "token_endpoint": f"{ISSUER}/token",
    }
    register_response = {"client_id": "fresh-client-id"}

    async def _gateway_post(path, payload):
        if path == "/v1/oauth/metadata":
            return asm
        assert path == "/v1/oauth/register"
        assert payload["redirect_uris"] == [REDIRECT_URI]
        return register_response

    with patch.object(mcp_oauth, "_gateway_oauth_post", AsyncMock(side_effect=_gateway_post)):
        result = await mcp_oauth.register_client(db, issuer=ISSUER, redirect_uri=REDIRECT_URI)

    assert result.created is True
    assert result.client.client_id == "fresh-client-id"
    assert result.client.registration_source == "dcr"
    # Never persists whatever the provider's raw registration response held.
    assert "client_secret" not in result.client.metadata_json
    assert db.commit_calls == 1


@pytest.mark.asyncio
async def test_register_client_raises_a_distinct_error_without_registration_endpoint():
    db = _Db(scalars=[None])
    asm = {"issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize"}

    with patch.object(mcp_oauth, "_gateway_oauth_post", AsyncMock(return_value=asm)):
        with pytest.raises(mcp_oauth.NoDynamicRegistration) as excinfo:
            await mcp_oauth.register_client(db, issuer=ISSUER, redirect_uri=REDIRECT_URI)

    assert excinfo.value.code == "mcp_oauth_no_dynamic_registration"
    assert db.added == []


@pytest.mark.asyncio
async def test_register_client_resolves_a_concurrent_registration_race():
    """Two logins from the same origin race DCR; the loser reuses the winner's row."""

    winner = _existing_client()
    db = _Db(
        scalars=[None, winner],
        commit_exc=IntegrityError("insert", {}, Exception("duplicate key")),
    )
    asm = {"issuer": ISSUER, "registration_endpoint": f"{ISSUER}/register"}

    with patch.object(
        mcp_oauth,
        "_gateway_oauth_post",
        AsyncMock(side_effect=[asm, {"client_id": "loser-client-id"}]),
    ):
        result = await mcp_oauth.register_client(db, issuer=ISSUER, redirect_uri=REDIRECT_URI)

    assert result.created is False
    assert result.client is winner
    assert db.rollback_calls == 1


@pytest.mark.asyncio
async def test_authorization_server_metadata_is_cached_across_calls():
    asm = {"issuer": ISSUER, "registration_endpoint": f"{ISSUER}/register"}

    with patch.object(mcp_oauth, "_gateway_oauth_post", AsyncMock(return_value=asm)) as gateway_post:
        first = await mcp_oauth.fetch_authorization_server_metadata(ISSUER)
        second = await mcp_oauth.fetch_authorization_server_metadata(ISSUER)

    assert first == asm
    assert second == asm
    gateway_post.assert_awaited_once()


@pytest.mark.asyncio
async def test_protected_resource_metadata_has_its_own_cache_key():
    """A resource URL and its issuer never collide in the metadata cache."""

    resource = f"{ISSUER}/mcp"
    prm = {"resource": resource, "authorization_servers": [ISSUER]}
    asm = {"issuer": ISSUER, "registration_endpoint": f"{ISSUER}/register"}

    with patch.object(
        mcp_oauth, "_gateway_oauth_post", AsyncMock(side_effect=[prm, asm])
    ) as gateway_post:
        got_prm = await mcp_oauth.fetch_protected_resource_metadata(resource)
        got_asm = await mcp_oauth.fetch_authorization_server_metadata(ISSUER)

    assert got_prm == prm
    assert got_asm == asm
    assert gateway_post.await_count == 2
