"""internal design notes Stage 3 — login start and callback exchange."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_oauth
from app.core.secrets.store import BindingMeta, BindingTarget, PostgresEncryptedSecretStore, SecretMeta
from app.models.mcp_oauth import McpOAuthClient, McpOAuthFlow
from app.models.market import UserInstalledItem
from app.models.secret import SecretKind, SecretPurpose, SecretResourceType, SecretState

ISSUER = "https://mcp.example.com"
RESOURCE = "https://mcp.example.com/mcp"
REDIRECT_URI = "https://legion.example/api/v1/user-installed-items/mcp/oauth/callback"


class _Db:
    def __init__(self, *, scalars=None, gets=None):
        self._scalars = list(scalars or [])
        self._gets = dict(gets or {})
        self.added: list = []
        self.commit_calls = 0

    async def scalar(self, *_args, **_kwargs):
        return self._scalars.pop(0) if self._scalars else None

    async def get(self, _model, item_id):
        return self._gets.get(item_id)

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commit_calls += 1

    async def refresh(self, _value):
        return None

    async def execute(self, *_args, **_kwargs):
        # _deactivate_stale_oauth_secrets's lookup for a prior active
        # binding: these tests all start from a first-ever login.
        class _Result:
            def all(self_inner):
                return []

        return _Result()


def _item(user_id, *, item_id=None, config=None):
    return UserInstalledItem(
        id=item_id or uuid4(),
        user_id=user_id,
        type="mcp",
        is_custom=True,
        name="Custom MCP",
        custom_config=config if config is not None else {"url": RESOURCE},
    )


def _flow(*, state="the-state", consumed_at=None, expires_at=None, item_id=None, user_id=None):
    return McpOAuthFlow(
        state=state,
        code_verifier="verifier",
        installed_item_id=item_id or uuid4(),
        user_id=user_id or uuid4(),
        redirect_uri=REDIRECT_URI,
        issuer=ISSUER,
        client_id="client-1",
        resource=RESOURCE,
        scope=None,
        expires_at=expires_at or (datetime.now(timezone.utc) + timedelta(minutes=10)),
        consumed_at=consumed_at,
    )


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    mcp_oauth._metadata_cache.clear()
    yield
    mcp_oauth._metadata_cache.clear()


# --- start_login ------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_login_builds_pkce_authorization_url_and_persists_the_flow():
    user_id = uuid4()
    item = _item(user_id)
    db = _Db()
    provider = mcp_oauth.ProviderMetadata(
        issuer=ISSUER,
        authorization_endpoint=f"{ISSUER}/authorize",
        token_endpoint=f"{ISSUER}/token",
        scope="tools",
    )
    registration = mcp_oauth.ClientRegistration(
        client=McpOAuthClient(issuer=ISSUER, redirect_uri=REDIRECT_URI, client_id="client-1"),
        created=True,
    )

    with patch.object(mcp_oauth, "resolve_provider", AsyncMock(return_value=provider)), patch.object(
        mcp_oauth, "register_client", AsyncMock(return_value=registration)
    ):
        result = await mcp_oauth.start_login(db, item=item, user_id=user_id, redirect_uri=REDIRECT_URI)

    assert result.authorization_url.startswith(f"{ISSUER}/authorize?")
    assert "code_challenge=" in result.authorization_url
    assert "code_challenge_method=S256" in result.authorization_url
    assert f"state={result.flow.state}" in result.authorization_url
    assert result.flow.installed_item_id == item.id
    assert result.flow.user_id == user_id
    assert result.flow.code_verifier  # a real PKCE verifier was minted
    assert db.added == [result.flow]
    assert db.commit_calls == 1


@pytest.mark.asyncio
async def test_start_login_requires_the_items_own_endpoint_url():
    item = _item(uuid4(), config={"mode": "hosted"})
    db = _Db()

    with pytest.raises(mcp_oauth.McpOAuthError) as excinfo:
        await mcp_oauth.start_login(db, item=item, user_id=item.user_id, redirect_uri=REDIRECT_URI)

    assert excinfo.value.code == "mcp_oauth_endpoint_missing"


@pytest.mark.asyncio
async def test_start_login_propagates_no_dynamic_registration():
    item = _item(uuid4())
    db = _Db()
    provider = mcp_oauth.ProviderMetadata(
        issuer=ISSUER, authorization_endpoint=f"{ISSUER}/authorize", token_endpoint=f"{ISSUER}/token", scope=None
    )

    with patch.object(mcp_oauth, "resolve_provider", AsyncMock(return_value=provider)), patch.object(
        mcp_oauth, "register_client", AsyncMock(side_effect=mcp_oauth.NoDynamicRegistration(ISSUER))
    ):
        with pytest.raises(mcp_oauth.NoDynamicRegistration):
            await mcp_oauth.start_login(db, item=item, user_id=item.user_id, redirect_uri=REDIRECT_URI)


# --- complete_login -----------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_login_rejects_an_unknown_or_already_consumed_state():
    db = _Db(scalars=[None])

    with pytest.raises(mcp_oauth.InvalidFlow):
        await mcp_oauth.complete_login(db, state="does-not-exist", code="the-code", keyring=object())


@pytest.mark.asyncio
async def test_complete_login_rejects_an_expired_flow():
    flow = _flow(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
    db = _Db(scalars=[flow])

    with pytest.raises(mcp_oauth.InvalidFlow):
        await mcp_oauth.complete_login(db, state=flow.state, code="the-code", keyring=object())

    assert flow.outcome == "error"
    assert flow.error_code == "mcp_oauth_flow_expired"
    assert flow.code_verifier == ""
    assert db.commit_calls == 1


@pytest.mark.asyncio
async def test_complete_login_surfaces_the_gateways_rejection_code():
    user_id = uuid4()
    item = _item(user_id)
    flow = _flow(item_id=item.id, user_id=user_id)
    db = _Db(scalars=[flow], gets={item.id: item})
    asm = {"issuer": ISSUER, "token_endpoint": f"{ISSUER}/token"}

    with patch.object(mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)), patch.object(
        mcp_oauth,
        "_gateway_oauth_token_post",
        AsyncMock(side_effect=mcp_oauth.McpOAuthError("mcp_oauth_token_rejected", "no")),
    ):
        outcome = await mcp_oauth.complete_login(db, state=flow.state, code="the-code", keyring=object())

    assert outcome.ok is False
    assert outcome.error_code == "mcp_oauth_token_rejected"
    assert flow.outcome == "error"
    assert flow.code_verifier == ""


@pytest.mark.asyncio
async def test_complete_login_stores_tokens_and_marks_the_item_oauth_on_success():
    user_id = uuid4()
    item = _item(user_id)
    flow = _flow(item_id=item.id, user_id=user_id)
    client_row = McpOAuthClient(issuer=ISSUER, redirect_uri=REDIRECT_URI, client_id="client-1")
    # scalar() is called for: flow lookup, then the McpOAuthClient lookup.
    db = _Db(scalars=[flow, client_row], gets={item.id: item})
    asm = {"issuer": ISSUER, "token_endpoint": f"{ISSUER}/token"}
    token_response = {
        "access_token": "the-access-token",
        "refresh_token": "the-refresh-token",
        "expires_in": 3600,
    }

    def _meta(kind, purpose):
        return SecretMeta(
            id=uuid4(),
            owner_id=user_id,
            kind=kind,
            label="x",
            provider=None,
            configured=True,
            version=1,
            expires_at=None,
            state=SecretState.ACTIVE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    create_calls: list = []
    bind_calls: list = []

    async def _fake_create(self, **kwargs):
        create_calls.append(kwargs)
        return _meta(kwargs["kind"], None)

    async def _fake_bind(self, **kwargs):
        bind_calls.append(kwargs)
        target = kwargs["target"]
        return BindingMeta(
            id=uuid4(),
            secret_id=kwargs["secret_id"],
            owner_id=user_id,
            resource_type=target.resource_type,
            resource_id=target.resource_id,
            purpose=target.purpose,
            injection_config=target.injection_config,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    with patch.object(
        mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)
    ), patch.object(
        mcp_oauth, "_gateway_oauth_token_post", AsyncMock(return_value=token_response)
    ), patch.object(
        PostgresEncryptedSecretStore, "create", _fake_create
    ), patch.object(
        PostgresEncryptedSecretStore, "bind", _fake_bind
    ):
        outcome = await mcp_oauth.complete_login(db, state=flow.state, code="the-code", keyring=object())

    assert outcome.ok is True
    assert outcome.installed_item_id == item.id
    assert flow.outcome == "success"
    assert flow.consumed_at is not None
    assert flow.code_verifier == ""

    # Access token bound as a plain bearer, matching resolve_mcp_upstream_auth's
    # existing (unmodified) MCP_AUTH_HEADER path.
    assert len(create_calls) == 2
    assert create_calls[0]["kind"] == SecretKind.MCP_HEADER
    assert bind_calls[0]["target"].purpose == SecretPurpose.MCP_AUTH_HEADER
    assert bind_calls[0]["target"].injection_config == {"mode": "bearer"}
    assert bind_calls[1]["target"].purpose == SecretPurpose.MCP_OAUTH_REFRESH
    assert bind_calls[1]["target"].resource_type == SecretResourceType.INSTALLED_MCP

    auth = item.custom_config["auth"]
    assert auth["mode"] == "oauth"
    assert auth["issuer"] == ISSUER
    assert auth["needs_reauth"] is False
    assert auth["client_ref"] == str(client_row.id)


@pytest.mark.asyncio
async def test_complete_login_skips_refresh_binding_when_the_provider_omits_one():
    user_id = uuid4()
    item = _item(user_id)
    flow = _flow(item_id=item.id, user_id=user_id)
    db = _Db(scalars=[flow, None], gets={item.id: item})
    asm = {"issuer": ISSUER, "token_endpoint": f"{ISSUER}/token"}
    token_response = {"access_token": "the-access-token"}

    async def _fake_create(self, **kwargs):
        return SecretMeta(
            id=uuid4(),
            owner_id=user_id,
            kind=kwargs["kind"],
            label="x",
            provider=None,
            configured=True,
            version=1,
            expires_at=None,
            state=SecretState.ACTIVE,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    bind_calls: list = []

    async def _fake_bind(self, **kwargs):
        bind_calls.append(kwargs)
        target = kwargs["target"]
        return BindingMeta(
            id=uuid4(),
            secret_id=kwargs["secret_id"],
            owner_id=user_id,
            resource_type=target.resource_type,
            resource_id=target.resource_id,
            purpose=target.purpose,
            injection_config=target.injection_config,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    with patch.object(
        mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)
    ), patch.object(
        mcp_oauth, "_gateway_oauth_token_post", AsyncMock(return_value=token_response)
    ), patch.object(
        PostgresEncryptedSecretStore, "create", _fake_create
    ), patch.object(
        PostgresEncryptedSecretStore, "bind", _fake_bind
    ):
        outcome = await mcp_oauth.complete_login(db, state=flow.state, code="the-code", keyring=object())

    assert outcome.ok is True
    assert len(bind_calls) == 1
    assert bind_calls[0]["target"].purpose == SecretPurpose.MCP_AUTH_HEADER
