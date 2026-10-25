"""internal design notes Stage 8 — manual client registration for providers without
dynamic client registration.

Covers: registering (and re-registering) a manual client, resolving its
confidential secret at token-exchange and refresh time, and that a DCR
(public) client is never asked for a secret it does not have.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_oauth
from app.core.secrets import runtime as secrets_runtime
from app.core.secrets.crypto import SecretValue
from app.core.secrets.store import (
    BindingMeta,
    PostgresEncryptedSecretStore,
    SecretAccessDenied,
    SecretMeta,
)
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


def _item(user_id, *, item_id=None) -> UserInstalledItem:
    return UserInstalledItem(
        id=item_id or uuid4(),
        user_id=user_id,
        type="mcp",
        is_custom=True,
        name="Custom MCP",
        custom_config={"url": RESOURCE},
    )


def _secret_meta(kind) -> SecretMeta:
    return SecretMeta(
        id=uuid4(),
        owner_id=uuid4(),
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


def _binding_meta(target, secret_id) -> BindingMeta:
    return BindingMeta(
        id=uuid4(),
        secret_id=secret_id,
        owner_id=uuid4(),
        resource_type=target.resource_type,
        resource_id=target.resource_id,
        purpose=target.purpose,
        injection_config=target.injection_config,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    mcp_oauth._metadata_cache.clear()
    yield
    mcp_oauth._metadata_cache.clear()


# --- register_manual_client -------------------------------------------------


@pytest.mark.asyncio
async def test_register_manual_client_stores_the_secret_and_creates_the_client_row():
    user_id = uuid4()
    item = _item(user_id)
    db = _Db(scalars=[None])  # no existing McpOAuthClient row
    prm = {"authorization_servers": [ISSUER]}
    asm = {"issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize", "token_endpoint": f"{ISSUER}/token"}

    create_calls: list = []
    bind_calls: list = []

    async def _fake_create(self, **kwargs):
        create_calls.append(kwargs)
        return _secret_meta(kwargs["kind"])

    async def _fake_bind(self, **kwargs):
        bind_calls.append(kwargs)
        return _binding_meta(kwargs["target"], kwargs["secret_id"])

    with patch.object(
        mcp_oauth, "fetch_protected_resource_metadata", AsyncMock(return_value=prm)
    ), patch.object(
        mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)
    ), patch.object(
        PostgresEncryptedSecretStore, "create", _fake_create
    ), patch.object(
        PostgresEncryptedSecretStore, "bind", _fake_bind
    ):
        result = await mcp_oauth.register_manual_client(
            db,
            item=item,
            user_id=user_id,
            redirect_uri=REDIRECT_URI,
            client_id="manual-client-id",
            client_secret="manual-client-secret",
            keyring=object(),
        )

    assert result.created is True
    assert result.client.registration_source == "manual"
    assert result.client.client_id == "manual-client-id"
    assert result.client.client_secret_binding_id is not None
    assert db.commit_calls == 1

    assert len(create_calls) == 1
    assert create_calls[0]["kind"] == SecretKind.CUSTOM
    assert bind_calls[0]["target"].purpose == SecretPurpose.MCP_OAUTH_CLIENT_SECRET
    assert bind_calls[0]["target"].resource_type == SecretResourceType.INSTALLED_MCP
    assert bind_calls[0]["target"].resource_id == item.id

    # The raw provider registration response is never persisted — only the
    # trimmed endpoints, same discipline as a DCR registration.
    assert "client_secret" not in result.client.metadata_json


@pytest.mark.asyncio
async def test_register_manual_client_without_a_secret_is_a_public_client():
    user_id = uuid4()
    item = _item(user_id)
    db = _Db(scalars=[None])
    prm = {"authorization_servers": [ISSUER]}
    asm = {"issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize", "token_endpoint": f"{ISSUER}/token"}

    with patch.object(
        mcp_oauth, "fetch_protected_resource_metadata", AsyncMock(return_value=prm)
    ), patch.object(
        mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)
    ), patch.object(
        PostgresEncryptedSecretStore, "create", AsyncMock()
    ) as create_mock:
        result = await mcp_oauth.register_manual_client(
            db,
            item=item,
            user_id=user_id,
            redirect_uri=REDIRECT_URI,
            client_id="manual-client-id",
            client_secret=None,
            keyring=object(),
        )

    create_mock.assert_not_awaited()
    assert result.client.client_secret_binding_id is None


@pytest.mark.asyncio
async def test_register_manual_client_rejects_a_blank_client_id():
    item = _item(uuid4())
    db = _Db()

    with pytest.raises(mcp_oauth.ManualClientInvalid):
        await mcp_oauth.register_manual_client(
            db,
            item=item,
            user_id=item.user_id,
            redirect_uri=REDIRECT_URI,
            client_id="   ",
            client_secret=None,
            keyring=object(),
        )
    assert db.added == []


@pytest.mark.asyncio
async def test_register_manual_client_rotates_an_existing_rows_credentials():
    user_id = uuid4()
    item = _item(user_id)
    existing = McpOAuthClient(
        issuer=ISSUER, redirect_uri=REDIRECT_URI, client_id="old-client-id", registration_source="manual"
    )
    db = _Db(scalars=[existing])
    prm = {"authorization_servers": [ISSUER]}
    asm = {"issuer": ISSUER, "authorization_endpoint": f"{ISSUER}/authorize", "token_endpoint": f"{ISSUER}/token"}

    async def _fake_create(self, **kwargs):
        return _secret_meta(kwargs["kind"])

    async def _fake_bind(self, **kwargs):
        return _binding_meta(kwargs["target"], kwargs["secret_id"])

    with patch.object(
        mcp_oauth, "fetch_protected_resource_metadata", AsyncMock(return_value=prm)
    ), patch.object(
        mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)
    ), patch.object(
        PostgresEncryptedSecretStore, "create", _fake_create
    ), patch.object(
        PostgresEncryptedSecretStore, "bind", _fake_bind
    ):
        result = await mcp_oauth.register_manual_client(
            db,
            item=item,
            user_id=user_id,
            redirect_uri=REDIRECT_URI,
            client_id="new-client-id",
            client_secret="new-secret",
            keyring=object(),
        )

    assert result.created is False
    assert result.client is existing
    assert existing.client_id == "new-client-id"
    assert existing.client_secret_binding_id is not None


# --- resolve_manual_client_secret --------------------------------------------


@pytest.mark.asyncio
async def test_resolve_manual_client_secret_is_none_for_a_dcr_client():
    client = McpOAuthClient(issuer=ISSUER, redirect_uri=REDIRECT_URI, client_id="c1")
    secret = await mcp_oauth.resolve_manual_client_secret(
        object(), client=client, owner_id=uuid4(), installed_item_id=uuid4(), keyring=object()
    )
    assert secret is None


@pytest.mark.asyncio
async def test_resolve_manual_client_secret_is_none_when_no_secret_was_stored():
    client = McpOAuthClient(
        issuer=ISSUER, redirect_uri=REDIRECT_URI, client_id="c1", registration_source="manual"
    )
    secret = await mcp_oauth.resolve_manual_client_secret(
        object(), client=client, owner_id=uuid4(), installed_item_id=uuid4(), keyring=object()
    )
    assert secret is None


@pytest.mark.asyncio
async def test_resolve_manual_client_secret_reveals_the_bound_value():
    binding_id = uuid4()
    owner_id = uuid4()
    item_id = uuid4()
    client = McpOAuthClient(
        issuer=ISSUER,
        redirect_uri=REDIRECT_URI,
        client_id="c1",
        registration_source="manual",
        client_secret_binding_id=binding_id,
    )

    resolve_mock = AsyncMock(return_value=SecretValue("the-client-secret"))
    with patch.object(PostgresEncryptedSecretStore, "resolve_for_use", resolve_mock):
        secret = await mcp_oauth.resolve_manual_client_secret(
            object(), client=client, owner_id=owner_id, installed_item_id=item_id, keyring=object()
        )

    assert secret == "the-client-secret"
    _args, kwargs = resolve_mock.call_args
    assert kwargs["owner_id"] == owner_id
    assert kwargs["target"].resource_id == item_id
    assert kwargs["target"].purpose == SecretPurpose.MCP_OAUTH_CLIENT_SECRET
    assert kwargs["target"].binding_id == binding_id


@pytest.mark.asyncio
async def test_resolve_manual_client_secret_fails_closed_on_access_denied():
    client = McpOAuthClient(
        issuer=ISSUER,
        redirect_uri=REDIRECT_URI,
        client_id="c1",
        registration_source="manual",
        client_secret_binding_id=uuid4(),
    )

    with patch.object(
        PostgresEncryptedSecretStore, "resolve_for_use", AsyncMock(side_effect=SecretAccessDenied("gone"))
    ):
        secret = await mcp_oauth.resolve_manual_client_secret(
            object(), client=client, owner_id=uuid4(), installed_item_id=uuid4(), keyring=object()
        )

    assert secret is None


# --- complete_login carries the manual secret into the exchange -------------


def _flow(*, item_id, user_id) -> McpOAuthFlow:
    return McpOAuthFlow(
        state="the-state",
        code_verifier="verifier",
        installed_item_id=item_id,
        user_id=user_id,
        redirect_uri=REDIRECT_URI,
        issuer=ISSUER,
        client_id="manual-client-id",
        resource=RESOURCE,
        scope=None,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )


@pytest.mark.asyncio
async def test_complete_login_sends_the_manual_clients_secret_to_the_gateway():
    user_id = uuid4()
    item = _item(user_id)
    flow = _flow(item_id=item.id, user_id=user_id)
    client_row = McpOAuthClient(
        issuer=ISSUER,
        redirect_uri=REDIRECT_URI,
        client_id="manual-client-id",
        registration_source="manual",
        client_secret_binding_id=uuid4(),
    )
    # scalar() calls, in order: flow lookup, then the McpOAuthClient lookup.
    db = _Db(scalars=[flow, client_row], gets={item.id: item})
    asm = {"issuer": ISSUER, "token_endpoint": f"{ISSUER}/token"}
    token_response = {"access_token": "the-access-token", "expires_in": 3600}

    async def _fake_create(self, **kwargs):
        return _secret_meta(kwargs["kind"])

    async def _fake_bind(self, **kwargs):
        return _binding_meta(kwargs["target"], kwargs["secret_id"])

    gateway_post = AsyncMock(return_value=token_response)

    with patch.object(
        mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)
    ), patch.object(
        mcp_oauth, "resolve_manual_client_secret", AsyncMock(return_value="the-client-secret")
    ), patch.object(
        mcp_oauth, "_gateway_oauth_token_post", gateway_post
    ), patch.object(
        PostgresEncryptedSecretStore, "create", _fake_create
    ), patch.object(
        PostgresEncryptedSecretStore, "bind", _fake_bind
    ):
        outcome = await mcp_oauth.complete_login(db, state=flow.state, code="the-code", keyring=object())

    assert outcome.ok is True
    sent_payload = gateway_post.await_args.args[0]
    assert sent_payload["client_secret"] == "the-client-secret"


@pytest.mark.asyncio
async def test_complete_login_omits_client_secret_for_a_dcr_client():
    user_id = uuid4()
    item = _item(user_id)
    flow = _flow(item_id=item.id, user_id=user_id)
    client_row = McpOAuthClient(issuer=ISSUER, redirect_uri=REDIRECT_URI, client_id="manual-client-id")
    db = _Db(scalars=[flow, client_row], gets={item.id: item})
    asm = {"issuer": ISSUER, "token_endpoint": f"{ISSUER}/token"}
    token_response = {"access_token": "the-access-token"}

    async def _fake_create(self, **kwargs):
        return _secret_meta(kwargs["kind"])

    async def _fake_bind(self, **kwargs):
        return _binding_meta(kwargs["target"], kwargs["secret_id"])

    gateway_post = AsyncMock(return_value=token_response)

    with patch.object(
        mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)
    ), patch.object(
        mcp_oauth, "_gateway_oauth_token_post", gateway_post
    ), patch.object(
        PostgresEncryptedSecretStore, "create", _fake_create
    ), patch.object(
        PostgresEncryptedSecretStore, "bind", _fake_bind
    ):
        outcome = await mcp_oauth.complete_login(db, state=flow.state, code="the-code", keyring=object())

    assert outcome.ok is True
    sent_payload = gateway_post.await_args.args[0]
    assert "client_secret" not in sent_payload


# --- refresh carries the manual secret too -----------------------------------


@pytest.mark.asyncio
async def test_refresh_oauth_tokens_resolves_and_forwards_the_manual_secret():
    from app.core.secrets.store import BindingTarget

    owner_id = uuid4()
    item_id = uuid4()
    client_id = uuid4()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=30)
    item = UserInstalledItem(
        id=item_id,
        user_id=owner_id,
        type="mcp",
        is_custom=True,
        name="Custom MCP",
        custom_config={"url": RESOURCE},
    )
    auth = {
        "mode": "oauth",
        "issuer": ISSUER,
        "resource": RESOURCE,
        "client_ref": str(client_id),
        "scope": None,
        "expires_at": expires_at.isoformat(),
        "needs_reauth": False,
    }
    client_row = McpOAuthClient(
        id=client_id,
        issuer=ISSUER,
        redirect_uri=REDIRECT_URI,
        client_id="manual-client-id",
        registration_source="manual",
        client_secret_binding_id=uuid4(),
    )

    class _RefreshDb:
        async def get(self, model, obj_id):
            if model is McpOAuthClient and obj_id == client_id:
                return client_row
            return None

        def add(self, _value):
            return None

        async def commit(self):
            return None

    exchange = AsyncMock(return_value={"access_token": "new-access", "expires_in": 3600})

    with patch.object(
        secrets_runtime, "_active_mcp_binding", AsyncMock(return_value=(uuid4(), uuid4()))
    ), patch.object(
        secrets_runtime, "resolve_for_use", AsyncMock(return_value=SecretValue("the-refresh-token"))
    ), patch.object(
        mcp_oauth, "resolve_manual_client_secret", AsyncMock(return_value="the-client-secret")
    ) as resolve_secret, patch.object(
        mcp_oauth, "refresh_access_token", exchange
    ), patch.object(
        PostgresEncryptedSecretStore, "replace_value", AsyncMock()
    ), patch.object(
        PostgresEncryptedSecretStore, "update_metadata", AsyncMock()
    ):
        refreshed = await secrets_runtime._refresh_oauth_tokens(
            db=_RefreshDb(),
            owner_id=owner_id,
            item=item,
            auth=auth,
            request_id=uuid4(),
            source_ip=None,
        )

    assert refreshed is True
    resolve_secret.assert_awaited_once()
    assert resolve_secret.await_args.kwargs["installed_item_id"] == item_id
    exchange.assert_awaited_once()
    assert exchange.await_args.kwargs["client_secret"] == "the-client-secret"
