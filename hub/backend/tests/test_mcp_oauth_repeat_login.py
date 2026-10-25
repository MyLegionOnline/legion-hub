"""internal design notes — a repeat login must replace the previous access/refresh
secret, not accumulate a second active one next to it.

Found live: a user logged into the same Cloudflare MCP item twice (two
distinct "Sign in" clicks that both completed). Both logins created their own
active MCP_AUTH_HEADER binding — nothing ever disabled the first one.
``resolve_mcp_upstream_auth``'s ``_active_mcp_binding_id`` demands *exactly
one* active binding for the purpose and silently returns ``None`` the moment
there are two, so the item permanently failed every tool call with
``mcp_secret_unavailable`` even though the login itself had succeeded.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_oauth
from app.core.secrets.store import BindingMeta, PostgresEncryptedSecretStore, SecretMeta
from app.models.mcp_oauth import McpOAuthClient, McpOAuthFlow
from app.models.market import UserInstalledItem
from app.models.secret import SecretBinding, SecretKind, SecretPurpose, SecretResourceType, SecretState, UserSecret

ISSUER = "https://mcp.example.com"
RESOURCE = "https://mcp.example.com/mcp"
REDIRECT_URI = "https://legion.example/api/v1/user-installed-items/mcp/oauth/callback"


class _Db:
    def __init__(self, *, scalars, gets, existing_active):
        self._scalars = list(scalars)
        self._gets = dict(gets)
        self._existing_active = existing_active  # {purpose: [(binding, secret)]}
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

    async def execute(self, statement, *_args, **_kwargs):
        # Cheap way to tell which purpose _deactivate_stale_oauth_secrets is
        # asking about without parsing SQL: inspect the compiled WHERE
        # clause's literal values isn't worth it here — the two purposes are
        # queried in a fixed order (MCP_AUTH_HEADER then MCP_OAUTH_REFRESH),
        # so pop them off in that same order.
        rows = self._existing_active.pop(0) if self._existing_active else []

        class _Result:
            def all(self_inner):
                return rows

        return _Result()


def _flow(*, item_id, user_id) -> McpOAuthFlow:
    return McpOAuthFlow(
        state="the-state",
        code_verifier="verifier",
        installed_item_id=item_id,
        user_id=user_id,
        redirect_uri=REDIRECT_URI,
        issuer=ISSUER,
        client_id="client-1",
        resource=RESOURCE,
        scope=None,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )


def _item(user_id) -> UserInstalledItem:
    return UserInstalledItem(
        id=uuid4(), user_id=user_id, type="mcp", is_custom=True, name="OAuth MCP",
        custom_config={"url": RESOURCE},
    )


def _secret_meta(kind, secret_id=None) -> SecretMeta:
    return SecretMeta(
        id=secret_id or uuid4(), owner_id=uuid4(), kind=kind, label="x", provider=None,
        configured=True, version=1, expires_at=None, state=SecretState.ACTIVE,
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_second_login_disables_the_first_active_binding_before_binding_a_new_one():
    user_id = uuid4()
    item = _item(user_id)
    flow = _flow(item_id=item.id, user_id=user_id)
    client_row = McpOAuthClient(issuer=ISSUER, redirect_uri=REDIRECT_URI, client_id="client-1")

    stale_access_secret = UserSecret(
        id=uuid4(), owner_id=user_id, kind=SecretKind.MCP_HEADER, label="old", ciphertext=b"x", key_id="k",
    )
    stale_binding = SecretBinding(
        id=uuid4(), secret_id=stale_access_secret.id, owner_id=user_id,
        resource_type=SecretResourceType.INSTALLED_MCP, resource_id=item.id,
        purpose=SecretPurpose.MCP_AUTH_HEADER,
    )

    db = _Db(
        scalars=[flow, client_row],
        gets={item.id: item},
        existing_active=[[(stale_binding, stale_access_secret)], []],
    )
    asm = {"issuer": ISSUER, "token_endpoint": f"{ISSUER}/token"}
    token_response = {"access_token": "new-access-token", "expires_in": 3600}

    disable_calls: list = []

    async def _fake_update_metadata(self, *, owner_id, secret_id, **kwargs):
        disable_calls.append((owner_id, secret_id, kwargs))
        return _secret_meta(SecretKind.MCP_HEADER, secret_id)

    async def _fake_create(self, **kwargs):
        return _secret_meta(kwargs["kind"])

    async def _fake_bind(self, **kwargs):
        target = kwargs["target"]
        return BindingMeta(
            id=uuid4(), secret_id=kwargs["secret_id"], owner_id=user_id,
            resource_type=target.resource_type, resource_id=target.resource_id,
            purpose=target.purpose, injection_config=target.injection_config,
            created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
        )

    with patch.object(
        mcp_oauth, "fetch_authorization_server_metadata", AsyncMock(return_value=asm)
    ), patch.object(
        mcp_oauth, "_gateway_oauth_token_post", AsyncMock(return_value=token_response)
    ), patch.object(
        PostgresEncryptedSecretStore, "update_metadata", _fake_update_metadata
    ), patch.object(
        PostgresEncryptedSecretStore, "create", _fake_create
    ), patch.object(
        PostgresEncryptedSecretStore, "bind", _fake_bind
    ):
        outcome = await mcp_oauth.complete_login(db, state=flow.state, code="the-code", keyring=object())

    assert outcome.ok is True
    assert len(disable_calls) == 1
    owner_id, secret_id, kwargs = disable_calls[0]
    assert owner_id == user_id
    assert secret_id == stale_access_secret.id
    assert kwargs["enabled"] is False
