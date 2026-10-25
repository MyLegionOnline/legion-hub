"""internal design notes Stage 5 — lifting the bearer-only restriction.

Everything OAuth wrote in Stages 1/3/4 (custom_config.auth mode "oauth" with
issuer/resource/client_ref/scope/expires_at/needs_reauth) must now be able to
round-trip through the same validation, discovery, and gateway boundaries a
bearer item already uses, without loosening what those boundaries reject.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import user_installed_items as registry
from app.core import mcp_discovery, mcp_oauth
from app.core.mcp_gateway import _mcp_auth_required
from app.core.secrets import runtime as secrets_runtime
from app.core.secrets.crypto import SecretValue
from app.core.secrets.store import PostgresEncryptedSecretStore
from app.models.mcp_oauth import McpOAuthClient
from app.models.market import UserInstalledItem
from app.models.secret import SecretPurpose

ISSUER = "https://mcp.example.com"
RESOURCE = "https://mcp.example.com/mcp"


def _oauth_auth(*, needs_reauth: bool = False, expires_in_seconds: float = 3600) -> dict:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    return {
        "mode": "oauth",
        "issuer": ISSUER,
        "resource": RESOURCE,
        "client_ref": str(uuid4()),
        "scope": "mcp.read",
        "expires_at": expires_at.isoformat(),
        "needs_reauth": needs_reauth,
    }


def test_validate_custom_mcp_config_accepts_the_oauth_auth_shape():
    config = registry._validate_custom_mcp_config({"url": RESOURCE, "auth": _oauth_auth()})
    assert config["auth"]["mode"] == "oauth"
    assert config["auth"]["issuer"] == ISSUER


def test_validate_custom_mcp_config_rejects_an_unknown_oauth_field():
    auth = _oauth_auth()
    auth["extra"] = "not part of the shape"
    with pytest.raises(HTTPException):
        registry._validate_custom_mcp_config({"url": RESOURCE, "auth": auth})


def test_validate_custom_mcp_config_still_rejects_basic_auth():
    with pytest.raises(HTTPException, match="Only bearer"):
        registry._validate_custom_mcp_config({"url": RESOURCE, "auth": {"mode": "basic"}})


def test_bearer_config_still_rejects_unknown_fields():
    with pytest.raises(HTTPException):
        registry._validate_custom_mcp_config(
            {"url": RESOURCE, "auth": {"mode": "bearer", "issuer": ISSUER}}
        )


def test_safe_auth_accepts_oauth_mode_without_a_binding():
    item = UserInstalledItem(
        id=uuid4(), user_id=uuid4(), type="mcp", is_custom=True, name="OAuth MCP",
        custom_config={"url": RESOURCE, "auth": _oauth_auth()},
    )
    mode, binding_id = mcp_discovery._safe_auth(item)
    assert mode == "oauth"
    assert binding_id is None


def test_mcp_auth_required_is_true_for_oauth_mode():
    assert _mcp_auth_required({"auth": {"mode": "oauth"}}) is True
    assert _mcp_auth_required({"auth": {"mode": "bearer"}}) is True
    assert _mcp_auth_required({"auth": {"mode": "none"}}) is False
    assert _mcp_auth_required(None) is False


class _ItemDb:
    def __init__(self, item: UserInstalledItem, client: McpOAuthClient | None = None):
        self.item = item
        self.client = client
        self.commit_calls = 0

    async def get(self, model, obj_id):
        if model is UserInstalledItem and obj_id == self.item.id:
            return self.item
        if self.client is not None and model is McpOAuthClient and obj_id == self.client.id:
            return self.client
        return None

    async def refresh(self, _obj):
        return None

    def add(self, _value):
        return None

    async def commit(self):
        self.commit_calls += 1


def _oauth_item(*, mode: str = "oauth", expires_in_seconds: float = 3600) -> tuple[UserInstalledItem, uuid4]:
    client_id = uuid4()
    auth = _oauth_auth(expires_in_seconds=expires_in_seconds)
    auth["mode"] = mode
    auth["client_ref"] = str(client_id)
    item = UserInstalledItem(
        id=uuid4(), user_id=uuid4(), type="mcp", is_custom=True, name="OAuth MCP",
        custom_config={"url": RESOURCE, "auth": auth},
    )
    return item, client_id


@pytest.mark.asyncio
async def test_force_refresh_ignores_a_far_off_expiry():
    """A live 401 means the recorded expiry lied; force must not defer to it,
    unlike the proactive `_maybe_refresh_oauth_token` path."""

    item, client_id = _oauth_item(expires_in_seconds=3600)
    db = _ItemDb(item, McpOAuthClient(id=client_id, issuer=ISSUER, redirect_uri="https://legion.example/cb", client_id="c1"))
    exchange = AsyncMock(return_value={"access_token": "fresh-token", "expires_in": 3600})

    with patch.object(
        secrets_runtime, "_active_mcp_binding", AsyncMock(return_value=(uuid4(), uuid4()))
    ), patch.object(
        secrets_runtime, "resolve_for_use", AsyncMock(return_value=SecretValue("the-refresh-token"))
    ), patch.object(mcp_oauth, "refresh_access_token", exchange), patch.object(
        PostgresEncryptedSecretStore, "replace_value", AsyncMock()
    ), patch.object(
        PostgresEncryptedSecretStore, "update_metadata", AsyncMock()
    ):
        refreshed = await secrets_runtime.force_refresh_mcp_oauth_token(
            db=db, owner_id=item.user_id, installed_item_id=item.id, request_id=uuid4(),
        )

    assert refreshed is True
    exchange.assert_awaited_once()
    assert item.custom_config["auth"]["needs_reauth"] is False


@pytest.mark.asyncio
async def test_force_refresh_is_a_noop_for_bearer_items():
    item, _client_id = _oauth_item(mode="bearer")
    db = _ItemDb(item)
    exchange = AsyncMock()

    with patch.object(mcp_oauth, "refresh_access_token", exchange):
        refreshed = await secrets_runtime.force_refresh_mcp_oauth_token(
            db=db, owner_id=item.user_id, installed_item_id=item.id, request_id=uuid4(),
        )

    assert refreshed is False
    exchange.assert_not_awaited()
    assert db.commit_calls == 0


@pytest.mark.asyncio
async def test_mark_needs_reauth_sets_the_flag_once():
    item, _client_id = _oauth_item()
    db = _ItemDb(item)

    await secrets_runtime.mark_mcp_oauth_needs_reauth(
        db=db, owner_id=item.user_id, installed_item_id=item.id,
    )

    assert item.custom_config["auth"]["needs_reauth"] is True
    assert db.commit_calls == 1

    # Already-set stays a no-op: Stage 6's pause card should not see spurious
    # extra writes every time a call keeps failing.
    await secrets_runtime.mark_mcp_oauth_needs_reauth(
        db=db, owner_id=item.user_id, installed_item_id=item.id,
    )
    assert db.commit_calls == 1


@pytest.mark.asyncio
async def test_mark_needs_reauth_is_a_noop_for_bearer_items():
    item, _client_id = _oauth_item(mode="bearer")
    db = _ItemDb(item)

    await secrets_runtime.mark_mcp_oauth_needs_reauth(
        db=db, owner_id=item.user_id, installed_item_id=item.id,
    )

    assert item.custom_config["auth"]["needs_reauth"] is False
    assert db.commit_calls == 0
