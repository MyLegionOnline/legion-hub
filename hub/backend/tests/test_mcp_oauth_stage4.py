"""internal design notes Stage 4 — token storage and proactive refresh.

``resolve_mcp_upstream_auth`` (app/core/secrets/runtime.py) is the boundary
Stage 4 adds to: before it ever reveals a bearer header for an OAuth-mode
item, it must refresh a soon-to-expire access token exactly once even under
concurrent callers, and a failed refresh must never erase the still-active
access token or raise anything but ``McpSecretUnavailable``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_oauth
from app.core.secrets import runtime as secrets_runtime
from app.core.secrets.crypto import SecretValue
from app.core.secrets.store import PostgresEncryptedSecretStore
from app.models.mcp_oauth import McpOAuthClient
from app.models.market import UserInstalledItem
from app.models.secret import SecretPurpose, SecretResourceType

ISSUER = "https://mcp.example.com"
RESOURCE = "https://mcp.example.com/mcp"


class _ItemDb:
    """A minimal async-session double: one mutable ``UserInstalledItem`` plus
    a ``McpOAuthClient`` row, nothing else queried by the refresh path."""

    def __init__(self, item: UserInstalledItem, client: McpOAuthClient):
        self.item = item
        self.client = client
        self.commit_calls = 0
        self.added: list = []

    async def get(self, model, obj_id):
        if model is UserInstalledItem and obj_id == self.item.id:
            return self.item
        if model is McpOAuthClient and obj_id == self.client.id:
            return self.client
        return None

    async def refresh(self, _obj):
        return None

    def add(self, value):
        self.added.append(value)

    async def commit(self):
        self.commit_calls += 1


def _oauth_item(*, expires_in_seconds: float, mode: str = "oauth") -> UserInstalledItem:
    user_id = uuid4()
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    client_id = uuid4()
    return UserInstalledItem(
        id=uuid4(),
        user_id=user_id,
        type="mcp",
        is_custom=True,
        name="Custom MCP",
        custom_config={
            "url": RESOURCE,
            "auth": {
                "mode": mode,
                "issuer": ISSUER,
                "resource": RESOURCE,
                "client_ref": str(client_id),
                "scope": None,
                "expires_at": expires_at.isoformat(),
                "needs_reauth": False,
            },
        },
    ), client_id


def _client_row(client_id) -> McpOAuthClient:
    return McpOAuthClient(id=client_id, issuer=ISSUER, redirect_uri="https://legion.example/cb", client_id="client-1")


def _refresh_binding(item_id, owner_id):
    return (uuid4(), uuid4())  # (binding_id, secret_id)


@pytest.mark.asyncio
async def test_ten_concurrent_calls_refresh_exactly_once():
    item, client_id = _oauth_item(expires_in_seconds=30)  # inside the refresh skew
    db = _ItemDb(item, _client_row(client_id))
    refresh_ids = _refresh_binding(item.id, item.user_id)
    access_ids = (uuid4(), uuid4())

    async def _active_binding(*, purpose, **_kwargs):
        return refresh_ids if purpose == SecretPurpose.MCP_OAUTH_REFRESH else access_ids

    token_response = {"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600}
    exchange = AsyncMock(return_value=token_response)

    with patch.object(secrets_runtime, "_active_mcp_binding", AsyncMock(side_effect=_active_binding)), patch.object(
        secrets_runtime, "resolve_for_use", AsyncMock(return_value=SecretValue("the-refresh-token"))
    ), patch.object(mcp_oauth, "refresh_access_token", exchange), patch.object(
        PostgresEncryptedSecretStore, "replace_value", AsyncMock()
    ), patch.object(
        PostgresEncryptedSecretStore, "update_metadata", AsyncMock()
    ):
        await asyncio.gather(
            *[
                secrets_runtime._maybe_refresh_oauth_token(
                    db=db,
                    owner_id=item.user_id,
                    installed_item_id=item.id,
                    request_id=uuid4(),
                    source_ip=None,
                )
                for _ in range(10)
            ]
        )

    assert exchange.await_count == 1
    assert db.commit_calls == 1
    auth = item.custom_config["auth"]
    assert auth["needs_reauth"] is False
    new_expires_at = datetime.fromisoformat(auth["expires_at"])
    assert new_expires_at > datetime.now(timezone.utc) + timedelta(minutes=30)


@pytest.mark.asyncio
async def test_failed_token_exchange_leaves_the_access_token_untouched():
    item, client_id = _oauth_item(expires_in_seconds=10)
    original_auth = dict(item.custom_config["auth"])
    db = _ItemDb(item, _client_row(client_id))
    refresh_ids = _refresh_binding(item.id, item.user_id)

    async def _active_binding(*, purpose, **_kwargs):
        return refresh_ids if purpose == SecretPurpose.MCP_OAUTH_REFRESH else None

    replace_value = AsyncMock()

    with patch.object(secrets_runtime, "_active_mcp_binding", AsyncMock(side_effect=_active_binding)), patch.object(
        secrets_runtime, "resolve_for_use", AsyncMock(return_value=SecretValue("the-refresh-token"))
    ), patch.object(
        mcp_oauth, "refresh_access_token", AsyncMock(side_effect=mcp_oauth.McpOAuthError("mcp_oauth_token_rejected", "no"))
    ), patch.object(
        PostgresEncryptedSecretStore, "replace_value", replace_value
    ):
        # Must not raise: a refresh failure is swallowed, the caller keeps
        # using whatever access token is already bound.
        await secrets_runtime._maybe_refresh_oauth_token(
            db=db,
            owner_id=item.user_id,
            installed_item_id=item.id,
            request_id=uuid4(),
            source_ip=None,
        )

    replace_value.assert_not_awaited()
    assert db.commit_calls == 0
    assert item.custom_config["auth"] == original_auth


@pytest.mark.asyncio
async def test_bearer_mode_items_are_never_refreshed():
    item, client_id = _oauth_item(expires_in_seconds=10, mode="bearer")
    db = _ItemDb(item, _client_row(client_id))
    exchange = AsyncMock()

    with patch.object(mcp_oauth, "refresh_access_token", exchange):
        await secrets_runtime._maybe_refresh_oauth_token(
            db=db, owner_id=item.user_id, installed_item_id=item.id, request_id=uuid4(), source_ip=None
        )

    exchange.assert_not_awaited()


@pytest.mark.asyncio
async def test_oauth_item_far_from_expiry_is_not_refreshed():
    item, client_id = _oauth_item(expires_in_seconds=3600)
    db = _ItemDb(item, _client_row(client_id))
    exchange = AsyncMock()

    with patch.object(mcp_oauth, "refresh_access_token", exchange):
        await secrets_runtime._maybe_refresh_oauth_token(
            db=db, owner_id=item.user_id, installed_item_id=item.id, request_id=uuid4(), source_ip=None
        )

    exchange.assert_not_awaited()


@pytest.mark.asyncio
async def test_oauth_item_without_a_recorded_expiry_is_never_proactively_refreshed():
    item, client_id = _oauth_item(expires_in_seconds=10)
    item.custom_config["auth"]["expires_at"] = None
    db = _ItemDb(item, _client_row(client_id))
    exchange = AsyncMock()

    with patch.object(mcp_oauth, "refresh_access_token", exchange):
        await secrets_runtime._maybe_refresh_oauth_token(
            db=db, owner_id=item.user_id, installed_item_id=item.id, request_id=uuid4(), source_ip=None
        )

    exchange.assert_not_awaited()
