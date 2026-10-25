"""internal design notes — a successful login must refresh the item's MCP
connection snapshot itself, not rely on the popup's opener tab staying open
and polling until it notices.

Found live: a user completed a real Cloudflare OAuth login (backend logs
confirmed tokens stored, custom_config.auth updated), but the card kept
showing "Provider sign-in required" because nothing ever re-ran discovery —
the stale pre-login McpConnectionSnapshot (last_error_code=mcp_oauth_required)
just sat there forever. The callback route now probes the endpoint itself
right after a successful token exchange.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import user_installed_items as registry
from app.core import mcp_oauth
from app.models.market import UserInstalledItem


class _FakeQueryParams(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _FakeAppState:
    def __init__(self, keyring):
        self.secret_keyring = keyring


class _FakeApp:
    def __init__(self, keyring):
        self.state = _FakeAppState(keyring)


class _FakeRequest:
    def __init__(self, *, query_params, keyring=object()):
        self.query_params = _FakeQueryParams(query_params)
        self.app = _FakeApp(keyring)


class _Db:
    def __init__(self, item):
        self.item = item

    async def get(self, model, obj_id):
        if model is UserInstalledItem and obj_id == self.item.id:
            return self.item
        return None


def _item() -> UserInstalledItem:
    return UserInstalledItem(
        id=uuid4(), user_id=uuid4(), type="mcp", is_custom=True, name="OAuth MCP",
        custom_config={"url": "https://mcp.example.com/mcp"},
    )


@pytest.mark.asyncio
async def test_successful_callback_forces_a_fresh_connection_probe():
    item = _item()
    db = _Db(item)
    request = _FakeRequest(query_params={"state": "the-state", "code": "the-code"})
    outcome = mcp_oauth.LoginOutcome(ok=True, installed_item_id=item.id)

    with patch.object(mcp_oauth, "complete_login", AsyncMock(return_value=outcome)), patch.object(
        registry, "verify_mcp_connection", AsyncMock()
    ) as verify:
        response = await registry.mcp_oauth_callback(request, db)

    assert response.status_code == 200
    verify.assert_awaited_once()
    _args, kwargs = verify.call_args
    assert _args[0] is db
    assert _args[1] is item
    assert kwargs.get("force") is True


@pytest.mark.asyncio
async def test_a_failed_probe_after_login_does_not_turn_success_into_an_error():
    item = _item()
    db = _Db(item)
    request = _FakeRequest(query_params={"state": "the-state", "code": "the-code"})
    outcome = mcp_oauth.LoginOutcome(ok=True, installed_item_id=item.id)

    with patch.object(mcp_oauth, "complete_login", AsyncMock(return_value=outcome)), patch.object(
        registry, "verify_mcp_connection", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        response = await registry.mcp_oauth_callback(request, db)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_a_failed_login_never_triggers_a_probe():
    item = _item()
    db = _Db(item)
    request = _FakeRequest(query_params={"state": "the-state", "code": "the-code"})
    outcome = mcp_oauth.LoginOutcome(ok=False, error_code="mcp_oauth_token_rejected")

    with patch.object(mcp_oauth, "complete_login", AsyncMock(return_value=outcome)), patch.object(
        registry, "verify_mcp_connection", AsyncMock()
    ) as verify:
        response = await registry.mcp_oauth_callback(request, db)

    assert response.status_code == 502
    verify.assert_not_awaited()
