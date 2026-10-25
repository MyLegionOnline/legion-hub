"""Regressions from a Cloudflare HTTP MCP stuck on "Secret unavailable".

1. Login requested the authorization server's *entire* ``scopes_supported``
   (~7.5k chars for Cloudflare), overflowing ``McpOAuthFlow.scope`` → 500 on
   "Sign in". The scope now comes from the protected resource's metadata.
2. Once the access token expired unused, the refresh path could not find the
   access-token binding (expired == inactive), so it never refreshed again.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from app.core import mcp_oauth
from app.core.secrets import runtime as secrets_runtime
from app.models.secret import SecretPurpose, SecretState


def test_login_scope_uses_protected_resource_scopes_plus_offline_access():
    prm = {"scopes_supported": ["user:read", "account:read"]}
    asm = {"scopes_supported": ["offline_access", "user:read", "account:read"] + [f"s{i}" for i in range(2000)]}
    assert mcp_oauth._login_scope(prm, asm) == "user:read account:read offline_access"


def test_login_scope_never_falls_back_to_the_authorization_servers_list():
    asm = {"scopes_supported": ["offline_access", "everything"]}
    assert mcp_oauth._login_scope({}, asm) is None
    assert mcp_oauth._login_scope(None, asm) is None


def test_login_scope_omitted_when_too_long_to_store():
    prm = {"scopes_supported": [f"scope-{i}" for i in range(200)]}
    assert mcp_oauth._login_scope(prm, None) is None


class _Binding:
    def __init__(self):
        self.id = uuid4()


class _Secret:
    def __init__(self, *, state: SecretState, expires_at: datetime | None):
        self.id = uuid4()
        self.state = state
        self.expires_at = expires_at


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _Db:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _stmt):
        return _Result(self._rows)


def _expired_access_db():
    binding = _Binding()
    secret = _Secret(state=SecretState.ACTIVE, expires_at=datetime.now(timezone.utc) - timedelta(days=14))
    return _Db([(binding, secret)]), binding, secret


@pytest.mark.asyncio
async def test_expired_access_binding_is_found_for_refresh():
    db, binding, secret = _expired_access_db()
    found = await secrets_runtime._active_mcp_binding(
        db=db,
        owner_id=uuid4(),
        installed_item_id=uuid4(),
        purpose=SecretPurpose.MCP_AUTH_HEADER,
        allow_expired=True,
    )
    assert found == (binding.id, secret.id)


@pytest.mark.asyncio
async def test_expired_access_binding_is_still_not_active_for_use():
    db, _binding, _secret = _expired_access_db()
    found = await secrets_runtime._active_mcp_binding(
        db=db,
        owner_id=uuid4(),
        installed_item_id=uuid4(),
        purpose=SecretPurpose.MCP_AUTH_HEADER,
    )
    assert found is None


@pytest.mark.asyncio
async def test_disabled_binding_is_never_found_even_when_expired_is_allowed():
    db = _Db([(_Binding(), _Secret(state=SecretState.DISABLED, expires_at=None))])
    found = await secrets_runtime._active_mcp_binding(
        db=db,
        owner_id=uuid4(),
        installed_item_id=uuid4(),
        purpose=SecretPurpose.MCP_AUTH_HEADER,
        allow_expired=True,
    )
    assert found is None
