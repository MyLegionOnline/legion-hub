from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_gateway
from app.core.mcp_gateway import McpToolBinding, execute_mcp_tool
from app.core.secrets.crypto import SecretValue
from app.core.secrets.runtime import (
    MCP_UPSTREAM_AUTH_HEADER,
    McpSecretUnavailable,
    resolve_mcp_upstream_auth,
)
from app.models.secret import SecretBinding, SecretKind, SecretPurpose, SecretResourceType, SecretState, UserSecret


class _Result:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def all(self):
        return self.rows


class _BindingDb:
    def __init__(self, rows=()):
        self.rows = rows

    async def execute(self, _statement):
        return _Result(self.rows)


class _RoomDb:
    def __init__(self, item):
        self.item = item
        self.calls = 0

    async def execute(self, _statement):
        self.calls += 1
        if self.calls == 1:
            result = _Result()
            result.first = lambda: (object(),)
            return result
        result = _Result()
        result.all = lambda: [(self.item, object())]
        return result

    async def get(self, *_args):
        return None

    async def scalar(self, *_args):
        return "full"


def _binding(*, owner_id, item_id, binding_id=None):
    return SecretBinding(
        id=binding_id or uuid4(),
        secret_id=uuid4(),
        owner_id=owner_id,
        resource_type=SecretResourceType.INSTALLED_MCP,
        resource_id=item_id,
        purpose=SecretPurpose.MCP_AUTH_HEADER,
        injection_config={"mode": "bearer"},
    )


def _secret(*, owner_id, state=SecretState.ACTIVE):
    return UserSecret(
        id=uuid4(),
        owner_id=owner_id,
        kind=SecretKind.MCP_HEADER,
        label="GitHub token",
        ciphertext=b"ciphertext",
        key_id="test",
        state=state,
    )


@pytest.mark.asyncio
async def test_resolve_mcp_upstream_auth_is_owner_item_and_purpose_scoped(monkeypatch):
    owner_id = uuid4()
    item_id = uuid4()
    binding = _binding(owner_id=owner_id, item_id=item_id)
    secret = _secret(owner_id=owner_id)
    db = _BindingDb([(binding, secret)])
    calls = []

    async def resolve(**kwargs):
        calls.append(kwargs)
        return SecretValue("fixture-upstream-token")

    monkeypatch.setattr("app.core.secrets.runtime.resolve_for_use", resolve)

    headers = await resolve_mcp_upstream_auth(
        db=db,
        owner_id=owner_id,
        installed_item_id=item_id,
        request_id=uuid4(),
    )

    assert headers == {MCP_UPSTREAM_AUTH_HEADER: "Bearer fixture-upstream-token"}
    assert calls[0]["owner_id"] == owner_id
    assert calls[0]["resource_id"] == item_id
    assert calls[0]["resource_type"] == SecretResourceType.INSTALLED_MCP
    assert calls[0]["purpose"] == SecretPurpose.MCP_AUTH_HEADER
    assert calls[0]["binding_id"] == binding.id
    assert "fixture-upstream-token" not in repr(binding)
    assert "fixture-upstream-token" not in repr(secret)


@pytest.mark.asyncio
async def test_resolve_mcp_upstream_auth_fails_closed_for_missing_or_disabled_binding(monkeypatch):
    owner_id = uuid4()
    item_id = uuid4()
    binding = _binding(owner_id=owner_id, item_id=item_id)
    disabled = _secret(owner_id=owner_id, state=SecretState.DISABLED)
    resolve = AsyncMock(return_value=SecretValue("must-not-be-used"))
    monkeypatch.setattr("app.core.secrets.runtime.resolve_for_use", resolve)

    with pytest.raises(McpSecretUnavailable):
        await resolve_mcp_upstream_auth(
            db=_BindingDb([(binding, disabled)]),
            owner_id=owner_id,
            installed_item_id=item_id,
        )
    resolve.assert_awaited_once()


@pytest.mark.asyncio
async def test_rotated_secret_is_used_by_the_next_gateway_boundary(monkeypatch):
    owner_id = uuid4()
    item_id = uuid4()
    binding_id = uuid4()
    resolve = AsyncMock(side_effect=[SecretValue("old-token"), SecretValue("rotated-token")])
    monkeypatch.setattr("app.core.secrets.runtime.resolve_for_use", resolve)

    first = await resolve_mcp_upstream_auth(
        db=_BindingDb(),
        owner_id=owner_id,
        installed_item_id=item_id,
        binding_id=binding_id,
    )
    second = await resolve_mcp_upstream_auth(
        db=_BindingDb(),
        owner_id=owner_id,
        installed_item_id=item_id,
        binding_id=binding_id,
    )

    assert first[MCP_UPSTREAM_AUTH_HEADER] == "Bearer old-token"
    assert second[MCP_UPSTREAM_AUTH_HEADER] == "Bearer rotated-token"
    assert resolve.await_count == 2


@pytest.mark.asyncio
async def test_execute_uses_header_boundary_and_never_runtime_auth_payload(monkeypatch):
    owner_id = uuid4()
    item_id = uuid4()
    secret = "fixture-upstream-token"
    binding = McpToolBinding(
        installed_item_id=item_id,
        market_item_id=None,
        market_slug=None,
        tool_name="get_commit",
        custom_config={
            "url": "https://api.github.example/mcp",
            "auth": {"mode": "bearer", "secret_binding_id": str(uuid4())},
        },
        isolation="stateless",
        auth_binding_id=uuid4(),
    )
    post = AsyncMock(return_value={"ok": True})

    with (
        patch(
            "app.core.mcp_gateway.resolve_mcp_upstream_auth",
            new_callable=AsyncMock,
            return_value={MCP_UPSTREAM_AUTH_HEADER: f"Bearer {secret}"},
        ) as resolve,
        patch("app.core.mcp_gateway._gateway_post", post),
    ):
        result = await execute_mcp_tool(binding, owner_id, uuid4(), {"repository": "legion"}, db=object())

    assert result == json.dumps({"ok": True}, ensure_ascii=False)
    resolve.assert_awaited_once()
    assert resolve.await_args.kwargs["owner_id"] == owner_id
    assert resolve.await_args.kwargs["installed_item_id"] == item_id
    assert resolve.await_args.kwargs["binding_id"] == binding.auth_binding_id
    payload = post.await_args.args[3]
    assert "runtime_auth" not in json.dumps(payload)
    assert secret not in json.dumps(payload)
    extra_headers = post.await_args.kwargs["extra_headers"]
    assert extra_headers[MCP_UPSTREAM_AUTH_HEADER] == f"Bearer {secret}"
    assert extra_headers["X-Legion-Correlation-ID"].startswith("mcp_call_")
    assert post.await_args.kwargs["timeout"] == mcp_gateway.MCP_GATEWAY_TIMEOUT


@pytest.mark.asyncio
async def test_execute_does_not_call_gateway_when_mcp_secret_is_unavailable(monkeypatch):
    binding = McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=None,
        market_slug=None,
        tool_name="get_commit",
        custom_config={"url": "https://api.github.example/mcp", "auth": {"mode": "bearer"}},
        isolation="stateless",
    )

    with (
        patch(
            "app.core.mcp_gateway.resolve_mcp_upstream_auth",
            new_callable=AsyncMock,
            side_effect=McpSecretUnavailable(),
        ),
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock) as post,
    ):
        result = await execute_mcp_tool(binding, uuid4(), uuid4(), {}, db=object())

    assert result == "Error: MCP authorization is not configured"
    post.assert_not_awaited()


@pytest.mark.asyncio
async def test_discovery_resolves_secret_once_and_sends_only_gateway_header(monkeypatch):
    from app.core.mcp_gateway import room_mcp_tools

    owner_id = uuid4()
    item_id = uuid4()
    binding_id = uuid4()
    item = SimpleNamespace(
        id=item_id,
        user_id=owner_id,
        type="mcp",
        status="active",
        is_custom=True,
        custom_config={
            "url": "https://api.github.example/mcp",
            "auth": {"mode": "bearer", "secret_binding_id": str(binding_id)},
        },
        market_item_id=None,
        market_slug=None,
        name="GitHub (read-only)",
        isolation="stateless",
    )
    post = AsyncMock(
        return_value={
            "status": "verified",
            "tools": [{"name": "get_commit", "description": "Read commit", "inputSchema": {}}],
        }
    )
    token = "fixture-upstream-token"

    with (
        patch(
            "app.core.mcp_gateway.resolve_mcp_upstream_auth",
            new_callable=AsyncMock,
            return_value={MCP_UPSTREAM_AUTH_HEADER: f"Bearer {token}"},
        ) as resolve,
        patch("app.core.mcp_gateway._gateway_post", post),
        patch("app.core.mcp_gateway._gateway_get", new_callable=AsyncMock) as legacy_get,
    ):
        ant, oai, bindings = await room_mcp_tools(uuid4(), owner_id, _RoomDb(item))

    assert [tool["name"] for tool in ant] == ["mcp__GitHub_read-only__get_commit"]
    assert oai[0]["function"]["name"] == ant[0]["name"]
    assert next(iter(bindings.values())).auth_binding_id == binding_id
    resolve.assert_awaited_once()
    assert resolve.await_args.kwargs["owner_id"] == owner_id
    assert resolve.await_args.kwargs["installed_item_id"] == item_id
    assert resolve.await_args.kwargs["binding_id"] == binding_id
    post.assert_awaited_once()
    assert post.await_args.args[0] == "/v1/tools/discover"
    payload = post.await_args.args[3]
    assert payload["custom_config"]["auth"] == {"mode": "bearer"}
    assert token not in json.dumps(payload)
    extra_headers = post.await_args.kwargs["extra_headers"]
    assert extra_headers[MCP_UPSTREAM_AUTH_HEADER] == f"Bearer {token}"
    assert extra_headers["X-Legion-Correlation-ID"].startswith("mcpd_hub_")
    legacy_get.assert_not_awaited()
