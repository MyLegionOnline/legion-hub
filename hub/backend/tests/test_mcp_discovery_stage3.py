"""Stage 3: HTTP MCP verification snapshots and safe availability state."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_discovery
from app.core.secrets.runtime import McpSecretUnavailable
from app.models.market import UserInstalledItem
from app.models.mcp_connection import McpConnectionSnapshot


class _Db:
    def __init__(self, snapshot=None):
        self.snapshot = snapshot
        self.commits = 0

    async def get(self, model, item_id):
        if model is McpConnectionSnapshot:
            return self.snapshot
        return None

    def add(self, value):
        if isinstance(value, McpConnectionSnapshot):
            self.snapshot = value

    async def commit(self):
        self.commits += 1


def _item(*, user_id=None, config=None):
    return UserInstalledItem(
        id=uuid4(),
        user_id=user_id or uuid4(),
        type="mcp",
        is_custom=True,
        name="GitHub (read-only)",
        custom_config=config or {"url": "https://mcp.example.test", "auth": {"mode": "none"}},
    )


@pytest.mark.asyncio
async def test_missing_snapshot_is_unknown_and_not_available():
    item = _item()
    status = await mcp_discovery.mcp_connection_status(_Db(), item)

    assert status.status == "unknown"
    assert status.available is False
    assert status.ready_for_room is False
    assert status.tools == []


@pytest.mark.asyncio
async def test_verification_persists_bounded_safe_tools_and_ttl():
    item = _item()
    db = _Db()
    gateway = AsyncMock(
        return_value={
            "status": "verified",
            "tools": [
                {
                    "name": "get_commit",
                    "description": "Read a commit.",
                    "inputSchema": {"type": "object"},
                }
            ],
        }
    )

    with patch("app.core.mcp_discovery._gateway_post", gateway):
        status = await mcp_discovery.verify_mcp_connection(db, item, force=True)

    assert status.status == "verified"
    assert status.available is True
    assert status.tool_count == 1
    assert status.tools[0].name == "get_commit"
    assert status.expires_at is not None
    assert status.ttl_seconds <= mcp_discovery.SNAPSHOT_TTL_SECONDS
    assert db.snapshot.source_fingerprint.startswith("sha256:")
    assert "mcp.example.test" not in repr(db.snapshot)
    assert gateway.await_args.args[0] == "/v1/tools/discover"
    assert gateway.await_args.kwargs["extra_headers"]["X-Legion-Correlation-ID"].startswith("mcpd_hub_")


@pytest.mark.asyncio
async def test_auth_failure_is_safe_and_gateway_is_not_called():
    item = _item(
        config={
            "url": "https://mcp.example.test",
            "auth": {"mode": "bearer"},
        }
    )
    db = _Db()
    gateway = AsyncMock()

    with (
        patch(
            "app.core.mcp_discovery.resolve_mcp_upstream_auth",
            new_callable=AsyncMock,
            side_effect=McpSecretUnavailable(),
        ),
        patch("app.core.mcp_discovery._gateway_post", gateway),
    ):
        status = await mcp_discovery.verify_mcp_connection(db, item, force=True)

    assert status.status == "error"
    assert status.available is False
    assert status.last_error_code == "mcp_secret_unavailable"
    assert status.last_error is not None
    assert status.last_error.details == {}
    gateway.assert_not_awaited()
    assert "mcp.example.test" not in status.model_dump_json()


@pytest.mark.asyncio
async def test_oauth_item_without_usable_token_asks_for_provider_login():
    item = _item(config={"url": "https://mcp.example.test", "auth": {"mode": "oauth"}})
    gateway = AsyncMock()

    with (
        patch(
            "app.core.mcp_discovery.resolve_mcp_upstream_auth",
            new_callable=AsyncMock,
            side_effect=McpSecretUnavailable(),
        ),
        patch("app.core.mcp_discovery._gateway_post", gateway),
    ):
        status = await mcp_discovery.verify_mcp_connection(_Db(), item, force=True)

    assert status.last_error_code == "mcp_oauth_required"
    gateway.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_verified_snapshot_is_expired_and_tools_are_hidden():
    item = _item()
    checked_at = datetime.now(timezone.utc) - timedelta(minutes=2)
    snapshot = McpConnectionSnapshot(
        installed_item_id=item.id,
        owner_id=item.user_id,
        status="verified",
        tools_snapshot=[{"name": "get_commit", "description": "", "inputSchema": {}}],
        tool_fingerprint="sha256:tool",
        source_fingerprint=mcp_discovery._source_fingerprint(item),
        checked_at=checked_at,
        expires_at=checked_at + timedelta(seconds=mcp_discovery.SNAPSHOT_TTL_SECONDS),
    )

    status = await mcp_discovery.mcp_connection_status(_Db(snapshot), item)

    assert status.status == "expired"
    assert status.available is False
    assert status.tools == []
    assert status.tool_fingerprint is None


@pytest.mark.asyncio
async def test_invalidation_clears_verified_tools():
    item = _item()
    snapshot = McpConnectionSnapshot(
        installed_item_id=item.id,
        owner_id=item.user_id,
        status="verified",
        tools_snapshot=[{"name": "get_commit", "description": "", "inputSchema": {}}],
        tool_fingerprint="sha256:tool",
        source_fingerprint=mcp_discovery._source_fingerprint(item),
        checked_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
    )
    db = _Db(snapshot)

    await mcp_discovery.invalidate_mcp_snapshot(db, item.id, owner_id=item.user_id)

    assert snapshot.status == "unknown"
    assert snapshot.tools_snapshot == []
    assert snapshot.tool_fingerprint is None
    assert snapshot.expires_at is None


@pytest.mark.asyncio
async def test_tool_size_limit_fails_closed_without_persisting_tools():
    item = _item()
    db = _Db()
    tools = [
        {"name": f"tool_{idx}", "description": "", "inputSchema": {"type": "object"}}
        for idx in range(mcp_discovery.MAX_TOOLS + 1)
    ]

    with patch(
        "app.core.mcp_discovery._gateway_post",
        new_callable=AsyncMock,
        return_value={"status": "verified", "tools": tools},
    ):
        status = await mcp_discovery.verify_mcp_connection(db, item, force=True)

    assert status.status == "error"
    assert status.last_error_code == "mcp_bad_response"
    assert db.snapshot.tools_snapshot == []
