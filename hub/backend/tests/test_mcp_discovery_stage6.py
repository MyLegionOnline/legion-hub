"""Stage 6: rollout flag, correlation propagation and bounded metrics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_discovery
from app.core.http_mcp_rollout import rollout_policy
from app.core.mcp_gateway import execute_mcp_tool, room_mcp_tools
from app.models.market import UserInstalledItem
from app.models.mcp_connection import McpConnectionSnapshot


class _Db:
    def __init__(self, snapshot=None):
        self.snapshot = snapshot
        self.commits = 0

    async def get(self, model, _item_id):
        return self.snapshot if model is McpConnectionSnapshot else None

    def add(self, value):
        if isinstance(value, McpConnectionSnapshot):
            self.snapshot = value

    async def commit(self):
        self.commits += 1

    async def scalar(self, *_args):
        return "readonly"


class _RoomResult:
    def __init__(self, *, first=None, rows=()):
        self._first = first
        self._rows = list(rows)

    def first(self):
        return self._first

    def all(self):
        return self._rows


class _RoomDb(_Db):
    def __init__(self, item, snapshot):
        super().__init__(snapshot)
        self.item = item
        self._execute_calls = 0

    async def execute(self, _statement):
        self._execute_calls += 1
        if self._execute_calls == 1:
            return _RoomResult(first=(object(),))
        return _RoomResult(rows=[(self.item, object())])


def _item():
    return UserInstalledItem(
        id=uuid4(),
        user_id=uuid4(),
        type="mcp",
        status="active",
        is_custom=True,
        name="GitHub (read-only)",
        custom_config={
            "url": "https://mcp.example.test/github/readonly",
            "auth": {"mode": "none"},
        },
    )


@pytest.mark.asyncio
async def test_rollout_off_hides_existing_snapshot_without_deleting_it(monkeypatch):
    item = _item()
    checked = datetime.now(timezone.utc)
    snapshot = McpConnectionSnapshot(
        installed_item_id=item.id,
        owner_id=item.user_id,
        status="verified",
        tools_snapshot=[{"name": "get_commit", "description": "", "inputSchema": {}}],
        tool_fingerprint="sha256:tool",
        source_fingerprint=mcp_discovery._source_fingerprint(item),
        checked_at=checked,
        expires_at=checked + timedelta(seconds=60),
    )
    monkeypatch.setenv("HTTP_MCP_VERIFIED_DISCOVERY_V1", "off")

    status = await mcp_discovery.mcp_connection_status(_Db(snapshot), item)

    assert rollout_policy().enabled is False
    assert status.status == "unknown"
    assert status.available is False
    assert status.tools == []
    assert snapshot.status == "verified"
    assert snapshot.tools_snapshot


@pytest.mark.asyncio
async def test_discovery_correlation_is_forwarded_and_measured(monkeypatch):
    monkeypatch.setenv("HTTP_MCP_VERIFIED_DISCOVERY_V1", "on")
    item = _item()
    db = _Db()
    gateway = AsyncMock(
        return_value={
            "status": "verified",
            "tools": [{"name": "get_commit", "description": "", "inputSchema": {}}],
        }
    )

    with (
        patch("app.core.mcp_discovery._gateway_post", gateway),
        patch("app.core.mcp_discovery.record_mcp_discovery") as record,
    ):
        status = await mcp_discovery.verify_mcp_connection(db, item, force=True)

    correlation = gateway.await_args.kwargs["extra_headers"]["X-Legion-Correlation-ID"]
    assert correlation == status.correlation_id
    assert correlation.startswith("mcpd_hub_")
    record.assert_called_once()
    assert record.call_args.kwargs["outcome"] == "success"
    assert record.call_args.kwargs["code"] == "verified"


@pytest.mark.asyncio
async def test_stage6_smoke_create_bind_verify_room_and_call(monkeypatch):
    """The Hub path keeps one binding and one verified snapshot end to end."""

    monkeypatch.setenv("HTTP_MCP_VERIFIED_DISCOVERY_V1", "on")
    item = _item()
    item.custom_config["auth"] = {"mode": "bearer", "secret_binding_id": str(uuid4())}
    db = _Db()
    upstream_header = {"X-Legion-MCP-Upstream-Authorization": "Bearer fixture-upstream-token"}
    discovery_gateway = AsyncMock(
        return_value={
            "status": "verified",
            "tools": [{"name": "get_commit", "description": "", "inputSchema": {}}],
        }
    )

    with (
        patch("app.core.mcp_gateway.resolve_mcp_upstream_auth", new_callable=AsyncMock, return_value=upstream_header),
        patch("app.core.mcp_discovery._gateway_post", discovery_gateway),
    ):
        verified = await mcp_discovery.verify_mcp_connection(db, item, force=True)

    room_db = _RoomDb(item, db.snapshot)
    ant, _oai, bindings = await room_mcp_tools(uuid4(), item.user_id, room_db)
    assert verified.available is True
    assert [tool["name"] for tool in ant] == ["mcp__GitHub_read-only__get_commit"]

    call_gateway = AsyncMock(return_value={"content": [{"type": "text", "text": "fixture commit"}]})
    with (
        patch("app.core.mcp_gateway.resolve_mcp_upstream_auth", new_callable=AsyncMock, return_value=upstream_header),
        patch("app.core.mcp_gateway._gateway_post", call_gateway),
    ):
        result = await execute_mcp_tool(
            next(iter(bindings.values())),
            item.user_id,
            uuid4(),
            {"repository": "legion/legion"},
            db=room_db,
        )

    assert "fixture commit" in result
    assert call_gateway.await_args.kwargs["extra_headers"]["X-Legion-MCP-Upstream-Authorization"] == upstream_header["X-Legion-MCP-Upstream-Authorization"]
    assert call_gateway.await_args.kwargs["extra_headers"]["X-Legion-Correlation-ID"] == verified.correlation_id
