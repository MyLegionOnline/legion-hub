"""Stage 4: runtime preflight, safe degradation, and cancellation fences."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_discovery
from app.core.mcp_gateway import (
    McpCapabilityFailure,
    McpToolBinding,
    execute_mcp_tool,
    format_mcp_failure_note,
    room_mcp_tools,
    task_requires_mcp_capability,
)
from app.core.proxy_session import ProxyRunTimeout, _call_with_heartbeat
from app.models.market import UserInstalledItem
from app.models.mcp_connection import McpConnectionSnapshot


class _Result:
    def __init__(self, *, first=None, rows=()):
        self._first = first
        self._rows = list(rows)

    def first(self):
        return self._first

    def all(self):
        return self._rows


class _RoomDb:
    def __init__(self, item, snapshot):
        self.item = item
        self.snapshot = snapshot
        self.calls = 0

    async def execute(self, _statement):
        self.calls += 1
        if self.calls == 1:
            return _Result(first=(object(),))
        return _Result(rows=[(self.item, object())])

    async def get(self, model, _item_id):
        if model is McpConnectionSnapshot:
            return self.snapshot
        return None

    async def scalar(self, *_args):
        return "readonly"


def _item(owner_id=None):
    return UserInstalledItem(
        id=uuid4(),
        user_id=owner_id or uuid4(),
        type="mcp",
        status="active",
        is_custom=True,
        name="GitHub (read-only)",
        custom_config={
            "url": "https://mcp.example.test",
            "auth": {"mode": "none"},
        },
        isolation="stateless",
    )


def _verified_snapshot(item):
    checked = datetime.now(timezone.utc)
    tools = [{"name": "get_commit", "description": "Read", "inputSchema": {}}]
    return McpConnectionSnapshot(
        installed_item_id=item.id,
        owner_id=item.user_id,
        status="verified",
        tools_snapshot=tools,
        tool_fingerprint=mcp_discovery._tools_fingerprint(tools),
        source_fingerprint=mcp_discovery._source_fingerprint(item),
        checked_at=checked,
        expires_at=checked + timedelta(seconds=60),
    )


@pytest.mark.asyncio
async def test_room_tools_use_fresh_verified_snapshot_without_gateway_call():
    item = _item()
    db = _RoomDb(item, _verified_snapshot(item))

    with patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock) as gateway:
        ant, oai, bindings = await room_mcp_tools(uuid4(), item.user_id, db)

    assert [tool["name"] for tool in ant] == ["mcp__GitHub_read-only__get_commit"]
    assert oai[0]["function"]["name"] == ant[0]["name"]
    assert next(iter(bindings.values())).custom_config["auth"] == {"mode": "none"}
    gateway.assert_not_awaited()


@pytest.mark.asyncio
async def test_expired_snapshot_runs_one_bounded_probe_then_uses_cache():
    item = _item()
    snapshot = _verified_snapshot(item)
    snapshot.checked_at = datetime.now(timezone.utc) - timedelta(minutes=2)
    snapshot.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db = SimpleNamespace(snapshot=snapshot, commits=0)

    async def get(model, _item_id):
        return db.snapshot if model is McpConnectionSnapshot else None

    def add(value):
        db.snapshot = value

    async def commit():
        db.commits += 1

    db.get = get
    db.add = add
    db.commit = commit
    gateway = AsyncMock(
        return_value={
            "status": "verified",
            "tools": [{"name": "get_commit", "description": "", "inputSchema": {}}],
        }
    )

    with patch("app.core.mcp_discovery._gateway_post", gateway):
        first = await mcp_discovery.fresh_mcp_connection_status(db, item)
        second = await mcp_discovery.fresh_mcp_connection_status(db, item)

    assert first.available is True
    assert second.available is True
    gateway.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_execution_rechecks_snapshot_even_without_bearer_auth():
    item = _item()
    snapshot = _verified_snapshot(item)
    db = SimpleNamespace()

    async def get(model, _item_id):
        return snapshot if model is McpConnectionSnapshot else None

    db.get = get
    binding = McpToolBinding(
        installed_item_id=item.id,
        market_item_id=None,
        market_slug=None,
        tool_name="get_commit",
        custom_config={"url": "https://mcp.example.test", "auth": {"mode": "none"}},
        isolation="stateless",
        expected_fingerprint=snapshot.tool_fingerprint,
    )

    with patch(
        "app.core.mcp_gateway._gateway_post",
        new_callable=AsyncMock,
        return_value={"ok": True},
    ):
        result = await execute_mcp_tool(binding, item.user_id, uuid4(), {}, db=db)

    assert result == '{"ok": true}'


def test_failure_note_is_safe_and_explicit_requirement_is_detected():
    failure = McpCapabilityFailure(
        installed_item_id=uuid4(),
        display_name="GitHub (read-only)",
        code="mcp_auth_failed",
        message="ignored",
        remedy="ignored",
        correlation_id="corr-safe",
    )
    note = format_mcp_failure_note([failure])

    assert "GitHub (read-only) is not available: MCP authorization failed" in note
    assert "raw HTTP/curl/requests" in note
    assert "corr-safe" not in note
    assert task_requires_mcp_capability("get the first commit from GitHub", [failure])


@pytest.mark.asyncio
async def test_heartbeat_deadline_cancels_provider_task():
    cancelled = asyncio.Event()

    async def slow_provider():
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    with pytest.raises(ProxyRunTimeout):
        await _call_with_heartbeat(
            slow_provider(),
            room_id=uuid4(),
            agent_id=None,
            session_id=uuid4(),
            task_id=uuid4(),
            interval=0.01,
            deadline_monotonic=time.monotonic() + 0.03,
        )

    assert cancelled.is_set()
