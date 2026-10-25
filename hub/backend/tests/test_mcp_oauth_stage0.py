"""internal design notes Stage 0 — the card names the provider instead of a silent failure."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_discovery
from app.models.market import UserInstalledItem
from app.models.mcp_connection import McpConnectionSnapshot


class _Db:
    def __init__(self):
        self.snapshot = None

    async def get(self, model, item_id):
        if model is McpConnectionSnapshot:
            return self.snapshot
        return None

    def add(self, value):
        if isinstance(value, McpConnectionSnapshot):
            self.snapshot = value

    async def commit(self):
        return None


def _item():
    return UserInstalledItem(
        id=uuid4(),
        user_id=uuid4(),
        type="mcp",
        is_custom=True,
        name="Provider with sign-in",
        custom_config={"url": "https://provider.example/mcp", "auth": {"mode": "none"}},
    )


def _gateway_oauth_body(provider: str | None):
    return {
        "status": "error",
        "error": {
            "code": "mcp_oauth_required",
            "message": "MCP server requires an OAuth login",
            "retryable": False,
            "provider": provider,
        },
    }


@pytest.mark.asyncio
async def test_oauth_required_names_the_provider_on_the_card():
    item = _item()
    db = _Db()

    with patch.object(
        mcp_discovery, "_gateway_post", AsyncMock(return_value=_gateway_oauth_body("Cloudflare API MCP Server"))
    ):
        status = await mcp_discovery.verify_mcp_connection(db, item, force=True)

    assert status.status == "error"
    assert status.last_error_code == "mcp_oauth_required"
    assert status.last_error.message == "The server requires a sign-in at the provider (Cloudflare API MCP Server)."
    assert status.last_error.details == {"provider": "Cloudflare API MCP Server"}
    # A repeat check will not create a sign-in — the card must not promise it.
    assert status.last_error.retryable is False
    assert status.available is False
    assert status.tools == []


@pytest.mark.asyncio
async def test_oauth_required_without_a_label_still_explains_the_cause():
    item = _item()
    db = _Db()

    with patch.object(mcp_discovery, "_gateway_post", AsyncMock(return_value=_gateway_oauth_body(None))):
        status = await mcp_discovery.verify_mcp_connection(db, item, force=True)

    assert status.last_error_code == "mcp_oauth_required"
    assert status.last_error.message == "The server requires a sign-in at the provider."
    assert status.last_error.details == {}


@pytest.mark.asyncio
async def test_auth_failure_is_unchanged_and_carries_no_provider():
    """A bad bearer stays a bad bearer — Stage 0 does not re-label it."""

    item = _item()
    db = _Db()
    body = {"status": "error", "error": {"code": "mcp_auth_failed", "provider": "Provider"}}

    with patch.object(mcp_discovery, "_gateway_post", AsyncMock(return_value=body)):
        status = await mcp_discovery.verify_mcp_connection(db, item, force=True)

    assert status.last_error_code == "mcp_auth_failed"
    assert status.last_error.message == "HTTP MCP authorization failed."
    assert status.last_error.details == {}


@pytest.mark.parametrize(
    "value",
    [
        None,
        123,
        "",
        "   ",
        "https://evil.example/phish",
        "x" * 65,
        "provider\x00with\x07control",
    ],
)
def test_provider_label_is_re_sanitized_on_this_side_of_the_gateway(value):
    """The label came from upstream via the gateway — the Hub does not take it on faith."""

    cleaned = mcp_discovery._safe_provider(value)

    assert cleaned is None or (
        "://" not in cleaned
        and len(cleaned) <= mcp_discovery._MAX_PROVIDER_CHARS
        and all(ord(char) >= 0x20 and ord(char) != 0x7F for char in cleaned)
    )


def test_provider_is_ignored_for_every_other_error_code():
    error = mcp_discovery._safe_error("mcp_unreachable", "mcpd_test", "Cloudflare")

    assert error["details"] == {}
    assert error["message"] == "HTTP MCP is unavailable."


def test_unknown_code_from_the_gateway_still_collapses_to_a_safe_one():
    error = mcp_discovery._safe_error("mcp_totally_new_code", "mcpd_test", "Cloudflare")

    assert error["code"] == "mcp_http_error"
    assert error["details"] == {}
