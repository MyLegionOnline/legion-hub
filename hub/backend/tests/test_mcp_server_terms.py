"""The Server Terms of an optional Market MCP (Terms §6,
legion_market/legal/PENDING.md item 56), seen from the Hub.

The Gateway refuses a call of a server whose Server Terms the Legion account
has not accepted in Market. The Hub names that account to the Gateway
(``X-Legion-Account-Id``) on a call with its service token, and passes the
refusal to the agent with the Market page to accept on. Fully mocked — no
database.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core import legion_wallet
from app.core.mcp_gateway import McpToolBinding, execute_mcp_tool
from tests.test_legion_wallet import KEY, _binding, _status_error, central, local_wallet_untouched  # noqa: F401

URL = "https://market.example/server-terms/search"
REFUSAL = {"code": "server_terms_required", "document": "search", "version": "2026-10-25", "url": URL,
           "message": f"Read and accept the Search MCP Server Terms at {URL} before this server can be used"}


@pytest.fixture()
def hub_wallet(monkeypatch):
    monkeypatch.setattr(legion_wallet, "MCP_CREDITS_WALLET", "hub")
    with (
        patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=0),
        patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
        patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock),
        patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock),
    ):
        yield


@pytest.mark.asyncio
async def test_our_hub_names_the_legion_account_to_the_gateway(hub_wallet):
    with (
        patch("app.core.mcp_gateway._legion_account_id", new_callable=AsyncMock, return_value="acct-1") as lookup,
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock, return_value={"ok": True}) as post,
    ):
        user_id = uuid.uuid4()
        await execute_mcp_tool(_binding(), user_id, uuid.uuid4(), {"query": "legion"})
    lookup.assert_awaited_once_with(user_id)
    assert post.await_args.kwargs["extra_headers"]["X-Legion-Account-Id"] == "acct-1"


@pytest.mark.asyncio
async def test_a_user_without_a_legion_account_sends_no_header(hub_wallet):
    with (
        patch("app.core.mcp_gateway._legion_account_id", new_callable=AsyncMock, return_value=None),
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock, return_value={"ok": True}) as post,
    ):
        await execute_mcp_tool(_binding(), uuid.uuid4(), uuid.uuid4(), {})
    assert "X-Legion-Account-Id" not in post.await_args.kwargs["extra_headers"]


@pytest.mark.asyncio
async def test_a_personal_mcp_does_not_look_up_the_account(hub_wallet):
    personal = McpToolBinding(installed_item_id=uuid.uuid4(), market_item_id=None, market_slug=None,
                              tool_name="t", custom_config=None, isolation="stateless", managed_stdio=False)
    with (
        patch("app.core.mcp_gateway._legion_account_id", new_callable=AsyncMock) as lookup,
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock, return_value={"ok": True}),
    ):
        await execute_mcp_tool(personal, uuid.uuid4(), uuid.uuid4(), {})
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_refusal_tells_the_agent_where_to_accept(hub_wallet):
    with (
        patch("app.core.mcp_gateway._legion_account_id", new_callable=AsyncMock, return_value="acct-1"),
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock,
              side_effect=_status_error(403, REFUSAL)),
    ):
        result = await execute_mcp_tool(_binding(), uuid.uuid4(), uuid.uuid4(), {})
    assert result.startswith("Error: [server_terms_required]") and URL in result


@pytest.mark.asyncio
async def test_a_key_paid_call_names_no_account_and_gets_the_same_refusal(central, local_wallet_untouched):  # noqa: F811
    with (
        patch.object(legion_wallet, "key_for", new_callable=AsyncMock, return_value=KEY),
        patch("app.core.mcp_gateway._legion_account_id", new_callable=AsyncMock) as lookup,
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock,
              side_effect=_status_error(403, REFUSAL)),
    ):
        result = await execute_mcp_tool(_binding(), uuid.uuid4(), uuid.uuid4(), {})
    lookup.assert_not_awaited()
    assert result.startswith("Error: [server_terms_required]") and URL in result
