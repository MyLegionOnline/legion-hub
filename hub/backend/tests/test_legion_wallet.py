"""A self-hosted Hub pays Market MCP from the Legion account's central wallet
(internal design notes §3.10, stage 3b).

With ``MCP_CREDITS_WALLET=legion`` on a Hub with local users, a Market MCP
call goes to the Legion Gateway with the owner's Legion API key; the local
wallet is never read or written. Fully mocked — no database.
"""
from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core import legion_auth, legion_link, legion_wallet
from app.core.mcp_gateway import McpToolBinding, _notify_hidden_mcp_items, execute_mcp_tool

KEY = "lgn_" + "b" * 40


def _binding(market_item_id: uuid.UUID | None = None) -> McpToolBinding:
    return McpToolBinding(
        installed_item_id=uuid.uuid4(),
        market_item_id=market_item_id if market_item_id is not None else uuid.uuid4(),
        market_slug="mcp-searxng",
        tool_name="search",
        custom_config=None,
        isolation="stateless",
        managed_stdio=False,
    )


def _status_error(status_code: int, detail: Any) -> httpx.HTTPStatusError:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"detail": detail}
    return httpx.HTTPStatusError("refused", request=MagicMock(), response=response)


@pytest.fixture()
def central(monkeypatch):
    monkeypatch.setattr(legion_wallet, "MCP_CREDITS_WALLET", "legion")
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "local")
    monkeypatch.setattr(legion_wallet, "LEGION_MCP_GATEWAY_URL", "https://mcp.legion.example")
    legion_wallet._balances.clear()
    yield
    legion_wallet._balances.clear()


@pytest.fixture()
def local_wallet_untouched():
    """Any touch of the Hub's own wallet fails the test."""
    with (
        patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock) as price,
        patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock) as reserve,
        patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as settle,
        patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock) as refund,
    ):
        yield
        for mock in (price, reserve, settle, refund):
            mock.assert_not_awaited()


def test_the_mode_needs_the_setting_and_local_users(monkeypatch):
    monkeypatch.setattr(legion_wallet, "MCP_CREDITS_WALLET", "hub")
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "local")
    assert not legion_wallet.central()
    monkeypatch.setattr(legion_wallet, "MCP_CREDITS_WALLET", "legion")
    assert legion_wallet.central()
    # A Hub that signs in with Legion is our Hub: its own wallet is trusted.
    monkeypatch.setattr(legion_auth, "AUTH_MODE", "legion")
    assert not legion_wallet.central()


@pytest.mark.asyncio
async def test_a_market_call_goes_to_the_legion_gateway_with_the_key(central, local_wallet_untouched):
    binding = _binding()
    with (
        patch.object(legion_wallet, "key_for", new_callable=AsyncMock, return_value=KEY),
        patch("app.core.mcp_gateway.httpx.AsyncClient") as client_cls,
    ):
        client = client_cls.return_value.__aenter__.return_value
        response = MagicMock()
        response.json.return_value = {"content": [{"type": "text", "text": "found"}]}
        response.raise_for_status.return_value = None
        client.post = AsyncMock(return_value=response)
        result = await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {"query": "legion"})

    assert "found" in result
    url = client.post.await_args.args[0]
    kwargs = client.post.await_args.kwargs
    assert url == "https://mcp.legion.example/v1/execute"
    assert kwargs["headers"]["Authorization"] == f"Bearer {KEY}"
    # The Gateway reserves in the central wallet under the Hub's call id.
    assert uuid.UUID(kwargs["json"]["call_id"])
    assert kwargs["json"]["market_item_id"] == str(binding.market_item_id)


@pytest.mark.asyncio
async def test_a_personal_mcp_keeps_the_hubs_own_gateway(central):
    binding = McpToolBinding(installed_item_id=uuid.uuid4(), market_item_id=None, market_slug=None,
                             tool_name="t", custom_config=None, isolation="stateless", managed_stdio=False)
    with (
        patch.object(legion_wallet, "key_for", new_callable=AsyncMock) as key_for,
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock, return_value={"ok": True}) as post,
    ):
        await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})
    key_for.assert_not_awaited()
    assert post.await_args.kwargs.get("legion_key") is None


@pytest.mark.asyncio
async def test_without_a_linked_account_the_call_is_refused_before_the_gateway(central, local_wallet_untouched):
    with (
        patch.object(legion_wallet, "key_for", new_callable=AsyncMock, side_effect=legion_wallet.NoLegionAccount()),
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock) as post,
    ):
        result = await execute_mcp_tool(_binding(), uuid.uuid4(), uuid.uuid4(), {})
    post.assert_not_awaited()
    assert result.startswith("Error:") and "link the account" in result


@pytest.mark.parametrize("status, detail, words", [
    (402, {"code": "insufficient_credits", "required_mcr": 3000, "available_mcr": 1000}, "Own Hubs"),
    (401, {"code": "key_rejected"}, "no longer accepts"),
    (503, {"code": "wallet_unavailable"}, "cannot be checked"),
    (402, {"code": "install_required"}, "install"),
])
@pytest.mark.asyncio
async def test_gateway_refusals_are_told_in_plain_words(central, local_wallet_untouched, status, detail, words):
    with (
        patch.object(legion_wallet, "key_for", new_callable=AsyncMock, return_value=KEY),
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock,
              side_effect=_status_error(status, detail)),
    ):
        result = await execute_mcp_tool(_binding(), uuid.uuid4(), uuid.uuid4(), {})
    assert result.startswith("Error:") and words in result


@pytest.mark.asyncio
async def test_an_ordinary_gateway_failure_stays_ordinary(central, local_wallet_untouched):
    with (
        patch.object(legion_wallet, "key_for", new_callable=AsyncMock, return_value=KEY),
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock,
              side_effect=_status_error(502, {"code": "mcp_unreachable"})),
    ):
        result = await execute_mcp_tool(_binding(), uuid.uuid4(), uuid.uuid4(), {})
    assert result == "Error: MCP Gateway rejected the call"


@pytest.mark.asyncio
async def test_hub_mode_is_unchanged(monkeypatch):
    monkeypatch.setattr(legion_wallet, "MCP_CREDITS_WALLET", "hub")
    with (
        patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
        patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock) as reserve,
        patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock),
        patch.object(legion_wallet, "key_for", new_callable=AsyncMock) as key_for,
        patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock, return_value={"ok": True}) as post,
    ):
        result = await execute_mcp_tool(_binding(), uuid.uuid4(), uuid.uuid4(), {})
    assert result == json.dumps({"ok": True})
    reserve.assert_awaited_once()
    key_for.assert_not_awaited()
    assert post.await_args.kwargs.get("legion_key") is None


@pytest.mark.asyncio
async def test_the_key_comes_only_from_a_live_link(monkeypatch):
    row = MagicMock(status="linked")
    session = MagicMock()
    session.get = AsyncMock(return_value=row)
    factory = MagicMock(return_value=MagicMock(__aenter__=AsyncMock(return_value=session),
                                                __aexit__=AsyncMock(return_value=False)))
    monkeypatch.setattr("app.core.mcp_credits._short_session_factory", lambda: factory)
    monkeypatch.setattr(legion_link, "stored_key", lambda r: MagicMock(reveal=lambda: KEY))
    assert await legion_wallet.key_for(uuid.uuid4()) == KEY

    row.status = "rejected"
    with pytest.raises(legion_wallet.NoLegionAccount) as rejected:
        await legion_wallet.key_for(uuid.uuid4())
    assert rejected.value.rejected

    session.get = AsyncMock(return_value=None)
    with pytest.raises(legion_wallet.NoLegionAccount) as missing:
        await legion_wallet.key_for(uuid.uuid4())
    assert not missing.value.rejected


@pytest.mark.asyncio
async def test_the_balance_is_read_from_legion_and_cached(central):
    user_id = uuid.uuid4()
    fetch = AsyncMock(return_value={"wallet": {"balance_mcr": 4_200}})
    with (
        patch.object(legion_wallet, "optional_key", new_callable=AsyncMock, return_value=KEY),
        patch.object(legion_wallet, "fetch_wallet", fetch),
    ):
        assert await legion_wallet.balance_mcr(user_id) == 4_200
        assert await legion_wallet.balance_mcr(user_id) == 4_200
    assert fetch.await_count == 1


@pytest.mark.asyncio
async def test_an_unknown_balance_hides_nothing(central):
    with patch.object(legion_wallet, "optional_key", new_callable=AsyncMock, return_value=None):
        assert await legion_wallet.balance_mcr(uuid.uuid4()) is None
    with (
        patch.object(legion_wallet, "optional_key", new_callable=AsyncMock, return_value=KEY),
        patch.object(legion_wallet, "fetch_wallet", new_callable=AsyncMock,
                     side_effect=legion_link.LegionLinkError("down", status=503, code="unreachable")),
    ):
        assert await legion_wallet.balance_mcr(uuid.uuid4()) is None


@pytest.mark.asyncio
async def test_fetch_wallet_maps_legion_answers(monkeypatch):
    def transport(status: int, body: dict | None = None):
        return httpx.MockTransport(lambda request: httpx.Response(status, json=body or {}))

    real_client = httpx.AsyncClient

    for status, body, code in [(401, None, "key_rejected"), (500, None, "bad_answer")]:
        monkeypatch.setattr(legion_wallet.httpx, "AsyncClient",
                            lambda *a, _t=transport(status, body), **k: real_client(transport=_t))
        with pytest.raises(legion_link.LegionLinkError) as exc:
            await legion_wallet.fetch_wallet(KEY)
        assert exc.value.code == code

    ok = {"account": {"email": "a@example.com"}, "wallet": {"balance_mcr": 1000, "balance": 1.0}}
    monkeypatch.setattr(legion_wallet.httpx, "AsyncClient",
                        lambda *a, _t=transport(200, ok), **k: real_client(transport=_t))
    assert (await legion_wallet.fetch_wallet(KEY))["wallet"]["balance"] == 1.0


@pytest.mark.asyncio
async def test_hidden_market_tools_without_an_account_say_why():
    room_id, user_id = uuid.uuid4(), uuid.uuid4()
    with patch("app.core.mcp_gateway.emit_event", new_callable=AsyncMock) as emit:
        await _notify_hidden_mcp_items(MagicMock(), room_id, user_id,
                                       [{"slug": "mcp-searxng", "reason": "no_account"}])
    text = emit.await_args.args[2]
    assert "Legion account" in text and "mcp-searxng" in text
