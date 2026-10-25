"""Hub side of the paid-package gate (internal design notes stage 6.1).

Market gives a paid skill package only to a Hub it trusts or to an account that
bought it. The core Hub presents no credentials of its own; the cloud plugin
adds its service token through ``PluginHost.add_market_credentials``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.core import market_client
from app.core.skill_market_install import MarketInstallError, ensure_shared_skill
from app.plugins import PluginHost, reset_registry


@pytest.fixture(autouse=True)
def _market(monkeypatch):
    monkeypatch.setenv("MARKET_ENABLED", "true")
    monkeypatch.setenv("MARKET_API_URL", "http://market.test")
    reset_registry()
    yield
    reset_registry()


def _serve(monkeypatch, status: int, body: bytes = b"zip", seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, content=body, headers={"X-Checksum-SHA256": "abc"})

    real = httpx.AsyncClient
    monkeypatch.setattr(market_client.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_a_core_hub_asks_without_credentials(monkeypatch):
    seen: list[httpx.Request] = []
    _serve(monkeypatch, 200, seen=seen)
    data, checksum = await market_client.fetch_package("some-skill", "1.0.0")
    assert (data, checksum) == (b"zip", "abc")
    assert "authorization" not in seen[0].headers and "x-billing-token" not in seen[0].headers


@pytest.mark.asyncio
async def test_a_plugin_can_present_the_hubs_service_token(monkeypatch):
    seen: list[httpx.Request] = []
    _serve(monkeypatch, 200, seen=seen)
    registry = reset_registry()
    PluginHost(registry, "cloud").add_market_credentials(lambda: {"X-Billing-Token": "secret"})
    await market_client.fetch_package("some-skill", "1.0.0")
    assert seen[0].headers["x-billing-token"] == "secret"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 402, 403])
async def test_a_refusal_is_an_access_error_not_an_outage(monkeypatch, status):
    _serve(monkeypatch, status)
    with pytest.raises(market_client.MarketAccessError):
        await market_client.fetch_package("paid-skill", "1.0.0")


@pytest.mark.asyncio
async def test_a_missing_package_is_still_not_found(monkeypatch):
    _serve(monkeypatch, 404)
    with pytest.raises(market_client.MarketNotFoundError):
        await market_client.fetch_package("nope", "1.0.0")


@pytest.mark.asyncio
async def test_install_says_the_skill_is_paid(monkeypatch):
    monkeypatch.setattr(market_client, "fetch_package",
                        AsyncMock(side_effect=market_client.MarketAccessError("refused")))
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    db.execute = AsyncMock(return_value=result)
    with pytest.raises(MarketInstallError) as refused:
        await ensure_shared_skill("paid-skill", db, version="1.0.0")
    assert refused.value.status_code == 403
    assert "paid skill" in refused.value.detail
