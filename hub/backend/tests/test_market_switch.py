"""MARKET_ENABLED: with it off the Hub never talks to Legion (internal design notes stage 4.1)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from app.core import legion_link, legion_wallet, market_client
from app.core.market_switch import MarketSwitchError, market_enabled, market_url


@pytest.fixture
def off(monkeypatch):
    monkeypatch.setenv("MARKET_ENABLED", "false")
    # Any socket the Hub opens in these tests is a bug.
    def boom(*_a, **_k):
        raise AssertionError("the Hub contacted the network while the Market integration is off")
    monkeypatch.setattr(httpx.AsyncClient, "__init__", boom)
    monkeypatch.setattr(httpx.Client, "__init__", boom)


def test_off_by_default_and_parsed_strictly():
    assert market_enabled({}) is False
    assert market_enabled({"MARKET_ENABLED": ""}) is False
    for word in ("1", "true", "YES", " on "):
        assert market_enabled({"MARKET_ENABLED": word}) is True
    for word in ("0", "false", "No", "off"):
        assert market_enabled({"MARKET_ENABLED": word}) is False
    with pytest.raises(MarketSwitchError):
        market_enabled({"MARKET_ENABLED": "maybe"})


def test_public_market_is_the_default_address(monkeypatch):
    monkeypatch.delenv("MARKET_API_URL", raising=False)
    assert market_url() == "https://market.mylegion.online"
    monkeypatch.setenv("MARKET_API_URL", "http://market-backend:8300/")
    assert market_url() == "http://market-backend:8300"


@pytest.mark.asyncio
async def test_catalogue_is_empty_and_silent(off):
    page = await market_client.list_catalog(type="skill", page=2, page_size=5)
    assert page == {"items": [], "total": 0, "page": 2, "page_size": 5}
    assert await market_client.list_categories(type="mcp") == []


@pytest.mark.asyncio
async def test_items_are_not_found_and_silent(off):
    with pytest.raises(market_client.MarketNotFoundError):
        await market_client.get_skill("anything")
    with pytest.raises(market_client.MarketNotFoundError):
        await market_client.get_catalog_item("anything")
    with pytest.raises(market_client.MarketNotFoundError):
        await market_client.fetch_package("anything", "1.0.0")


def test_nothing_can_be_linked_or_paid_from_legion(off, monkeypatch):
    monkeypatch.setattr("app.core.legion_auth.AUTH_MODE", "local")
    monkeypatch.setattr(legion_wallet, "MCP_CREDITS_WALLET", "legion")
    assert legion_link.available() is False
    assert legion_wallet.central() is False


def test_on_with_local_users_a_hub_can_link_and_pay_from_legion(monkeypatch):
    monkeypatch.setenv("MARKET_ENABLED", "true")
    monkeypatch.setattr("app.core.legion_auth.AUTH_MODE", "local")
    monkeypatch.setattr(legion_wallet, "MCP_CREDITS_WALLET", "legion")
    assert legion_link.available() is True
    assert legion_wallet.central() is True


def test_defaults_point_at_the_public_services_for_a_hub_that_is_not_ours():
    """A fresh process with an empty environment: self-hosted defaults."""
    code = (
        "from app.core import legion_auth, legion_wallet;"
        "print(legion_auth.AUTH_MODE, legion_auth.LEGION_ISSUER, legion_auth.LEGION_INTERNAL_URL,"
        " legion_wallet.MCP_CREDITS_WALLET, legion_wallet.LEGION_MCP_GATEWAY_URL)"
    )
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA")
    env = {k: v for k, v in os.environ.items() if k in keep}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent.parent, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "local", "https://market.mylegion.online", "https://market.mylegion.online",
        "legion", "https://gateway.mylegion.online",
    ]


def test_signing_in_through_legion_needs_the_integration_on():
    code = "import app.core.legion_auth"
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA")
    base = {k: v for k, v in os.environ.items() if k in keep}
    cwd = Path(__file__).resolve().parent.parent
    off = subprocess.run([sys.executable, "-c", code], env={**base, "AUTH_MODE": "legion"},
                         capture_output=True, text=True, cwd=cwd, timeout=120)
    assert off.returncode != 0 and "needs MARKET_ENABLED=true" in off.stderr
    on = subprocess.run([sys.executable, "-c", code], env={**base, "AUTH_MODE": "legion", "MARKET_ENABLED": "true"},
                        capture_output=True, text=True, cwd=cwd, timeout=120)
    assert on.returncode == 0, on.stderr
