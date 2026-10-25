from __future__ import annotations

import asyncio
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from app import config
from app.quota import verify_call_credits


def _fake_client(status_code: int, body: dict | None = None, *, calls: list | None = None):
    class Response:
        def __init__(self):
            self.status_code = status_code
            self._body = body or {}

        def json(self):
            return self._body

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json, headers):
            if calls is not None:
                calls.append({"url": url, "json": json, "headers": headers})
            return Response()

    return Client


def _fake_client_raising():
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json, headers):
            raise httpx.ConnectError("connection refused")

    return Client


def test_skips_personal_mcp_without_market_item_id(monkeypatch):
    """Never billed, so the check must not even attempt a Hub round trip."""
    monkeypatch.setattr(config, "ADMIN_API_URL", "http://backend:8000")

    def boom(**kwargs):
        raise AssertionError("must not call Hub for a non-billable item")

    monkeypatch.setattr("app.quota.httpx.AsyncClient", boom)
    asyncio.run(
        verify_call_credits(
            user_id=str(uuid4()), call_id=None, market_item_id=None,
            market_slug=None, tool="search",
        )
    )


def test_skips_when_admin_api_url_unset(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_URL", "")

    def boom(**kwargs):
        raise AssertionError("must not call Hub when ADMIN_API_URL is unset")

    monkeypatch.setattr("app.quota.httpx.AsyncClient", boom)
    asyncio.run(
        verify_call_credits(
            user_id=str(uuid4()), call_id=uuid4(), market_item_id=str(uuid4()),
            market_slug="mcp-toolbox", tool="query",
        )
    )


def test_skips_when_verify_disabled_for_local_dev(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_URL", "http://backend:8000")
    monkeypatch.setattr(config, "MCP_CREDITS_VERIFY", False)

    def boom(**kwargs):
        raise AssertionError("MCP_CREDITS_VERIFY=0 must skip the Hub round trip")

    monkeypatch.setattr("app.quota.httpx.AsyncClient", boom)
    asyncio.run(
        verify_call_credits(
            user_id=str(uuid4()), call_id=uuid4(), market_item_id=str(uuid4()),
            market_slug="mcp-toolbox", tool="query",
        )
    )


def test_allows_call_with_active_reservation(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_URL", "http://backend:8000")
    monkeypatch.setattr(config, "MCP_CREDITS_VERIFY", True)
    calls: list = []
    monkeypatch.setattr("app.quota.httpx.AsyncClient", lambda **kw: _fake_client(200, calls=calls)())
    asyncio.run(
        verify_call_credits(
            user_id="u1", call_id=uuid4(), market_item_id=str(uuid4()),
            market_slug="mcp-toolbox", tool="query",
        )
    )
    assert calls[0]["url"].endswith("/internal/mcp-credits/verify")
    assert calls[0]["headers"]["X-Admin-Token"] == config.ADMIN_API_TOKEN


def test_denies_call_without_active_reservation(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_URL", "http://backend:8000")
    monkeypatch.setattr(config, "MCP_CREDITS_VERIFY", True)
    monkeypatch.setattr(
        "app.quota.httpx.AsyncClient",
        lambda **kw: _fake_client(402, {"detail": "Not enough MCP credits."})(),
    )
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            verify_call_credits(
                user_id="u1", call_id=None, market_item_id=str(uuid4()),
                market_slug="mcp-toolbox", tool="query",
            )
        )
    assert exc_info.value.status_code == 402
    assert exc_info.value.detail == "Not enough MCP credits."


def test_fails_closed_when_hub_is_unreachable(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_URL", "http://backend:8000")
    monkeypatch.setattr(config, "MCP_CREDITS_VERIFY", True)
    monkeypatch.setattr("app.quota.httpx.AsyncClient", lambda **kw: _fake_client_raising()())
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            verify_call_credits(
                user_id="u1", call_id=uuid4(), market_item_id=str(uuid4()),
                market_slug="mcp-toolbox", tool="query",
            )
        )
    assert exc_info.value.status_code == 402


def test_fails_closed_on_unexpected_hub_error_status(monkeypatch):
    monkeypatch.setattr(config, "ADMIN_API_URL", "http://backend:8000")
    monkeypatch.setattr(config, "MCP_CREDITS_VERIFY", True)
    monkeypatch.setattr("app.quota.httpx.AsyncClient", lambda **kw: _fake_client(500)())
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            verify_call_credits(
                user_id="u1", call_id=uuid4(), market_item_id=str(uuid4()),
                market_slug="mcp-toolbox", tool="query",
            )
        )
    assert exc_info.value.status_code == 402
