"""The Server Terms of an optional MCP server before a call (Terms §6,
legion_market/legal/PENDING.md item 56).

Our Hub names the caller's Legion account in ``X-Legion-Account-Id``; for an
item Market marks with ``server_terms`` Gateway asks Market whether that
account accepted them, and refuses the call otherwise — before the upstream
MCP is reached. A self-hosted Hub's key gets the same refusal from the
central wallet's reserve.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, server_terms
from app.main import app, market
from tests.test_account_keys import ITEM_ID, _execute, fake_market, upstream  # noqa: F401  (fixtures)

REFUSAL = {"code": "server_terms_required", "message": "Read and accept the Vision MCP Server Terms at "
           "https://market.example/server-terms/vision before this server can be used",
           "document": "vision", "version": "2026-10-25", "url": "https://market.example/server-terms/vision"}


class Market:
    """Market's /internal/server-terms/check: who accepted what."""

    def __init__(self) -> None:
        self.accepted: set[str] = set()
        self.asked: list[dict[str, Any]] = []
        self.down = False

    # Patched onto httpx.AsyncClient as a bound method of this fake: the client is not passed.
    async def post(self, url, json=None, headers=None):  # noqa: A002 - httpx's own name
        assert url.endswith("/internal/server-terms/check")
        assert headers == {"X-Market-Gateway-Token": config.MARKET_API_TOKEN}
        self.asked.append(json)
        if self.down:
            raise httpx.ConnectError("market down")
        if json["account_id"] in self.accepted:
            return httpx.Response(200, json={"allowed": True})
        return httpx.Response(403, json={"detail": REFUSAL})


@pytest.fixture()
def checks(monkeypatch, upstream):  # noqa: F811
    fake = Market()
    monkeypatch.setattr(httpx.AsyncClient, "post", fake.post)
    server_terms.reset_cache()
    terms: dict[str, Any] = {"server_terms": "vision"}

    async def resolve(installed_item_id, *, market_item_id=None, slug=None):
        return {"id": ITEM_ID, "type": "mcp", "slug": "mcp-vision", "isolation": "stateless",
                "internal_endpoint": "http://mcp.internal/mcp", "is_active": True, **terms}

    async def no_credit_check(**kwargs):
        return None

    monkeypatch.setattr(market, "resolve", resolve)
    monkeypatch.setattr("app.main.verify_call_credits", no_credit_check)
    yield fake, terms
    server_terms.reset_cache()


def _service_call(client, account_id: str | None = "acct-1"):
    headers = {"Authorization": f"Bearer {config.GATEWAY_TOKEN}", "X-Legion-User-Id": "hub-user"}
    if account_id:
        headers[server_terms.ACCOUNT_HEADER] = account_id
    return client.post("/v1/execute", headers=headers, json={
        "installed_item_id": str(uuid4()), "market_item_id": ITEM_ID, "market_slug": "mcp-vision",
        "tool": "analyze", "call_id": str(uuid4())})


def test_our_hub_is_refused_until_the_account_accepts(checks, upstream):  # noqa: F811
    fake, _ = checks
    seen, _ = upstream
    with TestClient(app) as client:
        refused = _service_call(client)
        assert refused.status_code == 403 and refused.json()["detail"] == REFUSAL
        assert seen == []
        # Not remembered: the acceptance in Market takes effect at the next call.
        fake.accepted.add("acct-1")
        assert _service_call(client).status_code == 200
        # An acceptance is remembered briefly, so a run of calls does not ask each time.
        assert _service_call(client).status_code == 200
    assert len(fake.asked) == 2 and len(seen) == 2
    assert fake.asked[0] == {"account_id": "acct-1", "market_item_id": ITEM_ID, "market_slug": "mcp-vision"}


def test_a_user_without_a_legion_account_is_asked_about_and_refused(checks, upstream):  # noqa: F811
    fake, _ = checks
    with TestClient(app) as client:
        assert _service_call(client, account_id=None).status_code == 403
    assert fake.asked[0]["account_id"] is None


def test_a_server_without_server_terms_is_not_asked_about(checks, upstream):  # noqa: F811
    fake, terms = checks
    terms["server_terms"] = None
    with TestClient(app) as client:
        assert _service_call(client).status_code == 200
    assert fake.asked == []


def test_market_down_fails_closed(checks, upstream):  # noqa: F811
    fake, _ = checks
    seen, _ = upstream
    fake.down = True
    with TestClient(app) as client:
        response = _service_call(client)
    assert response.status_code == 503 and response.json()["detail"]["code"] == "server_terms_unavailable"
    assert seen == []


def test_a_key_paid_call_carries_the_wallets_refusal(fake_market, upstream, monkeypatch):  # noqa: F811
    seen, _ = upstream
    original = fake_market.post

    async def post(path, payload):
        if path == "reserve":
            fake_market.calls.append((path, payload))
            return httpx.Response(403, json={"detail": REFUSAL})
        return await original(path, payload)

    monkeypatch.setattr("app.account_keys._market_post", post)
    with TestClient(app) as client:
        refused = _execute(client)
    assert refused.status_code == 403 and refused.json()["detail"] == REFUSAL
    assert seen == [] and "settle" not in fake_market.paths()
