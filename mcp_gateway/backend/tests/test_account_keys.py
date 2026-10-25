"""A self-hosted Hub calls with a Legion account key (internal design notes §3.10, stage 3b).

Market is replaced by an in-memory wallet behind ``account_keys._market_post``;
the upstream MCP by a patched ``McpHttpClient``.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import account_keys, config
from app.main import app, market
from app.mcp_client import McpHttpClient

KEY = "lgn_" + "a" * 40
ITEM_ID = str(uuid4())


class FakeMarket:
    """The central wallet as Market keeps it: one account, prices per tool."""

    def __init__(self, balance_mcr: int = 2_000, price_mcr: int = 500) -> None:
        self.balance = balance_mcr
        self.price = price_mcr
        self.keys = {KEY: "acct-1"}
        self.calls: list[tuple[str, dict]] = []
        self.spends: dict[str, dict] = {}
        self.down = False
        self.terms_outstanding = False

    async def post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        self.calls.append((path, payload))
        if self.down:
            raise httpx.ConnectError("market down")
        if path in ("authorize", "reserve") and payload["api_key"] not in self.keys:
            return httpx.Response(401, json={"detail": {"code": "key_rejected"}})
        if path in ("authorize", "reserve") and self.terms_outstanding:
            return httpx.Response(403, json={"detail": {"code": "terms_update_required", "message": "accept"}})
        if path == "authorize":
            return httpx.Response(200, json={"account_id": self.keys[payload["api_key"]], "key_prefix": KEY[:12],
                                             "balance_mcr": self.balance})
        if path == "reserve":
            if self.price == 0:
                return httpx.Response(200, json={"reserved": False, "price_mcr": 0, "balance_mcr": self.balance})
            if self.balance < self.price:
                return httpx.Response(402, json={"detail": {
                    "code": "insufficient_credits", "message": "Not enough Legion credits",
                    "required_mcr": self.price, "available_mcr": self.balance}})
            self.balance -= self.price
            self.spends[payload["call_id"]] = {"status": "reserved", **payload}
            return httpx.Response(200, json={"reserved": True, "price_mcr": self.price, "balance_mcr": self.balance})
        if path == "settle":
            self.spends[payload["call_id"]]["status"] = "settled"
            return httpx.Response(200, json={"status": "settled"})
        if path == "refund":
            spend = self.spends[payload["call_id"]]
            if spend["status"] == "reserved":
                spend["status"] = "refunded"
                self.balance += self.price
            return httpx.Response(200, json={"status": spend["status"]})
        return httpx.Response(404)

    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]


@pytest.fixture()
def fake_market(monkeypatch):
    fake = FakeMarket()
    monkeypatch.setattr(account_keys, "_market_post", fake.post)
    monkeypatch.setattr(config, "ACCOUNT_KEYS_ENABLED", True)
    monkeypatch.setattr(account_keys, "limiter", account_keys.RateLimiter(6000, 100))
    account_keys.reset_cache()

    async def resolve(installed_item_id, *, market_item_id=None, slug=None):
        return {"id": ITEM_ID, "type": "mcp", "slug": "paid-search", "isolation": "tenant_aware",
                "internal_endpoint": "http://mcp.internal/mcp", "is_active": True}

    monkeypatch.setattr(market, "resolve", resolve)

    async def no_hub_verification(**kwargs):
        raise AssertionError("an account call must not ask our Hub about credits")

    monkeypatch.setattr("app.main.verify_call_credits", no_hub_verification)
    yield fake
    account_keys.reset_cache()


@pytest.fixture()
def upstream(monkeypatch):
    seen: list[dict] = []
    outcome: dict[str, Any] = {"result": {"content": [{"type": "text", "text": "found"}]}}

    async def call(self, tool, arguments, *, meta=None):
        seen.append({"tool": tool, "arguments": arguments, "meta": meta})
        if isinstance(outcome.get("raise"), HTTPException):
            raise outcome["raise"]
        return {"jsonrpc": "2.0", "id": 1, "result": outcome["result"]}

    async def tools(self):
        return [{"name": "search", "description": "", "inputSchema": {}}]

    monkeypatch.setattr(McpHttpClient, "call", call)
    monkeypatch.setattr(McpHttpClient, "tools", tools)
    return seen, outcome


def _key_headers(key: str = KEY, user_id: str | None = None) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "X-Legion-User-Id": user_id or str(uuid4()),
            "X-Legion-Room-Id": "room-1"}


def _execute(client, key=KEY, call_id=None, **extra):
    body = {"installed_item_id": str(uuid4()), "market_item_id": ITEM_ID, "market_slug": "paid-search",
            "tool": "search", "arguments": {"query": "legion"}, **extra}
    if call_id:
        body["call_id"] = str(call_id)
    return client.post("/v1/execute", headers=_key_headers(key), json=body)


def test_a_paid_call_is_reserved_in_market_and_settled(fake_market, upstream):
    seen, _ = upstream
    call_id = uuid4()
    with TestClient(app) as client:
        response = _execute(client, call_id=call_id)
    assert response.status_code == 200, response.text
    assert response.json() == {"content": [{"type": "text", "text": "found"}]}
    assert fake_market.paths() == ["authorize", "reserve", "settle"]
    reserve = fake_market.calls[1][1]
    # The item reserved is the catalogue's, the price is Market's, the call id the Hub's.
    assert reserve["market_item_id"] == ITEM_ID and reserve["tool"] == "search"
    assert reserve["call_id"] == str(call_id)
    assert fake_market.spends[str(call_id)]["status"] == "settled" and fake_market.balance == 1_500
    # A tenant-aware MCP sees the account, never the user id the foreign Hub claimed.
    assert seen[0]["meta"] == {"user_id": "legion-account:acct-1", "room_id": "room-1"}


def test_no_credits_refuses_before_the_upstream_is_called(fake_market, upstream):
    seen, _ = upstream
    fake_market.balance = 100
    with TestClient(app) as client:
        response = _execute(client)
    assert response.status_code == 402
    detail = response.json()["detail"]
    assert detail["code"] == "insufficient_credits" and detail["required_mcr"] == 500
    assert seen == []


@pytest.mark.parametrize("failure, expected", [
    (HTTPException(502, {"code": "mcp_unreachable", "reached_upstream": False, "message": "x"}), "refunded"),
    (HTTPException(502, {"code": "mcp_rpc_error", "reached_upstream": True, "message": "x"}), "settled"),
])
def test_upstream_failures_refund_or_settle_like_the_hub(fake_market, upstream, failure, expected):
    _, outcome = upstream
    outcome["raise"] = failure
    call_id = uuid4()
    with TestClient(app) as client:
        response = _execute(client, call_id=call_id)
    assert response.status_code == 502
    assert fake_market.spends[str(call_id)]["status"] == expected
    assert fake_market.balance == (2_000 if expected == "refunded" else 1_500)


def test_a_tool_error_is_refunded(fake_market, upstream):
    _, outcome = upstream
    outcome["result"] = {"isError": True, "content": [{"type": "text", "text": "bad query"}]}
    call_id = uuid4()
    with TestClient(app) as client:
        response = _execute(client, call_id=call_id)
    assert response.status_code == 200
    assert fake_market.spends[str(call_id)]["status"] == "refunded" and fake_market.balance == 2_000


def test_a_free_tool_reserves_nothing_and_settles_nothing(fake_market, upstream):
    fake_market.price = 0
    with TestClient(app) as client:
        response = _execute(client)
    assert response.status_code == 200
    assert fake_market.paths() == ["authorize", "reserve"]


def test_without_a_key_or_with_a_revoked_one_there_is_no_call(fake_market, upstream):
    seen, _ = upstream
    with TestClient(app) as client:
        no_key = client.post("/v1/execute", headers={"X-Legion-User-Id": "u"}, json={
            "installed_item_id": str(uuid4()), "tool": "search"})
        assert no_key.status_code == 401
        assert _execute(client).status_code == 200
        # Revoked in Market: the next paid call is refused even though the key was just accepted.
        fake_market.keys.clear()
        revoked = _execute(client)
        assert revoked.status_code == 401 and revoked.json()["detail"]["code"] == "key_rejected"
        # ...and the key is forgotten, so even the tool list is refused now.
        listed = client.get("/v1/tools", params={"installed_item_id": str(uuid4()), "market_item_id": ITEM_ID},
                            headers=_key_headers())
        assert listed.status_code == 401
    assert len(seen) == 1


def test_a_key_whose_owner_has_not_accepted_the_new_terms_is_refused(fake_market, upstream):
    seen, _ = upstream
    with TestClient(app) as client:
        assert _execute(client).status_code == 200
        # The grace period ended in Market: the next call is refused with the reason, not as a bad key.
        fake_market.terms_outstanding = True
        refused = _execute(client)
        assert refused.status_code == 403 and refused.json()["detail"]["code"] == "terms_update_required"
        # ...and it is not remembered as accepted, so the owner's acceptance takes effect at once.
        fake_market.terms_outstanding = False
        assert _execute(client).status_code == 200
    assert len(seen) == 2


def test_market_down_fails_closed(fake_market, upstream):
    seen, _ = upstream
    fake_market.down = True
    with TestClient(app) as client:
        response = _execute(client)
    assert response.status_code == 503 and response.json()["detail"]["code"] == "wallet_unavailable"
    assert seen == []


def test_a_key_reaches_catalogue_items_only(fake_market, upstream):
    with TestClient(app) as client:
        custom = _execute(client, custom_config={"url": "https://example.com/mcp", "auth": {"mode": "none"}})
        assert custom.status_code == 403 and custom.json()["detail"]["code"] == "custom_not_allowed"
        listed = client.get("/v1/tools", headers=_key_headers(), params={
            "installed_item_id": str(uuid4()), "custom_config": '{"url": "https://example.com/mcp"}'})
        assert listed.status_code == 403
        upstream_auth = client.post("/v1/execute", headers={
            **_key_headers(), "X-Legion-MCP-Upstream-Authorization": "Bearer stolen"}, json={
            "installed_item_id": str(uuid4()), "market_item_id": ITEM_ID, "tool": "search"})
        assert upstream_auth.status_code == 403
        # Operator and Hub-only routes stay service-token only.
        assert client.get("/v1/egress-posture", headers=_key_headers()).status_code == 401
        assert client.get("/v1/diagnostics", headers=_key_headers()).status_code == 401
        assert client.post("/v1/oauth/metadata", headers=_key_headers(), json={
            "kind": "authorization_server", "issuer": "https://example.com"}).status_code == 401


def test_the_tool_list_needs_an_accepted_key_and_is_cached_briefly(fake_market, upstream):
    with TestClient(app) as client:
        for _ in range(3):
            listed = client.get("/v1/tools", params={"installed_item_id": str(uuid4()), "market_item_id": ITEM_ID},
                                headers=_key_headers())
            assert listed.status_code == 200
            assert [t["name"] for t in listed.json()["tools"]] == ["search"]
    assert fake_market.paths() == ["authorize"]


def test_a_flood_from_one_key_is_rate_limited(fake_market, upstream, monkeypatch):
    monkeypatch.setattr(account_keys, "limiter", account_keys.RateLimiter(60, 2))
    with TestClient(app) as client:
        codes = [_execute(client).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[2:] == [429, 429]
    # The flood never reached Market past the bucket.
    assert fake_market.paths().count("reserve") == 2


def test_keys_can_be_switched_off(fake_market, upstream, monkeypatch):
    monkeypatch.setattr(config, "ACCOUNT_KEYS_ENABLED", False)
    with TestClient(app) as client:
        assert _execute(client).status_code == 401
    assert fake_market.calls == []


def test_the_service_token_path_is_unchanged(fake_market, upstream, monkeypatch):
    checked: list[dict] = []

    async def hub_verification(**kwargs):
        checked.append(kwargs)

    monkeypatch.setattr("app.main.verify_call_credits", hub_verification)
    with TestClient(app) as client:
        response = client.post("/v1/execute", headers={
            "Authorization": f"Bearer {config.GATEWAY_TOKEN}", "X-Legion-User-Id": "hub-user"}, json={
            "installed_item_id": str(uuid4()), "market_item_id": ITEM_ID, "tool": "search",
            "call_id": str(uuid4())})
    assert response.status_code == 200
    assert len(checked) == 1 and fake_market.calls == []


def test_rate_limiter_refills():
    limiter = account_keys.RateLimiter(60, 1)  # one a second
    assert limiter.allow("k", now=0.0) == 0
    assert limiter.allow("k", now=0.5) == pytest.approx(0.5)
    assert limiter.allow("k", now=1.01) == 0
    assert limiter.allow("other", now=1.01) == 0


def test_only_key_shaped_values_go_to_market():
    assert account_keys.looks_like_key(KEY)
    assert not account_keys.looks_like_key("dev-mcp-gateway-token")
    assert not account_keys.looks_like_key("lgn_short")
