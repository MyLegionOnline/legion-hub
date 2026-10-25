from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import user_installed_items as registry
from app.models.market import UserInstalledItem
from app.models.mcp_credits import McpPriceSnapshot


def test_custom_mcp_config_requires_valid_url_and_allowlisted_auth():
    assert registry._validate_custom_mcp_config({"url": "https://example.test/mcp", "auth": {"mode": "bearer"}})["url"] == "https://example.test/mcp"
    with pytest.raises(HTTPException, match="headers/env"):
        registry._validate_custom_mcp_config({"url": "https://example.test/mcp", "headers": {"Authorization": "Bearer x"}})
    with pytest.raises(HTTPException, match="headers/env"):
        registry._validate_custom_mcp_config({"url": "https://example.test/mcp", "env": {"TOKEN": "secret"}})
    with pytest.raises(HTTPException, match="absolute HTTP"):
        registry._validate_custom_mcp_config({"url": "/mcp"})
    with pytest.raises(HTTPException, match="Only bearer"):
        registry._validate_custom_mcp_config({"url": "https://example.test", "auth": {"mode": "basic"}})


def test_as_out_exposes_call_price_only_for_market_origin():
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    market = UserInstalledItem(user_id=uuid4(), type="mcp", name="Market", market_item_id=uuid4())
    linked = UserInstalledItem(user_id=uuid4(), type="mcp", name="Remote", custom_config={"url": "https://example.test/mcp"})
    for item in (market, linked):
        item.created_at = item.updated_at = now
    snapshot = McpPriceSnapshot(
        market_item_id=market.market_item_id, market_slug="mcp-toolbox",
        call_price_mcr=600, tool_price_overrides={"expensive_tool": 1200},
    )

    priced = registry._as_out(market, snapshot)
    assert priced.call_price == 0.6
    assert priced.tool_prices == {"expensive_tool": 1.2}

    # A price snapshot handed to a non-market item must never leak through —
    # only origin="legion" is billable (internal design notes).
    unbilled = registry._as_out(linked, snapshot)
    assert unbilled.call_price is None
    assert unbilled.tool_prices is None

    # No snapshot at all (e.g. Market unreachable) — price stays unknown, not zero.
    unpriced = registry._as_out(market)
    assert unpriced.call_price is None


def test_installed_item_origin_is_safe_and_unambiguous():
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    linked = UserInstalledItem(user_id=uuid4(), type="mcp", name="Remote", custom_config={"url": "https://example.test/mcp"})
    hosted = UserInstalledItem(user_id=uuid4(), type="mcp", name="Personal", custom_config={"mode": "hosted", "image": "example/mcp", "runtime": {"status": "running"}})
    market = UserInstalledItem(user_id=uuid4(), type="mcp", name="Market", market_item_id=uuid4())
    for item in (linked, hosted, market):
        item.created_at = item.updated_at = now
    assert registry._as_out(linked).origin == "link"
    assert registry._as_out(hosted).origin == "personal_image"
    assert registry._as_out(hosted).personal_status == "running"
    assert registry._as_out(market).origin == "legion"


@pytest.mark.asyncio
async def test_list_installed_items_batches_price_snapshots_by_market_item():
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    market_item_id = uuid4()
    market_item = UserInstalledItem(user_id=uuid4(), type="mcp", name="Market", market_item_id=market_item_id)
    market_item.created_at = market_item.updated_at = now
    personal_item = UserInstalledItem(user_id=uuid4(), type="mcp", name="Personal", custom_config={"mode": "hosted", "image": "x", "port": 8000})
    personal_item.created_at = personal_item.updated_at = now
    snapshot = McpPriceSnapshot(market_item_id=market_item_id, market_slug="mcp-toolbox", call_price_mcr=600)

    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = [market_item, personal_item]
    snap_result = MagicMock()
    snap_result.scalars.return_value.all.return_value = [snapshot]

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[items_result, snap_result])
    user = SimpleNamespace(id=uuid4())

    out = await registry.list_user_installed_items(db=db, current_user=user)

    assert db.execute.await_count == 2  # one query for items, one batched query for snapshots
    priced = next(o for o in out if o.name == "Market")
    unpriced = next(o for o in out if o.name == "Personal")
    assert priced.call_price == 0.6
    assert unpriced.call_price is None


@pytest.mark.asyncio
async def test_browse_market_mcps_exposes_credit_prices(monkeypatch):
    from app.core import market_client

    item_id = uuid4()
    catalog = {
        "items": [{
            "id": str(item_id), "slug": "mcp-toolbox", "name": "MCP Toolbox",
            "call_price_mcr": 600, "tool_price_overrides": {"expensive_tool": 1200},
        }],
        "total": 1, "page": 1, "page_size": 12,
    }
    monkeypatch.setattr(market_client, "list_catalog", AsyncMock(return_value=catalog))

    installed_result = MagicMock()
    installed_result.all.return_value = []
    db = AsyncMock()
    db.execute = AsyncMock(return_value=installed_result)
    user = SimpleNamespace(id=uuid4())

    out = await registry.browse_market_mcps(
        q=None, category=[], publisher=None, price=None, access=None, sort=None,
        page=1, page_size=12, db=db, current_user=user,
    )

    assert out.items[0].call_price == 0.6
    assert out.items[0].tool_prices == {"expensive_tool": 1.2}


@pytest.mark.asyncio
async def test_browse_market_mcps_forwards_taxonomy_filters(monkeypatch):
    from app.core import market_client

    list_catalog = AsyncMock(return_value={"items": [], "total": 0, "page": 1, "page_size": 12})
    monkeypatch.setattr(market_client, "list_catalog", list_catalog)
    db = AsyncMock()
    user = SimpleNamespace(id=uuid4())

    await registry.browse_market_mcps(
        q="pg", category=["databases", "cloud"], publisher="vendor", price="free",
        access="ready", sort="name", page=2, page_size=12, db=db, current_user=user,
    )

    kwargs = list_catalog.await_args.kwargs
    assert kwargs["type"] == "mcp"
    assert kwargs["category"] == ["databases", "cloud"]
    assert (kwargs["publisher"], kwargs["price"], kwargs["access"], kwargs["sort"]) == ("vendor", "free", "ready", "name")


@pytest.mark.asyncio
async def test_list_market_mcp_categories_proxies_faceted_counts(monkeypatch):
    from app.core import market_client

    rows = [{"slug": "databases", "emoji": "🗄️", "label_ru": "Databases (ru)", "label_en": "Databases", "count": 2, "empty": False}]
    list_categories = AsyncMock(return_value=rows)
    monkeypatch.setattr(market_client, "list_categories", list_categories)

    out = await registry.list_market_mcp_categories(
        q=None, publisher=None, price="free", access=None, current_user=SimpleNamespace(id=uuid4()),
    )

    assert out[0].slug == "databases" and out[0].count == 2
    assert list_categories.await_args.kwargs["type"] == "mcp"
    assert list_categories.await_args.kwargs["price"] == "free"


@pytest.mark.asyncio
async def test_create_custom_mcp_does_not_expose_credentials(monkeypatch):
    # The link-MCP cap (T7) has its own tests in test_tariff_enforcement.py.
    monkeypatch.setattr(registry, "_check_link_mcp_limit", AsyncMock())
    db = AsyncMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    user = SimpleNamespace(id=uuid4())
    body = registry.CustomInstalledItemIn(
        type="mcp", name="Private MCP", custom_config={"url": "https://mcp.example.test", "auth": {"mode": "bearer"}},
    )
    response = await registry.create_user_installed_item(body, db, user)
    created = db.add.call_args.args[0]
    # Test doubles do not populate SQLModel defaults until persistence.
    created.id = uuid4()
    created.created_at = created.updated_at = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    response = registry._as_out(created)
    assert response.has_custom_config is True
    assert "secret" not in response.model_dump_json()
    assert "headers" not in created.custom_config
    assert created.custom_config["auth"]["mode"] == "bearer"


@pytest.mark.asyncio
async def test_install_market_mcp_resolves_the_canonical_slug_endpoint(monkeypatch):
    from app.core import market_client

    market_item_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    market_item = {
        "id": str(market_item_id),
        "slug": "mcp-toolbox",
        "type": "mcp",
        "name": "MCP Toolbox for Databases",
        "isolation": "stateless",
        "is_active": True,
        "call_price_mcr": 600,
        "tool_price_overrides": {"expensive_tool": 1200},
    }
    get_item = AsyncMock(return_value=market_item)
    monkeypatch.setattr(market_client, "get_catalog_item", get_item)

    db = AsyncMock()
    existing_result = MagicMock()
    existing_result.scalar_one_or_none.return_value = None
    db.execute = AsyncMock(return_value=existing_result)
    db.add = MagicMock()
    # No cached price snapshot yet — store_price_snapshot must create one,
    # not mutate whatever AsyncMock.get() would fabricate by default.
    db.get = AsyncMock(return_value=None)

    result = await registry.install_market_mcp(
        registry.MarketMcpInstallIn(slug="mcp-toolbox"), db=db, current_user=user
    )

    get_item.assert_awaited_once_with("mcp-toolbox")
    added = [call.args[0] for call in db.add.call_args_list]
    created = next(obj for obj in added if isinstance(obj, UserInstalledItem))
    assert created.market_item_id == market_item_id
    assert created.market_slug == "mcp-toolbox"
    assert result.name == "MCP Toolbox for Databases"

    # Stage 4 (internal design notes): the call price is cached from the
    # catalog item already in hand, in the same transaction as the install —
    # execute_mcp_tool refuses to charge for a call it can't price.
    snapshot = next(obj for obj in added if isinstance(obj, McpPriceSnapshot))
    assert snapshot.market_item_id == market_item_id
    assert snapshot.market_slug == "mcp-toolbox"
    assert snapshot.call_price_mcr == 600
    assert snapshot.tool_price_overrides == {"expensive_tool": 1200}
