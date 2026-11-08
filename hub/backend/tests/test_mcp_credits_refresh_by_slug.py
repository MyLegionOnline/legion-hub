"""internal design notes Stage 4.5: the Admin Center's manual
"refresh price" action only has a Market slug at hand (not the
``market_item_id`` a prior discovery/install round trip would have cached),
so it needs a lookup-by-slug variant of ``refresh_price_snapshot``.

Fully mocked — no database, no network — mirroring the orchestration-layer
style of ``TestExecuteMcpToolOrchestration`` in test_mcp_credits.py rather
than the ``@pytest.mark.db`` class, which needs a real Postgres instance.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from app.core.mcp_credits import refresh_price_snapshot_by_slug


class _FakeSession:
    """Just enough of AsyncSession for store_price_snapshot's read/add/flush."""

    def __init__(self, existing=None):
        self.existing = existing
        self.added = []

    async def get(self, model, key):
        return self.existing

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass


@pytest.mark.asyncio
async def test_refresh_by_slug_looks_up_market_item_id_from_the_catalog_response(monkeypatch):
    market_item_id = uuid.uuid4()
    catalog_item = {
        "id": str(market_item_id),
        "call_price_mcr": 500,
        "tool_price_overrides": {"extract_design_spec": 900, "critique_design": 700},
    }
    get_catalog_item = AsyncMock(return_value=catalog_item)
    monkeypatch.setattr("app.core.market_client.get_catalog_item", get_catalog_item)

    db = _FakeSession()
    snapshot = await refresh_price_snapshot_by_slug(db, market_slug="mcp-vision")

    get_catalog_item.assert_awaited_once_with("mcp-vision")
    assert snapshot.market_item_id == market_item_id
    assert snapshot.market_slug == "mcp-vision"
    assert snapshot.call_price_mcr == 500
    assert snapshot.tool_price_overrides == {"extract_design_spec": 900, "critique_design": 700}
    assert db.added == [snapshot]


@pytest.mark.asyncio
async def test_refresh_by_slug_defaults_missing_price_fields_to_empty(monkeypatch):
    market_item_id = uuid.uuid4()
    monkeypatch.setattr(
        "app.core.market_client.get_catalog_item",
        AsyncMock(return_value={"id": str(market_item_id)}),
    )

    db = _FakeSession()
    snapshot = await refresh_price_snapshot_by_slug(db, market_slug="mcp-searxng")

    assert snapshot.call_price_mcr == 0
    assert snapshot.tool_price_overrides == {}


@pytest.mark.asyncio
async def test_refresh_by_slug_propagates_market_client_failures(monkeypatch):
    from app.core import market_client

    monkeypatch.setattr(
        "app.core.market_client.get_catalog_item",
        AsyncMock(side_effect=market_client.MarketNotFoundError("mcp-vision")),
    )

    db = _FakeSession()
    with pytest.raises(market_client.MarketNotFoundError):
        await refresh_price_snapshot_by_slug(db, market_slug="mcp-vision")


class _CommittingSession(_FakeSession):
    def __init__(self, existing=None):
        super().__init__(existing)
        self.committed = False

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
async def test_discovery_refresh_commits_in_its_own_session(monkeypatch):
    """The charge reads the price in a separate session, so a refresh left
    uncommitted in the caller's turn-long transaction would keep billing the
    old price — the refresh must commit on its own."""
    from app.core import mcp_credits
    from app.models.mcp_credits import McpPriceSnapshot

    market_item_id = uuid.uuid4()
    stale = McpPriceSnapshot(market_item_id=market_item_id, market_slug="legion-vps-mcp", call_price_mcr=300)
    session = _CommittingSession(existing=stale)
    monkeypatch.setattr(mcp_credits, "_short_session_factory", lambda: (lambda: session))
    monkeypatch.setattr(
        "app.core.market_client.get_catalog_item",
        AsyncMock(return_value={"id": str(market_item_id), "call_price_mcr": 200, "tool_price_overrides": {}}),
    )

    snapshot = await mcp_credits.refresh_price_snapshot(market_item_id=market_item_id, market_slug="legion-vps-mcp")

    assert snapshot.call_price_mcr == 200
    assert session.committed is True
