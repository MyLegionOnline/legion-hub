"""Default skill auto-provisioning (internal design notes stage 3,
Market-free since internal design notes stage 3)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import default_catalog, market_client, skill_defaults
from app.models.market import UserDefaultProvision


def _ledger_rows(rows):
    """(market_slug, market_item_id) rows of user_default_provision."""
    result = MagicMock()
    result.all.return_value = list(rows)
    return result


@pytest.fixture(autouse=True)
def _fresh_catalog():
    default_catalog.reset()
    yield
    default_catalog.reset()


@pytest.fixture(autouse=True)
def _provisioning_enabled(monkeypatch):
    monkeypatch.setattr(skill_defaults, "SKILL_DEFAULTS_PROVISION", True)


@pytest.fixture(autouse=True)
def _market_must_not_be_called(monkeypatch):
    for name in ("list_catalog", "get_skill", "get_catalog_item", "fetch_package"):
        monkeypatch.setattr(market_client, name, AsyncMock(side_effect=AssertionError(f"Market {name} called")))


def _synced(skills=(), mcps=()):
    default_catalog._state["skills"] = list(skills)
    default_catalog._state["mcps"] = list(mcps)
    default_catalog._state["ids"] = {s["slug"]: str(s["id"]) for s in skills}


@pytest.mark.asyncio
async def test_before_any_sync_installs_the_shipped_defaults(monkeypatch):
    user_id = uuid4()
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_ledger_rows([]))
    db.add = MagicMock()
    install = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(skill_defaults, "install_shared_skill_for_user", install)

    await skill_defaults.reconcile_default_skills(user_id, db)

    slugs = [call.args[1] for call in install.await_args_list]
    assert slugs == ["skill-arch", "skill-coder", "skill-writer"]
    assert all(call.kwargs["market_item_id"] is None for call in install.await_args_list)
    ledger = [c.args[0] for c in db.add.call_args_list]
    assert all(isinstance(r, UserDefaultProvision) for r in ledger)
    assert [r.market_slug for r in ledger] == slugs
    # Placeholder ledger ids are stable per slug, so a repeat stays exact.
    assert ledger[0].market_item_id == skill_defaults._ledger_item_id("skill-arch")


@pytest.mark.asyncio
async def test_after_sync_keeps_the_shipped_defaults_and_adds_market_ones(monkeypatch):
    xlsx = {"id": uuid4(), "slug": "xlsx"}
    _synced(skills=[xlsx])
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_ledger_rows([]))
    db.add = MagicMock()
    install = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(skill_defaults, "install_shared_skill_for_user", install)

    await skill_defaults.reconcile_default_skills(uuid4(), db)

    assert [(c.args[1], c.kwargs["market_item_id"]) for c in install.await_args_list] == [
        ("skill-arch", None), ("skill-coder", None), ("skill-writer", None), ("xlsx", xlsx["id"]),
    ]
    assert [r.market_item_id for r in (c.args[0] for c in db.add.call_args_list)][-1] == xlsx["id"]


@pytest.mark.asyncio
async def test_skips_defaults_already_in_the_ledger_by_slug_or_id(monkeypatch):
    xlsx = {"id": uuid4(), "slug": "xlsx"}
    _synced(skills=[xlsx])
    db = AsyncMock()
    # skill-arch provisioned from the bundle (placeholder id), the other two
    # shipped ones by slug, xlsx by an old ledger row that only matches by
    # Market id.
    db.execute = AsyncMock(return_value=_ledger_rows([
        ("skill-arch", skill_defaults.uuid.uuid5(skill_defaults._BUNDLED_LEDGER_NS, "skill-arch")),
        ("skill-coder", uuid4()),
        ("skill-writer", uuid4()),
        (None, xlsx["id"]),
    ]))
    install = AsyncMock()
    monkeypatch.setattr(skill_defaults, "install_shared_skill_for_user", install)

    await skill_defaults.reconcile_default_skills(uuid4(), db)

    install.assert_not_awaited()
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_market_only_default_not_downloaded_yet_is_left_for_later(monkeypatch):
    _synced(skills=[{"id": uuid4(), "slug": "xlsx"}])
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_ledger_rows([]))
    db.add = MagicMock()
    monkeypatch.setattr(skill_defaults, "install_shared_skill_for_user", AsyncMock(return_value=None))

    await skill_defaults.reconcile_default_skills(uuid4(), db)

    db.add.assert_not_called()  # no ledger row → offered again after the sync


@pytest.mark.asyncio
async def test_continues_after_one_install_fails(monkeypatch):
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_ledger_rows([]))
    db.add = MagicMock()

    async def fake_install(uid, slug, database, market_item_id=None):
        if slug == "skill-arch":
            raise RuntimeError("boom")
        return MagicMock()

    monkeypatch.setattr(skill_defaults, "install_shared_skill_for_user", AsyncMock(side_effect=fake_install))

    await skill_defaults.reconcile_default_skills(uuid4(), db)

    ledger = [c.args[0].market_slug for c in db.add.call_args_list]
    assert ledger == ["skill-coder", "skill-writer"]


@pytest.mark.asyncio
async def test_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(skill_defaults, "SKILL_DEFAULTS_PROVISION", False)
    db = AsyncMock()
    await skill_defaults.reconcile_default_skills(uuid4(), db)
    db.execute.assert_not_awaited()


def test_default_skill_slugs_never_calls_market():
    assert skill_defaults.default_skill_slugs() == {
        "skill-arch", "skill-coder", "skill-writer",
    }
    _synced(skills=[{"id": uuid4(), "slug": "xlsx"}])
    assert skill_defaults.default_skill_slugs() == {
        "skill-arch", "skill-coder", "skill-writer", "xlsx",
    }


@pytest.mark.asyncio
async def test_reconcile_default_mcps_is_noop_before_any_sync(monkeypatch):
    install = AsyncMock()
    monkeypatch.setattr(skill_defaults, "install_market_mcp_for_user", install)
    db = AsyncMock()

    await skill_defaults.reconcile_default_mcps(uuid4(), db)

    install.assert_not_awaited()
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_default_mcps_installs_item_and_records_ledger(monkeypatch):
    user_id = uuid4()
    item = {
        "id": uuid4(),
        "slug": "mcp-vision",
        "name": "Image understanding",
        "type": "mcp",
        "isolation": "stateless",
        "call_price_mcr": 500,
    }
    _synced(mcps=[item])
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_ledger_rows([]))
    db.add = MagicMock()
    install = AsyncMock()
    monkeypatch.setattr(skill_defaults, "install_market_mcp_for_user", install)

    await skill_defaults.reconcile_default_mcps(user_id, db)

    install.assert_awaited_once_with(user_id, item, db)
    ledger_rows = [call.args[0] for call in db.add.call_args_list]
    assert len(ledger_rows) == 1
    assert isinstance(ledger_rows[0], UserDefaultProvision)
    assert ledger_rows[0].market_item_id == item["id"]
    assert ledger_rows[0].market_slug == "mcp-vision"


@pytest.mark.asyncio
async def test_reconcile_default_mcps_skips_provisioned_item(monkeypatch):
    item_id = uuid4()
    _synced(mcps=[{"id": item_id, "slug": "mcp-vision"}])
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_ledger_rows([("mcp-vision", item_id)]))
    install = AsyncMock()
    monkeypatch.setattr(skill_defaults, "install_market_mcp_for_user", install)

    await skill_defaults.reconcile_default_mcps(uuid4(), db)

    install.assert_not_awaited()
    db.add.assert_not_called()
