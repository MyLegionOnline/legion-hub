"""internal design notes Stage 4A: charging a one-time Market item
install purchase — a new ledger kind (``purchase``), independent of the
per-call reserve/settle/refund path exercised in ``test_mcp_credits.py``.

Same DB-backed convention as ``TestChargingDB`` there: a real PostgreSQL
instance is required (``@pytest.mark.db``), skipped automatically when
``DATABASE_URL`` isn't set (see ``feedback_pytest_windows_env`` in project
memory — this class was written and reviewed but not run on this machine,
same limitation noted throughout this plan's other DB-backed stages).

Run with:
    DATABASE_URL=postgresql+asyncpg://... pytest tests/test_mcp_credits_purchase.py -m db
"""
from __future__ import annotations

import os
import uuid
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from app.core.mcp_credits import InstallPurchaseDenied, credits_to_mcr

DB_URL = os.environ.get("DATABASE_URL", "")
pytestmark_db = pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import SQLModel

    import app.models  # noqa: F401  (populate metadata)

    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
@pytest.mark.db
class TestInstallPurchaseDB:
    @pytest_asyncio.fixture(autouse=True)
    async def _reset_process_engine_pool(self):
        """Same reset as TestChargingDB in test_mcp_credits.py — charge_install/
        refund_install go through app.db.session.engine's own process-global
        connection pool, not db_session's fixture-scoped one."""
        from app.db.session import engine

        await engine.dispose()
        yield

    async def _make_user(self, db) -> uuid.UUID:
        from app.models.user import User

        user_id = uuid.uuid4()
        db.add(User(id=user_id, email=f"{user_id}@test.com", hashed_password="x"))
        await db.commit()
        return user_id

    async def test_charge_install_debits_exactly_the_price_and_writes_purchase_kind(self, db_session):
        from app.core.mcp_credits import charge_install, get_wallet_balance_mcr, list_ledger

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)  # issues the first weekly grant
        price = credits_to_mcr(3)  # 3000 mcr, the plan's ui-from-design price

        result = await charge_install(user_id, price, market_slug="ui-from-design")

        assert result.charged is True
        assert result.already_purchased is False
        assert result.purchase_id is not None

        after = await get_wallet_balance_mcr(user_id)
        assert after == starting - price

        rows, total = await list_ledger(user_id, kind="purchase")
        assert total == 1
        assert rows[0]["market_slug"] == "ui-from-design"
        assert rows[0]["amount"] == pytest.approx(-3.0)

    async def test_reinstall_after_delete_does_not_charge_twice(self, db_session):
        """Stage 4A.3 — the whole point of this stage: idempotency keyed on the
        ledger, not on UserInstalledItem (which the caller has already deleted
        by the time a reinstall reaches this function)."""
        from app.core.mcp_credits import charge_install, get_wallet_balance_mcr

        user_id = await self._make_user(db_session)
        price = credits_to_mcr(3)

        first = await charge_install(user_id, price, market_slug="ui-from-design")
        after_first = await get_wallet_balance_mcr(user_id)

        second = await charge_install(user_id, price, market_slug="ui-from-design")
        after_second = await get_wallet_balance_mcr(user_id)

        assert first.charged is True
        assert second.charged is False
        assert second.already_purchased is True
        assert second.purchase_id == first.purchase_id
        assert after_second == after_first  # not charged again

    async def test_refund_then_retry_charges_again(self, db_session):
        """Stage 4A.4 — a refunded purchase must be payable again, otherwise a
        failed install (charged, but package parse blew up) permanently locks
        the user out of ever owning the item."""
        from app.core.mcp_credits import charge_install, get_wallet_balance_mcr, refund_install

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)
        price = credits_to_mcr(3)

        first = await charge_install(user_id, price, market_slug="ui-from-design")
        assert first.charged is True
        await refund_install(first.purchase_id, note="install failed after charge, auto-refund")
        after_refund = await get_wallet_balance_mcr(user_id)
        assert after_refund == starting

        second = await charge_install(user_id, price, market_slug="ui-from-design")
        assert second.charged is True
        assert second.purchase_id != first.purchase_id
        after_second = await get_wallet_balance_mcr(user_id)
        assert after_second == starting - price

    async def test_refund_install_is_idempotent(self, db_session):
        from app.core.mcp_credits import charge_install, get_wallet_balance_mcr, refund_install

        user_id = await self._make_user(db_session)
        price = credits_to_mcr(3)
        charge = await charge_install(user_id, price, market_slug="ui-from-design")

        await refund_install(charge.purchase_id)
        once = await get_wallet_balance_mcr(user_id)
        await refund_install(charge.purchase_id)  # defensive double-call
        twice = await get_wallet_balance_mcr(user_id)

        assert once == twice

    async def test_insufficient_balance_denies_and_touches_nothing(self, db_session):
        from app.core.mcp_credits import charge_install, get_wallet_balance_mcr

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)  # hobby default: 100 credits
        price = credits_to_mcr(1000)  # far more than the weekly grant

        with pytest.raises(InstallPurchaseDenied) as exc:
            await charge_install(user_id, price, market_slug="ui-from-design")
        assert exc.value.required_mcr == price

        unchanged = await get_wallet_balance_mcr(user_id)
        assert unchanged == starting

    async def test_zero_price_is_a_pure_noop(self, db_session):
        from app.core.mcp_credits import charge_install, get_wallet_balance_mcr, list_ledger

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)

        result = await charge_install(user_id, 0, market_slug="skill-arch")

        assert result.charged is False
        assert result.already_purchased is False
        assert result.purchase_id is None
        assert await get_wallet_balance_mcr(user_id) == starting
        _rows, total = await list_ledger(user_id, kind="purchase")
        assert total == 0

    async def test_different_items_are_charged_independently(self, db_session):
        """The idempotency key is (user_id, market_slug) — owning one paid
        item must never make a second paid item look already-purchased."""
        from app.core.mcp_credits import charge_install, get_wallet_balance_mcr

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)
        price = credits_to_mcr(3)

        await charge_install(user_id, price, market_slug="ui-from-design")
        second = await charge_install(user_id, price, market_slug="some-other-paid-skill")

        assert second.charged is True
        after = await get_wallet_balance_mcr(user_id)
        assert after == starting - 2 * price
