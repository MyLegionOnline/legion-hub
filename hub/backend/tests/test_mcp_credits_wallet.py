"""Stage 3 of internal design notes: the wallet itself.

The rolling/grant/plan-change rules are pure functions (``roll_period``,
``apply_plan_change``) precisely so they can be pinned down here without a
database — the DB-backed half (``get_or_create_wallet``, ``roll_period_if_due``,
``wallet_summary``, ``topup``, ``adjust``) is exercised separately against a
real PostgreSQL instance, same convention as ``test_session_models.py``.

Run the DB-backed class with:
    DATABASE_URL=postgresql+asyncpg://... pytest tests/test_mcp_credits_wallet.py -m db
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio

from app.core.mcp_credits import (
    WEEK,
    LedgerDraft,
    WalletFields,
    apply_plan_change,
    credits_to_mcr,
    roll_period,
)

_NOW = datetime(2026, 8, 7, 12, 0, 0, tzinfo=timezone.utc)


def _fields(
    *,
    grant=0,
    purchased=0,
    period_grant=0,
    period_spent=0,
    lifetime_spent=0,
    started=None,
    ends=None,
) -> WalletFields:
    started = started if started is not None else _NOW - timedelta(days=3)
    ends = ends if ends is not None else _NOW - timedelta(days=1)
    return WalletFields(
        grant_balance_mcr=grant,
        purchased_balance_mcr=purchased,
        period_grant_mcr=period_grant,
        period_spent_mcr=period_spent,
        lifetime_spent_mcr=lifetime_spent,
        period_started_at=started,
        period_ends_at=ends,
    )


# ---------- roll_period: no-op when not due -------------------------------------


def test_roll_is_a_noop_before_the_window_ends():
    fields = _fields(grant=50_000, ends=_NOW + timedelta(days=1))
    new_fields, entries = roll_period(fields, weekly_grant_mcr=100_000, now=_NOW)

    assert new_fields == fields
    assert entries == []


def test_roll_is_a_noop_exactly_at_the_boundary_minus_one_second():
    ends = _NOW + timedelta(seconds=1)
    fields = _fields(grant=50_000, ends=ends)
    new_fields, entries = roll_period(fields, weekly_grant_mcr=100_000, now=_NOW)
    assert new_fields == fields
    assert entries == []


# ---------- roll_period: one week ------------------------------------------------


def test_roll_after_one_week_expires_leftover_and_grants_exactly_once():
    started = _NOW - timedelta(days=8)
    ends = _NOW - timedelta(days=1)  # due
    fields = _fields(grant=20_000, purchased=5_000, period_grant=100_000, period_spent=80_000,
                      lifetime_spent=500_000, started=started, ends=ends)

    new_fields, entries = roll_period(fields, weekly_grant_mcr=100_000, now=_NOW)

    kinds = [e.kind for e in entries]
    assert kinds == ["expire", "grant"]

    expire, grant = entries
    assert expire.amount_mcr == -20_000
    assert expire.grant_delta_mcr == -20_000
    assert expire.purchased_delta_mcr == 0
    assert expire.balance_after_mcr == 5_000  # grant zeroed, purchased untouched

    assert grant.amount_mcr == 100_000
    assert grant.grant_delta_mcr == 100_000
    assert grant.balance_after_mcr == 105_000

    assert new_fields.grant_balance_mcr == 100_000
    assert new_fields.period_grant_mcr == 100_000
    assert new_fields.period_spent_mcr == 0
    assert new_fields.period_started_at == ends
    assert new_fields.period_ends_at == ends + WEEK
    # purchased balance and lifetime spend are untouched by a roll
    assert new_fields.purchased_balance_mcr == 5_000
    assert new_fields.lifetime_spent_mcr == 500_000


def test_roll_with_nothing_left_to_expire_only_grants():
    fields = _fields(grant=0, purchased=1_000, period_grant=100_000, ends=_NOW - timedelta(hours=1))
    new_fields, entries = roll_period(fields, weekly_grant_mcr=100_000, now=_NOW)

    assert [e.kind for e in entries] == ["grant"]
    assert new_fields.grant_balance_mcr == 100_000


# ---------- roll_period: five missed weeks → exactly one grant ------------------


def test_five_missed_weeks_issue_exactly_one_grant_not_five():
    started = _NOW - timedelta(weeks=6)
    ends = _NOW - timedelta(weeks=5, days=1)  # five full windows overdue
    fields = _fields(grant=30_000, purchased=0, period_grant=100_000, started=started, ends=ends)

    new_fields, entries = roll_period(fields, weekly_grant_mcr=100_000, now=_NOW)

    grant_entries = [e for e in entries if e.kind == "grant"]
    expire_entries = [e for e in entries if e.kind == "expire"]
    assert len(grant_entries) == 1, "skipped weeks must not accumulate into multiple grants"
    assert len(expire_entries) == 1, "only the one balance that existed should expire, once"
    assert grant_entries[0].amount_mcr == 100_000
    assert new_fields.grant_balance_mcr == 100_000
    assert new_fields.period_spent_mcr == 0

    # Window lands on the boundary that actually contains `now`.
    assert new_fields.period_started_at <= _NOW < new_fields.period_ends_at
    # And it advanced in whole-week steps from the original boundary.
    assert (new_fields.period_ends_at - ends) % WEEK == timedelta(0)


def test_many_missed_weeks_with_zero_grant_leftover_still_grants_once():
    """Grant already at zero going in (nothing to expire) across a long gap."""
    started = _NOW - timedelta(weeks=20)
    ends = _NOW - timedelta(weeks=19)
    fields = _fields(grant=0, period_grant=0, started=started, ends=ends)

    new_fields, entries = roll_period(fields, weekly_grant_mcr=150_000, now=_NOW)

    assert [e.kind for e in entries] == ["grant"]
    assert new_fields.grant_balance_mcr == 150_000
    assert new_fields.period_grant_mcr == 150_000


# ---------- roll_period: free tier (zero grant) ----------------------------------


def test_roll_on_free_tier_grants_nothing_and_writes_no_grant_row():
    fields = _fields(grant=0, period_grant=0, ends=_NOW - timedelta(days=1))
    new_fields, entries = roll_period(fields, weekly_grant_mcr=0, now=_NOW)

    assert entries == []  # zero-amount ledger rows would just be noise
    assert new_fields.grant_balance_mcr == 0
    assert new_fields.period_grant_mcr == 0


# ---------- roll_period: purchased balance is never touched ---------------------


def test_purchased_balance_survives_every_shape_of_roll():
    for grant, weekly in [(0, 0), (0, 100_000), (40_000, 0), (40_000, 100_000)]:
        fields = _fields(grant=grant, purchased=77_000, ends=_NOW - timedelta(days=1))
        new_fields, _ = roll_period(fields, weekly_grant_mcr=weekly, now=_NOW)
        assert new_fields.purchased_balance_mcr == 77_000


# ---------- apply_plan_change ----------------------------------------------------


def test_plan_change_upgrade_tops_up_the_difference():
    fields = _fields(grant=10_000, purchased=2_000, period_grant=100_000)
    new_fields, entries = apply_plan_change(fields, new_weekly_grant_mcr=150_000)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.kind == "grant"
    assert entry.note == "plan change"
    assert entry.amount_mcr == 50_000
    assert entry.grant_delta_mcr == 50_000
    assert entry.purchased_delta_mcr == 0
    assert entry.balance_after_mcr == 62_000  # 10_000 + 50_000 + 2_000

    assert new_fields.grant_balance_mcr == 60_000
    assert new_fields.period_grant_mcr == 150_000


def test_plan_change_downgrade_does_not_claw_back():
    fields = _fields(grant=80_000, purchased=0, period_grant=100_000)
    new_fields, entries = apply_plan_change(fields, new_weekly_grant_mcr=30_000)

    assert entries == []
    assert new_fields == fields  # nothing already granted this window is taken back


def test_plan_change_to_the_same_tier_is_a_noop():
    fields = _fields(grant=80_000, period_grant=100_000)
    new_fields, entries = apply_plan_change(fields, new_weekly_grant_mcr=100_000)
    assert entries == []
    assert new_fields == fields


# ---------- wallet summary rendering (pure) --------------------------------------


def test_wallet_summary_dict_reports_credits_not_millicredits():
    from app.core.mcp_credits import _wallet_summary_dict

    fields = _fields(grant=37_400, purchased=50_000, period_grant=100_000, period_spent=62_600)
    summary = _wallet_summary_dict(fields, weekly_grant_credits=100.0)

    assert summary["balance"] == 87.4
    assert summary["grant_remaining"] == 37.4
    assert summary["grant_total"] == 100.0
    assert summary["purchased"] == 50.0
    assert summary["spent_this_period"] == 62.6
    assert summary["weekly_grant"] == 100.0
    assert summary["period_started_at"] == fields.period_started_at.isoformat()
    assert summary["period_ends_at"] == fields.period_ends_at.isoformat()


# ---------- DB integration (requires PostgreSQL) ---------------------------------

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
class TestWalletDB:
    @pytest_asyncio.fixture(autouse=True)
    async def _reset_process_engine_pool(self):
        """wallet_summary/topup/adjust use the process-global engine; each test
        has its own event loop, so connections pooled by an earlier test must go
        (the same reason as TestChargingDB in test_mcp_credits.py)."""
        from app.db.session import engine

        await engine.dispose()
        yield

    async def _make_user(self, db) -> uuid.UUID:
        from sqlalchemy import text

        user_id = uuid.uuid4()
        await db.execute(
            text("INSERT INTO \"user\" (id, email, hashed_password) VALUES (:id, :email, 'x')"),
            {"id": str(user_id), "email": f"{user_id}@test.com"},
        )
        await db.flush()
        return user_id

    async def test_first_touch_creates_a_due_wallet(self, db_session):
        from app.core.mcp_credits import get_or_create_wallet

        user_id = await self._make_user(db_session)
        wallet = await get_or_create_wallet(db_session, user_id)
        await db_session.commit()

        assert wallet.user_id == user_id
        assert wallet.period_started_at == wallet.period_ends_at
        assert wallet.grant_balance_mcr == 0

    async def test_get_or_create_is_idempotent(self, db_session):
        from app.core.mcp_credits import get_or_create_wallet

        user_id = await self._make_user(db_session)
        first = await get_or_create_wallet(db_session, user_id)
        await db_session.commit()
        second = await get_or_create_wallet(db_session, user_id)

        assert first.user_id == second.user_id
        assert first.period_started_at == second.period_started_at

    async def test_roll_period_if_due_issues_the_first_grant(self, db_session):
        from app.core.mcp_credits import get_or_create_wallet, roll_period_if_due

        user_id = await self._make_user(db_session)
        wallet = await get_or_create_wallet(db_session, user_id)
        wallet = await roll_period_if_due(db_session, wallet, credits_to_mcr(100))
        await db_session.commit()

        assert wallet.grant_balance_mcr == credits_to_mcr(100)
        assert wallet.period_grant_mcr == credits_to_mcr(100)
        assert wallet.period_ends_at > wallet.period_started_at

    async def test_wallet_summary_creates_and_rolls_in_one_call(self, db_session):
        from app.core.mcp_credits import wallet_summary

        user_id = await self._make_user(db_session)
        await db_session.commit()  # wallet_summary opens its own session and must see the user
        # wallet_summary opens its own session; DATABASE_URL points at the same DB.
        summary = await wallet_summary(user_id)

        assert summary["balance"] >= 0.0
        assert "period_ends_at" in summary

    async def test_topup_adds_to_purchased_and_survives_expiry(self, db_session):
        from app.core.mcp_credits import get_or_create_wallet, topup

        user_id = await self._make_user(db_session)
        await get_or_create_wallet(db_session, user_id)
        await db_session.commit()

        summary = await topup(user_id, credits_to_mcr(50), note="test topup")
        assert summary["purchased"] == 50.0

    async def test_adjust_clamps_at_zero(self, db_session):
        from app.core.mcp_credits import adjust, get_or_create_wallet

        user_id = await self._make_user(db_session)
        await get_or_create_wallet(db_session, user_id)
        await db_session.commit()

        summary = await adjust(user_id, -credits_to_mcr(9999), bucket="purchased", note="correction")
        assert summary["purchased"] == 0.0
