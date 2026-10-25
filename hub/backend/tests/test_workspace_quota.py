"""Unit tests for workspace quota logic (Admin Center stage 0).

Covers:
- the free/hobby default mismatch fix (model default == no-row fallback)
- resolve_tier: plan expiry semantics
- check_start_allowed: 402/429 enforcement against effective limits

All tests are pure/mocked — no database required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import app.core.workspace_quota as wq
from app.models.user_quota import DEFAULT_PLAN_TIER, UserQuota


@pytest.fixture(autouse=True)
def _no_extra_hours(monkeypatch):
    """No plugin here: nothing adds hours to the plan's month (bought hour
    packs are covered by the cloud plugin tests)."""
    from app.core import work_gates

    monkeypatch.setattr(work_gates, "extra_hours", AsyncMock(return_value=None))
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(wq, "get_hours_window", AsyncMock(return_value=(now, now)))


# ---------------------------------------------------------------------------
# Defaults consistency (the original 402 bug)
# ---------------------------------------------------------------------------


class TestDefaultsConsistency:
    def test_new_row_default_matches_no_row_fallback(self):
        """A userquota row created as a side effect (skills/logs) must give the
        user the same tier as having no row at all."""
        row_tier = UserQuota(user_id=uuid.uuid4()).plan_tier
        no_row_tier, _ = wq.resolve_tier(None)
        assert row_tier == no_row_tier == DEFAULT_PLAN_TIER

# ---------------------------------------------------------------------------
# resolve_tier
# ---------------------------------------------------------------------------


class TestResolveTier:
    def test_no_row(self):
        assert wq.resolve_tier(None) == (DEFAULT_PLAN_TIER, False)

    def test_plain_tier(self):
        q = UserQuota(user_id=uuid.uuid4(), plan_tier="pro")
        assert wq.resolve_tier(q) == ("pro", False)

    def test_future_expiry_keeps_tier(self):
        q = UserQuota(
            user_id=uuid.uuid4(), plan_tier="pro",
            plan_expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        assert wq.resolve_tier(q) == ("pro", False)

    def test_past_expiry_reverts_to_default(self):
        q = UserQuota(
            user_id=uuid.uuid4(), plan_tier="pro",
            plan_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
        assert wq.resolve_tier(q) == (wq.EXPIRED_PLAN_TIER, True)

    def test_naive_expiry_treated_as_utc(self):
        q = UserQuota(
            user_id=uuid.uuid4(), plan_tier="pro",
            plan_expires_at=datetime.utcnow() - timedelta(minutes=1),
        )
        tier, expired = wq.resolve_tier(q)
        assert (tier, expired) == (wq.EXPIRED_PLAN_TIER, True)

    def test_empty_tier_falls_back_to_default(self):
        q = UserQuota(user_id=uuid.uuid4(), plan_tier="")
        assert wq.resolve_tier(q) == (DEFAULT_PLAN_TIER, False)


# ---------------------------------------------------------------------------
# check_start_allowed (enforcement against effective limits)
# ---------------------------------------------------------------------------


def _limits(**over):
    base = {
        "plan_tier": "hobby", "override_active": False,
        "max_running": 1, "max_rooms": 3, "monthly_hours": 40, "mem_limit_mb": 1024,
    }
    base.update(over)
    return base


def _user():
    u = MagicMock()
    u.id = uuid.uuid4()
    return u


class TestCheckStartAllowed:
    @pytest.mark.asyncio
    async def test_zero_max_running_gives_402(self, monkeypatch):
        monkeypatch.setattr(
            wq, "get_effective_limits",
            AsyncMock(return_value=_limits(plan_tier="free", max_running=0)),
        )
        with pytest.raises(HTTPException) as exc:
            await wq.check_start_allowed(_user(), MagicMock())
        assert exc.value.status_code == 402

    @pytest.mark.asyncio
    async def test_running_at_limit_gives_429(self, monkeypatch):
        monkeypatch.setattr(wq, "get_effective_limits", AsyncMock(return_value=_limits()))
        monkeypatch.setattr(wq, "count_running_workspaces_for_user", AsyncMock(return_value=1))
        with pytest.raises(HTTPException) as exc:
            await wq.check_start_allowed(_user(), MagicMock())
        assert exc.value.status_code == 429

    @pytest.mark.asyncio
    async def test_hours_exhausted_gives_402(self, monkeypatch):
        monkeypatch.setattr(wq, "get_effective_limits", AsyncMock(return_value=_limits()))
        monkeypatch.setattr(wq, "count_running_workspaces_for_user", AsyncMock(return_value=0))
        monkeypatch.setattr(wq, "monthly_hours_consumed", AsyncMock(return_value=40.0))
        with pytest.raises(HTTPException) as exc:
            await wq.check_start_allowed(_user(), MagicMock())
        assert exc.value.status_code == 402

    @pytest.mark.asyncio
    async def test_override_unblocks_hours(self, monkeypatch):
        """The dev scenario: hobby user exhausted 40h, admin sets an override."""
        monkeypatch.setattr(
            wq, "get_effective_limits",
            AsyncMock(return_value=_limits(monthly_hours=100, override_active=True)),
        )
        monkeypatch.setattr(wq, "count_running_workspaces_for_user", AsyncMock(return_value=0))
        monkeypatch.setattr(wq, "monthly_hours_consumed", AsyncMock(return_value=40.0))
        await wq.check_start_allowed(_user(), MagicMock())  # must not raise

    @pytest.mark.asyncio
    async def test_unlimited_hours_skips_check(self, monkeypatch):
        monkeypatch.setattr(
            wq, "get_effective_limits",
            AsyncMock(return_value=_limits(plan_tier="pro", monthly_hours=None)),
        )
        monkeypatch.setattr(wq, "count_running_workspaces_for_user", AsyncMock(return_value=0))
        hours = AsyncMock(return_value=9999.0)
        monkeypatch.setattr(wq, "monthly_hours_consumed", hours)
        await wq.check_start_allowed(_user(), MagicMock())
        hours.assert_not_awaited()
