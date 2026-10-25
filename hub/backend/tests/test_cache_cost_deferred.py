"""Provider-cache price corrections land together with the turn's own charge.

A direct video-agent room (2026-10-24): an OpenRouter tool loop wrote a negative
cache correction to the session after every request, while the turn's tokens
were charged only at the end.  The turn was paused, so the session kept the
corrections alone: cost_usd = −$2.29 with no tokens behind it.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core import token_budget
from app.core.context_assembler import persist_provider_cache_usage
from app.core.token_budget import defer_cache_cost, pending_cache_cost, track_llm_usage


@pytest.fixture(autouse=True)
def _clean_pending():
    token_budget._PENDING_CACHE_COST.clear()
    yield
    token_budget._PENDING_CACHE_COST.clear()


def _session() -> MagicMock:
    session = MagicMock()
    session.id = uuid.uuid4()
    session.kind = "user_agent_p2"
    session.tokens_in = 0
    session.tokens_out = 0
    session.cost_usd = Decimal("0")
    session.context_snapshot = {}
    return session


def _db_for(session) -> AsyncMock:
    db = AsyncMock()
    db.get = AsyncMock(return_value=session)
    db.add = MagicMock()
    return db


@pytest.mark.asyncio
async def test_persist_defers_correction_instead_of_writing_cost():
    session = _session()

    await persist_provider_cache_usage(
        session_id=session.id, db=_db_for(session), read_tokens=90_000, cost_usd=-0.026
    )

    assert session.cost_usd == Decimal("0")
    assert session.context_snapshot["context_gateway"]["provider_cache_read_tokens"] == 90_000
    assert pending_cache_cost(session.id) == Decimal("-0.026")


def test_turn_charge_lands_with_its_corrections():
    session = _session()
    defer_cache_cost(session.id, -0.02)
    defer_cache_cost(session.id, -0.01)

    track_llm_usage(session, 200_000, 1_000, 0.0612)

    assert session.cost_usd == Decimal("0.0312")
    assert pending_cache_cost(session.id) == Decimal("0")


def test_uncharged_turn_leaves_no_negative_cost():
    session = _session()
    defer_cache_cost(session.id, -2.29)

    # The turn is paused before its charge: nothing reaches the session.
    assert session.cost_usd == Decimal("0")


def test_stale_correction_never_drives_a_charge_below_zero():
    session = _session()
    defer_cache_cost(session.id, -2.29)  # left by a turn that was never charged

    track_llm_usage(session, 1_500, 200, 0.0007)  # small close-time call

    assert session.cost_usd == Decimal("0")


def test_zero_token_tracking_does_not_consume_corrections():
    session = _session()
    defer_cache_cost(session.id, -0.01)

    track_llm_usage(session, 0, 0, 0.0)

    assert pending_cache_cost(session.id) == Decimal("-0.01")


def test_positive_correction_is_added_in_full():
    session = _session()
    defer_cache_cost(session.id, 0.004)  # Anthropic cache write, outside input_tokens

    track_llm_usage(session, 1_000, 100, 0.001)

    assert session.cost_usd == Decimal("0.005")


def test_pending_map_is_bounded(monkeypatch):
    monkeypatch.setattr(token_budget, "_PENDING_CACHE_COST_MAX", 3)
    ids = [uuid.uuid4() for _ in range(5)]
    for session_id in ids:
        defer_cache_cost(session_id, -0.01)

    assert len(token_budget._PENDING_CACHE_COST) == 3
    assert pending_cache_cost(ids[0]) == Decimal("0")
    assert pending_cache_cost(ids[-1]) == Decimal("-0.01")
