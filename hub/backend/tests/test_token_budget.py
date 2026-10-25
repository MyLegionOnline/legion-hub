"""
Unit tests for token_budget module (Step 6 / §2, §7.3).

Tests:
  - TestEstimateNextCallTokens — heuristic correctness
  - TestCheckTokenBudget       — legacy guard fires exactly at threshold, sets close_reason,
                                 emits SESSION_TOKEN_LIMIT_REACHED, skips closed sessions
  - TestTrackLlmUsage          — cumulative counters are correct across several replicas
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.token_budget import (
    TOKEN_LIMIT,
    _CHARS_PER_TOKEN,
    _DEFAULT_OUTPUT_ESTIMATE,
    check_token_budget,
    estimate_next_call_tokens,
    track_llm_usage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(tokens_in: int = 0, tokens_out: int = 0, closed: bool = False) -> MagicMock:
    session = MagicMock()
    session.id = uuid.uuid4()
    session.room_id = uuid.uuid4()
    session.kind = "user_agent_p1"
    session.tokens_in = tokens_in
    session.tokens_out = tokens_out
    session.cost_usd = Decimal("0")
    session.closed_at = MagicMock() if closed else None
    session.close_reason = None
    session.is_open = not closed
    return session


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


# ---------------------------------------------------------------------------
# TestEstimateNextCallTokens
# ---------------------------------------------------------------------------

class TestEstimateNextCallTokens:
    def test_empty_input_returns_at_least_one_plus_output_estimate(self):
        session = _make_session()
        est = estimate_next_call_tokens(session, "")
        # max(1, 0) + _DEFAULT_OUTPUT_ESTIMATE
        assert est == 1 + _DEFAULT_OUTPUT_ESTIMATE

    def test_short_text(self):
        session = _make_session()
        text = "a" * _CHARS_PER_TOKEN  # exactly 1 token
        est = estimate_next_call_tokens(session, text)
        assert est == 1 + _DEFAULT_OUTPUT_ESTIMATE

    def test_proportional_to_input_length(self):
        session = _make_session()
        text = "a" * (100 * _CHARS_PER_TOKEN)  # 100 tokens
        est = estimate_next_call_tokens(session, text)
        assert est == 100 + _DEFAULT_OUTPUT_ESTIMATE

    def test_always_positive(self):
        session = _make_session()
        for length in [0, 1, 100, 10_000]:
            assert estimate_next_call_tokens(session, "x" * length) > 0

    def test_large_input(self):
        session = _make_session()
        text = "x" * 400_000  # 100 000 tokens
        est = estimate_next_call_tokens(session, text)
        assert est == 100_000 + _DEFAULT_OUTPUT_ESTIMATE


# ---------------------------------------------------------------------------
# TestCheckTokenBudget
# ---------------------------------------------------------------------------

class TestCheckTokenBudget:
    @pytest.fixture(autouse=True)
    def _enable_legacy_cumulative_mode(self, monkeypatch):
        """These assertions cover the rollback-only cumulative branch."""
        monkeypatch.setattr(
            "app.core.token_budget.uses_cumulative_limit", lambda: True
        )

    @pytest.mark.asyncio
    async def test_allows_when_well_under_limit(self):
        session = _make_session(tokens_in=1_000, tokens_out=1_000)
        db = _make_db()
        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as mock_emit:
            result = await check_token_budget(session, "hello", db)
        assert result is True
        mock_emit.assert_not_called()
        assert session.closed_at is None

    @pytest.mark.asyncio
    async def test_blocks_exactly_at_threshold(self):
        # Arrange so that tokens_in + tokens_out + estimate == TOKEN_LIMIT exactly
        input_text = "a" * _CHARS_PER_TOKEN          # 1 input token
        estimate = 1 + _DEFAULT_OUTPUT_ESTIMATE        # = 1 001
        tokens_already_used = TOKEN_LIMIT - estimate   # = 98 999
        session = _make_session(tokens_in=tokens_already_used, tokens_out=0)
        db = _make_db()
        with (
            patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as mock_emit,
            patch("app.core.metrics.record_legacy_cumulative_guard_hit") as metric,
        ):
            result = await check_token_budget(session, input_text, db)
        assert result is False
        mock_emit.assert_called_once()
        metric.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_allows_one_token_below_threshold(self):
        input_text = "a" * _CHARS_PER_TOKEN           # 1 input token → estimate = 1 001
        tokens_already_used = TOKEN_LIMIT - 1_002      # projected = 99 999 < 100 000
        session = _make_session(tokens_in=tokens_already_used, tokens_out=0)
        db = _make_db()
        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as mock_emit:
            result = await check_token_budget(session, input_text, db)
        assert result is True
        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_sets_close_reason_token_limit(self):
        session = _make_session(tokens_in=99_999, tokens_out=0)
        db = _make_db()
        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock):
            await check_token_budget(session, "anything", db)
        assert session.close_reason == "token_limit"
        assert session.closed_at is not None

    @pytest.mark.asyncio
    async def test_does_not_set_close_reason_when_under_limit(self):
        session = _make_session(tokens_in=0, tokens_out=0)
        db = _make_db()
        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock):
            await check_token_budget(session, "hi", db)
        assert session.close_reason is None

    @pytest.mark.asyncio
    async def test_emits_correct_event_type(self):
        session = _make_session(tokens_in=99_000, tokens_out=0)
        db = _make_db()
        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as mock_emit:
            await check_token_budget(session, "x" * (1_000 * _CHARS_PER_TOKEN), db)
        if mock_emit.called:
            assert mock_emit.call_args.kwargs["event_type"] == "SESSION_TOKEN_LIMIT_REACHED"

    @pytest.mark.asyncio
    async def test_event_metadata_contains_required_fields(self):
        session = _make_session(tokens_in=99_500, tokens_out=0)
        db = _make_db()
        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as mock_emit:
            await check_token_budget(session, "x" * 400, db)
        if mock_emit.called:
            metadata = mock_emit.call_args.kwargs["metadata"]
            for key in ("session_id", "kind", "tokens_in", "tokens_out", "estimate", "projected", "limit"):
                assert key in metadata, f"missing metadata key: {key}"
            assert metadata["limit"] == TOKEN_LIMIT

    @pytest.mark.asyncio
    async def test_skips_already_closed_session(self):
        session = _make_session(tokens_in=0, tokens_out=0, closed=True)
        db = _make_db()
        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as mock_emit:
            result = await check_token_budget(session, "hello", db)
        assert result is False
        mock_emit.assert_not_called()
        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_flushes_session_before_emitting(self):
        session = _make_session(tokens_in=99_999, tokens_out=0)
        db = _make_db()
        call_order: list[str] = []
        db.flush.side_effect = lambda: call_order.append("flush")

        async def fake_emit(**kwargs):
            call_order.append("emit")

        with patch("app.core.token_budget.emit_event", side_effect=fake_emit):
            await check_token_budget(session, "x", db)

        assert call_order == ["flush", "emit"], "flush must happen before emit"


class TestCheckTokenBudgetWindowMode:
    @pytest.mark.asyncio
    async def test_cumulative_usage_never_closes_a_window_session(self):
        session = _make_session(tokens_in=110_000, tokens_out=5_000)
        db = _make_db()
        with (
            patch("app.core.token_budget.uses_cumulative_limit", return_value=False),
            patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as mock_emit,
        ):
            result = await check_token_budget(session, "next compact prompt", db)

        assert result is True
        assert session.close_reason is None
        mock_emit.assert_not_called()


# ---------------------------------------------------------------------------
# TestTrackLlmUsage
# ---------------------------------------------------------------------------

class TestTrackLlmUsage:
    def test_increments_tokens_in(self):
        session = _make_session(tokens_in=100, tokens_out=0)
        track_llm_usage(session, tokens_in=200, tokens_out=0)
        assert session.tokens_in == 300

    def test_increments_tokens_out(self):
        session = _make_session(tokens_in=0, tokens_out=50)
        track_llm_usage(session, tokens_in=0, tokens_out=150)
        assert session.tokens_out == 200

    def test_increments_cost_usd(self):
        session = _make_session()
        track_llm_usage(session, tokens_in=1_000, tokens_out=500, cost_usd=0.00500)
        assert session.cost_usd == Decimal("0.005")

    def test_default_cost_is_zero(self):
        session = _make_session()
        track_llm_usage(session, tokens_in=100, tokens_out=100)
        assert session.cost_usd == Decimal("0")

    def test_cumulative_over_multiple_calls(self):
        session = _make_session()
        track_llm_usage(session, tokens_in=100, tokens_out=50,  cost_usd=0.001)
        track_llm_usage(session, tokens_in=200, tokens_out=100, cost_usd=0.002)
        track_llm_usage(session, tokens_in=300, tokens_out=150, cost_usd=0.003)
        assert session.tokens_in  == 600
        assert session.tokens_out == 300
        assert session.cost_usd   == Decimal("0.006")

    def test_zero_tokens_does_not_change_counts(self):
        session = _make_session(tokens_in=500, tokens_out=300)
        track_llm_usage(session, tokens_in=0, tokens_out=0)
        assert session.tokens_in  == 500
        assert session.tokens_out == 300

    def test_cost_accumulates_without_float_drift(self):
        session = _make_session()
        for _ in range(100):
            track_llm_usage(session, tokens_in=0, tokens_out=0, cost_usd=0.001)
        assert session.cost_usd == Decimal("0.1")
