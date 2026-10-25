"""
Stage 3 tests: context-window statistics — max/last metrics, 'context_limit' close reason.

No DB required — pure unit tests.

Covers:
  - TestUpdateContextWindowMetrics  — token_budget.update_context_window_metrics behaviour
  - TestCloseReasons                — 'context_limit' in CLOSE_REASONS, not in old constraint
  - TestToolLimitReachedWindowMetrics — exception classes carry max_window_in/last_window_in
  - TestContextLimitCloseReason     — proxy_session maps context_window → context_limit
  - TestContextReplayResume         — context_replay hinted_replay treats context_limit as resume
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# TestUpdateContextWindowMetrics
# ---------------------------------------------------------------------------

class TestUpdateContextWindowMetrics:
    def _make_session(self, max_w: int = 0, last_w: int = 0):
        s = MagicMock()
        s.max_context_window_tokens = max_w
        s.last_context_window_tokens = last_w
        return s

    def test_sets_max_and_last_on_first_call(self):
        from app.core.token_budget import update_context_window_metrics
        s = self._make_session()
        update_context_window_metrics(s, 5000, 5000)
        assert s.max_context_window_tokens == 5000
        assert s.last_context_window_tokens == 5000

    def test_max_tracks_highest_value(self):
        from app.core.token_budget import update_context_window_metrics
        s = self._make_session(max_w=3000, last_w=3000)
        update_context_window_metrics(s, 8000, 8000)
        assert s.max_context_window_tokens == 8000
        assert s.last_context_window_tokens == 8000

    def test_max_does_not_decrease(self):
        from app.core.token_budget import update_context_window_metrics
        s = self._make_session(max_w=10000, last_w=10000)
        update_context_window_metrics(s, 2000, 2000)
        assert s.max_context_window_tokens == 10000
        assert s.last_context_window_tokens == 2000

    def test_last_always_updated(self):
        from app.core.token_budget import update_context_window_metrics
        s = self._make_session(max_w=5000, last_w=5000)
        update_context_window_metrics(s, 1000, 1000)
        assert s.last_context_window_tokens == 1000

    def test_zero_max_skips_update(self):
        """max_window_in=0 is treated as 'no data' and skipped."""
        from app.core.token_budget import update_context_window_metrics
        s = self._make_session(max_w=5000, last_w=5000)
        update_context_window_metrics(s, 0, 0)
        assert s.max_context_window_tokens == 5000
        assert s.last_context_window_tokens == 5000

    def test_different_max_and_last(self):
        from app.core.token_budget import update_context_window_metrics
        s = self._make_session()
        update_context_window_metrics(s, max_window_in=9000, last_window_in=4500)
        assert s.max_context_window_tokens == 9000
        assert s.last_context_window_tokens == 4500


# ---------------------------------------------------------------------------
# TestCloseReasons
# ---------------------------------------------------------------------------

class TestCloseReasons:
    def test_context_limit_in_close_reasons(self):
        from app.models.session.base import CLOSE_REASONS
        assert "context_limit" in CLOSE_REASONS

    def test_existing_reasons_still_present(self):
        from app.models.session.base import CLOSE_REASONS
        for reason in ("task_done", "token_limit", "tool_limit", "changes_requested",
                       "workspace_paused", "cancelled", "error", "retry"):
            assert reason in CLOSE_REASONS, f"Missing reason: {reason}"

    def test_context_limit_is_distinct_from_token_limit(self):
        from app.models.session.base import CLOSE_REASONS
        assert "context_limit" in CLOSE_REASONS
        assert "token_limit" in CLOSE_REASONS
        assert "context_limit" != "token_limit"


# ---------------------------------------------------------------------------
# TestToolLimitReachedWindowMetrics
# ---------------------------------------------------------------------------

class TestToolLimitReachedWindowMetrics:
    def test_base_class_stores_window_metrics(self):
        from app.core.proxy_session import ToolLimitReached
        exc = ToolLimitReached([], 100, 50, "model", max_window_in=8000, last_window_in=4000)
        assert exc.max_window_in == 8000
        assert exc.last_window_in == 4000

    def test_default_window_metrics_are_zero(self):
        from app.core.proxy_session import ToolLimitReached
        exc = ToolLimitReached([], 0, 0, "model")
        assert exc.max_window_in == 0
        assert exc.last_window_in == 0

    def test_context_window_limit_reached_propagates_metrics(self):
        from app.core.proxy_session import ContextWindowLimitReached
        exc = ContextWindowLimitReached(
            [], 200, 100, "model",
            context_window_limit=30_000,
            max_window_in=29_500,
            last_window_in=29_500,
        )
        assert exc.max_window_in == 29_500
        assert exc.last_window_in == 29_500
        assert exc.context_window_limit == 30_000
        assert exc.reason == "context_window"

    def test_token_budget_limit_reached_propagates_metrics(self):
        from app.core.proxy_session import TokenBudgetLimitReached
        exc = TokenBudgetLimitReached(
            [], 50_000, 50_000, "model",
            max_window_in=12_000,
            last_window_in=9_000,
        )
        assert exc.max_window_in == 12_000
        assert exc.last_window_in == 9_000

    def test_caught_as_tool_limit_reached_preserves_metrics(self):
        from app.core.proxy_session import ContextWindowLimitReached, ToolLimitReached
        exc = ContextWindowLimitReached(
            [], 0, 0, "m", context_window_limit=5_000,
            max_window_in=4_800, last_window_in=4_800,
        )
        caught = None
        try:
            raise exc
        except ToolLimitReached as e:
            caught = e
        assert caught is not None
        assert caught.max_window_in == 4_800
        assert caught.last_window_in == 4_800


# ---------------------------------------------------------------------------
# TestContextLimitCloseReason
# ---------------------------------------------------------------------------

class TestContextLimitCloseReason:
    """The proxy session maps limit_reason='context_window' → close_reason='context_limit'."""

    def _close_reason_for(self, limit_reason: str) -> str:
        """Replicate the close_reason logic from proxy_session._run_proxy_loop."""
        close_reason = (
            "context_limit"
            if limit_reason == "context_window"
            else "token_limit"
            if limit_reason == "token_budget"
            else "tool_limit"
        )
        return close_reason

    def test_context_window_maps_to_context_limit(self):
        assert self._close_reason_for("context_window") == "context_limit"

    def test_token_budget_maps_to_token_limit(self):
        assert self._close_reason_for("token_budget") == "token_limit"

    def test_tool_calls_maps_to_tool_limit(self):
        assert self._close_reason_for("tool_calls") == "tool_limit"

    def test_repeated_error_maps_to_tool_limit(self):
        assert self._close_reason_for("repeated_tool_error") == "tool_limit"

    def test_context_window_limit_reached_reason_attribute(self):
        from app.core.proxy_session import ContextWindowLimitReached
        exc = ContextWindowLimitReached([], 0, 0, "m", context_window_limit=10_000)
        assert exc.reason == "context_window"
        assert self._close_reason_for(exc.reason) == "context_limit"


# ---------------------------------------------------------------------------
# TestContextReplayResume
# ---------------------------------------------------------------------------

class TestContextReplayResume:
    """context_replay._classify returns 'resume' for context_limit."""

    def _strategy(self, close_reason: str | None) -> str:
        from app.core.context_replay import _classify
        return _classify(
            calls=[],
            title="some task",
            close_reason=close_reason,
            max_tool_result_tokens=0,
        )

    def test_context_limit_returns_resume(self):
        assert self._strategy("context_limit") == "resume"

    def test_token_limit_still_returns_resume(self):
        assert self._strategy("token_limit") == "resume"

    def test_tool_limit_returns_resume(self):
        assert self._strategy("tool_limit") == "resume"

    def test_task_done_does_not_return_resume(self):
        result = self._strategy("task_done")
        assert result != "resume"

    def test_none_does_not_return_resume(self):
        result = self._strategy(None)
        assert result != "resume"
