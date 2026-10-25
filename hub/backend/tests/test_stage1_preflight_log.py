"""
Unit tests for Stage 1 — preflight log and user message fixes.

Tests:
  - TestPreflightLimitDisplay   — _preflight_limit_display returns correct limit per reason
  - TestPromptCapLimitReached   — exception carries estimated_input_tokens, cap, cap_env_var
  - TestUserMessageFormat       — user-visible message contains all required info for prompt_cap
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.proxy_session import (
    PromptCapLimitReached,
    _preflight_limit_display,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_policy(input_cap: int, model_context_limit: int, context_window_limit: int):
    return SimpleNamespace(
        input_cap=input_cap,
        model_context_limit=model_context_limit,
        context_window_limit=context_window_limit,
    )


# ---------------------------------------------------------------------------
# TestPreflightLimitDisplay
# ---------------------------------------------------------------------------

class TestPreflightLimitDisplay:
    """_preflight_limit_display returns the threshold that actually caused rejection."""

    @pytest.mark.parametrize(
        "reason, input_cap, model_limit, window_limit, expected",
        [
            # prompt_cap: min(input_cap, model_context_limit)
            ("prompt_cap", 16_000, 200_000, 100_000, 16_000),
            # prompt_cap when model_limit is the tighter bound
            ("prompt_cap", 16_000, 10_000, 100_000, 10_000),
            # model_context_limit: model_context_limit
            ("model_context_limit", 16_000, 8_000, 100_000, 8_000),
            # context_window_limit: context_window_limit
            ("context_window_limit", 16_000, 200_000, 100_000, 100_000),
            # session_budget: context_window_limit (fallback)
            ("session_budget", 16_000, 200_000, 100_000, 100_000),
            # unknown reason: context_window_limit (safe default)
            ("unknown_reason", 16_000, 200_000, 100_000, 100_000),
        ],
    )
    def test_limit_per_reason(
        self, reason, input_cap, model_limit, window_limit, expected
    ):
        policy = _make_policy(input_cap, model_limit, window_limit)
        assert _preflight_limit_display(reason, policy) == expected

    def test_none_reason_returns_window_limit(self):
        policy = _make_policy(8_000, 200_000, 100_000)
        assert _preflight_limit_display(None, policy) == 100_000


# ---------------------------------------------------------------------------
# TestPromptCapLimitReached
# ---------------------------------------------------------------------------

class TestPromptCapLimitReached:
    """PromptCapLimitReached carries diagnostic fields for the user message."""

    def _make_exc(self, **kwargs) -> PromptCapLimitReached:
        defaults = dict(
            working=[],
            tokens_in=0,
            tokens_out=0,
            model="test-model",
        )
        return PromptCapLimitReached(**defaults, **kwargs)

    def test_defaults(self):
        exc = self._make_exc()
        assert exc.rejected_reason == "prompt_cap"
        assert exc.estimated_input_tokens == 0
        assert exc.cap == 0
        assert exc.cap_env_var == "CONTEXT_AGENT_INPUT_CAP"

    def test_custom_fields(self):
        exc = self._make_exc(
            rejected_reason="model_context_limit",
            estimated_input_tokens=18_000,
            cap=16_000,
            cap_env_var="CONTEXT_CHAT_MASTER_INPUT_CAP",
        )
        assert exc.rejected_reason == "model_context_limit"
        assert exc.estimated_input_tokens == 18_000
        assert exc.cap == 16_000
        assert exc.cap_env_var == "CONTEXT_CHAT_MASTER_INPUT_CAP"

    @pytest.mark.parametrize(
        "context_role, expected_env_var",
        [
            ("chat_master", "CONTEXT_CHAT_MASTER_INPUT_CAP"),
            ("agent", "CONTEXT_AGENT_INPUT_CAP"),
            ("coder", "CONTEXT_AGENT_INPUT_CAP"),
            ("unknown", "CONTEXT_AGENT_INPUT_CAP"),
        ],
    )
    def test_cap_env_var_by_role(self, context_role, expected_env_var):
        cap_env_var = (
            "CONTEXT_CHAT_MASTER_INPUT_CAP"
            if context_role == "chat_master"
            else "CONTEXT_AGENT_INPUT_CAP"
        )
        exc = self._make_exc(cap_env_var=cap_env_var)
        assert exc.cap_env_var == expected_env_var


# ---------------------------------------------------------------------------
# TestUserMessageFormat
# ---------------------------------------------------------------------------

class TestUserMessageFormat:
    """The user-visible prompt_cap message contains estimated_input, cap, and env_var."""

    def _build_message(self, exc: PromptCapLimitReached) -> str:
        return (
            f"cannot send the request: the prompt is estimated at "
            f"{getattr(exc, 'estimated_input_tokens', 0):,} tokens, "
            f"the limit of {getattr(exc, 'cap', 0):,} tokens is exceeded "
            f"(adjustable: {getattr(exc, 'cap_env_var', 'CONTEXT_AGENT_INPUT_CAP')}). "
            f"Reduce the task context size or simplify the handoff."
        )

    @pytest.mark.parametrize(
        "estimated, cap, env_var",
        [
            (16_630, 16_000, "CONTEXT_AGENT_INPUT_CAP"),
            (27_546, 8_000, "CONTEXT_CHAT_MASTER_INPUT_CAP"),
            (1_200, 1_000, "CONTEXT_AGENT_INPUT_CAP"),  # acceptance test: cap=1000 visible
        ],
    )
    def test_message_contains_all_fields(self, estimated, cap, env_var):
        exc = PromptCapLimitReached(
            [],
            0,
            0,
            "test-model",
            estimated_input_tokens=estimated,
            cap=cap,
            cap_env_var=env_var,
        )
        msg = self._build_message(exc)

        assert f"{estimated:,}" in msg, "estimated_input_tokens missing from message"
        assert f"{cap:,}" in msg, "cap missing from message"
        assert env_var in msg, "cap_env_var name missing from message"
        assert "Reduce" in msg, "actionable hint missing"

    def test_acceptance_cap_1000_visible_not_100000(self):
        """Acceptance: when CONTEXT_AGENT_INPUT_CAP=1000, message shows 1,000, not 100,000."""
        exc = PromptCapLimitReached(
            [],
            0,
            0,
            "test-model",
            estimated_input_tokens=1_200,
            cap=1_000,
            cap_env_var="CONTEXT_AGENT_INPUT_CAP",
        )
        msg = self._build_message(exc)
        assert "1,000" in msg
        assert "100,000" not in msg
