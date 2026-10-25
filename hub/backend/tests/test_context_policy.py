"""
Unit tests for context_policy module — Stage 0 (context_window_limit field).

Tests:
  - TestEnvHelpers             — _env_int/_env_bool parse correctly
  - TestPolicyForRole          — context_window_limit set from env default
  - TestApplyPolicyOverrides   — primary alias, short aliases, clamp, priority, no-op None
  - TestContextLimitMode       — CONTEXT_LIMIT_MODE is validated and selects runtime semantics
  - TestResolveContextPolicy   — agent.llm_config wins; room < agent < llm_config
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import app.core.context_policy as policy_mod
from app.core.context_policy import (
    ContextPolicy,
    apply_policy_overrides,
    policy_for_role,
    resolve_context_policy,
    _env_int,
    _OVERRIDE_FIELDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_policy(**overrides) -> ContextPolicy:
    """Return a minimal policy; override any field via kwargs."""
    defaults = dict(
        enabled=True,
        input_cap=16_000,
        active_tool_result_cap=6_000,
        keep_raw_results=3,
        output_reserve=4_096,
        model_context_limit=200_000,
        checkpoint_cap=600,
        context_window_limit=100_000,
    )
    defaults.update(overrides)
    return ContextPolicy(**defaults)


def _make_db(room=None, agent=None) -> AsyncMock:
    db = AsyncMock()

    async def _get(model, pk):
        if hasattr(model, "__name__") and model.__name__ == "Room":
            return room
        if hasattr(model, "__name__") and model.__name__ == "Agent":
            return agent
        return None

    db.get = AsyncMock(side_effect=_get)
    return db


# ---------------------------------------------------------------------------
# TestEnvHelpers
# ---------------------------------------------------------------------------

class TestEnvHelpers:
    def test_env_int_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("_TEST_MISSING_VAR", raising=False)
        assert _env_int("_TEST_MISSING_VAR", 42) == 42

    def test_env_int_parses_integer(self, monkeypatch):
        monkeypatch.setenv("_TEST_INT_VAR", "12345")
        assert _env_int("_TEST_INT_VAR", 0) == 12_345

    def test_env_int_returns_default_on_invalid(self, monkeypatch):
        monkeypatch.setenv("_TEST_INT_VAR", "not-a-number")
        assert _env_int("_TEST_INT_VAR", 99) == 99

    def test_env_int_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("_TEST_INT_VAR", "  500  ")
        assert _env_int("_TEST_INT_VAR", 0) == 500


# ---------------------------------------------------------------------------
# TestPolicyForRole
# ---------------------------------------------------------------------------

class TestPolicyForRole:
    def test_default_context_window_limit_is_100k(self, monkeypatch):
        monkeypatch.delenv("CONTEXT_WINDOW_LIMIT", raising=False)
        # Reload module so _CONTEXT_WINDOW_LIMIT_DEFAULT picks up env change.
        importlib.reload(policy_mod)
        p = policy_mod.policy_for_role("agent")
        assert p.context_window_limit == 100_000

    def test_env_overrides_context_window_limit(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_WINDOW_LIMIT", "50000")
        importlib.reload(policy_mod)
        p = policy_mod.policy_for_role("agent")
        assert p.context_window_limit == 50_000

    def test_context_window_limit_clamped_to_1024(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_WINDOW_LIMIT", "100")
        importlib.reload(policy_mod)
        p = policy_mod.policy_for_role("agent")
        assert p.context_window_limit == 1024

    def test_field_present_on_dataclass(self):
        p = policy_for_role("agent")
        assert hasattr(p, "context_window_limit")
        assert isinstance(p.context_window_limit, int)

    def test_context_window_limit_in_override_fields(self):
        assert "context_window_limit" in _OVERRIDE_FIELDS
        assert "context_window_limit" in _OVERRIDE_FIELDS["context_window_limit"]
        assert "context_window" in _OVERRIDE_FIELDS["context_window_limit"]
        assert "context_limit" in _OVERRIDE_FIELDS["context_window_limit"]

    # --- Stage 2: input_cap defaults to context_window_limit ---

    def test_default_input_cap_equals_context_window_limit(self, monkeypatch):
        monkeypatch.delenv("CONTEXT_AGENT_INPUT_CAP", raising=False)
        monkeypatch.delenv("CONTEXT_WINDOW_LIMIT", raising=False)
        importlib.reload(policy_mod)
        p = policy_mod.policy_for_role("agent")
        assert p.input_cap == p.context_window_limit == 100_000

    def test_cm_default_input_cap_equals_context_window_limit(self, monkeypatch):
        monkeypatch.delenv("CONTEXT_CHAT_MASTER_INPUT_CAP", raising=False)
        monkeypatch.delenv("CONTEXT_WINDOW_LIMIT", raising=False)
        importlib.reload(policy_mod)
        p = policy_mod.policy_for_role("chat_master")
        assert p.input_cap == p.context_window_limit == 100_000

    def test_explicit_input_cap_env_overrides_default(self, monkeypatch):
        """Admin setting CONTEXT_AGENT_INPUT_CAP=20000 is respected even though CWL=100000."""
        monkeypatch.setenv("CONTEXT_AGENT_INPUT_CAP", "20000")
        monkeypatch.delenv("CONTEXT_WINDOW_LIMIT", raising=False)
        importlib.reload(policy_mod)
        p = policy_mod.policy_for_role("agent")
        assert p.input_cap == 20_000
        assert p.context_window_limit == 100_000


# ---------------------------------------------------------------------------
# TestApplyPolicyOverrides
# ---------------------------------------------------------------------------

class TestApplyPolicyOverrides:
    def test_primary_alias_sets_context_window_limit(self):
        base = _base_policy(context_window_limit=100_000)
        result = apply_policy_overrides(base, {"context_window_limit": 60_000}, role="agent")
        assert result.context_window_limit == 60_000

    def test_short_alias_context_window(self):
        base = _base_policy(context_window_limit=100_000)
        result = apply_policy_overrides(base, {"context_window": 55_000}, role="agent")
        assert result.context_window_limit == 55_000

    def test_short_alias_context_limit(self):
        base = _base_policy(context_window_limit=100_000)
        result = apply_policy_overrides(base, {"context_limit": 45_000}, role="agent")
        assert result.context_window_limit == 45_000

    def test_clamp_to_minimum_1024(self):
        base = _base_policy(context_window_limit=100_000)
        result = apply_policy_overrides(base, {"context_window_limit": 100}, role="agent")
        assert result.context_window_limit == 1024

    def test_none_config_is_skipped(self):
        base = _base_policy(context_window_limit=80_000)
        result = apply_policy_overrides(base, None, role="agent")
        assert result.context_window_limit == 80_000

    def test_empty_config_is_skipped(self):
        base = _base_policy(context_window_limit=70_000)
        result = apply_policy_overrides(base, {}, role="agent")
        assert result.context_window_limit == 70_000

    def test_later_config_wins_over_earlier(self):
        base = _base_policy(context_window_limit=100_000)
        room_cfg = {"context_window_limit": 40_000}
        agent_cfg = {"context_window_limit": 75_000}
        result = apply_policy_overrides(base, room_cfg, agent_cfg, role="agent")
        assert result.context_window_limit == 75_000

    def test_three_layers_room_agent_llm(self):
        base = _base_policy(context_window_limit=100_000)
        room_cfg = {"context_window_limit": 30_000}
        agent_cfg = {"context_window_limit": 50_000}
        llm_cfg = {"context_window_limit": 90_000}
        result = apply_policy_overrides(base, room_cfg, agent_cfg, llm_cfg, role="agent")
        assert result.context_window_limit == 90_000

    def test_invalid_string_value_is_ignored(self):
        base = _base_policy(context_window_limit=80_000)
        result = apply_policy_overrides(base, {"context_window_limit": "bad"}, role="agent")
        assert result.context_window_limit == 80_000

    def test_string_number_is_accepted(self):
        base = _base_policy(context_window_limit=100_000)
        result = apply_policy_overrides(base, {"context_window_limit": "32000"}, role="agent")
        assert result.context_window_limit == 32_000

    def test_other_fields_unchanged(self):
        base = _base_policy(context_window_limit=100_000, input_cap=16_000)
        result = apply_policy_overrides(base, {"context_window_limit": 50_000}, role="agent")
        assert result.input_cap == 16_000

    def test_alias_priority_primary_over_short(self):
        # When both the primary alias and a short alias appear in the same config
        # dict, the _first_ matching alias wins (next() short-circuits).
        base = _base_policy(context_window_limit=100_000)
        cfg = {"context_window_limit": 60_000, "context_window": 30_000}
        result = apply_policy_overrides(base, cfg, role="agent")
        # primary alias appears first in the tuple → 60_000
        assert result.context_window_limit == 60_000

    # --- Stage 2: input_cap normalization in apply_policy_overrides ---

    def test_input_cap_clamped_when_context_window_limit_narrowed(self):
        """When context_window_limit is overridden to 50k, input_cap (100k) is clamped."""
        base = _base_policy(context_window_limit=100_000, input_cap=100_000)
        result = apply_policy_overrides(base, {"context_window_limit": 50_000}, role="agent")
        assert result.context_window_limit == 50_000
        assert result.input_cap == 50_000

    def test_input_cap_below_context_window_limit_unchanged(self):
        """input_cap < context_window_limit is not raised by normalization."""
        base = _base_policy(context_window_limit=100_000, input_cap=16_000)
        result = apply_policy_overrides(base, {"context_window_limit": 50_000}, role="agent")
        assert result.input_cap == 16_000

    def test_explicit_input_cap_override_below_cwl_respected(self):
        """Explicit input_cap override stays if it's below context_window_limit."""
        base = _base_policy(context_window_limit=100_000, input_cap=100_000)
        result = apply_policy_overrides(
            base, {"input_cap": 20_000, "context_window_limit": 100_000}, role="agent"
        )
        assert result.input_cap == 20_000

    def test_provider_context_length_sets_model_ceiling_for_existing_agents(self):
        base = _base_policy(model_context_limit=200_000)
        result = apply_policy_overrides(
            base,
            {"context_length": 32_000, "context_window_limit": 16_000},
            role="agent",
        )
        assert result.model_context_limit == 32_000
        assert result.context_window_limit == 16_000

    def test_unknown_provider_context_length_does_not_narrow_model_ceiling(self):
        base = _base_policy(model_context_limit=200_000)
        result = apply_policy_overrides(base, {"context_length": 0}, role="agent")
        assert result.model_context_limit == 200_000


# ---------------------------------------------------------------------------
# TestContextLimitMode
# ---------------------------------------------------------------------------

class TestContextLimitMode:
    def test_default_mode_is_window(self, monkeypatch):
        monkeypatch.delenv("CONTEXT_LIMIT_MODE", raising=False)
        importlib.reload(policy_mod)
        assert policy_mod.context_limit_mode() == "window"
        assert policy_mod.uses_window_limit()
        assert not policy_mod.uses_cumulative_limit()

    def test_env_sets_window_mode(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "window")
        importlib.reload(policy_mod)
        assert policy_mod.context_limit_mode() == "window"

    def test_mode_is_lowercased(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "WINDOW")
        importlib.reload(policy_mod)
        assert policy_mod.context_limit_mode() == "window"

    def test_cumulative_mode_is_an_explicit_legacy_branch(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "cumulative")
        importlib.reload(policy_mod)
        assert policy_mod.context_limit_mode() == "cumulative"
        assert not policy_mod.uses_window_limit()
        assert policy_mod.uses_cumulative_limit()

    def test_invalid_mode_fails_fast(self, monkeypatch):
        monkeypatch.setenv("CONTEXT_LIMIT_MODE", "per_session")
        with pytest.raises(ValueError, match="Invalid CONTEXT_LIMIT_MODE"):
            importlib.reload(policy_mod)
        monkeypatch.delenv("CONTEXT_LIMIT_MODE")
        importlib.reload(policy_mod)


# ---------------------------------------------------------------------------
# TestResolveContextPolicy
# ---------------------------------------------------------------------------

class TestResolveContextPolicy:
    @pytest.mark.asyncio
    async def test_no_agent_returns_policy_for_role(self):
        room = SimpleNamespace(coordinator_config=None)
        db = _make_db(room=room, agent=None)
        policy = await resolve_context_policy(
            role="agent",
            db=db,
            room_id=uuid4(),
            agent_id=None,
            llm_config=None,
        )
        # Use the live class (may differ after importlib.reload in sibling tests).
        assert isinstance(policy, policy_mod.ContextPolicy)
        assert policy.context_window_limit >= 1024

    @pytest.mark.asyncio
    async def test_agent_llm_config_overrides_default(self):
        room = SimpleNamespace(coordinator_config=None)
        agent = SimpleNamespace(config=None, llm_config={"context_window_limit": 55_000})
        db = _make_db(room=room, agent=agent)
        policy = await resolve_context_policy(
            role="agent",
            db=db,
            room_id=uuid4(),
            agent_id=uuid4(),
            llm_config=None,
        )
        assert policy.context_window_limit == 55_000

    @pytest.mark.asyncio
    async def test_request_llm_config_wins_over_agent(self):
        room = SimpleNamespace(coordinator_config=None)
        agent = SimpleNamespace(config=None, llm_config={"context_window_limit": 55_000})
        db = _make_db(room=room, agent=agent)
        policy = await resolve_context_policy(
            role="agent",
            db=db,
            room_id=uuid4(),
            agent_id=uuid4(),
            llm_config={"context_window_limit": 80_000},
        )
        assert policy.context_window_limit == 80_000

    @pytest.mark.asyncio
    async def test_room_config_overrides_global(self):
        room = SimpleNamespace(coordinator_config={"context_window_limit": 40_000})
        agent = SimpleNamespace(config=None, llm_config=None)
        db = _make_db(room=room, agent=agent)
        policy = await resolve_context_policy(
            role="agent",
            db=db,
            room_id=uuid4(),
            agent_id=uuid4(),
            llm_config=None,
        )
        assert policy.context_window_limit == 40_000

    @pytest.mark.asyncio
    async def test_agent_config_overrides_room(self):
        room = SimpleNamespace(coordinator_config={"context_window_limit": 30_000})
        agent = SimpleNamespace(config={"context_window_limit": 60_000}, llm_config=None)
        db = _make_db(room=room, agent=agent)
        policy = await resolve_context_policy(
            role="agent",
            db=db,
            room_id=uuid4(),
            agent_id=uuid4(),
            llm_config=None,
        )
        assert policy.context_window_limit == 60_000

    @pytest.mark.asyncio
    async def test_short_alias_via_agent_llm_config(self):
        room = SimpleNamespace(coordinator_config=None)
        agent = SimpleNamespace(config=None, llm_config={"context_window": 35_000})
        db = _make_db(room=room, agent=agent)
        policy = await resolve_context_policy(
            role="agent",
            db=db,
            room_id=uuid4(),
            agent_id=uuid4(),
            llm_config=None,
        )
        assert policy.context_window_limit == 35_000


# ---------------------------------------------------------------------------
# TestIncidentScenario (Stage 2)
# ---------------------------------------------------------------------------

class TestIncidentScenario:
    """Regression: room b8a143ec — Designer rejected at 16 630 tokens (prompt_cap)."""

    def test_designer_16630_tokens_no_prompt_cap(self, monkeypatch):
        """With default settings the hard_input_limit must be ≥ 16 630 tokens."""
        monkeypatch.delenv("CONTEXT_AGENT_INPUT_CAP", raising=False)
        monkeypatch.delenv("CONTEXT_WINDOW_LIMIT", raising=False)
        importlib.reload(policy_mod)
        policy = policy_mod.policy_for_role("agent")
        INCIDENT_TOKENS = 16_630
        hard_input_limit = min(policy.input_cap, policy.model_context_limit)
        assert INCIDENT_TOKENS <= hard_input_limit, (
            f"Incident regression: {INCIDENT_TOKENS} tokens triggers prompt_cap "
            f"(hard_input_limit={hard_input_limit}, input_cap={policy.input_cap})"
        )

    def test_cm_review_27546_tokens_no_prompt_cap(self, monkeypatch):
        """Chat Master review at 27 546 tokens (incident session 2472d44d) must not prompt_cap."""
        monkeypatch.delenv("CONTEXT_CHAT_MASTER_INPUT_CAP", raising=False)
        monkeypatch.delenv("CONTEXT_WINDOW_LIMIT", raising=False)
        importlib.reload(policy_mod)
        policy = policy_mod.policy_for_role("chat_master")
        INCIDENT_TOKENS = 27_546
        hard_input_limit = min(policy.input_cap, policy.model_context_limit)
        assert INCIDENT_TOKENS <= hard_input_limit, (
            f"Incident regression: {INCIDENT_TOKENS} tokens triggers prompt_cap for CM "
            f"(hard_input_limit={hard_input_limit}, input_cap={policy.input_cap})"
        )

    def test_admin_narrow_cap_still_rejects_over_limit(self, monkeypatch):
        """Admin can intentionally set a strict input cap to limit context size."""
        monkeypatch.setenv("CONTEXT_AGENT_INPUT_CAP", "10000")
        monkeypatch.delenv("CONTEXT_WINDOW_LIMIT", raising=False)
        importlib.reload(policy_mod)
        policy = policy_mod.policy_for_role("agent")
        assert policy.input_cap == 10_000
        assert min(policy.input_cap, policy.model_context_limit) == 10_000


def test_eviction_mode_defaults_to_on_demand_and_is_overridable(monkeypatch):
    from app.core.context_policy import apply_policy_overrides, policy_for_role

    monkeypatch.delenv("CONTEXT_EVICTION_MODE", raising=False)
    monkeypatch.delenv("CONTEXT_AGENT_EVICTION_MODE", raising=False)
    assert policy_for_role("agent").eviction_mode == "on_demand"

    monkeypatch.setenv("CONTEXT_AGENT_EVICTION_MODE", "eager")
    assert policy_for_role("agent").eviction_mode == "eager"

    resolved = apply_policy_overrides(
        policy_for_role("agent"),
        {"context_eviction_mode": "on_demand"},
        role="agent",
    )
    assert resolved.eviction_mode == "on_demand"

    # Junk configuration keeps the resolved value rather than corrupting it.
    assert (
        apply_policy_overrides(
            resolved, {"context_eviction_mode": "nonsense"}, role="agent"
        ).eviction_mode
        == "on_demand"
    )
