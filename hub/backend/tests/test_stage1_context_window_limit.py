"""
Stage 1 tests: context_window_limit in Agent schema (llm_config passthrough).

No DB required — pure pydantic / schema checks.
Verifies that AgentCreate / AgentUpdate / AgentOut pass context_window_limit
through llm_config unchanged, and that an absent key does not inject a default
(the default 100_000 lives in ContextPolicy, not in the schema layer).
"""
from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.schemas.agent import AgentCreate, AgentOut, AgentUpdate


class TestAgentCreateContextWindowLimit:
    def test_preserves_explicit_value(self):
        data = AgentCreate(
            nickname="test-agent",
            llm_config={"context_window_limit": 50_000, "model": "claude-sonnet-5"},
        )
        assert data.llm_config["context_window_limit"] == 50_000

    def test_absent_key_not_injected(self):
        data = AgentCreate(
            nickname="test-no-limit",
            llm_config={"model": "claude-sonnet-5"},
        )
        assert "context_window_limit" not in data.llm_config

    def test_accepts_large_value(self):
        data = AgentCreate(
            nickname="large-limit",
            llm_config={"context_window_limit": 200_000},
        )
        assert data.llm_config["context_window_limit"] == 200_000

    def test_accepts_minimum_value(self):
        data = AgentCreate(
            nickname="small-limit",
            llm_config={"context_window_limit": 1024},
        )
        assert data.llm_config["context_window_limit"] == 1024

    def test_other_llm_config_fields_unaffected(self):
        data = AgentCreate(
            nickname="full-config",
            llm_config={
                "model": "gpt-4o",
                "context_length": 128_000,
                "input_price": 5.0,
                "context_window_limit": 60_000,
            },
        )
        assert data.llm_config["context_window_limit"] == 60_000
        assert data.llm_config["context_length"] == 128_000
        assert data.llm_config["model"] == "gpt-4o"

    def test_rejects_window_limit_below_minimum(self):
        with pytest.raises(ValidationError, match="at least 1024"):
            AgentCreate(nickname="test", llm_config={"context_window_limit": 1023})

    def test_rejects_window_limit_above_known_model_limit(self):
        with pytest.raises(ValidationError, match="cannot exceed the model_context_limit"):
            AgentCreate(
                nickname="test",
                llm_config={"context_window_limit": 32_000, "context_length": 16_000},
            )


class TestAgentUpdateContextWindowLimit:
    def test_update_preserves_value(self):
        update = AgentUpdate(llm_config={"context_window_limit": 75_000})
        assert update.llm_config["context_window_limit"] == 75_000

    def test_update_absent_key_not_injected(self):
        update = AgentUpdate(llm_config={"model": "claude-sonnet-5"})
        assert "context_window_limit" not in update.llm_config

    def test_none_llm_config_stays_none(self):
        update = AgentUpdate(llm_config=None)
        assert update.llm_config is None

    def test_update_validates_explicit_model_context_limit(self):
        with pytest.raises(ValidationError, match="cannot exceed the model_context_limit"):
            AgentUpdate(
                llm_config={"context_window_limit": 64_000, "model_context_limit": 32_000}
            )


class TestAgentOutContextWindowLimit:
    def _make_out(self, llm_config: dict) -> AgentOut:
        return AgentOut(
            id=uuid.uuid4(),
            owner_id=uuid.uuid4(),
            nickname="test",
            agent_uid="test-abc123",
            agent_type="internal",
            status="offline",
            version="0.1.0",
            update_policy="manual",
            short_description="",
            config={},
            llm_config=llm_config,
            tools_allowed=None,
            token="tok_test",
        )

    def test_roundtrip_preserves_value(self):
        out = self._make_out({"context_window_limit": 80_000, "model": "claude-sonnet-5"})
        assert out.llm_config["context_window_limit"] == 80_000

    def test_absent_key_not_injected_in_response(self):
        out = self._make_out({"model": "claude-sonnet-5"})
        assert "context_window_limit" not in out.llm_config

    def test_serialization_preserves_value(self):
        out = self._make_out({"context_window_limit": 50_000})
        dumped = out.model_dump()
        assert dumped["llm_config"]["context_window_limit"] == 50_000
