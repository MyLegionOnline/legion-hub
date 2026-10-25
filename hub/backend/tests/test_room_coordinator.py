"""Unit tests for core/room_coordinator.py — the ChatMaster key triple and the
planner llm_config resolution (internal design notes §8.1)."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.core.room_coordinator import get_cm_agent_id, resolve_planner_llm_config


class _FakeDB:
    def __init__(self, agent):
        self._agent = agent

    async def get(self, model, ident):
        return self._agent


def _room(coordinator_config):
    return SimpleNamespace(coordinator_config=coordinator_config)


CM_ID = uuid.uuid4()


@pytest.mark.parametrize("key", ["agent_id", "chat_master_agent_id", "master_agent_id"])
def test_get_cm_agent_id_reads_all_three_keys(key):
    assert get_cm_agent_id(_room({key: str(CM_ID)})) == CM_ID


def test_get_cm_agent_id_none_when_absent_or_invalid():
    assert get_cm_agent_id(_room({})) is None
    assert get_cm_agent_id(_room({"agent_id": "not-a-uuid"})) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["agent_id", "chat_master_agent_id", "master_agent_id"])
async def test_resolve_planner_uses_chatmaster_config(key):
    agent = SimpleNamespace(llm_config={"provider": "anthropic", "model": "claude-opus-4-8"})
    got_agent, cfg = await resolve_planner_llm_config(
        _room({key: str(CM_ID)}), _FakeDB(agent)
    )
    assert got_agent is agent
    assert cfg["model"] == "claude-opus-4-8"


@pytest.mark.asyncio
async def test_resolve_planner_model_missing_when_chatmaster_has_no_model():
    agent = SimpleNamespace(llm_config={"provider": "anthropic"})
    _a, cfg = await resolve_planner_llm_config(
        _room({"agent_id": str(CM_ID)}), _FakeDB(agent)
    )
    assert not cfg.get("model")


@pytest.mark.asyncio
async def test_resolve_planner_model_override_string_wins():
    agent = SimpleNamespace(llm_config={"provider": "anthropic", "model": "claude-haiku-4-5-20251001"})
    _a, cfg = await resolve_planner_llm_config(
        _room({"agent_id": str(CM_ID), "planner_model": "claude-opus-4-8"}), _FakeDB(agent)
    )
    assert cfg["model"] == "claude-opus-4-8"
    assert cfg["provider"] == "anthropic"  # rest of the CM config preserved


@pytest.mark.asyncio
async def test_resolve_planner_no_chatmaster_returns_empty():
    _a, cfg = await resolve_planner_llm_config(_room({}), _FakeDB(None))
    assert cfg == {}
    assert _a is None
