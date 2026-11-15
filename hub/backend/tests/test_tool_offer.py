"""Per-session tool offer and deferred tools (internal design notes S2/S3)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.core import tool_offer as to
from app.core.tool_offer import LOAD_TOOLS, TaskToolHints, ToolOffer, plan_agent_offer
from app.core.tools import REGISTRY

RARE = {"publish_for_deploy", "release_tag", "room_config", "plan_dialogue", "run_subtask"}


def _mcp(name: str, chars: int = 1200) -> dict:
    return {"name": name, "description": "x" * chars, "input_schema": {"type": "object"}}


# ── S2: the decision ─────────────────────────────────────────────────────────


def test_default_agent_defers_only_the_rare_groups():
    offer = plan_agent_offer(registry=REGISTRY, allowed=None, hints=TaskToolHints())
    assert offer is not None
    assert offer.deferred == RARE
    core = {t.name for t in REGISTRY.tools_for("agent")} - offer.deferred
    # The tools the prod frontend agent actually used stay in full.
    assert {"bash", "edit_file", "write_file", "read_file", "screenshot", "workspace_service"} <= core


def test_deploy_group_is_offered_when_the_vps_mcp_is_bound():
    bindings = {"mcp__legion-vps-mcp__vps_exec": SimpleNamespace(market_slug="legion-vps-mcp")}
    hints = TaskToolHints(can_deploy=to.can_deploy(bindings))
    offer = plan_agent_offer(registry=REGISTRY, allowed=None, hints=hints)
    assert "publish_for_deploy" not in offer.deferred
    assert "release_tag" not in offer.deferred
    assert "room_config" in offer.deferred


def test_dialogue_archive_offers_plan_dialogue():
    offer = plan_agent_offer(
        registry=REGISTRY, allowed=None, hints=TaskToolHints(has_dialogue_archive=True)
    )
    assert "plan_dialogue" not in offer.deferred


def test_tools_named_in_agent_settings_are_never_deferred():
    allowed = ["bash", "read_file", "run_subtask", "release_tag"]
    offer = plan_agent_offer(registry=REGISTRY, allowed=allowed, hints=TaskToolHints())
    # Only tools the allowlist lets through can be deferred, and those it
    # names explicitly are the owner's choice: nothing is left to defer.
    assert offer is None


def test_mcp_tools_are_deferred_only_above_the_threshold(monkeypatch):
    monkeypatch.setattr(to, "MCP_DEFER_MIN_CHARS", 3000)
    small = [_mcp("mcp__a__one", 500)]
    big = [_mcp("mcp__a__one"), _mcp("mcp__a__two"), _mcp("mcp__b__three")]
    offer_small = plan_agent_offer(
        registry=REGISTRY, allowed=None, hints=TaskToolHints(), mcp_schemas=small
    )
    offer_big = plan_agent_offer(
        registry=REGISTRY, allowed=None, hints=TaskToolHints(), mcp_schemas=big
    )
    assert "mcp__a__one" not in offer_small.deferred
    assert {"mcp__a__one", "mcp__a__two", "mcp__b__three"} <= offer_big.deferred


def test_kill_switch_offers_everything(monkeypatch):
    monkeypatch.setattr(to, "DEFERRED_TOOLS_ENABLED", False)
    assert plan_agent_offer(registry=REGISTRY, allowed=None, hints=TaskToolHints()) is None


def test_the_decision_is_deterministic():
    a = plan_agent_offer(registry=REGISTRY, allowed=None, hints=TaskToolHints())
    b = plan_agent_offer(registry=REGISTRY, allowed=None, hints=TaskToolHints())
    assert a.deferred == b.deferred
