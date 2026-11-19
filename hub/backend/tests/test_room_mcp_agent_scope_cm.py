"""Stage E4 of internal design notes — ChatMaster
knows which agent has which room MCP server and who can view images.

No DB: room_mcp_briefs is patched, the rest is pure functions.
"""

from __future__ import annotations

import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import media_size
from app.core.mcp_gateway import RoomMcpBrief, assignment_capability_warnings
from app.core.memory_layer import _attach_agents_mcp, _render_agents_list
from app.core.vision_dispatch import agent_vision_hint

A = uuid.uuid4()
B = uuid.uuid4()


def _brief(name: str, *, slug: str | None = None, mode: str = "all", agents: tuple = ()) -> RoomMcpBrief:
    return RoomMcpBrief(uuid.uuid4(), name, slug, mode, frozenset(agents))


BRIEFS = [
    _brief("Web search (SearXNG)"),
    _brief("cloudflare full", mode="selected", agents=(A,)),
    _brief("Image understanding", slug="mcp-vision", mode="selected", agents=(A,)),
    _brief("Postgres toolbox", mode="selected"),
]


# ── agent_vision_hint ─────────────────────────────────────────────────────

def test_vision_hint_override_or_server_is_yes():
    override = {"vision": {"enabled": True, "provider": "openrouter", "model": "x/vl"}}
    assert agent_vision_hint(override, has_vision_server=False) == "yes"
    assert agent_vision_hint({"provider": "deepseek"}, has_vision_server=True) == "yes"


def test_vision_hint_unknown_without_catalog_and_for_other_providers():
    assert agent_vision_hint({"provider": "anthropic", "model": "m"}, has_vision_server=False) == "unknown"
    assert agent_vision_hint(None, has_vision_server=False) == "unknown"
    assert agent_vision_hint({"provider": "openrouter", "model": "never/seen"}, has_vision_server=False) == "unknown"


def test_vision_hint_reads_the_cached_catalog_answer():
    key = ("https://openrouter.ai/api/v1", "blind/model")
    media_size._chat_vision_cache[key] = (time.monotonic(), False)
    try:
        assert agent_vision_hint({"provider": "openrouter", "model": "blind/model"}, has_vision_server=False) == "no"
        media_size._chat_vision_cache[key] = (time.monotonic(), True)
        assert agent_vision_hint({"provider": "openrouter", "model": "blind/model"}, has_vision_server=False) == "yes"
        media_size._chat_vision_cache[key] = (time.monotonic() - 10**7, False)  # expired
        assert agent_vision_hint({"provider": "openrouter", "model": "blind/model"}, has_vision_server=False) == "unknown"
    finally:
        media_size._chat_vision_cache.pop(key, None)


# ── memory layer: agents list ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_attach_agents_mcp_names_per_agent_and_unassigned():
    agents = [
        {"id": str(A), "nickname": "seer", "short_description": "", "sees_images": "unknown"},
        {"id": str(B), "nickname": "coder", "short_description": "", "sees_images": "no"},
    ]
    with patch("app.core.mcp_gateway.room_mcp_briefs", AsyncMock(return_value=BRIEFS)):
        unassigned = await _attach_agents_mcp(uuid.uuid4(), agents, MagicMock())
    assert agents[0]["mcp"] == ["Web search (SearXNG)", "cloudflare full", "Image understanding"]
    assert agents[0]["sees_images"] == "yes"  # the vision server is in its scope
    assert agents[1]["mcp"] == ["Web search (SearXNG)"]
    assert agents[1]["sees_images"] == "no"
    assert unassigned == ["Postgres toolbox"]


@pytest.mark.asyncio
async def test_attach_agents_mcp_failure_leaves_list_alone():
    agents = [{"id": str(A), "nickname": "seer", "short_description": ""}]
    with patch("app.core.mcp_gateway.room_mcp_briefs", AsyncMock(side_effect=RuntimeError("db"))):
        assert await _attach_agents_mcp(uuid.uuid4(), agents, MagicMock()) == []
    assert "mcp" not in agents[0]


def test_render_agents_list_shows_names_vision_and_rule_but_no_schemas():
    agents = [
        {"id": str(A), "nickname": "seer", "short_description": "Designer", "capability": "text",
         "mcp": ["cloudflare full", "Image understanding"], "sees_images": "yes"},
        {"id": str(B), "nickname": "coder", "short_description": "Codes", "capability": "text",
         "mcp": [], "sees_images": "unknown"},
    ]
    out = _render_agents_list(agents, ["Postgres toolbox"])
    assert "MCP: cloudflare full, Image understanding · sees images: yes" in out
    assert "MCP: none" in out
    assert "sees images: unknown" not in out  # unknown costs tokens and says nothing
    assert "MCP connected but assigned to no agent: Postgres toolbox" in out
    assert "sees images: no" in out  # the rule line names it
    assert "input_schema" not in out and "mcp__" not in out


def test_render_agents_list_unchanged_without_mcp_data():
    agents = [{"id": str(A), "nickname": "coder", "short_description": "Codes", "capability": "text"}]
    out = _render_agents_list(agents)
    assert out == f"## Agents in Room\n\n- **@coder** (`{A}`): Codes"


# ── assignment warnings ───────────────────────────────────────────────────

def test_warning_for_server_outside_assignee_scope():
    w = assignment_capability_warnings(
        task_text="Purge the cache via Cloudflare", nickname="coder", agent_id=B,
        vision_hint="unknown", briefs=BRIEFS,
    )
    assert len(w) == 1 and "'cloudflare full'" in w[0] and "@coder" in w[0]


def test_no_warning_when_server_in_scope_or_shared():
    assert assignment_capability_warnings(
        task_text="Purge the cache via Cloudflare", nickname="seer", agent_id=A,
        vision_hint="yes", briefs=BRIEFS,
    ) == []
    assert assignment_capability_warnings(
        task_text="searxng lookup", nickname="coder", agent_id=B,
        vision_hint="unknown", briefs=BRIEFS,
    ) == []


def test_vision_warning_only_for_a_blind_assignee():
    text = "Compare /shots/home.png with the mockup"
    assert assignment_capability_warnings(
        task_text=text, nickname="coder", agent_id=B, vision_hint="no", briefs=[],
    )
    assert assignment_capability_warnings(
        task_text=text, nickname="coder", agent_id=B, vision_hint="unknown", briefs=[],
    ) == []
    assert assignment_capability_warnings(
        task_text="Write the README", nickname="coder", agent_id=B, vision_hint="no", briefs=[],
    ) == []


# ── task_create tool ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_create_helper_returns_warnings_for_assignee():
    from app.core.tools import _task_create

    agent = SimpleNamespace(id=B, nickname="coder", llm_config={"provider": "deepseek"})
    db = MagicMock()
    db.get = AsyncMock(return_value=agent)
    ctx = SimpleNamespace(db=db, room_id=uuid.uuid4())
    task = SimpleNamespace(id=uuid.uuid4(), assigned_agent_id=B, title="Deploy", description="via cloudflare")
    with patch("app.core.mcp_gateway.room_mcp_briefs", AsyncMock(return_value=BRIEFS)):
        warnings = await _task_create._assignee_capability_warnings(ctx, task)
    assert len(warnings) == 1 and "cloudflare full" in warnings[0]

    task.assigned_agent_id = None
    assert await _task_create._assignee_capability_warnings(ctx, task) == []


# ── plan_tasks prompt ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_tasks_prompt_has_agent_capabilities_and_rule():
    from app.core import chat_master

    room = SimpleNamespace(id=uuid.uuid4(), coordinator_config={})
    agents = [
        SimpleNamespace(id=A, nickname="seer", short_description="Designer", llm_config={"provider": "anthropic"}),
        SimpleNamespace(id=B, nickname="coder", short_description=None, llm_config={"provider": "anthropic"}),
    ]
    captured: dict = {}

    async def fake_completion(config, messages, schema, secret_value=None):
        captured["system"] = messages[0]["content"]
        return SimpleNamespace(text=json.dumps({"needs_tasks": False, "reply": "hi", "confirmation_message": None, "tasks": []}))

    with patch.object(chat_master, "get_project_memory", AsyncMock(return_value=None)), \
         patch.object(chat_master, "get_repo_snapshot", AsyncMock(return_value=[])), \
         patch.object(chat_master, "release_db", AsyncMock()), \
         patch.object(chat_master, "get_llm_json_completion", fake_completion), \
         patch.object(chat_master, "_save_master_session", AsyncMock()), \
         patch("app.core.mcp_gateway.room_mcp_briefs", AsyncMock(return_value=BRIEFS)):
        await chat_master.plan_tasks(room, "hello", agents, {}, MagicMock())

    system = captured["system"]
    assert "- nickname: seer — Designer · MCP: Web search (SearXNG), cloudflare full, Image understanding · sees images: yes" in system
    assert "- nickname: coder · MCP: Web search (SearXNG)" in system
    assert 'never to an agent with "sees images: no"' in system


# ── cold catalog warm-up (live run 7.4 finding) ───────────────────────────

@pytest.mark.asyncio
async def test_refresh_vision_hints_turns_unknown_into_no_from_catalog():
    from app.core.memory_layer import _refresh_vision_hints

    key = ("https://openrouter.ai/api/v1", "text/only-model")
    media_size._chat_vision_cache.pop(key, None)

    async def fake_fetch(base_url, model):
        media_size._chat_vision_cache[(base_url, model)] = (time.monotonic(), False)
        return False

    agents = [
        {"id": str(A), "sees_images": "unknown", "_vision_key": key},
        {"id": str(B), "sees_images": "yes", "_vision_key": None},
    ]
    try:
        with patch.object(media_size, "chat_model_supports_image_input", fake_fetch):
            await _refresh_vision_hints(agents)
    finally:
        media_size._chat_vision_cache.pop(key, None)
    assert agents[0]["sees_images"] == "no"
    assert agents[1]["sees_images"] == "yes"
    assert all("_vision_key" not in a for a in agents)


@pytest.mark.asyncio
async def test_warm_vision_catalog_is_bounded_and_quiet():
    import asyncio

    from app.core.vision_dispatch import warm_vision_catalog

    async def slow(base_url, model):
        await asyncio.sleep(5)

    with patch.object(media_size, "chat_model_supports_image_input", slow):
        started = time.monotonic()
        await warm_vision_catalog([("https://x.test", "never/cached")], timeout=0.1)
    assert time.monotonic() - started < 1
