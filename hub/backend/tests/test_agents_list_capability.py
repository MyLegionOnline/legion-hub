"""Unit tests for Stage 3: capability annotation in Agents List.

No DB required — mocks the execute() result.
Verifies:
  - [image_generation] label rendered when llm_config.media.capability is set
  - No label rendered for text-only agents (no media block)
  - capability="text" (default) never produces a label
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.memory_layer import _agents_list_short, _render_agents_list


def _make_row(agent_id, nickname, short_description, llm_config):
    row = MagicMock()
    row.id = agent_id
    row.nickname = nickname
    row.short_description = short_description
    row.llm_config = llm_config
    return row


def _make_db(rows):
    result = MagicMock()
    result.__iter__ = MagicMock(return_value=iter(rows))
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_image_generation_capability_in_agents_list():
    agent_id = uuid.uuid4()
    room_id = uuid.uuid4()
    row = _make_row(
        agent_id,
        "canvas-designer",
        "Generates images via AI",
        {"provider": "openrouter", "model": "claude-sonnet-5", "media": {"capability": "image_generation"}},
    )
    db = _make_db([row])

    agents = await _agents_list_short(room_id, db)

    assert len(agents) == 1
    assert agents[0]["capability"] == "image_generation"


@pytest.mark.asyncio
async def test_text_agent_has_text_capability():
    agent_id = uuid.uuid4()
    room_id = uuid.uuid4()
    row = _make_row(
        agent_id,
        "coder",
        "Writes code",
        {"provider": "openrouter", "model": "claude-sonnet-5"},
    )
    db = _make_db([row])

    agents = await _agents_list_short(room_id, db)

    assert agents[0]["capability"] == "text"


@pytest.mark.asyncio
async def test_no_llm_config_defaults_to_text():
    agent_id = uuid.uuid4()
    room_id = uuid.uuid4()
    row = _make_row(agent_id, "planner", "Plans stuff", None)
    db = _make_db([row])

    agents = await _agents_list_short(room_id, db)

    assert agents[0]["capability"] == "text"


def test_render_agents_list_shows_capability_label():
    agents = [
        {"id": str(uuid.uuid4()), "nickname": "canvas-designer", "short_description": "Generates images", "capability": "image_generation"},
        {"id": str(uuid.uuid4()), "nickname": "coder", "short_description": "Writes code", "capability": "text"},
    ]
    rendered = _render_agents_list(agents)

    assert "[**image_generation**]" in rendered
    assert "@canvas-designer" in rendered
    # text agents must NOT get a label
    assert "@coder" in rendered
    lines = rendered.splitlines()
    coder_line = next(l for l in lines if "coder" in l)
    assert "[**" not in coder_line


def test_render_agents_list_no_label_for_text_agents():
    agents = [
        {"id": str(uuid.uuid4()), "nickname": "architect", "short_description": "Plans tasks", "capability": "text"},
    ]
    rendered = _render_agents_list(agents)

    assert "[**" not in rendered
    assert "architect" in rendered


def test_render_agents_list_empty():
    rendered = _render_agents_list([])
    assert "*(none)*" in rendered
