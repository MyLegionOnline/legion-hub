"""Agents without any way to look at an image must not attempt visual work."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.tools import ToolContext
from app.core.vision_dispatch import (
    agent_can_see_images,
    has_vision_mcp,
    has_vision_override,
    no_vision_note,
)

_OVERRIDE = {"enabled": True, "provider": "openai", "model": "gpt-vision"}
_VISION_BINDING = SimpleNamespace(market_slug="mcp-vision", tool_name="describe_image")
_OTHER_BINDING = SimpleNamespace(market_slug="mcp-search", tool_name="search")
_OPENROUTER = {"provider": "openrouter", "model": "deepseek/text-only"}


def test_vision_override_needs_enabled_provider_and_model():
    assert has_vision_override({"vision": _OVERRIDE})
    assert not has_vision_override({"vision": {**_OVERRIDE, "enabled": False}})
    assert not has_vision_override({"vision": {**_OVERRIDE, "model": ""}})
    assert not has_vision_override({})
    assert not has_vision_override(None)


def test_vision_mcp_is_recognized_only_by_market_slug_and_tool():
    assert has_vision_mcp({"mcp__mcp-vision__describe_image": _VISION_BINDING})
    assert not has_vision_mcp({"mcp__mcp-search__search": _OTHER_BINDING})
    assert not has_vision_mcp(None)


@pytest.mark.asyncio
async def test_override_or_vision_mcp_means_can_see_without_a_model_lookup():
    with patch(
        "app.core.media_size.chat_model_supports_image_input", new_callable=AsyncMock
    ) as lookup:
        assert await agent_can_see_images({**_OPENROUTER, "vision": _OVERRIDE}, {})
        assert await agent_can_see_images(_OPENROUTER, {"t": _VISION_BINDING})
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_only_openrouter_model_without_vision_tools_is_blind():
    with patch(
        "app.core.media_size.chat_model_supports_image_input",
        new_callable=AsyncMock,
        return_value=False,
    ):
        assert not await agent_can_see_images(_OPENROUTER, {"t": _OTHER_BINDING})


@pytest.mark.asyncio
async def test_unknown_model_capability_stays_optimistic():
    with patch(
        "app.core.media_size.chat_model_supports_image_input",
        new_callable=AsyncMock,
        return_value=None,
    ):
        assert await agent_can_see_images(_OPENROUTER, {})
    # Non-OpenRouter providers are never second-guessed.
    assert await agent_can_see_images({"provider": "anthropic", "model": "x"}, {})
    assert await agent_can_see_images(None, None)


def test_no_vision_note_names_the_task_and_the_block_reason():
    task_id = uuid4()
    note = no_vision_note(task_id)

    assert str(task_id) in note
    assert 'state="blocked"' in note
    assert 'blocked_reason="needs_vision"' in note


@pytest.mark.asyncio
async def test_screenshot_result_tells_a_blind_agent_it_cannot_judge_the_frame():
    from app.core.tools._screenshot import ScreenshotTool

    def _ctx(llm_config):
        return ToolContext(
            room_id=uuid4(),
            session_id=uuid4(),
            task_id=None,
            db=MagicMock(),
            llm_config=llm_config,
            mcp_bindings={},
            mcp_user_id=uuid4(),
        )

    async def _run(llm_config) -> str:
        with (
            patch("app.core.screenshot.capture_screenshot", new_callable=AsyncMock, return_value=b"png"),
            patch("app.core.workspace_fs.fs_write_file", new_callable=AsyncMock, return_value=SimpleNamespace(id=uuid4())),
            patch("app.core.events.emit_event", new_callable=AsyncMock),
            patch(
                "app.core.media_size.chat_model_supports_image_input",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            return await ScreenshotTool().execute(_ctx(llm_config), {"service_slug": "catalog-web"})

    blind = await _run(_OPENROUTER)
    assert blind.startswith("Screenshot saved to /project/screenshots/catalog-web-")
    assert "cannot view images" in blind
    assert 'blocked_reason="needs_vision"' in blind

    sighted = await _run({**_OPENROUTER, "vision": _OVERRIDE})
    assert sighted.endswith(".png")
