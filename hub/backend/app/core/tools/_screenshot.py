"""Agent screenshot tool for declared workspace services."""
from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext

logger = logging.getLogger(__name__)


class ScreenshotTool(Tool):
    name = "screenshot"
    description = (
        "Capture a PNG screenshot of a ready declared web/API workspace service "
        "through the Browser service-origin gateway and save it in Repository. "
        "Call workspace_service declare, start, and wait_ready first."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "service_slug": {
                "type": "string",
                "description": "Slug returned by workspace_service declare/list (not an internal port)",
            },
            "path": {
                "type": "string",
                "description": "Absolute path to render on the service (default: its declared default_path)",
            },
            "width": {
                "type": "integer",
                "description": "Viewport width in px, 320-3840 (default 1280). Use 375 to check a phone layout.",
            },
            "height": {
                "type": "integer",
                "description": "Viewport height in px, 320-3840 (default 720).",
            },
        },
        "required": ["service_slug"],
        "additionalProperties": False,
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = True

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core.events import emit_event
        from app.core.screenshot import ScreenshotError, capture_screenshot
        from app.core.workspace_fs import fs_write_file

        service_slug = str(tool_input.get("service_slug") or "").strip()
        if not service_slug:
            return "Error: service_slug is required"
        if ctx.mcp_user_id is None:
            return "Error: screenshot requires an agent identity"

        try:
            png = await capture_screenshot(
                ctx.room_id,
                service_slug,
                ctx.db,
                user_id=ctx.mcp_user_id,
                path=tool_input.get("path"),
                width=tool_input.get("width"),
                height=tool_input.get("height"),
            )
        except ScreenshotError as exc:
            return f"Error: screenshot failed — {exc}"
        except Exception as exc:
            logger.warning(
                "screenshot tool failed room=%s service=%s (%s)",
                ctx.room_id,
                service_slug,
                type(exc).__name__,
            )
            return "Error: screenshot failed — the preview could not be captured"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = f"/project/screenshots/{service_slug}-{timestamp}.png"
        data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        try:
            await fs_write_file(
                ctx.room_id, output_path, data_uri, ctx.db, agent_id=ctx.actor_agent_id
            )
            await emit_event(
                room_id=ctx.room_id,
                event_type="FILE_CHANGED",
                content=f"Screenshot saved: {output_path}",
                db=ctx.db,
                metadata={
                    "path": output_path,
                    "source": "screenshot",
                },
            )
        except Exception as exc:
            logger.warning(
                "screenshot tool could not save room=%s service=%s (%s)",
                ctx.room_id,
                service_slug,
                type(exc).__name__,
            )
            return "Error: screenshot was captured but could not be saved to Repository"

        from app.core.vision_dispatch import agent_can_see_images

        if not await agent_can_see_images(ctx.llm_config, ctx.mcp_bindings):
            return (
                f"Screenshot saved to {output_path}. You cannot view images (no vision override, "
                "no mcp-vision tool, model without image input), so you cannot judge this frame. "
                "Do not report a visual verdict from it; if the task needs one, "
                'call task_update(state="blocked", blocked_reason="needs_vision").'
            )
        return f"Screenshot saved to {output_path}"


REGISTRY.register(ScreenshotTool())
