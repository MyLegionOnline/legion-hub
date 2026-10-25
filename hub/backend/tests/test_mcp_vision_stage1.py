from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_gateway


def _binding() -> mcp_gateway.McpToolBinding:
    return mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=uuid4(),
        market_slug="mcp-vision",
        tool_name="describe_image",
        custom_config=None,
        isolation="stateless",
    )


def test_vision_schema_accepts_workspace_path_without_requiring_base64():
    schema = mcp_gateway._vision_agent_schema({
        "type": "object",
        "properties": {"image_base64": {"type": "string"}},
        "required": ["image_base64"],
    })
    assert "image_path" in schema["properties"]
    assert "image_base64" not in schema["required"]
    assert {"required": ["image_path"]} in schema["anyOf"]


def test_vision_workspace_path_is_resolved_before_forward(monkeypatch):
    png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    encoded = base64.b64encode(png).decode()

    async def fake_read(room_id, path, db):
        assert path == "screenshots/app.png"
        return f"data:image/png;base64,{encoded}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    result = asyncio.run(mcp_gateway._resolve_vision_arguments(
        _binding(),
        {"image_path": "screenshots/app.png", "question": "What is visible?"},
        uuid4(),
        object(),
    ))
    assert result == {
        "image_base64": encoded,
        "mime_type": "image/png",
        "question": "What is visible?",
    }


def test_vision_corrects_mislabeled_extension_instead_of_rejecting(monkeypatch):
    """A JPEG saved as ``.png`` must be forwarded as image/jpeg, not rejected.

    Regression: ``_validate_forward_image`` used to raise "invalid image data"
    whenever the declared MIME (derived from the file extension by workspaced)
    disagreed with the magic bytes, sending the agent on a fruitless retry
    loop over a perfectly good mockup.
    """
    jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46])
    encoded = base64.b64encode(jpeg).decode()

    async def fake_read(room_id, path, db):
        # workspaced still labels by extension → declares png for jpeg bytes
        return f"data:image/png;base64,{encoded}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    result = asyncio.run(mcp_gateway._resolve_vision_arguments(
        _binding(), {"image_path": "project/images/mockup.png"}, uuid4(), object()
    ))
    assert result == {"image_base64": encoded, "mime_type": "image/jpeg"}


def test_vision_rejects_truly_unrecognized_bytes_with_actionable_message(monkeypatch):
    junk = base64.b64encode(b"not an image at all, just bytes").decode()

    async def fake_read(room_id, path, db):
        return f"data:image/png;base64,{junk}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with pytest.raises(mcp_gateway.McpVisionInputError) as exc:
        asyncio.run(mcp_gateway._resolve_vision_arguments(
            _binding(), {"image_path": "a.png"}, uuid4(), object()
        ))
    assert "extension" in str(exc.value).lower()


def test_vision_resolution_rejects_non_image_without_echoing_path(monkeypatch):
    async def fake_read(room_id, path, db):
        return "plain text"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with pytest.raises(mcp_gateway.McpVisionInputError) as exc:
        asyncio.run(mcp_gateway._resolve_vision_arguments(
            _binding(), {"image_path": "private/name.txt"}, uuid4(), object()
        ))
    assert "private/name.txt" not in str(exc.value)


@pytest.mark.asyncio
async def test_standard_vision_path_is_resolved_and_forwarded_to_gateway(monkeypatch):
    """Marketplace vision has no auth/fingerprint flags, but still needs db."""

    png = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
    encoded = base64.b64encode(png).decode()
    room_id, user_id, request_id = uuid4(), uuid4(), uuid4()
    db = object()

    async def fake_read(read_room_id, path, read_db):
        assert read_room_id == room_id
        assert path == "1111.png"
        assert read_db is db
        return f"data:image/png;base64,{encoded}"

    gateway_post = AsyncMock(return_value={"description": "a boot", "ocr_text": ""})
    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with (
        patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=0),
        patch("app.core.mcp_gateway._gateway_post", gateway_post),
    ):
        result = await mcp_gateway.execute_mcp_tool(
            _binding(), user_id, room_id, {"image_path": "1111.png"}, db=db, request_id=request_id
        )

    assert json.loads(result) == {"description": "a boot", "ocr_text": ""}
    payload = gateway_post.await_args.args[3]
    assert payload["arguments"] == {"image_base64": encoded, "mime_type": "image/png"}
    assert "image_path" not in payload["arguments"]
    assert gateway_post.await_args.kwargs["timeout"] == mcp_gateway.MCP_VISION_GATEWAY_TIMEOUT
