"""Stage 7 — Regression / E2E tests for the reference-image fix pipeline.

Covers the full path from a raw Gateway response (MCP CallToolResult envelope)
through execute_mcp_tool to the value the agent sees, in both the happy path
and the negative (isError) path.  Each test is named after the regression it
guards against — the comment at the top of each test describes the old
behaviour that the fix eliminated.

Tests are pure unit tests: _gateway_post is patched, no DB or network needed.
"""
from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_gateway

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PNG_BYTES = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"rest"
PNG_B64 = base64.b64encode(PNG_BYTES).decode()

VALID_SPEC = {
    "version": 1,
    "layout": [
        {"id": "n1", "role": "header"},
        {"id": "n2", "role": "card"},
        {"id": "n3", "role": "card"},
    ],
    "tokens": {
        "colors": [
            {"hex": "#3B82F6", "source": "matched"},
            {"hex": "#ffffff", "source": "new"},
        ]
    },
    "uncertain": ["hover state"],
}

DESCRIBE_RESULT = {
    "description": (
        "A 3-column card grid with a blue (#3B82F6) header banner. "
        "Aspect ratio 16:9. Each card shows a thumbnail on the left "
        "and two lines of text on the right."
    )
}

IS_ERROR_ENVELOPE = {
    "isError": True,
    "content": [{"type": "text", "text": "Vision provider is unavailable"}],
}


def _non_billable_binding(tool_name: str, slug: str = "mcp-vision") -> mcp_gateway.McpToolBinding:
    return mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=None,  # non-billable → skips all credit logic
        market_slug=slug,
        tool_name=tool_name,
        custom_config=None,
        isolation="stateless",
    )


def _billable_binding(tool_name: str, slug: str = "mcp-vision") -> mcp_gateway.McpToolBinding:
    return mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=uuid4(),
        market_slug=slug,
        tool_name=tool_name,
        custom_config=None,
        isolation="stateless",
    )


async def _fake_read(room_id, path, db):
    return f"data:image/png;base64,{PNG_B64}"


# ===========================================================================
# 1. Happy path — describe_image
# ===========================================================================


@pytest.mark.asyncio
async def test_describe_image_happy_path_returns_description_not_envelope(monkeypatch):
    """Regression: before Stage 1 the raw envelope was serialised and returned to
    the agent ({"content": [...], "isError": ...}) instead of the description text.
    Now the agent receives the description string."""
    envelope = {"content": [{"type": "text", "text": json.dumps(DESCRIBE_RESULT)}]}

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", _fake_read)
    with patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)):
        result = await mcp_gateway.execute_mcp_tool(
            _non_billable_binding("describe_image"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png"},
            db=object(),
        )

    parsed = json.loads(result)
    assert "description" in parsed, "agent must see the description, not an envelope"
    assert "3-column card grid" in parsed["description"]
    assert "content" not in parsed, "raw envelope key must not leak to the agent"
    assert "isError" not in parsed


@pytest.mark.asyncio
async def test_describe_image_happy_path_does_not_write_spec_file(monkeypatch):
    """describe_image never produces a design spec — no .spec.json must be written."""
    envelope = {"content": [{"type": "text", "text": json.dumps(DESCRIBE_RESULT)}]}
    write = AsyncMock()

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", _fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)),
        patch("app.core.workspace_fs.fs_write_file", write),
    ):
        await mcp_gateway.execute_mcp_tool(
            _non_billable_binding("describe_image"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png"},
            db=object(),
        )

    # No file of any kind should be written for a plain describe_image result.
    write.assert_not_awaited()


# ===========================================================================
# 2. Happy path — extract_design_spec
# ===========================================================================


@pytest.mark.asyncio
async def test_extract_design_spec_happy_path_writes_spec_and_returns_summary(monkeypatch):
    """Regression: before Stage 1 the envelope was passed to persist_design_spec,
    which wrote {"content": [...]} to design/ref.spec.json with nodes=0.
    Now the unwrapped spec is written and the summary is correct."""
    envelope = {"content": [{"type": "text", "text": json.dumps(VALID_SPEC)}]}
    write = AsyncMock(return_value=type("F", (), {"id": uuid4()})())
    emit = AsyncMock()

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", _fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)),
        patch("app.core.workspace_fs.fs_write_file", write),
        patch("app.core.events.emit_event", emit),
        patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={})),
    ):
        result = await mcp_gateway.execute_mcp_tool(
            _non_billable_binding("extract_design_spec"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png", "name": "ref"},
            db=object(),
        )

    parsed = json.loads(result)
    assert parsed["spec_path"] == "design/ref.spec.json"
    summary = parsed["summary"]
    assert summary["nodes"] == 3, "must count all 3 layout nodes, not 0"
    assert set(summary["roles"]) == {"header", "card"}
    assert summary["tokens_matched"] == 1
    assert summary["tokens_new"] == 1
    assert "uncertain" in summary

    # Spec file was written exactly once
    write.assert_awaited_once()
    written_path = write.await_args.args[1]
    assert written_path == "design/ref.spec.json"

    # FILE_CHANGED event was emitted
    emit.assert_awaited_once()
    assert emit.await_args.kwargs["event_type"] == "FILE_CHANGED"


@pytest.mark.asyncio
async def test_extract_design_spec_written_file_contains_real_spec_not_envelope(monkeypatch):
    """Regression: the junk file contained the transport envelope, not the spec.
    Now the written content must be the spec payload."""
    envelope = {"content": [{"type": "text", "text": json.dumps(VALID_SPEC)}]}
    written_content: list[str] = []

    async def capture_write(room_id, path, content, db, **kw):
        written_content.append(content)
        return type("F", (), {"id": uuid4()})()

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", _fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)),
        patch("app.core.workspace_fs.fs_write_file", capture_write),
        patch("app.core.events.emit_event", AsyncMock()),
        patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={})),
    ):
        await mcp_gateway.execute_mcp_tool(
            _non_billable_binding("extract_design_spec"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png", "name": "ref"},
            db=object(),
        )

    assert written_content, "spec file must have been written"
    saved = json.loads(written_content[0])
    assert saved.get("version") == 1, "written file must be the real spec, not the envelope"
    assert isinstance(saved.get("layout"), list)
    assert "content" not in saved, "envelope key must not appear in the written spec"


# ===========================================================================
# 3. Negative path — isError on describe_image
# ===========================================================================


@pytest.mark.asyncio
async def test_is_error_describe_image_returns_error_string(monkeypatch):
    """isError envelope → agent receives 'Error: ...' not a JSON envelope dict."""
    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", _fake_read)
    with patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=IS_ERROR_ENVELOPE)):
        result = await mcp_gateway.execute_mcp_tool(
            _non_billable_binding("describe_image"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png"},
            db=object(),
        )

    assert result == "Error: [provider_error] Vision provider is unavailable"


@pytest.mark.asyncio
async def test_is_error_describe_image_does_not_write_any_file(monkeypatch):
    """Regression: before Stage 2 the error envelope was written to disk as a
    .spec.json (even for describe_image which has no spec step). Now nothing
    is written."""
    write = AsyncMock()
    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", _fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=IS_ERROR_ENVELOPE)),
        patch("app.core.workspace_fs.fs_write_file", write),
    ):
        await mcp_gateway.execute_mcp_tool(
            _non_billable_binding("describe_image"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png"},
            db=object(),
        )

    write.assert_not_awaited()


# ===========================================================================
# 4. Negative path — isError on extract_design_spec
# ===========================================================================


@pytest.mark.asyncio
async def test_is_error_extract_design_spec_does_not_write_junk_spec_file(monkeypatch):
    """Regression: before Stages 1+2 an isError envelope was serialised into
    design/<name>.spec.json, committed to the room repo, and announced as a
    FILE_CHANGED event.  Now nothing is written."""
    write = AsyncMock()
    emit = AsyncMock()
    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", _fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=IS_ERROR_ENVELOPE)),
        patch("app.core.workspace_fs.fs_write_file", write),
        patch("app.core.events.emit_event", emit),
        patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={})),
    ):
        result = await mcp_gateway.execute_mcp_tool(
            _non_billable_binding("extract_design_spec"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png", "name": "ref"},
            db=object(),
        )

    assert result == "Error: [provider_error] Vision provider is unavailable"
    write.assert_not_awaited()
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_is_error_on_billed_extract_design_spec_refunds_not_settles(monkeypatch):
    """Regression: before Stage 3 credits were settled on a 200 with isError —
    every cold-restart loop burned 900 mcr for nothing."""
    reserve = AsyncMock()
    settle = AsyncMock()
    refund = AsyncMock()

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", _fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=IS_ERROR_ENVELOPE)),
        patch("app.core.mcp_credits.resolve_price_mcr", AsyncMock(return_value=900)),
        patch("app.core.mcp_credits.reserve_call", reserve),
        patch("app.core.mcp_credits.settle_call", settle),
        patch("app.core.mcp_credits.refund_call", refund),
        patch("app.core.metrics.record_mcp_credits_refund"),
        patch("app.core.metrics.record_mcp_credits_spent"),
        patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={})),
    ):
        result = await mcp_gateway.execute_mcp_tool(
            _billable_binding("extract_design_spec"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png", "name": "ref"},
            db=object(),
        )

    assert result == "Error: [provider_error] Vision provider is unavailable"
    reserve.assert_awaited_once()
    refund.assert_awaited_once()
    assert refund.await_args.kwargs.get("note") == "tool returned isError"
    settle.assert_not_awaited()


# ===========================================================================
# 5. No-regression — non-vision MCP tools are unaffected
# ===========================================================================


@pytest.mark.asyncio
async def test_non_vision_tool_bare_dict_unaffected_by_unwrap():
    """A non-envelope dict from mcp-searxng (or any HTTP-MCP) passes through
    _unwrap_mcp_tool_result unchanged — no spurious content/isError parsing."""
    searxng = mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=None,
        market_slug="mcp-searxng",
        tool_name="search",
        custom_config=None,
        isolation="stateless",
    )
    search_result = {"results": [{"title": "Foo", "url": "https://example.com", "content": "bar"}]}
    with patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=search_result)):
        result_text = await mcp_gateway.execute_mcp_tool(
            searxng, uuid4(), uuid4(), {"query": "css grid layout"}, db=object(),
        )

    parsed = json.loads(result_text)
    # The results list must be intact — bare dict pass-through preserved it.
    assert parsed["results"][0]["title"] == "Foo"


@pytest.mark.asyncio
async def test_non_vision_tool_envelope_still_unwraps_correctly():
    """If a non-vision Market MCP starts returning a CallToolResult envelope
    (a common MCP spec pattern), the generic unwrap still works — this is the
    'latent double-wrapping' fix from Stage 1 that benefits every tool."""
    non_vision = mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=None,
        market_slug="mcp-host",
        tool_name="list_files",
        custom_config=None,
        isolation="stateless",
    )
    payload = {"files": ["/app/main.py", "/app/models.py"]}
    envelope = {"content": [{"type": "text", "text": json.dumps(payload)}]}
    with patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)):
        result_text = await mcp_gateway.execute_mcp_tool(
            non_vision, uuid4(), uuid4(), {}, db=object(),
        )

    parsed = json.loads(result_text)
    assert parsed["files"] == ["/app/main.py", "/app/models.py"]
    assert "content" not in parsed


@pytest.mark.asyncio
async def test_non_vision_is_error_envelope_returns_error_string_not_written():
    """isError from any Market MCP → error string; no files, no crash."""
    non_vision = mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=None,
        market_slug="mcp-host",
        tool_name="run_command",
        custom_config=None,
        isolation="stateless",
    )
    error_env = {"isError": True, "content": [{"type": "text", "text": "command failed"}]}
    with patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=error_env)):
        result = await mcp_gateway.execute_mcp_tool(
            non_vision, uuid4(), uuid4(), {"cmd": "make test"}, db=object(),
        )

    assert result == "Error: command failed"
