from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import mcp_gateway, vision_artifacts


def _binding(tool_name: str) -> mcp_gateway.McpToolBinding:
    return mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=uuid4(),
        market_slug="mcp-vision",
        tool_name=tool_name,
        custom_config=None,
        isolation="stateless",
    )


PNG_BYTES = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) + b"rest"
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


# ---------------------------------------------------------------------------
# 2.1 / 2.2 — registry-driven argument resolution
# ---------------------------------------------------------------------------


def test_cv_path_is_resolved_into_cv_field_and_removed_from_arguments(monkeypatch):
    room_id = uuid4()
    cv_tree = {"regions": [{"id": "r1", "box": {"x": 0, "y": 0, "w": 10, "h": 10}}]}

    async def fake_read(read_room_id, path, db):
        assert read_room_id == room_id
        if path == "images/dashboard.png":
            return f"data:image/png;base64,{PNG_B64}"
        assert path == "design/dashboard.cv.json"
        return json.dumps(cv_tree)

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    result = asyncio.run(mcp_gateway._resolve_vision_arguments(
        _binding("extract_design_spec"),
        {"image_path": "images/dashboard.png", "cv_path": "design/dashboard.cv.json", "name": "dashboard"},
        room_id,
        object(),
    ))
    assert "cv_path" not in result
    assert result["cv"] == cv_tree
    # The raw tree text must never appear as a literal string anywhere in
    # the resolved arguments — only under the structured `cv` key.
    assert "regions" not in json.dumps({k: v for k, v in result.items() if k != "cv"})


@pytest.mark.parametrize("content,expected", [
    (None, "Error: [not_found] image file was not found at 'images/wrong.png'"),
    ("not a data uri", "Error: [not_an_image] workspace file is not a supported image (PNG/JPEG/WebP/GIF)"),
])
@pytest.mark.asyncio
async def test_execute_mcp_tool_vision_path_errors_carry_a_code(monkeypatch, content, expected):
    """The Frontend developer case: a wrong image_path must say [not_found] with the
    path echoed back, not collapse to an opaque failure."""

    async def fake_read(room_id, path, db):
        return content

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    reserve = AsyncMock()
    with patch("app.core.mcp_credits.reserve_call", reserve):
        result = await mcp_gateway.execute_mcp_tool(
            _binding("describe_image"), uuid4(), uuid4(),
            {"image_path": "images/wrong.png"},
            db=object(),
        )
    assert result.startswith(expected)
    reserve.assert_not_awaited()  # resolution fails before any credit is held


def test_critique_design_resolves_image_and_diff_together(monkeypatch):
    room_id = uuid4()
    diff_tree = {"diff_ratio": 0.03}

    async def fake_read(read_room_id, path, db):
        if path == "screenshots/app.png":
            return f"data:image/png;base64,{PNG_B64}"
        assert path == "design/app.diff.json"
        return json.dumps(diff_tree)

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    result = asyncio.run(mcp_gateway._resolve_vision_arguments(
        _binding("critique_design"),
        {"image_path": "screenshots/app.png", "diff_path": "design/app.diff.json"},
        room_id,
        object(),
    ))
    assert result["image_base64"] == PNG_B64
    assert result["diff"] == diff_tree
    assert "diff_path" not in result and "image_path" not in result


@pytest.mark.parametrize("broken_content,expected_message", [
    (None, "cv file was not found"),
    ("not json at all", "cv file is not valid JSON"),
    ("[1, 2, 3]", "cv file must contain a JSON object"),
])
def test_cv_path_resolution_rejects_bad_json_file(monkeypatch, broken_content, expected_message):
    async def fake_read(room_id, path, db):
        if path == "x.png":
            return f"data:image/png;base64,{PNG_B64}"
        return broken_content

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with pytest.raises(mcp_gateway.McpVisionInputError) as exc:
        asyncio.run(mcp_gateway._resolve_vision_arguments(
            _binding("extract_design_spec"),
            {"image_path": "x.png", "cv_path": "design/x.cv.json"},
            uuid4(),
            object(),
        ))
    assert str(exc.value) == expected_message


def test_cv_path_over_size_limit_is_rejected(monkeypatch):
    huge = json.dumps({"data": "x" * (mcp_gateway._VISION_JSON_FIELD_MAX_BYTES + 10)})

    async def fake_read(room_id, path, db):
        if path == "x.png":
            return f"data:image/png;base64,{PNG_B64}"
        return huge

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with pytest.raises(mcp_gateway.McpVisionInputError):
        asyncio.run(mcp_gateway._resolve_vision_arguments(
            _binding("extract_design_spec"),
            {"image_path": "x.png", "cv_path": "design/x.cv.json"},
            uuid4(),
            object(),
        ))


@pytest.mark.asyncio
async def test_bad_cv_path_fails_before_credit_reservation(monkeypatch):
    """§2.6 — resolution happens before `reserve_call`, so a bad cv_path never
    burns a credit."""

    async def fake_read(room_id, path, db):
        if path == "x.png":
            return f"data:image/png;base64,{PNG_B64}"
        return "not json"

    reserve = AsyncMock()
    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with patch("app.core.mcp_credits.reserve_call", reserve):
        result = await mcp_gateway.execute_mcp_tool(
            _binding("extract_design_spec"),
            uuid4(),
            uuid4(),
            {"image_path": "x.png", "cv_path": "design/x.cv.json"},
            db=object(),
        )
    assert result.startswith("Error:")
    reserve.assert_not_awaited()


def test_unknown_vision_tool_passes_through_unresolved():
    """§2.1 — the MCP server may ship a tool ahead of the Hub registry; the
    call must still go through, unresolved, rather than fail."""

    binding = _binding("some_future_tool")
    arguments = {"whatever": "value"}
    result = asyncio.run(mcp_gateway._resolve_vision_arguments(binding, arguments, uuid4(), object()))
    assert result is arguments


def test_non_vision_binding_is_unaffected():
    binding = mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(), market_item_id=None, market_slug=None,
        tool_name="describe_image", custom_config=None, isolation="stateless",
    )
    arguments = {"image_path": "x.png"}
    result = asyncio.run(mcp_gateway._resolve_vision_arguments(binding, arguments, uuid4(), object()))
    assert result is arguments


def test_vision_agent_schema_advertises_json_fields_for_extract_design_spec():
    spec = mcp_gateway._VISION_TOOLS["extract_design_spec"]
    schema = mcp_gateway._vision_agent_schema({"type": "object", "properties": {}}, spec)
    assert "image_path" in schema["properties"]
    assert "cv_path" in schema["properties"]
    assert "diff_path" not in schema["properties"]


# ---------------------------------------------------------------------------
# Design Vision Toolkit Stage 6 — R7 design_system injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_design_system_is_injected_when_agent_did_not_supply_one(monkeypatch):
    room_id = uuid4()
    tokens = {"colors": {"--color-primary": "#3B82F6"}}

    async def fake_read(read_room_id, path, db):
        return f"data:image/png;base64,{PNG_B64}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with patch("app.core.design_tokens.get_design_system", AsyncMock(return_value=tokens)) as getter:
        result = await mcp_gateway._resolve_vision_arguments(
            _binding("extract_design_spec"),
            {"image_path": "images/dashboard.png", "name": "dashboard"},
            room_id,
            object(),
        )
    getter.assert_awaited_once()
    assert result["design_system"] == tokens


@pytest.mark.asyncio
async def test_agent_supplied_design_system_is_never_overwritten(monkeypatch):
    own_tokens = {"colors": {"--brand": "#000"}}

    async def fake_read(read_room_id, path, db):
        return f"data:image/png;base64,{PNG_B64}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={"colors": {"x": "y"}})) as getter:
        result = await mcp_gateway._resolve_vision_arguments(
            _binding("critique_design"),
            {"image_path": "images/app.png", "design_system": own_tokens},
            uuid4(),
            object(),
        )
    getter.assert_not_awaited()
    assert result["design_system"] == own_tokens


@pytest.mark.asyncio
async def test_describe_image_never_gets_design_system_injected(monkeypatch):
    async def fake_read(read_room_id, path, db):
        return f"data:image/png;base64,{PNG_B64}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={"colors": {"x": "y"}})) as getter:
        result = await mcp_gateway._resolve_vision_arguments(
            _binding("describe_image"), {"image_path": "images/app.png"}, uuid4(), object(),
        )
    getter.assert_not_called()
    assert "design_system" not in result


# ---------------------------------------------------------------------------
# 2.3 — artifact insurance (base64 never reaches the agent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inline_artifact_base64_is_persisted_and_replaced_with_path():
    room_id = uuid4()
    fake_file = type("F", (), {"id": uuid4()})()
    upsert = AsyncMock(return_value=fake_file)
    emit = AsyncMock()
    with (
        patch("app.core.workspace_fs.fs_write_file", upsert),
        patch("app.core.events.emit_event", emit),
    ):
        result = await vision_artifacts.persist_vision_artifacts(
            {"artifacts": [{"kind": "crop", "mime_type": "image/png", "base64": PNG_B64}]},
            room_id=room_id, tool_name="describe_image", db=object(),
        )
    assert result["artifacts"][0]["path"].startswith("design/artifacts/describe_image-")
    assert "base64" not in json.dumps(result)
    upsert.assert_awaited_once()
    emit.assert_awaited_once()
    assert emit.await_args.kwargs["event_type"] == "FILE_CHANGED"


@pytest.mark.asyncio
async def test_artifact_write_failure_does_not_raise_and_reports_error():
    with (
        patch("app.core.workspace_fs.fs_write_file", AsyncMock(side_effect=RuntimeError("disk full"))),
        patch("app.core.events.emit_event", AsyncMock()),
    ):
        result = await vision_artifacts.persist_vision_artifacts(
            {"artifacts": [{"kind": "heatmap", "mime_type": "image/png", "base64": PNG_B64}]},
            room_id=uuid4(), tool_name="critique_design", db=object(),
        )
    assert result["artifacts"] == [{"kind": "heatmap", "error": "failed to save artifact"}]


@pytest.mark.asyncio
async def test_result_without_artifacts_key_is_untouched():
    result = await vision_artifacts.persist_vision_artifacts(
        {"description": "a boot"}, room_id=uuid4(), tool_name="describe_image", db=object(),
    )
    assert result == {"description": "a boot"}


# ---------------------------------------------------------------------------
# 2.4 — design spec becomes a file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_design_spec_is_saved_and_summary_returned():
    fake_file = type("F", (), {"id": uuid4()})()
    upsert = AsyncMock(return_value=fake_file)
    emit = AsyncMock()
    spec = {
        "version": 1,
        "layout": [{"id": "n1", "role": "header"}, {"id": "n2", "role": "card"}],
        "tokens": {"colors": [{"hex": "#fff", "source": "matched"}, {"hex": "#000", "source": "new"}]},
        "uncertain": ["hover state"],
    }
    with (
        patch("app.core.workspace_fs.fs_write_file", upsert),
        patch("app.core.events.emit_event", emit),
    ):
        result = await vision_artifacts.persist_design_spec(
            spec, room_id=uuid4(), tool_name="extract_design_spec", name="My Dashboard!", db=object(),
        )
    assert result["spec_path"] == "design/my-dashboard.spec.json"
    assert result["summary"] == {
        "nodes": 2, "roles": ["card", "header"], "uncertain": ["hover state"],
        "tokens_matched": 1, "tokens_new": 1,
    }
    # The full spec (layout/tokens tree) never appears in the returned value.
    assert "layout" not in result


@pytest.mark.asyncio
async def test_design_spec_write_failure_returns_original_result():
    spec = {"version": 1, "layout": []}
    with (
        patch("app.core.workspace_fs.fs_write_file", AsyncMock(side_effect=RuntimeError("boom"))),
        patch("app.core.events.emit_event", AsyncMock()),
    ):
        result = await vision_artifacts.persist_design_spec(
            spec, room_id=uuid4(), tool_name="extract_design_spec", name="x", db=object(),
        )
    assert result is spec


# ---------------------------------------------------------------------------
# Stage 2 — positive shape guards on persist_design_spec / persist_vision_artifacts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_design_spec_rejects_missing_version():
    write = AsyncMock()
    result = {"layout": [{"id": "n1"}]}  # no version field
    with patch("app.core.workspace_fs.fs_write_file", write):
        out = await vision_artifacts.persist_design_spec(
            result, room_id=uuid4(), tool_name="extract_design_spec", name="x", db=object(),
        )
    assert out is result
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_design_spec_rejects_wrong_version():
    write = AsyncMock()
    result = {"version": 2, "layout": []}  # version ≠ 1
    with patch("app.core.workspace_fs.fs_write_file", write):
        out = await vision_artifacts.persist_design_spec(
            result, room_id=uuid4(), tool_name="extract_design_spec", name="x", db=object(),
        )
    assert out is result
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_design_spec_rejects_non_list_layout():
    write = AsyncMock()
    result = {"version": 1, "layout": "not a list"}
    with patch("app.core.workspace_fs.fs_write_file", write):
        out = await vision_artifacts.persist_design_spec(
            result, room_id=uuid4(), tool_name="extract_design_spec", name="x", db=object(),
        )
    assert out is result
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_design_spec_rejects_non_dict():
    write = AsyncMock()
    with patch("app.core.workspace_fs.fs_write_file", write):
        out = await vision_artifacts.persist_design_spec(
            "Error: Vision provider is unavailable",
            room_id=uuid4(), tool_name="extract_design_spec", name="x", db=object(),
        )
    assert out == "Error: Vision provider is unavailable"
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_design_spec_accepts_valid_spec_and_emits_file_changed():
    write = AsyncMock(return_value=type("F", (), {"id": uuid4()})())
    emit = AsyncMock()
    spec = {"version": 1, "layout": [{"id": "n1", "role": "card"}], "tokens": {}, "uncertain": []}
    with (
        patch("app.core.workspace_fs.fs_write_file", write),
        patch("app.core.events.emit_event", emit),
    ):
        out = await vision_artifacts.persist_design_spec(
            spec, room_id=uuid4(), tool_name="extract_design_spec", name="card-grid", db=object(),
        )
    assert out["spec_path"] == "design/card-grid.spec.json"
    assert out["summary"]["nodes"] == 1
    write.assert_awaited_once()
    emit.assert_awaited_once()
    assert emit.await_args.kwargs["event_type"] == "FILE_CHANGED"


@pytest.mark.asyncio
async def test_persist_vision_artifacts_bails_on_non_dict_items_only():
    """artifacts=[42, "hello"] — no dicts → bail without touching anything."""
    write = AsyncMock()
    result = {"artifacts": [42, "hello"]}
    with patch("app.core.workspace_fs.fs_write_file", write):
        out = await vision_artifacts.persist_vision_artifacts(
            result, room_id=uuid4(), tool_name="describe_image", db=object(),
        )
    assert out is result
    write.assert_not_awaited()


# ---------------------------------------------------------------------------
# 2.5 — structural safety net for ALL MCP results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_small_result_is_unaffected_by_safety_net():
    result = {"ok": True}
    out = await mcp_gateway._apply_mcp_result_safety_net(
        result, room_id=uuid4(), tool_name="get_commit", db=object(), session_id=uuid4(),
    )
    # Secret masking/redaction runs first and may hand back a copy; below the
    # cap the content must come through untouched.
    assert out == result


@pytest.mark.asyncio
async def test_oversized_result_from_any_mcp_is_truncated_structurally(monkeypatch):
    monkeypatch.setattr(mcp_gateway, "MCP_MAX_RESULT_CHARS", 200)
    upsert = AsyncMock(return_value=type("F", (), {"id": uuid4()})())
    big_result = {"items": [{"n": i, "text": "x" * 50} for i in range(50)]}
    with patch("app.core.workspace_fs.fs_write_file", upsert):
        out = await mcp_gateway._apply_mcp_result_safety_net(
            big_result, room_id=uuid4(), tool_name="some_http_mcp_tool", db=object(), session_id=uuid4(),
        )
    assert out["truncated"] is True
    assert out["path"].startswith(".mcp-results/some_http_mcp_tool-")
    assert "hint" in out
    # Result must still be valid, well-shaped JSON.
    json.dumps(out)
    upsert.assert_awaited_once()
    written_path, written_content = upsert.await_args.args[1], upsert.await_args.args[2]
    assert json.loads(written_content) == big_result
    assert written_path == out["path"]


@pytest.mark.asyncio
async def test_safety_net_write_failure_degrades_to_preview_without_path():
    with patch("app.core.workspace_fs.fs_write_file", AsyncMock(side_effect=RuntimeError("no disk"))):
        out = await mcp_gateway._apply_mcp_result_safety_net(
            {"data": "x" * (mcp_gateway.MCP_MAX_RESULT_CHARS + 100)},
            room_id=uuid4(), tool_name="anything", db=object(), session_id=uuid4(),
        )
    assert out["truncated"] is True
    assert "path" not in out


@pytest.mark.asyncio
async def test_safety_net_without_db_returns_preview_only():
    out = await mcp_gateway._apply_mcp_result_safety_net(
        {"data": "x" * (mcp_gateway.MCP_MAX_RESULT_CHARS + 100)},
        room_id=uuid4(), tool_name="anything", db=None, session_id=None,
    )
    assert out["truncated"] is True
    assert "path" not in out


@pytest.mark.asyncio
async def test_execute_mcp_tool_applies_safety_net_to_non_vision_binding(monkeypatch):
    """R4/S4 must not be vision-specific."""

    binding = mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(), market_item_id=None, market_slug=None,
        tool_name="search", custom_config=None, isolation="stateless",
    )
    big = {"results": [{"snippet": "x" * 200} for _ in range(50)]}
    # Non-envelope dict (no "content" key) passes through _unwrap_mcp_tool_result
    # unchanged — matches non-MCP-spec transports such as mcp-host.
    monkeypatch.setattr(mcp_gateway, "MCP_MAX_RESULT_CHARS", 200)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=big)),
        patch("app.core.workspace_fs.fs_write_file", AsyncMock(return_value=type("F", (), {"id": uuid4()})())),
    ):
        result_text = await mcp_gateway.execute_mcp_tool(
            binding, uuid4(), uuid4(), {}, db=object(), request_id=uuid4(),
        )
    parsed = json.loads(result_text)
    assert parsed["truncated"] is True
    assert parsed["path"].startswith(".mcp-results/search-")


# ---------------------------------------------------------------------------
# Stage 1 — _unwrap_mcp_tool_result + execute_mcp_tool envelope path
# ---------------------------------------------------------------------------


def test_unwrap_success_envelope_returns_parsed_dict():
    spec = {"version": 1, "layout": [{"id": "n1", "role": "header"}]}
    envelope = {"content": [{"type": "text", "text": json.dumps(spec)}]}
    assert mcp_gateway._unwrap_mcp_tool_result(envelope) == spec


def test_unwrap_is_error_envelope_returns_error_string():
    envelope = {"isError": True, "content": [{"type": "text", "text": "Vision provider is unavailable"}]}
    result = mcp_gateway._unwrap_mcp_tool_result(envelope)
    assert result == "Error: Vision provider is unavailable"


def test_unwrap_non_envelope_passes_through():
    bare = {"results": [{"snippet": "hello"}]}
    assert mcp_gateway._unwrap_mcp_tool_result(bare) is bare


def test_unwrap_non_json_text_returns_raw_string():
    envelope = {"content": [{"type": "text", "text": "plain text, not JSON"}]}
    assert mcp_gateway._unwrap_mcp_tool_result(envelope) == "plain text, not JSON"


def test_unwrap_non_dict_passes_through():
    assert mcp_gateway._unwrap_mcp_tool_result("already a string") == "already a string"
    assert mcp_gateway._unwrap_mcp_tool_result(None) is None


@pytest.mark.asyncio
async def test_extract_design_spec_success_via_mcp_envelope(monkeypatch):
    """Stage 1: a real MCP success envelope from the Gateway is unwrapped before
    vision post-processing, so persist_design_spec receives the actual spec dict
    (not the transport shell) and emits a correct summary."""
    spec = {
        "version": 1,
        "layout": [{"id": "n1", "role": "header"}, {"id": "n2", "role": "card"}],
        "tokens": {"colors": []},
        "uncertain": [],
    }
    envelope = {"content": [{"type": "text", "text": json.dumps(spec)}]}
    # Non-billable binding: market_item_id=None skips all credit logic so this
    # test stays a pure unit test without DB or credit-service mocks.
    binding = mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(), market_item_id=None, market_slug="mcp-vision",
        tool_name="extract_design_spec", custom_config=None, isolation="stateless",
    )

    async def fake_read(room_id, path, db):
        return f"data:image/png;base64,{PNG_B64}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)),
        patch("app.core.workspace_fs.fs_write_file", AsyncMock(return_value=type("F", (), {"id": uuid4()})())),
        patch("app.core.events.emit_event", AsyncMock()),
        patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={})),
    ):
        result_text = await mcp_gateway.execute_mcp_tool(
            binding, uuid4(), uuid4(),
            {"image_path": "images/ref.png", "name": "ref"},
            db=object(),
        )
    parsed = json.loads(result_text)
    assert parsed["spec_path"] == "design/ref.spec.json"
    assert parsed["summary"]["nodes"] == 2


@pytest.mark.asyncio
async def test_is_error_envelope_returns_error_string_and_skips_vision_processing(monkeypatch):
    """Stage 1: isError envelope reaches execute_mcp_tool → error string returned,
    no file written, no FILE_CHANGED emitted."""
    envelope = {"isError": True, "content": [{"type": "text", "text": "Vision provider is unavailable"}]}
    binding = mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(), market_item_id=None, market_slug="mcp-vision",
        tool_name="extract_design_spec", custom_config=None, isolation="stateless",
    )

    async def fake_read(room_id, path, db):
        return f"data:image/png;base64,{PNG_B64}"

    write = AsyncMock()
    emit = AsyncMock()
    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)),
        patch("app.core.workspace_fs.fs_write_file", write),
        patch("app.core.events.emit_event", emit),
        patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={})),
    ):
        result = await mcp_gateway.execute_mcp_tool(
            binding, uuid4(), uuid4(),
            {"image_path": "images/ref.png", "name": "ref"},
            db=object(),
        )
    assert result == "Error: [provider_error] Vision provider is unavailable"
    write.assert_not_awaited()
    emit.assert_not_awaited()


# ---------------------------------------------------------------------------
# Stage 3 — credit refund on isError (HTTP 200 but tool reported failure)
# ---------------------------------------------------------------------------

def _billable_vision_binding(tool_name: str) -> mcp_gateway.McpToolBinding:
    return mcp_gateway.McpToolBinding(
        installed_item_id=uuid4(),
        market_item_id=uuid4(),
        market_slug="mcp-vision",
        tool_name=tool_name,
        custom_config=None,
        isolation="stateless",
    )


@pytest.mark.asyncio
async def test_is_error_on_billed_call_refunds_not_settles(monkeypatch):
    """Stage 3: a billed tool returning isError → refund_call, NOT settle_call."""
    envelope = {"isError": True, "content": [{"type": "text", "text": "Vision provider is unavailable"}]}
    reserve = AsyncMock()
    settle = AsyncMock()
    refund = AsyncMock()

    async def fake_read(room_id, path, db):
        return f"data:image/png;base64,{PNG_B64}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)),
        patch("app.core.mcp_credits.resolve_price_mcr", AsyncMock(return_value=900)),
        patch("app.core.mcp_credits.reserve_call", reserve),
        patch("app.core.mcp_credits.settle_call", settle),
        patch("app.core.mcp_credits.refund_call", refund),
        patch("app.core.metrics.record_mcp_credits_refund"),
        patch("app.core.metrics.record_mcp_credits_spent"),
        patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={})),
    ):
        result = await mcp_gateway.execute_mcp_tool(
            _billable_vision_binding("extract_design_spec"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png", "name": "ref"},
            db=object(),
        )
    assert result == "Error: [provider_error] Vision provider is unavailable"
    reserve.assert_awaited_once()
    refund.assert_awaited_once()
    assert refund.await_args.kwargs.get("note") == "tool returned isError"
    settle.assert_not_awaited()


@pytest.mark.asyncio
async def test_success_on_billed_call_settles_not_refunds(monkeypatch):
    """Stage 3 inverse: a successful billed call → settle_call, NOT refund_call."""
    spec = {"version": 1, "layout": [{"id": "n1", "role": "card"}], "tokens": {}, "uncertain": []}
    envelope = {"content": [{"type": "text", "text": json.dumps(spec)}]}
    reserve = AsyncMock()
    settle = AsyncMock()
    refund = AsyncMock()

    async def fake_read(room_id, path, db):
        return f"data:image/png;base64,{PNG_B64}"

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)
    with (
        patch("app.core.mcp_gateway._gateway_post", AsyncMock(return_value=envelope)),
        patch("app.core.mcp_credits.resolve_price_mcr", AsyncMock(return_value=900)),
        patch("app.core.mcp_credits.reserve_call", reserve),
        patch("app.core.mcp_credits.settle_call", settle),
        patch("app.core.mcp_credits.refund_call", refund),
        patch("app.core.metrics.record_mcp_credits_refund"),
        patch("app.core.metrics.record_mcp_credits_spent"),
        patch("app.core.workspace_fs.fs_write_file", AsyncMock(return_value=type("F", (), {"id": uuid4()})())),
        patch("app.core.events.emit_event", AsyncMock()),
        patch("app.core.design_tokens.get_design_system", AsyncMock(return_value={})),
    ):
        result = await mcp_gateway.execute_mcp_tool(
            _billable_vision_binding("extract_design_spec"),
            uuid4(), uuid4(),
            {"image_path": "images/ref.png", "name": "ref"},
            db=object(),
        )
    parsed = json.loads(result)
    assert parsed["spec_path"] == "design/ref.spec.json"
    reserve.assert_awaited_once()
    settle.assert_awaited_once()
    refund.assert_not_awaited()
