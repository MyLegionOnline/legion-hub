from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core import screenshot
from app.core.tools import REGISTRY, ToolContext
from app.core.tools.registry import canonicalize_tool_input


def _service(**overrides):
    values = {
        "id": uuid4(),
        "slug": "catalog-web",
        "title": "Catalog",
        "kind": "web",
        "runtime_state": "ready",
        "default_path": "/",
        "listen": {"protocol": "http", "assigned_port": 3010},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _db(workspace):
    result = MagicMock()
    result.scalar_one_or_none.return_value = workspace
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


def test_screenshot_tool_is_registered_for_internal_agents():
    tool = REGISTRY.get("screenshot")

    assert tool is not None
    assert tool.roles == frozenset({"agent"})
    assert tool.requires_container is False
    assert tool.input_schema["required"] == ["service_slug"]
    assert "service_slug" in tool.input_schema["properties"]


def test_ticket_url_keeps_service_path_query_and_fragment():
    url = screenshot._ticket_url(
        "http://svc-service.preview.localhost",
        "/dashboard?tab=errors#overview",
        "ticket with spaces",
    )

    assert url == (
        "http://svc-service.preview.localhost/dashboard?tab=errors&ticket=ticket%20with%20spaces#overview"
    )

    assert "evil" not in screenshot._ticket_url(
        "http://svc-service.preview.localhost", "/?ticket=evil", "fresh"
    )


def test_preview_path_rejects_external_targets():
    with pytest.raises(screenshot.ScreenshotError):
        screenshot._preview_path("https://attacker.example/", "/")
    with pytest.raises(screenshot.ScreenshotError):
        screenshot._preview_path("//attacker.example/", "/")


def test_internal_gateway_url_accepts_only_an_origin(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SCREENSHOT_GATEWAY_URL", "http://backend:8000/")
    assert screenshot._screenshot_gateway_url() == "http://backend:8000"
    monkeypatch.setenv("WORKSPACE_SCREENSHOT_GATEWAY_URL", "http://backend:8000/internal")
    assert screenshot._screenshot_gateway_url() is None


def test_gateway_host_rule_maps_preview_domain_to_gateway(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_DOMAIN", "preview.localhost")
    monkeypatch.setenv("WORKSPACE_SCREENSHOT_GATEWAY_URL", "http://backend:8000")
    assert screenshot._gateway_host_rule() == (
        "--host-resolver-rules=MAP *.preview.localhost backend:8000"
    )
    assert "--host-resolver-rules=MAP *.preview.localhost backend:8000" in screenshot._launch_args()

    monkeypatch.delenv("WORKSPACE_SCREENSHOT_GATEWAY_URL")
    assert screenshot._gateway_host_rule() is None
    assert not any(arg.startswith("--host-resolver-rules") for arg in screenshot._launch_args())


def test_screenshot_url_path_is_not_canonicalized_as_a_workspace_file():
    assert canonicalize_tool_input(
        "screenshot", {"service_slug": "catalog-web", "path": "/dashboard?tab=errors"}
    )["path"] == "/dashboard?tab=errors"


@pytest.mark.asyncio
async def test_capture_uses_ready_service_origin_ticket_and_isolated_context(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SERVICES_ENABLED", "true")
    monkeypatch.setenv("WORKSPACE_PREVIEW_GATEWAY_ENABLED", "true")
    workspace = SimpleNamespace(id=uuid4())
    service = _service(default_path="/dashboard")
    db = _db(workspace)

    page = MagicMock()
    page.goto = AsyncMock(return_value=SimpleNamespace(status=200))
    page.wait_for_load_state = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"png-bytes")
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)

    monkeypatch.setattr(screenshot.WorkspaceServiceRegistry, "get", AsyncMock(return_value=service))
    monkeypatch.setattr(screenshot, "_browser", AsyncMock(return_value=browser))
    monkeypatch.setattr(screenshot, "issue_ticket", lambda **_kwargs: "one-time-ticket")

    result = await screenshot.capture_screenshot(
        uuid4(), "catalog-web", db, user_id=uuid4(), path="/dashboard?tab=errors"
    )

    assert result == b"png-bytes"
    page.goto.assert_awaited_once()
    browser.new_context.assert_awaited_once_with(
        viewport={"width": 1280, "height": 720}, device_scale_factor=1
    )
    assert page.goto.await_args.args[0].endswith(
        "/dashboard?tab=errors&ticket=one-time-ticket"
    )
    context.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_capture_honours_requested_viewport(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SERVICES_ENABLED", "true")
    monkeypatch.setenv("WORKSPACE_PREVIEW_GATEWAY_ENABLED", "true")
    page = MagicMock()
    page.goto = AsyncMock(return_value=SimpleNamespace(status=200))
    page.wait_for_load_state = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"png-bytes")
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    monkeypatch.setattr(screenshot.WorkspaceServiceRegistry, "get", AsyncMock(return_value=_service()))
    monkeypatch.setattr(screenshot, "_browser", AsyncMock(return_value=browser))
    monkeypatch.setattr(screenshot, "issue_ticket", lambda **_kwargs: "t")

    await screenshot.capture_screenshot(
        uuid4(), "catalog-web", _db(SimpleNamespace(id=uuid4())), user_id=uuid4(), width=375, height=99999
    )

    browser.new_context.assert_awaited_once_with(
        viewport={"width": 375, "height": 3840}, device_scale_factor=1
    )


@pytest.mark.asyncio
async def test_capture_rejects_non_ready_service_before_launch(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SERVICES_ENABLED", "true")
    monkeypatch.setenv("WORKSPACE_PREVIEW_GATEWAY_ENABLED", "true")
    service = _service(runtime_state="starting")
    monkeypatch.setattr(screenshot.WorkspaceServiceRegistry, "get", AsyncMock(return_value=service))
    browser = AsyncMock()
    monkeypatch.setattr(screenshot, "_browser", browser)

    with pytest.raises(screenshot.ScreenshotError, match="not ready"):
        await screenshot.capture_screenshot(
            uuid4(), service.slug, _db(SimpleNamespace(id=uuid4())), user_id=uuid4()
        )

    browser.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_saves_png_data_uri_and_emits_file_changed(monkeypatch):
    from app.core.tools._screenshot import ScreenshotTool

    ctx = ToolContext(
        room_id=uuid4(),
        session_id=uuid4(),
        task_id=None,
        db=MagicMock(),
        mcp_user_id=uuid4(),
    )
    saved_file = SimpleNamespace(id=uuid4())

    with (
        patch("app.core.screenshot.capture_screenshot", new_callable=AsyncMock, return_value=b"png-bytes"),
        patch("app.core.workspace_fs.fs_write_file", new_callable=AsyncMock, return_value=saved_file) as upsert,
        patch("app.core.events.emit_event", new_callable=AsyncMock) as emit,
    ):
        result = await ScreenshotTool().execute(ctx, {"service_slug": "catalog-web"})

    assert result.startswith("Screenshot saved to /project/screenshots/catalog-web-")
    assert result.endswith(".png")
    assert upsert.await_args.args[0] == ctx.room_id
    assert upsert.await_args.args[1].startswith("/project/screenshots/catalog-web-")
    assert upsert.await_args.args[2] == "data:image/png;base64,cG5nLWJ5dGVz"
    assert emit.await_args.kwargs["event_type"] == "FILE_CHANGED"
    assert emit.await_args.kwargs["metadata"]["source"] == "screenshot"
