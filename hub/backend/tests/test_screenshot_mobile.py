"""screenshot: engine, device presets, full_page, measure and the WebKit gateway proxy."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core import screenshot
from app.core.tools import REGISTRY, ToolContext


def _service():
    return SimpleNamespace(
        id=uuid4(),
        slug="catalog-web",
        kind="web",
        runtime_state="ready",
        default_path="/",
        listen={"protocol": "http", "assigned_port": 3010},
    )


def _db():
    result = MagicMock()
    result.scalar_one_or_none.return_value = SimpleNamespace(id=uuid4())
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


def _fake_browser(*, evaluate=None):
    page = MagicMock()
    page.goto = AsyncMock(return_value=SimpleNamespace(status=200))
    page.wait_for_load_state = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.screenshot = AsyncMock(return_value=b"png-bytes")
    page.evaluate = AsyncMock(side_effect=evaluate)
    context = MagicMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    browser = MagicMock()
    browser.new_context = AsyncMock(return_value=context)
    return browser, page


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("WORKSPACE_SERVICES_ENABLED", "true")
    monkeypatch.setenv("WORKSPACE_PREVIEW_GATEWAY_ENABLED", "true")
    monkeypatch.setattr(screenshot.WorkspaceServiceRegistry, "get", AsyncMock(return_value=_service()))
    monkeypatch.setattr(screenshot, "issue_ticket", lambda **_kwargs: "t")


async def _capture(**kwargs):
    return await screenshot.capture_screenshot(
        uuid4(), "catalog-web", _db(), user_id=uuid4(), **kwargs
    )


def test_tool_schema_exposes_engine_device_full_page_and_measure():
    tool = REGISTRY.get("screenshot")
    props = tool.input_schema["properties"]

    assert props["engine"]["enum"] == ["chromium", "webkit"]
    assert {"iphone-15", "iphone-14", "iphone-se", "pixel-7"} <= set(props["device"]["enum"])
    assert props["full_page"]["type"] == "boolean"
    assert props["measure"]["type"] == "array"
    assert props["measure"]["maxItems"] == screenshot.MAX_MEASURE_SELECTORS
    assert "Do not install Playwright" in tool.description


@pytest.mark.asyncio
async def test_device_preset_sets_a_mobile_context_on_the_requested_engine(monkeypatch, enabled):
    browser, page = _fake_browser()
    launcher = AsyncMock(return_value=browser)
    monkeypatch.setattr(screenshot, "_browser", launcher)

    result = await _capture(engine="WebKit", device="iphone-14")

    launcher.assert_awaited_once_with("webkit")
    kwargs = browser.new_context.await_args.kwargs
    assert kwargs["viewport"] == {"width": 390, "height": 664}
    assert kwargs["screen"] == {"width": 390, "height": 844}
    assert kwargs["device_scale_factor"] == 3
    assert kwargs["is_mobile"] is True
    assert kwargs["has_touch"] is True
    assert "iPhone" in kwargs["user_agent"]
    assert result.engine == "webkit"
    assert result.device == "iphone-14"
    assert result.summary() == "webkit, iphone-14, viewport 390x664 @3x"
    page.evaluate.assert_not_awaited()
    assert "full_page" not in page.screenshot.await_args.kwargs


@pytest.mark.asyncio
async def test_explicit_size_overrides_the_device_viewport(monkeypatch, enabled):
    browser, _page = _fake_browser()
    monkeypatch.setattr(screenshot, "_browser", AsyncMock(return_value=browser))

    await _capture(device="pixel-7", height=1200)

    kwargs = browser.new_context.await_args.kwargs
    assert kwargs["viewport"] == {"width": 412, "height": 1200}
    assert kwargs["screen"] == {"width": 412, "height": 1200}
    assert kwargs["device_scale_factor"] == 2.625
    assert "Android" in kwargs["user_agent"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"engine": "firefox"}, "engine must be one of"),
        ({"device": "nokia-3310"}, "device must be one of"),
        ({"measure": [f".c{i}" for i in range(21)]}, "at most 20 selectors"),
        ({"measure": ["." + "a" * 400]}, "at most 300 characters"),
        ({"measure": 5}, "list of CSS selectors"),
    ],
)
async def test_invalid_options_fail_before_any_browser_launch(monkeypatch, enabled, kwargs, message):
    launcher = AsyncMock()
    monkeypatch.setattr(screenshot, "_browser", launcher)

    with pytest.raises(screenshot.ScreenshotError, match=message):
        await _capture(**kwargs)
    launcher.assert_not_awaited()


@pytest.mark.asyncio
async def test_full_page_is_requested_and_capped_for_endless_pages(monkeypatch, enabled):
    browser, page = _fake_browser(evaluate=lambda *_args: 50_000)
    monkeypatch.setattr(screenshot, "_browser", AsyncMock(return_value=browser))

    result = await _capture(device="iphone-se", full_page=True)

    shot = page.screenshot.await_args.kwargs
    assert shot["full_page"] is True
    assert shot["scale"] == "css"
    assert shot["clip"] == {"x": 0, "y": 0, "width": 375, "height": screenshot.MAX_FULL_PAGE_HEIGHT}
    assert result.full_page is True
    assert result.summary() == "chromium, iphone-se, viewport 375x548 @2x, full page"


@pytest.mark.asyncio
async def test_full_page_of_a_normal_page_is_not_clipped(monkeypatch, enabled):
    browser, page = _fake_browser(evaluate=lambda *_args: 2_400)
    monkeypatch.setattr(screenshot, "_browser", AsyncMock(return_value=browser))

    await _capture(full_page=True)

    assert "clip" not in page.screenshot.await_args.kwargs


@pytest.mark.asyncio
async def test_measure_returns_boxes_and_styles_as_text(monkeypatch, enabled):
    measured = {
        "viewport": {"width": 390, "height": 664, "dpr": 3},
        "document": {"scrollWidth": 412, "scrollHeight": 1800, "clientWidth": 390},
        "items": [
            {
                "selector": ".hero",
                "count": 7,
                "matches": [
                    {
                        "rect": {"x": 0, "y": 64, "width": 412, "height": 600.5},
                        "styles": {"display": "flex", "flex-direction": "column", "height": "600.5px"},
                    }
                ],
            },
            {"selector": ".missing", "count": 0, "matches": []},
            {"selector": "a[", "error": "invalid selector"},
        ],
    }
    browser, page = _fake_browser(evaluate=lambda *_args: measured)
    monkeypatch.setattr(screenshot, "_browser", AsyncMock(return_value=browser))

    result = await _capture(measure=[".hero", ".missing", ".hero", "a[", "  "])

    selectors, max_matches = page.evaluate.await_args.args[1]
    assert selectors == [".hero", ".missing", "a["]
    assert max_matches == screenshot.MAX_MEASURE_MATCHES
    text = result.measurements
    assert "viewport 390x664 @3x" in text
    assert "horizontal overflow: 22px wider than the viewport" in text
    assert ".hero: 7 matches, first 1" in text
    assert (
        "[0] x=0 y=64 w=412 h=600.5 | display: flex; flex-direction: column; height: 600.5px"
        in text
    )
    assert ".missing: no match" in text
    assert "a[: invalid selector" in text


def test_measurements_report_no_overflow_when_the_page_fits():
    text = screenshot.format_measurements(
        {
            "viewport": {"width": 390, "height": 664, "dpr": 3},
            "document": {"scrollWidth": 390, "scrollHeight": 900, "clientWidth": 390},
            "items": [],
        }
    )
    assert "horizontal overflow: none" in text
    assert screenshot.format_measurements(None) == "Layout measurements: unavailable"


@pytest.mark.asyncio
async def test_tool_passes_new_options_and_appends_measurements():
    from app.core.tools._screenshot import ScreenshotTool

    ctx = ToolContext(
        room_id=uuid4(), session_id=uuid4(), task_id=None, db=MagicMock(), mcp_user_id=uuid4()
    )
    capture = screenshot.ScreenshotCapture(
        png=b"png",
        engine="webkit",
        device="iphone-15",
        viewport={"width": 393, "height": 659},
        device_scale_factor=3,
        full_page=True,
        measurements="Layout measurements: ok",
    )
    with (
        patch("app.core.screenshot.capture_screenshot", new_callable=AsyncMock, return_value=capture) as cap,
        patch("app.core.workspace_fs.fs_write_file", new_callable=AsyncMock),
        patch("app.core.events.emit_event", new_callable=AsyncMock),
        patch("app.core.vision_dispatch.agent_can_see_images", new_callable=AsyncMock, return_value=True),
    ):
        result = await ScreenshotTool().execute(
            ctx,
            {
                "service_slug": "catalog-web",
                "engine": "webkit",
                "device": "iphone-15",
                "full_page": True,
                "measure": [".hero"],
            },
        )

    kwargs = cap.await_args.kwargs
    assert kwargs["engine"] == "webkit"
    assert kwargs["device"] == "iphone-15"
    assert kwargs["full_page"] is True
    assert kwargs["measure"] == [".hero"]
    assert "(webkit, iphone-15, viewport 393x659 @3x, full page)." in result
    assert result.endswith("\n\nLayout measurements: ok")


def test_gateway_proxy_maps_only_the_preview_domain():
    proxy = screenshot._GatewayProxy("preview.localhost", "backend:8000")

    assert proxy.target("svc-abc.preview.localhost", 8001) == ("backend", 8000)
    assert proxy.target("SVC-ABC.Preview.Localhost.", 8001) == ("backend", 8000)
    assert proxy.target("fonts.example.com", 443) == ("fonts.example.com", 443)
    assert proxy.target("preview.localhost.evil.example", 80) == (
        "preview.localhost.evil.example",
        80,
    )


def test_gateway_proxy_rewrites_absolute_form_and_keeps_the_service_host():
    head = (
        b"GET http://svc-abc.preview.localhost:8001/app?x=1 HTTP/1.1\r\n"
        b"Host: svc-abc.preview.localhost:8001\r\n"
        b"Proxy-Connection: keep-alive\r\n"
        b"Connection: keep-alive\r\n"
        b"Accept: */*\r\n\r\n"
    )

    host, new_head, _scheme, port = screenshot._GatewayProxy.rewrite_request_head(head)

    assert (host, port) == ("svc-abc.preview.localhost", 8001)
    assert new_head == (
        b"GET /app?x=1 HTTP/1.1\r\n"
        b"Host: svc-abc.preview.localhost:8001\r\n"
        b"Accept: */*\r\n"
        b"Connection: close\r\n\r\n"
    )


def test_gateway_proxy_keeps_websocket_upgrades_and_rejects_origin_form():
    head = (
        b"GET http://svc-abc.preview.localhost:8001/hmr HTTP/1.1\r\n"
        b"Host: svc-abc.preview.localhost:8001\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: websocket\r\n\r\n"
    )

    _host, new_head, _scheme, _port = screenshot._GatewayProxy.rewrite_request_head(head)

    assert b"Connection: Upgrade" in new_head
    assert b"Connection: close" not in new_head
    with pytest.raises(ValueError):
        screenshot._GatewayProxy.rewrite_request_head(b"GET /relative HTTP/1.1\r\n\r\n")


@pytest.mark.asyncio
async def test_gateway_proxy_forwards_preview_requests_to_the_gateway():
    received: list[bytes] = []

    async def gateway(reader, writer):
        received.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(gateway, "127.0.0.1", 0)
    gateway_port = server.sockets[0].getsockname()[1]
    proxy = screenshot._GatewayProxy("preview.localhost", f"127.0.0.1:{gateway_port}")
    proxy_url = await proxy.start()
    try:
        host, _, port = proxy_url.removeprefix("http://").rpartition(":")
        reader, writer = await asyncio.open_connection(host, int(port))
        writer.write(
            b"GET http://svc-abc.preview.localhost:8001/ HTTP/1.1\r\n"
            b"Host: svc-abc.preview.localhost:8001\r\n\r\n"
        )
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=5)
        writer.close()
    finally:
        await proxy.close()
        server.close()
        await server.wait_closed()

    assert response.endswith(b"\r\n\r\nok")
    assert received[0].startswith(
        b"GET / HTTP/1.1\r\nHost: svc-abc.preview.localhost:8001\r\n"
    )


def test_clean_url_drops_ticket_and_session_parameters():
    url = screenshot._clean_url(
        "http://svc-x.preview.localhost:8001",
        "/blog?page=2&ticket=t&_legion_preview_session=s#top",
        "/",
    )
    assert url == "http://svc-x.preview.localhost:8001/blog?page=2#top"
    assert screenshot._clean_url("http://o", None, "/") == "http://o/"


def test_session_is_read_from_a_multi_cookie_set_cookie_header():
    headers = {"set-cookie": "other=1; Path=/\nlegion-preview=abc.def; HttpOnly; Secure; SameSite=none"}
    assert screenshot._session_from_set_cookie(headers, "legion-preview") == "abc.def"
    assert screenshot._session_from_set_cookie({}, "legion-preview") is None


@pytest.mark.asyncio
async def test_refused_secure_cookie_is_re_added_and_the_clean_url_reloaded(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_SCHEME", "http")
    context = MagicMock()
    context.cookies = AsyncMock(return_value=[])
    context.add_cookies = AsyncMock()
    page = MagicMock()
    page.goto = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    response = MagicMock()
    response.all_headers = AsyncMock(
        return_value={"set-cookie": "legion-preview=sess; HttpOnly; Secure; SameSite=none"}
    )

    await screenshot._ensure_preview_cookie(
        context, page, response, "http://svc-x.preview.localhost:8001", "http://svc-x.preview.localhost:8001/a", 1000
    )

    cookie = context.add_cookies.await_args.args[0][0]
    assert cookie["name"] == "legion-preview"
    assert cookie["value"] == "sess"
    assert cookie["secure"] is False
    assert cookie["url"] == "http://svc-x.preview.localhost:8001"
    page.goto.assert_awaited_once_with(
        "http://svc-x.preview.localhost:8001/a", wait_until="domcontentloaded", timeout=1000
    )


@pytest.mark.asyncio
async def test_stored_cookie_needs_no_second_navigation(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_SCHEME", "http")
    context = MagicMock()
    context.cookies = AsyncMock(return_value=[{"name": "legion-preview", "value": "sess"}])
    context.add_cookies = AsyncMock()
    page = MagicMock()
    page.goto = AsyncMock()

    await screenshot._ensure_preview_cookie(context, page, MagicMock(), "http://o", "http://o/", 1000)

    context.add_cookies.assert_not_awaited()
    page.goto.assert_not_awaited()
