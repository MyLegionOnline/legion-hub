"""Capture a declared workspace service through the preview gateway.

The Browser tab and this module intentionally share the same service-origin
ticket flow.  The screenshot worker never connects to a service port directly
and never uses the deprecated ``workspace_preview`` endpoint.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.preview_gateway import (
    _clean_preview_query,
    _domain as _preview_domain,
    issue_ticket,
    preview_enabled,
    service_origin,
)
from app.core.workspace_services import WorkspaceServiceRegistry, services_enabled
from app.models.workspace import Workspace

logger = logging.getLogger(__name__)

_BROWSER_LOCK = asyncio.Lock()
_PLAYWRIGHT: Any = None
_BROWSER: Any = None


class ScreenshotError(RuntimeError):
    """A safe, user-facing screenshot failure without ticket/URL details."""


def _timeout_ms() -> int:
    try:
        return max(5_000, min(120_000, int(os.getenv("WORKSPACE_SCREENSHOT_TIMEOUT_MS", "30000"))))
    except ValueError:
        return 30_000


def _settle_ms() -> int:
    try:
        return max(0, min(5_000, int(os.getenv("WORKSPACE_SCREENSHOT_SETTLE_MS", "750"))))
    except ValueError:
        return 750


def _viewport() -> dict[str, int]:
    values: dict[str, int] = {}
    for key, default in (("WORKSPACE_SCREENSHOT_WIDTH", 1280), ("WORKSPACE_SCREENSHOT_HEIGHT", 720)):
        try:
            values[key] = max(320, min(3840, int(os.getenv(key, str(default)))))
        except ValueError:
            values[key] = default
    return {"width": values["WORKSPACE_SCREENSHOT_WIDTH"], "height": values["WORKSPACE_SCREENSHOT_HEIGHT"]}


def _clamp_dimension(value: Any, default: int) -> int:
    try:
        return max(320, min(3840, int(value)))
    except (TypeError, ValueError):
        return default


def _launch_args() -> list[str]:
    # The backend image runs as root in the shipped Compose deployments.  The
    # Chromium sandbox cannot initialize in that configuration; local non-root
    # runs keep the normal sandbox instead.
    args = ["--disable-dev-shm-usage"]
    geteuid = getattr(os, "geteuid", None)
    if geteuid is not None:
        try:
            if geteuid() == 0:
                args.append("--no-sandbox")
        except OSError:
            pass
    rule = _gateway_host_rule()
    if rule:
        args.append(rule)
    return args


def _gateway_host_rule() -> str | None:
    """Point ``*.<preview domain>`` at the Hub gateway for containerized Chromium.

    Inside Compose the advertised origin (``svc-<id>.preview.localhost:8001``)
    is not reachable, but the gateway is at ``backend:8000``.  The mapping is
    done by the browser's own resolver so the request keeps the service-origin
    ``Host`` header the gateway routes on.  Overriding ``Host`` through
    ``route.continue_`` does not work: Chromium drops it, and the request then
    lands on the Hub API itself (``/`` returns its JSON banner, any other path
    404s).
    """
    gateway_url = _screenshot_gateway_url()
    if not gateway_url:
        return None
    return f"--host-resolver-rules=MAP *.{_preview_domain()} {urlsplit(gateway_url).netloc}"


async def _browser() -> Any:
    """Return the one lazily-created Chromium instance for this backend."""
    global _PLAYWRIGHT, _BROWSER

    async with _BROWSER_LOCK:
        if _BROWSER is not None and _BROWSER.is_connected():
            return _BROWSER

        if _BROWSER is not None:
            _BROWSER = None
        if _PLAYWRIGHT is None:
            try:
                from playwright.async_api import async_playwright

                _PLAYWRIGHT = await async_playwright().start()
            except ImportError as exc:
                raise ScreenshotError("Playwright is not installed in the Hub backend") from exc
            except Exception as exc:
                logger.warning("screenshot: Playwright runtime failed to start (%s)", type(exc).__name__)
                raise ScreenshotError("Playwright runtime is not available in the Hub backend") from exc

        try:
            _BROWSER = await _PLAYWRIGHT.chromium.launch(headless=True, args=_launch_args())
        except Exception as exc:
            logger.warning("screenshot: Chromium launch failed (%s)", type(exc).__name__)
            try:
                await _PLAYWRIGHT.stop()
            finally:
                _PLAYWRIGHT = None
            raise ScreenshotError("Chromium is not available in the Hub backend") from exc
        return _BROWSER


async def close_browser() -> None:
    """Close the shared browser during Hub shutdown."""
    global _PLAYWRIGHT, _BROWSER

    async with _BROWSER_LOCK:
        browser, playwright = _BROWSER, _PLAYWRIGHT
        _BROWSER = None
        _PLAYWRIGHT = None
        if browser is not None:
            try:
                await browser.close()
            except Exception as exc:
                logger.warning("screenshot: failed to close Chromium (%s)", type(exc).__name__)
        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                logger.warning("screenshot: failed to stop Playwright")


def _preview_path(value: str | None, default: str) -> str:
    raw = str(value or default or "/").strip() or "/"
    parsed = urlsplit(raw)
    if (
        not raw.startswith("/")
        or raw.startswith("//")
        or "\\" in raw
        or any(ord(char) < 0x20 for char in raw)
        or parsed.scheme
        or parsed.netloc
    ):
        raise ScreenshotError("preview path must be an absolute path on the declared service")
    return raw


def _ticket_url(origin: str, path: str, ticket: str) -> str:
    parsed = urlsplit(path)
    pathname = parsed.path or "/"
    clean_query = _clean_preview_query(parsed.query)
    query = f"{clean_query}&" if clean_query else ""
    url = f"{origin}{pathname}?{query}ticket={quote(ticket, safe='')}"
    if parsed.fragment:
        url += f"#{parsed.fragment}"
    return url


def _screenshot_gateway_url() -> str | None:
    """Return an optional backend-reachable origin for containerized Chromium."""
    raw = os.getenv("WORKSPACE_SCREENSHOT_GATEWAY_URL", "").strip().rstrip("/")
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
        logger.warning("screenshot: ignoring invalid WORKSPACE_SCREENSHOT_GATEWAY_URL")
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


async def _service_for(db: AsyncSession, room_id: UUID, service_slug: str):
    result = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    workspace = result.scalar_one_or_none()
    if workspace is None:
        raise ScreenshotError("workspace not found for this room")
    service = await WorkspaceServiceRegistry(db).get(workspace.id, service_slug)
    if service is None:
        raise ScreenshotError(f"service {service_slug!r} is not declared")
    return service


async def capture_screenshot(
    room_id: UUID,
    service_slug: str,
    db: AsyncSession,
    *,
    user_id: UUID,
    path: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> bytes:
    """Render a ready declared HTTP service and return a PNG byte string.

    ``width``/``height`` override the default viewport (e.g. 375 for a phone).
    """
    if not services_enabled():
        raise ScreenshotError("workspace services are disabled")
    if not preview_enabled():
        raise ScreenshotError("workspace preview gateway is disabled")
    if not user_id:
        raise ScreenshotError("agent identity is unavailable")

    slug = str(service_slug or "").strip()
    if not slug:
        raise ScreenshotError("service_slug is required")
    service = await _service_for(db, room_id, slug)
    if service.kind not in {"web", "api"}:
        raise ScreenshotError("screenshots are available only for web and api services")
    if service.runtime_state != "ready":
        raise ScreenshotError(f"service {slug!r} is not ready; call workspace_service wait_ready first")
    if (service.listen or {}).get("protocol", "http") != "http":
        raise ScreenshotError("screenshots are available only for HTTP services")

    origin = service_origin(service.id)
    ticket = issue_ticket(
        user_id=user_id,
        room_id=room_id,
        service_id=service.id,
        origin=origin,
    )
    browser_url = _ticket_url(origin, _preview_path(path, service.default_path), ticket)

    browser = await _browser()
    context = None
    try:
        viewport = _viewport()
        if width is not None:
            viewport["width"] = _clamp_dimension(width, viewport["width"])
        if height is not None:
            viewport["height"] = _clamp_dimension(height, viewport["height"])
        context = await browser.new_context(viewport=viewport, device_scale_factor=1)
        page = await context.new_page()
        timeout = _timeout_ms()
        try:
            response = await page.goto(browser_url, wait_until="domcontentloaded", timeout=timeout)
            if response is not None and response.status >= 400:
                raise ScreenshotError("preview gateway returned an error")
            await page.wait_for_load_state("load", timeout=timeout)
            # The gateway's ticket bootstrap is a tiny HTML page with a
            # client-side location.replace. Playwright normally follows it,
            # but waiting explicitly prevents a screenshot of that transient
            # page when the first navigation resolves before the replacement.
            current_url = getattr(page, "url", "")
            if isinstance(current_url, str) and "ticket=" in current_url:
                await page.wait_for_url(
                    lambda candidate: "ticket=" not in candidate,
                    timeout=timeout,
                )
                await page.wait_for_load_state("load", timeout=timeout)
            if _settle_ms():
                await page.wait_for_timeout(_settle_ms())
            return await page.screenshot(type="png", full_page=False, animations="disabled")
        except ScreenshotError:
            raise
        except Exception as exc:
            logger.warning(
                "screenshot: preview navigation failed room=%s service=%s (%s)",
                room_id,
                slug,
                type(exc).__name__,
            )
            raise ScreenshotError("preview page could not be loaded") from exc
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:
                logger.warning("screenshot: failed to close page context room=%s service=%s", room_id, slug)
