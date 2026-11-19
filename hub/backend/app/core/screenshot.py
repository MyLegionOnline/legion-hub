"""Capture a declared workspace service through the preview gateway.

The Browser tab and this module intentionally share the same service-origin
ticket flow.  The screenshot worker never connects to a service port directly
and never uses the deprecated ``workspace_preview`` endpoint.

Rendering runs here, in the Hub backend, with Chromium or WebKit (both are
installed in the backend image).  Room workspaces deliberately carry no
browsers: an agent that needs to see or measure a page calls the
``screenshot`` tool instead of installing Playwright in its workspace.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.preview_gateway import (
    _clean_preview_query,
    _domain as _preview_domain,
    _scheme as _preview_scheme,
    issue_ticket,
    preview_cookie_name,
    preview_enabled,
    service_origin,
)
from app.core.workspace_services import WorkspaceServiceRegistry, services_enabled
from app.models.workspace import Workspace

logger = logging.getLogger(__name__)

_BROWSER_LOCK = asyncio.Lock()
_PLAYWRIGHT: Any = None
# One lazily-launched browser per engine ("chromium", "webkit").
_BROWSERS: dict[str, Any] = {}
# Loopback forward proxy that gives WebKit the gateway mapping Chromium gets
# from ``--host-resolver-rules``; started with the first WebKit launch.
_WEBKIT_PROXY: "_GatewayProxy | None" = None
_GATEWAY_SCHEME_WARNED = False

ENGINES = ("chromium", "webkit")

_IOS_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"
)
_ANDROID_UA = (
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Mobile Safari/537.36"
)

# Phone presets.  ``viewport`` is the page area below the browser chrome (what
# ``100vh`` layouts actually get), ``screen`` the whole display; the values
# follow Playwright's device descriptors.  All presets are mobile with touch.
DEVICES: dict[str, dict[str, Any]] = {
    "iphone-15": {
        "viewport": {"width": 393, "height": 659},
        "screen": {"width": 393, "height": 852},
        "device_scale_factor": 3,
        "user_agent": _IOS_UA,
    },
    "iphone-14": {
        "viewport": {"width": 390, "height": 664},
        "screen": {"width": 390, "height": 844},
        "device_scale_factor": 3,
        "user_agent": _IOS_UA,
    },
    "iphone-se": {
        "viewport": {"width": 375, "height": 548},
        "screen": {"width": 375, "height": 667},
        "device_scale_factor": 2,
        "user_agent": _IOS_UA,
    },
    "pixel-7": {
        "viewport": {"width": 412, "height": 839},
        "screen": {"width": 412, "height": 915},
        "device_scale_factor": 2.625,
        "user_agent": _ANDROID_UA,
    },
}

# A full-page capture of an endless feed would otherwise be an enormous PNG.
MAX_FULL_PAGE_HEIGHT = 16_384
MAX_MEASURE_SELECTORS = 20
MAX_MEASURE_SELECTOR_LENGTH = 300
MAX_MEASURE_MATCHES = 5

# Computed styles reported for every measured element.  Shorthands are built
# from longhands in the page because engines disagree on computed shorthands.
_MEASURE_JS = """
([selectors, maxMatches]) => {
  const longhands = {
    margin: ['margin-top', 'margin-right', 'margin-bottom', 'margin-left'],
    padding: ['padding-top', 'padding-right', 'padding-bottom', 'padding-left'],
    flex: ['flex-grow', 'flex-shrink', 'flex-basis'],
    gap: ['row-gap', 'column-gap'],
    overflow: ['overflow-x', 'overflow-y'],
  };
  const props = ['display', 'position', 'box-sizing', 'width', 'height', 'min-height',
    'max-height', 'margin', 'padding', 'flex-direction', 'flex-wrap', 'flex',
    'justify-content', 'align-items', 'gap', 'overflow', 'font-size', 'line-height'];
  const round = (v) => Math.round(v * 10) / 10;
  const root = document.documentElement;
  const out = {
    viewport: {width: window.innerWidth, height: window.innerHeight, dpr: window.devicePixelRatio},
    document: {
      scrollWidth: Math.max(root.scrollWidth, document.body ? document.body.scrollWidth : 0),
      scrollHeight: Math.max(root.scrollHeight, document.body ? document.body.scrollHeight : 0),
      clientWidth: root.clientWidth,
    },
    items: [],
  };
  for (const selector of selectors) {
    let nodes;
    try {
      nodes = Array.from(document.querySelectorAll(selector));
    } catch (e) {
      out.items.push({selector, error: 'invalid selector'});
      continue;
    }
    out.items.push({
      selector,
      count: nodes.length,
      matches: nodes.slice(0, maxMatches).map((el) => {
        const r = el.getBoundingClientRect();
        const cs = getComputedStyle(el);
        const styles = {};
        for (const p of props) {
          styles[p] = longhands[p]
            ? longhands[p].map((name) => cs.getPropertyValue(name)).join(' ')
            : cs.getPropertyValue(p);
        }
        return {
          rect: {x: round(r.left + window.scrollX), y: round(r.top + window.scrollY),
                 width: round(r.width), height: round(r.height)},
          styles,
        };
      }),
    });
  }
  return out;
}
"""

_PAGE_HEIGHT_JS = (
    "() => Math.max(document.documentElement.scrollHeight, "
    "document.body ? document.body.scrollHeight : 0)"
)


class ScreenshotError(RuntimeError):
    """A safe, user-facing screenshot failure without ticket/URL details."""


@dataclass(frozen=True)
class ScreenshotCapture:
    """A captured frame plus the rendering settings that produced it."""

    png: bytes
    engine: str = "chromium"
    device: str | None = None
    viewport: dict[str, int] = field(default_factory=dict)
    device_scale_factor: float = 1
    full_page: bool = False
    measurements: str | None = None

    def summary(self) -> str:
        parts = [self.engine]
        if self.device:
            parts.append(self.device)
        scale = f" @{self.device_scale_factor:g}x" if self.device_scale_factor != 1 else ""
        parts.append(f"viewport {self.viewport.get('width')}x{self.viewport.get('height')}{scale}")
        if self.full_page:
            parts.append("full page")
        return ", ".join(parts)


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


def _normalize_engine(value: Any) -> str:
    engine = str(value or "chromium").strip().lower()
    if engine not in ENGINES:
        raise ScreenshotError(f"engine must be one of: {', '.join(ENGINES)}")
    return engine


def _normalize_device(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    device = str(value).strip().lower()
    if device not in DEVICES:
        raise ScreenshotError(f"device must be one of: {', '.join(DEVICES)}")
    return device


def _normalize_measure(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ScreenshotError("measure must be a list of CSS selectors")
    selectors: list[str] = []
    for item in value:
        selector = str(item or "").strip()
        if not selector:
            continue
        if len(selector) > MAX_MEASURE_SELECTOR_LENGTH:
            raise ScreenshotError(
                f"measure selectors must be at most {MAX_MEASURE_SELECTOR_LENGTH} characters"
            )
        if selector not in selectors:
            selectors.append(selector)
    if len(selectors) > MAX_MEASURE_SELECTORS:
        raise ScreenshotError(f"measure accepts at most {MAX_MEASURE_SELECTORS} selectors")
    return selectors


def _context_options(
    device: str | None, width: Any, height: Any
) -> dict[str, Any]:
    """Browser-context options for a device preset with optional overrides."""
    if device:
        preset = DEVICES[device]
        viewport = dict(preset["viewport"])
        options: dict[str, Any] = {
            "screen": dict(preset["screen"]),
            "device_scale_factor": preset["device_scale_factor"],
            "user_agent": preset["user_agent"],
            "is_mobile": True,
            "has_touch": True,
        }
    else:
        viewport = _viewport()
        options = {"device_scale_factor": 1}
    if width is not None:
        viewport["width"] = _clamp_dimension(width, viewport["width"])
    if height is not None:
        viewport["height"] = _clamp_dimension(height, viewport["height"])
    if "screen" in options:
        options["screen"] = {
            "width": max(options["screen"]["width"], viewport["width"]),
            "height": max(options["screen"]["height"], viewport["height"]),
        }
    return {"viewport": viewport, **options}


def format_measurements(data: Any) -> str:
    """Render the in-page measurement result as compact text for the agent."""
    if not isinstance(data, dict):
        return "Layout measurements: unavailable"
    viewport = data.get("viewport") or {}
    document = data.get("document") or {}
    scroll_width = document.get("scrollWidth")
    client_width = document.get("clientWidth") or viewport.get("width")
    overflow = "none"
    if isinstance(scroll_width, (int, float)) and isinstance(client_width, (int, float)):
        if scroll_width > client_width:
            overflow = f"{scroll_width - client_width:g}px wider than the viewport"
    lines = [
        "Layout measurements (CSS px; x/y relative to the document top-left):",
        f"viewport {viewport.get('width')}x{viewport.get('height')} @{viewport.get('dpr')}x; "
        f"document {scroll_width}x{document.get('scrollHeight')}; horizontal overflow: {overflow}",
    ]
    for item in data.get("items") or []:
        selector = item.get("selector")
        if item.get("error"):
            lines.append(f"{selector}: {item['error']}")
            continue
        count = item.get("count") or 0
        matches = item.get("matches") or []
        if not count:
            lines.append(f"{selector}: no match")
            continue
        shown = f", first {len(matches)}" if count > len(matches) else ""
        lines.append(f"{selector}: {count} match{'es' if count != 1 else ''}{shown}")
        for index, match in enumerate(matches):
            rect = match.get("rect") or {}
            styles = match.get("styles") or {}
            style_text = "; ".join(f"{key}: {value}" for key, value in styles.items())
            lines.append(
                f"  [{index}] x={rect.get('x'):g} y={rect.get('y'):g} "
                f"w={rect.get('width'):g} h={rect.get('height'):g} | {style_text}"
            )
    return "\n".join(lines)


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
    404s).  WebKit gets the same mapping through :class:`_GatewayProxy`.
    """
    gateway_url = _screenshot_gateway_url()
    if not gateway_url:
        return None
    return f"--host-resolver-rules=MAP *.{_preview_domain()} {urlsplit(gateway_url).netloc}"


class _GatewayProxy:
    """Loopback HTTP forward proxy mapping ``*.<preview domain>`` to the gateway.

    WebKit has no ``--host-resolver-rules``, so it is launched with this proxy
    instead: requests for the preview domain go to the gateway with their
    service-origin ``Host`` header intact, everything else (fonts, CDNs) goes
    to its real host, as Chromium would resolve it.
    """

    def __init__(self, domain: str, gateway_netloc: str) -> None:
        self.domain = domain.lower().strip(".")
        host, _, port = gateway_netloc.rpartition(":")
        if not host or not port.isdigit():
            host, port = gateway_netloc, "80"
        self.gateway = (host.strip("[]"), int(port))
        self.url: str | None = None
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> str:
        if self._server is None:
            self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
            port = self._server.sockets[0].getsockname()[1]
            self.url = f"http://127.0.0.1:{port}"
        assert self.url is not None
        return self.url

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=5)
            except Exception:
                pass
            self._server = None
            self.url = None

    def target(self, host: str, port: int) -> tuple[str, int]:
        name = host.lower().strip("[]").rstrip(".")
        if name == self.domain or name.endswith("." + self.domain):
            return self.gateway
        return host.strip("[]"), port

    @staticmethod
    def rewrite_request_head(head: bytes) -> tuple[str, bytes, str, int]:
        """Turn an absolute-form proxy request head into an origin-form one.

        Returns ``(host, new_head, scheme, port)``.  Without an ``Upgrade`` the
        upstream connection is closed after one response, because the next
        request on a kept-alive browser connection would be absolute-form too.
        """
        lines = head.rstrip(b"\r\n").split(b"\r\n")
        method, target, version = lines[0].decode("latin-1").split(" ", 2)
        parts = urlsplit(target)
        if parts.scheme != "http" or not parts.hostname:
            raise ValueError("proxy request target must be an absolute http URL")
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        upgrade = any(line.lower().startswith(b"upgrade:") for line in lines[1:])
        headers = [
            line
            for line in lines[1:]
            if line
            and not line.lower().startswith((b"proxy-connection:", b"keep-alive:"))
            and (upgrade or not line.lower().startswith(b"connection:"))
        ]
        if not any(line.lower().startswith(b"host:") for line in headers):
            headers.insert(0, b"Host: " + parts.netloc.encode("latin-1"))
        if not upgrade:
            headers.append(b"Connection: close")
        new_head = (
            f"{method} {path} {version}\r\n".encode("latin-1") + b"\r\n".join(headers) + b"\r\n\r\n"
        )
        return parts.hostname, new_head, parts.scheme, parts.port or 80

    @staticmethod
    async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            chunk = await reader.read(65_536)
            if not chunk:
                return
            writer.write(chunk)
            await writer.drain()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
            method = head.split(b" ", 1)[0].decode("latin-1").upper()
            if method == "CONNECT":
                target = head.split(b" ", 2)[1].decode("latin-1")
                host, _, port = target.rpartition(":")
                upstream_reader, upstream_writer = await asyncio.open_connection(
                    *self.target(host, int(port or 443))
                )
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                try:
                    host, new_head, _scheme, port = self.rewrite_request_head(head)
                except ValueError:
                    writer.write(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
                    return
                upstream_reader, upstream_writer = await asyncio.open_connection(
                    *self.target(host, port)
                )
                upstream_writer.write(new_head)
                await upstream_writer.drain()
            tasks = {
                asyncio.create_task(self._pipe(reader, upstream_writer)),
                asyncio.create_task(self._pipe(upstream_reader, writer)),
            }
            _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception:
            pass
        finally:
            for stream in (upstream_writer, writer):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass


async def _start_playwright() -> Any:
    global _PLAYWRIGHT

    if _PLAYWRIGHT is None:
        try:
            from playwright.async_api import async_playwright

            _PLAYWRIGHT = await async_playwright().start()
        except ImportError as exc:
            raise ScreenshotError("Playwright is not installed in the Hub backend") from exc
        except Exception as exc:
            logger.warning("screenshot: Playwright runtime failed to start (%s)", type(exc).__name__)
            raise ScreenshotError("Playwright runtime is not available in the Hub backend") from exc
    return _PLAYWRIGHT


async def _launch(playwright: Any, engine: str) -> Any:
    global _WEBKIT_PROXY

    if engine == "chromium":
        return await playwright.chromium.launch(headless=True, args=_launch_args())
    options: dict[str, Any] = {"headless": True}
    gateway_url = _screenshot_gateway_url()
    if gateway_url:
        if _WEBKIT_PROXY is None:
            _WEBKIT_PROXY = _GatewayProxy(_preview_domain(), urlsplit(gateway_url).netloc)
        options["proxy"] = {"server": await _WEBKIT_PROXY.start()}
    return await playwright.webkit.launch(**options)


async def _browser(engine: str = "chromium") -> Any:
    """Return the one lazily-created browser of ``engine`` for this backend."""
    global _PLAYWRIGHT

    async with _BROWSER_LOCK:
        browser = _BROWSERS.get(engine)
        if browser is not None and browser.is_connected():
            return browser
        _BROWSERS.pop(engine, None)

        playwright = await _start_playwright()
        label = "Chromium" if engine == "chromium" else "WebKit"
        try:
            browser = await _launch(playwright, engine)
        except Exception as exc:
            logger.warning("screenshot: %s launch failed (%s)", label, type(exc).__name__)
            if not _BROWSERS:
                try:
                    await playwright.stop()
                finally:
                    _PLAYWRIGHT = None
            raise ScreenshotError(f"{label} is not available in the Hub backend") from exc
        _BROWSERS[engine] = browser
        return browser


async def close_browser() -> None:
    """Close the shared browsers during Hub shutdown."""
    global _PLAYWRIGHT, _WEBKIT_PROXY

    async with _BROWSER_LOCK:
        browsers = list(_BROWSERS.items())
        _BROWSERS.clear()
        playwright, proxy = _PLAYWRIGHT, _WEBKIT_PROXY
        _PLAYWRIGHT = None
        _WEBKIT_PROXY = None
        for engine, browser in browsers:
            try:
                await browser.close()
            except Exception as exc:
                logger.warning("screenshot: failed to close %s (%s)", engine, type(exc).__name__)
        if proxy is not None:
            await proxy.close()
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


def _clean_url(origin: str, path: str | None, default: str) -> str:
    """The service URL without ticket or bootstrap-session parameters."""
    parsed = urlsplit(_preview_path(path, default))
    query = _clean_preview_query(parsed.query)
    url = f"{origin}{parsed.path or '/'}" + (f"?{query}" if query else "")
    return url + (f"#{parsed.fragment}" if parsed.fragment else "")


def _session_from_set_cookie(headers: dict[str, str], name: str) -> str | None:
    """Pick the preview session value out of a (newline-joined) Set-Cookie header."""
    for line in str(headers.get("set-cookie") or "").split("\n"):
        pair = line.split(";", 1)[0].strip()
        key, sep, value = pair.partition("=")
        if sep and key.strip() == name and value:
            return value.strip().strip('"')
    return None


async def _ensure_preview_cookie(
    context: Any, page: Any, response: Any, origin: str, clean_url: str, timeout: int
) -> None:
    """Keep the preview session in a cookie when the engine refused to store it.

    On a local ``http://*.preview.localhost`` origin the gateway sets a
    ``Secure; SameSite=None`` cookie.  Chromium treats ``*.localhost`` as
    trustworthy and stores it; WebKit does not.  The gateway then falls back
    to a ``_legion_preview_session`` query parameter, which stays in the
    page location and breaks client-side routers.  For this short-lived,
    isolated context the same session is re-added as a plain host cookie and
    the clean URL is loaded again.
    """
    try:
        cookie_name = preview_cookie_name()
        if any(cookie.get("name") == cookie_name for cookie in await context.cookies(origin)):
            return
        headers = await response.all_headers() if response is not None else {}
    except Exception:
        return
    session = _session_from_set_cookie(headers, cookie_name)
    if not session:
        return
    await context.add_cookies(
        [
            {
                "name": cookie_name,
                "value": session,
                "url": origin,
                "httpOnly": True,
                "secure": origin.startswith("https://"),
                "sameSite": "Lax",
            }
        ]
    )
    await page.goto(clean_url, wait_until="domcontentloaded", timeout=timeout)
    await page.wait_for_load_state("load", timeout=timeout)


def _screenshot_gateway_url() -> str | None:
    """Return an optional backend-reachable origin for the containerized browsers."""
    raw = os.getenv("WORKSPACE_SCREENSHOT_GATEWAY_URL", "").strip().rstrip("/")
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
        logger.warning("screenshot: ignoring invalid WORKSPACE_SCREENSHOT_GATEWAY_URL")
        return None
    preview_scheme = _preview_scheme()
    if parsed.scheme != preview_scheme:
        # The browser keeps the service origin's scheme when its host is mapped
        # to the gateway, so an https preview mapped onto a plain-http gateway
        # fails the TLS handshake (ERR_SSL_PROTOCOL_ERROR).  Behind a TLS
        # terminator the public origin itself is reachable; use it unmapped.
        global _GATEWAY_SCHEME_WARNED
        if not _GATEWAY_SCHEME_WARNED:
            _GATEWAY_SCHEME_WARNED = True
            logger.info(
                "screenshot: WORKSPACE_SCREENSHOT_GATEWAY_URL is %s but previews are %s; "
                "browsers load the public preview origin directly",
                parsed.scheme,
                preview_scheme,
            )
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


_NET_ERROR_RE = re.compile(r"\b(NS_ERROR_[A-Z_]+|net::ERR_[A-Z0-9_]+)")


def _browser_error_code(exc: BaseException) -> str:
    """The engine's network error code (``net::ERR_…``), ``timeout`` or ``-``."""
    match = _NET_ERROR_RE.search(str(exc))
    if match:
        return match.group(1)
    first_line = (str(exc).splitlines() or [""])[0]
    if type(exc).__name__ == "TimeoutError" or "Timeout" in first_line:
        return "timeout"
    return "-"


def _load_failure_message(exc: BaseException) -> str:
    """Agent-facing text: say where the failure is so the room is not 'fixed'."""
    code = _browser_error_code(exc)
    if code == "timeout":
        return (
            "preview page could not be loaded (timed out). Check that the service "
            "answers on its declared port; retry once after workspace_service wait_ready"
        )
    detail = f" ({code})" if code != "-" else ""
    return (
        f"preview page could not be loaded{detail}. If the service is ready and answers "
        "locally, this is a Hub-side browser or gateway problem, not something to fix "
        "inside the room; do not install browsers in the workspace — report it instead"
    )


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
    engine: str | None = None,
    device: str | None = None,
    full_page: bool = False,
    measure: list[str] | None = None,
) -> ScreenshotCapture:
    """Render a ready declared HTTP service and return the PNG with its settings.

    ``device`` applies a phone preset (viewport, DPR, mobile UA, touch);
    ``width``/``height`` override its or the default viewport.  ``measure``
    lists CSS selectors whose boxes and computed styles are reported as text.
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
    engine_name = _normalize_engine(engine)
    device_name = _normalize_device(device)
    selectors = _normalize_measure(measure)
    options = _context_options(device_name, width, height)

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

    browser = await _browser(engine_name)
    context = None
    try:
        context = await browser.new_context(**options)
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
            await _ensure_preview_cookie(
                context, page, response, origin, _clean_url(origin, path, service.default_path), timeout
            )
            if _settle_ms():
                await page.wait_for_timeout(_settle_ms())

            measurements = None
            if selectors:
                measurements = format_measurements(
                    await page.evaluate(_MEASURE_JS, [selectors, MAX_MEASURE_MATCHES])
                )

            # scale="css" keeps a DPR-3 phone frame at its CSS size so the PNG
            # stays readable for vision models; layout still renders at the DPR.
            shot: dict[str, Any] = {"type": "png", "animations": "disabled", "scale": "css"}
            if full_page:
                shot["full_page"] = True
                page_height = await page.evaluate(_PAGE_HEIGHT_JS)
                if isinstance(page_height, (int, float)) and page_height > MAX_FULL_PAGE_HEIGHT:
                    shot["clip"] = {
                        "x": 0,
                        "y": 0,
                        "width": options["viewport"]["width"],
                        "height": MAX_FULL_PAGE_HEIGHT,
                    }
            png = await page.screenshot(**shot)
            return ScreenshotCapture(
                png=png,
                engine=engine_name,
                device=device_name,
                viewport=dict(options["viewport"]),
                device_scale_factor=options["device_scale_factor"],
                full_page=bool(full_page),
                measurements=measurements,
            )
        except ScreenshotError:
            raise
        except Exception as exc:
            logger.warning(
                "screenshot: preview navigation failed room=%s service=%s engine=%s (%s %s)",
                room_id,
                slug,
                engine_name,
                type(exc).__name__,
                _browser_error_code(exc),
            )
            raise ScreenshotError(_load_failure_message(exc)) from exc
    finally:
        if context is not None:
            try:
                await context.close()
            except Exception:
                logger.warning("screenshot: failed to close page context room=%s service=%s", room_id, slug)
