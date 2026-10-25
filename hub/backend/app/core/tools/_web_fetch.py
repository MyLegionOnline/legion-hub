"""web_fetch tool — HTTP GET from the backend (not the container).

SSRF protection: hostname is resolved before the request and every IP in the
result set is checked against private/reserved ranges. The check is re-run on
every redirect hop so a chain example.com → 169.254.x cannot bypass it.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_MAX_BYTES = 5 * 1024 * 1024   # download cap: 5 MB
_MAX_CHARS = 50_000              # markdown cap before _truncate
_TIMEOUT_SECS = 15.0
_MAX_REDIRECTS = 5
_USER_AGENT = "Legion-Agent/1.0 (+web-research)"

_PRIVATE_NETS: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local + cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),     # carrier-grade NAT
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),           # ULA
    ipaddress.ip_network("fe80::/10"),          # link-local IPv6
)


def _ssrf_check(url: str) -> str | None:
    """Return an error message if the URL is SSRF-blocked, None if it's safe.

    Blocking DNS resolution synchronously — call via asyncio.to_thread in async code.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return f"Invalid URL: {exc}"

    if parsed.scheme not in ("http", "https"):
        return "Only http:// and https:// URLs are allowed"

    host = parsed.hostname
    if not host:
        return "Invalid URL: missing hostname"

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return f"DNS resolution failed: {exc}"

    for info in infos:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return f"Blocked: {ip_str} is a restricted address"
        for net in _PRIVATE_NETS:
            if ip.version == net.version and ip in net:
                return f"Blocked: {ip_str} is in private range {net}"

    return None


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown text. Falls back to tag-stripping if markdownify unavailable."""
    try:
        import markdownify  # type: ignore
        return markdownify.markdownify(html, heading_style="ATX", bullets="-")
    except ImportError:
        import re
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


async def _do_fetch(url: str) -> tuple[int, str, str]:
    """
    Fetch url with SSRF check on every hop.

    Returns (status_code, content_type, body_text).
    On error, returns (0, '', error_message).
    """
    seen: set[str] = set()

    while True:
        if url in seen:
            return 0, "", "Error: redirect loop detected"
        seen.add(url)

        ssrf_err = await asyncio.to_thread(_ssrf_check, url)
        if ssrf_err:
            return 0, "", f"Error: {ssrf_err}"

        try:
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=_TIMEOUT_SECS,
                headers={"User-Agent": _USER_AGENT},
            ) as client:
                resp = await client.get(url)
        except httpx.TimeoutException:
            return 0, "", f"Error: request timed out after {_TIMEOUT_SECS:.0f}s"
        except Exception as exc:
            return 0, "", f"Error: {exc}"

        if resp.is_redirect:
            if len(seen) >= _MAX_REDIRECTS:
                return 0, "", "Error: too many redirects"
            location = resp.headers.get("location", "")
            if not location:
                return resp.status_code, "", "Error: redirect missing Location header"
            url = urljoin(str(resp.url), location)
            continue

        status = resp.status_code
        ct = resp.headers.get("content-type", "")

        if len(resp.content) > _MAX_BYTES:
            raw = resp.content[:_MAX_BYTES]
        else:
            raw = resp.content

        if "json" in ct:
            body = raw.decode("utf-8", errors="replace")
        elif "html" in ct:
            body = _html_to_markdown(raw.decode("utf-8", errors="replace"))
        else:
            body = raw.decode("utf-8", errors="replace")

        return status, ct, body


class WebFetchTool(Tool):
    name = "web_fetch"
    description = (
        "Fetch a public web URL and return its content as markdown (or raw text/JSON). "
        "Use to read documentation, GitHub READMEs, API references, blog posts, changelogs, etc. "
        "Content is truncated to ~50 000 characters. "
        "SSRF-protected: private/internal IPs and cloud metadata endpoints are blocked."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Full URL to fetch (http or https only)",
            },
            "max_chars": {
                "type": "integer",
                "description": (
                    f"Max characters to return (default {_MAX_CHARS}; max 100 000). "
                    "Increase if the content you need is cut off."
                ),
                "default": _MAX_CHARS,
            },
        },
        "required": ["url"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        url = str(tool_input.get("url") or "").strip()
        if not url:
            return "Error: url is required"

        max_chars = int(tool_input.get("max_chars") or _MAX_CHARS)
        max_chars = max(1, min(max_chars, 100_000))

        status, ct, body = await _do_fetch(url)

        if status == 0:
            # body contains the error message
            return body

        if status >= 400:
            return f"HTTP {status} from {url}"

        if len(body) > max_chars:
            body = body[:max_chars] + f"\n[...content truncated at {max_chars} chars — use max_chars to request more]"

        return _truncate(f"URL: {url}\nStatus: {status}\nContent-Type: {ct}\n\n{body}")


REGISTRY.register(WebFetchTool())
