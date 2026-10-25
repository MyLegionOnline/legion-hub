"""HTTP client for Hub -> Legion Market calls (catalog browsing, package fetch).

Complements api/v1/internal_market.py, which handles the reverse direction
(Market -> Hub, authenticated with X-Market-Token). This client talks to
Market's public, unauthenticated read endpoints, so it carries no credentials.

With ``MARKET_ENABLED`` off (the default of a self-hosted Hub, see
``app.core.market_switch``) nothing here touches the network: the catalogue
and the category list are empty and every item is "not found".
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from app.core.market_switch import market_enabled, market_url
from app.core.skill_package import MAX_PACKAGE_BYTES
from app.plugins import get_registry

logger = logging.getLogger(__name__)

MARKET_TIMEOUT_SEC = float(os.getenv("MARKET_API_TIMEOUT_SEC", "10"))


class MarketClientError(Exception):
    """Market request failed (unreachable, timed out, or non-2xx response)."""


class MarketNotFoundError(MarketClientError):
    """Market responded 404 — the slug/version does not exist."""


class MarketAccessError(MarketClientError):
    """Market refused the request for want of credentials or a purchase (401/402/403).

    A paid skill package goes only to a Hub Market trusts or to an account that
    bought it (internal design notes stage 6.1).
    """


def _require_enabled(what: str) -> None:
    if not market_enabled():
        raise MarketNotFoundError(f"{what}: the Market integration is off on this Hub")


async def _get(path: str, params: Optional[dict[str, Any]] = None) -> Any:
    _require_enabled(path)  # the one place every read goes through
    url = f"{market_url()}{path}"
    try:
        async with httpx.AsyncClient(timeout=MARKET_TIMEOUT_SEC) as client:
            response = await client.get(url, params=params)
    except httpx.HTTPError as exc:
        raise MarketClientError(f"Market request failed: {exc}") from exc
    if response.status_code == 404:
        raise MarketNotFoundError(f"Market resource not found: {path}")
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise MarketClientError(f"Market request failed: {exc}") from exc
    return response.json()


async def list_catalog(
    *,
    type: Optional[str] = None,
    q: Optional[str] = None,
    is_default: Optional[bool] = None,
    category: Optional[list[str]] = None,
    publisher: Optional[str] = None,
    price: Optional[str] = None,
    access: Optional[str] = None,
    sort: Optional[str] = None,
    page: int = 1,
    page_size: int = 24,
) -> dict[str, Any]:
    """Proxy of GET {MARKET_API_URL}/api/catalog."""
    if not market_enabled():
        return {"items": [], "total": 0, "page": page, "page_size": page_size}
    params: dict[str, Any] = {"page": page, "page_size": page_size}
    if type:
        params["type"] = type
    if q:
        params["q"] = q
    if is_default is not None:
        params["is_default"] = is_default
    if category:
        params["category"] = list(category)
    for key, value in (("publisher", publisher), ("price", price), ("access", access), ("sort", sort)):
        if value:
            params[key] = value
    return await _get("/api/catalog", params)


async def list_categories(
    *,
    type: Optional[str] = None,
    q: Optional[str] = None,
    publisher: Optional[str] = None,
    price: Optional[str] = None,
    access: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Proxy of GET {MARKET_API_URL}/api/catalog/categories (faceted counts)."""
    if not market_enabled():
        return []
    params: dict[str, Any] = {}
    if type:
        params["type"] = type
    if q:
        params["q"] = q
    for key, value in (("publisher", publisher), ("price", price), ("access", access)):
        if value:
            params[key] = value
    return await _get("/api/catalog/categories", params)


async def get_catalog_item(slug: str) -> dict[str, Any]:
    """Fetch one public Market item by its canonical slug."""
    _require_enabled(f"Market item {slug}")
    return await _get(f"/api/catalog/{slug}")


async def get_skill(slug: str) -> dict[str, Any]:
    """Proxy of GET {MARKET_API_URL}/api/skills/{slug} — metadata + versions[]."""
    _require_enabled(f"Market skill {slug}")
    return await _get(f"/api/skills/{slug}")


def _package_headers() -> dict[str, str]:
    """What plugins say about this Hub to Market; empty for a self-hosted Hub."""
    headers: dict[str, str] = {}
    for provider in get_registry().market_credentials:
        headers.update(provider())
    return headers


async def fetch_package(slug: str, version: str) -> tuple[bytes, str]:
    """Download one skill package version.

    Returns (zip bytes, X-Checksum-SHA256 header value). Enforces
    MAX_PACKAGE_BYTES against both the declared Content-Length and the actual
    bytes read, since Market's declared length isn't itself a security
    guarantee here.
    """
    _require_enabled(f"Skill package {slug}@{version}")
    url = f"{market_url()}/api/skills/{slug}/{version}/package"
    try:
        async with httpx.AsyncClient(timeout=MARKET_TIMEOUT_SEC) as client:
            async with client.stream("GET", url, headers=_package_headers()) as response:
                if response.status_code == 404:
                    raise MarketNotFoundError(f"Skill package not found: {slug}@{version}")
                if response.status_code in (401, 402, 403):
                    raise MarketAccessError(f"Market refused the package {slug}@{version}: {response.status_code}")
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > MAX_PACKAGE_BYTES:
                    raise MarketClientError(f"Skill package exceeds {MAX_PACKAGE_BYTES} bytes")
                checksum = response.headers.get("X-Checksum-SHA256", "")
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > MAX_PACKAGE_BYTES:
                        raise MarketClientError(f"Skill package exceeds {MAX_PACKAGE_BYTES} bytes")
    except httpx.HTTPError as exc:
        raise MarketClientError(f"Market request failed: {exc}") from exc
    return bytes(chunks), checksum
