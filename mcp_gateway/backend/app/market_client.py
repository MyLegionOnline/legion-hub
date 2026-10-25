from __future__ import annotations

from typing import Any
from uuid import UUID

import httpx
from fastapi import HTTPException

from app import config
from app.cache import CatalogCache


class MarketClient:
    def __init__(self, cache: CatalogCache) -> None:
        self.cache = cache

    async def resolve(
        self,
        installed_item_id: UUID,
        *,
        market_item_id: UUID | None = None,
        slug: str | None = None,
    ) -> dict[str, Any]:
        catalog_id = market_item_id or installed_item_id
        key = f"item:{catalog_id}"
        cached = await self.cache.get(key)
        if cached:
            return cached
        # The Market public API intentionally omits internal_endpoint. The
        # internal endpoint is therefore fetched through the Gateway-only
        # internal route, authenticated with a distinct service token.
        urls = [
            f"{config.MARKET_API_URL.rstrip('/')}/api/internal/gateway/items/{catalog_id}",
        ]
        if slug:
            urls.append(f"{config.MARKET_API_URL.rstrip('/')}/api/internal/gateway/items/slug/{slug}")
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SEC) as client:
            response = None
            for url in urls:
                try:
                    candidate = await client.get(url, headers={"X-Market-Gateway-Token": config.MARKET_API_TOKEN})
                except httpx.RequestError as exc:
                    raise HTTPException(status_code=502, detail="Market catalog is unavailable") from exc
                if candidate.status_code != 404:
                    response = candidate
                    break
            if response is None:
                raise HTTPException(status_code=404, detail="MCP catalog item not found")
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="MCP catalog item not found")
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail="Market catalog rejected the lookup")
        item = response.json()
        if item.get("type") != "mcp" or not item.get("is_active", True):
            raise HTTPException(status_code=404, detail="MCP catalog item not found")
        if not item.get("internal_endpoint"):
            raise HTTPException(status_code=503, detail="MCP endpoint is not configured")
        await self.cache.put(key, item)
        return item

    async def resolve_by_slug(self, slug: str) -> dict[str, Any]:
        """Look up a catalog item by slug alone (diagnostics/operator tooling).

        Unlike resolve(), this does not require a Hub-issued installed_item_id —
        it is used by the Gateway's own readiness checks (§9.2 of the market plan).
        """
        key = f"slug:{slug}"
        cached = await self.cache.get(key)
        if cached:
            return cached
        url = f"{config.MARKET_API_URL.rstrip('/')}/api/internal/gateway/items/slug/{slug}"
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SEC) as client:
            try:
                response = await client.get(url, headers={"X-Market-Gateway-Token": config.MARKET_API_TOKEN})
            except httpx.RequestError as exc:
                raise HTTPException(status_code=502, detail="Market catalog is unavailable") from exc
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="MCP catalog item not found")
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail="Market catalog rejected the lookup")
        item = response.json()
        if item.get("type") != "mcp" or not item.get("is_active", True):
            raise HTTPException(status_code=404, detail="MCP catalog item not found")
        if not item.get("internal_endpoint"):
            raise HTTPException(status_code=503, detail="MCP endpoint is not configured")
        await self.cache.put(key, item)
        return item
