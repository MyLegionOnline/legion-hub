from __future__ import annotations

import json
import time
from typing import Any

from app import config


class CatalogCache:
    """Small in-process cache with optional Redis backing.

    Redis is an optimization only; Gateway remains functional when Redis is
    unavailable and never stores custom MCP configuration or secrets.
    """

    def __init__(self) -> None:
        self._local: dict[str, tuple[float, dict[str, Any]]] = {}
        self._redis = None

    async def start(self) -> None:
        if not config.REDIS_URL:
            return
        try:
            from redis.asyncio import Redis

            self._redis = Redis.from_url(config.REDIS_URL, decode_responses=True)
            await self._redis.ping()
        except Exception:
            self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def get(self, key: str) -> dict[str, Any] | None:
        now = time.monotonic()
        local = self._local.get(key)
        if local and local[0] > now:
            return local[1]
        if self._redis is not None:
            try:
                value = await self._redis.get(f"mcp-gateway:{key}")
                if value:
                    parsed = json.loads(value)
                    self._local[key] = (now + config.CACHE_TTL_SEC, parsed)
                    return parsed
            except Exception:
                pass
        return None

    async def put(self, key: str, value: dict[str, Any]) -> None:
        self._local[key] = (time.monotonic() + config.CACHE_TTL_SEC, value)
        if self._redis is not None:
            try:
                await self._redis.setex(
                    f"mcp-gateway:{key}", int(max(1, config.CACHE_TTL_SEC)), json.dumps(value)
                )
            except Exception:
                pass
