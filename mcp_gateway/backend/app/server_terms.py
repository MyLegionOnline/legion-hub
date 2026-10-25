"""The Server Terms of an optional MCP server, before a call from our Hub
(Terms §6, legion_market/legal/PENDING.md item 56).

Some catalogue servers (Vision, Search) send the input to an outside service
and have their own terms, which a person accepts in Market before the first
call. Market marks such an item with ``server_terms``; for those only,
Gateway asks Market whether the caller's Legion account accepted the current
version (``/internal/server-terms/check``). Our Hub names the account in
``X-Legion-Account-Id`` next to its own user id, and only the service token
may say it. A self-hosted Hub's key is checked by the central wallet when the
call is reserved (``account_keys.reserve``).

Fail-closed: no account, a refusal or Market not answering means no call.
An acceptance is remembered for a minute per account and server, so a run of
calls does not ask Market each time; a refusal is never remembered.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from fastapi import HTTPException

from app import config

logger = logging.getLogger("mcp_gateway.server_terms")

SERVER_TERMS_REQUIRED = "server_terms_required"
ACCOUNT_HEADER = "X-Legion-Account-Id"

#: (account id, terms slug) -> expires at.
_accepted: dict[tuple[str, str], float] = {}


def reset_cache() -> None:
    _accepted.clear()


def _unavailable() -> HTTPException:
    return HTTPException(status_code=503, detail={
        "code": "server_terms_unavailable", "message": "The server terms cannot be checked right now",
    })


async def verify(item: dict[str, Any], account_id: str | None) -> None:
    """Refuse a call of ``item`` whose Server Terms ``account_id`` has not accepted."""
    terms = item.get("server_terms")
    if not terms:
        return
    key = (account_id or "", str(terms))
    if account_id and _accepted.get(key, 0) > time.monotonic():
        return
    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SEC) as client:
            response = await client.post(
                f"{config.MARKET_API_URL.rstrip('/')}/internal/server-terms/check",
                json={"account_id": account_id or None, "market_item_id": item.get("id"),
                      "market_slug": item.get("slug")},
                headers={"X-Market-Gateway-Token": config.MARKET_API_TOKEN},
            )
    except httpx.RequestError:
        raise _unavailable() from None
    if response.status_code == 200:
        if account_id:
            _accepted[key] = time.monotonic() + config.SERVER_TERMS_CACHE_SEC
        return
    if response.status_code == 403:
        try:
            detail = response.json().get("detail")
        except Exception:  # noqa: BLE001 - an unreadable refusal is still a refusal
            detail = None
        if isinstance(detail, dict) and detail.get("code") == SERVER_TERMS_REQUIRED:
            raise HTTPException(status_code=403, detail=detail)
    logger.warning("server terms check: Market answered %s", response.status_code)
    raise _unavailable()
