"""Calls from self-hosted Hubs, paid from the central wallet (internal design notes §3.10, stage 3b).

Our own Hub authenticates with the service token and keeps its users'
credits in its own database; Gateway double-checks that through
``app.quota``. A self-hosted Hub's database is its owner's, so nothing it
says about money can be believed. It sends the Legion account's API key
instead (``Authorization: Bearer lgn_…``), and Gateway:

1. asks Market whose key it is (``/internal/wallet/authorize``) — a revoked
   key, an unknown one or a blocked account is a 401;
2. limits how often one key may call (:class:`RateLimiter`);
3. reserves the call in the account's central wallet *before* the upstream
   MCP is reached (``/internal/wallet/reserve``; Market prices it from its
   own catalogue);
4. settles it after the MCP answered, or refunds it when the call never got
   there or the tool reported an error — the same outcomes as the Hub's own
   ledger.

Fail-closed: Market not answering means the call is refused, never free.
A key only ever reaches catalogue items: a custom MCP link through our
Gateway would make it an open proxy.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx
from fastapi import HTTPException

from app import config

logger = logging.getLogger("mcp_gateway.account_keys")

KEY_MARK = "lgn_"
_TOKEN_HEADER = "X-Market-Gateway-Token"


@dataclass(frozen=True)
class Caller:
    """Who is calling: our Hub (``service``) or a self-hosted Hub with an account key."""

    kind: str  # "service" | "account"
    api_key: str | None = None
    account_id: str | None = None
    key_prefix: str | None = None

    @property
    def is_account(self) -> bool:
        return self.kind == "account"

    def tenant_id(self) -> str:
        """The ``_meta.user_id`` a tenant-aware MCP sees for an account caller.

        Not the user id a self-hosted Hub sends: that is the Hub's own claim
        and could name one of our Cloud users. One account is one tenant.
        """
        return f"legion-account:{self.account_id}"


SERVICE = Caller(kind="service")


def looks_like_key(value: str | None) -> bool:
    return bool(value) and value.startswith(KEY_MARK) and 16 < len(value) <= 200


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _error(status_code: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message, **extra})


TERMS_REQUIRED = "terms_update_required"
SERVER_TERMS_REQUIRED = "server_terms_required"


def _terms_error() -> HTTPException:
    return _error(403, TERMS_REQUIRED,
                  "The owner of this Legion API key has to accept the updated Terms of Service and Privacy Policy "
                  "at the Legion Market before the key can be used")


# --- how often one key may call ------------------------------------------------------

class RateLimiter:
    """A token bucket per key, in this process: ``rate`` calls a minute with
    bursts of ``burst``. Several Gateway processes each keep their own bucket,
    so the effective limit is a multiple of it — a guard against a runaway
    Hub, not an exact quota (the wallet is the quota)."""

    def __init__(self, rate_per_min: float, burst: int) -> None:
        self.rate = max(rate_per_min, 0.0) / 60.0
        self.burst = max(burst, 1)
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key_id: str, *, now: float | None = None) -> float:
        """0 when the call may go now, else the seconds to wait."""
        if self.rate <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key_id, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - last) * self.rate)
            if tokens >= 1:
                self._buckets[key_id] = (tokens - 1, now)
                return 0.0
            self._buckets[key_id] = (tokens, now)
            if len(self._buckets) > 10_000:  # forget idle keys rather than grow forever
                cutoff = now - self.burst / self.rate
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] >= cutoff}
            return (1 - tokens) / self.rate


limiter = RateLimiter(config.KEY_RATE_PER_MIN, config.KEY_RATE_BURST)


# --- Market ----------------------------------------------------------------------------

#: digest of a key -> (expires at, Caller). Only accepted keys are cached, and
#: only briefly: a revoked key still reaches no paid call (``reserve`` asks
#: Market every time), it can at most list tools for KEY_CACHE_SEC.
_accepted: dict[str, tuple[float, Caller]] = {}


def reset_cache() -> None:
    _accepted.clear()


async def _market_post(path: str, payload: dict[str, Any]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SEC) as client:
        return await client.post(
            f"{config.MARKET_API_URL.rstrip('/')}/internal/wallet/{path}",
            json=payload,
            headers={_TOKEN_HEADER: config.MARKET_API_TOKEN},
        )


def _market_detail(response: httpx.Response) -> dict[str, Any]:
    try:
        detail = response.json().get("detail")
    except Exception:  # noqa: BLE001 - any unreadable body is just "no detail"
        return {}
    return detail if isinstance(detail, dict) else {}


async def authorize(key: str) -> Caller:
    """The account behind ``key``; 401 when Market does not accept it, 503 when Market is down."""
    digest = _digest(key)
    cached = _accepted.get(digest)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]
    try:
        response = await _market_post("authorize", {"api_key": key})
    except httpx.RequestError:
        raise _error(503, "wallet_unavailable", "Legion credits cannot be checked right now") from None
    if response.status_code == 401:
        _accepted.pop(digest, None)
        raise _error(401, "key_rejected", "The Legion API key is invalid or revoked")
    if response.status_code == 403 and _market_detail(response).get("code") == TERMS_REQUIRED:
        # Not cached: the owner accepts the updated terms and the next call works.
        _accepted.pop(digest, None)
        raise _terms_error()
    if response.status_code != 200:
        logger.warning("account key authorize: Market answered %s", response.status_code)
        raise _error(503, "wallet_unavailable", "Legion credits cannot be checked right now")
    body = response.json()
    caller = Caller(kind="account", api_key=key, account_id=str(body.get("account_id")),
                    key_prefix=body.get("key_prefix"))
    _accepted[digest] = (time.monotonic() + config.KEY_CACHE_SEC, caller)
    return caller


async def caller_for_key(key: str) -> Caller:
    """Rate-limit first (a flood never reaches Market), then authorize."""
    wait = limiter.allow(_digest(key))
    if wait > 0:
        raise HTTPException(
            status_code=429,
            detail={"code": "rate_limited", "message": "Too many MCP calls with this key", "retry_after": round(wait, 1)},
            headers={"Retry-After": str(max(1, int(wait + 0.999)))},
        )
    return await authorize(key)


async def reserve(caller: Caller, *, call_id: UUID, market_item_id: str | None, market_slug: str | None,
                  tool: str) -> dict[str, Any]:
    """Take the call's price from the central wallet before the upstream MCP is called."""
    try:
        response = await _market_post("reserve", {
            "api_key": caller.api_key, "call_id": str(call_id), "market_item_id": market_item_id,
            "market_slug": market_slug, "tool": tool,
        })
    except httpx.RequestError:
        raise _error(503, "wallet_unavailable", "Legion credits cannot be checked right now") from None
    if response.status_code == 200:
        return response.json()
    detail = _market_detail(response)
    code = detail.get("code")
    if response.status_code == 401:
        _accepted.pop(_digest(caller.api_key or ""), None)
        raise _error(401, "key_rejected", "The Legion API key is invalid or revoked")
    if response.status_code == 403 and code == TERMS_REQUIRED:
        _accepted.pop(_digest(caller.api_key or ""), None)
        raise _terms_error()
    if response.status_code == 403 and code == SERVER_TERMS_REQUIRED:
        # Terms §6: the server's own terms, accepted in Market before its first call.
        raise HTTPException(status_code=403, detail=detail)
    if response.status_code == 402:
        extra = {k: detail[k] for k in ("required_mcr", "available_mcr") if k in detail}
        raise _error(402, code or "insufficient_credits", detail.get("message") or "Not enough Legion credits", **extra)
    if response.status_code == 404:
        raise _error(404, "unknown_item", "No such MCP in the Legion catalogue")
    logger.warning("account key reserve: Market answered %s", response.status_code)
    raise _error(503, "wallet_unavailable", "Legion credits cannot be checked right now")


async def _finish(path: str, call_id: UUID, note: str | None = None) -> None:
    """Settle or refund, a few tries: a lost settle leaves the call charged
    (the reservation already took the money), a lost refund overcharges —
    worth retrying, never worth failing the caller's answer over."""
    payload = {"call_id": str(call_id), **({"note": note[:500]} if note else {})}
    for attempt in range(3):
        try:
            response = await _market_post(path, payload)
            if response.status_code == 200:
                return
            logger.warning("account key %s %s: Market answered %s", path, call_id, response.status_code)
        except httpx.RequestError as exc:
            logger.warning("account key %s %s: Market unreachable (%s)", path, call_id, type(exc).__name__)
        await asyncio.sleep(0.2 * (attempt + 1))
    logger.error("account key %s %s failed after retries", path, call_id)


async def settle(call_id: UUID) -> None:
    await _finish("settle", call_id)


async def refund(call_id: UUID, note: str) -> None:
    await _finish("refund", call_id, note)
