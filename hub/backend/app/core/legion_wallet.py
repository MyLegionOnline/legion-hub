"""Market MCP on a self-hosted Hub, paid from the Legion account's central wallet
(internal design notes §3.10, stage 3b).

On our Hub, credits live in its own database (``app.core.mcp_credits``) and
the MCP Gateway trusts it. A self-hosted Hub's database is its owner's — a
balance kept there would be one ``UPDATE`` away from any number — so with
``MCP_CREDITS_WALLET=legion`` such a Hub does not count Market MCP credits
at all. Each user's call goes to Legion's public Gateway
(``LEGION_MCP_GATEWAY_URL``) with that user's Legion API key (Settings →
Legion account, ``app.core.legion_link``); the Gateway reserves, settles or
refunds it in the account's "Own Hubs" wallet in Market. The local wallet is
not touched, and editing it changes nothing.

Personal MCP servers (links, images, stdio) are never billed and keep going
through the Hub's own Gateway (``MCP_GATEWAY_URL``).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional
from uuid import UUID

import httpx

from app.core import legion_auth, legion_link
from app.core.market_switch import DEFAULT_MCP_GATEWAY_URL
from app.core.mcp_credits import InsufficientCredits, McpCreditError, format_credits

logger = logging.getLogger(__name__)

#: ``hub`` — credits in this Hub's database (our Hub; the default);
#: ``legion`` — the central wallet in Market (a self-hosted Hub).
#: A Hub that is not ours defaults to the central wallet; Legion Cloud sets ``hub``.
MCP_CREDITS_WALLET = (os.getenv("MCP_CREDITS_WALLET", "legion") or "legion").strip().lower()
#: Legion's public MCP Gateway.
LEGION_MCP_GATEWAY_URL = (os.getenv("LEGION_MCP_GATEWAY_URL", "").strip() or DEFAULT_MCP_GATEWAY_URL)
#: How long a balance read from Market is reused for hiding unaffordable tools.
BALANCE_CACHE_SEC = float(os.getenv("LEGION_WALLET_CACHE_SEC", "30"))


def central() -> bool:
    """Market MCP calls are paid from the Legion account (and only a Hub with
    local users has per-user Legion keys to pay with)."""
    return MCP_CREDITS_WALLET == "legion" and legion_link.available()


def gateway_url() -> str:
    from app.core.mcp_gateway import MCP_GATEWAY_URL

    return (LEGION_MCP_GATEWAY_URL or MCP_GATEWAY_URL).rstrip("/")


# --- refusals, in the words the room and the agent see -------------------------------

class NoLegionAccount(McpCreditError):
    """The user has not linked a Legion account (or Legion rejected its key)."""

    reason = "no_account"

    def __init__(self, *, rejected: bool = False) -> None:
        super().__init__("no linked Legion account to pay for a Market MCP call")
        self.rejected = rejected

    def user_message(self) -> str:
        if self.rejected:
            return ("Legion no longer accepts the account key: link the account again "
                    "(Settings → Profile → Legion account) to use Market MCPs.")
        return ("Market MCPs on this Hub are paid with Legion account credits: "
                "link the account in Settings → Profile → Legion account.")


class CentralInsufficient(InsufficientCredits):
    """The Own Hubs wallet does not cover the call (Market said so)."""

    def user_message(self) -> str:
        return (f"Not enough Legion credits in the \"Own Hubs\" wallet (need "
                f"{format_credits(self.required_mcr)}, available {format_credits(self.available_mcr)}). "
                "You can top up in your Legion profile.")


class CentralUnavailable(McpCreditError):
    """Legion could not check the credits; the call is refused, never free."""

    reason = "no_price"

    def user_message(self) -> str:
        return "Legion credits cannot be checked right now — the Market MCP call was rejected. Try again later."


class InstallNotSold(McpCreditError):
    reason = "plan_zero"

    def user_message(self) -> str:
        return "Paid installs from Market are not yet available on your own Hub."


def refusal_from_gateway(status_code: int, detail: Any) -> McpCreditError | None:
    """What a Gateway refusal of a key-paid call means, or None for an ordinary failure."""
    code = detail.get("code") if isinstance(detail, dict) else None
    if status_code == 401:
        return NoLegionAccount(rejected=True)
    if status_code == 402 and code == "install_required":
        return InstallNotSold("item is bought once at install")
    if status_code == 402:
        required = int(detail.get("required_mcr") or 0) if isinstance(detail, dict) else 0
        available = int(detail.get("available_mcr") or 0) if isinstance(detail, dict) else 0
        return CentralInsufficient(required, available)
    if status_code == 503 and code == "wallet_unavailable":
        return CentralUnavailable("Legion wallet unavailable")
    return None


# --- the user's key ----------------------------------------------------------------------

async def key_for(user_id: UUID) -> str:
    """The user's Legion API key, decrypted; :class:`NoLegionAccount` without a live link."""
    from app.core.mcp_credits import _short_session_factory
    from app.models.legion_link import LegionLink

    async with _short_session_factory()() as db:
        row = await db.get(LegionLink, user_id)
    if row is None:
        raise NoLegionAccount()
    if row.status != "linked":
        raise NoLegionAccount(rejected=True)
    try:
        return legion_link.stored_key(row).reveal()
    except legion_link.LegionLinkError:
        raise NoLegionAccount(rejected=True) from None


async def optional_key(user_id: UUID) -> Optional[str]:
    try:
        return await key_for(user_id)
    except NoLegionAccount:
        return None


# --- the balance ---------------------------------------------------------------------------

_balances: dict[UUID, tuple[float, int]] = {}


def forget_balance(user_id: UUID) -> None:
    _balances.pop(user_id, None)


async def fetch_wallet(key: str, cursor: Optional[str] = None) -> dict[str, Any]:
    """Market's ``GET /api/wallet`` for the key: the account and its Own Hubs wallet.
    Raises ``legion_link.LegionLinkError`` (``key_rejected`` / ``unreachable`` / ``bad_answer``)."""
    try:
        async with httpx.AsyncClient(timeout=legion_auth.LEGION_HTTP_TIMEOUT_SEC) as client:
            response = await client.get(
                f"{legion_auth.LEGION_INTERNAL_URL}/api/wallet",
                params={"cursor": cursor} if cursor else None,
                headers={"Authorization": f"Bearer {key}"},
            )
    except httpx.HTTPError as exc:
        raise legion_link.LegionLinkError("Legion is unreachable. Try again shortly.", status=503,
                                          code="unreachable") from exc
    if response.status_code in (401, 403):
        raise legion_link.LegionLinkError("Legion did not accept this key.", code="key_rejected")
    try:
        body = response.json() if response.status_code == 200 else None
        wallet = body["wallet"] if body else None
        int(wallet["balance_mcr"])
    except (ValueError, KeyError, TypeError):
        raise legion_link.LegionLinkError("Legion gave an unexpected answer.", status=502,
                                          code="bad_answer") from None
    return body


async def balance_mcr(user_id: UUID) -> Optional[int]:
    """The Own Hubs balance for hiding unaffordable tools; None when unknown
    (no key, Market down) — then nothing is hidden and the Gateway decides."""
    cached = _balances.get(user_id)
    if cached is not None and cached[0] > time.monotonic():
        return cached[1]
    key = await optional_key(user_id)
    if key is None:
        return None
    try:
        body = await fetch_wallet(key)
    except legion_link.LegionLinkError as exc:
        logger.info("legion wallet: balance of user %s unknown (%s)", user_id, exc.code)
        return None
    value = int(body["wallet"]["balance_mcr"])
    _balances[user_id] = (time.monotonic() + BALANCE_CACHE_SEC, value)
    return value
