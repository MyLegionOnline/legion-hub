"""Money units, wallet and domain errors for marketplace MCP call credits.

Stage 0 of ``internal design notes`` built the units and the errors.
Stage 3 added the wallet itself: lazy weekly rolling, the ledger, and the
``/me/usage`` read path. Stage 4 adds the charging path: the Market price
cache (:func:`resolve_price_mcr`, :func:`store_price_snapshot`,
:func:`refresh_price_snapshot`) and reserve/settle/refund on an actual tool
call (:func:`reserve_call`, :func:`settle_call`, :func:`refund_call`),
composed by ``execute_mcp_tool`` in ``app.core.mcp_gateway``.

Internally a balance is always an integer count of **millicredits** (``mcr``);
``1 credit = 1000 mcr``.  Floats never hold a balance: ``0.3 + 0.6`` already
drifts in binary, and over millions of tool calls the drift is real money.
Everything crossing the boundary to a human — API payloads, the Admin Center,
plan limits — speaks *credits*, and this module is the only place that
converts between the two.  If you find yourself writing ``* 1000`` anywhere
else, import from here instead.

Every wallet mutation (:func:`topup`, :func:`adjust`, :func:`wallet_summary`,
:func:`reserve_call`, :func:`settle_call`, :func:`refund_call`,
:func:`resolve_price_mcr`) opens its **own** short-lived DB session and
commits immediately, rather than depending on a session handed in by the
caller — the call sites in ``proxy_session.py``/``user_sessions.py``
frequently don't have one to hand, and holding the caller's transaction open
across an MCP round trip would violate ``internal design notes``. The
lower-level functions (:func:`get_or_create_wallet`, :func:`roll_period_if_due`,
:func:`apply_plan_change`, :func:`get_price_snapshot`) take an explicit
``db`` and don't commit, so a caller that already has a session open — like
``room_mcp_tools`` during discovery — can compose them without an extra
round trip.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, ROUND_HALF_UP
from typing import Optional, Union
from uuid import UUID, uuid4

from sqlalchemy import func, select, tuple_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

# ---------- Units --------------------------------------------------------------

MCR_PER_CREDIT = 1000

#: 1 mcr is the smallest representable amount, so three decimal places is a
#: lossless representation of any balance in credits.
CREDIT_INPUT_PLACES = 3
#: Humans read two decimal places; used for display only, never for storage.
CREDIT_DISPLAY_PLACES = 2

#: Guard against values that could not survive the BIGINT columns anyway.
MAX_MCR = 2**63 - 1

_MCR_PER_CREDIT_DEC = Decimal(MCR_PER_CREDIT)
_DISPLAY_QUANTUM = Decimal(1).scaleb(-CREDIT_DISPLAY_PLACES)

CreditAmount = Union[int, float, str, Decimal]


def _to_decimal(value: CreditAmount) -> Decimal:
    """Parse a credit amount into an exact decimal.

    Floats go through ``str()`` on purpose: ``Decimal(0.3)`` is
    ``0.299999999999999988897769753748...`` while ``Decimal(str(0.3))`` is
    exactly ``0.3``, which is what the caller typed and what they mean.
    """
    if isinstance(value, bool):  # bool is an int subclass; never a money amount
        raise TypeError("credit amount must be a number, got bool")
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, int):
        dec = Decimal(value)
    elif isinstance(value, float):
        dec = Decimal(str(value))
    elif isinstance(value, str):
        try:
            dec = Decimal(value.strip())
        except InvalidOperation:
            raise ValueError(f"not a credit amount: {value!r}") from None
    else:
        raise TypeError(f"credit amount must be a number, got {type(value).__name__}")
    if not dec.is_finite():
        raise ValueError("credit amount must be finite")
    return dec


def credits_to_mcr(value: CreditAmount) -> int:
    """Convert credits to millicredits, rounding **down** to the whole mcr.

    Rounding is toward negative infinity, not toward zero, so the rule is the
    same sentence for every sign: the stored amount is never larger than what
    was asked for.  A fraction finer than 0.001 credits is below the resolution
    of the system and is dropped, not accumulated.
    """
    dec = _to_decimal(value) * _MCR_PER_CREDIT_DEC
    mcr = int(dec.to_integral_value(rounding=ROUND_FLOOR))
    if abs(mcr) > MAX_MCR:
        raise ValueError(f"credit amount out of range: {value!r}")
    return mcr


def mcr_to_credits(value: int) -> float:
    """Convert millicredits to credits, exactly.

    Lossless (every mcr is a whole 0.001 of a credit), so this is what API
    payloads and arithmetic use.  Use :func:`format_credits` for display.
    """
    return float(Decimal(int(value)) / _MCR_PER_CREDIT_DEC)


def format_credits(value: int) -> float:
    """Round millicredits to the two decimal places shown to a human.

    Display only — never feed the result back into a balance, or the rounding
    becomes a slow leak.
    """
    dec = (Decimal(int(value)) / _MCR_PER_CREDIT_DEC).quantize(
        _DISPLAY_QUANTUM, rounding=ROUND_HALF_UP
    )
    return float(dec)


# ---------- Domain errors ------------------------------------------------------

#: Labels for ``legion_mcp_credits_denied_total{reason}`` (stage 8).  Every
#: denial must map onto exactly one of these, so operators can tell "user is
#: out of money" apart from "we failed to price the call".
DENIAL_REASONS: tuple[str, ...] = ("insufficient", "no_price", "no_owner", "plan_zero", "no_account")


class McpCreditError(RuntimeError):
    """Base class for anything that stops a marketplace MCP call being paid for."""

    #: One of :data:`DENIAL_REASONS`; used as a metric label and in logs.
    reason: str = "insufficient"

    def user_message(self) -> str:
        """Human-readable reason, shown in the room and in tool results."""
        return "Not enough MCP credits."


class InsufficientCredits(McpCreditError):
    """The wallet does not cover the price of this call."""

    reason = "insufficient"

    def __init__(
        self,
        required_mcr: int,
        available_mcr: int,
        *,
        user_id: Optional[UUID] = None,
    ) -> None:
        super().__init__(
            f"insufficient MCP credits: need {required_mcr} mcr, have {available_mcr} mcr"
        )
        self.required_mcr = int(required_mcr)
        self.available_mcr = int(available_mcr)
        self.user_id = user_id

    def user_message(self) -> str:
        return (
            f"Not enough MCP credits (need {format_credits(self.required_mcr)}, "
            f"available {format_credits(self.available_mcr)})."
        )


class PriceUnknown(McpCreditError):
    """No price snapshot for this item, and Market could not be asked.

    A paid call whose price we cannot establish is refused rather than let
    through: a free paid call is a silent loss, a refused one is a bug report.
    """

    reason = "no_price"

    def __init__(
        self,
        *,
        market_item_id: Optional[UUID] = None,
        market_slug: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> None:
        target = market_slug or (str(market_item_id) if market_item_id else "unknown item")
        if tool_name:
            target = f"{target}/{tool_name}"
        super().__init__(f"no price snapshot for {target}")
        self.market_item_id = market_item_id
        self.market_slug = market_slug
        self.tool_name = tool_name

    def user_message(self) -> str:
        return "The MCP call price is unknown — the call was rejected."


class NoWalletOwner(McpCreditError):
    """A billable call arrived without an owner to charge.

    Guards the ``mcp_user_id or UUID(int=0)`` fallback in the session code:
    an unknown owner must be a hard refusal, not a free ride.
    """

    reason = "no_owner"

    def __init__(self, *, market_slug: Optional[str] = None, tool_name: Optional[str] = None) -> None:
        super().__init__(
            "billable MCP call has no owner to charge"
            + (f" (item={market_slug})" if market_slug else "")
        )
        self.market_slug = market_slug
        self.tool_name = tool_name

    def user_message(self) -> str:
        return "The MCP call has no owner to charge credits to — the call was rejected."


class InstallPurchaseDenied(InsufficientCredits):
    """The wallet does not cover a one-time Market item install (Stage 4A.5).

    Same wallet math as :class:`InsufficientCredits` — reuses its
    ``required_mcr``/``available_mcr``/:func:`format_credits` — but a
    distinct ``user_message`` so a support ticket about "why can't I install
    this" doesn't read like a normal call-spend failure. When the cause is
    the plan itself (weekly grant is 0 — free tier), ``reason`` becomes the
    ``"plan_zero"`` label that :data:`DENIAL_REASONS` has reserved but no
    caller has emitted yet, and the message names the tariff instead of
    saying "insufficient".
    """

    def __init__(
        self,
        required_mcr: int,
        available_mcr: int,
        *,
        user_id: Optional[UUID] = None,
        plan_zero: bool = False,
    ) -> None:
        super().__init__(required_mcr, available_mcr, user_id=user_id)
        self.plan_zero = plan_zero
        if plan_zero:
            self.reason = "plan_zero"

    def user_message(self) -> str:
        if self.plan_zero:
            return (
                "Installing a paid add-on is unavailable on the current plan "
                "(0 MCP credits per week) — upgrade the plan to buy it."
            )
        return (
            f"Not enough credits to purchase (need {format_credits(self.required_mcr)}, "
            f"available {format_credits(self.available_mcr)})."
        )


# ---------- Wallet: weekly window, grant, ledger --------------------------------

WEEK = timedelta(days=7)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: datetime) -> datetime:
    """SQLite in tests hands back naive datetimes; Postgres never does."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _trunc_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


@dataclass(frozen=True)
class WalletFields:
    """Pure snapshot of the columns :func:`roll_period` and :func:`apply_plan_change`
    read and write. Kept separate from the ORM row so the rolling/grant logic
    — the part with the bugs — is unit-testable without a database.
    """

    grant_balance_mcr: int
    purchased_balance_mcr: int
    period_grant_mcr: int
    period_spent_mcr: int
    lifetime_spent_mcr: int
    period_started_at: datetime
    period_ends_at: datetime

    @property
    def balance_mcr(self) -> int:
        return self.grant_balance_mcr + self.purchased_balance_mcr


@dataclass(frozen=True)
class LedgerDraft:
    """A ledger row not yet written; ``balance_after_mcr`` is always filled."""

    kind: str
    amount_mcr: int
    grant_delta_mcr: int
    purchased_delta_mcr: int
    balance_after_mcr: int
    note: Optional[str] = None
    #: Stage 8 Stripe seam — see ``topup``. Never set by ``roll_period`` or
    #: ``apply_plan_change``, only by an explicit paid top-up.
    source: Optional[str] = None
    external_ref: Optional[str] = None


def roll_period(
    fields: WalletFields, weekly_grant_mcr: int, now: datetime
) -> tuple[WalletFields, list[LedgerDraft]]:
    """Advance ``fields`` to the window containing ``now``, lazily.

    Pure function — no I/O, no clock reads. Mirrors the pseudocode in the
    plan exactly: any remaining grant expires on the boundary it was due, the
    window rolls forward one week at a time without accumulating skipped
    weeks, and exactly one fresh grant is issued once the loop lands on the
    window that actually contains ``now`` — whether that took one roll or
    five. ``purchased_balance_mcr`` is never touched here.
    """
    entries: list[LedgerDraft] = []
    grant = fields.grant_balance_mcr
    purchased = fields.purchased_balance_mcr
    started = fields.period_started_at
    ends = fields.period_ends_at
    spent = fields.period_spent_mcr
    period_grant = fields.period_grant_mcr

    rolled = False
    while now >= ends:
        rolled = True
        if grant > 0:
            entries.append(
                LedgerDraft(
                    kind="expire",
                    amount_mcr=-grant,
                    grant_delta_mcr=-grant,
                    purchased_delta_mcr=0,
                    balance_after_mcr=purchased,
                    note="weekly grant expired",
                )
            )
            grant = 0
        started = ends
        ends = ends + WEEK
        spent = 0

    if rolled:
        period_grant = max(0, int(weekly_grant_mcr))
        grant = period_grant
        if period_grant:
            entries.append(
                LedgerDraft(
                    kind="grant",
                    amount_mcr=period_grant,
                    grant_delta_mcr=period_grant,
                    purchased_delta_mcr=0,
                    balance_after_mcr=grant + purchased,
                    note="weekly grant",
                )
            )

    new_fields = WalletFields(
        grant_balance_mcr=grant,
        purchased_balance_mcr=purchased,
        period_grant_mcr=period_grant,
        period_spent_mcr=spent,
        lifetime_spent_mcr=fields.lifetime_spent_mcr,
        period_started_at=started,
        period_ends_at=ends,
    )
    return new_fields, entries


def apply_plan_change(
    fields: WalletFields, new_weekly_grant_mcr: int
) -> tuple[WalletFields, list[LedgerDraft]]:
    """Mid-window plan change: top up the grant bucket if it went up, leave it
    alone if it went down.

    A downgrade never claws back a grant already issued this window — that
    would read as "charged for upgrading in reverse". An upgrade tops up the
    difference immediately rather than making the user wait for the next
    roll. See the plan's "Changing the plan in the middle of a window" note.
    """
    new_weekly_grant_mcr = max(0, int(new_weekly_grant_mcr))
    if new_weekly_grant_mcr <= fields.period_grant_mcr:
        return fields, []
    delta = new_weekly_grant_mcr - fields.period_grant_mcr
    new_grant_balance = fields.grant_balance_mcr + delta
    entry = LedgerDraft(
        kind="grant",
        amount_mcr=delta,
        grant_delta_mcr=delta,
        purchased_delta_mcr=0,
        balance_after_mcr=new_grant_balance + fields.purchased_balance_mcr,
        note="plan change",
    )
    new_fields = replace(
        fields, grant_balance_mcr=new_grant_balance, period_grant_mcr=new_weekly_grant_mcr
    )
    return new_fields, [entry]


def _fields_from_wallet(wallet: "McpCreditWallet") -> WalletFields:
    return WalletFields(
        grant_balance_mcr=wallet.grant_balance_mcr,
        purchased_balance_mcr=wallet.purchased_balance_mcr,
        period_grant_mcr=wallet.period_grant_mcr,
        period_spent_mcr=wallet.period_spent_mcr,
        lifetime_spent_mcr=wallet.lifetime_spent_mcr,
        period_started_at=_aware(wallet.period_started_at),
        period_ends_at=_aware(wallet.period_ends_at),
    )


def _apply_fields(wallet: "McpCreditWallet", fields: WalletFields, *, now: datetime) -> None:
    wallet.grant_balance_mcr = fields.grant_balance_mcr
    wallet.purchased_balance_mcr = fields.purchased_balance_mcr
    wallet.period_grant_mcr = fields.period_grant_mcr
    wallet.period_spent_mcr = fields.period_spent_mcr
    wallet.lifetime_spent_mcr = fields.lifetime_spent_mcr
    wallet.period_started_at = fields.period_started_at
    wallet.period_ends_at = fields.period_ends_at
    wallet.updated_at = now


def _add_ledger_rows(
    db: AsyncSession, user_id: UUID, entries: list[LedgerDraft], *, actor: str = "system"
) -> None:
    from app.models.mcp_credits import McpCreditLedger

    for draft in entries:
        db.add(
            McpCreditLedger(
                user_id=user_id,
                kind=draft.kind,
                amount_mcr=draft.amount_mcr,
                grant_delta_mcr=draft.grant_delta_mcr,
                purchased_delta_mcr=draft.purchased_delta_mcr,
                balance_after_mcr=draft.balance_after_mcr,
                status="settled",
                actor=actor,
                note=draft.note,
                source=draft.source,
                external_ref=draft.external_ref,
            )
        )


async def get_or_create_wallet(db: AsyncSession, user_id: UUID, *, now: Optional[datetime] = None) -> "McpCreditWallet":
    """Fetch the wallet row for ``user_id``, creating it on first touch.

    A brand new wallet is created with its window already "due"
    (``period_started_at == period_ends_at``), so the very next
    :func:`roll_period_if_due` call issues the first grant through the exact
    same code path as every later roll — no separate "initial grant" branch
    to keep in sync with it.
    """
    from app.models.mcp_credits import McpCreditWallet

    result = await db.execute(
        select(McpCreditWallet).where(McpCreditWallet.user_id == user_id).with_for_update()
    )
    wallet = result.scalar_one_or_none()
    if wallet is not None:
        return wallet

    started = _trunc_to_hour(now or _utcnow())
    wallet = McpCreditWallet(user_id=user_id, period_started_at=started, period_ends_at=started)
    db.add(wallet)
    try:
        await db.flush()
    except IntegrityError:
        # Lost a first-touch race to a concurrent request; use the winner's row.
        await db.rollback()
        result = await db.execute(
            select(McpCreditWallet).where(McpCreditWallet.user_id == user_id).with_for_update()
        )
        wallet = result.scalar_one()
    return wallet


async def roll_period_if_due(
    db: AsyncSession,
    wallet: "McpCreditWallet",
    weekly_grant_mcr: int,
    *,
    now: Optional[datetime] = None,
    actor: str = "system",
) -> "McpCreditWallet":
    """Lazily roll ``wallet`` forward to the window containing ``now``.

    A no-op when the current window hasn't ended yet — which is the common
    case, since this runs on every wallet read as well as every charge.
    Writes ledger rows and flushes; does not commit, so callers composing a
    larger transaction (Stage 4's reserve/settle) control the boundary.
    """
    now = now or _utcnow()
    fields = _fields_from_wallet(wallet)
    if now < fields.period_ends_at:
        return wallet
    new_fields, entries = roll_period(fields, weekly_grant_mcr, now)
    _apply_fields(wallet, new_fields, now=now)
    db.add(wallet)
    _add_ledger_rows(db, wallet.user_id, entries, actor=actor)
    await db.flush()
    return wallet


async def _weekly_grant_mcr_for(user_id: UUID, db: AsyncSession) -> int:
    from app.core.workspace_quota import get_effective_limits

    limits = await get_effective_limits(user_id, db)
    return credits_to_mcr(limits.get("mcp_weekly_credits") or 0)


async def apply_plan_change_if_needed(
    db: AsyncSession,
    wallet: "McpCreditWallet",
    weekly_grant_mcr: int,
    *,
    now: Optional[datetime] = None,
    actor: str = "system",
) -> "McpCreditWallet":
    """Top up the current window's grant if the effective plan grant rose.

    Wired in next to every :func:`roll_period_if_due` call so a plan tier
    change, a limits override edit, or a fix to the plan table itself (see
    Stage 6's jsonb/json migration bug) is picked up on the very next wallet
    read/charge instead of waiting for the natural weekly roll. A no-op when
    the effective grant is unchanged or lower — mirrors :func:`apply_plan_change`,
    which never claws back a grant already issued this window.
    """
    fields = _fields_from_wallet(wallet)
    new_fields, entries = apply_plan_change(fields, weekly_grant_mcr)
    if not entries:
        return wallet
    _apply_fields(wallet, new_fields, now=now or _utcnow())
    db.add(wallet)
    _add_ledger_rows(db, wallet.user_id, entries, actor=actor)
    await db.flush()
    return wallet


def _wallet_summary_dict(wallet: "McpCreditWallet", weekly_grant_credits: float) -> dict:
    return {
        "balance": mcr_to_credits(wallet.balance_mcr),
        "grant_remaining": mcr_to_credits(wallet.grant_balance_mcr),
        "grant_total": mcr_to_credits(wallet.period_grant_mcr),
        "purchased": mcr_to_credits(wallet.purchased_balance_mcr),
        "spent_this_period": mcr_to_credits(wallet.period_spent_mcr),
        "weekly_grant": float(weekly_grant_credits),
        "period_started_at": wallet.period_started_at.isoformat(),
        "period_ends_at": wallet.period_ends_at.isoformat(),
    }


def _short_session_factory():
    from sqlalchemy.orm import sessionmaker

    from app.db.session import engine

    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def _rolled_wallet(user_id: UUID) -> tuple["McpCreditWallet", int]:
    """Shared body of :func:`wallet_summary` and :func:`get_wallet_balance_mcr`.

    Rolls the window if due (so a user who hasn't touched anything in 8 days
    still sees a fresh grant on next login) and commits that roll — a read
    that silently failed to persist its own state change would re-roll, and
    re-expire an already-expired grant, on every subsequent read.
    """
    async_session = _short_session_factory()
    async with async_session() as db:
        weekly_grant_mcr = await _weekly_grant_mcr_for(user_id, db)
        wallet = await get_or_create_wallet(db, user_id)
        wallet = await roll_period_if_due(db, wallet, weekly_grant_mcr)
        wallet = await apply_plan_change_if_needed(db, wallet, weekly_grant_mcr)
        await db.commit()
        return wallet, weekly_grant_mcr


async def wallet_summary(user_id: UUID) -> dict:
    """The block ``GET /api/v1/users/me/usage`` merges in under ``mcp_credits``."""
    wallet, weekly_grant_mcr = await _rolled_wallet(user_id)
    return _wallet_summary_dict(wallet, mcr_to_credits(weekly_grant_mcr))


async def get_wallet_balance_mcr(user_id: UUID) -> int:
    """Rolled balance in mcr — ``room_mcp_tools`` uses this to decide what to hide."""
    wallet, _ = await _rolled_wallet(user_id)
    return wallet.balance_mcr


# ---------- Price snapshot: Market is the source of truth, Hub caches it -------


async def get_price_snapshot(db: AsyncSession, market_item_id: UUID) -> Optional["McpPriceSnapshot"]:
    """Read the cached Market price for one item, using the caller's own session.

    For ``room_mcp_tools``, which already has a request-scoped session open
    for a dozen other queries in the same discovery pass — no need for the
    short-session dance here since this is a plain read with no mutation.
    """
    from app.models.mcp_credits import McpPriceSnapshot

    return await db.get(McpPriceSnapshot, market_item_id)


def price_from_snapshot(
    snapshot: Optional["McpPriceSnapshot"],
    tool_name: str,
    *,
    market_item_id: Optional[UUID] = None,
    market_slug: Optional[str] = None,
) -> int:
    """Millicredit price of one tool call. Raises :class:`PriceUnknown` if uncached."""
    if snapshot is None:
        raise PriceUnknown(market_item_id=market_item_id, market_slug=market_slug, tool_name=tool_name)
    overrides = snapshot.tool_price_overrides or {}
    price = overrides.get(tool_name, snapshot.call_price_mcr)
    return max(0, int(price))


async def get_price_snapshot_by_slug(db: AsyncSession, market_slug: str) -> Optional["McpPriceSnapshot"]:
    """Look up the cached price snapshot by Market slug rather than item id.

    Used where only the slug is known — the legacy ``/internal/admin/mcp-quota``
    alias (Stage 7) predates carrying a ``market_item_id`` in its payload.
    """
    from app.models.mcp_credits import McpPriceSnapshot

    return (
        await db.execute(
            select(McpPriceSnapshot).where(McpPriceSnapshot.market_slug == market_slug)
        )
    ).scalars().first()


async def resolve_price_mcr(
    *, market_item_id: Optional[UUID], market_slug: Optional[str], tool_name: str
) -> int:
    """Look up one tool's mcr price for a charging call, opening its own session.

    ``execute_mcp_tool`` frequently has no ``db`` to hand (see the module
    docstring), so — unlike :func:`get_price_snapshot` — this never depends
    on one.
    """
    if market_item_id is None:
        raise PriceUnknown(market_slug=market_slug, tool_name=tool_name)
    async_session = _short_session_factory()
    async with async_session() as db:
        snapshot = await get_price_snapshot(db, market_item_id)
    return price_from_snapshot(
        snapshot, tool_name, market_item_id=market_item_id, market_slug=market_slug
    )


async def store_price_snapshot(
    db: AsyncSession,
    *,
    market_item_id: UUID,
    market_slug: str,
    call_price_mcr: int,
    tool_price_overrides: Optional[dict] = None,
) -> "McpPriceSnapshot":
    """Upsert the cached price from an already-fetched Market catalog item.

    Used at install time, where the caller just fetched the item from Market
    and a second round-trip would be wasted. Flushes but does not commit —
    the caller's own transaction decides the boundary.
    """
    from app.models.mcp_credits import McpPriceSnapshot

    snapshot = await db.get(McpPriceSnapshot, market_item_id)
    if snapshot is None:
        snapshot = McpPriceSnapshot(market_item_id=market_item_id, market_slug=market_slug)
    snapshot.market_slug = market_slug
    snapshot.call_price_mcr = max(0, int(call_price_mcr))
    snapshot.tool_price_overrides = {
        str(key): max(0, int(value)) for key, value in (tool_price_overrides or {}).items()
    }
    snapshot.fetched_at = _utcnow()
    db.add(snapshot)
    await db.flush()
    return snapshot


async def refresh_price_snapshot(
    db: AsyncSession, *, market_item_id: UUID, market_slug: str
) -> "McpPriceSnapshot":
    """Re-fetch one item's price from Market and update the cached snapshot.

    Raises whatever ``app.core.market_client`` raises on failure — the caller
    (``room_mcp_tools`` during discovery) decides whether to fall back to a
    stale snapshot or hide the item.
    """
    from app.core import market_client

    catalog_item = await market_client.get_catalog_item(market_slug)
    return await store_price_snapshot(
        db,
        market_item_id=market_item_id,
        market_slug=market_slug,
        call_price_mcr=int(catalog_item.get("call_price_mcr") or 0),
        tool_price_overrides=catalog_item.get("tool_price_overrides") or {},
    )


async def refresh_price_snapshot_by_slug(db: AsyncSession, *, market_slug: str) -> "McpPriceSnapshot":
    """Re-fetch and cache one item's price from Market, looking it up by slug.

    Used by the Admin Center's manual "refresh price" action
    (internal design notes Stage 4.5), where the operator only has the
    slug at hand — unlike :func:`refresh_price_snapshot`, which assumes the
    caller already knows ``market_item_id`` from a prior discovery/install
    round trip. Raises whatever ``market_client`` raises (unreachable, 404)
    and does not commit — the caller decides the transaction boundary.
    """
    from app.core import market_client

    catalog_item = await market_client.get_catalog_item(market_slug)
    market_item_id = UUID(str(catalog_item["id"]))
    return await store_price_snapshot(
        db,
        market_item_id=market_item_id,
        market_slug=market_slug,
        call_price_mcr=int(catalog_item.get("call_price_mcr") or 0),
        tool_price_overrides=catalog_item.get("tool_price_overrides") or {},
    )


# ---------- Charging one call: reserve, then settle or refund ------------------


async def verify_reserved_call(
    *,
    call_id: Optional[UUID],
    user_id: UUID,
    market_item_id: Optional[UUID],
    market_slug: Optional[str],
    tool_name: str,
) -> None:
    """Confirm Hub actually reserved credits for this call (Stage 7's second
    line of defense — see ``/internal/mcp-credits/verify``).

    Called by Gateway right before it proxies a call to the upstream MCP
    server, so a request that skipped ``execute_mcp_tool``'s reservation step
    (e.g. a leaked service token hitting Gateway directly) is refused instead
    of executed for free. Personal MCPs (``market_item_id is None``) are
    never billed and always pass — mirrors the ``billable`` predicate in
    ``execute_mcp_tool``. A price that resolves to 0 needs no reservation,
    mirroring :func:`reserve_call`'s own no-op for ``price_mcr <= 0``.
    Raises :class:`PriceUnknown` or :class:`InsufficientCredits` — both are
    :class:`McpCreditError`, which the caller maps to HTTP 402.
    """
    if market_item_id is None:
        return
    price_mcr = await resolve_price_mcr(
        market_item_id=market_item_id, market_slug=market_slug, tool_name=tool_name
    )
    if price_mcr <= 0:
        return
    if call_id is None:
        raise InsufficientCredits(price_mcr, 0, user_id=user_id)

    from app.models.mcp_credits import McpCreditLedger

    async_session = _short_session_factory()
    async with async_session() as db:
        row = (
            await db.execute(
                select(McpCreditLedger).where(
                    McpCreditLedger.call_id == call_id,
                    McpCreditLedger.kind == "spend",
                )
            )
        ).scalar_one_or_none()
    if (
        row is None
        or row.user_id != user_id
        or row.tool_name != tool_name
        or row.status not in ("reserved", "settled")
    ):
        raise InsufficientCredits(price_mcr, 0, user_id=user_id)


async def reserve_call(
    user_id: UUID,
    price_mcr: int,
    *,
    call_id: UUID,
    installed_item_id: UUID,
    market_item_id: Optional[UUID],
    market_slug: Optional[str],
    tool_name: str,
    room_id: Optional[UUID] = None,
    session_id: Optional[UUID] = None,
    actor: str = "agent",
) -> None:
    """Reserve ``price_mcr`` against the wallet before a call leaves the Hub.

    Raises :class:`InsufficientCredits` and touches nothing if the wallet
    can't cover it — a no-op price (``<= 0``) skips the wallet entirely, so a
    free item never litters the ledger with zero-value rows. On success,
    writes one ``spend``/``reserved`` ledger row under ``call_id``; the
    caller settles or refunds it once the call's outcome is known (see the
    outcome table in the plan). The wallet row lock taken by
    :func:`get_or_create_wallet` serializes concurrent reservations for the
    same user — that's what keeps parallel tool calls from overdrawing the
    balance, not a hand-rolled atomic ``UPDATE``.
    """
    from app.models.mcp_credits import McpCreditLedger

    if price_mcr <= 0:
        return
    async_session = _short_session_factory()
    async with async_session() as db:
        weekly_grant_mcr = await _weekly_grant_mcr_for(user_id, db)
        wallet = await get_or_create_wallet(db, user_id)
        wallet = await roll_period_if_due(db, wallet, weekly_grant_mcr)
        wallet = await apply_plan_change_if_needed(db, wallet, weekly_grant_mcr)
        if wallet.balance_mcr < price_mcr:
            raise InsufficientCredits(price_mcr, wallet.balance_mcr, user_id=user_id)
        grant_delta = -min(wallet.grant_balance_mcr, price_mcr)
        purchased_delta = -(price_mcr + grant_delta)
        wallet.grant_balance_mcr += grant_delta
        wallet.purchased_balance_mcr += purchased_delta
        wallet.period_spent_mcr += price_mcr
        wallet.lifetime_spent_mcr += price_mcr
        wallet.updated_at = _utcnow()
        db.add(wallet)
        db.add(
            McpCreditLedger(
                user_id=user_id,
                kind="spend",
                amount_mcr=-price_mcr,
                grant_delta_mcr=grant_delta,
                purchased_delta_mcr=purchased_delta,
                balance_after_mcr=wallet.balance_mcr,
                status="reserved",
                call_id=call_id,
                installed_item_id=installed_item_id,
                market_slug=market_slug,
                tool_name=tool_name,
                room_id=room_id,
                session_id=session_id,
                actor=actor,
            )
        )
        await db.commit()


async def settle_call(call_id: UUID) -> None:
    """Confirm a reservation — the call reached the upstream MCP. Idempotent:
    a no-op if it was already settled/refunded, or was never reserved (a
    free, price-0 call)."""
    from app.models.mcp_credits import McpCreditLedger

    async_session = _short_session_factory()
    async with async_session() as db:
        row = (
            await db.execute(
                select(McpCreditLedger).where(
                    McpCreditLedger.call_id == call_id, McpCreditLedger.kind == "spend"
                )
            )
        ).scalar_one_or_none()
        if row is None or row.status != "reserved":
            return
        row.status = "settled"
        db.add(row)
        await db.commit()


async def refund_call(call_id: UUID, *, note: Optional[str] = None) -> None:
    """Reverse a reservation exactly — the call failed before reaching the MCP.

    A no-op if the reservation was already settled/refunded, or was never
    reserved, so a defensive double-call from an outer error handler is safe.
    """
    from app.models.mcp_credits import McpCreditLedger, McpCreditWallet

    async_session = _short_session_factory()
    async with async_session() as db:
        spend = (
            await db.execute(
                select(McpCreditLedger)
                .where(McpCreditLedger.call_id == call_id, McpCreditLedger.kind == "spend")
                .with_for_update()
            )
        ).scalar_one_or_none()
        if spend is None or spend.status != "reserved":
            return
        wallet = (
            await db.execute(
                select(McpCreditWallet)
                .where(McpCreditWallet.user_id == spend.user_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if wallet is None:
            return
        # Reverses grant_delta_mcr/purchased_delta_mcr exactly rather than
        # re-deriving the split — the original reservation already recorded
        # how the price was divided between the two buckets.
        wallet.grant_balance_mcr -= spend.grant_delta_mcr
        wallet.purchased_balance_mcr -= spend.purchased_delta_mcr
        wallet.period_spent_mcr = max(0, wallet.period_spent_mcr + spend.amount_mcr)
        wallet.lifetime_spent_mcr = max(0, wallet.lifetime_spent_mcr + spend.amount_mcr)
        wallet.updated_at = _utcnow()
        db.add(wallet)
        spend.status = "refunded"
        db.add(spend)
        db.add(
            McpCreditLedger(
                user_id=spend.user_id,
                kind="refund",
                amount_mcr=-spend.amount_mcr,
                grant_delta_mcr=-spend.grant_delta_mcr,
                purchased_delta_mcr=-spend.purchased_delta_mcr,
                balance_after_mcr=wallet.balance_mcr,
                status="settled",
                call_id=call_id,
                installed_item_id=spend.installed_item_id,
                market_slug=spend.market_slug,
                tool_name=spend.tool_name,
                room_id=spend.room_id,
                session_id=spend.session_id,
                actor="system",
                note=note,
            )
        )
        await db.commit()


# ---------- Charging an install: one-time purchase, then refund on failure -----


@dataclass(frozen=True)
class InstallCharge:
    """Result of :func:`charge_install`."""

    #: True only if this call actually debited the wallet just now.
    charged: bool
    #: True if a live (non-refunded) purchase already existed — a no-op.
    already_purchased: bool
    #: The purchase row's ``call_id`` (new or pre-existing); ``None`` when
    #: ``amount_mcr <= 0`` skipped the wallet entirely.
    purchase_id: Optional[UUID]


async def _find_active_purchase(
    db: AsyncSession, user_id: UUID, market_slug: str
) -> Optional["McpCreditLedger"]:
    """Latest un-refunded ``purchase`` row for ``(user_id, market_slug)``, if any.

    "Un-refunded" means no ``refund`` row shares its ``call_id`` — the same
    linkage :func:`refund_call` uses for a spend and its reversal. Reusing it
    here is what makes install idempotency independent of ``UserInstalledItem``
    (Stage 4A.3): a deleted-then-reinstalled item still finds its old purchase
    and skips charging again, while a purchase that had to be refunded
    (Stage 4A.4, install failed after charging) is free to be paid for again.
    """
    from app.models.mcp_credits import McpCreditLedger

    purchase_rows = (
        await db.execute(
            select(McpCreditLedger)
            .where(
                McpCreditLedger.user_id == user_id,
                McpCreditLedger.market_slug == market_slug,
                McpCreditLedger.kind == "purchase",
            )
            .order_by(McpCreditLedger.created_at.desc())
        )
    ).scalars().all()
    if not purchase_rows:
        return None
    call_ids = [row.call_id for row in purchase_rows if row.call_id is not None]
    refunded_call_ids: set[UUID] = set()
    if call_ids:
        refunded_call_ids = set(
            (
                await db.execute(
                    select(McpCreditLedger.call_id).where(
                        McpCreditLedger.user_id == user_id,
                        McpCreditLedger.kind == "refund",
                        McpCreditLedger.call_id.in_(call_ids),
                    )
                )
            ).scalars().all()
        )
    for row in purchase_rows:
        if row.call_id not in refunded_call_ids:
            return row
    return None


async def charge_install(
    user_id: UUID,
    amount_mcr: int,
    *,
    market_slug: str,
    actor: str = "system",
    note: Optional[str] = None,
) -> InstallCharge:
    """Charge a one-time Market item install purchase (internal design notes Stage 4A).

    Idempotent per ``(user_id, market_slug)`` via :func:`_find_active_purchase`
    — never via the caller's ``UserInstalledItem`` row, which is recreated on
    every reinstall and would double-charge a delete-then-reinstall (4A.3).
    ``amount_mcr <= 0`` is a pure no-op and touches nothing: the overwhelming
    majority of items never call this at all.

    Debits grant before purchased — same order as :func:`reserve_call` — but
    unlike a call, a purchase does not touch ``period_spent_mcr``/
    ``lifetime_spent_mcr``: those track per-call spend metrics, and a
    one-time install is a different thing being measured.

    Raises :class:`InstallPurchaseDenied` (a subclass of
    :class:`InsufficientCredits`) and touches nothing if the wallet can't
    cover it; ``plan_zero`` is set when the caller's current weekly grant is
    0, so the free tier gets a message about the plan rather than "insufficient".
    """
    from app.models.mcp_credits import McpCreditLedger

    if amount_mcr <= 0:
        return InstallCharge(charged=False, already_purchased=False, purchase_id=None)

    async_session = _short_session_factory()
    async with async_session() as db:
        weekly_grant_mcr = await _weekly_grant_mcr_for(user_id, db)
        wallet = await get_or_create_wallet(db, user_id)
        wallet = await roll_period_if_due(db, wallet, weekly_grant_mcr)
        wallet = await apply_plan_change_if_needed(db, wallet, weekly_grant_mcr)

        existing = await _find_active_purchase(db, user_id, market_slug)
        if existing is not None:
            await db.commit()
            return InstallCharge(charged=False, already_purchased=True, purchase_id=existing.call_id)

        if wallet.balance_mcr < amount_mcr:
            raise InstallPurchaseDenied(
                amount_mcr, wallet.balance_mcr, user_id=user_id, plan_zero=weekly_grant_mcr <= 0
            )

        grant_delta = -min(wallet.grant_balance_mcr, amount_mcr)
        purchased_delta = -(amount_mcr + grant_delta)
        wallet.grant_balance_mcr += grant_delta
        wallet.purchased_balance_mcr += purchased_delta
        wallet.updated_at = _utcnow()
        db.add(wallet)

        purchase_id = uuid4()
        db.add(
            McpCreditLedger(
                user_id=user_id,
                kind="purchase",
                amount_mcr=-amount_mcr,
                grant_delta_mcr=grant_delta,
                purchased_delta_mcr=purchased_delta,
                balance_after_mcr=wallet.balance_mcr,
                status="settled",
                call_id=purchase_id,
                market_slug=market_slug,
                actor=actor,
                note=note,
            )
        )
        await db.commit()
        return InstallCharge(charged=True, already_purchased=False, purchase_id=purchase_id)


async def refund_install(purchase_id: UUID, *, note: Optional[str] = None) -> None:
    """Reverse one install purchase by its ``call_id`` (Stage 4A.4).

    Called when a charge succeeded but the rest of the install failed
    (package fetch/parse, ``SkillFile`` write) — money and installed state
    must sink or swim together, never "charged but nothing installed". The
    retry that follows goes through :func:`charge_install` again and succeeds,
    since the refund row makes the earlier purchase inactive (see
    :func:`_find_active_purchase`). Idempotent: a no-op if this purchase
    does not exist or was already refunded.
    """
    from app.models.mcp_credits import McpCreditLedger, McpCreditWallet

    async_session = _short_session_factory()
    async with async_session() as db:
        purchase = (
            await db.execute(
                select(McpCreditLedger)
                .where(McpCreditLedger.call_id == purchase_id, McpCreditLedger.kind == "purchase")
                .with_for_update()
            )
        ).scalar_one_or_none()
        if purchase is None:
            return
        already_refunded = (
            await db.execute(
                select(McpCreditLedger.id).where(
                    McpCreditLedger.call_id == purchase_id, McpCreditLedger.kind == "refund"
                )
            )
        ).scalar_one_or_none()
        if already_refunded is not None:
            return
        wallet = (
            await db.execute(
                select(McpCreditWallet)
                .where(McpCreditWallet.user_id == purchase.user_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if wallet is None:
            return
        wallet.grant_balance_mcr -= purchase.grant_delta_mcr
        wallet.purchased_balance_mcr -= purchase.purchased_delta_mcr
        wallet.updated_at = _utcnow()
        db.add(wallet)
        db.add(
            McpCreditLedger(
                user_id=purchase.user_id,
                kind="refund",
                amount_mcr=-purchase.amount_mcr,
                grant_delta_mcr=-purchase.grant_delta_mcr,
                purchased_delta_mcr=-purchase.purchased_delta_mcr,
                balance_after_mcr=wallet.balance_mcr,
                status="settled",
                call_id=purchase_id,
                market_slug=purchase.market_slug,
                actor="system",
                note=note,
            )
        )
        await db.commit()


async def _adjust_bucket(
    user_id: UUID,
    amount_mcr: int,
    *,
    bucket: str,
    kind: str,
    note: Optional[str],
    actor: str,
    source: Optional[str] = None,
    external_ref: Optional[str] = None,
) -> dict:
    if bucket not in ("grant", "purchased"):
        raise ValueError(f"unknown wallet bucket: {bucket!r}")

    async_session = _short_session_factory()
    async with async_session() as db:
        weekly_grant_mcr = await _weekly_grant_mcr_for(user_id, db)
        wallet = await get_or_create_wallet(db, user_id)
        wallet = await roll_period_if_due(db, wallet, weekly_grant_mcr)
        wallet = await apply_plan_change_if_needed(db, wallet, weekly_grant_mcr)

        current = wallet.grant_balance_mcr if bucket == "grant" else wallet.purchased_balance_mcr
        # Clamped at zero: an admin correction larger than the bucket empties
        # it rather than going negative, so a signed adjustment can never
        # invent debt that the wallet has to dig itself out of.
        new_value = max(0, current + amount_mcr)
        applied = new_value - current
        if bucket == "grant":
            wallet.grant_balance_mcr = new_value
            grant_delta, purchased_delta = applied, 0
        else:
            wallet.purchased_balance_mcr = new_value
            grant_delta, purchased_delta = 0, applied
        wallet.updated_at = _utcnow()
        db.add(wallet)
        _add_ledger_rows(
            db,
            user_id,
            [
                LedgerDraft(
                    kind=kind,
                    amount_mcr=applied,
                    grant_delta_mcr=grant_delta,
                    purchased_delta_mcr=purchased_delta,
                    balance_after_mcr=wallet.balance_mcr,
                    note=note,
                    source=source,
                    external_ref=external_ref,
                )
            ],
            actor=actor,
        )
        await db.commit()
        await db.refresh(wallet)
        return _wallet_summary_dict(wallet, mcr_to_credits(weekly_grant_mcr))


async def topup(
    user_id: UUID,
    amount_mcr: int,
    *,
    bucket: str = "purchased",
    note: Optional[str] = None,
    actor: str = "system",
    source: Optional[str] = None,
    external_ref: Optional[str] = None,
) -> dict:
    """Add credits that were paid for — defaults to the bucket that never expires.

    Positive amount only; use :func:`adjust` for a signed correction. This is
    the seam Stage 8's Stripe webhook and Stage 5's admin "credit" button
    both call — the ledger row is what makes either idempotent later.

    ``source``/``external_ref`` are the Stage 8 Stripe seam and unused by any
    caller today (Admin Center top-ups pass neither, so both stay ``NULL``).
    A future payment webhook passes ``source="stripe"`` and
    ``external_ref=<payment_intent id>``; the partial unique index on
    ``mcp_credit_ledger.external_ref`` then makes a retried webhook delivery
    idempotent, but only if the caller catches the resulting
    ``IntegrityError`` and treats it as "already applied" — this function
    does not do that for you.
    """
    if amount_mcr <= 0:
        raise ValueError("topup amount must be positive")
    return await _adjust_bucket(
        user_id,
        amount_mcr,
        bucket=bucket,
        kind="topup",
        note=note,
        actor=actor,
        source=source,
        external_ref=external_ref,
    )


async def adjust(
    user_id: UUID,
    amount_mcr: int,
    *,
    bucket: str = "purchased",
    note: Optional[str] = None,
    actor: str = "system",
    source: Optional[str] = None,
    external_ref: Optional[str] = None,
) -> dict:
    """Signed admin correction to a bucket (e.g. refunding an incident by hand).

    ``source``/``external_ref`` as in :func:`topup`: billing revokes credits
    of a refunded payment with ``external_ref=<entitlement id>``.
    """
    if amount_mcr == 0:
        raise ValueError("adjustment amount must be non-zero")
    return await _adjust_bucket(
        user_id, amount_mcr, bucket=bucket, kind="adjustment", note=note, actor=actor,
        source=source, external_ref=external_ref,
    )


def _ledger_entry_dict(row: "McpCreditLedger") -> dict:
    return {
        "id": str(row.id),
        "kind": row.kind,
        "amount": mcr_to_credits(row.amount_mcr),
        "grant_delta": mcr_to_credits(row.grant_delta_mcr),
        "purchased_delta": mcr_to_credits(row.purchased_delta_mcr),
        "balance_after": mcr_to_credits(row.balance_after_mcr),
        "status": row.status,
        "call_id": str(row.call_id) if row.call_id else None,
        "installed_item_id": str(row.installed_item_id) if row.installed_item_id else None,
        "market_slug": row.market_slug,
        "tool_name": row.tool_name,
        "room_id": str(row.room_id) if row.room_id else None,
        "actor": row.actor,
        "note": row.note,
        "source": row.source,
        "external_ref": row.external_ref,
        "created_at": row.created_at.isoformat(),
    }


async def list_ledger(
    user_id: UUID,
    *,
    limit: int = 50,
    offset: int = 0,
    kind: Optional[str] = None,
    since: Optional[datetime] = None,
) -> tuple[list[dict], int]:
    """Page through one user's ledger, newest first — the Admin Center's view.

    Opens its own short session like every other function here (see the
    module docstring); a read-only listing has no reason to depend on a
    caller's transaction. ``since`` (inclusive) scopes to entries at or after
    a timestamp — the ``/me/mcp-credits/ledger`` endpoint uses it to show
    only the current weekly grant window.
    """
    from app.models.mcp_credits import McpCreditLedger

    async_session = _short_session_factory()
    async with async_session() as db:
        stmt = select(McpCreditLedger).where(McpCreditLedger.user_id == user_id)
        if kind is not None:
            stmt = stmt.where(McpCreditLedger.kind == kind)
        if since is not None:
            stmt = stmt.where(McpCreditLedger.created_at >= since)
        total = (
            await db.execute(select(func.count()).select_from(stmt.subquery()))
        ).scalar_one()
        rows = (
            await db.execute(
                stmt.order_by(McpCreditLedger.created_at.desc()).offset(offset).limit(limit)
            )
        ).scalars().all()
        return [_ledger_entry_dict(r) for r in rows], total


def user_facing_actor(actor: Optional[str]) -> str:
    """Collapse the ledger's ``agent|admin:<email>|system`` actor into a label
    safe to show the wallet owner — admin emails aren't the user's business."""
    if actor and actor.startswith("admin:"):
        return "support"
    return actor or "system"


def _owner_ledger_entry(row: "McpCreditLedger") -> dict:
    """A ledger row as its owner sees it outside the Hub (the Legion profile,
    internal design notes §2.5): no admin notes, payment references or room ids."""
    return {
        "id": str(row.id),
        "kind": row.kind,
        "amount": mcr_to_credits(row.amount_mcr),
        "grant_delta": mcr_to_credits(row.grant_delta_mcr),
        "purchased_delta": mcr_to_credits(row.purchased_delta_mcr),
        "balance_after": mcr_to_credits(row.balance_after_mcr),
        "status": row.status,
        "market_slug": row.market_slug,
        "tool_name": row.tool_name,
        "actor": user_facing_actor(row.actor),
        "created_at": row.created_at.isoformat(),
    }


def encode_ledger_cursor(created_at: datetime, row_id: UUID) -> str:
    raw = f"{_aware(created_at).isoformat()}|{row_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_ledger_cursor(cursor: str) -> tuple[datetime, UUID]:
    """Raises ``ValueError`` on anything :func:`encode_ledger_cursor` did not make."""
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        stamp, row_id = raw.split("|", 1)
        return _aware(datetime.fromisoformat(stamp)), UUID(row_id)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid ledger cursor") from exc


async def ledger_page(
    user_id: UUID, *, limit: int = 50, cursor: Optional[str] = None
) -> tuple[list[dict], Optional[str]]:
    """One page of a user's whole ledger, newest first, for the wallet owner.

    Keyset paging on ``(created_at, id)``: the ledger takes a row per MCP call,
    so an offset would both slow down and shift under a reader while the user
    keeps working. Returns the entries and the cursor of the next page
    (``None`` on the last one).
    """
    from app.models.mcp_credits import McpCreditLedger

    stmt = select(McpCreditLedger).where(McpCreditLedger.user_id == user_id)
    if cursor:
        before_at, before_id = decode_ledger_cursor(cursor)
        stmt = stmt.where(
            tuple_(McpCreditLedger.created_at, McpCreditLedger.id) < tuple_(before_at, before_id)
        )
    async_session = _short_session_factory()
    async with async_session() as db:
        rows = (
            await db.execute(
                stmt.order_by(McpCreditLedger.created_at.desc(), McpCreditLedger.id.desc()).limit(limit + 1)
            )
        ).scalars().all()
    next_cursor = None
    if len(rows) > limit:
        rows = rows[:limit]
        next_cursor = encode_ledger_cursor(rows[-1].created_at, rows[-1].id)
    return [_owner_ledger_entry(r) for r in rows], next_cursor


__all__ = [
    "MCR_PER_CREDIT",
    "CREDIT_INPUT_PLACES",
    "CREDIT_DISPLAY_PLACES",
    "MAX_MCR",
    "CreditAmount",
    "credits_to_mcr",
    "mcr_to_credits",
    "format_credits",
    "DENIAL_REASONS",
    "McpCreditError",
    "InsufficientCredits",
    "PriceUnknown",
    "NoWalletOwner",
    "InstallPurchaseDenied",
    "InstallCharge",
    "WEEK",
    "WalletFields",
    "LedgerDraft",
    "roll_period",
    "apply_plan_change",
    "apply_plan_change_if_needed",
    "get_or_create_wallet",
    "roll_period_if_due",
    "wallet_summary",
    "get_wallet_balance_mcr",
    "get_price_snapshot",
    "get_price_snapshot_by_slug",
    "price_from_snapshot",
    "resolve_price_mcr",
    "store_price_snapshot",
    "refresh_price_snapshot",
    "refresh_price_snapshot_by_slug",
    "verify_reserved_call",
    "reserve_call",
    "settle_call",
    "refund_call",
    "charge_install",
    "refund_install",
    "topup",
    "adjust",
    "list_ledger",
    "user_facing_actor",
    "encode_ledger_cursor",
    "decode_ledger_cursor",
    "ledger_page",
]
