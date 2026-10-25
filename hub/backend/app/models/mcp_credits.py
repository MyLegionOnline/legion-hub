"""Wallet, ledger and price snapshot for marketplace MCP call credits.

Stage 0 of ``internal design notes``.  The Hub is the source of truth for
*balances*; the Market service is the source of truth for *prices*, which is
why :class:`McpPriceSnapshot` exists — a charge must not fail just because
Market is momentarily unreachable, and it must not silently become free either.

All amounts are integer millicredits (``BIGINT``); see
``app/core/mcp_credits.py`` for the one permitted conversion to credits.
Nothing here has a foreign key into the Market database, same as the models in
``app/models/market.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


#: ``grant``/``expire`` — the weekly grant being issued and burnt;
#: ``topup``/``adjustment`` — money added or removed by an admin (later: Stripe);
#: ``spend``/``refund`` — one tool call and its reversal;
#: ``purchase`` — a one-time Market item install charge (internal design notes
#: Stage 4A), reversed by a plain ``refund`` row sharing the same ``call_id``.
LEDGER_KINDS: frozenset[str] = frozenset(
    {"grant", "expire", "topup", "adjustment", "spend", "refund", "purchase"}
)

#: ``reserved`` is written before the call leaves the Hub; it becomes
#: ``settled`` once the call reached the upstream MCP, or ``refunded`` when it
#: failed on our side.  See the outcome table in the plan.
LEDGER_STATUSES: frozenset[str] = frozenset({"reserved", "settled", "refunded"})


def _mcr_column(**kwargs: Any) -> sa.Column:
    return sa.Column(sa.BigInteger, nullable=False, server_default=sa.text("0"), **kwargs)


class McpCreditWallet(SQLModel, table=True):
    """One credit wallet per user, with two buckets.

    ``grant_balance_mcr`` is the weekly allowance and burns when the window
    rolls; ``purchased_balance_mcr`` is bought and never burns.  Spending draws
    on the grant first, so a user cannot lose money they paid for by being
    idle.  The window is anchored to the user's first activation and rolls
    lazily on read — there is deliberately no cron pass over every wallet.
    """

    __tablename__ = "mcp_credit_wallet"

    user_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    grant_balance_mcr: int = Field(default=0, sa_column=_mcr_column())
    purchased_balance_mcr: int = Field(default=0, sa_column=_mcr_column())
    #: How much grant was issued for the current window — the "Y" of "X of Y".
    period_grant_mcr: int = Field(default=0, sa_column=_mcr_column())
    period_spent_mcr: int = Field(default=0, sa_column=_mcr_column())
    lifetime_spent_mcr: int = Field(default=0, sa_column=_mcr_column())
    period_started_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False)
    )
    period_ends_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )

    @property
    def balance_mcr(self) -> int:
        return self.grant_balance_mcr + self.purchased_balance_mcr


class McpCreditLedger(SQLModel, table=True):
    """Append-only record of every movement of credits.

    ``balance_after_mcr`` is filled on every row on purpose: reconstructing a
    balance from deltas during an incident is exactly the moment you do not
    want to be doing arithmetic.
    """

    __tablename__ = "mcp_credit_ledger"
    __table_args__ = (
        sa.Index(
            "ix_mcp_credit_ledger_user_created",
            "user_id",
            sa.text("created_at DESC"),
        ),
        # One spend per call_id — the retry of an LLM loop must not charge
        # twice, and the refund finds its spend by the same id.
        sa.Index(
            "uq_mcp_credit_ledger_call",
            "call_id",
            unique=True,
            postgresql_where=sa.text("call_id IS NOT NULL AND kind = 'spend'"),
        ),
        # Stage 8 seam: one ledger row per payment-provider event. Nothing
        # populates external_ref yet, so this index is a no-op until then.
        sa.Index(
            "uq_mcp_credit_ledger_external_ref",
            "external_ref",
            unique=True,
            postgresql_where=sa.text("external_ref IS NOT NULL"),
        ),
    )

    # No index=True on the primary key: the ledger takes a row per MCP call,
    # and the PK btree already covers it.
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    kind: str = Field(max_length=16)
    #: Signed: a spend is negative, a grant positive.
    amount_mcr: int = Field(
        sa_column=sa.Column(sa.BigInteger, nullable=False)
    )
    #: How the amount was split across the two buckets, so a refund can
    #: reverse it exactly instead of guessing.
    grant_delta_mcr: int = Field(default=0, sa_column=_mcr_column())
    purchased_delta_mcr: int = Field(default=0, sa_column=_mcr_column())
    balance_after_mcr: int = Field(
        sa_column=sa.Column(sa.BigInteger, nullable=False)
    )
    status: str = Field(
        default="settled",
        sa_column=sa.Column(
            sa.String(16), nullable=False, server_default=sa.text("'settled'")
        ),
    )
    call_id: Optional[UUID] = Field(default=None)
    installed_item_id: Optional[UUID] = Field(default=None)
    market_slug: Optional[str] = Field(default=None, max_length=100)
    tool_name: Optional[str] = Field(default=None, max_length=255)
    room_id: Optional[UUID] = Field(default=None)
    session_id: Optional[UUID] = Field(default=None)
    #: ``agent`` | ``admin:<email>`` | ``system``.
    actor: str = Field(
        default="system",
        sa_column=sa.Column(
            sa.String(255), nullable=False, server_default=sa.text("'system'")
        ),
    )
    #: Stage 8 seam for paid top-ups: ``None``/``"admin"`` today, ``"stripe"``
    #: once a payment provider is wired up. Not read by anything yet — see
    #: app.core.mcp_credits.topup and hub/MCP_CREDITS_REFERENCE.md.
    source: Optional[str] = Field(default=None, max_length=32)
    #: Payment-provider reference (e.g. a Stripe PaymentIntent id). The
    #: partial unique index below is what will make a future webhook retry
    #: idempotent — a second delivery of the same event hits the same row
    #: instead of double-crediting the wallet.
    external_ref: Optional[str] = Field(default=None, max_length=255)
    note: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text, nullable=True))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )


class McpPriceSnapshot(SQLModel, table=True):
    """Last known Market price for one marketplace item, cached in the Hub.

    Refreshed on install, on discovery and by hand from the Admin Center.  When
    there is no snapshot and Market is unreachable, the call is refused rather
    than served free — see :class:`app.core.mcp_credits.PriceUnknown`.
    """

    __tablename__ = "mcp_price_snapshot"
    __table_args__ = (sa.Index("ix_mcp_price_snapshot_slug", "market_slug"),)

    #: Market item id; intentionally no FK — the catalogue lives in another
    #: database (same rule as ``UserInstalledItem.market_item_id``).
    market_item_id: UUID = Field(primary_key=True)
    market_slug: str = Field(max_length=100)
    call_price_mcr: int = Field(default=0, sa_column=_mcr_column())
    #: ``{"<tool name>": <price in mcr>}`` — per-tool overrides of the base
    #: price (mcp-toolbox charges differently per tool).
    tool_price_overrides: dict[str, int] = Field(
        default_factory=dict,
        sa_column=sa.Column(
            JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
    )
    fetched_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )


__all__ = [
    "McpCreditWallet",
    "McpCreditLedger",
    "McpPriceSnapshot",
    "LEDGER_KINDS",
    "LEDGER_STATUSES",
]
