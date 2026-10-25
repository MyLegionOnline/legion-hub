"""Money the operator spent on a marketplace MCP server (manual entry).

The Admin Center "Finance" tab sets this against the credits users paid for the
same server (``McpCreditLedger``) to show profit and profitability per MCP.
Stage 1 is manual: the operator types in a deposit or an invoice by hand.
Automatic per-call cost is a later stage (``internal design notes`` §5).

Amounts are integer US cents; a payment made in another currency is converted
by the operator at entry time and the original is kept in ``note``.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class McpExpense(SQLModel, table=True):
    __tablename__ = "mcp_expense"
    __table_args__ = (
        sa.Index("ix_mcp_expense_slug_date", "market_slug", "incurred_on"),
        sa.CheckConstraint("amount_cents > 0", name="ck_mcp_expense_amount_positive"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    #: Market slug of the MCP server the money was spent on (``mcp-searxng``).
    #: No FK: the catalogue lives in the Market database.
    market_slug: str = Field(max_length=100)
    amount_cents: int = Field(sa_column=sa.Column(sa.BigInteger, nullable=False))
    #: The day the money left, not the day it was typed in.
    incurred_on: date = Field(sa_column=sa.Column(sa.Date, nullable=False))
    note: Optional[str] = Field(default=None, sa_column=sa.Column(sa.Text, nullable=True))
    #: ``admin_center:<login>``, same value the Hub audit trail records.
    actor: str = Field(max_length=255)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )


__all__ = ["McpExpense"]
