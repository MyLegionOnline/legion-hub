"""Profit and profitability per marketplace MCP server (Admin Center "Finance").

Revenue is what users *paid* for: net spend from the purchased bucket of the
credit ledger, valued at :data:`CREDIT_PRICE_USD`. Spend covered by the weekly
grant is reported separately as ``grant_credits`` — it is already paid for by
the subscription, not by the MCP, so it is not revenue of the server, but the
server's expenses still include the calls it served.

Expenses are the manual entries of :class:`app.models.mcp_expense.McpExpense`.
Both sides are filtered by the same inclusive ``[date_from, date_to]`` range
(UTC days).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.mcp_credits import mcr_to_credits

#: ``internal design notes`` R8: 1 $ = 100 credits, so 1 credit = 1 cent.
CREDIT_PRICE_USD = 0.01

#: Ledger rows that carry money for a server: a call, a one-time install
#: charge and the reversal of either. ``reserved`` rows are excluded below —
#: a call that has not settled is not revenue yet.
_REVENUE_KINDS = ("spend", "purchase", "refund")


def day_bounds(date_from: Optional[date], date_to: Optional[date]) -> tuple[Optional[datetime], Optional[datetime]]:
    """Inclusive date range -> half-open ``[start, end)`` UTC datetimes."""
    start = datetime.combine(date_from, time.min, tzinfo=timezone.utc) if date_from else None
    end = (
        datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc)
        if date_to
        else None
    )
    return start, end


def _usd(cents: float) -> float:
    return round(cents / 100.0, 4)


def build_rows(
    revenue: dict[str, dict[str, int]],
    expenses_cents: dict[str, int],
) -> list[dict[str, Any]]:
    """Merge the two sides by slug and derive profit and profitability.

    ``margin_pct`` is profit / revenue, ``roi_pct`` is profit / expenses; each
    is ``None`` when its denominator is zero (an honest "n/a", not 0 %).
    """
    rows: list[dict[str, Any]] = []
    for slug in sorted(set(revenue) | set(expenses_cents)):
        rev = revenue.get(slug, {})
        paid_credits = mcr_to_credits(rev.get("paid_mcr", 0))
        grant_credits = mcr_to_credits(rev.get("grant_mcr", 0))
        revenue_usd = round(paid_credits * CREDIT_PRICE_USD, 4)
        expenses_usd = _usd(expenses_cents.get(slug, 0))
        profit_usd = round(revenue_usd - expenses_usd, 4)
        rows.append(
            {
                "market_slug": slug,
                "calls": rev.get("calls", 0),
                "paid_credits": paid_credits,
                "grant_credits": grant_credits,
                "revenue_usd": revenue_usd,
                "expenses_usd": expenses_usd,
                "profit_usd": profit_usd,
                "margin_pct": round(profit_usd / revenue_usd * 100, 1) if revenue_usd else None,
                "roi_pct": round(profit_usd / expenses_usd * 100, 1) if expenses_usd else None,
            }
        )
    return rows


def totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    revenue = round(sum(r["revenue_usd"] for r in rows), 4)
    expenses = round(sum(r["expenses_usd"] for r in rows), 4)
    profit = round(revenue - expenses, 4)
    return {
        "calls": sum(r["calls"] for r in rows),
        "paid_credits": round(sum(r["paid_credits"] for r in rows), 3),
        "grant_credits": round(sum(r["grant_credits"] for r in rows), 3),
        "revenue_usd": revenue,
        "expenses_usd": expenses,
        "profit_usd": profit,
        "margin_pct": round(profit / revenue * 100, 1) if revenue else None,
        "roi_pct": round(profit / expenses * 100, 1) if expenses else None,
    }


_REVENUE_SQL = text(
    """
    SELECT market_slug,
           COALESCE(SUM(-purchased_delta_mcr), 0) AS paid_mcr,
           COALESCE(SUM(-grant_delta_mcr), 0)     AS grant_mcr,
           COUNT(*) FILTER (WHERE kind = 'spend' AND status = 'settled') AS calls
      FROM mcp_credit_ledger
     WHERE market_slug IS NOT NULL
       AND kind IN :kinds
       AND status IN ('settled', 'refunded')
       AND (CAST(:start AS timestamptz) IS NULL OR created_at >= :start)
       AND (CAST(:end   AS timestamptz) IS NULL OR created_at <  :end)
     GROUP BY market_slug
    """
).bindparams(bindparam("kinds", expanding=True))

_EXPENSE_SQL = text(
    """
    SELECT market_slug, COALESCE(SUM(amount_cents), 0) AS cents
      FROM mcp_expense
     WHERE (CAST(:d_from AS date) IS NULL OR incurred_on >= :d_from)
       AND (CAST(:d_to   AS date) IS NULL OR incurred_on <= :d_to)
     GROUP BY market_slug
    """
)

_KNOWN_SLUGS_SQL = text(
    """
    SELECT market_slug FROM mcp_price_snapshot
    UNION SELECT market_slug FROM mcp_credit_ledger WHERE market_slug IS NOT NULL
    UNION SELECT market_slug FROM mcp_expense
    """
)


async def summary(
    db: AsyncSession, date_from: Optional[date], date_to: Optional[date]
) -> dict[str, Any]:
    start, end = day_bounds(date_from, date_to)
    revenue_rows = (
        await db.execute(_REVENUE_SQL, {"kinds": list(_REVENUE_KINDS), "start": start, "end": end})
    ).all()
    expense_rows = (
        await db.execute(_EXPENSE_SQL, {"d_from": date_from, "d_to": date_to})
    ).all()
    known = (await db.execute(_KNOWN_SLUGS_SQL)).scalars().all()

    rows = build_rows(
        {
            r.market_slug: {"paid_mcr": int(r.paid_mcr), "grant_mcr": int(r.grant_mcr), "calls": int(r.calls)}
            for r in revenue_rows
        },
        {r.market_slug: int(r.cents) for r in expense_rows},
    )
    return {
        "date_from": date_from.isoformat() if date_from else None,
        "date_to": date_to.isoformat() if date_to else None,
        "credit_price_usd": CREDIT_PRICE_USD,
        "rows": rows,
        "totals": totals(rows),
        "known_slugs": sorted(known),
    }
