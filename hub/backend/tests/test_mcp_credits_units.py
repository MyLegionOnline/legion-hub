"""Stage 0 of internal design notes: units, conversion and table shape.

The point of the millicredit is that 0.3 + 0.6 must never drift, so these
tests pin the conversion against exactly the values that float arithmetic gets
wrong (0.29 * 1000 == 289.99999999999994 in binary), and the DDL against what
migrate_mcp_credits.sql declares.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from app.core.mcp_credits import (
    MAX_MCR,
    MCR_PER_CREDIT,
    InsufficientCredits,
    McpCreditError,
    NoWalletOwner,
    PriceUnknown,
    credits_to_mcr,
    format_credits,
    mcr_to_credits,
)


# ---------- credits -> mcr -----------------------------------------------------


@pytest.mark.parametrize(
    "credits, expected_mcr",
    [
        (0, 0),
        (0.3, 300),      # mcp-searxng
        (0.6, 600),      # mcp-toolbox
        (0.29, 290),     # 0.29 * 1000 == 289.99999999999994 as a float
        (0.07, 70),
        (1, 1000),
        (100, 100_000),
        (0.001, 1),      # one millicredit, the smallest amount there is
        (-50, -50_000),  # admin adjustment
    ],
)
def test_credits_to_mcr_is_exact_for_real_prices(credits, expected_mcr):
    assert credits_to_mcr(credits) == expected_mcr


def test_charging_a_balance_repeatedly_does_not_drift():
    """The reason the whole module exists: a float balance leaks, an int does not."""
    float_balance = 100.0
    for _ in range(1000):
        float_balance -= 0.3
    assert float_balance != -200.0  # -200.00000000000128 after 1000 charges

    mcr_balance = credits_to_mcr(100)
    for _ in range(1000):
        mcr_balance -= credits_to_mcr(0.3)
    assert mcr_balance == credits_to_mcr(-200)
    assert mcr_to_credits(mcr_balance) == -200.0


def test_spending_a_balance_to_exactly_zero_is_not_negative():
    """A float wallet emptied by three 0.1 calls reads as overdrawn."""
    float_balance = 0.3
    for _ in range(3):
        float_balance -= 0.1
    assert float_balance < 0  # -2.78e-17: "insufficient credits" on an empty-but-square wallet

    mcr_balance = credits_to_mcr(0.3)
    for _ in range(3):
        mcr_balance -= credits_to_mcr(0.1)
    assert mcr_balance == 0


@pytest.mark.parametrize(
    "credits, expected_mcr",
    [
        (0.0004, 0),      # finer than 1 mcr — dropped, not accumulated
        (0.0016, 1),
        (1.9999, 1999),
        (-0.0004, -1),    # rounds toward -inf, same sentence for either sign
        (-1.0005, -1001),
    ],
)
def test_input_rounds_down_below_one_millicredit(credits, expected_mcr):
    assert credits_to_mcr(credits) == expected_mcr


@pytest.mark.parametrize("value", ["0.3", Decimal("0.3"), 0.3])
def test_credits_to_mcr_accepts_str_decimal_and_float_alike(value):
    assert credits_to_mcr(value) == 300


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "not a number"])
def test_credits_to_mcr_rejects_non_amounts(value):
    with pytest.raises(ValueError):
        credits_to_mcr(value)


@pytest.mark.parametrize("value", [True, None, [1]])
def test_credits_to_mcr_rejects_wrong_types(value):
    with pytest.raises(TypeError):
        credits_to_mcr(value)


def test_credits_to_mcr_rejects_amounts_a_bigint_cannot_hold():
    with pytest.raises(ValueError):
        credits_to_mcr(MAX_MCR)  # * 1000 would overflow the column


# ---------- mcr -> credits -----------------------------------------------------


@pytest.mark.parametrize(
    "mcr, expected_credits",
    [(0, 0.0), (300, 0.3), (600, 0.6), (1, 0.001), (87_400, 87.4), (-50_000, -50.0)],
)
def test_mcr_to_credits_is_lossless(mcr, expected_credits):
    assert mcr_to_credits(mcr) == expected_credits


@pytest.mark.parametrize("mcr", [0, 1, 300, 999, 87_412, -6_251])
def test_conversion_round_trips_through_credits(mcr):
    assert credits_to_mcr(mcr_to_credits(mcr)) == mcr


def test_format_credits_rounds_to_two_places_for_humans():
    assert format_credits(87_412) == 87.41
    assert format_credits(87_415) == 87.42  # half up
    assert format_credits(300) == 0.3
    assert format_credits(0) == 0.0


def test_mcr_per_credit_is_the_only_scale_factor():
    assert MCR_PER_CREDIT == 1000
    assert credits_to_mcr(1) == MCR_PER_CREDIT


# ---------- domain errors ------------------------------------------------------


def test_insufficient_credits_reports_both_amounts_in_credits():
    err = InsufficientCredits(600, 200, user_id=uuid4())

    assert err.reason == "insufficient"
    assert (err.required_mcr, err.available_mcr) == (600, 200)
    assert "0.6" in err.user_message() and "0.2" in err.user_message()


def test_price_unknown_and_no_owner_carry_metric_reasons():
    assert PriceUnknown(market_slug="mcp-toolbox", tool_name="run").reason == "no_price"
    assert NoWalletOwner(market_slug="mcp-searxng").reason == "no_owner"


@pytest.mark.parametrize(
    "err",
    [InsufficientCredits(1, 0), PriceUnknown(market_slug="x"), NoWalletOwner()],
)
def test_every_credit_error_is_catchable_as_one_type(err):
    assert isinstance(err, McpCreditError)
    assert err.user_message()


# ---------- table shape --------------------------------------------------------


def test_models_declare_the_tables_the_migration_creates():
    from app.models import McpCreditLedger, McpCreditWallet, McpPriceSnapshot

    wallet = McpCreditWallet.__table__
    assert wallet.name == "mcp_credit_wallet"
    # Balances must be BIGINT: INTEGER caps at ~2.1M credits of lifetime spend.
    for column in ("grant_balance_mcr", "purchased_balance_mcr", "lifetime_spent_mcr"):
        assert wallet.c[column].type.__class__.__name__ == "BigInteger"

    ledger = McpCreditLedger.__table__
    call_index = next(i for i in ledger.indexes if i.name == "uq_mcp_credit_ledger_call")
    assert call_index.unique, "a retried call must not be able to charge twice"

    snapshot = McpPriceSnapshot.__table__
    assert snapshot.name == "mcp_price_snapshot"
    assert "tool_price_overrides" in snapshot.c
