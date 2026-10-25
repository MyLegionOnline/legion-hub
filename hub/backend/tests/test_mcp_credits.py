"""Stage 4 of internal design notes: charging a marketplace MCP call.

Two layers, same split as test_mcp_credits_wallet.py:

* ``TestExecuteMcpToolOrchestration`` — pure, mocked. Exercises the
  reserve/settle/refund *decisions* ``execute_mcp_tool`` makes for every row
  of the outcome table, without a database.
* ``TestChargingDB`` — against a real PostgreSQL instance (``@pytest.mark.db``,
  same convention as ``test_session_models.py`` / ``test_mcp_credits_wallet.py``).
  Exercises what actually happens to the wallet and ledger: a real
  reserve/settle/refund round trip, concurrent reservations not overdrawing,
  a retried call_id not double-spending, and ``room_mcp_tools`` hiding a
  marketplace item once the balance can't cover its cheapest tool.

Run the DB-backed class with:
    DATABASE_URL=postgresql+asyncpg://... pytest tests/test_mcp_credits.py -m db
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from app.core.mcp_credits import InsufficientCredits, credits_to_mcr
from app.core.mcp_gateway import McpToolBinding, execute_mcp_tool, room_mcp_tools


def _binding(
    *,
    market_item_id: uuid.UUID | None,
    market_slug: str | None = "mcp-searxng",
    tool_name: str = "search",
    managed_stdio: bool = False,
) -> McpToolBinding:
    return McpToolBinding(
        installed_item_id=uuid.uuid4(),
        market_item_id=market_item_id,
        market_slug=market_slug,
        tool_name=tool_name,
        custom_config=None,
        isolation="stateless",
        managed_stdio=managed_stdio,
    )


def _http_status_error(status_code: int, detail: Any) -> httpx.HTTPStatusError:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"detail": detail}
    return httpx.HTTPStatusError("boom", request=MagicMock(), response=response)


# ---------- Orchestration: reserve/settle/refund decisions, fully mocked --------


@pytest.mark.asyncio
class TestExecuteMcpToolOrchestration:
    async def test_marketplace_call_reserves_then_settles_on_success(self):
        binding = _binding(market_item_id=uuid.uuid4())
        user_id = uuid.uuid4()
        room_id = uuid.uuid4()

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock) as mock_reserve,
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as mock_settle,
            patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock) as mock_refund,
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"ok": True},
            ) as mock_post,
        ):
            result = await execute_mcp_tool(binding, user_id, room_id, {"query": "legion"})

        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        mock_reserve.assert_awaited_once()
        reserve_kwargs = mock_reserve.await_args.kwargs
        assert reserve_kwargs["installed_item_id"] == binding.installed_item_id
        assert reserve_kwargs["market_item_id"] == binding.market_item_id
        assert reserve_kwargs["market_slug"] == binding.market_slug
        assert reserve_kwargs["tool_name"] == binding.tool_name
        assert reserve_kwargs["room_id"] == room_id
        call_id = reserve_kwargs["call_id"]
        mock_settle.assert_awaited_once_with(call_id)
        mock_refund.assert_not_awaited()
        mock_post.assert_awaited_once()
        # Stage 7: Gateway's verify call matches this reservation's call_id
        # back to what Hub actually sent it in the /v1/execute payload.
        gateway_payload = mock_post.await_args.args[-1]
        assert gateway_payload["call_id"] == str(call_id)

    async def test_free_marketplace_call_forwards_no_call_id_to_gateway(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=0),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"ok": True},
            ) as mock_post,
        ):
            await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        gateway_payload = mock_post.await_args.args[-1]
        assert gateway_payload["call_id"] is None

    async def test_personal_mcp_never_touches_the_wallet(self):
        binding = _binding(market_item_id=None, market_slug=None, managed_stdio=False)
        user_id = uuid.uuid4()
        room_id = uuid.uuid4()

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock) as mock_price,
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock) as mock_reserve,
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as mock_settle,
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"ok": True},
            ),
        ):
            result = await execute_mcp_tool(binding, user_id, room_id, {})

        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        mock_price.assert_not_awaited()
        mock_reserve.assert_not_awaited()
        mock_settle.assert_not_awaited()

    async def test_free_marketplace_item_skips_the_wallet_but_still_calls_gateway(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=0),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock) as mock_reserve,
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as mock_settle,
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"ok": True},
            ),
        ):
            result = await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        mock_reserve.assert_not_awaited()
        mock_settle.assert_not_awaited()

    async def test_insufficient_credits_denies_before_touching_the_gateway(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=1000),
            patch(
                "app.core.mcp_credits.reserve_call",
                new_callable=AsyncMock,
                side_effect=InsufficientCredits(1000, 200),
            ),
            patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock) as mock_post,
        ):
            result = await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        assert "Not enough MCP credits" in result
        mock_post.assert_not_awaited()

    async def test_zero_uuid_owner_is_a_hard_refusal_not_a_free_ride(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock) as mock_price,
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock) as mock_reserve,
            patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock) as mock_post,
        ):
            result = await execute_mcp_tool(binding, uuid.UUID(int=0), uuid.uuid4(), {})

        assert "no owner" in result
        mock_price.assert_not_awaited()
        mock_reserve.assert_not_awaited()
        mock_post.assert_not_awaited()

    async def test_gateway_unreachable_refunds_the_reservation(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as mock_settle,
            patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock) as mock_refund,
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("no route"),
            ),
        ):
            result = await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        assert result == "Error: MCP Gateway unavailable"
        mock_refund.assert_awaited_once()
        assert mock_refund.await_args.kwargs["note"] == "gateway unreachable"
        mock_settle.assert_not_awaited()

    async def test_isolated_tool_error_refunds_the_charge(self):
        """Stage 3: an in-band ``isError`` envelope (HTTP 200, but the MCP tool
        reported failure) is unwrapped to an ``"Error: ..."`` string and refunds
        the reservation — it does not settle (VISION_REFERENCE §"Stage 3 —
        Credit refund on in-band isError")."""
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as mock_settle,
            patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock) as mock_refund,
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"isError": True, "content": [{"type": "text", "text": "tool failed"}]},
            ),
        ):
            result = await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        assert result == "Error: tool failed"
        mock_refund.assert_awaited_once()
        assert mock_refund.await_args.kwargs["note"] == "tool returned isError"
        mock_settle.assert_not_awaited()

    async def test_plain_text_success_settles_the_charge(self):
        """A successful non-JSON text result (``web_url_read`` returns a page as
        markdown) also unwraps to a ``str``; it is charged, not refunded."""
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as mock_settle,
            patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock) as mock_refund,
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"content": [{"type": "text", "text": "# Page title\n\nSome article text."}]},
            ),
        ):
            result = await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        assert result == "# Page title\n\nSome article text."
        mock_settle.assert_awaited_once()
        mock_refund.assert_not_awaited()

    @pytest.mark.parametrize(
        "code,expect_settle",
        [
            ("mcp_unreachable", False),
            ("mcp_auth_failed", False),
            ("mcp_http_error", False),
            ("mcp_bad_response", False),
            ("mcp_bad_request", False),
            ("mcp_rpc_error", True),
        ],
    )
    async def test_structured_gateway_error_codes_pick_settle_or_refund(self, code, expect_settle):
        binding = _binding(market_item_id=uuid.uuid4())
        error = _http_status_error(502, {"code": code, "reached_upstream": True, "message": "x"})

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as mock_settle,
            patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock) as mock_refund,
            patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock, side_effect=error),
        ):
            result = await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        assert result == "Error: MCP Gateway rejected the call"
        if expect_settle:
            mock_settle.assert_awaited_once()
            mock_refund.assert_not_awaited()
        else:
            mock_refund.assert_awaited_once()
            mock_settle.assert_not_awaited()


# ---------- Stage 8: metrics wiring, fully mocked ---------------------------------


@pytest.mark.asyncio
class TestMcpCreditsMetrics:
    async def test_settled_call_records_spend_in_mcr(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock),
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"ok": True},
            ),
            patch("app.core.mcp_gateway.record_mcp_credits_spent") as mock_spent,
            patch("app.core.mcp_gateway.record_mcp_credits_denied") as mock_denied,
            patch("app.core.mcp_gateway.record_mcp_credits_refund") as mock_refund,
        ):
            await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        mock_spent.assert_called_once_with(binding.market_slug, 300)
        mock_denied.assert_not_called()
        mock_refund.assert_not_called()

    async def test_free_call_records_no_spend(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=0),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"ok": True},
            ),
            patch("app.core.mcp_gateway.record_mcp_credits_spent") as mock_spent,
        ):
            await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        mock_spent.assert_not_called()

    async def test_insufficient_credits_records_denial_with_reason(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=1000),
            patch(
                "app.core.mcp_credits.reserve_call",
                new_callable=AsyncMock,
                side_effect=InsufficientCredits(1000, 200),
            ),
            patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock),
            patch("app.core.mcp_gateway.record_mcp_credits_denied") as mock_denied,
        ):
            await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        mock_denied.assert_called_once_with("insufficient")

    async def test_gateway_unreachable_records_refund_reason(self):
        binding = _binding(market_item_id=uuid.uuid4())

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock),
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("no route"),
            ),
            patch("app.core.mcp_gateway.record_mcp_credits_refund") as mock_refund_metric,
        ):
            await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        mock_refund_metric.assert_called_once_with("gateway_unreachable")

    async def test_structured_gateway_error_records_refund_reason_as_the_code(self):
        binding = _binding(market_item_id=uuid.uuid4())
        error = _http_status_error(
            502, {"code": "mcp_unreachable", "reached_upstream": False, "message": "x"}
        )

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock),
            patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock, side_effect=error),
            patch("app.core.mcp_gateway.record_mcp_credits_refund") as mock_refund_metric,
        ):
            await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        mock_refund_metric.assert_called_once_with("mcp_unreachable")

    async def test_structured_rpc_error_settles_and_records_spend_not_refund(self):
        binding = _binding(market_item_id=uuid.uuid4())
        error = _http_status_error(
            502, {"code": "mcp_rpc_error", "reached_upstream": True, "message": "x"}
        )

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=300),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock),
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock),
            patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock, side_effect=error),
            patch("app.core.mcp_gateway.record_mcp_credits_spent") as mock_spent,
            patch("app.core.mcp_gateway.record_mcp_credits_refund") as mock_refund_metric,
        ):
            await execute_mcp_tool(binding, uuid.uuid4(), uuid.uuid4(), {})

        mock_spent.assert_called_once_with(binding.market_slug, 300)
        mock_refund_metric.assert_not_called()



# ---------- DB integration (requires PostgreSQL) ---------------------------------

DB_URL = os.environ.get("DATABASE_URL", "")
pytestmark_db = pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator:
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import SQLModel

    import app.models  # noqa: F401  (populate metadata)

    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
@pytest.mark.db
class TestChargingDB:
    @pytest_asyncio.fixture(autouse=True)
    async def _reset_process_engine_pool(self):
        """The credit helpers under test (reserve_call, settle_call, ...) go
        through app.db.session.engine's own connection pool, not db_session's
        fixture-scoped one — it's a process-global engine, deliberately never
        recreated per test. pytest-asyncio hands each test function a fresh
        event loop, so a pooled asyncpg connection left over from a previous
        test's (now-closed) loop makes the next test blow up with
        "cannot perform operation: another operation is in progress" the
        moment a short-session function tries to reuse it. Disposing the pool
        before each test forces fresh, current-loop connections.
        """
        from app.db.session import engine

        await engine.dispose()
        yield

    async def _make_user(self, db) -> uuid.UUID:
        from app.models.user import User

        user_id = uuid.uuid4()
        db.add(User(id=user_id, email=f"{user_id}@test.com", hashed_password="x"))
        await db.commit()
        return user_id

    async def test_reserve_settle_round_trip_matches_ledger_and_balance(self, db_session):
        from app.core.mcp_credits import (
            get_wallet_balance_mcr,
            reserve_call,
            settle_call,
            wallet_summary,
        )

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)  # issues the first weekly grant
        price = credits_to_mcr(1)
        call_id = uuid.uuid4()

        await reserve_call(
            user_id, price, call_id=call_id, installed_item_id=uuid.uuid4(),
            market_item_id=uuid.uuid4(), market_slug="mcp-searxng", tool_name="search",
        )
        after_reserve = await get_wallet_balance_mcr(user_id)
        assert after_reserve == starting - price

        await settle_call(call_id)  # settling must not move the balance again
        after_settle = await get_wallet_balance_mcr(user_id)
        assert after_settle == after_reserve

        summary = await wallet_summary(user_id)
        assert summary["spent_this_period"] == pytest.approx(1.0)

    async def test_wallet_summary_self_heals_a_grant_frozen_below_the_current_plan(self, db_session):
        """Regression for a wallet whose window was rolled while the effective
        plan grant was wrongly 0 (e.g. Stage 6's jsonb/json plan-table migration
        bug) — a real wallet in this state stayed stuck reporting a 0 grant for
        the rest of its week because nothing ever re-applied the corrected plan
        limit mid-window. ``apply_plan_change_if_needed`` (wired into
        ``wallet_summary`` via ``_rolled_wallet``) must top it up on the very
        next read instead of waiting for the natural weekly roll.
        """
        from app.core.mcp_credits import get_or_create_wallet, roll_period_if_due, wallet_summary

        user_id = await self._make_user(db_session)
        wallet = await get_or_create_wallet(db_session, user_id)
        # Simulate the historical bug: the window rolled while the effective
        # grant resolved to 0, so both the grant balance and the frozen
        # period_grant_mcr landed on 0.
        wallet = await roll_period_if_due(db_session, wallet, 0)
        await db_session.commit()
        assert wallet.period_grant_mcr == 0
        assert wallet.grant_balance_mcr == 0

        summary = await wallet_summary(user_id)  # _make_user has no UserQuota -> DEFAULT_PLAN_TIER (hobby, 50cr)

        assert summary["weekly_grant"] == 50.0
        assert summary["grant_total"] == 50.0
        assert summary["grant_remaining"] == 50.0
        assert summary["balance"] == 50.0

        # A second read is a no-op — the top-up must not double-apply.
        summary_again = await wallet_summary(user_id)
        assert summary_again["grant_total"] == 50.0
        assert summary_again["balance"] == 50.0

    async def test_refund_restores_exactly_what_was_reserved(self, db_session):
        from app.core.mcp_credits import get_wallet_balance_mcr, refund_call, reserve_call

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)
        price = credits_to_mcr(2)
        call_id = uuid.uuid4()

        await reserve_call(
            user_id, price, call_id=call_id, installed_item_id=uuid.uuid4(),
            market_item_id=uuid.uuid4(), market_slug="mcp-searxng", tool_name="search",
        )
        await refund_call(call_id, note="gateway unreachable")
        restored = await get_wallet_balance_mcr(user_id)
        assert restored == starting

    async def test_settle_and_refund_are_idempotent_after_the_first_call(self, db_session):
        from app.core.mcp_credits import get_wallet_balance_mcr, refund_call, reserve_call, settle_call

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)
        price = credits_to_mcr(1)
        call_id = uuid.uuid4()

        await reserve_call(
            user_id, price, call_id=call_id, installed_item_id=uuid.uuid4(),
            market_item_id=uuid.uuid4(), market_slug="mcp-searxng", tool_name="search",
        )
        await settle_call(call_id)
        # A refund arriving late (e.g. a stray retry) after settlement must not
        # claw back money that was already earned by the upstream call.
        await refund_call(call_id, note="late refund attempt")
        assert await get_wallet_balance_mcr(user_id) == starting - price

    async def test_retrying_the_same_call_id_does_not_double_spend(self, db_session):
        from app.core.mcp_credits import get_wallet_balance_mcr, reserve_call

        user_id = await self._make_user(db_session)
        starting = await get_wallet_balance_mcr(user_id)
        price = credits_to_mcr(1)
        call_id = uuid.uuid4()

        await reserve_call(
            user_id, price, call_id=call_id, installed_item_id=uuid.uuid4(),
            market_item_id=uuid.uuid4(), market_slug="mcp-searxng", tool_name="search",
        )
        with pytest.raises(Exception):  # unique index on (call_id) where kind='spend'
            await reserve_call(
                user_id, price, call_id=call_id, installed_item_id=uuid.uuid4(),
                market_item_id=uuid.uuid4(), market_slug="mcp-searxng", tool_name="search",
            )
        # The failed retry's own transaction rolled back — only the first
        # reservation's deduction survived.
        assert await get_wallet_balance_mcr(user_id) == starting - price

    async def test_concurrent_reservations_do_not_overdraw_the_wallet(self, db_session):
        from app.core.mcp_credits import get_or_create_wallet, get_wallet_balance_mcr, reserve_call, roll_period_if_due

        user_id = await self._make_user(db_session)
        # Force a small, known balance so two concurrent full-price reservations
        # cannot both succeed.
        wallet = await get_or_create_wallet(db_session, user_id)
        wallet = await roll_period_if_due(db_session, wallet, credits_to_mcr(100))
        wallet.grant_balance_mcr = credits_to_mcr(1)
        db_session.add(wallet)
        await db_session.commit()

        price = credits_to_mcr(1)
        results = await asyncio.gather(
            reserve_call(
                user_id, price, call_id=uuid.uuid4(), installed_item_id=uuid.uuid4(),
                market_item_id=uuid.uuid4(), market_slug="mcp-searxng", tool_name="search",
            ),
            reserve_call(
                user_id, price, call_id=uuid.uuid4(), installed_item_id=uuid.uuid4(),
                market_item_id=uuid.uuid4(), market_slug="mcp-searxng", tool_name="search",
            ),
            return_exceptions=True,
        )
        failures = [r for r in results if isinstance(r, InsufficientCredits)]
        successes = [r for r in results if r is None]
        assert len(failures) == 1
        assert len(successes) == 1
        assert await get_wallet_balance_mcr(user_id) == 0

    async def test_room_mcp_tools_hides_marketplace_item_at_zero_balance(self, db_session, monkeypatch):
        from app.core.mcp_credits import store_price_snapshot
        from app.models.market import RoomMcpServer, UserInstalledItem
        from app.models.room import Room

        user_id = await self._make_user(db_session)
        market_item_id = uuid.uuid4()

        room = Room(name="r", creator_id=user_id)
        db_session.add(room)
        await db_session.flush()

        item = UserInstalledItem(
            user_id=user_id, market_item_id=market_item_id, market_slug="mcp-searxng",
            type="mcp", name="SearXNG", status="active", is_custom=False,
        )
        db_session.add(item)
        await db_session.flush()

        db_session.add(RoomMcpServer(room_id=room.id, user_installed_item_id=item.id, added_by=user_id))
        # Balance is zero (fresh free-tier-equivalent wallet); price is not.
        await store_price_snapshot(
            db_session, market_item_id=market_item_id, market_slug="mcp-searxng",
            call_price_mcr=credits_to_mcr(1), tool_price_overrides={},
        )
        await db_session.commit()
        room_id = room.id

        monkeypatch.setattr(
            "app.core.mcp_credits.get_wallet_balance_mcr", AsyncMock(return_value=0)
        )
        monkeypatch.setattr(
            "app.core.mcp_gateway._gateway_get",
            AsyncMock(return_value={"tools": [{"name": "search", "description": "d", "inputSchema": {}}]}),
        )
        monkeypatch.setattr("app.core.mcp_gateway.emit_event", AsyncMock())

        with patch("app.core.mcp_gateway.record_mcp_credits_denied") as mock_denied:
            ant, oai, bindings = await room_mcp_tools(room_id, user_id, db_session)

        assert ant == []
        assert oai == []
        assert bindings == {}
        mock_denied.assert_called_once_with("insufficient")

    async def test_topup_with_source_and_external_ref_lands_on_the_ledger_row(self, db_session):
        """Stage 8's Stripe seam: unused by any caller today, but the plumbing
        from topup() through to the ledger row must actually work once a
        payment webhook starts passing these."""
        from app.core.mcp_credits import list_ledger, topup

        user_id = await self._make_user(db_session)
        await topup(
            user_id, credits_to_mcr(20), bucket="purchased", note="checkout",
            source="stripe", external_ref="pi_test_123",
        )

        entries, _ = await list_ledger(user_id, kind="topup")
        assert len(entries) == 1
        assert entries[0]["source"] == "stripe"
        assert entries[0]["external_ref"] == "pi_test_123"

    async def test_duplicate_external_ref_topup_is_rejected_by_the_unique_index(self, db_session):
        """A retried webhook delivery must not double-credit the wallet — the
        partial unique index on external_ref is what stops it."""
        from app.core.mcp_credits import topup

        user_id = await self._make_user(db_session)
        await topup(
            user_id, credits_to_mcr(5), bucket="purchased",
            source="stripe", external_ref="pi_test_dup",
        )
        with pytest.raises(Exception):  # unique index on external_ref where not null
            await topup(
                user_id, credits_to_mcr(5), bucket="purchased",
                source="stripe", external_ref="pi_test_dup",
            )

    async def test_list_ledger_pages_newest_first_and_filters_by_kind(self, db_session):
        """Stage 5: the Admin Center's ledger table relies on this ordering,
        pagination and kind filter directly. The free-tier default plan grants
        0 weekly credits, which issues no ``grant`` ledger row at all (see
        ``roll_period``'s ``if period_grant:`` guard) — so only the two topups
        below are guaranteed to exist, and the assertions don't depend on the
        default plan tier's grant amount."""
        from app.core.mcp_credits import list_ledger, topup

        user_id = await self._make_user(db_session)
        await topup(user_id, credits_to_mcr(10), bucket="purchased", note="first")
        await topup(user_id, credits_to_mcr(5), bucket="purchased", note="second")

        page, total = await list_ledger(user_id, limit=1, offset=0)
        assert total >= 2
        assert len(page) == 1
        assert page[0]["note"] == "second"  # newest first

        topups, topup_total = await list_ledger(user_id, kind="topup")
        assert topup_total == 2
        assert [e["note"] for e in topups] == ["second", "first"]
        assert all(e["kind"] == "topup" for e in topups)

    async def test_adjust_records_the_admin_actor_on_the_ledger_row(self, db_session):
        """Stage 5: the admin endpoint passes actor="admin:<login>" through to
        adjust()/topup() — confirm it lands on the ledger row untouched."""
        from app.core.mcp_credits import adjust, get_wallet_balance_mcr, list_ledger, topup

        user_id = await self._make_user(db_session)
        await get_wallet_balance_mcr(user_id)
        await topup(user_id, credits_to_mcr(5), bucket="purchased")
        result = await adjust(
            user_id, -credits_to_mcr(1), bucket="purchased", note="incident refund",
            actor="admin:alice",
        )
        assert result["purchased"] == pytest.approx(4.0)

        entries, _ = await list_ledger(user_id, kind="adjustment")
        assert len(entries) == 1
        assert entries[0]["actor"] == "admin:alice"
        assert entries[0]["note"] == "incident refund"


# ---------- Stage 9 (internal design notes): the VPS transport bills like any item ----


@pytest.mark.asyncio
class TestVpsMcpCharging:
    """The whole point of shipping legion-vps-mcp as a Market item with an
    internal_endpoint (§3.2/§18.1) is that billing needs *no* new code: the
    same reserve -> call -> settle path an mcp-vision call takes. These two
    tests pin that, and pin that a policy refusal still costs nothing."""

    @staticmethod
    def _vps_binding(tool_name: str = "vps_list_dir") -> McpToolBinding:
        return _binding(
            market_item_id=uuid.uuid4(),
            market_slug="legion-vps-mcp",
            tool_name=tool_name,
        )

    @staticmethod
    def _db(mode: str) -> MagicMock:
        db = MagicMock()
        db.scalar = AsyncMock(return_value=mode)
        db.get = AsyncMock(return_value=None)
        return db

    async def test_vps_call_reserves_and_settles_like_any_market_item(self):
        from app.core import vps_inventory

        binding = self._vps_binding()
        conn = vps_inventory.VpsConnection(
            legion_conn={"room_id": "r", "server": "PROD", "host": "203.0.113.10", "port": 22},
            mask_values={"VPS_PROD_KEY": "secret-key-material"},
        )

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock, return_value=3),
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock) as mock_reserve,
            patch("app.core.mcp_credits.settle_call", new_callable=AsyncMock) as mock_settle,
            patch("app.core.mcp_credits.refund_call", new_callable=AsyncMock) as mock_refund,
            patch(
                "app.core.vps_inventory.resolve_connection",
                new_callable=AsyncMock,
                return_value=conn,
            ),
            patch(
                "app.core.mcp_gateway._gateway_post",
                new_callable=AsyncMock,
                return_value={"ok": True},
            ) as mock_post,
        ):
            result = await execute_mcp_tool(
                binding, uuid.uuid4(), uuid.uuid4(), {"server": "PROD", "path": "/etc/nginx"},
                db=self._db("readonly"),
            )

        assert result == json.dumps({"ok": True}, ensure_ascii=False)
        mock_reserve.assert_awaited_once()
        assert mock_reserve.await_args.kwargs["market_slug"] == "legion-vps-mcp"
        mock_settle.assert_awaited_once_with(mock_reserve.await_args.kwargs["call_id"])
        mock_refund.assert_not_awaited()
        # The injected connection goes to the Gateway, never the price path.
        gateway_payload = mock_post.await_args.args[-1]
        assert gateway_payload["arguments"]["_legion_conn"] == conn.legion_conn

    async def test_policy_denial_costs_nothing(self):
        # §3.7/Stage 5: a write tool in a readonly Room is refused *before*
        # anything reserves credits — the AC "a policy refusal charges
        # nothing", re-checked here from the billing side.
        binding = self._vps_binding("vps_write_file")

        with (
            patch("app.core.mcp_credits.resolve_price_mcr", new_callable=AsyncMock) as mock_price,
            patch("app.core.mcp_credits.reserve_call", new_callable=AsyncMock) as mock_reserve,
            patch("app.core.vps_inventory.resolve_connection", new_callable=AsyncMock) as mock_conn,
            patch("app.core.mcp_gateway._gateway_post", new_callable=AsyncMock) as mock_post,
        ):
            result = await execute_mcp_tool(
                binding, uuid.uuid4(), uuid.uuid4(),
                {"server": "PROD", "path": "/etc/nginx/nginx.conf", "content": "x"},
                db=self._db("readonly"),
            )

        assert result.startswith("Error: [policy_denied]")
        mock_price.assert_not_awaited()
        mock_reserve.assert_not_awaited()
        mock_conn.assert_not_awaited()
        mock_post.assert_not_awaited()
