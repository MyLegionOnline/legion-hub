"""
Session usage tracking and the temporary legacy cumulative guard.

Public API:
    TOKEN_LIMIT                  — historical default for the context window
    estimate_next_call_tokens    — heuristic token estimate for the next LLM call
    check_token_budget           — guard: True = proceed, False = limit reached + close triggered
    track_llm_usage              — increment session.tokens_in/tokens_out/cost_usd after a call
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import emit_event
from app.core.context_policy import uses_cumulative_limit

if TYPE_CHECKING:
    from app.models.session.base import Session

logger = logging.getLogger(__name__)

TOKEN_LIMIT = 100_000
_DEFAULT_OUTPUT_ESTIMATE = 1_000   # conservative fixed estimate for the model reply
_CHARS_PER_TOKEN = 4               # standard rule-of-thumb: 1 token ≈ 4 chars


# Provider-cache price corrections waiting for the turn's own charge.
#
# A tool loop reports cache usage after every request but charges the turn's
# tokens only once, when the turn ends.  A correction for cached input is
# negative (a cache read is cheaper than input), so writing it to the session
# straight away left a negative cost whenever the turn never reached its charge
# (workspace pause, restart) — a direct video-agent room, 2026-10-24: -$2.29.
# Corrections wait here and land together with the tokens they belong to; a
# turn that is never charged loses its correction along with its tokens.
_PENDING_CACHE_COST: "OrderedDict[str, Decimal]" = OrderedDict()
_PENDING_CACHE_COST_MAX = 10_000


def defer_cache_cost(session_id: object, cost_usd: float) -> None:
    """Hold a cache price correction until ``track_llm_usage`` charges the turn."""
    if not cost_usd:
        return
    key = str(session_id)
    _PENDING_CACHE_COST[key] = _PENDING_CACHE_COST.get(key, Decimal("0")) + Decimal(
        str(cost_usd)
    )
    _PENDING_CACHE_COST.move_to_end(key)
    # Turns that never get charged leave entries behind; keep the map bounded.
    while len(_PENDING_CACHE_COST) > _PENDING_CACHE_COST_MAX:
        _PENDING_CACHE_COST.popitem(last=False)


def pending_cache_cost(session_id: object) -> Decimal:
    return _PENDING_CACHE_COST.get(str(session_id), Decimal("0"))


def estimate_next_call_tokens(session: "Session", next_input: str) -> int:
    """Estimate the token cost of the next LLM call.

    Input token estimate: len(next_input) // 4, at minimum 1.
    Output token estimate: a fixed conservative DEFAULT_OUTPUT_ESTIMATE.
    The session parameter is accepted for future context-aware implementations.
    """
    input_tokens = max(1, len(next_input) // _CHARS_PER_TOKEN)
    return input_tokens + _DEFAULT_OUTPUT_ESTIMATE


async def check_token_budget(
    session: "Session",
    next_input: str,
    db: AsyncSession,
) -> bool:
    """Run the legacy cumulative-session guard when it is explicitly enabled.

    In the default ``window`` mode, cumulative usage is accounting only: this
    function deliberately permits the call regardless of the accumulated
    counters.  Per-request admission is performed by ``ContextAssembler``.

    Returns True  — window mode is active or legacy budget is sufficient.
    Returns False — the projected total (tokens_in + tokens_out + estimate)
                    would reach or exceed TOKEN_LIMIT.  In that case the session
                    is closed with close_reason='token_limit' and a
                    SESSION_TOKEN_LIMIT_REACHED event is emitted before returning.
    """
    if not session.is_open:
        logger.debug(
            "token_budget: session %s is already closed — skip budget check", session.id
        )
        return False

    if not uses_cumulative_limit():
        return True

    estimate = estimate_next_call_tokens(session, next_input)
    projected = session.tokens_in + session.tokens_out + estimate

    if projected < TOKEN_LIMIT:
        return True

    logger.info(
        "token_budget: limit reached for session %s "
        "(tokens_in=%d tokens_out=%d estimate=%d projected=%d limit=%d)",
        session.id,
        session.tokens_in,
        session.tokens_out,
        estimate,
        projected,
        TOKEN_LIMIT,
    )
    try:
        from app.core.metrics import record_legacy_cumulative_guard_hit

        record_legacy_cumulative_guard_hit()
    except Exception:  # pragma: no cover - metrics must not affect closing
        logger.debug("token_budget: legacy-guard metric failed", exc_info=True)

    now = datetime.now(timezone.utc)
    session.closed_at = now
    session.close_reason = "token_limit"
    db.add(session)
    await db.flush()

    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_TOKEN_LIMIT_REACHED",
        content=(
            f"Session {session.id} reached the {TOKEN_LIMIT:,}-token limit "
            f"({session.tokens_in + session.tokens_out:,} tokens used). "
            "Close flow triggered."
        ),
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": session.kind,
            "tokens_in": session.tokens_in,
            "tokens_out": session.tokens_out,
            "estimate": estimate,
            "projected": projected,
            "limit": TOKEN_LIMIT,
        },
    )
    return False


def track_llm_usage(
    session: "Session",
    tokens_in: int,
    tokens_out: int,
    cost_usd: float = 0.0,
    reasoning_tokens: int = 0,
    model: str | None = None,
) -> None:
    """Increment session token counters and cost after a completed LLM call.

    Mutates *session* in-place.  The caller is responsible for flushing /
    committing the change to the database.

    ``model`` is the model the call actually ran on, recorded on the session
    row.  Without it the Sessions tab had to derive the model from the agent's
    *current* ``llm_config``, so rebinding an agent silently rewrote the model
    of every session it had ever run — work done by one model was reported as
    the other's.  Last write wins: a session that rotated models mid-way is
    labelled by its most recent turn.

    ``reasoning_tokens`` is recorded on its own column and deliberately NOT
    added to ``tokens_out``: the provider already counts thinking inside the
    completion tokens (reasoning_and_router_model_plan §7.3), so adding it
    again would bill the user twice for the same tokens.
    """
    session.tokens_in += tokens_in
    session.tokens_out += tokens_out
    if model:
        try:
            session.model_id = str(model)[:200]
        except (AttributeError, TypeError):
            # Plain stand-ins in unit tests may not carry the column.
            pass
    if reasoning_tokens:
        session.reasoning_tokens = (getattr(session, "reasoning_tokens", 0) or 0) + int(
            reasoning_tokens
        )
    session.cost_usd += Decimal(str(cost_usd))
    if tokens_in or tokens_out:
        # The turn's tokens are charged now, so its cache corrections land too.
        # A correction left by an uncharged turn may still be waiting; it can
        # discount this charge at most down to zero, never below.
        pending = _PENDING_CACHE_COST.pop(str(getattr(session, "id", "")), Decimal("0"))
        session.cost_usd += max(pending, -Decimal(str(cost_usd)))
    try:
        from app.core.metrics import record_session_cumulative_tokens

        record_session_cumulative_tokens(
            kind=str(getattr(session, "kind", "unknown") or "unknown"),
            tokens=session.tokens_in + session.tokens_out,
        )
    except Exception:  # pragma: no cover - accounting must not fail for metrics
        logger.debug("token_budget: cumulative-usage metric failed", exc_info=True)


def log_llm_call(
    *,
    purpose: str,
    model: str | None,
    session: "Session | None" = None,
    room_id: object = None,
    kind: str | None = None,
    agent_nick: str | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    """One structured log line per LLM call — ``internal design notes`` §8.1.

    Makes "who planned this, with which model" a checked fact instead of a guess
    from the model name in a stack of logs. ``purpose`` is one of
    ``route | dialogue | plan_decompose | plan_digest | session_summary |
    proxy_turn``.
    """
    if session is not None:
        room_id = room_id or getattr(session, "room_id", None)
        kind = kind or getattr(session, "kind", None)
    sid = getattr(session, "id", None)
    logger.info(
        "llm_call room=%s session=%s kind=%s purpose=%s agent=%s model=%s "
        "tokens_in=%s tokens_out=%s",
        room_id,
        sid,
        kind or "-",
        purpose,
        f"@{agent_nick}" if agent_nick else "-",
        model or "-",
        tokens_in or 0,
        tokens_out or 0,
    )


def update_context_window_metrics(
    session: "Session",
    max_window_in: int,
    last_window_in: int,
) -> None:
    """Update max/last context-window size metrics on the session.

    Call after track_llm_usage to record the per-call prompt size.
    Mutates *session* in-place; caller flushes.
    """
    if max_window_in > 0:
        session.max_context_window_tokens = max(
            session.max_context_window_tokens, max_window_in
        )
    if last_window_in > 0:
        session.last_context_window_tokens = last_window_in
