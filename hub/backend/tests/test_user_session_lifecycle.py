"""Stage 1 of internal design notes — user-session lifecycle.

Covers the five defects that together made a user-facing session immortal:
  * 1.1 accepting a plan closes the session that proposed it (C4)
  * 1.2 the user_cm_p2 prompt no longer forbids close_session (C3)
  * 1.3 turn/age/window limits rotate the session before the next LLM call (C2)
  * 1.4 the watchdog closes idle user sessions and orphans left by a restart (C5)
  * 5.2 context-window metrics are written for user sessions (O2)

All provider and database access is stubbed; no network, no PostgreSQL.
"""
from __future__ import annotations

import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.user_session_limits import (
    ROTATE_MAX_AGE,
    ROTATE_MAX_TURNS,
    ROTATE_WINDOW_RATIO,
    rotation_reason,
)
from app.core.user_sessions import (
    _LLMTurnResult,
    _get_history,
    _maybe_warn_growth,
    process_user_turn,
)
from app.models.session.base import CLOSE_REASONS

NOW = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    kind: str = "user_cm_p2",
    *,
    messages: int = 0,
    age_hours: float = 0.0,
    last_window: int = 0,
) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.side = None
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = last_window
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.started_at = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    s.last_activity_at = None
    s.user_id = uuid.uuid4()
    s.agent_id = uuid.uuid4() if "agent" in kind else None
    s.chat_master_agent_id = uuid.uuid4() if "cm" in kind else None
    s.goal_id = None
    s.task_id = None
    s.summary = None
    s.pending_actions = None
    s.context_snapshot = {
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(messages)
        ]
    }
    return s


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    result.scalars.return_value.all.return_value = []
    db.execute = AsyncMock(return_value=result)
    db.get = AsyncMock(return_value=None)
    return db


def _text_result(text: str = "reply", tokens_in: int = 70_000) -> _LLMTurnResult:
    return _LLMTurnResult(
        text=text, tool_input=None, tokens_in=tokens_in, tokens_out=100,
        model="claude-haiku-4-5-20251001",
        max_window_in=tokens_in, last_window_in=tokens_in,
    )


def _close_result() -> _LLMTurnResult:
    return _LLMTurnResult(
        text=None,
        tool_input={"agent_memory_patch": {"content": "state"}, "summary": "dialogue summary"},
        tokens_in=500, tokens_out=50, model="claude-haiku-4-5-20251001",
        max_window_in=500, last_window_in=500,
    )


# ---------------------------------------------------------------------------
# 1.5 — close reasons
# ---------------------------------------------------------------------------


def test_new_close_reasons_registered():
    for reason in ("rotated", "idle", "stale_cleanup"):
        assert reason in CLOSE_REASONS


# ---------------------------------------------------------------------------
# 1.3 — the limit policy itself
# ---------------------------------------------------------------------------


class TestRotationPolicy:
    def test_no_rotation_for_a_fresh_session(self):
        assert rotation_reason(message_count=4, started_at=NOW, now=NOW) is None

    def test_turn_limit(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_MAX_TURNS", "40")
        assert rotation_reason(message_count=40, started_at=NOW, now=NOW) == ROTATE_MAX_TURNS
        assert rotation_reason(message_count=39, started_at=NOW, now=NOW) is None

    def test_age_limit(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_MAX_AGE_HOURS", "24")
        old = NOW - timedelta(hours=25)
        assert rotation_reason(message_count=2, started_at=old, now=NOW) == ROTATE_MAX_AGE

    def test_naive_started_at_is_read_as_utc(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_MAX_AGE_HOURS", "24")
        old = (NOW - timedelta(hours=25)).replace(tzinfo=None)
        assert rotation_reason(message_count=2, started_at=old, now=NOW) == ROTATE_MAX_AGE

    def test_window_share(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_WINDOW_SOFT_RATIO", "0.5")
        assert rotation_reason(
            message_count=2, started_at=NOW, last_window_tokens=70_000,
            window_limit=100_000, now=NOW,
        ) == ROTATE_WINDOW_RATIO
        assert rotation_reason(
            message_count=2, started_at=NOW, last_window_tokens=20_000,
            window_limit=100_000, now=NOW,
        ) is None

    def test_unknown_window_limit_never_rotates(self):
        assert rotation_reason(
            message_count=2, started_at=NOW, last_window_tokens=999_999,
            window_limit=0, now=NOW,
        ) is None

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_LIFECYCLE_ENFORCE", "false")
        assert rotation_reason(message_count=10_000, started_at=NOW, now=NOW) is None


# ---------------------------------------------------------------------------
# 1.3 — rotation inside process_user_turn
# ---------------------------------------------------------------------------


@contextmanager
def _turn_env(llm_results: list[_LLMTurnResult], successor: MagicMock):
    """Stub every collaborator of a user turn; yield the session repository mock.

    ``llm_results`` are handed out in order: the graceful close of the rotating
    session first, then the replayed turn in its successor.
    """
    calls = iter(llm_results)

    async def _fake_call(*_args: Any, **_kwargs: Any) -> _LLMTurnResult:
        return next(calls)

    repo_instance = AsyncMock()
    repo_instance.create = AsyncMock(return_value=successor)

    with ExitStack() as stack:
        for target, kwargs in (
            ("build_memory_layer", {"new_callable": AsyncMock, "return_value": {}}),
            ("build_system_prompt", {"return_value": "SYS"}),
            ("_get_agent_llm_config", {"new_callable": AsyncMock, "return_value": {}}),
            ("_call_llm_with_tools", {"side_effect": _fake_call}),
            ("apply_close_actions", {"new_callable": AsyncMock}),
            ("emit_event", {"new_callable": AsyncMock}),
            ("_context_window_limit", {"new_callable": AsyncMock, "return_value": 100_000}),
            ("SessionRepository", {"return_value": repo_instance}),
        ):
            stack.enter_context(patch(f"app.core.user_sessions.{target}", **kwargs))
        manager = stack.enter_context(patch("app.core.user_sessions.manager"))
        manager.broadcast = AsyncMock()
        yield repo_instance


@pytest.fixture(autouse=True)
def _no_workspace():
    with patch(
        "app.core.workspace_tools.workspace_tools_available",
        new_callable=AsyncMock, return_value=False,
    ):
        yield


class TestRotation:
    @pytest.mark.asyncio
    async def test_turn_limit_rotates_and_answers_in_the_successor(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_MAX_TURNS", "40")
        session = _make_session("user_cm_p2", messages=40)
        successor = _make_session("user_cm_p2")
        db = _make_db()

        with _turn_env(
            [_close_result(), _text_result("reply in the new session", tokens_in=800)],
            successor,
        ) as repo_instance:
            still_open = await process_user_turn(session, "and what next?", db)

        # Old session closed with the new reason, summary preserved for the successor.
        assert session.closed_at is not None
        assert session.close_reason == "rotated"
        assert session.summary == "dialogue summary"

        # A successor of the same kind was created for the same participants.
        payload = repo_instance.create.call_args[0][0]
        assert payload.kind == "user_cm_p2"
        assert payload.user_id == session.user_id
        assert payload.chat_master_agent_id == session.chat_master_agent_id

        # The user's message was answered in the successor, not dropped.
        assert still_open is True
        successor_history = _get_history(successor)
        assert successor_history[0]["role"] == "user"
        assert "and what next?" in successor_history[0]["content"]
        # ...carrying the previous session's summary, not its transcript.
        assert "dialogue summary" in successor_history[0]["content"]
        assert "m0" not in successor_history[0]["content"]
        assert successor_history[-1]["content"] == "reply in the new session"

    @pytest.mark.asyncio
    async def test_window_share_rotates(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_WINDOW_SOFT_RATIO", "0.5")
        session = _make_session("user_cm_p2", messages=6, last_window=70_000)
        successor = _make_session("user_cm_p2")
        db = _make_db()

        with _turn_env([_close_result(), _text_result(tokens_in=900)], successor):
            await process_user_turn(session, "next", db)

        assert session.close_reason == "rotated"

    @pytest.mark.asyncio
    async def test_successor_never_rotates_again(self, monkeypatch):
        """The replayed turn must not rotate a second time (no recursion)."""
        monkeypatch.setenv("USER_SESSION_MAX_TURNS", "40")
        session = _make_session("user_cm_p2", messages=40)
        # A successor that would itself trip the limit if it were re-checked.
        successor = _make_session("user_cm_p2", messages=40)
        db = _make_db()

        with _turn_env([_close_result(), _text_result()], successor) as repo_instance:
            await process_user_turn(session, "next", db)

        assert repo_instance.create.await_count == 1
        assert successor.close_reason is None

    @pytest.mark.asyncio
    async def test_under_the_limits_nothing_rotates(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_MAX_TURNS", "40")
        session = _make_session("user_cm_p2", messages=4)
        db = _make_db()

        with _turn_env([_text_result(tokens_in=900)], _make_session()) as repo_instance:
            still_open = await process_user_turn(session, "hi", db)

        assert still_open is True
        assert session.close_reason is None
        repo_instance.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kill_switch_disables_rotation(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_LIFECYCLE_ENFORCE", "false")
        monkeypatch.setenv("USER_SESSION_MAX_TURNS", "40")
        session = _make_session("user_cm_p2", messages=400)
        db = _make_db()

        with _turn_env([_text_result(tokens_in=900)], _make_session()) as repo_instance:
            await process_user_turn(session, "hi", db)

        assert session.close_reason is None
        repo_instance.create.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5.2 — window metrics are recorded for user sessions
# ---------------------------------------------------------------------------


class TestWindowMetrics:
    @pytest.mark.asyncio
    async def test_turn_records_max_and_last_window(self):
        session = _make_session("user_cm_p2", messages=2)
        db = _make_db()

        with _turn_env([_text_result(tokens_in=12_345)], _make_session()):
            await process_user_turn(session, "question", db)

        assert session.last_context_window_tokens == 12_345
        assert session.max_context_window_tokens == 12_345

    @pytest.mark.asyncio
    async def test_turn_stamps_last_activity(self):
        session = _make_session("user_cm_p2", messages=2)
        db = _make_db()

        with _turn_env([_text_result(tokens_in=900)], _make_session()):
            await process_user_turn(session, "question", db)

        assert session.last_activity_at is not None


# ---------------------------------------------------------------------------
# 1.1 — accepting a plan closes the session
# ---------------------------------------------------------------------------


class TestConfirmPlanClosesSession:
    @pytest.mark.asyncio
    async def test_confirm_plan_closes_and_announces(self):
        from app.api.v1.chat import confirm_plan_session

        session = _make_session("user_cm_p2", messages=6)
        session.context_snapshot = {
            **session.context_snapshot,
            "pending_plan": {"goal_title": "SEO", "goal_description": "optimization"},
        }
        db = _make_db()
        result = MagicMock()
        result.scalars.return_value.first.return_value = session
        db.execute = AsyncMock(return_value=result)

        user = MagicMock()
        user.full_name = "Test User"
        user.email = "t@example.com"

        with (
            patch("app.core.session_close.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.user_sessions._trigger_plan_pipeline", new_callable=AsyncMock),
            patch("app.core.events.emit_event", new_callable=AsyncMock) as emit,
        ):
            out = await confirm_plan_session(session.id, db=db, current_user=user)

        assert out["status"] == "confirmed"
        assert session.closed_at is not None
        assert session.close_reason == "task_done"
        assert "SEO" in (session.summary or "")
        assert "pending_plan" not in session.context_snapshot
        assert emit.await_args.kwargs["event_type"] == "SESSION_CLOSED"


# ---------------------------------------------------------------------------
# 1.4 — watchdog
# ---------------------------------------------------------------------------


class TestWatchdogUserSessions:
    @pytest.mark.asyncio
    async def test_idle_sessions_are_closed(self, monkeypatch):
        from app.core.watchdog import close_idle_user_sessions

        monkeypatch.setenv("USER_SESSION_IDLE_MINUTES", "90")
        idle = _make_session("user_cm_p2")
        idle.last_activity_at = NOW - timedelta(minutes=120)
        db = _make_db()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [idle]
        db.execute = AsyncMock(return_value=result)

        with patch("app.core.watchdog.emit_event", new_callable=AsyncMock):
            closed = await close_idle_user_sessions(db, now=NOW)

        assert closed == 1
        assert idle.close_reason == "idle"
        assert idle.closed_at == NOW

    @pytest.mark.asyncio
    async def test_kill_switch_skips_idle_close(self, monkeypatch):
        from app.core.watchdog import close_idle_user_sessions

        monkeypatch.setenv("USER_SESSION_LIFECYCLE_ENFORCE", "false")
        db = _make_db()

        closed = await close_idle_user_sessions(db, now=NOW)

        assert closed == 0
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_restart_closes_orphaned_user_sessions(self):
        from app.core.watchdog import cleanup_orphaned_sessions

        orphan = _make_session("user_agent_p2")
        db = _make_db()
        proxy_result = MagicMock()
        proxy_result.scalars.return_value.all.return_value = []
        user_result = MagicMock()
        user_result.scalars.return_value.all.return_value = [orphan]
        db.execute = AsyncMock(side_effect=[proxy_result, user_result])

        with patch("app.core.watchdog.emit_event", new_callable=AsyncMock), patch(
            "app.core.watchdog.close_stale_plan_sessions", new_callable=AsyncMock, return_value=0
        ):
            closed = await cleanup_orphaned_sessions(db, now=NOW)

        assert closed == 1
        assert orphan.close_reason == "orphaned_restart"
        assert orphan.closed_at == NOW


# ---------------------------------------------------------------------------
# 1.2 — the prompt no longer forbids close_session
# ---------------------------------------------------------------------------


def test_user_cm_p2_prompt_allows_close_session():
    from app.core.memory_layer import _KIND_HEADERS

    header = _KIND_HEADERS["user_cm_p2"]
    assert "Do NOT call close_session — the Accept button" not in header
    assert "DO call close_session" in header


# ---------------------------------------------------------------------------
# 5.6 — growth warning banner (fires at the same threshold rotation uses)
# ---------------------------------------------------------------------------


class TestGrowthWarning:
    @pytest.mark.asyncio
    async def test_fires_once_past_the_soft_ratio(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_WINDOW_SOFT_RATIO", "0.5")
        session = _make_session("user_cm_p2", last_window=60_000)
        db = _make_db()
        emitted = AsyncMock()

        with patch(
            "app.core.user_sessions._context_window_limit",
            new_callable=AsyncMock, return_value=100_000,
        ), patch("app.core.user_sessions.emit_event", emitted):
            await _maybe_warn_growth(session, db)

        emitted.assert_awaited_once()
        assert emitted.await_args.kwargs["event_type"] == "SESSION_GROWING"
        assert session.context_snapshot["growth_warned"] is True

    @pytest.mark.asyncio
    async def test_does_not_fire_below_the_soft_ratio(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_WINDOW_SOFT_RATIO", "0.5")
        session = _make_session("user_cm_p2", last_window=10_000)
        db = _make_db()
        emitted = AsyncMock()

        with patch(
            "app.core.user_sessions._context_window_limit",
            new_callable=AsyncMock, return_value=100_000,
        ), patch("app.core.user_sessions.emit_event", emitted):
            await _maybe_warn_growth(session, db)

        emitted.assert_not_awaited()
        assert "growth_warned" not in (session.context_snapshot or {})

    @pytest.mark.asyncio
    async def test_debounced_to_once_per_session(self, monkeypatch):
        monkeypatch.setenv("USER_SESSION_WINDOW_SOFT_RATIO", "0.5")
        session = _make_session("user_cm_p2", last_window=90_000)
        session.context_snapshot["growth_warned"] = True
        db = _make_db()
        emitted = AsyncMock()

        with patch(
            "app.core.user_sessions._context_window_limit",
            new_callable=AsyncMock, return_value=100_000,
        ), patch("app.core.user_sessions.emit_event", emitted):
            await _maybe_warn_growth(session, db)

        emitted.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_non_user_facing_kinds(self, monkeypatch):
        session = _make_session("cm_agent_proxy", last_window=90_000)
        db = _make_db()
        emitted = AsyncMock()

        with patch(
            "app.core.user_sessions._context_window_limit",
            new_callable=AsyncMock, return_value=100_000,
        ), patch("app.core.user_sessions.emit_event", emitted):
            await _maybe_warn_growth(session, db)

        emitted.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_closed_session_is_skipped(self):
        session = _make_session("user_cm_p2", last_window=90_000)
        session.is_open = False
        db = _make_db()
        emitted = AsyncMock()

        with patch("app.core.user_sessions.emit_event", emitted):
            await _maybe_warn_growth(session, db)

        emitted.assert_not_awaited()
