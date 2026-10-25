"""
Stage 6.2 of internal design notes — "a month of the
project" end-to-end scenario.

The plan's fact-finding section (§2) measured the room "blog" over 8 real days:
one immortal user session (185 messages, 3.18M cumulative input tokens) and a
Room memory that went stale for the same 8 days. This test drives a comparable
shape of traffic — 60 user messages across ten simulated days, each day a
burst of messages followed by a long silence, plus five task/goal completions
during the month — through the same collaborators production uses
(process_user_turn, maybe_refresh_room_memory) and asserts the three numbers
the plan promised instead (§6 "Target contract", §7 Stage 6.2):

  * user-session count over the month >= 8 (not 1)
  * no user session's window ever grows past 25k tokens
  * Room memory is refreshed >= 5 times during the month (not 0)

No database — scripted AsyncMock throughout, reusing the turn-driving harness
from tests/test_user_session_lifecycle.py and the Room-memory-refresh harness
from tests/test_session_close_memory_flush.py.
"""
from __future__ import annotations

import types
import uuid
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.session_close import maybe_refresh_room_memory
from app.core.user_sessions import _LLMTurnResult, process_user_turn

_ROOM_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()
_CM_AGENT_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# Turn-driving harness (mirrors tests/test_user_session_lifecycle.py)
# ---------------------------------------------------------------------------


def _make_session(kind: str = "user_cm_p2") -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = _ROOM_ID
    s.kind = kind
    s.side = None
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = 0
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.started_at = datetime.now(timezone.utc)
    s.last_activity_at = None
    s.user_id = _USER_ID
    s.agent_id = None
    s.chat_master_agent_id = _CM_AGENT_ID
    s.goal_id = None
    s.task_id = None
    s.summary = None
    s.pending_actions = None
    s.context_snapshot = {"messages": []}
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


def _text_result(tokens_in: int) -> _LLMTurnResult:
    return _LLMTurnResult(
        text="ChatMaster reply", tool_input=None, tokens_in=tokens_in, tokens_out=150,
        model="claude-haiku-4-5-20251001",
        max_window_in=tokens_in, last_window_in=tokens_in,
    )


def _close_result() -> _LLMTurnResult:
    return _LLMTurnResult(
        text=None,
        tool_input={"agent_memory_patch": {"content": "state"}, "summary": "summary of the day"},
        tokens_in=500, tokens_out=50, model="claude-haiku-4-5-20251001",
        max_window_in=500, last_window_in=500,
    )


@contextmanager
def _turn_env(llm_results: list[_LLMTurnResult], successor: MagicMock):
    calls = iter(llm_results)

    async def _fake_call(*_args, **_kwargs) -> _LLMTurnResult:
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


# ---------------------------------------------------------------------------
# The scenario: 10 days x 6 messages = 60 user messages
# ---------------------------------------------------------------------------

_DAYS = 10
_MESSAGES_PER_DAY = 6
_MAX_WINDOW_TOKENS = 25_000


class TestSixtyMessagesOverTenDays:
    """§6.2 — the room never collapses back into one immortal session."""

    @pytest.mark.asyncio
    async def test_the_month_spans_at_least_eight_sessions_within_the_window_budget(
        self, monkeypatch,
    ):
        # A session may live at most 6 real hours — short enough that the long
        # silence between two simulated days always forces a rotation, long
        # enough that a single day's own burst of messages never rotates on
        # its own (so day-boundaries are the only source of rotation here).
        monkeypatch.setenv("USER_SESSION_MAX_AGE_HOURS", "6")
        monkeypatch.setenv("USER_SESSION_MAX_TURNS", "40")
        monkeypatch.setenv("USER_SESSION_WINDOW_SOFT_RATIO", "0.5")

        db = _make_db()
        current = _make_session("user_cm_p2")
        all_sessions = [current]

        for day in range(_DAYS):
            for msg_in_day in range(_MESSAGES_PER_DAY):
                tokens = 1_500 + msg_in_day * 200  # varies, always well under 25k
                will_rotate = day > 0 and msg_in_day == 0
                results = (
                    [_close_result(), _text_result(tokens)]
                    if will_rotate else [_text_result(tokens)]
                )
                successor = _make_session("user_cm_p2")

                with _turn_env(results, successor):
                    still_open = await process_user_turn(
                        current, f"day {day}, message {msg_in_day}", db,
                    )

                assert still_open is True
                if will_rotate:
                    assert current.closed_at is not None
                    assert current.close_reason == "rotated"
                    current = successor
                    all_sessions.append(current)
                else:
                    assert current.closed_at is None

            if day < _DAYS - 1:
                # The user comes back the next day: simulate the silence by
                # aging the session past the limit before its first message.
                current.started_at = datetime.now(timezone.utc) - timedelta(hours=30)

        assert len(all_sessions) >= 8, (
            f"a month of traffic must span >= 8 user sessions, got "
            f"{len(all_sessions)} — the room 'blog' fact-finding case had 1"
        )
        for session in all_sessions:
            assert session.max_context_window_tokens <= _MAX_WINDOW_TOKENS, (
                f"session {session.id} window grew to "
                f"{session.max_context_window_tokens} tokens"
            )


# ---------------------------------------------------------------------------
# Room memory during the same month
# ---------------------------------------------------------------------------


class TestRoomMemoryStaysFreshAcrossTheMonth:
    """§6.2 — Room memory is refreshed on task/goal completions, not left
    stale for the whole month the way it was in the room 'blog' (updated once,
    on day 1, then never again for 8 days — defect M4).
    """

    @pytest.mark.asyncio
    async def test_five_task_and_goal_completions_each_refresh_room_memory(self):
        triggers = [
            "task_completed:sitemap",
            "task_completed:hero-section",
            "task_blocked:e2e-tests",
            "goal_completed:devlog",
            "task_completed:seo-audit",
            "goal_completed:seo",
        ]
        assert len(triggers) >= 5

        board_snapshot = {
            "goals": [{
                "title": "Blog SEO optimization", "status": "active", "total": 4,
                "collapsed": {"completed": 2}, "columns": {},
            }],
            "without_goal": {"total": 0},
            "counters": {"total": 4, "completed": 2, "blocked": 1},
        }

        refreshed = 0
        for trigger in triggers:
            stale_row = types.SimpleNamespace(
                room_id=_ROOM_ID,
                content="# Room memory\n\nmanual notes about the project.",
                token_count=10,
                updated_at=datetime.utcnow() - timedelta(days=1),
            )
            result = MagicMock()
            result.scalars.return_value.first.return_value = stale_row
            db = AsyncMock()
            db.add = MagicMock()
            db.flush = AsyncMock()
            db.execute = AsyncMock(return_value=result)

            with (
                patch(
                    "app.core.board_snapshot.build_board_snapshot",
                    new=AsyncMock(return_value=board_snapshot),
                ),
                patch("app.core.events.emit_event", new_callable=AsyncMock),
            ):
                refreshed_now = await maybe_refresh_room_memory(
                    _ROOM_ID, db, trigger=trigger,
                )

            assert refreshed_now is True, f"refresh must succeed for trigger={trigger!r}"
            assert "Status (updated automatically)" in stale_row.content
            refreshed += 1

        assert refreshed >= 5, (
            f"Room memory must refresh >= 5 times over the month, got {refreshed} "
            f"— the room 'blog' fact-finding case had 1 update in 8 days"
        )
