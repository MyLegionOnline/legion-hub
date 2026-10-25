"""
Stage 4 of internal design notes — deterministic memory flush.

flush_session_memory (§4.1) is the single point every session-close path calls,
so a close that never gave the model a chance to write agent_memory_patch
(tool_limit, timeout, orphaned_restart, idle, context_limit, ...) still leaves a
non-empty AgentMemory record. maybe_refresh_room_memory (§4.2) is a cheaper,
debounced, LLM-free trigger for Room memory on task/goal terminal events.

No database: db is a scripted AsyncMock, following the style already used by
test_board_snapshot.py and test_watchdog.py for this codebase.
"""
from __future__ import annotations

import types
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.session_close import (
    MemoryPatch,
    SessionCloseActions,
    _flush_deterministic_summary,
    _memory_flush_forced,
    _render_room_status_digest,
    _upsert_room_memory_auto_section,
    flush_session_memory,
    maybe_refresh_room_memory,
)

_ROOM_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()
_TASK_ID = uuid.uuid4()


def _session(kind: str, **overrides) -> object:
    common = dict(
        id=uuid.uuid4(),
        room_id=_ROOM_ID,
        kind=kind,
        agent_id=_AGENT_ID,
        chat_master_agent_id=None,
        task_id=None,
        context_snapshot=None,
        summary=None,
        side="agent",
    )
    common.update(overrides)
    return types.SimpleNamespace(**common)


def _result(first_row=None, all_rows=None) -> MagicMock:
    m = MagicMock()
    m.scalars.return_value.first.return_value = first_row
    m.scalars.return_value.all.return_value = all_rows or []
    return m


def _db(*results) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(side_effect=list(results))
    db.get = AsyncMock(return_value=None)
    return db


# ---------------------------------------------------------------------------
# flush_session_memory — no-op branches (must not touch the database at all)
# ---------------------------------------------------------------------------


class TestFlushNoOpBranches:
    @pytest.mark.asyncio
    async def test_kill_switch_disables_flush(self):
        session = _session("user_cm_p2")
        db = _db()
        with patch.dict("os.environ", {"MEMORY_FLUSH_FORCED": "false"}):
            result = await flush_session_memory(session, db, reason="idle")
        assert result is False
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_agent_memory_patch_already_applied(self):
        session = _session("user_cm_p2")
        db = _db()
        actions = SessionCloseActions(agent_memory_patch=MemoryPatch(content="already written"))
        result = await flush_session_memory(
            session, db, reason="task_done", applied_actions=actions
        )
        assert result is False
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_for_kind_without_agent_memory(self):
        """cm_self_plan's close-actions matrix has no agent_memory_patch at all."""
        session = _session("cm_self_plan", chat_master_agent_id=_AGENT_ID, agent_id=None)
        db = _db()
        result = await flush_session_memory(session, db, reason="token_limit")
        assert result is False
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_session_has_no_agent_identity(self):
        session = _session("user_cm_p2", agent_id=None, chat_master_agent_id=None)
        db = _db()
        result = await flush_session_memory(session, db, reason="idle")
        assert result is False
        db.execute.assert_not_awaited()


# ---------------------------------------------------------------------------
# flush_session_memory — tiers: reused summary / deterministic fallback
# ---------------------------------------------------------------------------


class TestFlushTiers:
    @pytest.mark.asyncio
    async def test_reuses_applied_actions_summary_without_llm_call(self):
        session = _session("user_agent_p2")
        actions = SessionCloseActions(summary="Fixed the login bug and wrote a test.")
        db = _db(
            _result(first_row=None),  # Agent lookup — irrelevant for this tier
            _result(first_row=None),  # AgentMemory lookup — create new row
        )
        with (
            patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
            patch(
                "app.core.session_close._flush_cheap_summary", new_callable=AsyncMock
            ) as mock_cheap,
            patch(
                "app.core.metrics.record_session_close_without_memory"
            ) as mock_metric,
        ):
            result = await flush_session_memory(
                session, db, reason="timeout", applied_actions=actions
            )

        assert result is True
        mock_cheap.assert_not_awaited()  # a usable summary was already at hand
        added = [c.args[0] for c in db.add.call_args_list]
        assert added, "AgentMemory row must be added"
        memory_row = added[-1]
        assert "Fixed the login bug and wrote a test." in memory_row.content
        assert "timeout" in memory_row.content
        mock_metric.assert_called_once()
        assert mock_metric.call_args.kwargs["tier"] == "reused_summary"
        mock_emit.assert_awaited_once()
        assert mock_emit.call_args.kwargs["event_type"] == "MEMORY_FLUSHED"

    @pytest.mark.asyncio
    async def test_reuses_session_summary_when_no_applied_actions(self):
        session = _session("user_cm_p1", summary="Previously saved context.")
        db = _db(_result(first_row=None), _result(first_row=None))
        with (
            patch("app.core.events.emit_event", new_callable=AsyncMock),
            patch(
                "app.core.session_close._flush_cheap_summary", new_callable=AsyncMock
            ) as mock_cheap,
        ):
            result = await flush_session_memory(session, db, reason="idle")
        assert result is True
        mock_cheap.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_to_deterministic_when_nothing_else_available(self):
        session = _session("cm_agent_proxy", side="agent", task_id=_TASK_ID)
        task_row = types.SimpleNamespace(title="Ship the widget")
        db = _db(
            _result(first_row=None),  # Agent lookup — None disables the cheap tier
            _result(first_row=None),  # AgentMemory lookup — create new row
        )
        db.get = AsyncMock(return_value=task_row)
        with patch("app.core.events.emit_event", new_callable=AsyncMock):
            result = await flush_session_memory(session, db, reason="tool_limit")

        assert result is True
        memory_row = db.add.call_args_list[-1].args[0]
        assert "tool_limit" in memory_row.content
        assert "Ship the widget" in memory_row.content

    @pytest.mark.asyncio
    async def test_appended_content_is_bounded(self):
        """A very long pre-existing memory must not grow the row without limit."""
        session = _session("user_agent_p2", summary="short new note")
        huge_existing = types.SimpleNamespace(
            content="x" * 9000, room_id=_ROOM_ID, agent_id=_AGENT_ID,
        )
        db = _db(_result(first_row=None), _result(first_row=huge_existing))
        with patch("app.core.events.emit_event", new_callable=AsyncMock):
            await flush_session_memory(session, db, reason="idle")
        assert len(huge_existing.content) <= 8000


# ---------------------------------------------------------------------------
# _flush_deterministic_summary — pure function
# ---------------------------------------------------------------------------


class TestDeterministicSummary:
    def test_includes_task_title_and_files(self):
        session = _session("cm_agent_proxy")
        text = _flush_deterministic_summary(
            session,
            reason="tool_limit",
            task_title="Refactor auth module",
            changed_paths=["app/auth.py", "tests/test_auth.py"],
        )
        assert "tool_limit" in text
        assert "Refactor auth module" in text
        assert "app/auth.py" in text

    def test_never_empty_with_nothing_to_report(self):
        session = _session("user_cm_p2")
        text = _flush_deterministic_summary(
            session, reason="orphaned_restart", task_title=None, changed_paths=[]
        )
        assert text.strip()
        assert "orphaned_restart" in text


# ---------------------------------------------------------------------------
# Room memory digest rendering — pure functions
# ---------------------------------------------------------------------------


class TestRoomStatusDigest:
    def test_renders_goal_status_and_blocked_reasons(self):
        snapshot = {
            "counters": {"total": 5, "completed": 2, "blocked": 1},
            "goals": [
                {
                    "title": "SEO",
                    "status": "active",
                    "total": 3,
                    "collapsed": {"completed": 1},
                    "columns": {
                        "blocked": [
                            {"title": "Fix sitemap", "blocked_reason": "Not started — Retry"}
                        ]
                    },
                }
            ],
            "without_goal": {"total": 2},
        }
        digest = _render_room_status_digest(snapshot)
        assert "SEO" in digest
        assert "1/3" in digest
        assert "Fix sitemap" in digest
        assert "No goal" in digest

    def test_upsert_replaces_only_the_auto_section(self):
        original = "# Project Summary\n\nHuman-written decisions go here.\n"
        with_digest = _upsert_room_memory_auto_section(original, "Total tasks: 3.")
        assert "Human-written decisions go here." in with_digest
        assert "Total tasks: 3." in with_digest

        refreshed = _upsert_room_memory_auto_section(with_digest, "Total tasks: 4.")
        assert "Human-written decisions go here." in refreshed
        assert "Total tasks: 4." in refreshed
        assert "Total tasks: 3." not in refreshed


# ---------------------------------------------------------------------------
# maybe_refresh_room_memory
# ---------------------------------------------------------------------------


class TestMaybeRefreshRoomMemory:
    @pytest.mark.asyncio
    async def test_kill_switch_disables_refresh(self):
        db = _db()
        with patch.dict("os.environ", {"MEMORY_FLUSH_FORCED": "false"}):
            result = await maybe_refresh_room_memory(_ROOM_ID, db, trigger="task_blocked")
        assert result is False
        db.execute.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_debounced_when_recently_updated(self):
        recent_row = types.SimpleNamespace(
            room_id=_ROOM_ID, content="# Project Summary\n", updated_at=datetime.utcnow()
        )
        db = _db(_result(first_row=recent_row))
        result = await maybe_refresh_room_memory(_ROOM_ID, db, trigger="task_blocked")
        assert result is False
        db.execute.assert_awaited_once()  # only the cheap pre-check, no lock taken

    @pytest.mark.asyncio
    async def test_refreshes_when_stale(self):
        stale_row = types.SimpleNamespace(
            room_id=_ROOM_ID,
            content="# Project Summary\n\nHand-written notes.\n",
            updated_at=datetime.utcnow() - timedelta(hours=1),
        )
        db = _db(
            _result(first_row=stale_row),  # pre-check
            MagicMock(),                   # resource_lock's SELECT pg_advisory_xact_lock
            _result(first_row=stale_row),  # re-check under lock
        )
        fake_snapshot = {
            "counters": {"total": 1, "completed": 0, "blocked": 1},
            "goals": [],
            "without_goal": {"total": 0},
        }
        with (
            patch(
                "app.core.board_snapshot.build_board_snapshot",
                new_callable=AsyncMock, return_value=fake_snapshot,
            ),
            patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
        ):
            result = await maybe_refresh_room_memory(_ROOM_ID, db, trigger="task_blocked")

        assert result is True
        assert "Hand-written notes." in stale_row.content
        assert "Total tasks: 1" in stale_row.content
        mock_emit.assert_awaited_once()
        assert mock_emit.call_args.kwargs["event_type"] == "MEMORY_FLUSHED"


def test_memory_flush_forced_reads_env_truthy_values():
    with patch.dict("os.environ", {"MEMORY_FLUSH_FORCED": "0"}):
        assert _memory_flush_forced() is False
    with patch.dict("os.environ", {"MEMORY_FLUSH_FORCED": "true"}):
        assert _memory_flush_forced() is True
    with patch.dict("os.environ", {}, clear=False):
        # Default (no env var set) is "true" — see os.environ.pop below.
        import os as _os

        _os.environ.pop("MEMORY_FLUSH_FORCED", None)
        assert _memory_flush_forced() is True
