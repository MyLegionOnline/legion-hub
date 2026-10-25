"""Tests for Stage 4: ask_user tool and registry.

Coverage:
  - registry register / resolve / expire / is_session_waiting
  - tool: answer resumes the loop
  - tool: timeout resumes the loop with fallback message
  - watchdog: does not kill a session that is waiting for user input
  - watchdog: kills a session that has no pending question after kill threshold
  - two parallel questions in different sessions do not interfere
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# ask_user_registry tests
# ---------------------------------------------------------------------------

from app.core import ask_user_registry as reg


def _make_ids():
    return str(uuid.uuid4()), uuid.uuid4(), uuid.uuid4()


def test_register_creates_pending_question():
    qid, sid, rid = _make_ids()
    pq = reg.register(qid, sid, rid, {"questions": []})
    assert pq.question_id == qid
    assert pq.session_id == sid
    assert not pq.event.is_set()
    assert reg.is_session_waiting(sid)
    # cleanup
    reg.expire(qid)


def test_resolve_sets_answer_and_fires_event():
    qid, sid, rid = _make_ids()
    pq = reg.register(qid, sid, rid, {})
    result = reg.resolve(qid, {"selected_options": ["Option A"], "free_text": None})
    assert result is True
    assert pq.event.is_set()
    assert pq.answer == {"selected_options": ["Option A"], "free_text": None}
    assert not reg.is_session_waiting(sid)


def test_resolve_missing_question_returns_false():
    assert reg.resolve("does-not-exist", {}) is False


def test_expire_cleans_up():
    qid, sid, rid = _make_ids()
    reg.register(qid, sid, rid, {})
    assert reg.is_session_waiting(sid)
    reg.expire(qid)
    assert not reg.is_session_waiting(sid)


def test_two_sessions_independent():
    qid1, sid1, rid1 = _make_ids()
    qid2, sid2, rid2 = _make_ids()
    reg.register(qid1, sid1, rid1, {})
    reg.register(qid2, sid2, rid2, {})
    # Resolve only the first
    reg.resolve(qid1, {"selected_options": []})
    assert not reg.is_session_waiting(sid1)
    assert reg.is_session_waiting(sid2)
    # cleanup
    reg.expire(qid2)


# ---------------------------------------------------------------------------
# Tool execute tests (async)
# ---------------------------------------------------------------------------


class _FakeDB:
    pass


def _make_ctx(sid=None, rid=None):
    from app.core.tools.registry import ToolContext
    return ToolContext(
        room_id=rid or uuid.uuid4(),
        session_id=sid or uuid.uuid4(),
        task_id=None,
        db=_FakeDB(),
    )


@pytest.mark.asyncio
async def test_ask_user_answer_resumes_loop():
    """User answers promptly — tool returns the selected option JSON."""
    from app.core.tools._ask_user import AskUserTool

    tool = AskUserTool()
    ctx = _make_ctx()

    questions = [
        {
            "question": "Which approach?",
            "header": "Approach",
            "options": [
                {"label": "A", "description": "Option A"},
                {"label": "B", "description": "Option B"},
            ],
        }
    ]

    async def _emit_event(**kwargs):
        pass

    with patch("app.core.events.emit_event", side_effect=_emit_event):
        # Run the tool; answer it almost immediately from another coroutine
        async def _answer_soon():
            await asyncio.sleep(0.05)
            qid = list(reg._PENDING.keys())[-1]
            reg.resolve(qid, {"selected_options": ["A"], "free_text": None})

        tool_task = asyncio.create_task(tool.execute(ctx, {"questions": questions}))
        asyncio.create_task(_answer_soon())
        result = await tool_task

    import json
    data = json.loads(result)
    assert data["selected_options"] == ["A"]
    assert data["free_text"] is None
    assert not reg.is_session_waiting(ctx.session_id)


@pytest.mark.asyncio
async def test_ask_user_timeout_returns_fallback():
    """Timeout fires before user answers — tool returns the 'user is away' message."""
    import importlib, sys
    import os

    # Temporarily override timeout to 0.1 s
    with patch.dict(os.environ, {"ASK_USER_TIMEOUT_SECONDS": "0"}):
        # Force reimport to pick up env var (module already imported, so patch the constant)
        from app.core.tools import _ask_user as _au_mod
        original_timeout = _au_mod._TIMEOUT
        _au_mod._TIMEOUT = 0  # 0 seconds → immediate timeout

        tool = _au_mod.AskUserTool()
        ctx = _make_ctx()

        async def _emit_event(**kwargs):
            pass

        with patch("app.core.events.emit_event", side_effect=_emit_event):
            result = await tool.execute(ctx, {"questions": [
                {"question": "?", "header": "Q", "options": [
                    {"label": "X", "description": "x"},
                    {"label": "Y", "description": "y"},
                ]}
            ]})

        _au_mod._TIMEOUT = original_timeout  # restore

    assert "best judgement" in result.lower() or "did not respond" in result.lower()
    assert not reg.is_session_waiting(ctx.session_id)


# ---------------------------------------------------------------------------
# Watchdog integration: skip waiting sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_watchdog_skips_waiting_session():
    """A session waiting for user input must NOT be killed by the watchdog."""
    from app.core.watchdog import run_watchdog_cycle
    from app.models.session.cm_agent_proxy import CmAgentProxySession

    now = datetime.now(timezone.utc)
    very_old = now - timedelta(hours=2)

    sid = uuid.uuid4()
    rid = uuid.uuid4()

    # Register in ask_user_registry to mark session as waiting
    qid = str(uuid.uuid4())
    reg.register(qid, sid, rid, {})

    try:
        # Build a fake stale session
        fake_session = MagicMock(spec=CmAgentProxySession)
        fake_session.id = sid
        fake_session.room_id = rid
        fake_session.closed_at = None
        fake_session.last_activity_at = very_old
        fake_session.started_at = very_old
        fake_session.task_id = uuid.uuid4()
        fake_session.agent_id = uuid.uuid4()
        fake_session.peer_session_id = None
        fake_session.side = "agent"
        fake_session.kind = "cm_agent_proxy"

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[fake_session])))))

        killed = []

        async def fake_kill(session, now_, handled, db):
            killed.append(session.id)

        with patch("app.core.watchdog._kill_stalled_pair", side_effect=fake_kill):
            await run_watchdog_cycle(mock_db, now=now)

        assert sid not in killed, "Watchdog must not kill a session that is waiting for user input"
    finally:
        reg.expire(qid)


@pytest.mark.asyncio
async def test_watchdog_kills_truly_stale_session():
    """A session not in the waiting registry IS killed after kill threshold."""
    from app.core.watchdog import run_watchdog_cycle
    from app.models.session.cm_agent_proxy import CmAgentProxySession

    now = datetime.now(timezone.utc)
    very_old = now - timedelta(hours=2)

    sid = uuid.uuid4()
    rid = uuid.uuid4()
    # NOT registered in ask_user_registry

    fake_session = MagicMock(spec=CmAgentProxySession)
    fake_session.id = sid
    fake_session.room_id = rid
    fake_session.closed_at = None
    fake_session.last_activity_at = very_old
    fake_session.started_at = very_old
    fake_session.task_id = uuid.uuid4()
    fake_session.agent_id = uuid.uuid4()
    fake_session.peer_session_id = None
    fake_session.side = "agent"
    fake_session.kind = "cm_agent_proxy"

    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[fake_session])))))

    killed = []

    async def fake_kill(session, now_, handled, db):
        killed.append(session.id)

    with patch("app.core.watchdog._kill_stalled_pair", side_effect=fake_kill):
        await run_watchdog_cycle(mock_db, now=now)

    assert sid in killed
