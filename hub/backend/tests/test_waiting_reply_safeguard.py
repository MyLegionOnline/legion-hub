"""waiting_reply_safeguard — a user's reply to a task that is waiting for them.

Covers the incident where "finish", sent to an agent whose dialogue session
had rotated out, was cold-routed to a fresh ChatMaster that just closed itself:

  * S1  classify_reply / mark / find / handle
  * S3  is_continuable waives a size limit for a turn that answers a question
  * S4  a forced-text turn parks the task BLOCKED, not WAITING_REVIEW

Provider and database access is stubbed; no network, no PostgreSQL.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import session_work, waiting_reply
from app.core.user_session_limits import (
    ROTATE_MAX_AGE,
    ROTATE_MAX_TURNS,
    is_continuable,
)
from app.core.user_sessions import _looks_like_open_question
from app.models.task import DispatchStatus, TaskState

NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)


# ── S1: classify_reply ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("finish", "approve"),
        ("Finish.", "approve"),
        ("ok", "approve"),
        ("yes, I confirm", "approve"),
        ("go", "approve"),
        ("all great", "approve"),
        ("lgtm", "approve"),
        ("finish it up", "continue"),
        ("keep going", "continue"),
        ("finish what you started", "continue"),
        ("bring it to the end", "continue"),
        ("and now make a dark theme", None),
        ("why doesn't the button work?", None),
        ("", None),
        ("x" * 250, None),
    ],
)
def test_classify_reply(text, expected):
    assert waiting_reply.classify_reply(text) == expected


# ── S1/S3: open-question detection ──────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Done. Would you like me to finish these steps?", True),
        ("Did X. The script hookup is still left to finish.", True),
        ("As soon as the tools are available again, I will continue.", True),
        ("All done, the task is complete.", False),
        ("I fixed the padding and the background color.", False),
        ("", False),
    ],
)
def test_looks_like_open_question(text, expected):
    assert _looks_like_open_question(text) is expected


# ── S3: is_continuable grace for an open question ───────────────────────────


def test_open_question_waives_turn_limit_but_not_age(monkeypatch):
    monkeypatch.setenv("USER_SESSION_MAX_TURNS", "80")
    over_turns = dict(message_count=200, started_at=NOW, now=NOW)

    assert is_continuable(**over_turns) is False
    assert is_continuable(**over_turns, has_open_question=True) is True

    # Age is a hard stop even with a pending question — the answer is carried
    # forward by waiting_reply instead.
    old = dict(message_count=4, started_at=NOW - timedelta(hours=48), now=NOW)
    monkeypatch.setenv("USER_SESSION_MAX_AGE_HOURS", "24")
    assert is_continuable(**old, has_open_question=True) is False


# ── S1: the task-side flag ─────────────────────────────────────────────────


class _FakeDB:
    def __init__(self):
        self.added: list[object] = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass


def _task(**kw):
    base = dict(
        id=uuid4(),
        room_id=uuid4(),
        title="HERO caption",
        state=TaskState.WAITING_REVIEW,
        origin="direct_session",
        origin_session_id=uuid4(),
        assigned_agent_id=uuid4(),
        awaiting_user_since=None,
        awaiting_user_kind=None,
        awaiting_user_prompt=None,
        branch_name="task/x",
        dispatch_status=DispatchStatus.READY,
        goal_id=None,
        merge_commit=None,
        updated_at=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_mark_and_clear_awaiting():
    db = _FakeDB()
    task = _task()

    await waiting_reply.mark_task_awaiting(task, "review", db, prompt="finish up?")
    assert task.awaiting_user_since is not None
    assert task.awaiting_user_kind == "review"
    assert task.awaiting_user_prompt == "finish up?"

    await waiting_reply.clear_task_awaiting(task, db)
    assert task.awaiting_user_since is None
    assert task.awaiting_user_kind is None


@pytest.mark.asyncio
async def test_mark_awaiting_noop_when_task_not_waiting():
    db = _FakeDB()
    task = _task(state=TaskState.COMPLETED)
    await waiting_reply.mark_task_awaiting(task, "review", db)
    assert task.awaiting_user_since is None


# ── S1: find_waiting_task prefers the user's own session ────────────────────


@pytest.mark.asyncio
async def test_find_waiting_task_prefers_session():
    sid = uuid4()
    mine = _task(awaiting_user_since=NOW, origin_session_id=sid)
    other = _task(awaiting_user_since=NOW + timedelta(minutes=5))

    class _DB:
        async def execute(self, _stmt):
            return SimpleNamespace(
                scalars=lambda: SimpleNamespace(all=lambda: [other, mine])
            )

    picked = await waiting_reply.find_waiting_task(
        mine.room_id, _DB(), prefer_session_id=sid
    )
    assert picked is mine

    picked_default = await waiting_reply.find_waiting_task(mine.room_id, _DB())
    assert picked_default is other  # freshest wins with no preference


# ── S1: handle_waiting_reply — approve merges, continue re-opens ────────────


@pytest.mark.asyncio
async def test_handle_waiting_reply_approve(monkeypatch):
    db = _FakeDB()
    task = _task(awaiting_user_since=NOW, goal_id=None)
    room = SimpleNamespace(id=task.room_id)
    user = SimpleNamespace(id=uuid4())

    merge = AsyncMock(return_value={"ok": True, "sha": "abc123", "target": "dev"})
    monkeypatch.setattr("app.core.room_git_flow.merge_task_branch", merge)
    monkeypatch.setattr("app.core.room_git_flow.maybe_open_promotion", AsyncMock())
    monkeypatch.setattr("app.core.events.emit_event", AsyncMock())

    handled = await waiting_reply.handle_waiting_reply(
        room, user, "finish", db, task=task
    )

    assert handled is True
    assert task.state == TaskState.COMPLETED
    assert task.merge_commit == "abc123"
    assert task.awaiting_user_since is None
    merge.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_waiting_reply_continue_direct_session(monkeypatch):
    db = _FakeDB()
    task = _task(awaiting_user_since=NOW, state=TaskState.BLOCKED, origin="direct_session")
    room = SimpleNamespace(id=task.room_id)
    user = SimpleNamespace(id=uuid4())

    started: dict = {}

    async def _fake_start(_room, _user, decision, _db):
        started["route"] = decision.route
        started["agent"] = decision.mentioned_agent_ids[0]
        return SimpleNamespace(id=uuid4(), context_snapshot={})

    turns: dict = {}

    async def _fake_turn(session, content, _db):
        turns["content"] = content
        turns["snapshot"] = dict(session.context_snapshot)

    monkeypatch.setattr("app.core.user_sessions.start_user_session", _fake_start)
    monkeypatch.setattr("app.core.user_sessions.process_user_turn", _fake_turn)
    monkeypatch.setattr("app.core.events.emit_event", AsyncMock())
    monkeypatch.setattr(waiting_reply, "_nick", AsyncMock(return_value="Frontender"))

    handled = await waiting_reply.handle_waiting_reply(
        room, user, "finish it", db, task=task
    )

    assert handled is True
    assert task.state == TaskState.IN_PROGRESS
    assert task.awaiting_user_since is None
    assert started["route"] == "single_agent_p2"
    assert started["agent"] == task.assigned_agent_id
    assert turns["snapshot"].get(session_work.SNAPSHOT_TASK_ID) == str(task.id)


# ── S4: a forced-text turn parks the task BLOCKED, not WAITING_REVIEW ───────


@pytest.mark.asyncio
async def test_incomplete_turn_parks_blocked(monkeypatch):
    from app.core import room_git_flow

    task = _task(state=TaskState.IN_PROGRESS, origin="direct_session")
    session = SimpleNamespace(
        id=uuid4(), room_id=task.room_id, kind="user_agent_p2",
        context_snapshot={session_work.SNAPSHOT_WORK_TOUCHED: True},
        goal_id=None, agent_id=task.assigned_agent_id,
    )

    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(id=task.room_id, git_config={})),
    )
    monkeypatch.setattr(
        room_git_flow, "get_flow_config",
        lambda _room: SimpleNamespace(
            direct_session_mode="gate1", direct_session_auto_merge="risk",
        ),
    )
    monkeypatch.setattr(
        session_work, "adopt_session_branch_as_task", AsyncMock(return_value=task)
    )
    monkeypatch.setattr(session_work, "_remember", AsyncMock())
    gate = AsyncMock()
    monkeypatch.setattr(room_git_flow, "finalize_task_gate1", gate)
    monkeypatch.setattr("app.core.events.emit_event", AsyncMock())

    result = await session_work.finalize_session_work(
        session, _FakeDB(), reason="turn_end",
        agent_reply="Partly done. Some is left to finish. Continue?",
        incomplete=True,
    )

    assert result["incomplete"] is True
    assert task.state == TaskState.BLOCKED
    assert task.awaiting_user_kind == "continuation_offer"
    gate.assert_not_awaited()  # the review gate never ran
