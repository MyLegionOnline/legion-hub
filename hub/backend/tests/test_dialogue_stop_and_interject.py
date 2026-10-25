"""Stopping a direct dialogue, and talking to it while it works.

Incident (direct room, agent working through ``run_subtask`` steps): the user
typed "stop" and the agent kept calling tools — the message started a second
turn beside the running one — and there was no task card to cancel because the
task of a dialogue is only recorded at the end of the turn.

* a message that arrives mid-turn is queued for the loop, not run in parallel;
* the Stop button ends the turn but keeps (and gates) the work, session open;
* a ``run_subtask`` child of a dialogue stops with its parent.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import session_interject, user_sessions


@pytest.fixture(autouse=True)
def _clean():
    session_interject._PENDING.clear()
    session_interject._STOP_REQUESTED.clear()
    user_sessions._ACTIVE_TURNS.clear()
    yield
    session_interject._PENDING.clear()
    session_interject._STOP_REQUESTED.clear()
    user_sessions._ACTIVE_TURNS.clear()


def _session(**kwargs):
    base = {"id": uuid4(), "room_id": uuid4(), "kind": "user_agent_p2", "is_open": True}
    base.update(kwargs)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# queue / append helpers
# ---------------------------------------------------------------------------


def test_queue_drain_and_stop_flag():
    sid = uuid4()
    session_interject.queue(sid, "first")
    session_interject.queue(sid, "   ")  # blank is not a message
    session_interject.queue(sid, "second")
    assert session_interject.has_pending(sid)
    assert session_interject.drain(sid) == ["first", "second"]
    assert session_interject.drain(sid) == []

    session_interject.queue(sid, "dropped by stop")
    session_interject.request_stop(sid)
    assert session_interject.stop_requested(sid)
    assert session_interject.drain(sid) == []
    assert session_interject.consume_stop(sid) is True
    assert session_interject.consume_stop(sid) is False


def test_append_user_note_merges_into_tool_result_turn():
    tool_turn = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
    messages = [{"role": "user", "content": "go"}, {"role": "assistant", "content": []}, tool_turn]
    session_interject.append_user_note(messages, "stop")
    assert len(messages) == 3  # no second consecutive user turn
    assert messages[-1]["content"][-1] == {"type": "text", "text": "stop"}
    assert messages[-1]["content"][0]["type"] == "tool_result"


def test_append_user_note_after_plain_turn_adds_a_message():
    messages = [{"role": "assistant", "content": "x"}]
    session_interject.append_user_note(messages, "stop")
    assert messages[-1] == {"role": "user", "content": "stop"}


# ---------------------------------------------------------------------------
# process_user_turn: no parallel turn on a busy session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_message_during_a_running_turn_is_queued_not_run(monkeypatch):
    session = _session()
    inner = AsyncMock(return_value=True)
    monkeypatch.setattr(user_sessions, "_process_user_turn", inner)
    user_sessions._ACTIVE_TURNS.add(session.id)

    assert await user_sessions.process_user_turn(session, "stop", AsyncMock()) is True

    inner.assert_not_called()
    assert session_interject.drain(session.id) == ["stop"]
    # The running turn still owns the busy mark.
    assert session.id in user_sessions._ACTIVE_TURNS


@pytest.mark.asyncio
async def test_leftover_message_is_answered_as_a_follow_up_turn(monkeypatch):
    session = _session()
    seen: list[str] = []

    async def fake_turn(sess, text, db, **_kw):
        seen.append(text)
        if len(seen) == 1:  # arrives after the loop's last model request
            session_interject.queue(sess.id, "one more thing")
        return True

    monkeypatch.setattr(user_sessions, "_process_user_turn", fake_turn)

    assert await user_sessions.process_user_turn(session, "do it", AsyncMock()) is True

    assert seen == ["do it", "one more thing"]
    assert session.id not in user_sessions._ACTIVE_TURNS
    assert not session_interject.has_pending(session.id)


@pytest.mark.asyncio
async def test_busy_mark_and_queue_are_released_after_a_turn(monkeypatch):
    session = _session()
    monkeypatch.setattr(user_sessions, "_process_user_turn", AsyncMock(return_value=False))
    session_interject.request_stop(session.id)

    await user_sessions.process_user_turn(session, "hi", AsyncMock())

    assert session.id not in user_sessions._ACTIVE_TURNS
    assert not session_interject.stop_requested(session.id)


# ---------------------------------------------------------------------------
# stop_dialogue_turn
# ---------------------------------------------------------------------------


def _patch_open_session(monkeypatch, session):
    from app.repositories import session as session_repo

    async def find(self, _room_id, _user_id):
        return session

    monkeypatch.setattr(session_repo.SessionRepository, "find_open_user_session", find)


@pytest.mark.asyncio
async def test_stop_with_nothing_open_is_idle(monkeypatch):
    _patch_open_session(monkeypatch, None)
    out = await user_sessions.stop_dialogue_turn(uuid4(), uuid4(), AsyncMock())
    assert out == {"status": "idle"}


@pytest.mark.asyncio
async def test_stop_during_a_turn_requests_the_stop_and_keeps_the_session(monkeypatch):
    session = _session()
    _patch_open_session(monkeypatch, session)
    finalize = AsyncMock()
    monkeypatch.setattr(user_sessions, "_finalize_session_work", finalize)
    user_sessions._ACTIVE_TURNS.add(session.id)
    session_interject.queue(session.id, "stale")

    out = await user_sessions.stop_dialogue_turn(session.room_id, uuid4(), AsyncMock())

    assert out["status"] == "stopped"
    assert session_interject.stop_requested(session.id)
    assert not session_interject.has_pending(session.id)
    assert session.is_open  # the conversation continues
    finalize.assert_not_called()  # the unwinding turn does it, after its last write


@pytest.mark.asyncio
async def test_stop_on_an_idle_session_finalizes_its_unrecorded_work(monkeypatch):
    session = _session()
    _patch_open_session(monkeypatch, session)
    finalize = AsyncMock()
    monkeypatch.setattr(user_sessions, "_finalize_session_work", finalize)
    db = AsyncMock()

    out = await user_sessions.stop_dialogue_turn(session.room_id, uuid4(), db)

    assert out["status"] == "finalized"
    finalize.assert_awaited_once()
    assert finalize.await_args.kwargs["reason"] == user_sessions.USER_STOP_REASON
    assert "incomplete" not in finalize.await_args.kwargs  # a stop keeps the result, it is not "unfinished"


# ---------------------------------------------------------------------------
# the unwinding turn keeps the work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stopped_turn_saves_history_and_runs_the_gate(monkeypatch):
    session = _session(context_snapshot={}, last_activity_at=None)
    finalize = AsyncMock()
    saved: list[object] = []
    monkeypatch.setattr(user_sessions, "_finalize_session_work", finalize)
    monkeypatch.setattr(user_sessions, "_save_history", lambda s, h: saved.append(list(h)))
    session_interject.request_stop(session.id)
    result = SimpleNamespace(tool_notes=[{"role": "assistant", "content": "note"}])
    history = [{"role": "user", "content": "make a pdf"}]
    db = AsyncMock()
    db.add = lambda _o: None

    handled = await user_sessions._end_turn_on_user_stop(session, "make a pdf", result, history, db)

    assert handled is True
    assert history[-1]["content"] == user_sessions._USER_STOP_NOTE
    assert {"role": "assistant", "content": "note"} in history  # what was done is remembered
    assert saved
    finalize.assert_awaited_once()
    assert finalize.await_args.kwargs["user_request"] == "make a pdf"
    assert not session_interject.stop_requested(session.id)


@pytest.mark.asyncio
async def test_turn_closed_for_another_reason_is_not_treated_as_a_stop(monkeypatch):
    session = _session()
    finalize = AsyncMock()
    monkeypatch.setattr(user_sessions, "_finalize_session_work", finalize)

    handled = await user_sessions._end_turn_on_user_stop(
        session, "x", SimpleNamespace(tool_notes=[]), [], AsyncMock()
    )

    assert handled is False
    finalize.assert_not_called()
