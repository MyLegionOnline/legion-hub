"""Stage 4 of internal design notes — resilience.

Every normal turn already runs Gate 1 on its way out (Stage 3). This stage
covers the paths that end a dialogue *without* a normal turn ever completing:

* the session closes itself voluntarily but through rotation, not
  ``close_session`` (``close_user_session_gracefully``);
* the watchdog closes it (`idle`, `orphaned_restart`) — no model turn runs at
  all;
* neither of the above could reach workspaced (a restart, a room whose
  container is not up yet) — the next direct session in the room sweeps it up.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import session_work
from app.core.watchdog import _finalize_dialogue_work, close_idle_user_sessions

NOW = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# watchdog._finalize_dialogue_work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_dialogue_work_delegates_to_session_work(monkeypatch):
    session = MagicMock(id=uuid.uuid4())
    db = MagicMock(name="db")
    finalize = AsyncMock(return_value={"merged": True})
    monkeypatch.setattr("app.core.session_work.finalize_session_work", finalize)

    await _finalize_dialogue_work(session, db, reason="idle")

    finalize.assert_awaited_once_with(session, db, reason="idle")


@pytest.mark.asyncio
async def test_finalize_dialogue_work_swallows_failures(monkeypatch):
    """workspaced unreachable must not stop the watchdog from closing the row —
    sweep_orphan_session_work is the recovery path for this case."""
    session = MagicMock(id=uuid.uuid4())
    monkeypatch.setattr(
        "app.core.session_work.finalize_session_work",
        AsyncMock(side_effect=RuntimeError("workspaced unreachable")),
    )

    await _finalize_dialogue_work(session, db=object(), reason="orphaned_restart")  # must not raise


# ---------------------------------------------------------------------------
# watchdog._close_user_sessions (via close_idle_user_sessions)
# ---------------------------------------------------------------------------


def _idle_dialogue_session(*, work_touched: bool) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "user_agent_p2"
    s.closed_at = None
    s.close_reason = None
    s.context_snapshot = (
        {session_work.SNAPSHOT_WORK_TOUCHED: True} if work_touched else {}
    )
    return s


@pytest.mark.asyncio
async def test_idle_close_finalizes_unfinished_dialogue_work(monkeypatch):
    idle = _idle_dialogue_session(work_touched=True)
    result = MagicMock()
    result.scalars.return_value.all.return_value = [idle]
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(return_value=result)

    finalize = AsyncMock(return_value={"merged": True})
    with (
        patch("app.core.watchdog.flush_session_memory", new_callable=AsyncMock),
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock),
        patch("app.core.session_work.finalize_session_work", finalize),
    ):
        closed = await close_idle_user_sessions(db, now=NOW)

    assert closed == 1
    finalize.assert_awaited_once()
    assert finalize.await_args.args[0] is idle
    assert finalize.await_args.kwargs["reason"] == "idle"


@pytest.mark.asyncio
async def test_idle_close_is_a_noop_for_a_reading_only_session(monkeypatch):
    """No writing tool ever ran, so there is nothing for Gate 1 to look at —
    finalize_session_work itself no-ops, but it must still be safe to call."""
    idle = _idle_dialogue_session(work_touched=False)
    result = MagicMock()
    result.scalars.return_value.all.return_value = [idle]
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(return_value=result)

    with (
        patch("app.core.watchdog.flush_session_memory", new_callable=AsyncMock),
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock),
    ):
        closed = await close_idle_user_sessions(db, now=NOW)

    assert closed == 1
    assert idle.close_reason == "idle"


# ---------------------------------------------------------------------------
# session_work.sweep_orphan_session_work
# ---------------------------------------------------------------------------


def _closed_session(*, kind="user_agent_p2", work_touched, closed_at=None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        room_id=uuid.uuid4(),
        kind=kind,
        closed_at=closed_at or NOW,
        context_snapshot=(
            {session_work.SNAPSHOT_WORK_TOUCHED: True} if work_touched else {"messages": []}
        ),
    )


class _RowsDB:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _stmt):
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: self._rows))


@pytest.mark.asyncio
async def test_sweep_finalizes_only_sessions_with_unfinished_work(monkeypatch):
    touched = _closed_session(work_touched=True)
    clean = _closed_session(work_touched=False)
    db = _RowsDB([touched, clean])

    finalize = AsyncMock(return_value={"merged": True})
    monkeypatch.setattr(session_work, "finalize_session_work", finalize)

    swept = await session_work.sweep_orphan_session_work(uuid.uuid4(), db)

    assert swept == 1
    finalize.assert_awaited_once_with(touched, db, reason="sweep")


@pytest.mark.asyncio
async def test_sweep_counts_only_sessions_that_actually_produced_something(monkeypatch):
    """A branch that turned out to carry no commits (finalize returns None) —
    picked up, but not counted as recovered work."""
    touched = _closed_session(work_touched=True)
    db = _RowsDB([touched])
    monkeypatch.setattr(session_work, "finalize_session_work", AsyncMock(return_value=None))

    swept = await session_work.sweep_orphan_session_work(uuid.uuid4(), db)

    assert swept == 0


@pytest.mark.asyncio
async def test_sweep_is_resilient_to_a_single_failure(monkeypatch):
    """One session's finalize blowing up (workspaced still down) must not stop
    the sweep from recovering the rest."""
    broken = _closed_session(work_touched=True)
    fine = _closed_session(work_touched=True)
    db = _RowsDB([broken, fine])

    async def _finalize(session, _db, *, reason):
        if session is broken:
            raise RuntimeError("workspaced unreachable")
        return {"merged": True}

    monkeypatch.setattr(session_work, "finalize_session_work", _finalize)

    swept = await session_work.sweep_orphan_session_work(uuid.uuid4(), db)

    assert swept == 1


@pytest.mark.asyncio
async def test_sweep_ignores_sessions_without_a_touched_flag(monkeypatch):
    db = _RowsDB([_closed_session(work_touched=False), _closed_session(work_touched=False)])
    finalize = AsyncMock()
    monkeypatch.setattr(session_work, "finalize_session_work", finalize)

    swept = await session_work.sweep_orphan_session_work(uuid.uuid4(), db)

    assert swept == 0
    finalize.assert_not_awaited()


# ---------------------------------------------------------------------------
# user_sessions.start_user_session — lazy sweep on a new direct session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_user_session_sweeps_the_room_for_direct_routes(monkeypatch):
    from app.core import user_sessions
    from app.core.router import RouteDecision

    room = SimpleNamespace(id=uuid.uuid4())
    user = SimpleNamespace(id=uuid.uuid4())
    agent_id = uuid.uuid4()
    route_decision = RouteDecision(
        route="single_agent_p2", mentioned_agent_ids=[agent_id],
        matched_task_id=None, matched_goal_id=None,
        confidence=1.0, reasoning="test",
    )

    created = SimpleNamespace(id=uuid.uuid4(), kind="user_agent_p2")
    repo = MagicMock()
    repo.create = AsyncMock(return_value=created)
    db = MagicMock(name="db")

    sweep = AsyncMock(return_value=0)
    with (
        patch("app.core.user_sessions.SessionRepository", return_value=repo),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch("app.core.session_work.sweep_orphan_session_work", sweep),
    ):
        result = await user_sessions.start_user_session(room, user, route_decision, db)

    assert result is created
    sweep.assert_awaited_once_with(room.id, db)


@pytest.mark.asyncio
async def test_start_user_session_sweep_failure_does_not_block_session_creation(monkeypatch):
    from app.core import user_sessions
    from app.core.router import RouteDecision

    room = SimpleNamespace(id=uuid.uuid4())
    user = SimpleNamespace(id=uuid.uuid4())
    agent_id = uuid.uuid4()
    route_decision = RouteDecision(
        route="single_agent_p1", mentioned_agent_ids=[agent_id],
        matched_task_id=uuid.uuid4(), matched_goal_id=uuid.uuid4(),
        confidence=1.0, reasoning="test",
    )

    created = SimpleNamespace(id=uuid.uuid4(), kind="user_agent_p1")
    repo = MagicMock()
    repo.create = AsyncMock(return_value=created)

    with (
        patch("app.core.user_sessions.SessionRepository", return_value=repo),
        patch("app.core.user_sessions.emit_event", new_callable=AsyncMock),
        patch(
            "app.core.session_work.sweep_orphan_session_work",
            AsyncMock(side_effect=RuntimeError("db down")),
        ),
    ):
        result = await user_sessions.start_user_session(room, user, route_decision, db=object())

    assert result is created


# ---------------------------------------------------------------------------
# user_sessions.close_user_session_gracefully — rotation also finalizes work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rotation_close_finalizes_dialogue_work(monkeypatch):
    from app.core import user_sessions
    from app.core.session_close import SessionCloseActions

    session = MagicMock()
    session.id = uuid.uuid4()
    session.room_id = uuid.uuid4()
    session.kind = "user_agent_p2"
    session.summary = None

    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    finalize = AsyncMock()
    with (
        patch("app.core.user_sessions._get_agent_llm_config", AsyncMock(return_value={})),
        patch("app.core.user_sessions.build_memory_layer", AsyncMock(return_value={})),
        patch("app.core.user_sessions.build_system_prompt", MagicMock(return_value="sys")),
        patch("app.core.user_sessions._get_history", MagicMock(return_value=[])),
        patch(
            "app.core.user_sessions._final_memory_save",
            AsyncMock(return_value=SessionCloseActions(summary="dialogue summary")),
        ),
        patch("app.core.user_sessions.flush_session_memory", AsyncMock()),
        patch("app.core.user_sessions.emit_event", AsyncMock()),
        patch("app.core.user_sessions._finalize_session_work", finalize),
    ):
        summary = await user_sessions.close_user_session_gracefully(
            session, db, rotation_reason="turn_limit"
        )

    assert summary == "dialogue summary"
    finalize.assert_awaited_once()
    args, kwargs = finalize.await_args
    assert args[0] is session
    assert kwargs["reason"] == "session_closed"
    assert kwargs["agent_reply"] == "dialogue summary"
