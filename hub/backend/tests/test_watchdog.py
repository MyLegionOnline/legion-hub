"""Unit tests for chat upgrade stage 4: watchdog for stalled sessions and tasks.

Covers:
- run_watchdog_cycle: stalled-but-alive session → ephemeral session_stalled status,
  session stays open
- run_watchdog_cycle: session past the kill threshold → both sides closed with
  reason='timeout', task → BLOCKED, persistent TASK_STALLED emitted
- run_watchdog_cycle: nothing stalled → no statuses, no events
- cleanup_orphaned_sessions: open proxy sessions closed with reason='orphaned_restart',
  IN_PROGRESS task → BLOCKED + TASK_STALLED
- block_stalled_legacy_tasks (internal design notes, part 3, stage 3):
  IN_PROGRESS tasks left behind by the legacy approve_and_start pipeline
  (no CmAgentProxySession, no run_id) get recovered the same way
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.events import AgentStatus
from app.core.watchdog import (
    block_stalled_legacy_tasks,
    cleanup_orphaned_sessions,
    close_stale_plan_sessions,
    run_watchdog_cycle,
)
from app.models.task import TaskState

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(all_rows=None, first_row=None) -> MagicMock:
    m = MagicMock()
    m.scalars.return_value.all.return_value = all_rows or []
    m.scalars.return_value.first.return_value = first_row
    return m


def _make_db(results: list) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock(side_effect=results)
    return db


def _make_proxy_session(
    last_activity_minutes_ago: float | None, peer_id=None, started_minutes_ago: float = 60
) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_agent_proxy"
    s.side = "agent"
    s.agent_id = uuid.uuid4()
    s.task_id = uuid.uuid4()
    s.peer_session_id = peer_id
    s.closed_at = None
    s.close_reason = None
    s.started_at = NOW - timedelta(minutes=started_minutes_ago)
    s.last_activity_at = (
        None
        if last_activity_minutes_ago is None
        else NOW - timedelta(minutes=last_activity_minutes_ago)
    )
    return s


def _make_proxy_message(sender_type: str, minutes_ago: float) -> MagicMock:
    """A PROXY_MESSAGE row as _cm_awaiting_agent_since sees it.

    timestamp is naive UTC to match the real column (datetime.utcnow default).
    """
    m = MagicMock()
    m.event_metadata = {"sender_type": sender_type}
    m.timestamp = NOW.replace(tzinfo=None) - timedelta(minutes=minutes_ago)
    return m


def _make_task(state=TaskState.IN_PROGRESS, run_id=None) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "Test task"
    t.state = state
    t.run_id = run_id
    t.updated_at = NOW - timedelta(hours=1)
    return t


@pytest.fixture(autouse=True)
def _no_memory_flush():
    """Isolate watchdog's own logic from §4's memory-flush side effects.

    flush_session_memory / maybe_refresh_room_memory (session lifecycle
    design, §4) make their own db.execute calls whenever a session closes
    or a task is blocked — every test in this file builds `db` from a strict,
    hand-counted queue of mocked results, so an uncounted extra call here would
    desync a later, unrelated assertion elsewhere in the same test. The wiring
    itself is covered separately by test_watchdog_memory_flush.py.
    """
    with (
        patch("app.core.watchdog.flush_session_memory", new_callable=AsyncMock) as mock_flush,
        patch(
            "app.core.watchdog.maybe_refresh_room_memory", new_callable=AsyncMock
        ) as mock_refresh,
        # The cm_self_plan sweep has its own query; it is covered by the
        # close_stale_plan_sessions tests at the bottom of this file.
        patch(
            "app.core.watchdog.close_stale_plan_sessions",
            new_callable=AsyncMock,
            return_value=0,
        ),
    ):
        yield mock_flush, mock_refresh


# ---------------------------------------------------------------------------
# run_watchdog_cycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stalled_session_emits_ephemeral_status_and_stays_open():
    """Past stall threshold (3 min) but before kill threshold (15 min)."""
    session = _make_proxy_session(last_activity_minutes_ago=5)
    db = _make_db([
        _result(all_rows=[]),          # block_stalled_legacy_tasks scan — none
        _result(all_rows=[]),          # idle user-session scan — none
        _result(all_rows=[session]),
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    mock_status.assert_awaited_once()
    args, kwargs = mock_status.await_args
    assert args[0] == session.room_id
    assert args[1] == AgentStatus.SESSION_STALLED
    assert kwargs["session_id"] == session.id
    assert kwargs["task_id"] == session.task_id
    assert kwargs["extra"]["stalled_for_s"] == 300

    mock_event.assert_not_awaited()
    assert session.closed_at is None, "stalled-but-alive session must not be closed"


@pytest.mark.asyncio
async def test_dead_session_killed_task_blocked_and_stalled_event_emitted(_no_memory_flush):
    """Past kill threshold: pair closed with reason='timeout', task → BLOCKED."""
    mock_flush, mock_refresh = _no_memory_flush
    peer = _make_proxy_session(last_activity_minutes_ago=20)
    peer.side = "chat_master"
    session = _make_proxy_session(last_activity_minutes_ago=20, peer_id=peer.id)
    task = _make_task(TaskState.IN_PROGRESS)
    session.task_id = task.id

    db = _make_db([
        _result(all_rows=[]),                # block_stalled_legacy_tasks scan — none
        _result(all_rows=[]),                # idle user-session scan — none
        _result(all_rows=[session, peer]),   # stalled select
        _result(first_row=peer),             # peer lookup
        _result(first_row=task),             # task lookup
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    assert session.closed_at == NOW
    assert session.close_reason == "timeout"
    assert peer.closed_at == NOW
    assert peer.close_reason == "timeout"
    assert task.state == TaskState.BLOCKED

    event_types = [c.kwargs["event_type"] for c in mock_event.await_args_list]
    assert event_types.count("SESSION_CLOSED") == 2
    assert "TASK_STALLED" in event_types
    stalled_call = next(
        c for c in mock_event.await_args_list if c.kwargs["event_type"] == "TASK_STALLED"
    )
    assert stalled_call.kwargs["metadata"]["reason"] == "timeout"
    assert stalled_call.kwargs["task_id"] == task.id

    # No ephemeral stall status for a killed session — and the peer must not
    # be processed twice (it was handled as part of the pair).
    mock_status.assert_not_awaited()
    # legacy scan + idle user scan + stalled select + peer lookup + task lookup
    assert db.execute.await_count == 5

    # §4: both closed sides get an Agent-memory flush attempt, and the newly
    # BLOCKED task triggers a Room memory refresh attempt.
    assert mock_flush.await_count == 2
    flushed_reasons = {c.kwargs["reason"] for c in mock_flush.await_args_list}
    assert flushed_reasons == {"timeout"}
    mock_refresh.assert_awaited_once_with(task.room_id, db, trigger="task_blocked")


@pytest.mark.asyncio
async def test_no_stalled_sessions_is_a_noop():
    db = _make_db([
        _result(all_rows=[]),  # block_stalled_legacy_tasks scan — none
        _result(all_rows=[]),  # idle user-session scan — none
        _result(all_rows=[]),
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    mock_status.assert_not_awaited()
    mock_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_dead_session_with_non_in_progress_task_does_not_emit_task_stalled():
    """Killed session whose task already left IN_PROGRESS → no TASK_STALLED."""
    session = _make_proxy_session(last_activity_minutes_ago=30)
    task = _make_task(TaskState.COMPLETED)
    session.task_id = task.id

    db = _make_db([
        _result(all_rows=[]),  # block_stalled_legacy_tasks scan — none
        _result(all_rows=[]),  # idle user-session scan — none
        _result(all_rows=[session]),
        _result(first_row=task),
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock),
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    assert task.state == TaskState.COMPLETED
    event_types = [c.kwargs["event_type"] for c in mock_event.await_args_list]
    assert "TASK_STALLED" not in event_types
    assert event_types.count("SESSION_CLOSED") == 1


@pytest.mark.asyncio
async def test_null_activity_session_falls_back_to_started_at_stall():
    """last_activity_at=NULL (loop crashed before first commit) → stall via started_at."""
    session = _make_proxy_session(last_activity_minutes_ago=None, started_minutes_ago=5)
    db = _make_db([
        _result(all_rows=[]),  # block_stalled_legacy_tasks scan — none
        _result(all_rows=[]),  # idle user-session scan — none
        _result(all_rows=[session]),
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    mock_status.assert_awaited_once()
    args, kwargs = mock_status.await_args
    assert args[1] == AgentStatus.SESSION_STALLED
    assert kwargs["extra"]["stalled_for_s"] == 300
    mock_event.assert_not_awaited()
    assert session.closed_at is None


@pytest.mark.asyncio
async def test_null_activity_session_falls_back_to_started_at_kill():
    """last_activity_at=NULL and started_at past the kill threshold → pair killed."""
    session = _make_proxy_session(last_activity_minutes_ago=None, started_minutes_ago=20)
    task = _make_task(TaskState.IN_PROGRESS)
    session.task_id = task.id

    db = _make_db([
        _result(all_rows=[]),  # block_stalled_legacy_tasks scan — none
        _result(all_rows=[]),  # idle user-session scan — none
        _result(all_rows=[session]),
        _result(first_row=task),
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    assert session.closed_at == NOW
    assert session.close_reason == "timeout"
    assert task.state == TaskState.BLOCKED
    event_types = [c.kwargs["event_type"] for c in mock_event.await_args_list]
    assert "TASK_STALLED" in event_types
    mock_status.assert_not_awaited()


# ---------------------------------------------------------------------------
# chat_master side: idle-by-design while the agent works (proxy dialogue)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cm_side_stale_while_agent_works_is_waiting_not_stalled():
    """CM proxy side past the stall threshold, but no pending agent message →
    ephemeral 'waiting_peer' (waiting for the agent), session stays open, no kill."""
    cm = _make_proxy_session(last_activity_minutes_ago=12, peer_id=uuid.uuid4())
    cm.side = "chat_master"
    db = _make_db([
        _result(all_rows=[]),          # legacy scan
        _result(all_rows=[]),          # idle user-session scan
        _result(all_rows=[cm]),        # stalled select
        _result(first_row=None),       # _cm_awaiting_agent_since — no proxy messages yet
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    mock_status.assert_awaited_once()
    args, kwargs = mock_status.await_args
    assert args[1] == AgentStatus.WAITING_PEER
    assert kwargs["extra"] == {"side": "chat_master", "waiting_for": "agent"}
    mock_event.assert_not_awaited()
    assert cm.closed_at is None


@pytest.mark.asyncio
async def test_cm_side_idle_when_last_proxy_message_was_its_own():
    """Last PROXY_MESSAGE came from the CM side → agent owes progress, not CM →
    treated as waiting, never stalled."""
    cm = _make_proxy_session(last_activity_minutes_ago=30, peer_id=uuid.uuid4())
    cm.side = "chat_master"
    db = _make_db([
        _result(all_rows=[]),
        _result(all_rows=[]),
        _result(all_rows=[cm]),
        _result(first_row=_make_proxy_message("chat_master", minutes_ago=25)),
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    assert mock_status.await_args[0][1] == AgentStatus.WAITING_PEER
    mock_event.assert_not_awaited()
    assert cm.closed_at is None


@pytest.mark.asyncio
async def test_cm_side_recently_asked_is_left_to_its_own_llm_status():
    """Agent handed CM a question inside the stall window → watchdog stays quiet
    (the CM turn's own llm_request_sent status covers it), session stays open."""
    cm = _make_proxy_session(last_activity_minutes_ago=10, peer_id=uuid.uuid4())
    cm.side = "chat_master"
    db = _make_db([
        _result(all_rows=[]),
        _result(all_rows=[]),
        _result(all_rows=[cm]),
        _result(first_row=_make_proxy_message("agent", minutes_ago=1)),
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    mock_status.assert_not_awaited()
    mock_event.assert_not_awaited()
    assert cm.closed_at is None


@pytest.mark.asyncio
async def test_cm_side_stalled_since_agent_message_emits_stall():
    """Agent's message to CM is older than the stall threshold but younger than
    the kill threshold → SESSION_STALLED dated from that message, session open."""
    cm = _make_proxy_session(last_activity_minutes_ago=40, peer_id=uuid.uuid4())
    cm.side = "chat_master"
    db = _make_db([
        _result(all_rows=[]),
        _result(all_rows=[]),
        _result(all_rows=[cm]),
        _result(first_row=_make_proxy_message("agent", minutes_ago=6)),
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    mock_status.assert_awaited_once()
    args, kwargs = mock_status.await_args
    assert args[1] == AgentStatus.SESSION_STALLED
    assert kwargs["extra"]["stalled_for_s"] == 360
    mock_event.assert_not_awaited()
    assert cm.closed_at is None


@pytest.mark.asyncio
async def test_cm_side_dead_since_agent_message_kills_pair(_no_memory_flush):
    """Agent's message to CM is older than the kill threshold → pair torn down."""
    # Same coroutine drives both sides: a CM answer-call that hangs 20 min blocks
    # the agent turn too, so the agent side is stale here as well.
    peer = _make_proxy_session(last_activity_minutes_ago=20)
    peer.side = "agent"
    cm = _make_proxy_session(last_activity_minutes_ago=40, peer_id=peer.id)
    cm.side = "chat_master"
    task = _make_task(TaskState.IN_PROGRESS)
    cm.task_id = task.id

    db = _make_db([
        _result(all_rows=[]),
        _result(all_rows=[]),
        _result(all_rows=[cm]),                                   # stalled select
        _result(first_row=_make_proxy_message("agent", 20)),      # _cm_awaiting_agent_since
        _result(first_row=peer),                                  # _kill_stalled_pair peer lookup
        _result(first_row=task),                                  # block_stalled_task
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    assert cm.closed_at == NOW and cm.close_reason == "timeout"
    assert peer.closed_at == NOW and peer.close_reason == "timeout"
    assert task.state == TaskState.BLOCKED
    mock_status.assert_not_awaited()
    event_types = [c.kwargs["event_type"] for c in mock_event.await_args_list]
    assert "TASK_STALLED" in event_types
    assert event_types.count("SESSION_CLOSED") == 2


# ---------------------------------------------------------------------------
# block_stalled_legacy_tasks (part 3, stage 3 — approve_and_start recovery)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_task_past_kill_threshold_is_blocked():
    """IN_PROGRESS, run_id=None, stale updated_at → BLOCKED + TASK_STALLED."""
    task = _make_task(TaskState.IN_PROGRESS, run_id=None)

    db = _make_db([
        _result(all_rows=[task.id]),  # legacy scan
        _result(first_row=task),      # block_stalled_task's own lookup
    ])

    with patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event:
        blocked = await block_stalled_legacy_tasks(db, now=NOW)

    assert blocked == 1
    assert task.state == TaskState.BLOCKED
    assert task.runtime_state == "stop_requested"
    stalled_call = next(
        c for c in mock_event.await_args_list if c.kwargs["event_type"] == "TASK_STALLED"
    )
    assert stalled_call.kwargs["metadata"]["reason"] == "legacy_pipeline_stalled"
    assert stalled_call.kwargs["task_id"] == task.id


@pytest.mark.asyncio
async def test_no_stale_legacy_tasks_is_a_noop():
    db = _make_db([_result(all_rows=[])])

    with patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event:
        blocked = await block_stalled_legacy_tasks(db, now=NOW)

    assert blocked == 0
    mock_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_watchdog_cycle_recovers_legacy_task_with_no_stalled_sessions():
    """A stuck legacy task must be recovered even when no proxy session is stalled —
    the legacy scan must not depend on / be short-circuited by the session check."""
    task = _make_task(TaskState.IN_PROGRESS, run_id=None)

    db = _make_db([
        _result(all_rows=[task.id]),  # legacy scan
        _result(first_row=task),      # block_stalled_task's own lookup
        _result(all_rows=[]),         # idle user-session scan — none
        _result(all_rows=[]),         # stalled CmAgentProxySession select — none
    ])

    with (
        patch("app.core.watchdog.emit_status", new_callable=AsyncMock) as mock_status,
        patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event,
    ):
        await run_watchdog_cycle(db, now=NOW)

    assert task.state == TaskState.BLOCKED
    event_types = [c.kwargs["event_type"] for c in mock_event.await_args_list]
    assert "TASK_STALLED" in event_types
    mock_status.assert_not_awaited()


# ---------------------------------------------------------------------------
# cleanup_orphaned_sessions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orphan_cleanup_closes_sessions_and_blocks_tasks(_no_memory_flush):
    mock_flush, mock_refresh = _no_memory_flush
    task = _make_task(TaskState.IN_PROGRESS)
    s1 = _make_proxy_session(last_activity_minutes_ago=1)
    s2 = _make_proxy_session(last_activity_minutes_ago=1)
    s2.side = "chat_master"
    s1.task_id = task.id
    s2.task_id = task.id

    db = _make_db([
        _result(all_rows=[s1, s2]),  # open proxy sessions
        _result(first_row=task),     # task lookup (single shared task_id)
        _result(all_rows=[]),        # open user-facing sessions — none
    ])

    with patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event:
        closed = await cleanup_orphaned_sessions(db, now=NOW)

    assert closed == 2
    for s in (s1, s2):
        assert s.closed_at == NOW
        assert s.close_reason == "orphaned_restart"
    assert task.state == TaskState.BLOCKED

    event_types = [c.kwargs["event_type"] for c in mock_event.await_args_list]
    assert event_types.count("SESSION_CLOSED") == 2
    assert event_types.count("TASK_STALLED") == 1

    # §4: both orphaned sides get a memory-flush attempt; the newly BLOCKED
    # task triggers a Room memory refresh attempt.
    assert mock_flush.await_count == 2
    flushed_reasons = {c.kwargs["reason"] for c in mock_flush.await_args_list}
    assert flushed_reasons == {"orphaned_restart"}
    mock_refresh.assert_awaited_once_with(task.room_id, db, trigger="task_blocked")

    stalled_call = next(
        c for c in mock_event.await_args_list if c.kwargs["event_type"] == "TASK_STALLED"
    )
    assert stalled_call.kwargs["metadata"]["reason"] == "orphaned_restart"


@pytest.mark.asyncio
async def test_orphan_cleanup_noop_when_no_open_sessions():
    db = _make_db([
        _result(all_rows=[]),  # open proxy sessions — none
        _result(all_rows=[]),  # open user-facing sessions — none
    ])

    with patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event:
        closed = await cleanup_orphaned_sessions(db, now=NOW)

    assert closed == 0
    mock_event.assert_not_awaited()


# ---------------------------------------------------------------------------
# close_stale_plan_sessions — cm_self_plan rows orphaned by a process kill
# ---------------------------------------------------------------------------


def _make_plan_session(last_activity_minutes_ago: float | None) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_self_plan"
    s.closed_at = None
    s.close_reason = None
    s.started_at = NOW - timedelta(hours=5)
    s.last_activity_at = (
        None
        if last_activity_minutes_ago is None
        else NOW - timedelta(minutes=last_activity_minutes_ago)
    )
    return s


@pytest.mark.asyncio
async def test_stale_plan_sessions_closed_with_given_reason():
    stale = _make_plan_session(last_activity_minutes_ago=None)
    db = _make_db([_result(all_rows=[stale])])

    with patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event:
        closed = await close_stale_plan_sessions(db, reason="orphaned_restart", now=NOW)

    assert closed == 1
    assert stale.closed_at == NOW
    assert stale.close_reason == "orphaned_restart"
    meta = mock_event.await_args.kwargs["metadata"]
    assert meta["kind"] == "cm_self_plan"
    assert meta["close_reason"] == "orphaned_restart"


@pytest.mark.asyncio
async def test_stale_plan_sweep_filters_by_kind_and_idle_threshold(monkeypatch):
    monkeypatch.setenv("PLAN_SESSION_STALE_MINUTES", "60")
    db = _make_db([_result(all_rows=[])])

    with patch("app.core.watchdog.emit_event", new_callable=AsyncMock) as mock_event:
        closed = await close_stale_plan_sessions(db, reason="timeout", now=NOW)

    assert closed == 0
    mock_event.assert_not_awaited()
    sql = str(
        db.execute.await_args.args[0].compile(compile_kwargs={"literal_binds": True})
    )
    assert "cm_self_plan" in sql
    assert "closed_at IS NULL" in sql
    assert "coalesce(session.last_activity_at, session.started_at)" in sql
    assert "2026-07-17 11:00:00" in sql  # NOW - 60 min


@pytest.mark.asyncio
async def test_orphan_cleanup_sweeps_stale_plan_sessions():
    db = _make_db([
        _result(all_rows=[]),  # open proxy sessions — none
        _result(all_rows=[]),  # open user-facing sessions — none
    ])
    with patch(
        "app.core.watchdog.close_stale_plan_sessions",
        new_callable=AsyncMock,
        return_value=3,
    ) as sweep:
        closed = await cleanup_orphaned_sessions(db, now=NOW)

    assert closed == 3
    sweep.assert_awaited_once_with(db, reason="orphaned_restart", now=NOW)
