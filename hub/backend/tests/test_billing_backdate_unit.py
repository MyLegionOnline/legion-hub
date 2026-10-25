"""Back-dated closing of a billing interval (economics.md §2).

An interval is closed when the Hub NOTICES that the work has ended. While the control plane
is watching, that is within a tick. After a window when nobody was watching (the backend or
Docker were not running, auto-stop did not fire), closing at the moment `now`
billed the entire gap: on a live stack 40.2 h for a room that
was used for 5 minutes. Under internal design notes §1 "Unit of accounting" such time
is not paid — the interval ends at the last evidence of use
plus the tail that the rule gives that evidence.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.models.workspace import Workspace, WorkspaceUsage

_IDLE = 900.0


def _result(first=None, scalars_all=None, scalar=None):
    res = MagicMock()
    res.first.return_value = first
    res.all.return_value = []
    res.scalars.return_value.all.return_value = scalars_all or []
    res.scalar_one_or_none.return_value = scalar
    return res


def _db(results):
    """Results in order, then empty ones (the reconciler tick's tail)."""
    queue = iter(results)

    async def execute(_stmt):
        return next(queue, _result())

    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    db.rollback = AsyncMock()
    db.execute = AsyncMock(side_effect=execute)
    return db


@pytest.fixture
def idle_window(monkeypatch):
    monkeypatch.setattr("app.core.workspace_lifecycle.WORKSPACE_IDLE_TIMEOUT_SEC", int(_IDLE))


@pytest.fixture
def recorded(monkeypatch):
    hours: list[float] = []
    monkeypatch.setattr("app.core.metrics.record_workspace_session_hours", hours.append)
    return hours


def _ts(**delta):
    return datetime(2026, 9, 27, 10, 41, tzinfo=timezone.utc) - timedelta(**delta)


# ---------- billed_stop_at: pure function ----------


def test_live_case_from_economics():
    """A live case: the interval 25.09 18:32, last activity 18:37,
    the container vanished, noticed 27.09 10:41 — paid until 18:52, not 40 h."""
    from app.core.billing_interval import billed_stop_at

    started = datetime(2026, 9, 25, 18, 32, tzinfo=timezone.utc)
    activity = datetime(2026, 9, 25, 18, 37, tzinfo=timezone.utc)
    now = datetime(2026, 9, 27, 10, 41, tzinfo=timezone.utc)

    stop = billed_stop_at(started, now, activity=(activity,), idle_sec=_IDLE)

    assert stop == activity + timedelta(seconds=_IDLE)
    assert (stop - started).total_seconds() / 60 == pytest.approx(20.0)


def test_latest_evidence_wins_with_its_own_tail():
    """Presence extends the bill by the socket's freshness, activity — by the idle
    window; whichever ends later is taken."""
    from app.core.billing_interval import PRESENCE_FRESH_SEC, billed_stop_at

    now, started, presence = _ts(), _ts(hours=10), _ts(hours=2)

    stop = billed_stop_at(
        started, now, activity=(None, _ts(hours=3)), presence_at=presence, idle_sec=_IDLE
    )
    assert stop == presence + timedelta(seconds=PRESENCE_FRESH_SEC)

    stop = billed_stop_at(
        started, now, activity=(_ts(minutes=100),), presence_at=presence, idle_sec=_IDLE
    )
    assert stop == _ts(minutes=100) + timedelta(seconds=_IDLE)


def test_recent_use_closes_at_now():
    """The normal case: the work ended a minute ago — closing at `now`."""
    from app.core.billing_interval import billed_stop_at

    now = _ts()
    assert billed_stop_at(_ts(hours=1), now, activity=(_ts(minutes=1),), idle_sec=_IDLE) == now


def test_never_before_start_and_naive_tolerant():
    from app.core.billing_interval import billed_stop_at

    now, started = _ts(), _ts(hours=5)
    # Activity long before the interval started (an old task) — not earlier than the start.
    assert billed_stop_at(started, now, activity=(_ts(days=10),), idle_sec=_IDLE) == started
    # Naive timestamps of legacy rows are UTC by construction.
    stop = billed_stop_at(
        started.replace(tzinfo=None), now,
        activity=(_ts(hours=4).replace(tzinfo=None),), idle_sec=_IDLE,
    )
    assert stop == _ts(hours=4) + timedelta(seconds=_IDLE)


def test_no_evidence_is_one_idle_window_capped_by_now():
    from app.core.billing_interval import billed_stop_at

    now = _ts()
    assert billed_stop_at(_ts(hours=40), now, idle_sec=_IDLE) == _ts(hours=40) + timedelta(
        seconds=_IDLE
    )
    assert billed_stop_at(_ts(minutes=3), now, idle_sec=_IDLE) == now


# ---------- close_usage: the row, the counter and container_seconds agree ----------


@pytest.mark.asyncio
async def test_close_usage_backdates_counter_and_container(idle_window, recorded):
    from app.core.billing_interval import close_usage

    now = datetime.now(timezone.utc)
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    ws.last_activity_at = now - timedelta(hours=40) + timedelta(minutes=5)
    # The reconciler's status probe is the container's life, not use of the room.
    ws.last_heartbeat_at = now - timedelta(minutes=1)
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = now - timedelta(hours=40)
    usage.container_started_at = usage.started_at  # the container lived all 40 h
    db = _db([_result(first=(None, None))])

    hours = await close_usage(usage, ws, db, stop_reason="reconciler", now=now)

    assert usage.stopped_at == ws.last_activity_at + timedelta(seconds=_IDLE)
    assert usage.stop_reason == "reconciler"
    assert hours == pytest.approx(20 / 60)
    assert recorded == [pytest.approx(20 / 60)]
    assert usage.container_started_at is None
    assert usage.container_seconds == pytest.approx(20 * 60)


@pytest.mark.asyncio
async def test_close_usage_uses_session_and_task_activity(idle_window, recorded):
    from app.core.billing_interval import close_usage

    now = datetime.now(timezone.utc)
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = now - timedelta(hours=6)
    session_touch = now - timedelta(hours=5)
    task_touch = (now - timedelta(hours=4)).replace(tzinfo=None)  # Task.updated_at is naive
    db = _db([_result(first=(session_touch, task_touch))])

    await close_usage(usage, ws, db, stop_reason="work_ended", now=now)

    assert usage.stopped_at == now - timedelta(hours=4) + timedelta(seconds=_IDLE)
    assert recorded == [pytest.approx(2 + _IDLE / 3600)]


@pytest.mark.asyncio
async def test_close_usage_clamps_previously_folded_container_time(idle_window, recorded):
    """A segment already added at the moment `now` when the container stopped does not
    exceed the paid duration — the §2 share stays ≤ 100 %."""
    from app.core.billing_interval import close_usage

    now = datetime.now(timezone.utc)
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = now - timedelta(hours=30)
    usage.container_seconds = 30 * 3600.0
    db = _db([_result(first=(None, now - timedelta(hours=29)))])

    await close_usage(usage, ws, db, stop_reason="work_ended", now=now)

    billed = (usage.stopped_at - usage.started_at).total_seconds()
    assert billed == pytest.approx(3600 + _IDLE)
    assert usage.container_seconds == pytest.approx(billed)
    assert recorded == [pytest.approx(billed / 3600)]


# ---------- Closing paths ----------


def _gone_container(monkeypatch, live: bool):
    from app.core import workspaced_client

    monkeypatch.setattr(
        workspaced_client, "get_workspace_status",
        AsyncMock(side_effect=httpx.RequestError("container gone")),
    )
    monkeypatch.setattr(workspaced_client, "list_workspace_ids", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "app.core.workspace_lifecycle.room_has_live_work", AsyncMock(return_value=live)
    )


@pytest.mark.asyncio
async def test_reconciler_backdates_externally_stopped_container(
    monkeypatch, idle_window, recorded
):
    """The "container stopped externally" branch: an unobserved gap is not billed."""
    from app.core import workspace_reconciler as rec

    _gone_container(monkeypatch, live=False)
    now = datetime.now(timezone.utc)
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4(), status="running")
    ws.last_activity_at = now - timedelta(hours=40) + timedelta(minutes=5)
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = now - timedelta(hours=40)
    usage.container_started_at = usage.started_at
    db = _db([_result(scalars_all=[ws]), _result(scalar=usage)])

    await rec._reconcile_tick(db)

    assert usage.stop_reason == "reconciler"
    assert usage.stopped_at == ws.last_activity_at + timedelta(seconds=_IDLE)
    assert recorded == [pytest.approx(20 / 60, abs=1e-3)]
    assert usage.container_seconds == pytest.approx(20 * 60, abs=1)
    assert ws.status == "stopped"


@pytest.mark.asyncio
async def test_reconciler_keeps_interval_open_while_work_continues(monkeypatch, recorded):
    from app.core import workspace_reconciler as rec

    _gone_container(monkeypatch, live=True)
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4(), status="running")
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.container_started_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    db = _db([_result(scalars_all=[ws]), _result(scalar=usage)])

    await rec._reconcile_tick(db)

    assert usage.stopped_at is None
    assert usage.container_started_at is None
    assert usage.container_seconds == pytest.approx(600, abs=5)
    assert recorded == []


@pytest.mark.asyncio
async def test_stale_guard_bills_only_up_to_the_last_activity(monkeypatch, idle_window, recorded):
    """The 24 h safeguard: a stuck row held the "work" but there was no activity
    — only the time up to the last activity is billed."""
    from app.core import workspace_reconciler as rec

    monkeypatch.setattr(
        "app.core.workspace_lifecycle.room_has_live_work", AsyncMock(return_value=True)
    )
    now = datetime.now(timezone.utc)
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4(), status="stopped")
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = now - timedelta(seconds=rec.STALE_OPEN_INTERVAL_SEC + 3600)
    last_session = usage.started_at + timedelta(hours=2)
    rows = _result()
    rows.all.return_value = [(usage, ws)]
    db = _db([rows, _result(first=(last_session, None))])

    await rec._close_finished_billing_intervals(db)

    assert usage.stop_reason == "stale_open"
    assert usage.stopped_at == last_session + timedelta(seconds=_IDLE)
    assert recorded == [pytest.approx(2 + _IDLE / 3600, abs=1e-3)]


@pytest.mark.asyncio
async def test_late_idle_stop_is_backdated(monkeypatch, idle_window, recorded):
    """auto_stop_idle_workspaces, delayed by sleep or a backend crash,
    closes the interval at the last activity, not at the moment of stopping."""
    from app.core.workspace_lifecycle import _close_workspace_usage

    now = datetime.now(timezone.utc)
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    ws.last_activity_at = now - timedelta(hours=8)
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = now - timedelta(hours=9)
    # room_has_live_work: no tasks, sessions, presence; open intervals;
    # session/task activity.
    db = _db([
        _result(), _result(), _result(),
        _result(scalars_all=[usage]),
        _result(first=(None, None)),
    ])

    await _close_workspace_usage(ws, db, stop_reason="idle")

    assert usage.stopped_at == ws.last_activity_at + timedelta(seconds=_IDLE)
    assert recorded == [pytest.approx(1 + _IDLE / 3600, abs=1e-3)]
