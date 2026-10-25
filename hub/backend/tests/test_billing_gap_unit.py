"""An open billing interval is not carried across a window nobody watched.

Live case (2026-10-15/16): an interval opened at 18:55, the agents' last work
at 19:27, then the Hub was down until the owner opened the room the next
afternoon at 15:13. The presence beat found the evening's interval still open
and reused it; when the room went idle at 15:53 it was closed with 20.97 h,
the whole night included. Closing back-dates only the END of an interval, so
the gap inside it has to be split off when the interval is picked up again.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.workspace import Workspace, WorkspaceUsage

_IDLE = 900.0


def _result(first=None, scalars_first=None, scalar=None, all_rows=None):
    res = MagicMock()
    res.first.return_value = first
    res.all.return_value = all_rows or []
    res.scalars.return_value.first.return_value = scalars_first
    res.scalar.return_value = scalar
    return res


def _db(results):
    queue = iter(results)

    async def execute(_stmt):
        return next(queue, _result())

    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    db.execute = AsyncMock(side_effect=execute)
    return db


@pytest.fixture(autouse=True)
def idle_window(monkeypatch):
    monkeypatch.setattr("app.core.workspace_lifecycle.WORKSPACE_IDLE_TIMEOUT_SEC", int(_IDLE))


@pytest.fixture
def recorded(monkeypatch):
    hours: list[float] = []
    monkeypatch.setattr("app.core.metrics.record_workspace_session_hours", hours.append)
    return hours


def _room(now, *, started_h, activity=None, presence=None, container=False):
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4(), mem_limit_mb=1024)
    ws.last_activity_at = activity
    ws.presence_at = presence
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = now - timedelta(hours=started_h)
    usage.container_seconds = 0.0
    if container:
        usage.container_started_at = usage.started_at
    return ws, usage


def _added(db, cls):
    return [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], cls)]


# ---------- ensure_usage_open ----------


@pytest.mark.asyncio
async def test_gap_splits_the_interval(recorded):
    """The live case: billed up to the last work plus the idle window, then anew."""
    from app.core.billing_interval import ensure_usage_open

    now = datetime.now(timezone.utc)
    last_work = now - timedelta(hours=19, minutes=46)
    ws, usage = _room(now, started_h=20.3, activity=last_work)
    # _open_row, then session/task activity (the task finished at 19:27 too).
    db = _db([_result(scalars_first=usage), _result(first=(last_work, last_work))])

    new = await ensure_usage_open(ws, db)

    assert usage.stop_reason == "gap"
    assert usage.stopped_at == last_work + timedelta(seconds=_IDLE)
    assert recorded == [pytest.approx(
        (usage.stopped_at - usage.started_at).total_seconds() / 3600
    )]
    assert new is not usage
    assert new.stopped_at is None
    assert abs((new.started_at - now).total_seconds()) < 5
    assert new.container_started_at is None
    assert _added(db, WorkspaceUsage)[-1] is new


@pytest.mark.asyncio
async def test_no_gap_reuses_the_interval(recorded):
    from app.core.billing_interval import ensure_usage_open

    now = datetime.now(timezone.utc)
    ws, usage = _room(now, started_h=3, activity=now - timedelta(minutes=10))
    db = _db([_result(scalars_first=usage), _result(first=(None, None))])

    assert await ensure_usage_open(ws, db) is usage
    assert usage.stopped_at is None
    assert recorded == []


@pytest.mark.asyncio
async def test_quiet_but_watched_session_activity_keeps_the_interval(recorded):
    """The room's own stamp is old, but an agent session worked minutes ago."""
    from app.core.billing_interval import ensure_usage_open

    now = datetime.now(timezone.utc)
    ws, usage = _room(now, started_h=5, activity=now - timedelta(hours=4))
    db = _db([
        _result(scalars_first=usage),
        _result(first=(now - timedelta(minutes=2), None)),
    ])

    assert await ensure_usage_open(ws, db) is usage
    assert usage.stopped_at is None


@pytest.mark.asyncio
async def test_presence_only_room(recorded):
    """A room only looked at: the last beat is the evidence, with its freshness tail."""
    from app.core.billing_interval import PRESENCE_FRESH_SEC, ensure_usage_open

    now = datetime.now(timezone.utc)
    # A beat a minute ago — the socket is live, the interval carries on.
    ws, usage = _room(now, started_h=1, presence=now - timedelta(seconds=60))
    db = _db([_result(scalars_first=usage), _result(first=(None, None))])
    assert await ensure_usage_open(ws, db) is usage
    assert usage.stopped_at is None

    # The last beat was three hours ago — the tab was closed, the Hub went down.
    last_beat = now - timedelta(hours=3)
    ws, usage = _room(now, started_h=4, presence=last_beat)
    db = _db([_result(scalars_first=usage), _result(first=(None, None))])
    new = await ensure_usage_open(ws, db)
    assert new is not usage
    assert usage.stop_reason == "gap"
    assert usage.stopped_at == last_beat + timedelta(seconds=PRESENCE_FRESH_SEC)
    assert recorded == [pytest.approx(1 + PRESENCE_FRESH_SEC / 3600)]


@pytest.mark.asyncio
async def test_container_start_uses_activity_from_before_its_stamp(recorded):
    """The start path stamps last_activity_at first; the previous value is the evidence."""
    from app.core.billing_interval import ensure_usage_open

    now = datetime.now(timezone.utc)
    before = now - timedelta(hours=10)
    ws, usage = _room(now, started_h=11, activity=now)  # already stamped
    db = _db([_result(scalars_first=usage), _result(first=(None, None))])

    new = await ensure_usage_open(ws, db, container=True, last_activity_at=before)

    assert usage.stopped_at == before + timedelta(seconds=_IDLE)
    assert new is not usage
    assert new.container_started_at is not None
    assert abs((new.container_started_at - now).total_seconds()) < 5


@pytest.mark.asyncio
async def test_container_time_stays_within_each_interval(recorded):
    """A container the Hub thought was running across the gap: its stretch is
    cut at the old interval's end and starts again with the new one."""
    from app.core.billing_interval import ensure_usage_open

    now = datetime.now(timezone.utc)
    last_work = now - timedelta(hours=6)
    ws, usage = _room(now, started_h=7, activity=last_work, container=True)
    db = _db([_result(scalars_first=usage), _result(first=(None, None))])

    new = await ensure_usage_open(ws, db)

    billed = (usage.stopped_at - usage.started_at).total_seconds()
    assert usage.container_started_at is None
    assert usage.container_seconds == pytest.approx(billed)
    assert new.container_started_at is not None


@pytest.mark.asyncio
async def test_no_open_row_opens_one_as_before(recorded):
    from app.core.billing_interval import ensure_usage_open

    now = datetime.now(timezone.utc)
    ws, _ = _room(now, started_h=1)
    db = _db([_result(scalars_first=None), _result(scalar=None)])

    new = await ensure_usage_open(ws, db)

    assert new.stopped_at is None
    assert abs((new.started_at - now).total_seconds()) < 5
    assert recorded == []


# ---------- touch_presence: the old beat is read before it is overwritten ----------


def _presence_env(monkeypatch, ws, db):
    @asynccontextmanager
    async def session():
        yield db

    monkeypatch.setattr(
        "sqlalchemy.ext.asyncio.async_sessionmaker", lambda *a, **k: session
    )
    monkeypatch.setattr(
        "app.core.workspace_lifecycle._get_or_create_workspace",
        AsyncMock(return_value=ws),
    )


@pytest.mark.asyncio
async def test_touch_presence_after_a_gap_splits(monkeypatch, recorded):
    from app.core.billing_interval import PRESENCE_FRESH_SEC, touch_presence

    now = datetime.now(timezone.utc)
    last_beat = now - timedelta(hours=20)
    ws, usage = _room(now, started_h=20.5, presence=last_beat)
    db = _db([_result(scalars_first=usage), _result(first=(None, None))])
    _presence_env(monkeypatch, ws, db)

    assert await touch_presence(ws.room_id) is True

    assert usage.stop_reason == "gap"
    assert usage.stopped_at == last_beat + timedelta(seconds=PRESENCE_FRESH_SEC)
    assert abs((ws.presence_at - now).total_seconds()) < 5
    opened = [u for u in _added(db, WorkspaceUsage) if u is not usage]
    assert len(opened) == 1 and opened[0].stopped_at is None
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_touch_presence_of_a_live_socket_reuses(monkeypatch, recorded):
    from app.core.billing_interval import PRESENCE_BEAT_SEC, touch_presence

    now = datetime.now(timezone.utc)
    ws, usage = _room(now, started_h=2, presence=now - timedelta(seconds=PRESENCE_BEAT_SEC))
    db = _db([_result(scalars_first=usage), _result(first=(None, None))])
    _presence_env(monkeypatch, ws, db)

    assert await touch_presence(ws.room_id) is True

    assert usage.stopped_at is None
    assert _added(db, WorkspaceUsage) == []
    assert recorded == []


# ---------- The reconciler's pass after a blind window ----------


@pytest.mark.asyncio
async def test_unobserved_pass_reopens_only_rooms_that_still_work(monkeypatch, recorded):
    from app.core.billing_interval import close_unobserved_intervals

    now = datetime.now(timezone.utc)
    working_ws, working = _room(
        now, started_h=12, activity=now - timedelta(hours=11), container=True
    )
    idle_ws, idle = _room(now, started_h=12, activity=now - timedelta(hours=11))
    fresh_ws, fresh = _room(now, started_h=1, activity=now - timedelta(minutes=1))
    db = _db([
        _result(all_rows=[(working, working_ws), (idle, idle_ws), (fresh, fresh_ws)]),
        _result(first=(None, None)),
        _result(first=(None, None)),
        _result(first=(None, None)),
    ])
    monkeypatch.setattr(
        "app.core.workspace_lifecycle.room_has_live_work",
        AsyncMock(side_effect=lambda room_id, _db: room_id == working_ws.room_id),
    )

    assert await close_unobserved_intervals(db) == 2

    assert working.stop_reason == idle.stop_reason == "gap"
    assert fresh.stopped_at is None
    opened = [u for u in _added(db, WorkspaceUsage) if u not in (working, idle, fresh)]
    assert len(opened) == 1
    assert opened[0].workspace_id == working_ws.id
    assert opened[0].container_started_at is not None
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconciler_runs_the_pass_only_after_a_blind_window(monkeypatch):
    from app.core import workspace_reconciler as rec

    calls: list[str] = []
    sleeps = iter([0, 0, 0])
    monkeypatch.setattr(rec, "_INTERVAL_SEC", 0)
    monkeypatch.setattr(rec, "_BLIND_TICK_SEC", 3600)

    async def fake_sleep(_sec):
        if next(sleeps, None) is None:
            raise __import__("asyncio").CancelledError

    monkeypatch.setattr(rec.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(
        rec, "_split_unobserved_intervals", AsyncMock(side_effect=lambda _f: calls.append("split"))
    )
    monkeypatch.setattr(
        rec, "_reconcile_tick", AsyncMock(side_effect=lambda _db: calls.append("tick"))
    )
    monkeypatch.setattr("app.core.bg_wake.wake_finished_processes", AsyncMock())

    @asynccontextmanager
    async def session():
        yield MagicMock()

    monkeypatch.setattr(rec, "sessionmaker", lambda *a, **k: session)

    await rec.run_reconciler_loop(engine=None)

    # The first tick after start splits; the punctual ones after it do not.
    assert calls == ["split", "tick", "tick", "tick"]
