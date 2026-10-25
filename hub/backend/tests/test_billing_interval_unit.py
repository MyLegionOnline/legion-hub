"""The billing unit of accounting is "the room is working", not "the container is running".

Product rule (decision of 2026-09-23): hibernation is an internal tool of the
Hub, the user does not know about it, and all working time of the room is billed.
Hence the invariant these tests check: stopping the container by itself
does NOT close the billing interval while the room has live work;
the end of the work does.

Before, the link was direct (the interval lived exactly as long as the container), and
matched reality only because the idle reaper stops the container exactly
when the work ends. Two cases break this coincidence: an agent
working with files (the tools go past the container there — 0.3 % of session time
on a live stack), and future hibernation, which stops the container in the middle of work
by design.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.workspace import Workspace, WorkspaceUsage


def _result(first=None, all_rows=None, scalars_all=None):
    res = MagicMock()
    res.first.return_value = first
    res.all.return_value = all_rows or []
    res.scalars.return_value.all.return_value = scalars_all or []
    res.scalar_one_or_none.return_value = None
    return res


def _db(execute_side_effect):
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.flush = AsyncMock()
    db.execute = AsyncMock(side_effect=execute_side_effect)
    return db


@pytest.mark.asyncio
async def test_container_stop_keeps_interval_open_while_work_runs():
    """Hibernation/stopping the container in the middle of work does not cut the bill."""
    from app.core.workspace_lifecycle import _close_workspace_usage

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)

    usage.container_started_at = datetime.now(timezone.utc) - timedelta(minutes=30)
    # First room_has_live_work query — an active task is found; then
    # the selection of open intervals.
    db = _db([_result(first=(uuid.uuid4(),)), _result(scalars_all=[usage])])

    await _close_workspace_usage(ws, db, stop_reason="hibernate")

    assert usage.stopped_at is None, "the interval was closed although the room is still working"
    # The container segment did end — for the cost model.
    assert usage.container_started_at is None
    assert usage.container_seconds == pytest.approx(1800, abs=5)


@pytest.mark.asyncio
async def test_container_stop_closes_interval_when_idle():
    """When there is no work, stopping the container closes the interval as before."""
    from app.core.workspace_lifecycle import _close_workspace_usage

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)

    # No tasks, no running sessions, nobody holds the room open,
    # then the selection of open intervals.
    db = _db([
        _result(first=None), _result(first=None), _result(first=None),
        _result(scalars_all=[usage]),
        _result(first=None),  # close_usage: last session/task activity
    ])

    await _close_workspace_usage(ws, db, stop_reason="idle")

    assert usage.stopped_at is not None
    assert usage.stop_reason == "idle"


@pytest.mark.asyncio
async def test_sweep_closes_interval_after_work_ends(monkeypatch):
    """The work ended while the container was down — the reconciler sweeps.

    Without this pass an interval that outlived its container would no longer be closed by
    anyone: the container stop happened earlier and will not happen a second time.
    """
    from app.core import workspace_reconciler as rec

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    ws.status = "stopped"
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = datetime.now(timezone.utc) - timedelta(hours=3)

    # The session worked until the last minute — the interval is paid in full.
    last_session = datetime.now(timezone.utc) - timedelta(minutes=1)
    db = _db([_result(all_rows=[(usage, ws)]), _result(first=(last_session, None))])
    monkeypatch.setattr(
        "app.core.workspace_lifecycle.room_has_live_work", AsyncMock(return_value=False)
    )
    recorded: list[float] = []
    monkeypatch.setattr(
        "app.core.metrics.record_workspace_session_hours", recorded.append
    )

    await rec._close_finished_billing_intervals(db)

    assert usage.stop_reason == "work_ended"
    assert usage.stopped_at is not None
    assert recorded == [pytest.approx(3.0, abs=0.01)]


@pytest.mark.asyncio
async def test_sweep_leaves_interval_open_while_work_runs(monkeypatch):
    from app.core import workspace_reconciler as rec

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    ws.status = "stopped"
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = datetime.now(timezone.utc) - timedelta(minutes=20)

    db = _db([_result(all_rows=[(usage, ws)])])
    monkeypatch.setattr(
        "app.core.workspace_lifecycle.room_has_live_work", AsyncMock(return_value=True)
    )

    await rec._close_finished_billing_intervals(db)

    assert usage.stopped_at is None


@pytest.mark.asyncio
async def test_sweep_force_closes_a_stuck_interval(monkeypatch):
    """Safeguard: a session that never closes must not
    be billed forever — the quota counts an open interval as running."""
    from app.core import workspace_reconciler as rec

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    ws.status = "stopped"
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = datetime.now(timezone.utc) - timedelta(
        seconds=rec.STALE_OPEN_INTERVAL_SEC + 60
    )

    db = _db([_result(all_rows=[(usage, ws)]), _result(first=None)])
    # The room persistently reports work — this is exactly the case we insure against.
    monkeypatch.setattr(
        "app.core.workspace_lifecycle.room_has_live_work", AsyncMock(return_value=True)
    )
    monkeypatch.setattr("app.core.metrics.record_workspace_session_hours", lambda _h: None)

    await rec._close_finished_billing_intervals(db)

    assert usage.stop_reason == "stale_open"
    assert usage.stopped_at is not None


# ---------- Interval without a container (owner's decision 2026-09-25) ----------
#
# The limit counts all the time while the client uses the room: the room is open in
# the browser (including while the person thinks over an answer to the agent) and then, while
# the agents work after the browser is closed. Whether a container exists
# does not matter. Before, an interval was born only at container start, and
# ChatMaster's work with files while the workspace was stopped was not billed at all.


@pytest.mark.asyncio
async def test_open_room_counts_as_live_work():
    """A person keeps the room open — the bill runs even when nobody is working."""
    from app.core.workspace_lifecycle import room_has_live_work

    # No tasks, no running sessions, but the presence is fresh.
    db = _db([_result(first=None), _result(first=None), _result(first=(uuid.uuid4(),))])

    assert await room_has_live_work(uuid.uuid4(), db) is True


def test_fold_container_time_accumulates_and_is_idempotent():
    from app.core.billing_interval import fold_container_time

    now = datetime.now(timezone.utc)
    usage = WorkspaceUsage(workspace_id=uuid.uuid4(), mem_limit_mb=1024)
    usage.container_seconds = 100.0
    usage.container_started_at = now - timedelta(seconds=60)

    fold_container_time(usage, now)
    fold_container_time(usage, now)

    assert usage.container_seconds == pytest.approx(160.0)
    assert usage.container_started_at is None


@pytest.mark.asyncio
async def test_ensure_usage_open_never_overlaps_previous_interval():
    """Back-dated, yes, but not on top of an already paid interval."""
    from app.core.billing_interval import ensure_usage_open

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    last_stop = datetime.now(timezone.utc) - timedelta(minutes=2)
    no_open = _result()
    no_open.scalars.return_value.first.return_value = None
    last = MagicMock()
    last.scalar.return_value = last_stop
    db = _db([no_open, last])

    usage = await ensure_usage_open(
        ws, db, started_at=last_stop - timedelta(minutes=5), container=True
    )

    assert usage.started_at == last_stop
    assert usage.stopped_at is None
    assert usage.container_started_at is not None
    db.add.assert_called()


def _sweep_db(session_rows, task_rows, present_rows, open_row=None, last_stop=None):
    """The order of open_intervals_for_working_rooms queries for one room."""
    open_res = _result()
    open_res.scalars.return_value.first.return_value = open_row
    last = MagicMock()
    last.scalar.return_value = last_stop
    seq = [
        _result(all_rows=session_rows),
        _result(all_rows=task_rows),
        _result(all_rows=present_rows),
        open_res,          # does the room already have an open interval?
        last,              # end of the previous interval
        open_res,          # ensure_usage_open: re-check
        last,
    ]
    return _db(seq)


@pytest.mark.asyncio
async def test_sweep_opens_interval_for_containerless_session(monkeypatch):
    """ChatMaster reads files while the workspace is stopped — the interval
    opens and is dated by the session start."""
    from app.core import billing_interval as bi

    room_id = uuid.uuid4()
    ws = Workspace(id=uuid.uuid4(), room_id=room_id)
    monkeypatch.setattr(
        "app.core.workspace_lifecycle._get_or_create_workspace", AsyncMock(return_value=ws)
    )
    session_start = datetime.now(timezone.utc) - timedelta(seconds=50)
    db = _sweep_db([(room_id, session_start)], [], [])

    assert await bi.open_intervals_for_working_rooms(db) == 1
    usage = db.add.call_args_list[-1].args[0]
    assert isinstance(usage, WorkspaceUsage)
    assert usage.started_at == session_start
    assert usage.container_started_at is None


@pytest.mark.asyncio
async def test_sweep_caps_backdating(monkeypatch):
    from app.core import billing_interval as bi

    room_id = uuid.uuid4()
    ws = Workspace(id=uuid.uuid4(), room_id=room_id)
    monkeypatch.setattr(
        "app.core.workspace_lifecycle._get_or_create_workspace", AsyncMock(return_value=ws)
    )
    db = _sweep_db([(room_id, datetime.now(timezone.utc) - timedelta(days=60))], [], [])

    await bi.open_intervals_for_working_rooms(db)

    usage = db.add.call_args_list[-1].args[0]
    age = (datetime.now(timezone.utc) - usage.started_at).total_seconds()
    assert age == pytest.approx(bi.MAX_BACKDATE_SEC, abs=5)


@pytest.mark.asyncio
async def test_sweep_does_not_reopen_on_work_older_than_last_interval(monkeypatch):
    """The 24 h safeguard closed a stuck session's interval — the same session
    must not open it again, otherwise the safeguard is pointless."""
    from app.core import billing_interval as bi

    room_id = uuid.uuid4()
    ws = Workspace(id=uuid.uuid4(), room_id=room_id)
    monkeypatch.setattr(
        "app.core.workspace_lifecycle._get_or_create_workspace", AsyncMock(return_value=ws)
    )
    now = datetime.now(timezone.utc)
    db = _sweep_db(
        [(room_id, now - timedelta(hours=30))], [], [], last_stop=now - timedelta(hours=6)
    )

    assert await bi.open_intervals_for_working_rooms(db) == 0


@pytest.mark.asyncio
async def test_sweep_opens_on_presence(monkeypatch):
    from app.core import billing_interval as bi

    room_id = uuid.uuid4()
    ws = Workspace(id=uuid.uuid4(), room_id=room_id)
    monkeypatch.setattr(
        "app.core.workspace_lifecycle._get_or_create_workspace", AsyncMock(return_value=ws)
    )
    db = _sweep_db([], [], [(room_id,)], last_stop=datetime.now(timezone.utc) - timedelta(hours=1))

    assert await bi.open_intervals_for_working_rooms(db) == 1


# ---------- The dot in the room list = the bill ----------


def test_billing_change_is_published_only_after_commit(monkeypatch):
    """Opening/closing an interval reaches the dot after the commit, and a rollback
    drops the change — the dot will not show what did not happen."""
    import asyncio

    from app.core import billing_interval as bi

    sent: list = []

    async def fake_publish(room_id, active):
        sent.append((room_id, active))

    monkeypatch.setattr("app.core.workspace_lifecycle.publish_room_billing", fake_publish)

    class _Sync:
        def __init__(self):
            self.info = {}

    class _Db:
        def __init__(self):
            self.sync_session = _Sync()

    room_a, room_b = uuid.uuid4(), uuid.uuid4()

    async def scenario():
        committed, rolled_back = _Db(), _Db()
        bi.mark_billing_change(committed, room_a, True)
        bi.mark_billing_change(rolled_back, room_b, True)
        bi._drop_after_rollback(rolled_back.sync_session)
        bi._publish_after_commit(rolled_back.sync_session)
        bi._publish_after_commit(committed.sync_session)
        await asyncio.sleep(0)
        # A repeat commit of the same session does not send a second time.
        bi._publish_after_commit(committed.sync_session)
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert sent == [(room_a, True)]


def test_listener_is_registered_once():
    from sqlalchemy import event
    from sqlalchemy.orm import Session as SyncSession

    from app.core import billing_interval as bi

    bi._register_listeners()
    assert event.contains(SyncSession, "after_commit", bi._publish_after_commit)
