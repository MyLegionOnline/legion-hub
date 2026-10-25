"""internal design notes T0.16 / internal design notes T10: the disk watchdog."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.dialects import postgresql

from app.core import disk_watch as dw

GIB = 1024 ** 3
MIB = 1024 ** 2
NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.setattr(dw, "_last_level", None)
    monkeypatch.setattr(dw, "_last_stop_at", None)


@pytest.mark.parametrize("free, level", [
    (50, "ok"), (15, "ok"), (14.9, "low"), (5, "low"), (4.9, "emergency"), (0, "emergency"),
])
def test_levels(free, level):
    assert dw.VolumeState(1000 * MIB, int(free * 10 * MIB)).level == level


def test_zero_size_volume_is_an_emergency():
    assert dw.VolumeState(0, 0).level == "emergency"


def test_growth_is_last_minus_first_sample_per_room():
    a, b = uuid4(), uuid4()
    samples = [
        (a, NOW - timedelta(minutes=50), 1 * GIB),
        (a, NOW, 3 * GIB),
        (a, NOW - timedelta(minutes=20), 2 * GIB),
        (b, NOW - timedelta(minutes=30), 2 * GIB),
        (b, NOW, GIB),
    ]
    assert dw.growth_by_room(samples) == {a: 2 * GIB, b: -GIB}


def test_pick_the_fastest_room_above_the_floor():
    a, b = uuid4(), uuid4()
    assert dw.pick_room_to_stop({a: 100 * MIB, b: 2 * GIB}) == (b, 2 * GIB)
    assert dw.pick_room_to_stop({a: 10 * MIB}, min_growth=64 * MIB) is None
    assert dw.pick_room_to_stop({}) is None


def test_notice_names_the_growth_and_the_way_out():
    text = dw.notice_text(dw.VolumeState(100 * GIB, 4 * GIB), int(2.5 * GIB))
    assert "4.0 %" in text and "2.5 GiB" in text and "Clean up dependencies" in text


def _db():
    db = MagicMock()
    result = MagicMock()
    result.rowcount = 0
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    return db


def _probe(monkeypatch, size, free):
    from app.core import workspaced_client as wsd

    monkeypatch.setattr(wsd, "get_data_volume", AsyncMock(return_value={
        "size_bytes": size, "free_bytes": free,
    }))


@pytest.mark.asyncio
async def test_healthy_volume_publishes_and_stops_nothing(monkeypatch):
    from app.core import metrics

    gauges = {}
    for name in ("data_volume_size_bytes", "data_volume_free_bytes",
                 "data_volume_checked_timestamp_seconds"):
        gauges[name] = MagicMock()
        monkeypatch.setattr(metrics, name, gauges[name])
    _probe(monkeypatch, 100 * GIB, 40 * GIB)
    stop = AsyncMock()
    monkeypatch.setattr(dw, "_stop_fastest_room", stop)
    result = await dw.watch_data_volume(_db(), now=NOW)
    assert result["level"] == "ok" and result["stopped"] is None
    stop.assert_not_awaited()
    gauges["data_volume_free_bytes"].set.assert_called_once_with(40 * GIB)
    gauges["data_volume_size_bytes"].set.assert_called_once_with(100 * GIB)
    gauges["data_volume_checked_timestamp_seconds"].set.assert_called_once_with(NOW.timestamp())


@pytest.mark.asyncio
async def test_low_volume_alerts_without_stopping(monkeypatch):
    _probe(monkeypatch, 100 * GIB, 10 * GIB)
    stop = AsyncMock()
    monkeypatch.setattr(dw, "_stop_fastest_room", stop)
    assert (await dw.watch_data_volume(_db(), now=NOW))["level"] == "low"
    stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_probe_failure_is_survived(monkeypatch):
    from app.core import workspaced_client as wsd

    monkeypatch.setattr(wsd, "get_data_volume", AsyncMock(side_effect=OSError("down")))
    assert await dw.watch_data_volume(MagicMock(), now=NOW) == {"level": "unknown", "stopped": None}


def _rows(rows):
    result = MagicMock()
    result.all.return_value = rows
    return result


@pytest.mark.asyncio
async def test_emergency_stops_the_fastest_growing_room(monkeypatch):
    from app.core import room_close, room_storage

    _probe(monkeypatch, 100 * GIB, 3 * GIB)
    fast, slow, stale = uuid4(), uuid4(), uuid4()
    seen = []

    async def execute(stmt):
        seen.append(stmt)
        if len(seen) == 1:      # candidates
            return _rows([
                (fast, NOW - timedelta(seconds=30)),
                (slow, NOW - timedelta(seconds=30)),
                (stale, NOW - timedelta(minutes=9)),
            ])
        return _rows([          # samples of the last hour
            (fast, NOW - timedelta(minutes=50), 1 * GIB),
            (fast, NOW, 4 * GIB),
            (slow, NOW - timedelta(minutes=50), 1 * GIB),
            (slow, NOW, 1 * GIB + 300 * MIB),
            (stale, NOW, 2 * GIB),
        ])

    db = MagicMock()
    db.execute = execute
    db.commit = AsyncMock()
    measure = AsyncMock(return_value=0)
    monkeypatch.setattr(room_storage, "measure_room", measure)
    close = AsyncMock(return_value={
        "paused_task_ids": [], "container_stopped": True, "billing_closed": False,
    })
    monkeypatch.setattr(room_close, "close_room", close)

    result = await dw.watch_data_volume(db, now=NOW)
    for stmt in seen:
        str(stmt.compile(dialect=postgresql.dialect()))
    # Only the room not measured in the last two minutes is re-measured.
    assert [c.args[0] for c in measure.await_args_list] == [stale]
    assert result["level"] == "emergency"
    assert result["stopped"]["room_id"] == fast
    assert result["stopped"]["grew_bytes"] == 3 * GIB
    close.assert_awaited_once()
    assert close.await_args.args[0] == fast
    kwargs = close.await_args.kwargs
    assert kwargs["pause_reason"] == dw.PAUSE_REASON
    assert kwargs["notice_event"] == dw.DISK_EMERGENCY_EVENT
    # The room stays readable: the work stops, the view does not close.
    assert kwargs["close_view"] is False
    # Marked before the stop, so it does not start again while the disk is full.
    mark = str(seen[2].compile(dialect=postgresql.dialect()))
    assert mark.startswith("UPDATE workspace SET disk_stopped_at")


@pytest.mark.asyncio
async def test_emergency_cooldown_between_stops(monkeypatch):
    _probe(monkeypatch, 100 * GIB, 3 * GIB)
    monkeypatch.setattr(dw, "_last_stop_at", NOW - timedelta(seconds=dw.STOP_COOLDOWN_SEC - 1))
    db = MagicMock()
    db.execute = AsyncMock()
    assert (await dw.watch_data_volume(db, now=NOW))["stopped"] is None
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_emergency_without_a_growing_room_stops_nothing(monkeypatch):
    from app.core import room_close

    _probe(monkeypatch, 100 * GIB, 3 * GIB)
    room = uuid4()
    db = _db()
    db.execute = AsyncMock(side_effect=[
        _rows([(room, NOW)]),
        _rows([(room, NOW - timedelta(minutes=40), GIB), (room, NOW, GIB + 10 * MIB)]),
    ])
    close = AsyncMock()
    monkeypatch.setattr(room_close, "close_room", close)
    assert (await dw.watch_data_volume(db, now=NOW))["stopped"] is None
    close.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_room_disk_keeps_a_sample(monkeypatch):
    from app.core import room_storage, workspace_lifecycle
    from app.models.workspace import WorkspaceDiskSample

    ws = MagicMock()
    monkeypatch.setattr(workspace_lifecycle, "_get_or_create_workspace", AsyncMock(return_value=ws))
    db = MagicMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()
    room_id = uuid4()
    await room_storage.record_room_disk(room_id, 5 * MIB, db)
    samples = [c.args[0] for c in db.add.call_args_list if isinstance(c.args[0], WorkspaceDiskSample)]
    assert len(samples) == 1
    assert samples[0].room_id == room_id and samples[0].disk_bytes == 5 * MIB
    # Old samples of the room are pruned in the same transaction.
    stmt = db.execute.await_args.args[0]
    assert "DELETE FROM workspace_disk_sample" in str(stmt.compile(dialect=postgresql.dialect()))
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_volume_with_room_again_lifts_the_stop_marks(monkeypatch):
    _probe(monkeypatch, 100 * GIB, 6 * GIB)
    db = _db()
    db.execute.return_value.rowcount = 2
    assert (await dw.watch_data_volume(db, now=NOW))["level"] == "low"
    stmt = str(db.execute.await_args.args[0].compile(dialect=postgresql.dialect()))
    assert stmt.startswith("UPDATE workspace SET disk_stopped_at")
    assert "IS NOT NULL" in stmt
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_emergency_keeps_the_stop_marks(monkeypatch):
    _probe(monkeypatch, 100 * GIB, 3 * GIB)
    lift = AsyncMock()
    monkeypatch.setattr(dw, "_lift_stop_marks", lift)
    monkeypatch.setattr(dw, "_stop_fastest_room", AsyncMock(return_value=None))
    await dw.watch_data_volume(_db(), now=NOW)
    lift.assert_not_awaited()


def _scalar(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
async def test_disk_stop_refusal():
    db = MagicMock()
    db.execute = AsyncMock(return_value=_scalar(NOW))
    assert await dw.disk_stop_refusal(uuid4(), db) == dw.refusal_text()
    db.execute = AsyncMock(return_value=_scalar(None))
    assert await dw.disk_stop_refusal(uuid4(), db) is None
    db.execute = AsyncMock(side_effect=RuntimeError("db"))
    assert await dw.disk_stop_refusal(uuid4(), db) is None


@pytest.mark.asyncio
async def test_room_work_and_writes_refused_while_stopped(monkeypatch):
    from app.core import room_storage

    monkeypatch.setattr(dw, "disk_stop_refusal", AsyncMock(return_value="stopped"))
    owner = AsyncMock()
    monkeypatch.setattr(room_storage, "_room_owner", owner)
    assert await room_storage.room_storage_refusal(uuid4(), MagicMock()) == "stopped"
    owner.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_stopped_room_opens_without_a_workspace(monkeypatch):
    from types import SimpleNamespace

    from app.api.v1 import rooms
    from app.core import work_gates, workspace_lifecycle
    from app.core import workspace_quota as wq

    monkeypatch.setattr(work_gates, "open_refusal", AsyncMock(return_value=None))
    monkeypatch.setattr(wq, "get_user_tier", AsyncMock(return_value="hobby"))
    monkeypatch.setattr(
        workspace_lifecycle, "ensure_workspace_started",
        AsyncMock(side_effect=workspace_lifecycle.WorkspaceStartError(
            dw.PAUSE_REASON, "stopped", status_code=507,
        )),
    )
    monkeypatch.setattr(
        workspace_lifecycle, "_get_or_create_workspace", AsyncMock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(workspace_lifecycle, "_workspace_payload", lambda w: {"status": "stopped"})
    monkeypatch.setattr(rooms.RoomOut, "model_validate", classmethod(lambda cls, r: {"id": "r"}))
    member = MagicMock()
    member.scalar_one_or_none.return_value = object()
    db = MagicMock()
    db.execute = AsyncMock(return_value=member)
    db.get = AsyncMock(return_value=SimpleNamespace(deletion_started_at=None, creator_id=uuid4()))
    db.commit = AsyncMock()

    out = await rooms.open_room(uuid4(), db=db, current_user=SimpleNamespace(id=uuid4()))
    assert out["disk_stopped"] == {"reason": dw.PAUSE_REASON, "message": "stopped"}
    assert out["workspace"] == {"status": "stopped"}
