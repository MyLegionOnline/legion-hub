"""internal design notes T3: room files quota, level 1 (accounting in the app).

Run: pytest tests/test_room_storage.py -v
Unit tests only (no database): the DB is a mock wherever one is needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.core import room_storage as rs
from app.core import workspace_quota as wq

MIB = 1024 * 1024
GIB = 1024 * MIB


def _status(used: int, quota_mb, tier: str = "hobby") -> rs.StorageStatus:
    return rs.StorageStatus(tier=tier, used_bytes=used, quota_mb=quota_mb)


# ---------- Thresholds ------------------------------------------------------------

@pytest.mark.parametrize(
    "used, level",
    [
        (0, "ok"),
        (int(3 * GIB * 0.79), "ok"),
        (3 * GIB * 81 // 100, "warn"),
        (3 * GIB * 96 // 100, "critical"),
        (3 * GIB, "full"),
        (4 * GIB, "full"),
    ],
)
def test_levels_on_hobby(used, level):
    status = _status(used, 3072)
    assert status.level == level
    assert status.full is (level == "full")


def test_no_cap_is_never_full():
    status = _status(500 * GIB, None, tier="selfhosted")
    assert status.percent is None
    assert status.level == "ok" and not status.full


def test_an_expired_plans_zero_quota_does_not_pause_for_disk():
    """TR4: an expired plan is read-only already; the disk does not add a pause."""
    assert not _status(GIB, 0, tier="free").full


def test_refusal_text_names_the_way_out():
    text = rs.refusal_text(_status(3 * GIB, 3072))
    assert "3 GiB of 3 GiB" in text
    assert "Clean up dependencies" in text


def test_usage_payload():
    room = rs.RoomDisk(uuid4(), "Shop", 300 * MIB, datetime(2026, 10, 2, tzinfo=timezone.utc))
    payload = rs.usage_payload(
        rs.StorageStatus(tier="hobby", used_bytes=300 * MIB, quota_mb=3072, rooms=[room])
    )
    assert payload["storage_used_mb"] == 300.0
    assert payload["storage_quota_mb"] == 3072
    assert payload["storage_level"] == "ok"
    assert payload["storage_rooms"][0]["name"] == "Shop"
    assert payload["storage_rooms"][0]["used_mb"] == 300.0


@pytest.mark.asyncio
async def test_storage_status_sums_the_owners_rooms(monkeypatch):
    monkeypatch.setattr(wq, "get_effective_limits", AsyncMock(return_value={
        "plan_tier": "hobby", "room_storage_quota_mb": 3072,
    }))
    rows = MagicMock()
    rows.all.return_value = [
        (uuid4(), "a", GIB, None),
        (uuid4(), "b", 2 * GIB, None),
        (uuid4(), "never measured", None, None),
    ]
    db = MagicMock()
    db.execute = AsyncMock(return_value=rows)
    status = await rs.storage_status(uuid4(), db)
    assert status.used_bytes == 3 * GIB
    assert status.full
    assert [r.name for r in status.rooms] == ["b", "a", "never measured"]


# ---------- Gates -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_room_refusal_fails_open(monkeypatch):
    monkeypatch.setattr(rs, "_room_owner", AsyncMock(side_effect=RuntimeError("db")))
    assert await rs.room_storage_refusal(uuid4(), MagicMock()) is None


@pytest.mark.asyncio
async def test_file_api_gate_is_a_402(monkeypatch):
    monkeypatch.setattr(rs, "_room_owner", AsyncMock(return_value=uuid4()))
    monkeypatch.setattr(rs, "storage_status", AsyncMock(return_value=_status(3 * GIB, 3072)))
    with pytest.raises(HTTPException) as exc:
        await rs.require_room_storage(uuid4(), MagicMock())
    assert exc.value.status_code == 402
    assert exc.value.detail["error_code"] == "storage_full"

    monkeypatch.setattr(rs, "storage_status", AsyncMock(return_value=_status(GIB, 3072)))
    await rs.require_room_storage(uuid4(), MagicMock())


@pytest.mark.asyncio
async def test_upload_refused_at_the_quota(monkeypatch):
    from app.api.v1 import workspace_files

    monkeypatch.setattr(workspace_files, "_get_room", AsyncMock())
    monkeypatch.setattr(workspace_files, "_verify_membership", AsyncMock())
    monkeypatch.setattr(
        workspace_files, "require_room_storage",
        AsyncMock(side_effect=HTTPException(status_code=402, detail="full")),
    )
    write = AsyncMock()
    monkeypatch.setattr(workspace_files.wsd, "write_workspace_file", write)
    upload = SimpleNamespace(filename="a.txt", content_type="text/plain", read=AsyncMock(return_value=b"x"))
    with pytest.raises(HTTPException) as exc:
        await workspace_files.workspace_fs_upload(
            uuid4(), file=upload, target_path="/", relative_path="", commit_message=None,
            db=MagicMock(), current_user=SimpleNamespace(id=uuid4()),
        )
    assert exc.value.status_code == 402
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_is_allowed_and_remeasures(monkeypatch):
    from app.api.v1 import workspace_files

    monkeypatch.setattr(workspace_files, "_get_room", AsyncMock())
    monkeypatch.setattr(workspace_files, "_verify_membership", AsyncMock())
    gate = AsyncMock(side_effect=HTTPException(status_code=402, detail="full"))
    monkeypatch.setattr(workspace_files, "require_room_storage", gate)
    monkeypatch.setattr(workspace_files.wsd, "delete_workspace_file", AsyncMock(return_value=True))
    monkeypatch.setattr(workspace_files, "emit_event", AsyncMock())
    measure = MagicMock()
    monkeypatch.setattr(workspace_files, "schedule_room_measure", measure)
    room_id = uuid4()
    out = await workspace_files.workspace_fs_delete_file(
        room_id, path="big.bin", commit_message=None,
        db=MagicMock(), current_user=SimpleNamespace(id=uuid4()),
    )
    assert out.ok
    gate.assert_not_awaited()
    measure.assert_called_once_with(room_id)


@pytest.mark.asyncio
async def test_no_task_dispatch_at_the_quota(monkeypatch):
    from app.core import work_gates
    from app.core.proxy_session import TaskDispatchPreflightError, prepare_task_dispatch
    from app.models.task import TaskState

    monkeypatch.setattr(work_gates, "work_refusal", AsyncMock(return_value=None))
    monkeypatch.setattr(rs, "room_storage_refusal", AsyncMock(return_value="full"))
    task = SimpleNamespace(assigned_agent_id=uuid4(), state=TaskState.ASSIGNED, room_id=uuid4())
    with pytest.raises(TaskDispatchPreflightError) as exc:
        await prepare_task_dispatch(task, MagicMock())
    assert exc.value.code == "storage_full"


@pytest.mark.asyncio
async def test_refuse_room_work_posts_a_storage_notice(monkeypatch):
    from app.core import events, work_gates

    monkeypatch.setattr(work_gates, "work_refusal", AsyncMock(return_value=None))
    monkeypatch.setattr(rs, "room_storage_refusal", AsyncMock(return_value="full"))
    emit = AsyncMock()
    monkeypatch.setattr(events, "emit_event", emit)
    room_id = uuid4()
    assert await work_gates.refuse_room_work(room_id, MagicMock())
    assert emit.await_args.args[:3] == (room_id, rs.STORAGE_FULL_EVENT, "full")


@pytest.mark.asyncio
async def test_write_file_tool_refuses_at_the_quota(monkeypatch):
    from app.core.tools._write_file import WriteFileTool

    monkeypatch.setattr(rs, "room_storage_refusal", AsyncMock(return_value="full"))
    from app.core import workspaced_client as wsd

    write = AsyncMock()
    monkeypatch.setattr(wsd, "write_workspace_file", write)
    ctx = SimpleNamespace(room_id=uuid4(), db=MagicMock(), session_id="s", task_id=None)
    out = await WriteFileTool().execute(ctx, {"path": "a.txt", "content": "x"})
    assert out == "Error: full"
    write.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_room_at_the_quota_opens_without_a_workspace(monkeypatch):
    """§5.3: the room opens, so its files can be deleted; nothing starts."""
    from app.api.v1 import rooms
    from app.core import work_gates, workspace_lifecycle

    monkeypatch.setattr(work_gates, "open_refusal", AsyncMock(return_value=None))
    monkeypatch.setattr(wq, "get_user_tier", AsyncMock(return_value="hobby"))
    monkeypatch.setattr(
        workspace_lifecycle, "ensure_workspace_started",
        AsyncMock(side_effect=workspace_lifecycle.WorkspaceStartError(
            "storage_full", "full", status_code=402,
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
    assert out["storage_full"] == {"reason": "storage_full", "message": "full"}
    assert out["workspace"] == {"status": "stopped"}


# ---------- Measuring and the reconciler pass --------------------------------------

@pytest.mark.asyncio
async def test_measure_room_records_bytes(monkeypatch):
    from app.core import workspaced_client as wsd

    monkeypatch.setattr(wsd, "measure_workspace_disk", AsyncMock(return_value={"bytes": 42, "exists": True}))
    record = AsyncMock()
    monkeypatch.setattr(rs, "record_room_disk", record)
    room_id = uuid4()
    assert await rs.measure_room(room_id, MagicMock()) == 42
    record.assert_awaited_once()
    assert record.await_args.args[:2] == (room_id, 42)


@pytest.mark.asyncio
async def test_measure_room_survives_workspaced_down(monkeypatch):
    from app.core import workspaced_client as wsd

    monkeypatch.setattr(wsd, "measure_workspace_disk", AsyncMock(side_effect=OSError("down")))
    record = AsyncMock()
    monkeypatch.setattr(rs, "record_room_disk", record)
    assert await rs.measure_room(uuid4(), MagicMock()) is None
    record.assert_not_awaited()


def test_stale():
    now = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)
    assert rs._stale(None, now)
    assert rs._stale(now - timedelta(minutes=10), now)
    assert not rs._stale(now - timedelta(minutes=9), now)
    assert not rs._stale((now - timedelta(minutes=1)).replace(tzinfo=None), now)


def _rows(rows):
    result = MagicMock()
    result.all.return_value = rows
    return result


@pytest.mark.asyncio
async def test_sweep_measures_stale_rooms_and_pauses_full_owners(monkeypatch):
    now = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)
    full_owner, ok_owner = uuid4(), uuid4()
    stale_room, fresh_room, ok_room, backfill_room = uuid4(), uuid4(), uuid4(), uuid4()
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[
        _rows([
            (stale_room, full_owner, now - timedelta(hours=1)),
            (fresh_room, full_owner, now - timedelta(minutes=1)),
            (ok_room, ok_owner, now),
        ]),
        _rows([(backfill_room, ok_owner)]),
        _rows([]),  # no disk-paused tasks
    ])
    measure = AsyncMock(return_value=1)
    monkeypatch.setattr(rs, "measure_room", measure)
    statuses = {full_owner: _status(3 * GIB, 3072), ok_owner: _status(GIB, 3072)}
    monkeypatch.setattr(rs, "storage_status", AsyncMock(side_effect=lambda o, _db: statuses[o]))
    pause = AsyncMock(return_value=1)
    monkeypatch.setattr(rs, "_pause_owner_work", pause)

    assert await rs.enforce_storage_quota(db, now=now) == 1
    assert [c.args[0] for c in measure.await_args_list] == [stale_room, backfill_room]
    pause.assert_awaited_once()
    assert pause.await_args.args[0] == full_owner


@pytest.mark.asyncio
async def test_pause_owner_work_posts_a_notice_where_tasks_paused(monkeypatch):
    from app.core import events, workspace_lifecycle

    busy, idle = uuid4(), uuid4()
    rooms = MagicMock()
    rooms.scalars.return_value.all.return_value = [busy, idle]
    db = MagicMock()
    db.execute = AsyncMock(return_value=rooms)
    db.commit = AsyncMock()
    task_id = uuid4()
    pause = AsyncMock(side_effect=lambda room_id, _db, pause_reason: [task_id] if room_id == busy else [])
    monkeypatch.setattr(workspace_lifecycle, "pause_workspace_tasks", pause)
    emit = AsyncMock()
    monkeypatch.setattr(events, "emit_event", emit)

    assert await rs._pause_owner_work(uuid4(), "full", db) == 1
    assert {c.kwargs["pause_reason"] for c in pause.await_args_list} == {"storage_full"}
    emit.assert_awaited_once()
    assert emit.await_args.args[:3] == (busy, rs.STORAGE_FULL_EVENT, "full")


@pytest.mark.asyncio
async def test_sweep_queries_compile_for_postgres():
    from sqlalchemy.dialects import postgresql

    seen = []

    async def execute(stmt):
        seen.append(stmt)
        return _rows([])

    db = MagicMock()
    db.execute = execute
    assert await rs.enforce_storage_quota(db) == 0
    for stmt in seen:
        str(stmt.compile(dialect=postgresql.dialect()))
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_clean_endpoint_returns_megabytes(monkeypatch):
    from app.api.v1 import workspace_files
    from app.core import workspaced_client as wsd

    monkeypatch.setattr(workspace_files, "_get_room", AsyncMock())
    monkeypatch.setattr(workspace_files, "_verify_membership", AsyncMock())
    monkeypatch.setattr(wsd, "clean_workspace_dependencies", AsyncMock(return_value={
        "removed": ["node_modules"], "freed_bytes": 276 * MIB, "bytes": 6 * MIB,
    }))
    record = AsyncMock()
    monkeypatch.setattr(rs, "record_room_disk", record)
    monkeypatch.setattr(rs, "_room_owner", AsyncMock(return_value=None))
    out = await workspace_files.workspace_clean_dependencies(
        uuid4(), db=MagicMock(), current_user=SimpleNamespace(id=uuid4()),
    )
    assert out["removed"] == ["node_modules"]
    assert out["freed_mb"] == 276.0 and out["used_mb"] == 6.0
    assert record.await_args.args[1] == 6 * MIB


# ---------- Resuming after space was freed ----------------------------------------


@pytest.mark.asyncio
async def test_sweep_resumes_disk_paused_work_once_under_the_quota(monkeypatch):
    freed, still_full = uuid4(), uuid4()
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[_rows([]), _rows([])])
    monkeypatch.setattr(rs, "_rooms_with_paused_work",
                        AsyncMock(return_value=[(uuid4(), freed), (uuid4(), still_full)]))
    statuses = {freed: _status(GIB, 3072), still_full: _status(3 * GIB, 3072)}
    monkeypatch.setattr(rs, "storage_status", AsyncMock(side_effect=lambda o, _db: statuses[o]))
    monkeypatch.setattr(rs, "notify_storage_warning", AsyncMock(return_value=False))
    resume = AsyncMock(return_value=1)
    monkeypatch.setattr(rs, "resume_owner_storage_work", resume)
    pause = AsyncMock(return_value=1)
    monkeypatch.setattr(rs, "_pause_owner_work", pause)

    assert await rs.enforce_storage_quota(db) == 0
    resume.assert_awaited_once()
    assert resume.await_args.args[0] == freed
    # A full owner with nothing live left is not paused again.
    pause.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_owner_work_resumes_and_posts_a_notice(monkeypatch):
    from app.core import events, workspace_lifecycle

    owner, busy, idle = uuid4(), uuid4(), uuid4()
    task_id = uuid4()
    monkeypatch.setattr(rs, "storage_status", AsyncMock(return_value=_status(GIB, 3072)))
    monkeypatch.setattr(rs, "_rooms_with_paused_work", AsyncMock(return_value=[(busy, owner), (idle, owner)]))
    resume = AsyncMock(side_effect=lambda room_id, _db, **kw: [task_id] if room_id == busy else [])
    monkeypatch.setattr(workspace_lifecycle, "resume_paused_tasks", resume)
    emit = AsyncMock()
    monkeypatch.setattr(events, "emit_event", emit)

    assert await rs.resume_owner_storage_work(owner, MagicMock()) == 1
    assert {c.kwargs["pause_reason"] for c in resume.await_args_list} == {"storage_full"}
    emit.assert_awaited_once()
    assert emit.await_args.args[:2] == (busy, rs.STORAGE_FREED_EVENT)


@pytest.mark.asyncio
async def test_resume_owner_work_waits_while_still_full(monkeypatch):
    monkeypatch.setattr(rs, "storage_status", AsyncMock(return_value=_status(3 * GIB, 3072)))
    rooms = AsyncMock()
    monkeypatch.setattr(rs, "_rooms_with_paused_work", rooms)
    assert await rs.resume_owner_storage_work(uuid4(), MagicMock()) == 0
    rooms.assert_not_awaited()


@pytest.mark.asyncio
async def test_clean_dependencies_schedules_the_resume(monkeypatch):
    from app.core import workspaced_client as wsd
    from app.db import release

    owner = uuid4()
    monkeypatch.setattr(release, "release_db", AsyncMock())
    monkeypatch.setattr(wsd, "clean_workspace_dependencies", AsyncMock(return_value={"bytes": 5}))
    monkeypatch.setattr(rs, "record_room_disk", AsyncMock())
    monkeypatch.setattr(rs, "_room_owner", AsyncMock(return_value=owner))
    scheduled = MagicMock()
    monkeypatch.setattr(rs, "schedule_owner_resume", scheduled)

    await rs.clean_room_dependencies(uuid4(), MagicMock())
    scheduled.assert_called_once_with(owner)


@pytest.mark.asyncio
async def test_resume_paused_tasks_query_compiles_for_postgres():
    from sqlalchemy.dialects import postgresql

    seen = []

    async def execute(stmt):
        seen.append(stmt)
        return _rows([])

    db = MagicMock()
    db.execute = execute
    assert await rs._rooms_with_paused_work(db, uuid4()) == []
    str(seen[0].compile(dialect=postgresql.dialect()))


# ---------- The 80 % letter -------------------------------------------------------


@pytest.fixture()
def letters(monkeypatch):
    from app.plugins import get_registry, reset_registry

    rs._warned.clear()
    sent = AsyncMock(return_value=True)
    reset_registry()
    get_registry().event_handlers[rs.USER_NOTICE_EVENT] = [sent]
    monkeypatch.setattr(wq, "get_hours_window", AsyncMock(return_value=(
        datetime(2026, 10, 1, tzinfo=timezone.utc), datetime(2026, 11, 1, tzinfo=timezone.utc))))
    yield sent
    rs._warned.clear()
    reset_registry()


def _account_db(account_id):
    result = MagicMock()
    result.scalar_one_or_none.return_value = account_id
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_warning_letter_goes_once_per_hour_window(letters):
    owner, account = uuid4(), uuid4()
    status = _status(3 * GIB * 85 // 100, 3072)
    assert await rs.notify_storage_warning(owner, status, _account_db(account)) is True
    assert await rs.notify_storage_warning(owner, status, _account_db(account)) is False
    letters.assert_awaited_once()
    account_id, kind, ref, params = letters.await_args.args
    assert (account_id, kind, ref) == (account, "storage_warning", "2026-10-01")
    assert params["quota_mb"] == 3072 and params["percent"] == 85.0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [
    _status(GIB, 3072),                          # under 80 %
    _status(500 * GIB, None, tier="selfhosted"),  # no cap
    _status(GIB, 0, tier="free"),                # expired plan
])
async def test_no_warning_letter(letters, status):
    assert await rs.notify_storage_warning(uuid4(), status, _account_db(uuid4())) is False
    letters.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refused_letter_is_asked_for_again(letters):
    owner = uuid4()
    status = _status(3 * GIB, 3072)
    letters.return_value = False
    assert await rs.notify_storage_warning(owner, status, _account_db(uuid4())) is False
    letters.return_value = True
    assert await rs.notify_storage_warning(owner, status, _account_db(uuid4())) is True
    assert letters.await_count == 2


@pytest.mark.asyncio
async def test_no_letter_without_a_notice_handler(monkeypatch):
    """Self-hosted (no cloud plugin): nobody to ask, no letter."""
    from app.plugins import reset_registry

    reset_registry()
    rs._warned.clear()
    status = _status(3 * GIB, 3072)
    assert await rs.notify_storage_warning(uuid4(), status, _account_db(uuid4())) is False


def test_kernel_enospc_hint_only_for_failed_commands_and_remeasures(monkeypatch):
    scheduled = []
    monkeypatch.setattr(rs, "schedule_room_measure", scheduled.append)
    room = uuid4()

    hint = rs.kernel_enospc_hint(room, 1, "", "dd: error writing 'f': No space left on device")
    assert hint == rs.KERNEL_ENOSPC_HINT
    assert scheduled == [room]

    assert rs.kernel_enospc_hint(room, 1, "", "cp: Disk quota exceeded") != ""
    # A successful command that merely printed the phrase, or an unrelated
    # failure, gets no note and no extra measurement.
    scheduled.clear()
    assert rs.kernel_enospc_hint(room, 0, "No space left on device") == ""
    assert rs.kernel_enospc_hint(room, 2, "", "command not found") == ""
    assert scheduled == []
