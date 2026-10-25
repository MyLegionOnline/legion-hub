"""Rooms memory pool (internal design notes §3a, T2a).

- a room's limit is the plan's mem_limit_mb, or its own setting up to
  room_max_memory_mb in 256 MiB steps;
- a start must fit the owner's rooms_memory_pool_mb next to max_running,
  else 402 ``memory_pool_full`` with "Memory pool is full: …";
- a new setting reaches a running room at once (docker update) when it fits
  the pool and the room's resident memory, otherwise at the next start;
- /me/usage reports the rooms' resident memory ("Currently used").

No DB required — db and collaborators are mocked, per
[[feedback_pytest_windows_env]] / [[feedback_no_tests_in_prod]].
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import room_memory as rm
from app.core import storage_projects as sp
from app.core import workspace_lifecycle as wl
from app.core import workspace_quota as wq
from app.models.workspace import Workspace

HOBBY = {"plan_tier": "hobby", "max_running": 1, "mem_limit_mb": 1024,
         "rooms_memory_pool_mb": 1024, "room_max_memory_mb": 1024, "monthly_hours": None}
PRO = {"plan_tier": "pro", "max_running": 3, "mem_limit_mb": 1024,
       "rooms_memory_pool_mb": 3072, "room_max_memory_mb": 2048, "monthly_hours": None}
SELFHOSTED = {"plan_tier": "selfhosted", "max_running": 5, "mem_limit_mb": 2048,
              "rooms_memory_pool_mb": None, "room_max_memory_mb": None, "monthly_hours": None}


# ── Limits ───────────────────────────────────────────────────────────────────

def test_room_limit_defaults_to_plan_and_is_capped_by_room_ceiling():
    assert rm.effective_mb(PRO, None) == 1024
    assert rm.effective_mb(PRO, 2048) == 2048
    # A Pro setting after a downgrade to Hobby: the Hobby ceiling wins.
    assert rm.effective_mb(HOBBY, 2048) == 1024
    # An explicit start request is clamped the same way.
    assert rm.effective_mb(PRO, None, 4096) == 2048
    # Selfhosted has no room ceiling below workspaced's own.
    assert rm.max_mb(SELFHOSTED) == rm.HARD_MAX_MB
    assert rm.effective_mb(SELFHOSTED, 4096) == 4096


def test_setting_must_fit_range_and_step():
    assert rm.validate_setting(PRO, None) is None
    assert rm.validate_setting(PRO, 1536) == 1536
    with pytest.raises(rm.RoomMemoryError) as too_big:
        rm.validate_setting(PRO, 2304)
    assert too_big.value.error_code == "memory_out_of_range"
    with pytest.raises(rm.RoomMemoryError):
        rm.validate_setting(HOBBY, 128)
    with pytest.raises(rm.RoomMemoryError) as step:
        rm.validate_setting(PRO, 1000)
    assert step.value.error_code == "memory_step"


def test_pool_full_text():
    assert rm.format_mb(2048) == "2 GiB"
    assert rm.format_mb(1536) == "1.5 GiB"
    assert rm.format_mb(512) == "512 MiB"
    assert rm.pool_full_text(3072, 2048) == (
        "Memory pool is full: 2 GiB of 3 GiB running. "
        "Stop a room or lower its limit."
    )


# ── Pool check at start ──────────────────────────────────────────────────────

def _quota_patches(limits, slots, memory):
    return (
        patch.object(wq, "get_effective_limits", new=AsyncMock(return_value=limits)),
        patch.object(wq, "reserved_workspace_slots", new=AsyncMock(return_value=slots)),
        patch.object(wq, "reserved_workspace_memory_mb", new=AsyncMock(return_value=memory)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("allocated,requested,fits", [
    (2048, 1024, True),    # 2 GiB + 1 GiB
    (0, 2048, True),       # one room at 2 GiB
    (2048, 2048, False),   # 2 + 2 > 3
    (1024 * 2, 1280, False),
])
async def test_start_must_fit_the_pool(allocated, requested, fits):
    operation = AsyncMock(return_value="started")
    p1, p2, p3 = _quota_patches(PRO, 1, allocated)
    with p1, p2, p3:
        if fits:
            assert await wq.quota_manager.reserve_and_run(
                uuid.uuid4(), uuid.uuid4(), AsyncMock(), operation, mem_mb=requested,
            ) == "started"
        else:
            with pytest.raises(wq.WorkspaceMemoryPoolExceeded) as exc:
                await wq.quota_manager.reserve_and_run(
                    uuid.uuid4(), uuid.uuid4(), AsyncMock(), operation, mem_mb=requested,
                )
            assert exc.value.pool_mb == 3072 and exc.value.allocated_mb == allocated
            operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_max_running_is_still_checked_first():
    p1, p2, p3 = _quota_patches(PRO, 3, 0)
    with p1, p2, p3, pytest.raises(wq.WorkspaceQuotaExceeded):
        await wq.quota_manager.reserve_and_run(
            uuid.uuid4(), uuid.uuid4(), AsyncMock(), AsyncMock(), mem_mb=256,
        )


@pytest.mark.asyncio
async def test_no_pool_means_no_memory_check():
    p1, p2, p3 = _quota_patches(SELFHOSTED, 0, 999_999)
    with p1, p2, p3:
        assert await wq.quota_manager.reserve_and_run(
            uuid.uuid4(), uuid.uuid4(), AsyncMock(), AsyncMock(return_value=1), mem_mb=8192,
        ) == 1


def _db_returning(ws) -> AsyncMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = ws
    result.scalar_one.return_value = ws
    result.scalars.return_value.all.return_value = []
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    db.add = MagicMock()
    return db


def _start_patches(
    ws,
    limits,
    reserve,
    *,
    start_operation=None,
    storage_project_for_owner=None,
):
    return [
        patch.object(wl, "_restart_for_image_mismatch", new=AsyncMock()),
        patch.object(wl, "_get_or_create_workspace", new=AsyncMock(return_value=ws)),
        patch("app.core.workspace_quota.get_effective_limits", new=AsyncMock(return_value=limits)),
        patch("app.core.room_storage.storage_status",
              new=AsyncMock(return_value=SimpleNamespace(full=False))),
        # These tests exercise memory admission.  Keep the new T9 transport
        # separate from their mock-only database, which deliberately has no
        # UserQuota/Room rows behind it.
        patch(
            "app.core.storage_projects.project_for_owner",
            new=storage_project_for_owner or AsyncMock(
                return_value=sp.StorageProject(10042, 3072, [str(ws.room_id)])
            ),
        ),
        patch("app.core.workspace_quota.quota_manager.reserve_and_run", new=reserve),
        patch.object(wl, "_broadcast_workspace_status", new=AsyncMock()),
        patch.object(wl, "_run_start_operation", new=start_operation or AsyncMock()),
        patch.object(wl, "_track_start_task"),
    ]


@pytest.mark.asyncio
async def test_open_with_a_full_pool_answers_402():
    ws = Workspace(room_id=uuid.uuid4(), status="stopped", mem_setting_mb=2048)
    ws.id = uuid.uuid4()
    owner = uuid.uuid4()
    reserve = AsyncMock(side_effect=wq.WorkspaceMemoryPoolExceeded(owner, 3072, 2048, 2048))
    patches = _start_patches(ws, PRO, reserve)
    for p in patches:
        p.start()
    try:
        with pytest.raises(wl.WorkspaceStartError) as exc:
            await wl.ensure_workspace_started(ws.room_id, _db_returning(ws), quota_owner_id=owner)
    finally:
        for p in patches:
            p.stop()
    assert exc.value.status_code == 402
    assert exc.value.error_code == "memory_pool_full"
    assert exc.value.message.startswith("Memory pool is full: 2 GiB of 3 GiB running.")
    assert reserve.await_args.kwargs["mem_mb"] == 2048


@pytest.mark.asyncio
async def test_start_records_the_room_limit_before_the_container_runs():
    """The starting row carries its limit, so a concurrent start counts it."""
    ws = Workspace(room_id=uuid.uuid4(), status="stopped", mem_setting_mb=1536, mem_limit_mb=1024)
    ws.id = uuid.uuid4()

    async def run(owner_id, room_id, db, operation, *, mem_mb=None):
        return await operation()

    patches = _start_patches(ws, PRO, AsyncMock(side_effect=run))
    for p in patches:
        p.start()
    try:
        await wl.ensure_workspace_started(ws.room_id, _db_returning(ws), quota_owner_id=uuid.uuid4())
    finally:
        for p in patches:
            p.stop()
    assert ws.status == "starting"
    assert ws.mem_limit_mb == 1536


@pytest.mark.asyncio
async def test_plan_layer_limit_overrides_legacy_disk_request():
    ws = Workspace(room_id=uuid.uuid4(), status="stopped")
    ws.id = uuid.uuid4()
    start_operation = AsyncMock()

    async def run(_owner_id, _room_id, _db, operation, *, mem_mb=None):
        return await operation()

    patches = _start_patches(
        ws,
        PRO | {"container_layer_mb": 4096},
        AsyncMock(side_effect=run),
        start_operation=start_operation,
    )
    for p in patches:
        p.start()
    try:
        await wl.ensure_workspace_started(ws.room_id, _db_returning(ws), quota_owner_id=uuid.uuid4())
    finally:
        for p in patches:
            p.stop()

    assert start_operation.call_args.kwargs["storage_project_id"] == 10042
    assert start_operation.call_args.kwargs["storage_quota_mb"] == 3072
    assert start_operation.call_args.kwargs["storage_project_room_ids"] == [str(ws.room_id)]
    assert start_operation.call_args.kwargs["container_layer_mb"] == 4096
    # API callers still submit the former 2 GiB default, but the DB/API
    # contract must show the tariffed writable layer that Docker will apply.
    assert start_operation.call_args.kwargs["disk_limit_mb"] == 4096
    assert ws.disk_limit_mb == 4096


@pytest.mark.asyncio
async def test_storage_project_always_belongs_to_the_room_creator():
    ws = Workspace(room_id=uuid.uuid4(), status="stopped")
    ws.id = uuid.uuid4()
    creator, caller = uuid.uuid4(), uuid.uuid4()
    db = _db_returning(ws)
    db.get = AsyncMock(return_value=SimpleNamespace(creator_id=creator))
    project_for_owner = AsyncMock(
        return_value=sp.StorageProject(10042, 3072, [str(ws.room_id)])
    )

    async def run(_owner_id, _room_id, _db, operation, *, mem_mb=None):
        return await operation()

    patches = _start_patches(
        ws,
        PRO | {"container_layer_mb": 2048},
        AsyncMock(side_effect=run),
        storage_project_for_owner=project_for_owner,
    )
    for p in patches:
        p.start()
    try:
        await wl.ensure_workspace_started(ws.room_id, db, quota_owner_id=caller)
    finally:
        for p in patches:
            p.stop()

    assert project_for_owner.await_args.args[0] == creator


# ── Setting a running room's limit ───────────────────────────────────────────

def _setting_patches(ws, limits, *, others=0, resident=None, update=None):
    ctx = rm._Context(ws.room_id, uuid.uuid4(), limits, ws)
    lock = MagicMock()
    lock.__aenter__ = AsyncMock(return_value=None)
    lock.__aexit__ = AsyncMock(return_value=False)
    return [
        patch.object(rm, "_context", new=AsyncMock(return_value=ctx)),
        patch.object(wq, "quota_owner_lock", return_value=lock),
        patch.object(wq, "reserved_workspace_memory_mb", new=AsyncMock(return_value=others)),
        patch.object(rm, "_resident_mb", new=AsyncMock(return_value=resident)),
        patch("app.core.workspaced_client.update_workspace_memory", new=update or AsyncMock()),
    ]


async def _set(ws, limits, value, **kw):
    update = kw.pop("update", None) or AsyncMock(return_value={})
    patches = _setting_patches(ws, limits, update=update, **kw)
    for p in patches:
        p.start()
    try:
        result = await rm.set_room_memory(ws.room_id, value, AsyncMock(add=MagicMock()))
    finally:
        for p in patches:
            p.stop()
    return result, update


@pytest.mark.asyncio
async def test_raise_of_a_running_room_that_fits_the_pool_applies_at_once():
    ws = Workspace(room_id=uuid.uuid4(), status="running", mem_limit_mb=1024)
    result, update = await _set(ws, PRO, 2048, others=1024, resident=300.0)

    update.assert_awaited_once_with(str(ws.room_id), 2048)
    assert ws.mem_setting_mb == 2048 and ws.mem_limit_mb == 2048
    assert result["applied"] is True and result["pending"] is False


@pytest.mark.asyncio
async def test_raise_beyond_the_pool_waits_for_the_next_start():
    ws = Workspace(room_id=uuid.uuid4(), status="running", mem_limit_mb=1024)
    result, update = await _set(ws, PRO, 2048, others=2048, resident=300.0)

    update.assert_not_awaited()
    assert ws.mem_setting_mb == 2048 and ws.mem_limit_mb == 1024
    assert result["applied"] is False and result["pending"] is True
    assert "next room start" in result["message"]


@pytest.mark.asyncio
async def test_lowering_below_what_the_room_holds_waits_for_the_next_start():
    ws = Workspace(room_id=uuid.uuid4(), status="running", mem_limit_mb=2048)
    result, update = await _set(ws, PRO, 512, resident=900.0)

    update.assert_not_awaited()
    assert ws.mem_limit_mb == 2048 and result["applied"] is False


@pytest.mark.asyncio
async def test_lowering_above_what_the_room_holds_applies_at_once():
    ws = Workspace(room_id=uuid.uuid4(), status="running", mem_limit_mb=2048)
    result, update = await _set(ws, PRO, 1024, resident=400.0)

    update.assert_awaited_once()
    assert ws.mem_limit_mb == 1024 and result["applied"] is True


@pytest.mark.asyncio
async def test_failed_live_update_keeps_the_setting_for_the_next_start():
    ws = Workspace(room_id=uuid.uuid4(), status="running", mem_limit_mb=1024)
    result, _ = await _set(ws, PRO, 1536, update=AsyncMock(side_effect=RuntimeError("409")))

    assert ws.mem_setting_mb == 1536 and ws.mem_limit_mb == 1024
    assert result["applied"] is False and result["pending"] is True


@pytest.mark.asyncio
async def test_stopped_room_only_stores_the_setting():
    ws = Workspace(room_id=uuid.uuid4(), status="stopped", mem_limit_mb=1024)
    result, update = await _set(ws, PRO, None)

    update.assert_not_awaited()
    assert ws.mem_setting_mb is None
    assert result["applied"] is True and result["effective_mb"] == 1024


@pytest.mark.asyncio
async def test_expired_plan_cannot_change_the_limit():
    ws = Workspace(room_id=uuid.uuid4(), status="stopped")
    with pytest.raises(rm.RoomMemoryError) as exc:
        await _set(ws, {**HOBBY, "plan_tier": wq.EXPIRED_PLAN_TIER}, 512)
    assert exc.value.status_code == 402


# ── "Currently used" ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_usage_reports_resident_memory_of_running_rooms():
    a, b = uuid.uuid4(), uuid.uuid4()
    rows = MagicMock()
    rows.all.return_value = [(a, "blog", 2048, "running"), (b, "ranobe", 1024, "starting")]
    db = AsyncMock()
    db.execute = AsyncMock(return_value=rows)
    resident = AsyncMock(return_value={str(a): 640.0})

    with patch.object(rm, "resident_memory", new=resident):
        block = await wq._memory_allocation(uuid.uuid4(), db)

    resident.assert_awaited_once_with([a])  # a starting room has no container yet
    assert block["memory_allocated_mb"] == 3072
    assert block["memory_used_mb"] == 640.0
    assert block["memory_rooms"][0]["rss_mb"] == 640.0
    assert block["memory_rooms"][1]["rss_mb"] is None


@pytest.mark.asyncio
async def test_resident_memory_survives_an_unreachable_workspaced():
    a, b = uuid.uuid4(), uuid.uuid4()

    async def stats(room_id):
        if room_id == str(b):
            raise ConnectionError("down")
        return {"memory_rss_mb": 210.5}

    with patch("app.core.workspaced_client.get_workspace_stats", new=stats):
        result = await rm.resident_memory([a, b])

    assert result == {str(a): 210.5, str(b): None}
