"""Server-wide capacity (internal design notes stage 12.1).

- ``[capacity]`` in legion.toml and ``LEGION_CAPACITY_*``: no cap by default,
  the variable wins, an unknown key or a bad value stops the startup;
- a workspace start must fit ``max_workspaces`` and ``memory_mb`` next to the
  owner's own limits; otherwise 503 ``server_busy`` and nothing is written;
- a personal MCP start must fit ``memory_mb``.

No DB required — db and collaborators are mocked, per
[[feedback_pytest_windows_env]] / [[feedback_no_tests_in_prod]].
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import host_capacity as hc
from app.core import workspace_lifecycle as wl
from app.core import workspace_quota as wq
from app.core.tools._workspace_wake import start_error_message
from app.models.workspace import Workspace

SELFHOSTED = {"plan_tier": "selfhosted", "max_running": 5, "mem_limit_mb": 2048,
              "rooms_memory_pool_mb": None, "room_max_memory_mb": None, "monthly_hours": None}


@pytest.fixture(autouse=True)
def _fresh_capacity():
    hc.reset_capacity()
    yield
    hc.reset_capacity()


def _use(capacity: hc.HostCapacity):
    return patch.object(hc, "get_capacity", return_value=capacity)


def _usage(workspaces: int, workspace_mb: int, mcp_mb: int = 0):
    return (
        patch.object(hc, "workspace_usage", new=AsyncMock(return_value=(workspaces, workspace_mb))),
        patch.object(hc, "personal_mcp_memory", new=AsyncMock(return_value=mcp_mb)),
    )


# ── Configuration ────────────────────────────────────────────────────────────

def test_no_file_and_no_env_means_no_cap():
    capacity = hc.load(environ={})
    assert capacity == hc.HostCapacity(None, None)
    assert capacity.unlimited


def test_file_then_env(tmp_path):
    config = tmp_path / "legion.toml"
    config.write_text('[capacity]\nmax_workspaces = 3\nmemory_mb = "unlimited"\n', encoding="utf-8")
    assert hc.load(path=str(config), environ={}) == hc.HostCapacity(3, None)
    env = {"LEGION_CAPACITY_MEMORY_MB": "3072", "LEGION_CAPACITY_MAX_WORKSPACES": "unlimited"}
    assert hc.load(path=str(config), environ=env) == hc.HostCapacity(None, 3072)
    # An empty variable (compose passes unset ones as "") leaves the file's value.
    env = {"LEGION_CAPACITY_MAX_WORKSPACES": "", "LEGION_CAPACITY_MEMORY_MB": " "}
    assert hc.load(path=str(config), environ=env) == hc.HostCapacity(3, None)


@pytest.mark.parametrize("body,needle", [
    ('[capacity]\nmax_workspace = 3\n', "Did you mean 'max_workspaces'"),
    ('[capacity]\nmemory_mb = -1\n', "must not be negative"),
    ('[capacity]\nmemory_mb = "lots"\n', "whole number or 'unlimited'"),
    ('[capacity]\nmemory_mb = true\n', "whole number or 'unlimited'"),
    ('capacity = 5\n', "[capacity] must be a table"),
])
def test_bad_file_stops_startup(tmp_path, body, needle):
    config = tmp_path / "legion.toml"
    config.write_text(body, encoding="utf-8")
    with pytest.raises(hc.CapacityConfigError) as exc:
        hc.load(path=str(config), environ={})
    assert needle in str(exc.value)


def test_unknown_env_key_stops_startup():
    with pytest.raises(hc.CapacityConfigError):
        hc.load(environ={"LEGION_CAPACITY_MEMORY": "1024"})


def test_the_example_config_loads():
    from pathlib import Path

    example = Path(__file__).resolve().parents[2] / "legion.example.toml"
    assert hc.load(path=str(example), environ={}).unlimited


# ── Workspace start ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("capacity,workspaces,workspace_mb,mcp_mb,requested,kind", [
    (hc.HostCapacity(3, None), 2, 0, 0, 1024, None),
    (hc.HostCapacity(3, None), 3, 0, 0, 1024, "workspaces"),
    (hc.HostCapacity(None, 3072), 2, 2048, 0, 1024, None),
    (hc.HostCapacity(None, 3072), 2, 2048, 512, 1024, "memory"),  # personal MCPs count too
    (hc.HostCapacity(None, 3072), 0, 0, 0, 4096, "memory"),
    (hc.HostCapacity(None, None), 99, 999_999, 999_999, 8192, None),
])
async def test_workspace_start_must_fit(capacity, workspaces, workspace_mb, mcp_mb, requested, kind):
    u1, u2 = _usage(workspaces, workspace_mb, mcp_mb)
    with _use(capacity), u1, u2:
        if kind is None:
            await hc.check_workspace_start(AsyncMock(), uuid.uuid4(), requested)
        else:
            with pytest.raises(hc.HostCapacityExceeded) as exc:
                await hc.check_workspace_start(AsyncMock(), uuid.uuid4(), requested)
            assert exc.value.kind == kind


@pytest.mark.asyncio
async def test_the_starting_room_is_not_counted_twice():
    room_id = uuid.uuid4()
    usage = AsyncMock(return_value=(0, 0))
    with _use(hc.HostCapacity(1, None)), patch.object(hc, "workspace_usage", new=usage):
        await hc.check_workspace_start(AsyncMock(), room_id, 1024)
    assert usage.await_args.kwargs["exclude_room_id"] == room_id


def _quota_patches(limits):
    return (
        patch.object(wq, "get_effective_limits", new=AsyncMock(return_value=limits)),
        patch.object(wq, "reserved_workspace_slots", new=AsyncMock(return_value=0)),
        patch.object(wq, "reserved_workspace_memory_mb", new=AsyncMock(return_value=0)),
    )


@pytest.mark.asyncio
async def test_reserve_and_run_refuses_a_full_server_without_writing():
    operation = AsyncMock(return_value="started")
    p1, p2, p3 = _quota_patches(SELFHOSTED)
    u1, u2 = _usage(2, 2048)
    with p1, p2, p3, u1, u2, _use(hc.HostCapacity(2, None)):
        with pytest.raises(hc.HostCapacityExceeded):
            await wq.quota_manager.reserve_and_run(
                uuid.uuid4(), uuid.uuid4(), AsyncMock(), operation, mem_mb=1024,
            )
    operation.assert_not_awaited()


@pytest.mark.asyncio
async def test_reserve_and_run_writes_under_the_host_lock():
    seen = {}

    async def operation():
        seen["locked"] = hc.host_lock().locked()
        return "started"

    p1, p2, p3 = _quota_patches(SELFHOSTED)
    u1, u2 = _usage(0, 0)
    with p1, p2, p3, u1, u2, _use(hc.HostCapacity(2, 4096)):
        assert await wq.quota_manager.reserve_and_run(
            uuid.uuid4(), uuid.uuid4(), AsyncMock(), operation, mem_mb=1024,
        ) == "started"
    assert seen["locked"] is True
    assert not hc.host_lock().locked()


def _db_returning(ws) -> AsyncMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = ws
    result.scalar_one.return_value = ws
    result.scalars.return_value.all.return_value = []
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    db.add = MagicMock()
    return db


@pytest.mark.asyncio
async def test_open_on_a_full_server_answers_503_server_busy():
    ws = Workspace(room_id=uuid.uuid4(), status="stopped")
    ws.id = uuid.uuid4()
    reserve = AsyncMock(side_effect=hc.HostCapacityExceeded("memory", 3072, 3072, 1024))
    patches = [
        patch.object(wl, "_restart_for_image_mismatch", new=AsyncMock()),
        patch.object(wl, "_get_or_create_workspace", new=AsyncMock(return_value=ws)),
        patch("app.core.workspace_quota.get_effective_limits", new=AsyncMock(return_value=SELFHOSTED)),
        patch("app.core.room_storage.storage_status",
              new=AsyncMock(return_value=SimpleNamespace(full=False))),
        patch("app.core.workspace_quota.quota_manager.reserve_and_run", new=reserve),
        patch.object(wl, "_broadcast_workspace_status", new=AsyncMock()),
        patch.object(wl, "_run_start_operation", new=AsyncMock()),
        patch.object(wl, "_track_start_task"),
    ]
    for p in patches:
        p.start()
    try:
        with pytest.raises(wl.WorkspaceStartError) as exc:
            await wl.ensure_workspace_started(ws.room_id, _db_returning(ws), quota_owner_id=uuid.uuid4())
    finally:
        for p in patches:
            p.stop()
    assert exc.value.status_code == 503
    assert exc.value.error_code == hc.SERVER_BUSY
    assert exc.value.message == hc.SERVER_BUSY_TEXT
    assert ws.status == "stopped"


def test_agent_is_told_the_server_is_full_not_the_account():
    error = wl.WorkspaceStartError(hc.SERVER_BUSY, hc.SERVER_BUSY_TEXT, status_code=503)
    text = start_error_message(error)
    assert "(server_busy)" in text
    assert "whole server is full" in text
    assert "Do not retry in a loop" in text


# ── Personal MCP start ───────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("capacity,workspace_mb,mcp_mb,requested,fits", [
    (hc.HostCapacity(None, 3072), 2048, 512, 512, True),
    (hc.HostCapacity(None, 3072), 2048, 512, 1024, False),
    # max_workspaces is about rooms only.
    (hc.HostCapacity(0, None), 999_999, 0, 1024, True),
])
async def test_personal_mcp_start_must_fit_memory(capacity, workspace_mb, mcp_mb, requested, fits):
    item_id = uuid.uuid4()
    mcp = AsyncMock(return_value=mcp_mb)
    with _use(capacity), \
            patch.object(hc, "workspace_usage", new=AsyncMock(return_value=(5, workspace_mb))), \
            patch.object(hc, "personal_mcp_memory", new=mcp):
        if fits:
            await hc.check_personal_mcp_start(AsyncMock(), item_id, requested)
        else:
            with pytest.raises(hc.HostCapacityExceeded):
                await hc.check_personal_mcp_start(AsyncMock(), item_id, requested)
    if capacity.memory_mb is not None:
        assert mcp.await_args.kwargs["exclude_item_id"] == item_id


@pytest.mark.asyncio
async def test_legacy_personal_servers_count_their_memory():
    managed = MagicMock()
    managed.scalars.return_value.all.return_value = [512, None]
    legacy = MagicMock()
    skip_id, keep_id = uuid.uuid4(), uuid.uuid4()
    legacy.all.return_value = [
        (keep_id, {"runtime": {"status": "running", "mem_limit_mb": 1024}}),
        (uuid.uuid4(), {"mem_limit_mb": 768, "runtime": {"status": "starting"}}),
        (uuid.uuid4(), {"runtime": {"status": "stopped", "mem_limit_mb": 4096}}),
        (skip_id, {"runtime": {"status": "running", "mem_limit_mb": 2048}}),
        (uuid.uuid4(), None),
    ]
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[managed, legacy])
    assert await hc.personal_mcp_memory(db, exclude_item_id=skip_id) == 512 + 1024 + 768
