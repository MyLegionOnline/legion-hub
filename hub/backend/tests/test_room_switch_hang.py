"""A room switch must not hang, and a stop it abandons must not be undone.

Incident (2026-10-01): switching "blog" → "Ranobe" with the workspace quota
full, the user confirmed the stop, yet "blog" kept running and "Ranobe" never
started; only reloading the page and pressing Stop by hand got out of it.

- WS sends had no timeout, and the switch awaits status broadcasts while
  holding its lock: one client that stopped reading hung the whole switch.
- A second switch between the same rooms queued behind the hung one.
- A stop abandoned in "draining"/"stopping" was revived to "running" by the
  next room open (the container was still alive), and stop_workspace itself
  returned early on "stopping", so even the Stop button could not finish it.

No DB required — db.execute and collaborators are mocked, per
[[feedback_pytest_windows_env]] / [[feedback_no_tests_in_prod]].
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import workspace_lifecycle as wl
from app.models.workspace import Workspace
from app.websockets import manager as manager_module
from app.websockets.manager import ConnectionManager


class _HangingSocket:
    """A client that stopped reading: every send waits forever."""

    def __init__(self):
        self.closed = False

    async def send_text(self, payload):
        await asyncio.Event().wait()

    async def send_json(self, message):
        await asyncio.Event().wait()

    async def close(self):
        self.closed = True


class _OkSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(payload)

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_broadcast_drops_a_socket_that_stopped_reading():
    mgr = ConnectionManager()
    room_id = uuid.uuid4()
    deaf, alive = _HangingSocket(), _OkSocket()
    mgr.active_connections[room_id] = {uuid.uuid4(): [deaf], uuid.uuid4(): [alive]}

    with patch.object(manager_module, "SEND_TIMEOUT_SEC", 0.05):
        await asyncio.wait_for(mgr.broadcast({"type": "x"}, room_id), timeout=2)

    assert alive.sent, "a live client still gets the frame"
    remaining = [s for sockets in mgr.active_connections[room_id].values() for s in sockets]
    assert deaf not in remaining
    assert deaf.closed


@pytest.mark.asyncio
async def test_user_channel_drops_a_socket_that_stopped_reading():
    mgr = ConnectionManager()
    user_id = uuid.uuid4()
    deaf = _HangingSocket()
    mgr.connect_user_events(deaf, user_id)

    with patch.object(manager_module, "SEND_TIMEOUT_SEC", 0.05):
        await asyncio.wait_for(mgr.send_to_users({"type": "x"}, [user_id]), timeout=2)

    assert user_id not in mgr.user_event_sockets
    assert deaf.closed


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
async def test_stop_finishes_an_abandoned_stopping_workspace():
    room_id = uuid.uuid4()
    ws = Workspace(room_id=room_id, status="stopping", container_id="c1")
    ws.id = uuid.uuid4()
    db = _db_returning(ws)

    with patch("app.core.workspaced_client.stop_workspace", new=AsyncMock()) as wsd_stop, \
         patch.object(wl, "_close_workspace_usage", new=AsyncMock()), \
         patch.object(wl, "_broadcast_workspace_status", new=AsyncMock()):
        payload = await wl.stop_workspace(room_id, db, reason="user")

    wsd_stop.assert_awaited_once_with(str(room_id))
    assert ws.status == "stopped"
    assert payload["status"] == "stopped"


@pytest.mark.asyncio
@pytest.mark.parametrize("stale_status", ["draining", "stopping"])
async def test_open_finishes_an_abandoned_stop_instead_of_reviving_it(stale_status):
    room_id = uuid.uuid4()
    owner_id = uuid.uuid4()
    ws = Workspace(room_id=room_id, status=stale_status, container_id="c1")
    ws.id = uuid.uuid4()
    db = _db_returning(ws)

    async def finish_stop(*args, **kwargs):
        ws.status = "stopped"
        return {"status": "stopped"}

    starting = {"room_id": room_id, "status": "starting"}
    with patch.object(wl, "_restart_for_image_mismatch", new=AsyncMock()), \
         patch.object(wl, "_get_or_create_workspace", new=AsyncMock(return_value=ws)), \
         patch.object(wl, "_stop_workspace_locked", new=AsyncMock(side_effect=finish_stop)) as stop_locked, \
         patch(
             "app.core.workspaced_client.get_workspace_status",
             new=AsyncMock(return_value={"status": "running", "container_id": "c1"}),
         ) as live_status, \
         patch(
             "app.core.workspace_quota.get_effective_limits",
             new=AsyncMock(return_value={"mem_limit_mb": 1024, "monthly_hours": None}),
         ), \
         patch(
             "app.core.workspace_quota.quota_manager.reserve_and_run",
             new=AsyncMock(return_value=starting),
         ), \
         patch.object(wl, "_broadcast_workspace_status", new=AsyncMock()), \
         patch.object(wl, "_run_start_operation", new=AsyncMock()), \
         patch.object(wl, "_track_start_task"):
        result = await wl.ensure_workspace_started(room_id, db, quota_owner_id=owner_id)

    stop_locked.assert_awaited_once()
    assert stop_locked.await_args.kwargs["force"] is True
    live_status.assert_not_awaited()  # no reconcile back to "running"
    assert result == starting


@pytest.mark.asyncio
async def test_second_switch_between_same_rooms_answers_busy_instead_of_queueing():
    a, b = uuid.uuid4(), uuid.uuid4()
    db = AsyncMock()

    async with wl._switch_lock(a, b):
        payload = await asyncio.wait_for(
            wl.request_switch(b, a, uuid.uuid4(), db, confirm_stop=True),
            timeout=1,
        )

    assert payload["status"] == "busy"
    db.get.assert_not_awaited()
