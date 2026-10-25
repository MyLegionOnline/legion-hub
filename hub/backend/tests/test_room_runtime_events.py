"""Per-user event channel — internal design notes, stages 1-2."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.websockets.manager import ConnectionManager


class _FakeSocket:
    def __init__(self, fail: bool = False):
        self.sent: list[dict] = []
        self.fail = fail
        self.closed = False

    async def send_text(self, payload: str) -> None:
        if self.fail:
            raise RuntimeError("socket gone")
        self.sent.append(json.loads(payload))

    async def close(self) -> None:
        self.closed = True


# ── Stage 1: registry ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_every_tab_of_the_user_hears_the_event_and_nobody_else_does():
    mgr = ConnectionManager()
    alice, bob = uuid.uuid4(), uuid.uuid4()
    tab1, tab2, bob_tab = _FakeSocket(), _FakeSocket(), _FakeSocket()
    mgr.connect_user_events(tab1, alice)
    mgr.connect_user_events(tab2, alice)
    mgr.connect_user_events(bob_tab, bob)

    await mgr.send_to_users({"type": "room_runtime"}, [alice])

    assert tab1.sent == [{"type": "room_runtime"}]
    assert tab2.sent == [{"type": "room_runtime"}]
    assert bob_tab.sent == []


def test_closing_the_last_tab_leaves_nothing_in_the_registry():
    mgr = ConnectionManager()
    alice = uuid.uuid4()
    tab1, tab2 = _FakeSocket(), _FakeSocket()
    mgr.connect_user_events(tab1, alice)
    mgr.connect_user_events(tab2, alice)

    mgr.disconnect_user_events(tab1, alice)
    assert mgr.user_event_sockets == {alice: {tab2}}
    mgr.disconnect_user_events(tab2, alice)
    assert mgr.user_event_sockets == {}
    assert not mgr.has_user_event_listeners()
    # A late duplicate disconnect is harmless.
    mgr.disconnect_user_events(tab2, alice)


@pytest.mark.asyncio
async def test_a_dead_socket_is_dropped_and_closed_so_the_client_reconnects():
    mgr = ConnectionManager()
    alice = uuid.uuid4()
    dead, alive = _FakeSocket(fail=True), _FakeSocket()
    mgr.connect_user_events(dead, alice)
    mgr.connect_user_events(alive, alice)

    await mgr.send_to_users({"type": "room_runtime"}, [alice])

    assert dead.closed
    assert mgr.user_event_sockets == {alice: {alive}}
    assert alive.sent == [{"type": "room_runtime"}]


def _app_with_session(db):
    from app.api.v1 import users
    from app.db.session import get_session

    app = FastAPI()
    app.include_router(users.router, prefix="/api/v1/users")

    async def _override():
        yield db

    app.dependency_overrides[get_session] = _override
    return app


def test_socket_registers_releases_the_db_connection_and_unregisters_on_close():
    from app.api.v1 import users

    user_id = uuid.uuid4()
    db = MagicMock()
    db.rollback = AsyncMock()
    mgr = ConnectionManager()
    app = _app_with_session(db)

    with patch.object(users, "manager", mgr), \
         patch.object(users, "_user_id_from_token", AsyncMock(return_value=user_id)):
        with TestClient(app).websocket_connect("/api/v1/users/me/events?token=t") as ws:
            ws.send_text("ping")  # round-trip so the handler is surely registered
            assert user_id in mgr.user_event_sockets
            db.rollback.assert_awaited()
        # The handler's finally runs on the server side of the closed socket.
        assert mgr.user_event_sockets == {}


def test_bad_token_is_refused():
    from app.api.v1 import users

    mgr = ConnectionManager()
    app = _app_with_session(MagicMock(rollback=AsyncMock()))
    with patch.object(users, "manager", mgr), \
         patch.object(users, "_user_id_from_token", AsyncMock(return_value=None)):
        with TestClient(app).websocket_connect("/api/v1/users/me/events?token=bad") as ws:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws.receive_text()
    assert exc.value.code == 1008
    assert mgr.user_event_sockets == {}


# ── Stage 2: publishing ──────────────────────────────────────────────────────

def _fake_factory(member_ids):
    result = MagicMock()
    result.scalars.return_value.all.return_value = member_ids
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)

    class _Ctx:
        async def __aenter__(self):
            return db

        async def __aexit__(self, *exc):
            return False

    return MagicMock(return_value=_Ctx()), db


@pytest.mark.asyncio
async def test_members_get_only_room_id_and_runtime_status():
    from app.core import workspace_lifecycle as wl

    room_id, member, outsider = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    mgr = ConnectionManager()
    member_tab, outsider_tab = _FakeSocket(), _FakeSocket()
    mgr.connect_user_events(member_tab, member)
    mgr.connect_user_events(outsider_tab, outsider)
    factory, _db = _fake_factory([member])

    with patch("app.websockets.manager.manager", mgr), \
         patch.object(wl, "async_sessionmaker", return_value=factory):
        await wl.publish_room_runtime(room_id, "stopped")

    assert member_tab.sent == [
        {"type": "room_runtime", "room_id": str(room_id), "runtime_status": "stopped"}
    ]
    assert outsider_tab.sent == []


@pytest.mark.asyncio
async def test_no_listeners_means_no_membership_lookup():
    from app.core import workspace_lifecycle as wl

    factory, db = _fake_factory([uuid.uuid4()])
    with patch("app.websockets.manager.manager", ConnectionManager()), \
         patch.object(wl, "async_sessionmaker", return_value=factory):
        await wl.publish_room_runtime(uuid.uuid4(), "running")
    db.execute.assert_not_called()


@pytest.mark.asyncio
async def test_status_transitions_publish_but_semantic_events_do_not():
    from app.core import workspace_lifecycle as wl

    ws = SimpleNamespace(
        id=uuid.uuid4(), room_id=uuid.uuid4(), status="running", workspace_generation=1,
    )
    with patch.object(wl, "publish_room_runtime", AsyncMock()) as publish:
        await wl._broadcast_workspace_status(ws, previous_status="starting", operation_id=None)
        publish.assert_awaited_once_with(ws.room_id, "running")

        publish.reset_mock()
        await wl._broadcast_workspace_status(
            ws, previous_status="running", operation_id=None,
            event_type="workspace.snapshot.created",
        )
        publish.assert_not_awaited()
