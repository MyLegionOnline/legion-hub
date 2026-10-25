"""A revoked connector loses its open socket.

Revoking a connector — by its owner, or with the owner's data on erasure —
used to leave a live WebSocket running: the token was refused only on the
next connect, and frames kept arriving meanwhile. Now the revoke closes the
socket, and the socket loop re-checks the connector on every frame.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.api.v1 import connector_ws
from app.websockets.manager import ConnectionManager


class FakeSocket:
    def __init__(self, frames=()):
        self.frames = list(frames)
        self.closed_with = None
        self.sent = []

    async def accept(self):
        pass

    async def receive_text(self):
        if not self.frames:
            raise AssertionError("the loop read past the revoked frame")
        return self.frames.pop(0)

    async def send_json(self, frame):
        self.sent.append(frame)

    async def close(self, code=1000):
        self.closed_with = code


class FakeDb:
    def add(self, _row):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass


@pytest.mark.asyncio
async def test_close_connector_closes_and_forgets_the_socket():
    manager = ConnectionManager()
    connector_id = uuid4()
    ws = FakeSocket()
    manager.connect_connector(ws, connector_id)

    assert await manager.close_connector(connector_id) is True
    assert ws.closed_with == 1008
    assert connector_id not in manager.connector_websockets
    # Nothing open: harmless.
    assert await manager.close_connector(connector_id) is False


@pytest.mark.asyncio
async def test_a_frame_after_the_revoke_closes_the_socket_unhandled(monkeypatch):
    connector = SimpleNamespace(id=uuid4(), version="0.1.0", capabilities={"protocol_version": "2"},
                                status="offline", last_seen_at=None)
    monkeypatch.setattr(connector_ws, "_authenticate", AsyncMock(return_value=connector))
    monkeypatch.setattr(connector_ws, "_still_active", AsyncMock(return_value=False))
    monkeypatch.setattr(connector_ws.connector_broker, "deliver", AsyncMock(return_value=0))
    handled = AsyncMock()
    monkeypatch.setattr(connector_ws.connector_reverse, "handle_inbound_frame", handled)

    ws = FakeSocket(['{"type": "result", "session_id": null, "payload": {}}'])
    await connector_ws.connector_websocket_endpoint(ws, token="t", db=FakeDb())

    assert ws.closed_with == 1008
    handled.assert_not_called()
    assert connector.id not in connector_ws.manager.connector_websockets
