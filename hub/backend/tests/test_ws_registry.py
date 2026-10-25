"""The room socket registry must not lose a live socket to a dead one.

Symptom (2026-10-13): after a page reload the chat showed "Connected" but new
events never appeared until the room was reopened. The registry held ONE socket
per user per room and the last socket to *register* won. Registration follows
several awaited DB checks, so it is not connection order: React StrictMode's
first socket (already closed by the client) could register after the live one,
replace it, and then deregister itself — leaving nothing. A second tab or window
of the same user displaced the first the same way.

No DB, no network — plain fake sockets.
"""

from __future__ import annotations

import uuid

import pytest

from app.websockets.manager import ConnectionManager


class _Socket:
    def __init__(self):
        self.sent = []
        self.closed = False

    async def send_text(self, payload):
        if self.closed:
            raise RuntimeError("closed")
        self.sent.append(payload)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_a_dead_socket_registering_last_does_not_hide_the_live_one():
    mgr = ConnectionManager()
    room, user = uuid.uuid4(), uuid.uuid4()
    first, live = _Socket(), _Socket()   # `first` was opened first but is already abandoned

    await mgr.connect(live, room, user)
    await mgr.connect(first, room, user)   # registers after `live`, as in the race
    first.closed = True
    mgr.disconnect(room, user, False, first)

    await mgr.broadcast({"type": "x"}, room)

    assert live.sent, "the live socket must still receive the frame"


@pytest.mark.asyncio
async def test_two_sockets_of_one_user_in_a_room_both_receive():
    mgr = ConnectionManager()
    room, user = uuid.uuid4(), uuid.uuid4()
    tab_a, tab_b = _Socket(), _Socket()

    await mgr.connect(tab_a, room, user)
    await mgr.connect(tab_b, room, user)
    await mgr.broadcast({"type": "x"}, room)

    assert tab_a.sent and tab_b.sent


@pytest.mark.asyncio
async def test_late_disconnect_of_an_old_socket_keeps_the_newer_one_registered():
    mgr = ConnectionManager()
    room, user = uuid.uuid4(), uuid.uuid4()
    old, new = _Socket(), _Socket()

    await mgr.connect(old, room, user)
    await mgr.connect(new, room, user)
    mgr.disconnect(room, user, False, old)

    assert mgr.user_websockets[user] is new       # send_to_client still reaches the newest
    await mgr.broadcast({"type": "x"}, room)
    assert new.sent and not old.sent


@pytest.mark.asyncio
async def test_a_failed_send_drops_only_that_socket():
    mgr = ConnectionManager()
    room, user = uuid.uuid4(), uuid.uuid4()
    broken, healthy = _Socket(), _Socket()
    broken.closed = True

    await mgr.connect(broken, room, user)
    await mgr.connect(healthy, room, user)
    await mgr.broadcast({"type": "x"}, room)

    assert healthy.sent
    assert mgr.active_connections[room][user] == [healthy]


@pytest.mark.asyncio
async def test_the_last_disconnect_leaves_no_room_entry_behind():
    mgr = ConnectionManager()
    room, user = uuid.uuid4(), uuid.uuid4()
    sock = _Socket()

    await mgr.connect(sock, room, user)
    mgr.disconnect(room, user, False, sock)

    assert room not in mgr.active_connections
    assert user not in mgr.user_websockets
