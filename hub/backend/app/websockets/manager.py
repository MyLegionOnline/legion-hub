import asyncio
import json
import logging
from typing import Dict, Iterable, List, Set, Union
from uuid import UUID
from fastapi import WebSocket

logger = logging.getLogger(__name__)

# A client that stopped reading (a suspended tab, a half-open TCP connection
# after sleep or a network change) makes a send wait for the transport to
# drain forever. Every publisher awaits these sends — including the room
# switch, which holds its lock across them — so one deaf socket could hang the
# switch until the page was reloaded. Such a socket is treated as dead.
SEND_TIMEOUT_SEC = 5.0
CLOSE_TIMEOUT_SEC = 1.0


async def _send_text(connection: WebSocket, payload: str) -> None:
    await asyncio.wait_for(connection.send_text(payload), timeout=SEND_TIMEOUT_SEC)


async def _send_json(connection: WebSocket, message: dict) -> None:
    await asyncio.wait_for(connection.send_json(message), timeout=SEND_TIMEOUT_SEC)


async def _close_quietly(connection: WebSocket) -> None:
    try:
        await asyncio.wait_for(connection.close(), timeout=CLOSE_TIMEOUT_SEC)
    except Exception:
        pass


class ConnectionManager:
    def __init__(self):
        # active_connections: {room_id: {client_id: [websocket, ...]}}
        # A LIST per client, not one socket: the same user can legitimately have
        # several sockets in one room (a second tab or window, a reload whose old
        # socket the server has not noticed dying yet, React StrictMode's
        # double mount). With one slot per client the last socket to *register*
        # won — and registration order is not connection order, because it
        # follows several awaited DB checks — so the live page could be
        # replaced by a socket the client had already closed and then go deaf
        # while still showing "Connected".
        self.active_connections: Dict[UUID, Dict[UUID, List[WebSocket]]] = {}
        # user_websockets: {user_id: websocket}
        self.user_websockets: Dict[UUID, WebSocket] = {}
        # agent_websockets: {agent_id: websocket}
        self.agent_websockets: Dict[UUID, WebSocket] = {}
        # connector_websockets: {connector_id: websocket}
        # Stage 2. Deliberately separate from agent_websockets/active_connections:
        # a connector socket is not per-room (one socket serves every Room the
        # connector is granted into), so it doesn't fit the room-keyed registry.
        self.connector_websockets: Dict[UUID, WebSocket] = {}
        # user_event_sockets: {user_id: {websocket, ...}} — the per-user
        # channel (internal design notes). Not per-room, and a SET per
        # user: every open tab/browser of the same user holds its own socket
        # and all of them must hear the event. Deliberately separate from
        # user_websockets above, which is keyed by client_id and holds only
        # the latest room socket.
        self.user_event_sockets: Dict[UUID, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: UUID, client_id: UUID, is_agent: bool = False):
        sockets = self.active_connections.setdefault(room_id, {}).setdefault(client_id, [])
        if websocket not in sockets:
            sockets.append(websocket)

        if is_agent:
            self.agent_websockets[client_id] = websocket
        else:
            self.user_websockets[client_id] = websocket

    def disconnect(self, room_id: UUID, client_id: UUID, is_agent: bool = False, websocket: WebSocket = None):
        # Remove exactly this socket. A late disconnect of an old socket must
        # not take a newer one of the same client with it (it used to, for the
        # per-client slots below).
        room = self.active_connections.get(room_id)
        if room is not None and client_id in room:
            if websocket is None:
                del room[client_id]
            else:
                room[client_id] = [s for s in room[client_id] if s is not websocket]
                if not room[client_id]:
                    del room[client_id]
            if not room:
                del self.active_connections[room_id]

        # The "latest socket" slots used by send_to_client: drop one only if it
        # is this socket (or on an unconditional disconnect of the client).
        # Checked in both because a caller that only has the socket, such as
        # broadcast's stale cleanup, does not know whether the client is an agent.
        if websocket is None:
            (self.agent_websockets if is_agent else self.user_websockets).pop(client_id, None)
        else:
            for slots in (self.agent_websockets, self.user_websockets):
                if slots.get(client_id) is websocket:
                    del slots[client_id]

    async def broadcast(self, message: dict, room_id: UUID):
        if room_id not in self.active_connections:
            return

        # Serialise once, defensively. A non-JSON value in ``message`` would make
        # every per-client ``send_json`` raise, which used to silently drop every
        # socket in the room (the client never sees an ``onclose`` and goes deaf
        # until it reloads). Fall back to a text-only frame instead.
        try:
            payload = json.dumps(message)
        except (TypeError, ValueError):
            logger.warning(
                "manager.broadcast: message for room %s is not JSON-serialisable "
                "(event_type=%s) — sending a reduced frame",
                room_id, message.get("event_type") if isinstance(message, dict) else "?",
            )
            payload = json.dumps(message, default=str)

        stale = []
        for client_id, sockets in list(self.active_connections[room_id].items()):
            for connection in list(sockets):
                try:
                    await _send_text(connection, payload)
                except Exception:
                    stale.append((client_id, connection))
        for cid, conn in stale:
            self.disconnect(room_id, cid, websocket=conn)
            # Close so the client's onclose fires and it reconnects (and reloads
            # history). A removed-but-not-closed socket is the "deaf until F5" bug.
            await _close_quietly(conn)

    async def send_to_client(self, message: dict, client_id: UUID, is_agent: bool = False):
        try:
            if is_agent:
                if client_id in self.agent_websockets:
                    await _send_json(self.agent_websockets[client_id], message)
            else:
                if client_id in self.user_websockets:
                    await _send_json(self.user_websockets[client_id], message)
        except Exception:
            pass

    def connect_connector(self, websocket: WebSocket, connector_id: UUID):
        self.connector_websockets[connector_id] = websocket

    def disconnect_connector(self, connector_id: UUID, websocket: WebSocket = None):
        # Guard against a stale disconnect racing a newer reconnect's registration,
        # same pattern as disconnect()'s websocket-identity check above.
        if connector_id in self.connector_websockets:
            if websocket is None or self.connector_websockets[connector_id] is websocket:
                del self.connector_websockets[connector_id]

    async def close_connector(self, connector_id: UUID, code: int = 1008) -> bool:
        """Close a revoked or erased connector's live socket now, instead of
        letting it run until its next frame. Returns whether one was open."""
        ws = self.connector_websockets.pop(connector_id, None)
        if ws is None:
            return False
        try:
            await asyncio.wait_for(ws.close(code=code), timeout=CLOSE_TIMEOUT_SEC)
        except Exception:
            pass
        return True

    async def send_to_connector(self, message: dict, connector_id: UUID) -> bool:
        """Best-effort push to a live connector socket. Callers needing durable
        delivery go through core/connector_broker.py's outbox instead — this
        is only the "is a socket open right now" primitive it builds on."""
        ws = self.connector_websockets.get(connector_id)
        if ws is None:
            return False
        try:
            await _send_json(ws, message)
            return True
        except Exception:
            return False

    def connect_user_events(self, websocket: WebSocket, user_id: UUID) -> None:
        self.user_event_sockets.setdefault(user_id, set()).add(websocket)

    def disconnect_user_events(self, websocket: WebSocket, user_id: UUID) -> None:
        # Drop the user's key once their last socket is gone, so a closed tab
        # leaves nothing behind in the registry.
        sockets = self.user_event_sockets.get(user_id)
        if sockets is None:
            return
        sockets.discard(websocket)
        if not sockets:
            del self.user_event_sockets[user_id]

    def has_user_event_listeners(self) -> bool:
        """Cheap pre-check so publishers can skip the membership lookup when
        nobody has the user channel open."""
        return bool(self.user_event_sockets)

    async def send_to_users(self, message: dict, user_ids: Iterable[UUID]) -> None:
        """Best-effort push to every open user-channel socket of ``user_ids``.

        Callers resolve the recipients from room membership themselves, so the
        payload must carry nothing that needs a finer permission check. A
        socket that fails to send is dropped and closed, same as in
        ``broadcast``, so the client reconnects instead of going deaf.
        """
        try:
            payload = json.dumps(message)
        except (TypeError, ValueError):
            payload = json.dumps(message, default=str)

        for user_id in set(user_ids):
            for connection in list(self.user_event_sockets.get(user_id, ())):
                try:
                    await _send_text(connection, payload)
                except Exception:
                    self.disconnect_user_events(connection, user_id)
                    await _close_quietly(connection)

manager = ConnectionManager()
