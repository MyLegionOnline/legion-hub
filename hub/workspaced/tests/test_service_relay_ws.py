"""Service WebSocket relay: HMR subprotocol handshake and volume ownership."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app import config, generation, main


class _Tunnel:
    """Fake service socket: answers the HTTP upgrade, then reports EOF."""

    def __init__(self, accept_protocol: str | None):
        self.accept_protocol = accept_protocol
        self.written: list[bytes] = []
        self._replies = [self._reply()]

    def _reply(self) -> bytes:
        head = "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        if self.accept_protocol:
            head += f"Sec-WebSocket-Protocol: {self.accept_protocol}\r\n"
        return (head + "\r\n").encode("ascii")

    async def write(self, data: bytes) -> None:
        self.written.append(data)

    async def read(self) -> bytes | None:
        return self._replies.pop(0) if self._replies else None

    async def aclose(self) -> None:
        pass


@pytest.fixture
def relay(monkeypatch):
    class _Supervisor:
        async def service_port(self, room_id, service_id):
            return 3000

    tunnels: list[_Tunnel] = []

    def _install(accept_protocol):
        tunnel = _Tunnel(accept_protocol)
        tunnels.append(tunnel)

        async def _open(room_id, port):
            return tunnel

        monkeypatch.setattr(main.docker_mgr, "open_service_tcp_tunnel", _open)
        return tunnel

    monkeypatch.setattr(main, "_service_supervisor", lambda: _Supervisor())
    return _install


def _url() -> str:
    return f"/workspaces/room/services/svc/relay-ws?token={config.WORKSPACED_TOKEN}&path=/&query="


def test_relay_offers_the_browser_subprotocol_to_the_service(relay):
    tunnel = relay("vite-hmr")
    client = TestClient(main.app)  # no ``with``: the lifespan wants a Docker daemon
    with client.websocket_connect(_url(), subprotocols=["vite-hmr"]) as ws:
        assert ws.accepted_subprotocol == "vite-hmr"
    handshake = tunnel.written[0].decode("ascii")
    assert "Sec-WebSocket-Protocol: vite-hmr\r\n" in handshake


def test_relay_without_subprotocol_sends_no_protocol_header(relay):
    tunnel = relay(None)
    client = TestClient(main.app)
    with client.websocket_connect(_url()) as ws:
        assert ws.accepted_subprotocol is None
    assert "Sec-WebSocket-Protocol" not in tunnel.written[0].decode("ascii")


def test_generation_metadata_belongs_to_the_container_user(tmp_path: Path, monkeypatch):
    """A root-owned ``.legion`` stops the container user creating ``.legion/services``."""
    monkeypatch.setattr(generation, "WORKSPACE_DATA_DIR", tmp_path)
    chowned: list[tuple[Path, int, int]] = []
    monkeypatch.setattr(os, "lchown", lambda path, uid, gid: chowned.append((Path(path), uid, gid)), raising=False)

    generation.ensure_generation("room")

    paths = {path for path, _uid, _gid in chowned}
    assert tmp_path / "room" / ".legion" in paths
    assert tmp_path / "room" / ".legion" / "generation.json" in paths
    assert {(uid, gid) for _p, uid, gid in chowned} == {(generation.WORKSPACE_UID, generation.WORKSPACE_GID)}
