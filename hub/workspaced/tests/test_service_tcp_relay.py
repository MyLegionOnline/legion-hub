"""The in-container TCP relay behind the Browser preview tunnel."""

import os
import socket
import subprocess
import sys
import threading
import time

from app.docker_manager import SERVICE_TCP_RELAY


def _start_relay(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", SERVICE_TCP_RELAY],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        env={**os.environ, "SERVICE_PORT": str(port)},
    )


def test_idle_connection_outlives_the_connect_timeout_and_ends_with_the_tunnel():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    peer_closed = threading.Event()

    def service():
        conn, _ = server.accept()
        # A quiet HMR WebSocket: nothing is sent for longer than the 10 s
        # connect timeout, then a frame arrives.
        time.sleep(11)
        conn.sendall(b"hmr-update")
        # The relay half-closes once its stdin ends; the service then closes.
        while conn.recv(1024):
            pass
        peer_closed.set()
        conn.close()

    threading.Thread(target=service, daemon=True).start()
    relay = _start_relay(port)
    try:
        assert relay.stdout.read(len(b"hmr-update")) == b"hmr-update"
        relay.stdin.close()
        assert relay.wait(timeout=5) == 0
        assert peer_closed.wait(timeout=5)
    finally:
        if relay.poll() is None:
            relay.kill()
        server.close()
