"""Memory pressure inside a room (internal design notes, E0)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import WORKSPACED_TOKEN
from app.docker_manager import DockerManager
from app.room_pressure import parse_pressure

AUTH_HEADER = {"Authorization": f"Bearer {WORKSPACED_TOKEN}"}
ROOM_ID = "test-room-00000000"
MIB = 1024 * 1024

CGROUP_V2 = f"""#current
{1600 * MIB}
#max
{2048 * MIB}
#stat
anon {1400 * MIB}
inactive_file {100 * MIB}
shmem {64 * MIB}
#pressure
some avg10=12.50 avg60=4.00 avg300=1.00 total=123
full avg10=3.25 avg60=1.00 avg300=0.10 total=45
#events
low 0
high 0
max 7
oom 2
oom_kill 2
#agents
307200
#ps
  412 798720 node /workspace/node_modules/.bin/next dev
  518 307200 node /workspace/node_modules/.bin/next build
    1   1024 /sbin/docker-init -- sleep infinity
"""


def test_parse_cgroup_v2_with_psi_and_processes():
    p = parse_pressure(CGROUP_V2)
    assert p["mem_used_mb"] == 1500.0  # current minus inactive page cache
    assert p["mem_max_mb"] == 2048
    assert p["pct"] == 73.2
    assert p["psi_some_avg10"] == 12.5 and p["psi_full_avg10"] == 3.25
    assert p["oom_kills"] == 2
    assert p["agent_rss_mb"] == 300.0  # tagged processes, KiB in the script
    assert p["tmp_mb"] == 64.0  # tmpfs pages are shmem
    assert p["top"][0] == {
        "pid": 412, "rss_mb": 780.0, "cmd": "node /workspace/node_modules/.bin/next dev",
    }
    assert len(p["top"]) == 3


def test_unlimited_cgroup_falls_back_to_docker_limit():
    text = f"#current\n{512 * MIB}\n#max\nmax\n#stat\n#pressure\n#events\n#ps\n"
    p = parse_pressure(text, docker_limit_mb=1024)
    assert p["mem_max_mb"] == 1024 and p["pct"] == 50.0


def test_missing_cgroup_v2_gives_unknown_not_full():
    p = parse_pressure("#current\n#max\n#stat\n#pressure\n#events\n#ps\n")
    assert p["pct"] is None and p["mem_used_mb"] is None
    assert p["psi_some_avg10"] is None and p["oom_kills"] is None
    assert p["top"] == []


def _container(text: str, running: bool = True):
    container = AsyncMock()
    container.show = AsyncMock(return_value={
        "State": {"Running": running}, "HostConfig": {"Memory": 2048 * MIB},
    })
    messages = [SimpleNamespace(stream=1, data=text.encode()), None]

    stream = MagicMock()
    stream.read_out = AsyncMock(side_effect=messages)

    @asynccontextmanager
    async def _start(detach=False):
        yield stream

    ex = MagicMock()
    ex.start = _start
    container.exec = AsyncMock(return_value=ex)
    return container


@pytest.fixture
def mgr():
    m = DockerManager()
    m._docker = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_pressure_is_read_once_and_cached(mgr):
    container = _container(CGROUP_V2)
    mgr._docker.containers.get = AsyncMock(return_value=container)

    first = await mgr.pressure(ROOM_ID)
    second = await mgr.pressure(ROOM_ID)

    assert first["status"] == "running" and first["pct"] == 73.2
    assert first["cached"] is False and second["cached"] is True
    container.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_pressure_falls_back_to_docker_stats_when_exec_times_out(mgr):
    """E4: under load the exec misses its timeout; the percentage must survive."""
    container = _container(CGROUP_V2)
    container.exec = AsyncMock(side_effect=TimeoutError())
    container.stats = AsyncMock(return_value=[{"memory_stats": {
        "usage": 900 * 1024 * 1024, "limit": 1024 * 1024 * 1024,
        "stats": {"inactive_file": 100 * 1024 * 1024},
    }}])
    mgr._docker.containers.get = AsyncMock(return_value=container)

    result = await mgr.pressure(ROOM_ID)

    assert result["status"] == "running" and result["source"] == "docker_stats"
    assert result["mem_used_mb"] == 800.0 and result["mem_max_mb"] == 1024
    assert result["pct"] == 78.1
    assert result["psi_some_avg10"] is None and result["top"] == []
    assert result["agent_rss_mb"] is None


@pytest.mark.asyncio
async def test_pressure_of_a_stopped_room(mgr):
    mgr._docker.containers.get = AsyncMock(return_value=_container("", running=False))

    result = await mgr.pressure(ROOM_ID)

    assert result["status"] == "stopped" and "pct" not in result
    assert ROOM_ID not in mgr._pressure_cache


@pytest.mark.asyncio
async def test_pressure_endpoint(mgr, monkeypatch):
    import app.main as main

    monkeypatch.setattr(main, "docker_mgr", mgr)
    mgr._docker.containers.get = AsyncMock(return_value=_container(CGROUP_V2))
    async with AsyncClient(transport=ASGITransport(app=main.app), base_url="http://t") as client:
        resp = await client.get(f"/workspaces/{ROOM_ID}/pressure", headers=AUTH_HEADER)
        unauth = await client.get(f"/workspaces/{ROOM_ID}/pressure")
    assert resp.status_code == 200 and resp.json()["pct"] == 73.2
    assert unauth.status_code in (401, 403, 422)  # no token header
