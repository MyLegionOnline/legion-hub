"""Room memory: resident figure, stats cache, live limit change (T2a).

internal design notes §3a — "In use now" is the rooms' resident
memory, and a room's memory limit can be changed while it runs.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import aiodocker
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import WORKSPACED_TOKEN
from app.docker_manager import DockerManager, memory_from_stats

AUTH_HEADER = {"Authorization": f"Bearer {WORKSPACED_TOKEN}"}
ROOM_ID = "test-room-00000000"
MIB = 1024 * 1024


def _not_found():
    return aiodocker.exceptions.DockerError(404, {"message": "not found"})


def _container(running: bool = True, usage_mb: int = 300, inactive_mb: int = 100):
    container = AsyncMock()
    container.id = "abc123"
    container.show = AsyncMock(return_value={"State": {"Running": running}})
    container.stats = AsyncMock(return_value=[{
        "memory_stats": {
            "usage": usage_mb * MIB,
            "limit": 1024 * MIB,
            "stats": {"inactive_file": inactive_mb * MIB},
        },
    }])
    return container


@pytest.fixture
def mgr():
    m = DockerManager()
    m._docker = AsyncMock()
    return m


def test_resident_memory_drops_inactive_page_cache():
    v2 = memory_from_stats([{"memory_stats": {
        "usage": 300 * MIB, "limit": 2048 * MIB, "stats": {"inactive_file": 100 * MIB},
    }}])
    assert v2 == {"memory_rss_mb": 200.0, "memory_usage_mb": 300.0, "memory_limit_mb": 2048}
    v1 = memory_from_stats({"memory_stats": {
        "usage": 300 * MIB, "stats": {"total_inactive_file": 50 * MIB},
    }})
    assert v1["memory_rss_mb"] == 250.0 and v1["memory_limit_mb"] is None
    assert memory_from_stats([{}])["memory_rss_mb"] is None


@pytest.mark.asyncio
async def test_stats_are_cached_for_a_running_room(mgr):
    container = _container()
    mgr._docker.containers.get = AsyncMock(return_value=container)

    first = await mgr.stats(ROOM_ID)
    second = await mgr.stats(ROOM_ID)

    assert first["status"] == "running" and first["memory_rss_mb"] == 200.0
    assert first["cached"] is False and second["cached"] is True
    container.stats.assert_awaited_once()


@pytest.mark.asyncio
async def test_stats_of_a_stopped_room_are_not_cached(mgr):
    mgr._docker.containers.get = AsyncMock(side_effect=_not_found())

    result = await mgr.stats(ROOM_ID)

    assert result["status"] == "stopped" and result["memory_rss_mb"] is None
    assert ROOM_ID not in mgr._stats_cache


@pytest.mark.asyncio
async def test_update_memory_moves_swap_with_memory(mgr):
    container = _container()
    mgr._docker.containers.get = AsyncMock(return_value=container)
    mgr._docker._query_json = AsyncMock(return_value={"Warnings": []})
    await mgr.stats(ROOM_ID)

    result = await mgr.update_memory(ROOM_ID, 2048)

    assert result == {"room_id": ROOM_ID, "memory_limit_mb": 2048}
    args, kwargs = mgr._docker._query_json.call_args
    assert args[0] == "containers/abc123/update"
    assert kwargs["method"] == "POST"
    assert kwargs["data"] == {"Memory": 2048 * MIB, "MemorySwap": 4096 * MIB}
    # The next figure is measured against the new limit, not the cached one.
    assert ROOM_ID not in mgr._stats_cache


@pytest.mark.asyncio
async def test_update_memory_of_a_stopped_room_is_refused(mgr):
    mgr._docker.containers.get = AsyncMock(return_value=_container(running=False))
    mgr._docker._query_json = AsyncMock()

    with pytest.raises(RuntimeError):
        await mgr.update_memory(ROOM_ID, 1024)
    mgr._docker._query_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_stats_and_memory():
    from app.main import app, docker_mgr

    docker_mgr._docker = AsyncMock()
    docker_mgr._stats_cache.clear()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        docker_mgr._docker.containers.get = AsyncMock(return_value=_container())
        docker_mgr._docker._query_json = AsyncMock(return_value={})

        stats = await c.get(f"/workspaces/{ROOM_ID}/stats", headers=AUTH_HEADER)
        assert stats.status_code == 200 and stats.json()["memory_rss_mb"] == 200.0

        updated = await c.post(
            f"/workspaces/{ROOM_ID}/memory", headers=AUTH_HEADER, json={"mem_limit_mb": 1536},
        )
        assert updated.status_code == 200 and updated.json()["memory_limit_mb"] == 1536

        docker_mgr._docker.containers.get = AsyncMock(side_effect=_not_found())
        stopped = await c.post(
            f"/workspaces/{ROOM_ID}/memory", headers=AUTH_HEADER, json={"mem_limit_mb": 1024},
        )
        assert stopped.status_code == 409

        too_big = await c.post(
            f"/workspaces/{ROOM_ID}/memory", headers=AUTH_HEADER, json={"mem_limit_mb": 99999},
        )
        assert too_big.status_code == 422
