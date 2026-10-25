"""Tests for the workspaced microservice.

Unit tests mock aiodocker; no real Docker daemon required.
Run: pytest tests/test_workspaced.py -v
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import WORKSPACED_TOKEN
from app.docker_manager import DockerManager
from app.schemas import StartRequest, ExecRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AUTH_HEADER = {"Authorization": f"Bearer {WORKSPACED_TOKEN}"}
ROOM_ID = "test-room-00000000"


def _make_container_mock(running: bool = True, container_id: str = "abc123"):
    container = AsyncMock()
    container.id = container_id
    container._container = {"Names": [f"/ws-{ROOM_ID}"]}

    state = {"Running": running, "StartedAt": "2026-07-14T10:00:00Z"}
    container.show = AsyncMock(return_value={"State": state})
    container.start = AsyncMock()
    container.stop = AsyncMock()
    container.delete = AsyncMock()

    stats_data = {
        "memory_stats": {"usage": 50 * 1024 * 1024, "limit": 1024 * 1024 * 1024},
    }
    container.stats = AsyncMock(return_value=[stats_data])

    return container


def _make_exec_mock(exit_code: int = 0, output: bytes = b"hello\n"):
    ex = AsyncMock()

    stream_mock = MagicMock()
    stream_mock.read_out = AsyncMock(
        side_effect=[
            SimpleNamespace(stream=1, data=output),
            None,
        ]
    )
    stream_mock.__aenter__ = AsyncMock(return_value=stream_mock)
    stream_mock.__aexit__ = AsyncMock(return_value=False)

    ex.start = MagicMock(return_value=stream_mock)
    ex.inspect = AsyncMock(return_value={"ExitCode": exit_code})
    return ex


# ---------------------------------------------------------------------------
# DockerManager unit tests
# ---------------------------------------------------------------------------


class TestDockerManager:
    @pytest.fixture
    def mgr(self):
        m = DockerManager()
        m._docker = AsyncMock()
        return m

    @pytest.mark.asyncio
    async def test_start_creates_and_starts_container(self, mgr):
        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(
            side_effect=_not_found_error()
        )
        mgr._docker.containers.create_or_replace = AsyncMock(return_value=container)

        result = await mgr.start(ROOM_ID, StartRequest())

        assert result.status == "running"
        assert result.room_id == ROOM_ID
        container.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_returns_running_if_already_up(self, mgr):
        container = _make_container_mock(running=True)
        mgr._docker.containers.get = AsyncMock(return_value=container)

        result = await mgr.start(ROOM_ID, StartRequest())

        assert result.status == "running"
        mgr._docker.containers.create_or_replace.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_applies_xfs_project_and_overlay_layer_limit(self, mgr, monkeypatch):
        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())
        mgr._docker.containers.create_or_replace = AsyncMock(return_value=container)
        quotas = SimpleNamespace(configure=AsyncMock(return_value=[ROOM_ID]))
        monkeypatch.setattr("app.project_quota.project_quotas", quotas)
        monkeypatch.setattr("app.docker_manager.WORKSPACE_PROJECT_QUOTA_MODE", "required")

        await mgr.start(
            ROOM_ID,
            StartRequest(
                storage_project_id=10042,
                storage_quota_mb=3072,
                storage_project_room_ids=[ROOM_ID],
                container_layer_mb=2048,
            ),
        )

        quotas.configure.assert_awaited_once_with(
            project_id=10042,
            quota_mb=3072,
            room_ids=[ROOM_ID],
        )
        _name, config = mgr._docker.containers.create_or_replace.await_args.args
        assert config["HostConfig"]["StorageOpt"] == {"size": "2048M"}

    @pytest.mark.asyncio
    async def test_start_reaps_orphans_and_uses_configured_pids_limit(self, mgr, monkeypatch):
        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())
        mgr._docker.containers.create_or_replace = AsyncMock(return_value=container)
        monkeypatch.setattr("app.docker_manager.WORKSPACE_PIDS_LIMIT", 777)

        await mgr.start(ROOM_ID, StartRequest())

        _, config = mgr._docker.containers.create_or_replace.await_args.args
        # tini as PID 1: orphans of killed renders are reaped instead of
        # lingering as zombies that hold the container's PID budget.
        assert config["HostConfig"]["Init"] is True
        assert config["HostConfig"]["PidsLimit"] == 777

    @pytest.mark.asyncio
    async def test_workspace_network_isolated_and_off_switch_wins(self, mgr, monkeypatch):
        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())
        mgr._docker.containers.create_or_replace = AsyncMock(return_value=container)
        monkeypatch.setattr("app.docker_manager.WORKSPACE_NETWORK", "legion_workspace-network")

        await mgr.start(ROOM_ID, StartRequest(network_enabled=True))
        _, config = mgr._docker.containers.create_or_replace.await_args.args
        assert config["HostConfig"]["NetworkMode"] == "legion_workspace-network"

        await mgr.start(ROOM_ID, StartRequest(network_enabled=False))
        _, config = mgr._docker.containers.create_or_replace.await_args.args
        assert config["HostConfig"]["NetworkMode"] == "none"

    @pytest.mark.asyncio
    async def test_required_quota_mode_rejects_legacy_unlimited_start(self, mgr, monkeypatch):
        monkeypatch.setattr("app.docker_manager.WORKSPACE_PROJECT_QUOTA_MODE", "required")

        with pytest.raises(RuntimeError, match="required project quotas"):
            await mgr.start(ROOM_ID, StartRequest())

    @pytest.mark.asyncio
    async def test_stop_stops_and_removes(self, mgr):
        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(return_value=container)

        status = await mgr.stop(ROOM_ID)

        assert status == "stopped"
        container.stop.assert_awaited_once()
        container.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_running(self, mgr):
        mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())

        status = await mgr.stop(ROOM_ID)
        assert status == "stopped"

    @pytest.mark.asyncio
    async def test_destroy_removes_container_and_room_data(self, mgr, tmp_path, monkeypatch):
        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(return_value=container)
        room_root = tmp_path / ROOM_ID
        room_root.mkdir()
        (room_root / "project.txt").write_text("permanent data", encoding="utf-8")
        monkeypatch.setattr("app.docker_manager.WORKSPACE_DATA_DIR", str(tmp_path))

        result = await mgr.destroy(ROOM_ID)

        assert result == {
            "room_id": ROOM_ID,
            "status": "destroyed",
            "container_removed": True,
            "data_removed": True,
        }
        container.stop.assert_awaited_once()
        container.delete.assert_awaited_once_with(force=True)
        assert not room_root.exists()

    @pytest.mark.asyncio
    async def test_destroy_is_idempotent_without_container_or_data(self, mgr, tmp_path, monkeypatch):
        mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())
        monkeypatch.setattr("app.docker_manager.WORKSPACE_DATA_DIR", str(tmp_path))

        result = await mgr.destroy(ROOM_ID)

        assert result["status"] == "destroyed"
        assert result["container_removed"] is False
        assert result["data_removed"] is False

    @pytest.mark.asyncio
    async def test_status_returns_stopped_when_no_container(self, mgr):
        mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())

        result = await mgr.status(ROOM_ID)
        assert result.status == "stopped"

    @pytest.mark.asyncio
    async def test_status_returns_running_with_stats(self, mgr):
        container = _make_container_mock(running=True)
        mgr._docker.containers.get = AsyncMock(return_value=container)

        result = await mgr.status(ROOM_ID)

        assert result.status == "running"
        assert result.memory_usage_mb == 50.0
        assert result.container_id == "abc123"

    @pytest.mark.asyncio
    async def test_exec_runs_command(self, mgr):
        container = _make_container_mock(running=True)
        mgr._docker.containers.get = AsyncMock(return_value=container)

        ex = _make_exec_mock(exit_code=0, output=b"hello\n")
        container.exec = AsyncMock(return_value=ex)

        result = await mgr.exec(ROOM_ID, ExecRequest(cmd=["echo", "hello"]))

        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_exec_raises_when_not_running(self, mgr):
        mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())

        with pytest.raises(RuntimeError, match="not running"):
            await mgr.exec(ROOM_ID, ExecRequest(cmd=["echo", "hi"]))

    @pytest.mark.asyncio
    async def test_activity_tracking(self, mgr):
        assert mgr.last_activity(ROOM_ID) is None

        mgr.touch_activity(ROOM_ID)
        ts = mgr.last_activity(ROOM_ID)
        assert ts is not None
        assert time.time() - ts < 2

    @pytest.mark.asyncio
    async def test_list_running(self, mgr):
        c1 = MagicMock()
        c1._container = {"Names": ["/ws-room-a"]}
        c2 = MagicMock()
        c2._container = {"Names": ["/ws-room-b"]}
        mgr._docker.containers.list = AsyncMock(return_value=[c1, c2])

        result = await mgr.list_running()
        assert set(result) == {"room-a", "room-b"}


# ---------------------------------------------------------------------------
# API endpoint tests (via TestClient)
# ---------------------------------------------------------------------------


class TestAPI:
    @pytest.fixture
    async def client(self):
        from app.main import app, docker_mgr

        docker_mgr._docker = AsyncMock()

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_start_requires_auth(self, client):
        resp = await client.post(f"/workspaces/{ROOM_ID}/start")
        assert resp.status_code == 422 or resp.status_code == 401

    @pytest.mark.asyncio
    async def test_start_with_auth(self, client):
        from app.main import docker_mgr
        from app.schemas import StatusResponse

        docker_mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())
        container = _make_container_mock()
        docker_mgr._docker.containers.create_or_replace = AsyncMock(return_value=container)

        resp = await client.post(
            f"/workspaces/{ROOM_ID}/start",
            headers=AUTH_HEADER,
            json={"mem_limit_mb": 512, "cpu_limit_milli": 500},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["room_id"] == ROOM_ID

    @pytest.mark.asyncio
    async def test_stop_with_auth(self, client):
        from app.main import docker_mgr

        container = _make_container_mock()
        docker_mgr._docker.containers.get = AsyncMock(return_value=container)

        resp = await client.post(
            f"/workspaces/{ROOM_ID}/stop",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_destroy_with_auth(self, client):
        from app.main import docker_mgr

        docker_mgr.destroy = AsyncMock(
            return_value={
                "room_id": ROOM_ID,
                "status": "destroyed",
                "container_removed": True,
                "data_removed": True,
            }
        )

        resp = await client.delete(
            f"/workspaces/{ROOM_ID}",
            headers=AUTH_HEADER,
        )

        assert resp.status_code == 200
        assert resp.json()["status"] == "destroyed"
        docker_mgr.destroy.assert_awaited_once_with(ROOM_ID)

    @pytest.mark.asyncio
    async def test_status_with_auth(self, client):
        from app.main import docker_mgr

        docker_mgr._docker.containers.get = AsyncMock(side_effect=_not_found_error())

        resp = await client.get(
            f"/workspaces/{ROOM_ID}/status",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_exec_with_auth(self, client):
        from app.main import docker_mgr

        container = _make_container_mock(running=True)
        docker_mgr._docker.containers.get = AsyncMock(return_value=container)

        ex = _make_exec_mock(exit_code=0, output=b"hi\n")
        container.exec = AsyncMock(return_value=ex)

        resp = await client.post(
            f"/workspaces/{ROOM_ID}/exec",
            headers=AUTH_HEADER,
            json={"cmd": ["echo", "hi"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["exit_code"] == 0


# ---------------------------------------------------------------------------
# Reaper test
# ---------------------------------------------------------------------------


class TestReaper:
    @pytest.mark.asyncio
    async def test_reaper_stops_idle_containers(self):
        mgr = DockerManager()
        mgr._docker = AsyncMock()

        mgr._last_activity[ROOM_ID] = time.time() - 1000

        c = MagicMock()
        c._container = {"Names": [f"/ws-{ROOM_ID}"]}
        mgr._docker.containers.list = AsyncMock(return_value=[c])

        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(return_value=container)

        from app.reaper import idle_reaper
        import app.config as cfg

        orig_interval = cfg.REAPER_INTERVAL_SEC
        cfg.REAPER_INTERVAL_SEC = 0.01

        task = asyncio.create_task(idle_reaper(mgr))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        cfg.REAPER_INTERVAL_SEC = orig_interval

        container.stop.assert_awaited()


class TestExitedSweep:
    @pytest.mark.asyncio
    async def test_removes_exited_container(self):
        mgr = DockerManager()
        mgr._docker = AsyncMock()
        listed = MagicMock()
        listed._container = {"Names": [f"/ws-{ROOM_ID}"]}
        mgr._docker.containers.list = AsyncMock(return_value=[listed])
        container = _make_container_mock(running=False)
        container.show = AsyncMock(return_value={"State": {"Running": False, "Status": "exited"}})
        mgr._docker.containers.get = AsyncMock(return_value=container)

        assert await mgr.remove_exited() == [ROOM_ID]
        container.delete.assert_awaited_once_with(force=True)
        filters = mgr._docker.containers.list.await_args.kwargs["filters"]
        assert filters["status"] == ["exited", "dead"]

    @pytest.mark.asyncio
    async def test_keeps_container_restarted_meanwhile(self):
        mgr = DockerManager()
        mgr._docker = AsyncMock()
        listed = MagicMock()
        listed._container = {"Names": [f"/ws-{ROOM_ID}"]}
        mgr._docker.containers.list = AsyncMock(return_value=[listed])
        container = _make_container_mock(running=True)
        container.show = AsyncMock(return_value={"State": {"Running": True, "Status": "running"}})
        mgr._docker.containers.get = AsyncMock(return_value=container)

        assert await mgr.remove_exited() == []
        container.delete.assert_not_awaited()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _not_found_error():
    import aiodocker.exceptions
    return aiodocker.exceptions.DockerError(404, "not found")
