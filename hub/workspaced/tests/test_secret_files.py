"""Tests for internal design notes Stage 5 — deploy-key tmpfs files.

Unit tests mock aiodocker; no real Docker daemon required.
Run: pytest tests/test_secret_files.py -v
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.docker_manager import SECRETS_DIR, DockerManager
from app.schemas import ExecRequest, SecretFile

ROOM_ID = "test-room-secrets-0001"


def _make_container_mock(running: bool = True):
    container = AsyncMock()
    container.id = "abc123"
    state = {"Running": running}
    container.show = AsyncMock(return_value={"State": state})
    return container


def _make_stream_mock(output: bytes = b""):
    stream = MagicMock()
    stream.read_out = AsyncMock(
        side_effect=[SimpleNamespace(stream=1, data=output), None] if output else [None]
    )
    stream.write_in = AsyncMock()
    stream.close = AsyncMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=False)
    return stream


def _make_exec_mock(exit_code: int = 0, output: bytes = b"", running: bool = False):
    ex = AsyncMock()
    stream = _make_stream_mock(output)
    ex.start = MagicMock(return_value=stream)
    ex.inspect = AsyncMock(return_value={"ExitCode": exit_code, "Running": running})
    return ex


@pytest.fixture
def mgr():
    m = DockerManager()
    m._docker = AsyncMock()
    return m


def _secret_file() -> SecretFile:
    return SecretFile(path=f"{SECRETS_DIR}/DEPLOY_KEY", content="-----BEGIN KEY-----\nabc\n-----END KEY-----", mode="0600")


class TestSecretFiles:
    @pytest.mark.asyncio
    async def test_secret_file_written_and_deleted_after_exec(self, mgr):
        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(return_value=container)

        calls: list[list[str]] = []

        def _exec(cmd, **kwargs):
            calls.append(list(cmd))
            return _make_exec_mock(exit_code=0, output=b"ok\n" if cmd[:1] == ["git"] else b"")

        container.exec = AsyncMock(side_effect=_exec)

        req = ExecRequest(cmd=["git", "push"], secret_files=[_secret_file()])
        result = await mgr.exec(ROOM_ID, req)

        assert result.exit_code == 0

        # write (cat), chmod, main cmd, then rm — in that order.
        write_calls = [c for c in calls if c[:2] == ["sh", "-c"] and "cat" in c[2]]
        chmod_calls = [c for c in calls if c[0] == "chmod"]
        rm_calls = [c for c in calls if c[0] == "rm"]

        assert len(write_calls) == 1
        assert write_calls[0][-1] == f"{SECRETS_DIR}/DEPLOY_KEY"

        assert len(chmod_calls) == 1
        assert chmod_calls[0] == ["chmod", "0600", f"{SECRETS_DIR}/DEPLOY_KEY"]

        assert len(rm_calls) == 1
        assert rm_calls[0] == ["rm", "-f", f"{SECRETS_DIR}/DEPLOY_KEY"]

        # Content travelled over stdin, never as a command-line argument.
        for c in calls:
            assert "-----BEGIN KEY-----" not in " ".join(c)

        # rm must be the very last exec call — after the main command ran.
        assert calls[-1][0] == "rm"

    @pytest.mark.asyncio
    async def test_secret_file_deleted_on_timeout_once_process_exits(self, mgr):
        container = _make_container_mock()
        mgr._docker.containers.get = AsyncMock(return_value=container)

        write_ex = _make_exec_mock(exit_code=0)
        chmod_ex = _make_exec_mock(exit_code=0)
        rm_ex = _make_exec_mock(exit_code=0)

        # The main command's stream never yields a message before the
        # asyncio.timeout fires — read_out blocks forever.
        main_ex = AsyncMock()
        main_stream = MagicMock()

        async def _blocked_read_out():
            await asyncio.sleep(10)

        main_stream.read_out = AsyncMock(side_effect=_blocked_read_out)
        main_stream.__aenter__ = AsyncMock(return_value=main_stream)
        main_stream.__aexit__ = AsyncMock(return_value=False)
        main_ex.start = MagicMock(return_value=main_stream)
        # Still "Running" the first few inspects, then finished.
        main_ex.inspect = AsyncMock(
            side_effect=[
                {"ExitCode": None, "Running": True},
                {"ExitCode": None, "Running": True},
                {"ExitCode": 0, "Running": False},
            ]
        )

        exec_calls: list[Any] = []

        def _exec(cmd, **kwargs):
            exec_calls.append(list(cmd))
            if cmd[0] == "chmod":
                return chmod_ex
            if cmd[0] == "rm":
                return rm_ex
            if cmd[:2] == ["sh", "-c"]:
                return write_ex
            return main_ex

        container.exec = AsyncMock(side_effect=_exec)

        req = ExecRequest(
            cmd=["git", "push"], timeout_sec=1, secret_files=[_secret_file()]
        )

        result = await mgr.exec(ROOM_ID, req)
        assert result.truncated is True

        # Deletion is scheduled as a background task tied to the exec's own
        # completion (plan §3.3), not to this request returning — wait for
        # whatever mgr.exec() just spawned.
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        await asyncio.gather(*pending)

        rm_calls = [c for c in exec_calls if c[0] == "rm"]
        assert len(rm_calls) == 1
        assert rm_calls[0] == ["rm", "-f", f"{SECRETS_DIR}/DEPLOY_KEY"]
