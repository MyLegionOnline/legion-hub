"""bash tool emits FILE_CHANGED when a command advances the volume generation.

Unlike write_file, bash writes straight to the volume and auto-commits nothing,
so without this nudge the Repository panel and any live git-flow view never
learn a shell command touched the working tree.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.tools import REGISTRY, ToolContext


def _make_ctx():
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = "running"
    ws.network_enabled = True

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    def _make_result(obj):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=obj)
        r.scalars.return_value.all.return_value = []
        return r

    db.execute = AsyncMock(side_effect=lambda q: _make_result(ws))
    return ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)


@pytest.mark.asyncio
async def test_emits_file_changed_when_generation_advances():
    ctx = _make_ctx()
    tool = REGISTRY.get("bash")

    with (
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch(
            "app.core.workspaced_client.get_workspace_generation",
            new_callable=AsyncMock,
            side_effect=[
                {"epoch": "e1", "revision": 4},
                {"epoch": "e1", "revision": 5},
            ],
        ),
        patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": "done", "stderr": ""}
        await tool.execute(ctx, {"cmd": "python download.py"})

    mock_emit.assert_awaited_once()
    assert mock_emit.await_args.kwargs["event_type"] == "FILE_CHANGED"
    assert mock_emit.await_args.kwargs["metadata"]["source"] == "bash"


@pytest.mark.asyncio
async def test_no_event_when_generation_unchanged():
    ctx = _make_ctx()
    tool = REGISTRY.get("bash")

    with (
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch(
            "app.core.workspaced_client.get_workspace_generation",
            new_callable=AsyncMock,
            return_value={"epoch": "e1", "revision": 7},
        ),
        patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": "nothing", "stderr": ""}
        await tool.execute(ctx, {"cmd": "ls -la"})

    mock_emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_event_when_generation_lookup_unavailable():
    ctx = _make_ctx()
    tool = REGISTRY.get("bash")

    with (
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch(
            "app.core.workspaced_client.get_workspace_generation",
            new_callable=AsyncMock,
            side_effect=RuntimeError("workspaced down"),
        ),
        patch("app.core.events.emit_event", new_callable=AsyncMock) as mock_emit,
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": "ok", "stderr": ""}
        result = await tool.execute(ctx, {"cmd": "echo hi"})

    assert "exit_code: 0" in result
    mock_emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_emit_failure_does_not_break_the_tool():
    ctx = _make_ctx()
    tool = REGISTRY.get("bash")

    with (
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch(
            "app.core.workspaced_client.get_workspace_generation",
            new_callable=AsyncMock,
            side_effect=[
                {"epoch": "e1", "revision": 1},
                {"epoch": "e1", "revision": 2},
            ],
        ),
        patch(
            "app.core.events.emit_event",
            new_callable=AsyncMock,
            side_effect=RuntimeError("broadcast failed"),
        ),
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": "done", "stderr": ""}
        result = await tool.execute(ctx, {"cmd": "make build"})

    assert "exit_code: 0" in result
