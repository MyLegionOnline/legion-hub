"""
Tests for Stage 2: background bash (bash_output, bash_kill), large-output handling, cwd param.

All tests use mocked workspaced_client and DB — no real workspace needed.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.tools import REGISTRY, ToolContext
from app.models.workspace import WorkspaceBgProc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ws(status="running", network_enabled=True):
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = status
    ws.network_enabled = network_enabled
    return ws


def _make_ctx(ws=None) -> tuple[ToolContext, MagicMock, MagicMock]:
    """Return (ctx, db_mock, ws_mock)."""
    ws = ws or _make_ws()
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    # Default execute: return workspace scalar
    def _make_result(obj):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=obj)
        r.scalars.return_value.all.return_value = []
        return r

    db.execute = AsyncMock(side_effect=lambda q: _make_result(ws))

    ctx = ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)
    return ctx, db, ws


# ---------------------------------------------------------------------------
# bash — cwd parameter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_cwd_passed_to_exec():
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")

    with (
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": "ok", "stderr": ""}
        result = await tool.execute(ctx, {"cmd": "pwd", "cwd": "/workspace/src"})

    assert "exit_code: 0" in result
    called_kwargs = mock_exec.call_args
    assert called_kwargs.kwargs.get("cwd") == "/workspace/src" or called_kwargs.args[2] == "/workspace/src"


@pytest.mark.asyncio
async def test_bash_default_cwd_is_workspace():
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")

    with (
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": "ok", "stderr": ""}
        await tool.execute(ctx, {"cmd": "ls"})

    cwd_arg = mock_exec.call_args.kwargs.get("cwd") or mock_exec.call_args.args[2]
    assert cwd_arg == "/workspace"


# ---------------------------------------------------------------------------
# bash — large output (> 16 KB)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_large_output_saved_to_log():
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")

    big_stdout = "x" * 20_000

    with (
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock) as mock_write,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": big_stdout, "stderr": ""}
        mock_write.return_value = {}
        result = await tool.execute(ctx, {"cmd": "bigcmd"})

    assert "saved to /workspace/.legion/logs/" in result
    assert "read_file" in result
    mock_write.assert_called_once()


@pytest.mark.asyncio
async def test_bash_small_output_not_saved():
    ctx, db, ws = _make_ctx()
    tool = REGISTRY.get("bash")

    with (
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec,
        patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock) as mock_write,
        patch("app.core.workspaced_client.get_workspace_generation", new_callable=AsyncMock, return_value=None),
    ):
        mock_exec.return_value = {"exit_code": 0, "stdout": "small output", "stderr": ""}
        mock_write.return_value = {}
        result = await tool.execute(ctx, {"cmd": "echo hi"})

    assert "small output" in result
    mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# bash — background=True
# ---------------------------------------------------------------------------


def _make_ctx_bg(bg_procs=None) -> tuple[ToolContext, MagicMock, MagicMock]:
    """Context where workspace is running and bg procs list is customizable."""
    ws = _make_ws()
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    bg_procs = bg_procs or []

    call_count = [0]

    def _side_effect(q):
        r = MagicMock()
        c = call_count[0]
        call_count[0] += 1
        if c == 0:
            # First call: workspace lookup
            r.scalar_one_or_none = MagicMock(return_value=ws)
            r.scalars.return_value.all.return_value = bg_procs
        else:
            # Subsequent calls: bg_proc list
            r.scalar_one_or_none = MagicMock(return_value=ws)
            r.scalars.return_value.all.return_value = bg_procs
        return r

    db.execute = AsyncMock(side_effect=_side_effect)
    ctx = ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)
    return ctx, db, ws


@pytest.mark.asyncio
async def test_bash_background_returns_bg_id():
    ctx, db, ws = _make_ctx_bg()
    tool = REGISTRY.get("bash")

    with patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"exit_code": 0, "stdout": "12345 12345\n", "stderr": ""}
        result = await tool.execute(ctx, {"cmd": "npm run dev", "background": True})

    assert "bg_id:" in result
    assert "pid: 12345" in result
    assert "bash_output" in result
    assert "bash_kill" in result
    # Verify a WorkspaceBgProc was added
    db.add.assert_called()
    db.flush.assert_called()


@pytest.mark.asyncio
async def test_bash_background_limit_enforced():
    # Create 5 fake running bg procs
    running = [MagicMock(bg_id=f"bg{i}", status="running") for i in range(5)]

    ws = _make_ws()
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    call_count = [0]

    def _side_effect(q):
        r = MagicMock()
        c = call_count[0]
        call_count[0] += 1
        r.scalar_one_or_none = MagicMock(return_value=ws)
        r.scalars.return_value.all.return_value = running if c >= 1 else []
        return r

    db.execute = AsyncMock(side_effect=_side_effect)
    ctx = ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)

    tool = REGISTRY.get("bash")
    result = await tool.execute(ctx, {"cmd": "npm run dev", "background": True})

    assert "maximum" in result.lower() and "5" in result


@pytest.mark.asyncio
async def test_bash_background_workspace_not_running():
    ws = _make_ws(status="stopped")
    ctx, db, _ = _make_ctx(ws)
    tool = REGISTRY.get("bash")
    # bash wakes a stopped workspace itself; a failed start is an error.
    with patch(
        "app.core.workspace_lifecycle.ensure_workspace_started",
        new_callable=AsyncMock,
        side_effect=RuntimeError("workspaced unreachable"),
    ):
        result = await tool.execute(ctx, {"cmd": "npm run dev", "background": True})
    assert "Error" in result
    assert "could not be started" in result


# ---------------------------------------------------------------------------
# bash_output
# ---------------------------------------------------------------------------


def _make_ctx_with_proc(proc) -> tuple[ToolContext, MagicMock]:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    def _side_effect(q):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=proc)
        return r

    db.execute = AsyncMock(side_effect=_side_effect)
    ctx = ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)
    return ctx, db


@pytest.mark.asyncio
async def test_bash_output_reads_log():
    proc = MagicMock(
        bg_id="abc123",
        pid=999,
        pgid=999,
        log_path=".legion/bg/abc123.log",
        cmd="npm run dev",
        status="running",
    )
    ctx, db = _make_ctx_with_proc(proc)
    tool = REGISTRY.get("bash_output")

    log_content = "\n".join(f"line {i}" for i in range(200))

    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read:
        mock_read.return_value = log_content
        result = await tool.execute(ctx, {"bg_id": "abc123", "tail": 50})

    assert "bg_id: abc123" in result
    assert "status: running" in result
    assert "pid: 999" in result
    assert "200" in result  # total lines
    mock_read.assert_called_once()
    # Should have path relative to workspace root
    call_args = mock_read.call_args
    assert ".legion/bg/abc123.log" in str(call_args)


@pytest.mark.asyncio
async def test_bash_output_missing_proc():
    ctx, db = _make_ctx_with_proc(None)
    tool = REGISTRY.get("bash_output")
    result = await tool.execute(ctx, {"bg_id": "nonexistent"})
    assert "Error" in result
    assert "nonexistent" in result


@pytest.mark.asyncio
async def test_bash_output_empty_log():
    proc = MagicMock(
        bg_id="abc123",
        pid=999,
        log_path=".legion/bg/abc123.log",
        cmd="npm run dev",
        status="running",
    )
    ctx, db = _make_ctx_with_proc(proc)
    tool = REGISTRY.get("bash_output")

    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read:
        mock_read.return_value = None  # not yet created
        result = await tool.execute(ctx, {"bg_id": "abc123"})

    assert "not yet created" in result
    assert "running" in result


@pytest.mark.asyncio
async def test_bash_output_tail_param():
    proc = MagicMock(
        bg_id="abc123",
        pid=999,
        log_path=".legion/bg/abc123.log",
        cmd="ls",
        status="running",
    )
    ctx, db = _make_ctx_with_proc(proc)
    tool = REGISTRY.get("bash_output")

    log_content = "\n".join(f"line {i}" for i in range(500))

    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read:
        mock_read.return_value = log_content
        result = await tool.execute(ctx, {"bg_id": "abc123", "tail": 10})

    # Should say we show last 10 of 500
    assert "10" in result
    assert "line 490" in result
    assert "line 0" not in result


# ---------------------------------------------------------------------------
# bash_kill
# ---------------------------------------------------------------------------


def _make_ctx_kill(proc, ws_status="running") -> tuple[ToolContext, MagicMock]:
    ws = _make_ws(status=ws_status)
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()

    call_count = [0]

    def _side_effect(q):
        r = MagicMock()
        c = call_count[0]
        call_count[0] += 1
        if c == 0:
            r.scalar_one_or_none = MagicMock(return_value=proc)
        else:
            r.scalar_one_or_none = MagicMock(return_value=ws)
        return r

    db.execute = AsyncMock(side_effect=_side_effect)
    ctx = ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)
    return ctx, db


@pytest.mark.asyncio
async def test_bash_kill_marks_killed():
    proc = MagicMock(bg_id="abc123", pid=999, pgid=999, status="running")
    ctx, db = _make_ctx_kill(proc)
    tool = REGISTRY.get("bash_kill")

    with patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = {"exit_code": 0, "stdout": "", "stderr": ""}
        result = await tool.execute(ctx, {"bg_id": "abc123"})

    assert "killed" in result
    assert proc.status == "killed"
    db.flush.assert_called()


@pytest.mark.asyncio
async def test_bash_kill_already_stopped():
    proc = MagicMock(bg_id="abc123", pid=999, status="killed")
    ctx, db = _make_ctx_kill(proc)
    tool = REGISTRY.get("bash_kill")

    result = await tool.execute(ctx, {"bg_id": "abc123"})
    assert "already stopped" in result


@pytest.mark.asyncio
async def test_bash_kill_missing_proc():
    ctx, db = _make_ctx_with_proc(None)
    tool = REGISTRY.get("bash_kill")
    result = await tool.execute(ctx, {"bg_id": "nonexistent"})
    assert "Error" in result


@pytest.mark.asyncio
async def test_bash_kill_workspace_stopped():
    proc = MagicMock(bg_id="abc123", pid=999, pgid=999, status="running")
    ctx, db = _make_ctx_kill(proc, ws_status="stopped")
    tool = REGISTRY.get("bash_kill")

    result = await tool.execute(ctx, {"bg_id": "abc123"})
    # Should mark as failed when workspace is gone
    assert "failed" in result or "killed" in result


# ---------------------------------------------------------------------------
# Registry checks
# ---------------------------------------------------------------------------


def test_registry_has_bash_output():
    assert REGISTRY.get("bash_output") is not None


def test_registry_has_bash_kill():
    assert REGISTRY.get("bash_kill") is not None


def test_bash_output_not_in_cm_tools():
    cm_names = {t.name for t in REGISTRY.tools_for("cm")}
    assert "bash_output" not in cm_names
    assert "bash_kill" not in cm_names


def test_bash_output_not_requires_container():
    tool = REGISTRY.get("bash_output")
    assert tool.requires_container is False


def test_bash_kill_requires_container():
    tool = REGISTRY.get("bash_kill")
    assert tool.requires_container is True
