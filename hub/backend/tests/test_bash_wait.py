"""bash_wait: blocks on a background process, free of budget and wall-clock."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core import run_clock
from app.core.tools import REGISTRY, ToolContext
from app.core.tools import _bash_wait


def _ctx(proc):
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=proc)
    db.execute = AsyncMock(return_value=result)
    return ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db), db


def _proc(status="running"):
    return MagicMock(bg_id="abc", log_path=".legion/bg/abc.log", pid=42, status=status)


def test_run_clock_credit_shifts_deadline_and_is_capped():
    tid = uuid4()
    run_clock.reset(tid)
    assert run_clock.effective_deadline(100.0, tid) == 100.0
    run_clock.add_credit(tid, 30)
    assert run_clock.effective_deadline(100.0, tid) == 130.0
    run_clock.add_credit(tid, run_clock.MAX_CREDIT_SECONDS * 2)
    assert run_clock.effective_deadline(100.0, tid) == 100.0 + run_clock.MAX_CREDIT_SECONDS
    run_clock.reset(tid)
    assert run_clock.effective_deadline(100.0, tid) == 100.0
    assert run_clock.effective_deadline(None, tid) is None


@pytest.mark.asyncio
async def test_bash_wait_returns_on_exit_and_marks_done():
    proc = _proc()
    ctx, db = _ctx(proc)

    async def read(rid, path):
        return "0\n" if path.endswith(".exit") else "installed\nOK\n"

    with patch("app.core.workspaced_client.read_workspace_file", side_effect=read):
        out = await REGISTRY.get("bash_wait").execute(ctx, {"bg_id": "abc", "max_wait_sec": 30})

    assert "status: done" in out and "exit_code: 0" in out and "installed" in out
    assert proc.status == "done"


@pytest.mark.asyncio
async def test_bash_wait_failed_exit_code():
    ctx, _ = _ctx(_proc())

    async def read(rid, path):
        return "2\n" if path.endswith(".exit") else "boom\n"

    with patch("app.core.workspaced_client.read_workspace_file", side_effect=read):
        out = await REGISTRY.get("bash_wait").execute(ctx, {"bg_id": "abc"})

    assert "status: failed" in out and "exit_code: 2" in out


@pytest.mark.asyncio
async def test_bash_wait_timeout_reports_running_and_credits_clock():
    proc = _proc()
    ctx, _ = _ctx(proc)
    run_clock.reset(ctx.task_id)

    async def read(rid, path):
        return None if path.endswith(".exit") else "still going\n"

    with (
        patch("app.core.workspaced_client.read_workspace_file", side_effect=read),
        patch.object(_bash_wait, "_POLL_SECONDS", 0.05),
    ):
        out = await REGISTRY.get("bash_wait").execute(ctx, {"bg_id": "abc", "max_wait_sec": 1})

    assert "status: running" in out and "bash_wait again" in out
    assert proc.status == "running"
    assert run_clock.effective_deadline(0.0, ctx.task_id) >= 0.9
    run_clock.reset(ctx.task_id)


@pytest.mark.asyncio
async def test_bash_wait_dead_pid_without_exit_file_is_failed():
    ctx, _ = _ctx(_proc())

    async def read(rid, path):
        return None

    with (
        patch("app.core.workspaced_client.read_workspace_file", side_effect=read),
        patch("app.core.workspaced_client.exec_in_workspace", new_callable=AsyncMock,
              return_value={"exit_code": 1, "stdout": "", "stderr": ""}),
        patch.object(_bash_wait, "_POLL_SECONDS", 0.05),
        patch.object(_bash_wait, "_LIVENESS_EVERY_SECONDS", 0.0),
    ):
        out = await REGISTRY.get("bash_wait").execute(ctx, {"bg_id": "abc", "max_wait_sec": 30})

    assert "status: failed" in out


@pytest.mark.asyncio
async def test_bash_wait_unknown_bg_id():
    ctx, _ = _ctx(None)
    out = await REGISTRY.get("bash_wait").execute(ctx, {"bg_id": "nope"})
    assert out.startswith("Error: no background process")


@pytest.mark.asyncio
async def test_run_clock_accrues_while_the_block_runs_not_after():
    import asyncio

    tid = uuid4()
    run_clock.reset(tid)
    async with run_clock.accruing(tid, interval=0.05):
        await asyncio.sleep(0.3)
        mid = run_clock.effective_deadline(0.0, tid)
    assert mid >= 0.2  # already credited before the block finished
    assert run_clock.effective_deadline(0.0, tid) >= 0.28
    run_clock.reset(tid)


@pytest.mark.asyncio
async def test_bash_wait_stops_when_the_run_is_no_longer_current():
    ctx, _ = _ctx(_proc())
    ctx.run_id = "run-1"

    async def read(rid, path):
        return None if path.endswith(".exit") else "x\n"

    with (
        patch("app.core.workspaced_client.read_workspace_file", side_effect=read),
        patch("app.core.task_runtime.is_run_current", AsyncMock(return_value=False)),
        patch.object(_bash_wait, "_POLL_SECONDS", 0.05),
        patch.object(_bash_wait, "_FENCE_EVERY_SECONDS", 0.0),
    ):
        out = await REGISTRY.get("bash_wait").execute(ctx, {"bg_id": "abc", "max_wait_sec": 300})

    assert "stopped or paused" in out
