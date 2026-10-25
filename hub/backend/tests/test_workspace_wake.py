"""A stopped workspace is started on demand by tools that exec in it.

Precondition for hibernation (internal design notes stage 7): the agent has no
tool to start the container, so ``bash`` (and ``workspace_service``) must wake
it through the quota-checked ``ensure_workspace_started`` — releasing the
loop's DB connection first — and explain quota failures in plain words.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.tools import REGISTRY, ToolContext
from app.core.tools import _workspace_wake as wake
from app.core.tools._workspace_service import WorkspaceServiceTool, WorkspaceServiceToolError
from app.core.workspace_lifecycle import WorkspaceStartError


def _ws(status: str):
    ws = MagicMock()
    ws.id = uuid4()
    ws.status = status
    ws.network_enabled = True
    ws.last_error = None
    return ws


def _ctx(ws) -> ToolContext:
    db = AsyncMock()
    db.add = MagicMock()

    def _result(_q):
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=ws)
        r.scalars.return_value.all.return_value = []
        return r

    db.execute = AsyncMock(side_effect=_result)
    return ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)


def _exec_patches(mock_exec):
    return (
        patch("app.core.workspaced_client.exec_in_workspace", mock_exec),
        patch(
            "app.core.workspaced_client.get_workspace_generation",
            new_callable=AsyncMock,
            return_value=None,
        ),
    )


@pytest.mark.asyncio
async def test_running_workspace_is_not_restarted():
    ws = _ws("running")
    ctx = _ctx(ws)
    with patch(
        "app.core.workspace_lifecycle.ensure_workspace_started", new_callable=AsyncMock
    ) as start:
        got, err = await wake.ensure_workspace_running(ctx)
    assert err is None and got is ws
    start.assert_not_awaited()


@pytest.mark.asyncio
async def test_bash_starts_stopped_workspace_after_releasing_db():
    ws = _ws("stopped")
    ctx = _ctx(ws)
    order: list[str] = []

    async def _release(_ctx, *, reason):
        order.append(f"release:{reason}")
        return True

    async def _start(room_id, db, *, background):
        assert room_id == ctx.room_id and db is ctx.db and background is False
        order.append("start")
        ws.status = "running"
        return {}

    mock_exec = AsyncMock(return_value={"exit_code": 0, "stdout": "ok", "stderr": ""})
    p_exec, p_gen = _exec_patches(mock_exec)
    with (
        patch.object(wake, "release_tool_db", side_effect=_release),
        patch("app.core.workspace_lifecycle.ensure_workspace_started", side_effect=_start),
        p_exec,
        p_gen,
    ):
        result = await REGISTRY.get("bash").execute(ctx, {"cmd": "git status"})

    assert order[:2] == ["release:bash/workspace_start", "start"]
    assert "exit_code: 0" in result
    mock_exec.assert_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["quota_exceeded", "monthly_hours_exceeded"])
async def test_bash_quota_error_is_explained_and_nothing_runs(code):
    ctx = _ctx(_ws("stopped"))
    mock_exec = AsyncMock()
    p_exec, p_gen = _exec_patches(mock_exec)
    with (
        patch(
            "app.core.workspace_lifecycle.ensure_workspace_started",
            side_effect=WorkspaceStartError(code, "Limit exhausted.", status_code=429),
        ),
        p_exec,
        p_gen,
    ):
        result = await REGISTRY.get("bash").execute(ctx, {"cmd": "npm test"})

    assert result.startswith("Error:")
    assert code in result and "Limit exhausted." in result
    assert "cannot change" in result and "do not retry" in result
    assert "read_file" in result  # points at what still works
    assert "Start it first" not in result
    mock_exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_bash_background_also_wakes_workspace():
    ws = _ws("stopped")
    ctx = _ctx(ws)

    async def _start(*_a, **_k):
        ws.status = "running"

    with (
        patch("app.core.workspace_lifecycle.ensure_workspace_started", side_effect=_start) as start,
        patch.object(REGISTRY.get("bash"), "_run_background", new_callable=AsyncMock,
                     return_value="bg_id: x") as run_bg,
    ):
        result = await REGISTRY.get("bash").execute(ctx, {"cmd": "npm run dev", "background": True})

    assert result == "bg_id: x"
    start.assert_awaited_once()
    run_bg.assert_awaited_once()


@pytest.mark.asyncio
async def test_waits_for_start_owned_by_another_caller(monkeypatch):
    ws = _ws("starting")
    ctx = _ctx(ws)
    monkeypatch.setattr(wake, "_POLL_INTERVAL_SEC", 0)
    reads = {"n": 0}
    orig = ctx.db.execute.side_effect

    def _result(q):
        reads["n"] += 1
        if reads["n"] >= 4:
            ws.status = "running"
        return orig(q)

    ctx.db.execute.side_effect = _result
    with patch("app.core.workspace_lifecycle.ensure_workspace_started", new_callable=AsyncMock):
        got, err = await wake.ensure_workspace_running(ctx)
    assert err is None and got is ws


@pytest.mark.asyncio
async def test_start_that_ends_in_error_reports_last_error():
    ws = _ws("stopped")
    ctx = _ctx(ws)

    async def _start(*_a, **_k):
        ws.status = "error"
        ws.last_error = "image pull failed"

    with patch("app.core.workspace_lifecycle.ensure_workspace_started", side_effect=_start):
        got, err = await wake.ensure_workspace_running(ctx)
    assert got is None
    assert "status='error'" in err and "image pull failed" in err


@pytest.mark.asyncio
async def test_unexpected_start_failure_becomes_tool_error():
    ctx = _ctx(_ws("stopped"))
    with patch(
        "app.core.workspace_lifecycle.ensure_workspace_started",
        side_effect=RuntimeError("workspaced unreachable"),
    ):
        got, err = await wake.ensure_workspace_running(ctx)
    assert got is None and err.startswith("Error:") and "workspaced unreachable" in err


@pytest.mark.asyncio
async def test_workspace_service_start_error_uses_error_code():
    """Regression: the old helper read ``exc.code`` and crashed with AttributeError."""
    ctx = _ctx(_ws("stopped"))
    with patch(
        "app.core.workspace_lifecycle.ensure_workspace_started",
        side_effect=WorkspaceStartError("quota_exceeded", "Limit.", status_code=429),
    ):
        with pytest.raises(WorkspaceServiceToolError) as info:
            await WorkspaceServiceTool()._ensure_workspace_started(ctx)
    text = str(info.value)
    assert "quota_exceeded" in text and not text.startswith("Error:")


@pytest.mark.asyncio
async def test_disabled_services_message_is_not_actionable_by_agent(monkeypatch):
    monkeypatch.delenv("WORKSPACE_SERVICES_ENABLED", raising=False)
    result = await WorkspaceServiceTool().execute(_ctx(_ws("running")), {"action": "list"})
    assert result.startswith("Error:")
    assert "cannot change" in result and "not a room variable" in result
    assert "background=true" in result
    assert "Set WORKSPACE_SERVICES_ENABLED" not in result
