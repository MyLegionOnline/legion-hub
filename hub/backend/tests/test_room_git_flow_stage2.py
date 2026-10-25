"""Stage 2 of internal design notes — Gate 1 pre-merge checks (hub side).

Covers ``room_git_flow.run_task_checks``:
  * a room with no checks → ``passed`` / ``ran=False`` / ``CI_SKIPPED``,
    ``ci_status`` left None;
  * a transport error is fail-open (``passed`` + ``CI_SKIPPED``);
  * green checks → ``ci_status="passed"`` + ``CI_PASSED``;
  * red checks → ``ci_status="failed"`` + ``CI_FAILED``, log key recorded.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow


class _FakeDB:
    def __init__(self):
        self.added: list[object] = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass


def _task():
    return SimpleNamespace(
        id=uuid4(), title="T", branch_name=f"task/{uuid4()}",
        ci_status=None, ci_log_key=None,
    )


@pytest.fixture(autouse=True)
def _no_s3(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_archive_ci_log", AsyncMock(return_value="ci/key.jsonl"))


@pytest.mark.asyncio
async def test_no_checks_skips_and_leaves_ci_status_none(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(return_value={"ran": False, "passed": True, "results": []}),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    task = _task()
    out = await room_git_flow.run_task_checks(str(uuid4()), task, _FakeDB())

    assert out["passed"] is True and out["ran"] is False
    assert task.ci_status is None
    assert emit.await_args.kwargs["event_type"] == "CI_SKIPPED"


@pytest.mark.asyncio
async def test_transport_error_is_fail_open(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    task = _task()
    out = await room_git_flow.run_task_checks(str(uuid4()), task, _FakeDB())

    assert out["passed"] is True and out["ran"] is False
    assert emit.await_args.kwargs["event_type"] == "CI_SKIPPED"


@pytest.mark.asyncio
async def test_green_checks_mark_passed(monkeypatch):
    run = AsyncMock(return_value={
        "ran": True, "passed": True,
        "results": [{"name": "pytest", "exit_code": 0, "tail": "ok", "timed_out": False}],
    })
    monkeypatch.setattr("app.core.workspaced_client.run_workspace_checks", run)
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    task = _task()
    out = await room_git_flow.run_task_checks(str(uuid4()), task, _FakeDB())

    assert out["passed"] is True and out["ran"] is True
    assert task.ci_status == "passed"
    assert task.ci_log_key == "ci/key.jsonl"
    run.assert_awaited_once()
    assert run.await_args.kwargs["ref"] == task.branch_name
    assert emit.await_args.kwargs["event_type"] == "CI_PASSED"


@pytest.mark.asyncio
async def test_red_checks_mark_failed_and_emit_ci_failed(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(return_value={
            "ran": True, "passed": False,
            "results": [{"name": "pytest", "exit_code": 1, "tail": "E   assert 0", "timed_out": False}],
        }),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    task = _task()
    out = await room_git_flow.run_task_checks(str(uuid4()), task, _FakeDB())

    assert out["passed"] is False
    assert task.ci_status == "failed"
    assert task.ci_log_key == "ci/key.jsonl"
    kwargs = emit.await_args.kwargs
    assert kwargs["event_type"] == "CI_FAILED"
    assert "assert 0" in kwargs["content"]
