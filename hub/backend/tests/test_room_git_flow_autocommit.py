"""``room_git_flow.commit_working_tree`` — fold bash-only work into the task
branch before Gate 1 checks and before the merge.

bash (unlike write_file) auto-commits nothing, so without this the work never
enters the task branch and is invisible to checks, risk classification and the
promotion flow.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow


class _FakeDB:
    def add(self, obj):  # pragma: no cover - trivial
        pass

    async def flush(self):  # pragma: no cover - trivial
        pass

    async def commit(self):  # pragma: no cover - trivial
        pass


def _task(branch=None):
    return SimpleNamespace(
        id=uuid4(),
        title="T",
        branch_name=branch or f"task/{uuid4()}",
        ci_status=None,
        ci_log_key=None,
        goal_id=None,
    )


@pytest.mark.asyncio
async def test_commits_and_emits_when_head_on_task_branch(monkeypatch):
    task = _task()
    monkeypatch.setattr(room_git_flow, "_current_branch", AsyncMock(return_value=task.branch_name))
    commit = AsyncMock(return_value={"sha": "abc123", "committed": True})
    monkeypatch.setattr("app.core.workspaced_client.git_commit_workspace", commit)
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    sha = await room_git_flow.commit_working_tree(str(uuid4()), task, _FakeDB(), reason="x")

    assert sha == "abc123"
    commit.assert_awaited_once()
    assert emit.await_args.kwargs["event_type"] == "FILE_CHANGED"
    assert emit.await_args.kwargs["metadata"]["source"] == "auto_commit"


@pytest.mark.asyncio
async def test_skips_when_head_on_other_branch(monkeypatch):
    task = _task()
    monkeypatch.setattr(room_git_flow, "_current_branch", AsyncMock(return_value="dev"))
    commit = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.git_commit_workspace", commit)
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    sha = await room_git_flow.commit_working_tree(str(uuid4()), task, _FakeDB(), reason="x")

    assert sha is None
    commit.assert_not_awaited()
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_event_when_nothing_to_commit(monkeypatch):
    task = _task()
    monkeypatch.setattr(room_git_flow, "_current_branch", AsyncMock(return_value=task.branch_name))
    monkeypatch.setattr(
        "app.core.workspaced_client.git_commit_workspace",
        AsyncMock(return_value={"sha": None, "committed": False}),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    sha = await room_git_flow.commit_working_tree(str(uuid4()), task, _FakeDB(), reason="x")

    assert sha is None
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit_transport_error_is_swallowed(monkeypatch):
    task = _task()
    monkeypatch.setattr(room_git_flow, "_current_branch", AsyncMock(return_value=task.branch_name))
    monkeypatch.setattr(
        "app.core.workspaced_client.git_commit_workspace",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    sha = await room_git_flow.commit_working_tree(str(uuid4()), task, _FakeDB(), reason="x")

    assert sha is None
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_task_branch_commits_pending_tree_first(monkeypatch):
    task = _task()
    monkeypatch.setattr(room_git_flow, "_task_base_branch", AsyncMock(return_value="dev"))
    ct = AsyncMock(return_value="sha1")
    monkeypatch.setattr(room_git_flow, "commit_working_tree", ct)
    merge = AsyncMock(return_value={"ok": True, "sha": "mergesha"})
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", merge)

    out = await room_git_flow.merge_task_branch(str(uuid4()), task, _FakeDB())

    ct.assert_awaited_once()
    # refresh (base → task branch) + the real merge back into the base.
    assert merge.await_count == 2
    assert out["ok"] is True


@pytest.mark.asyncio
async def test_run_task_checks_commits_pending_tree_first(monkeypatch):
    task = _task()
    ct = AsyncMock(return_value="sha1")
    monkeypatch.setattr(room_git_flow, "commit_working_tree", ct)
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(return_value={"ran": False, "passed": True, "results": []}),
    )
    monkeypatch.setattr("app.core.events.emit_event", AsyncMock())

    await room_git_flow.run_task_checks(str(uuid4()), task, _FakeDB())

    ct.assert_awaited_once()
