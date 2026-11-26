"""internal design notes E2 — agent tools go to the task's own tree.

A task with ``task.worktree`` reads, writes, searches and commits in its worktree;
a task without one (and any non-task session) keeps the shared-tree behaviour.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow, task_trees
from app.core import workspaced_client as wsd
from app.core.tools.registry import ToolContext, ctx_tree


class _Scalar:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value


class _DB:
    """Answers ``select(Task.worktree)`` / ``select(Task)`` with one fixed task."""

    def __init__(self, task=None):
        self.task = task
        self.queries = 0

    async def execute(self, _stmt):
        self.queries += 1
        if self.task is None:
            return _Scalar(None)
        # tree_of_task_id selects the column; the git flow selects the row.
        if len(_stmt.selected_columns) == 1:
            return _Scalar(self.task.worktree)
        return _Scalar(self.task)

    async def commit(self):
        pass


def _task(worktree=None, state="in_progress"):
    return SimpleNamespace(
        id=uuid4(), room_id=uuid4(), worktree=worktree, state=state, branch_name=None
    )


def _ctx(task, db=None):
    return ToolContext(
        room_id=task.room_id, session_id=uuid4(), task_id=task.id, db=db or _DB(task)
    )


@pytest.mark.asyncio
async def test_ctx_tree_resolves_once_from_the_task():
    task = _task(worktree="/workspace/.legion/wt/x")
    db = _DB(task)
    ctx = _ctx(task, db)
    assert await ctx_tree(ctx) == task_trees.task8_of(task.id)
    assert await ctx_tree(ctx) == task_trees.task8_of(task.id)
    assert db.queries == 1


@pytest.mark.asyncio
async def test_ctx_tree_none_without_tree_or_task():
    task = _task(worktree=None)
    ctx = _ctx(task)
    assert await ctx_tree(ctx) is None
    # a negative answer is cached too
    assert ctx._tree_resolved is True
    no_task = ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=None, db=_DB())
    assert await ctx_tree(no_task) is None


def test_container_path():
    assert task_trees.tree_container_path("ab12cd34") == "/workspace/.legion/wt/ab12cd34"
    assert task_trees.tree_container_path("wt/ab12cd34") == "/workspace/.legion/wt/ab12cd34"


@pytest.mark.asyncio
async def test_read_file_reads_the_task_tree(monkeypatch):
    from app.core.tools._read_file import ReadFileTool

    task = _task(worktree="/workspace/.legion/wt/x")
    read = AsyncMock(return_value=None)
    monkeypatch.setattr(wsd, "read_workspace_file", read)
    out = await ReadFileTool().execute(_ctx(task), {"path": "a.txt"})
    assert "not found" in out.lower()
    assert read.await_args.kwargs.get("tree") == task_trees.task8_of(task.id)


@pytest.mark.asyncio
async def test_read_file_shared_tree_when_no_worktree(monkeypatch):
    from app.core.tools._read_file import ReadFileTool

    task = _task(worktree=None)
    read = AsyncMock(return_value=None)
    monkeypatch.setattr(wsd, "read_workspace_file", read)
    await ReadFileTool().execute(_ctx(task), {"path": "a.txt"})
    assert "tree" not in read.await_args.kwargs


def test_tree_for_path_keeps_saved_logs_shared():
    assert task_trees.tree_for_path("ab12cd34", ".legion/logs/x.log") is None
    assert task_trees.tree_for_path("ab12cd34", "/workspace/.legion/logs/x.log") is None
    assert task_trees.tree_for_path("ab12cd34", ".legion/services.json") == "ab12cd34"
    assert task_trees.tree_for_path("ab12cd34", "src/a.ts") == "ab12cd34"
    assert task_trees.tree_for_path(None, ".legion/logs/x.log") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [".legion/logs/e1.log", "/workspace/.legion/logs/e1.log"])
async def test_read_file_saved_bash_log_from_shared_workspace(monkeypatch, raw):
    # bash saves big outputs to /workspace/.legion/logs and tells the agent to
    # read_file('.legion/logs/<id>.log'); a task tree has no such dir.
    from app.core.tools._read_file import ReadFileTool

    task = _task(worktree="/workspace/.legion/wt/x")
    read = AsyncMock(return_value="line\n")
    monkeypatch.setattr(wsd, "read_workspace_file", read)
    out = await ReadFileTool().execute(_ctx(task), {"path": raw})
    assert "line" in out
    assert "tree" not in read.await_args.kwargs


@pytest.mark.asyncio
async def test_read_files_mixes_tree_and_saved_log(monkeypatch):
    from app.core.tools._read_files import ReadFilesTool

    task = _task(worktree="/workspace/.legion/wt/x")
    read = AsyncMock(return_value="x\n")
    monkeypatch.setattr(wsd, "read_workspace_file", read)
    await ReadFilesTool().execute(_ctx(task), {"paths": ["src/a.ts", ".legion/logs/e1.log"]})
    trees = [c.kwargs.get("tree") for c in read.await_args_list]
    assert trees == [task_trees.task8_of(task.id), None]


@pytest.mark.asyncio
async def test_write_file_commits_in_tree_and_does_not_pin(monkeypatch):
    from app.core import room_storage
    from app.core.tools._write_file import WriteFileTool

    task = _task(worktree="/workspace/.legion/wt/x")
    write = AsyncMock(return_value={"ok": True})
    pin = AsyncMock()
    monkeypatch.setattr(wsd, "write_workspace_file", write)
    monkeypatch.setattr(room_git_flow, "pin_task_head", pin)
    monkeypatch.setattr(room_storage, "room_storage_refusal", AsyncMock(return_value=None))
    out = await WriteFileTool().execute(_ctx(task), {"path": "src/a.txt", "content": "hi"})
    assert out.startswith("Written")
    assert write.await_args.kwargs["tree"] == task_trees.task8_of(task.id)
    pin.assert_not_awaited()


@pytest.mark.asyncio
async def test_write_file_old_mode_still_pins(monkeypatch):
    from app.core import room_storage
    from app.core.tools._write_file import WriteFileTool

    task = _task(worktree=None)
    write = AsyncMock(return_value={"ok": True})
    pin = AsyncMock()
    monkeypatch.setattr(wsd, "write_workspace_file", write)
    monkeypatch.setattr(room_git_flow, "pin_task_head", pin)
    monkeypatch.setattr(room_storage, "room_storage_refusal", AsyncMock(return_value=None))
    await WriteFileTool().execute(_ctx(task), {"path": "src/a.txt", "content": "hi"})
    pin.assert_awaited_once()
    assert "tree" not in write.await_args.kwargs


@pytest.mark.asyncio
async def test_git_diff_runs_in_the_task_tree(monkeypatch):
    from app.core.tools._git_diff import GitDiffTool

    task = _task(worktree="/workspace/.legion/wt/x")
    diff = AsyncMock(return_value={"diff": " a.txt | 1 +"})
    monkeypatch.setattr(wsd, "git_diff_workspace", diff)
    await GitDiffTool().execute(_ctx(task), {})
    assert diff.await_args.kwargs["tree"] == task_trees.task8_of(task.id)


@pytest.mark.asyncio
async def test_list_files_lists_the_task_tree(monkeypatch):
    from app.core.tools._list_files import GlobTool

    task = _task(worktree="/workspace/.legion/wt/x")
    tree = AsyncMock(return_value=[{"path": "a.txt", "mtime": 1}])
    monkeypatch.setattr(wsd, "get_workspace_tree", tree)
    out = await GlobTool().execute(_ctx(task), {"pattern": "**/*"})
    assert "a.txt" in out
    assert tree.await_args.kwargs["tree"] == task_trees.task8_of(task.id)


# ── git flow: nothing to pin, nothing to hold ────────────────────────────────

@pytest.mark.asyncio
async def test_pin_task_head_skips_a_task_with_a_tree(monkeypatch):
    task = _task(worktree="/workspace/.legion/wt/x")
    current = AsyncMock(return_value="dev")
    monkeypatch.setattr(room_git_flow, "_current_branch", current)
    got = await room_git_flow.pin_task_head(task.room_id, task.id, _DB(task))
    assert got == room_git_flow.task_branch_name(task.id)
    current.assert_not_awaited()  # the shared HEAD is not even looked at


@pytest.mark.asyncio
async def test_task_with_tree_does_not_hold_the_shared_head(monkeypatch):
    task = _task(worktree="/workspace/.legion/wt/x")
    branch = room_git_flow.task_branch_name(task.id)
    held = await room_git_flow._head_held_by_live_work(task.room_id, branch, _DB(task))
    assert held is False
    old = _task(worktree=None)
    held = await room_git_flow._head_held_by_live_work(
        old.room_id, room_git_flow.task_branch_name(old.id), _DB(old)
    )
    assert held is True


@pytest.mark.asyncio
async def test_commit_working_tree_commits_in_the_tree(monkeypatch):
    task = _task(worktree="/workspace/.legion/wt/x")
    commit = AsyncMock(return_value={"sha": None})
    monkeypatch.setattr(wsd, "git_commit_workspace", commit)
    current = AsyncMock(return_value="dev")  # shared HEAD elsewhere: irrelevant
    monkeypatch.setattr(room_git_flow, "_current_branch", current)
    await room_git_flow.commit_working_tree(task.room_id, task, _DB(task), reason="r")
    assert commit.await_args.kwargs["tree"] == task_trees.task8_of(task.id)
    current.assert_not_awaited()


@pytest.mark.asyncio
async def test_commit_working_tree_old_mode_needs_head_on_branch(monkeypatch):
    task = _task(worktree=None)
    commit = AsyncMock(return_value={"sha": None})
    monkeypatch.setattr(wsd, "git_commit_workspace", commit)
    monkeypatch.setattr(room_git_flow, "_current_branch", AsyncMock(return_value="dev"))
    assert await room_git_flow.commit_working_tree(task.room_id, task, _DB(task), reason="r") is None
    commit.assert_not_awaited()
