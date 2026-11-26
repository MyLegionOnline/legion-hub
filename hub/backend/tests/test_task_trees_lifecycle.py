"""internal design notes E1 — worktree lifecycle on the backend.

Covers ``core.task_trees`` and its hooks: ``ensure_task_branch`` (own tree vs.
checkout), the removal before a merge, and the watchdog sweep.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow, task_trees


class _FakeDB:
    def __init__(self, tasks=()):
        self.added = []
        self.commits = 0
        self._tasks = list(tasks)

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        self.commits += 1

    async def rollback(self):
        pass

    async def execute(self, _stmt):
        rows = self._tasks

        class _R:
            def scalars(self_inner):
                return SimpleNamespace(all=lambda: rows)

        return _R()


def _task(**kw):
    base = dict(id=uuid4(), room_id=uuid4(), goal_id=None, branch_name=None,
                worktree=None, state="in_progress")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("ROOM_TASK_WORKTREES", "1")


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("ROOM_TASK_WORKTREES", raising=False)
    assert task_trees.task_worktrees_enabled() is False
    monkeypatch.setenv("ROOM_TASK_WORKTREES", "true")
    assert task_trees.task_worktrees_enabled() is True


def test_task8_is_first_eight_hex():
    tid = uuid4()
    assert task_trees.task8_of(tid) == tid.hex[:8]


def test_wants_tree_rules(monkeypatch):
    monkeypatch.delenv("ROOM_TASK_WORKTREES", raising=False)
    assert task_trees.wants_task_tree(_task()) is False
    # an existing tree is always kept, flag or not
    assert task_trees.wants_task_tree(_task(worktree="/workspace/.legion/wt/x")) is True
    monkeypatch.setenv("ROOM_TASK_WORKTREES", "1")
    assert task_trees.wants_task_tree(_task()) is True
    # begun before the flag: a branch but no tree → old mode to the end
    assert task_trees.wants_task_tree(_task(branch_name="task/old")) is False


@pytest.mark.asyncio
async def test_flag_off_checks_out_as_before(monkeypatch):
    monkeypatch.delenv("ROOM_TASK_WORKTREES", raising=False)
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    create = AsyncMock(return_value={"ok": True})
    add = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)
    monkeypatch.setattr("app.core.workspaced_client.add_task_worktree", add)
    task = _task()

    branch = await room_git_flow.ensure_task_branch("room-1", task, _FakeDB())

    create.assert_awaited_once_with("room-1", f"task/{task.id}", base="dev")
    add.assert_not_awaited()
    assert task.worktree is None
    assert task.branch_name == branch


@pytest.mark.asyncio
async def test_flag_on_makes_tree_and_leaves_shared_head(flag_on, monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    create = AsyncMock()
    path = "/workspace/.legion/wt/abcd1234"
    add = AsyncMock(return_value={"ok": True, "created": True, "path": path})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)
    monkeypatch.setattr("app.core.workspaced_client.add_task_worktree", add)
    task = _task()

    branch = await room_git_flow.ensure_task_branch("room-1", task, _FakeDB())

    add.assert_awaited_once_with("room-1", task.id.hex[:8], f"task/{task.id}", base="dev")
    create.assert_not_awaited()  # the shared tree's HEAD is not switched
    assert task.worktree == path
    assert task.branch_name == branch


@pytest.mark.asyncio
async def test_goal_branch_is_the_tree_base(flag_on, monkeypatch):
    gid = uuid4()
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    monkeypatch.setattr(
        room_git_flow, "_load_goal",
        AsyncMock(return_value=SimpleNamespace(id=gid, title="G", branch_name="goal/x")),
    )
    add = AsyncMock(return_value={"ok": True, "created": True, "path": "/p"})
    monkeypatch.setattr("app.core.workspaced_client.add_task_worktree", add)
    task = _task(goal_id=gid)

    await room_git_flow.ensure_task_branch("room-1", task, _FakeDB())

    assert add.await_args.kwargs["base"] == "goal/x"


@pytest.mark.asyncio
async def test_refused_tree_falls_back_to_checkout(flag_on, monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)
    monkeypatch.setattr(
        "app.core.workspaced_client.add_task_worktree",
        AsyncMock(side_effect=RuntimeError("409")),
    )
    task = _task()

    await room_git_flow.ensure_task_branch("room-1", task, _FakeDB())

    create.assert_awaited_once()
    assert task.worktree is None


@pytest.mark.asyncio
async def test_resume_reuses_tree_even_with_flag_off(monkeypatch):
    monkeypatch.delenv("ROOM_TASK_WORKTREES", raising=False)
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    path = "/workspace/.legion/wt/abcd1234"
    add = AsyncMock(return_value={"ok": True, "created": False, "path": path})
    monkeypatch.setattr("app.core.workspaced_client.add_task_worktree", add)
    task = _task(worktree=path, branch_name="task/x")

    await room_git_flow.ensure_task_branch("room-1", task, _FakeDB())

    add.assert_awaited_once()
    assert task.worktree == path


@pytest.mark.asyncio
async def test_pre_flag_task_stays_in_shared_tree(flag_on, monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    create = AsyncMock(return_value={"ok": True})
    add = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)
    monkeypatch.setattr("app.core.workspaced_client.add_task_worktree", add)
    task = _task(branch_name="task/old")

    await room_git_flow.ensure_task_branch("room-1", task, _FakeDB())

    add.assert_not_awaited()
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_remove_clears_task_worktree(monkeypatch):
    remove = AsyncMock(return_value={"ok": True, "removed": True})
    monkeypatch.setattr("app.core.workspaced_client.remove_task_worktree", remove)
    task = _task(worktree="/p")

    assert await task_trees.remove_task_tree("r", task, _FakeDB()) is True

    remove.assert_awaited_once_with("r", task.id.hex[:8], force=False)
    assert task.worktree is None


@pytest.mark.asyncio
async def test_remove_keeps_record_when_workspaced_refuses(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.remove_task_worktree",
        AsyncMock(return_value={"ok": False, "removed": False, "error": "dirty"}),
    )
    task = _task(worktree="/p")
    assert await task_trees.remove_task_tree("r", task, _FakeDB()) is False
    assert task.worktree == "/p"


@pytest.mark.asyncio
async def test_remove_survives_transport_error(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.remove_task_worktree",
        AsyncMock(side_effect=RuntimeError("down")),
    )
    task = _task(worktree="/p")
    assert await task_trees.remove_task_tree("r", task, _FakeDB()) is False
    assert task.worktree == "/p"


@pytest.mark.asyncio
async def test_merge_removes_the_tree_only_after_a_clean_merge(monkeypatch):
    order = []

    async def _remove(room_id, task, db, **_):
        order.append("remove")
        task.worktree = None
        return True

    async def _merge(room_id, task, db):
        order.append("merge")
        return {"ok": True}

    monkeypatch.setattr(task_trees, "remove_task_tree", _remove)
    monkeypatch.setattr(room_git_flow, "_merge_task_branch", _merge)
    task = _task(worktree="/p", branch_name="task/x")

    res = await room_git_flow.merge_task_branch("room-1", task, _FakeDB())

    assert res == {"ok": True}
    assert order == ["merge", "remove"]


@pytest.mark.asyncio
async def test_a_conflicting_merge_keeps_the_tree(monkeypatch):
    remove = AsyncMock(return_value=True)
    monkeypatch.setattr(task_trees, "remove_task_tree", remove)
    monkeypatch.setattr(
        room_git_flow, "_merge_task_branch",
        AsyncMock(return_value={"ok": False, "conflict": True}),
    )
    task = _task(worktree="/p", branch_name="task/x")

    res = await room_git_flow.merge_task_branch("room-1", task, _FakeDB())

    assert res["conflict"] is True
    remove.assert_not_awaited()
    assert task.worktree == "/p"


def _merge_env(monkeypatch, *, tree):
    """Stub everything ``_merge_task_branch`` talks to; return the merge-call recorder."""
    calls = []

    async def _wsd_merge(room_id, name, **kw):
        calls.append({"name": name, **kw})
        return {"ok": True, "sha": "abc"}

    monkeypatch.setattr(room_git_flow, "_task_base_branch", AsyncMock(return_value="dev"))
    monkeypatch.setattr(room_git_flow, "commit_working_tree", AsyncMock(return_value=None))
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", _wsd_merge)
    monkeypatch.setattr(
        "app.core.remote_divergence.close_merged_pull_rework", AsyncMock(return_value=None)
    )
    return calls


@pytest.mark.asyncio
async def test_base_refresh_happens_inside_the_task_tree(monkeypatch):
    calls = _merge_env(monkeypatch, tree=True)
    task = _task(worktree="/p", branch_name="task/x", title="t")

    res = await room_git_flow._merge_task_branch("room-1", task, _FakeDB())

    assert res["ok"] is True
    refresh, fold = calls
    # base → task branch: in the tree, on the task's branch, plain 3-way
    assert (refresh["name"], refresh["target"], refresh["theirs"]) == ("dev", "task/x", False)
    assert refresh["tree"] == task.id.hex[:8]
    # task branch → base: in the shared tree (nobody works there any more)
    assert (fold["name"], fold["target"], fold["theirs"]) == ("task/x", "dev", True)
    assert "tree" not in fold


@pytest.mark.asyncio
async def test_old_mode_refresh_stays_in_the_shared_tree(monkeypatch):
    calls = _merge_env(monkeypatch, tree=False)
    task = _task(branch_name="task/x", title="t")

    await room_git_flow._merge_task_branch("room-1", task, _FakeDB())

    assert all("tree" not in c for c in calls)


@pytest.mark.asyncio
async def test_checks_run_in_the_task_tree_without_checkout(monkeypatch):
    seen = {}

    async def _run(room_id, **kw):
        seen.update(kw)
        return {"ran": False}

    monkeypatch.setattr(room_git_flow, "commit_working_tree", AsyncMock(return_value=None))
    monkeypatch.setattr(room_git_flow, "_resolve_checks_env", AsyncMock(return_value=({}, {})))
    monkeypatch.setattr("app.core.workspaced_client.run_workspace_checks", _run)
    task = _task(worktree="/p", branch_name="task/x", title="t")

    await room_git_flow.run_task_checks(str(uuid4()), task, _FakeDB(), emit_skipped=False)
    assert seen["tree"] == task.id.hex[:8]

    seen.clear()
    old = _task(branch_name="task/y", title="t")
    await room_git_flow.run_task_checks(str(uuid4()), old, _FakeDB(), emit_skipped=False)
    assert "tree" not in seen and seen["ref"] == "task/y"


@pytest.mark.asyncio
async def test_sweep_removes_only_terminal_tasks(monkeypatch):
    remove = AsyncMock(return_value={"ok": True, "removed": True})
    monkeypatch.setattr("app.core.workspaced_client.remove_task_worktree", remove)
    live = _task(worktree="/a", state="paused")
    review = _task(worktree="/b", state="waiting_review")
    done = _task(worktree="/c", state="completed")
    cancelled = _task(worktree="/d", state=SimpleNamespace(value="cancelled"))
    db = _FakeDB([live, review, done, cancelled])

    assert await task_trees.sweep_task_trees(db) == 2

    assert live.worktree == "/a" and review.worktree == "/b"
    assert done.worktree is None and cancelled.worktree is None
    assert remove.await_count == 2
