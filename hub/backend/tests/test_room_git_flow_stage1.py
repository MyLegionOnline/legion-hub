"""Stage 1 of internal design notes — task / goal branches (hub side).

Covers ``room_git_flow``:
  * branch-name helpers (``task_branch_name`` / ``goal_branch_name`` / ``_slug``);
  * ``ensure_task_branch`` picks ``working_branch`` as base, or the goal
    branch when the task belongs to a goal;
  * ``ensure_goal_branch`` is idempotent once ``goal.branch_name`` is set;
  * ``merge_task_branch`` targets the right base and raises ``GIT_CONFLICT``
    on a real conflict.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
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


# ── name helpers ────────────────────────────────────────────────────────────

def test_slug_ascii_and_empty():
    assert room_git_flow._slug("Add Login Page!") == "add-login-page"
    assert room_git_flow._slug("添加页面") == ""  # no ascii word chars
    assert room_git_flow._slug("  ") == ""


def test_task_branch_name():
    tid = uuid4()
    assert room_git_flow.task_branch_name(tid) == f"task/{tid}"


def test_goal_branch_name_with_and_without_slug():
    gid = uuid4()
    assert room_git_flow.goal_branch_name(gid, "Ship v2") == f"goal/{gid}-ship-v2"
    assert room_git_flow.goal_branch_name(gid, "目标") == f"goal/{gid}"


# ── ensure_task_branch ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ensure_task_branch_no_goal_uses_working_branch(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=None, branch_name=None)
    db = _FakeDB()

    branch = await room_git_flow.ensure_task_branch("room-1", task, db)

    assert branch == f"task/{tid}"
    create.assert_awaited_once_with("room-1", f"task/{tid}", base="dev")
    assert task.branch_name == f"task/{tid}"
    assert task in db.added


@pytest.mark.asyncio
async def test_ensure_task_branch_with_goal_branches_off_goal(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    gid = uuid4()
    goal = SimpleNamespace(id=gid, title="Ship v2", branch_name=None)
    monkeypatch.setattr(room_git_flow, "_load_goal", AsyncMock(return_value=goal))
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=gid, branch_name=None)
    db = _FakeDB()

    branch = await room_git_flow.ensure_task_branch("room-1", task, db)

    goal_branch = f"goal/{gid}-ship-v2"
    assert branch == f"task/{tid}"
    assert goal.branch_name == goal_branch
    create.assert_any_await("room-1", goal_branch, base="dev")
    create.assert_any_await("room-1", f"task/{tid}", base=goal_branch)


@pytest.mark.asyncio
async def test_ensure_goal_branch_idempotent(monkeypatch):
    create = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)
    goal = SimpleNamespace(id=uuid4(), title="x", branch_name="goal/existing")

    out = await room_git_flow.ensure_goal_branch("room-1", goal, _FakeDB())

    assert out == "goal/existing"
    create.assert_not_awaited()


# ── merge_task_branch ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_merge_task_branch_targets_working_branch(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    merge = AsyncMock(return_value={"ok": True, "conflict": False})
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", merge)

    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=None, branch_name=f"task/{tid}", title="T")
    out = await room_git_flow.merge_task_branch("room-1", task, _FakeDB())

    assert out["target"] == "dev"
    # 1) refresh: base merged into the task branch with a plain 3-way merge;
    # 2) the real merge of the task branch back into its base.
    assert merge.await_count == 2
    refresh_kwargs = merge.await_args_list[0].kwargs
    assert refresh_kwargs["target"] == f"task/{tid}"
    assert refresh_kwargs["theirs"] is False
    assert merge.await_args_list[0].args[1] == "dev"
    kwargs = merge.await_args_list[1].kwargs
    assert kwargs["target"] == "dev"
    assert kwargs["theirs"] is True


@pytest.mark.asyncio
async def test_merge_task_branch_targets_goal_branch(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    gid = uuid4()
    goal = SimpleNamespace(id=gid, title="g", branch_name=f"goal/{gid}-g")
    monkeypatch.setattr(room_git_flow, "_load_goal", AsyncMock(return_value=goal))
    merge = AsyncMock(return_value={"ok": True, "conflict": False})
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", merge)

    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=gid, branch_name=f"task/{tid}", title="T")
    out = await room_git_flow.merge_task_branch("room-1", task, _FakeDB())

    assert out["target"] == f"goal/{gid}-g"


@pytest.mark.asyncio
async def test_merge_task_branch_conflict_emits_git_conflict(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    monkeypatch.setattr(
        "app.core.workspaced_client.merge_workspace_branch",
        AsyncMock(return_value={"ok": False, "conflict": True}),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=None, branch_name=f"task/{tid}", title="T")
    out = await room_git_flow.merge_task_branch(str(uuid4()), task, _FakeDB())

    assert out["conflict"] is True
    emit.assert_awaited_once()
    assert emit.await_args.kwargs["event_type"] == "GIT_CONFLICT"


@pytest.mark.asyncio
async def test_merge_task_branch_stale_base_refresh_conflict_blocks_theirs_merge(monkeypatch):
    """A base that diverged from the branch must not be folded back with
    ``-X theirs`` — the pre-merge refresh has to stop and hand it to a human."""
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))

    calls: list[dict] = []

    async def merge(room_id, name, **kwargs):
        calls.append({"name": name, **kwargs})
        # First call is the base→branch refresh; report a real overlap.
        if kwargs.get("theirs") is False:
            return {"ok": False, "conflict": True}
        return {"ok": True, "conflict": False}

    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", merge)
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=None, branch_name=f"task/{tid}", title="T")
    out = await room_git_flow.merge_task_branch(str(uuid4()), task, _FakeDB())

    assert out["conflict"] is True
    # Only the refresh ran — the destructive `theirs=True` merge never happened.
    assert len(calls) == 1
    assert calls[0]["theirs"] is False
    emit.assert_awaited_once()
    assert emit.await_args.kwargs["event_type"] == "GIT_CONFLICT"
    assert emit.await_args.kwargs["metadata"]["stage"] == "base_refresh"
