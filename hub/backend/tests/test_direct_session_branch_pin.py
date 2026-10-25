"""Branch pinning for direct sessions and for "Pull from Git".

Follow-up to internal design notes §0.1 — two paths ignored the branch
model entirely:

  * a direct session (an ``@nickname`` mention, or a ChatMaster chat) never
    went through task dispatch, so agent tool writes auto-committed onto
    whatever branch the previous dispatch left checked out —
    ``room_git_flow.ensure_session_branch`` pins it, except when a *live* task
    still owns the working tree;
  * ``git_fetch`` merged the remote into the current HEAD, so a sync fired
    while a task session held ``task/<id>`` checked out landed the remote
    repository on that task branch — ``fetch_workspace_from_git`` now pins the
    landing branch to the room working branch.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import room_git_flow
from app.core.git_sync import fetch_workspace_from_git
from app.models.task import TaskState


class _FakeDB:
    """Minimal db double: ``execute`` returns whatever was staged."""

    def __init__(self, scalar_result=None):
        self.added: list[object] = []
        self._scalar_result = scalar_result

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def execute(self, _stmt):
        result = self._scalar_result
        return SimpleNamespace(scalar_one_or_none=lambda: result)


def _head(monkeypatch, branch: str | None):
    monkeypatch.setattr(room_git_flow, "_current_branch", AsyncMock(return_value=branch))


# ── ensure_session_branch ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_without_task_is_pinned_to_working_branch(monkeypatch):
    """The bug this closes: HEAD left on a finished task branch, so writes from
    an @-mention silently landed there instead of on ``dev``."""
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    _head(monkeypatch, f"task/{uuid4()}")
    create = AsyncMock(return_value={"ok": True, "branch": "dev"})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    session = SimpleNamespace(id=uuid4())  # user_agent_p2 / user_cm_*: no task_id
    db = _FakeDB(scalar_result=SimpleNamespace(state=TaskState.COMPLETED))

    branch = await room_git_flow.ensure_session_branch("room-1", session, db)

    assert branch == "dev"
    create.assert_awaited_once_with("room-1", "dev")


@pytest.mark.asyncio
async def test_pin_is_a_no_op_when_head_is_already_right(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    _head(monkeypatch, "dev")
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    branch = await room_git_flow.ensure_session_branch(
        "room-1", SimpleNamespace(id=uuid4()), _FakeDB()
    )

    assert branch == "dev"
    create.assert_not_awaited()  # no needless checkout of the shared work tree


@pytest.mark.asyncio
async def test_session_with_task_reuses_the_task_branch(monkeypatch):
    """A ``user_agent_p1`` session continues the work of its task, so it
    belongs on the task branch — where Gate 1 picks it up on merge."""
    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=None, branch_name=None)
    ensure_task = AsyncMock(return_value=f"task/{tid}")
    monkeypatch.setattr(room_git_flow, "ensure_task_branch", ensure_task)
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    session = SimpleNamespace(id=uuid4(), task_id=tid)
    db = _FakeDB(scalar_result=task)

    branch = await room_git_flow.ensure_session_branch("room-1", session, db)

    assert branch == f"task/{tid}"
    ensure_task.assert_awaited_once_with("room-1", task, db)
    create.assert_not_awaited()  # no second, conflicting checkout


@pytest.mark.asyncio
async def test_session_with_dangling_task_id_falls_back_to_working_branch(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    _head(monkeypatch, "some-branch")
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    session = SimpleNamespace(id=uuid4(), task_id=uuid4())
    branch = await room_git_flow.ensure_session_branch("room-1", session, _FakeDB(None))

    assert branch == "dev"
    create.assert_awaited_once_with("room-1", "dev")


@pytest.mark.asyncio
async def test_running_task_keeps_the_working_tree(monkeypatch):
    """A live task holds the single working tree — switching it away mid-edit
    would strand the work of that task, so the pin stands down."""
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    _head(monkeypatch, f"task/{uuid4()}")
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    db = _FakeDB(scalar_result=SimpleNamespace(state=TaskState.IN_PROGRESS))
    branch = await room_git_flow.ensure_session_branch(
        "room-1", SimpleNamespace(id=uuid4()), db
    )

    assert branch is None
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_awaiting_review_does_not_hold_the_tree(monkeypatch):
    """A task parked in the review gate is not editing anything: its branch is
    already complete, so HEAD is free to move back to ``dev``."""
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    _head(monkeypatch, f"task/{uuid4()}")
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    db = _FakeDB(scalar_result=SimpleNamespace(state=TaskState.WAITING_REVIEW))
    branch = await room_git_flow.ensure_session_branch(
        "room-1", SimpleNamespace(id=uuid4()), db
    )

    assert branch == "dev"
    create.assert_awaited_once_with("room-1", "dev")


@pytest.mark.asyncio
async def test_failed_checkout_reports_none(monkeypatch):
    """A failed checkout must be visible to the caller rather than silently
    leaving HEAD on a stale branch."""
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    _head(monkeypatch, "stale")
    monkeypatch.setattr(
        "app.core.workspaced_client.create_workspace_branch",
        AsyncMock(return_value={"ok": False, "branch": "stale"}),
    )

    session = SimpleNamespace(id=uuid4())
    assert await room_git_flow.ensure_session_branch("room-1", session, _FakeDB()) is None


# ── fetch_workspace_from_git ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fetch_defaults_target_to_the_room_working_branch():
    git_config = {"repo_url": "https://github.com/o/r", "branch": "main"}
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {"ok": True, "conflict": False, "branch": "dev"}
        result = await fetch_workspace_from_git(
            room_id=uuid4(), git_config=git_config, access_token="t"
        )

    assert mf.call_args.kwargs["branch"] == "main"        # remote side
    assert mf.call_args.kwargs["target_branch"] == "dev"  # local landing side
    assert result["synced"] == 1


@pytest.mark.asyncio
async def test_fetch_honours_a_custom_working_branch():
    git_config = {
        "repo_url": "https://github.com/o/r",
        "branch": "trunk",
        "working_branch": "integration",
    }
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {"ok": True, "conflict": False}
        await fetch_workspace_from_git(room_id=uuid4(), git_config=git_config)

    assert mf.call_args.kwargs["target_branch"] == "integration"


@pytest.mark.asyncio
async def test_fetch_target_can_be_overridden_for_the_release_branch():
    """Stage 7 catches ``main`` up after a merged PR — that fetch must land on
    ``main``, not on the working branch."""
    git_config = {"repo_url": "https://github.com/o/r", "branch": "main"}
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {"ok": True, "conflict": False}
        await fetch_workspace_from_git(
            room_id=uuid4(), git_config=git_config, target_branch="main"
        )

    assert mf.call_args.kwargs["target_branch"] == "main"
