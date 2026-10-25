"""Stage 1 of internal design notes — the dialogue's own branch.

A direct `@nickname` session used to write straight onto ``dev``. Now the first
tool that could touch the working tree pins HEAD to ``session/<session_id>``
first, so the work can be adopted as a task later (Stage 2) instead of leaking
into the working branch unreviewed.

Covers:
  * ``tool_touches_worktree`` — default-deny classification of tool names;
  * ``ensure_worktree_branch`` — the just-in-time pin and what it records;
  * the new priority order in ``room_git_flow.ensure_session_branch``;
  * ``_live_session_owns_head`` — one dialogue does not yank the working tree
    out from under another.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow, session_work
from app.models.task import TaskState


class _FakeDB:
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


def _session(**kwargs):
    base = {
        "id": uuid4(),
        "room_id": "room-1",
        "kind": "user_agent_p2",
        "context_snapshot": {},
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _room(**git_config):
    return SimpleNamespace(git_config=git_config)


def _patch_git(monkeypatch, *, head=None, room=None, create=None, commit=None):
    """Stub the git surface ``ensure_worktree_branch`` reaches for."""
    monkeypatch.setattr(room_git_flow, "_current_branch", AsyncMock(return_value=head))
    monkeypatch.setattr(
        room_git_flow, "_load_room", AsyncMock(return_value=room if room is not None else _room())
    )
    create = create or AsyncMock(return_value={"ok": True})
    commit = commit or AsyncMock(return_value={"sha": None})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)
    monkeypatch.setattr("app.core.workspaced_client.git_commit_workspace", commit)
    return create, commit


# ── which tools force a branch ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "tool",
    ["grep", "glob", "read_file", "read_files", "git_diff", "repo_map", "web_search",
     "todo_write", "task_update", "bash_output", "ask_user"],
)
def test_reading_tools_do_not_need_a_branch(tool):
    assert session_work.tool_touches_worktree(tool) is False


@pytest.mark.parametrize(
    "tool", ["write_file", "edit_file", "bash", "screenshot", "workspace_service", "run_subtask"]
)
def test_writing_tools_need_a_branch(tool):
    assert session_work.tool_touches_worktree(tool) is True


def test_unknown_tools_are_treated_as_writers():
    """Default-deny: a tool added later lands on the safe side of this line
    rather than silently committing onto the working branch."""
    assert session_work.tool_touches_worktree("some_new_tool") is True


# ── ensure_worktree_branch ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chatmaster_session_is_left_alone(monkeypatch):
    """The ``cm`` role has no file-writing tools — no work to record, and no
    reason to change what ChatMaster chats do today."""
    create, _ = _patch_git(monkeypatch, head="dev")

    branch = await session_work.ensure_worktree_branch(
        _session(kind="user_cm_p2"), _FakeDB()
    )

    assert branch is None
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_room_can_opt_out(monkeypatch):
    create, _ = _patch_git(monkeypatch, head="dev", room=_room(direct_session_mode="off"))

    assert await session_work.ensure_worktree_branch(_session(), _FakeDB()) is None
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_bound_to_a_task_uses_the_task_branch(monkeypatch):
    tid = uuid4()
    task = SimpleNamespace(id=tid, goal_id=None, branch_name=None)
    _patch_git(monkeypatch, head="dev")
    ensure_task = AsyncMock(return_value=f"task/{tid}")
    monkeypatch.setattr(room_git_flow, "ensure_task_branch", ensure_task)

    branch = await session_work.ensure_worktree_branch(
        _session(task_id=tid), _FakeDB(scalar_result=task)
    )

    assert branch == f"task/{tid}"
    ensure_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_unfinished_dialogue_task_is_continued(monkeypatch):
    """A task this dialogue already opened (red checks, or waiting for review)
    is the current work: the next writing turn builds on its branch."""
    tid = uuid4()
    task = SimpleNamespace(
        id=tid, goal_id=None, branch_name=f"task/{tid}",
        state=TaskState.BLOCKED, merge_commit=None,
    )
    create, _ = _patch_git(monkeypatch, head="dev")
    ensure_task = AsyncMock(return_value=f"task/{tid}")
    monkeypatch.setattr(room_git_flow, "ensure_task_branch", ensure_task)

    session = _session(context_snapshot={session_work.SNAPSHOT_TASK_ID: str(tid)})
    branch = await session_work.ensure_worktree_branch(session, _FakeDB(scalar_result=task))

    assert branch == f"task/{tid}"
    ensure_task.assert_awaited_once()
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_merged_dialogue_task_starts_a_fresh_branch(monkeypatch):
    """Once the work is in ``dev``, the burst is over: the next edit is its own
    piece of work and gets its own branch off the working branch."""
    tid = uuid4()
    task = SimpleNamespace(
        id=tid, state=TaskState.COMPLETED, merge_commit="abc123", branch_name=f"task/{tid}"
    )
    create, _ = _patch_git(monkeypatch, head="dev")

    session = _session(context_snapshot={session_work.SNAPSHOT_TASK_ID: str(tid)})
    branch = await session_work.ensure_worktree_branch(session, _FakeDB(scalar_result=task))

    assert branch == f"session/{session.id}"
    create.assert_awaited_once_with("room-1", f"session/{session.id}", base="dev")


@pytest.mark.asyncio
async def test_first_write_opens_the_dialogue_branch_off_the_working_branch(monkeypatch):
    create, commit = _patch_git(monkeypatch, head="dev")
    session = _session()
    db = _FakeDB()

    branch = await session_work.ensure_worktree_branch(
        session, db, user_request="make the padding like in the reference"
    )

    assert branch == f"session/{session.id}"
    create.assert_awaited_once_with("room-1", f"session/{session.id}", base="dev")
    # what the dialogue is building on, and the request the task will be named
    # after — both read back in Stage 2
    assert session.context_snapshot[session_work.SNAPSHOT_WORK_BRANCH] == branch
    assert (
        session.context_snapshot[session_work.SNAPSHOT_WORK_REQUEST]
        == "make the padding like in the reference"
    )
    assert session in db.added
    # leaving `dev` with a dirty tree parks it there instead of dragging the
    # changes onto the new branch
    commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_working_branch_is_the_base(monkeypatch):
    create, _ = _patch_git(monkeypatch, head="integration", room=_room(working_branch="integration"))
    session = _session()

    await session_work.ensure_worktree_branch(session, _FakeDB())

    assert create.await_args.kwargs["base"] == "integration"


@pytest.mark.asyncio
async def test_second_writing_tool_in_the_same_turn_is_a_no_op(monkeypatch):
    session = _session()
    branch = f"session/{session.id}"
    create, commit = _patch_git(monkeypatch, head=branch)
    session.context_snapshot = {
        session_work.SNAPSHOT_WORK_BRANCH: branch,
        session_work.SNAPSHOT_WORK_REQUEST: "first request",
    }

    assert await session_work.ensure_worktree_branch(session, _FakeDB()) == branch
    create.assert_not_awaited()
    commit.assert_not_awaited()
    # the request that opened the burst is not overwritten by a later turn
    assert session.context_snapshot[session_work.SNAPSHOT_WORK_REQUEST] == "first request"


@pytest.mark.asyncio
async def test_refused_checkout_records_nothing(monkeypatch):
    """If git would not give us the branch, the dialogue must not believe it
    has one — Stage 2 would otherwise look for work that never existed."""
    create, _ = _patch_git(
        monkeypatch, head="dev", create=AsyncMock(return_value={"ok": False})
    )
    session = _session()

    assert await session_work.ensure_worktree_branch(session, _FakeDB()) is None
    assert session_work.SNAPSHOT_WORK_BRANCH not in session.context_snapshot


@pytest.mark.asyncio
async def test_workspace_outage_never_breaks_the_turn(monkeypatch):
    _patch_git(monkeypatch, head="dev", create=AsyncMock(side_effect=RuntimeError("boom")))
    session = _session()

    assert await session_work.ensure_worktree_branch(session, _FakeDB()) is None
    assert session_work.SNAPSHOT_WORK_BRANCH not in session.context_snapshot


# ── ensure_session_branch: priority order ───────────────────────────────────

@pytest.mark.asyncio
async def test_turn_start_returns_to_the_dialogue_branch(monkeypatch):
    """The turn-start pin must not drag a dialogue with unfinished work back to
    ``dev`` — the next turn continues where the previous one stopped."""
    session = _session()
    branch = f"session/{session.id}"
    session.context_snapshot = {session_work.SNAPSHOT_WORK_BRANCH: branch}
    monkeypatch.setattr(room_git_flow, "_current_branch", AsyncMock(return_value="dev"))
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    result = await room_git_flow.ensure_session_branch("room-1", session, _FakeDB())

    assert result == branch
    # checkout of an existing branch — never re-created off the working branch,
    # which would throw away the commits already on it
    create.assert_awaited_once_with("room-1", branch)


@pytest.mark.asyncio
async def test_turn_start_prefers_the_open_dialogue_task(monkeypatch):
    tid = uuid4()
    task = SimpleNamespace(
        id=tid, state=TaskState.WAITING_REVIEW, merge_commit=None, branch_name=f"task/{tid}"
    )
    session = _session(context_snapshot={
        session_work.SNAPSHOT_TASK_ID: str(tid),
        session_work.SNAPSHOT_WORK_BRANCH: f"session/{uuid4()}",
    })
    ensure_task = AsyncMock(return_value=f"task/{tid}")
    monkeypatch.setattr(room_git_flow, "ensure_task_branch", ensure_task)

    result = await room_git_flow.ensure_session_branch(
        "room-1", session, _FakeDB(scalar_result=task)
    )

    assert result == f"task/{tid}"
    ensure_task.assert_awaited_once()


@pytest.mark.asyncio
async def test_another_live_dialogue_keeps_the_working_tree(monkeypatch):
    """Same rule as a live task: the other conversation may be mid-edit."""
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    monkeypatch.setattr(
        room_git_flow, "_current_branch", AsyncMock(return_value=f"session/{uuid4()}")
    )
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    db = _FakeDB(scalar_result=SimpleNamespace(closed_at=None))
    result = await room_git_flow.ensure_session_branch("room-1", _session(), db)

    assert result is None
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_branch_of_a_closed_dialogue_does_not_hold_the_tree(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_working_branch", AsyncMock(return_value="dev"))
    monkeypatch.setattr(
        room_git_flow, "_current_branch", AsyncMock(return_value=f"session/{uuid4()}")
    )
    create = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", create)

    db = _FakeDB(scalar_result=SimpleNamespace(closed_at=datetime.now(timezone.utc)))
    result = await room_git_flow.ensure_session_branch("room-1", _session(), db)

    assert result == "dev"
    create.assert_awaited_once_with("room-1", "dev")


# ── _live_session_owns_head ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_own_branch_is_never_somebody_elses():
    sid = uuid4()
    db = _FakeDB(scalar_result=SimpleNamespace(closed_at=None))

    owned = await room_git_flow._live_session_owns_head(
        "room-1", f"session/{sid}", db, exclude_session_id=sid
    )
    assert owned is False


@pytest.mark.asyncio
async def test_non_session_branches_are_ignored():
    db = _FakeDB(scalar_result=SimpleNamespace(closed_at=None))
    assert await room_git_flow._live_session_owns_head("room-1", "dev", db) is False
    assert await room_git_flow._live_session_owns_head("room-1", "session/nope", db) is False
    assert await room_git_flow._live_session_owns_head("room-1", None, db) is False
