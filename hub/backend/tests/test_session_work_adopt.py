"""Stage 2 of internal design notes — the branch becomes a task.

Once a dialogue's ``session/<id>`` branch actually carries commits, the work is
written down as a ``Task(origin="direct_session")``: no spec, no approval, no
dispatch of its own, title and description taken straight from the
conversation. The branch is then renamed to the canonical ``task/<id>``.

Covers ``session_work``:
  * ``derive_task_title`` / ``derive_task_description`` — what the board shows;
  * ``adopt_session_branch_as_task`` — the "nothing happened" case, the happy
    path, the rename (and its failure), snapshot bookkeeping, and the silent
    ``TASK_CREATED``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow, session_work
from app.models.task import DispatchStatus, Task, TaskState


class _FakeDB:
    def __init__(self, scalar_result=None):
        self.added: list[object] = []
        self._scalar_result = scalar_result

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def get(self, _model, _pk):
        return SimpleNamespace(nickname="Frontender")

    async def execute(self, _stmt):
        result = self._scalar_result
        return SimpleNamespace(scalar_one_or_none=lambda: result)


def _session(**kwargs):
    base = {
        "id": uuid4(),
        "room_id": uuid4(),
        "kind": "user_agent_p2",
        "agent_id": uuid4(),
        "goal_id": None,
        "context_snapshot": {},
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _with_work_branch(session, request="make the padding like in the reference"):
    session.context_snapshot = {
        session_work.SNAPSHOT_WORK_BRANCH: f"session/{session.id}",
        session_work.SNAPSHOT_WORK_REQUEST: request,
    }
    return session


def _patch_adopt(monkeypatch, *, diff="1 file changed, 6 insertions(+)", rename=None):
    monkeypatch.setattr(room_git_flow, "_load_room", AsyncMock(return_value=SimpleNamespace(git_config={})))
    commit = AsyncMock(return_value=None)
    monkeypatch.setattr(room_git_flow, "commit_branch_worktree", commit)
    monkeypatch.setattr(
        "app.core.workspaced_client.git_diff_workspace",
        AsyncMock(return_value={"diff": diff}),
    )
    rename = rename or AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.rename_workspace_branch", rename)
    events = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", events)
    return commit, rename, events


# ── title / description ─────────────────────────────────────────────────────

def test_title_drops_the_mention_and_keeps_the_ask():
    title = session_work.derive_task_title(
        "@Frontender fix the padding in the article. Everything is pressed to the left now"
    )
    assert title == "fix the padding in the article"


def test_title_is_cut_on_a_word_boundary():
    title = session_work.derive_task_title("make " + "very " * 40 + "nicely")
    assert len(title) <= 81  # 80 + ellipsis
    assert title.endswith("…")
    assert not title.endswith(" …")


def test_title_falls_back_when_there_is_no_request():
    assert session_work.derive_task_title(None) == "Follow-up from the dialogue"
    assert session_work.derive_task_title("   ") == "Follow-up from the dialogue"


def test_description_carries_the_ask_and_the_result():
    desc = session_work.derive_task_description("fix the padding", "Done: fix in style.css")
    assert "fix the padding" in desc
    assert "**Agent's result:**" in desc
    assert "Done: fix in style.css" in desc


def test_description_without_a_reply_is_just_the_ask():
    desc = session_work.derive_task_description("fix the padding", None)
    assert desc == "fix the padding"


# ── adopt: nothing to record ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dialogue_without_a_branch_records_nothing(monkeypatch):
    _, _, events = _patch_adopt(monkeypatch)
    db = _FakeDB()

    assert await session_work.adopt_session_branch_as_task(_session(), db) is None
    assert db.added == []
    events.assert_not_awaited()


@pytest.mark.asyncio
async def test_branch_without_commits_records_nothing(monkeypatch):
    """A turn that opened a branch and then wrote nothing: no task, no event,
    and the empty branch is discarded and forgotten."""
    commit, rename, events = _patch_adopt(monkeypatch, diff="")
    discard = AsyncMock(return_value={"ok": True, "deleted": True})
    monkeypatch.setattr("app.core.workspaced_client.git_discard_branch_workspace", discard)
    session = _with_work_branch(_session())
    db = _FakeDB()

    assert await session_work.adopt_session_branch_as_task(session, db) is None
    commit.assert_awaited_once()  # bash-only work folded in before the check
    rename.assert_not_awaited()
    events.assert_not_awaited()
    discard.assert_awaited_once_with(str(session.room_id), f"session/{session.id}")
    assert session_work.SNAPSHOT_WORK_BRANCH not in session.context_snapshot


@pytest.mark.asyncio
async def test_refused_discard_keeps_the_recorded_branch(monkeypatch):
    _patch_adopt(monkeypatch, diff="")
    monkeypatch.setattr(
        "app.core.workspaced_client.git_discard_branch_workspace",
        AsyncMock(return_value={"ok": False, "deleted": False}),
    )
    session = _with_work_branch(_session())

    assert await session_work.adopt_session_branch_as_task(session, _FakeDB()) is None
    assert session.context_snapshot[session_work.SNAPSHOT_WORK_BRANCH] == f"session/{session.id}"


@pytest.mark.asyncio
async def test_unreadable_workspace_records_nothing_and_keeps_the_branch(monkeypatch):
    _patch_adopt(monkeypatch)
    monkeypatch.setattr(
        "app.core.workspaced_client.git_diff_workspace",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )
    discard = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.git_discard_branch_workspace", discard)
    db = _FakeDB()

    assert await session_work.adopt_session_branch_as_task(_with_work_branch(_session()), db) is None
    assert db.added == []
    discard.assert_not_awaited()  # a failed diff is not proof the branch is empty


@pytest.mark.asyncio
async def test_already_adopted_dialogue_returns_the_same_task(monkeypatch):
    """The second writing turn of a burst must not open a second task."""
    tid = uuid4()
    open_task = SimpleNamespace(
        id=tid, state=TaskState.WAITING_REVIEW, merge_commit=None, branch_name=f"task/{tid}"
    )
    _, _, events = _patch_adopt(monkeypatch)
    session = _session(context_snapshot={session_work.SNAPSHOT_TASK_ID: str(tid)})

    result = await session_work.adopt_session_branch_as_task(
        session, _FakeDB(scalar_result=open_task)
    )

    assert result is open_task
    events.assert_not_awaited()


# ── adopt: the happy path ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_work_becomes_a_task_without_a_spec(monkeypatch):
    _, rename, events = _patch_adopt(monkeypatch)
    session = _with_work_branch(_session(), request="@Frontender fix the padding in the article")
    db = _FakeDB()

    task = await session_work.adopt_session_branch_as_task(
        session, db, agent_reply="Done: margin in style.css"
    )

    assert isinstance(task, Task)
    assert task.title == "fix the padding in the article"
    assert "Done: margin in style.css" in task.description
    assert task.origin == "direct_session"
    assert task.origin_session_id == session.id
    assert task.assigned_agent_id == session.agent_id
    assert task.goal_id is None                      # a task without a Goal
    assert task.state == TaskState.IN_PROGRESS
    # the dialogue owns the tree until Stage 3 finalizes it — the board must
    # not start a second executor on the same branch meanwhile
    assert task.dispatch_status == DispatchStatus.DISPATCHED
    assert task.branch_name == f"task/{task.id}"
    rename.assert_awaited_once_with(str(session.room_id), f"session/{session.id}", f"task/{task.id}")


@pytest.mark.asyncio
async def test_session_goal_is_inherited(monkeypatch):
    """A dialogue held inside a goal keeps its work in that goal's branch —
    ``ensure_task_branch`` bases the task on ``goal/<id>``."""
    _patch_adopt(monkeypatch)
    goal_id = uuid4()
    session = _with_work_branch(_session(goal_id=goal_id))

    task = await session_work.adopt_session_branch_as_task(session, _FakeDB())

    assert task.goal_id == goal_id


@pytest.mark.asyncio
async def test_failed_rename_keeps_the_dialogue_branch(monkeypatch):
    """The rename is cosmetic: every gate reads ``task.branch_name``, so a
    refusal must not cost the work."""
    _patch_adopt(monkeypatch, rename=AsyncMock(return_value={"ok": False}))
    session = _with_work_branch(_session())

    task = await session_work.adopt_session_branch_as_task(session, _FakeDB())

    assert task.branch_name == f"session/{session.id}"


@pytest.mark.asyncio
async def test_rename_outage_is_not_fatal(monkeypatch):
    _patch_adopt(monkeypatch, rename=AsyncMock(side_effect=RuntimeError("boom")))
    session = _with_work_branch(_session())

    task = await session_work.adopt_session_branch_as_task(session, _FakeDB())

    assert task.branch_name == f"session/{session.id}"


@pytest.mark.asyncio
async def test_snapshot_moves_from_branch_to_task(monkeypatch):
    """After adoption the dialogue points at its task, not at a branch name
    that no longer exists — and the next burst captures a fresh request."""
    _patch_adopt(monkeypatch)
    session = _with_work_branch(_session())

    task = await session_work.adopt_session_branch_as_task(session, _FakeDB())

    assert session.context_snapshot[session_work.SNAPSHOT_TASK_ID] == str(task.id)
    assert session_work.SNAPSHOT_WORK_BRANCH not in session.context_snapshot
    assert session_work.SNAPSHOT_WORK_REQUEST not in session.context_snapshot


@pytest.mark.asyncio
async def test_task_created_is_silent_in_chat(monkeypatch):
    """The board refreshes on the event; the chat card for a dialogue burst is
    the Gate 1 result, not the bookkeeping."""
    _, _, events = _patch_adopt(monkeypatch)
    session = _with_work_branch(_session())

    task = await session_work.adopt_session_branch_as_task(session, _FakeDB())

    events.assert_awaited_once()
    kwargs = events.await_args.kwargs
    assert kwargs["event_type"] == "TASK_CREATED"
    assert kwargs["task_id"] == task.id
    assert kwargs["metadata"]["silent"] is True
    assert kwargs["metadata"]["origin"] == "direct_session"
    assert kwargs["metadata"]["session_id"] == str(session.id)


@pytest.mark.asyncio
async def test_the_request_is_read_from_the_snapshot_when_not_passed(monkeypatch):
    """The burst may have started several turns ago — the request that opened
    it was recorded then."""
    _patch_adopt(monkeypatch)
    session = _with_work_branch(_session(), request='rewrite the "About us" block')

    task = await session_work.adopt_session_branch_as_task(session, _FakeDB())

    assert task.title == 'rewrite the "About us" block'
