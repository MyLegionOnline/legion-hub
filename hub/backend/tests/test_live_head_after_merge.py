"""A gate merge must not strand a running task's agent on the merge target.

Prod 2026-11-16, goal ec3cc5aa: two tasks of one goal ran in parallel in the
room's single working tree. The writer's task finished first and its Gate 1
merge checked out the goal branch — and left HEAD there. The second task's
agent was still mid-session; every ``edit_file`` it made afterwards
auto-committed onto the goal branch. Its own ``task/<id>`` froze at the first
half of the work, so its merge hit a "divergence" with its own second half and
the divergence rework redid the whole task.

Two guards, both covered here:

* ``keep_live_head`` around every gate merge puts HEAD back on the branch a
  running task (or open dialogue) held before the merge;
* ``pin_task_head`` before a task's ``write_file`` / ``edit_file`` returns HEAD
  to that task's branch whatever moved it.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import room_git_flow
from app.models.task import TaskState


class _FakeDB:
    """``execute`` answers every task lookup from a ``{task_id: task}`` map."""

    def __init__(self, tasks: dict | None = None):
        self.tasks = tasks or {}
        self.added: list[object] = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def execute(self, stmt):
        params = stmt.compile().params
        wanted = next(iter(params.values()), None)
        task = self.tasks.get(str(wanted))
        return SimpleNamespace(scalar_one_or_none=lambda: task)


def _task(state: TaskState, **extra):
    tid = uuid4()
    return SimpleNamespace(
        id=tid, state=state, branch_name=f"task/{tid}", goal_id=None, title="t", **extra
    )


def _heads(monkeypatch, *branches):
    """``_current_branch`` answers these in order, then repeats the last one."""
    seq = list(branches)

    async def _current(_room_id):
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(room_git_flow, "_current_branch", _current)


@pytest.fixture
def create(monkeypatch):
    mock = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", mock)
    return mock


# ── keep_live_head around merge_task_branch ─────────────────────────────────

@pytest.mark.asyncio
async def test_merge_returns_head_to_a_sibling_task_still_running(monkeypatch, create):
    """The incident: writer's merge ran while Sara's task held HEAD."""
    running = _task(TaskState.IN_PROGRESS)
    finished = _task(TaskState.WAITING_REVIEW)
    db = _FakeDB({str(running.id): running, str(finished.id): finished})
    _heads(monkeypatch, running.branch_name, "goal/g-1")
    monkeypatch.setattr(
        room_git_flow, "_merge_task_branch", AsyncMock(return_value={"ok": True})
    )

    out = await room_git_flow.merge_task_branch("room-1", finished, db)

    assert out == {"ok": True}
    create.assert_awaited_once_with("room-1", running.branch_name)


@pytest.mark.asyncio
async def test_head_is_returned_after_a_conflicting_merge_too(monkeypatch, create):
    running = _task(TaskState.IN_PROGRESS)
    finished = _task(TaskState.WAITING_REVIEW)
    db = _FakeDB({str(running.id): running, str(finished.id): finished})
    _heads(monkeypatch, running.branch_name, finished.branch_name)
    monkeypatch.setattr(
        room_git_flow,
        "_merge_task_branch",
        AsyncMock(return_value={"ok": False, "conflict": True}),
    )

    await room_git_flow.merge_task_branch("room-1", finished, db)

    create.assert_awaited_once_with("room-1", running.branch_name)


@pytest.mark.asyncio
async def test_merged_branch_itself_is_not_checked_back_out(monkeypatch, create):
    task = _task(TaskState.IN_PROGRESS)  # e.g. merged straight from a live session
    db = _FakeDB({str(task.id): task})
    _heads(monkeypatch, task.branch_name, "goal/g-1")
    monkeypatch.setattr(room_git_flow, "_merge_task_branch", AsyncMock(return_value={"ok": True}))

    await room_git_flow.merge_task_branch("room-1", task, db)

    create.assert_not_awaited()


@pytest.mark.parametrize("state", [TaskState.COMPLETED, TaskState.CONTINUED, TaskState.CANCELLED])
@pytest.mark.asyncio
async def test_branch_of_a_task_nobody_works_on_is_left(monkeypatch, create, state):
    """A continued task's work moved to another branch — nobody writes on its own."""
    idle = _task(state)
    finished = _task(TaskState.WAITING_REVIEW)
    db = _FakeDB({str(idle.id): idle, str(finished.id): finished})
    _heads(monkeypatch, idle.branch_name, "goal/g-1")
    monkeypatch.setattr(room_git_flow, "_merge_task_branch", AsyncMock(return_value={"ok": True}))

    await room_git_flow.merge_task_branch("room-1", finished, db)

    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_checkout_when_head_did_not_move(monkeypatch, create):
    running = _task(TaskState.IN_PROGRESS)
    finished = _task(TaskState.WAITING_REVIEW)
    db = _FakeDB({str(running.id): running, str(finished.id): finished})
    _heads(monkeypatch, running.branch_name)
    monkeypatch.setattr(room_git_flow, "_merge_task_branch", AsyncMock(return_value={"ok": True}))

    await room_git_flow.merge_task_branch("room-1", finished, db)

    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_goal_promotion_returns_head_to_a_running_task(monkeypatch, create):
    """Prod 12:02 the same day: another goal's goal → dev merge moved HEAD to
    ``dev`` under the running rework task."""
    running = _task(TaskState.IN_PROGRESS)
    db = _FakeDB({str(running.id): running})
    _heads(monkeypatch, running.branch_name, "dev")
    monkeypatch.setattr(room_git_flow, "_load_room", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(
        room_git_flow, "get_flow_config", lambda _room: SimpleNamespace(working_branch="dev")
    )
    monkeypatch.setattr(
        "app.core.workspaced_client.merge_workspace_branch",
        AsyncMock(return_value={"ok": False, "conflict": True}),
    )
    monkeypatch.setattr("app.core.events.emit_event", AsyncMock())
    goal = SimpleNamespace(id=uuid4(), branch_name="goal/g-2", title="other goal")

    out = await room_git_flow._promote_goal_branch_to_dev(
        "room-1", goal, db, rid=uuid4(), log_prefix="test"
    )

    assert out["conflict"] is True
    create.assert_awaited_once_with("room-1", running.branch_name)


# ── pin_task_head before a task's write ─────────────────────────────────────

@pytest.mark.asyncio
async def test_pin_returns_a_running_task_to_its_branch(monkeypatch):
    """HEAD left on the goal branch: the write must still land on task/<id>."""
    task = _task(TaskState.IN_PROGRESS)
    db = _FakeDB({str(task.id): task})
    _heads(monkeypatch, "goal/g-1")
    ensure = AsyncMock(return_value=task.branch_name)
    monkeypatch.setattr(room_git_flow, "ensure_task_branch", ensure)

    branch = await room_git_flow.pin_task_head("room-1", task.id, db)

    assert branch == task.branch_name
    ensure.assert_awaited_once_with("room-1", task, db)


@pytest.mark.asyncio
async def test_pin_is_a_no_op_on_the_right_branch(monkeypatch):
    task = _task(TaskState.IN_PROGRESS)
    db = _FakeDB({str(task.id): task})
    _heads(monkeypatch, task.branch_name)
    ensure = AsyncMock()
    monkeypatch.setattr(room_git_flow, "ensure_task_branch", ensure)

    assert await room_git_flow.pin_task_head("room-1", task.id, db) == task.branch_name
    ensure.assert_not_awaited()


@pytest.mark.asyncio
async def test_pin_does_not_yank_another_running_task(monkeypatch):
    mine = _task(TaskState.IN_PROGRESS)
    other = _task(TaskState.IN_PROGRESS)
    db = _FakeDB({str(mine.id): mine, str(other.id): other})
    _heads(monkeypatch, other.branch_name)
    ensure = AsyncMock()
    monkeypatch.setattr(room_git_flow, "ensure_task_branch", ensure)

    assert await room_git_flow.pin_task_head("room-1", mine.id, db) is None
    ensure.assert_not_awaited()


@pytest.mark.asyncio
async def test_pin_skips_a_task_that_is_not_running(monkeypatch):
    task = _task(TaskState.COMPLETED)
    db = _FakeDB({str(task.id): task})
    _heads(monkeypatch, "dev")
    ensure = AsyncMock()
    monkeypatch.setattr(room_git_flow, "ensure_task_branch", ensure)

    assert await room_git_flow.pin_task_head("room-1", task.id, db) is None
    ensure.assert_not_awaited()


@pytest.mark.asyncio
async def test_pin_never_raises(monkeypatch):
    db = MagicMock()
    db.execute = AsyncMock(side_effect=RuntimeError("db down"))

    assert await room_git_flow.pin_task_head("room-1", uuid4(), db) is None


# ── the tools call the pin ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_file_pins_the_task_branch_before_writing(monkeypatch):
    from app.core import room_storage
    from app.core import workspaced_client as wsd
    from app.core.tools._write_file import WriteFileTool

    order: list[str] = []
    monkeypatch.setattr(room_storage, "room_storage_refusal", AsyncMock(return_value=None))
    monkeypatch.setattr(
        room_git_flow, "pin_task_head", AsyncMock(side_effect=lambda *a: order.append("pin"))
    )
    monkeypatch.setattr(
        wsd, "write_workspace_file", AsyncMock(side_effect=lambda *a, **k: order.append("write"))
    )
    tid = uuid4()
    ctx = SimpleNamespace(room_id=uuid4(), db=MagicMock(), session_id="s", task_id=tid)

    await WriteFileTool().execute(ctx, {"path": "a.txt", "content": "x"})

    assert order == ["pin", "write"]
    room_git_flow.pin_task_head.assert_awaited_once_with(ctx.room_id, tid, ctx.db)


@pytest.mark.asyncio
async def test_edit_file_pins_before_it_reads(monkeypatch):
    from app.core import workspaced_client as wsd
    from app.core.tools import _session_state
    from app.core.tools._edit_file import EditFileTool

    order: list[str] = []
    monkeypatch.setattr(_session_state, "was_read", lambda *_a: True)
    monkeypatch.setattr(_session_state, "read_fingerprint", lambda *_a: None)
    monkeypatch.setattr(_session_state, "mark_read", lambda *_a: None)
    monkeypatch.setattr(
        room_git_flow, "pin_task_head", AsyncMock(side_effect=lambda *a: order.append("pin"))
    )

    async def _read(*_a, **_k):
        order.append("read")
        return "old"

    monkeypatch.setattr(wsd, "read_workspace_file", _read)
    monkeypatch.setattr(
        wsd, "write_workspace_file", AsyncMock(side_effect=lambda *a, **k: order.append("write"))
    )
    ctx = SimpleNamespace(room_id=uuid4(), db=MagicMock(), session_id="s", task_id=uuid4())

    await EditFileTool().execute(
        ctx, {"path": "a.txt", "old_string": "old", "new_string": "new"}
    )

    assert order[:3] == ["pin", "read", "write"]
