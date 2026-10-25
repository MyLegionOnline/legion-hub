"""DELETE /rooms/{id}/git/branch — the user discards a branch from the Git panel.

Covers:
  * protected branches (release/working) are refused;
  * a live session branch is refused;
  * a running task's branch is refused;
  * happy path: workspaced tag-backup + delete, task cancelled, open goal gate
    closed, BRANCH_DELETED emitted.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import git as git_api
from app.models.task import TaskState


def _room():
    r = MagicMock()
    r.id = uuid4()
    r.git_config = {}
    return r


def _user():
    u = MagicMock()
    u.id = uuid4()
    return u


def _task(room_id, branch, state=TaskState.WAITING_REVIEW):
    t = MagicMock()
    t.id = uuid4()
    t.room_id = room_id
    t.branch_name = branch
    t.state = state
    t.divergence_state = None
    return t


def _goal(room_id, branch, review_state="waiting_review"):
    g = MagicMock()
    g.id = uuid4()
    g.room_id = room_id
    g.branch_name = branch
    g.review_state = review_state
    return g


def _db(*rows_by_call):
    """db.execute() yielding successive scalars().all()/first() results."""
    idx = [0]
    results = list(rows_by_call)

    async def execute(_stmt):
        i = idx[0]
        idx[0] += 1
        rows = results[i] if i < len(results) else []
        scalars = MagicMock()
        scalars.all.return_value = rows if isinstance(rows, list) else [rows]
        scalars.first.return_value = (
            rows[0] if isinstance(rows, list) and rows else (None if isinstance(rows, list) else rows)
        )
        res = MagicMock()
        res.scalars.return_value = scalars
        return res

    db = AsyncMock()
    db.execute = execute
    db.add = MagicMock()  # sync in SQLAlchemy — keep it out of the await chain
    return db


async def _call(db, room, user, name, discard_result=None):
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "emit_event", new_callable=AsyncMock), \
         patch.object(
             git_api.wsd, "git_discard_branch_workspace", new_callable=AsyncMock,
             return_value=discard_result or {"ok": True, "deleted": True,
                                             "backup_ref": "refs/legion/trash/x@t"},
         ):
        return await git_api.delete_git_branch(room.id, name, db=db, current_user=user)


@pytest.mark.asyncio
async def test_refuses_protected_branches():
    room, user = _room(), _user()
    for name in ("main", "dev"):
        db = _db()
        db.get = AsyncMock(return_value=room)
        with pytest.raises(HTTPException) as exc:
            await _call(db, room, user, name)
        assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_refuses_live_session_branch():
    room, user = _room(), _user()
    sess = MagicMock()
    sess.id = uuid4()
    branch = git_api.session_branch_name(sess.id)

    db = _db([sess])  # live sessions
    db.get = AsyncMock(return_value=room)
    with pytest.raises(HTTPException) as exc:
        await _call(db, room, user, branch)
    assert exc.value.status_code == 409
    assert "session" in exc.value.detail


@pytest.mark.asyncio
async def test_refuses_running_task_branch():
    room, user = _room(), _user()
    t = _task(room.id, "task/5", state=TaskState.IN_PROGRESS)

    db = _db([], [t])  # no live sessions, then the task lookup
    db.get = AsyncMock(return_value=room)
    with pytest.raises(HTTPException) as exc:
        await _call(db, room, user, "task/5")
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_deletes_goal_branch_and_closes_gate():
    room, user = _room(), _user()
    goal = _goal(room.id, "goal/abc", review_state="waiting_review")

    db = _db([], [], [goal])  # sessions, task lookup (none), goal lookup
    db.get = AsyncMock(return_value=room)

    res = await _call(db, room, user, "goal/abc")

    assert res.ok is True and res.deleted is True
    assert res.backup_ref == "refs/legion/trash/x@t"
    assert goal.review_state == "rejected"
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_deletes_task_branch_and_cancels_task():
    room, user = _room(), _user()
    t = _task(room.id, "task/7", state=TaskState.WAITING_REVIEW)

    db = _db([], [t], [])  # sessions, task lookup, goal lookup (none)
    db.get = AsyncMock(return_value=room)

    res = await _call(db, room, user, "task/7")

    assert res.ok is True
    assert t.state == TaskState.CANCELLED
    assert t.divergence_state == "resolved"


@pytest.mark.asyncio
async def test_workspaced_failure_surfaces_as_409():
    room, user = _room(), _user()
    db = _db([], [], [])
    db.get = AsyncMock(return_value=room)

    with pytest.raises(HTTPException) as exc:
        await _call(db, room, user, "spike/x",
                    discard_result={"ok": False, "deleted": False,
                                    "stderr": "branch is checked out — cannot discard the active branch"})
    assert exc.value.status_code == 409
    assert "checked out" in exc.value.detail
