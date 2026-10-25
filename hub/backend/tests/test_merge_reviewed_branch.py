"""internal design notes — POST /rooms/{id}/tasks/{tid}/merge-branch.

Direct "review already happened, just fold it in" merge for a COMPLETED task
whose branch never landed.

Covers:
  * COMPLETED + unmerged → merge_task_branch called, merge_commit stamped;
  * COMPLETED + already merged → 422;
  * non-COMPLETED state (CANCELLED / IN_PROGRESS) → 422;
  * no branch_name → 422;
  * merge_task_branch reports a conflict → 409;
  * merge_task_branch reports a plain failure → 422.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import tasks as tasks_api
from app.models.task import TaskState


def _task(state=TaskState.COMPLETED, branch="task/70"):
    t = MagicMock()
    t.id = uuid4()
    t.room_id = uuid4()
    t.branch_name = branch
    t.state = state
    t.goal_id = None
    t.merge_commit = None
    t.title = "Hero section"
    return t


def _user():
    u = MagicMock()
    u.id = uuid4()
    return u


def _db_with_task(task):
    db = AsyncMock()
    db.get = AsyncMock(return_value=MagicMock(git_config={}))
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    scalars = MagicMock()
    scalars.first.return_value = task
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars
    db.execute = AsyncMock(return_value=execute_result)
    return db


# ── success ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_completed_unmerged_merges():
    task = _task(state=TaskState.COMPLETED)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[{"name": "task/70", "merged": False}]), \
         patch("app.core.room_git_flow._task_base_branch",
               new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.room_git_flow.merge_task_branch", new_callable=AsyncMock,
               return_value={"ok": True, "sha": "deadbee", "branch": "task/70", "target": "dev"}), \
         patch("app.api.v1.tasks._enrich_task", new_callable=AsyncMock, return_value="ENRICHED"):
        result = await tasks_api.merge_reviewed_task_branch(
            room_id=task.room_id, task_id=task.id, db=db, current_user=_user(),
        )

    assert result == "ENRICHED"
    assert task.merge_commit == "deadbee"


# ── already merged ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_already_merged_422():
    task = _task(state=TaskState.COMPLETED)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[{"name": "task/70", "merged": True}]), \
         patch("app.core.room_git_flow._task_base_branch",
               new_callable=AsyncMock, return_value="dev"):
        with pytest.raises(HTTPException) as exc:
            await tasks_api.merge_reviewed_task_branch(
                room_id=task.room_id, task_id=task.id, db=db, current_user=_user(),
            )
    assert exc.value.status_code == 422
    assert "already merged" in exc.value.detail


# ── wrong state ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("state", [TaskState.CANCELLED, TaskState.IN_PROGRESS, TaskState.WAITING_REVIEW])
async def test_non_completed_422(state):
    task = _task(state=state)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock):
        with pytest.raises(HTTPException) as exc:
            await tasks_api.merge_reviewed_task_branch(
                room_id=task.room_id, task_id=task.id, db=db, current_user=_user(),
            )
    assert exc.value.status_code == 422


# ── no branch ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_branch_422():
    task = _task(state=TaskState.COMPLETED, branch=None)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock):
        with pytest.raises(HTTPException) as exc:
            await tasks_api.merge_reviewed_task_branch(
                room_id=task.room_id, task_id=task.id, db=db, current_user=_user(),
            )
    assert exc.value.status_code == 422
    assert "no branch" in exc.value.detail.lower()


# ── conflict ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_merge_conflict_409():
    task = _task(state=TaskState.COMPLETED)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[{"name": "task/70", "merged": False}]), \
         patch("app.core.room_git_flow._task_base_branch",
               new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.room_git_flow.merge_task_branch", new_callable=AsyncMock,
               return_value={"ok": False, "conflict": True, "branch": "task/70", "target": "dev"}):
        with pytest.raises(HTTPException) as exc:
            await tasks_api.merge_reviewed_task_branch(
                room_id=task.room_id, task_id=task.id, db=db, current_user=_user(),
            )
    assert exc.value.status_code == 409


# ── plain failure ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_merge_failure_422():
    task = _task(state=TaskState.COMPLETED)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[{"name": "task/70", "merged": False}]), \
         patch("app.core.room_git_flow._task_base_branch",
               new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.room_git_flow.merge_task_branch", new_callable=AsyncMock,
               return_value={"ok": False, "stderr": "boom"}):
        with pytest.raises(HTTPException) as exc:
            await tasks_api.merge_reviewed_task_branch(
                room_id=task.room_id, task_id=task.id, db=db, current_user=_user(),
            )
    assert exc.value.status_code == 422
    assert "boom" in exc.value.detail
