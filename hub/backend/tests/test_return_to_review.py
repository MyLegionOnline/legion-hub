"""internal design notes Stage 6 — return-to-review endpoint.

Covers:
  * COMPLETED task with unmerged branch → 200, state becomes WAITING_REVIEW,
    ci_status reset to None;
  * CANCELLED task → same success path;
  * COMPLETED task with merged branch → 422;
  * non-terminal task (IN_PROGRESS) → 422;
  * task with no branch_name → 422;
  * branch not found in workspace → 422.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import tasks as tasks_api
from app.models.task import TaskState


# ── helpers ──────────────────────────────────────────────────────────────────

def _task(state=TaskState.COMPLETED, branch="task/48", goal_id=None):
    t = MagicMock()
    t.id = uuid4()
    t.room_id = uuid4()
    t.branch_name = branch
    t.state = state
    t.goal_id = goal_id
    t.merge_commit = "abc123"
    t.ci_status = "passed"
    t.title = "Fix something"
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
    # _get_task calls db.execute().scalars().first()
    scalars = MagicMock()
    scalars.first.return_value = task
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars
    db.execute = AsyncMock(return_value=execute_result)
    return db


# ── shared patches ────────────────────────────────────────────────────────────

_BASE_PATCHES = {
    "app.api.v1.tasks._verify_membership": AsyncMock(),
    "app.api.v1.tasks.emit_event": AsyncMock(),
}


# ── test: COMPLETED + unmerged → success ─────────────────────────────────────

@pytest.mark.asyncio
async def test_return_completed_unmerged():
    task = _task(state=TaskState.COMPLETED)
    db = _db_with_task(task)
    room_id = task.room_id

    branch_info = {"name": "task/48", "merged": False, "ahead": 2, "behind": 0}

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[branch_info]), \
         patch("app.core.room_git_flow._task_base_branch",
               new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.room_git_flow.run_task_checks", new_callable=AsyncMock), \
         patch("asyncio.create_task"):

        enriched = MagicMock()
        with patch("app.api.v1.tasks._enrich_task", new_callable=AsyncMock, return_value=enriched):
            result = await tasks_api.return_task_to_review(
                room_id=room_id,
                task_id=task.id,
                db=db,
                current_user=_user(),
            )

    assert result is enriched
    assert task.state == TaskState.WAITING_REVIEW
    assert task.merge_commit is None
    assert task.ci_status is None


# ── test: CANCELLED + unmerged → success ─────────────────────────────────────

@pytest.mark.asyncio
async def test_return_cancelled_unmerged():
    task = _task(state=TaskState.CANCELLED)
    db = _db_with_task(task)

    branch_info = {"name": "task/48", "merged": False, "ahead": 1, "behind": 0}

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.api.v1.tasks.emit_event", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[branch_info]), \
         patch("app.core.room_git_flow._task_base_branch",
               new_callable=AsyncMock, return_value="dev"), \
         patch("app.core.room_git_flow.run_task_checks", new_callable=AsyncMock), \
         patch("asyncio.create_task"):

        enriched = MagicMock()
        with patch("app.api.v1.tasks._enrich_task", new_callable=AsyncMock, return_value=enriched):
            result = await tasks_api.return_task_to_review(
                room_id=task.room_id,
                task_id=task.id,
                db=db,
                current_user=_user(),
            )

    assert result is enriched
    assert task.state == TaskState.WAITING_REVIEW


# ── test: COMPLETED + merged branch → 422 ────────────────────────────────────

@pytest.mark.asyncio
async def test_return_merged_branch_raises_422():
    from fastapi import HTTPException

    task = _task(state=TaskState.COMPLETED)
    db = _db_with_task(task)

    branch_info = {"name": "task/48", "merged": True, "ahead": 0, "behind": 0}

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[branch_info]), \
         patch("app.core.room_git_flow._task_base_branch",
               new_callable=AsyncMock, return_value="dev"):

        with pytest.raises(HTTPException) as exc_info:
            await tasks_api.return_task_to_review(
                room_id=task.room_id,
                task_id=task.id,
                db=db,
                current_user=_user(),
            )

    assert exc_info.value.status_code == 422
    assert "already merged" in exc_info.value.detail


# ── test: non-terminal state → 422 ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_return_non_terminal_raises_422():
    from fastapi import HTTPException

    task = _task(state=TaskState.IN_PROGRESS)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock):
        with pytest.raises(HTTPException) as exc_info:
            await tasks_api.return_task_to_review(
                room_id=task.room_id,
                task_id=task.id,
                db=db,
                current_user=_user(),
            )

    assert exc_info.value.status_code == 422
    assert "terminal" in exc_info.value.detail


# ── test: no branch_name → 422 ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_return_no_branch_raises_422():
    from fastapi import HTTPException

    task = _task(state=TaskState.COMPLETED, branch=None)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock):
        with pytest.raises(HTTPException) as exc_info:
            await tasks_api.return_task_to_review(
                room_id=task.room_id,
                task_id=task.id,
                db=db,
                current_user=_user(),
            )

    assert exc_info.value.status_code == 422
    assert "no branch" in exc_info.value.detail.lower()


# ── test: branch not found in workspace → 422 ────────────────────────────────

@pytest.mark.asyncio
async def test_return_branch_not_found_raises_422():
    from fastapi import HTTPException

    task = _task(state=TaskState.COMPLETED)
    db = _db_with_task(task)

    with patch("app.api.v1.tasks._verify_membership", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[]), \
         patch("app.core.room_git_flow._task_base_branch",
               new_callable=AsyncMock, return_value="dev"):

        with pytest.raises(HTTPException) as exc_info:
            await tasks_api.return_task_to_review(
                room_id=task.room_id,
                task_id=task.id,
                db=db,
                current_user=_user(),
            )

    assert exc_info.value.status_code == 422
    assert "not found" in exc_info.value.detail.lower()
