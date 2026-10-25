"""internal design notes follow-up — POST /rooms/{id}/goals/{gid}/merge-branch.

Direct fold of a finished goal's still-unmerged branch into dev, for a goal
whose Stage 5 gate never opened. HTTP-level guards + error mapping.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import goals as goals_api
from app.models.goal import GoalStatus


def _goal(**kw):
    base = dict(
        id=uuid4(),
        title="Hero section",
        branch_name=f"goal/{uuid4()}-hero",
        status=GoalStatus.COMPLETED,
        review_state=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _room():
    return MagicMock(git_config={"repo_url": "https://github.com/o/r"}, creator_id=uuid4())


def _db():
    db = AsyncMock()
    db.add = MagicMock()
    return db


async def _call(goal, *, branches, merge_result):
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock, return_value=goal), \
         patch.object(goals_api, "_get_room", new_callable=AsyncMock, return_value=_room()), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=branches), \
         patch("app.core.room_git_flow.get_flow_config",
               return_value=SimpleNamespace(working_branch="dev", release_branch="main")), \
         patch("app.core.room_git_flow.merge_goal_branch_direct",
               new_callable=AsyncMock, return_value=merge_result) as m:
        out = await goals_api.merge_goal_branch_endpoint(
            room_id=uuid4(), goal_id=goal.id, db=_db(), current_user=MagicMock(id=uuid4()),
        )
        return out, m


@pytest.mark.asyncio
async def test_merge_ok_returns_goal():
    goal = _goal()
    out, m = await _call(
        goal,
        branches=[{"name": goal.branch_name, "merged": False}],
        merge_result={"ok": True, "merge_commit": "abc"},
    )
    assert out is goal
    m.assert_awaited_once()


@pytest.mark.asyncio
async def test_merge_422_when_not_completed():
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock, return_value=_goal(status=GoalStatus.ACTIVE)):
        with pytest.raises(HTTPException) as ei:
            await goals_api.merge_goal_branch_endpoint(
                room_id=uuid4(), goal_id=uuid4(), db=_db(), current_user=MagicMock(id=uuid4()),
            )
    assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_merge_422_when_review_pending():
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock,
                      return_value=_goal(review_state="waiting_review")):
        with pytest.raises(HTTPException) as ei:
            await goals_api.merge_goal_branch_endpoint(
                room_id=uuid4(), goal_id=uuid4(), db=_db(), current_user=MagicMock(id=uuid4()),
            )
    assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_merge_422_when_no_branch():
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock, return_value=_goal(branch_name=None)):
        with pytest.raises(HTTPException) as ei:
            await goals_api.merge_goal_branch_endpoint(
                room_id=uuid4(), goal_id=uuid4(), db=_db(), current_user=MagicMock(id=uuid4()),
            )
    assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_merge_422_when_already_merged():
    goal = _goal()
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock, return_value=goal), \
         patch.object(goals_api, "_get_room", new_callable=AsyncMock, return_value=_room()), \
         patch("app.core.workspaced_client.list_workspace_branches",
               new_callable=AsyncMock, return_value=[{"name": goal.branch_name, "merged": True}]), \
         patch("app.core.room_git_flow.get_flow_config",
               return_value=SimpleNamespace(working_branch="dev", release_branch="main")):
        with pytest.raises(HTTPException) as ei:
            await goals_api.merge_goal_branch_endpoint(
                room_id=uuid4(), goal_id=goal.id, db=_db(), current_user=MagicMock(id=uuid4()),
            )
    assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_merge_409_on_conflict():
    goal = _goal()
    with pytest.raises(HTTPException) as ei:
        await _call(
            goal,
            branches=[{"name": goal.branch_name, "merged": False}],
            merge_result={"ok": False, "conflict": True},
        )
    assert ei.value.status_code == 409
