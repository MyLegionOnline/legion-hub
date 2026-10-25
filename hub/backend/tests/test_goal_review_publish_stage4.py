"""internal design notes Stage 4 — POST /rooms/{id}/goals/{gid}/review/publish.

The endpoint is a thin wrapper over ``room_git_flow.publish_goal_branch``
(covered in ``test_room_git_flow_stage5.py``); here we check the HTTP-level
guards and error mapping, mirroring ``test_git_publish_guards.py``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import goals as goals_api


def _goal(**kw):
    base = dict(
        id=uuid4(),
        title="G",
        branch_name=f"goal/{uuid4()}-x",
        review_state="waiting_review",
        review_ci_status="passed",
        review_branch_pushed_at=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _room(repo_url="https://github.com/o/r"):
    return MagicMock(git_config={"repo_url": repo_url} if repo_url else {}, creator_id=uuid4())


def _request():
    return MagicMock(client=MagicMock(host="127.0.0.1"))


def _db():
    db = AsyncMock()
    db.add = MagicMock()
    return db


async def _call(goal, room):
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock, return_value=goal), \
         patch.object(goals_api, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch("app.api.v1.git._resolve_git_access_token", new_callable=AsyncMock, return_value=None), \
         patch(
             "app.core.room_git_flow.publish_goal_branch",
             new_callable=AsyncMock,
         ) as pub:
        pub.return_value = {"ok": True, "branch": goal.branch_name}
        result = await goals_api.publish_goal_review_branch(
            room_id=uuid4(), goal_id=goal.id, request=_request(),
            db=_db(), current_user=MagicMock(id=uuid4()),
        )
        return result, pub


@pytest.mark.asyncio
async def test_publish_ok_returns_goal():
    goal = _goal()
    out, pub = await _call(goal, _room())
    assert out is goal
    pub.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_422_when_gate_not_open():
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock, return_value=_goal(review_state="approved")), \
         patch.object(goals_api, "_get_room", new_callable=AsyncMock, return_value=_room()):
        with pytest.raises(HTTPException) as ei:
            await goals_api.publish_goal_review_branch(
                room_id=uuid4(), goal_id=uuid4(), request=_request(),
                db=_db(), current_user=MagicMock(id=uuid4()),
            )
    assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_publish_422_without_repo():
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock, return_value=_goal()), \
         patch.object(goals_api, "_get_room", new_callable=AsyncMock, return_value=_room(repo_url=None)):
        with pytest.raises(HTTPException) as ei:
            await goals_api.publish_goal_review_branch(
                room_id=uuid4(), goal_id=uuid4(), request=_request(),
                db=_db(), current_user=MagicMock(id=uuid4()),
            )
    assert ei.value.status_code == 422


@pytest.mark.asyncio
async def test_publish_409_on_non_fast_forward():
    goal = _goal()
    with patch.object(goals_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(goals_api, "_get_goal", new_callable=AsyncMock, return_value=goal), \
         patch.object(goals_api, "_get_room", new_callable=AsyncMock, return_value=_room()), \
         patch("app.api.v1.git._resolve_git_access_token", new_callable=AsyncMock, return_value=None), \
         patch(
             "app.core.room_git_flow.publish_goal_branch",
             new_callable=AsyncMock,
             return_value={"ok": False, "error": "rejected", "hint": "sync_then_promote"},
         ):
        with pytest.raises(HTTPException) as ei:
            await goals_api.publish_goal_review_branch(
                room_id=uuid4(), goal_id=goal.id, request=_request(),
                db=_db(), current_user=MagicMock(id=uuid4()),
            )
    assert ei.value.status_code == 409
