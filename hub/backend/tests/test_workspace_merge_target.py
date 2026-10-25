"""internal design notes Stage 0 — /workspace/merge cannot target the release branch.

Covers:
  * POST /workspace/merge {name:'x'} without target → merges into working_branch (dev)
  * POST /workspace/merge {name:'x', target:'main'} → 422, main not touched
  * POST /workspace/merge {name:'x', target:'dev'} → allowed, passes target to workspaced
  * GIT_CONFLICT event uses actual target, not hardcoded 'main'
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import workspace as ws_api


def _room(working_branch="dev"):
    gc = {"working_branch": working_branch}
    return MagicMock(id=uuid4(), git_config=gc)


def _user():
    return MagicMock(id=uuid4(), email="u@e")


def _make_body(name="feat/x", target=None, message=None):
    body = MagicMock()
    body.name = name
    body.target = target
    body.message = message
    return body


# ── no target → merges into working_branch ──────────────────────────────────

@pytest.mark.asyncio
async def test_no_target_merges_into_working_branch():
    room = _room(working_branch="dev")
    merge_mock = AsyncMock(return_value={"ok": True, "conflict": False, "stdout": "", "stderr": ""})

    with patch.object(ws_api, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch.object(ws_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(ws_api.wsd, "merge_workspace_branch", merge_mock):
        result = await ws_api.workspace_git_merge_branch(
            room_id=room.id,
            body=_make_body(target=None),
            db=AsyncMock(),
            current_user=_user(),
        )

    assert result["ok"] is True
    merge_mock.assert_awaited_once()
    _, call_kwargs = merge_mock.call_args
    assert call_kwargs.get("target") == "dev" or merge_mock.call_args[0][2:] == () or \
        merge_mock.call_args.kwargs.get("target") == "dev" or \
        merge_mock.call_args[0][2] == "dev" if len(merge_mock.call_args[0]) > 2 else True
    # Verify by inspecting all args
    all_args = list(merge_mock.call_args[0]) + list(merge_mock.call_args[1].values())
    assert "dev" in all_args


# ── target='main' → 422 ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_target_main_raises_422():
    room = _room(working_branch="dev")
    merge_mock = AsyncMock(return_value={"ok": True, "conflict": False, "stdout": "", "stderr": ""})

    with patch.object(ws_api, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch.object(ws_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(ws_api.wsd, "merge_workspace_branch", merge_mock):
        with pytest.raises(HTTPException) as exc_info:
            await ws_api.workspace_git_merge_branch(
                room_id=room.id,
                body=_make_body(target="main"),
                db=AsyncMock(),
                current_user=_user(),
            )

    assert exc_info.value.status_code == 422
    assert "promotion gate" in exc_info.value.detail
    merge_mock.assert_not_awaited()


# ── target='dev' → allowed ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_explicit_dev_target_allowed():
    room = _room(working_branch="dev")
    merge_mock = AsyncMock(return_value={"ok": True, "conflict": False, "stdout": "", "stderr": ""})

    with patch.object(ws_api, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch.object(ws_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(ws_api.wsd, "merge_workspace_branch", merge_mock):
        result = await ws_api.workspace_git_merge_branch(
            room_id=room.id,
            body=_make_body(target="dev"),
            db=AsyncMock(),
            current_user=_user(),
        )

    assert result["ok"] is True
    merge_mock.assert_awaited_once()
    assert merge_mock.call_args.kwargs.get("target") == "dev"


# ── GIT_CONFLICT event uses actual target ───────────────────────────────────

@pytest.mark.asyncio
async def test_conflict_event_uses_actual_target():
    room = _room(working_branch="dev")
    merge_mock = AsyncMock(return_value={"ok": False, "conflict": True, "stdout": "", "stderr": ""})
    emit_mock = AsyncMock()

    with patch.object(ws_api, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch.object(ws_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(ws_api.wsd, "merge_workspace_branch", merge_mock), \
         patch.object(ws_api, "emit_event", emit_mock):
        await ws_api.workspace_git_merge_branch(
            room_id=room.id,
            body=_make_body(name="feat/y", target=None),
            db=AsyncMock(),
            current_user=_user(),
        )

    emit_mock.assert_awaited_once()
    call_kwargs = emit_mock.call_args.kwargs
    assert "dev" in call_kwargs.get("content", "")
    assert "main" not in call_kwargs.get("content", "")
    assert call_kwargs.get("metadata", {}).get("target") == "dev"
