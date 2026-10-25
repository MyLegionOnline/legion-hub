"""internal design notes Stage 1 — adopt an external repo as the room base.

Covers:
  * ``git_sync.adopt_remote_repository`` resolves ``working_branch`` from the
    room flow config, forwards ``force``, and normalises the workspaced result;
  * ``POST /rooms/{id}/git/adopt`` stamps ``git_config.adopted_at`` and emits
    ``GIT_ADOPTED`` only when the transplant actually happened;
  * a ``diverged`` result emits ``GIT_CONFLICT`` and leaves ``adopted_at`` unset;
  * the endpoint forwards ``{"force": true}``.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import git as git_api
from app.core import git_sync


# ── adopt_remote_repository ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_adopt_repo_forwards_working_branch_and_force():
    cfg = {"repo_url": "https://github.com/o/r", "branch": "main", "working_branch": "trunk"}
    with patch("app.core.workspaced_client.git_adopt_workspace", new_callable=AsyncMock) as mp:
        mp.return_value = {
            "ok": True, "adopted": True, "diverged": False, "dropped_init": True,
            "main_sha": "abc", "working_sha": "abc",
        }
        result = await git_sync.adopt_remote_repository(
            room_id=uuid4(), git_config=cfg, access_token="ghp", force=True
        )

    assert mp.await_args.kwargs["working_branch"] == "trunk"
    assert mp.await_args.kwargs["force"] is True
    assert mp.await_args.kwargs["branch"] == "main"
    assert result == {
        "ok": True, "adopted": True, "diverged": False, "dropped_init": True,
        "main_sha": "abc", "working_sha": "abc", "errors": [],
    }


@pytest.mark.asyncio
async def test_adopt_repo_reports_divergence_without_error():
    with patch("app.core.workspaced_client.git_adopt_workspace", new_callable=AsyncMock) as mp:
        mp.return_value = {
            "ok": True, "adopted": False, "diverged": True,
            "stderr": "local `main` has diverged from the remote; not moved",
        }
        result = await git_sync.adopt_remote_repository(
            room_id=uuid4(),
            git_config={"repo_url": "https://github.com/o/r"},
        )

    assert result["diverged"] is True
    assert result["adopted"] is False
    assert result["errors"] == []


@pytest.mark.asyncio
async def test_adopt_repo_surfaces_workspaced_exception():
    with patch(
        "app.core.workspaced_client.git_adopt_workspace",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = await git_sync.adopt_remote_repository(
            room_id=uuid4(), git_config={"repo_url": "https://github.com/o/r"}
        )
    assert result == {"ok": False, "adopted": False, "diverged": False, "errors": ["boom"]}


# ── POST /rooms/{id}/git/adopt ─────────────────────────────────────────────

def _room():
    return MagicMock(git_config={"repo_url": "https://github.com/o/r", "branch": "main"})


def _user():
    return MagicMock(id=uuid4(), email="u@e")


def _request():
    return MagicMock(client=MagicMock(host="127.0.0.1"))


@pytest.mark.asyncio
async def test_adopt_endpoint_stamps_adopted_at_and_emits_event():
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "flag_modified"), \
         patch.object(git_api, "emit_event", new_callable=AsyncMock) as emit, \
         patch.object(git_api, "adopt_remote_repository", new_callable=AsyncMock) as adopt:
        adopt.return_value = {"ok": True, "adopted": True, "diverged": False, "dropped_init": True, "errors": []}
        db = AsyncMock()
        resp = await git_api.git_adopt(
            room_id=uuid4(), request=_request(),
            body=git_api.GitAdoptBody(force=False), db=db, current_user=_user(),
        )

    assert resp.adopted is True
    assert "adopted_at" in room.git_config
    db.commit.assert_awaited()
    assert emit.await_args.kwargs["event_type"] == "GIT_ADOPTED"


@pytest.mark.asyncio
async def test_adopt_endpoint_emits_conflict_on_divergence():
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "flag_modified"), \
         patch.object(git_api, "emit_event", new_callable=AsyncMock) as emit, \
         patch.object(git_api, "adopt_remote_repository", new_callable=AsyncMock) as adopt:
        adopt.return_value = {"ok": True, "adopted": False, "diverged": True, "errors": []}
        resp = await git_api.git_adopt(
            room_id=uuid4(), request=_request(), body=None, db=AsyncMock(), current_user=_user(),
        )

    assert resp.diverged is True
    assert "adopted_at" not in room.git_config
    assert emit.await_args.kwargs["event_type"] == "GIT_CONFLICT"


@pytest.mark.asyncio
async def test_adopt_endpoint_forwards_force():
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "flag_modified"), \
         patch.object(git_api, "emit_event", new_callable=AsyncMock), \
         patch.object(git_api, "adopt_remote_repository", new_callable=AsyncMock) as adopt:
        adopt.return_value = {"ok": False, "adopted": False, "diverged": True, "errors": ["task/x present"]}
        await git_api.git_adopt(
            room_id=uuid4(), request=_request(),
            body=git_api.GitAdoptBody(force=True), db=AsyncMock(), current_user=_user(),
        )

    assert adopt.await_args.kwargs["force"] is True
