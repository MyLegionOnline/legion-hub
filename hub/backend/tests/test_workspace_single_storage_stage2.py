"""
Stage 2 of internal design notes — git on the volume.

Stage 6 removed the legacy branch; these tests now pin the single
behaviour of the ``/rooms/{id}/git/*`` endpoints:

  * ``POST /git/sync`` drives ``fetch_workspace_from_git`` and emits
    ``GIT_CONFLICT`` when the merge cannot be resolved automatically
    (diverged histories are merged, only real content conflicts surface);
  * ``POST /git/push`` drives ``push_workspace_to_git`` with ``mode``,
    refuses an empty workspace, and forwards ``mode="force"``;
  * ``POST /git/webhook`` drives ``fetch_workspace_from_git`` (per-file
    ``paths`` no longer needed).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException

from app.api.v1 import git as git_api
from app.core.git_sync import push_workspace_to_git, fetch_workspace_from_git


def _room():
    return MagicMock(
        git_config={
            "repo_url": "https://github.com/randypostman/blog",
            "branch": "main",
            "provider": "github",
        },
        creator_id=uuid4(),
    )


def _user():
    return MagicMock(id=uuid4(), email="u@e")


def _request():
    return MagicMock(client=MagicMock(host="127.0.0.1"))


# ── /git/sync ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sync_calls_fetch_and_returns_ok():
    room_id = uuid4()
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock) as mf:
        mf.return_value = {"synced": 1, "errors": [], "conflict": False}
        result = await git_api.git_sync(
            room_id=room_id, request=_request(), db=AsyncMock(), current_user=_user()
        )

    mf.assert_awaited_once()
    assert mf.call_args.kwargs["git_config"] is room.git_config
    assert mf.call_args.kwargs["access_token"] == "ghp_token"
    assert result.ok is True
    assert result.synced == 1
    assert result.conflict is False


@pytest.mark.asyncio
async def test_sync_emits_git_conflict_event_on_non_fast_forward():
    room_id = uuid4()
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"
    db = AsyncMock()

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock) as mf, \
         patch.object(git_api, "emit_event", new_callable=AsyncMock) as me:
        mf.return_value = {"synced": 0, "errors": ["merge: not a fast-forward"], "conflict": True}
        result = await git_api.git_sync(
            room_id=room_id, request=_request(), db=db, current_user=_user()
        )

    me.assert_awaited_once()
    assert me.call_args.kwargs["event_type"] == "GIT_CONFLICT"
    assert result.conflict is True
    assert result.ok is False


@pytest.mark.asyncio
async def test_sync_surfaces_workspaced_4xx_as_is():
    room_id = uuid4()
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"

    request = httpx.Request("POST", "http://x/fetch")
    response = httpx.Response(404, text="no git repo", request=request)
    exc = httpx.HTTPStatusError("404", request=request, response=response)

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock, side_effect=exc):
        with pytest.raises(HTTPException) as raised:
            await git_api.git_sync(
                room_id=room_id, request=_request(), db=AsyncMock(), current_user=_user()
            )

    assert raised.value.status_code == 404


# ── /git/push ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_push_calls_push_with_default_mode():
    room_id = uuid4()
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "_branch_has_content", new_callable=AsyncMock, return_value=True), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as mp:
        mp.return_value = {"pushed": 1, "errors": []}
        result = await git_api.git_push(
            room_id=room_id, request=_request(), db=AsyncMock(), current_user=_user()
        )

    mp.assert_awaited_once()
    assert mp.call_args.kwargs["mode"] == "push"
    assert mp.call_args.kwargs["access_token"] == "ghp_token"
    assert result["ok"] is True
    assert result["pushed"] == 1


@pytest.mark.asyncio
async def test_push_forwards_force_mode():
    room_id = uuid4()
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "_branch_has_content", new_callable=AsyncMock, return_value=True), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as mp:
        mp.return_value = {"pushed": 1, "errors": []}
        await git_api.git_push(
            room_id=room_id, request=_request(), mode="force",
            db=AsyncMock(), current_user=_user(),
        )

    assert mp.call_args.kwargs["mode"] == "force"


@pytest.mark.asyncio
async def test_push_refuses_empty_branch_with_409():
    room_id = uuid4()
    room = _room()

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_branch_has_content", new_callable=AsyncMock, return_value=False), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as mp:
        with pytest.raises(HTTPException) as exc:
            await git_api.git_push(
                room_id=room_id, request=_request(), db=AsyncMock(), current_user=_user()
            )

    assert exc.value.status_code == 409
    assert "has no changes yet" in exc.value.detail
    mp.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_surfaces_rejected_remote_as_error_not_force():
    """Default mode must NOT paper over a non-fast-forward rejection — the
    user has to ask for ``mode=force`` explicitly (stage 2.2)."""
    room_id = uuid4()
    room = _room()
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "_branch_has_content", new_callable=AsyncMock, return_value=True), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as mp:
        mp.return_value = {"pushed": 0, "errors": ["! [rejected] HEAD -> main (non-fast-forward)"]}
        result = await git_api.git_push(
            room_id=room_id, request=_request(), db=AsyncMock(), current_user=_user()
        )

    assert mp.call_args.kwargs["mode"] == "push"
    assert result["ok"] is False
    assert result["pushed"] == 0
    assert "non-fast-forward" in result["errors"][0]


# ── /git/webhook ──────────────────────────────────────────────────────────────

def _webhook_request(payload: dict):
    import json as _json
    request = MagicMock()
    request.body = AsyncMock(return_value=_json.dumps(payload).encode())
    request.headers = {}
    request.client = MagicMock(host="127.0.0.1")
    return request


@pytest.mark.asyncio
async def test_webhook_calls_fetch():
    room_id = uuid4()
    room = _room()
    payload = {"commits": [{"added": ["a.py"], "modified": ["b.py"]}]}

    secret_value = MagicMock()
    secret_value.reveal.return_value = b"wh_secret"
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp_token"

    with patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "binding_id_from_config", return_value=None), \
         patch.object(git_api, "legacy_secret_value", return_value=secret_value), \
         patch.object(git_api, "verify_git_webhook_signature", return_value=True), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock) as mf:
        mf.return_value = {"synced": 1, "errors": [], "conflict": False}
        result = await git_api.git_webhook(
            room_id=room_id, request=_webhook_request(payload), db=AsyncMock()
        )

    mf.assert_awaited_once()
    assert result["ok"] is True
    assert result["synced"] == 1


# ── closed workspace.py endpoints ────────────────────────────────────────────

def test_workspace_git_push_endpoint_removed():
    """Stage 2.6: the token-in-body git endpoints are gone — a single
    ``/git/*`` with Hub-side token resolution replaces them so there is no
    way to bypass the secret store."""
    from app.api.v1 import workspace as ws

    routes = {r.path for r in ws.router.routes}
    assert "/rooms/{room_id}/workspace/git/push" not in routes
    assert "/rooms/{room_id}/workspace/git/fetch" not in routes


# ── push_workspace_to_git forwards mode to the client ───────────────────────

@pytest.mark.asyncio
async def test_push_workspace_to_git_forwards_mode_to_client():
    room_id = uuid4()
    git_config = {"repo_url": "https://github.com/o/r", "branch": "main"}
    with patch("app.core.workspaced_client.git_push_workspace", new_callable=AsyncMock) as mp:
        mp.return_value = {"ok": True, "stdout": "", "stderr": ""}
        await push_workspace_to_git(
            room_id=room_id, git_config=git_config, access_token="t", mode="force"
        )
    assert mp.call_args.kwargs["mode"] == "force"


@pytest.mark.asyncio
async def test_push_endpoint_returns_409_for_tracked_private_files():
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "_branch_has_content", new_callable=AsyncMock, return_value=True), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as mp:
        mp.return_value = {
            "pushed": 0,
            "errors": ["Hub blocked push"],
            "userfiles_push_blocked": True,
            "userfiles_tracked": [".userfiles/secret.txt"],
        }
        with pytest.raises(HTTPException) as exc:
            await git_api.git_push(
                room_id=uuid4(), request=_request(), db=AsyncMock(), current_user=_user()
            )

    assert exc.value.status_code == 409
    assert exc.value.detail["error_code"] == "userfiles_tracked"
    assert exc.value.detail["paths"] == [".userfiles/secret.txt"]


@pytest.mark.asyncio
async def test_fetch_workspace_from_git_returns_conflict_flag():
    room_id = uuid4()
    git_config = {"repo_url": "https://github.com/o/r", "branch": "main"}
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {"ok": False, "conflict": True, "stderr": "merge: not a fast-forward"}
        result = await fetch_workspace_from_git(
            room_id=room_id, git_config=git_config, access_token="t"
        )
    assert result["conflict"] is True
    assert result["synced"] == 0
