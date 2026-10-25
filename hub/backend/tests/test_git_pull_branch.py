"""Per-branch Pull from the Git panel.

Covers:
  * ``git_sync.fetch_workspace_from_git`` fetches ``remote_branch`` when given
    (and the configured branch otherwise);
  * ``POST /rooms/{id}/git/sync`` with body ``{"branch": "dev"}`` fetches the
    remote ``dev`` into the local ``dev``; without a body nothing changes.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import git as git_api
from app.core import git_sync

_CFG = {"repo_url": "https://github.com/o/r", "branch": "main"}


@pytest.mark.asyncio
async def test_fetch_uses_remote_branch_override():
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {"ok": True, "branch": "task/7"}
        await git_sync.fetch_workspace_from_git(
            room_id=uuid4(), git_config=_CFG, remote_branch="task/7", target_branch="task/7",
        )
    assert mf.await_args.kwargs["branch"] == "task/7"
    assert mf.await_args.kwargs["target_branch"] == "task/7"


@pytest.mark.asyncio
async def test_fetch_defaults_to_config_branch_into_working_branch():
    with patch("app.core.workspaced_client.git_fetch_workspace", new_callable=AsyncMock) as mf:
        mf.return_value = {"ok": True}
        await git_sync.fetch_workspace_from_git(room_id=uuid4(), git_config=_CFG)
    assert mf.await_args.kwargs["branch"] == "main"
    assert mf.await_args.kwargs["target_branch"] == "dev"


def _token():
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp"
    return token_value


async def _sync(body):
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock,
                      return_value=MagicMock(git_config=dict(_CFG))), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock,
                      return_value=_token()), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock) as mf:
        mf.return_value = {"synced": 1, "errors": [], "conflict": False, "branch": "dev"}
        resp = await git_api.git_sync(
            room_id=uuid4(),
            request=MagicMock(client=MagicMock(host="127.0.0.1")),
            body=body,
            db=AsyncMock(),
            current_user=MagicMock(id=uuid4(), email="u@e"),
        )
    return mf, resp


@pytest.mark.asyncio
async def test_sync_endpoint_pulls_named_branch_into_itself():
    mf, resp = await _sync(git_api.GitSyncBody(branch="dev"))
    assert mf.await_args.kwargs["remote_branch"] == "dev"
    assert mf.await_args.kwargs["target_branch"] == "dev"
    assert resp.ok and resp.branch == "dev"


@pytest.mark.asyncio
async def test_sync_endpoint_without_body_keeps_default_routing():
    mf, _ = await _sync(None)
    assert "remote_branch" not in mf.await_args.kwargs
    assert "target_branch" not in mf.await_args.kwargs
