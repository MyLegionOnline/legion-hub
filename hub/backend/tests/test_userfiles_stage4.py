"""Stage 4 — backend policy state and private upload fail-closed behaviour."""
from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.v1 import git as git_api
from app.api.v1 import workspace_files


def _room(*, conflict: bool = False):
    config = {
        "repo_url": "https://github.com/example/project",
        "branch": "main",
    }
    if conflict:
        config.update(
            {
                "userfiles_policy": "conflict",
                "userfiles_policy_paths": [".userfiles/remote.txt"],
            }
        )
    return SimpleNamespace(id=uuid4(), creator_id=uuid4(), git_config=config)


def _user():
    return SimpleNamespace(id=uuid4())


def _request():
    return MagicMock(client=MagicMock(host="127.0.0.1"))


@pytest.mark.asyncio
async def test_sync_surfaces_remote_userfiles_policy_conflict():
    room_id = uuid4()
    room = _room()
    db = AsyncMock()
    token = MagicMock()
    token.reveal.return_value = "token"
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=room), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock) as fetch, \
         patch.object(git_api, "apply_userfiles_policy", new_callable=AsyncMock) as apply:
        fetch.return_value = {
            "synced": 0,
            "errors": ["private-files policy conflict"],
            "conflict": False,
            "userfiles_policy_conflict": True,
            "userfiles_tracked": [".userfiles/remote.txt"],
            "userfiles_tracked_details": [
                {
                    "path": ".userfiles/remote.txt",
                    "size": 7,
                    "source": "remote",
                    "error": "remote_tracks_userfiles",
                }
            ],
        }
        result = await git_api.git_sync(
            room_id=room_id,
            request=_request(),
            db=db,
            current_user=_user(),
        )

    assert result.ok is False
    assert result.userfiles_policy_conflict is True
    assert result.userfiles_tracked == [".userfiles/remote.txt"]
    apply.assert_awaited_once()


@pytest.mark.asyncio
async def test_private_upload_is_rejected_while_policy_conflict_is_marked():
    room_id = uuid4()
    room = _room(conflict=True)
    upload = UploadFile(BytesIO(b"secret"), filename="secret.txt")
    db = AsyncMock()

    with patch.object(workspace_files, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch.object(workspace_files, "_verify_membership", new_callable=AsyncMock), \
         patch.object(workspace_files.wsd, "write_workspace_file", new_callable=AsyncMock) as write:
        with pytest.raises(HTTPException) as raised:
            await workspace_files.workspace_fs_upload(
                room_id=room_id,
                file=upload,
                target_path="/.userfiles",
                relative_path="secret.txt",
                commit_message=None,
                db=db,
                current_user=_user(),
            )

    assert raised.value.status_code == 409
    assert raised.value.detail["error_code"] == "userfiles_policy_conflict"
    assert "remove_remote_tracking" in raised.value.detail["options"]
    write.assert_not_awaited()
