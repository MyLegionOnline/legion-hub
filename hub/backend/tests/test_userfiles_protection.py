"""A private write is refused unless workspaced confirms Git exclusion.

Regression for the 2026-10-06 leak: the backend already sent uploads to
``/.userfiles`` while workspaced still ran pre-exclusion code, so the file was
swept into a commit by ``git add -A`` and pushed.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException, UploadFile

from app.api.v1 import workspace_files
from app.core import userfiles_policy


def _user():
    return SimpleNamespace(id=uuid4())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backfill_result",
    [
        {"ok": True, "updated": False},  # pre-protection workspaced omits the key
        {"ok": True, "userfiles_protected": False},
    ],
)
async def test_guard_fails_closed_when_workspaced_does_not_confirm(backfill_result):
    with patch(
        "app.core.workspaced_client.backfill_local_excludes",
        new=AsyncMock(return_value=backfill_result),
    ):
        with pytest.raises(userfiles_policy.UserfilesProtectionUnavailable):
            await userfiles_policy.ensure_userfiles_protected(uuid4())


@pytest.mark.asyncio
async def test_guard_passes_when_workspaced_confirms():
    with patch(
        "app.core.workspaced_client.backfill_local_excludes",
        new=AsyncMock(return_value={"ok": True, "userfiles_protected": True}),
    ):
        result = await userfiles_policy.ensure_userfiles_protected(uuid4())
    assert result["userfiles_protected"] is True


@pytest.mark.asyncio
async def test_private_upload_is_503_and_nothing_is_written_without_confirmation():
    room = SimpleNamespace(creator_id=None, git_config={})
    upload = UploadFile(BytesIO(b"secret"), filename="sara.jpg")
    with patch.object(workspace_files, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch.object(workspace_files, "_verify_membership", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_writable", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_storage", new_callable=AsyncMock), \
         patch.object(workspace_files, "reserve_room_storage", new_callable=AsyncMock) as reserve, \
         patch(
             "app.core.workspaced_client.backfill_local_excludes",
             new=AsyncMock(return_value={"ok": True, "updated": False}),
         ), \
         patch.object(workspace_files.wsd, "write_workspace_file", new_callable=AsyncMock) as write:
        with pytest.raises(HTTPException) as raised:
            await workspace_files.workspace_fs_upload(
                room_id=uuid4(),
                file=upload,
                target_path="/.userfiles",
                relative_path="sara.jpg",
                commit_message=None,
                db=AsyncMock(),
                current_user=_user(),
            )
    assert raised.value.status_code == 503
    assert raised.value.detail["error_code"] == "userfiles_protection_unavailable"
    write.assert_not_awaited()
    reserve.assert_not_awaited()


async def _upload(target_path: str, commit_message):
    room = SimpleNamespace(creator_id=None, git_config={})
    reservation = SimpleNamespace(release=MagicMock())
    upload = UploadFile(BytesIO(b"x"), filename="a.txt")
    with patch.object(workspace_files, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch.object(workspace_files, "_verify_membership", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_writable", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_storage", new_callable=AsyncMock), \
         patch.object(workspace_files, "reserve_room_storage", new_callable=AsyncMock, return_value=reservation), \
         patch.object(workspace_files, "ensure_userfiles_protected", new_callable=AsyncMock), \
         patch.object(workspace_files, "emit_event", new_callable=AsyncMock), \
         patch.object(workspace_files, "schedule_room_measure"), \
         patch.object(
             workspace_files.wsd,
             "write_workspace_file",
             new_callable=AsyncMock,
             return_value={"size": 1, "commit_sha": "abc"},
         ) as write:
        await workspace_files.workspace_fs_upload(
            room_id=uuid4(),
            file=upload,
            target_path=target_path,
            relative_path="a.txt",
            commit_message=commit_message,
            db=AsyncMock(),
            current_user=_user(),
        )
    return write.call_args.kwargs["commit_message"]


@pytest.mark.asyncio
async def test_private_upload_never_commits_even_if_caller_asks():
    assert await _upload("/.userfiles", "user: upload .userfiles/a.txt") is None


@pytest.mark.asyncio
async def test_project_upload_still_commits():
    assert await _upload("/src", None) == "user: upload src/a.txt"


@pytest.mark.asyncio
async def test_private_put_is_guarded_and_put_and_delete_never_commit():
    room = SimpleNamespace(creator_id=None, git_config={})
    body = SimpleNamespace(content="hi", commit_message="user: write .userfiles/n.txt")
    with patch.object(workspace_files, "_get_room", new_callable=AsyncMock, return_value=room), \
         patch.object(workspace_files, "_verify_membership", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_writable", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_storage", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_files", new_callable=AsyncMock), \
         patch.object(workspace_files, "ensure_userfiles_protected", new_callable=AsyncMock) as guard, \
         patch.object(workspace_files, "emit_event", new_callable=AsyncMock), \
         patch.object(workspace_files, "schedule_room_measure"), \
         patch.object(
             workspace_files.wsd,
             "write_workspace_file",
             new_callable=AsyncMock,
             return_value={"size": 2, "commit_sha": "abc"},
         ) as write, \
         patch.object(
             workspace_files.wsd, "delete_workspace_file", new_callable=AsyncMock, return_value=True
         ) as delete:
        out = await workspace_files.workspace_fs_write_file(
            room_id=uuid4(), path=".userfiles/n.txt", body=body, db=AsyncMock(), current_user=_user()
        )
        await workspace_files.workspace_fs_delete_file(
            room_id=uuid4(),
            path=".userfiles/n.txt",
            commit_message="user: delete .userfiles/n.txt",
            db=AsyncMock(),
            current_user=_user(),
        )

    guard.assert_awaited_once()
    assert write.call_args.kwargs["commit_message"] is None
    assert out.commit_sha is None
    assert delete.call_args.kwargs["commit_message"] is None
