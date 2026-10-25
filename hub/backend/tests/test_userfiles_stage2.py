"""Stage 2 — upload defaults and private-file response/event metadata."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import workspace_files
from app.core.tools import _repo_map


def _user():
    return SimpleNamespace(id=uuid4())


@pytest.mark.asyncio
async def test_upload_defaults_to_private_room_files_and_marks_response_and_event():
    room_id = uuid4()
    upload = SimpleNamespace(
        filename="reference.md",
        content_type="text/markdown",
        read=AsyncMock(side_effect=[b"# reference", b""]),
    )
    reservation = SimpleNamespace(release=MagicMock())
    db = AsyncMock()
    db.get = AsyncMock(return_value=SimpleNamespace(creator_id=None, git_config={}))

    with patch.object(workspace_files, "_get_room", new_callable=AsyncMock, return_value=SimpleNamespace(creator_id=None, git_config={})), \
         patch.object(workspace_files, "_verify_membership", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_writable", new_callable=AsyncMock), \
         patch.object(workspace_files, "require_room_storage", new_callable=AsyncMock), \
         patch.object(workspace_files, "reserve_room_storage", new_callable=AsyncMock, return_value=reservation), \
         patch.object(workspace_files, "ensure_userfiles_protected", new_callable=AsyncMock), \
         patch.object(workspace_files.wsd, "write_workspace_file", new_callable=AsyncMock) as write, \
         patch.object(workspace_files, "emit_event", new_callable=AsyncMock) as emit, \
         patch.object(workspace_files, "schedule_room_measure"):
        write.return_value = {
            "path": ".userfiles/reference.md",
            "size": 11,
            "commit_sha": "should-not-be-presented",
            "sync_ignored": False,
            "sync_ignore_source": None,
        }
        result = await workspace_files.workspace_fs_upload(
            room_id=room_id,
            file=upload,
            relative_path="",
            commit_message=None,
            db=db,
            current_user=_user(),
        )

    assert write.call_args.args[1] == ".userfiles/reference.md"
    # Private files are written to the volume only — never committed.
    assert write.call_args.kwargs["commit_message"] is None
    assert result.path == ".userfiles/reference.md"
    assert result.commit_sha is None
    assert result.storage_scope == "room_private"
    assert result.git_excluded is True
    metadata = emit.call_args.kwargs["metadata"]
    assert metadata["storage_scope"] == "room_private"
    assert metadata["git_excluded"] is True
    assert metadata["commit_sha"] is None
    reservation.release.assert_called_once_with()


def test_repo_map_allows_private_text_but_keeps_other_dot_paths_hidden():
    assert _repo_map._ignored(".userfiles/readme.md") is False
    assert _repo_map._ignored(".userfiles/nested/reference.txt") is False
    assert _repo_map._ignored(".git/config") is True
    assert _repo_map._ignored(".legion/plan.md") is True
    assert _repo_map._indexable(".userfiles/readme.md", 12) is True
    assert _repo_map._indexable(".userfiles/image.png", 12) is False
