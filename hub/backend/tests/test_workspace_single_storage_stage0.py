"""
Stage 0 of internal design notes — stop the bleeding.

Stage 6 removed the legacy branch entirely; the surviving guard from stage 0
is the *empty-workspace* check: the endpoint refuses to push when the branch
being pushed has no real content yet, because a push would otherwise overwrite
every remote file with an empty tree — the same failure mode that was already
repaired for ``sync_files_to_workspace`` in ``internal design notes`` Stage 0.
internal design notes §4.1 retargeted the check from "HEAD has commits"
to "the pushed branch's tree has content" (``_branch_has_content``).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import git as git_api


@pytest.mark.asyncio
async def test_git_push_refuses_when_workspace_has_no_commits():
    """An empty workspace must not be pushed over a live remote."""
    room_id = uuid4()

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock) as vm, \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock) as gr, \
         patch.object(git_api, "_branch_has_content", new_callable=AsyncMock) as hc:
        vm.return_value = None
        gr.return_value = MagicMock(git_config={"repo_url": "https://github.com/o/r"})
        hc.return_value = False
        with pytest.raises(HTTPException) as exc:
            await git_api.git_push(
                room_id=room_id,
                request=MagicMock(),
                db=AsyncMock(),
                current_user=MagicMock(),
            )

    assert exc.value.status_code == 409
    assert "empty tree" in exc.value.detail
