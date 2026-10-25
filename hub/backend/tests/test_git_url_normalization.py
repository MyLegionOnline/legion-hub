"""Regression: a browser-pasted repository URL must never reach ``git``.

Bug: a room was connected with ``repo_url`` copied from the GitHub address bar
(``https://github.com/owner/repo/tree/main``). ``_parse_repo_info`` stripped the
``/tree/…`` suffix for the forge API, but ``push_workspace_to_git`` /
``fetch_workspace_from_git`` / ``adopt_remote_repository`` only trimmed a trailing
slash and ``.git``, so ``git fetch https://github.com/owner/repo/tree/main/ main``
failed with ``fatal: repository '.../tree/main/' not found``.

Now ``git_sync._clone_url`` is the single normaliser used by every path, and the
``RoomUpdate`` / ``RoomBase`` schema normalises ``repo_url`` at the write boundary.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from app.core import git_sync
from app.core.git_sync import _clone_url
from app.schemas.room import RoomUpdate


CLEAN = "https://github.com/randypostman/blog"


@pytest.mark.parametrize(
    "raw",
    [
        "https://github.com/randypostman/blog",
        "https://github.com/randypostman/blog/",
        "https://github.com/randypostman/blog.git",
        "https://github.com/randypostman/blog/tree/main",
        "https://github.com/randypostman/blog/tree/main/",
        "https://github.com/randypostman/blog/tree/feature/x",
        "https://github.com/randypostman/blog/blob/main/README.md",
        "https://github.com/randypostman/blog/commit/deadbeef",
        "https://github.com/randypostman/blog/pull/7",
        "  https://github.com/randypostman/blog/tree/main  ",
    ],
)
def test_clone_url_strips_web_ui_suffixes(raw):
    assert _clone_url(raw) == CLEAN


def test_clone_url_handles_gitlab_dash_paths():
    assert (
        _clone_url("https://gitlab.com/group/sub/proj/-/blob/main/app.py")
        == "https://gitlab.com/group/sub/proj"
    )
    assert (
        _clone_url("https://gitlab.com/group/proj/-/tree/main")
        == "https://gitlab.com/group/proj"
    )


def test_clone_url_empty():
    assert _clone_url("") == ""
    assert _clone_url(None) == ""


def test_room_update_normalizes_repo_url():
    upd = RoomUpdate(git_config={"repo_url": f"{CLEAN}/tree/main", "branch": "main"})
    assert upd.git_config["repo_url"] == CLEAN
    assert upd.git_config["branch"] == "main"


def test_room_update_leaves_clean_url_untouched():
    upd = RoomUpdate(git_config={"repo_url": CLEAN})
    assert upd.git_config["repo_url"] == CLEAN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fn, wsd_name, kw",
    [
        ("push_workspace_to_git", "git_push_workspace", {}),
        ("fetch_workspace_from_git", "git_fetch_workspace", {}),
        ("adopt_remote_repository", "git_adopt_workspace", {}),
    ],
)
async def test_sync_paths_pass_clean_url_to_workspaced(fn, wsd_name, kw):
    cfg = {"repo_url": f"{CLEAN}/tree/main", "branch": "main"}
    with patch(
        f"app.core.workspaced_client.{wsd_name}", new_callable=AsyncMock
    ) as mp:
        mp.return_value = {"ok": True}
        await getattr(git_sync, fn)(room_id=uuid4(), git_config=cfg, **kw)
    assert mp.await_args.kwargs["remote_url"] == CLEAN
