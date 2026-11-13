"""``git_fetch`` reports whether the checked-out commit moved.

A running workspace service keeps the code it loaded at launch. The backend
restarts running services after a Pull only when ``head_changed`` is True,
so a no-op Pull must report False and a real one True.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops
from tests.test_git_fetch_branch_pin import _seeded_remote


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "head-room"
    git_ops.git_init(room_id)
    return room_id


def test_pull_that_brings_commits_reports_head_changed(isolated_repo, tmp_path):
    remote = _seeded_remote(tmp_path)

    first = git_ops.git_fetch(
        isolated_repo, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )
    assert first["ok"] is True, first["stderr"]
    assert first["head_changed"] is True

    again = git_ops.git_fetch(
        isolated_repo, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )
    assert again["ok"] is True, again["stderr"]
    assert again["head_changed"] is False
