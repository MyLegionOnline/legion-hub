"""Stage 4 — remote private-files policy conflicts are fail-closed."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "userfiles-stage4"
    git_ops.git_init(room_id)
    return room_id, tmp_path / room_id


def _git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def _remote_with_tracked_userfiles(tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], capture_output=True, check=True)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", str(remote), str(clone)], capture_output=True, check=True)
    _git(clone, "config", "user.name", "Remote Author")
    _git(clone, "config", "user.email", "remote@example.com")
    private = clone / ".userfiles" / "remote-secret.txt"
    private.parent.mkdir(parents=True)
    private.write_text("remote only\n", encoding="utf-8")
    _git(clone, "add", "-f", ".userfiles/remote-secret.txt")
    _git(clone, "commit", "-m", "track private path")
    _git(clone, "push", "origin", "HEAD:main")
    return remote


def test_pull_blocks_before_head_or_private_files_change(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    local_private = root / ".userfiles" / "local.txt"
    local_private.write_text("keep me\n", encoding="utf-8")
    before = _git(root, "rev-parse", "HEAD")
    remote = _remote_with_tracked_userfiles(tmp_path)

    result = git_ops.git_fetch(
        room_id,
        f"file://{remote}",
        branch="main",
        target_branch="dev",
    )

    assert result["ok"] is False
    assert result["userfiles_policy_conflict"] is True
    assert result["userfiles_tracked"] == [".userfiles/remote-secret.txt"]
    assert result["userfiles_tracked_details"][0]["size"] == len("remote only\n".encode())
    assert _git(root, "rev-parse", "HEAD") == before
    assert local_private.read_text(encoding="utf-8") == "keep me\n"
    assert not (root / ".userfiles" / "remote-secret.txt").exists()


def test_backfill_reports_already_tracked_local_private_path(isolated_repo):
    room_id, root = isolated_repo
    private = root / ".userfiles" / "forced.txt"
    private.write_text("forced\n", encoding="utf-8")
    _git(root, "add", "-f", ".userfiles/forced.txt")

    result = git_ops.backfill_local_excludes(room_id)

    assert result["userfiles_policy_conflict"] is True
    assert result["userfiles_tracked"] == [".userfiles/forced.txt"]
    assert result["userfiles_tracked_details"][0]["source"] == "local_index"
