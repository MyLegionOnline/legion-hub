"""Stage 6 of internal design notes — ``git_push`` sends the *local* branch.

The refspec changed from ``HEAD:{branch}`` to
``refs/heads/{branch}:refs/heads/{branch}`` so Gate 2 can push ``dev`` to the
remote for an external CI while the working tree stays on ``dev`` — and a
manual ``main`` push sends ``main`` even though HEAD is on ``dev``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "stage6-room"
    git_ops.git_init(room_id)          # HEAD → dev, main also exists
    return room_id, tmp_path / room_id


def _bare_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], capture_output=True, check=True)
    return remote


def _remote_branches(remote: Path) -> set[str]:
    out = subprocess.run(
        ["git", "-C", str(remote), "for-each-ref", "--format=%(refname:short)", "refs/heads"],
        capture_output=True, text=True, check=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


def test_push_sends_named_branch_not_head(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    # A commit on dev (HEAD); main stays on the init commit only.
    (root / "feature.txt").write_text("work", encoding="utf-8")
    git_ops.git_commit(room_id, "feature on dev")

    remote = _bare_remote(tmp_path)

    # Push `main` while HEAD is on `dev` — the remote must get `main` only,
    # and it must NOT carry the dev-only commit.
    result = git_ops.git_push(room_id, f"file://{remote}", branch="main", token=None, mode="push")
    assert result["ok"] is True, result["stderr"]
    assert _remote_branches(remote) == {"main"}

    log = subprocess.run(
        ["git", "-C", str(remote), "log", "--format=%s", "main"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "feature on dev" not in log

    # Now push `dev` — same working tree, different branch shipped.
    result = git_ops.git_push(room_id, f"file://{remote}", branch="dev", token=None, mode="push")
    assert result["ok"] is True, result["stderr"]
    assert _remote_branches(remote) == {"main", "dev"}
    dev_log = subprocess.run(
        ["git", "-C", str(remote), "log", "--format=%s", "dev"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "feature on dev" in dev_log


def test_push_blocks_indexed_private_file_before_network(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    private_file = root / ".userfiles" / "secret.txt"
    private_file.write_text("do not publish", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", "-f", ".userfiles/secret.txt"],
        check=True,
    )

    result = git_ops.git_push(
        room_id, f"file://{tmp_path / 'unused.git'}", branch="dev", token=None, mode="force"
    )

    assert result["ok"] is False
    assert result["userfiles_push_blocked"] is True
    assert result["userfiles_tracked"] == [".userfiles/secret.txt"]
    assert "git rm -r --cached .userfiles" in result["stderr"]
