"""Stage 2 of internal design notes — renaming a dialogue branch.

A direct dialogue accumulates its work on ``session/<session_id>``; when that
work is written down as a task, the branch takes the canonical ``task/<id>``
name. The rename happens while the branch is checked out and must never
overwrite an existing branch.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "rename-room"
    (tmp_path / room_id).mkdir(parents=True)
    git_ops.git_init(room_id)  # main + dev, HEAD on dev
    return room_id, tmp_path / room_id


def _write(root: Path, rel: str, content: str) -> None:
    (root / rel).write_text(content, encoding="utf-8")


def _branches(root: Path) -> set[str]:
    r = git_ops._run(["branch", "--format=%(refname:short)"], cwd=root)
    return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}


def test_rename_keeps_the_commits_and_the_checkout(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_create_branch(room_id, "session/s1", base="dev")
    _write(root, "style.css", ".article { margin: 0 auto; }")
    sha = git_ops.git_commit(room_id, "agent: fix margins")

    assert git_ops.git_rename_branch(room_id, "session/s1", "task/t1") is True

    assert _branches(root) >= {"task/t1", "dev", "main"}
    assert "session/s1" not in _branches(root)
    # still checked out, still holding the work
    assert git_ops.git_current_branch(room_id) == "task/t1"
    assert (root / "style.css").exists()
    head = git_ops._run(["rev-parse", "HEAD"], cwd=root).stdout.strip()
    assert head == sha


def test_rename_of_a_branch_that_is_not_checked_out(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_create_branch(room_id, "session/s2", base="dev")
    _write(root, "a.txt", "x")
    git_ops.git_commit(room_id, "work")
    git_ops.git_create_branch(room_id, "dev")  # move HEAD away

    assert git_ops.git_rename_branch(room_id, "session/s2", "task/t2") is True
    assert "task/t2" in _branches(root)
    assert git_ops.git_current_branch(room_id) == "dev"


def test_rename_refuses_to_overwrite_an_existing_branch(isolated_repo):
    """A name collision means somebody else's history is at stake — the caller
    keeps the old name instead."""
    room_id, root = isolated_repo
    git_ops.git_create_branch(room_id, "task/taken", base="dev")
    git_ops.git_create_branch(room_id, "session/s3", base="dev")

    assert git_ops.git_rename_branch(room_id, "session/s3", "task/taken") is False
    assert {"task/taken", "session/s3"} <= _branches(root)


def test_rename_is_idempotent_when_already_done(isolated_repo):
    room_id, _ = isolated_repo
    git_ops.git_create_branch(room_id, "task/t4", base="dev")

    # the source is gone and the destination is there: a repeated call (a retry
    # after a lost response) reports success rather than a phantom failure
    assert git_ops.git_rename_branch(room_id, "session/s4", "task/t4") is True


def test_rename_of_a_missing_branch_fails(isolated_repo):
    room_id, _ = isolated_repo
    assert git_ops.git_rename_branch(room_id, "session/nope", "task/nope") is False


def test_rename_rejects_degenerate_arguments(isolated_repo):
    room_id, _ = isolated_repo
    git_ops.git_create_branch(room_id, "session/s5", base="dev")

    assert git_ops.git_rename_branch(room_id, "session/s5", "session/s5") is False
    assert git_ops.git_rename_branch(room_id, "", "task/t5") is False
    assert git_ops.git_rename_branch(room_id, "session/s5", "") is False
