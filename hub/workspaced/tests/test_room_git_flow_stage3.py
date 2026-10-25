"""Stage 3 of internal design notes — risk diff + auto-merge revert (git layer).

Covers:
  * ``git_diff(merge_base=True)`` with two refs → three-dot diff: only the
    changes the task branch introduced, not commits that landed on the base
    afterwards;
  * ``git_merge_branch`` reports the new merge commit sha;
  * ``git_revert_commit`` undoes a merge on ``dev`` with a fresh revert commit
    and leaves the tree clean.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "stage3-room"
    (tmp_path / room_id).mkdir(parents=True)
    git_ops.git_init(room_id)  # main + dev, HEAD on dev
    return room_id, tmp_path / room_id


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_three_dot_diff_excludes_later_base_commits(isolated_repo):
    room_id, root = isolated_repo
    _write(root, "seed.txt", "seed\n")
    git_ops.git_commit(room_id, "seed on dev")

    git_ops.git_create_branch(room_id, "task/abc", base="dev")
    _write(root, "feature.txt", "hi\n")
    git_ops.git_commit(room_id, "task adds feature.txt")

    # a commit lands on dev *after* the task branched
    git_ops._run(["checkout", "dev"], cwd=root)
    _write(root, "unrelated.txt", "later\n")
    git_ops.git_commit(room_id, "dev moves on")
    git_ops._run(["checkout", "task/abc"], cwd=root)

    stat = git_ops.git_diff(
        room_id, from_ref="dev", to_ref="task/abc", stat=True, merge_base=True
    )
    assert "feature.txt" in stat
    assert "unrelated.txt" not in stat  # two-dot would show it as a deletion


def test_merge_returns_sha_and_revert_undoes_it(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_create_branch(room_id, "task/abc", base="dev")
    _write(root, "feature.txt", "hello\n")
    git_ops.git_commit(room_id, "task work")

    merge = git_ops.git_merge_branch(room_id, "task/abc", target="dev")
    assert merge["ok"] and merge["sha"]
    assert (root / "feature.txt").exists()

    revert = git_ops.git_revert_commit(room_id, merge["sha"], target="dev")
    assert revert["ok"] and not revert["conflict"]
    assert not (root / "feature.txt").exists()

    status = git_ops._run(["status", "--porcelain"], cwd=root).stdout.strip()
    assert status == ""
    # main never involved
    assert git_ops.git_current_branch(room_id) == "dev"
