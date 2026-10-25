"""Stage 1 of internal design notes — task/goal branches (git layer).

Covers:
  * ``git_create_branch(base=...)`` branches off an explicit ref, not HEAD;
  * it is idempotent (a second call just checks the branch out);
  * ``git_merge_branch(target=...)`` merges into an arbitrary branch and
    leaves the working tree on that target;
  * ``theirs=True`` auto-resolves overlapping hunks; a genuine add/add
    conflict is reported and the tree is left clean.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "stage1-room"
    (tmp_path / room_id).mkdir(parents=True)
    git_ops.git_init(room_id)  # main + dev, HEAD on dev
    return room_id, tmp_path / room_id


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _branches(root: Path) -> set[str]:
    r = git_ops._run(["branch", "--format=%(refname:short)"], cwd=root)
    return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}


def test_create_branch_off_explicit_base(isolated_repo):
    room_id, root = isolated_repo
    # a commit that lives only on dev
    _write(root, "dev_only.txt", "x")
    git_ops.git_commit(room_id, "dev commit")

    # branch a task off `dev`
    assert git_ops.git_create_branch(room_id, "task/abc", base="dev")
    assert git_ops.git_current_branch(room_id) == "task/abc"
    assert (root / "dev_only.txt").exists()

    # branch another task off `main` — must NOT see the dev-only file
    assert git_ops.git_create_branch(room_id, "task/def", base="main")
    assert git_ops.git_current_branch(room_id) == "task/def"
    assert not (root / "dev_only.txt").exists()


def test_create_branch_is_idempotent(isolated_repo):
    room_id, root = isolated_repo
    assert git_ops.git_create_branch(room_id, "task/abc", base="dev")
    git_ops._run(["checkout", "dev"], cwd=root)
    # second call: branch exists → just check it out, base ignored
    assert git_ops.git_create_branch(room_id, "task/abc", base="main")
    assert git_ops.git_current_branch(room_id) == "task/abc"


def test_merge_into_arbitrary_target(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_create_branch(room_id, "task/abc", base="dev")
    _write(root, "feature.txt", "hello")
    git_ops.git_commit(room_id, "task work")

    result = git_ops.git_merge_branch(room_id, "task/abc", target="dev")
    assert result["ok"] and not result["conflict"]
    assert git_ops.git_current_branch(room_id) == "dev"
    assert (root / "feature.txt").read_text() == "hello"

    # main untouched
    git_ops._run(["checkout", "main"], cwd=root)
    assert not (root / "feature.txt").exists()


def test_merge_theirs_resolves_overlap(isolated_repo):
    room_id, root = isolated_repo
    _write(root, "shared.txt", "base\n")
    git_ops.git_commit(room_id, "seed shared.txt on dev")

    git_ops.git_create_branch(room_id, "task/abc", base="dev")
    _write(root, "shared.txt", "from-task\n")
    git_ops.git_commit(room_id, "task edit")

    git_ops._run(["checkout", "dev"], cwd=root)
    _write(root, "shared.txt", "from-dev\n")
    git_ops.git_commit(room_id, "dev edit")

    result = git_ops.git_merge_branch(room_id, "task/abc", target="dev", theirs=True)
    assert result["ok"] and not result["conflict"]
    assert (root / "shared.txt").read_text() == "from-task\n"


def test_merge_genuine_conflict_is_reported_and_tree_clean(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_create_branch(room_id, "task/abc", base="dev")
    _write(root, "new.txt", "task version\n")
    git_ops.git_commit(room_id, "task adds new.txt")

    git_ops._run(["checkout", "dev"], cwd=root)
    _write(root, "new.txt", "dev version\n")
    git_ops.git_commit(room_id, "dev adds new.txt")

    result = git_ops.git_merge_branch(room_id, "task/abc", target="dev", theirs=False)
    assert result["conflict"] is True
    assert result["ok"] is False
    # tree left clean (merge aborted)
    status = git_ops._run(["status", "--porcelain"], cwd=root).stdout.strip()
    assert status == ""
