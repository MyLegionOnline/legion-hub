"""Stage 0 of internal design notes — branch-model foundation (git layer).

Covers:
  * ``git_init`` creates both ``main`` and ``dev`` and leaves HEAD on ``dev``.
  * ``git_commit`` refuses a commit on ``main`` unless ``allow_main_commit``.
  * ``git_revert`` (an explicit human operation) may still touch ``main``.
  * ``ensure_base_branches`` backfills ``dev`` for a repo that predates this
    stage, is idempotent, and never changes HEAD.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "stage0-room"
    (tmp_path / room_id).mkdir(parents=True)
    return room_id, tmp_path / room_id


def _branches(root: Path) -> set[str]:
    r = git_ops._run(["branch", "--format=%(refname:short)"], cwd=root)
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_git_init_creates_main_and_dev_with_head_on_dev(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_init(room_id)

    assert _branches(root) == {"main", "dev"}
    assert git_ops.git_current_branch(room_id) == "dev"


def test_commit_on_main_is_rejected_without_override(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_init(room_id)
    git_ops._run(["checkout", "main"], cwd=root)

    _write(root, "a.txt", "hello")
    with pytest.raises(git_ops.MainBranchProtectedError):
        git_ops.git_commit(room_id, "should not land on main")

    # nothing was committed
    log = git_ops.git_log(room_id, limit=10)
    assert all("should not land" not in e["message"] for e in log)


def test_commit_on_main_allowed_with_override(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_init(room_id)
    git_ops._run(["checkout", "main"], cwd=root)

    _write(root, "a.txt", "hello")
    sha = git_ops.git_commit(room_id, "human commit", allow_main_commit=True)
    assert sha


def test_commit_on_dev_is_allowed(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_init(room_id)  # HEAD == dev

    _write(root, "a.txt", "hello")
    sha = git_ops.git_commit(room_id, "work on dev")
    assert sha
    assert git_ops.git_current_branch(room_id) == "dev"


def test_revert_still_works_on_main(isolated_repo):
    room_id, root = isolated_repo
    git_ops.git_init(room_id)
    _write(root, "a.txt", "v1")
    git_ops.git_commit(room_id, "add a.txt on dev")

    # promote to main by hand so there is something to revert against
    git_ops._run(["checkout", "main"], cwd=root)
    git_ops._run(["merge", "dev", "-m", "merge dev"], cwd=root)
    _write(root, "a.txt", "v2")
    git_ops.git_commit(room_id, "change a.txt", allow_main_commit=True)

    ok = git_ops.git_revert(room_id, "HEAD~1", path="a.txt")
    assert ok
    assert (root / "a.txt").read_text() == "v1"


def test_ensure_base_branches_backfills_dev(isolated_repo):
    room_id, root = isolated_repo
    # Simulate a pre-Stage-0 repo: init then drop `dev`.
    git_ops.git_init(room_id)
    git_ops._run(["checkout", "main"], cwd=root)
    git_ops._run(["branch", "-D", "dev"], cwd=root)
    assert _branches(root) == {"main"}

    result = git_ops.ensure_base_branches(room_id)
    assert result == {"ok": True, "created_dev": True}
    assert _branches(root) == {"main", "dev"}
    # HEAD untouched
    assert git_ops.git_current_branch(room_id) == "main"

    # idempotent
    again = git_ops.ensure_base_branches(room_id)
    assert again == {"ok": True, "created_dev": False}


def test_ensure_base_branches_no_repo(isolated_repo):
    room_id, _ = isolated_repo
    result = git_ops.ensure_base_branches(room_id)
    assert result["ok"] is False
