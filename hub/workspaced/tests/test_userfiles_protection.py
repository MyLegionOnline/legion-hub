"""``.userfiles`` can never be staged, even when the exclude rule is missing.

Regression for the 2026-10-06 leak: a room whose ``.git/info/exclude`` predated
the ``.userfiles/`` rule committed uploaded photos with ``git add -A`` and the
next push published them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import file_ops, generation, git_ops


@pytest.fixture
def legacy_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A repo whose exclude file lacks the private-files rule."""
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_ops, "WORKSPACE_DATA_DIR", tmp_path)
    monkeypatch.setattr(generation, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "userfiles-protection"
    git_ops.git_init(room_id)
    root = tmp_path / room_id
    exclude = root / ".git" / "info" / "exclude"
    exclude.write_text(
        "\n".join(
            line for line in exclude.read_text(encoding="utf-8").splitlines()
            if line != ".userfiles/"
        )
        + "\n",
        encoding="utf-8",
    )
    assert ".userfiles/" not in exclude.read_text(encoding="utf-8")
    return room_id, root


def _tracked(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--", ".userfiles"], cwd=root, capture_output=True, text=True
    )
    return out.stdout.split()


def test_commit_does_not_stage_userfiles_when_exclude_rule_is_missing(legacy_repo):
    room_id, root = legacy_repo
    (root / ".userfiles").mkdir(exist_ok=True)
    (root / ".userfiles" / "sara.jpg").write_bytes(b"\xff\xd8\xff")
    (root / "index.html").write_text("<p>hi</p>", encoding="utf-8")

    sha = git_ops.git_commit(room_id, "user: change", allow_main_commit=True)

    assert sha
    assert _tracked(root) == []
    # The commit repaired the missing local rule instead of relying on a backfill.
    assert ".userfiles/" in (root / ".git" / "info" / "exclude").read_text(encoding="utf-8")


def test_commit_does_not_stage_changes_to_an_already_tracked_private_file(legacy_repo):
    room_id, root = legacy_repo
    (root / ".userfiles").mkdir(exist_ok=True)
    private = root / ".userfiles" / "old.txt"
    private.write_text("v1", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".userfiles/old.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "tracked by accident"], cwd=root, check=True,
                   capture_output=True)
    private.write_text("v2 secret", encoding="utf-8")

    assert git_ops.git_commit(room_id, "user: change", allow_main_commit=True) is None

    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                            capture_output=True, text=True).stdout.split()
    assert staged == []


def test_private_write_and_delete_never_commit(legacy_repo):
    room_id, root = legacy_repo
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True).stdout

    written = file_ops.write_file(room_id, ".userfiles/a.txt", "x", commit_message="user: upload")
    assert "commit_sha" not in written
    assert file_ops.delete_file(room_id, ".userfiles/a.txt", commit_message="user: delete")

    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True).stdout == head
    assert _tracked(root) == []


def test_project_write_still_commits(legacy_repo):
    room_id, root = legacy_repo
    written = file_ops.write_file(room_id, "src/a.txt", "x", commit_message="user: upload src/a.txt")
    assert written["commit_sha"]


def test_park_and_untrack_stage_paths_skip_userfiles(legacy_repo):
    room_id, root = legacy_repo
    (root / ".userfiles").mkdir(exist_ok=True)
    (root / ".userfiles" / "kevin.jpg").write_bytes(b"\xff\xd8\xff")

    git_ops._stage_all(root)

    assert subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                          capture_output=True, text=True).stdout.split() == []


def test_backfill_reports_protection_from_the_file_on_disk(legacy_repo):
    room_id, root = legacy_repo

    result = git_ops.backfill_local_excludes(room_id)

    assert result["updated"] is True
    assert result["userfiles_protected"] is True


def test_backfill_without_a_repo_is_protected_because_git_init_writes_the_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_ops, "WORKSPACE_DATA_DIR", tmp_path)

    result = git_ops.backfill_local_excludes("never-inited")

    assert result["error"] == "no git repo"
    assert result["userfiles_protected"] is True
