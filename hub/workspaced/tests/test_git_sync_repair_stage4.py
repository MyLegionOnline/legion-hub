"""
Stage 4 of internal design notes — restore/dry-run building block.

`git_ls_tree()` is what the hub backend's recovery preview (see
hub/backend/tests/test_git_sync_repair_stage4.py) uses to compare volume
sizes against a candidate restore ref without touching the working tree —
exact byte sizes straight from the git object store, so a file zeroed out by
the sync-files incident shows up as size 0 at the bad commit and its real
size at the good one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "stage4-room"
    root = tmp_path / room_id
    root.mkdir(parents=True)
    git_ops.git_init(room_id)
    return room_id, root


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_git_ls_tree_returns_exact_blob_sizes(isolated_repo):
    room_id, root = isolated_repo
    _write(root, "a.txt", "hello")       # 5 bytes
    _write(root, "src/b.txt", "hi there")  # 8 bytes
    sha = git_ops.git_commit(room_id, "add files")
    assert sha

    entries = {e["path"]: e["size"] for e in git_ops.git_ls_tree(room_id, sha)}
    assert entries["a.txt"] == 5
    assert entries["src/b.txt"] == 8


def test_git_ls_tree_reflects_a_zeroed_file(isolated_repo):
    """Reproduces the incident shape: a file goes from real content to 0
    bytes in a later commit. ls-tree at the good ref must still report the
    real size, so a restore preview can flag the later state as dangerous."""
    room_id, root = isolated_repo
    content = "console.log(1)"
    _write(root, "app.js", content)
    good_sha = git_ops.git_commit(room_id, "good state")
    assert good_sha

    _write(root, "app.js", "")  # zeroed out, like the sync bug did
    bad_sha = git_ops.git_commit(room_id, "bad sync")
    assert bad_sha

    good_entries = {e["path"]: e["size"] for e in git_ops.git_ls_tree(room_id, good_sha)}
    bad_entries = {e["path"]: e["size"] for e in git_ops.git_ls_tree(room_id, bad_sha)}
    assert good_entries["app.js"] == len(content)
    assert bad_entries["app.js"] == 0


def test_git_ls_tree_scoped_to_path(isolated_repo):
    room_id, root = isolated_repo
    _write(root, "a.txt", "x")
    _write(root, "src/b.txt", "yy")
    sha = git_ops.git_commit(room_id, "add")

    entries = git_ops.git_ls_tree(room_id, sha, path="src")
    assert [e["path"] for e in entries] == ["src/b.txt"]


def test_git_ls_tree_missing_repo_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    assert git_ops.git_ls_tree("no-such-room", "HEAD") == []


def test_git_revert_all_files_restores_zeroed_content(isolated_repo):
    """End-to-end: git_revert(ref, path=None) — the primitive Stage 4's
    /restore/apply calls — brings a zeroed file back to its good content."""
    room_id, root = isolated_repo
    _write(root, "app.js", "console.log(1)")
    good_sha = git_ops.git_commit(room_id, "good state")
    _write(root, "app.js", "")
    git_ops.git_commit(room_id, "bad sync")
    assert (root / "app.js").read_text(encoding="utf-8") == ""

    ok = git_ops.git_revert(room_id, good_sha, path=None)

    assert ok is True
    assert (root / "app.js").read_text(encoding="utf-8") == "console.log(1)"
