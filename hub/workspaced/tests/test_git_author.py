"""Room commits are authored by the room's person and co-authored by Legion."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops
from app.config import LEGION_GIT_EMAIL, LEGION_GIT_NAME

_TRAILER = f"Co-Authored-By: {LEGION_GIT_NAME} <{LEGION_GIT_EMAIL}>"


def _g(root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(root), *args],
        capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


@pytest.fixture
def room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "author-room"
    git_ops.git_init(room_id)
    return room_id, tmp_path / room_id


def _head(root: Path, ref: str = "HEAD") -> tuple[str, str, str]:
    out = _g(root, "log", "-1", "--format=%an <%ae>%x00%cn <%ce>%x00%B", ref)
    author, committer, body = out.split("\x00")
    return author, committer, body


def test_without_author_legion_authors_and_no_trailer(room):
    room_id, root = room
    (root / "a.txt").write_text("a\n")
    assert git_ops.git_commit(room_id, "add a")

    author, committer, body = _head(root)
    assert author == f"{LEGION_GIT_NAME} <{LEGION_GIT_EMAIL}>"
    assert committer == author
    assert "Co-Authored-By" not in body


def test_person_authors_legion_commits_and_coauthors(room):
    room_id, root = room
    git_ops.git_set_author(room_id, "Ada", "1+ada@users.noreply.github.com")
    (root / "a.txt").write_text("a\n")
    assert git_ops.git_commit(room_id, "add a")

    author, committer, body = _head(root)
    assert author == "Ada <1+ada@users.noreply.github.com>"
    assert committer == f"{LEGION_GIT_NAME} <{LEGION_GIT_EMAIL}>"
    assert body.splitlines()[-1] == _TRAILER


def test_merge_commit_gets_trailer(room):
    room_id, root = room
    git_ops.git_set_author(room_id, "Ada", "ada@example.com")
    _g(root, "checkout", "-b", "task/1")
    (root / "t.txt").write_text("t\n")
    assert git_ops.git_commit(room_id, "task work")

    result = git_ops.git_merge_branch(room_id, "task/1", target="dev")
    assert result["ok"], result

    author, _, body = _head(root, "dev")
    assert author == "Ada <ada@example.com>"
    assert body.count(_TRAILER) == 1


def test_clearing_author_returns_to_legion(room):
    room_id, root = room
    git_ops.git_set_author(room_id, "Ada", "ada@example.com")
    git_ops.git_set_author(room_id, None, None)
    (root / "a.txt").write_text("a\n")
    assert git_ops.git_commit(room_id, "add a")

    author, _, body = _head(root)
    assert author == f"{LEGION_GIT_NAME} <{LEGION_GIT_EMAIL}>"
    assert "Co-Authored-By" not in body


def test_set_author_replaces_legacy_identity(room):
    room_id, root = room
    _g(root, "config", "user.name", "Legion Hub")
    _g(root, "config", "user.email", "hub@legion.ai")
    git_ops.git_set_author(room_id, "Ada", "ada@example.com")
    assert _g(root, "config", "user.email") == LEGION_GIT_EMAIL
    assert _g(root, "config", "user.name") == LEGION_GIT_NAME
