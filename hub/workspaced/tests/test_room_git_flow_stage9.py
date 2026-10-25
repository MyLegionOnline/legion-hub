"""Stage 9 of internal design notes — ``git_log`` accepts a ``rev_range``.

The deploy hook lists the commits a promotion carries by asking for
``git log main..dev``. ``rev_range`` restricts the log to that range; a value
that looks like a git option is ignored.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "stage9-room"
    git_ops.git_init(room_id)  # HEAD → dev, main exists at the init commit
    root = tmp_path / room_id
    return room_id, root


def _commit(room_id: str, root: Path, name: str, msg: str) -> None:
    (root / name).write_text(name, encoding="utf-8")
    git_ops.git_commit(room_id, msg)


def test_git_log_rev_range_lists_only_range(repo):
    room_id, root = repo
    _commit(room_id, root, "a.txt", "first on dev")
    _commit(room_id, root, "b.txt", "second on dev")

    msgs = [e["message"] for e in git_ops.git_log(room_id, rev_range="main..dev", limit=50)]
    assert msgs == ["second on dev", "first on dev"]

    # Whole history (no range) still includes the init commit.
    assert len(git_ops.git_log(room_id, limit=50)) >= 3


def test_git_log_rev_range_empty_when_nothing_ahead(repo):
    room_id, _ = repo
    assert git_ops.git_log(room_id, rev_range="main..dev", limit=50) == []


def test_git_log_rejects_option_like_rev_range(repo):
    room_id, root = repo
    _commit(room_id, root, "a.txt", "on dev")
    # A leading dash must not be handed to `git log` as an option.
    out = git_ops.git_log(room_id, rev_range="--all", limit=50)
    assert [e["message"] for e in out] == [e["message"] for e in git_ops.git_log(room_id, limit=50)]
