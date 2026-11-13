"""Legion's own bookkeeping commit never sweeps up an agent's pending edits.

The Hub commits ``.legion/services.json`` whenever the Browser catalog changes,
which can happen while an agent is mid-task. ``git_commit(only_path=...)``
must commit that one file and leave every other change in the tree untouched.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import file_ops, git_ops


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_ops, "WORKSPACE_DATA_DIR", tmp_path, raising=False)
    room_id = "only-path-room"
    (tmp_path / room_id).mkdir(parents=True)
    git_ops.git_init(room_id)  # main + dev, HEAD on dev
    return room_id, tmp_path / room_id


def _g(root: Path, *args: str) -> str:
    return git_ops._run(list(args), cwd=root).stdout.strip()


def test_commit_only_path_leaves_other_changes_pending(repo):
    room_id, root = repo
    (root / "app.py").write_text("print('half done')\n", encoding="utf-8")
    (root / ".legion").mkdir(exist_ok=True)
    (root / ".legion" / "services.json").write_text('{"version": 1, "services": []}\n', encoding="utf-8")

    sha = git_ops.git_commit(room_id, "legion: update Browser services manifest",
                             only_path=".legion/services.json")

    assert sha
    assert _g(root, "show", "--name-only", "--format=", "HEAD").splitlines() == [".legion/services.json"]
    assert "app.py" in _g(root, "status", "--porcelain")


def test_commit_only_path_with_a_room_author(repo):
    # A room author adds a --trailer option; it must land before the pathspec.
    room_id, root = repo
    git_ops.git_set_author(room_id, "Ada", "ada@example.com")
    (root / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "m.json").write_text("{}\n", encoding="utf-8")

    assert git_ops.git_commit(room_id, "legion: manifest", only_path="m.json")
    assert _g(root, "show", "--name-only", "--format=", "HEAD").splitlines() == ["m.json"]
    assert _g(root, "log", "-1", "--format=%an") == "Ada"
    assert "app.py" in _g(root, "status", "--porcelain")


def test_commit_only_path_still_refuses_main(repo):
    room_id, root = repo
    _g(root, "checkout", "main")
    (root / "x.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(git_ops.MainBranchProtectedError):
        git_ops.git_commit(room_id, "m", only_path="x.json")
