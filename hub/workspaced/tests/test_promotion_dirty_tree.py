"""git_ops.py — a promotion / gate merge must not dead-end on a dirty working
tree, and Legion's per-volume metadata must never enter git history.

Regression for the field failure where ``.legion/generation.json`` (rewritten by
workspaced on every mutation) was tracked, left the tree permanently dirty, and
made every ``dev → main`` promotion abort at ``git checkout main`` with
"Your local changes to the following files would be overwritten by checkout".

Fixes:
  1. ``git_merge_branch`` / ``git_revert_commit`` now park a dirty tree before
     the checkout — the same guard ``git_fetch`` ("Pull from Git") already used.
  2. ``.legion/generation.json`` and ``.legion/runs/`` joined
     ``_LEGION_LOCAL_EXCLUDES``; a legacy room that still tracks them is healed
     in place the first time it runs a gate merge.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "dirty-tree-room"
    (tmp_path / room_id).mkdir(parents=True)
    git_ops.git_init(room_id)  # main + dev, HEAD on dev
    return room_id, tmp_path / room_id


def _g(root: Path, *args: str) -> str:
    return git_ops._run(list(args), cwd=root).stdout.strip()


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# ── the excludes ────────────────────────────────────────────────────────────

def test_volatile_legion_paths_are_excluded():
    lines = git_ops._LEGION_LOCAL_EXCLUDES.splitlines()
    assert ".legion/generation.json" in lines
    assert ".legion/runs/" in lines
    # a real, shared CI config stays tracked
    assert ".legion/run.json" not in lines
    assert set(git_ops._VOLATILE_LEGION_PATHS) == {".legion/generation.json", ".legion/runs"}


def test_fresh_room_keeps_volatile_metadata_out_of_git(repo):
    room_id, root = repo
    _write(root, ".legion/generation.json", '{"epoch":"e","revision":3}')
    _write(root, ".legion/runs/dev.log", "started\n")
    _write(root, "index.html", "<h1>hi</h1>")
    git_ops.git_commit(room_id, "add page")

    tracked = _g(root, "ls-files").splitlines()
    assert "index.html" in tracked
    assert ".legion/generation.json" not in tracked
    assert not any(t.startswith(".legion/runs/") for t in tracked)
    assert _g(root, "status", "--porcelain") == ""


# ── promotion over a dirty tree ─────────────────────────────────────────────

def test_promotion_parks_a_dirty_tracked_file(repo):
    room_id, root = repo
    _write(root, "app.py", "v1\n")
    git_ops.git_commit(room_id, "seed")
    git_ops.git_merge_branch(room_id, "dev", target="main", theirs=False)  # main gets content
    git_ops._run(["checkout", "dev"], cwd=root)

    _write(root, "feature.txt", "shipped\n")
    git_ops.git_commit(room_id, "feature")
    _write(root, "app.py", "v2-uncommitted\n")  # dirty tracked file

    res = git_ops.git_merge_branch(room_id, "dev", target="main", theirs=False)

    assert res["ok"] and not res["conflict"], res
    assert _g(root, "show", "main:feature.txt") == "shipped"
    # the uncommitted edit was parked on `dev`, not dropped
    assert "v2-uncommitted" in _g(root, "show", "dev:app.py")


def test_promotion_never_commits_a_dirty_tree_directly_onto_main(repo):
    """§0.1 — only a promotion moves ``main``. A dirty tree while HEAD sits on
    ``main`` is parked on ``dev`` and only reaches ``main`` through the merge,
    never as a plain commit written straight onto the release branch."""
    room_id, root = repo
    _write(root, "app.py", "v1\n")
    git_ops.git_commit(room_id, "seed")
    git_ops.git_merge_branch(room_id, "dev", target="main", theirs=False)
    # HEAD now on main; leave an uncommitted file there
    _write(root, "stray.txt", "oops\n")

    res = git_ops.git_merge_branch(room_id, "dev", target="main", theirs=False)

    assert res["ok"], res
    # parked on `dev`, not dropped
    assert _g(root, "show", "dev:stray.txt") == "oops"
    # `main` advanced only by the merge commit (sha + two parents = 3 tokens)
    assert len(_g(root, "rev-list", "--parents", "-n", "1", "main").split()) == 3


def test_promotion_heals_a_legacy_tracked_generation_json(repo):
    room_id, root = repo
    _write(root, "app.py", "v1\n")
    git_ops.git_commit(room_id, "seed")
    git_ops.git_merge_branch(room_id, "dev", target="main", theirs=False)

    # pre-fix room: generation.json force-committed on BOTH branches
    for branch, rev in (("main", "1"), ("dev", "2")):
        git_ops._run(["checkout", branch], cwd=root)
        _write(root, ".legion/generation.json", f'{{"epoch":"e","revision":{rev}}}')
        git_ops._run(["add", "-f", "--", ".legion/generation.json"], cwd=root)
        git_ops._run(["commit", "-m", f"legacy: track generation.json on {branch}"], cwd=root)

    git_ops._run(["checkout", "dev"], cwd=root)
    _write(root, "feature.txt", "shipped\n")
    git_ops.git_commit(room_id, "feature")
    _write(root, ".legion/generation.json", '{"epoch":"e","revision":99}')  # workspaced rewrote it

    res = git_ops.git_merge_branch(room_id, "dev", target="main", theirs=False)

    assert res["ok"] and not res["conflict"], res
    assert _g(root, "show", "main:feature.txt") == "shipped"
    # generation.json is gone from history on both branches
    assert ".legion/generation.json" not in _g(root, "ls-tree", "-r", "--name-only", "main")
    assert ".legion/generation.json" not in _g(root, "ls-tree", "-r", "--name-only", "dev")


def test_revert_commit_parks_a_dirty_tree(repo):
    room_id, root = repo
    git_ops.git_create_branch(room_id, "task/x", base="dev")
    _write(root, "f.txt", "feature\n")
    git_ops.git_commit(room_id, "feature")
    merge = git_ops.git_merge_branch(room_id, "task/x", target="dev")
    assert merge["ok"], merge

    _write(root, "scratch.txt", "uncommitted\n")  # dirty tree
    res = git_ops.git_revert_commit(room_id, merge["sha"], target="dev", mainline=1)

    assert res["ok"] and not res["conflict"], res
    assert "f.txt" not in _g(root, "ls-tree", "-r", "--name-only", "dev")  # reverted
    assert _g(root, "show", "dev:scratch.txt") == "uncommitted"  # parked, not lost
