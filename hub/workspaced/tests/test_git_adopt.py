"""``git_adopt_remote`` — take an external repo as the room's base.

internal design notes §3. A plain "Pull from Git" merges the remote
into ``dev`` with ``--allow-unrelated-histories``, leaving the room with two
roots: the first Gate 2 then shows the whole external repo as ``main..dev``
and the promotion push carries a synthetic root commit. ``git_adopt_remote``
transplants the remote onto ``main`` while the room is still pristine so no
unrelated history is ever created.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops

# ``git_ops._run`` pins ``core.autocrlf=false``; mirror that in the test's own
# git calls so a CRLF-rewriting global config can't leave the tree "dirty"
# between the two.
_CFG = ["-c", "core.autocrlf=false"]


def _g(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(
        ["git", *_CFG, "-C", str(root), *args],
        capture_output=True, text=True, check=check,
    )
    return r.stdout.strip()


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "adopt-room"
    git_ops.git_init(room_id)
    return room_id, tmp_path / room_id


def _seeded_remote(tmp_path: Path, branch: str = "main", commits: int = 2) -> Path:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], capture_output=True, check=True)
    clone = tmp_path / "seed-clone"
    subprocess.run(["git", "clone", str(remote), str(clone)], capture_output=True, check=True)
    _g(clone, "config", "user.name", "Remote Author")
    _g(clone, "config", "user.email", "remote@example.com")
    for i in range(commits):
        (clone / f"file{i}.txt").write_text(f"remote content {i}\n", encoding="utf-8")
        _g(clone, "add", "-A")
        _g(clone, "commit", "-m", f"remote commit {i}")
    _g(clone, "push", "origin", f"HEAD:{branch}")
    return remote


def _commit_on_main(root: Path, rel: str, content: str) -> None:
    _g(root, "checkout", "main")
    (root / rel).write_text(content, encoding="utf-8")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", f"local: {rel}")


def _file_on(root: Path, ref: str, rel: str) -> str | None:
    r = subprocess.run(
        ["git", *_CFG, "-C", str(root), "show", f"{ref}:{rel}"],
        capture_output=True, text=True,
    )
    return r.stdout if r.returncode == 0 else None


def test_pristine_room_transplants_remote_onto_main(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path, commits=3)
    remote_tip = _g(tmp_path / "seed-clone", "rev-parse", "HEAD")

    result = git_ops.git_adopt_remote(room_id, f"file://{remote}", branch="main", working_branch="dev")

    assert result["ok"] and result["adopted"] and not result["diverged"]
    assert result["dropped_init"] is True
    assert _g(root, "rev-parse", "main") == remote_tip
    assert _g(root, "rev-parse", "dev") == remote_tip
    # No synthetic Legion init commit anywhere in main's history.
    assert "init: Legion Hub workspace" not in _g(root, "log", "--format=%s", "main")
    # main..dev is empty — Gate 2 will diff only real room changes.
    assert _g(root, "rev-list", "--count", "main..dev") == "0"


def test_adopt_is_idempotent(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)

    first = git_ops.git_adopt_remote(room_id, f"file://{remote}", branch="main")
    main_after_first = _g(root, "rev-parse", "main")

    second = git_ops.git_adopt_remote(room_id, f"file://{remote}", branch="main")

    assert first["adopted"] and second["ok"]
    assert not second["diverged"]
    assert _g(root, "rev-parse", "main") == main_after_first


def test_room_with_work_and_divergence_leaves_main_untouched(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)
    _commit_on_main(root, "local.txt", "local work\n")
    main_before = _g(root, "rev-parse", "main")

    result = git_ops.git_adopt_remote(room_id, f"file://{remote}", branch="main")

    assert result["ok"] and not result["adopted"] and result["diverged"]
    assert _g(root, "rev-parse", "main") == main_before, "main must not move on divergence"


def test_force_transplants_a_room_with_work(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)
    remote_tip = _g(tmp_path / "seed-clone", "rev-parse", "HEAD")
    _commit_on_main(root, "local.txt", "local work\n")

    result = git_ops.git_adopt_remote(room_id, f"file://{remote}", branch="main", force=True)

    assert result["ok"] and result["adopted"]
    assert _g(root, "rev-parse", "main") == remote_tip


def test_force_refused_while_a_task_branch_is_open(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)

    git_ops.git_create_branch(room_id, "task/abc", base="dev")
    _commit_on_main(root, "local.txt", "local\n")
    main_before = _g(root, "rev-parse", "main")

    result = git_ops.git_adopt_remote(room_id, f"file://{remote}", branch="main", force=True)

    assert not result["ok"] and not result["adopted"]
    assert "task/abc" in result["stderr"]
    assert _g(root, "rev-parse", "main") == main_before


def test_fast_forward_when_main_is_a_strict_ancestor(isolated_repo, tmp_path):
    """A room adopted once, then the remote advances — a second adopt just
    fast-forwards main (no rewrite, no divergence)."""
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path, commits=1)
    git_ops.git_adopt_remote(room_id, f"file://{remote}", branch="main")

    clone = tmp_path / "seed-clone"
    (clone / "more.txt").write_text("more\n", encoding="utf-8")
    _g(clone, "add", "-A")
    _g(clone, "commit", "-m", "advance")
    _g(clone, "push", "origin", "HEAD:main")
    new_tip = _g(clone, "rev-parse", "HEAD")

    result = git_ops.git_adopt_remote(room_id, f"file://{remote}", branch="main")

    assert result["ok"] and result["adopted"] and not result["diverged"]
    assert _g(root, "rev-parse", "main") == new_tip
