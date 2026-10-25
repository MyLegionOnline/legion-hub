"""``room_git_lock`` — serialise HEAD-moving git ops on a room's one tree.

internal design notes §5 (Stage 3). The room volume is the single
working directory and many code paths run ``git checkout`` on it. Without a
lock two operations in the same room can switch each other's branch mid-run
and leave a half-merged tree or a clobbered index behind.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import pytest

from app import git_lock, git_ops
from app.git_lock import GitWorkspaceBusyError, room_git_lock

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
    room_id = "lock-room"
    git_ops.git_init(room_id)  # commit on main, HEAD → dev
    return room_id, tmp_path / room_id


def _run_all(workers: list[threading.Thread]) -> None:
    for t in workers:
        t.start()
    for t in workers:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in workers), "a worker hung on the lock"


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_commits_and_branches_leave_a_consistent_repo(isolated_repo):
    room_id, root = isolated_repo
    errors: list[Exception] = []
    made_branches: list[str] = []

    def commit_worker(i: int) -> None:
        try:
            (root / f"file_{i}.txt").write_text(f"content {i}\n", encoding="utf-8")
            git_ops.git_commit(room_id, f"commit {i}")
        except Exception as exc:  # noqa: BLE001 - recorded and asserted below
            errors.append(exc)

    def branch_worker(i: int) -> None:
        try:
            name = f"task/b{i}"
            if git_ops.git_create_branch(room_id, name, base="dev"):
                made_branches.append(name)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    workers: list[threading.Thread] = []
    for i in range(20):
        target = commit_worker if i % 2 == 0 else branch_worker
        workers.append(threading.Thread(target=target, args=(i,)))
    _run_all(workers)

    assert not errors, f"workers raised: {errors}"

    # No half-finished state: no stray index/merge locks, fsck is clean.
    assert not (root / ".git" / "index.lock").exists()
    assert not (root / ".git" / "MERGE_HEAD").exists()
    _g(root, "fsck", "--full")  # raises on corruption

    # Every commit worker's file made it into history on some branch (branch
    # workers switch HEAD, and each `git add -A` sweeps the whole tree, so a
    # file may be committed on a task branch or absorbed by another worker's
    # commit — but none may be lost to an index-lock race).
    committed = _g(root, "log", "--all", "--name-only", "--pretty=format:")
    for i in range(0, 20, 2):
        assert f"file_{i}.txt" in committed, f"file_{i}.txt was lost"

    # Every branch that reported success actually exists.
    existing = {
        line.lstrip("* ").strip()
        for line in _g(root, "branch", "--list").splitlines()
    }
    for name in made_branches:
        assert name in existing


def test_two_concurrent_merges_both_complete(isolated_repo):
    room_id, root = isolated_repo

    # Two independent task branches, each with its own file.
    for name, rel in (("task/one", "one.txt"), ("task/two", "two.txt")):
        git_ops.git_create_branch(room_id, name, base="dev")
        (root / rel).write_text(f"{rel}\n", encoding="utf-8")
        git_ops.git_commit(room_id, f"work on {name}")
    git_ops.git_create_branch(room_id, "dev")  # HEAD back on dev

    results: dict[str, dict] = {}

    def merge_worker(name: str) -> None:
        results[name] = git_ops.git_merge_branch(
            room_id, name, target="dev", theirs=True
        )

    _run_all([
        threading.Thread(target=merge_worker, args=("task/one",)),
        threading.Thread(target=merge_worker, args=("task/two",)),
    ])

    assert results["task/one"]["ok"] and results["task/two"]["ok"]
    assert not results["task/one"].get("busy") and not results["task/two"].get("busy")
    assert not results["task/one"]["conflict"] and not results["task/two"]["conflict"]

    _g(root, "checkout", "dev")
    assert (root / "one.txt").exists() and (root / "two.txt").exists()
    assert not (root / ".git" / "index.lock").exists()


# ---------------------------------------------------------------------------
# Stale lock
# ---------------------------------------------------------------------------

def test_stale_lock_file_is_broken(isolated_repo, monkeypatch):
    room_id, root = isolated_repo
    lock = root / ".legion" / "git.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("999999 2000-01-01T00:00:00+00:00\n", encoding="utf-8")
    old = time.time() - (git_lock.STALE_TTL + 60)
    import os
    os.utime(lock, (old, old))

    (root / "after_stale.txt").write_text("x\n", encoding="utf-8")
    sha = git_ops.git_commit(room_id, "commit past a stale lock")

    assert sha, "commit should have proceeded after breaking the stale lock"
    assert not lock.exists(), "the lock file must be released after the op"


def test_fresh_lock_is_respected_until_timeout(isolated_repo, monkeypatch):
    room_id, root = isolated_repo
    monkeypatch.setattr(git_lock, "ACQUIRE_TIMEOUT", 0.5)

    released = threading.Event()

    def holder() -> None:
        with room_git_lock(room_id):
            released.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    time.sleep(0.1)  # let the holder take the lock

    try:
        result = git_ops.git_merge_branch(room_id, "dev", target="dev")
        assert result["busy"] is True and result["ok"] is False

        with pytest.raises(GitWorkspaceBusyError):
            git_ops.git_commit(room_id, "should not get the lock")
    finally:
        released.set()
        t.join(timeout=5)


# ---------------------------------------------------------------------------
# Re-entrancy
# ---------------------------------------------------------------------------

def test_lock_is_reentrant_within_a_thread(isolated_repo):
    room_id, _ = isolated_repo
    with room_git_lock(room_id):
        with room_git_lock(room_id, timeout=0.2):  # must not block or raise
            pass


def test_git_revert_takes_the_lock_and_still_commits(isolated_repo):
    """``git_revert`` holds the lock and calls ``git_commit`` (also locked) —
    the nested acquire must not deadlock."""
    room_id, root = isolated_repo
    (root / "doc.txt").write_text("v1\n", encoding="utf-8")
    git_ops.git_commit(room_id, "add doc")
    good = _g(root, "rev-parse", "HEAD")
    (root / "doc.txt").write_text("v2 broken\n", encoding="utf-8")
    git_ops.git_commit(room_id, "break doc")

    ok = git_ops.git_revert(room_id, good, path="doc.txt")

    assert ok
    assert (root / "doc.txt").read_text(encoding="utf-8") == "v1\n"


def test_lock_file_is_git_ignored(isolated_repo):
    room_id, root = isolated_repo
    with room_git_lock(room_id):
        status = _g(root, "status", "--porcelain")
        assert ".legion/git.lock" not in status
        # `git add -A` inside the lock must not stage it.
        _g(root, "add", "-A")
        staged = _g(root, "diff", "--cached", "--name-only")
        assert ".legion/git.lock" not in staged
