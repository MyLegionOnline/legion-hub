"""With ``GIT_RUN_AS_WORKSPACE_USER`` on, workspaced's git runs as the room
user, not root (app/git_exec.py) — so a planted link inside ``.git`` can only
reach files the agent could write anyway.

Needs root on POSIX (that is how workspaced runs); skipped elsewhere.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from app import git_exec, git_ops
from app.config import WORKSPACE_GID, WORKSPACE_UID

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="needs root on POSIX, like workspaced itself",
)


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch):
    # pytest's tmp_path sits under 0700 root dirs the workspace user cannot
    # enter; workspaced's data dir is world-traversable.
    base = Path(tempfile.mkdtemp(prefix="ws-as-user-", dir="/var/tmp"))
    base.chmod(0o755)
    home = base / "home"
    monkeypatch.setenv("GIT_RUN_AS_WORKSPACE_USER", "1")
    monkeypatch.setenv("GIT_USER_HOME", str(home))
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", base / "data")
    (base / "data").mkdir()
    (base / "data").chmod(0o755)
    git_exec._owned_roots.clear()
    yield base / "data"
    git_exec._owned_roots.clear()
    shutil.rmtree(base, ignore_errors=True)


def _owner(path: Path) -> int:
    return os.lstat(path).st_uid


def test_git_runs_as_the_workspace_user(data_dir):
    room_id = "as-user"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "work")

    assert _owner(root / ".git" / "index") == WORKSPACE_UID
    r = git_exec.git_exec(["rev-parse", "HEAD"], root)
    assert r.returncode == 0
    obj = r.stdout.strip()
    assert _owner(root / ".git" / "objects" / obj[:2] / obj[2:]) == WORKSPACE_UID


def test_root_owned_leftovers_are_handed_over(data_dir):
    room_id = "leftovers"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    os.chown(root / ".git" / "index", 0, 0)
    os.chmod(root / ".git" / "index", 0o644)
    git_exec._owned_roots.clear()  # a fresh workspaced process

    (root / "b.txt").write_text("b\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "work after the switch")
    assert _owner(root / ".git" / "index") == WORKSPACE_UID


def test_a_root_file_written_later_is_fixed_by_the_retry(data_dir):
    room_id = "retry"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    (root / "c.txt").write_text("c\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "first")
    # The tree was already handed over in this process; a root-made dir
    # appears afterwards (workspaced's Python writes run as root).
    objects = root / ".git" / "objects"
    sub = objects / "ff"
    sub.mkdir(exist_ok=True)
    os.chown(sub, 0, 0)
    sub.chmod(0o755)
    for d in objects.iterdir():
        if d.is_dir() and len(d.name) == 2:
            os.chown(d, 0, 0)
            d.chmod(0o755)

    (root / "d.txt").write_text("d\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "second")


def test_a_symlink_raced_into_dotgit_cannot_reach_root_files(data_dir, tmp_path):
    """Even past the structure check, git as the user cannot write root's file."""
    room_id = "race"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    victim = data_dir.parent / "victim.py"
    victim.write_text("ORIGINAL = True\n", encoding="utf-8")
    victim.chmod(0o644)  # root-owned, like workspaced's code

    # Simulate the race the check cannot close: bypass it and let git write
    # through the link.
    msg = root / ".git" / "COMMIT_EDITMSG"
    if msg.exists():
        msg.unlink()
    os.symlink(victim, msg)
    uid_env = git_exec.base_env() | git_exec._user_env()
    r = subprocess.run(
        ["git", "-c", "safe.directory=*", "commit", "--allow-empty", "-m", "import os"],
        cwd=root, env=uid_env | {"GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@x",
                                 "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@x"},
        user=WORKSPACE_UID, group=WORKSPACE_GID, extra_groups=[],
        capture_output=True, text=True,
    )
    assert victim.read_text(encoding="utf-8") == "ORIGINAL = True\n", r.stderr


def test_pull_rework_worktree_works_as_the_user(data_dir, tmp_path):
    """The one temporary directory workspaced hands to git is given to the user."""
    room_id = "rework"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    (root / "x.txt").write_text("x\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "x")
    head = git_exec.git_exec(["rev-parse", "HEAD"], root).stdout.strip()

    with tempfile.TemporaryDirectory(prefix="legion-pull-rework-") as temp:
        git_exec.hand_to_git_user(temp)
        checkout = Path(temp) / "merge"
        r = git_exec.git_exec(["worktree", "add", "--detach", str(checkout), head], root)
        assert r.returncode == 0, r.stderr
        assert (checkout / "x.txt").exists()
        assert git_exec.git_exec(["status", "--porcelain"], checkout).returncode == 0
        git_exec.git_exec(["worktree", "remove", "--force", str(checkout)], root)


def test_flag_off_keeps_root(monkeypatch):
    monkeypatch.delenv("GIT_RUN_AS_WORKSPACE_USER", raising=False)
    assert git_exec.run_as_workspace_user() is False
