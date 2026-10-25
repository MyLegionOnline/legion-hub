"""git_ops.git_discard_branch — tag-backup + delete, including the branch the
workspace is currently sitting on.

Acceptance criteria:
  * a branch nobody is standing on is tagged under refs/legion/trash/ and gone;
  * the CHECKED-OUT branch is discarded too: the worktree steps off onto dev
    first (uncommitted work is parked into the backup), instead of the old
    "branch is checked out — cannot discard the active branch" dead end;
  * main / dev are still refused;
  * a missing branch is a no-op success (idempotent).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops

_CFG = ["-c", "core.autocrlf=false"]


def _g(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(
        ["git", *_CFG, "-C", str(root), *args],
        capture_output=True, text=True, check=check,
    )
    return r.stdout.strip()


@pytest.fixture
def room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    # chown is a no-op outside the real volume (and needs root) — skip it.
    monkeypatch.setattr(git_ops, "chown_workspace_tree", lambda *_a, **_kw: None)
    room_id = "discard-branch-room"
    git_ops.git_init(room_id)
    root = tmp_path / room_id
    _g(root, "config", "user.name", "Test")
    _g(root, "config", "user.email", "test@example.com")
    (root / "dev.txt").write_text("dev seed\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "dev: seed")
    return room_id, root


def _branches(root: Path) -> list[str]:
    return _g(root, "for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()


def test_discards_idle_branch_with_backup(room):
    room_id, root = room
    _g(root, "checkout", "-b", "task/10", "dev")
    (root / "t10.txt").write_text("work\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "task 10 work")
    tip = _g(root, "rev-parse", "task/10")
    _g(root, "checkout", "dev")

    res = git_ops.git_discard_branch(room_id, "task/10")

    assert res["ok"] is True and res["deleted"] is True
    assert "task/10" not in _branches(root)
    # the tip survives under the backup ref
    assert _g(root, "rev-parse", res["backup_ref"]) == tip


def test_discards_the_checked_out_branch(room):
    """The common case: the workspace is still standing on the branch the user
    wants gone. It must be discarded, not refused."""
    room_id, root = room
    _g(root, "checkout", "-b", "goal/abc", "dev")
    (root / "goal.txt").write_text("goal work\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "goal work")
    # …and leave an uncommitted edit on top: it gets parked, not dropped.
    (root / "goal.txt").write_text("goal work, uncommitted\n")

    res = git_ops.git_discard_branch(room_id, "goal/abc")

    assert res["ok"] is True and res["deleted"] is True
    assert "goal/abc" not in _branches(root)
    assert git_ops.git_current_branch(room_id) == "dev"
    # the parked edit is reachable from the backup ref
    blob = _g(root, "show", f"{res['backup_ref']}:goal.txt")
    assert blob == "goal work, uncommitted"
    # dev itself was not touched by the parking commit
    assert not (root / "goal.txt").exists()


def test_refuses_protected_branches(room):
    room_id, _root = room
    for name in ("main", "dev"):
        res = git_ops.git_discard_branch(room_id, name)
        assert res["ok"] is False
        assert "refusing" in res["stderr"]


def test_missing_branch_is_a_noop(room):
    room_id, _root = room
    res = git_ops.git_discard_branch(room_id, "task/nope")
    assert res["ok"] is True
    assert res["deleted"] is False
