"""internal design notes Stage 1-2 — git_divergence_report / git_discard_branch.

Acceptance:
  * diverged branches with an overlapping edit → conflict=True, the file listed
    in conflict_files and overlap_files, per-side commits populated;
  * diverged branches touching disjoint files → conflict=False but still
    ahead>0 and behind>0;
  * git_discard_branch tags the branch under refs/legion/trash/ then deletes it;
  * git_discard_branch refuses main / dev and the checked-out branch;
  * report on a room with no repo → ok=False, no exception.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops

_CFG = ["-c", "core.autocrlf=false"]


def _g(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *_CFG, "-C", str(root), *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "divergence-room"
    git_ops.git_init(room_id)
    root = tmp_path / room_id
    _g(root, "config", "user.name", "Test")
    _g(root, "config", "user.email", "test@example.com")
    (root / "f.txt").write_text("l1\nl2\nl3\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "dev: seed")
    return room_id, root


def _diverge(root: Path, *, overlap: bool):
    # task branch edits f.txt line 2
    _g(root, "checkout", "-b", "task/1", "dev")
    (root / "f.txt").write_text("l1\nTASK\nl3\n")
    (root / "task-only.txt").write_text("t\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "task: work")
    # dev moves forward
    _g(root, "checkout", "dev")
    if overlap:
        (root / "f.txt").write_text("l1\nDEV\nl3\n")
    else:
        (root / "dev-only.txt").write_text("d\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "dev: later work")


def test_report_overlapping_edit_is_conflict(room):
    room_id, root = room
    _diverge(root, overlap=True)

    rep = git_ops.git_divergence_report(room_id, "task/1", "dev")

    assert rep["ok"] is True
    assert rep["conflict"] is True
    assert "f.txt" in rep["conflict_files"]
    assert "f.txt" in rep["overlap_files"]
    assert rep["ahead"] == 1 and rep["behind"] == 1
    assert len(rep["branch_commits"]) == 1
    assert len(rep["base_commits"]) == 1
    assert any(d["path"] == "f.txt" and d["branch_diff"] for d in rep["conflict_details"])


def test_report_disjoint_edits_no_conflict(room):
    room_id, root = room
    _diverge(root, overlap=False)

    rep = git_ops.git_divergence_report(room_id, "task/1", "dev")

    assert rep["ok"] is True
    assert rep["conflict"] is False
    assert rep["overlap_files"] == []
    assert rep["ahead"] == 1 and rep["behind"] == 1
    assert rep["redundant"] is False


def test_report_redundant_when_branch_work_already_in_base(room):
    room_id, root = room
    # task branch adds a file; then the same content lands on dev independently
    _g(root, "checkout", "-b", "task/1", "dev")
    (root / "feat.txt").write_text("hello\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "task: add feat")
    _g(root, "checkout", "dev")
    (root / "feat.txt").write_text("hello\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "dev: same feat via rework")

    rep = git_ops.git_divergence_report(room_id, "task/1", "dev")

    assert rep["ok"] is True
    assert rep["redundant"] is True  # base...branch diff is empty


def test_report_no_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    rep = git_ops.git_divergence_report("ghost", "task/1", "dev")
    assert rep["ok"] is False and rep["error"]


def test_discard_branch_tags_then_deletes(room):
    room_id, root = room
    _diverge(root, overlap=True)
    _g(root, "checkout", "dev")  # not on task/1

    res = git_ops.git_discard_branch(room_id, "task/1")

    assert res["ok"] is True and res["deleted"] is True
    assert res["backup_ref"].startswith("refs/legion/trash/task-1@")
    assert not git_ops.git_branch_exists(room_id, "task/1")
    # backup ref still resolves
    assert _g(root, "rev-parse", "--verify", res["backup_ref"])


def test_discard_branch_refuses_protected_but_leaves_current(room, monkeypatch):
    """``main``/``dev`` stay protected, but the branch that happens to be
    checked out is discarded — the worktree steps off onto ``dev`` first
    (see tests/test_git_discard_branch.py)."""
    room_id, root = room
    monkeypatch.setattr(git_ops, "chown_workspace_tree", lambda *_a, **_kw: None)
    _diverge(root, overlap=True)  # leaves HEAD on dev after last checkout

    assert git_ops.git_discard_branch(room_id, "dev")["ok"] is False
    assert git_ops.git_discard_branch(room_id, "main")["ok"] is False

    _g(root, "checkout", "task/1")
    res = git_ops.git_discard_branch(room_id, "task/1")
    assert res["ok"] is True and res["deleted"] is True
    assert not git_ops.git_branch_exists(room_id, "task/1")
    assert git_ops.git_current_branch(room_id) == "dev"


def test_discard_missing_branch_is_idempotent(room):
    room_id, _ = room
    res = git_ops.git_discard_branch(room_id, "task/nope")
    assert res["ok"] is True and res["deleted"] is False
