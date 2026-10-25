"""internal design notes Stage 1 — git_list_branches and git_ahead_behind_remote.

Acceptance criteria:
  * room with main, dev, merged task/44 and live task/48 → task/44 merged=True,
    task/48 merged=False and ahead>0;
  * branch whose merge failed by conflict comes with merged=False;
  * call in a room without a git repo → empty list, no exception;
  * git_ahead_behind_remote returns None for room without remote tracking ref.
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
    """An initialised git room with main and dev branches.

    git_init already creates both main (empty commit) and dev (branched off
    main), with HEAD on dev. We add a seed commit on dev so task branches
    branched off dev have a non-trivial ancestor graph.
    """
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "list-branches-room"
    git_ops.git_init(room_id)
    root = tmp_path / room_id

    # Override identity for reproducible commits
    _g(root, "config", "user.name", "Test")
    _g(root, "config", "user.email", "test@example.com")

    # HEAD is on dev after git_init — add a seed commit there
    (root / "dev.txt").write_text("dev seed\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "dev: seed")

    return room_id, root


# ── merged branch ────────────────────────────────────────────────────────────

def test_merged_branch_reported(room):
    room_id, root = room

    # task/44: branch off dev, commit, then merge into dev → merged
    _g(root, "checkout", "-b", "task/44", "dev")
    (root / "task44.txt").write_text("task 44\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "task 44 work")

    _g(root, "checkout", "dev")
    _g(root, "merge", "--no-ff", "task/44", "-m", "Merge task/44")

    branches = git_ops.git_list_branches(room_id, {"task/44": "dev"})
    by_name = {b["name"]: b for b in branches}

    assert "task/44" in by_name
    assert by_name["task/44"]["merged"] is True
    assert by_name["task/44"]["ahead"] == 0


# ── live (not merged) branch ─────────────────────────────────────────────────

def test_live_branch_reported(room):
    room_id, root = room

    # task/48: branch off dev, add commit, do NOT merge
    _g(root, "checkout", "-b", "task/48", "dev")
    (root / "task48.txt").write_text("task 48\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "task 48 work")

    branches = git_ops.git_list_branches(room_id, {"task/48": "dev"})
    by_name = {b["name"]: b for b in branches}

    assert "task/48" in by_name
    t = by_name["task/48"]
    assert t["merged"] is False
    assert t["ahead"] is not None and t["ahead"] > 0


# ── both merged and live ──────────────────────────────────────────────────────

def test_merged_and_live_together(room):
    room_id, root = room

    # task/44 merged into dev
    _g(root, "checkout", "-b", "task/44", "dev")
    (root / "t44.txt").write_text("44\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "44")
    _g(root, "checkout", "dev")
    _g(root, "merge", "--no-ff", "task/44", "-m", "Merge task/44")

    # task/48 not merged
    _g(root, "checkout", "-b", "task/48", "dev")
    (root / "t48.txt").write_text("48\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "48")

    bases = {"task/44": "dev", "task/48": "dev"}
    branches = git_ops.git_list_branches(room_id, bases)
    by_name = {b["name"]: b for b in branches}

    assert by_name["task/44"]["merged"] is True
    assert by_name["task/48"]["merged"] is False
    assert by_name["task/48"]["ahead"] > 0


# ── conflicted (not merged) branch ───────────────────────────────────────────

def test_conflict_branch_not_merged(room):
    """A branch whose merge was attempted but aborted comes as merged=False."""
    room_id, root = room

    # Create conflicting content: both dev and task/conflict touch shared.txt
    _g(root, "checkout", "dev")
    (root / "shared.txt").write_text("dev version\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "dev: add shared")

    _g(root, "checkout", "-b", "task/conflict", "dev")
    # Move dev forward with different content
    _g(root, "checkout", "dev")
    (root / "shared.txt").write_text("dev updated\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "dev: update shared")

    # Write conflicting content on task/conflict
    _g(root, "checkout", "task/conflict")
    (root / "shared.txt").write_text("task conflict version\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "task: conflict version")

    # No merge happened → task/conflict has ahead > 0 → merged=False
    branches = git_ops.git_list_branches(room_id, {"task/conflict": "dev"})
    by_name = {b["name"]: b for b in branches}

    assert "task/conflict" in by_name
    assert by_name["task/conflict"]["merged"] is False


# ── room without git repo → empty list ───────────────────────────────────────

def test_no_repo_returns_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    result = git_ops.git_list_branches("no-git-room", {"main": "origin/main"})
    assert result == []


# ── unknown branch (no base) → merged=None ───────────────────────────────────

def test_unknown_base_yields_none(room):
    room_id, root = room

    _g(root, "checkout", "-b", "other/branch", "dev")
    (root / "other.txt").write_text("x\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "other")

    # bases dict does not include other/branch
    branches = git_ops.git_list_branches(room_id, {})
    by_name = {b["name"]: b for b in branches}

    assert "other/branch" in by_name
    t = by_name["other/branch"]
    assert t["merged"] is None
    assert t["ahead"] is None
    assert t["behind"] is None


# ── stray refs/heads/FETCH_HEAD is filtered out ──────────────────────────────

def test_pseudo_ref_branch_filtered(room):
    """A ``refs/heads/FETCH_HEAD`` left by a bad refspec never appears."""
    room_id, root = room

    _g(root, "branch", "FETCH_HEAD", "dev")          # the accident
    _g(root, "checkout", "-b", "task/50", "dev")
    (root / "t50.txt").write_text("50\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "50")

    branches = git_ops.git_list_branches(room_id, {"task/50": "dev"})
    names = {b["name"] for b in branches}

    assert not any(n.rsplit("/", 1)[-1] == "FETCH_HEAD" for n in names)
    assert "task/50" in names


def test_prune_pseudo_ref_branches_deletes_stray(room):
    room_id, root = room
    _g(root, "branch", "FETCH_HEAD", "dev")
    assert "FETCH_HEAD" in _g(root, "branch", "--list", "FETCH_HEAD")

    git_ops._prune_pseudo_ref_branches(root)

    assert _g(root, "branch", "--list", "FETCH_HEAD") == ""


# ── git_ahead_behind_remote: no remote → None ────────────────────────────────

def test_ahead_behind_remote_no_remote(room):
    room_id, _ = room
    result = git_ops.git_ahead_behind_remote(room_id, "dev")
    assert result is None


# ── git_ahead_behind_remote: no repo → None ──────────────────────────────────

def test_ahead_behind_remote_no_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    result = git_ops.git_ahead_behind_remote("no-repo", "main")
    assert result is None
