"""
Stage 2 of internal design notes — workspaced side.

``git_push`` now takes a ``mode`` argument (see
internal design notes stage 2.2):

  * ``mode="push"`` (default) — a plain ``git push`` that lets the remote
    reject a non-fast-forward instead of clobbering someone else's history;
  * ``mode="force"`` — ``--force-with-lease`` for an explicit, acknowledged
    override of divergence.

Both modes still retry with ``-u`` when the remote branch doesn't exist yet
(first push to an empty remote).

The push refspec is ``refs/heads/{branch}:refs/heads/{branch}`` since
internal design notes Stage 6 — the *local* branch of that name is pushed
regardless of which branch HEAD is on. ``git_init`` leaves HEAD on ``dev``,
so these tests exercise ``branch="dev"``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "stage2-room"
    git_ops.git_init(room_id)
    return room_id, tmp_path / room_id


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bare_remote(tmp_path: Path, name: str = "remote.git") -> Path:
    remote = tmp_path / name
    subprocess.run(
        ["git", "init", "--bare", str(remote)],
        capture_output=True,
        text=True,
        check=True,
    )
    return remote


def test_push_default_mode_seeds_empty_remote(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    _write(root, "a.txt", "hello")
    git_ops.git_commit(room_id, "add a")

    remote = _bare_remote(tmp_path)
    result = git_ops.git_push(
        room_id, f"file://{remote}", branch="dev", token=None, mode="push"
    )
    assert result["ok"] is True, result["stderr"]


def test_push_default_mode_rejects_non_fast_forward(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    _write(root, "a.txt", "hello")
    git_ops.git_commit(room_id, "add a")
    remote = _bare_remote(tmp_path)

    # Seed the remote from this workspace.
    seeded = git_ops.git_push(
        room_id, f"file://{remote}", branch="dev", token=None, mode="push"
    )
    assert seeded["ok"]

    # Diverge both sides: remote gets an independent commit, workspace gets
    # its own. A plain push from the workspace must be rejected.
    _diverge_remote(tmp_path, remote)
    _write(root, "a.txt", "workspace-side change")
    git_ops.git_commit(room_id, "workspace change")

    rejected = git_ops.git_push(
        room_id, f"file://{remote}", branch="dev", token=None, mode="push"
    )
    assert rejected["ok"] is False
    assert "rejected" in rejected["stderr"].lower() or "non-fast-forward" in rejected["stderr"].lower()


def test_push_force_mode_overrides_divergence(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    _write(root, "a.txt", "hello")
    git_ops.git_commit(room_id, "add a")
    remote = _bare_remote(tmp_path)

    seeded = git_ops.git_push(
        room_id, f"file://{remote}", branch="dev", token=None, mode="push"
    )
    assert seeded["ok"]

    _diverge_remote(tmp_path, remote, conflict_file="a.txt")

    # Diverge the workspace side BEFORE fetching, so the merge is a genuine
    # non-fast-forward (both sides moved). Real flow: the user fetches (sees
    # GIT_CONFLICT), then chooses force. The fetch refreshes the
    # remote-tracking ref so --force-with-lease can verify the lease against
    # the actual remote tip.
    _write(root, "a.txt", "workspace-side change")
    git_ops.git_commit(room_id, "workspace change")

    fetched = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="dev", token=None
    )
    # Both sides edited a.txt differently → a real content conflict that
    # git_fetch cannot resolve on its own; the tree is rolled back clean.
    assert fetched["ok"] is False
    assert fetched["conflict"] is True

    forced = git_ops.git_push(
        room_id, f"file://{remote}", branch="dev", token=None, mode="force"
    )
    assert forced["ok"] is True, forced["stderr"]


def test_fetch_merges_diverged_history_without_conflict(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    _write(root, "a.txt", "hello")
    git_ops.git_commit(room_id, "add a")
    remote = _bare_remote(tmp_path)

    assert git_ops.git_push(
        room_id, f"file://{remote}", branch="dev", token=None, mode="push"
    )["ok"]

    # Both sides move, but on different files → mergeable divergence.
    _diverge_remote(tmp_path, remote)
    _write(root, "b.txt", "workspace only")
    git_ops.git_commit(room_id, "workspace change")

    fetched = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="dev", token=None
    )
    assert fetched["ok"] is True, fetched["stderr"]
    assert fetched["conflict"] is False
    # Remote commit is now present in the workspace, local work is preserved.
    assert (root / "remote-side.txt").read_text(encoding="utf-8") == "from remote"
    assert (root / "b.txt").read_text(encoding="utf-8") == "workspace only"


def test_fetch_pulls_into_freshly_seeded_room(isolated_repo, tmp_path):
    """A brand-new room repo (only the empty init commit) pointed at an
    existing remote must still pull the remote's history in."""
    room_id, root = isolated_repo
    remote = _bare_remote(tmp_path)

    # Seed the remote independently — unrelated history to the room's init commit.
    clone = tmp_path / "seed-clone"
    subprocess.run(["git", "clone", str(remote), str(clone)],
                   capture_output=True, text=True, check=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.name", "Remote Author"],
                   capture_output=True, text=True, check=True)
    subprocess.run(["git", "-C", str(clone), "config", "user.email", "remote@example.com"],
                   capture_output=True, text=True, check=True)
    (clone / "README.md").write_text("# blog", encoding="utf-8")
    subprocess.run(["git", "-C", str(clone), "add", "-A"],
                   capture_output=True, text=True, check=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-m", "initial"],
                   capture_output=True, text=True, check=True)
    subprocess.run(["git", "-C", str(clone), "push", "origin", "HEAD:main"],
                   capture_output=True, text=True, check=True)

    fetched = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None
    )
    assert fetched["ok"] is True, fetched["stderr"]
    assert fetched["conflict"] is False
    assert (root / "README.md").read_text(encoding="utf-8") == "# blog"


def _diverge_remote(tmp_path: Path, remote: Path, conflict_file: str | None = None) -> None:
    """Make an independent commit on the bare remote so the workspace's
    next push is non-fast-forward.

    When ``conflict_file`` is given, that path is also rewritten on the
    remote so a workspace edit to the same file produces a genuine merge
    conflict (not just a divergent-but-mergeable history).
    """
    clone = tmp_path / "remote-clone"
    subprocess.run(
        ["git", "clone", str(remote), str(clone)],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "config", "user.name", "Remote Author"],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "config", "user.email", "remote@example.com"],
        capture_output=True, text=True, check=True,
    )
    # The bare remote's HEAD may point at a different default branch name
    # (system init.defaultBranch), leaving the clone with no checked-out
    # branch — pin it to the branch we actually seeded.
    subprocess.run(
        ["git", "-C", str(clone), "checkout", "-B", "dev", "origin/dev"],
        capture_output=True, text=True, check=True,
    )
    (clone / "remote-side.txt").write_text("from remote", encoding="utf-8")
    if conflict_file is not None:
        (clone / conflict_file).write_text("remote-side change", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(clone), "add", "-A"],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "commit", "-m", "remote-side change"],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        ["git", "-C", str(clone), "push", "origin", "dev"],
        capture_output=True, text=True, check=True,
    )
