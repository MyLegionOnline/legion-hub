"""Branch pinning for ``git_fetch`` (internal design notes §0.1 follow-up).

Before this, "Pull from Git" merged the remote into whatever branch happened
to be checked out. A sync fired while a task session held ``task/<id>``
checked out landed the entire remote repository on that task branch, and
``dev`` never saw it. ``target_branch`` pins the landing branch.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "pin-room"
    git_ops.git_init(room_id)
    return room_id, tmp_path / room_id


def _write(root: Path, rel: str, content: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seeded_remote(tmp_path: Path, branch: str = "main") -> Path:
    """A bare remote carrying one commit on ``branch`` (unrelated history)."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)],
                   capture_output=True, text=True, check=True)
    clone = tmp_path / "seed-clone"
    subprocess.run(["git", "clone", str(remote), str(clone)],
                   capture_output=True, text=True, check=True)
    for key, value in (("user.name", "Remote Author"), ("user.email", "remote@example.com")):
        subprocess.run(["git", "-C", str(clone), "config", key, value],
                       capture_output=True, text=True, check=True)
    (clone / "README.md").write_text("# from remote", encoding="utf-8")
    subprocess.run(["git", "-C", str(clone), "add", "-A"],
                   capture_output=True, text=True, check=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-m", "initial"],
                   capture_output=True, text=True, check=True)
    subprocess.run(["git", "-C", str(clone), "push", "origin", f"HEAD:{branch}"],
                   capture_output=True, text=True, check=True)
    return remote


def _file_on(root: Path, branch: str, rel: str) -> str | None:
    r = subprocess.run(["git", "-C", str(root), "show", f"{branch}:{rel}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def test_fetch_lands_on_target_branch_not_on_checked_out_task_branch(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)

    # A task session left HEAD on its own branch, as task dispatch does.
    git_ops.git_create_branch(room_id, "task/abc", base="dev")
    assert git_ops.git_current_branch(room_id) == "task/abc"

    fetched = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )

    assert fetched["ok"] is True, fetched["stderr"]
    assert fetched["branch"] == "dev"
    assert git_ops.git_current_branch(room_id) == "dev"
    assert _file_on(root, "dev", "README.md") == "# from remote"
    # The task branch is untouched — the remote did not leak into it.
    assert _file_on(root, "task/abc", "README.md") is None


def test_fetch_without_target_keeps_historic_behaviour(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)
    git_ops.git_create_branch(room_id, "task/abc", base="dev")

    fetched = git_ops.git_fetch(room_id, f"file://{remote}", branch="main", token=None)

    assert fetched["ok"] is True, fetched["stderr"]
    assert git_ops.git_current_branch(room_id) == "task/abc"
    assert _file_on(root, "task/abc", "README.md") == "# from remote"


def test_fetch_creates_missing_target_branch(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)

    fetched = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None, target_branch="integration"
    )

    assert fetched["ok"] is True, fetched["stderr"]
    assert git_ops.git_current_branch(room_id) == "integration"
    assert _file_on(root, "integration", "README.md") == "# from remote"


def test_fetch_parks_uncommitted_work_before_switching(isolated_repo, tmp_path):
    """Uncommitted edits made on a task branch must survive the switch to the
    target branch — the "never destroy content" contract."""
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)
    git_ops.git_create_branch(room_id, "task/abc", base="dev")
    _write(root, "wip.txt", "unsaved work")

    fetched = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )

    assert fetched["ok"] is True, fetched["stderr"]
    # Parked as a commit on the branch it was made on, not carried onto `dev`.
    assert _file_on(root, "task/abc", "wip.txt") == "unsaved work"


def test_fetch_never_commits_a_dirty_tree_onto_main(isolated_repo, tmp_path):
    """§0.1: nothing but a promotion moves ``main``. A dirty tree sitting on
    ``main`` is parked on ``dev`` instead of being committed where it is."""
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path)
    subprocess.run(["git", "-C", str(root), "checkout", "main"],
                   capture_output=True, text=True, check=True)
    _write(root, "wip.txt", "unsaved work")

    before = subprocess.run(["git", "-C", str(root), "rev-parse", "main"],
                            capture_output=True, text=True, check=True).stdout.strip()

    fetched = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )

    after = subprocess.run(["git", "-C", str(root), "rev-parse", "main"],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert fetched["ok"] is True, fetched["stderr"]
    assert after == before, "`main` must not move on a fetch"
    assert _file_on(root, "dev", "wip.txt") == "unsaved work"
