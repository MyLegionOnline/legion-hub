"""Explicit rename of a room's integration branch.

internal design notes Stage 3: ``rename_working_branch``
moves the branch with ``git branch -m`` (work and a HEAD on it follow) and
records the new name; ``remote_branch_exists`` asks the remote with
``ls-remote`` before the Hub renames onto a name the remote already has.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops

_CFG = ["-c", "core.autocrlf=false", "-c", "safe.directory=*"]


def _g(root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *_CFG, "-C", str(root), *args], capture_output=True, text=True, check=True,
    )
    return r.stdout.strip()


def _branches(root: Path) -> set[str]:
    out = _g(root, "branch", "--format=%(refname:short)")
    return {line.strip() for line in out.splitlines() if line.strip()}


def _commit(root: Path, rel: str, content: str, message: str) -> None:
    (root / rel).write_text(content, encoding="utf-8")
    git_ops._run(["add", "-A"], cwd=root)
    git_ops._run(["commit", "-m", message], cwd=root)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def dev_room(data_dir: Path):
    room_id = "rename-room"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    _commit(root, "a.txt", "work\n", "feat: work on dev")
    return room_id, root


def test_rename_carries_work_and_head(dev_room):
    room_id, root = dev_room
    tip = _g(root, "rev-parse", "dev")

    result = git_ops.rename_working_branch(room_id, "dev", "core")

    assert result["ok"] and result["renamed"] and not result["created"]
    assert _branches(root) == {"main", "core"}
    assert _g(root, "rev-parse", "core") == tip
    assert git_ops.git_current_branch(room_id) == "core"
    assert result["head"] == "core"
    assert git_ops.working_branch_of(root) == "core"


def test_rename_from_another_head_leaves_head_alone(dev_room):
    room_id, root = dev_room
    _g(root, "checkout", "-b", "task/1")

    result = git_ops.rename_working_branch(room_id, "dev", "core")

    assert result["ok"] and result["renamed"]
    assert git_ops.git_current_branch(room_id) == "task/1"
    assert "core" in _branches(root) and "dev" not in _branches(root)


def test_rename_drops_the_old_upstream(dev_room):
    room_id, root = dev_room
    _g(root, "config", "branch.dev.remote", "https://example.invalid/r.git")
    _g(root, "config", "branch.dev.merge", "refs/heads/dev")

    assert git_ops.rename_working_branch(room_id, "dev", "core")["ok"]

    r = subprocess.run(
        ["git", *_CFG, "-C", str(root), "config", "--get", "branch.core.merge"],
        capture_output=True, text=True,
    )
    assert r.stdout.strip() == ""


def test_rename_back_restores_the_old_name(dev_room):
    room_id, root = dev_room
    tip = _g(root, "rev-parse", "dev")
    assert git_ops.rename_working_branch(room_id, "dev", "core")["ok"]

    result = git_ops.rename_working_branch(room_id, "core", "dev")

    assert result["ok"] and result["renamed"]
    assert _branches(root) == {"main", "dev"}
    assert _g(root, "rev-parse", "dev") == tip
    assert git_ops.working_branch_of(root) == "dev"


@pytest.mark.parametrize("name", ["main", "task/1", "\u044f\u0434\u0440\u043e", "a..b", ""])
def test_rename_rejects_an_invalid_name(dev_room, name):
    room_id, root = dev_room

    result = git_ops.rename_working_branch(room_id, "dev", name)

    assert not result["ok"] and result["code"] == "invalid_name"
    assert _branches(root) == {"main", "dev"}


def test_rename_refuses_an_existing_branch(dev_room):
    room_id, root = dev_room
    _g(root, "branch", "core", "main")

    result = git_ops.rename_working_branch(room_id, "dev", "core")

    assert not result["ok"] and result["code"] == "branch_exists"
    assert {"dev", "core"} <= _branches(root)
    assert git_ops.working_branch_of(root) == "dev"


def test_rename_without_the_old_branch_starts_from_main(data_dir):
    room_id = "no-old"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    _g(root, "checkout", "main")
    _g(root, "branch", "-D", "dev")

    result = git_ops.rename_working_branch(room_id, "dev", "core")

    assert result["ok"] and result["created"] and not result["renamed"]
    assert _g(root, "rev-parse", "core") == _g(root, "rev-parse", "main")
    assert git_ops.working_branch_of(root) == "core"


def test_rename_refuses_when_the_repo_works_on_another_branch(data_dir):
    room_id = "mismatch"
    git_ops.git_init(room_id, "transport")
    root = data_dir / room_id

    # The Hub still thinks the room is on `dev` (e.g. a second, stale rename).
    result = git_ops.rename_working_branch(room_id, "dev", "core")

    assert not result["ok"] and result["code"] == "working_branch_mismatch"
    assert _branches(root) == {"main", "transport"}


def test_rename_in_a_room_without_a_repo_creates_one(data_dir):
    result = git_ops.rename_working_branch("fresh", "dev", "core")

    root = data_dir / "fresh"
    assert result["ok"] and result["created"]
    assert _branches(root) == {"main", "core"}
    assert git_ops.git_current_branch("fresh") == "core"
    assert git_ops.working_branch_of(root) == "core"


def test_same_name_only_records_it(dev_room):
    room_id, root = dev_room

    result = git_ops.rename_working_branch(room_id, "dev", "dev")

    assert result["ok"] and not result["renamed"] and not result["created"]
    assert _branches(root) == {"main", "dev"}


def test_ensure_keeps_the_renamed_branch(dev_room):
    room_id, root = dev_room
    assert git_ops.rename_working_branch(room_id, "dev", "core")["ok"]

    # The startup backfill after a rename must not bring `dev` back.
    result = git_ops.ensure_base_branches(room_id, "core")

    assert result["ok"] and not result["created_dev"]
    assert _branches(root) == {"main", "core"}


def test_remote_branch_exists(dev_room, tmp_path):
    room_id, root = dev_room
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _g(root, "push", str(remote), "refs/heads/dev:refs/heads/core")
    url = f"file://{remote}"

    assert git_ops.remote_branch_exists(room_id, url, "core") == {
        "ok": True, "exists": True, "error": None,
    }
    assert git_ops.remote_branch_exists(room_id, url, "transport")["exists"] is False
    # `core` must not match a longer name that merely ends with it.
    _g(root, "push", str(remote), "refs/heads/dev:refs/heads/team/frontend")
    assert git_ops.remote_branch_exists(room_id, url, "frontend")["exists"] is False


def test_remote_branch_exists_reports_an_unreachable_remote(dev_room, tmp_path):
    room_id, _ = dev_room

    result = git_ops.remote_branch_exists(room_id, f"file://{tmp_path / 'missing.git'}", "core")

    assert result["ok"] is False and result["exists"] is False and result["error"]
