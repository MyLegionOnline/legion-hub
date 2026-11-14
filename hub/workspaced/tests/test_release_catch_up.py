"""Several rooms publishing into one remote ``main``.

internal design notes Stage 5: ``git_catch_up_release``
fast-forwards a room's ``main`` to the remote's (never merges) so a publication
after another room's does not end in a non-fast-forward push;
``release_remote_status`` reports what the last fetched remote ``main`` has.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from app import git_ops

_CFG = ["-c", "core.autocrlf=false", "-c", "safe.directory=*"]
# Setup git stands in for workspaced itself: the room's ref guard lets it move `main`.
_TRUSTED = {**os.environ, "LEGION_GIT_TRUSTED": "1"}


def _g(root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *_CFG, "-C", str(root), *args],
        capture_output=True, text=True, check=True, env=_TRUSTED,
    )
    return r.stdout.strip()


def _commit(root: Path, rel: str, content: str, message: str) -> None:
    (root / rel).write_text(content, encoding="utf-8")
    git_ops._run(["add", "-A"], cwd=root)
    git_ops._run(["commit", "-m", message], cwd=root)


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path / "rooms")
    (tmp_path / "rooms").mkdir()
    return tmp_path / "rooms"


@pytest.fixture
def remote(tmp_path: Path) -> str:
    """A bare repo whose ``main`` has one commit, as GitHub would."""
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)], check=True, capture_output=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "init", "-b", "main", str(seed)], check=True, capture_output=True)
    _g(seed, "config", "user.name", "Seed")
    _g(seed, "config", "user.email", "seed@example.invalid")
    (seed / "README.md").write_text("shared\n", encoding="utf-8")
    _g(seed, "add", "-A")
    _g(seed, "commit", "-m", "init")
    _g(seed, "push", str(bare), "main")
    return f"file://{bare}"


def _room(data_dir: Path, remote: str, room_id: str, working: str) -> Path:
    result = git_ops.git_adopt_remote(room_id, remote, "main", working_branch=working)
    assert result["ok"] and result["adopted"], result
    return data_dir / room_id


def _publish(room_id: str, root: Path, working: str, remote: str) -> dict:
    """What an approved local promotion + publish does: merge, then push ``main``."""
    merged = git_ops.git_merge_branch(room_id, working, target="main", theirs=False)
    assert merged["ok"], merged
    _g(root, "checkout", working)
    return git_ops.git_push(room_id, remote, branch="main")


def test_second_room_publishes_after_catching_up(data_dir, remote):
    core_root = _room(data_dir, remote, "room-core", "core")
    tr_root = _room(data_dir, remote, "room-transport", "transport")

    _commit(core_root, "core.py", "core\n", "feat: core")
    assert _publish("room-core", core_root, "core", remote)["ok"]

    _commit(tr_root, "transport.py", "transport\n", "feat: transport")
    remote_main = _g(core_root, "rev-parse", "main")

    caught = git_ops.git_catch_up_release("room-transport", remote)

    assert caught["ok"] and caught["state"] == "fast_forwarded"
    assert caught["main_sha"] == caught["remote_sha"] == remote_main
    assert _g(tr_root, "rev-parse", "main") == remote_main
    # The ref moved in place: HEAD stays on the working branch.
    assert git_ops.git_current_branch("room-transport") == "transport"
    assert _g(tr_root, "rev-parse", "refs/legion/remote/main") == remote_main

    pushed = _publish("room-transport", tr_root, "transport", remote)
    assert pushed["ok"], pushed
    assert (tr_root / "core.py").exists() is False  # transport never took core in
    files = _g(tr_root, "ls-tree", "--name-only", "main").splitlines()
    assert {"README.md", "core.py", "transport.py"} <= set(files)


def test_without_catching_up_the_push_is_rejected(data_dir, remote):
    """The failure Stage 5 removes — kept to show the test reproduces it."""
    core_root = _room(data_dir, remote, "room-core", "core")
    tr_root = _room(data_dir, remote, "room-transport", "transport")
    _commit(core_root, "core.py", "core\n", "feat: core")
    assert _publish("room-core", core_root, "core", remote)["ok"]

    _commit(tr_root, "transport.py", "transport\n", "feat: transport")
    assert _publish("room-transport", tr_root, "transport", remote)["ok"] is False


def test_up_to_date_and_ahead(data_dir, remote):
    root = _room(data_dir, remote, "room-core", "core")

    assert git_ops.git_catch_up_release("room-core", remote)["state"] == "up_to_date"

    _commit(root, "core.py", "core\n", "feat: core")
    merged = git_ops.git_merge_branch("room-core", "core", target="main", theirs=False)
    assert merged["ok"]
    _g(root, "checkout", "core")
    local_main = _g(root, "rev-parse", "main")

    result = git_ops.git_catch_up_release("room-core", remote)

    assert result["state"] == "ahead"
    assert _g(root, "rev-parse", "main") == local_main


def test_diverged_main_is_left_alone(data_dir, remote):
    core_root = _room(data_dir, remote, "room-core", "core")
    tr_root = _room(data_dir, remote, "room-transport", "transport")
    _commit(core_root, "core.py", "core\n", "feat: core")
    assert _publish("room-core", core_root, "core", remote)["ok"]

    _commit(tr_root, "transport.py", "transport\n", "feat: transport")
    merged = git_ops.git_merge_branch("room-transport", "transport", target="main", theirs=False)
    assert merged["ok"]
    _g(tr_root, "checkout", "transport")
    local_main = _g(tr_root, "rev-parse", "main")

    result = git_ops.git_catch_up_release("room-transport", remote)

    assert result["ok"] and result["state"] == "diverged"
    assert _g(tr_root, "rev-parse", "main") == local_main


def test_head_on_main_fast_forwards_the_checkout(data_dir, remote):
    core_root = _room(data_dir, remote, "room-core", "core")
    tr_root = _room(data_dir, remote, "room-transport", "transport")
    _commit(core_root, "core.py", "core\n", "feat: core")
    assert _publish("room-core", core_root, "core", remote)["ok"]
    _g(tr_root, "checkout", "main")

    result = git_ops.git_catch_up_release("room-transport", remote)

    assert result["state"] == "fast_forwarded"
    assert (tr_root / "core.py").read_text(encoding="utf-8") == "core\n"


def test_fetch_failure_is_reported(data_dir, remote, tmp_path):
    _room(data_dir, remote, "room-core", "core")

    result = git_ops.git_catch_up_release("room-core", f"file://{tmp_path / 'missing.git'}")

    assert result["ok"] is False and result["state"] == "fetch_failed"
    assert "credentials" in result["error"]


def test_release_remote_status_counts_what_a_pull_brings(data_dir, remote):
    core_root = _room(data_dir, remote, "room-core", "core")
    tr_root = _room(data_dir, remote, "room-transport", "transport")
    _commit(core_root, "core.py", "core\n", "feat: core")
    assert _publish("room-core", core_root, "core", remote)["ok"]

    # Nothing fetched since adopt: the saved remote ref still matches the room.
    before = git_ops.release_remote_status("room-transport", "transport")
    assert before["working_behind"] in (None, 0)

    git_ops.git_fetch("room-transport", remote, "main", fetch_only=True, target_branch="main")
    status = git_ops.release_remote_status("room-transport", "transport")

    assert status["ok"] is True
    assert status["main_behind"] == 2  # the feature commit + the promotion merge
    assert status["working_behind"] == 2

    git_ops.git_catch_up_release("room-transport", remote)
    after = git_ops.release_remote_status("room-transport", "transport")
    assert after["main_behind"] == 0 and after["working_behind"] == 2
    assert git_ops.release_remote_status("room-transport", "missing")["working_behind"] is None
    assert tr_root.exists()


def test_release_remote_status_without_a_fetch(data_dir):
    git_ops.git_init("lonely")

    assert git_ops.release_remote_status("lonely", "dev") == {
        "ok": False, "remote_sha": None, "main_behind": None, "working_behind": None,
    }
