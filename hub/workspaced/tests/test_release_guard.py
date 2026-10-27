"""The room's guard hooks keep ``main`` and release tags out of an agent's
reach, and release tags are created on ``main`` only, never moved."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from app import git_ops, release_guard


def _agent_git(root: Path, *args: str) -> subprocess.CompletedProcess:
    """git as the agent's bash runs it: no LEGION_GIT_TRUSTED."""
    env = {k: v for k, v in os.environ.items() if k != release_guard.TRUSTED_ENV}
    env.update(GIT_AUTHOR_NAME="a", GIT_AUTHOR_EMAIL="a@x", GIT_COMMITTER_NAME="a", GIT_COMMITTER_EMAIL="a@x")
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(root), *args],
        capture_output=True, text=True, env=env,
    )


def _sha(root: Path, ref: str) -> str:
    return _agent_git(root, "rev-parse", f"{ref}^{{commit}}").stdout.strip()


def _bare(tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    return remote


@pytest.fixture
def room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "guard-room"
    git_ops.git_init(room_id)
    return room_id, tmp_path / room_id


def test_git_init_installs_the_hooks(room):
    _, root = room
    for name, script in release_guard.HOOKS.items():
        path = root / ".git" / "hooks" / name
        assert path.read_text() == script
        assert os.access(path, os.X_OK)
    assert release_guard.ensure_ref_guard(root) is False  # idempotent


def test_agent_cannot_commit_to_main_even_with_no_verify(room):
    _, root = room
    main_before = _sha(root, "main")
    assert _agent_git(root, "checkout", "main").returncode == 0  # switching is harmless
    (root / "f.txt").write_text("x\n")
    _agent_git(root, "add", "f.txt")
    r = _agent_git(root, "commit", "-m", "on main")
    assert r.returncode != 0 and "promotion" in r.stderr and "task/<task id>" in r.stderr
    assert _agent_git(root, "commit", "--no-verify", "-m", "on main").returncode != 0
    assert _sha(root, "main") == main_before


def test_agent_still_works_on_its_own_branches(room):
    _, root = room
    assert _agent_git(root, "checkout", "-b", "task/1").returncode == 0
    (root / "f.txt").write_text("x\n")
    _agent_git(root, "add", "f.txt")
    assert _agent_git(root, "commit", "-m", "work").returncode == 0


def test_moving_main_by_branch_or_update_ref_is_refused(room):
    _, root = room
    _agent_git(root, "commit", "--allow-empty", "-m", "work")  # on dev
    main_before = _sha(root, "main")
    assert _agent_git(root, "branch", "-f", "main", "dev").returncode != 0
    assert _agent_git(root, "update-ref", "refs/heads/main", "dev").returncode != 0
    assert _agent_git(root, "branch", "-D", "main").returncode != 0
    assert _sha(root, "main") == main_before


def test_workspaced_itself_still_moves_main(room):
    room_id, root = room
    git_ops._run(["checkout", "main"], cwd=root)
    (root / "r.txt").write_text("r\n")
    assert git_ops.git_commit(room_id, "restore", allow_main_commit=True)


def test_tags_are_created_but_never_moved_or_deleted(room):
    _, root = room
    assert _agent_git(root, "tag", "v0.1.0", "main").returncode == 0
    _agent_git(root, "commit", "--allow-empty", "-m", "next")  # on dev
    r = _agent_git(root, "tag", "-f", "v0.1.0", "dev")
    assert r.returncode != 0 and "immutable" in r.stderr
    assert _agent_git(root, "tag", "-d", "v0.1.0").returncode != 0
    assert _sha(root, "v0.1.0") == _sha(root, "main")


def test_push_of_main_or_a_tag_is_refused_from_the_container(room, tmp_path):
    _, root = room
    remote = _bare(tmp_path)
    r = _agent_git(root, "push", str(remote), "main")
    assert r.returncode != 0 and "publish_for_deploy" in r.stderr
    _agent_git(root, "tag", "v0.1.0", "main")
    r = _agent_git(root, "push", str(remote), "v0.1.0")
    assert r.returncode != 0 and "release_tag" in r.stderr
    assert _agent_git(root, "push", str(remote), "dev").returncode == 0


def test_a_repo_with_its_own_hooks_path_is_left_alone(tmp_path):
    root = tmp_path / "husky"
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(root), "config", "core.hooksPath", ".husky"], check=True)
    assert release_guard.ensure_ref_guard(root) is False
    assert not (root / ".git" / "hooks" / "reference-transaction").exists()


def test_a_project_hook_of_the_same_name_is_not_overwritten(tmp_path):
    root = tmp_path / "own"
    subprocess.run(["git", "init", str(root)], check=True, capture_output=True)
    own = root / ".git" / "hooks" / "pre-push"
    own.write_text("#!/bin/sh\nexit 0\n")
    release_guard.ensure_ref_guard(root)
    assert own.read_text() == "#!/bin/sh\nexit 0\n"
    assert (root / ".git" / "hooks" / "reference-transaction").exists()


def test_main_bypass_lists_only_commits_that_skipped_the_promotion(room):
    room_id, root = room
    # promoted work: committed on dev, merged into main by workspaced
    git_ops._run(["commit", "--allow-empty", "-m", "feature"], cwd=root)
    git_ops._run(["checkout", "main"], cwd=root)
    git_ops._run(["merge", "--no-ff", "-m", "promote: dev -> main", "dev"], cwd=root)
    assert release_guard.main_bypass_commits(room_id)["commits"] == []
    # a direct commit on main (the guard was got round)
    git_ops._run(["commit", "--allow-empty", "-m", "sneaky"], cwd=root)
    found = release_guard.main_bypass_commits(room_id)["commits"]
    assert [c["message"] for c in found] == ["sneaky"]
    # once the forge has it, it is not reported again; an unknown sha is ignored
    assert release_guard.main_bypass_commits(room_id, exclude=[found[0]["sha"]])["commits"] == []
    assert release_guard.main_bypass_commits(room_id, exclude=["deadbeef" * 5])["commits"] != []


def test_release_tag_goes_on_main_only_and_is_idempotent(room):
    room_id, root = room
    assert release_guard.check_release_tag(room_id, "1.0")["code"] == "tag_invalid"
    assert release_guard.check_release_tag(room_id, "v1.0.0", "nope")["code"] == "ref_unknown"
    git_ops._run(["commit", "--allow-empty", "-m", "dev only"], cwd=root)
    assert release_guard.check_release_tag(room_id, "v1.0.0", "dev")["code"] == "not_on_main"

    made = release_guard.create_release_tag(room_id, "v1.0.0", message="First release")
    assert made["ok"] and made["created"] and made["sha"] == _sha(root, "main")
    assert _agent_git(root, "cat-file", "-t", "v1.0.0").stdout.strip() == "tag"  # annotated
    again = release_guard.create_release_tag(room_id, "v1.0.0")
    assert again["ok"] and again["exists"] and "created" not in again


def test_an_existing_tag_on_another_commit_is_refused(room):
    room_id, root = room
    release_guard.create_release_tag(room_id, "v1.0.0")
    git_ops._run(["checkout", "main"], cwd=root)
    git_ops._run(["commit", "--allow-empty", "-m", "next"], cwd=root)
    assert release_guard.check_release_tag(room_id, "v1.0.0")["code"] == "tag_exists"


def test_push_release_tag_pushes_exactly_that_tag(room, tmp_path):
    room_id, root = room
    remote = _bare(tmp_path)
    release_guard.create_release_tag(room_id, "v1.0.0")
    _agent_git(root, "tag", "scratch", "main")
    assert release_guard.push_release_tag(room_id, str(remote), "v1.0.0")["ok"]
    tags = subprocess.run(["git", "-C", str(remote), "tag"], capture_output=True, text=True).stdout.split()
    assert tags == ["v1.0.0"]
    assert not release_guard.push_release_tag(room_id, str(remote), "v9.9.9")["ok"]



def test_latest_release_suggestions_ignore_prereleases_and_other_branches(tmp_path, monkeypatch):
    from types import SimpleNamespace
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(release_guard, "_workspace_root", lambda room: tmp_path)
    calls = []
    def run(args, *, cwd):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="v1.9.9\nv1.10.0\nv9.0.0-rc.1\n", stderr="")
    monkeypatch.setattr(release_guard, "_run", run)
    assert release_guard.latest_release_tag("room") == {"ok": True, "latest": "v1.10.0", "patch": "v1.10.1", "minor": "v1.11.0"}
    assert calls == [["tag", "--merged", "refs/heads/main", "--list", "v*"]]
    monkeypatch.setattr(release_guard, "_run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""))
    assert release_guard.latest_release_tag("room")["patch"] == "v0.1.0"
