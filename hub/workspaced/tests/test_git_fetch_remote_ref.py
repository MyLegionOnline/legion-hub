"""Persistent pull refs and the Git operations used to resolve pull conflicts."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import git_ops
from app.config import WORKSPACED_TOKEN
from app.main import app


def _git(root: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-C", str(root), *args],
        capture_output=True, text=True, encoding="utf-8", check=check,
    ).stdout.strip()


def _commit(root: Path, path: str, content: str) -> str:
    (root / path).write_text(content + "\n", encoding="utf-8")
    _git(root, "add", "--", path)
    _git(root, "commit", "-m", content)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "pull-conflict"
    git_ops.git_init(room_id)
    root = tmp_path / room_id
    remote = tmp_path / "remote"
    _git(tmp_path, "init", "-b", "main", str(remote))
    _git(remote, "config", "user.name", "Remote Author")
    _git(remote, "config", "user.email", "remote@example.com")
    return room_id, root, remote


def _conflict(repos):
    room_id, root, remote = repos
    path = "shared file.txt"
    local_sha = _commit(root, path, "local")
    _commit(remote, path, "remote")
    remote_sha = _commit(remote, "remote-only.txt", "incoming change")
    result = git_ops.git_fetch(room_id, str(remote), branch="main", target_branch="dev")
    assert not result["ok"] and result["conflict"], result
    assert result["branch"] == "dev"
    assert result["conflict_files"] == [path]
    assert result["remote_ref"] == "refs/legion/remote/main"
    assert result["remote_sha"] == remote_sha
    assert result["local_sha"] == local_sha == _git(root, "rev-parse", "dev")
    assert not (root / ".git" / "MERGE_HEAD").exists()
    assert _git(root, "status", "--porcelain") == ""
    return local_sha, remote_sha, result["remote_ref"]


def test_remote_ref_survives_another_branch_fetch(repos):
    room_id, root, remote = repos
    _, first_sha, first_ref = _conflict(repos)
    _git(remote, "checkout", "--orphan", "review/other")
    _git(remote, "rm", "-rf", ".")
    other_sha = _commit(remote, "other.txt", "other")
    result = git_ops.git_fetch(room_id, str(remote), branch="review/other", target_branch="dev")

    assert result["ok"], result
    assert result["remote_ref"] == "refs/legion/remote/review/other"
    assert result["remote_sha"] == other_sha == _git(root, "rev-parse", "FETCH_HEAD")
    assert result["local_sha"] == _git(root, "rev-parse", "dev")
    assert _git(root, "rev-parse", first_ref) == first_sha
    assert _git(root, "rev-parse", result["remote_ref"]) == other_sha


@pytest.mark.parametrize(("strategy", "expected"), [("ours", "local"), ("theirs", "remote")])
def test_strategy_merges_both_unrelated_histories(repos, strategy, expected):
    room_id, root, _ = repos
    local_sha, remote_sha, ref = _conflict(repos)
    result = git_ops.git_merge_branch(room_id, ref, target="dev", strategy=strategy)
    assert result["ok"], result
    assert _git(root, "show", "-s", "--format=%P", "dev").split() == [local_sha, remote_sha]
    assert (root / "shared file.txt").read_text(encoding="utf-8").strip() == expected
    assert (root / "remote-only.txt").read_text(encoding="utf-8").strip() == "incoming change"


@pytest.mark.parametrize("checked_out", [True, False])
def test_reset_preserves_branch_tip_and_dirty_work(repos, checked_out):
    room_id, root, _ = repos
    local_sha, remote_sha, ref = _conflict(repos)
    if not checked_out:
        _git(root, "checkout", "main")
    (root / "draft.txt").write_text("uncommitted work", encoding="utf-8")

    result = git_ops.git_reset_branch_to(room_id, "dev", ref)

    assert result["ok"], result
    assert result["sha"] == _git(root, "rev-parse", "dev") == remote_sha
    backup = result["backup_ref"]
    assert backup.startswith("refs/legion/trash/dev@")
    if checked_out:
        assert _git(root, "rev-parse", backup + "^") == local_sha
        assert _git(root, "show", f"{backup}:draft.txt") == "uncommitted work"
        assert not (root / "draft.txt").exists()
        assert (root / "shared file.txt").read_text(encoding="utf-8").strip() == "remote"
    else:
        assert _git(root, "rev-parse", backup) == local_sha
        assert _git(root, "branch", "--show-current") == "main"
        assert (root / "draft.txt").read_text(encoding="utf-8") == "uncommitted work"


def test_reset_main_parks_dirty_work_on_dev(repos):
    room_id, root, remote = repos
    remote_sha = _commit(remote, "from-remote.txt", "incoming change")
    fetch = git_ops.git_fetch(room_id, str(remote), branch="main", target_branch="dev")
    assert fetch["ok"], fetch
    _git(root, "checkout", "main")
    local_sha = _git(root, "rev-parse", "main")
    (root / "draft.txt").write_text("uncommitted work", encoding="utf-8")

    result = git_ops.git_reset_branch_to(room_id, "main", fetch["remote_ref"])

    assert result["ok"], result
    assert _git(root, "rev-parse", "main") == remote_sha
    assert _git(root, "rev-parse", result["backup_ref"]) == local_sha
    assert _git(root, "show", "dev:draft.txt") == "uncommitted work"


def test_reset_refuses_unknown_commit_without_parking(repos):
    room_id, root, _ = repos
    local_sha = _git(root, "rev-parse", "dev")
    (root / "draft.txt").write_text("uncommitted work", encoding="utf-8")

    result = git_ops.git_reset_branch_to(room_id, "dev", "refs/legion/remote/missing")

    assert not result["ok"]
    assert _git(root, "rev-parse", "dev") == local_sha
    assert _git(root, "for-each-ref", "refs/legion/trash") == ""
    assert _git(root, "status", "--porcelain") == "?? draft.txt"


def test_saved_remote_ref_cannot_bypass_private_files_policy(repos):
    room_id, root, remote = repos
    (remote / ".userfiles").mkdir()
    _commit(remote, ".userfiles/from-remote.txt", "tracked remote file")
    private = root / ".userfiles" / "local.txt"
    private.write_text("room private data", encoding="utf-8")
    local_sha = _git(root, "rev-parse", "dev")

    fetch = git_ops.git_fetch(room_id, str(remote), branch="main", target_branch="dev")
    assert fetch["userfiles_policy_conflict"]
    ref = fetch["remote_ref"]
    assert _git(root, "rev-parse", ref) == fetch["remote_sha"]

    merge = git_ops.git_merge_branch(room_id, ref, target="dev", strategy="theirs")
    assert not merge["ok"] and merge["userfiles_policy_conflict"]
    reset = git_ops.git_reset_branch_to(room_id, "dev", ref)
    assert not reset["ok"] and "private-files policy conflict" in reset["stderr"]
    assert _git(root, "rev-parse", "dev") == local_sha
    assert private.read_text(encoding="utf-8") == "room private data"


def test_workspaced_http_fetch_merge_and_reset(repos):
    room_id, root, remote = repos
    _commit(root, "shared file.txt", "local")
    remote_sha = _commit(remote, "shared file.txt", "remote")
    api = TestClient(app, headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"})
    url = f"/workspaces/{room_id}"

    fetch = api.post(f"{url}/git/fetch", json={"remote_url": str(remote), "target_branch": "dev"})
    assert fetch.status_code == 200, fetch.text
    fetched = fetch.json()
    assert fetched["conflict_files"] == ["shared file.txt"]
    assert fetched["local_sha"] == _git(root, "rev-parse", "dev")
    assert fetched["remote_sha"] == remote_sha

    body = {"name": fetched["remote_ref"], "target": "dev"}
    neutral = api.post(f"{url}/merge", json={**body, "strategy": None})
    assert neutral.status_code == 200 and neutral.json()["conflict"]
    merge = api.post(f"{url}/merge", json={
        **body, "strategy": "ours", "expected_tips": [fetched["local_sha"], remote_sha],
    })
    assert merge.status_code == 200 and merge.json()["ok"], merge.text
    reset = api.post(f"{url}/git/reset-branch", json={
        "branch": "dev", "ref": fetched["remote_ref"],
        "expected_tips": [merge.json()["sha"], remote_sha],
    })
    assert reset.status_code == 200 and reset.json()["ok"], reset.text
    assert reset.json()["sha"] == remote_sha
    assert _git(root, "rev-parse", reset.json()["backup_ref"]) == merge.json()["sha"]


def test_fetch_review_refreshes_refs_without_moving_or_parking_checkout(repos):
    room_id, root, remote = repos
    local_sha, _, ref = _conflict(repos)
    remote_sha = _commit(remote, "new.txt", "new remote commit")
    _git(root, "checkout", "main")
    head = _git(root, "rev-parse", "HEAD")
    (root / "draft.txt").write_text("uncommitted", encoding="utf-8")
    api = TestClient(app, headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"})

    response = api.post(
        f"/workspaces/{room_id}/git/fetch-review",
        json={"remote_url": str(remote), "target_branch": "dev"},
    )

    assert response.status_code == 200, response.text
    fetched = response.json()
    assert fetched["ok"] and not fetched["head_changed"]
    assert fetched["local_sha"] == local_sha == _git(root, "rev-parse", "dev")
    assert fetched["remote_sha"] == remote_sha == _git(root, "rev-parse", ref)
    assert _git(root, "rev-parse", "HEAD") == head
    assert _git(root, "status", "--porcelain") == "?? draft.txt"


@pytest.mark.parametrize("moved_side", ["remote", "local"])
def test_reviewed_operations_reject_changed_tips(repos, moved_side):
    room_id, root, remote = repos
    local_sha, remote_sha, ref = _conflict(repos)
    if moved_side == "remote":
        _commit(remote, "new.txt", "new remote commit")
        assert git_ops.git_fetch(room_id, str(remote), target_branch="dev", fetch_only=True)["ok"]
    else:
        _commit(root, "new.txt", "new local commit")
    head = _git(root, "rev-parse", "HEAD")
    api = TestClient(app, headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"})
    url = f"/workspaces/{room_id}"
    expected = [local_sha, remote_sha]

    merge = api.post(f"{url}/merge", json={
        "name": ref, "target": "dev", "strategy": "ours", "expected_tips": expected,
    })
    reset = api.post(f"{url}/git/reset-branch", json={
        "branch": "dev", "ref": ref, "expected_tips": expected,
    })

    for response in (merge, reset):
        assert response.status_code == 200 and response.json()["stale"], response.text
    assert _git(root, "rev-parse", "HEAD") == head
    assert _git(root, "for-each-ref", "refs/legion/trash") == ""


def test_reviewed_reset_parks_new_local_work_and_requests_another_review(repos):
    room_id, root, _ = repos
    local_sha, remote_sha, ref = _conflict(repos)
    (root / "draft.txt").write_text("uncommitted", encoding="utf-8")

    reset = git_ops.git_reset_branch_to(
        room_id, "dev", ref, expected_tips=(local_sha, remote_sha),
    )

    assert reset["stale"]
    assert _git(root, "rev-parse", "dev^") == local_sha
    assert _git(root, "show", "dev:draft.txt") == "uncommitted"
    assert _git(root, "for-each-ref", "refs/legion/trash") == ""


def test_remote_report_handles_unrelated_histories_and_flags_binary_paths(repos):
    room_id, root, remote = repos
    (root / "local package.deb").write_bytes(b"\0binary local")
    (remote / "remote dump").write_bytes(b"\0binary remote")
    _git(root, "add", "local package.deb")
    _git(remote, "add", "remote dump")
    local_sha, remote_sha, ref = _conflict(repos)

    report = git_ops.git_divergence_report(room_id, ref, "dev")

    assert report["ok"] and report["conflict"], report
    assert report["merge_base"] == {}
    assert report["base_head"]["sha"] == local_sha
    assert report["branch_head"]["sha"] == remote_sha
    assert report["conflict_files"] == ["shared file.txt"]
    assert "CONFLICT" in report["merge_messages"]
    assert "local" in report["conflict_details"][0]["base_diff"]
    assert "remote" in report["conflict_details"][0]["branch_diff"]
    assert report["base_binary_files"] == ["local package.deb"]
    assert report["branch_binary_files"] == ["remote dump"]



def test_pull_rework_seeds_both_histories_without_touching_the_room_checkout(repos):
    room_id, root, remote = repos
    local_sha, remote_sha, ref = _conflict(repos)
    (root / "draft.txt").write_text("uncommitted", encoding="utf-8")
    api = TestClient(app, headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"})
    url = f"/workspaces/{room_id}/git/prepare-pull-rework"
    body = {
        "task_branch": "task/rework", "branch": "dev", "remote_ref": ref,
        "expected_tips": [local_sha, remote_sha], "conflict_files": ["shared file.txt"],
    }

    response = api.post(url, json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["ok"], result
    sha = result["sha"]
    assert _git(root, "show", "-s", "--format=%P", sha).split() == [local_sha, remote_sha]
    assert _git(root, "show", f"{sha}:shared file.txt") == "local"
    assert _git(root, "show", f"{sha}:remote-only.txt") == "incoming change"
    assert "+remote" in result["remote_patches"]["shared file.txt"]
    assert _git(root, "rev-parse", "HEAD") == local_sha
    assert _git(root, "status", "--porcelain") == "?? draft.txt"
    assert len(_git(root, "worktree", "list").splitlines()) == 1
    assert api.post(url, json=body).json()["sha"] == sha
    assert not api.post(url, json={**body, "branch": "main"}).json()["ok"]

    _git(root, "checkout", "task/rework")
    _commit(root, "shared file.txt", "local and remote")
    assert git_ops.git_merge_branch(room_id, "task/rework", target="dev")["ok"]
    _git(remote, "checkout", "--detach")
    _git(root, "push", str(remote), "dev:main")
    assert _git(remote, "rev-parse", "main") == _git(root, "rev-parse", "dev")
    stale = api.post(url, json={**body, "task_branch": "task/stale"}).json()
    assert stale["stale"] and not git_ops.git_branch_exists(room_id, "task/stale")


@pytest.mark.parametrize("remote_deleted", [True, False])
def test_pull_rework_seeds_modify_delete_conflict_with_local_state(repos, remote_deleted):
    room_id, root, _ = repos
    ancestor = _commit(root, "shared.txt", "original")
    _git(root, "checkout", "-b", "incoming")
    if remote_deleted:
        _git(root, "rm", "shared.txt")
        _git(root, "commit", "-m", "remote deletion")
    else:
        _commit(root, "shared.txt", "remote edit")
    remote_sha = _git(root, "rev-parse", "HEAD")
    ref = "refs/legion/remote/dev"
    _git(root, "update-ref", ref, remote_sha)
    _git(root, "checkout", "dev")
    if remote_deleted:
        _commit(root, "shared.txt", "local edit")
    else:
        _git(root, "rm", "shared.txt")
        _git(root, "commit", "-m", "local deletion")
    local_sha = _git(root, "rev-parse", "dev")

    result = git_ops.git_prepare_pull_rework(room_id, "task/rework", "dev", ref, (local_sha, remote_sha), [])

    assert result["ok"], result
    sha = result["sha"]
    assert _git(root, "show", "-s", "--format=%P", sha).split() == [local_sha, remote_sha]
    assert _git(root, "diff", "--name-only", local_sha, sha) == ""
    assert result["diff_base"] == ancestor
    assert ("-original" if remote_deleted else "+remote edit") in result["remote_patches"]["shared.txt"]
    assert len(_git(root, "worktree", "list").splitlines()) == 1
