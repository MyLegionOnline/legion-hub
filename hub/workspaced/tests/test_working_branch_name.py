"""A room's integration branch may be named other than ``dev``.

internal design notes Stage 1. The Hub's
``git_config.working_branch`` is the source of truth; workspaced records it in
the repo's local config (``legion.workingBranch``) through
``ensure_base_branches`` and uses it wherever it has no name from the Hub:
parking a dirty ``main``, stepping off a discarded branch, the default revert
target and the ``main`` bypass check.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops, release_guard

_CFG = ["-c", "core.autocrlf=false"]


def _g(root: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *_CFG, "-c", "safe.directory=*", "-C", str(root), *args],
        capture_output=True, text=True, check=True,
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
def core_room(data_dir: Path):
    room_id = "core-room"
    git_ops.git_init(room_id, "core")
    return room_id, data_dir / room_id


@pytest.mark.parametrize("name", ["dev", "core", "team/frontend", "release-2.x", "a_b"])
def test_valid_working_branch_names(name):
    assert git_ops.valid_working_branch_name(name)


@pytest.mark.parametrize(
    "name",
    ["", None, "main", "HEAD", "task/1", "goal/x", "session/y", "\u044f\u0434\u0440\u043e", "-x", "a..b",
     "a//b", "a/", "a.", ".a", "a/.b", "a.lock", "a b", "a@{1}", "x" * 101],
)
def test_invalid_working_branch_names(name):
    assert not git_ops.valid_working_branch_name(name)


def test_git_init_uses_and_records_the_name(core_room):
    room_id, root = core_room
    assert _branches(root) == {"main", "core"}
    assert git_ops.git_current_branch(room_id) == "core"
    assert git_ops.working_branch_of(root) == "core"


def test_git_init_default_and_invalid_name_fall_back_to_dev(data_dir):
    git_ops.git_init("plain")
    git_ops.git_init("bad", "main")
    for room_id in ("plain", "bad"):
        root = data_dir / room_id
        assert _branches(root) == {"main", "dev"}
        assert git_ops.working_branch_of(root) == "dev"


def test_an_edited_marker_falls_back_to_dev(core_room):
    _, root = core_room
    git_ops._run(["config", "--local", "legion.workingBranch", "task/123"], cwd=root)
    assert git_ops.working_branch_of(root) == "dev"


def test_ensure_does_not_bring_dev_back(core_room):
    room_id, root = core_room
    result = git_ops.ensure_base_branches(room_id, "core")
    assert result["ok"] and not result["created_dev"]
    assert _branches(root) == {"main", "core"}
    # Without a name it follows the recorded one, not `dev`.
    assert git_ops.ensure_base_branches(room_id)["working_branch"] == "core"
    assert _branches(root) == {"main", "core"}


def test_ensure_renames_an_integration_branch_without_work(data_dir):
    room_id = "renamed-early"
    git_ops.git_init(room_id)
    root = data_dir / room_id

    result = git_ops.ensure_base_branches(room_id, "transport")

    assert result["ok"] and result["renamed_from"] == "dev"
    assert _branches(root) == {"main", "transport"}
    assert git_ops.git_current_branch(room_id) == "transport"
    assert git_ops.working_branch_of(root) == "transport"


def test_ensure_refuses_to_leave_work_behind(data_dir):
    room_id = "has-work"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    _commit(root, "a.txt", "work\n", "feature on dev")
    dev_tip = _g(root, "rev-parse", "dev")

    result = git_ops.ensure_base_branches(room_id, "core")

    assert result["ok"] is False and result["error"] == "working branch mismatch"
    assert _branches(root) == {"main", "dev"}
    assert _g(root, "rev-parse", "dev") == dev_tip
    assert git_ops.working_branch_of(root) == "dev"


def test_ensure_rejects_an_invalid_name(core_room):
    room_id, root = core_room
    assert git_ops.ensure_base_branches(room_id, "main")["ok"] is False
    assert git_ops.working_branch_of(root) == "core"


def test_legacy_repo_without_marker_records_dev(data_dir):
    room_id = "legacy"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    git_ops._run(["config", "--local", "--unset", "legion.workingBranch"], cwd=root)

    assert git_ops.ensure_base_branches(room_id, "dev")["ok"]
    assert _g(root, "config", "--local", "--get", "legion.workingBranch") == "dev"


def test_a_dirty_main_is_parked_on_the_working_branch(core_room):
    room_id, root = core_room
    git_ops._run(["checkout", "main"], cwd=root)
    (root / "draft.txt").write_text("unsaved\n", encoding="utf-8")

    assert git_ops._park_worktree_before_switch(root, room_id, "main") is None

    assert git_ops.git_current_branch(room_id) == "core"
    assert "draft.txt" in _g(root, "ls-tree", "--name-only", "core")
    assert "draft.txt" not in _g(root, "ls-tree", "--name-only", "main")


def test_discard_steps_off_onto_the_working_branch(core_room):
    room_id, root = core_room
    git_ops._run(["checkout", "-b", "task/1"], cwd=root)
    _commit(root, "t.txt", "task\n", "task work")

    result = git_ops.git_discard_branch(room_id, "task/1")

    assert result["ok"] and result["deleted"]
    assert git_ops.git_current_branch(room_id) == "core"


def test_discard_refuses_the_working_branch_but_not_an_old_dev(core_room):
    room_id, root = core_room
    assert git_ops.git_discard_branch(room_id, "core")["ok"] is False
    assert git_ops.git_discard_branch(room_id, "main")["ok"] is False
    # The Hub's name wins over the recorded one.
    assert git_ops.git_discard_branch(room_id, "core", working_branch="core")["ok"] is False

    git_ops._run(["branch", "dev", "main"], cwd=root)
    result = git_ops.git_discard_branch(room_id, "dev")
    assert result["ok"] and result["deleted"]
    assert _branches(root) == {"main", "core"}


def test_revert_defaults_to_the_working_branch(core_room):
    room_id, root = core_room
    _commit(root, "a.txt", "one\n", "add a")
    sha = _g(root, "rev-parse", "core")

    result = git_ops.git_revert_commit(room_id, sha)

    assert result["ok"]
    assert not (root / "a.txt").exists()
    assert git_ops.git_current_branch(room_id) == "core"


def test_main_bypass_measures_against_the_working_branch(core_room):
    room_id, root = core_room
    git_ops._run(["commit", "--allow-empty", "-m", "feature"], cwd=root)
    git_ops._run(["checkout", "main"], cwd=root)
    git_ops._run(["merge", "--no-ff", "-m", "promote: core -> main", "core"], cwd=root)

    assert release_guard.main_bypass_commits(room_id)["commits"] == []
    assert release_guard.main_bypass_commits(room_id, integration_branch="core")["commits"] == []
    # Measured against a branch that does not carry the promoted work, the
    # promotion itself would look like a bypass — the bug this stage fixes.
    git_ops._run(["branch", "dev", "main~1"], cwd=root)
    found = release_guard.main_bypass_commits(room_id, integration_branch="dev")["commits"]
    assert [c["message"] for c in found] == ["feature"]


def _seeded_remote(tmp_path: Path) -> tuple[Path, str]:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], capture_output=True, check=True)
    clone = tmp_path / "seed-clone"
    subprocess.run(["git", "clone", str(remote), str(clone)], capture_output=True, check=True)
    _g(clone, "config", "user.name", "Remote Author")
    _g(clone, "config", "user.email", "remote@example.com")
    (clone / "app.txt").write_text("remote\n", encoding="utf-8")
    _g(clone, "add", "-A")
    _g(clone, "commit", "-m", "remote commit")
    _g(clone, "push", "origin", "HEAD:main")
    return remote, _g(clone, "rev-parse", "HEAD")


def test_adopt_into_a_pristine_room_under_another_name(data_dir, tmp_path):
    room_id = "adopt-core"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    remote, tip = _seeded_remote(tmp_path)

    result = git_ops.git_adopt_remote(room_id, f"file://{remote}", working_branch="core")

    assert result["ok"] and result["adopted"]
    assert _g(root, "rev-parse", "core") == tip
    assert _g(root, "rev-parse", "main") == tip
    # The old `dev` only held the dropped init commit.
    assert _branches(root) == {"main", "core"}
    assert git_ops.working_branch_of(root) == "core"


def test_adopt_does_not_transplant_over_work_on_the_old_branch(data_dir, tmp_path):
    room_id = "adopt-work"
    git_ops.git_init(room_id)
    root = data_dir / room_id
    _commit(root, "mine.txt", "local work\n", "work on dev")
    dev_tip = _g(root, "rev-parse", "dev")
    remote, _ = _seeded_remote(tmp_path)

    result = git_ops.git_adopt_remote(room_id, f"file://{remote}", working_branch="core")

    # Not pristine: the work on `dev` is kept and `main` is not transplanted.
    assert result["adopted"] is False or result.get("diverged")
    assert _g(root, "rev-parse", "dev") == dev_tip
    assert "init: Legion Hub workspace" in _g(root, "log", "--format=%s", "main")


# ── Stage 2: texts an agent reads never assume `dev` ─────────────────────────

def test_guard_hooks_do_not_name_dev():
    import re

    for script in release_guard.HOOKS.values():
        assert not re.search(r"\bdev\b", script)
        assert "legion-ref-guard v2" in script


def test_an_old_guard_hook_is_upgraded(core_room):
    _, root = core_room
    old = "#!/bin/sh\n# legion-ref-guard v1 - installed by Legion; old wording about dev.\nexit 0\n"
    for name in release_guard.HOOKS:
        (root / ".git" / "hooks" / name).write_text(old, encoding="utf-8")
    assert release_guard.ensure_ref_guard(root) is True
    for name, script in release_guard.HOOKS.items():
        assert (root / ".git" / "hooks" / name).read_text(encoding="utf-8") == script


def test_main_commit_refusal_names_the_room_working_branch(core_room):
    room_id, root = core_room
    assert git_ops.working_branch_of(root) == "core"
    git_ops._run(["checkout", "main"], cwd=root)
    (root / "f.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(git_ops.MainBranchProtectedError) as exc:
        git_ops.git_commit(room_id, "on main")
    assert "`core`" in str(exc.value)
    assert "`dev`" not in str(exc.value)
