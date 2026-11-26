"""Task worktrees in workspaced (internal design notes, E0).

A task gets ``<room>/.legion/wt/<task8>`` — a ``git worktree`` on its own
branch. Acceptance criteria:
  * adding a tree never moves the room's shared HEAD, and a commit in the tree
    lands on the task's branch only;
  * the tree's ``.git`` file is relative (it must resolve inside the room
    container too) and the tree is locked against a container-side prune;
  * file operations with ``tree`` read and write that tree, never the shared
    one; ``.git`` and paths that leave the tree are refused;
  * removing a tree commits pending work first and keeps the branch;
  * a ``.git`` file that points anywhere but the room's own worktree entry is
    refused by the git guard;
  * checks run against a tree read its ``run.json`` and check nothing out.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest

import app.config as config
import app.generation as generation
from app import checks, file_ops, git_exec, git_ops
from app.task_tree import TaskTreeError, container_path

_CFG = ["-c", "core.autocrlf=false"]
T1 = "t1abc123"
T2 = "t2def456"

posix_only = pytest.mark.skipif(os.name != "posix", reason="needs POSIX symlinks")


def _g(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(
        ["git", *_CFG, "-C", str(root), *args],
        capture_output=True, text=True, check=check,
    )
    return r.stdout.strip()


@pytest.fixture
def room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for module in (git_ops, config, generation, file_ops):
        monkeypatch.setattr(module, "WORKSPACE_DATA_DIR", tmp_path, raising=False)
    # chown needs root; the tests run as an ordinary user.
    monkeypatch.setattr(git_ops, "chown_workspace_tree", lambda *_a, **_kw: None)
    monkeypatch.setattr(git_ops, "_chown_walk", lambda *_a, **_kw: None)
    room_id = "wt-room"
    git_ops.git_init(room_id)
    root = tmp_path / room_id
    _g(root, "config", "user.name", "Test")
    _g(root, "config", "user.email", "test@example.com")
    (root / "app.txt").write_text("seed\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "dev: seed")
    git_ops.ensure_base_branches(room_id)
    return room_id, root


def _branch(root: Path) -> str:
    return _g(root, "rev-parse", "--abbrev-ref", "HEAD")


def test_add_gives_the_task_its_own_tree_and_leaves_shared_head_alone(room):
    room_id, root = room
    before_branch, before_head = _branch(root), _g(root, "rev-parse", "HEAD")

    info = git_ops.worktree_add(room_id, T1, f"task/{T1}")

    tree = root / ".legion" / "wt" / T1
    assert info["created"] is True
    assert info["path"] == container_path(T1) == f"/workspace/.legion/wt/{T1}"
    assert info["branch"] == f"task/{T1}"
    assert (tree / "app.txt").read_text() == "seed\n"
    # The shared tree did not move.
    assert _branch(root) == before_branch
    assert _g(root, "rev-parse", "HEAD") == before_head
    assert git_ops.git_current_branch(room_id) == before_branch
    assert git_ops.git_current_branch(room_id, tree=T1) == f"task/{T1}"


def test_tree_git_file_is_relative_and_the_tree_is_locked(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")

    tree = root / ".legion" / "wt" / T1
    assert (tree / ".git").read_text().strip() == f"gitdir: ../../../.git/worktrees/{T1}"
    # git resolves the relative file from the tree as well.
    assert _g(tree, "rev-parse", "--abbrev-ref", "HEAD") == f"task/{T1}"
    # A container-side `git worktree prune` must not drop the tree.
    assert (root / ".git" / "worktrees" / T1 / "locked").is_file()
    _g(root, "worktree", "prune")
    assert (root / ".git" / "worktrees" / T1).is_dir()


def test_the_shared_tree_never_sees_the_tree_directory(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    assert _g(root, "status", "--porcelain") == ""
    assert git_ops.git_commit(room_id, "should be nothing") is None


def test_commit_in_the_tree_lands_on_the_task_branch_only(room):
    room_id, root = room
    shared = _branch(root)
    shared_tip = _g(root, "rev-parse", shared)
    git_ops.worktree_add(room_id, T1, f"task/{T1}")

    result = file_ops.write_file(
        room_id, "/src/new.txt", "hello\n", commit_message="task: add new", tree=T1,
    )

    assert result["commit_sha"]
    assert not (root / "src" / "new.txt").exists()
    assert (root / ".legion" / "wt" / T1 / "src" / "new.txt").read_text() == "hello\n"
    assert _g(root, "rev-parse", f"task/{T1}") == result["commit_sha"]
    assert _g(root, "rev-parse", shared) == shared_tip
    assert _branch(root) == shared


def test_two_trees_do_not_see_each_others_work(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    git_ops.worktree_add(room_id, T2, f"task/{T2}")

    file_ops.write_file(room_id, "one.txt", "1\n", commit_message="t1", tree=T1)
    file_ops.write_file(room_id, "two.txt", "2\n", commit_message="t2", tree=T2)

    assert (file_ops.read_file(room_id, "one.txt", tree=T1) or "").strip() == "1"
    assert file_ops.read_file(room_id, "one.txt", tree=T2) is None
    assert (file_ops.read_file(room_id, "two.txt", tree=T2) or "").strip() == "2"
    assert file_ops.read_file(room_id, "one.txt") is None
    assert [e["path"] for e in file_ops.get_tree(room_id, tree=T1) if e["path"].endswith(".txt")] == [
        "/app.txt", "/one.txt",
    ]
    assert git_ops.git_current_branch(room_id, tree=T2) == f"task/{T2}"


def test_status_diff_log_and_snapshot_follow_the_tree(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    tree = root / ".legion" / "wt" / T1
    (tree / "app.txt").write_text("changed\n")

    status = git_ops.git_status(room_id, tree=T1)
    assert status["branch"] == f"task/{T1}" and status["dirty"] is True
    assert status["changes"] == ["app.txt"]
    assert git_ops.git_status(room_id)["dirty"] is False
    assert "+changed" in git_ops.git_diff(room_id, tree=T1)
    assert git_ops.git_diff(room_id) == ""

    sha = git_ops.git_commit(room_id, "task: change", tree=T1)
    assert sha == _g(tree, "rev-parse", "HEAD")
    assert git_ops.git_log(room_id, limit=1, tree=T1)[0]["message"] == "task: change"
    assert git_ops.git_log(room_id, limit=1)[0]["message"] != "task: change"

    snap = file_ops.get_snapshot(room_id, preview_paths=["app.txt"], tree=T1)
    assert snap["git"]["branch"] == f"task/{T1}"
    assert snap["previews"]["/app.txt"].startswith("changed")


def test_agent_paths_cannot_reach_git_or_leave_the_tree(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")

    assert file_ops.read_file(room_id, ".git", tree=T1) is None
    with pytest.raises(ValueError):
        file_ops.write_file(room_id, ".git", "gitdir: /elsewhere\n", tree=T1)
    with pytest.raises(ValueError):
        file_ops.write_file(room_id, ".git/hooks/pre-commit", "x", tree=T1)
    assert file_ops.read_file(room_id, "../../../.git/config", tree=T1) is None
    assert (root / ".legion" / "wt" / T1 / ".git").read_text().startswith("gitdir: ../../../")


@pytest.mark.parametrize("name", ["", "../x", "a/b", "wt/../x", "x y", "-rf", ".hidden"])
def test_bad_tree_names_are_refused(room, name):
    room_id, _ = room
    if name == "":
        # "no tree" means the shared tree — the one name that is allowed.
        assert (file_ops.read_file(room_id, "app.txt", tree=name) or "").strip() == "seed"
        return
    with pytest.raises(TaskTreeError):
        file_ops.read_file(room_id, "app.txt", tree=name)
    with pytest.raises(TaskTreeError):
        git_ops.worktree_add(room_id, name, "task/x")


def test_a_missing_tree_is_an_error_not_the_shared_tree(room):
    room_id, _ = room
    with pytest.raises(TaskTreeError) as caught:
        file_ops.write_file(room_id, "x.txt", "x", commit_message="m", tree="nosuch1")
    assert caught.value.status == 404
    with pytest.raises(TaskTreeError):
        git_ops.git_commit(room_id, "m", tree="nosuch1")
    assert file_ops.read_file(room_id, "x.txt") is None


def test_add_is_idempotent_and_refuses_a_second_branch(room):
    room_id, root = room
    first = git_ops.worktree_add(room_id, T1, f"task/{T1}")
    again = git_ops.worktree_add(room_id, T1, f"task/{T1}")
    assert first["created"] is True and again["created"] is False
    assert again["head"] == first["head"]

    with pytest.raises(TaskTreeError) as caught:
        git_ops.worktree_add(room_id, T1, "task/other")
    assert caught.value.status == 409


def test_add_reuses_an_existing_branch(room):
    room_id, root = room
    _g(root, "branch", f"task/{T1}", _branch(root))
    info = git_ops.worktree_add(room_id, T1, f"task/{T1}")
    assert info["created"] is True and info["branch"] == f"task/{T1}"


def test_a_branch_checked_out_elsewhere_is_a_409(room):
    room_id, root = room
    _g(root, "checkout", "-b", f"task/{T1}")
    with pytest.raises(TaskTreeError) as caught:
        git_ops.worktree_add(room_id, T1, f"task/{T1}")
    assert caught.value.status == 409
    assert _branch(root) == f"task/{T1}"  # nothing was moved


def test_add_refuses_main_and_odd_branch_names(room):
    room_id, _ = room
    for branch in ("main", "HEAD", "", "-x", "a..b", "x.lock", "x/"):
        with pytest.raises(TaskTreeError):
            git_ops.worktree_add(room_id, T1, branch)


def test_list_reports_only_task_trees(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T2, f"task/{T2}")
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    (root / ".legion" / "wt" / T1 / "dirty.txt").write_text("x\n")

    listed = git_ops.worktree_list(room_id)
    assert [e["task8"] for e in listed] == [T1, T2]
    first = listed[0]
    assert first["branch"] == f"task/{T1}"
    assert first["path"] == container_path(T1)
    assert first["locked"] is True and first["dirty"] is True
    assert listed[1]["dirty"] is False


def test_remove_parks_pending_work_and_keeps_the_branch(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    tree = root / ".legion" / "wt" / T1
    (tree / "wip.txt").write_text("unfinished\n")

    result = git_ops.worktree_remove(room_id, T1)

    assert result["ok"] is True and result["removed"] is True
    assert result["parked_sha"] == _g(root, "rev-parse", f"task/{T1}")
    assert not tree.exists()
    assert git_ops.worktree_list(room_id) == []
    assert "wip.txt" in _g(root, "ls-tree", "-r", "--name-only", f"task/{T1}")
    assert "park task tree" in _g(root, "log", "-1", "--format=%s", f"task/{T1}")
    assert not (root / ".git" / "worktrees" / T1).exists()
    # The same task can get its tree back on the same branch.
    again = git_ops.worktree_add(room_id, T1, f"task/{T1}")
    assert (root / ".legion" / "wt" / T1 / "wip.txt").read_text() == "unfinished\n"
    assert again["created"] is True


def test_remove_of_a_clean_or_missing_tree(room):
    room_id, root = room
    assert git_ops.worktree_remove(room_id, "never001") == {
        "ok": True, "removed": False, "parked_sha": None, "task8": "never001",
    }
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    tip = _g(root, "rev-parse", f"task/{T1}")
    result = git_ops.worktree_remove(room_id, T1)
    assert result["removed"] is True and result["parked_sha"] is None
    assert _g(root, "rev-parse", f"task/{T1}") == tip


def test_remove_keeps_a_tree_it_cannot_park_unless_forced(room, monkeypatch):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    tree = root / ".legion" / "wt" / T1
    (tree / "wip.txt").write_text("unfinished\n")
    # A broken .git file: git is refused here, so nothing can be committed.
    (tree / ".git").write_text("gitdir: /nowhere\n")

    refused = git_ops.worktree_remove(room_id, T1)
    assert refused["ok"] is False and "error" in refused
    assert tree.is_dir()

    forced = git_ops.worktree_remove(room_id, T1, force=True)
    assert forced["ok"] is True and forced["removed"] is True
    assert not tree.exists()
    assert not (root / ".git" / "worktrees" / T1).exists()


def test_prune_forgets_a_tree_whose_directory_vanished(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    import shutil

    shutil.rmtree(root / ".legion" / "wt" / T1)
    # Locked, so git alone would keep it forever.
    _g(root, "worktree", "prune")
    assert (root / ".git" / "worktrees" / T1).is_dir()

    result = git_ops.worktree_prune(room_id)
    assert result["pruned"] == [T1]
    assert not (root / ".git" / "worktrees" / T1).exists()
    assert git_ops.worktree_list(room_id) == []


# ---------------------------------------------------------------------------
# The git guard
# ---------------------------------------------------------------------------

def test_a_tree_of_the_room_passes_the_guard(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    assert git_exec.repo_safety_error(root / ".legion" / "wt" / T1) == ""


def test_a_dotgit_file_pointing_at_another_room_is_refused(room, tmp_path):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    other = tmp_path / "other-room"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    tree = root / ".legion" / "wt" / T1

    (tree / ".git").write_text(f"gitdir: {other / '.git'}\n")
    assert git_exec.repo_safety_error(tree) != ""

    (tree / ".git").write_text("gitdir: ../../../../other-room/.git\n")
    assert git_exec.repo_safety_error(tree) != ""
    assert git_ops.git_commit(room_id, "x", tree=T1) is None


def test_a_forged_worktree_entry_is_refused(room, tmp_path):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    tree = root / ".legion" / "wt" / T1
    admin = root / ".git" / "worktrees" / T1
    (admin / "commondir").write_text("../../../other-room/.git\n")
    assert git_exec.repo_safety_error(tree) != ""


def test_a_task_tree_dir_elsewhere_in_the_volume_is_refused(room, tmp_path):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    fake = root / "not-wt" / T1
    fake.mkdir(parents=True)
    (fake / ".git").write_text((root / ".legion" / "wt" / T1 / ".git").read_text())
    assert "linked worktree" in git_exec.repo_safety_error(fake)


@posix_only
def test_a_symlinked_wt_dir_is_refused(room, tmp_path):
    room_id, root = room
    other = tmp_path / "other-room"
    (other / ".legion" / "wt").mkdir(parents=True)
    (root / ".legion").mkdir(exist_ok=True)
    os.symlink(other / ".legion" / "wt", root / ".legion" / "wt")
    with pytest.raises(TaskTreeError):
        git_ops.worktree_add(room_id, T1, f"task/{T1}")


@posix_only
def test_a_symlink_leaving_the_tree_cannot_be_written_through(room, tmp_path):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    victim = tmp_path / "victim"
    victim.mkdir()
    os.symlink(victim, root / ".legion" / "wt" / T1 / "out")
    with pytest.raises(ValueError):
        file_ops.write_file(room_id, "out/pwn.txt", "x", tree=T1)
    assert not (victim / "pwn.txt").exists()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def test_checks_in_a_tree_read_its_run_json_and_check_nothing_out(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    file_ops.write_file(
        room_id, ".legion/run.json",
        json.dumps({"checks": [{"name": "t", "command": "echo tree-check"}]}),
        commit_message="task: checks", tree=T1,
    )
    calls: list[list[str]] = []

    async def exec_fn(cmd, timeout, env):
        calls.append(cmd)
        return {"exit_code": 0, "stdout": "ok", "stderr": "", "timed_out": False}

    outcome = asyncio.run(
        checks.run_checks(room_id, "quick", "task/x", 30, exec_fn=exec_fn, tree=T1)
    )

    assert outcome["passed"] is True and outcome["ran"] is True
    assert calls == [["bash", "-lc", "echo tree-check"]]  # no `git checkout`
    # The shared tree has no such file, so without `tree` nothing would run.
    outcome = asyncio.run(checks.run_checks(room_id, "quick", None, 30, exec_fn=exec_fn))
    assert outcome["ran"] is False


# --- E4: dependencies shared with the main tree -----------------------------


def _seed_node(root: Path, lock: str = "v1\n") -> None:
    (root / "package.json").write_text('{"name": "x"}\n')
    (root / "package-lock.json").write_text(lock)
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "deps: seed")
    (root / "node_modules" / "left-pad").mkdir(parents=True)
    (root / "node_modules" / "left-pad" / "index.js").write_text("x\n")


@posix_only
def test_node_modules_is_linked_when_the_lock_matches(room):
    room_id, root = room
    _seed_node(root)

    info = git_ops.worktree_add(room_id, T1, f"task/{T1}")

    link = root / ".legion" / "wt" / T1 / "node_modules"
    assert info["linked"] == ["node_modules"]
    assert link.is_symlink() and not os.path.isabs(os.readlink(link))
    assert (link / "left-pad" / "index.js").read_text() == "x\n"  # resolves from the tree
    assert link.resolve() == (root / "node_modules").resolve()


@posix_only
def test_the_link_is_never_committed_or_shown(room):
    room_id, root = room
    _seed_node(root)
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    tree = root / ".legion" / "wt" / T1
    (tree / "new.txt").write_text("n\n")

    assert "node_modules" not in _g(tree, "status", "--porcelain")
    git_ops.git_commit(room_id, "task: new", tree=T1)
    assert _g(tree, "ls-tree", "-r", "--name-only", "HEAD").count("node_modules") == 0
    # removing the tree leaves the shared directory alone
    git_ops.worktree_remove(room_id, T1)
    assert (root / "node_modules" / "left-pad" / "index.js").read_text() == "x\n"


@posix_only
def test_a_different_lock_means_no_link(room):
    room_id, root = room
    _seed_node(root, "v1\n")
    # the main tree moved on after the commit the task starts from
    (root / "package-lock.json").write_text("v2\n")

    info = git_ops.worktree_add(room_id, T1, f"task/{T1}")

    assert info["linked"] == []
    assert not (root / ".legion" / "wt" / T1 / "node_modules").exists()


@posix_only
def test_no_lock_files_and_no_main_install_mean_no_link(room):
    room_id, root = room
    (root / "node_modules").mkdir()  # nothing pins it: no package.json at all
    assert git_ops.worktree_add(room_id, T1, f"task/{T1}")["linked"] == []
    assert git_ops.worktree_add(room_id, T2, f"task/{T2}")["linked"] == []


@posix_only
def test_subproject_dependencies_and_venv_are_linked(room):
    room_id, root = room
    (root / "frontend").mkdir()
    (root / "frontend" / "package.json").write_text("{}\n")
    (root / "frontend" / "yarn.lock").write_text("y\n")
    (root / "requirements.txt").write_text("flask\n")
    _g(root, "add", "-A")
    _g(root, "commit", "-m", "deps")
    (root / "frontend" / "node_modules").mkdir()
    (root / ".venv").mkdir()

    info = git_ops.worktree_add(room_id, T1, f"task/{T1}")

    assert sorted(info["linked"]) == [".venv", "frontend/node_modules"]
    tree = root / ".legion" / "wt" / T1
    assert (tree / "frontend" / "node_modules").resolve() == (root / "frontend" / "node_modules").resolve()


@posix_only
def test_resume_keeps_an_own_install_and_adds_a_missing_link_only(room):
    room_id, root = room
    _seed_node(root)
    git_ops.worktree_add(room_id, T1, f"task/{T1}")
    link = root / ".legion" / "wt" / T1 / "node_modules"
    link.unlink()
    link.mkdir()  # the task installed its own copy
    (link / "own.txt").write_text("own\n")

    info = git_ops.worktree_add(room_id, T1, f"task/{T1}")

    assert info["created"] is False and info["linked"] == []
    assert link.is_dir() and not link.is_symlink() and (link / "own.txt").exists()


# ---------------------------------------------------------------------------
# E5: merges and branch operations around task trees
# ---------------------------------------------------------------------------


def _advance_dev(root: Path, name: str, text: str) -> None:
    """Land a commit on ``dev`` from the shared tree (what a sibling task's merge does)."""
    _g(root, "checkout", "dev")
    (root / name).write_text(text)
    _g(root, "add", "-A")
    _g(root, "commit", "-m", f"dev: {name}")


def test_base_refresh_merges_inside_the_tree_and_leaves_shared_head(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}", base="dev")
    file_ops.write_file(room_id, "/mine.txt", "mine\n", commit_message="task: mine", tree=T1)
    _advance_dev(root, "sibling.txt", "sibling\n")
    shared_head = _branch(root)
    dev_tip = _g(root, "rev-parse", "dev")

    res = git_ops.git_merge_branch(
        room_id, "dev", target=f"task/{T1}", theirs=False, tree=T1,
        message="refresh",
    )

    tree = root / ".legion" / "wt" / T1
    assert res["ok"] is True and res["conflict"] is False and res["head_changed"] is True
    assert (tree / "sibling.txt").read_text() == "sibling\n"  # the base's work arrived in the tree
    assert (tree / "mine.txt").exists()
    assert _branch(root) == shared_head  # no checkout in the shared tree
    assert _g(root, "rev-parse", "dev") == dev_tip  # the base itself did not move
    assert _g(root, "rev-parse", f"task/{T1}") == res["sha"]


def test_a_conflict_in_the_tree_is_aborted_and_leaves_it_clean(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}", base="dev")
    file_ops.write_file(room_id, "/app.txt", "task side\n", commit_message="task: edit", tree=T1)
    _advance_dev(root, "app.txt", "dev side\n")
    before = _g(root, "rev-parse", f"task/{T1}")

    res = git_ops.git_merge_branch(
        room_id, "dev", target=f"task/{T1}", theirs=False, tree=T1,
    )

    tree = root / ".legion" / "wt" / T1
    assert res["ok"] is False and res["conflict"] is True
    assert _g(tree, "status", "--porcelain") == ""  # no half-merged state
    assert _g(root, "rev-parse", f"task/{T1}") == before
    assert (tree / "app.txt").read_text() == "task side\n"


def test_merge_in_a_tree_refuses_a_tree_on_another_branch(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}", base="dev")

    res = git_ops.git_merge_branch(room_id, "dev", target="task/other", tree=T1)

    assert res["ok"] is False and "task/other" in res["stderr"]


def test_pending_files_are_saved_before_the_tree_merges(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}", base="dev")
    tree = root / ".legion" / "wt" / T1
    (tree / "bash-made.txt").write_text("from bash\n")  # written without a commit
    _advance_dev(root, "sibling.txt", "sibling\n")

    res = git_ops.git_merge_branch(room_id, "dev", target=f"task/{T1}", theirs=False, tree=T1)

    assert res["ok"] is True
    assert _g(tree, "status", "--porcelain") == ""
    assert "bash-made.txt" in _g(root, "ls-tree", "-r", "--name-only", f"task/{T1}")


def test_a_missing_tree_is_a_404_not_a_shared_tree_merge(room):
    room_id, root = room
    with pytest.raises(TaskTreeError):
        git_ops.git_merge_branch(room_id, "dev", target=f"task/{T1}", tree=T1)


def test_folding_a_tree_branch_into_the_base_works_from_the_shared_tree(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}", base="dev")
    file_ops.write_file(room_id, "/mine.txt", "mine\n", commit_message="task: mine", tree=T1)
    git_ops.git_merge_branch(room_id, "dev", target=f"task/{T1}", theirs=False, tree=T1)

    res = git_ops.git_merge_branch(room_id, f"task/{T1}", target="dev", theirs=True)

    assert res["ok"] is True, res
    assert _g(root, "show", "dev:mine.txt") == "mine"
    # the tree still holds its branch; removing it afterwards leaves the branch merged
    assert git_ops.worktree_remove(room_id, T1)["ok"] is True
    assert _g(root, "branch", "--merged", "dev", "--list", f"task/{T1}")


def test_a_branch_held_by_a_tree_cannot_be_discarded_until_it_is_released(room):
    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}", base="dev")
    file_ops.write_file(room_id, "/mine.txt", "mine\n", commit_message="task: mine", tree=T1)

    held = git_ops.git_discard_branch(room_id, f"task/{T1}")
    assert held["ok"] is False and held["deleted"] is False
    assert _g(root, "rev-parse", "--verify", f"task/{T1}")  # still there

    assert git_ops.worktree_remove(room_id, T1)["ok"] is True
    freed = git_ops.git_discard_branch(room_id, f"task/{T1}")
    assert freed["ok"] is True and freed["deleted"] is True and freed["backup_ref"]


def test_a_room_archive_carries_the_pending_files_of_its_trees(room, tmp_path, monkeypatch):
    """The archive packs the whole volume, so nothing needs committing first."""
    import tarfile

    from app import room_archive

    room_id, root = room
    git_ops.worktree_add(room_id, T1, f"task/{T1}", base="dev")
    (root / ".legion" / "wt" / T1 / "unsaved.txt").write_text("pending\n")
    store = tmp_path / "store"
    monkeypatch.setattr(room_archive, "workspace_root", lambda rid: tmp_path / rid)
    monkeypatch.setattr(room_archive, "STAGING_DIR", str(tmp_path / "staging"))
    monkeypatch.setattr(room_archive, "_store", room_archive._DirStore(str(store)))

    assert room_archive.archive_room(room_id)["archived"] is True

    with tarfile.open(store / f"{room_id}.tar.gz") as tar:
        names = tar.getnames()
    assert f"./.legion/wt/{T1}/unsaved.txt" in names
    assert any(n.startswith(f"./.git/worktrees/") for n in names)
