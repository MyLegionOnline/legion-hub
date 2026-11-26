"""bash inside a task's own worktree (internal design notes E3)."""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from app.core.task_trees import rewrite_cmd_for_tree, tree_cwd

ROOT = "/workspace/.legion/wt/abcd1234"


def test_rewrite_maps_workspace_but_keeps_legion():
    cmd = "cd /workspace && ls /workspace/src /workspace/.legion/logs /workspace/.legion"
    new, changed = rewrite_cmd_for_tree(cmd, "abcd1234")
    assert changed
    assert new == f"cd {ROOT} && ls {ROOT}/src /workspace/.legion/logs /workspace/.legion"


@pytest.mark.parametrize("cmd", [
    "ls /workspacex", "cat a/workspace/x", "echo ~/workspace", "ls",
    f"cd {ROOT} && ls",
])
def test_rewrite_leaves_other_text(cmd):
    assert rewrite_cmd_for_tree(cmd, "abcd1234") == (cmd, False)


def test_tree_cwd():
    assert tree_cwd("/workspace", "abcd1234") == ROOT
    assert tree_cwd("/workspace/web", "abcd1234") == f"{ROOT}/web"
    assert tree_cwd(f"{ROOT}/web", "abcd1234") == f"{ROOT}/web"
    assert tree_cwd("/workspace/.legion/bg", "abcd1234") == "/workspace/.legion/bg"


@pytest.mark.asyncio
async def test_bash_runs_in_task_tree(monkeypatch):
    from app.core.tools import _bash

    ws = SimpleNamespace(id=uuid.uuid4())

    async def _ensure(ctx, reason):
        return ws, None

    async def _env(self, ctx, secrets, environment):
        return {}, {}, [], None

    seen = {}

    async def _sync(self, ctx, ws_, cmd, cwd, timeout, sid, wsd, env, rs, sf):
        seen.update(cmd=cmd, cwd=cwd, env=env)
        return "ran"

    monkeypatch.setattr(_bash, "ensure_workspace_running", _ensure)
    monkeypatch.setattr(_bash.BashTool, "_build_exec_env", _env)
    monkeypatch.setattr(_bash.BashTool, "_run_sync", _sync)

    ctx = SimpleNamespace(
        room_id=uuid.uuid4(), session_id=uuid.uuid4(), task_id=None, db=None,
        tree="abcd1234", agent_id=None,
    )
    out = await _bash.BashTool().execute(ctx, {"cmd": "cd /workspace && ls"})
    assert seen["cwd"] == ROOT
    assert seen["cmd"] == f"cd {ROOT} && ls"
    assert seen["env"]["LEGION_TASK_TREE"] == ROOT
    assert out.startswith("ran") and "your task tree" in out

    # No rewrite, no note.
    out = await _bash.BashTool().execute(ctx, {"cmd": "ls"})
    assert out == "ran" and seen["cwd"] == ROOT

    # Old mode: no tree, shared /workspace untouched.
    ctx2 = SimpleNamespace(
        room_id=uuid.uuid4(), session_id=uuid.uuid4(), task_id=None, db=None, agent_id=None,
    )
    out = await _bash.BashTool().execute(ctx2, {"cmd": "cd /workspace && ls"})
    assert seen["cwd"] == "/workspace" and seen["cmd"] == "cd /workspace && ls"
    assert "LEGION_TASK_TREE" not in seen["env"] and out == "ran"


# --- E4: dependency links ---------------------------------------------------

import shutil
import subprocess

from app.core.task_trees import (
    DEP_UNLINK_MARK,
    dep_unlink_prelude,
    take_dep_unlink_note,
)

_TREE = "abcd1234"
_ROOT = f"/workspace/.legion/wt/{_TREE}"


@pytest.mark.parametrize("cmd", [
    "npm install", "npm i lodash", "cd app && npm ci", "pnpm add x", "yarn", "yarn add x",
    "sudo npm install -g y", "bun install",
])
def test_node_installs_get_a_prelude(cmd):
    assert "node_modules" in dep_unlink_prelude(cmd, _ROOT, _TREE)


@pytest.mark.parametrize("cmd", [
    "pip install x", "python3 -m pip install x", "uv sync", "poetry add y", "python -m venv .venv",
])
def test_python_installs_get_a_prelude(cmd):
    assert ".venv" in dep_unlink_prelude(cmd, _ROOT, _TREE)


@pytest.mark.parametrize("cmd", [
    "npm run build", "npm test", "yarn build", "ls node_modules", "git status", "echo npm install",
])
def test_other_commands_get_none(cmd):
    assert dep_unlink_prelude(cmd, _ROOT, _TREE) == ""


def test_prelude_covers_cwd_root_and_cd_targets_inside_the_tree_only():
    p = dep_unlink_prelude("cd frontend && npm i; cd /etc && npm i; cd ~ && npm i", _ROOT + "/x", _TREE)
    assert f"'{_ROOT}'" in p and f"'{_ROOT}/x'" in p and f"'{_ROOT}/x/frontend'" in p
    assert "/etc" not in p and "~" not in p


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_prelude_removes_only_links_and_reports_them(tmp_path):
    shared = tmp_path / "shared_nm"
    shared.mkdir()
    (shared / "pkg").write_text("x")
    tree = tmp_path / "wt" / _TREE
    tree.mkdir(parents=True)
    (tree / "node_modules").symlink_to(shared, target_is_directory=True)
    real = tree / "sub"
    (real / "node_modules").mkdir(parents=True)
    pre = dep_unlink_prelude("npm i; cd sub && npm i", str(tree), _TREE)
    # the prelude only accepts the real container path; rebuild it for the temp tree
    pre = pre.replace(_ROOT, str(tree))
    pre = pre.replace(f"'{tree}/x'", "")
    out = subprocess.run(["sh", "-c", pre + "echo done"], capture_output=True, text=True).stdout
    assert not (tree / "node_modules").exists() and not (tree / "node_modules").is_symlink()
    assert (shared / "pkg").exists()  # the shared copy is untouched
    assert (real / "node_modules").is_dir()  # a real directory is never removed
    cleaned, note = take_dep_unlink_note(out)
    assert DEP_UNLINK_MARK not in cleaned and cleaned.strip() == "done"
    assert "node_modules" in note and "shared" in note


def test_note_is_empty_without_markers():
    assert take_dep_unlink_note("hello\n") == ("hello\n", "")
