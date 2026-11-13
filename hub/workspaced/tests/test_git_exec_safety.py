"""workspaced's git must not run what the room's agents planted in ``.git``.

The agent writes the room volume, ``.git`` included, and workspaced holds the
Docker socket. A hook, a config-driven program, a symlink inside ``.git`` or a
token in a URL would each turn a room into a way out of its sandbox
(app/git_exec.py).
"""
from __future__ import annotations

import base64
import os
import stat
import subprocess
from pathlib import Path

import pytest

from app import git_exec, git_ops

posix_only = pytest.mark.skipif(os.name != "posix", reason="needs POSIX shell scripts / symlinks")


@pytest.fixture
def room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path / "data")
    room_id = "safety-room"
    git_ops.git_init(room_id)
    return room_id, tmp_path / "data" / room_id


def _cfg(root: Path, key: str, value: str) -> None:
    subprocess.run(
        ["git", "-c", "safe.directory=*", "config", "--file", str(root / ".git" / "config"), key, value],
        check=True, capture_output=True,
    )


def _marker_script(tmp_path: Path, marker: Path) -> Path:
    script = tmp_path / "evil.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\ncat\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


# ---------------------------------------------------------------------------
# Programs the repo asks git to run
# ---------------------------------------------------------------------------

@posix_only
def test_room_hooks_never_run_under_workspaced(room, tmp_path):
    room_id, root = room
    marker = tmp_path / "hook-ran"
    for name in ("pre-commit", "post-commit", "post-checkout", "post-merge"):
        hook = root / ".git" / "hooks" / name
        hook.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
        hook.chmod(0o755)

    (root / "a.txt").write_text("a\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "work")
    assert git_ops.git_create_branch(room_id, "task/x", base="dev")
    assert not marker.exists()


@posix_only
def test_fsmonitor_and_editor_from_repo_config_are_overridden(room, tmp_path):
    room_id, root = room
    marker = tmp_path / "ran"
    script = _marker_script(tmp_path, marker)
    _cfg(root, "core.fsmonitor", str(script))
    _cfg(root, "core.editor", str(script))
    _cfg(root, "core.sshCommand", str(script))

    (root / "b.txt").write_text("b\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "work")
    assert not marker.exists()


@posix_only
def test_credential_helper_from_repo_config_is_not_called(room, tmp_path):
    _, root = room
    marker = tmp_path / "helper-ran"
    script = _marker_script(tmp_path, marker)
    _cfg(root, "credential.helper", f"!{script}")
    _cfg(root, "credential.https://example.com.helper", f"!{script}")

    r = git_exec.git_exec(
        ["credential", "fill"], root, input="protocol=https\nhost=example.com\n\n",
    )
    assert r.returncode != 0
    assert not marker.exists()


@posix_only
@pytest.mark.parametrize(
    "key, value",
    [
        ("filter.evil.clean", "{script}"),
        ("diff.evil.textconv", "{script}"),
        ("merge.evil.driver", "{script} %A"),
        ("url.https://attacker.example/.insteadOf", "https://github.com/"),
        ("http.https://github.com.proxy", "http://attacker.example:3128"),
        ("include.path", "/etc/gitconfig"),
        ("core.worktree", "/"),
        ("gpg.program", "{script}"),
        ("remote.origin.uploadpack", "{script}"),
        ("extensions.worktreeConfig", "true"),
    ],
)
def test_unsafe_config_keys_refuse_the_operation(room, tmp_path, key, value):
    room_id, root = room
    marker = tmp_path / "filter-ran"
    script = _marker_script(tmp_path, marker)
    (root / ".gitattributes").write_text("* filter=evil diff=evil merge=evil\n", encoding="utf-8")
    _cfg(root, key, value.format(script=script))

    (root / "c.txt").write_text("c\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "work") is None
    assert not marker.exists()
    r = git_exec.git_exec(["status", "--porcelain"], root)
    assert r.returncode == git_exec.UNSAFE_REPO_RETURNCODE
    assert key.split(".")[0] in r.stderr.lower()
    assert "git config --unset" in r.stderr


def test_everyday_agent_config_is_still_accepted(room):
    room_id, root = room
    for key, value in (
        ("user.name", "Agent"),
        ("pull.rebase", "false"),
        ("push.autoSetupRemote", "true"),
        ("credential.helper", "store"),
        ("http.postBuffer", "524288000"),
        ("core.editor", "vim"),
        ("core.hooksPath", ".husky"),
        ("remote.origin.url", "https://github.com/o/r.git"),
        ("remote.origin.fetch", "+refs/heads/*:refs/remotes/origin/*"),
        ("branch.dev.remote", "origin"),
        ("filter.lfs.clean", "git-lfs clean -- %f"),
        ("filter.lfs.smudge", "git-lfs smudge -- %f"),
        ("filter.lfs.process", "git-lfs filter-process"),
        ("filter.lfs.required", "true"),
    ):
        _cfg(root, key, value)
    assert git_exec.repo_safety_error(root) == ""
    (root / "d.txt").write_text("d\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "work")


def test_a_fixed_config_is_accepted_again(room):
    _, root = room
    _cfg(root, "url.https://attacker.example/.insteadOf", "https://github.com/")
    assert git_exec.repo_safety_error(root)
    subprocess.run(
        ["git", "-c", "safe.directory=*", "config", "--file", str(root / ".git" / "config"),
         "--remove-section", "url.https://attacker.example/"],
        check=True, capture_output=True,
    )
    assert git_exec.repo_safety_error(root) == ""


# ---------------------------------------------------------------------------
# .git structure
# ---------------------------------------------------------------------------

def test_commondir_and_alternates_are_refused(room, tmp_path):
    _, root = room
    (root / ".git" / "commondir").write_text("../../other/.git\n", encoding="utf-8")
    assert "commondir" in git_exec.repo_safety_error(root)
    (root / ".git" / "commondir").unlink()

    alternates = root / ".git" / "objects" / "info" / "alternates"
    alternates.parent.mkdir(parents=True, exist_ok=True)
    alternates.write_text(str(tmp_path / "other.git" / "objects") + "\n", encoding="utf-8")
    assert "alternates" in git_exec.repo_safety_error(root)


def test_a_dotgit_file_inside_the_volume_is_refused(room, tmp_path, monkeypatch):
    import app.config as config

    _, root = room
    monkeypatch.setattr(config, "WORKSPACE_DATA_DIR", str(tmp_path / "data"))
    other = tmp_path / "data" / "other-room"
    other.mkdir()
    (other / ".git").write_text(f"gitdir: {root / '.git'}\n", encoding="utf-8")
    assert "linked worktree" in git_exec.repo_safety_error(other)


@posix_only
def test_symlinks_inside_dotgit_are_refused(room, tmp_path):
    room_id, root = room
    target = tmp_path / "victim.py"
    target.write_text("ORIGINAL = True\n", encoding="utf-8")
    msg = root / ".git" / "COMMIT_EDITMSG"
    if msg.exists():
        msg.unlink()
    os.symlink(target, msg)

    (root / "e.txt").write_text("e\n", encoding="utf-8")
    assert git_ops.git_commit(room_id, "import os; os.system('id')") is None
    assert target.read_text(encoding="utf-8") == "ORIGINAL = True\n"
    assert "COMMIT_EDITMSG" in git_exec.repo_safety_error(root)


@posix_only
def test_dotgit_symlink_is_refused(tmp_path):
    real = tmp_path / "real"
    subprocess.run(["git", "init", "-q", str(real)], check=True)
    room = tmp_path / "room"
    room.mkdir()
    os.symlink(real / ".git", room / ".git")
    assert "symbolic link" in git_exec.repo_safety_error(room)


@posix_only
def test_workspaced_writes_into_dotgit_do_not_follow_links(room, tmp_path):
    _, root = room
    target = tmp_path / "victim.txt"
    target.write_text("untouched\n", encoding="utf-8")
    exclude = root / ".git" / "info" / "exclude"
    exclude.unlink()
    os.symlink(target, exclude)

    assert git_ops._write_local_excludes(root) is False
    assert target.read_text(encoding="utf-8") == "untouched\n"

    hook = root / ".git" / "hooks" / "pre-push"
    hook.unlink()
    os.symlink(target, hook)
    from app import release_guard

    release_guard.ensure_ref_guard(root)
    assert target.read_text(encoding="utf-8") == "untouched\n"


# ---------------------------------------------------------------------------
# Environment, token, timeout
# ---------------------------------------------------------------------------

def test_workspaced_secrets_do_not_reach_git(monkeypatch):
    monkeypatch.setenv("ROOM_ARCHIVE_S3_SECRET_KEY", "s3-secret")
    monkeypatch.setenv("WORKSPACED_TOKEN", "ws-secret")
    env = git_exec.base_env()
    assert "ROOM_ARCHIVE_S3_SECRET_KEY" not in env
    assert "WORKSPACED_TOKEN" not in env
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_token_goes_in_an_origin_scoped_header():
    env = git_exec.auth_env("https://github.com/o/r.git", "tok123")
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraHeader"
    basic = env["GIT_CONFIG_VALUE_0"].removeprefix("Authorization: Basic ")
    assert base64.b64decode(basic).decode() == "x-access-token:tok123"

    gl = git_exec.auth_env("https://gitlab.example:8443/g/r.git", "t")
    assert gl["GIT_CONFIG_KEY_0"] == "http.https://gitlab.example:8443/.extraHeader"
    assert base64.b64decode(gl["GIT_CONFIG_VALUE_0"].split()[-1]).decode() == "oauth2:t"

    assert git_exec.auth_env("https://github.com/o/r.git", None) == {}
    assert git_exec.auth_env("git@github.com:o/r.git", "t") == {}


def test_push_keeps_the_token_out_of_argv(room, monkeypatch):
    room_id, _ = room
    seen: list[tuple[list[str], dict]] = []
    real = git_ops.git_exec

    def spy(args, cwd, **kw):
        if args and args[0] == "push":
            seen.append((list(args), dict(kw.get("env") or {})))
            return subprocess.CompletedProcess(args, 0, "", "")
        return real(args, cwd, **kw)

    monkeypatch.setattr(git_ops, "git_exec", spy)
    r = git_ops.git_push(room_id, "https://github.com/o/r.git", branch="dev", token="SECRET-TOKEN")
    assert r["ok"]
    assert seen
    for args, env in seen:
        assert not any("SECRET-TOKEN" in a for a in args)
        assert "https://github.com/o/r.git" in args
        header = env["GIT_CONFIG_VALUE_0"]
        assert base64.b64decode(header.split()[-1]).decode() == "x-access-token:SECRET-TOKEN"


@posix_only
def test_a_hung_git_is_killed_at_the_timeout(room, tmp_path):
    _, root = room
    slow = tmp_path / "slow-askpass.sh"
    slow.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    slow.chmod(0o755)
    r = git_exec.git_exec(
        ["credential", "fill"], root,
        input="protocol=https\nhost=example.com\n\n",
        env={"GIT_ASKPASS": str(slow)}, timeout=1,
    )
    assert r.returncode == git_exec.TIMEOUT_RETURNCODE
    assert "timed out" in r.stderr


def test_git_failures_come_back_as_results_not_exceptions(tmp_path):
    r = git_exec.git_exec(["rev-parse", "HEAD"], tmp_path)
    assert r.returncode != 0
    assert isinstance(r.stderr, str)

