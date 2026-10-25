"""`.gitignore` model — Stages 0, 1, 2 & 3.

internal design notes. Before this plan ``git_init`` committed a ``.gitignore``
into the synthetic root commit, which wedged "Merge histories" on an ``add/add``
conflict and leaked a synthetic root into adopted repos.

* Stage 0 split the seed into two disjoint constants (``_PROJECT_GITIGNORE_DEFAULT``
  — safe to commit and hand out; ``_LEGION_LOCAL_EXCLUDES`` — infra, never
  committed).
* Stage 1 moves the seed out of the commit: ``git_init`` now produces an *empty*
  root commit and writes ``_LOCAL_EXCLUDE_SEED`` to ``.git/info/exclude`` so the
  agent's ``git add -A`` / ``git status`` stay clean without committing anything.
* Stage 2 puts a real, shared ``.gitignore`` back — but only as an explicit
  commit on ``dev`` the first time Legion itself publishes a greenfield repo, so
  a human who clones it gets the ordinary project ignores.
* Stage 3 backfills ``.git/info/exclude`` (``backfill_local_excludes``) for
  rooms whose ``git_init`` predates Stage 1, without rewriting any history.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from app import git_ops

_CFG = ["-c", "core.autocrlf=false"]


def _g(root: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(
        ["git", *_CFG, "-C", str(root), *args],
        capture_output=True, text=True, check=check,
    )
    return r.stdout.strip()


@pytest.fixture
def isolated_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    room_id = "gitignore-model-room"
    git_ops.git_init(room_id)  # empty commit on main, HEAD → dev
    return room_id, tmp_path / room_id


# ---------------------------------------------------------------------------
# 0.2 — the constants
# ---------------------------------------------------------------------------

def test_project_default_has_no_legion_infrastructure():
    assert ".legion/git.lock" not in git_ops._PROJECT_GITIGNORE_DEFAULT
    assert "node_modules/" in git_ops._PROJECT_GITIGNORE_DEFAULT
    assert git_ops._PROJECT_GITIGNORE_DEFAULT.endswith("\n")


def test_legion_local_excludes_is_only_infrastructure():
    lines = git_ops._LEGION_LOCAL_EXCLUDES.splitlines()
    # Legion's own per-volume infrastructure only — never project files.
    assert ".userfiles/" in lines
    assert ".legion/git.lock" in lines
    assert ".legion/generation.json" in lines
    assert ".legion/runs/" in lines
    # `.legion/run.json` is a real, shared CI config and stays tracked.
    assert ".legion/run.json" not in lines
    assert "node_modules/" not in lines


def test_local_exclude_seed_is_the_union():
    assert git_ops._LOCAL_EXCLUDE_SEED == (
        git_ops._PROJECT_GITIGNORE_DEFAULT + git_ops._LEGION_LOCAL_EXCLUDES
    )


def test_old_constant_name_is_gone():
    assert not hasattr(git_ops, "_GITIGNORE_DEFAULT")


def test_known_seed_gitignores_cover_current_and_previous_seed():
    # Historical *committed* `.gitignore` seeds (pre-Stage-1 rooms). Stage 1 no
    # longer commits any seed, and the local-exclude seed now also carries
    # volatile `.legion/` paths that were never committed — so it is
    # intentionally not required to be a member here.
    # v1 predates the git lock line.
    assert git_ops._PROJECT_GITIGNORE_DEFAULT in git_ops._KNOWN_SEED_GITIGNORES
    # v2 adds only the git lock line (never the volatile per-volume paths).
    assert (
        git_ops._PROJECT_GITIGNORE_DEFAULT + ".legion/git.lock\n"
    ) in git_ops._KNOWN_SEED_GITIGNORES
    assert len(git_ops._KNOWN_SEED_GITIGNORES) == len(set(git_ops._KNOWN_SEED_GITIGNORES))


# ---------------------------------------------------------------------------
# 1.1 — git_init: empty root commit + local excludes only
# ---------------------------------------------------------------------------

def test_git_init_main_tree_is_empty(isolated_repo):
    _room_id, root = isolated_repo

    assert _g(root, "rev-list", "--count", "main") == "1"
    tree = [p for p in _g(root, "ls-tree", "-r", "--name-only", "main").splitlines() if p]
    assert tree == []


def test_git_init_writes_the_seed_to_info_exclude(isolated_repo):
    _room_id, root = isolated_repo

    body = (root / ".git" / "info" / "exclude").read_text(encoding="utf-8")
    for line in git_ops._LOCAL_EXCLUDE_SEED.splitlines():
        assert line in body.splitlines()
    # ...and nothing was committed.
    assert _g(root, "ls-tree", "-r", "--name-only", "main") == ""


def test_git_init_creates_private_directory(isolated_repo):
    _room_id, root = isolated_repo

    assert (root / ".userfiles").is_dir()
    assert (root / ".git" / "info" / "exclude").read_text(encoding="utf-8").splitlines().count(
        ".userfiles/"
    ) == 1


def test_write_local_excludes_repairs_duplicate_private_rules(isolated_repo):
    _room_id, root = isolated_repo
    exclude = root / ".git" / "info" / "exclude"
    exclude.write_text(".userfiles/\n.userfiles/\n", encoding="utf-8")

    assert git_ops._write_local_excludes(root) is True
    assert exclude.read_text(encoding="utf-8").splitlines().count(".userfiles/") == 1


def test_git_init_head_is_dev(isolated_repo):
    _room_id, root = isolated_repo
    assert _g(root, "rev-parse", "--abbrev-ref", "HEAD") == "dev"


def test_git_init_no_synthetic_gitignore_commit(isolated_repo):
    _room_id, root = isolated_repo
    assert "init: Legion Hub workspace" in _g(root, "log", "--format=%s", "main")
    assert _g(root, "log", "--format=%s", "main").count("\n") == 0  # exactly one commit


# ---------------------------------------------------------------------------
# 1.2 — working-tree hygiene before publication
# ---------------------------------------------------------------------------

def test_ignored_junk_and_lock_file_keep_status_clean(isolated_repo):
    _room_id, root = isolated_repo

    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("junk\n", encoding="utf-8")
    (root / ".legion").mkdir(exist_ok=True)
    (root / ".legion" / "git.lock").write_text("123 2026-08-30T00:00:00+00:00\n", encoding="utf-8")

    assert _g(root, "status", "--porcelain") == ""


def test_write_local_excludes_is_idempotent(isolated_repo):
    _room_id, root = isolated_repo
    exclude = root / ".git" / "info" / "exclude"

    before = exclude.read_text(encoding="utf-8")
    git_ops._write_local_excludes(root)
    git_ops._write_local_excludes(root)
    assert exclude.read_text(encoding="utf-8") == before


def test_write_local_excludes_extra_lines_deduped(isolated_repo):
    _room_id, root = isolated_repo
    exclude = root / ".git" / "info" / "exclude"

    git_ops._write_local_excludes(root, "secrets.txt\n")
    git_ops._write_local_excludes(root, "secrets.txt\n")
    body = exclude.read_text(encoding="utf-8")
    assert body.splitlines().count("secrets.txt") == 1


def test_write_local_excludes_preserves_existing_entries(isolated_repo):
    _room_id, root = isolated_repo
    exclude = root / ".git" / "info" / "exclude"
    exclude.write_text("# hand-written\nmy-own-rule/\n", encoding="utf-8")

    git_ops._write_local_excludes(root)
    body = exclude.read_text(encoding="utf-8").splitlines()
    assert "my-own-rule/" in body
    assert "node_modules/" in body
    assert ".legion/git.lock" in body


# ---------------------------------------------------------------------------
# 1.1 / 1.4 — a pristine room adopts an external repo with zero conflicts
# ---------------------------------------------------------------------------

def _seeded_remote(tmp_path: Path, branch: str = "main", commits: int = 2) -> Path:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], capture_output=True, check=True)
    clone = tmp_path / "seed-clone"
    subprocess.run(["git", "clone", str(remote), str(clone)], capture_output=True, check=True)
    _g(clone, "config", "user.name", "Remote Author")
    _g(clone, "config", "user.email", "remote@example.com")
    # Any real repo carries its own .gitignore — the exact file that used to
    # collide with Legion's committed seed.
    (clone / ".gitignore").write_text("dist/\ncoverage/\n", encoding="utf-8")
    for i in range(commits):
        (clone / f"file{i}.txt").write_text(f"remote content {i}\n", encoding="utf-8")
        _g(clone, "add", "-A")
        _g(clone, "commit", "-m", f"remote commit {i}")
    _g(clone, "push", "origin", f"HEAD:{branch}")
    return remote


def test_pristine_room_adopts_external_repo_without_conflict(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path, commits=3)
    remote_tip = _g(tmp_path / "seed-clone", "rev-parse", "HEAD")

    result = git_ops.git_adopt_remote(
        room_id, f"file://{remote}", branch="main", working_branch="dev"
    )

    assert result["ok"] and result["adopted"] and not result["diverged"]
    assert result["dropped_init"] is True
    assert _g(root, "rev-parse", "main") == remote_tip
    assert _g(root, "rev-parse", "dev") == remote_tip
    # The remote's own .gitignore is the only one, verbatim.
    assert _g(root, "show", "main:.gitignore") == "dist/\ncoverage/"
    assert "init: Legion Hub workspace" not in _g(root, "log", "--format=%s", "main")


# ---------------------------------------------------------------------------
# 2.1 — ensure_shared_gitignore: seed the shared file once, on publication
# ---------------------------------------------------------------------------

def test_ensure_shared_gitignore_commits_project_default_on_dev(isolated_repo):
    room_id, root = isolated_repo

    result = git_ops.ensure_shared_gitignore(room_id, "dev")

    assert result["ok"] and result["added"] and result["branch"] == "dev"
    assert result["sha"] == _g(root, "rev-parse", "dev")
    assert _g(root, "show", "dev:.gitignore") == git_ops._PROJECT_GITIGNORE_DEFAULT.strip()
    assert _g(root, "log", "--format=%s", "dev").splitlines()[0] == "chore: add default .gitignore"
    # infra line is never committed — it belongs in .git/info/exclude only
    assert ".legion/git.lock" not in _g(root, "show", "dev:.gitignore")
    # HEAD stays on dev (it started there)
    assert _g(root, "rev-parse", "--abbrev-ref", "HEAD") == "dev"


def test_ensure_shared_gitignore_is_idempotent(isolated_repo):
    room_id, root = isolated_repo
    git_ops.ensure_shared_gitignore(room_id, "dev")
    tip = _g(root, "rev-parse", "dev")

    again = git_ops.ensure_shared_gitignore(room_id, "dev")

    assert again["ok"] and not again["added"] and again["sha"] is None
    assert _g(root, "rev-parse", "dev") == tip  # no second commit


def test_ensure_shared_gitignore_keeps_agent_authored_file(isolated_repo):
    room_id, root = isolated_repo
    (root / ".gitignore").write_text("*.tmp\nsecret/\n", encoding="utf-8")

    result = git_ops.ensure_shared_gitignore(room_id, "dev")

    assert result["added"]
    assert _g(root, "show", "dev:.gitignore") == "*.tmp\nsecret/"


def test_ensure_shared_gitignore_restores_prior_head(isolated_repo):
    room_id, root = isolated_repo
    _g(root, "checkout", "main")

    git_ops.ensure_shared_gitignore(room_id, "dev")

    assert _g(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    # committed on dev, not on main
    assert _g(root, "ls-tree", "-r", "--name-only", "main") == ""
    assert _g(root, "show", "dev:.gitignore") == git_ops._PROJECT_GITIGNORE_DEFAULT.strip()


def test_ensure_shared_gitignore_flows_to_main_on_promotion_merge(isolated_repo):
    """The seed lands on `dev`; a clean `dev → main` merge carries it to `main`,
    which is what the promotion push publishes."""
    room_id, root = isolated_repo
    git_ops.ensure_shared_gitignore(room_id, "dev")

    git_ops.git_merge_branch(room_id, "dev", target="main", theirs=False)

    assert _g(root, "show", "main:.gitignore") == git_ops._PROJECT_GITIGNORE_DEFAULT.strip()


def test_ensure_shared_gitignore_missing_branch_is_reported(isolated_repo):
    room_id, _root = isolated_repo
    result = git_ops.ensure_shared_gitignore(room_id, "nope")
    assert result["ok"] is False and result["added"] is False and result["stderr"]


# ---------------------------------------------------------------------------
# 3.1 — backfill_local_excludes: catch up a pre-Stage-1 repo
# ---------------------------------------------------------------------------

def _strip_excludes(root: Path) -> None:
    """Simulate a repo whose ``git_init`` predates Stage 1: drop every line the
    local-exclude seed would have added, keeping any hand-written content."""
    exclude = root / ".git" / "info" / "exclude"
    seeded = set(git_ops._LOCAL_EXCLUDE_SEED.splitlines())
    kept = [
        line
        for line in exclude.read_text(encoding="utf-8").splitlines()
        if line not in seeded
    ]
    exclude.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def test_backfill_local_excludes_catches_up_old_repo(isolated_repo):
    room_id, root = isolated_repo
    _strip_excludes(root)

    result = git_ops.backfill_local_excludes(room_id)

    assert result == {"ok": True, "updated": True, "userfiles_protected": True}
    body = (root / ".git" / "info" / "exclude").read_text(encoding="utf-8").splitlines()
    for line in git_ops._LOCAL_EXCLUDE_SEED.splitlines():
        assert line in body


def test_backfill_local_excludes_keeps_status_clean(isolated_repo):
    room_id, root = isolated_repo
    _strip_excludes(root)
    git_ops.backfill_local_excludes(room_id)

    (root / "node_modules").mkdir()
    (root / "node_modules" / "x.js").write_text("junk\n", encoding="utf-8")
    (root / ".legion").mkdir(exist_ok=True)
    (root / ".legion" / "git.lock").write_text("1 x\n", encoding="utf-8")

    assert _g(root, "status", "--porcelain") == ""


def test_backfill_local_excludes_is_idempotent_noop_on_fresh_repo(isolated_repo):
    room_id, _root = isolated_repo
    # git_init already wrote the seed — nothing to do.
    assert git_ops.backfill_local_excludes(room_id) == {
        "ok": True,
        "updated": False,
        "userfiles_protected": True,
    }


def test_backfill_local_excludes_does_not_touch_history(isolated_repo):
    room_id, root = isolated_repo
    before_main = _g(root, "rev-parse", "main")
    before_dev = _g(root, "rev-parse", "dev")
    _strip_excludes(root)

    git_ops.backfill_local_excludes(room_id)

    assert _g(root, "rev-parse", "main") == before_main
    assert _g(root, "rev-parse", "dev") == before_dev
    assert _g(root, "rev-parse", "--abbrev-ref", "HEAD") == "dev"


def test_backfill_local_excludes_preserves_hand_written_lines(isolated_repo):
    room_id, root = isolated_repo
    exclude = root / ".git" / "info" / "exclude"
    exclude.write_text("# mine\ncustom-rule/\n", encoding="utf-8")

    git_ops.backfill_local_excludes(room_id)

    body = exclude.read_text(encoding="utf-8").splitlines()
    assert "custom-rule/" in body
    assert ".legion/git.lock" in body


def test_backfill_local_excludes_no_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(git_ops, "WORKSPACE_DATA_DIR", tmp_path)
    result = git_ops.backfill_local_excludes("never-inited")
    assert result["ok"] is False and result["updated"] is False


# ---------------------------------------------------------------------------
# 4.1 — "Merge histories" auto-resolves an untouched seed .gitignore
# ---------------------------------------------------------------------------

def _make_pre_stage1_room(root: Path, body: str | None = None) -> None:
    """Reproduce a room whose ``git_init`` (pre-Stage-1) committed the seed
    ``.gitignore`` into the synthetic root commit, inherited by ``dev``."""
    body = body or git_ops._PROJECT_GITIGNORE_DEFAULT
    _g(root, "checkout", "main")
    (root / ".gitignore").write_text(body, encoding="utf-8")
    _g(root, "add", ".gitignore")
    _g(root, "commit", "--amend", "--no-edit")
    _g(root, "branch", "-f", "dev", "main")
    _g(root, "checkout", "dev")


def test_merge_histories_autoresolves_untouched_seed_gitignore(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    _make_pre_stage1_room(root)
    remote = _seeded_remote(tmp_path, commits=2)  # carries its own dist/\ncoverage/

    result = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )

    assert result["ok"] is True, result["stderr"]
    assert result["conflict"] is False
    assert result["gitignore_autoresolved"] is True
    # the remote's file won, verbatim; the merge actually completed
    assert _g(root, "show", "dev:.gitignore") == "dist/\ncoverage/"
    assert _g(root, "show", "dev:file0.txt") == "remote content 0"
    assert _g(root, "status", "--porcelain") == ""


@pytest.mark.parametrize("seed", git_ops._KNOWN_SEED_GITIGNORES)
def test_merge_histories_autoresolves_every_known_seed(isolated_repo, tmp_path, seed):
    room_id, root = isolated_repo
    _make_pre_stage1_room(root, body=seed)
    remote = _seeded_remote(tmp_path, commits=1)

    result = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )

    assert result["ok"] is True and result["gitignore_autoresolved"] is True
    assert _g(root, "show", "dev:.gitignore") == "dist/\ncoverage/"


def test_merge_histories_edited_gitignore_still_conflicts(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    _make_pre_stage1_room(root)
    # the agent tuned the ignores — no longer a pristine seed
    (root / ".gitignore").write_text("*.custom\nsecret/\n", encoding="utf-8")
    _g(root, "add", ".gitignore")
    _g(root, "commit", "-m", "chore: tune ignores")
    remote = _seeded_remote(tmp_path, commits=1)
    before = _g(root, "rev-parse", "dev")

    result = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )

    assert result["ok"] is False
    assert result["conflict"] is True
    assert result.get("gitignore_autoresolved", False) is False
    # rolled back to a clean, usable tree
    assert _g(root, "status", "--porcelain") == ""
    assert _g(root, "rev-parse", "dev") == before
    assert _g(root, "show", "dev:.gitignore") == "*.custom\nsecret/"


def test_merge_histories_clean_merge_leaves_flag_false(isolated_repo, tmp_path):
    """A Stage-1 room has no committed ``.gitignore`` — the remote's file just
    gets added, no conflict, and the flag is absent/False."""
    room_id, root = isolated_repo
    remote = _seeded_remote(tmp_path, commits=2)

    result = git_ops.git_fetch(
        room_id, f"file://{remote}", branch="main", token=None, target_branch="dev"
    )

    assert result["ok"] is True and result["conflict"] is False
    assert result.get("gitignore_autoresolved", False) is False
    assert _g(root, "show", "dev:.gitignore") == "dist/\ncoverage/"


def test_seed_gitignore_autoresolvable_predicate(isolated_repo, tmp_path):
    room_id, root = isolated_repo
    _make_pre_stage1_room(root)
    remote = _seeded_remote(tmp_path, commits=1)
    # drive the merge to the conflicted state by hand, then check the predicate
    _g(root, "fetch", f"file://{remote}", "main:FETCH_HEAD")
    _g(
        root, "merge", "--no-ff", "--no-edit", "--allow-unrelated-histories",
        "-m", "m", "FETCH_HEAD", check=False,
    )
    assert git_ops._seed_gitignore_autoresolvable(root) is True
    _g(root, "merge", "--abort")
