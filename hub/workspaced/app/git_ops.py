"""Git operations on workspace volumes.

All functions are synchronous and operate directly on the host filesystem path
WORKSPACE_DATA_DIR/<room_id>/.git — no running container required.

Token-bearing URLs are never logged: callers must strip tokens before logging.
"""
from __future__ import annotations

import functools
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Optional

from app.config import LEGION_GIT_EMAIL, LEGION_GIT_NAME, WORKSPACE_DATA_DIR
from app.file_ops import chown_workspace_tree, ensure_userfiles_dir
from app.git_exec import auth_env, git_exec, hand_to_git_user, read_repo_file, write_repo_file
from app.git_lock import GitWorkspaceBusyError, room_git_lock
from app.path_contract import USERFILES_DIR, normalize_workspace_path

logger = logging.getLogger("workspaced.git")


def _serialized(*, busy_dict: bool):
    """Run the wrapped git op under the room's working-tree lock.

    internal design notes §5 — every operation that moves HEAD or writes
    to the shared tree is serialised so a concurrent op in the same room cannot
    switch its branch mid-run. ``room_id`` must be the first positional arg.

    On lock timeout: ``busy_dict=True`` ops return ``{ok: False, busy: True}``
    (they already return a dict); the rest re-raise ``GitWorkspaceBusyError``
    for the endpoint / caller to surface — never a silent skip.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(room_id, *args, **kwargs):
            try:
                with room_git_lock(room_id):
                    return fn(room_id, *args, **kwargs)
            except GitWorkspaceBusyError as exc:
                if busy_dict:
                    return {
                        "ok": False,
                        "busy": True,
                        "conflict": False,
                        "stdout": "",
                        "stderr": str(exc),
                    }
                raise

        return wrapper

    return decorator

# internal design notes Stage 0 — branch model foundation.
#   `main` is the release branch and only moves through an explicit human
#   promotion (`dev → main`). `dev` is the room's integration branch: every
#   agent / auto commit lands here (or on a task branch that merges here).
RELEASE_BRANCH = "main"
# Default name of the integration branch. A room may use another name
# (internal design notes): the Hub's
# ``git_config.working_branch`` is the source of truth and workspaced keeps a
# copy in the repo's local config (``legion.workingBranch``) so the paths that
# do not get the name from the Hub — parking a dirty ``main``, stepping off a
# discarded branch — land on the right branch.
INTEGRATION_BRANCH = "dev"
_WORKING_BRANCH_KEY = "legion.workingBranch"
# Prefixes Legion names its own short-lived branches with.
_RESERVED_BRANCH_PREFIXES = ("task/", "goal/", "session/")
_REMOTE_REF_PREFIX = "refs/legion/remote/"
_BRANCH_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,99}$")


def valid_working_branch_name(name: Optional[str]) -> bool:
    """Whether ``name`` may be a room's integration branch.

    ASCII only, a valid ``refs/heads/`` name (``git check-ref-format`` rules,
    checked without a repo), not ``main`` / ``HEAD`` and not one of the
    prefixes Legion names task, goal and session branches with.
    """
    if not name or not isinstance(name, str) or not _BRANCH_NAME_RE.match(name):
        return False
    if name in (RELEASE_BRANCH, "HEAD") or name.startswith(_RESERVED_BRANCH_PREFIXES):
        return False
    if ".." in name or "//" in name or "@{" in name:
        return False
    for part in name.split("/"):
        if not part or part.startswith(".") or part.endswith(".lock"):
            return False
    return not name.endswith((".", "/"))


def working_branch_of(root: Path) -> str:
    """The room's integration branch as recorded in its repo (default ``dev``).

    A missing or unusable value — an old repo, or an agent that edited
    ``.git/config`` — falls back to the default instead of failing.
    """
    if not (root / ".git").is_dir():
        return INTEGRATION_BRANCH
    value = _run(["config", "--local", "--get", _WORKING_BRANCH_KEY], cwd=root).stdout.strip()
    return value if valid_working_branch_name(value) else INTEGRATION_BRANCH


def _record_working_branch(root: Path, name: str) -> None:
    """Store ``name`` as the repo's integration branch (local config only)."""
    current = _run(["config", "--local", "--get", _WORKING_BRANCH_KEY], cwd=root).stdout.strip()
    if current != name:
        _run(["config", "--local", _WORKING_BRANCH_KEY, name], cwd=root)


def _working_branch_arg(root: Path, working_branch: Optional[str]) -> str:
    """``working_branch`` when the caller passed a usable one, else the recorded one."""
    if working_branch and valid_working_branch_name(working_branch):
        return working_branch
    return working_branch_of(root)


class MainBranchProtectedError(RuntimeError):
    """Raised when an agent or auto commit targets `main` directly.

    internal design notes §0.1 invariant: neither an agent nor an auto-merge
    ever writes to `main`. Only an explicit human action (revert / restore,
    or the `dev → main` promotion gate) passes ``allow_main_commit=True``.
    """

# Flags applied to every git invocation to avoid dubious-ownership errors and
# ensure consistent defaults regardless of the system gitconfig.
_BASE_FLAGS = [
    "-c", "safe.directory=*",
    "-c", "init.defaultBranch=main",
    "-c", "core.autocrlf=false",
]

# Overlay on ``git_exec``'s minimal environment (never ``os.environ``: git and
# anything it might start must not see workspaced's own secrets).
_GIT_ENV = {
    "GIT_AUTHOR_NAME": LEGION_GIT_NAME,
    "GIT_AUTHOR_EMAIL": LEGION_GIT_EMAIL,
    "GIT_COMMITTER_NAME": LEGION_GIT_NAME,
    "GIT_COMMITTER_EMAIL": LEGION_GIT_EMAIL,
    # workspaced is the one party allowed to move ``main`` and tags; the
    # room's guard hooks (app/release_guard.py) let this git through.
    "LEGION_GIT_TRUSTED": "1",
}

# The person a room's commits are made for. Stored in the repo's own
# ``.git/config`` (never committed) by ``git_set_author``; when present it
# becomes the author of every commit Legion creates in that room, Legion stays
# the committer and is added as ``Co-Authored-By``. Without it Legion authors.
_AUTHOR_NAME_KEY = "legion.authorName"
_AUTHOR_EMAIL_KEY = "legion.authorEmail"
_COAUTHOR_TRAILER = f"Co-Authored-By: {LEGION_GIT_NAME} <{LEGION_GIT_EMAIL}>"
# Subcommands that can write a commit object.
_COMMIT_WRITING = frozenset({"commit", "merge", "revert"})

# internal design notes Stage 0 — two disjoint rule sets:
#
#   * ``_PROJECT_GITIGNORE_DEFAULT`` — ordinary project ignores. Safe to commit
#     and hand to anyone who clones the repo (Stage 2 seeds it as a real
#     ``.gitignore`` commit once Legion itself publishes a repo).
#   * ``_LEGION_LOCAL_EXCLUDES`` — Legion's own infrastructure. Never committed;
#     only ever written to ``.git/info/exclude`` inside the room clone.
_PROJECT_GITIGNORE_DEFAULT = (
    "node_modules/\n__pycache__/\n*.pyc\n.venv/\nvenv/\n"
    ".env\n*.log\ndist/\nbuild/\n.DS_Store\n"
)
# internal design notes §5 — the working-tree serialisation lock, plus
# the per-volume metadata ``workspaced`` rewrites on every mutation
# (``generation.json``) and its disposable run logs (``runs/``). None of it may
# ever enter git history: ``generation.json`` in particular changes on every
# single operation, so a tracked copy leaves the working tree permanently dirty
# and aborts the ``git checkout`` that every promotion / gate merge performs.
# ``.legion/run.json`` is deliberately *not* here — it is a real, shared CI
# config and stays tracked.
_LEGION_LOCAL_EXCLUDES = (
    f"{USERFILES_DIR}/\n"
    ".legion/git.lock\n"
    ".legion/generation.json\n"
    ".legion/runs/\n"
    # Background ``bash`` job logs and their ``.exit`` markers. An agent's
    # ``git add -A`` used to commit them; each room run writes new ones, so
    # ``dev`` and ``main`` collected diverging copies and the ``dev → main``
    # gate hit add/add conflicts on pure noise.
    ".legion/bg/\n"
    # internal design notes Stage 0 — task-formation dialogue archives
    # and drafts. Legion's own infrastructure: written into the room clone only,
    # never committed. If they became tracked they would vanish on branch switch
    # and an agent's ``git add -A`` would push them into the client's repo.
    ".legion/plans/\n"
    ".legion/drafts/\n"
    # Everything else under ``.legion/`` is room-local too: venvs
    # (``environments/``), materialized skills, ``bash`` output logs, media
    # caches, and whatever an agent stashes there to survive a restart — e.g. a
    # hand-extracted ``libroot/`` of ``.deb`` packages. ``.legion/`` is
    # therefore an allowlist: only the shared configs below are ever committed.
    # The negations must follow ``.legion/*``; ``_write_local_excludes`` appends
    # missing lines in this order.
    ".legion/*\n"
    + "".join(f"!{p}\n" for p in ("/.legion/run.json", "/.legion/services.json"))
)

# ``.legion/`` files that belong to the project and stay tracked: the CI config
# (``run.json``) and the Browser services manifest (``services.json``).
_TRACKED_LEGION_FILES = (".legion/run.json", ".legion/services.json")
# Until the first outgoing push there is no shared ``.gitignore`` yet, so the
# room clone's ``.git/info/exclude`` carries the union — that keeps the agent's
# ``git add -A`` / ``git status`` clean without committing anything.
_LOCAL_EXCLUDE_SEED = _PROJECT_GITIGNORE_DEFAULT + _LEGION_LOCAL_EXCLUDES

# Every ``.gitignore`` body Legion has ever seeded into a room's initial commit.
# Stage 4 uses this to auto-resolve the ``add/add`` conflict when an external
# repo is merged in: a seed file that nobody touched yields to the remote's.
_KNOWN_SEED_GITIGNORES = (
    # v1 (predates internal design notes §5, commits a84757c…a7d2ced)
    "node_modules/\n__pycache__/\n*.pyc\n.venv/\nvenv/\n.env\n*.log\ndist/\nbuild/\n.DS_Store\n",
    # v2 (followup §5 — adds the git lock)
    "node_modules/\n__pycache__/\n*.pyc\n.venv/\nvenv/\n.env\n*.log\ndist/\nbuild/\n.DS_Store\n.legion/git.lock\n",
)

# Paths under ``.legion/`` that a room created before they joined
# ``_LEGION_LOCAL_EXCLUDES`` may still *track* — and ``.git/info/exclude`` has no
# effect on an already-tracked file. These are actively dropped from the index
# whenever a branch switch runs (``_park_worktree_before_switch`` and the
# ``git_merge_branch`` / ``git_revert_commit`` gate paths), so a legacy room
# heals itself the first time it promotes. ``.legion/run.json`` is absent on
# purpose: it is a shared CI config and must stay tracked.
_VOLATILE_LEGION_PATHS = (".legion/generation.json", ".legion/runs")

# Legion paths that must leave the index but whose on-disk copy must survive a
# branch switch: background job logs are still read by ``bash_output`` /
# ``bash_wait`` (a running job keeps writing to its log), so unlike
# ``_VOLATILE_LEGION_PATHS`` they are never deleted from the working tree. Once
# untracked they are covered by ``.git/info/exclude``, and git treats ignored
# files as expendable on checkout, so they cannot block the switch either.
#
# Since ``.legion/`` became an allowlist this is the whole directory minus
# ``_TRACKED_LEGION_FILES`` — a room that already committed, say, an agent's
# ``.legion/libroot/`` drops it from the index on its next commit or switch.
_UNTRACK_LEGION_PATHS = (
    ".legion",
    *(f":(exclude){p}" for p in _TRACKED_LEGION_FILES),
)


def _workspace_root(room_id: str) -> Path:
    return Path(WORKSPACE_DATA_DIR) / room_id


def _write_local_excludes(root: Path, extra: str = "") -> bool:
    """Idempotently add Legion's local ignore rules to ``.git/info/exclude``.

    internal design notes Stage 1: before a repo is ever published there is no
    shared ``.gitignore`` commit, so the ordinary project ignores plus Legion's
    own infrastructure (``_LOCAL_EXCLUDE_SEED``) live only in the room clone's
    ``.git/info/exclude`` — that keeps the agent's ``git add -A`` / ``git
    status`` clean without committing anything.

    Lines already present (in any order) are left untouched; only missing ones
    are appended. ``extra`` contributes caller-specific lines under the same
    dedup. Best-effort: a filesystem error is swallowed, never raised.

    Returns ``True`` when the file was actually modified, ``False`` otherwise
    (already complete, no ``.git`` dir, or a swallowed filesystem error).
    """
    exclude = root / ".git" / "info" / "exclude"
    try:
        if not exclude.parent.is_dir():
            return False
        # The agent owns .git: never read or write through a link it planted.
        existing = read_repo_file(root, ".git", "info", "exclude") or ""
        lines = existing.splitlines()
        # The private-files rule is a policy invariant: repair duplicate
        # copies left by interrupted/older backfills while preserving every
        # other local rule and its order.
        private_rule = f"{USERFILES_DIR}/"
        deduped_lines: list[str] = []
        private_seen = False
        duplicate_private_rule = False
        for line in lines:
            if line == private_rule:
                if private_seen:
                    duplicate_private_rule = True
                    continue
                private_seen = True
            deduped_lines.append(line)

        present = set(deduped_lines)
        missing = [
            line
            for line in (_LOCAL_EXCLUDE_SEED + extra).splitlines()
            if line.strip() and line not in present
        ]
        if not missing and not duplicate_private_rule:
            return False
        if missing:
            deduped_lines.extend(missing)
        write_repo_file(root, (".git", "info", "exclude"), "\n".join(deduped_lines) + "\n")
        return True
    except (OSError, UnicodeDecodeError):
        return False


def _userfiles_excluded(root: Path) -> bool:
    """Whether ``.git/info/exclude`` currently carries the private-files rule."""
    try:
        text = read_repo_file(root, ".git", "info", "exclude") or ""
        return f"{USERFILES_DIR}/" in text.splitlines()
    except (OSError, UnicodeDecodeError):
        return False


def _stage_all(root: Path) -> list[str]:
    """``git add -A`` that can never stage room-private files.

    ``.git/info/exclude`` is only a *local* rule that a one-off backfill wrote,
    so it can be missing (a repo older than the rule, an interrupted backfill,
    a workspaced process still running pre-rule code). Every staging point
    therefore re-asserts the rule first and, independently of it, carves
    ``.userfiles`` out with an exclude pathspec — that also covers a file which
    is already tracked, which the exclude file would not.

    Room-local ``.legion/`` content a legacy room still tracks is unstaged too
    (the working copy stays), and newly added machine binaries are held back by
    ``_hold_back_binaries``. Returns the held-back paths.
    """
    _write_local_excludes(root)
    _untrack_volatile_legion_paths(root)
    _run(["add", "-A", "--", ".", f":(exclude){USERFILES_DIR}"], cwd=root)
    return _hold_back_binaries(root)


# Package archives and native libraries: build/runtime artefacts of *this*
# container, never project source. ``.so`` also matches versioned names such as
# ``libX11.so.6.4.0`` and the symlinks pointing at them.
_BINARY_SUFFIXES = (".deb", ".rpm", ".so")
_SO_VERSIONED = re.compile(r"\.so(\.\d+)+$")
_ELF_MAGIC = b"\x7fELF"
_HOLD_BACK_HEADER = "# Legion: machine binaries kept out of git (auto)"
# Cap per call: a dump of thousands of files is reported by count beyond this.
_HOLD_BACK_REPORT_MAX = 50


def _is_machine_binary(root: Path, rel: str) -> bool:
    """A native library, executable or OS package — by name or ELF header."""
    name = rel.rsplit("/", 1)[-1].lower()
    if name.endswith(_BINARY_SUFFIXES) or _SO_VERSIONED.search(name):
        return True
    p = root / rel
    try:
        if p.is_symlink() or not p.is_file():
            return False
        with p.open("rb") as fh:
            return fh.read(4) == _ELF_MAGIC
    except OSError:
        return False


def _gitignore_literal(rel: str) -> str:
    """An anchored ``.git/info/exclude`` line matching exactly ``rel``."""
    escaped = re.sub(r"([\\*?\[\]!#])", r"\\\1", rel)
    if escaped.endswith(" "):
        escaped = escaped[:-1] + "\\ "
    return "/" + escaped


def _hold_back_binaries(root: Path) -> list[str]:
    """Unstage newly added machine binaries and ignore them locally.

    Agents that lack root install system libraries by hand — ``apt-get
    download`` + ``dpkg -x`` into the workspace, the only path that survives a
    restart — and the next ``git add -A`` would ship hundreds of ``.deb`` /
    ``.so`` files into the client's repository. Only *added* paths are
    considered: a binary the project already tracks is its own business, and an
    explicit ``git add -f`` by a human still works because the rule is local.
    Each held-back path is written to ``.git/info/exclude`` so the agent's own
    ``git add -A`` skips it from then on. Best-effort, never raises.
    """
    try:
        r = _run(["diff", "--cached", "--name-only", "--diff-filter=A", "-z"], cwd=root)
        if r.returncode != 0:
            return []
        added = [p for p in r.stdout.split("\0") if p]
        held = [p for p in added if _is_machine_binary(root, p)]
        if not held:
            return []
        for i in range(0, len(held), 200):
            _run(["rm", "--cached", "--quiet", "--", *held[i:i + 200]], cwd=root)
        lines = [_HOLD_BACK_HEADER, *(_gitignore_literal(p) for p in held)]
        _write_local_excludes(root, "\n".join(lines) + "\n")
        logger.info("held back %d machine binaries from commit in %s", len(held), root)
        return held
    except Exception as exc:  # pragma: no cover — never block a commit
        logger.warning("binary hold-back failed in %s: %s", root, exc)
        return []


def is_path_tracked(room_id: str, path: str) -> bool:
    """Whether ``path`` is currently tracked in the git index.

    internal design notes §3.5 / Stage 8 — before ``sync_room_env_file``
    writes secrets into ``.env`` it refuses to touch a file that is already
    tracked (an externally-adopted repo, or a user who committed ``.env`` before
    Legion ever wrote to it): appending secrets to a tracked file would publish
    them on the next push. Checks the *index*, not a specific ref — a file
    ``git add``-ed but not yet committed already counts.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return False
    result = _run(["ls-files", "--error-unmatch", "--", path], cwd=root)
    return result.returncode == 0


def backfill_local_excludes(room_id: str) -> dict:
    """Backfill ``.git/info/exclude`` for a repo whose ``git_init`` predates
    internal design notes Stage 1 (idempotent, no history / HEAD change).

    Rooms created before Stage 1 committed a seed ``.gitignore`` and never
    wrote the project ignores or the ``.legion/git.lock`` line into
    ``.git/info/exclude``. This runs ``_write_local_excludes`` once so their
    working tree stays clean the same way a fresh ``git_init`` does. A fresh
    repo already carries the lines — this is a no-op there.

    Returns ``{ok, updated, error}`` in the style of ``ensure_base_branches``.
    """
    root = _workspace_root(room_id)
    userfiles_created = not (root / USERFILES_DIR).is_dir()
    ensure_userfiles_dir(room_id, root=root)
    if not (root / ".git").is_dir():
        # No repo yet: ``git_init`` writes the rule before its first commit, so
        # nothing can be staged unprotected.
        result = {
            "ok": False,
            "updated": False,
            "error": "no git repo",
            "userfiles_protected": True,
        }
        if userfiles_created:
            result["userfiles_created"] = True
        return result

    from app.release_guard import ensure_ref_guard

    updated = _write_local_excludes(root)
    if ensure_ref_guard(root):
        chown_workspace_tree(room_id)
    if updated:
        # The exclude file was written as root — keep it readable by the
        # container user, same as every other git-object mutation here.
        chown_workspace_tree(room_id)
    # Verified on disk, not assumed: the backend refuses private writes unless
    # this is True, so an old workspaced (which omits the key) fails closed.
    result = {"ok": True, "updated": updated, "userfiles_protected": _userfiles_excluded(root)}
    if userfiles_created:
        result["userfiles_created"] = True
    tracked = git_ls_tracked_userfiles_details(room_id)
    if tracked:
        result.update(
            {
                "userfiles_policy_conflict": True,
                "userfiles_tracked": [entry["path"] for entry in tracked],
                "userfiles_tracked_details": tracked,
            }
        )
    return result


def _room_author(cwd: Path) -> Optional[tuple[str, str]]:
    """The (name, email) ``git_set_author`` stored for this repo, if any."""
    r = git_exec(
        ["config", "--get-regexp", r"^legion\.author"],
        cwd, env=_GIT_ENV, flags=_BASE_FLAGS,
    )
    found: dict[str, str] = {}
    for line in r.stdout.splitlines():
        key, _, value = line.partition(" ")
        found[key.lower()] = value.strip()
    name = found.get(_AUTHOR_NAME_KEY.lower())
    email = found.get(_AUTHOR_EMAIL_KEY.lower())
    if not name or not email:
        return None
    return name, email


def _with_coauthor(message: str) -> str:
    if _COAUTHOR_TRAILER.lower() in message.lower():
        return message
    return f"{message.rstrip()}\n\n{_COAUTHOR_TRAILER}"


def _authored_args(args: list[str], cwd: Path) -> tuple[list[str], dict[str, str]]:
    """Author a commit-writing command by the room's person, co-authored by Legion."""
    if not args or args[0] not in _COMMIT_WRITING:
        return args, _GIT_ENV
    author = _room_author(cwd)
    if author is None or author[1].lower() == LEGION_GIT_EMAIL.lower():
        return args, _GIT_ENV
    env = {**_GIT_ENV, "GIT_AUTHOR_NAME": author[0], "GIT_AUTHOR_EMAIL": author[1]}
    if args[0] == "commit":
        # Options go before a ``--`` pathspec separator, or git reads them as paths.
        cut = args.index("--") if "--" in args else len(args)
        args = args[:cut] + ["--trailer", _COAUTHOR_TRAILER] + args[cut:]
    elif args[0] == "merge" and "-m" in args:
        i = args.index("-m") + 1
        if i < len(args):
            args = args[:i] + [_with_coauthor(args[i])] + args[i + 1:]
    # ``revert --no-edit`` takes git's own message; it keeps the author only.
    return args, env


def _run(
    args: list[str],
    cwd: Path,
    input: Optional[str] = None,
    *,
    auth: Optional[tuple[str, Optional[str]]] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run git through ``git_exec`` (hardened flags, config check, timeout).

    ``auth=(remote_url, token)`` sends the token as an auth header for that
    URL's origin — the URL itself stays token-free.
    """
    args, env = _authored_args(args, cwd)
    if auth is not None:
        env = {**env, **auth_env(*auth)}
    return git_exec(args, cwd, input=input, env=env, timeout=timeout, flags=_BASE_FLAGS)


# ---------------------------------------------------------------------------
# Repo lifecycle
# ---------------------------------------------------------------------------

def is_git_repo(room_id: str) -> bool:
    return (_workspace_root(room_id) / ".git").is_dir()


def git_init(room_id: str, working_branch: Optional[str] = None) -> None:
    """Initialise a git repo in the workspace volume (idempotent).

    internal design notes Stage 1: the synthetic root commit is now *empty* —
    nothing is added, no ``.gitignore`` is committed. The project ignores plus
    Legion's own infrastructure lines go into ``.git/info/exclude`` instead. A
    shared ``.gitignore`` commit is seeded later and only when Legion itself
    publishes the repo (Stage 2); an externally adopted repo keeps its own.

    ``working_branch`` names the integration branch a new repo is checked out
    on (default ``dev``). An existing repo is left as it is —
    ``ensure_base_branches`` reconciles its name with the Hub.
    """
    root = _workspace_root(room_id)
    root.mkdir(parents=True, exist_ok=True)
    ensure_userfiles_dir(room_id, root=root)

    from app.release_guard import ensure_ref_guard

    if (root / ".git").is_dir():
        _write_local_excludes(root)
        ensure_ref_guard(root)
        return

    integration = (
        working_branch if valid_working_branch_name(working_branch) else INTEGRATION_BRANCH
    )

    _run(["init"], cwd=root)
    _run(["config", "user.name", LEGION_GIT_NAME], cwd=root)
    _run(["config", "user.email", LEGION_GIT_EMAIL], cwd=root)

    _write_local_excludes(root)
    ensure_ref_guard(root)

    _run(["commit", "--allow-empty", "-m", "init: Legion Hub workspace"], cwd=root)

    # internal design notes Stage 0: the room develops on its
    # integration branch; `main` stays put as the release branch. The empty
    # initial commit above landed on `main` (init.defaultBranch=main) — branch
    # the integration branch off it and check it out.
    _run(["checkout", "-b", integration], cwd=root)
    _record_working_branch(root, integration)

    # .git objects were created as root — hand them to the container user
    chown_workspace_tree(room_id)

    logger.info("git_init: initialised repo for room %s (HEAD=%s)", room_id, integration)


@_serialized(busy_dict=False)
def git_set_author(room_id: str, name: Optional[str], email: Optional[str]) -> None:
    """Record the person this room's commits are authored by (None clears it).

    Lives in the repo's ``.git/config`` only. Also resets ``user.name`` /
    ``user.email`` to Legion, replacing the pre-rename ``Legion Hub`` identity
    older repos were initialised with.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        git_init(room_id)
    _run(["config", "user.name", LEGION_GIT_NAME], cwd=root)
    _run(["config", "user.email", LEGION_GIT_EMAIL], cwd=root)
    if name and email:
        _run(["config", _AUTHOR_NAME_KEY, name], cwd=root)
        _run(["config", _AUTHOR_EMAIL_KEY, email], cwd=root)
    else:
        _run(["config", "--unset", _AUTHOR_NAME_KEY], cwd=root)
        _run(["config", "--unset", _AUTHOR_EMAIL_KEY], cwd=root)


def _local_branch_exists(root: Path, name: str) -> bool:
    return _run(["rev-parse", "--verify", "--quiet", f"refs/heads/{name}"], cwd=root).returncode == 0


@_serialized(busy_dict=True)
def ensure_base_branches(room_id: str, working_branch: Optional[str] = None) -> dict:
    """Guarantee `main` and the integration branch both exist (idempotent).

    Backfill for rooms whose repo was initialised before internal design notes
    Stage 0, and the point where the repo learns the integration branch name
    the Hub has (``working_branch``; omitted → the name the repo already
    records, default ``dev``). Never switches branches — a concurrent agent
    session may be working on the current HEAD.

    When the Hub names a branch the repo does not have while the branch the
    repo used so far does exist, the old one is renamed only if it carries no
    commits ``main`` lacks (nothing to lose — a room renamed before any work).
    With work on it the call refuses (``error="working branch mismatch"``) and
    changes nothing: moving that work is the explicit rename's job, never a
    side effect of a backfill that would leave it behind on the old name.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "created_dev": False, "error": "no git repo"}

    if working_branch is not None and not valid_working_branch_name(working_branch):
        return {
            "ok": False, "created_dev": False,
            "error": f"invalid working branch name {working_branch!r}",
        }
    previous = working_branch_of(root)
    target = working_branch or previous
    result = {"ok": True, "created_dev": False, "working_branch": target, "renamed_from": None}

    if _local_branch_exists(root, target):
        _record_working_branch(root, target)
        return result

    if target != previous and _local_branch_exists(root, previous):
        has_main = _local_branch_exists(root, RELEASE_BRANCH)
        ahead = _rev_count(root, f"{RELEASE_BRANCH}..{previous}") if has_main else None
        if ahead != 0:
            logger.warning(
                "ensure_base_branches: room %s works on %s but the Hub names %s; "
                "left as is (%s commits not on %s)",
                room_id, previous, target, ahead, RELEASE_BRANCH,
            )
            return {
                "ok": False, "created_dev": False, "working_branch": previous,
                "renamed_from": None, "error": "working branch mismatch",
            }
        r = _run(["branch", "-m", previous, target], cwd=root)
        if r.returncode != 0:
            return {"ok": False, "created_dev": False, "error": r.stderr[:300]}
        _record_working_branch(root, target)
        chown_workspace_tree(room_id)
        logger.info(
            "ensure_base_branches: renamed empty %s to %s for room %s", previous, target, room_id,
        )
        return {**result, "renamed_from": previous}

    base = RELEASE_BRANCH if _local_branch_exists(root, RELEASE_BRANCH) else "HEAD"
    r = _run(["branch", target, base], cwd=root)
    if r.returncode != 0:
        logger.warning(
            "ensure_base_branches: could not create %s for room %s: %s",
            target, room_id, r.stderr[:300],
        )
        return {"ok": False, "created_dev": False, "error": r.stderr[:300]}
    _record_working_branch(root, target)
    chown_workspace_tree(room_id)
    return {**result, "created_dev": True}


@_serialized(busy_dict=True)
def rename_working_branch(room_id: str, from_branch: str, to_branch: str) -> dict:
    """Rename the room's integration branch ``from_branch`` → ``to_branch``
    (internal design notes Stage 3).

    ``git branch -m`` keeps the commits and the reflog, and a HEAD sitting on
    the branch follows it. The upstream recorded by an earlier ``push -u``
    points at the old remote name, so it is dropped — Legion always pushes and
    fetches with an explicit refspec. The new name is recorded as the repo's
    integration branch. A room without a repo gets one, checked out on the new
    name, so a later write does not create ``dev`` behind the Hub's back.

    Returns ``{ok, renamed, created, from_branch, to_branch, head, error,
    code}``; ``code`` is ``invalid_name`` / ``branch_exists`` on a refusal.
    The Hub renames back with the same call when it cannot store the name.
    """
    result: dict = {
        "ok": False, "renamed": False, "created": False,
        "from_branch": from_branch, "to_branch": to_branch,
        "head": None, "error": None, "code": None,
    }
    if not valid_working_branch_name(to_branch):
        return {**result, "error": f"invalid branch name {to_branch!r}", "code": "invalid_name"}
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        git_init(room_id, to_branch)
        return {**result, "ok": True, "created": True, "head": git_current_branch(room_id)}

    if from_branch == to_branch:
        if not _local_branch_exists(root, to_branch):
            base = RELEASE_BRANCH if _local_branch_exists(root, RELEASE_BRANCH) else "HEAD"
            r = _run(["branch", to_branch, base], cwd=root)
            if r.returncode != 0:
                return {**result, "error": r.stderr[:300]}
            result["created"] = True
        _record_working_branch(root, to_branch)
        chown_workspace_tree(room_id)
        return {**result, "ok": True, "head": git_current_branch(room_id)}

    if _local_branch_exists(root, to_branch):
        return {
            **result,
            "error": f"a branch named {to_branch!r} already exists in this room",
            "code": "branch_exists",
        }

    if _local_branch_exists(root, from_branch):
        r = _run(["branch", "-m", from_branch, to_branch], cwd=root)
        if r.returncode != 0:
            return {**result, "error": r.stderr[:300]}
        # No upstream is fine; any other failure leaves a harmless stale entry.
        _run(["branch", "--unset-upstream", to_branch], cwd=root)
        result["renamed"] = True
    else:
        recorded = working_branch_of(root)
        if recorded not in (from_branch, to_branch) and _local_branch_exists(root, recorded):
            # The repo works on a branch the Hub did not name: renaming
            # nothing and starting a fresh branch would strand that work.
            return {
                **result,
                "error": (
                    f"this room works on {recorded!r}, not {from_branch!r}; "
                    "nothing was renamed"
                ),
                "code": "working_branch_mismatch",
            }
        # Nothing to carry over: start the new branch where the old one would.
        base = RELEASE_BRANCH if _local_branch_exists(root, RELEASE_BRANCH) else "HEAD"
        r = _run(["branch", to_branch, base], cwd=root)
        if r.returncode != 0:
            return {**result, "error": r.stderr[:300]}
        result["created"] = True

    _record_working_branch(root, to_branch)
    chown_workspace_tree(room_id)
    logger.info(
        "rename_working_branch: room %s %s → %s (renamed=%s)",
        room_id, from_branch, to_branch, result["renamed"],
    )
    return {**result, "ok": True, "head": git_current_branch(room_id)}


def remote_branch_exists(
    room_id: str, remote_url: str, branch: str, token: Optional[str] = None
) -> dict:
    """Whether ``refs/heads/<branch>`` exists on ``remote_url`` (``ls-remote``).

    Read-only and network-bound, so it runs outside the room lock. Returns
    ``{ok, exists, error}``; ``ok=False`` when the remote could not be asked.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "exists": False, "error": "no git repo"}
    if not remote_url or not branch:
        return {"ok": False, "exists": False, "error": "remote url and branch are required"}
    r = _run(
        ["ls-remote", "--heads", remote_url, f"refs/heads/{branch}"],
        cwd=root, auth=(remote_url, token), timeout=60,
    )
    if r.returncode != 0:
        return {"ok": False, "exists": False, "error": (r.stderr or "ls-remote failed")[:300]}
    exists = any(
        line.split("\t", 1)[-1].strip() == f"refs/heads/{branch}"
        for line in r.stdout.splitlines()
    )
    return {"ok": True, "exists": exists, "error": None}


_SHARED_GITIGNORE_MESSAGE = "chore: add default .gitignore"


@_serialized(busy_dict=True)
def ensure_shared_gitignore(
    room_id: str, integration_branch: Optional[str] = None
) -> dict:
    """Commit a shared ``.gitignore`` on ``integration_branch`` once, if the
    repo does not already track one (internal design notes Stage 2).

    Called before the first outgoing push of a repo Legion itself publishes
    (greenfield — the user connected an empty remote). A human who later
    clones that repo from GitHub/GitLab then gets the ordinary project
    ignores, the same as on any professional team. For a repo adopted from an
    external remote this is never called — that ``.gitignore`` belongs to the
    upstream project.

    The committed body is ``_PROJECT_GITIGNORE_DEFAULT`` — never the
    Legion-infrastructure lines, which stay in ``.git/info/exclude``. An
    untracked working-tree ``.gitignore`` the agent already wrote is committed
    as-is rather than overwritten.

    Returns ``{ok, added, sha, branch, stderr}``. HEAD is left where it was.
    """
    root = _workspace_root(room_id)
    branch = integration_branch or working_branch_of(root)
    result: dict = {"ok": False, "added": False, "sha": None, "branch": branch, "stderr": ""}
    if not (root / ".git").is_dir():
        result["stderr"] = "no git repo"
        return result

    if _run(["rev-parse", "--verify", branch], cwd=root).returncode != 0:
        result["stderr"] = f"branch {branch!r} does not exist"
        return result

    tracked = _run(["ls-tree", "--name-only", branch, ".gitignore"], cwd=root)
    if tracked.returncode == 0 and tracked.stdout.strip() == ".gitignore":
        result["ok"] = True
        return result  # already there — nothing to do

    started_on = git_current_branch(room_id)
    if started_on != branch:
        co = _run(["checkout", branch], cwd=root)
        if co.returncode != 0:
            result["stderr"] = co.stderr[:2048]
            return result

    def _restore_head() -> None:
        if started_on and started_on != branch:
            if _run(["checkout", started_on], cwd=root).returncode == 0:
                chown_workspace_tree(room_id)

    try:
        # Not through a link the agent planted (app/git_exec.py).
        existing = read_repo_file(root, ".gitignore") or ""
    except (OSError, UnicodeDecodeError):
        existing = ""
    body = existing if existing.strip() else _PROJECT_GITIGNORE_DEFAULT
    if not body.endswith("\n"):
        body += "\n"
    try:
        write_repo_file(root, (".gitignore",), body)
    except OSError as exc:
        result["stderr"] = str(exc)
        _restore_head()
        return result

    _run(["add", "--", ".gitignore"], cwd=root)
    commit = _run(
        ["commit", "-m", _SHARED_GITIGNORE_MESSAGE, "--", ".gitignore"], cwd=root
    )
    if commit.returncode == 0:
        chown_workspace_tree(room_id)
        sha_r = _run(["rev-parse", "HEAD"], cwd=root)
        result["ok"] = True
        result["added"] = True
        result["sha"] = sha_r.stdout.strip() if sha_r.returncode == 0 else None
    elif "nothing to commit" in (commit.stdout + commit.stderr):
        result["ok"] = True  # an identical file was already tracked
    else:
        result["stderr"] = commit.stderr[:2048]
        logger.warning(
            "ensure_shared_gitignore: commit failed for room %s: %s",
            room_id, commit.stderr[:300],
        )

    _restore_head()
    return result


# ---------------------------------------------------------------------------
# Commits
# ---------------------------------------------------------------------------

@_serialized(busy_dict=False)
def git_commit(
    room_id: str,
    message: str,
    allow_main_commit: bool = False,
    only_path: Optional[str] = None,
    report: Optional[dict] = None,
) -> Optional[str]:
    """Stage everything and commit. Returns new HEAD sha, or None if nothing changed.

    ``report``, when given, receives ``held_back``: the machine binaries the
    staging step kept out of this commit (see ``_hold_back_binaries``).

    ``only_path`` commits just that one workspace-relative file and leaves every
    other pending change (an agent's half-done edit) staged or not exactly as it
    was — for Legion's own bookkeeping files such as ``.legion/services.json``.

    Raises ``MainBranchProtectedError`` when HEAD is on ``main`` and the
    caller did not pass ``allow_main_commit=True`` — agent and auto commits
    must never move ``main`` (internal design notes §0.1). Manual user
    operations (revert / restore) pass ``True``.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        git_init(room_id)

    if not allow_main_commit and git_current_branch(room_id) == RELEASE_BRANCH:
        raise MainBranchProtectedError(
            "refusing to commit on `main`: agent and auto commits must target "
            f"a task branch or `{working_branch_of(root)}` "
            "(internal design notes §0.1)"
        )

    if only_path:
        rel = normalize_workspace_path(only_path)
        _run(["add", "-A", "--", rel], cwd=root)
        r = _run(["commit", "-m", message, "--", rel], cwd=root)
    else:
        held = _stage_all(root)
        if report is not None:
            report["held_back"] = held
        r = _run(["commit", "-m", message], cwd=root)
    if r.returncode != 0:
        output = r.stdout + r.stderr
        if "nothing to commit" in output:
            return None
        logger.warning("git commit failed for room %s: %s", room_id, r.stderr[:500])
        return None

    # New .git objects were written as root — hand them to the container user
    chown_workspace_tree(room_id)

    sha_r = _run(["rev-parse", "HEAD"], cwd=root)
    return sha_r.stdout.strip() if sha_r.returncode == 0 else None


# ---------------------------------------------------------------------------
# History & diff
# ---------------------------------------------------------------------------

def git_log(
    room_id: str,
    path: Optional[str] = None,
    limit: int = 20,
    rev_range: Optional[str] = None,
) -> list[dict]:
    """Return list of {sha, short_sha, message, author, timestamp}.

    ``rev_range`` (e.g. ``"main..dev"``) restricts the log to that revision
    range — used by the agentic CI/CD deploy hook (internal design notes Stage 9)
    to list the commits a promotion carries. A value starting with ``-`` is
    ignored to keep it from being read as a git option.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return []

    fmt = "%H%x1f%s%x1f%an%x1f%aI"
    args = ["log", f"--max-count={limit}", f"--format={fmt}"]
    if rev_range and not str(rev_range).startswith("-"):
        args.append(str(rev_range))
    if path:
        normalized = normalize_workspace_path(path, allow_empty=True)
        if normalized:
            args += ["--", normalized]

    r = _run(args, cwd=root)
    if r.returncode != 0:
        return []

    entries: list[dict] = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("\x1f")
        if len(parts) < 4:
            continue
        sha = parts[0]
        entries.append({
            "sha": sha,
            "short_sha": sha[:8],
            "message": parts[1],
            "author": parts[2],
            "timestamp": parts[3],
        })
    return entries


def git_diff(
    room_id: str,
    from_ref: Optional[str] = None,
    to_ref: Optional[str] = None,
    path: Optional[str] = None,
    stat: bool = False,
    merge_base: bool = False,
) -> str:
    """Return unified diff text.

    - No refs: working tree vs HEAD
    - from_ref only: from_ref vs working tree
    - Both refs: from_ref..to_ref (or from_ref...to_ref when merge_base=True —
      only the changes to_ref introduced since it diverged from from_ref)
    - stat: return `--stat` summary instead of the full diff
    - merge_base (from_ref only): diff from merge-base(from_ref, HEAD) so
      commits that landed on from_ref after the branch diverged don't show
      up as reverse changes
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return ""

    args = ["diff"]
    if stat:
        args.append("--stat")
    if from_ref and to_ref:
        args.append(f"{from_ref}...{to_ref}" if merge_base else f"{from_ref}..{to_ref}")
    elif from_ref:
        base = from_ref
        if merge_base:
            mb = _run(["merge-base", from_ref, "HEAD"], cwd=root)
            if mb.returncode == 0 and mb.stdout.strip():
                base = mb.stdout.strip()
        args.append(base)
    else:
        args.append("HEAD")

    if path:
        normalized = normalize_workspace_path(path, allow_empty=True)
        if normalized:
            args += ["--", normalized]

    r = _run(args, cwd=root)
    return r.stdout


def git_ls_tree(room_id: str, ref: str, path: Optional[str] = None) -> list[dict]:
    """Return [{"path", "size"}] for every blob at `ref` (optionally scoped to `path`).

    Sizes come straight from the git object store, so this is exact byte
    length — used by the hub's restore dry-run to compare against the
    current volume tree without touching the working tree (see
    internal design notes Stage 4).
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return []

    args = ["ls-tree", "-r", "-l", "--full-tree", ref]
    if path:
        normalized = normalize_workspace_path(path, allow_empty=True)
        if normalized:
            args += ["--", normalized]

    r = _run(args, cwd=root)
    if r.returncode != 0:
        return []

    entries: list[dict] = []
    for line in r.stdout.splitlines():
        try:
            meta, file_path = line.split("\t", 1)
        except ValueError:
            continue
        parts = meta.split()
        if len(parts) < 4 or parts[1] != "blob":
            continue
        size = int(parts[3]) if parts[3].isdigit() else 0
        entries.append({"path": file_path, "size": size})
    return entries


def git_read_file_at(room_id: str, path: str, rev: str) -> Optional[str]:
    """Return content of `path` at revision `rev`, or None if not found."""
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return None

    normalized = normalize_workspace_path(path)
    r = _run(["show", f"{rev}:{normalized}"], cwd=root)
    if r.returncode != 0:
        return None
    return r.stdout


# ---------------------------------------------------------------------------
# Revert
# ---------------------------------------------------------------------------

@_serialized(busy_dict=False)
def git_revert(room_id: str, ref: str, path: Optional[str] = None) -> bool:
    """Restore file(s) to the state at `ref`, then auto-commit."""
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return False

    normalized = normalize_workspace_path(path, allow_empty=True) if path else ""
    target = [normalized] if normalized else ["."]
    r = _run(["checkout", ref, "--"] + target, cwd=root)
    if r.returncode != 0:
        logger.warning("git revert failed for room %s: %s", room_id, r.stderr[:500])
        return False

    label = path or "all files"
    # Revert is an explicit human operation — allowed to touch `main`.
    git_commit(room_id, f"revert: restore {label} to {ref[:8]}", allow_main_commit=True)
    chown_workspace_tree(room_id)  # checkout rewrote worktree files as root
    return True


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------

def _head_sha(root: Path) -> Optional[str]:
    r = _run(["rev-parse", "HEAD"], cwd=root)
    return r.stdout.strip() if r.returncode == 0 else None


def git_current_branch(room_id: str) -> Optional[str]:
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return None
    r = _run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    return r.stdout.strip() if r.returncode == 0 else None


def git_branch_exists(room_id: str, branch_name: str) -> bool:
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return False
    return bool(_run(["branch", "--list", branch_name], cwd=root).stdout.strip())


@_serialized(busy_dict=False)
def git_create_branch(room_id: str, branch_name: str, base: Optional[str] = None) -> bool:
    """Create and switch to ``branch_name``.

    ``base`` (a branch or commit-ish) is the starting point — internal design notes
    Stage 1 branches a task off ``dev`` or a goal branch, not off whatever HEAD
    happened to be. When ``base`` is omitted the current HEAD is used.

    Idempotent: if ``branch_name`` already exists (e.g. a resumed task), the
    branch is simply checked out and ``base`` is ignored.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        git_init(room_id)

    if git_branch_exists(room_id, branch_name):
        r = _run(["checkout", branch_name], cwd=root)
    elif base:
        r = _run(["checkout", "-b", branch_name, base], cwd=root)
    else:
        r = _run(["checkout", "-b", branch_name], cwd=root)

    if r.returncode != 0:
        logger.warning(
            "git_create_branch failed for room %s/%s (base=%s): %s",
            room_id, branch_name, base, r.stderr[:300],
        )
    else:
        chown_workspace_tree(room_id)
    return r.returncode == 0


@_serialized(busy_dict=False)
def git_rename_branch(room_id: str, name: str, new_name: str) -> bool:
    """Rename local branch ``name`` to ``new_name`` (``git branch -m``).

    internal design notes Stage 2: a dialogue accumulates its work on
    ``session/<session_id>`` and only becomes a task once that branch has
    commits — the rename gives it the canonical ``task/<id>`` name. Works while
    the branch is checked out. Refuses to overwrite an existing branch (no
    ``-M``): a name collision means somebody else's history is at stake.

    Returns True when the rename happened or ``new_name`` is already the only
    one that exists (idempotent re-run); False when it could not be done — the
    caller keeps using the old name, which is a cosmetic loss, not a failure.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return False
    if not name or not new_name or name == new_name:
        return False

    if not git_branch_exists(room_id, name):
        return git_branch_exists(room_id, new_name)
    if git_branch_exists(room_id, new_name):
        logger.warning(
            "git_rename_branch: %s already exists in room %s — leaving %s alone",
            new_name, room_id, name,
        )
        return False

    r = _run(["branch", "-m", name, new_name], cwd=root)
    if r.returncode != 0:
        logger.warning(
            "git_rename_branch failed for room %s (%s → %s): %s",
            room_id, name, new_name, r.stderr[:300],
        )
        return False
    return True


@_serialized(busy_dict=True)
def git_merge_branch(
    room_id: str,
    branch_name: str,
    target: str = RELEASE_BRANCH,
    theirs: bool = True,
    message: Optional[str] = None,
    *,
    strategy: Literal["ours", "theirs", "legacy"] | None = "legacy",
    expected_tips: tuple[str, str] | None = None,
) -> dict:
    """Merge a source ref into ``target`` with ``--no-ff``.

    With no explicit strategy, legacy task merges prefer the incoming side
    and gates passing ``theirs=False`` leave conflicts unresolved. Explicit
    ``None`` also leaves conflicts unresolved.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "conflict": False, "stdout": "", "stderr": "no git repo"}

    if strategy == "legacy":
        strategy = "theirs" if theirs else None

    remote_source = branch_name.startswith(_REMOTE_REF_PREFIX)
    if expected_tips is not None and not _review_tips_match(root, target, branch_name, expected_tips):
        return _stale_review()
    source = expected_tips[1] if expected_tips is not None else branch_name
    if remote_source and (
        git_ls_remote_userfiles(root, source)
        or git_ls_remote_userfiles(root, target)
        or git_ls_tracked_userfiles_details(room_id)
    ):
        return {
            "ok": False,
            "conflict": False,
            "userfiles_policy_conflict": True,
            "stderr": f"private-files policy conflict: a merge side or the index tracks {USERFILES_DIR}/",
        }

    # A tracked file with uncommitted edits — or per-volume Legion metadata a
    # legacy room still tracks (``.legion/generation.json`` changes on every
    # operation) — makes the checkout below abort with "Your local changes
    # would be overwritten by checkout". Park it first, the same guard "Pull
    # from Git" already uses, then clear the volatile remnants so the switch
    # cannot trip on an untracked-vs-tracked collision. The exclude rules are
    # refreshed first so a room older than a rule (``.legion/bg/``) has the
    # just-untracked files ignored — and therefore expendable — on checkout.
    head_before = _head_sha(root)
    _write_local_excludes(root)
    park_error = _park_worktree_before_switch(root, room_id, git_current_branch(room_id))
    if park_error:
        return {"ok": False, "conflict": False, "stdout": "", "stderr": park_error}
    _drop_volatile_worktree_remnants(root)

    co = _run(["checkout", target], cwd=root)
    if co.returncode != 0:
        logger.warning(
            "git_merge_branch: cannot checkout target %s for room %s: %s",
            target, room_id, co.stderr[:300],
        )
        return {
            "ok": False, "conflict": False,
            "stdout": co.stdout[:4096], "stderr": co.stderr[:4096],
        }

    # ``target`` (a legacy ``main`` / ``dev``) may still track the volatile
    # metadata too; unstage + commit it here so the upcoming merge does not hit
    # a modify/delete conflict on a file that changes every operation. This
    # commit on ``target`` is legitimate — ``git_merge_branch`` is the
    # sanctioned writer for every gate, ``main`` included.
    if _untrack_volatile_legion_paths(root):
        _stage_all(root)
        _run(["commit", "-m", "chore: untrack Legion volume metadata"], cwd=root)
    _drop_volatile_worktree_remnants(root)

    if expected_tips is not None and _head_sha(root) != expected_tips[0]:
        return _stale_review()
    msg = message or f"merge: {branch_name} → {target}"
    args = ["merge", "--no-ff"]
    if remote_source:
        args.append("--allow-unrelated-histories")
    if strategy:
        args += ["-X", strategy]
    args += ["-m", msg, "--", source]
    r = _run(args, cwd=root)
    conflict = r.returncode != 0 and "CONFLICT" in (r.stdout + r.stderr)

    if conflict:
        # Leave no half-merged state behind for the next session / gate.
        _run(["merge", "--abort"], cwd=root)
    elif r.returncode != 0:
        logger.warning("git_merge_branch failed for room %s: %s", room_id, r.stderr[:500])

    chown_workspace_tree(room_id)  # checkout/merge rewrote files as root

    sha: Optional[str] = None
    if r.returncode == 0:
        head = _run(["rev-parse", "HEAD"], cwd=root)
        sha = head.stdout.strip() if head.returncode == 0 else None

    return {
        "ok": r.returncode == 0,
        "conflict": conflict,
        "stdout": r.stdout[:4096],
        "stderr": r.stderr[:4096],
        "sha": sha,
        "head_changed": sha is not None and sha != head_before,
    }


def git_archive_branch(room_id: str, ref: str) -> Optional[bytes]:
    """Tar of ``ref``'s tree's base snapshot for a
    ``mode=worker`` connector dispatch.

    The connector's local CLI agent has no network path to this repository
    (it lives in this container's volume, reachable only from our own
    services — exposing it would undo the whole point of the connector's
    pull-only threat model). So instead of git access, a dispatch hands
    the connector this one-shot tarball of the task branch's tip to start
    local work from; the connector reports back a plain unified diff, which
    ``git_apply_patch`` below folds onto the real branch.

    Read-only — no HEAD movement, no working-tree write, so no
    ``room_git_lock`` is needed here (unlike every mutating op above).
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return None
    r = git_exec(["archive", "--format=tar", ref], root, text=False, env=_GIT_ENV, flags=_BASE_FLAGS)
    if r.returncode != 0:
        logger.warning(
            "git_archive_branch failed for room %s ref %s: %s",
            room_id, ref, r.stderr[:300].decode("utf-8", errors="replace"),
        )
        return None
    return r.stdout


@_serialized(busy_dict=True)
def git_apply_patch(
    room_id: str,
    branch_name: str,
    patch: str,
    message: Optional[str] = None,
) -> dict:
    """Apply a connector-produced unified diff onto ``branch_name`` and commit it.

    ``mode=worker``: the counterpart of
    ``git_archive_branch`` — the connector worked against a tar snapshot of
    this same branch in its own directory (no git access of its own) and
    reports back a plain ``git diff`` against that starting point. This
    applies it for real, so ``merge_task_branch``/Gate 1 see it exactly like
    a local agent's own commit — same shape as ``git_merge_branch``'s
    conflict handling: a failed apply never leaves a half-applied state.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "conflict": False, "stdout": "", "stderr": "no git repo"}
    if not patch.strip():
        return {"ok": False, "conflict": False, "stdout": "", "stderr": "empty patch"}
    if not git_branch_exists(room_id, branch_name):
        return {
            "ok": False, "conflict": False, "stdout": "",
            "stderr": f"branch {branch_name} does not exist",
        }

    park_error = _park_worktree_before_switch(root, room_id, git_current_branch(room_id))
    if park_error:
        return {"ok": False, "conflict": False, "stdout": "", "stderr": park_error}
    _drop_volatile_worktree_remnants(root)

    co = _run(["checkout", branch_name], cwd=root)
    if co.returncode != 0:
        return {"ok": False, "conflict": False, "stdout": co.stdout[:4096], "stderr": co.stderr[:4096]}

    apply_r = _run(["apply", "--whitespace=fix", "-"], cwd=root, input=patch)
    if apply_r.returncode != 0:
        # 3-way tolerates minor drift between the archive's tip and whatever
        # is currently on ``branch_name`` (e.g. a rework's second attempt).
        apply_r = _run(["apply", "--3way", "--whitespace=fix", "-"], cwd=root, input=patch)
    if apply_r.returncode != 0:
        _run(["checkout", "--", "."], cwd=root)  # discard any partial 3-way state
        return {
            "ok": False, "conflict": True,
            "stdout": apply_r.stdout[:4096], "stderr": apply_r.stderr[:4096],
        }

    chown_workspace_tree(room_id)
    sha = git_commit(room_id, message or f"connector: apply patch to {branch_name}")
    return {
        "ok": sha is not None, "conflict": False,
        "stdout": apply_r.stdout[:4096], "stderr": "", "sha": sha,
    }


@_serialized(busy_dict=True)
def git_revert_commit(
    room_id: str,
    sha: str,
    target: Optional[str] = None,
    mainline: int = 1,
) -> dict:
    """Revert commit ``sha`` on ``target`` with a fresh revert commit.

    internal design notes Stage 3: the one-click undo for a ``trivial``
    auto-merge. ``mainline=1`` reverts a merge commit against its first
    parent (the base branch side). A conflict is rolled back and reported —
    ``target`` is never left half-reverted.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "conflict": False, "stdout": "", "stderr": "no git repo"}
    target = target or working_branch_of(root)

    # Same dirty-tree guard as the merge path: park uncommitted work and clear
    # volatile Legion metadata so the checkout cannot abort.
    park_error = _park_worktree_before_switch(root, room_id, git_current_branch(room_id))
    if park_error:
        return {"ok": False, "conflict": False, "stdout": "", "stderr": park_error}
    _drop_volatile_worktree_remnants(root)

    co = _run(["checkout", target], cwd=root)
    if co.returncode != 0:
        return {
            "ok": False, "conflict": False,
            "stdout": co.stdout[:4096], "stderr": co.stderr[:4096],
        }

    r = _run(
        ["revert", "--no-edit", "-m", str(mainline), sha],
        cwd=root,
    )
    conflict = r.returncode != 0 and "CONFLICT" in (r.stdout + r.stderr)
    if r.returncode != 0:
        _run(["revert", "--abort"], cwd=root)
        if not conflict:
            logger.warning("git_revert_commit failed for room %s: %s", room_id, r.stderr[:500])

    chown_workspace_tree(room_id)

    new_sha: Optional[str] = None
    if r.returncode == 0:
        head = _run(["rev-parse", "HEAD"], cwd=root)
        new_sha = head.stdout.strip() if head.returncode == 0 else None

    return {
        "ok": r.returncode == 0,
        "conflict": conflict,
        "stdout": r.stdout[:4096],
        "stderr": r.stderr[:4096],
        "sha": new_sha,
    }


# ---------------------------------------------------------------------------
# Remote: push / fetch
# ---------------------------------------------------------------------------

@_serialized(busy_dict=True)
def git_push(
    room_id: str,
    remote_url: str,
    branch: str = "main",
    token: Optional[str] = None,
    mode: str = "push",
) -> dict:
    """Push workspace repo to remote. Returns {ok, stdout, stderr}.

    ``mode="push"`` (default) performs a plain ``git push`` and lets the
    remote reject a non-fast-forward update instead of clobbering someone
    else's history — the failure that prompted the ``mode`` split (see
    internal design notes stage 2.2). ``mode="force"`` uses
    ``--force`` so an explicit, acknowledged user choice can override
    divergence. ``--force`` (not ``--force-with-lease``) because
    ``workspaced`` never sets up a named remote, so there is no
    remote-tracking ref for a lease to check against — the safety here
    comes from the explicit ``mode`` choice in the API, not from git's
    lease mechanism. For both modes, if the remote branch doesn't exist
    yet (first push) the call is retried with ``-u`` so an empty workspace
    can still seed a new remote branch.

    The refspec is ``refs/heads/{branch}:refs/heads/{branch}`` — the *local*
    branch is pushed to the same-named remote branch regardless of which
    branch HEAD happens to be checked out (internal design notes Stage 6:
    Gate 2 pushes ``dev`` while the working tree stays on ``dev``, and the
    manual ``main`` push must send ``main`` even though HEAD is on ``dev``).
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "stdout": "", "stderr": "no git repo in workspace"}

    tracked_userfiles = git_ls_tracked_userfiles(room_id)
    if tracked_userfiles:
        paths = ", ".join(tracked_userfiles)
        return {
            "ok": False,
            "stdout": "",
            "stderr": (
                "Hub blocked push: Git tracks private room files under "
                f"{USERFILES_DIR}/ ({paths}). Remove them from the index with "
                f"`git rm -r --cached {USERFILES_DIR}` or move deliberately "
                "publishable files into the project before retrying."
            ),
            "userfiles_push_blocked": True,
            "userfiles_tracked": tracked_userfiles,
            "userfiles_policy_conflict": True,
            "userfiles_tracked_details": git_ls_tracked_userfiles_details(room_id),
        }

    push_url = remote_url
    auth = (remote_url, token)
    force = mode == "force"
    refspec = f"refs/heads/{branch}:refs/heads/{branch}"

    def _looks_like_missing_upstream(stderr: str) -> bool:
        # First-push failures (no upstream branch on the remote) are benign
        # and worth a `-u` retry; genuine rejections ("! [rejected]",
        # "non-fast-forward", "error:", "denied") must not be retried as a
        # force — the caller asked for an explicit mode for a reason.
        low = stderr.lower()
        if "rejected" in low or "non-fast-forward" in low:
            return False
        if "error" in low or "denied" in low or "forbidden" in low:
            return False
        return True

    if force:
        r = _run(["push", push_url, refspec, "--force"], cwd=root, auth=auth)
        if r.returncode != 0 and _looks_like_missing_upstream(r.stderr):
            r = _run(["push", "-u", push_url, refspec], cwd=root, auth=auth)
    else:
        r = _run(["push", push_url, refspec], cwd=root, auth=auth)
        if r.returncode != 0 and _looks_like_missing_upstream(r.stderr):
            r = _run(["push", "-u", push_url, refspec], cwd=root, auth=auth)

    return {
        "ok": r.returncode == 0,
        "stdout": r.stdout[:4096],
        "stderr": r.stderr[:4096],
    }


def git_ls_tracked_userfiles(room_id: str, *, limit: int = 20) -> list[str]:
    """Return the first indexed paths under the room-private namespace.

    ``.git/info/exclude`` only protects untracked files.  An explicit
    ``git add -f`` therefore has to be caught from the index immediately
    before any Hub-managed push or promotion sends data to a remote.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return []
    result = _run(["ls-files", "--", USERFILES_DIR], cwd=root)
    if result.returncode != 0:
        return []
    paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return paths[: max(1, int(limit))]


def git_ls_tracked_userfiles_details(room_id: str, *, limit: int = 20) -> list[dict]:
    """Return safe metadata for private paths currently in the local index.

    Index entries do not have a cheap, stable size query without reading the
    working tree. Size is therefore deliberately reported as zero here; the
    important observability property is that no file content or data URI is
    copied into a policy event.
    """
    return [
        {
            "path": path,
            "size": 0,
            "source": "local_index",
            "error": "local_index_tracks_userfiles",
        }
        for path in git_ls_tracked_userfiles(room_id, limit=limit)
    ]


def git_ls_remote_userfiles(root: Path, ref: str = "FETCH_HEAD", *, limit: int = 20) -> list[dict]:
    """Inspect a fetched tree for ``.userfiles`` without checking it out.

    ``git ls-tree`` reads only Git's object database, so this check can run
    before adopt/pull mutates HEAD or the working tree. Only path and blob
    size leave this layer; file contents never enter the response.
    """
    if not (root / ".git").is_dir():
        return []
    result = _run(
        ["ls-tree", "-r", "-l", "--full-tree", ref, "--", USERFILES_DIR],
        cwd=root,
    )
    if result.returncode != 0:
        return []
    entries: list[dict] = []
    for line in result.stdout.splitlines():
        if "\t" not in line:
            continue
        header, path = line.split("\t", 1)
        fields = header.split()
        if len(fields) < 4 or not path.strip():
            continue
        try:
            size = int(fields[3]) if fields[3] != "-" else 0
        except ValueError:
            size = 0
        entries.append(
            {
                "path": path.strip(),
                "size": max(0, size),
                "source": "remote",
                "error": "remote_tracks_userfiles",
            }
        )
        if len(entries) >= max(1, int(limit)):
            break
    return entries


def _untrack_volatile_legion_paths(root) -> bool:
    """Drop per-volume Legion metadata (``_UNTRACK_LEGION_PATHS``) from the
    index if a legacy room still tracks it. Idempotent; does not commit.

    ``git rm --cached`` only — the working-tree copies stay, and once they are
    out of the index ``.git/info/exclude`` keeps them out of every later
    ``git add -A``. Returns True when something was actually unstaged.
    """
    r = _run(
        ["rm", "-r", "--cached", "--ignore-unmatch", "--", *_UNTRACK_LEGION_PATHS],
        cwd=root,
    )
    return r.returncode == 0 and bool(r.stdout.strip())


def _drop_volatile_worktree_remnants(root) -> None:
    """Remove working-tree copies of volatile Legion metadata before a branch
    switch.

    After ``_untrack_volatile_legion_paths`` the current branch no longer tracks
    them, but the branch about to be checked out might (a legacy room that has
    not run a gate since the exclude landed), and git then refuses the switch
    with "untracked working tree files would be overwritten by checkout".
    ``workspaced`` regenerates ``generation.json`` on demand and run logs are
    disposable, so dropping the on-disk copy is safe.
    """
    for rel in _VOLATILE_LEGION_PATHS:
        p = Path(root) / rel
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        except OSError:
            pass


_PSEUDO_REF_BRANCH_NAMES = ("FETCH_HEAD", "ORIG_HEAD", "MERGE_HEAD", "CHERRY_PICK_HEAD", "HEAD")


def _prune_pseudo_ref_branches(root) -> None:
    """Delete a stray ``refs/heads/FETCH_HEAD`` (and friends).

    ``git fetch <url> <branch>:FETCH_HEAD`` writes an *actual* branch
    ``refs/heads/FETCH_HEAD`` — git only treats a bare ``FETCH_HEAD`` as the
    ``.git/FETCH_HEAD`` pseudo-ref when it is the refspec *source*, not the
    destination. Such a branch is never a real Legion branch, it clutters the
    branch tree, and it makes ``FETCH_HEAD`` an ambiguous ref for later
    ``merge`` / ``branch -f`` calls. Best-effort, never raises.
    """
    for nm in _PSEUDO_REF_BRANCH_NAMES:
        try:
            _run(["update-ref", "-d", f"refs/heads/{nm}"], cwd=root)
        except Exception:
            pass


def _park_worktree_before_switch(root, room_id: str, current: Optional[str]) -> Optional[str]:
    """Commit uncommitted work so a branch switch / merge can run.

    A ``git checkout`` or ``git merge`` refuses to run with a dirty tree, and
    dropping local edits would violate the "never destroy content" contract
    (internal design notes §2). Called before every branch switch a "Pull from
    Git" or a gate merge (``git_merge_branch`` / ``git_revert_commit``) makes.

    Per-volume Legion metadata a legacy room still tracks
    (``_UNTRACK_LEGION_PATHS``) is unstaged first so it never lands in project
    history — ``generation.json`` changes on every operation and would otherwise
    make the tree permanently dirty and dead-end every promotion.

    Returns an error string when the tree is dirty on ``main`` and could not be
    parked without writing to the protected release branch (§0.1), else None.
    """
    _untrack_volatile_legion_paths(root)

    if not _run(["status", "--porcelain"], cwd=root).stdout.strip():
        return None

    if current == RELEASE_BRANCH:
        # Committing here would move `main` outside a promotion. Park the work
        # on the integration branch instead — `git checkout` carries the
        # uncommitted changes across when they don't conflict.
        integration = working_branch_of(root)
        co = _run(["checkout", integration], cwd=root)
        if co.returncode != 0:
            logger.warning(
                "park worktree: dirty tree on %s for room %s, cannot park: %s",
                RELEASE_BRANCH, room_id, co.stderr[:300],
            )
            return (
                f"working tree has uncommitted changes on `{RELEASE_BRANCH}` and they "
                f"could not be moved to `{integration}`: {co.stderr[:300]}"
            )

    _stage_all(root)
    _run(["commit", "-m", "chore: park local changes before branch switch"], cwd=root)
    return None


def _unmerged_paths(root: Path) -> list[str]:
    # NUL delimiters preserve filenames with whitespace and avoid Git quoting.
    result = _run(["diff", "--name-only", "--diff-filter=U", "-z"], cwd=root)
    if result.returncode != 0 or not result.stdout:
        return []
    return result.stdout.rstrip("\0").split("\0")


def _seed_gitignore_autoresolvable(root: Path, conflict_files: list[str] | None = None) -> bool:
    """True when the *only* unmerged path is ``.gitignore`` and our side of it
    is byte-for-byte a known Legion seed (internal design notes Stage 4).

    An ``add/add`` conflict on ``.gitignore`` is the near-universal outcome of
    "Merge histories" against an external repo: every real repo carries its own
    ``.gitignore`` in its root, and a pre-Stage-1 room carried Legion's seed in
    its init commit. When nobody has touched that seed there is a single right
    answer — take the remote's file — so it can be resolved without a human. An
    agent or a user who edited ``.gitignore`` fails this check and the conflict
    still goes to manual resolution.
    """
    if conflict_files is None:
        conflict_files = _unmerged_paths(root)
    if conflict_files != [".gitignore"]:
        return False
    ours = _run(["show", ":2:.gitignore"], cwd=root)
    if ours.returncode != 0:
        return False
    return ours.stdout in _KNOWN_SEED_GITIGNORES


def _rev(root: Path, ref: str) -> Optional[str]:
    r = _run(["rev-parse", ref], cwd=root)
    return r.stdout.strip() if r.returncode == 0 else None


def _commit_sha(root: Path, ref: str) -> str | None:
    result = _run(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], cwd=root)
    return result.stdout.strip() if result.returncode == 0 else None


def _review_tips_match(root: Path, branch: str, ref: str, expected: tuple[str, str]) -> bool:
    return (_commit_sha(root, f"refs/heads/{branch}"), _commit_sha(root, ref)) == expected


def _stale_review() -> dict:
    return {"ok": False, "stale": True, "stderr": "A branch tip changed. Review the conflict again."}


def _rev_count(root: Path, ref: str) -> Optional[int]:
    r = _run(["rev-list", "--count", ref], cwd=root)
    if r.returncode != 0:
        return None
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


def _is_pristine_room(root: Path, working_branch: str) -> bool:
    """True when the room repo still holds only the synthetic ``git_init`` start.

    Criteria (internal design notes §3.1): ``main`` has exactly one
    commit, its tree is empty or only ``.gitignore``, and ``working_branch``
    (when it exists) carries nothing beyond that commit. Such a room can have
    the external repo transplanted straight onto ``main`` with no
    ``--allow-unrelated-histories`` merge and no synthetic root left behind.
    """
    if _rev_count(root, RELEASE_BRANCH) != 1:
        return False

    ls = _run(["ls-tree", "-r", "--name-only", RELEASE_BRANCH], cwd=root)
    if ls.returncode != 0:
        return False
    # A room initialised under internal design notes Stage 1 has an *empty*
    # root tree; rooms created before it carry a single seed ``.gitignore`` in
    # the init commit — both still count as pristine.
    if any(p.strip() and p.strip() != ".gitignore" for p in ls.stdout.splitlines()):
        return False

    if _run(["rev-parse", "--verify", working_branch], cwd=root).returncode == 0:
        ahead = _rev_count(root, f"{RELEASE_BRANCH}..{working_branch}")
        if ahead is None or ahead > 0:
            return False
    return True


@_serialized(busy_dict=True)
def git_adopt_remote(
    room_id: str,
    remote_url: str,
    branch: str = "main",
    token: Optional[str] = None,
    working_branch: Optional[str] = None,
    force: bool = False,
) -> dict:
    """Adopt an external repository as the room's base (internal design notes §3).

    Distinct from ``git_fetch`` ("Pull from Git"): the first time a room is
    pointed at an existing remote, its local ``main`` is only the synthetic
    ``init: Legion Hub workspace`` commit and has nothing to do with the
    remote's ``main``. Merging the two with ``--allow-unrelated-histories``
    (what ``git_fetch`` does) leaves the room with two roots — the first
    Gate 2 then shows the whole external repo as the ``main..dev`` diff, and
    the promotion push carries a synthetic root plus a merge into the real
    repo (rejected under "require linear history").

    ``git_adopt_remote`` instead *transplants* the remote onto ``main`` when
    the room is still pristine, so no unrelated history is ever created:

    - Pristine room (``_is_pristine_room``) → hard reset: ``main`` and
      ``working_branch`` both point at the fetched tip; the synthetic init
      commit is dropped (it was never pushed anywhere).
    - Room with real work → nothing is rewritten: ``main`` fast-forwards to
      the fetched tip only when it is an ancestor, otherwise ``diverged=True``
      and ``main`` is left alone. ``working_branch`` content is pulled in
      separately by the caller through ``git_fetch``.
    - ``force=True`` (an explicit human choice) allows the transplant on a
      room with work too, but only when no unfinished ``task/*`` branch exists.

    Returns ``{ok, adopted, diverged, main_sha, working_sha, dropped_init,
    stdout, stderr}``.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        git_init(room_id, working_branch)
    previous_working = working_branch_of(root)
    working_branch = _working_branch_arg(root, working_branch)

    fetch_url = remote_url
    auth = (remote_url, token)
    # Plain ``git fetch <url> <branch>`` — no ``:FETCH_HEAD`` dst refspec, which
    # would create ``refs/heads/FETCH_HEAD`` and make every later
    # ``FETCH_HEAD`` an ambiguous ref for ``branch -f`` / ``checkout -B``.
    r_fetch = _run(["fetch", fetch_url, branch], cwd=root, auth=auth)
    if r_fetch.returncode != 0:
        return {
            "ok": False, "adopted": False, "diverged": False,
            "stdout": r_fetch.stdout[:4096], "stderr": r_fetch.stderr[:4096],
        }

    out, err = r_fetch.stdout, r_fetch.stderr
    remote_userfiles = git_ls_remote_userfiles(root, "FETCH_HEAD")
    local_userfiles = git_ls_tracked_userfiles_details(room_id)
    if remote_userfiles or local_userfiles:
        # The policy check must happen before checkout/branch -f. A remote
        # which already owns .userfiles is an explicit policy conflict, not a
        # merge problem, and adopting it must never overwrite or delete the
        # room's existing private files.
        ensure_userfiles_dir(room_id, root=root)
        _write_local_excludes(root)
        chown_workspace_tree(room_id)
        entries = remote_userfiles or local_userfiles
        return {
            "ok": False,
            "adopted": False,
            "diverged": False,
            "userfiles_policy_conflict": True,
            "userfiles_tracked": [entry["path"] for entry in entries],
            "userfiles_tracked_details": entries,
            "stdout": out[:4096],
            "stderr": (
                "private-files policy conflict: the fetched repository tracks "
                f"{USERFILES_DIR}/; remove those paths from the remote or choose "
                "another private-files namespace before adopting"
            ),
        }
    # The branch the room worked on so far counts too: under another name it
    # may hold work the Hub-side name does not show yet.
    pristine = _is_pristine_room(root, working_branch) and (
        previous_working == working_branch or _is_pristine_room(root, previous_working)
    )
    task_branches = [
        b.strip().lstrip("* ").strip()
        for b in _run(["branch", "--list", "task/*"], cwd=root).stdout.splitlines()
        if b.strip()
    ]

    if pristine or force:
        if not pristine and task_branches:
            return {
                "ok": False, "adopted": False, "diverged": True,
                "stdout": out[:4096],
                "stderr": (
                    "refusing to transplant `main`: unfinished task branches "
                    f"present ({', '.join(task_branches)}) — merge or drop them first"
                ),
            }
        dropped_init = _rev_count(root, RELEASE_BRANCH) == 1
        # Hard transplant: the synthetic init commit is thrown away — it was
        # never pushed, so there is nothing to lose, and this is the only way
        # to keep unrelated histories from ever appearing.
        _run(["checkout", "-B", working_branch, "FETCH_HEAD"], cwd=root)
        _run(["branch", "-f", RELEASE_BRANCH, "FETCH_HEAD"], cwd=root)
        _record_working_branch(root, working_branch)
        if pristine and previous_working != working_branch:
            # The old integration branch only held the synthetic init commit
            # that was just dropped — do not leave it behind under its name.
            _run(["branch", "-D", previous_working], cwd=root)
        ensure_userfiles_dir(room_id, root=root)
        _write_local_excludes(root)
        chown_workspace_tree(room_id)
        logger.info(
            "git_adopt_remote: transplanted room %s onto remote %s (force=%s)",
            room_id, branch, force,
        )
        return {
            "ok": True, "adopted": True, "diverged": False,
            "dropped_init": bool(dropped_init),
            "main_sha": _rev(root, RELEASE_BRANCH),
            "working_sha": _rev(root, working_branch),
            "stdout": out[:4096], "stderr": err[:4096],
        }

    # Room with real work, no force: never rewrite history.
    is_ancestor = _run(
        ["merge-base", "--is-ancestor", RELEASE_BRANCH, "FETCH_HEAD"], cwd=root
    ).returncode == 0
    if not is_ancestor:
        return {
            "ok": True, "adopted": False, "diverged": True,
            "main_sha": _rev(root, RELEASE_BRANCH),
            "working_sha": _rev(root, working_branch),
            "stdout": out[:4096],
            "stderr": "local `main` has diverged from the remote; not moved",
        }

    current = git_current_branch(room_id)
    if current == RELEASE_BRANCH:
        ff = _run(["merge", "--ff-only", "FETCH_HEAD"], cwd=root)
    else:
        ff = _run(["branch", "-f", RELEASE_BRANCH, "FETCH_HEAD"], cwd=root)
    chown_workspace_tree(room_id)
    if ff.returncode != 0:
        return {
            "ok": False, "adopted": False, "diverged": False,
            "stdout": (out + ff.stdout)[:4096], "stderr": (err + ff.stderr)[:4096],
        }
    ensure_userfiles_dir(room_id, root=root)
    _write_local_excludes(root)
    if _local_branch_exists(root, working_branch):
        _record_working_branch(root, working_branch)
    return {
        "ok": True, "adopted": True, "diverged": False, "dropped_init": False,
        "main_sha": _rev(root, RELEASE_BRANCH),
        "working_sha": _rev(root, working_branch),
        "stdout": (out + ff.stdout)[:4096], "stderr": (err + ff.stderr)[:4096],
    }


@_serialized(busy_dict=True)
def git_fetch(
    room_id: str,
    remote_url: str,
    branch: str = "main",
    token: Optional[str] = None,
    target_branch: Optional[str] = None,
    *,
    fetch_only: bool = False,
) -> dict:
    """Fetch from remote and merge into ``target_branch`` (default: current HEAD).

    ``target_branch`` pins the branch the remote lands on and is checked out
    before the merge. Without it the remote would be merged into whatever HEAD
    happened to be — a "Pull from Git" fired while a task session holds
    ``task/<id>`` checked out would land the whole remote repository on that
    task branch instead of the integration branch. Callers pass the room's
    ``working_branch`` (user sync / push webhook) or ``release_branch``
    (catching ``main`` up after a PR-backed promotion).

    A fast-forward is taken when possible. When the workspace and the remote
    have diverged (the room made its own commits, or the room repo was seeded
    with an unrelated initial commit), a real merge commit is created instead
    of aborting — that is the whole point of "Pull from Git". Only a genuine
    content conflict, which cannot be resolved automatically, is reported back
    as ``conflict=True``; on that path the merge is rolled back so the
    workspace stays in a clean, usable state.

    The fetched tip is saved under ``refs/legion/remote/<branch>``. The result
    includes its ref and SHA, the local target tip, and any unresolved paths.
    In fetch-only mode no checkout, parking or merge is performed.
    ``branch`` in the result is the local branch the remote was merged into.
    ``gitignore_autoresolved`` (internal design notes Stage 4) is True when an
    ``add/add`` conflict on an untouched seed ``.gitignore`` was resolved in
    favour of the remote's file instead of being surfaced as a conflict.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "conflict": False, "stdout": "", "stderr": "no git repo in workspace"}

    fetch_url = remote_url
    auth = (remote_url, token)

    # Clear any stray refs/heads/FETCH_HEAD left by an earlier bad refspec so it
    # does not shadow the pseudo-ref below (or linger in the branch tree).
    if not fetch_only:
        _prune_pseudo_ref_branches(root)
    elif _run(["check-ref-format", f"refs/heads/{branch}"], cwd=root).returncode != 0:
        return {"ok": False, "stderr": "Invalid remote branch"}

    # Fetch first: a bad URL or token fails here without having touched the
    # working tree or moved HEAD. Plain ``git fetch <url> <branch>`` populates
    # the ``.git/FETCH_HEAD`` pseudo-ref. A ``:FETCH_HEAD`` destination would
    # create a real ``refs/heads/FETCH_HEAD`` branch (see ``git_adopt_remote``).
    source = f"refs/heads/{branch}" if fetch_only else branch
    r_fetch = _run(["fetch", "--", fetch_url, source], cwd=root, auth=auth)
    if r_fetch.returncode != 0:
        if fetch_only:
            return {"ok": False, "stderr": "Could not fetch the remote branch. Check the remote and its credentials."}
        return {
            "ok": False,
            "conflict": False,
            "stdout": r_fetch.stdout[:4096],
            "stderr": r_fetch.stderr[:4096],
        }

    out = r_fetch.stdout
    err = r_fetch.stderr

    remote_ref = f"{_REMOTE_REF_PREFIX}{branch}"
    remote_sha = _commit_sha(root, "FETCH_HEAD")
    if remote_sha is None:
        return {"ok": False, "conflict": False, "stdout": out[:4096], "stderr": "FETCH_HEAD is not a commit"}
    saved = _run(["update-ref", remote_ref, remote_sha], cwd=root)
    if saved.returncode != 0:
        return {"ok": False, "conflict": False, "stdout": out[:4096], "stderr": saved.stderr[:4096]}
    remote = {"remote_ref": remote_ref, "remote_sha": remote_sha}

    remote_userfiles = git_ls_remote_userfiles(root, remote_sha)
    local_userfiles = git_ls_tracked_userfiles_details(room_id)
    if remote_userfiles or local_userfiles:
        # Stop before parking, checkout, or merge. Besides preserving HEAD,
        # this keeps a remote .userfiles tree from replacing room-local files.
        if not fetch_only:
            ensure_userfiles_dir(room_id, root=root)
            _write_local_excludes(root)
            chown_workspace_tree(room_id)
        entries = remote_userfiles or local_userfiles
        return {
            "ok": False,
            "conflict": False,
            "userfiles_policy_conflict": True,
            "userfiles_tracked": [entry["path"] for entry in entries],
            "userfiles_tracked_details": entries,
            **remote,
            "stdout": out[:4096],
            "stderr": (
                "private-files policy conflict: the fetched repository tracks "
                f"{USERFILES_DIR}/; remove those paths from the remote or choose "
                "another private-files namespace before pulling"
            ),
        }

    if fetch_only:
        target = target_branch or git_current_branch(room_id)
        local_sha = _commit_sha(root, f"refs/heads/{target}") if target else None
        if local_sha is None:
            return {**_stale_review(), **remote}
        return {"ok": True, "branch": target, "local_sha": local_sha, **remote}

    # Whether the checked-out tree moved is what tells the caller that
    # running workspace services now serve stale code and need a restart.
    head_before = _head_sha(root)
    current = git_current_branch(room_id)
    target = target_branch or current

    park_error = _park_worktree_before_switch(root, room_id, current)
    if park_error:
        return {"ok": False, "conflict": False, "stdout": out[:4096], "stderr": park_error, **remote}
    current = git_current_branch(room_id)  # parking may have moved off `main`
    _drop_volatile_worktree_remnants(root)  # legacy `target` may still track them

    if target and target != current:
        if git_branch_exists(room_id, target):
            co = _run(["checkout", target], cwd=root)
        else:
            co = _run(["checkout", "-b", target], cwd=root)
        if co.returncode != 0:
            # Merging into the wrong branch is worse than not merging at all.
            logger.warning(
                "git_fetch: cannot check out target %s for room %s: %s",
                target, room_id, co.stderr[:300],
            )
            return {
                "ok": False,
                "conflict": False,
                "stdout": (out + co.stdout)[:4096],
                "stderr": (err + co.stderr)[:4096],
                "branch": current,
                **remote,
            }
        chown_workspace_tree(room_id)
    else:
        target = current

    def success(*, gitignore_autoresolved: bool = False) -> dict:
        ensure_userfiles_dir(room_id, root=root)
        _write_local_excludes(root)
        chown_workspace_tree(room_id)
        tip = _head_sha(root)
        return {
            "ok": True, "conflict": False, "stdout": out[:4096], "stderr": err[:4096],
            "branch": target, **remote, "local_sha": tip, "head_changed": tip != head_before,
            "gitignore_autoresolved": gitignore_autoresolved,
        }

    local_sha = _head_sha(root)
    # Try the cheap path first.
    r_ff = _run(["merge", "--ff-only", remote_sha], cwd=root)
    out += r_ff.stdout
    err += r_ff.stderr
    if r_ff.returncode == 0 or "up to date" in (r_ff.stdout + r_ff.stderr).lower():
        return success()

    # Diverged: create a real merge commit. --allow-unrelated-histories covers
    # a freshly-seeded room repo being pointed at an existing remote.
    r_merge = _run(
        [
            "merge", "--no-ff", "--no-edit", "--allow-unrelated-histories",
            "-m", f"merge: pull {branch} from remote", remote_sha,
        ],
        cwd=root,
    )
    out += r_merge.stdout
    err += r_merge.stderr

    if r_merge.returncode == 0:
        return success()

    # Capture conflicts before abort clears the index.
    conflict_files = _unmerged_paths(root)
    conflict = bool(conflict_files) or "CONFLICT" in (r_merge.stdout + r_merge.stderr)

    # internal design notes Stage 4: the one conflict that has a single right
    # answer is an ``add/add`` on a seed ``.gitignore`` nobody edited — take the
    # remote's file and finish the merge rather than dead-ending "Merge histories".
    if _seed_gitignore_autoresolvable(root, conflict_files):
        picked = _run(["checkout", "--theirs", "--", ".gitignore"], cwd=root)
        staged = _run(["add", "--", ".gitignore"], cwd=root)
        done = _run(["commit", "--no-edit"], cwd=root)
        if picked.returncode == 0 and staged.returncode == 0 and done.returncode == 0:
            out += done.stdout
            return success(gitignore_autoresolved=True)
        logger.warning(
            "git_fetch: seed .gitignore auto-resolve failed for room %s: %s",
            room_id, (picked.stderr + staged.stderr + done.stderr)[:300],
        )

    _run(["merge", "--abort"], cwd=root)
    chown_workspace_tree(room_id)
    if not conflict:
        logger.warning("git_fetch merge failed for room %s: %s", room_id, r_merge.stderr[:500])

    return {
        "ok": False,
        "conflict": conflict,
        "stdout": out[:4096],
        "stderr": err[:4096],
        "branch": target,
        **remote,
        "local_sha": local_sha,
        "conflict_files": conflict_files,
    }


# ---------------------------------------------------------------------------
# Branch listing (internal design notes Stage 1)
# ---------------------------------------------------------------------------

def git_list_branches(room_id: str, bases: dict[str, str]) -> list[dict]:
    """Return a list of branch descriptors for the room's git repo.

    ``bases`` maps branch name → base ref (e.g. ``{"task/48": "dev",
    "goal/12-slug": "dev"}``).  Computed by the backend from its Task/RoomGoal
    records.

    Each item:
      name, sha, committed_at, base, ahead, behind, merged

    Branches not in ``bases`` get base=None, ahead=None, behind=None,
    merged=None (shown, not hidden — we don't know their history).

    Pure read operation — no HEAD movement, no _serialized needed.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return []

    r = _run(
        ["for-each-ref",
         "--format=%(refname:short)%09%(objectname)%09%(committerdate:iso-strict)",
         "refs/heads"],
        cwd=root,
    )
    if r.returncode != 0:
        logger.warning("git_list_branches for-each-ref failed for room %s: %s", room_id, r.stderr[:300])
        return []

    result = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        name = parts[0]
        sha = parts[1]
        committed_at = parts[2] if len(parts) > 2 else None

        # Skip stray pseudo-ref branches (``refs/heads/FETCH_HEAD`` &c. — never
        # a real Legion branch, only ever an accident of a bad refspec). The
        # short name comes through as ``heads/FETCH_HEAD`` once ``.git/FETCH_HEAD``
        # makes the bare name ambiguous, so match on the last path segment.
        if name.rsplit("/", 1)[-1] in _PSEUDO_REF_BRANCH_NAMES:
            continue

        base = bases.get(name)
        ahead: Optional[int] = None
        behind: Optional[int] = None
        merged: Optional[bool] = None

        if base is not None:
            r_ahead = _run(
                ["rev-list", "--count", f"{base}..{name}"], cwd=root
            )
            r_behind = _run(
                ["rev-list", "--count", f"{name}..{base}"], cwd=root
            )
            if r_ahead.returncode == 0 and r_behind.returncode == 0:
                try:
                    ahead = int(r_ahead.stdout.strip())
                    behind = int(r_behind.stdout.strip())
                    merged = ahead == 0
                except ValueError:
                    pass

        result.append({
            "name": name,
            "sha": sha,
            "committed_at": committed_at,
            "base": base,
            "ahead": ahead,
            "behind": behind,
            "merged": merged,
        })

    return result


def git_ahead_behind_remote(room_id: str, branch: str) -> Optional[dict]:
    """Return ``{ahead, behind}`` of ``branch`` vs its remote tracking ref.

    Uses the already-fetched ``refs/remotes/origin/<branch>`` — no network
    call.  Returns None when the room has no repo, no remote ref exists, or
    the counts cannot be parsed.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return None

    remote_ref = f"refs/remotes/origin/{branch}"
    r_check = _run(["rev-parse", "--verify", remote_ref], cwd=root)
    if r_check.returncode != 0:
        return None

    r_ahead = _run(["rev-list", "--count", f"{remote_ref}..{branch}"], cwd=root)
    r_behind = _run(["rev-list", "--count", f"{branch}..{remote_ref}"], cwd=root)
    if r_ahead.returncode != 0 or r_behind.returncode != 0:
        return None
    try:
        return {
            "ahead": int(r_ahead.stdout.strip()),
            "behind": int(r_behind.stdout.strip()),
        }
    except ValueError:
        return None


# ── Catching ``main`` up with the remote
# (internal design notes Stage 5) ────────────────────

def _is_ancestor(root: Path, older: str, newer: str) -> bool:
    return _run(["merge-base", "--is-ancestor", older, newer], cwd=root).returncode == 0


def _count_between(root: Path, from_ref: str, to_ref: str) -> Optional[int]:
    """Commits ``to_ref`` has that ``from_ref`` does not (``from..to``)."""
    r = _run(["rev-list", "--count", f"{from_ref}..{to_ref}"], cwd=root)
    if r.returncode != 0:
        return None
    try:
        return int(r.stdout.strip())
    except ValueError:
        return None


@_serialized(busy_dict=True)
def git_catch_up_release(room_id: str, remote_url: str, token: Optional[str] = None) -> dict:
    """Fast-forward the local ``main`` to the remote's ``main``, never merging.

    Several rooms can publish into one remote ``main`` (one per working
    branch). After another room published, this room's ``main`` is behind and
    its next promotion would end in a non-fast-forward push. Called before a
    promotion is opened and before it is merged.

    The fetched tip is saved under ``refs/legion/remote/main`` (the same ref a
    Pull writes). ``state``:

    * ``up_to_date`` — local and remote ``main`` are the same commit;
    * ``fast_forwarded`` — local ``main`` was behind and now matches the remote;
    * ``ahead`` — local ``main`` has commits the remote does not (a promotion
      not pushed yet); left alone;
    * ``diverged`` — both sides have their own commits; left alone, a Pull of
      ``main`` resolves it through the divergence review;
    * ``userfiles_conflict`` — the remote ``main`` tracks the private-files
      directory; left alone, a Pull reports it;
    * ``no_local_main`` — the room has no ``main`` yet; nothing to compare.

    ``ok=False`` only when the fetch itself failed (network, credentials) or
    the fast-forward could not be written. HEAD never moves: the ref is
    updated in place unless ``main`` is the checked-out branch.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "state": "no_repo", "error": "no git repo"}
    if not remote_url:
        return {"ok": False, "state": "no_remote", "error": "remote url is required"}

    fetched = _run(
        ["fetch", "--", remote_url, f"refs/heads/{RELEASE_BRANCH}"],
        cwd=root, auth=(remote_url, token),
    )
    if fetched.returncode != 0:
        logger.warning(
            "git_catch_up_release: fetch failed for room %s: %s", room_id, fetched.stderr[:300]
        )
        return {
            "ok": False, "state": "fetch_failed",
            "error": "Could not fetch main from the remote. Check the remote and its credentials.",
        }
    remote_sha = _commit_sha(root, "FETCH_HEAD")
    if remote_sha is None:
        return {"ok": False, "state": "fetch_failed", "error": "FETCH_HEAD is not a commit"}
    _run(["update-ref", f"{_REMOTE_REF_PREFIX}{RELEASE_BRANCH}", remote_sha], cwd=root)

    local_sha = _commit_sha(root, f"refs/heads/{RELEASE_BRANCH}")
    result = {"ok": True, "local_sha": local_sha, "main_sha": local_sha, "remote_sha": remote_sha}
    if local_sha is None:
        return {**result, "state": "no_local_main"}
    if local_sha == remote_sha:
        return {**result, "state": "up_to_date"}
    if _is_ancestor(root, remote_sha, local_sha):
        return {**result, "state": "ahead"}
    if not _is_ancestor(root, local_sha, remote_sha):
        return {**result, "state": "diverged"}
    if git_ls_remote_userfiles(root, remote_sha):
        return {**result, "state": "userfiles_conflict"}

    if git_current_branch(room_id) == RELEASE_BRANCH:
        ff = _run(["merge", "--ff-only", remote_sha], cwd=root)
        chown_workspace_tree(room_id)
    else:
        ff = _run(
            ["update-ref", f"refs/heads/{RELEASE_BRANCH}", remote_sha, local_sha], cwd=root
        )
    if ff.returncode != 0:
        logger.warning(
            "git_catch_up_release: fast-forward failed for room %s: %s", room_id, ff.stderr[:300]
        )
        return {**result, "ok": False, "state": "ff_failed", "error": ff.stderr[:300]}
    logger.info(
        "git_catch_up_release: room %s main %s → %s", room_id, local_sha[:8], remote_sha[:8]
    )
    return {**result, "state": "fast_forwarded", "main_sha": remote_sha}


def release_remote_status(room_id: str, working_branch: str) -> dict:
    """What the last fetched remote ``main`` carries that this room lacks.

    Pure read of ``refs/legion/remote/main`` (written by a Pull of ``main``
    and by ``git_catch_up_release``) — no network. ``main_behind``: commits
    the local ``main`` lacks; ``working_behind``: commits ``working_branch``
    lacks, i.e. what a Pull would bring into it. ``None`` where unknown.
    """
    root = _workspace_root(room_id)
    empty = {"ok": False, "remote_sha": None, "main_behind": None, "working_behind": None}
    if not (root / ".git").is_dir():
        return empty
    remote_ref = f"{_REMOTE_REF_PREFIX}{RELEASE_BRANCH}"
    remote_sha = _commit_sha(root, remote_ref)
    if remote_sha is None:
        return empty
    main_behind = (
        _count_between(root, f"refs/heads/{RELEASE_BRANCH}", remote_ref)
        if _local_branch_exists(root, RELEASE_BRANCH) else None
    )
    working_behind = (
        _count_between(root, f"refs/heads/{working_branch}", remote_ref)
        if working_branch and _local_branch_exists(root, working_branch) else None
    )
    return {
        "ok": True, "remote_sha": remote_sha,
        "main_behind": main_behind, "working_behind": working_behind,
    }


# ── Divergence review (internal design notes Stage 1) ────────────────────

# Caps on what a divergence report carries into the ChatMaster turn — a large
# diverged branch would otherwise blow the model's context.
_DIVERGENCE_MAX_FILES = 12
_DIVERGENCE_MAX_DIFF_LINES = 200
_DIVERGENCE_MAX_COMMITS = 40


def _log_between(root: Path, rev_range: str) -> list[dict]:
    fmt = "%H%x1f%s%x1f%an%x1f%aI"
    r = _run(
        ["log", f"--max-count={_DIVERGENCE_MAX_COMMITS}", f"--format={fmt}", rev_range],
        cwd=root,
    )
    if r.returncode != 0:
        return []
    out: list[dict] = []
    for line in r.stdout.strip().splitlines():
        parts = line.split("\x1f")
        if len(parts) < 4:
            continue
        out.append({
            "sha": parts[0], "short_sha": parts[0][:8],
            "message": parts[1], "author": parts[2], "timestamp": parts[3],
        })
    return out


def _commit_meta(root: Path, ref: str) -> dict:
    r = _run(["show", "-s", "--format=%H%x1f%aI%x1f%s", ref], cwd=root)
    if r.returncode != 0 or not r.stdout.strip():
        return {"sha": None, "timestamp": None, "subject": None}
    parts = r.stdout.strip().split("\x1f")
    return {
        "sha": parts[0],
        "timestamp": parts[1] if len(parts) > 1 else None,
        "subject": parts[2] if len(parts) > 2 else None,
    }


def _truncate_diff(text: str, max_lines: int = _DIVERGENCE_MAX_DIFF_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    kept.append(f"… (+{len(lines) - max_lines} lines truncated)")
    return "\n".join(kept)


def git_divergence_report(room_id: str, branch: str, base: str) -> dict:
    """Read-only analysis of how ``branch`` and ``base`` have diverged.

    internal design notes Stage 1 — feeds the ChatMaster "Divergence review"
    turn. Pure read: no HEAD movement, no working-tree write, so no
    ``_serialized`` lock. ``git merge-tree --write-tree`` computes the merge
    result in the object store without touching the checkout.

    Returns::

        {
          "ok": bool, "error": str | None,
          "branch": str, "base": str,
          "merge_base": {"sha", "timestamp", "subject"},
          "base_head": {...}, "branch_head": {...},
          "ahead": int, "behind": int,          # branch vs base
          "branch_commits": [ {sha, message, ...} ],   # base..branch
          "base_commits": [ ... ],                      # branch..base
          "branch_files": [path], "base_files": [path], # changed since merge-base
          "overlap_files": [path],
          "conflict": bool,
          "conflict_files": [path],
          "conflict_details": [ {path, base_diff, branch_diff} ],
          "merge_messages": str,
        }
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "error": "no git repo", "branch": branch, "base": base}

    branch_sha = _commit_sha(root, branch)
    base_sha = _commit_sha(root, base)
    if not (branch_sha and base_sha):
        return {"ok": False, "error": f"ref not found: {branch if not branch_sha else base}",
                "branch": branch, "base": base}

    remote_review = branch.startswith(_REMOTE_REF_PREFIX)
    mb_run = _run(["merge-base", base_sha, branch_sha], cwd=root)
    merge_base_sha = mb_run.stdout.strip() if mb_run.returncode == 0 else None
    if not merge_base_sha and (not remote_review or mb_run.returncode != 1):
        return {"ok": False, "error": "no common ancestor", "branch": branch, "base": base}
    diff_base = merge_base_sha
    if diff_base is None:
        empty_tree = _run(["hash-object", "-t", "tree", "-w", "--stdin"], cwd=root, input="")
        if empty_tree.returncode != 0:
            return {"ok": False, "error": "could not read the empty tree", "branch": branch, "base": base}
        diff_base = empty_tree.stdout.strip()

    ahead = behind = 0
    ab = _run(["rev-list", "--left-right", "--count", f"{base_sha}...{branch_sha}"], cwd=root)
    if ab.returncode == 0:
        try:
            behind_s, ahead_s = ab.stdout.split()
            behind, ahead = int(behind_s), int(ahead_s)
        except ValueError:
            pass

    def _changed_files(sha: str) -> list[str]:
        r = _run(["diff", "--name-only", "-z", diff_base, sha, "--"], cwd=root)
        return r.stdout.rstrip("\0").split("\0") if r.returncode == 0 and r.stdout else []

    def _binary_files(sha: str) -> list[str]:
        r = _run(["diff", "--numstat", "--no-renames", "-z", diff_base, sha, "--"], cwd=root)
        return [
            entry.removeprefix("-\t-\t")
            for entry in r.stdout.split("\0")
            if entry.startswith("-\t-\t")
        ] if r.returncode == 0 else []

    branch_files = _changed_files(branch_sha)
    base_files = _changed_files(base_sha)
    overlap = sorted(set(branch_files) & set(base_files))

    # Merge preview without touching the working tree.
    args = ["merge-tree", "--write-tree", "--name-only", "--messages", "-z"]
    if not merge_base_sha:
        args.append("--allow-unrelated-histories")
    mt = _run([*args, base_sha, branch_sha], cwd=root)
    conflict = mt.returncode == 1
    conflict_files: list[str] = []
    merge_messages = ""
    merge_tree_oid = ""
    if mt.returncode in (0, 1):
        paths, _, messages = mt.stdout.partition("\0\0")
        merge_tree_oid, *conflict_files = paths.rstrip("\0").split("\0")
        fields = messages.rstrip("\0").split("\0")
        details = []
        offset = 0
        while offset < len(fields) and fields[offset].isdigit():
            # Each message has a path count, paths, a short type and a description.
            end = offset + int(fields[offset]) + 2
            if end >= len(fields):
                break
            details.append(fields[end].strip())
            offset = end + 1
        merge_messages = "\n".join(details)
    else:
        merge_messages = (mt.stderr or "").strip()

    # Does the branch still carry anything the base does not already have? After
    # a rework task lands, merging the diverged branch into base would produce
    # exactly base's own tree — nothing to salvage, safe to just delete.
    base_tree = _run(["rev-parse", f"{base_sha}^{{tree}}"], cwd=root).stdout.strip()
    redundant = bool(merge_tree_oid) and not conflict and merge_tree_oid == base_tree

    detail_paths = (conflict_files or overlap)[:_DIVERGENCE_MAX_FILES]
    conflict_details: list[dict] = []
    for path in detail_paths:
        base_diff = _run(["diff", diff_base, base_sha, "--", f":(literal){path}"], cwd=root).stdout
        branch_diff = _run(["diff", diff_base, branch_sha, "--", f":(literal){path}"], cwd=root).stdout
        conflict_details.append({
            "path": path,
            "base_diff": _truncate_diff(base_diff),
            "branch_diff": _truncate_diff(branch_diff),
        })

    return {
        "ok": True,
        "error": None,
        "branch": branch,
        "base": base,
        "merge_base": _commit_meta(root, merge_base_sha) if merge_base_sha else {},
        "base_head": _commit_meta(root, base_sha),
        "branch_head": _commit_meta(root, branch_sha),
        "ahead": ahead,
        "behind": behind,
        "branch_commits": _log_between(root, f"{base_sha}..{branch_sha}"),
        "base_commits": _log_between(root, f"{branch_sha}..{base_sha}"),
        "branch_files": branch_files,
        "base_files": base_files,
        "branch_binary_files": _binary_files(branch_sha) if remote_review else [],
        "base_binary_files": _binary_files(base_sha) if remote_review else [],
        "overlap_files": overlap,
        "conflict": conflict,
        "conflict_files": conflict_files,
        "conflict_details": conflict_details,
        "merge_messages": merge_messages[:4000],
        "redundant": redundant,
    }


@_serialized(busy_dict=True)
def git_prepare_pull_rework(
    room_id: str, task_branch: str, branch: str, remote_ref: str,
    expected_tips: tuple[str, str], conflict_files: list[str],
) -> dict:
    """Seed a task branch with both histories without switching the room checkout."""
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "stderr": "no git repo"}
    task_ref = f"refs/heads/{task_branch}"
    local_ref = f"refs/heads/{branch}"
    if (branch == RELEASE_BRANCH or not remote_ref.startswith(_REMOTE_REF_PREFIX)
            or not task_branch.startswith("task/")
            or any(_run(["check-ref-format", ref], cwd=root).returncode
                   for ref in (task_ref, local_ref, remote_ref))):
        return {"ok": False, "stderr": "Invalid branches for Pull conflict rework."}
    if not _review_tips_match(root, branch, remote_ref, expected_tips):
        return _stale_review()
    local_sha, remote_sha = expected_tips
    if (git_ls_remote_userfiles(root, local_sha) or git_ls_remote_userfiles(root, remote_sha)
            or git_ls_tracked_userfiles_details(room_id)):
        return {
            "ok": False, "userfiles_policy_conflict": True,
            "stderr": f"private-files policy conflict: a merge side or the index tracks {USERFILES_DIR}/",
        }

    with TemporaryDirectory(prefix="legion-pull-rework-") as temp:
        hand_to_git_user(temp)  # git may run as the room's user (git_exec)
        checkout = Path(temp) / "merge"
        result = _run(["worktree", "add", "--detach", str(checkout), local_sha], cwd=root)
        if result.returncode:
            return {"ok": False, "stderr": result.stderr[:4096]}
        try:
            result = _run([
                "merge", "--no-ff", "--allow-unrelated-histories", "-X", "ours",
                "-m", f"Merge {remote_ref} for Pull conflict rework", "--", remote_sha,
            ], cwd=checkout)
            remaining = _unmerged_paths(checkout) if result.returncode else []
            if remaining:
                # -X ours handles overlapping hunks. For conflicts such as
                # modify/delete, keep the local file or its local deletion.
                result = _keep_local_rework_paths(checkout, local_sha, remaining)
                if not result.returncode:
                    result = _run(["commit", "--no-edit"], cwd=checkout)
            if result.returncode:
                return {
                    "ok": False, "conflict": bool(remaining),
                    "stderr": result.stderr[:4096], "stdout": result.stdout[:4096],
                }
            sha = _head_sha(checkout)
            if sha is None:
                return {"ok": False, "stderr": "Could not read the rework merge commit."}
        finally:
            removed = _run(["worktree", "remove", "--force", str(checkout)], cwd=root)
            if removed.returncode:
                logger.warning("git_prepare_pull_rework: could not remove temporary checkout for %s", room_id)

    # Pin the diff to the same two commits as the seed merge.
    merge_base = _run(["merge-base", local_sha, remote_sha], cwd=root)
    if merge_base.returncode == 1:
        merge_base = _run(["hash-object", "-t", "tree", "-w", "--stdin"], cwd=root, input="")
    if merge_base.returncode:
        return {"ok": False, "stderr": "Could not read the Pull conflict merge base."}
    diff_base = merge_base.stdout.strip()
    paths = list(dict.fromkeys([*conflict_files, *remaining]))
    patches = _pull_rework_patches(root, diff_base, remote_sha, paths)

    if not _review_tips_match(root, branch, remote_ref, expected_tips):
        return _stale_review()
    existing = _commit_sha(root, task_ref)
    if existing:
        # A previous request may have seeded the ref before the database commit.
        def signature(commit: str) -> str:
            return _run(["show", "-s", "--format=%T %P", commit], cwd=root).stdout.strip()

        if signature(existing) != signature(sha):
            return {"ok": False, "existing_sha": existing,
                    "stderr": "The rework branch already contains different changes."}
        sha = existing
    else:
        transaction = f"verify {local_ref}\0{local_sha}\0verify {remote_ref}\0{remote_sha}\0create {task_ref}\0{sha}\0"
        created = _run(["update-ref", "--stdin", "-z"], cwd=root, input=transaction)
        if created.returncode:
            if not _review_tips_match(root, branch, remote_ref, expected_tips):
                return _stale_review()
            return {"ok": False, "stderr": created.stderr[:4096]}
    chown_workspace_tree(room_id)
    return {
        "ok": True, "sha": sha, "diff_base": diff_base,
        "remote_patches": patches, "conflict_files": paths,
    }


def _keep_local_rework_paths(
    checkout: Path, local_sha: str, paths: list[str],
) -> subprocess.CompletedProcess:
    pathspecs = [f":(literal){path}" for path in paths]
    local_files = _run(["ls-tree", "-r", "--name-only", "-z", local_sha, "--", *pathspecs], cwd=checkout)
    if local_files.returncode:
        return local_files
    # Removing the unmerged index entries first also handles files absent on
    # the local side, which git restore refuses to remove while unmerged.
    removed = _run(["rm", "-rf", "--ignore-unmatch", "--", *pathspecs], cwd=checkout)
    if removed.returncode or not local_files.stdout:
        return removed
    return _run([
        "restore", f"--source={local_sha}", "--staged", "--worktree", "--",
        *(f":(literal){path}" for path in local_files.stdout.rstrip("\0").split("\0")),
    ], cwd=checkout)


def _pull_rework_patches(root: Path, diff_base: str, remote_sha: str, paths: list[str]) -> dict[str, str]:
    patches = {}
    budget = 32000
    for path in paths:
        if budget <= 0:
            break
        diff = _run([
            "diff", "--no-ext-diff", "--no-textconv", diff_base, remote_sha, "--", f":(literal){path}",
        ], cwd=root)
        if diff.returncode:
            continue
        patch = _truncate_diff(diff.stdout)
        limit = min(budget, 8000)
        patches[path] = patch if len(patch) <= limit else patch[:limit] + "\n[Diff truncated]"
        budget -= len(patches[path])
    return patches


def _backup_branch_tip(root: Path, branch: str, sha: str) -> tuple[str, subprocess.CompletedProcess]:
    safe = branch.replace("/", "-").replace("~", "-").replace("^", "-").replace(":", "-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup_ref = f"refs/legion/trash/{safe}@{ts}"
    # Require a new ref so a retry cannot overwrite a previous backup.
    return backup_ref, _run(["update-ref", backup_ref, sha, ""], cwd=root)


@_serialized(busy_dict=True)
def git_reset_branch_to(
    room_id: str, branch: str, ref: str, *, expected_tips: tuple[str, str] | None = None,
) -> dict:
    """Move an existing local branch to a commit after saving its current tip."""
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "stderr": "no git repo"}

    if expected_tips is not None and not _review_tips_match(root, branch, ref, expected_tips):
        return _stale_review()
    head_before = _head_sha(root)
    branch_ref = f"refs/heads/{branch}"
    local = _run(["show-ref", "--verify", "--hash", branch_ref], cwd=root)
    if local.returncode != 0:
        return {"ok": False, "stderr": f"local branch does not exist: {branch!r}"}
    local_sha = local.stdout.strip()
    destination = _run(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], cwd=root)
    if destination.returncode != 0:
        return {"ok": False, "stderr": destination.stderr[:4096]}
    sha = destination.stdout.strip()
    if sha == local_sha:
        return {"ok": True, "branch": branch, "sha": sha}

    if git_ls_remote_userfiles(root, sha) or git_ls_remote_userfiles(root, local_sha):
        return {"ok": False, "stderr": f"private-files policy conflict: a reset tree tracks {USERFILES_DIR}/"}

    current = git_current_branch(room_id)
    was_checked_out = branch == current
    if was_checked_out:
        if git_ls_tracked_userfiles_details(room_id):
            return {"ok": False, "stderr": f"private-files policy conflict: the index tracks {USERFILES_DIR}/"}
        park_error = _park_worktree_before_switch(root, room_id, current)
        if park_error:
            return {"ok": False, "stderr": park_error}
        current = git_current_branch(room_id)  # parking may have moved off main
        status = _run(["status", "--porcelain"], cwd=root)
        if status.returncode != 0 or status.stdout:
            return {"ok": False, "stderr": status.stderr or "could not park local changes"}
        local_sha = _rev(root, branch_ref)
        if local_sha is None:
            return {"ok": False, "stderr": f"local branch disappeared: {branch!r}"}

    if expected_tips is not None and (local_sha, sha) != expected_tips:
        return _stale_review()
    backup_ref, saved = _backup_branch_tip(root, branch, local_sha)
    if saved.returncode != 0:
        return {"ok": False, "stderr": saved.stderr[:4096]}

    if branch == current:
        result = _run(["reset", "--hard", sha], cwd=root)
    else:
        result = _run(["branch", "-f", "--", branch, sha], cwd=root)
    if was_checked_out:
        chown_workspace_tree(room_id)
    return {
        "ok": result.returncode == 0,
        "branch": branch,
        "sha": sha if result.returncode == 0 else None,
        "backup_ref": backup_ref,
        "head_changed": _head_sha(root) != head_before,
        "stderr": result.stderr[:4096],
    }


@_serialized(busy_dict=True)
def git_discard_branch(room_id: str, branch: str, working_branch: Optional[str] = None) -> dict:
    """Tag ``branch`` under ``refs/legion/trash/`` then delete it.

    internal design notes Stage 2 — the "Delete branch" divergence
    resolution. The tag makes the delete recoverable (``git branch <name>
    refs/legion/trash/<...>``). Refuses to delete the branch that is currently
    checked out. Serialised: it moves refs the branch tree depends on.

    ``working_branch`` is the room's integration branch (omitted → the one
    the repo records): it is never discarded and is where the working tree
    steps to when ``branch`` is checked out.

    Returns ``{ok, deleted, backup_ref, busy, stderr}``.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "deleted": False, "stderr": "no git repo"}
    integration = _working_branch_arg(root, working_branch)
    if not branch or branch in (RELEASE_BRANCH, integration):
        return {"ok": False, "deleted": False, "stderr": f"refusing to discard {branch!r}"}
    if not git_branch_exists(room_id, branch):
        # Idempotent: nothing to do.
        return {"ok": True, "deleted": False, "stderr": "branch does not exist"}
    if git_current_branch(room_id) == branch:
        # The workspace is sitting on the branch the user wants gone (the usual
        # case right after a task/goal finished on it). Step off onto the
        # integration branch first — parking any uncommitted work as a commit
        # on the doomed branch, which the backup tag below then preserves — so
        # "delete branch" is not a dead end that needs a manual checkout.
        park_error = _park_worktree_before_switch(root, room_id, branch)
        if park_error:
            return {"ok": False, "deleted": False, "stderr": park_error}
        _drop_volatile_worktree_remnants(root)
        co = _run(["checkout", integration], cwd=root)
        if co.returncode != 0:
            logger.warning(
                "git_discard_branch: cannot leave %s for room %s: %s",
                branch, room_id, co.stderr[:300],
            )
            return {
                "ok": False, "deleted": False,
                "stderr": (
                    f"the branch is checked out in the working tree and leaving it for "
                    f"`{integration}` failed: {co.stderr[:300]}"
                ),
            }
        chown_workspace_tree(room_id)  # checkout rewrote worktree files as root

    backup_ref, tag = _backup_branch_tip(root, branch, f"refs/heads/{branch}")
    if tag.returncode != 0:
        return {"ok": False, "deleted": False, "stderr": tag.stderr[:500]}

    r = _run(["branch", "-D", branch], cwd=root)
    if r.returncode != 0:
        return {"ok": False, "deleted": False, "backup_ref": backup_ref, "stderr": r.stderr[:500]}
    return {"ok": True, "deleted": True, "backup_ref": backup_ref, "stderr": ""}
