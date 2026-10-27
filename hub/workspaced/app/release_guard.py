"""Keep ``main`` and release tags out of an agent's reach.

workspaced is the only party that may move ``main`` (the ``dev → main``
promotion, a restore, a Pull) — ``git_commit`` already refuses an agent's
write there. An agent's bash runs git directly in the room container, though,
and nothing stopped ``git checkout main && git commit`` or ``git push origin
main``. Two hooks, installed into every room repo, close that path:

* ``reference-transaction`` refuses any change to ``refs/heads/main`` and any
  move or deletion of an existing tag (release tags are immutable). It runs for
  every ref update — commit, merge, reset, ``update-ref``, ``branch -f`` — and
  ``--no-verify`` does not skip it.
* ``pre-push`` refuses pushing ``main`` or a tag from the container; those
  reach the forge only through ``publish_for_deploy`` and ``release_tag``.

workspaced's own git runs with ``LEGION_GIT_TRUSTED=1`` and passes through.
This guards against an agent's mistake, not against an agent set on getting
round it; the forge-side checks in the Hub (``publish_for_deploy`` asks a
person when ``main`` carries commits that skipped the promotion) are the
backstop.

Release tags are created here as well: annotated, on a commit of ``main``,
never moved once they exist, and pushed one at a time.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from app.git_ops import _inject_token, _run, _serialized, _workspace_root

logger = logging.getLogger("workspaced.release_guard")

TRUSTED_ENV = "LEGION_GIT_TRUSTED"
_HOOK_MARKER = "# legion-ref-guard v1"

_REFERENCE_TRANSACTION = f"""#!/bin/sh
{_HOOK_MARKER} - installed by Legion; keeps main and release tags safe.
[ "$1" = "prepared" ] || exit 0
[ "${{{TRUSTED_ENV}:-}}" = "1" ] && exit 0
status=0
while read -r old new ref; do
  # A deletion arrives as an all-zero new value, often with an all-zero old
  # one too (git does not always know it); only a real no-op is skipped.
  case "$new" in *[!0]*) deleting=0 ;; *) deleting=1 ;; esac
  [ "$old" = "$new" ] && [ $deleting = 0 ] && continue
  case "$ref" in
    refs/heads/main)
      echo "legion: branch 'main' moves only through the dev -> main promotion in Legion; agents never commit, merge or reset it." >&2
      echo "legion: switch back to your task branch (git checkout task/<task id>, or dev) and commit there. To bring in changes from the remote repository, ask the user to press Pull on the Git tab." >&2
      status=1
      ;;
    refs/tags/*)
      case "$old:$deleting" in
        *[!0]*:*|*:1)
          echo "legion: tag '${{ref#refs/tags/}}' already exists; release tags are immutable and are never moved or deleted. Use a new version number (the release_tag tool creates and publishes it)." >&2
          status=1
          ;;
      esac
      ;;
  esac
done
exit $status
"""

_PRE_PUSH = f"""#!/bin/sh
{_HOOK_MARKER} - installed by Legion; main and tags are published by Legion.
[ "${{{TRUSTED_ENV}:-}}" = "1" ] && exit 0
status=0
while read -r local_ref local_sha remote_ref remote_sha; do
  case "$remote_ref" in
    refs/heads/main)
      echo "legion: 'main' is published only by the publish_for_deploy tool, after the dev -> main promotion. Do not push it with git." >&2
      status=1
      ;;
    refs/tags/*)
      echo "legion: tags are published only by the release_tag tool. Do not push them with git." >&2
      status=1
      ;;
  esac
done
exit $status
"""

HOOKS = {"reference-transaction": _REFERENCE_TRANSACTION, "pre-push": _PRE_PUSH}

#: ``v1.2.3`` with an optional pre-release suffix (``v1.2.3-rc.1``).
RELEASE_TAG_RE = re.compile(r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z][0-9A-Za-z.-]*)?$")
RELEASE_BRANCH = "main"
INTEGRATION_BRANCH = "dev"


def _hook_current(path: Path, script: str) -> bool:
    try:
        return path.read_text(encoding="utf-8") == script
    except (OSError, UnicodeDecodeError):
        return False


def ensure_ref_guard(root: Path) -> bool:
    """Install the guard hooks into ``root/.git/hooks`` (idempotent).

    Returns ``True`` when a file was written. A repo with its own
    ``core.hooksPath`` (husky and friends) never reads ``.git/hooks``, and a
    project hook of the same name is the project's: both are left alone and
    logged, never overwritten. Best-effort — a filesystem error is swallowed.
    """
    git_dir = root / ".git"
    if not git_dir.is_dir():
        return False
    if all(_hook_current(git_dir / "hooks" / name, script) for name, script in HOOKS.items()):
        return False  # the common case: one stat + read per hook, no git call
    hooks_path = _run(["config", "--local", "--get", "core.hooksPath"], cwd=root).stdout.strip()
    if hooks_path:
        logger.info("ref guard: %s sets core.hooksPath=%s; hooks not installed", root, hooks_path)
        return False
    written = False
    try:
        hooks = git_dir / "hooks"
        hooks.mkdir(exist_ok=True)
        for name, script in HOOKS.items():
            path = hooks / name
            if path.exists():
                current = path.read_text(encoding="utf-8", errors="replace")
                if current == script:
                    continue
                if _HOOK_MARKER.split(" v")[0] not in current:
                    logger.warning("ref guard: %s has its own %s hook; left alone", root, name)
                    continue
            path.write_text(script, encoding="utf-8", newline="\n")
            path.chmod(0o755)
            written = True
    except OSError:
        logger.warning("ref guard: could not install hooks in %s", root, exc_info=True)
    return written


def _resolve(root: Path, rev: str) -> Optional[str]:
    if not rev or rev.startswith("-"):
        return None
    r = _run(["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], cwd=root)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def main_bypass_commits(room_id: str, exclude: Optional[list[str]] = None, limit: int = 20) -> dict:
    """Commits on ``main`` that did not come through the promotion.

    Every legitimate commit on ``main`` is either reachable from ``dev`` (the
    promotion merges it in), already on the forge (a Pull brought it) or a
    merge commit (the promotion / Pull merge itself). What is left — a
    non-merge commit that only ``main`` has — was made on ``main`` directly:
    by an agent that got round the guard, or by a restore from the Git tab.
    ``exclude`` are the commits the forge already has (its ``main`` head);
    unknown ones are ignored. Returns ``{ok, commits}``.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "commits": [], "error": "no git repo"}
    if _resolve(root, f"refs/heads/{RELEASE_BRANCH}") is None:
        return {"ok": True, "commits": []}
    nots = [f"^{sha}" for sha in (_resolve(root, x) for x in (exclude or [])) if sha]
    if _resolve(root, f"refs/heads/{INTEGRATION_BRANCH}"):
        nots.append(f"^refs/heads/{INTEGRATION_BRANCH}")
    fmt = "%H%x1f%s%x1f%an%x1f%aI"
    r = _run(
        ["log", "--no-merges", f"--max-count={max(1, min(limit, 200))}", f"--format={fmt}",
         f"refs/heads/{RELEASE_BRANCH}", *nots],
        cwd=root,
    )
    if r.returncode != 0:
        return {"ok": False, "commits": [], "error": r.stderr[:300]}
    commits = []
    for line in r.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            commits.append({"sha": parts[0], "short_sha": parts[0][:8], "message": parts[1],
                            "author": parts[2], "timestamp": parts[3]})
    return {"ok": True, "commits": commits}


def latest_release_tag(room_id: str) -> dict:
    """Highest final semver tag reachable from main, excluding prereleases."""
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "code": "tag_no_repo", "error": "the workspace has no git repository"}
    result = _run(["tag", "--merged", f"refs/heads/{RELEASE_BRANCH}", "--list", "v*"], cwd=root)
    if result.returncode:
        return {"ok": False, "code": "tag_failed", "error": result.stderr.strip()[:300]}
    versions = []
    for tag in result.stdout.splitlines():
        match = re.fullmatch(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", tag)
        if match:
            versions.append((tuple(map(int, match.groups())), tag))
    if not versions:
        return {"ok": True, "latest": None, "patch": "v0.1.0", "minor": "v0.1.0"}
    (major, minor, patch), tag = max(versions)
    return {"ok": True, "latest": tag, "patch": f"v{major}.{minor}.{patch + 1}",
            "minor": f"v{major}.{minor + 1}.0"}


def check_release_tag(room_id: str, tag: str, ref: str = RELEASE_BRANCH) -> dict:
    """Validate a release tag before anyone is asked about it.

    Returns ``{ok, tag, sha, exists}`` or ``{ok: False, code, error}``. ``exists``
    means the tag is already here *on the same commit* (asking again is
    harmless); on another commit it is ``tag_exists`` — tags never move.
    """
    root = _workspace_root(room_id)
    if not (root / ".git").is_dir():
        return {"ok": False, "code": "no_repo", "error": "the workspace has no git repository"}
    if not RELEASE_TAG_RE.match(tag or ""):
        return {"ok": False, "code": "tag_invalid",
                "error": f"'{tag}' is not a release tag; use vMAJOR.MINOR.PATCH, e.g. v1.4.0 or v1.4.0-rc.1"}
    sha = _resolve(root, ref)
    if sha is None:
        return {"ok": False, "code": "ref_unknown", "error": f"'{ref}' is not a commit in the workspace"}
    on_main = _run(["merge-base", "--is-ancestor", sha, f"refs/heads/{RELEASE_BRANCH}"], cwd=root)
    if on_main.returncode != 0:
        return {"ok": False, "code": "not_on_main",
                "error": f"{sha[:10]} is not on '{RELEASE_BRANCH}'; release tags go on {RELEASE_BRANCH} "
                         "only, after the dev -> main promotion"}
    existing = _resolve(root, f"refs/tags/{tag}")
    if existing is not None and existing != sha:
        return {"ok": False, "code": "tag_exists",
                "error": f"tag '{tag}' already exists on {existing[:10]}; release tags are never moved. "
                         "Use a new version number"}
    return {"ok": True, "tag": tag, "sha": sha, "exists": existing is not None}


@_serialized(busy_dict=True)
def create_release_tag(room_id: str, tag: str, ref: str = RELEASE_BRANCH, message: str = "") -> dict:
    """Create the annotated tag (idempotent on the same commit). Same result shape as the check."""
    checked = check_release_tag(room_id, tag, ref)
    if not checked["ok"] or checked["exists"]:
        return checked
    root = _workspace_root(room_id)
    r = _run(["tag", "-a", tag, "-m", (message or "").strip() or f"Release {tag}", checked["sha"]], cwd=root)
    if r.returncode != 0:
        return {"ok": False, "code": "tag_failed", "error": r.stderr.strip()[:300]}
    return {**checked, "created": True}


@_serialized(busy_dict=True)
def push_release_tag(room_id: str, remote_url: str, tag: str, token: Optional[str] = None) -> dict:
    """Push exactly ``refs/tags/<tag>`` — never ``--tags``, never forced.

    The forge refuses the push when it already has the tag on another commit,
    which is the immutability rule on the other side. Returns ``{ok, stdout, stderr}``.
    """
    root = _workspace_root(room_id)
    if not RELEASE_TAG_RE.match(tag or "") or _resolve(root, f"refs/tags/{tag}") is None:
        return {"ok": False, "stdout": "", "stderr": f"tag '{tag}' does not exist in the workspace"}
    refspec = f"refs/tags/{tag}:refs/tags/{tag}"
    r = _run(["push", _inject_token(remote_url, token), refspec], cwd=root)
    return {"ok": r.returncode == 0, "stdout": r.stdout[:4096], "stderr": r.stderr[:4096]}
