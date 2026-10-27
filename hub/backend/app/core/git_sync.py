"""
Git integration core: syncs files from GitHub/GitLab into the room repository.

Supported git_config structure (stored in Room.git_config):
{
  "provider": "github" | "gitlab",
  "repo_url": "https://github.com/user/project",
  "branch": "main",
  "access_token": "ghp_..."
}

Flow:
  POST /rooms/{id}/git/webhook  — GitHub/GitLab push webhook → sync changed files
  POST /rooms/{id}/git/sync     — manual full sync of the configured branch
"""

import base64
import logging
import mimetypes
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import emit_event
from app.core.git_author import sync_from_token

logger = logging.getLogger(__name__)


class GitTreeTruncatedError(RuntimeError):
    """Raised when the provider's recursive tree listing was truncated.

    GitHub's `git/trees?recursive=1` sets `truncated: true` and silently
    omits entries once a repo is too large for one response. Syncing that
    partial list would delete-by-omission or just miss files without any
    signal — see internal design notes Stage 2.2.
    """


def _clone_url(repo_url: str) -> str:
    """Normalize a user-supplied repository URL into one ``git`` can fetch/push.

    Users routinely paste the browser address bar rather than the clone URL,
    e.g. ``https://github.com/owner/repo/tree/main`` or a GitLab
    ``https://gitlab.com/owner/repo/-/blob/main/file.py``. Handed straight to
    ``git fetch <url> <branch>`` these fail with
    ``fatal: repository '.../tree/main/' not found``. Strip a trailing slash,
    a ``.git`` suffix, and any web-UI path segment (``/tree/``, ``/blob/``,
    ``/commit/``, ``/pull/``, GitLab's ``/-/...``) so only the
    ``scheme://host/owner/repo`` root reaches git. The branch is always taken
    from ``git_config["branch"]`` separately, so dropping ``/tree/<branch>`` here
    never loses information.
    """
    url = (repo_url or "").strip().rstrip("/")
    for marker in (
        "/-/tree/", "/-/blob/", "/-/commit/", "/-/merge_requests/", "/-/",
        "/tree/", "/blob/", "/commit/", "/commits/", "/pull/", "/pulls/",
        "/compare/", "/releases/",
    ):
        idx = url.find(marker)
        if idx != -1:
            url = url[:idx]
            break
    return url.rstrip("/").removesuffix(".git")


def _parse_repo_info(
    git_config: Dict[str, Any], access_token: str | None = None
) -> tuple[str, str, str, str]:
    """Extract repository metadata and a migration-only token fallback."""
    repo_url: str = git_config.get("repo_url", "")
    branch: str = git_config.get("branch", "main")
    token = access_token
    if token is None:
        from app.core.secrets.runtime import legacy_secret_value

        legacy = legacy_secret_value(git_config, "access_token")
        try:
            token = legacy.reveal() if legacy else ""
        finally:
            legacy = None
    provider: str = git_config.get("provider", "github")

    # Strip trailing .git and any /blob/... /tree/... web-UI path suffix so a URL
    # like https://github.com/owner/repo/blob/main/file still parses correctly.
    repo_url = _clone_url(repo_url)
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse repo owner/name from URL: {repo_url!r}")
    owner, repo = parts[-2], parts[-1]
    return owner, repo, branch, token


def _bytes_to_content(path: str, raw: bytes) -> str:
    """Decode raw file bytes for storage.

    Valid UTF-8 text is stored as-is. Anything else is wrapped as a
    ``data:<mime>;base64,...`` URI instead of being force-decoded with
    `errors="replace"`, which would corrupt binary files (images, fonts,
    archives...). ``workspaced``'s ``file_ops.write_file`` treats a `data:`
    prefix as the binary marker, so no downstream change is needed.
    See Stage 2.4.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "application/octet-stream"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"


async def push_workspace_to_git(
    room_id: UUID,
    git_config: Dict[str, Any],
    access_token: str | None = None,
    mode: str = "push",
    branch: str | None = None,
) -> Dict[str, Any]:
    """Push workspace git repo to remote using native git (v2 — workspace mode only).

    Replaces the file-by-file Contents API approach with a single git push,
    preserving full commit history. ``mode="push"`` (default) is a plain
    push and lets the remote reject a non-fast-forward; ``mode="force"``
    applies ``--force-with-lease`` so divergence can be overridden only by
    an explicit, acknowledged user choice (see
    internal design notes stage 2.2). Falls back gracefully
    if workspaced is unreachable.

    ``branch`` overrides the branch to push; when ``None`` the configured
    ``git_config["branch"]`` (``main`` by default) is used. internal design notes
    Stage 6 passes ``"dev"`` so an external CI can run on the integration
    branch without ever touching ``main``.
    """
    from app.core import workspaced_client as wsd

    repo_url = _clone_url(git_config.get("repo_url", ""))
    branch = branch or git_config.get("branch", "main")
    token = access_token or ""
    await sync_from_token(room_id, git_config, token)

    # Ensure .git suffix stripped for push URL
    try:
        result = await wsd.git_push_workspace(
            str(room_id),
            remote_url=repo_url,
            branch=branch,
            token=token or None,
            mode=mode,
        )
    except Exception as exc:
        logger.warning("push_workspace_to_git failed for room %s: %s", room_id, exc)
        return {"pushed": 0, "errors": [str(exc)]}

    if result.get("userfiles_push_blocked"):
        return {
            "pushed": 0,
            "errors": [result.get("stderr", "push blocked")[:500]],
            "userfiles_push_blocked": True,
            "userfiles_tracked": result.get("userfiles_tracked") or [],
            "userfiles_policy_conflict": True,
            "userfiles_tracked_details": result.get("userfiles_tracked_details") or [],
        }

    if result.get("ok"):
        return {"pushed": 1, "errors": []}
    return {"pushed": 0, "errors": [result.get("stderr", "push failed")[:500]]}


async def push_tag_to_git(
    room_id: UUID,
    git_config: Dict[str, Any],
    tag: str,
    access_token: str | None = None,
) -> Dict[str, Any]:
    """Push one release tag of the workspace to the remote (``release_tag``).

    Same transport as :func:`push_workspace_to_git`; only ``refs/tags/<tag>``
    goes, never forced, so a forge that already has the tag on another commit
    refuses it. Returns ``{pushed, errors}``.
    """
    from app.core import workspaced_client as wsd

    repo_url = _clone_url(git_config.get("repo_url", ""))
    token = access_token or ""
    await sync_from_token(room_id, git_config, token)
    try:
        result = await wsd.git_push_tag_workspace(
            str(room_id), remote_url=repo_url, tag=tag, token=token or None
        )
    except Exception as exc:
        logger.warning("push_tag_to_git failed for room %s: %s", room_id, exc)
        return {"pushed": 0, "errors": [str(exc)]}
    if result.get("ok"):
        return {"pushed": 1, "errors": []}
    return {"pushed": 0, "errors": [result.get("stderr", "push failed")[:500]]}


async def fetch_workspace_from_git(
    room_id: UUID,
    git_config: Dict[str, Any],
    access_token: str | None = None,
    target_branch: str | None = None,
    remote_branch: str | None = None,
) -> Dict[str, Any]:
    """Fetch + fast-forward merge from remote using native git (v2 — workspace mode only).

    The remote branch (``git_config["branch"]``) is merged into
    ``target_branch``, which is checked out first. When omitted it defaults to
    the room's ``working_branch`` (``dev``) — never "whatever HEAD happens to
    be", which would land the remote on a task branch if a task session had one
    checked out (internal design notes §0.1). ``room_git_flow`` passes the
    release branch explicitly when catching ``main`` up after a merged PR.

    ``remote_branch`` overrides which remote branch is fetched (default: the
    configured ``git_config["branch"]``). The Git panel's per-branch Pull passes
    the same name as ``target_branch``, so an edit made to ``dev`` on the remote
    lands on the local ``dev``.

    On conflict, returns conflict=True so the caller can emit a chat event.
    """
    from types import SimpleNamespace

    from app.core import workspaced_client as wsd
    from app.core.room_git_flow import get_flow_config

    repo_url = _clone_url(git_config.get("repo_url", ""))
    branch = remote_branch or git_config.get("branch", "main")
    token = access_token or ""
    await sync_from_token(room_id, git_config, token)
    landing = target_branch or get_flow_config(
        SimpleNamespace(git_config=git_config)
    ).working_branch

    try:
        result = await wsd.git_fetch_workspace(
            str(room_id),
            remote_url=repo_url,
            branch=branch,
            token=token or None,
            target_branch=landing,
        )
    except Exception as exc:
        logger.warning("fetch_workspace_from_git failed for room %s: %s", room_id, exc)
        return {"synced": 0, "errors": [str(exc)], "conflict": False}

    # internal design notes Stage 4 — surfaced by the caller as a chat event.
    gitignore_autoresolved = bool(result.get("gitignore_autoresolved"))
    if result.get("ok"):
        return {
            "synced": 1,
            "errors": [],
            "conflict": False,
            "branch": result.get("branch") or landing,
            "gitignore_autoresolved": gitignore_autoresolved,
        }
    return {
        "synced": 0,
        "errors": [result.get("stderr", "fetch failed")[:500]],
        "conflict": result.get("conflict", False),
        "userfiles_policy_conflict": bool(result.get("userfiles_policy_conflict")),
        "userfiles_tracked": result.get("userfiles_tracked") or [],
        "userfiles_tracked_details": result.get("userfiles_tracked_details") or [],
        "branch": result.get("branch") or landing,
        "gitignore_autoresolved": gitignore_autoresolved,
    }


async def adopt_remote_repository(
    room_id: UUID,
    git_config: Dict[str, Any],
    access_token: str | None = None,
    force: bool = False,
) -> Dict[str, Any]:
    """internal design notes §3 — take an external repo as the room's base.

    Called the first time a room is bound to an existing remote (before any
    ``adopted_at`` marker). Pins the room's local ``main`` to the remote's
    ``main`` instead of merging the remote into ``dev`` with unrelated
    histories, which is what the plain ``sync`` path does.

    Returns ``{ok, adopted, diverged, dropped_init, main_sha, working_sha,
    errors}``. ``diverged`` (``main`` left untouched) is *not* an error — the
    caller surfaces it as a ``GIT_CONFLICT`` chat event so the user can pick
    between merging histories and forcing the transplant.
    """
    from types import SimpleNamespace

    from app.core import workspaced_client as wsd
    from app.core.room_git_flow import get_flow_config

    repo_url = _clone_url(git_config.get("repo_url", ""))
    branch = git_config.get("branch", "main")
    token = access_token or ""
    await sync_from_token(room_id, git_config, token)
    working_branch = get_flow_config(
        SimpleNamespace(git_config=git_config)
    ).working_branch

    try:
        result = await wsd.git_adopt_workspace(
            str(room_id),
            remote_url=repo_url,
            branch=branch,
            token=token or None,
            working_branch=working_branch,
            force=force,
        )
    except Exception as exc:
        logger.warning("adopt_remote_repository failed for room %s: %s", room_id, exc)
        return {"ok": False, "adopted": False, "diverged": False, "errors": [str(exc)]}

    ok = bool(result.get("ok"))
    normalized = {
        "ok": ok,
        "adopted": bool(result.get("adopted")),
        "diverged": bool(result.get("diverged")),
        "dropped_init": bool(result.get("dropped_init")),
        "main_sha": result.get("main_sha"),
        "working_sha": result.get("working_sha"),
        "errors": [] if ok else [result.get("stderr", "adopt failed")[:500]],
    }
    if (
        result.get("userfiles_policy_conflict")
        or result.get("userfiles_tracked")
        or result.get("userfiles_tracked_details")
    ):
        normalized.update(
            {
                "userfiles_policy_conflict": bool(result.get("userfiles_policy_conflict")),
                "userfiles_tracked": result.get("userfiles_tracked") or [],
                "userfiles_tracked_details": result.get("userfiles_tracked_details") or [],
            }
        )
    return normalized


def extract_changed_paths_github(payload: Dict[str, Any]) -> List[str]:
    """Extract changed file paths from a GitHub push webhook payload."""
    paths: List[str] = []
    for commit in payload.get("commits", []):
        paths.extend(commit.get("added", []))
        paths.extend(commit.get("modified", []))
    return list(dict.fromkeys(paths))  # deduplicate, preserve order


def extract_changed_paths_gitlab(payload: Dict[str, Any]) -> List[str]:
    """Extract changed file paths from a GitLab push webhook payload."""
    paths: List[str] = []
    for commit in payload.get("commits", []):
        paths.extend(commit.get("added", []))
        paths.extend(commit.get("modified", []))
    return list(dict.fromkeys(paths))
