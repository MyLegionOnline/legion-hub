"""Deploy only what was published (internal design notes Stage 15).

A server that runs ``git pull`` or ``git checkout <ref>`` can only see commits
that reached the forge. What an agent built may still sit in the Room's
workspace — on ``dev``, on a task branch, or on a ``main`` that was merged
locally and never pushed — and then the server happily fast-forwards to the
*previous* state and reports success. Nothing in any log says the site shows
the old version. That is the worst failure a deploy has, so this module makes
it impossible to reach quietly:

* :func:`ensure_published` pushes a branch and returns the commit the server is
  obliged to end up on;
* :func:`parse_deploy_intent` / :func:`check_intent` recognise the one class of
  ``vps_exec`` command that moves the server's checkout and ask the forge,
  *before* the server is touched, whether the target exists there
  (``deploy_not_published``) and whether it is part of the Room's publication
  branch (a person decides otherwise);
* :func:`sha_matches` is the after-the-fact check that the server really
  landed on that commit (``deploy_sha_mismatch``).

It lives beside the gateway rather than inside it because the same facts are
needed by the ``publish_for_deploy`` tool, and none of it touches MCP.
"""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
#: Refs whose meaning only the server knows (``HEAD~1``, ``FETCH_HEAD``, a
#: ``stash@{0}``): there is nothing to ask the forge about, so they are not gated.
_SERVER_LOCAL_REF = re.compile(r"[~^@:{]|^(HEAD|FETCH_HEAD|ORIG_HEAD|MERGE_HEAD)$")
_REF_PREFIXES = ("refs/remotes/origin/", "refs/heads/", "origin/")
_GIT_VALUE_FLAGS = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}


class DeployPublishError(Exception):
    """A deploy that must not start, with the code the agent is told.

    ``deploy_not_published``       the target is not on the forge — publish it.
    ``deploy_needs_approval``      published, but not part of the publication
                                   branch — a person has to say yes.
    ``deploy_publication_unverified``  the forge could not be asked; nothing is
                                   assumed in either direction.
    """

    def __init__(self, code: str, message: str, *, commits: Optional[list[dict[str, Any]]] = None):
        super().__init__(message)
        self.code = code
        self.commits = commits or []


@dataclass(frozen=True)
class DeployIntent:
    """What a ``vps_exec`` git command is about to do to the server's checkout."""

    kind: str  # "ref" — move to a named commit/branch; "pull" — follow the branch
    ref: Optional[str] = None

    @property
    def ref_is_sha(self) -> bool:
        return bool(self.ref and _SHA_RE.match(self.ref.lower()))


@dataclass(frozen=True)
class DeployCheck:
    """The verdict on an intent that may go ahead (possibly after a card)."""

    sha: Optional[str]  # the commit the server must end up on; None = not knowable
    ref: Optional[str] = None
    unmerged: bool = False  # published, but outside the publication branch
    publication_branch: str = "main"
    commits: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class PublishedCommit:
    sha: str
    branch: str
    pushed: bool  # False = it was already on the forge
    verified: bool = True  # False = the push succeeded but the forge could not confirm
    publication_branch: str = "main"


# ── parsing ─────────────────────────────────────────────────────────────────


def _norm(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return posixpath.normpath(path.strip()).rstrip("/") or "/"


def parse_deploy_intent(
    command: Sequence[str], *, cwd: Optional[str], project_dir: Optional[str]
) -> Optional[DeployIntent]:
    """``DeployIntent`` for a git command that moves the Room's checkout, else ``None``.

    Only a command aimed at ``project_dir`` (by ``-C`` or ``cwd``) counts: the
    Room's checkout is the thing a deploy publishes to, and every other git
    invocation on the box is somebody else's business. Forms with no single
    target ref (``checkout -b``, ``checkout <ref> -- file``, ``reset`` without
    ``--hard``) restore files rather than move the deploy, and are left alone.
    """
    if not command or posixpath.basename(str(command[0])) != "git":
        return None
    tokens = [str(t) for t in command[1:]]
    workdir = _norm(cwd)
    i = 0
    while i < len(tokens) and tokens[i].startswith("-"):
        if tokens[i] == "-C" and i + 1 < len(tokens):
            target = tokens[i + 1]
            workdir = _norm(target if target.startswith("/") or not workdir else posixpath.join(workdir, target))
        if tokens[i] in _GIT_VALUE_FLAGS:
            i += 2
        else:
            i += 1
    if i >= len(tokens) or workdir is None or workdir != _norm(project_dir):
        return None
    sub, args = tokens[i], tokens[i + 1:]

    if sub == "pull":
        return DeployIntent("pull")
    if "--" in args:
        return None
    if sub in ("checkout", "switch"):
        if any(a in ("-b", "-B", "-c", "-C", "--orphan", "--create", "--force-create") for a in args):
            return None
    elif sub == "reset":
        if "--hard" not in args:
            return None
    elif sub != "merge":
        return None
    positional = [a for a in args if not a.startswith("-")]
    if len(positional) != 1 or _SERVER_LOCAL_REF.search(positional[0]):
        return None
    ref = positional[0]
    for prefix in _REF_PREFIXES:
        if ref.startswith(prefix):
            ref = ref[len(prefix):]
            break
    return DeployIntent("ref", ref)


def sha_matches(expected: Optional[str], actual: Optional[str]) -> bool:
    """True when ``actual`` (``git rev-parse HEAD`` output) is the ``expected`` commit."""
    if not expected or not actual:
        return False
    actual = actual.strip().lower()
    expected = expected.strip().lower()
    return len(actual) >= 7 and expected.startswith(actual)


# ── workspace facts ─────────────────────────────────────────────────────────


async def _workspace_commits(room_id: UUID, **kwargs: Any) -> list[dict[str, Any]]:
    from app.core import workspaced_client as wsd

    try:
        data = await wsd.git_log_workspace(str(room_id), **kwargs)
    except Exception:
        logger.warning("deploy_publish: workspace git log failed for %s", room_id, exc_info=True)
        return []
    return list(data.get("entries") or [])


async def local_head(room_id: UUID, branch: str) -> Optional[str]:
    """Full sha of ``branch`` in the Room's workspace, or ``None`` if it has no such branch."""
    entries = await _workspace_commits(room_id, limit=1, rev_range=branch)
    return entries[0].get("sha") if entries else None


async def _unpublished(room_id: UUID, local: str, remote: Optional[str]) -> list[dict[str, Any]]:
    """Commits the workspace has on ``local`` that the forge head ``remote`` lacks."""
    if remote is None:
        return await _workspace_commits(room_id, limit=20, rev_range=local)
    return await _workspace_commits(room_id, limit=20, rev_range=f"{remote}..{local}")


async def unmerged_commits(room_id: UUID, branch: str, base: str) -> list[dict[str, Any]]:
    """Commits on ``branch`` that are not in ``base`` (best effort; ``[]`` when unknown)."""
    if branch == base:
        return []
    from app.core import workspaced_client as wsd

    try:
        report = await wsd.git_divergence_workspace(str(room_id), branch, base)
    except Exception:
        return []
    if not report.get("ok"):
        return []
    return list(report.get("branch_commits") or [])


# ── context ─────────────────────────────────────────────────────────────────


async def _target(db: Any, room_id: UUID, request_id: Optional[UUID]):
    """``(room, RepoTarget, token)``; :class:`DeployKeyError` when the Room has no repository."""
    from app.core import git_forge, vps_deploy_key

    room, target = await vps_deploy_key.repo_target(db, room_id)
    token = await git_forge.resolve_forge_token(db, room, request_id=request_id or uuid4())
    return room, target, token


def _unverified(what: str) -> DeployPublishError:
    return DeployPublishError(
        "deploy_publication_unverified",
        f"could not confirm {what} with the Room's git host (unreachable, or the Room's git "
        "token is missing or expired). Nothing was assumed: ask the person to check the "
        "repository connection on the Room's Git tab, then try again",
    )


# ── the gate ────────────────────────────────────────────────────────────────


async def check_intent(
    db: Any, room_id: UUID, intent: DeployIntent, *, request_id: Optional[UUID] = None
) -> Optional[DeployCheck]:
    """Ask the forge whether ``intent`` deploys something that was published.

    Returns ``None`` when there is nothing to check (the Room has no external
    repository, or its forge is not one the Hub can ask — the old behaviour,
    unchanged). Raises :class:`DeployPublishError` when the deploy must not
    start. Otherwise a :class:`DeployCheck`; if ``unmerged`` is set the caller
    owes a person a card before going ahead.
    """
    from app.core import git_forge, vps_deploy_key

    try:
        _room, target, token = await _target(db, room_id, request_id)
    except vps_deploy_key.DeployKeyError:
        return None
    git_config = target.git_config
    publication = target.branch

    if intent.kind == "pull":
        state, remote = await git_forge.remote_branch_head(git_config, token, publication)
        if state == "unsupported":
            return None
        if state == "unknown":
            raise _unverified(f"that '{publication}' is published")
        local = await local_head(room_id, publication)
        remote_sha = remote if state == "found" else None
        if local and local != remote_sha:
            ahead = await _unpublished(room_id, local, remote_sha)
            if ahead:
                raise DeployPublishError(
                    "deploy_not_published",
                    _not_published_text(publication, len(ahead), published=state == "found"),
                    commits=ahead,
                )
        return DeployCheck(sha=None, ref=publication, publication_branch=publication)

    assert intent.ref is not None
    relation, full = await git_forge.commit_relation(git_config, token, intent.ref, publication)
    if relation == "unsupported":
        return None
    if relation == "unknown":
        raise _unverified(f"that '{intent.ref}' is published")
    if relation == "missing":
        raise DeployPublishError(
            "deploy_not_published",
            f"'{intent.ref}' does not exist on the Room's repository, so the server cannot fetch "
            "it — a deploy of it would fail or, worse, leave the old version running. Call "
            "publish_for_deploy first and deploy the commit it returns",
        )
    if not intent.ref_is_sha:
        # A branch name moves. The forge copy must be what the workspace has,
        # or the server gets a stale head and calls it a success.
        local = await local_head(room_id, intent.ref)
        if local and full and local != full:
            ahead = await _unpublished(room_id, local, full)
            if ahead:
                raise DeployPublishError(
                    "deploy_not_published",
                    _not_published_text(intent.ref, len(ahead), published=True),
                    commits=ahead,
                )
    unmerged = relation == "outside_branch"
    commits: list[dict[str, Any]] = []
    if unmerged:
        commits = await unmerged_commits(room_id, full or intent.ref, publication)
    return DeployCheck(
        sha=full, ref=intent.ref, unmerged=unmerged, publication_branch=publication, commits=commits
    )


def _not_published_text(branch: str, count: int, *, published: bool) -> str:
    where = "the copy on the repository is behind it" if published else "it was never pushed"
    return (
        f"'{branch}' has {count} commit(s) in the Room's workspace that are not on the "
        f"repository ({where}). A server can only fetch what is published, so this deploy "
        "would quietly ship the previous version. Call publish_for_deploy, then deploy the "
        "commit it returns"
    )


# ── publishing ──────────────────────────────────────────────────────────────


async def ensure_published(
    db: Any,
    room_id: UUID,
    *,
    branch: Optional[str] = None,
    request_id: Optional[UUID] = None,
    allow_unmerged: bool = False,
) -> PublishedCommit:
    """Make ``branch`` (default: the Room's publication branch) exist on the forge as-is.

    Returns the commit a server must be deployed at. A branch that is not part
    of the publication branch raises ``deploy_needs_approval`` (with the
    commits) unless the caller has already got a person's yes — pushing it is
    harmless, but it is the first step of putting unreviewed work on a server.
    """
    from app.core import git_forge, vps_deploy_key
    from app.core.git_sync import push_workspace_to_git
    from app.db.release import release_db

    try:
        _room, target, token = await _target(db, room_id, request_id)
    except vps_deploy_key.DeployKeyError as exc:
        raise DeployPublishError(
            "repo_not_configured",
            f"{exc} Do not retry: ask the person to connect a repository on the Room's Git tab",
        ) from None
    publication = target.branch
    branch = (branch or publication).strip()

    local = await local_head(room_id, branch)
    if local is None:
        raise DeployPublishError(
            "deploy_not_published",
            f"branch '{branch}' has no commits in this Room's workspace — there is nothing to "
            "publish. Check the branch name, and that the work was committed",
        )
    unmerged = await unmerged_commits(room_id, branch, publication)
    if unmerged and not allow_unmerged:
        raise DeployPublishError(
            "deploy_needs_approval",
            f"'{branch}' carries {len(unmerged)} commit(s) that are not part of '{publication}'",
            commits=unmerged,
        )

    git_config = target.git_config
    state, remote = await git_forge.remote_branch_head(git_config, token, branch)
    if state == "found" and remote == local:
        return PublishedCommit(local, branch, pushed=False, publication_branch=publication)

    await release_db(db, reason="deploy_publish/push")
    result = await push_workspace_to_git(
        room_id=room_id, git_config=dict(git_config), access_token=token, mode="push", branch=branch,
    )
    errors = result.get("errors") or []
    if errors:
        raise DeployPublishError(
            "deploy_not_published",
            f"pushing '{branch}' failed: {str(errors[0])[:300]}. If the remote moved on, a person "
            "has to sync the Room's repository on the Git tab; do not try another transport",
        )

    state, remote = await git_forge.remote_branch_head(git_config, token, branch)
    if state == "found" and remote != local:
        raise DeployPublishError(
            "deploy_not_published",
            f"the push of '{branch}' reported success, but the repository's '{branch}' is at "
            f"{(remote or '')[:10]} and not at {local[:10]}. Do not deploy until a person looks",
        )
    return PublishedCommit(
        local, branch, pushed=True, verified=state == "found", publication_branch=publication
    )


# ── the card ────────────────────────────────────────────────────────────────


async def confirm_unmerged_deploy(
    db: Any,
    *,
    room_id: UUID,
    session_id: UUID,
    ref: str,
    publication_branch: str,
    commits: Sequence[dict[str, Any]],
) -> bool:
    """Ask a person whether work outside the publication branch may be deployed.

    Same ask-user card the other MCP confirmations use. "Allow" is for this one
    deploy; nothing is remembered.
    """
    from app.core import ask_user_registry as _reg
    from app.core.events import emit_event

    shown = [f"- {str(c.get('sha', ''))[:8]} {str(c.get('message', ''))[:90]}" for c in list(commits)[:8]]
    more = len(commits) - len(shown)
    lines = [
        f"A deploy to a server is about to use '{ref}', which is not part of "
        f"'{publication_branch}' — the branch this Room publishes.",
    ]
    if shown:
        lines.append(f"Commits that '{publication_branch}' does not have ({len(commits)}):")
        lines.extend(shown)
        if more > 0:
            lines.append(f"…and {more} more")
    lines.append("Those changes have not been through the Room's promotion gate.")
    question_text = "\n".join(lines)
    questions = [
        {
            "question": question_text,
            "header": "Deploy unmerged",
            "options": [
                {
                    "label": "Allow",
                    "description": "Deploy this exact commit now. The permission is not saved.",
                },
                {
                    "label": "Deny",
                    "description": "Do not deploy. The agent is told to deploy a published, merged commit instead.",
                },
            ],
        }
    ]
    question_id = str(uuid4())
    timeout_seconds = int(os.getenv("MCP_POLICY_APPROVAL_TIMEOUT_SECONDS", "300"))
    try:
        await emit_event(
            room_id=room_id,
            event_type="AGENT_QUESTION",
            content=question_text,
            db=db,
            metadata={
                "question_id": question_id,
                "session_id": str(session_id),
                "questions": questions,
                "timeout_seconds": timeout_seconds,
            },
        )
    except Exception:
        logger.exception("deploy_publish: failed to emit the unmerged-deploy card (room=%s)", room_id)
        return False

    pq = _reg.register(question_id, session_id, room_id, {"questions": questions})
    try:
        await asyncio.wait_for(pq.event.wait(), timeout=float(timeout_seconds))
    except asyncio.TimeoutError:
        _reg.expire(question_id)
        return False
    _reg.expire(question_id)
    return "Allow" in ((pq.answer or {}).get("selected_options") or [])


__all__ = [
    "DeployCheck",
    "DeployIntent",
    "DeployPublishError",
    "PublishedCommit",
    "check_intent",
    "confirm_unmerged_deploy",
    "ensure_published",
    "local_head",
    "parse_deploy_intent",
    "sha_matches",
    "unmerged_commits",
]
