"""Pull-request helpers for the agentic CI/CD workflow (variant A).

internal design notes **Stage 7 — PR + external CI status.** When a room runs
``ci_provider == "github_actions"``, Gate 2 no longer relies on the in-container
check suite: ``open_promotion`` pushes ``dev`` to the remote (Stage 6) and this
module opens a real pull request ``dev → main`` so the user's own
``.github/workflows/ci.yml`` runs. GitHub then reports back via a
``workflow_run`` / ``check_suite`` webhook (handled in ``app.api.v1.git``), which
stamps ``Promotion.ci_status``. On approve, ``resolve_promotion`` merges the PR
through the forge API instead of the local tree and fetches ``main`` back so the
workspace volume catches up.

Everything here is best-effort: a forge that is unreachable or a token that is
missing must never wedge a promotion — the gate still opens and a human can fall
back to the workspace-only flow (``force``).

Only GitHub is wired end-to-end; GitLab merge-requests use the same shape
(``/merge_requests`` instead of ``/pulls``) and are left as a follow-up.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import httpx

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# A reference workflow the user drops into their own repository. Kept in sync
# with hub/examples/agentic_cicd/ci.yml.
EXAMPLE_CI_WORKFLOW = """\
# .github/workflows/ci.yml — runs on the `dev → main` pull request Legion opens.
name: ci
on:
  pull_request:
    branches: [main]
  push:
    branches: [dev]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest -q
"""


def parse_repo(git_config: Mapping[str, Any]) -> tuple[str, str, str]:
    """``(provider, owner, repo)`` from ``git_config['repo_url']``.

    Mirrors ``git_sync._parse_repo_info`` URL cleanup (drops ``.git`` and any
    ``/blob/…`` / ``/tree/…`` suffix). Raises ``ValueError`` on an unparseable
    URL.
    """
    from app.core.git_sync import _clone_url

    provider = str(git_config.get("provider") or "github").strip().lower()
    repo_url = _clone_url(str(git_config.get("repo_url") or ""))
    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2 or not parts[-1] or not parts[-2]:
        raise ValueError(f"cannot parse owner/repo from {repo_url!r}")
    return provider, parts[-2], parts[-1]


def _api_base(git_config: Mapping[str, Any]) -> str:
    return str(git_config.get("api_base") or _GITHUB_API).rstrip("/")


def _headers(token: str | None) -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


async def remote_branch_exists(
    git_config: Mapping[str, Any],
    token: str | None,
    branch: str,
) -> bool:
    """True if ``branch`` exists on the remote (GitHub).

    ``GET /repos/{owner}/{repo}/branches/{branch}`` — 200 → exists, 404 → not.
    Used by ``open_promotion`` to avoid opening a ``dev → main`` PR against a
    ``main`` that was never published (the forge would answer 422 and the
    promotion would sit there with no PR and no explanation —
    internal design notes §4.2). Any transport error or an
    unparseable / non-GitHub URL is treated as "cannot confirm" → ``False``,
    so the caller degrades to the workspace-only flow rather than guessing.
    """
    try:
        provider, owner, repo = parse_repo(git_config)
    except ValueError:
        return False
    if provider != "github":
        return False

    url = f"{_api_base(git_config)}/repos/{owner}/{repo}/branches/{branch}"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.get(url, headers=_headers(token))
            return r.status_code == 200
    except httpx.HTTPError as exc:
        logger.warning(
            "remote_branch_exists: %s/%s %s check failed: %s", owner, repo, branch, exc
        )
        return False


async def remote_branch_head(
    git_config: Mapping[str, Any],
    token: str | None,
    branch: str,
) -> tuple[str, str | None]:
    """The commit ``branch`` points at on the forge: ``(state, sha)``.

    ``state`` is ``"found"`` (``sha`` is the full head), ``"missing"`` (the forge
    answered 404 — the branch was never pushed), ``"unsupported"`` (not a forge
    we can ask) or ``"unknown"`` (the forge could not be reached or refused the
    token). The four are kept apart on purpose: a deploy gate treats "never
    pushed" as a verdict and "could not ask" as a reason to stop and say so,
    never as permission (internal design notes Stage 15).
    """
    try:
        provider, owner, repo = parse_repo(git_config)
    except ValueError:
        return "unsupported", None

    from urllib.parse import quote

    if provider == "github":
        url = f"{_api_base(git_config)}/repos/{owner}/{repo}/branches/{quote(branch, safe='')}"
        headers = _headers(token)
    elif provider == "gitlab":
        url = (
            f"{_gitlab_api_base(git_config)}/projects/{_gitlab_project(owner, repo)}"
            f"/repository/branches/{quote(branch, safe='')}"
        )
        headers = {"PRIVATE-TOKEN": token} if token else {}
    else:
        return "unsupported", None

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.get(url, headers=headers)
    except httpx.HTTPError as exc:
        logger.warning("remote_branch_head: %s/%s %s failed: %s", owner, repo, branch, exc)
        return "unknown", None
    if r.status_code == 404:
        # A private repository answers 404 to an unauthenticated caller too,
        # so only an authenticated 404 means "never pushed".
        return ("missing" if token else "unknown"), None
    if r.status_code != 200:
        return "unknown", None
    try:
        commit = r.json().get("commit") or {}
        sha = commit.get("sha") or commit.get("id")
    except ValueError:
        return "unknown", None
    return ("found", str(sha)) if sha else ("unknown", None)


async def commit_relation(
    git_config: Mapping[str, Any],
    token: str | None,
    sha: str,
    base_branch: str,
) -> tuple[str, str | None]:
    """Where commit ``sha`` stands on the forge relative to ``base_branch``.

    Returns ``(relation, full_sha)``. ``relation`` is ``"in_branch"`` (the
    commit is an ancestor of, or equal to, the base branch head),
    ``"outside_branch"`` (the forge has the commit, the base branch does not
    contain it), ``"missing"`` (the forge has never seen it — it was not
    pushed), ``"unsupported"`` or ``"unknown"`` as in :func:`remote_branch_head`.
    """
    try:
        provider, owner, repo = parse_repo(git_config)
    except ValueError:
        return "unsupported", None

    from urllib.parse import quote

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            if provider == "github":
                headers = _headers(token)
                base = f"{_api_base(git_config)}/repos/{owner}/{repo}"
                r = await c.get(f"{base}/commits/{quote(sha, safe='')}", headers=headers)
                if r.status_code in (404, 422):
                    return ("missing" if token else "unknown"), None
                if r.status_code != 200:
                    return "unknown", None
                full = r.json().get("sha")
                cmp_ = await c.get(
                    f"{base}/compare/{quote(base_branch, safe='')}...{full}", headers=headers
                )
                if cmp_.status_code != 200:
                    return "unknown", full
                status = cmp_.json().get("status")
                return ("in_branch" if status in ("identical", "behind") else "outside_branch"), full
            if provider == "gitlab":
                headers = {"PRIVATE-TOKEN": token} if token else {}
                base = f"{_gitlab_api_base(git_config)}/projects/{_gitlab_project(owner, repo)}"
                r = await c.get(f"{base}/repository/commits/{quote(sha, safe='')}", headers=headers)
                if r.status_code == 404:
                    return ("missing" if token else "unknown"), None
                if r.status_code != 200:
                    return "unknown", None
                full = r.json().get("id")
                refs = await c.get(
                    f"{base}/repository/commits/{full}/refs", headers=headers, params={"type": "branch"}
                )
                if refs.status_code != 200:
                    return "unknown", full
                names = {item.get("name") for item in refs.json() if isinstance(item, dict)}
                return ("in_branch" if base_branch in names else "outside_branch"), full
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("commit_relation: %s/%s %s failed: %s", owner, repo, sha, exc)
        return "unknown", None
    return "unsupported", None


async def create_pull_request(
    git_config: Mapping[str, Any],
    token: str | None,
    *,
    head: str,
    base: str,
    title: str,
    body: str = "",
) -> dict[str, Any]:
    """Open a PR ``head → base``. Returns ``{ok, url, number, error}``.

    A PR that already exists for the same ``head``/``base`` (GitHub 422) is not
    an error — the existing PR is looked up and returned.
    """
    try:
        provider, owner, repo = parse_repo(git_config)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "url": None, "number": None}
    if provider != "github":
        return {"ok": False, "error": f"forge {provider!r} not supported", "url": None, "number": None}

    url = f"{_api_base(git_config)}/repos/{owner}/{repo}/pulls"
    payload = {"title": title, "head": head, "base": base, "body": body}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.post(url, json=payload, headers=_headers(token))
            if r.status_code == 201:
                data = r.json()
                return {"ok": True, "url": data.get("html_url"), "number": data.get("number"), "error": None}
            if r.status_code == 422:
                existing = await _find_open_pr(c, git_config, token, owner, repo, head, base)
                if existing:
                    return {"ok": True, **existing, "error": None}
            return {
                "ok": False,
                "error": f"{r.status_code}: {r.text[:300]}",
                "url": None,
                "number": None,
            }
    except httpx.HTTPError as exc:
        logger.warning("create_pull_request: %s/%s %s→%s failed: %s", owner, repo, head, base, exc)
        return {"ok": False, "error": str(exc), "url": None, "number": None}


async def _find_open_pr(
    client: httpx.AsyncClient,
    git_config: Mapping[str, Any],
    token: str | None,
    owner: str,
    repo: str,
    head: str,
    base: str,
) -> dict[str, Any] | None:
    params = {"state": "open", "head": f"{owner}:{head}", "base": base}
    try:
        r = await client.get(
            f"{_api_base(git_config)}/repos/{owner}/{repo}/pulls",
            params=params,
            headers=_headers(token),
        )
        if r.status_code == 200 and r.json():
            pr = r.json()[0]
            return {"url": pr.get("html_url"), "number": pr.get("number")}
    except httpx.HTTPError as exc:
        logger.warning("_find_open_pr: %s/%s failed: %s", owner, repo, exc)
    return None


async def merge_pull_request(
    git_config: Mapping[str, Any],
    token: str | None,
    number: int,
    *,
    method: str = "merge",
) -> dict[str, Any]:
    """Merge PR ``number`` via the forge API. Returns ``{ok, sha, error}``."""
    try:
        provider, owner, repo = parse_repo(git_config)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "sha": None}
    if provider != "github":
        return {"ok": False, "error": f"forge {provider!r} not supported", "sha": None}

    url = f"{_api_base(git_config)}/repos/{owner}/{repo}/pulls/{number}/merge"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.put(url, json={"merge_method": method}, headers=_headers(token))
            if r.status_code == 200:
                return {"ok": True, "sha": r.json().get("sha"), "error": None}
            return {"ok": False, "error": f"{r.status_code}: {r.text[:300]}", "sha": None}
    except httpx.HTTPError as exc:
        logger.warning("merge_pull_request: %s/%s #%s failed: %s", owner, repo, number, exc)
        return {"ok": False, "error": str(exc), "sha": None}


# ── deploy keys (internal design notes Stage 13) ─────────────────────────────────
#
# A deploy key is what lets one prepared VPS read one repository, and nothing
# else: the Room's own `git_https` token is a full-account PAT and must never
# leave the Hub (§3.5). Everything below registers or removes the *public*
# half of a key pair whose private half exists only on that one server.


def _gitlab_api_base(git_config: Mapping[str, Any]) -> str:
    return str(git_config.get("api_base") or "https://gitlab.com/api/v4").rstrip("/")


def _gitlab_project(owner: str, repo: str) -> str:
    from urllib.parse import quote

    return quote(f"{owner}/{repo}", safe="")


def forge_host(git_config: Mapping[str, Any]) -> str:
    """The forge's own hostname, from the repository URL (``github.com``)."""
    from urllib.parse import urlparse

    from app.core.git_sync import _clone_url

    parsed = urlparse(_clone_url(str(git_config.get("repo_url") or "")))
    return (parsed.hostname or "").lower()


def ssh_repo_path(git_config: Mapping[str, Any]) -> str:
    """``owner/repo.git`` — the path half of an SSH clone URL."""
    _, owner, repo = parse_repo(git_config)
    return f"{owner}/{repo}.git"


def https_clone_url(git_config: Mapping[str, Any]) -> str:
    """The plain ``https://host/owner/repo.git`` URL — never with a token in it."""
    from app.core.git_sync import _clone_url

    return f"{_clone_url(str(git_config.get('repo_url') or ''))}.git"


async def add_deploy_key(
    git_config: Mapping[str, Any],
    token: str | None,
    *,
    title: str,
    public_key: str,
) -> dict[str, Any]:
    """Register ``public_key`` as a **read-only** deploy key. ``{ok, id, error}``.

    Read-only is not a preference here, it is the whole point: the key sits on
    a machine the Hub does not control, so the forge — not our policy layer —
    has to be the thing that refuses a push (Stage 13 acceptance). Both
    provider calls below therefore pass the write flag explicitly rather than
    relying on a default.
    """
    try:
        provider, owner, repo = parse_repo(git_config)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "id": None}
    if not token:
        return {"ok": False, "error": "no forge token available", "id": None}

    if provider == "github":
        url = f"{_api_base(git_config)}/repos/{owner}/{repo}/keys"
        payload: dict[str, Any] = {"title": title, "key": public_key, "read_only": True}
        headers = _headers(token)
    elif provider == "gitlab":
        url = f"{_gitlab_api_base(git_config)}/projects/{_gitlab_project(owner, repo)}/deploy_keys"
        payload = {"title": title, "key": public_key, "can_push": False}
        headers = {"PRIVATE-TOKEN": token}
    else:
        return {"ok": False, "error": f"forge {provider!r} not supported", "id": None}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.post(url, json=payload, headers=headers)
            if r.status_code in (200, 201):
                return {"ok": True, "id": str(r.json().get("id") or ""), "error": None}
            return {"ok": False, "error": f"{r.status_code}: {r.text[:300]}", "id": None}
    except httpx.HTTPError as exc:
        logger.warning("add_deploy_key: %s/%s failed: %s", owner, repo, exc)
        return {"ok": False, "error": str(exc), "id": None}


async def delete_deploy_key(
    git_config: Mapping[str, Any],
    token: str | None,
    key_id: str,
) -> dict[str, Any]:
    """Remove deploy key ``key_id``. ``{ok, error}``; a 404 counts as removed.

    Deletion is by id, stored in ``VPS_<NAME>_DEPLOY_KEY_ID`` when the key was
    issued — never by matching the title, which a person can edit in the
    forge's own UI and which would make "Revoke" quietly remove the wrong
    key or nothing at all.
    """
    try:
        provider, owner, repo = parse_repo(git_config)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if not token:
        return {"ok": False, "error": "no forge token available"}
    if not str(key_id or "").strip():
        return {"ok": False, "error": "no deploy key id recorded"}

    if provider == "github":
        url = f"{_api_base(git_config)}/repos/{owner}/{repo}/keys/{key_id}"
        headers = _headers(token)
    elif provider == "gitlab":
        url = f"{_gitlab_api_base(git_config)}/projects/{_gitlab_project(owner, repo)}/deploy_keys/{key_id}"
        headers = {"PRIVATE-TOKEN": token}
    else:
        return {"ok": False, "error": f"forge {provider!r} not supported"}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.delete(url, headers=headers)
            if r.status_code in (200, 202, 204, 404):
                return {"ok": True, "error": None}
            return {"ok": False, "error": f"{r.status_code}: {r.text[:300]}"}
    except httpx.HTTPError as exc:
        logger.warning("delete_deploy_key: %s/%s #%s failed: %s", owner, repo, key_id, exc)
        return {"ok": False, "error": str(exc)}


async def forge_ssh_host_keys(git_config: Mapping[str, Any]) -> list[str]:
    """The forge's published SSH host keys, as ``known_hosts`` key lines.

    TOFU has no place here (§3.5 is about *your* server, whose key nobody
    publishes): a forge's host keys are public, so the server is given them
    outright and ``StrictHostKeyChecking yes`` can stay on. GitHub publishes
    them at ``/meta``. Other forges do not publish keys in a machine-readable
    form we can trust, so this returns an empty list and the caller degrades
    to "add it yourself" rather than silently accepting whatever answers.
    """
    provider, _, _ = parse_repo(git_config)
    if provider != "github":
        return []
    url = f"{_api_base(git_config)}/meta"
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as c:
            r = await c.get(url, headers=_headers(None))
            if r.status_code != 200:
                return []
            keys = r.json().get("ssh_keys") or []
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("forge_ssh_host_keys: %s failed: %s", url, exc)
        return []
    return [str(k).strip() for k in keys if str(k).strip()]


async def resolve_forge_token(
    db, room, request=None, *, request_id=None, source_ip=None
) -> str | None:
    """Best-effort room Git token for a forge API call.

    A legacy inline ``access_token`` is returned directly. A secret-store
    binding needs audit context: either a ``request``, or the ``request_id`` /
    ``source_ip`` the caller already has (Stage 13's deploy-key issue runs
    inside the preparation wizard, which has no ``Request`` object but does
    have the run's request id). With neither — ``open_promotion`` fired from
    task completion — this returns ``None`` and the caller degrades.
    """
    from app.core.secrets.runtime import (
        binding_id_from_config,
        legacy_secret_value,
        resolve_for_use,
    )

    config = getattr(room, "git_config", None) or {}
    try:
        binding_id = binding_id_from_config(config, "access_token_secret_binding_id")
    except Exception:
        binding_id = None

    if binding_id is None:
        legacy = legacy_secret_value(config, "access_token")
        return legacy.reveal() if legacy else None

    if request is None and request_id is None:
        return None

    try:
        from app.core.request_id import request_id_from_request
        from app.models.secret import SecretPurpose, SecretResourceType

        if request is not None:
            request_id = request_id_from_request(request)
            source_ip = request.client.host if request.client else None

        value = await resolve_for_use(
            db=db,
            owner_id=room.creator_id,
            resource_type=SecretResourceType.ROOM,
            resource_id=room.id,
            purpose=SecretPurpose.GIT_HTTPS,
            binding_id=binding_id,
            request_id=request_id,
            source_ip=source_ip,
        )
        return value.reveal() if value else None
    except Exception as exc:
        logger.warning("resolve_forge_token: could not resolve binding for room %s: %s",
                       getattr(room, "id", "?"), exc)
        return None


__all__ = [
    "parse_repo",
    "remote_branch_exists",
    "remote_branch_head",
    "commit_relation",
    "create_pull_request",
    "merge_pull_request",
    "resolve_forge_token",
    "add_deploy_key",
    "delete_deploy_key",
    "forge_ssh_host_keys",
    "forge_host",
    "ssh_repo_path",
    "https_clone_url",
    "EXAMPLE_CI_WORKFLOW",
]
