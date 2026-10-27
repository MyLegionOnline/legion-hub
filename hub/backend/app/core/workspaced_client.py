from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx
from app.core.secret_config import configured_secret
from app.core.path_contract import normalize_workspace_path, workspace_absolute_path
from app.core.step_profile import record_workspaced_wait, workspaced_op

WORKSPACED_URL = os.getenv("WORKSPACED_URL", "http://localhost:8100")
# No built-in default: without a token workspaced refuses every call.
WORKSPACED_TOKEN = configured_secret("WORKSPACED_TOKEN") or ""

_TIMEOUT = httpx.Timeout(connect=5.0, read=130.0, write=10.0, pool=5.0)


class WorkspaceGitBusyError(RuntimeError):
    """The room's single working tree is locked by another git operation.

    internal design notes §5 — ``workspaced`` serialises every
    HEAD-moving git op per room; when this one timed out waiting it answers
    503 ``{"busy": true}`` (or, for the dict-returning ops, 200 with
    ``busy=true``). Raised here so callers get a clear "retry later" instead
    of mistaking it for a real failure or a silent no-op.
    """


def _raise_if_busy(r: httpx.Response) -> None:
    """Turn a ``workspaced`` busy signal into ``WorkspaceGitBusyError``."""
    busy = False
    if r.status_code == 503:
        try:
            busy = bool(r.json().get("busy"))
        except (ValueError, AttributeError):
            busy = False
    else:
        ctype = r.headers.get("content-type", "")
        if r.status_code == 200 and ctype.startswith("application/json"):
            try:
                busy = bool(r.json().get("busy"))
            except (ValueError, AttributeError):
                busy = False
    if busy:
        raise WorkspaceGitBusyError(
            "workspace git tree is busy — another git operation is in progress"
        )


class WorkspaceMainProtectedError(httpx.HTTPStatusError):
    """``workspaced`` refused to commit a file write because HEAD is ``main``.

    internal design notes §0.1: agents and auto commits never write
    to ``main``; ``workspaced`` answers 409 for that.  It happens when an agent
    runs ``git checkout main`` in bash and then edits a file.  A subclass of
    ``HTTPStatusError`` so callers that map statuses keep working.
    """


def _raise_if_main_protected(r: httpx.Response) -> None:
    """Turn the file endpoints' 409 (protected ``main``) into a typed error."""
    if r.status_code != 409:
        return
    try:
        detail = str(r.json().get("detail") or "")
    except (ValueError, AttributeError):
        detail = r.text
    raise WorkspaceMainProtectedError(
        detail or "commits to main are not allowed", request=r.request, response=r
    )


@dataclass
class ServiceRelayStream:
    """Open streaming response from the internal, service-bound relay."""

    client: httpx.AsyncClient
    response: httpx.Response

    async def aclose(self) -> None:
        await self.response.aclose()
        await self.client.aclose()


class _TimedTransport(httpx.AsyncHTTPTransport):
    """Times every workspaced call without touching the ~60 call sites.

    internal design notes stage 0.3 needs ``workspaced_wait`` per agent step.
    Wrapping the transport rather than each function means a call added later
    is measured for free, and there is exactly one place to get the timing
    right.  For a streamed response this measures up to the response headers:
    the body is consumed by the caller's own loop and belongs to that caller's
    time, not to the request's.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        started = time.perf_counter()
        try:
            return await super().handle_async_request(request)
        finally:
            record_workspaced_wait(
                workspaced_op(request.url.path), time.perf_counter() - started
            )


def _client(read_timeout: Optional[float] = None) -> httpx.AsyncClient:
    timeout = _TIMEOUT
    if read_timeout is not None:
        timeout = httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0)
    return httpx.AsyncClient(
        base_url=WORKSPACED_URL,
        headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"},
        timeout=timeout,
        transport=_TimedTransport(),
    )


async def start_workspace(
    room_id: str,
    mem_limit_mb: int = 1024,
    cpu_limit_milli: int = 500,
    disk_limit_mb: int = 2048,
    network_enabled: bool = True,
    image: Optional[str] = None,
    storage_project_id: Optional[int] = None,
    storage_quota_mb: Optional[int] = None,
    storage_project_room_ids: Optional[list[str]] = None,
    container_layer_mb: Optional[int] = None,
) -> dict[str, Any]:
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/start",
            json={
                "mem_limit_mb": mem_limit_mb,
                "cpu_limit_milli": cpu_limit_milli,
                "disk_limit_mb": disk_limit_mb,
                "network_enabled": network_enabled,
                "image": image,
                "storage_project_id": storage_project_id,
                "storage_quota_mb": storage_quota_mb,
                "storage_project_room_ids": storage_project_room_ids or [],
                "container_layer_mb": container_layer_mb,
            },
        )
        r.raise_for_status()
        return r.json()


async def configure_storage_project(
    *,
    project_id: int,
    quota_mb: int,
    room_ids: list[str],
) -> dict[str, Any]:
    """Apply a project's XFS limit and attach the owner's existing rooms."""
    async with _client(read_timeout=180.0) as c:
        r = await c.post(
            f"/storage-projects/{project_id}",
            json={"quota_mb": quota_mb, "room_ids": room_ids},
        )
        r.raise_for_status()
        return r.json()


async def stop_workspace(room_id: str) -> dict[str, Any]:
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/stop")
        r.raise_for_status()
        return r.json()


async def destroy_workspace(room_id: str) -> dict[str, Any]:
    """Permanently remove the room container and its volume subtree."""
    async with _client() as c:
        r = await c.delete(f"/workspaces/{room_id}")
        r.raise_for_status()
        return r.json()


async def measure_workspace_disk(room_id: str) -> dict[str, Any]:
    """Bytes the room's directory takes on the volume (internal design notes T3)."""
    async with _client(read_timeout=60.0) as c:
        r = await c.get(f"/workspaces/{room_id}/disk-usage")
        r.raise_for_status()
        return r.json()


async def archive_workspace(room_id: str) -> dict[str, Any]:
    """Pack the stopped room into the archive store and remove it from the
    volume (internal design notes §5.5, T11)."""
    async with _client(read_timeout=1800.0) as c:
        r = await c.post(f"/workspaces/{room_id}/archive")
        r.raise_for_status()
        return r.json()


async def unarchive_workspace(room_id: str) -> dict[str, Any]:
    """Bring an archived room back onto the volume (T11)."""
    async with _client(read_timeout=1800.0) as c:
        r = await c.post(f"/workspaces/{room_id}/unarchive")
        r.raise_for_status()
        return r.json()


async def open_workspace_export(room_id: str) -> tuple[httpx.AsyncClient, httpx.Response]:
    """Start streaming the room's zip ("Download the room"). The caller reads
    the response and closes both."""
    client = _client(read_timeout=600.0)
    try:
        response = await client.send(client.build_request("GET", f"/workspaces/{room_id}/export"), stream=True)
    except BaseException:
        await client.aclose()
        raise
    return client, response


async def delete_workspace_archive(room_id: str) -> dict[str, Any]:
    """Delete the room's archive for good (the 90th day, T11)."""
    async with _client(read_timeout=120.0) as c:
        r = await c.delete(f"/workspaces/{room_id}/archive")
        r.raise_for_status()
        return r.json()


async def get_data_volume() -> dict[str, Any]:
    """Size and free bytes of the volume holding the rooms (T0.16 watchdog)."""
    async with _client(read_timeout=10.0) as c:
        r = await c.get("/host/data-volume")
        r.raise_for_status()
        return r.json()


async def get_host_stats() -> dict[str, Any]:
    """Host memory and Docker OOM kills since workspaced started (capacity sizing)."""
    async with _client(read_timeout=10.0) as c:
        r = await c.get("/host/stats")
        r.raise_for_status()
        return r.json()


async def clean_workspace_dependencies(room_id: str) -> dict[str, Any]:
    """Remove the room's reproducible dependency directories (node_modules, …)."""
    async with _client(read_timeout=300.0) as c:
        r = await c.post(f"/workspaces/{room_id}/clean-dependencies")
        r.raise_for_status()
        return r.json()


async def get_workspace_stats(room_id: str) -> dict[str, Any]:
    """The room's resident memory, cached 15 s by workspaced (T2a)."""
    async with _client(read_timeout=5.0) as c:
        r = await c.get(f"/workspaces/{room_id}/stats")
        r.raise_for_status()
        return r.json()


async def update_workspace_memory(room_id: str, mem_limit_mb: int) -> dict[str, Any]:
    """Change a running room's memory limit without a restart (T2a)."""
    async with _client(read_timeout=30.0) as c:
        r = await c.post(f"/workspaces/{room_id}/memory", json={"mem_limit_mb": mem_limit_mb})
        r.raise_for_status()
        return r.json()


async def list_workspace_ids() -> list[str]:
    """Return room ids present in workspaced's containers or data volume."""
    async with _client() as c:
        r = await c.get("/workspaces")
        r.raise_for_status()
        payload = r.json()
        return [str(room_id) for room_id in payload.get("room_ids", [])]


async def get_workspace_status(room_id: str) -> dict[str, Any]:
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/status")
        r.raise_for_status()
        return r.json()


async def exec_in_workspace(
    room_id: str,
    cmd: list[str],
    cwd: str = "/workspace",
    timeout_sec: int = 30,
    env: Optional[dict[str, str]] = None,
    secret_files: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    cwd = workspace_absolute_path(cwd)
    payload: dict[str, Any] = {
        "cmd": cmd,
        "cwd": cwd,
        "timeout_sec": timeout_sec,
        "env": env or {},
    }
    if secret_files:
        payload["secret_files"] = secret_files
    # The default read timeout (130 s) covers the ordinary 120 s exec cap; a
    # longer exec (package installs) needs the HTTP read to outlive it.
    read_timeout = timeout_sec + 20.0 if timeout_sec > 100 else None
    async with _client(read_timeout=read_timeout) as c:
        r = await c.post(f"/workspaces/{room_id}/exec", json=payload)
        r.raise_for_status()
        return r.json()


async def wait_workspace_idle(room_id: str, max_wait_sec: float = 20.0) -> dict[str, Any]:
    """Wait for any exec that hit its tool timeout to actually finish in the container.

    See workspaced's /workspaces/{room_id}/idle: an exec that times out
    keeps running rather than being killed, so a snapshot taken right
    after can miss files that finish writing moments later. Call this
    before treating a snapshot as final.
    """
    timeout = httpx.Timeout(connect=5.0, read=max_wait_sec + 10.0, write=10.0, pool=5.0)
    async with httpx.AsyncClient(
        base_url=WORKSPACED_URL,
        headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"},
        timeout=timeout,
        transport=_TimedTransport(),
    ) as c:
        r = await c.get(f"/workspaces/{room_id}/idle", params={"max_wait_sec": max_wait_sec})
        r.raise_for_status()
        return r.json()


def exec_stream_client(
    room_id: str,
    cmd: list[str],
    cwd: str = "/workspace",
    timeout_sec: int = 300,
    env: Optional[dict[str, str]] = None,
):
    """Return an httpx streaming context for exec output (NDJSON, application/x-ndjson)."""
    cwd = workspace_absolute_path(cwd)
    read_timeout = timeout_sec + 60.0
    client = httpx.AsyncClient(
        base_url=WORKSPACED_URL,
        headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"},
        timeout=httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0),
    )
    return client.stream(
        "POST",
        f"/workspaces/{room_id}/exec-stream",
        json={"cmd": cmd, "cwd": cwd, "timeout_sec": timeout_sec, "env": env or {}},
    )


# ── File operations (work without a running container) ───────────────────────

async def get_workspace_tree(room_id: str) -> list[dict[str, Any]]:
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/tree")
        r.raise_for_status()
        return r.json()


async def list_dialogue_archive_files(room_id: str) -> list[dict[str, Any]]:
    """List ``.legion/plans`` + ``.legion/drafts`` files ``[{path, mtime, size}]``.

    Those directories are invisible to the tree/snapshot endpoints, so this is
    the only enumeration the plan-dialogue retention sweep can use.
    """
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/dialogue-archives")
        r.raise_for_status()
        return r.json()


async def read_workspace_file(room_id: str, path: str) -> Optional[str]:
    """Return file content, or None if not found (404)."""
    path = normalize_workspace_path(path)
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/file", params={"path": path})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()["content"]


async def get_workspace_outline(
    room_id: str, path: str, max_symbols: int = 40
) -> Optional[dict[str, Any]]:
    """Return symbols/imports for a file without transferring its contents."""
    path = normalize_workspace_path(path)
    async with _client() as c:
        r = await c.get(
            f"/workspaces/{room_id}/outline",
            params={"path": path, "max_symbols": max_symbols},
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()


async def get_workspace_snapshot(
    room_id: str,
    paths: Optional[list[str]] = None,
    max_previews: int = 20,
    preview_chars: int = 600,
) -> dict[str, Any]:
    """Return the authoritative volume snapshot and its generation."""
    params: dict[str, Any] = {
        "max_previews": max_previews,
        "preview_chars": preview_chars,
    }
    if paths:
        params["paths"] = [normalize_workspace_path(path) for path in paths]
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/snapshot", params=params)
        r.raise_for_status()
        return r.json()


async def get_workspace_generation(room_id: str) -> dict[str, Any]:
    """Cheap generation lookup backed directly by the volume."""
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/generation")
        r.raise_for_status()
        return r.json()


async def write_workspace_file(
    room_id: str,
    path: str,
    content: str,
    commit_message: Optional[str] = None,
) -> dict[str, Any]:
    path = normalize_workspace_path(path)
    body: dict[str, Any] = {"content": content}
    if commit_message:
        body["commit_message"] = commit_message
    async with _client() as c:
        r = await c.put(
            f"/workspaces/{room_id}/file",
            params={"path": path},
            json=body,
        )
        _raise_if_busy(r)
        _raise_if_main_protected(r)
        r.raise_for_status()
        return r.json()


async def delete_workspace_file(
    room_id: str,
    path: str,
    commit_message: Optional[str] = None,
) -> bool:
    path = normalize_workspace_path(path)
    params: dict[str, str] = {"path": path}
    if commit_message:
        params["commit_message"] = commit_message
    async with _client() as c:
        r = await c.delete(f"/workspaces/{room_id}/file", params=params)
        _raise_if_busy(r)
        _raise_if_main_protected(r)
        r.raise_for_status()
        return r.json().get("ok", False)


async def import_to_workspace(
    room_id: str,
    files: list[dict[str, str]],
    commit_message: Optional[str] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "files": [
            {**item, "path": normalize_workspace_path(str(item.get("path") or ""))}
            for item in files
        ]
    }
    if commit_message:
        body["commit_message"] = commit_message
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/import", json=body)
        _raise_if_busy(r)
        _raise_if_main_protected(r)
        r.raise_for_status()
        return r.json()


# ── Git operations ─────────────────────────────────────────────────────────────

async def git_commit_workspace(
    room_id: str,
    message: str,
    allow_main_commit: bool = False,
) -> dict[str, Any]:
    """Commit the workspace tree.

    ``allow_main_commit`` defaults to False so an agent / auto commit on
    ``main`` is rejected by ``workspaced`` with 409 (internal design notes
    §0.1). Explicit human operations pass True.
    """
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/git/commit",
            json={"message": message, "allow_main_commit": allow_main_commit},
        )
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def set_git_author(room_id: str, name: Optional[str], email: Optional[str]) -> dict[str, Any]:
    """Author the room's commits by a person; Legion co-authors (see git_author)."""
    async with _client() as c:
        r = await c.put(
            f"/workspaces/{room_id}/git/author",
            json={"name": name, "email": email},
        )
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def ensure_base_branches(room_id: str) -> dict[str, Any]:
    """Guarantee the room repo has both `main` and `dev` (idempotent).

    Backfill for repos created before internal design notes Stage 0; a fresh
    ``git_init`` already creates both.
    """
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/git/ensure-base-branches")
        r.raise_for_status()
        return r.json()


async def backfill_local_excludes(room_id: str) -> dict[str, Any]:
    """Backfill local Git policy and ``/.userfiles`` for a room.

    For a repository this also writes the local excludes; without a repository
    the endpoint still creates the private directory and returns ``no git
    repo`` for the Git portion.
    """
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/git/backfill-excludes")
        r.raise_for_status()
        return r.json()


async def git_is_path_tracked(room_id: str, path: str) -> bool:
    """internal design notes §3.5 / Stage 8 — is ``path`` already
    tracked in the git index? Used by ``sync_room_env_file`` to refuse writing
    secrets into a ``.env`` the room's repo (or an externally-adopted one)
    already commits.
    """
    path = normalize_workspace_path(path)
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/git/is-tracked", params={"path": path})
        r.raise_for_status()
        return bool(r.json()["tracked"])


async def git_log_workspace(
    room_id: str,
    path: Optional[str] = None,
    limit: int = 20,
    rev_range: Optional[str] = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if path:
        params["path"] = normalize_workspace_path(path, allow_empty=True)
    if rev_range:
        params["rev_range"] = rev_range
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/log", params=params)
        r.raise_for_status()
        return r.json()


async def git_diff_workspace(
    room_id: str,
    from_ref: Optional[str] = None,
    to_ref: Optional[str] = None,
    path: Optional[str] = None,
    stat: bool = False,
    merge_base: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if from_ref:
        params["from_ref"] = from_ref
    if to_ref:
        params["to_ref"] = to_ref
    if path:
        params["path"] = normalize_workspace_path(path, allow_empty=True)
    if stat:
        params["stat"] = "true"
    if merge_base:
        params["merge_base"] = "true"
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/diff", params=params)
        r.raise_for_status()
        return r.json()


async def git_ls_tree_workspace(
    room_id: str, ref: str, path: Optional[str] = None
) -> list[dict[str, Any]]:
    """Return [{"path", "size"}] for every blob at `ref` — exact byte sizes
    straight from the git object store, used to compare a candidate restore
    ref against the current volume tree without touching it."""
    params: dict[str, Any] = {"ref": ref}
    if path:
        params["path"] = normalize_workspace_path(path, allow_empty=True)
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/ls-tree", params=params)
        r.raise_for_status()
        return r.json()["entries"]


async def git_revert_workspace(room_id: str, ref: str, path: Optional[str] = None) -> dict[str, Any]:
    body: dict[str, Any] = {"ref": ref}
    if path:
        body["path"] = normalize_workspace_path(path, allow_empty=True)
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/revert", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def git_revert_commit_workspace(
    room_id: str,
    sha: str,
    target: Optional[str] = None,
    mainline: int = 1,
) -> dict[str, Any]:
    """internal design notes Stage 3 — revert a ``trivial`` auto-merge commit
    on ``target`` (``dev`` by default) with a fresh revert commit."""
    body: dict[str, Any] = {"sha": sha, "mainline": mainline}
    if target:
        body["target"] = target
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/revert-commit", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def create_workspace_branch(
    room_id: str,
    name: str,
    base: Optional[str] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name}
    if base:
        body["base"] = base
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/branch", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def rename_workspace_branch(
    room_id: str,
    name: str,
    new_name: str,
) -> dict[str, Any]:
    """Rename a local branch (internal design notes Stage 2).

    ``{"ok": false}`` when the rename could not be done — the caller keeps the
    old name rather than losing the branch.
    """
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/branch/rename",
            json={"name": name, "new_name": new_name},
        )
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def merge_workspace_branch(
    room_id: str,
    name: str,
    message: Optional[str] = None,
    target: Optional[str] = None,
    theirs: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {"name": name}
    if message:
        body["message"] = message
    if target:
        body["target"] = target
    if not theirs:
        body["theirs"] = False
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/merge", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def list_workspace_branches(
    room_id: str,
    bases: dict[str, str],
) -> list[dict[str, Any]]:
    """internal design notes Stage 1 — list branches with ahead/behind/merged.

    ``bases`` maps branch name → base ref, built by the backend from its
    Task/RoomGoal records.  Returns an empty list when the room has no repo
    or workspaced is unreachable (caller should degrade gracefully).
    """
    try:
        async with _client() as c:
            r = await c.post(f"/workspaces/{room_id}/git/branches", json={"bases": bases})
            r.raise_for_status()
            return r.json().get("branches", [])
    except Exception:
        return []


async def git_divergence_workspace(
    room_id: str,
    branch: str,
    base: str,
) -> dict[str, Any]:
    """internal design notes Stage 1 — read-only divergence report for the
    "Divergence review" ChatMaster turn.

    Degrades to ``{"ok": False, "error": ...}`` on any transport / HTTP error
    (a stale ``workspaced`` without this route, network) so the caller can show
    a degraded card instead of a silent failure."""
    try:
        async with _client() as c:
            r = await c.get(
                f"/workspaces/{room_id}/git/divergence",
                params={"branch": branch, "base": base},
            )
            if r.status_code >= 400:
                return {
                    "ok": False,
                    "error": f"workspaced /git/divergence → HTTP {r.status_code}",
                    "branch": branch, "base": base,
                }
            return r.json()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"workspaced unreachable: {exc}", "branch": branch, "base": base}


async def git_discard_branch_workspace(
    room_id: str,
    branch: str,
) -> dict[str, Any]:
    """internal design notes Stage 2 — tag-backup then delete ``branch``."""
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/git/discard-branch",
            json={"branch": branch},
        )
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def git_ahead_behind_remote_workspace(
    room_id: str,
    branch: str,
) -> Optional[dict[str, Any]]:
    """internal design notes Stage 2 — ahead/behind vs remote tracking ref.

    Uses already-fetched ``refs/remotes/origin/<branch>`` — no network call.
    Returns None when the room has no repo, no remote ref, or on any error.
    """
    try:
        async with _client() as c:
            r = await c.get(
                f"/workspaces/{room_id}/git/ahead-behind-remote",
                params={"branch": branch},
            )
            if r.status_code == 404:
                return None
            r.raise_for_status()
            data = r.json()
            return data if data.get("ahead") is not None else None
    except Exception:
        return None


async def git_push_workspace(
    room_id: str,
    remote_url: str,
    branch: str = "main",
    token: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"remote_url": remote_url, "branch": branch}
    if token:
        body["token"] = token
    if mode:
        body["mode"] = mode
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/git/push", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def git_main_bypass_workspace(
    room_id: str, exclude: Optional[list[str]] = None, limit: int = 20
) -> dict[str, Any]:
    """Non-merge commits on ``main`` that neither ``dev`` nor ``exclude`` (the
    forge's ``main`` head) has — work that skipped the promotion. Pure read."""
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/git/main-bypass",
            json={"exclude": list(exclude or []), "limit": limit},
        )
        r.raise_for_status()
        return r.json()


async def git_release_tags_workspace(room_id: str) -> dict[str, Any]:
    """Read the latest final release tag reachable from main and suggestions."""
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/git/release-tags")
        r.raise_for_status()
        return r.json()


async def git_release_tag_workspace(
    room_id: str, tag: str, ref: str = "main", message: str = "", create: bool = False
) -> dict[str, Any]:
    """Check (``create=False``) or create an annotated release tag on ``main``.

    Returns ``{ok, tag, sha, exists}`` or ``{ok: False, code, error}``.
    """
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/git/release-tag",
            json={"tag": tag, "ref": ref, "message": message, "create": create},
        )
        r.raise_for_status()
        return r.json()


async def git_push_tag_workspace(
    room_id: str, remote_url: str, tag: str, token: Optional[str] = None
) -> dict[str, Any]:
    """Push exactly ``refs/tags/<tag>`` (never forced)."""
    body: dict[str, Any] = {"remote_url": remote_url, "tag": tag}
    if token:
        body["token"] = token
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/git/push-tag", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def git_fetch_workspace(
    room_id: str,
    remote_url: str,
    branch: str = "main",
    token: Optional[str] = None,
    target_branch: Optional[str] = None,
) -> dict[str, Any]:
    """Fetch ``branch`` from the remote and merge it into ``target_branch``.

    ``target_branch`` pins the landing branch (internal design notes §0.1) so a
    sync fired while a task session holds ``task/<id>`` checked out cannot land
    the remote on that task branch. None keeps the historic behaviour.
    """
    body: dict[str, Any] = {"remote_url": remote_url, "branch": branch}
    if token:
        body["token"] = token
    if target_branch:
        body["target_branch"] = target_branch
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/git/fetch", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def archive_workspace_branch(room_id: str, ref: str) -> Optional[bytes]:
    """Tar snapshot of ``ref``'s tree.

    The base a ``mode=worker`` connector dispatch works against, in lieu of
    git access it will never get. Returns ``None`` on any failure —
    a missing repo, a bad ref, or workspaced being unreachable — so the
    caller can fail that one dispatch instead of guessing at a fallback.
    """
    import base64

    try:
        async with _client() as c:
            r = await c.get(f"/workspaces/{room_id}/git/archive", params={"ref": ref})
            r.raise_for_status()
            data = r.json()
    except Exception:
        return None
    if not data.get("ok"):
        return None
    try:
        return base64.b64decode(data.get("content_b64") or "")
    except Exception:
        return None


async def apply_patch_workspace(
    room_id: str,
    branch: str,
    patch: str,
    message: Optional[str] = None,
) -> dict[str, Any]:
    """Fold a connector's ``mode=worker`` diff onto ``branch``."""
    body: dict[str, Any] = {"branch": branch, "patch": patch}
    if message:
        body["message"] = message
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/git/apply-patch", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def git_adopt_workspace(
    room_id: str,
    remote_url: str,
    branch: str = "main",
    token: Optional[str] = None,
    working_branch: str = "dev",
    force: bool = False,
) -> dict[str, Any]:
    """internal design notes §3 — adopt an external repo as the room base.

    Unlike ``git_fetch_workspace`` (which merges the remote into
    ``working_branch``), this pins the room's local ``main`` to the remote's
    ``main`` so Gate 2 diffs and promotion pushes stay sane. Returns
    ``{ok, adopted, diverged, dropped_init, main_sha, working_sha, ...}``.
    """
    body: dict[str, Any] = {
        "remote_url": remote_url,
        "branch": branch,
        "working_branch": working_branch,
        "force": force,
    }
    if token:
        body["token"] = token
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/git/adopt", json=body)
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def git_ensure_shared_gitignore(
    room_id: str, integration_branch: str = "dev"
) -> dict[str, Any]:
    """internal design notes Stage 2 — commit a shared ``.gitignore`` on
    ``integration_branch`` once, for a repo Legion itself publishes (greenfield
    remote). Idempotent; never called for an externally adopted repo. Returns
    ``{ok, added, sha, branch, ...}``."""
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/git/ensure-shared-gitignore",
            json={"integration_branch": integration_branch},
        )
        _raise_if_busy(r)
        r.raise_for_status()
        return r.json()


async def run_workspace_checks(
    room_id: str,
    scope: str = "quick",
    ref: Optional[str] = None,
    timeout_sec: int = 600,
    env: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """internal design notes Stage 2 — run the room's declared CI checks.

    ``env`` (internal design notes Stage 6) carries resolved Room
    variables/secrets values for this one call only — never logged, never
    stored past this request.

    Returns ``{passed, ran, scope, results, error}``. ``ran`` is False when the
    room declared no matching checks.
    """
    body: dict[str, Any] = {"scope": scope, "timeout_sec": timeout_sec}
    if ref:
        body["ref"] = ref
    if env:
        body["env"] = env
    # Checks run real test suites — give the HTTP call headroom past workspaced's
    # own per-command timeout.
    async with _client(read_timeout=timeout_sec + 60) as c:
        r = await c.post(f"/workspaces/{room_id}/checks/run", json=body)
        r.raise_for_status()
        return r.json()


async def read_workspace_file_at_rev(room_id: str, path: str, rev: str) -> Optional[str]:
    """Return file content at a given git revision, or None if not found."""
    path = normalize_workspace_path(path)
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/file", params={"path": path, "rev": rev})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()["content"]


# ── HTTP preview proxy ────────────────────────────────────────────────────────

async def proxy_fetch(
    room_id: str,
    port: int,
    path: str = "",
    query: str = "",
) -> dict[str, Any]:
    """Fetch {status_code, content_type, body_b64, error} from a server inside the workspace."""
    params = {"qs": query} if query else None
    async with _client() as c:
        r = await c.get(
            f"/workspaces/{room_id}/proxy/{port}/{path.lstrip('/')}",
            params=params,
        )
        r.raise_for_status()
        return r.json()


# ── Run configuration (Stage 5) ───────────────────────────────────────────────

async def get_workspace_runs(room_id: str) -> dict[str, Any]:
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/runs")
        r.raise_for_status()
        return r.json()


async def save_workspace_runs(room_id: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    async with _client() as c:
        r = await c.put(f"/workspaces/{room_id}/runs", json={"runs": runs})
        r.raise_for_status()
        return r.json()


async def start_workspace_run(room_id: str, name: str) -> dict[str, Any]:
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/runs/{name}/start")
        r.raise_for_status()
        return r.json()


async def get_workspace_run_logs(room_id: str, name: str, lines: int = 200) -> dict[str, Any]:
    async with _client() as c:
        r = await c.get(f"/workspaces/{room_id}/runs/{name}/logs", params={"lines": lines})
        r.raise_for_status()
        return r.json()


async def stop_workspace_run(room_id: str, name: str) -> dict[str, Any]:
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/runs/{name}/stop")
        r.raise_for_status()
        return r.json()


# ── Browser service relay (stage 3) ─────────────────────────────────────────

async def open_workspace_service_relay(
    room_id: str,
    service_id: str,
    content: AsyncIterator[bytes],
) -> ServiceRelayStream:
    """Open a byte-stream relay to one already-declared service.

    The caller does not supply a port.  Workspaced derives it from the
    supervisor runtime keyed by ``service_id``; this prevents the preview
    gateway from becoming a generic internal-network proxy.
    """
    client = httpx.AsyncClient(
        base_url=WORKSPACED_URL,
        headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"},
        timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0),
    )
    try:
        request = client.build_request(
            "POST", f"/workspaces/{room_id}/services/{service_id}/relay", content=content
        )
        response = await client.send(request, stream=True)
        response.raise_for_status()
        return ServiceRelayStream(client=client, response=response)
    except Exception:
        await client.aclose()
        raise


def workspace_service_relay_ws_url(room_id: str, service_id: str, path: str, query: str) -> str:
    """Internal workspaced WebSocket URL for one service-bound frame relay."""
    from urllib.parse import quote
    base = WORKSPACED_URL.replace("https://", "wss://", 1).replace("http://", "ws://", 1).rstrip("/")
    return (
        f"{base}/workspaces/{room_id}/services/{service_id}/relay-ws"
        f"?token={quote(WORKSPACED_TOKEN, safe='')}&path={quote(path, safe='/')}&query={quote(query, safe='')}"
    )


# ── Declared workspace services (browser-services stage 1) ─────────────────

async def prepare_workspace_environment(room_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    timeout_sec = max(1, min(int(spec.get("timeout_sec") or 600), 1800))
    # Package installation legitimately takes longer than the normal bounded
    # service-start request. Keep the client timeout aligned with the
    # workspaced-side execution timeout instead of turning a slow but healthy
    # npm/pip install into a false supervisor outage after 130 seconds.
    timeout = httpx.Timeout(connect=5.0, read=timeout_sec + 60.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(
        base_url=WORKSPACED_URL,
        headers={"Authorization": f"Bearer {WORKSPACED_TOKEN}"},
        timeout=timeout,
        transport=_TimedTransport(),
    ) as c:
        r = await c.post(f"/workspaces/{room_id}/environments/prepare", json=spec)
        r.raise_for_status()
        return r.json()


async def get_workspace_environment_logs(
    room_id: str, environment_id: str, slug: str, cursor: int = 0,
) -> dict[str, Any]:
    async with _client() as c:
        r = await c.get(
            f"/workspaces/{room_id}/environments/{environment_id}/logs",
            params={"slug": slug, "cursor": max(0, cursor)},
        )
        r.raise_for_status()
        return r.json()

async def start_workspace_service(room_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/services/start", json=spec)
        r.raise_for_status()
        return r.json()


async def reserve_workspace_service_port(room_id: str, requested_port: int | None = None) -> int:
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/services/reserve-port",
            params={"requested_port": requested_port} if requested_port else None,
        )
        r.raise_for_status()
        return int(r.json()["port"])


async def stop_workspace_service(
    room_id: str, service_id: str, pid: int | None = None, pgid: int | None = None
) -> dict[str, Any]:
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/services/{service_id}/stop",
            json={"pid": pid, "pgid": pgid},
        )
        r.raise_for_status()
        return r.json()


async def wait_workspace_service_ready(
    room_id: str,
    service_id: str,
    readiness: dict[str, Any],
    timeout_sec: int = 45,
) -> dict[str, Any]:
    async with _client() as c:
        r = await c.post(
            f"/workspaces/{room_id}/services/{service_id}/wait-ready",
            params={"timeout_sec": max(1, min(timeout_sec, 300))},
            json=readiness,
        )
        r.raise_for_status()
        return r.json()


async def get_workspace_service_logs(
    room_id: str, service_id: str, cursor: int = 0, slug: str = ""
) -> dict[str, Any]:
    async with _client() as c:
        r = await c.get(
            f"/workspaces/{room_id}/services/{service_id}/logs",
            params={"cursor": cursor, "slug": slug},
        )
        r.raise_for_status()
        return r.json()


async def reconcile_workspace_services(room_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    async with _client() as c:
        r = await c.post(f"/workspaces/{room_id}/services/reconcile", json=records)
        r.raise_for_status()
        return r.json()
