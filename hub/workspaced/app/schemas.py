from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class StartRequest(BaseModel):
    mem_limit_mb: int = Field(default=1024, ge=128, le=8192)
    cpu_limit_milli: int = Field(default=500, ge=100, le=4000)
    disk_limit_mb: int = Field(default=2048, ge=256, le=16384)
    network_enabled: bool = True
    # Explicit image tag chosen by the Hub backend (internal design notes
    # Stage 5 §7.1 — base vs. office). None falls back to the daemon's own
    # WORKSPACE_IMAGE default, e.g. for callers that predate this field.
    image: Optional[str] = None
    # Tariffs T9: all room directories of an owner share an XFS project id.
    # The Hub resolves ownership and plan limits; only workspaced receives
    # these values because it mounts the data filesystem and Docker API.
    storage_project_id: Optional[int] = Field(default=None, ge=1, le=2_147_483_647)
    storage_quota_mb: Optional[int] = Field(default=None, ge=1, le=1_048_576)
    storage_project_room_ids: list[str] = Field(default_factory=list, max_length=512)
    # Docker overlay2 writable layer, separate from /workspace. Docker accepts
    # this only when its own data-root is on XFS with pquota.
    container_layer_mb: Optional[int] = Field(default=None, ge=1, le=1_048_576)


class StorageProjectRequest(BaseModel):
    """One owner's durable XFS project quota (tariffs T9)."""

    quota_mb: int = Field(ge=1, le=1_048_576)
    room_ids: list[str] = Field(default_factory=list, max_length=512)


class StorageProjectResponse(BaseModel):
    project_id: int
    quota_mb: int
    assigned_room_ids: list[str] = Field(default_factory=list)


class MemoryUpdateRequest(BaseModel):
    """A running room's new memory limit (internal design notes §3a, T2a)."""

    mem_limit_mb: int = Field(ge=128, le=8192)


class StatusResponse(BaseModel):
    room_id: str
    status: str  # stopped | starting | running | error
    container_id: Optional[str] = None
    uptime_sec: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    memory_limit_mb: Optional[int] = None


class SecretFile(BaseModel):
    """internal design notes Stage 5 — a value written to tmpfs for one exec.

    ``path`` must resolve under ``docker_manager.SECRETS_DIR``; the file is
    written just before the command runs and removed once it (really)
    finishes, including after a tool-level timeout.
    """

    path: str
    content: str
    mode: str = "0600"


class ExecRequest(BaseModel):
    cmd: list[str]
    cwd: str = "/workspace"
    timeout_sec: int = Field(default=30, ge=1, le=7200)
    env: Optional[dict[str, str]] = None
    secret_files: Optional[list[SecretFile]] = None


class ExecResponse(BaseModel):
    exec_id: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool = False


class StopResponse(BaseModel):
    room_id: str
    status: str
    stopped_at: str


class DestroyResponse(BaseModel):
    room_id: str
    status: str
    container_removed: bool = False
    data_removed: bool = False


class WorkspaceListResponse(BaseModel):
    room_ids: list[str]


class ErrorResponse(BaseModel):
    detail: str


# ── File operation schemas ───────────────────────────────────────────────────

class FileTreeEntry(BaseModel):
    path: str
    size: int
    mtime: float


class FileReadResponse(BaseModel):
    path: str
    content: str  # text, or data-URI for binary


class FileWriteRequest(BaseModel):
    content: str  # text, or data-URI for binary
    commit_message: Optional[str] = None


class FileWriteResponse(BaseModel):
    path: str
    size: int
    commit_sha: Optional[str] = None
    # Metadata-sync information consumed by backend agent tools. A write is
    # still successful when ignored; it simply must not create a RepoFile row.
    sync_ignored: bool = False
    sync_ignore_source: Optional[str] = None


class ImportRequest(BaseModel):
    files: list[dict]  # [{path: str, content: str}]
    commit_message: Optional[str] = None


class ImportResponse(BaseModel):
    imported: int
    errors: list[dict]
    commit_sha: Optional[str] = None


# ── Git schemas ──────────────────────────────────────────────────────────────

class GitLogEntry(BaseModel):
    sha: str
    short_sha: str
    message: str
    author: str
    timestamp: str  # ISO 8601


class GitLogResponse(BaseModel):
    entries: list[GitLogEntry]
    branch: Optional[str] = None


class GitDiffResponse(BaseModel):
    diff: str
    from_ref: Optional[str] = None
    to_ref: Optional[str] = None


class GitLsTreeEntry(BaseModel):
    path: str
    size: int


class GitLsTreeResponse(BaseModel):
    entries: list[GitLsTreeEntry]
    ref: str


class GitCommitRequest(BaseModel):
    message: str
    # internal design notes Stage 0: agent / auto commits leave this False and
    # are rejected on `main`; explicit human operations (revert / restore)
    # pass True.
    allow_main_commit: bool = False


class GitAuthorRequest(BaseModel):
    """Who the room's commits are authored by; both None clears it."""

    name: Optional[str] = Field(default=None, max_length=256)
    email: Optional[str] = Field(default=None, max_length=320)


class GitEnsureBaseBranchesResponse(BaseModel):
    ok: bool
    created_dev: bool = False
    error: Optional[str] = None


class GitBackfillExcludesResponse(BaseModel):
    """internal design notes Stage 3 — ``.git/info/exclude`` backfill result."""

    ok: bool
    updated: bool = False
    error: Optional[str] = None
    # internal design notes Stage 4 — policy findings are metadata only; no file
    # content is returned to the backend or persisted in an event.
    userfiles_created: bool = False
    # True only when ``.userfiles/`` is verifiably kept out of ``git add`` (or no
    # repo exists yet and ``git_init`` will write the rule first). Absent from
    # pre-protection workspaced builds, which the backend reads as "not safe".
    userfiles_protected: bool = False
    userfiles_policy_conflict: bool = False
    userfiles_tracked: list[str] = Field(default_factory=list)
    userfiles_tracked_details: list[dict] = Field(default_factory=list)


class GitIsTrackedResponse(BaseModel):
    """internal design notes §3.5 / Stage 8 — index-tracked check."""

    tracked: bool


class GitCommitResponse(BaseModel):
    sha: Optional[str] = None
    committed: bool


class GitRevertRequest(BaseModel):
    ref: str
    path: Optional[str] = None


class GitBranchRequest(BaseModel):
    name: str
    # internal design notes Stage 1: start point for the new branch (a branch
    # name or commit-ish). None → branch off current HEAD.
    base: Optional[str] = None


class GitBranchRenameRequest(BaseModel):
    """internal design notes Stage 2 — a dialogue's ``session/<id>``
    branch becomes the canonical ``task/<id>`` once its work is worth a task."""

    name: str
    new_name: str


class GitMergeRequest(BaseModel):
    name: str
    message: Optional[str] = None
    # internal design notes Stage 1: merge target. None keeps the historic
    # default (`main`) so the manual promote endpoint is unchanged.
    target: Optional[str] = None
    # theirs=True (default) auto-resolves overlapping hunks for task→dev/goal
    # merges; the dev→main / goal→dev gates pass False for a clean merge.
    theirs: bool = True


class GitRevertCommitRequest(BaseModel):
    """internal design notes Stage 3 — undo a ``trivial`` auto-merge."""

    sha: str
    target: Optional[str] = None
    mainline: int = 1


class GitApplyPatchRequest(BaseModel):
    """Fold a connector's ``mode=worker`` diff in."""

    branch: str
    patch: str
    message: Optional[str] = None


class GitArchiveResponse(BaseModel):
    """Tar snapshot of a branch for a worker dispatch."""

    ok: bool
    ref: str = ""
    content_b64: str = ""
    error: Optional[str] = None


class GitOpResponse(BaseModel):
    ok: bool
    conflict: bool = False
    # internal design notes §5 — the room's single working tree was
    # locked by another git operation and this one timed out waiting. Distinct
    # from a plain failure: the caller should retry, not treat it as done.
    busy: bool = False
    # internal design notes Stage 4 — "Merge histories" hit an ``add/add``
    # conflict on an untouched seed ``.gitignore`` and resolved it in favour of
    # the remote's file instead of aborting. Only ever True on a ``git_fetch``.
    gitignore_autoresolved: bool = False
    stdout: str = ""
    stderr: str = ""
    sha: Optional[str] = None
    branch: Optional[str] = None
    # internal design notes Stage 1 — a tracked private path blocks every
    # Hub-managed push/promotion before it can reach the network.
    userfiles_push_blocked: bool = False
    userfiles_tracked: list[str] = Field(default_factory=list)
    userfiles_policy_conflict: bool = False
    userfiles_tracked_details: list[dict] = Field(default_factory=list)


class GitMainBypassRequest(BaseModel):
    # Commits the forge already has (its ``main`` head); unknown ones are ignored.
    exclude: list[str] = []
    limit: int = 20


class GitReleaseTagRequest(BaseModel):
    tag: str
    ref: str = "main"
    message: str = ""
    # False: validate only (before a person is asked); True: create the tag.
    create: bool = False


class GitPushTagRequest(BaseModel):
    remote_url: str
    tag: str
    token: Optional[str] = None


class GitPushRequest(BaseModel):
    remote_url: str
    branch: str = "main"
    token: Optional[str] = None
    # "push" (default): a normal non-fast-forward-safe push; rejected if the
    # remote branch has diverged. "force": `--force-with-lease` so the user
    # can override divergence only with an explicit, acknowledged choice — see
    # internal design notes stage 2.2.
    mode: str = "push"


class GitFetchRequest(BaseModel):
    remote_url: str
    branch: str = "main"
    token: Optional[str] = None
    # internal design notes §0.1 — branch the remote is merged into. None keeps
    # the historic behaviour (merge into whatever HEAD happens to be).
    target_branch: Optional[str] = None


class GitAdoptRequest(BaseModel):
    """internal design notes §3 — take an external repo as the room's base."""

    remote_url: str
    branch: str = "main"
    token: Optional[str] = None
    working_branch: str = "dev"
    # An explicit human choice: allow the transplant even on a room that
    # already has work (still refused while an unfinished task/* branch exists).
    force: bool = False


class GitAdoptResponse(BaseModel):
    ok: bool
    adopted: bool = False
    diverged: bool = False
    busy: bool = False
    dropped_init: bool = False
    main_sha: Optional[str] = None
    working_sha: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    userfiles_policy_conflict: bool = False
    userfiles_tracked: list[str] = Field(default_factory=list)
    userfiles_tracked_details: list[dict] = Field(default_factory=list)


class GitEnsureSharedGitignoreRequest(BaseModel):
    """internal design notes Stage 2 — branch to seed the shared file on."""

    integration_branch: str = "dev"


class GitEnsureSharedGitignoreResponse(BaseModel):
    ok: bool
    added: bool = False
    busy: bool = False
    sha: Optional[str] = None
    branch: Optional[str] = None
    stderr: str = ""


# ── Branch listing (internal design notes Stage 1) ──────────────────────────────

class GitListBranchesRequest(BaseModel):
    """Maps branch name → base ref; computed by the backend from Task/RoomGoal records."""
    bases: dict[str, str] = Field(default_factory=dict)


class GitBranchInfo(BaseModel):
    name: str
    sha: str
    committed_at: Optional[str] = None
    base: Optional[str] = None
    ahead: Optional[int] = None
    behind: Optional[int] = None
    merged: Optional[bool] = None


class GitListBranchesResponse(BaseModel):
    branches: list[GitBranchInfo]


# ── Divergence review (internal design notes Stage 1-2) ──────────────────

class GitDivergenceResponse(BaseModel):
    ok: bool = False
    error: Optional[str] = None
    branch: str
    base: str
    merge_base: dict = Field(default_factory=dict)
    base_head: dict = Field(default_factory=dict)
    branch_head: dict = Field(default_factory=dict)
    ahead: int = 0
    behind: int = 0
    branch_commits: list[dict] = Field(default_factory=list)
    base_commits: list[dict] = Field(default_factory=list)
    branch_files: list[str] = Field(default_factory=list)
    base_files: list[str] = Field(default_factory=list)
    overlap_files: list[str] = Field(default_factory=list)
    conflict: bool = False
    conflict_files: list[str] = Field(default_factory=list)
    conflict_details: list[dict] = Field(default_factory=list)
    merge_messages: str = ""
    redundant: bool = False


class GitDiscardBranchRequest(BaseModel):
    branch: str


class GitDiscardBranchResponse(BaseModel):
    ok: bool = False
    deleted: bool = False
    backup_ref: Optional[str] = None
    busy: bool = False
    stderr: str = ""


# ── HTTP preview proxy ───────────────────────────────────────────────────────

class ProxyResponse(BaseModel):
    status_code: int
    content_type: str = ""
    body_b64: str = ""
    error: Optional[str] = None


# ── Run configuration schemas (Stage 5) ──────────────────────────────────────

class RunConfig(BaseModel):
    name: str
    cmd: str
    long_running: bool = False
    cwd: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    # internal design notes Stage 6 — names only (never values) of
    # Room secrets to resolve into this run's env at start time. Resolution
    # happens in the Hub backend, which alone has DB/secret-store access;
    # workspaced only ever sees the committed name list.
    secrets: list[str] = Field(default_factory=list)
    ports: list[int] = Field(default_factory=list)
    resume_policy: Literal["never", "restart"] = "never"
    readiness: dict[str, Any] = Field(default_factory=dict)


class CheckSpec(BaseModel):
    """internal design notes Stage 2 — one CI command in ``.legion/run.json``.

    ``quick`` commands run at Gate 1 (task branch → dev); the full set runs at
    Gate 2 (dev → main).
    """

    name: str
    command: str
    quick: bool = True
    # internal design notes Stage 6 — names only; resolved by the Hub
    # backend (room_git_flow.py) before the run and passed as ChecksRunRequest.env.
    secrets: list[str] = Field(default_factory=list)


class RunsConfig(BaseModel):
    runs: list[RunConfig]
    checks: list[CheckSpec] = Field(default_factory=list)


class ChecksRunRequest(BaseModel):
    scope: Literal["quick", "full"] = "quick"
    # Optional branch / commit-ish to check out before running; omitted → run
    # against whatever the shared working tree currently has checked out.
    ref: Optional[str] = None
    timeout_sec: int = Field(default=600, ge=1, le=3600)
    # internal design notes Stage 6 — resolved Room variables/secrets
    # values (never persisted; the Hub backend resolves CheckSpec.secrets
    # right before this call and forgets them once it returns).
    env: Optional[dict[str, str]] = None


class CheckRunResult(BaseModel):
    name: str
    command: str
    exit_code: int
    tail: str = ""
    timed_out: bool = False


class ChecksRunResponse(BaseModel):
    passed: bool
    ran: bool
    scope: str
    results: list[CheckRunResult] = Field(default_factory=list)
    error: Optional[str] = None


class RunStartResponse(BaseModel):
    name: str
    long_running: bool
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    pid: Optional[int] = None
    log_path: Optional[str] = None
    truncated: bool = False
    command: str = ""
    cwd: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    ports: list[int] = Field(default_factory=list)
    resume_policy: Literal["never", "restart"] = "never"
    readiness: dict[str, Any] = Field(default_factory=dict)


class RunLogsResponse(BaseModel):
    name: str
    log_path: str
    content: str
    running: bool


class RunStopResponse(BaseModel):
    name: str
    ok: bool
    message: str = ""


# ── Workspace services (browser-services stage 1) ──────────────────────────

class ServiceStartRequest(BaseModel):
    """A normalized service declaration supplied by the backend.

    This deliberately has no shell-command field: the supervisor receives an
    argv vector and starts it in a new session/process group.
    """

    service_id: str
    slug: str
    argv: list[str] = Field(min_length=1)
    cwd: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    port: int | Literal["auto"] = "auto"
    readiness: dict[str, Any] = Field(default_factory=dict)
    timeout_sec: int = Field(default=45, ge=1, le=300)

    @field_validator("port")
    @classmethod
    def validate_port(cls, value: int | str) -> int | str:
        if value != "auto" and (not isinstance(value, int) or not 1 <= value <= 65535):
            raise ValueError("port must be auto or a TCP port")
        return value


class EnvironmentPrepareRequest(BaseModel):
    environment_id: str
    slug: str
    profile: Literal["python-pip", "node-npm", "node-pnpm", "node-yarn", "none"]
    root: str = "/workspace"
    manifests: list[str] = Field(default_factory=list)
    timeout_sec: int = Field(default=600, ge=1, le=1800)

    @field_validator("root")
    @classmethod
    def validate_root(cls, value: str) -> str:
        if value != "/workspace" and not value.startswith("/workspace/"):
            raise ValueError("environment root must stay within /workspace")
        return value


class EnvironmentPrepareResponse(BaseModel):
    state: Literal["ready", "error"]
    fingerprint: Optional[str] = None
    runtime_fingerprint: Optional[str] = None
    runtime_env: dict[str, str] = Field(default_factory=dict)
    log_path: Optional[str] = None
    error_code: Optional[str] = None
    error: Optional[str] = None
    cached: bool = False


class EnvironmentLogsResponse(BaseModel):
    log_path: str
    content: str = ""
    cursor: int = 0
    rotated: bool = False


class ServiceStopRequest(BaseModel):
    """Durable fallback identity for a service the supervisor may have
    forgotten (e.g. after it restarted). Without this the caller can never
    reliably stop a real orphaned process it can no longer look up by an
    in-memory runtime map.
    """

    pid: Optional[int] = None
    pgid: Optional[int] = None
    log_path: Optional[str] = None


class ServiceRuntimeResponse(BaseModel):
    service_id: str
    slug: str
    state: Literal["starting", "ready", "stopped", "error"]
    port: Optional[int] = None
    pid: Optional[int] = None
    pgid: Optional[int] = None
    log_path: Optional[str] = None
    error_code: Optional[str] = None
    error: Optional[str] = None


class ServiceLogsResponse(BaseModel):
    service_id: str
    log_path: str
    content: str = ""
    cursor: int = 0
    rotated: bool = False
