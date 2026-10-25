from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Boolean, Column, DateTime, JSON, String, Text, UniqueConstraint
from sqlmodel import SQLModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Workspace(SQLModel, table=True):
    __tablename__ = "workspace"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: UUID = Field(foreign_key="room.id", unique=True, index=True)
    status: str = Field(default="stopped")  # stopped | starting | running | draining | stopping | error
    status_reason: Optional[str] = Field(default=None)
    last_error: Optional[str] = Field(default=None)
    operation_id: Optional[UUID] = Field(default=None, index=True)
    quota_owner_id: Optional[UUID] = Field(default=None, foreign_key="user.id", index=True)
    container_id: Optional[str] = Field(default=None)
    image: str = Field(default="legion-workspace:base")
    # The limit of the running container (or of its last run).
    mem_limit_mb: int = Field(default=1024)
    # The room's memory setting (internal design notes §3a, T2a): applied at
    # the next start, or at once when it fits the owner's pool. None — the
    # plan's default mem_limit_mb (core/room_memory.py).
    mem_setting_mb: Optional[int] = Field(default=None)
    cpu_limit_milli: int = Field(default=500)
    disk_limit_mb: int = Field(default=2048)
    network_enabled: bool = Field(default=True)
    last_activity_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    stopped_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    last_heartbeat_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # Last heartbeat of a member having this room open in a browser (see
    # core/billing_interval.py). An open room is billed even when idle.
    presence_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # When the room was closed (core/room_close.py): by its stop button or for
    # running out of hours. A browser view opened before this moment no
    # longer counts as presence and is shut on its next heartbeat.
    closed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # Room files on the volume (internal design notes §5.2, T3): the last
    # measurement by workspaced, summed per owner against
    # room_storage_quota_mb (core/room_storage.py).
    disk_bytes: int = Field(default=0, sa_column=Column(BigInteger, nullable=False, server_default="0"))
    disk_measured_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # Set when the disk watchdog stopped this room (core/disk_watch.py,
    # T0.16): no start and no new work until the volume has room again.
    disk_stopped_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # internal design notes §5.5 (T11): the room's files left the volume for
    # the archive store, 30 days after its owner's plan ended. A renewed plan
    # brings them back (legion_cloud/data_lifecycle.py).
    archived_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    archive_bytes: int = Field(default=0, sa_column=Column(BigInteger, nullable=False, server_default="0"))
    # The volume is authoritative; this is the last generation observed by
    # the backend and is deliberately kept as JSON so an epoch change cannot
    # be confused with a revision reset.
    workspace_generation: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    active_session_count: int = Field(default=0, nullable=False)
    draining_since: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class WorkspaceUsage(SQLModel, table=True):
    __tablename__ = "workspace_usage"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    started_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    stopped_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    mem_limit_mb: int
    stop_reason: Optional[str] = Field(default=None)  # idle | user | quota | error
    # How much of the interval ran with a container. The interval itself is
    # the room's work (billed); this split feeds the cost model only.
    container_seconds: float = Field(default=0.0)
    container_started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


class WorkspaceDiskSample(SQLModel, table=True):
    """One measurement of a room's files, kept two hours: the disk watchdog
    stops the room that grew fastest over the last hour (core/disk_watch.py)."""

    __tablename__ = "workspace_disk_sample"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: UUID = Field(index=True)
    measured_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    disk_bytes: int = Field(default=0, sa_column=Column(BigInteger, nullable=False))


class WorkspaceExec(SQLModel, table=True):
    __tablename__ = "workspace_exec"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    initiator: str  # 'user:<uuid>' | 'agent:<uuid>' | 'session:<uuid>'
    command: str
    tool_name: Optional[str] = Field(default=None)  # which tool produced this exec
    exit_code: Optional[int] = Field(default=None)
    status: str = Field(default="completed")  # running | completed | interrupted
    stdout_tail: Optional[str] = Field(default=None)
    stderr_tail: Optional[str] = Field(default=None)
    duration_ms: Optional[int] = Field(default=None)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class WorkspaceBgProc(SQLModel, table=True):
    __tablename__ = "workspace_bg_proc"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    session_id: str
    bg_id: str = Field(unique=True, index=True)  # short UUID used as tool handle
    pid: Optional[int] = Field(default=None)
    pgid: Optional[int] = Field(default=None)
    log_path: str  # relative to workspace root, e.g. ".legion/bg/<bg_id>.log"
    cmd: str
    status: str = Field(default="running")  # running | done | failed | killed
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    # Wake-on-exit (app/core/bg_wake.py): the agent ended its turn waiting for
    # this process; when it exits the owning direct session is resumed once.
    wake_on_exit: bool = Field(default=False, sa_column=Column(Boolean, nullable=False, server_default="false"))
    wake_note: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    exit_code: Optional[int] = Field(default=None)
    woken_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


class WorkspaceEnvironment(SQLModel, table=True):
    """Reproducible dependency environment shared by workspace services.

    The source tree lives on the workspace volume while a container's user
    home and global packages do not.  This record is the durable declaration
    and status for the profile-specific preparation step that bridges that
    gap before a service is launched.
    """

    __tablename__ = "workspace_environment"
    __table_args__ = (
        UniqueConstraint("workspace_id", "slug", name="uq_workspace_environment_slug"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    slug: str = Field(index=True)
    profile: str
    root: str = Field(default=".")
    manifests: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    config_hash: str = Field(default="")
    state: str = Field(default="unprepared")  # unprepared | preparing | ready | error
    prepared_fingerprint: Optional[str] = Field(default=None)
    runtime_fingerprint: Optional[str] = Field(default=None)
    runtime_env: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    log_path: Optional[str] = Field(default=None)
    log_cursor: int = Field(default=0)
    last_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class WorkspaceManagedProcess(SQLModel, table=True):
    """A restartable process explicitly registered through the Runs API.

    This is deliberately separate from ``WorkspaceBgProc``: arbitrary
    ``bash ... &`` processes are interrupted with the runtime and must never be
    resurrected automatically.
    """

    __tablename__ = "workspace_managed_process"
    __table_args__ = (
        UniqueConstraint("workspace_id", "name", name="uq_workspace_managed_process_name"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    name: str = Field(index=True)
    # ``name`` remains for the legacy Runs compatibility path.  New services
    # use the stable slug/title fields below and are still stored in this same
    # table, rather than creating a second competing process registry.
    slug: str = Field(default="", index=True)
    title: str = Field(default="")
    kind: str = Field(default="worker")  # web | api | worker
    command: str = Field(sa_column=Column(Text, nullable=False))
    argv: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    cwd: str = Field(default="/workspace")
    env: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    # internal design notes Stage 6 — names only (never values) of
    # Room secrets to resolve into this service's env at start time. See
    # ``_start_tree`` in tools/_workspace_service.py.
    secrets: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    ports: list[int] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    listen: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    resume_policy: str = Field(default="never")  # never | restart
    readiness: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    status: str = Field(default="stopped")  # starting | running | paused | stopped | error
    desired_state: str = Field(default="stopped")  # stopped | running
    runtime_state: str = Field(default="stopped")  # stopped | starting | ready | degraded | stopping | error
    pid: Optional[int] = Field(default=None)
    pgid: Optional[int] = Field(default=None)
    restart_policy: str = Field(default="manual")  # manual | on_workspace_restore
    default_path: str = Field(default="/")
    depends_on: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    declared_by: str = Field(default="agent")
    schema_version: int = Field(default=1)
    config_hash: str = Field(default="")
    environment_id: Optional[UUID] = Field(default=None, foreign_key="workspace_environment.id", index=True)
    log_path: Optional[str] = Field(default=None)
    log_cursor: int = Field(default=0)
    last_started_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    last_ready_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    last_accessed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True), nullable=True))
    last_error: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class WorkspaceServiceEvent(SQLModel, table=True):
    """Small, safe transition ledger for the Browser/catalog event stream."""

    __tablename__ = "workspace_service_event"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    service_id: UUID = Field(foreign_key="workspace_managed_process.id", index=True)
    event_type: str
    source: str = Field(default="system")
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime(timezone=True), nullable=False))


class WorkspaceCheckpoint(SQLModel, table=True):
    """Durable room/session state captured before a runtime is stopped."""

    __tablename__ = "workspace_checkpoint"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    workspace_id: UUID = Field(foreign_key="workspace.id", index=True)
    room_id: UUID = Field(foreign_key="room.id", index=True)
    operation_id: Optional[UUID] = Field(default=None, index=True)
    reason: str
    pause_reason: str
    generation: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON, nullable=False))
    validation: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    status: str = Field(default="created")  # created | validated | restored | stale | error
    force_stop: bool = Field(default=False, nullable=False)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    restored_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )


class WorkspaceSwitchRequest(SQLModel, table=True):
    """Durable idempotency record for a room switch operation.

    Keeping this row in the database means retries after a browser reconnect
    or backend restart observe the same result instead of stopping/starting a
    second pair of runtimes.
    """

    __tablename__ = "workspace_switch_request"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    request_id: UUID = Field(unique=True, index=True)
    from_room_id: UUID = Field(foreign_key="room.id", index=True)
    to_room_id: UUID = Field(foreign_key="room.id", index=True)
    quota_owner_id: UUID = Field(foreign_key="user.id", index=True)
    status: str = Field(default="pending", sa_column=Column(String, nullable=False))
    error_code: Optional[str] = Field(default=None)
    message: Optional[str] = Field(default=None)
    force_stop_confirmed: bool = Field(default=False, nullable=False)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
