"""Configuration contract for the room/workspace lifecycle.

The module contains the stage 0 policy/configuration contract and the stage 1
idempotent room-level startup operation. Later lifecycle stages should consume
this module instead of reading environment variables directly, so timeout and
transition semantics remain the same in every backend worker.

The number of concurrent workspaces is intentionally not configured here.
``N`` is ``max_running`` from the user's effective quota, resolved by
``app.core.workspace_quota.get_effective_limits`` from the Admin Center plan
and an optional per-user override.
"""

from __future__ import annotations

import logging
import os
import asyncio
import httpx
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, FrozenSet, Optional
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

DEFAULT_ACTIVE_OPERATIONS: FrozenSet[str] = frozenset(
    {
        "agent_tool_call",
        "background_process",
        "dev_server",
        "git_operation",
        "streaming_response",
    }
)

QUOTA_SOURCE = "effective_user_limits.max_running"
MANAGED_PROCESS_RESUME_POLICY = "restart_opt_in"


async def resolve_max_running_workspaces(user_id, db) -> int:
    """Return ``N`` from the user's Admin Center-backed effective quota."""
    from app.core.workspace_quota import get_effective_limits

    limits = await get_effective_limits(user_id, db)
    raw_limit = limits.get("max_running")
    if raw_limit is None:
        # ``None`` is unlimited for optional plan fields, but max_running is
        # required to be finite so the slot manager can reserve atomically.
        raise ValueError("effective quota max_running must be a finite integer")
    try:
        return max(0, int(raw_limit))
    except (TypeError, ValueError) as exc:
        raise ValueError("effective quota max_running must be an integer") from exc


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Read a non-negative integer env value and fall back predictably."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("workspace lifecycle: invalid %s=%r; using %s", name, raw, default)
        return default
    if value < minimum:
        logger.warning(
            "workspace lifecycle: %s=%s is below minimum %s; using %s",
            name,
            value,
            minimum,
            default,
        )
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env value, accepting the usual deployment spellings."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("workspace lifecycle: invalid %s=%r; using %s", name, raw, default)
    return default


def _env_transport(name: str = "WORKSPACE_STATUS_TRANSPORT") -> str:
    raw = os.getenv(name, "websocket").strip().lower()
    if raw in {"websocket", "sse", "polling"}:
        return raw
    logger.warning("workspace lifecycle: invalid %s=%r; using websocket", name, raw)
    return "websocket"


def _env_port_range(
    name: str = "WORKSPACE_SERVICE_PORT_RANGE",
) -> tuple[int, int]:
    """Read an inclusive TCP port range in ``first-last`` form."""
    default = (3000, 3999)
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        first_raw, last_raw = raw.strip().split("-", 1)
        first, last = int(first_raw), int(last_raw)
    except ValueError:
        logger.warning("workspace services: invalid %s=%r; using %s-%s", name, raw, *default)
        return default
    if not (1 <= first <= last <= 65535):
        logger.warning("workspace services: invalid %s=%r; using %s-%s", name, raw, *default)
        return default
    return first, last


@dataclass(frozen=True)
class WorkspaceLifecycleConfig:
    """Resolved backend policy for workspace lifecycle operations.

    A runtime in ``starting`` is counted by policy, as are ``running``,
    ``draining`` and ``stopping`` runtimes until their lease is released.
    Concurrency itself comes from the user's effective ``max_running`` quota.
    """

    starting_counts_toward_quota: bool = True
    allow_background_rooms: bool = True
    idle_timeout_sec: int = 900
    start_timeout_sec: int = 120
    drain_timeout_sec: int = 30
    stop_timeout_sec: int = 30
    recovery_timeout_sec: int = 60
    status_transport: str = "websocket"
    polling_initial_sec: int = 1
    polling_max_sec: int = 10
    stop_on_browser_disconnect: bool = False
    auto_resume_workspace_switch_tasks: bool = True
    force_stop_requires_confirmation: bool = True
    auto_force_stop_stale_runtime: bool = True
    bash_satisfies_read_before_edit: bool = False
    active_operations: FrozenSet[str] = DEFAULT_ACTIVE_OPERATIONS
    # Browser services are deliberately opt-in until the registry, supervisor
    # and preview gateway are rolled out in later stages.
    services_enabled: bool = False
    preview_gateway_enabled: bool = False
    service_idle_timeout_sec: int = 900
    service_port_range: tuple[int, int] = (3000, 3999)

    def __post_init__(self) -> None:
        if self.status_transport not in {"websocket", "sse", "polling"}:
            raise ValueError("status_transport must be websocket, sse or polling")
        if self.polling_initial_sec > self.polling_max_sec:
            raise ValueError("polling_initial_sec cannot exceed polling_max_sec")
        if not self.active_operations:
            raise ValueError("active_operations cannot be empty")
        if self.service_idle_timeout_sec < 1:
            raise ValueError("service_idle_timeout_sec must be positive")
        first_port, last_port = self.service_port_range
        if not (1 <= first_port <= last_port <= 65535):
            raise ValueError("service_port_range must be a valid inclusive TCP range")

    @property
    def counted_states(self) -> tuple[str, ...]:
        """Workspace states that reserve a quota slot."""
        if self.starting_counts_toward_quota:
            return ("starting", "running", "draining", "stopping")
        return ("running", "draining", "stopping")


def load_workspace_lifecycle_config() -> WorkspaceLifecycleConfig:
    """Resolve the lifecycle contract from backend environment variables."""
    return WorkspaceLifecycleConfig(
        starting_counts_toward_quota=_env_bool(
            "WORKSPACE_STARTING_COUNTS_TOWARD_QUOTA", True
        ),
        allow_background_rooms=_env_bool("WORKSPACE_ALLOW_BACKGROUND_ROOMS", True),
        idle_timeout_sec=_env_int("WORKSPACE_IDLE_TIMEOUT_SEC", 900, minimum=1),
        start_timeout_sec=_env_int("WORKSPACE_START_TIMEOUT_SEC", 120, minimum=1),
        drain_timeout_sec=_env_int("WORKSPACE_DRAIN_TIMEOUT_SEC", 30, minimum=1),
        stop_timeout_sec=_env_int("WORKSPACE_STOP_TIMEOUT_SEC", 30, minimum=1),
        recovery_timeout_sec=_env_int(
            "WORKSPACE_RECOVERY_TIMEOUT_SEC", 60, minimum=1
        ),
        status_transport=_env_transport(),
        polling_initial_sec=_env_int(
            "WORKSPACE_STATUS_POLL_INITIAL_SEC", 1, minimum=1
        ),
        polling_max_sec=_env_int("WORKSPACE_STATUS_POLL_MAX_SEC", 10, minimum=1),
        stop_on_browser_disconnect=False,
        auto_resume_workspace_switch_tasks=True,
        force_stop_requires_confirmation=True,
        auto_force_stop_stale_runtime=True,
        bash_satisfies_read_before_edit=False,
        active_operations=DEFAULT_ACTIVE_OPERATIONS,
        services_enabled=_env_bool("WORKSPACE_SERVICES_ENABLED", False),
        preview_gateway_enabled=_env_bool(
            "WORKSPACE_PREVIEW_GATEWAY_ENABLED", False
        ),
        service_idle_timeout_sec=_env_int(
            "WORKSPACE_SERVICE_IDLE_TIMEOUT_SEC", 900, minimum=1
        ),
        service_port_range=_env_port_range(),
    )


# Import-time constants make the contract convenient for callers and preserve
# the project's existing env-at-startup configuration style.  Tests and code
# that need to resolve a changed environment should call the loader directly.
CONFIG = load_workspace_lifecycle_config()
WORKSPACE_STARTING_COUNTS_TOWARD_QUOTA = CONFIG.starting_counts_toward_quota
WORKSPACE_ALLOW_BACKGROUND_ROOMS = CONFIG.allow_background_rooms
WORKSPACE_IDLE_TIMEOUT_SEC = CONFIG.idle_timeout_sec
WORKSPACE_START_TIMEOUT_SEC = CONFIG.start_timeout_sec
WORKSPACE_DRAIN_TIMEOUT_SEC = CONFIG.drain_timeout_sec
WORKSPACE_STOP_TIMEOUT_SEC = CONFIG.stop_timeout_sec
WORKSPACE_RECOVERY_TIMEOUT_SEC = CONFIG.recovery_timeout_sec
WORKSPACE_STATUS_TRANSPORT = CONFIG.status_transport
WORKSPACE_STATUS_POLL_INITIAL_SEC = CONFIG.polling_initial_sec
WORKSPACE_STATUS_POLL_MAX_SEC = CONFIG.polling_max_sec
WORKSPACE_STOP_ON_BROWSER_DISCONNECT = CONFIG.stop_on_browser_disconnect
WORKSPACE_AUTO_RESUME_SWITCH_TASKS = CONFIG.auto_resume_workspace_switch_tasks
WORKSPACE_FORCE_STOP_REQUIRES_CONFIRMATION = CONFIG.force_stop_requires_confirmation
WORKSPACE_AUTO_FORCE_STOP_STALE_RUNTIME = CONFIG.auto_force_stop_stale_runtime
WORKSPACE_BASH_SATISFIES_READ_BEFORE_EDIT = CONFIG.bash_satisfies_read_before_edit
WORKSPACE_SERVICES_ENABLED = CONFIG.services_enabled
WORKSPACE_PREVIEW_GATEWAY_ENABLED = CONFIG.preview_gateway_enabled
WORKSPACE_SERVICE_IDLE_TIMEOUT_SEC = CONFIG.service_idle_timeout_sec
WORKSPACE_SERVICE_PORT_RANGE = CONFIG.service_port_range


# ---------------------------------------------------------------------------
# Stage 1: idempotent room-level startup.
#
# The lock is only an optimisation for requests handled by one backend
# process.  The durable ``operation_id`` and ``quota_owner_id`` fields are
# the source of truth and make a retry safe after a process restart.
# ---------------------------------------------------------------------------

_ROOM_LOCKS: dict[UUID, asyncio.Lock] = {}
_START_TASKS: set[asyncio.Task] = set()
_EVENT_SEQUENCES: dict[UUID, int] = {}


class WorkspaceStartError(RuntimeError):
    """Stable, user-facing failure for an automatic workspace start."""

    def __init__(self, error_code: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


def _room_lock(room_id: UUID) -> asyncio.Lock:
    lock = _ROOM_LOCKS.get(room_id)
    if lock is None:
        lock = asyncio.Lock()
        _ROOM_LOCKS[room_id] = lock
    return lock


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _workspace_payload(ws: Any) -> dict[str, Any]:
    starting_duration = None
    if ws.status == "starting":
        anchor = getattr(ws, "started_at", None) or getattr(ws, "created_at", None)
        if anchor is not None:
            if anchor.tzinfo is None:
                anchor = anchor.replace(tzinfo=timezone.utc)
            starting_duration = max(0.0, (_now() - anchor).total_seconds())
    return {
        "workspace_id": ws.id,
        "room_id": ws.room_id,
        "status": ws.status,
        "status_reason": ws.status_reason,
        "last_error": ws.last_error,
        "operation_id": ws.operation_id,
        "quota_owner_id": ws.quota_owner_id,
        "container_id": ws.container_id,
        "image": ws.image,
        "mem_limit_mb": ws.mem_limit_mb,
        "cpu_limit_milli": ws.cpu_limit_milli,
        "disk_limit_mb": ws.disk_limit_mb,
        "network_enabled": ws.network_enabled,
        "last_activity_at": ws.last_activity_at,
        "started_at": ws.started_at,
        "stopped_at": ws.stopped_at,
        "last_heartbeat_at": ws.last_heartbeat_at,
        "workspace_generation": ws.workspace_generation,
        "active_session_count": ws.active_session_count,
        "draining_since": ws.draining_since,
        "starting_duration_sec": starting_duration,
        "metadata_synced_at": getattr(ws, "last_heartbeat_at", None),
        "background_processes": [],
        "managed_processes": [],
    }


async def _broadcast_workspace_status(
    ws: Any,
    *,
    previous_status: Optional[str],
    operation_id: Optional[UUID],
    reason: Optional[str] = None,
    error: Optional[dict[str, str]] = None,
    request_id: Optional[UUID] = None,
    event_type: Optional[str] = None,
) -> None:
    """Publish the v1 lifecycle envelope without creating chat history rows.

    Workspace lifecycle frames intentionally use a separate WS message type
    instead of being persisted as chat history.  ``event_type`` is overridable
    for semantic events such as ``workspace.start.requested`` and
    ``workspace.snapshot.created``; status transitions retain the stable
    ``workspace.<status>`` name.
    """
    try:
        from app.websockets.manager import manager

        sequence = _EVENT_SEQUENCES.get(ws.room_id, 0) + 1
        _EVENT_SEQUENCES[ws.room_id] = sequence
        await manager.broadcast(
            {
                "type": "workspace_status",
                "event_id": str(uuid4()),
                "event_type": event_type or f"workspace.{ws.status}",
                "schema_version": 1,
                "sequence": sequence,
                "room_id": str(ws.room_id),
                "workspace_id": str(ws.id),
                "operation_id": str(operation_id) if operation_id else None,
                "request_id": str(request_id) if request_id else None,
                "previous_status": previous_status,
                "status": ws.status,
                "generation": ws.workspace_generation,
                "reason": reason,
                "error": error,
                "created_at": _now().isoformat(),
            },
            ws.room_id,
        )
        logger.info(
            "workspace lifecycle event",
            extra={
                "room_id": str(ws.room_id),
                "workspace_operation_id": str(operation_id) if operation_id else None,
                "session_id": None,
                "task_id": None,
                "canonical_path": None,
                "generation": ws.workspace_generation,
                "duration": None,
                "error_category": (error or {}).get("error_code") if error else None,
                "event_type": event_type or f"workspace.{ws.status}",
            },
        )
    except Exception as exc:  # lifecycle status must never break startup
        logger.warning("Failed to broadcast workspace status for %s: %s", ws.room_id, exc)

    # Semantic events (start.requested, snapshot.created, ...) carry an
    # unchanged status; only real transitions move the sidebar dot.
    if event_type is None:
        await publish_room_runtime(ws.room_id, ws.status)


async def publish_room_runtime(room_id: UUID, status: Optional[str]) -> None:
    """Tell every member of the room, on the per-user channel, that its
    runtime status changed (internal design notes, stage 2).

    The room socket above reaches only whoever has this room open; the sidebar
    dot of every other member would stay stale until a reload — while the room
    keeps being billed. The frame is deliberately just ``{room_id,
    runtime_status}``: the user channel is not bound to a room, so it must not
    carry anything beyond what membership alone entitles one to see.
    """
    await _send_to_room_members(
        room_id,
        {"type": "room_runtime", "room_id": str(room_id), "runtime_status": status},
    )


async def publish_room_billing(room_id: UUID, active: bool) -> None:
    """Tell the room's members whether its hours are being drawn right now.

    This is what the sidebar dot shows: an open ``workspace_usage`` interval,
    container or not (core/billing_interval.py). Sent after the commit that
    opened or closed the interval, never before.
    """
    await _send_to_room_members(
        room_id,
        {"type": "room_billing", "room_id": str(room_id), "billing_active": bool(active)},
    )


def room_closed_frame(room_id: UUID, reason: str, message: Optional[str]) -> dict:
    return {
        "type": "room_closed",
        "room_id": str(room_id),
        "reason": reason,
        "message": message,
    }


async def publish_room_closed(room_id: UUID, reason: str, message: Optional[str] = None) -> None:
    """Tell every open view of the room that it was closed (core/room_close.py).

    Both channels: the room socket (the tab showing the room leaves it) and
    the user channel (every tab drops the room from view, whatever it shows).
    """
    from app.websockets.manager import manager

    frame = room_closed_frame(room_id, reason, message)
    try:
        await manager.broadcast(frame, room_id)
    except Exception as exc:
        logger.warning("Failed to broadcast room_closed for %s: %s", room_id, exc)
    await _send_to_room_members(room_id, frame)


async def _send_to_room_members(room_id: UUID, frame: dict) -> None:
    try:
        from app.websockets.manager import manager

        # Nobody listening on the user channel — skip the membership lookup.
        if not manager.has_user_event_listeners():
            return

        from app.db.session import engine
        from app.models.room import RoomUserLink

        # Own short session: callers are mid-transaction on theirs, and this
        # read must neither join nor disturb it.
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as db:
            result = await db.execute(
                select(RoomUserLink.user_id).where(RoomUserLink.room_id == room_id)
            )
            member_ids = list(result.scalars().all())
        if not member_ids:
            return
        await manager.send_to_users(frame, member_ids)
    except Exception as exc:  # the sidebar dot must never break the lifecycle
        logger.warning("Failed to publish %s for %s: %s", frame.get("type"), room_id, exc)


async def broadcast_workspace_event(
    ws: Any,
    event_type: str,
    *,
    reason: Optional[str] = None,
    error: Optional[dict[str, str]] = None,
    operation_id: Optional[UUID] = None,
    request_id: Optional[UUID] = None,
) -> None:
    """Publish a semantic stage-6 event using the lifecycle envelope."""
    await _broadcast_workspace_status(
        ws,
        previous_status=ws.status,
        operation_id=operation_id if operation_id is not None else ws.operation_id,
        reason=reason,
        error=error,
        request_id=request_id,
        event_type=event_type,
    )


async def _count_reserved_slots(
    room_id: UUID, quota_owner_id: UUID, db: AsyncSession
) -> int:
    """Count slots owned by the caller, including legacy unowned rows."""
    from app.models.room import RoomUserLink
    from app.models.workspace import Workspace

    counted = CONFIG.counted_states
    result = await db.execute(
        select(Workspace.id)
        .join(RoomUserLink, RoomUserLink.room_id == Workspace.room_id)
        .where(RoomUserLink.user_id == quota_owner_id)
        .where(Workspace.room_id != room_id)
        .where(Workspace.status.in_(counted))
        .where(
            or_(
                Workspace.quota_owner_id == quota_owner_id,
                Workspace.quota_owner_id.is_(None),
            )
        )
        .with_for_update()
    )
    return len(result.all())


async def _get_or_create_workspace(room_id: UUID, db: AsyncSession) -> Any:
    """Idempotent get-or-create, race-safe against a concurrent caller.

    _room_lock below only serializes callers *within this process*; it does
    not (and, since the actual commit happens later in the request/session
    lifecycle rather than at this flush, cannot cleanly) block a second
    concurrent /open request from also observing "no row yet" and racing
    this insert — that happened for real (POST /rooms/{id}/open 500,
    "duplicate key value violates unique constraint ix_workspace_room_id").
    A SAVEPOINT lets the losing insert be rolled back on its own without
    discarding whatever else this request's session has already staged,
    then re-reads the row the winner created.
    """
    from sqlalchemy.exc import IntegrityError
    from app.models.workspace import Workspace

    result = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ws = result.scalar_one_or_none()
    if ws is None:
        try:
            async with db.begin_nested():
                ws = Workspace(room_id=room_id)
                db.add(ws)
                await db.flush()
        except IntegrityError:
            result = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
            ws = result.scalar_one_or_none()
            if ws is None:
                raise
    return ws


async def _run_start_operation(
    room_id: UUID,
    operation_id: UUID,
    *,
    mem_limit_mb: int,
    cpu_limit_milli: int,
    disk_limit_mb: int,
    network_enabled: bool,
    owner_id: UUID,
    storage_project_id: Optional[int],
    storage_quota_mb: Optional[int],
    storage_project_room_ids: list[str],
    container_layer_mb: Optional[int],
    db_factory: Any,
) -> None:
    from app.core import workspaced_client as wsd
    from app.core.workspace_image import resolve_workspace_image
    from app.models.room import Room
    from app.models.workspace import Workspace

    async with db_factory() as db:
        result = await db.execute(
            select(Workspace).where(
                Workspace.room_id == room_id,
                Workspace.operation_id == operation_id,
            )
        )
        ws = result.scalar_one_or_none()
        if ws is None:
            return
        room = await db.get(Room, room_id)
        if room is None or room.deletion_started_at is not None:
            # A room deletion may have started after the public open/start
            # endpoint accepted this operation. Do not create a new runtime;
            # workspaced's per-room destroy lock handles the inverse ordering.
            return
        previous = ws.status
        try:
            image = await resolve_workspace_image(room_id, db)
            # Stage 1.2 (internal design notes §8.2): the Docker start can take
            # tens of seconds — with ``background=False`` this is the calling
            # tool loop's own session, so do not hold its connection meanwhile.
            from app.db.release import release_db

            await release_db(db, reason="workspace/start")
            live = await wsd.start_workspace(
                room_id=str(room_id),
                mem_limit_mb=mem_limit_mb,
                cpu_limit_milli=cpu_limit_milli,
                disk_limit_mb=disk_limit_mb,
                network_enabled=network_enabled,
                image=image,
                storage_project_id=storage_project_id,
                storage_quota_mb=storage_quota_mb,
                storage_project_room_ids=storage_project_room_ids,
                container_layer_mb=container_layer_mb,
            )
            now = _now()
            # The room's last use before this start: the stamp below would
            # hide an unobserved gap from the billing interval.
            prior_activity_at = ws.last_activity_at
            ws.status = live.get("status", "running")
            ws.status_reason = "ready"
            ws.last_error = None
            ws.container_id = live.get("container_id")
            ws.image = image
            if live.get("workspace_generation") is not None:
                ws.workspace_generation = live.get("workspace_generation")
            ws.mem_limit_mb = mem_limit_mb
            ws.cpu_limit_milli = cpu_limit_milli
            ws.disk_limit_mb = disk_limit_mb
            ws.network_enabled = network_enabled
            ws.last_activity_at = now
            ws.last_heartbeat_at = now
            ws.started_at = now
            ws.stopped_at = None
            ws.quota_owner_id = owner_id
            db.add(ws)
            # One open usage interval per room. It may already be open — the
            # room was working without a container (billing_interval) — in
            # which case the container just joins it — unless it outlived an
            # unobserved gap, which is split off unbilled.
            from app.core.billing_interval import ensure_usage_open

            await ensure_usage_open(
                ws, db, container=True, last_activity_at=prior_activity_at
            )
            await db.commit()
            from app.core.git_author import sync_room_author

            await sync_room_author(room, db)
            checkpoint_ready = True
            try:
                from app.core.workspace_checkpoint import restore_workspace_checkpoint

                generation = ws.workspace_generation
                if generation is None:
                    generation_payload = await wsd.get_workspace_generation(str(room_id))
                    generation = generation_payload.get("generation", generation_payload)
                    ws.workspace_generation = generation
                    db.add(ws)
                    await db.commit()
                checkpoint_validation = await restore_workspace_checkpoint(
                    room_id,
                    db,
                    generation=generation,
                )
                if checkpoint_validation and any(
                    item.get("status") == "error"
                    for item in checkpoint_validation.get("managed_processes") or []
                ):
                    checkpoint_ready = False
            except Exception:
                # Runtime availability wins over checkpoint restoration. The
                # checkpoint remains pending/stale and can be retried on open.
                logger.exception(
                    "workspace start: checkpoint restore failed for %s",
                    room_id,
                )
                checkpoint_ready = False
            if CONFIG.auto_resume_workspace_switch_tasks and checkpoint_ready:
                try:
                    await resume_workspace_switch_tasks(room_id, db)
                except Exception:
                    # A task checkpoint must never turn a successful runtime
                    # start into an error; it is retried on room open.
                    logger.exception(
                        "workspace start: failed to resume switch-paused tasks for %s",
                        room_id,
                    )
            await _broadcast_workspace_status(
                ws,
                previous_status=previous,
                operation_id=operation_id,
                reason="ready",
            )
        except WorkspaceStartError:
            raise
        except httpx.HTTPStatusError as exc:
            await _mark_start_error(
                db,
                ws,
                operation_id,
                "workspaced_error",
                f"Could not start the workspace (HTTP {exc.response.status_code}).",
                previous,
            )
        except httpx.RequestError:
            await _mark_start_error(
                db,
                ws,
                operation_id,
                "workspaced_unreachable",
                "The workspace is unavailable. Try again.",
                previous,
            )
        except Exception:
            logger.exception("Workspace start failed for room %s", room_id)
            await _mark_start_error(
                db,
                ws,
                operation_id,
                "start_failed",
                "Could not start the workspace. Try again.",
                previous,
            )


async def _mark_start_error(
    db: AsyncSession,
    ws: Any,
    operation_id: UUID,
    error_code: str,
    message: str,
    previous_status: str,
) -> None:
    ws.status = "error"
    ws.status_reason = error_code
    ws.last_error = message
    ws.stopped_at = _now()
    db.add(ws)
    await db.commit()
    await _broadcast_workspace_status(
        ws,
        previous_status=previous_status,
        operation_id=operation_id,
        reason=error_code,
        error={"error_code": error_code, "message": message},
    )


def _track_start_task(task: asyncio.Task) -> None:
    _START_TASKS.add(task)
    task.add_done_callback(_START_TASKS.discard)


@asynccontextmanager
async def _reuse_session(db: AsyncSession):
    """Adapter used by synchronous/unit-test startup calls."""
    yield db


async def _restart_for_image_mismatch(room_id: UUID, db: AsyncSession) -> None:
    """Best-effort: stop a running workspace whose image no longer matches what
    its task-assigned agents need (internal design notes Stage 5
    §7.1, "Note on changing the image" — e.g. an office/executable skill was
    assigned to an agent after the container already started on the base
    image).

    This deliberately runs *before* ``_room_lock`` is acquired in
    ``ensure_workspace_started`` below: ``stop_workspace`` takes the same
    per-room lock internally and ``asyncio.Lock`` is not reentrant, so it
    cannot be called from inside the locked section without deadlocking. A
    race here (another open() interleaving) just means the mismatch is
    retried on the next open rather than acted on immediately — the cold
    start path re-resolves the image itself regardless.
    """
    from app.models.workspace import Workspace

    result = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ws = result.scalar_one_or_none()
    if ws is None or ws.status != "running":
        return
    try:
        from app.core.workspace_image import resolve_workspace_image

        desired_image = await resolve_workspace_image(room_id, db)
    except Exception:
        logger.exception("workspace image resolution failed for %s", room_id)
        return
    if not desired_image or desired_image == ws.image:
        return
    if not ws.image:
        # We never recorded what this container actually started on — a legacy
        # row, or a reconcile that revived the workspace (below) without
        # stamping the image. ``None != "legion-workspace:base"`` would then
        # force-restart a perfectly healthy container on *every* room open.
        # Stamp the best estimate and leave the runtime alone; a genuine
        # base→office switch is caught on the next open once the image is known.
        ws.image = desired_image
        db.add(ws)
        await db.commit()
        logger.info(
            "workspace image unknown for room %s — stamped %s, no restart",
            room_id, desired_image,
        )
        return
    logger.info(
        "workspace image mismatch for room %s: running=%s desired=%s — restarting",
        room_id, ws.image, desired_image,
    )
    try:
        await pause_workspace_tasks(room_id, db, pause_reason="image_switch")
        await stop_workspace(room_id, db, reason="image_switch", force=True)
    except Exception:
        logger.exception(
            "workspace image restart failed for %s; keeping current runtime", room_id
        )


async def ensure_workspace_started(
    room_id: UUID,
    db: AsyncSession,
    *,
    quota_owner_id: Optional[UUID] = None,
    mem_limit_mb: Optional[int] = None,
    cpu_limit_milli: int = 500,
    disk_limit_mb: int = 2048,
    network_enabled: bool = True,
    background: bool = True,
) -> dict[str, Any]:
    """Ensure one room has a workspace start operation in progress.

    The operation is safe to call repeatedly.  By default it reserves the
    quota lease and returns ``starting`` immediately; the actual Docker call
    runs with a fresh DB session so the request connection is not held.

    *mem_limit_mb* overrides the room's memory setting for this start; by
    default the room's own limit is used (core/room_memory.py).
    """
    from app.core.host_capacity import SERVER_BUSY, SERVER_BUSY_TEXT, HostCapacityExceeded
    from app.core.room_memory import POOL_FULL, effective_mb, pool_full_text
    from app.core.workspace_quota import (
        EXPIRED_PLAN_TIER,
        PLAN_EXPIRED_READ_ONLY,
        WorkspaceMemoryPoolExceeded,
        WorkspaceQuotaExceeded,
        get_effective_limits,
        quota_manager,
    )

    from app.db.session import engine
    from app.models.room import Room
    from app.models.workspace import Workspace

    # Stage 5: a running workspace on the wrong image is restarted before the
    # normal open/start flow below runs (see _restart_for_image_mismatch for
    # why this must happen outside _room_lock).
    await _restart_for_image_mismatch(room_id, db)

    async with _room_lock(room_id):
        room = await db.get(Room, room_id)
        if room is None:
            raise WorkspaceStartError(
                "room_not_found",
                "Room not found.",
                status_code=404,
            )
        # T9 stays tied to the room's owner even if an old, shared-room row
        # still reaches this API.  Otherwise opening somebody else's room
        # could reassign its XFS subtree to the member's project id.
        storage_owner_id = room.creator_id
        if quota_owner_id is None:
            quota_owner_id = room.creator_id
        ws = await _get_or_create_workspace(room_id, db)
        # Serialize the lease transition at the database level as well as in
        # this process.  PostgreSQL honours the row lock; SQLite simply
        # ignores ``FOR UPDATE`` and remains covered by the process lock.
        locked = await db.execute(
            select(Workspace).where(Workspace.id == ws.id).with_for_update()
        )
        ws = locked.scalar_one()
        if ws.status in {"draining", "stopping"}:
            # A stop runs entirely under ``_room_lock``, which we hold now, so
            # this is a stop abandoned midway — e.g. a room switch whose request
            # died after the checkpoint. The reconcile below used to see the
            # still-live container and revive the room to "running": the stop
            # the user consented to was silently undone, and the room kept the
            # quota slot the switch had been freeing. Finish the stop; the
            # cold start below then restores from the checkpoint it took.
            logger.info(
                "workspace open: finishing abandoned stop for room %s (status=%s)",
                room_id, ws.status,
            )
            await _stop_workspace_locked(
                room_id, db, reason="stop_completed", force=True
            )
            ws = (
                await db.execute(
                    select(Workspace).where(Workspace.id == ws.id).with_for_update()
                )
            ).scalar_one()
        if ws.status == "running":
            if ws.quota_owner_id is None:
                ws.quota_owner_id = quota_owner_id
                ws.last_heartbeat_at = _now()
                db.add(ws)
                await db.commit()
            try:
                from app.core import workspaced_client as wsd
                from app.core.workspace_checkpoint import restore_workspace_checkpoint
                from app.models.workspace import WorkspaceCheckpoint

                pending_checkpoint = (
                    await db.execute(
                        select(WorkspaceCheckpoint.id)
                        .where(
                            WorkspaceCheckpoint.room_id == room_id,
                            WorkspaceCheckpoint.status.in_(["created", "stale"]),
                        )
                        .limit(1)
                    )
                ).first()
                if pending_checkpoint:
                    generation_payload = await wsd.get_workspace_generation(str(room_id))
                    generation = generation_payload.get("generation", generation_payload)
                    validation = await restore_workspace_checkpoint(
                        room_id,
                        db,
                        generation=generation,
                    )
                    processes_ready = not validation or not any(
                        item.get("status") == "error"
                        for item in validation.get("managed_processes") or []
                    )
                    if CONFIG.auto_resume_workspace_switch_tasks and processes_ready:
                        await resume_workspace_switch_tasks(room_id, db)
            except Exception:
                logger.exception(
                    "workspace open: pending checkpoint retry failed for %s",
                    room_id,
                )
            return _workspace_payload(ws)
        if ws.status == "starting" and ws.operation_id:
            return _workspace_payload(ws)

        # Reconcile unknown/stale DB state before reserving another slot.
        if ws.status not in {"stopped", "error"}:
            from app.core import workspaced_client as wsd

            try:
                live = await wsd.get_workspace_status(str(room_id))
            except Exception:
                live = {}
            if live.get("status") == "running":
                now = _now()
                ws.status = "running"
                ws.status_reason = "reconciled"
                ws.last_error = None
                ws.container_id = live.get("container_id")
                if live.get("workspace_generation") is not None:
                    ws.workspace_generation = live.get("workspace_generation")
                if not ws.image:
                    # Keep the image column truthful across a revive so the next
                    # open's mismatch check has something real to compare.
                    try:
                        from app.core.workspace_image import resolve_workspace_image

                        ws.image = await resolve_workspace_image(room_id, db)
                    except Exception:
                        logger.debug("reconcile: image resolve skipped for %s", room_id)
                ws.quota_owner_id = ws.quota_owner_id or quota_owner_id
                ws.started_at = ws.started_at or now
                ws.last_heartbeat_at = now
                await db.commit()
                await _broadcast_workspace_status(
                    ws,
                    previous_status="unknown",
                    operation_id=ws.operation_id,
                    reason="reconciled",
                )
                return _workspace_payload(ws)

        limits = await get_effective_limits(quota_owner_id, db)
        if limits.get("plan_tier") == EXPIRED_PLAN_TIER:
            # TR4: an expired plan's rooms are read-only — never "out of hours".
            raise WorkspaceStartError(
                "plan_expired", PLAN_EXPIRED_READ_ONLY, status_code=402,
            )
        # §5.5 (T11): a frozen room does not start; an archived one waits
        # for its files (the open path and the reconciler bring them back).
        from app.core import work_gates

        refusal = await work_gates.start_refusal(room_id, db)
        if refusal is not None:
            raise WorkspaceStartError(refusal.reason, refusal.text, status_code=refusal.status_code)
        # internal design notes §3a (T2a): the room's memory setting, up to
        # the plan's room ceiling; the pool is checked with the slot below.
        mem_limit_mb = effective_mb(limits, ws.mem_setting_mb, mem_limit_mb)
        monthly_limit = limits.get("monthly_hours")
        if monthly_limit is not None:
            from app.core.workspace_quota import hours_limit_exhausted

            # The month's hours and bought hour packs (payments plan §3.4).
            if await hours_limit_exhausted(quota_owner_id, db, monthly_limit):
                raise WorkspaceStartError(
                    "monthly_hours_exceeded",
                    "The monthly workspace hours limit is exhausted.",
                    status_code=402,
                )
        # internal design notes §5.3 (T3): room files at the quota — no
        # room starts until the owner frees space.
        from app.core.room_storage import PAUSE_REASON as STORAGE_FULL
        from app.core.room_storage import refusal_text, storage_status

        storage = await storage_status(quota_owner_id, db)
        if storage.full:
            raise WorkspaceStartError(
                STORAGE_FULL, refusal_text(storage), status_code=402,
            )
        # T9: application accounting above is user-facing and sampled; the
        # XFS project id below is the hard kernel limit.  It is only allocated
        # for a finite plan, so a normal self-hosted Hub retains its unlimited
        # storage behaviour even when workspaced has no XFS support.
        from app.core.storage_projects import finite_positive_mb, project_for_owner

        storage_limits = limits
        if storage_owner_id != quota_owner_id:
            storage_limits = await get_effective_limits(storage_owner_id, db)
        storage_project = await project_for_owner(
            storage_owner_id, db, limits=storage_limits
        )
        storage_project_id = storage_project.project_id if storage_project else None
        storage_quota_mb = storage_project.quota_mb if storage_project else None
        storage_project_room_ids = storage_project.room_ids if storage_project else []
        container_layer_mb = finite_positive_mb(storage_limits.get("container_layer_mb"))
        # ``disk_limit_mb`` predates tariffed storage and remains in the
        # workspace status contract.  For a finite cloud plan it must report
        # the same writable-layer limit that Docker actually receives, not a
        # caller-controlled legacy value (which otherwise stays at 2 GiB for
        # Pro while its StorageOpt is 4 GiB).
        if container_layer_mb is not None:
            disk_limit_mb = container_layer_mb
        # internal design notes T0.16: stopped by the disk watchdog — no
        # start until the data volume has room again.
        if getattr(ws, "disk_stopped_at", None) is not None:
            from app.core import disk_watch

            raise WorkspaceStartError(
                disk_watch.PAUSE_REASON, disk_watch.refusal_text(), status_code=507,
            )
        previous = ws.status
        operation_id = uuid4()

        # Let connected room clients render the opening sequence immediately,
        # before the durable ``starting`` transition is committed.
        await _broadcast_workspace_status(
            ws,
            previous_status=previous,
            operation_id=operation_id,
            reason="opening_room",
            event_type="workspace.start.requested",
        )

        async def _materialize_start() -> dict[str, Any]:
            ws.status = "starting"
            ws.status_reason = "opening_room"
            ws.last_error = None
            ws.operation_id = operation_id
            ws.quota_owner_id = quota_owner_id
            ws.container_id = None
            # Counted in the owner's memory pool from this moment on.
            ws.mem_limit_mb = mem_limit_mb
            # Keep the immediate ``starting`` API response truthful too;
            # waiting for the background Docker call would expose the stale
            # legacy default during the whole start operation.
            ws.disk_limit_mb = disk_limit_mb
            db.add(ws)
            await db.commit()
            await db.refresh(ws)
            return _workspace_payload(ws)

        try:
            starting_payload = await quota_manager.reserve_and_run(
                quota_owner_id, room_id, db, _materialize_start, mem_mb=mem_limit_mb
            )
        except ValueError as exc:
            raise WorkspaceStartError(
                "quota_unavailable",
                "Could not determine the workspace limit. Try again.",
                status_code=503,
            ) from exc
        except WorkspaceQuotaExceeded as exc:
            raise WorkspaceStartError(
                "quota_exceeded",
                "The limit of simultaneously running workspaces has been reached.",
                status_code=429,
            ) from exc
        except WorkspaceMemoryPoolExceeded as exc:
            raise WorkspaceStartError(
                POOL_FULL,
                pool_full_text(exc.pool_mb, exc.allocated_mb),
                status_code=402,
            ) from exc
        except HostCapacityExceeded as exc:
            # internal design notes stage 12.1: the machine is full, not the plan.
            logger.warning(
                "workspace start refused for room %s: server %s capacity %s, used %s, requested %s",
                room_id, exc.kind, exc.limit, exc.used, exc.requested,
            )
            raise WorkspaceStartError(
                SERVER_BUSY, SERVER_BUSY_TEXT, status_code=503,
            ) from exc
        await _broadcast_workspace_status(
            ws,
            previous_status=previous,
            operation_id=operation_id,
            reason="opening_room",
        )

        if not background:
            factory = lambda: _reuse_session(db)
            await _run_start_operation(
                room_id,
                operation_id,
                mem_limit_mb=mem_limit_mb,
                cpu_limit_milli=cpu_limit_milli,
                disk_limit_mb=disk_limit_mb,
                network_enabled=network_enabled,
                owner_id=quota_owner_id,
                storage_project_id=storage_project_id,
                storage_quota_mb=storage_quota_mb,
                storage_project_room_ids=storage_project_room_ids,
                container_layer_mb=container_layer_mb,
                db_factory=factory,
            )
            result = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
            return _workspace_payload(result.scalar_one())

        factory = async_sessionmaker(engine, expire_on_commit=False)
        task = asyncio.create_task(
            _run_start_operation(
                room_id,
                operation_id,
                mem_limit_mb=mem_limit_mb,
                cpu_limit_milli=cpu_limit_milli,
                disk_limit_mb=disk_limit_mb,
                network_enabled=network_enabled,
                owner_id=quota_owner_id,
                storage_project_id=storage_project_id,
                storage_quota_mb=storage_quota_mb,
                storage_project_room_ids=storage_project_room_ids,
                container_layer_mb=container_layer_mb,
                db_factory=factory,
            )
        )
        _track_start_task(task)
        return starting_payload


# ---------------------------------------------------------------------------
# Stage 2: quota-aware switching, draining and durable task pause/resume.
# ---------------------------------------------------------------------------

_SWITCH_LOCKS: dict[tuple[UUID, UUID], asyncio.Lock] = {}


def _switch_lock(from_room_id: UUID, to_room_id: UUID) -> asyncio.Lock:
    key = tuple(sorted((from_room_id, to_room_id), key=str))
    lock = _SWITCH_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _SWITCH_LOCKS[key] = lock
    return lock


def _switch_payload(record: Any) -> dict[str, Any]:
    return {
        "request_id": record.request_id,
        "from_room_id": record.from_room_id,
        "to_room_id": record.to_room_id,
        "status": record.status,
        "error_code": record.error_code,
        "message": record.message,
        "force_stop_confirmed": record.force_stop_confirmed,
    }


async def _persist_switch(
    db: AsyncSession,
    *,
    request_id: UUID,
    from_room_id: UUID,
    to_room_id: UUID,
    quota_owner_id: UUID,
) -> Any:
    from app.models.workspace import WorkspaceSwitchRequest

    result = await db.execute(
        select(WorkspaceSwitchRequest).where(
            WorkspaceSwitchRequest.request_id == request_id
        )
    )
    record = result.scalar_one_or_none()
    if record is None:
        record = WorkspaceSwitchRequest(
            request_id=request_id,
            from_room_id=from_room_id,
            to_room_id=to_room_id,
            quota_owner_id=quota_owner_id,
        )
        db.add(record)
        await db.flush()
    elif (
        record.from_room_id != from_room_id
        or record.to_room_id != to_room_id
        or record.quota_owner_id != quota_owner_id
    ):
        raise WorkspaceStartError(
            "request_id_conflict",
            "request_id is already used for another switch.",
            status_code=409,
        )
    return record


async def pause_workspace_tasks(
    room_id: UUID,
    db: AsyncSession,
    *,
    pause_reason: str = "workspace_switch",
    force_stop: bool = False,
) -> list[UUID]:
    """Checkpoint active tasks before a workspace is stopped.

    The transition is intentionally explicit (``pausing`` then ``paused``)
    so status subscribers never observe a task disappearing straight from
    ``in_progress``.
    """
    from app.models.task import DispatchStatus, Task, TaskState
    from app.models.workspace import Workspace

    result = await db.execute(
        select(Task).where(
            Task.room_id == room_id,
            Task.state.in_(
                [
                    TaskState.IN_PROGRESS,
                    TaskState.PAUSING,
                    TaskState.RESUMING,
                    # A watchdog can mark the lifecycle blocked before an
                    # inner tool-loop has observed the stop.  It still needs
                    # a user-initiated runtime stop request.
                    TaskState.BLOCKED,
                ]
            ),
        )
    )
    tasks = result.scalars().all()
    now = _now()
    ids: list[UUID] = []
    from app.core.task_runtime import (
        RUNTIME_PAUSED,
        RUNTIME_RUNNING,
        RUNTIME_STOP_REQUESTED,
        request_task_stop,
    )
    from app.models.session import CmAgentProxySession

    open_session_task_ids = {
        item
        for item in (
            await db.execute(
                select(CmAgentProxySession.task_id).where(
                    CmAgentProxySession.room_id == room_id,
                    CmAgentProxySession.closed_at.is_(None),
                    CmAgentProxySession.task_id.is_not(None),
                )
            )
        ).scalars().all()
        if item is not None
    }
    requested_ids: list[UUID] = []
    for task in tasks:
        live_run = (
            task.runtime_state in {RUNTIME_RUNNING, RUNTIME_STOP_REQUESTED}
            or task.id in open_session_task_ids
        )
        await request_task_stop(task.id, pause_reason, db)
        task.pause_reason = pause_reason
        task.paused_at = now
        task.updated_at = now.replace(tzinfo=None)
        # Preserve the diagnostic BLOCKED lifecycle state; it is distinct from
        # whether a stale runtime is being stopped.  Runnable tasks move into
        # PAUSING and become PAUSED only when their run acknowledges the fence.
        if task.state != TaskState.BLOCKED:
            task.state = TaskState.PAUSING if live_run else TaskState.PAUSED
        if not live_run:
            task.runtime_state = RUNTIME_PAUSED
            # No loop will run finalize_run for this task, so reset the stale
            # DISPATCHED marker here; a live run's finalize_run does it when the
            # fence is acknowledged.
            if task.dispatch_status == DispatchStatus.DISPATCHED:
                task.dispatch_status = DispatchStatus.READY
        db.add(task)
        ids.append(task.id)
        if live_run:
            requested_ids.append(task.id)
    if tasks:
        await db.flush()
    from app.core.workspace_checkpoint import checkpoint_workspace

    checkpoint = await checkpoint_workspace(
        room_id,
        db,
        pause_reason=pause_reason,
        reason="workspace_stop" if pause_reason == "user" else pause_reason,
        force_stop=force_stop,
    )
    if tasks:
        await db.commit()
    workspace_result = await db.execute(
        select(Workspace).where(Workspace.room_id == room_id)
    )
    workspace = workspace_result.scalar_one_or_none()
    if workspace is not None:
        await _broadcast_workspace_status(
            workspace,
            previous_status=workspace.status,
            operation_id=workspace.operation_id,
            reason=pause_reason,
            event_type="workspace.session.paused",
        )
        if checkpoint is not None:
            await _broadcast_workspace_status(
                workspace,
                previous_status=workspace.status,
                operation_id=workspace.operation_id,
                reason=getattr(checkpoint, "reason", None),
                event_type="workspace.snapshot.created",
            )
    if requested_ids:
        logger.info(
            "workspace_lifecycle: pause requested for %d live task run(s) in room %s",
            len(requested_ids), room_id,
        )
    return ids


async def resume_workspace_switch_tasks(
    room_id: UUID,
    db: AsyncSession,
) -> list[UUID]:
    """Resume only tasks paused by an automatic workspace switch."""
    return await resume_paused_tasks(
        room_id, db, pause_reason="workspace_switch", source="workspace_resume",
    )


async def resume_paused_tasks(
    room_id: UUID,
    db: AsyncSession,
    *,
    pause_reason: str,
    source: str,
) -> list[UUID]:
    """Resume the room's tasks paused with ``pause_reason``.

    A task whose dispatch preflight refuses (hours, disk, agent gone…) stays
    paused and visible.
    """
    from app.models.task import DispatchStatus, Task, TaskState

    result = await db.execute(
        select(Task).where(
            Task.room_id == room_id,
            Task.state == TaskState.PAUSED,
            Task.pause_reason == pause_reason,
        )
    )
    tasks = result.scalars().all()
    if not tasks:
        return []
    from app.core.proxy_session import (
        TaskDispatchPreflightError,
        prepare_task_dispatch,
        request_task_dispatch,
    )

    # Do not turn a paused task into In Progress until the proxy can actually
    # be recreated. Failed preflight leaves the task paused and visible.
    dispatches = []
    for task in tasks:
        previous_dispatch_status = task.dispatch_status
        task.dispatch_status = DispatchStatus.READY
        try:
            await prepare_task_dispatch(task, db)
        except TaskDispatchPreflightError as exc:
            task.dispatch_status = previous_dispatch_status
            logger.info(
                "workspace_lifecycle: task %s remains paused: %s", task.id, exc.code
            )
        else:
            dispatches.append(task)

    if not dispatches:
        return []

    now = _now()
    for task in dispatches:
        task.state = TaskState.RESUMING
        task.updated_at = now.replace(tzinfo=None)
        db.add(task)
    await db.flush()
    for task in dispatches:
        task.state = TaskState.IN_PROGRESS
        task.pause_reason = None
        task.paused_at = None
        task.dispatch_status = DispatchStatus.READY
        task.updated_at = now.replace(tzinfo=None)
        db.add(task)
    await db.commit()
    # A status transition alone does not resume provider work. Start a fresh
    # proxy pair seeded from the workspace_paused checkpoint history.
    for task in dispatches:
        outcome = await request_task_dispatch(
            task, room_id, db, resume=True, source=source,
        )
        logger.info(
            "workspace_lifecycle: resume dispatch for task %s outcome=%s",
            task.id, outcome,
        )
    return [task.id for task in dispatches]


async def room_has_live_work(room_id: UUID, db: AsyncSession) -> bool:
    """True while the room is doing work, regardless of its container.

    This is the billing unit's definition of "the room is working", and it is
    deliberately narrower than :func:`workspace_is_idle`: no container status,
    no background processes, no service access — only work that a user would
    call work, and that the Hub keeps doing after the browser is closed.

    Why it exists. Billing used to be tied to the container process: an
    interval opened when the container started and closed when it stopped.
    That conflated two different things — *whether the room is working*, which
    the user pays for, and *whether we happen to be running its container*,
    which is our own optimisation. They coincide today only because the idle
    reaper stops the container exactly when work ends. Two things break that
    coincidence: an agent doing file/git work needs no container at all (the
    tools go straight to the volume — measured at 0.3 % of session time on the
    live stack), and hibernation (internal design notes §3.3, lever 2) would
    stop containers *while work continues* by design. Under the product rule
    "hibernation is our tool, the user never sees it, all working time is
    billed", the interval must survive the container.

    What counts (owner's rule, 2026-09-25): all the time the client uses the
    room. That is three things, any one of which is enough:

    * the room is open in a member's browser (``workspace.presence_at``, see
      core/billing_interval.py) — including the minutes a person spends
      thinking over an agent's answer;
    * a task is running;
    * a session is actually working — it stamped activity within the idle
      window. An open session row alone is not work: on the live stack 42
      ``cm_self_plan`` rows were left open for up to two months with no
      activity ever recorded, and each would have billed its room until the
      24 h guard. Waiting for a reply is billed through presence, not through
      a session row nobody is looking at. ``cm_self_plan`` is excluded
      outright (NON_BILLING_SESSION_KINDS).
    """
    from datetime import timedelta

    from sqlalchemy import func

    from app.core.billing_interval import NON_BILLING_SESSION_KINDS, PRESENCE_FRESH_SEC
    from app.models.session.base import Session
    from app.models.task import Task, TaskState
    from app.models.workspace import Workspace

    now = _now()
    active_task = await db.execute(
        select(Task.id).where(
            Task.room_id == room_id,
            Task.state.in_(
                [TaskState.IN_PROGRESS, TaskState.PAUSING, TaskState.RESUMING]
            ),
        ).limit(1)
    )
    if active_task.first() is not None:
        return True
    working_session = await db.execute(
        select(Session.id).where(
            Session.room_id == room_id,
            Session.closed_at.is_(None),
            Session.kind.not_in(NON_BILLING_SESSION_KINDS),
            func.coalesce(Session.last_activity_at, Session.started_at)
            >= now - timedelta(seconds=WORKSPACE_IDLE_TIMEOUT_SEC),
        ).limit(1)
    )
    if working_session.first() is not None:
        return True
    present = await db.execute(
        select(Workspace.id).where(
            Workspace.room_id == room_id,
            Workspace.presence_at >= now - timedelta(seconds=PRESENCE_FRESH_SEC),
        ).limit(1)
    )
    return present.first() is not None


async def _close_workspace_usage(
    ws: Any,
    db: AsyncSession,
    *,
    stop_reason: str,
) -> None:
    """Close the open billing interval(s) of one workspace and count the hours.

    The counter matters as much as the row. Quota enforcement reads the
    ``workspace_usage`` table directly (``monthly_hours_consumed``), but every
    dashboard, alert and capacity figure reads ``workspace_session_hours_total``
    — and that counter used to be fed from exactly two places: the reconciler,
    which only fires when a container disappears behind the backend's back, and
    ``api/v1/workspace._close_usage``, which nothing has called since the
    lifecycle moved into this module. The ordinary stop path — the one that
    ends almost every room session — closed the row and told the counter
    nothing. The result was a counter stuck near zero while the table held
    hundreds of hours: measured 2026-09-23 on the live stack, 808.7 h across
    146 closed intervals against a counter reading ~0.

    Double counting is not a risk here: the reconciler closes its rows itself
    and never calls this helper, and the two paths are mutually exclusive by
    construction (the reconciler acts only when the container is already gone).
    Both go through billing_interval.close_usage, so the row, the counter and
    container_seconds agree on where the interval ended.
    """
    from app.models.workspace import WorkspaceUsage

    # The container is going away; the billing interval only follows it if the
    # room has actually finished working (see room_has_live_work). Otherwise
    # the row stays open and keeps accruing, and the reconciler closes it on
    # the tick after the last task/session ends. Either way the container's
    # own stretch ends here (container_seconds, cost model only).
    from app.core.billing_interval import fold_container_time

    still_working = (
        getattr(ws, "room_id", None) is not None
        and await room_has_live_work(ws.room_id, db)
    )
    result = await db.execute(
        select(WorkspaceUsage).where(
            WorkspaceUsage.workspace_id == ws.id,
            WorkspaceUsage.stopped_at.is_(None),
        )
    )
    now = _now()
    rows = result.scalars().all()
    if still_working:
        for usage in rows:
            fold_container_time(usage, now)
            db.add(usage)
        logger.info(
            "workspace_lifecycle: keeping billing interval open for room %s "
            "(container stopping, reason=%s, but the room is still working)",
            ws.room_id, stop_reason,
        )
        return

    # Closed at the last evidence of use rather than `now`: an idle stop or a
    # recovery that runs late (backend down, auto-stop failing) must not bill
    # the time nobody used the room. With recent use this is simply `now`.
    from app.core.billing_interval import close_usage

    for usage in rows:
        await close_usage(usage, ws, db, stop_reason=stop_reason, now=now)


async def stop_workspace(
    room_id: UUID,
    db: AsyncSession,
    *,
    reason: str = "user",
    force: bool = False,
    request_id: Optional[UUID] = None,
) -> dict[str, Any]:
    """Idempotently drain and stop one runtime, preserving its volume."""
    async with _room_lock(room_id):
        return await _stop_workspace_locked(
            room_id, db, reason=reason, force=force, request_id=request_id
        )


async def _stop_workspace_locked(
    room_id: UUID,
    db: AsyncSession,
    *,
    reason: str,
    force: bool,
    request_id: Optional[UUID] = None,
) -> dict[str, Any]:
    """Body of :func:`stop_workspace`; the caller holds ``_room_lock``."""
    from app.core import workspaced_client as wsd
    from app.models.workspace import Workspace

    result = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ws = result.scalar_one_or_none()
    if ws is None:
        return {"room_id": room_id, "status": "stopped"}
    if ws.status == "stopped":
        return _workspace_payload(ws)
    # A whole stop runs under ``_room_lock``, so "stopping" seen here is a stop
    # that was abandoned midway (its request died). Finish it rather than
    # report it as in progress forever.

    previous = ws.status
    if ws.status == "running":
        ws.status = "draining"
        ws.status_reason = "switch" if reason == "workspace_switch" else reason
        ws.draining_since = _now()
        db.add(ws)
        await db.commit()
        await _broadcast_workspace_status(
            ws,
            previous_status=previous,
            operation_id=ws.operation_id,
            reason="drain_started",
            request_id=request_id,
        )

    if ws.active_session_count and not force:
        # The caller can wait/retry or explicitly confirm a force stop.
        return {
            **_workspace_payload(ws),
            "status": "drain_pending",
            "error_code": "drain_timeout",
            "message": "The current room is still performing an operation.",
        }

    if force:
        from app.models.workspace import WorkspaceExec

        active_exec_result = await db.execute(
            select(WorkspaceExec).where(
                WorkspaceExec.workspace_id == ws.id,
                WorkspaceExec.status == "running",
            )
        )
        active_execs = active_exec_result.scalars().all()
        for execution in active_execs:
            execution.status = "interrupted"
            execution.exit_code = -1
            db.add(execution)
        if active_execs:
            await db.flush()

    ws.status = "stopping"
    ws.status_reason = reason
    db.add(ws)
    await db.commit()
    await _broadcast_workspace_status(
        ws,
        previous_status=previous,
        operation_id=ws.operation_id,
        reason="stopping",
        request_id=request_id,
    )

    try:
        await wsd.stop_workspace(str(room_id))
    except httpx.HTTPStatusError as exc:
        ws.status = "error"
        ws.status_reason = "stop_failed"
        ws.last_error = f"Could not stop the workspace (HTTP {exc.response.status_code})."
        await db.commit()
        await _broadcast_workspace_status(
            ws,
            previous_status="stopping",
            operation_id=ws.operation_id,
            reason="stop_failed",
            error={"error_code": "stop_failed", "message": ws.last_error},
            request_id=request_id,
        )
        raise WorkspaceStartError("stop_failed", ws.last_error, status_code=502) from exc
    except httpx.RequestError as exc:
        ws.status = "error"
        ws.status_reason = "workspaced_unreachable"
        ws.last_error = "The workspace is unavailable. Try again."
        await db.commit()
        raise WorkspaceStartError(
            "workspaced_unreachable", ws.last_error, status_code=503
        ) from exc

    now = _now()
    from app.models.workspace import WorkspaceBgProc, WorkspaceManagedProcess
    from app.core.workspace_services import WorkspaceServiceRegistry

    bg_rows = (
        await db.execute(
            select(WorkspaceBgProc).where(
                WorkspaceBgProc.workspace_id == ws.id,
                WorkspaceBgProc.status == "running",
            )
        )
    ).scalars().all()
    for process in bg_rows:
        process.status = "failed"
        db.add(process)
    managed_rows = (
        await db.execute(
            select(WorkspaceManagedProcess).where(
                WorkspaceManagedProcess.workspace_id == ws.id,
                WorkspaceManagedProcess.status.in_(["starting", "running"]),
            )
        )
    ).scalars().all()
    stopped_services = []
    registry = WorkspaceServiceRegistry(db)
    for process in managed_rows:
        is_service = bool(process.argv)
        process.status = "paused" if (
            process.restart_policy == "on_workspace_restore" if is_service
            else process.resume_policy == "restart"
        ) else "stopped"
        if is_service:
            process.runtime_state = "stopped"
            registry.record_event(process, "workspace.service.stopped", f"lifecycle:{reason}", {})
            stopped_services.append(process)
        process.pid = None
        process.pgid = None
        process.updated_at = now
        db.add(process)
    ws.status = "stopped"
    ws.status_reason = reason
    ws.last_error = None
    ws.stopped_at = now
    ws.last_heartbeat_at = now
    ws.draining_since = None
    ws.operation_id = None
    ws.container_id = None
    ws.active_session_count = 0
    db.add(ws)
    await _close_workspace_usage(ws, db, stop_reason=reason)
    await db.commit()
    for process in stopped_services:
        await registry.publish(process, "workspace.service.stopped")
    await _broadcast_workspace_status(
        ws,
        previous_status="stopping",
        operation_id=None,
        reason="stopped",
        request_id=request_id,
    )
    # internal design notes T3: a room's files are measured when it stops.
    from app.core.room_storage import schedule_room_measure

    schedule_room_measure(room_id)
    return _workspace_payload(ws)


async def heartbeat_workspace(
    room_id: UUID,
    db: AsyncSession,
    *,
    generation: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Persist a lease heartbeat without changing runtime state."""
    from app.models.workspace import Workspace

    result = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ws = result.scalar_one_or_none()
    if ws is None:
        raise WorkspaceStartError("workspace_not_found", "Workspace not found.", status_code=404)
    if ws.status not in {"starting", "running", "draining", "stopping"}:
        return _workspace_payload(ws)
    now = _now()
    ws.last_heartbeat_at = now
    if generation is not None:
        ws.workspace_generation = generation
    db.add(ws)
    await db.commit()
    return _workspace_payload(ws)


async def find_stale_workspaces(
    db: AsyncSession,
    *,
    recovery_timeout_sec: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Return stale leases for the recovery worker.

    Recovery deliberately does not force-stop here; only the recovery worker
    that has observed the stale lease for the configured timeout may do so.
    """
    from app.core.workspace_quota import quota_manager

    rows = await quota_manager.stale_workspaces(
        db,
        older_than_sec=recovery_timeout_sec or CONFIG.recovery_timeout_sec,
    )
    return [_workspace_payload(ws) for ws in rows]


async def recover_stale_workspaces(
    db: AsyncSession,
    *,
    recovery_timeout_sec: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Force-stop leases that have exceeded the recovery timeout.

    This is the only automatic force-stop path; user-initiated switching must
    pass ``force_stop=True`` explicitly.
    """
    stale = await find_stale_workspaces(
        db, recovery_timeout_sec=recovery_timeout_sec
    )
    recovered: list[dict[str, Any]] = []
    for item in stale:
        try:
            payload = await stop_workspace(
                UUID(str(item["room_id"])),
                db,
                reason="recovery",
                force=True,
            )
            recovered.append(payload)
        except WorkspaceStartError:
            logger.exception(
                "workspace recovery failed for room %s", item.get("room_id")
            )
    return recovered


async def workspace_is_idle(
    room_id: UUID,
    db: AsyncSession,
    *,
    idle_timeout_sec: Optional[int] = None,
    now: Optional[datetime] = None,
) -> bool:
    """Classify idleness without considering browser/WebSocket presence.

    A live proxy session is activity even when it is waiting between model
    turns.  The previous implementation only looked at ``Task.state`` and
    background processes, so an in-flight ChatMaster conversation could be
    reaped as idle during a long provider call.
    """
    from app.models.task import Task, TaskState
    from app.models.session.base import Session
    from app.models.workspace import (
        Workspace,
        WorkspaceBgProc,
        WorkspaceManagedProcess,
    )

    result = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ws = result.scalar_one_or_none()
    if ws is None or ws.status != "running" or ws.active_session_count:
        return False
    active_tasks = await db.execute(
        select(Task.id).where(
            Task.room_id == room_id,
            Task.state.in_(
                [TaskState.IN_PROGRESS, TaskState.PAUSING, TaskState.RESUMING]
            ),
        )
    )
    if active_tasks.first() is not None:
        # Keep the idle window anchored to the latest observed active work.
        # This makes the 15-minute timer start after the task really stops,
        # rather than from the last file operation before the task began.
        activity_at = now or _now()
        if hasattr(db, "add") and hasattr(db, "flush"):
            ws.last_activity_at = activity_at
            db.add(ws)
            await db.flush()
        return False
    active_sessions = await db.execute(
        select(Session.id).where(
            Session.room_id == room_id,
            Session.closed_at.is_(None),
        )
    )
    if active_sessions.first() is not None:
        activity_at = now or _now()
        if hasattr(db, "add") and hasattr(db, "flush"):
            ws.last_activity_at = activity_at
            db.add(ws)
            await db.flush()
        return False
    active_processes = await db.execute(
        select(WorkspaceBgProc.id).where(
            WorkspaceBgProc.workspace_id == ws.id,
            WorkspaceBgProc.status == "running",
        )
    )
    if active_processes.first() is not None:
        return False
    # A running service is deliberately *not* activity by itself: otherwise a
    # forgotten Vite process would keep a workspace alive forever.  A Browser
    # relay, agent request/log read, or service start stamps last_accessed_at,
    # which joins normal workspace activity in resetting the shared timeout.
    service_access = await db.execute(
        select(WorkspaceManagedProcess.last_accessed_at)
        .where(
            WorkspaceManagedProcess.workspace_id == ws.id,
            WorkspaceManagedProcess.last_accessed_at.is_not(None),
        )
        .order_by(WorkspaceManagedProcess.last_accessed_at.desc())
        .limit(1)
    )
    last_service_access = service_access.scalar_one_or_none()
    last_activity = ws.last_activity_at or ws.started_at
    if last_service_access is not None:
        if last_service_access.tzinfo is None:
            last_service_access = last_service_access.replace(tzinfo=timezone.utc)
        if last_activity is None or last_service_access > last_activity:
            last_activity = last_service_access
    if last_activity is None:
        return False
    if last_activity.tzinfo is None:
        last_activity = last_activity.replace(tzinfo=timezone.utc)
    now = now or _now()
    return (now - last_activity).total_seconds() >= (
        idle_timeout_sec or CONFIG.idle_timeout_sec
    )


async def auto_stop_idle_workspaces(
    db: AsyncSession,
    *,
    idle_timeout_sec: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Stop idle runtimes; callers can run this from a periodic worker."""
    from app.models.workspace import Workspace

    rows = await db.execute(select(Workspace).where(Workspace.status == "running"))
    stopped: list[dict[str, Any]] = []
    for ws in rows.scalars().all():
        if not await workspace_is_idle(
            ws.room_id, db, idle_timeout_sec=idle_timeout_sec
        ):
            continue
        await _broadcast_workspace_status(
            ws,
            previous_status="running",
            operation_id=ws.operation_id,
            reason="idle_warning",
        )
        stopped.append(
            await stop_workspace(ws.room_id, db, reason="idle")
        )
    return stopped


async def request_switch(
    from_room_id: UUID,
    to_room_id: UUID,
    request_id: UUID,
    db: AsyncSession,
    *,
    quota_owner_id: Optional[UUID] = None,
    confirm_stop: bool = False,
    force_stop: bool = False,
) -> dict[str, Any]:
    """Idempotently switch rooms, stopping the current one only on consent."""
    from app.models.room import Room
    from app.models.workspace import WorkspaceSwitchRequest
    from app.core.workspace_quota import (
        WorkspaceQuotaExceeded,
        quota_manager,
    )

    if not isinstance(request_id, UUID):
        request_id = UUID(str(request_id))
    if not isinstance(from_room_id, UUID):
        from_room_id = UUID(str(from_room_id))
    if not isinstance(to_room_id, UUID):
        to_room_id = UUID(str(to_room_id))

    if from_room_id == to_room_id:
        return {
            "request_id": request_id,
            "from_room_id": from_room_id,
            "to_room_id": to_room_id,
            "status": "noop",
        }

    switch_lock = _switch_lock(from_room_id, to_room_id)
    if switch_lock.locked():
        # Queueing behind a running switch made every extra click wait out the
        # first one (and all of them hang together if it hangs). Answer now;
        # the client already shows the switch in progress.
        return {
            "request_id": request_id,
            "from_room_id": from_room_id,
            "to_room_id": to_room_id,
            "status": "busy",
            "message": "A switch between these rooms is already in progress.",
        }

    async with switch_lock:
        from_room = await db.get(Room, from_room_id)
        to_room = await db.get(Room, to_room_id)
        if from_room is None or to_room is None:
            raise WorkspaceStartError("room_not_found", "Room not found.", status_code=404)
        owner = quota_owner_id or from_room.creator_id
        record = await _persist_switch(
            db,
            request_id=request_id,
            from_room_id=from_room_id,
            to_room_id=to_room_id,
            quota_owner_id=owner,
        )
        # A terminal result is returned verbatim for retries.
        if record.status in {"started", "completed", "cancelled"}:
            return _switch_payload(record)
        if force_stop:
            record.force_stop_confirmed = True

        from app.models.workspace import Workspace
        target_res = await db.execute(
            select(Workspace).where(Workspace.room_id == to_room_id)
        )
        target = target_res.scalar_one_or_none()
        target_running = target is not None and target.status in {"running", "starting"}

        if not target_running:
            try:
                await quota_manager.reserve(owner, to_room_id, db)
            except WorkspaceQuotaExceeded:
                if not confirm_stop:
                    record.status = "confirmation_required"
                    record.error_code = "quota_exceeded"
                    record.message = (
                        "The limit of simultaneously running workspaces has been reached. "
                        "Confirm stopping the current room."
                    )
                    record.updated_at = _now()
                    db.add(record)
                    await db.commit()
                    return _switch_payload(record)

                current_res = await db.execute(
                    select(Workspace).where(Workspace.room_id == from_room_id)
                )
                current = current_res.scalar_one_or_none()
                if current is not None and current.status in {
                    "running",
                    "starting",
                    "draining",
                    "stopping",
                }:
                    paused_ids = await pause_workspace_tasks(
                        from_room_id,
                        db,
                        pause_reason="force_stop" if force_stop else "workspace_switch",
                        force_stop=force_stop,
                    )
                    stopped = await stop_workspace(
                        from_room_id,
                        db,
                        reason="workspace_switch",
                        force=force_stop,
                        request_id=request_id,
                    )
                    if stopped.get("status") == "drain_pending":
                        record.status = "pending"
                        record.error_code = "drain_timeout"
                        record.message = stopped["message"]
                        record.updated_at = _now()
                        db.add(record)
                        await db.commit()
                        return _switch_payload(record) | {"paused_task_ids": paused_ids}

        try:
            target_payload = await ensure_workspace_started(
                to_room_id,
                db,
                quota_owner_id=owner,
                background=True,
            )
        except WorkspaceStartError as exc:
            record.status = "error"
            record.error_code = exc.error_code
            record.message = exc.message
            record.updated_at = _now()
            db.add(record)
            await db.commit()
            raise

        record.status = (
            "completed" if target_payload.get("status") == "running" else "started"
        )
        record.error_code = None
        record.message = None
        record.updated_at = _now()
        db.add(record)
        await db.commit()
        return _switch_payload(record) | {"workspace": target_payload}
