from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import UUID

import base64

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from jose import jwt, JWTError

from app.api import deps
from app.core import legion_auth, security, workspaced_client as wsd
from app.core.events import emit_event
from app.core.storage import DatabaseBackend, get_storage_backend
from app.core.path_contract import normalize_workspace_path
from app.core.workspace_recovery import apply_restore, compute_restore_preview
from app.core.workspace_lifecycle import (
    WorkspaceStartError,
    _get_or_create_workspace as lifecycle_get_or_create_workspace,
    broadcast_workspace_event,
    ensure_workspace_started,
    pause_workspace_tasks,
    stop_workspace as stop_workspace_lifecycle,
    heartbeat_workspace,
)
from app.core.workspace_services import (
    ServiceValidationError,
    WorkspaceServiceRegistry,
    normalize_declaration,
    reconcile_service_runtime as _reconcile_service_runtime,
    services_enabled,
)
from app.core.preview_gateway import issue_ticket, preview_enabled, service_origin, ticket_ttl_sec
from app.core.room_git_flow import get_flow_config
from app.db.session import get_session
from app.models.room import Room, RoomUserLink
from app.models.user import User
from app.models.workspace import (
    Workspace,
    WorkspaceBgProc,
    WorkspaceExec,
    WorkspaceManagedProcess,
    WorkspaceServiceEvent,
    WorkspaceUsage,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── helpers ──────────────────────────────────────────────────────────────────

async def _verify_membership(room_id: UUID, user: User, db: AsyncSession) -> None:
    res = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Not a room member")


async def _get_room(room_id: UUID, db: AsyncSession) -> Room:
    room = await db.get(Room, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    if room.deletion_started_at is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "error_code": "room_deleting",
                "message": "Room deletion is in progress; workspace access is disabled.",
            },
        )
    return room


async def _get_or_create_workspace(room_id: UUID, db: AsyncSession) -> Workspace:
    # Race-safe get-or-create lives in workspace_lifecycle (SAVEPOINT insert,
    # re-read the winner's row on IntegrityError). Opening a new room fires
    # several workspace endpoints at once, and the old select-then-insert here
    # lost the race with "duplicate key ... ix_workspace_room_id".
    ws = await lifecycle_get_or_create_workspace(room_id, db)
    await db.commit()
    return ws


# Billing intervals are opened and closed in app/core/workspace_lifecycle.py
# (_open in ensure_workspace_started, _close in _close_workspace_usage). Here
# there once lived its own _open_usage/_close_usage pair; after the lifecycle
# moved it stayed unreachable but looked functional — and because of that
# workspace_session_hours_total did not move for years: the only call of
# record_workspace_session_hours lived in exactly that dead code. Removed
# on 2026-09-23 together with the counter fix.


# ── schemas ───────────────────────────────────────────────────────────────────

class WorkspaceStatusOut(BaseModel):
    workspace_id: UUID
    room_id: UUID
    status: str
    status_reason: Optional[str] = None
    last_error: Optional[str] = None
    operation_id: Optional[UUID] = None
    quota_owner_id: Optional[UUID] = None
    container_id: Optional[str] = None
    image: str
    mem_limit_mb: int
    cpu_limit_milli: int
    disk_limit_mb: int
    network_enabled: bool
    last_activity_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_heartbeat_at: Optional[datetime] = None
    workspace_generation: Optional[dict[str, Any]] = None
    active_session_count: int = 0
    draining_since: Optional[datetime] = None
    uptime_sec: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    # Stage 6 observability fields.  They are derived from durable workspace
    # metadata and the process tables, so the REST snapshot remains useful
    # after a WebSocket reconnect.
    starting_duration_sec: Optional[float] = None
    metadata_synced_at: Optional[datetime] = None
    background_processes: list[dict[str, Any]] = Field(default_factory=list)
    managed_processes: list[dict[str, Any]] = Field(default_factory=list)
    # Lets the UI tailor sync-files wording/behavior. Other constructions of
    # this model (start/heartbeat) build it from a plain dict and don't pass
    # this explicitly, so the default must reflect the live env.
    fs_backend: str = Field(default_factory=lambda: "workspace")


class WorkspaceStartRequest(BaseModel):
    # None — the room's own memory limit (core/room_memory.py).
    mem_limit_mb: Optional[int] = Field(default=None, ge=128, le=8192)
    cpu_limit_milli: int = Field(default=500, ge=100, le=4000)
    # Compatibility input for self-hosted callers.  Cloud overrides this with
    # the plan's ``container_layer_mb`` before starting Docker and persisting
    # the status, so a client cannot advertise a different disk limit.
    disk_limit_mb: int = Field(default=2048, ge=256, le=16384)
    network_enabled: bool = True


class WorkspaceHeartbeatRequest(BaseModel):
    generation: Optional[dict[str, Any]] = None


class ExecRequest(BaseModel):
    cmd: list[str]
    cwd: str = "/workspace"
    timeout_sec: int = Field(default=30, ge=1, le=7200)
    env: Optional[dict[str, str]] = None


class ExecOut(BaseModel):
    exec_id: UUID
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    truncated: bool


# ── GET /rooms/{room_id}/workspace ────────────────────────────────────────────

@router.get("/rooms/{room_id}/workspace", response_model=WorkspaceStatusOut)
async def get_workspace(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    ws = await _get_or_create_workspace(room_id, db)

    live: dict[str, Any] = {}
    if ws.status == "running":
        try:
            live = await wsd.get_workspace_status(str(room_id))
        except Exception:
            pass

    now = datetime.now(timezone.utc)
    starting_duration = None
    if ws.status == "starting":
        # ``started_at`` is set once workspaced reports readiness.  While a
        # runtime is starting, ``created_at`` is the durable lower-bound
        # fallback and avoids requiring a schema migration for a transient
        # ``starting_at`` column.
        anchor = ws.started_at or ws.created_at
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=timezone.utc)
        starting_duration = max(0.0, (now - anchor).total_seconds())

    bg_rows = (
        await db.execute(
            select(WorkspaceBgProc).where(
                WorkspaceBgProc.workspace_id == ws.id,
                WorkspaceBgProc.status == "running",
            ).order_by(WorkspaceBgProc.created_at.asc())
        )
    ).scalars().all()
    managed_rows = (
        await db.execute(
            select(WorkspaceManagedProcess).where(
                WorkspaceManagedProcess.workspace_id == ws.id,
                WorkspaceManagedProcess.status.in_(["starting", "running", "paused"]),
            ).order_by(WorkspaceManagedProcess.name.asc())
        )
    ).scalars().all()

    return WorkspaceStatusOut(
        workspace_id=ws.id,
        room_id=ws.room_id,
        status=live.get("status", ws.status),
        status_reason=ws.status_reason,
        last_error=ws.last_error,
        operation_id=ws.operation_id,
        quota_owner_id=ws.quota_owner_id,
        container_id=live.get("container_id", ws.container_id),
        image=ws.image,
        mem_limit_mb=ws.mem_limit_mb,
        cpu_limit_milli=ws.cpu_limit_milli,
        disk_limit_mb=ws.disk_limit_mb,
        network_enabled=ws.network_enabled,
        last_activity_at=ws.last_activity_at,
        started_at=ws.started_at,
        stopped_at=ws.stopped_at,
        last_heartbeat_at=ws.last_heartbeat_at,
        workspace_generation=live.get("workspace_generation", ws.workspace_generation),
        active_session_count=ws.active_session_count,
        draining_since=ws.draining_since,
        uptime_sec=live.get("uptime_sec"),
        memory_usage_mb=live.get("memory_usage_mb"),
        starting_duration_sec=starting_duration,
        # Metadata sync is represented by the last durable heartbeat.  The
        # explicit name keeps the UI contract stable while older databases
        # remain compatible.
        metadata_synced_at=ws.last_heartbeat_at,
        background_processes=[
            {
                "id": str(proc.id),
                "bg_id": proc.bg_id,
                "cmd": proc.cmd,
                "status": proc.status,
                "pid": proc.pid,
                "log_path": proc.log_path,
            }
            for proc in bg_rows
        ],
        managed_processes=[
            {
                "id": str(proc.id),
                "name": proc.name,
                "command": proc.command,
                "cwd": proc.cwd,
                "ports": proc.ports or [],
                "resume_policy": proc.resume_policy,
                "status": proc.status,
                "pid": proc.pid,
                "last_error": proc.last_error,
            }
            for proc in managed_rows
        ],
    )


# ── POST /rooms/{room_id}/workspace/start ─────────────────────────────────────

@router.post("/rooms/{room_id}/workspace/start", response_model=WorkspaceStatusOut)
async def start_workspace(
    room_id: UUID,
    body: WorkspaceStartRequest = WorkspaceStartRequest(),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    # The creator owns the quota lease even when the room has multiple
    # members.  The lifecycle operation returns ``starting`` immediately and
    # publishes the eventual ``running``/``error`` status over room WS.
    try:
        result = await ensure_workspace_started(
            room_id,
            db,
            quota_owner_id=current_user.id,
            mem_limit_mb=body.mem_limit_mb,
            cpu_limit_milli=body.cpu_limit_milli,
            disk_limit_mb=body.disk_limit_mb,
            network_enabled=body.network_enabled,
        )
    except WorkspaceStartError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc
    return WorkspaceStatusOut(**result)


# ── POST /rooms/{room_id}/workspace/stop ──────────────────────────────────────

# ── Room memory limit (internal design notes §3a, T2a) ──────────────────────

class RoomMemoryRequest(BaseModel):
    # None — back to the plan's default.
    mem_limit_mb: Optional[int] = None


@router.get("/rooms/{room_id}/workspace/memory")
async def get_room_memory(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    from app.core.room_memory import room_memory_view

    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    view = await room_memory_view(room_id, db)
    await db.commit()
    return view


@router.put("/rooms/{room_id}/workspace/memory")
async def put_room_memory(
    room_id: UUID,
    body: RoomMemoryRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    from app.core.room_memory import RoomMemoryError, set_room_memory

    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        return await set_room_memory(room_id, body.mem_limit_mb, db)
    except RoomMemoryError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc


@router.post("/rooms/{room_id}/workspace/stop")
async def stop_workspace(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    try:
        existing_ws = (
            await db.execute(select(Workspace).where(Workspace.room_id == room_id))
        ).scalar_one_or_none()
        if existing_ws is not None and existing_ws.status != "stopped":
            await pause_workspace_tasks(room_id, db, pause_reason="user")
        payload = await stop_workspace_lifecycle(room_id, db, reason="user")
    except WorkspaceStartError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc

    ws = await _get_or_create_workspace(room_id, db)
    # Mark all running background processes as failed — container is gone.
    bg_res = await db.execute(
        select(WorkspaceBgProc).where(
            WorkspaceBgProc.workspace_id == ws.id,
            WorkspaceBgProc.status == "running",
        )
    )
    for proc in bg_res.scalars().all():
        proc.status = "failed"
        db.add(proc)
    await db.commit()

    await emit_event(
        room_id=room_id,
        event_type="WORKSPACE_STOPPED",
        content="Workspace stopped",
        db=db,
        metadata={"workspace_id": str(ws.id), "stop_reason": "user"},
    )

    return {"ok": True, **payload}


@router.post("/rooms/{room_id}/workspace/pause-agents")
async def pause_all_workspace_agents(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Checkpoint and request a cooperative stop for every active task runtime."""
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    existing_ws = (
        await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ).scalar_one_or_none()
    if existing_ws is None:
        raise HTTPException(status_code=409, detail="Workspace has not been created")
    paused_ids = await pause_workspace_tasks(room_id, db, pause_reason="user")
    await emit_event(
        room_id=room_id,
        event_type="WORKSPACE_AGENTS_PAUSED",
        content=(
            f"Agent pause requested for the room: tasks affected — {len(paused_ids)}. "
            "The actual stop of each live session will be confirmed separately."
        ),
        db=db,
        metadata={
            "pause_reason": "user",
            "task_ids": [str(task_id) for task_id in paused_ids],
        },
    )
    return {
        "ok": True,
        "pause_reason": "user",
        "paused_task_ids": paused_ids,
        "pause_requested_task_ids": paused_ids,
    }


@router.post("/rooms/{room_id}/workspace/heartbeat", response_model=WorkspaceStatusOut)
async def workspace_heartbeat(
    room_id: UUID,
    body: WorkspaceHeartbeatRequest = WorkspaceHeartbeatRequest(),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        payload = await heartbeat_workspace(
            room_id, db, generation=body.generation
        )
    except WorkspaceStartError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error_code": exc.error_code, "message": exc.message},
        ) from exc
    return WorkspaceStatusOut(**payload)


# ── POST /rooms/{room_id}/workspace/exec ──────────────────────────────────────

@router.post("/rooms/{room_id}/workspace/exec", response_model=ExecOut)
async def exec_in_workspace(
    room_id: UUID,
    body: ExecRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    ws = await _get_or_create_workspace(room_id, db)
    if ws.status != "running":
        raise HTTPException(status_code=409, detail="Workspace is not running")

    cmd_str = " ".join(body.cmd)

    await emit_event(
        room_id=room_id,
        event_type="EXEC_STARTED",
        content=f"Running: {cmd_str}",
        db=db,
        metadata={"workspace_id": str(ws.id), "cmd": cmd_str, "cwd": body.cwd},
    )

    try:
        result = await wsd.exec_in_workspace(
            room_id=str(room_id),
            cmd=body.cmd,
            cwd=body.cwd,
            timeout_sec=body.timeout_sec,
            env=body.env,
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")

    exec_record = WorkspaceExec(
        workspace_id=ws.id,
        initiator=f"user:{current_user.id}",
        command=cmd_str,
        exit_code=result.get("exit_code"),
        stdout_tail=result.get("stdout", "")[-8192:],
        stderr_tail=result.get("stderr", "")[-8192:],
        duration_ms=result.get("duration_ms"),
    )
    db.add(exec_record)

    ws.last_activity_at = datetime.now(timezone.utc)
    db.add(ws)
    await db.commit()
    await db.refresh(exec_record)

    await emit_event(
        room_id=room_id,
        event_type="EXEC_FINISHED",
        content=f"Finished (exit {result.get('exit_code', '?')}): {cmd_str}",
        db=db,
        metadata={
            "workspace_id": str(ws.id),
            "exec_id": str(exec_record.id),
            "cmd": cmd_str,
            "exit_code": result.get("exit_code"),
            "duration_ms": result.get("duration_ms"),
        },
    )

    return ExecOut(
        exec_id=exec_record.id,
        exit_code=result.get("exit_code", -1),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        duration_ms=result.get("duration_ms", 0),
        truncated=result.get("truncated", False),
    )


# ── POST /rooms/{room_id}/workspace/exec-stream ───────────────────────────────
#
# Streaming variant of exec: returns NDJSON chunks as the process runs so the
# frontend can display output in real time and allow the user to abort.

from starlette.responses import StreamingResponse as _StreamingResponse


@router.post("/rooms/{room_id}/workspace/exec-stream")
async def exec_stream_in_workspace(
    room_id: UUID,
    body: ExecRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    ws = await _get_or_create_workspace(room_id, db)
    if ws.status != "running":
        raise HTTPException(status_code=409, detail="Workspace is not running")

    async def _gen():
        try:
            ctx = wsd.exec_stream_client(
                str(room_id),
                cmd=body.cmd,
                cwd=body.cwd,
                timeout_sec=body.timeout_sec,
                env=body.env,
            )
            async with ctx as r:
                async for chunk in r.aiter_bytes():
                    yield chunk
        except Exception as exc:
            import json as _json
            yield (_json.dumps({"error": str(exc)}) + "\n").encode()

    return _StreamingResponse(_gen(), media_type="application/x-ndjson")


# ── GET /rooms/{room_id}/workspace/preview/{port}/{path} ─────────────────────
#
# Serves HTTP content from a server running inside the workspace container so
# the frontend can render it in an iframe (room "Browser" tab). Because the
# iframe (and its subresources) cannot send an Authorization header, the token
# is also accepted via ?token= query param or the "legion_token" cookie.

def _preview_error_html(message: str) -> str:
    return (
        "<!doctype html><html><body style=\"margin:0;display:flex;align-items:center;"
        "justify-content:center;height:100vh;font-family:system-ui,sans-serif;"
        "background:#0d1117;color:#8b949e\">"
        f"<div style=\"text-align:center;padding:24px\"><p style=\"font-size:14px\">{message}</p></div>"
        "</body></html>"
    )


async def _resolve_preview_user(request: Request, db: AsyncSession) -> Optional[User]:
    token = ""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:]
    if not token:
        token = request.query_params.get("token", "")
    if not token:
        token = request.cookies.get("legion_token", "")
    if not token:
        return None
    try:
        payload = security.decode_access_token(token)
        raw = payload.get("sub")
        user_id = UUID(raw) if raw else None
    except (JWTError, ValueError, TypeError):
        return None
    if not user_id:
        return None
    res = await db.execute(select(User).where(User.id == user_id, User.is_active.is_(True)))
    user = res.scalars().first()
    if user is not None and await legion_auth.session_revoked(db, user.legion_account_id, payload):
        return None
    return user


@router.get("/rooms/{room_id}/workspace/preview/{port}/{path:path}")
async def workspace_preview(
    room_id: UUID,
    port: int,
    request: Request,
    path: str = "",
    db: AsyncSession = Depends(get_session),
) -> Response:
    user = await _resolve_preview_user(request, db)
    if not user:
        return Response(
            content=_preview_error_html("Unauthorized — reload the Browser tab"),
            status_code=401, media_type="text/html",
        )

    await _get_room(room_id, db)
    await _verify_membership(room_id, user, db)

    if not (1 <= port <= 65535):
        raise HTTPException(status_code=400, detail="Invalid port")

    ws = await _get_or_create_workspace(room_id, db)
    if ws.status != "running":
        return Response(
            content=_preview_error_html("Workspace is not running — start it in the Workspace tab"),
            status_code=409, media_type="text/html",
        )

    # Forward the original query string (minus our auth token)
    qs = "&".join(
        p for p in (request.url.query or "").split("&")
        if p and not p.startswith("token=")
    )

    try:
        result = await wsd.proxy_fetch(str(room_id), port, path, query=qs)
    except httpx.HTTPStatusError as e:
        return Response(
            content=_preview_error_html(f"workspaced error: {e.response.status_code}"),
            status_code=502, media_type="text/html",
        )
    except httpx.RequestError:
        return Response(
            content=_preview_error_html("workspaced unavailable"),
            status_code=503, media_type="text/html",
        )

    if result.get("error"):
        return Response(
            content=_preview_error_html(f"Could not get a response from port {port}: {result['error']}"),
            status_code=result.get("status_code", 502), media_type="text/html",
        )

    body = base64.b64decode(result.get("body_b64", "") or "")
    return Response(
        content=body,
        status_code=result.get("status_code", 200),
        media_type=result.get("content_type") or "application/octet-stream",
    )


# ── Git API ───────────────────────────────────────────────────────────────────

class GitLogOut(BaseModel):
    entries: list[dict]
    branch: Optional[str] = None


class GitDiffOut(BaseModel):
    diff: str
    from_ref: Optional[str] = None
    to_ref: Optional[str] = None


class GitRevertIn(BaseModel):
    ref: str
    path: Optional[str] = None


class GitBranchIn(BaseModel):
    name: str


class GitMergeIn(BaseModel):
    name: str
    message: Optional[str] = None
    target: Optional[str] = None


class GitOpOut(BaseModel):
    ok: bool
    conflict: bool = False
    stdout: str = ""
    stderr: str = ""
    sha: Optional[str] = None
    branch: Optional[str] = None


class RestoreFileEntry(BaseModel):
    path: str
    current_size: int
    target_size: int
    status: str  # "restore" | "dangerous" | "change" | "unchanged"


class RestorePreviewOut(BaseModel):
    ref: str
    files: list[RestoreFileEntry]
    dangerous: int
    restored: int
    changed: int
    unchanged: int


class RestoreApplyIn(BaseModel):
    ref: str
    confirm_dangerous: bool = False


class RestoreApplyOut(BaseModel):
    ok: bool
    preview: RestorePreviewOut
    revert: Optional[dict] = None


class GitFileOut(BaseModel):
    path: str
    content: str
    rev: Optional[str] = None


@router.get("/rooms/{room_id}/workspace/log", response_model=GitLogOut)
async def workspace_git_log(
    room_id: UUID,
    path: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        data = await wsd.git_log_workspace(str(room_id), path=path, limit=limit)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")
    return data


@router.get("/rooms/{room_id}/workspace/diff", response_model=GitDiffOut)
async def workspace_git_diff(
    room_id: UUID,
    from_ref: Optional[str] = None,
    to_ref: Optional[str] = None,
    path: Optional[str] = None,
    merge_base: bool = False,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        data = await wsd.git_diff_workspace(
            str(room_id), from_ref=from_ref, to_ref=to_ref, path=path, merge_base=merge_base
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")
    return data


@router.get("/rooms/{room_id}/workspace/file", response_model=GitFileOut)
async def workspace_file_at_rev(
    room_id: UUID,
    path: str,
    rev: Optional[str] = None,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    if rev is None:
        raise HTTPException(status_code=400, detail="rev query parameter is required")
    try:
        content = await wsd.read_workspace_file_at_rev(str(room_id), path, rev)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")
    if content is None:
        raise HTTPException(status_code=404, detail=f"File {path!r} not found at revision {rev!r}")
    return GitFileOut(path=path, content=content, rev=rev)


@router.post("/rooms/{room_id}/workspace/revert", response_model=GitOpOut)
async def workspace_git_revert(
    room_id: UUID,
    body: GitRevertIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        data = await wsd.git_revert_workspace(str(room_id), body.ref, path=body.path)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")
    return data


# ── /rooms/{room_id}/workspace/restore ───────────────────────────────────────
#
# Recovery tool (internal design notes Stage 4): restores the whole
# workspace tree from an earlier commit in the room's own git history.
# /restore/preview is read-only — it never touches the volume — and flags
# any file that would go from non-empty to empty if the restore were
# applied (the exact failure mode of the incident that motivated this).
# /restore/apply refuses to proceed past a dangerous overwrite unless the
# caller explicitly sets confirm_dangerous. Stage 6: RepoFile metadata
# re-materialization is gone (table dropped); the volume is the single
# source of truth and the snapshot reflects the restore on the next read.

@router.get("/rooms/{room_id}/workspace/restore/preview", response_model=RestorePreviewOut)
async def workspace_restore_preview(
    room_id: UUID,
    ref: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        return await compute_restore_preview(room_id, ref, db)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")


@router.post("/rooms/{room_id}/workspace/restore/apply", response_model=RestoreApplyOut)
async def workspace_restore_apply(
    room_id: UUID,
    body: RestoreApplyIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        preview = await compute_restore_preview(room_id, body.ref, db)
        if preview["dangerous"] > 0 and not body.confirm_dangerous:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": (
                        f"{preview['dangerous']} file(s) would be overwritten with "
                        "empty content — resend with confirm_dangerous=true to proceed"
                    ),
                    "preview": preview,
                },
            )
        result = await apply_restore(room_id, body.ref, db)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")

    await emit_event(
        room_id=room_id,
        event_type="FILE_CHANGED",
        content=(
            f"Workspace restored to {body.ref[:8]}: {preview['restored']} file(s) "
            f"recovered, {preview['dangerous']} overwritten"
        ),
        db=db,
        metadata={"source": "restore", "ref": body.ref},
    )

    return RestoreApplyOut(ok=True, preview=preview, revert=result["revert"])


@router.post("/rooms/{room_id}/workspace/branch", response_model=GitOpOut)
async def workspace_git_create_branch(
    room_id: UUID,
    body: GitBranchIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        data = await wsd.create_workspace_branch(str(room_id), body.name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")
    return data


@router.post("/rooms/{room_id}/workspace/merge", response_model=GitOpOut)
async def workspace_git_merge_branch(
    room_id: UUID,
    body: GitMergeIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    room = await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    cfg = get_flow_config(room)
    target = body.target or cfg.working_branch
    if target == cfg.release_branch:
        raise HTTPException(
            status_code=422,
            detail=f"main merges only through a promotion gate",
        )
    result = await wsd.merge_workspace_branch(
        str(room_id), body.name, message=body.message, target=target
    )
    if result.get("conflict"):
        await emit_event(
            room_id=room_id,
            event_type="GIT_CONFLICT",
            content=f"Conflict while merging branch {body.name} into {target} — resolve manually",
            db=db,
            metadata={"branch": body.name, "target": target},
        )
    return result


# ── Declared workspace services (browser-services stage 1) ──────────────────

class ServiceDeclarationIn(BaseModel):
    slug: str
    title: str
    kind: Literal["web", "api", "worker"]
    argv: list[str]
    cwd: str
    env: dict[str, str] = Field(default_factory=dict)
    listen: dict[str, Any] = Field(default_factory=dict)
    readiness: dict[str, Any] = Field(default_factory=dict)
    default_path: str = "/"
    depends_on: list[str] = Field(default_factory=list)
    restart_policy: Literal["manual", "on_workspace_restore"] = "manual"
    environment: dict[str, Any]


class ServiceOut(BaseModel):
    id: UUID
    slug: str
    title: str
    kind: str
    desired_state: str
    runtime_state: str
    assigned_port: Optional[int] = None
    pid: Optional[int] = None
    pgid: Optional[int] = None
    default_path: str
    last_error: Optional[str] = None
    log_cursor: int = 0
    config_hash: str


class ServiceLogsOut(BaseModel):
    content: str = ""
    cursor: int = 0
    rotated: bool = False
    log_path: Optional[str] = None


class ServiceOpenTicketOut(BaseModel):
    browser_url: str
    expires_in_sec: int


def _service_out(service: WorkspaceManagedProcess) -> ServiceOut:
    listen = service.listen or {}
    return ServiceOut(
        id=service.id, slug=service.slug, title=service.title, kind=service.kind,
        desired_state=service.desired_state, runtime_state=service.runtime_state,
        assigned_port=listen.get("assigned_port"), pid=service.pid, pgid=service.pgid,
        default_path=service.default_path, last_error=service.last_error,
        log_cursor=service.log_cursor, config_hash=service.config_hash,
    )


def _require_services_enabled() -> None:
    if not services_enabled():
        raise HTTPException(status_code=404, detail="Workspace services are disabled")


@router.get("/rooms/{room_id}/workspace/services", response_model=list[ServiceOut])
async def list_workspace_services(
    room_id: UUID, db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[ServiceOut]:
    _require_services_enabled()
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    workspace = await _get_or_create_workspace(room_id, db)
    registry = WorkspaceServiceRegistry(db)
    await _reconcile_service_runtime(room_id, workspace, registry, db)
    return [_service_out(service) for service in await registry.list(workspace.id)]


@router.put("/rooms/{room_id}/workspace/services/{slug}", response_model=ServiceOut)
async def declare_workspace_service(
    room_id: UUID, slug: str, body: ServiceDeclarationIn,
    db: AsyncSession = Depends(get_session), current_user: User = Depends(deps.get_current_user),
) -> ServiceOut:
    _require_services_enabled()
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    if slug != body.slug:
        raise HTTPException(status_code=422, detail="URL slug must equal service.slug")
    workspace = await _get_or_create_workspace(room_id, db)
    registry = WorkspaceServiceRegistry(db)
    try:
        service, _changed = await registry.declare(workspace.id, body.model_dump(), source=f"user:{current_user.id}")
        await db.commit()
        await db.refresh(service)
    except ServiceValidationError as exc:
        await db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))
    return _service_out(service)


async def _start_declared_service(
    room_id: UUID, workspace: Workspace, service: WorkspaceManagedProcess,
    registry: WorkspaceServiceRegistry, db: AsyncSession, current_user: User,
    *, source: str, restart: bool = False,
) -> WorkspaceManagedProcess:
    from app.core.workspace_environments import WorkspaceEnvironmentError, prepare_service_environment

    async with registry.lock_for(workspace.id):
        if service.runtime_state in {"starting", "ready"} and not restart:
            return service
        if restart and service.runtime_state in {"starting", "ready", "degraded", "error"}:
            await registry.transition(service, "stopping", source)
            # Pass the durably stored pid/pgid as a fallback: the supervisor's
            # in-memory runtime map may have been wiped by its own restart
            # while the real process (in the separate workspace container)
            # kept running, which is exactly how a fixed port gets stuck.
            await wsd.stop_workspace_service(str(room_id), str(service.id), service.pid, service.pgid)
            service.pid = service.pgid = None
            await registry.transition(service, "stopped", source)
        # A service cannot start until its declared dependencies are healthy.
        for dep_slug in service.depends_on or []:
            dependency = await registry.get(workspace.id, dep_slug)
            if dependency is None or dependency.runtime_state != "ready":
                raise HTTPException(status_code=409, detail=f"dependency {dep_slug!r} is not ready")
        service.desired_state = "running"
        await registry.transition(service, "starting", source)
        service.last_accessed_at = datetime.now(timezone.utc)
        workspace.last_activity_at = service.last_accessed_at
        db.add(workspace)
        await db.commit()

        try:
            await ensure_workspace_started(room_id, db, quota_owner_id=current_user.id, background=False)
            runtime_env, _environment = await prepare_service_environment(
                str(room_id), workspace.id, service, registry, db,
            )
            requested = (service.listen or {}).get("port", "auto")
            # Relative cwd is an invariant of the durable declaration; the
            # internal supervisor receives only an absolute workspace path.
            spec = {
                "service_id": str(service.id), "slug": service.slug, "argv": service.argv,
                "cwd": f"/workspace/{service.cwd}", "env": {**service.env, **runtime_env},
                "port": requested, "readiness": service.readiness,
                "timeout_sec": int((service.readiness or {}).get("timeout_sec", 45)),
            }
            result = await wsd.start_workspace_service(str(room_id), spec)
        except WorkspaceEnvironmentError as exc:
            service.last_error = f"environment preparation failed ({exc.code}): {exc}"
            await registry.transition(service, "error", source, {"code": exc.code})
            await db.commit()
            return service
        except (httpx.HTTPError, WorkspaceStartError) as exc:
            service.last_error = str(exc)
            await registry.transition(service, "error", source, {"code": "start_failed"})
            await db.commit()
            raise HTTPException(status_code=502, detail=f"Could not start service: {exc}")

        # A duplicate Start can fail with ``port_conflict`` while the original
        # supervisor-owned process is still alive after a control-plane
        # restart.  Do not erase its durable PID/port/log metadata with the
        # empty error response: the next reconciliation needs it to recover.
        listen = dict(service.listen or {})
        if result.get("port") is not None:
            listen["assigned_port"] = result["port"]
        service.listen = listen
        if result.get("pid") is not None:
            service.pid = result["pid"]
        if result.get("pgid") is not None:
            service.pgid = result["pgid"]
        if result.get("log_path") is not None:
            service.log_path = result["log_path"]
        if result.get("state") == "ready":
            await registry.transition(service, "ready", source)
        else:
            service.last_error = result.get("error") or "service failed readiness"
            await registry.transition(service, "error", source, {"code": result.get("error_code", "readiness_failed")})
        await db.commit()
        return service


@router.post("/rooms/{room_id}/workspace/services/{slug}/start", response_model=ServiceOut)
async def start_declared_workspace_service(
    room_id: UUID, slug: str, db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> ServiceOut:
    _require_services_enabled()
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    workspace = await _get_or_create_workspace(room_id, db)
    registry = WorkspaceServiceRegistry(db)
    await _reconcile_service_runtime(room_id, workspace, registry, db)
    service = await registry.get(workspace.id, slug)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    return _service_out(await _start_declared_service(room_id, workspace, service, registry, db, current_user, source=f"user:{current_user.id}"))


@router.post("/rooms/{room_id}/workspace/services/{slug}/restart", response_model=ServiceOut)
async def restart_declared_workspace_service(
    room_id: UUID, slug: str, db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> ServiceOut:
    _require_services_enabled()
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    workspace = await _get_or_create_workspace(room_id, db)
    registry = WorkspaceServiceRegistry(db)
    await _reconcile_service_runtime(room_id, workspace, registry, db)
    service = await registry.get(workspace.id, slug)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    return _service_out(await _start_declared_service(room_id, workspace, service, registry, db, current_user, source=f"user:{current_user.id}", restart=True))


@router.post("/rooms/{room_id}/workspace/services/{slug}/stop", response_model=ServiceOut)
async def stop_declared_workspace_service(
    room_id: UUID, slug: str, db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> ServiceOut:
    _require_services_enabled()
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    workspace = await _get_or_create_workspace(room_id, db)
    registry = WorkspaceServiceRegistry(db)
    service = await registry.get(workspace.id, slug)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    async with registry.lock_for(workspace.id):
        try:
            await wsd.stop_workspace_service(str(room_id), str(service.id), service.pid, service.pgid)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Could not stop service: {exc}")
        service.desired_state = "stopped"
        service.pid = service.pgid = None
        if service.runtime_state != "stopped":
            await registry.transition(service, "stopping", f"user:{current_user.id}")
        await registry.transition(service, "stopped", f"user:{current_user.id}")
        await db.commit()
    return _service_out(service)


@router.get("/rooms/{room_id}/workspace/services/{slug}/logs", response_model=ServiceLogsOut)
async def declared_workspace_service_logs(
    room_id: UUID, slug: str, cursor: int = 0, db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> ServiceLogsOut:
    _require_services_enabled()
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    workspace = await _get_or_create_workspace(room_id, db)
    service = await WorkspaceServiceRegistry(db).get(workspace.id, slug)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    try:
        result = await wsd.get_workspace_service_logs(str(room_id), str(service.id), max(cursor, 0), service.slug)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not read service logs: {exc}")
    service.log_cursor = int(result.get("cursor", cursor))
    service.last_accessed_at = datetime.now(timezone.utc)
    db.add(service)
    await db.commit()
    return ServiceLogsOut(**result)


@router.get("/rooms/{room_id}/workspace/services/{slug}/environment-logs", response_model=ServiceLogsOut)
async def declared_workspace_environment_logs(
    room_id: UUID, slug: str, cursor: int = 0, db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> ServiceLogsOut:
    """Return the bounded, manager-owned dependency installation log."""
    _require_services_enabled()
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    workspace = await _get_or_create_workspace(room_id, db)
    registry = WorkspaceServiceRegistry(db)
    service = await registry.get(workspace.id, slug)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    environment = await registry.environment_for(workspace.id, service)
    if environment is None:
        raise HTTPException(status_code=404, detail="Service has no managed environment")
    from app.core.workspace_environments import WorkspaceEnvironmentError, environment_logs
    try:
        result = await environment_logs(str(room_id), environment, max(cursor, 0))
    except WorkspaceEnvironmentError as exc:
        raise HTTPException(status_code=502, detail=f"Could not read environment logs: {exc}") from exc
    environment.log_cursor = int(result.get("cursor", cursor))
    environment.updated_at = datetime.now(timezone.utc)
    db.add(environment)
    await db.commit()
    return ServiceLogsOut(**result)


@router.post(
    "/rooms/{room_id}/workspace/services/{slug}/open-ticket",
    response_model=ServiceOpenTicketOut,
)
async def open_declared_workspace_service_ticket(
    room_id: UUID, slug: str, db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> ServiceOpenTicketOut:
    """Issue a single-use service-origin ticket after membership validation."""
    _require_services_enabled()
    if not preview_enabled():
        raise HTTPException(status_code=404, detail="Workspace preview gateway is disabled")
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    workspace = await _get_or_create_workspace(room_id, db)
    registry = WorkspaceServiceRegistry(db)
    # Re-adopt the runtime in workspaced (or demote a dead one) before handing
    # out a ticket: workspaced forgets its runtime map when it restarts.
    await _reconcile_service_runtime(room_id, workspace, registry, db)
    service = await registry.get(workspace.id, slug)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    if service.kind not in {"web", "api"} or service.runtime_state != "ready":
        raise HTTPException(status_code=409, detail="Service is not ready for preview")
    origin = service_origin(service.id)
    ticket = issue_ticket(
        user_id=current_user.id, room_id=room_id, service_id=service.id, origin=origin
    )
    from urllib.parse import quote
    return ServiceOpenTicketOut(
        browser_url=f"{origin}{service.default_path}?ticket={quote(ticket, safe='')}",
        expires_in_sec=ticket_ttl_sec(),
    )


@router.delete("/rooms/{room_id}/workspace/services/{slug}", status_code=204)
async def delete_declared_workspace_service(
    room_id: UUID, slug: str, db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    _require_services_enabled()
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    workspace = await _get_or_create_workspace(room_id, db)
    service = await WorkspaceServiceRegistry(db).get(workspace.id, slug)
    if service is None:
        raise HTTPException(status_code=404, detail="Service not found")
    if service.runtime_state not in {"stopped", "error"}:
        raise HTTPException(status_code=409, detail="Stop service before deletion")
    await db.delete(service)
    await db.commit()
    return Response(status_code=204)


# ── Run configuration endpoints (Stage 5) ────────────────────────────────────

class RunConfigItem(BaseModel):
    name: str
    cmd: str
    long_running: bool = False
    cwd: str = "/workspace"
    env: dict[str, str] = Field(default_factory=dict)
    ports: list[int] = Field(default_factory=list)
    resume_policy: Literal["never", "restart"] = "never"
    readiness: dict[str, Any] = Field(default_factory=dict)


class RunsConfigIn(BaseModel):
    runs: list[RunConfigItem]


class RunStartOut(BaseModel):
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


class RunLogsOut(BaseModel):
    name: str
    log_path: str
    content: str
    running: bool


class RunStopOut(BaseModel):
    name: str
    ok: bool
    message: str = ""


@router.get("/rooms/{room_id}/workspace/runs", response_model=RunsConfigIn)
async def list_workspace_runs(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    try:
        data = await wsd.get_workspace_runs(str(room_id))
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")
    return data


@router.post("/rooms/{room_id}/workspace/runs", response_model=RunsConfigIn)
async def save_workspace_runs(
    room_id: UUID,
    body: RunsConfigIn,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)
    runs_list = [r.model_dump() for r in body.runs]
    try:
        data = await wsd.save_workspace_runs(str(room_id), runs_list)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")
    _ = data
    return body


@router.post("/rooms/{room_id}/workspace/runs/{name}/start", response_model=RunStartOut)
async def start_workspace_run(
    room_id: UUID,
    name: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    ws = await _get_or_create_workspace(room_id, db)
    if ws.status != "running":
        raise HTTPException(status_code=409, detail="Workspace is not running")

    try:
        result = await wsd.start_workspace_run(str(room_id), name)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        raise HTTPException(status_code=status if status in (404, 409) else 502, detail=e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")

    ws.last_activity_at = datetime.now(timezone.utc)
    db.add(ws)

    long_running = result.get("long_running", False)
    if long_running:
        managed = (
            await db.execute(
                select(WorkspaceManagedProcess).where(
                    WorkspaceManagedProcess.workspace_id == ws.id,
                    WorkspaceManagedProcess.name == name,
                )
            )
        ).scalar_one_or_none()
        if managed is None:
            managed = WorkspaceManagedProcess(
                workspace_id=ws.id,
                name=name,
                command=result.get("command") or "",
            )
        managed.command = result.get("command") or managed.command
        managed.cwd = result.get("cwd") or "/workspace"
        managed.env = dict(result.get("env") or {})
        managed.ports = [int(port) for port in (result.get("ports") or [])]
        managed.resume_policy = result.get("resume_policy") or "never"
        managed.readiness = dict(result.get("readiness") or {})
        managed.status = "running"
        managed.pid = result.get("pid")
        managed.last_error = None
        managed.updated_at = datetime.now(timezone.utc)
        db.add(managed)
    await emit_event(
        room_id=room_id,
        event_type="EXEC_STARTED",
        content=f"Run '{name}' started{' (background)' if long_running else ''}",
        db=db,
        metadata={"run_name": name, "long_running": long_running, "pid": result.get("pid")},
    )

    if not long_running:
        await emit_event(
            room_id=room_id,
            event_type="EXEC_FINISHED",
            content=f"Run '{name}' finished (exit {result.get('exit_code', '?')})",
            db=db,
            metadata={"run_name": name, "exit_code": result.get("exit_code")},
        )

    await db.commit()
    return result


@router.get("/rooms/{room_id}/workspace/runs/{name}/logs", response_model=RunLogsOut)
async def get_workspace_run_logs(
    room_id: UUID,
    name: str,
    lines: int = 200,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    ws = await _get_or_create_workspace(room_id, db)
    if ws.status != "running":
        raise HTTPException(status_code=409, detail="Workspace is not running")

    try:
        data = await wsd.get_workspace_run_logs(str(room_id), name, lines=lines)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")
    return data


@router.post("/rooms/{room_id}/workspace/runs/{name}/stop", response_model=RunStopOut)
async def stop_workspace_run(
    room_id: UUID,
    name: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    ws = await _get_or_create_workspace(room_id, db)
    if ws.status != "running":
        raise HTTPException(status_code=409, detail="Workspace is not running")

    try:
        data = await wsd.stop_workspace_run(str(room_id), name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")

    if data.get("ok"):
        managed = (
            await db.execute(
                select(WorkspaceManagedProcess).where(
                    WorkspaceManagedProcess.workspace_id == ws.id,
                    WorkspaceManagedProcess.name == name,
                )
            )
        ).scalar_one_or_none()
        if managed is not None:
            managed.status = "stopped"
            managed.pid = None
            managed.updated_at = datetime.now(timezone.utc)
            db.add(managed)
        await emit_event(
            room_id=room_id,
            event_type="EXEC_FINISHED",
            content=f"Run '{name}' stopped",
            db=db,
            metadata={"run_name": name, "stop_reason": "user"},
        )
        await db.commit()

    return data


# ── POST /rooms/{room_id}/workspace/sync-files ───────────────────────────────
#
# Stage 6 of internal design notes: the workspace volume is
# the single source of truth. There is no ``repofile`` table to import from
# or materialize into — this endpoint now refreshes the snapshot from the
# volume (a touch of the heartbeat) so the UI's "sync" button still does
# something useful (it forces a fresh snapshot fetch and a heartbeat ping).

@router.post("/rooms/{room_id}/workspace/sync-files")
async def sync_files_to_workspace(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    ws = await _get_or_create_workspace(room_id, db)
    if ws.status != "running":
        raise HTTPException(status_code=409, detail="Workspace is not running")

    try:
        await wsd.get_workspace_snapshot(str(room_id), max_previews=0)
    except httpx.HTTPStatusError as e:
        await broadcast_workspace_event(
            ws,
            "workspace.metadata.sync.failed",
            reason="workspaced_error",
            error={"error_code": "workspaced_error", "message": e.response.text[:500]},
        )
        raise HTTPException(status_code=502, detail=f"workspaced error: {e.response.text}")
    except httpx.RequestError as e:
        await broadcast_workspace_event(
            ws,
            "workspace.metadata.sync.failed",
            reason="workspaced_unreachable",
            error={"error_code": "workspaced_unreachable", "message": str(e)},
        )
        raise HTTPException(status_code=503, detail=f"workspaced unreachable: {e}")

    ws.last_heartbeat_at = datetime.now(timezone.utc)
    ws.last_activity_at = ws.last_heartbeat_at
    db.add(ws)
    await db.commit()

    return {"ok": True, "mode": "workspace", "synced": 0}


# ── GET /rooms/{room_id}/workspace/tool-calls ────────────────────────────────


class ToolCallItem(BaseModel):
    id: UUID
    tool_name: Optional[str] = None
    command: str
    exit_code: Optional[int] = None
    status: str = "completed"
    duration_ms: Optional[int] = None
    created_at: datetime
    initiator: str


@router.get("/rooms/{room_id}/workspace/tool-calls", response_model=list[ToolCallItem])
async def list_workspace_tool_calls(
    room_id: UUID,
    limit: int = 50,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Return recent tool calls for the workspace (from workspace_exec)."""
    await _get_room(room_id, db)
    await _verify_membership(room_id, current_user, db)

    ws_res = await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ws = ws_res.scalar_one_or_none()
    if ws is None:
        return []

    import sqlalchemy as _sa

    rows = (
        await db.execute(
            _sa.select(WorkspaceExec)
            .where(WorkspaceExec.workspace_id == ws.id)
            .order_by(WorkspaceExec.created_at.desc())
            .limit(max(1, min(limit, 200)))
        )
    ).scalars().all()

    return [
        ToolCallItem(
            id=r.id,
            tool_name=r.tool_name,
            command=r.command[:200],
            exit_code=r.exit_code,
            status=r.status,
            duration_ms=r.duration_ms,
            created_at=r.created_at,
            initiator=r.initiator,
        )
        for r in rows
    ]


# ── WS /rooms/{room_id}/workspace/term ───────────────────────────────────────

@router.websocket("/rooms/{room_id}/workspace/term")
async def workspace_terminal(
    websocket: WebSocket,
    room_id: UUID,
    token: str = "",
    db: AsyncSession = Depends(get_session),
) -> None:
    """Interactive terminal stub — full PTY forwarding implemented in Stage 6."""
    token = websocket.query_params.get("token", token)

    user_id: Optional[UUID] = None
    payload: dict = {}
    try:
        payload = security.decode_access_token(token)
        raw = payload.get("sub")
        if raw:
            user_id = UUID(raw)
    except (JWTError, Exception):
        pass

    if not user_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalars().first()
    if (not user or not user.is_active
            or await legion_auth.session_revoked(db, user.legion_account_id, payload)):
        await websocket.close(code=4001, reason="Unauthorized")
        return

    membership = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not membership.scalar_one_or_none():
        await websocket.close(code=4003, reason="Forbidden")
        return

    room = await db.get(Room, room_id)
    if not room:
        await websocket.close(code=4004, reason="Room not found")
        return

    ws_record = await _get_or_create_workspace(room_id, db)
    if ws_record.status != "running":
        await websocket.close(code=4009, reason="Workspace not running")
        return

    await websocket.accept()
    # PTY not yet implemented (Stage 6). Close immediately so the frontend
    # falls back to the REST exec endpoint, which actually runs commands.
    await websocket.close(code=4000, reason="PTY stub — use REST exec")
