"""Durable workspace/session checkpoints for lifecycle stage 5."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectin_polymorphic

from app.core.context_observations import fingerprint, invalidate_path
from app.core.path_contract import normalize_workspace_path

logger = logging.getLogger(__name__)


def _json_value(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    return value


def extract_open_tool_calls(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return provider tool calls that have no matching result yet."""
    pending: dict[str, dict[str, Any]] = {}
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    call_id = str(block.get("id") or "")
                    if call_id:
                        pending[call_id] = {
                            "id": call_id,
                            "name": block.get("name"),
                            "input": block.get("input") or {},
                            "provider": "anthropic",
                        }
                elif block.get("type") == "tool_result":
                    pending.pop(str(block.get("tool_use_id") or ""), None)
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            call_id = str(call.get("id") or "")
            function = call.get("function") or {}
            raw_arguments = function.get("arguments")
            try:
                arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else (raw_arguments or {})
            except (TypeError, json.JSONDecodeError):
                arguments = {"raw": str(raw_arguments)}
            if call_id:
                pending[call_id] = {
                    "id": call_id,
                    "name": function.get("name"),
                    "input": arguments,
                    "provider": "openai",
                }
        if message.get("role") == "tool":
            pending.pop(str(message.get("tool_call_id") or ""), None)
    return list(pending.values())


def _session_manifest(session: Any) -> dict[str, dict[str, Any]]:
    state = dict(getattr(session, "context_state", None) or {})
    ledger = dict(state.get("read_ledger") or {})
    try:
        from app.core.tools._session_state import export_reads

        for path, source_fp in export_reads(str(session.id)).items():
            entry = dict(ledger.get(path) or {})
            if source_fp:
                entry["fingerprint"] = source_fp
            entry.setdefault("ranges", [])
            entry.setdefault("observation_refs", [])
            ledger[path] = entry
    except Exception:
        logger.debug("workspace checkpoint: in-memory read export failed", exc_info=True)
    return ledger


async def checkpoint_workspace(
    room_id: UUID,
    db: AsyncSession,
    *,
    pause_reason: str,
    reason: str | None = None,
    force_stop: bool = False,
) -> Any:
    """Capture tasks, provider sessions, reads and pending operations atomically."""
    from app.models.message import Message
    from app.models.session import SESSION_SUBTYPES, Session
    from app.models.task import Task
    from app.models.workspace import (
        Workspace,
        WorkspaceBgProc,
        WorkspaceCheckpoint,
        WorkspaceExec,
        WorkspaceManagedProcess,
    )

    workspace = (
        await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ).scalar_one_or_none()
    if workspace is None:
        raise ValueError(f"workspace not found for room {room_id}")

    tasks = list(
        (
            await db.execute(
                select(Task)
                .where(Task.room_id == room_id)
                .order_by(Task.updated_at.desc())
            )
        ).scalars().all()
    )
    sessions = list(
        (
            await db.execute(
                select(Session)
                .where(Session.room_id == room_id)
                .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
                .order_by(Session.started_at.desc())
            )
        ).scalars().all()
    )
    messages = list(
        (
            await db.execute(
                select(Message)
                .where(Message.room_id == room_id)
                .order_by(Message.timestamp.desc())
                .limit(500)
            )
        ).scalars().all()
    )
    executions = list(
        (
            await db.execute(
                select(WorkspaceExec).where(
                    WorkspaceExec.workspace_id == workspace.id,
                    WorkspaceExec.status == "running",
                )
            )
        ).scalars().all()
    )
    background = list(
        (
            await db.execute(
                select(WorkspaceBgProc).where(
                    WorkspaceBgProc.workspace_id == workspace.id,
                    WorkspaceBgProc.status == "running",
                )
            )
        ).scalars().all()
    )
    managed = list(
        (
            await db.execute(
                select(WorkspaceManagedProcess).where(
                    WorkspaceManagedProcess.workspace_id == workspace.id,
                    WorkspaceManagedProcess.status.in_(["starting", "running", "paused"]),
                )
            )
        ).scalars().all()
    )

    session_payload: list[dict[str, Any]] = []
    combined_manifest: dict[str, dict[str, Any]] = {}
    for session in sessions:
        snapshot = dict(session.context_snapshot or {})
        ledger = _session_manifest(session)
        for path, entry in ledger.items():
            if entry.get("fingerprint"):
                try:
                    combined_manifest[normalize_workspace_path(path)] = dict(entry)
                except ValueError:
                    logger.warning(
                        "workspace checkpoint: skipping invalid read path %r in session %s",
                        path,
                        session.id,
                    )
        durable_state = dict(session.context_state or {})
        durable_state["read_ledger"] = ledger
        durable_state["workspace_checkpoint_pending"] = True
        session_payload.append(
            {
                "id": str(session.id),
                "kind": session.kind,
                "task_id": str(getattr(session, "task_id", "") or "") or None,
                "peer_session_id": str(session.peer_session_id or "") or None,
                "started_at": _json_value(session.started_at),
                "closed_at": _json_value(session.closed_at),
                "provider_session_metadata": _json_value(
                    session.provider_session_metadata or {}
                ),
                "context_snapshot": _json_value(snapshot),
                "context_state": _json_value(durable_state),
                "pending_actions": _json_value(session.pending_actions or {}),
                "open_tool_calls": extract_open_tool_calls(snapshot.get("messages") or []),
            }
        )
        # Room-facing user conversations are not workspace agent runtimes; keep
        # them open so a browser reconnect does not lose the chat attachment.
        is_workspace_agent = session.kind not in {"user_cm_p1", "user_cm_p2"}
        if is_workspace_agent and session.closed_at is None:
            session.closed_at = datetime.now(timezone.utc)
            session.close_reason = "workspace_paused"
        session.context_state = durable_state
        db.add(session)

    payload = {
        "schema_version": 1,
        "tasks": [
            {
                "id": str(task.id),
                "state": getattr(task.state, "value", str(task.state)),
                "pause_reason": task.pause_reason,
                "assigned_agent_id": str(task.assigned_agent_id or "") or None,
                "goal_id": str(task.goal_id or "") or None,
                "dispatch_status": getattr(
                    task.dispatch_status, "value", str(task.dispatch_status)
                ),
            }
            for task in tasks
        ],
        "sessions": session_payload,
        "messages": [
            {
                "id": str(message.id),
                "event_type": message.event_type,
                "task_id": str(message.task_id or "") or None,
                "content": message.content,
                "metadata": _json_value(message.event_metadata or {}),
                "timestamp": _json_value(message.timestamp),
            }
            for message in reversed(messages)
        ],
        "read_manifest": combined_manifest,
        "open_tool_calls": [
            call for item in session_payload for call in item["open_tool_calls"]
        ],
        "pending_operations": [
            {
                "id": str(item.id),
                "command": item.command,
                "tool_name": item.tool_name,
                "initiator": item.initiator,
            }
            for item in executions
        ],
        "interrupted_background_processes": [
            {
                "bg_id": item.bg_id,
                "command": item.cmd,
                "log_path": item.log_path,
            }
            for item in background
        ],
        "managed_processes": [
            {
                "id": str(item.id),
                "name": item.name,
                "command": item.command,
                "slug": item.slug,
                "title": item.title,
                "kind": item.kind,
                "argv": item.argv,
                "cwd": item.cwd,
                "env": item.env,
                "ports": item.ports,
                "listen": item.listen,
                "resume_policy": item.resume_policy,
                "restart_policy": item.restart_policy,
                "readiness": item.readiness,
                "config_hash": item.config_hash,
                "desired_state": item.desired_state,
                "status": item.status,
            }
            for item in managed
        ],
    }
    checkpoint = WorkspaceCheckpoint(
        workspace_id=workspace.id,
        room_id=room_id,
        operation_id=workspace.operation_id,
        reason=reason or pause_reason,
        pause_reason=pause_reason,
        generation=_json_value(workspace.workspace_generation),
        payload=payload,
        force_stop=force_stop,
    )
    workspace.active_session_count = 0
    db.add(workspace)
    db.add(checkpoint)
    await db.flush()
    await prune_room_checkpoints(room_id, db)
    await db.commit()
    await db.refresh(checkpoint)
    return checkpoint


# internal design notes §5.4: a checkpoint is written on every stop and room
# switch but only the latest is ever read (_latest_checkpoint). The previous
# one stays as a fallback; everything older is dead weight (~1 MB each).
CHECKPOINTS_KEPT_PER_ROOM = 2


async def prune_room_checkpoints(
    room_id: UUID, db: AsyncSession, *, keep: int = CHECKPOINTS_KEPT_PER_ROOM
) -> int:
    """Delete all but the ``keep`` newest checkpoints of a room; no commit."""
    from sqlalchemy import delete

    from app.models.workspace import WorkspaceCheckpoint

    newest = (
        select(WorkspaceCheckpoint.id)
        .where(WorkspaceCheckpoint.room_id == room_id)
        .order_by(WorkspaceCheckpoint.created_at.desc(), WorkspaceCheckpoint.id.desc())
        .limit(keep)
    )
    kept = list((await db.execute(newest)).scalars().all())
    result = await db.execute(
        delete(WorkspaceCheckpoint)
        .where(WorkspaceCheckpoint.room_id == room_id)
        .where(WorkspaceCheckpoint.id.not_in(kept))
        .execution_options(synchronize_session=False)
    )
    return int(result.rowcount or 0)


async def _latest_checkpoint(room_id: UUID, db: AsyncSession) -> Any | None:
    from app.models.workspace import WorkspaceCheckpoint

    return (
        await db.execute(
            select(WorkspaceCheckpoint)
            .where(WorkspaceCheckpoint.room_id == room_id)
            .order_by(WorkspaceCheckpoint.created_at.desc())
            .limit(1)
        )
    ).scalars().first()


async def _validate_fingerprints(
    room_id: UUID,
    manifest: dict[str, dict[str, Any]],
    db: AsyncSession,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    from app.core import workspaced_client as wsd

    valid: dict[str, str] = {}
    stale: list[dict[str, Any]] = []
    for raw_path, entry in manifest.items():
        path = normalize_workspace_path(raw_path)
        expected = str((entry or {}).get("fingerprint") or "")
        try:
            content = await wsd.read_workspace_file(str(room_id), path)
            actual = fingerprint(content) if content is not None else None
        except Exception as exc:
            stale.append(
                {"path": path, "expected": expected, "actual": None, "error": str(exc)}
            )
            continue
        if expected and actual == expected:
            valid[path] = expected
            continue
        stale.append({"path": path, "expected": expected, "actual": actual})
        await invalidate_path(db, room_id, path, new_fingerprint=actual)
    return valid, stale


async def _wait_process_ready(
    room_id: UUID,
    ports: list[int],
    *,
    timeout_sec: float = 30.0,
) -> bool:
    if not ports:
        return True
    from app.core import workspaced_client as wsd

    deadline = asyncio.get_running_loop().time() + timeout_sec
    checks = " && ".join(
        f"(echo >/dev/tcp/127.0.0.1/{int(port)})" for port in ports
    )
    while asyncio.get_running_loop().time() < deadline:
        try:
            result = await wsd.exec_in_workspace(
                str(room_id),
                ["bash", "-c", checks],
                timeout_sec=5,
            )
            if int(result.get("exit_code", 1)) == 0:
                return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


def should_restore_declared_service(process: Any, checkpoint_record: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Decide restore eligibility without trusting a stale checkpoint spec.

    The registry row remains authoritative.  A checkpoint merely proves that
    this exact declaration was healthy before the runtime stopped; a changed
    declaration must be started explicitly instead of resurrecting an older
    command/configuration.
    """
    if process.restart_policy != "on_workspace_restore":
        return False, "restart_policy_manual"
    if process.desired_state != "running":
        return False, "desired_state_stopped"
    if checkpoint_record is None:
        return False, "checkpoint_service_missing"
    if checkpoint_record.get("desired_state") != "running":
        return False, "checkpoint_desired_state_stopped"
    if checkpoint_record.get("config_hash") != process.config_hash:
        return False, "config_hash_mismatch"
    return True, None


async def restore_managed_processes(
    room_id: UUID,
    db: AsyncSession,
    *,
    checkpoint_services: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Restore legacy Runs and opt-in declared services through one registry."""
    from app.core import workspaced_client as wsd
    from app.core.workspace_environments import prepare_service_environment
    from app.core.workspace_services import WorkspaceServiceRegistry
    from app.models.workspace import Workspace, WorkspaceManagedProcess

    workspace = (
        await db.execute(select(Workspace).where(Workspace.room_id == room_id))
    ).scalar_one_or_none()
    if workspace is None:
        return []
    registry = WorkspaceServiceRegistry(db)
    processes = list(
        (
            await db.execute(
                select(WorkspaceManagedProcess).where(
                    WorkspaceManagedProcess.workspace_id == workspace.id,
                    WorkspaceManagedProcess.status.in_(["paused", "error", "stopped"]),
                )
            )
        ).scalars().all()
    )
    outcomes: list[dict[str, Any]] = []
    checkpoint_by_id = {
        str(item.get("id")): item
        for item in checkpoint_services
        if isinstance(item, dict) and item.get("id")
    }
    for process in processes:
        is_service = bool(process.argv)
        if is_service:
            should_restore, skip_reason = should_restore_declared_service(
                process, checkpoint_by_id.get(str(process.id))
            )
        else:
            should_restore, skip_reason = process.resume_policy == "restart", None
        if not should_restore:
            if is_service and skip_reason not in {"restart_policy_manual", "desired_state_stopped"}:
                process.status = "stopped"
                process.runtime_state = "stopped"
                process.last_error = (
                    "service configuration changed since checkpoint; start it explicitly"
                    if skip_reason == "config_hash_mismatch"
                    else None
                )
                process.updated_at = datetime.now(timezone.utc)
                db.add(process)
                outcomes.append({"name": process.name, "status": "skipped", "reason": skip_reason})
            continue
        process.status = "starting"
        process.runtime_state = "starting" if is_service else process.runtime_state
        process.last_error = None
        process.updated_at = datetime.now(timezone.utc)
        db.add(process)
        await db.flush()
        try:
            if is_service:
                runtime_env, _environment = await prepare_service_environment(
                    str(room_id), workspace.id, process, registry, db,
                )
                requested = (process.listen or {}).get("port", "auto")
                result = await wsd.start_workspace_service(str(room_id), {
                    "service_id": str(process.id), "slug": process.slug,
                    "argv": process.argv, "cwd": f"/workspace/{process.cwd.lstrip('/')}",
                    "env": {**process.env, **runtime_env}, "port": requested, "readiness": process.readiness,
                    "timeout_sec": int((process.readiness or {}).get("timeout_sec", 45)),
                })
                if result.get("state") != "ready":
                    raise RuntimeError(result.get("error") or "service readiness failed")
                process.pid = result.get("pid")
                process.pgid = result.get("pgid")
                process.log_path = result.get("log_path")
                process.listen = {**(process.listen or {}), "assigned_port": result.get("port")}
                process.runtime_state = "ready"
                outcomes.append({"name": process.name, "status": "ready"})
            else:
                result = await wsd.start_workspace_run(str(room_id), process.name)
                process.pid = result.get("pid")
                ready = await _wait_process_ready(
                    room_id,
                    [int(port) for port in (process.ports or [])],
                    timeout_sec=float((process.readiness or {}).get("timeout_sec", 30)),
                )
                if not ready:
                    raise RuntimeError("managed process readiness timeout")
                outcomes.append({"name": process.name, "status": "running"})
            process.status = "running"
        except Exception as exc:
            process.status = "error"
            process.runtime_state = "error" if is_service else process.runtime_state
            process.last_error = str(exc)[:2_000]
            outcomes.append(
                {"name": process.name, "status": "error", "error": process.last_error}
            )
        process.updated_at = datetime.now(timezone.utc)
        db.add(process)
    await db.commit()
    return outcomes


async def restore_workspace_checkpoint(
    room_id: UUID,
    db: AsyncSession,
    *,
    generation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate the latest checkpoint and prepare it for new provider sessions."""
    from app.core import workspaced_client as wsd

    checkpoint = await _latest_checkpoint(room_id, db)
    if checkpoint is None or checkpoint.status not in {"created", "validated", "stale"}:
        return None
    payload = dict(checkpoint.payload or {})
    valid, stale = await _validate_fingerprints(
        room_id, dict(payload.get("read_manifest") or {}), db
    )
    previous_generation = checkpoint.generation or {}
    generation = generation or {}
    epoch_changed = bool(
        previous_generation.get("epoch")
        and generation.get("epoch")
        and previous_generation.get("epoch") != generation.get("epoch")
    )
    if epoch_changed and valid:
        for path, source_fp in list(valid.items()):
            stale.append(
                {
                    "path": path,
                    "expected": source_fp,
                    "actual": source_fp,
                    "reason": "workspace_epoch_changed",
                }
            )
            await invalidate_path(db, room_id, path)
        valid = {}
    git_status = ""
    git_status_error = None
    if checkpoint.force_stop:
        try:
            result = await wsd.exec_in_workspace(
                str(room_id),
                ["git", "status", "--porcelain=v1"],
                timeout_sec=15,
            )
            git_status = str(result.get("stdout") or "")
            if int(result.get("exit_code", 0)) != 0:
                git_status_error = str(result.get("stderr") or "git status failed")
        except Exception as exc:
            git_status_error = str(exc)

    process_outcomes = await restore_managed_processes(
        room_id,
        db,
        checkpoint_services=payload.get("managed_processes") or [],
    )
    task_valid_reads: dict[str, dict[str, str]] = {}
    for source_session in payload.get("sessions") or []:
        task_id = str(source_session.get("task_id") or "")
        if not task_id:
            continue
        ledger = dict(
            (source_session.get("context_state") or {}).get("read_ledger") or {}
        )
        scoped = task_valid_reads.setdefault(task_id, {})
        for path in ledger:
            try:
                canonical_path = normalize_workspace_path(path)
            except ValueError:
                continue
            if canonical_path in valid:
                scoped[canonical_path] = valid[canonical_path]
    validation = {
        "generation_before": previous_generation,
        "generation_after": generation,
        "epoch_changed": epoch_changed,
        "valid_reads": valid,
        "task_valid_reads": task_valid_reads,
        "stale_paths": stale,
        "git_status": git_status,
        "git_status_error": git_status_error,
        "managed_processes": process_outcomes,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }
    checkpoint.validation = validation
    process_errors = [
        item for item in process_outcomes if item.get("status") == "error"
    ]
    process_config_mismatches = [
        item for item in process_outcomes
        if item.get("reason") == "config_hash_mismatch"
    ]
    checkpoint.status = (
        "stale" if epoch_changed or stale or process_errors or process_config_mismatches else "restored"
    )
    checkpoint.restored_at = datetime.now(timezone.utc)
    db.add(checkpoint)
    await db.commit()
    return validation


async def apply_checkpoint_to_session(
    room_id: UUID,
    session_id: UUID,
    db: AsyncSession,
    *,
    task_id: UUID | None = None,
) -> dict[str, Any] | None:
    """Restore only fingerprint-validated reads into a newly-created session."""
    from app.core.tools._session_state import restore_reads, restore_tool_failure
    from app.models.session import Session

    checkpoint = await _latest_checkpoint(room_id, db)
    if checkpoint is None or not checkpoint.validation:
        return None
    validation = dict(checkpoint.validation)
    valid_reads = dict(
        (validation.get("task_valid_reads") or {}).get(str(task_id), {})
        if task_id is not None
        else validation.get("valid_reads") or {}
    )
    restore_reads(str(session_id), valid_reads, replace=True)
    session = await db.get(Session, session_id)
    if session is not None:
        source_sessions = list((checkpoint.payload or {}).get("sessions") or [])
        source_state = next(
            (
                dict(item.get("context_state") or {})
                for item in source_sessions
                if item.get("task_id")
                and (task_id is None or str(item.get("task_id")) == str(task_id))
            ),
            {},
        )
        source_ledger = dict(source_state.get("read_ledger") or {})
        source_state["read_ledger"] = {
            path: source_ledger.get(path, {"fingerprint": source_fp, "ranges": []})
            for path, source_fp in valid_reads.items()
        }
        source_state["workspace_checkpoint_pending"] = False
        source_state["workspace_checkpoint_id"] = str(checkpoint.id)
        if source_state.get("tool_error_retry"):
            restore_tool_failure(str(session_id), source_state["tool_error_retry"])
        session.context_state = source_state
        db.add(session)
        await db.flush()
    return validation


async def checkpoint_resume_note(room_id: UUID, db: AsyncSession) -> str | None:
    checkpoint = await _latest_checkpoint(room_id, db)
    if checkpoint is None or not checkpoint.validation:
        return None
    validation = dict(checkpoint.validation)
    stale_paths = list(validation.get("stale_paths") or [])
    epoch_changed = bool(validation.get("epoch_changed"))
    process_errors = [
        item for item in validation.get("managed_processes") or []
        if item.get("status") == "error"
    ]
    if not stale_paths and not epoch_changed and not process_errors:
        return (
            "[Workspace restored from checkpoint. Previously read files were "
            "fingerprint-validated and remain valid; do not re-read them unless needed.]"
        )
    parts = ["[Workspace context validation detected stale state.]"]
    if epoch_changed:
        parts.append("The workspace volume epoch changed; treat prior filesystem context as stale.")
    if stale_paths:
        parts.append(
            "Changed or unavailable paths: "
            + ", ".join(str(item.get("path")) for item in stale_paths[:40])
            + ". Re-read these paths before relying on prior content."
        )
    if process_errors:
        parts.append(
            "Managed processes not ready: "
            + ", ".join(str(item.get("name")) for item in process_errors)
            + ". Diagnose them before continuing dependent work."
        )
    return "\n".join(parts)
