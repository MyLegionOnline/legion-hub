"""Authenticated Hub API for managed stdio personal MCPs."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.core.personal_mcp import (
    CONTRACT_VERSION,
    PersonalMcpError,
    launch_spec_from_candidate,
    owned_runtime,
    owner_lifecycle_lock,
    parse_import_payload,
    preview_response,
    record_event,
    require_feature,
    require_rollout_capacity,
    transition_runtime,
)
from app.core import host_capacity
from app.core.personal_mcp_host import PersonalMcpHostError, host_client
from app.core.metrics import (
    record_personal_mcp_failure,
    record_personal_mcp_import,
    record_personal_mcp_quota_denied,
    record_personal_mcp_verify,
    set_personal_mcp_usage,
)
from app.core.workspace_quota import get_effective_limits
from app.core.secrets.lease import ManagedMcpSecretLeaseService, SecretLeaseError
from app.core.secrets.room_secrets import EnvironmentLocked
from app.core.secrets.store import PostgresEncryptedSecretStore, SecretStoreError
from app.db.session import get_session
from app.models.market import RoomMcpServer, UserInstalledItem
from app.models.personal_mcp import PersonalMcpEvent, PersonalMcpImportPreview, PersonalMcpRuntime
from app.models.room import Room
from app.models.secret import (
    SecretActorType,
    SecretAudit,
    SecretAuditAction,
    SecretAuditOutcome,
    SecretBinding,
    SecretPurpose,
    SecretResourceType,
    SecretState,
    UserSecret,
)
from app.models.user import User
from app.schemas.personal_mcp import (
    CreatePersonalMcpRequest,
    PersonalMcpEventOut,
    PersonalMcpOut,
    PersonalMcpUsageOut,
    PersonalSecretBindingOut,
    ResourceUsageOut,
    RoomBindingOut,
    RoomBindingRequest,
    SecretBindingsRequest,
    ToolsSnapshotOut,
)


router = APIRouter(prefix="/personal-mcps")

# Keep the public resource presets honest for the operator-owned Node runner.
# Existing rows created with older resource presets are promoted on their next
# start and remain subject to the owner's normal memory/storage quotas.
MIN_MANAGED_MCP_MEMORY_MB = 512
MIN_MANAGED_MCP_STORAGE_MB = 512


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _active_status(value: Any) -> str:
    if isinstance(value, SecretState):
        return value.value
    raw = getattr(value, "value", value)
    return str(raw)


def _error(code: str, message: str, remedy: str, *, status_code: int = 422, details: dict[str, Any] | None = None) -> PersonalMcpError:
    return PersonalMcpError(code, message, remedy, status_code=status_code, details=details)


def _host_error(code: str, *, correlation_id: str) -> dict[str, Any]:
    retryable = code in {"runner_unavailable", "runner_not_ready", "protocol_handshake_timeout", "package_registry_unavailable"}
    messages = {
        "not_mcp_server": "The process did not respond as an MCP server.",
        "invalid_tools_response": "MCP returned an invalid tool list.",
        "protocol_handshake_timeout": "MCP did not complete the protocol handshake in time.",
        "runner_not_ready": "The isolated runner did not become ready.",
        "runner_unavailable": "The isolated runner is temporarily unavailable.",
        "runner_request_rejected": "The runner rejected the launch request.",
        "runner_invalid_response": "The runner returned an invalid status.",
        "package_not_pinned": "Could not pin the npm package version.",
        "package_registry_unavailable": "The npm registry is temporarily unavailable.",
        "secret_lease_required": "No Secret access was granted for launching the MCP.",
        "secret_lease_invalid": "The Secret access for the MCP is invalid.",
        "tool_fingerprint_changed": "The MCP tool list changed after verification.",
    }
    remedies = {
        "not_mcp_server": "Check the package and its MCP arguments.",
        "invalid_tools_response": "Update the package or choose another MCP.",
        "protocol_handshake_timeout": "Check the launch arguments and run the check again.",
        "runner_not_ready": "Run the check again later; use the correlation ID for diagnostics.",
        "runner_unavailable": "Try launching again later.",
        "runner_request_rejected": "Check the resource settings and try launching again.",
        "runner_invalid_response": "Try launching again; if the error persists, contact the operator.",
        "package_not_pinned": "Run the check again to pin the package version.",
        "package_registry_unavailable": "Run the check again later.",
        "secret_lease_required": "Check the active binding and run the check again.",
        "secret_lease_invalid": "Run the check again after updating the Secret binding.",
        "tool_fingerprint_changed": "Run the MCP check again before using it.",
    }
    return {
        "code": code if code in messages else "runner_unavailable",
        "message": messages.get(code, messages["runner_unavailable"]),
        "remedy": remedies.get(code, remedies["runner_unavailable"]),
        "retryable": retryable,
        "details": {},
        "correlation_id": correlation_id,
    }


def _host_tools_snapshot(response: dict[str, Any]) -> dict[str, Any]:
    """Validate the bounded tools snapshot produced by mcp_host.

    The host is the component attached to the private MCP network and performs
    the readiness plus protocol probe there. Hub must not call the returned
    runner endpoint directly because it intentionally has no mcp-network
    attachment.
    """

    raw = response.get("tools")
    if not isinstance(raw, dict):
        raise PersonalMcpHostError("invalid_tools_response", retryable=False)
    names = raw.get("names")
    fingerprint = raw.get("fingerprint")
    count = raw.get("count")
    if (
        not isinstance(names, list)
        or len(names) > 512
        or any(not isinstance(name, str) or not name or len(name) > 128 for name in names)
        or not isinstance(fingerprint, str)
        or len(fingerprint) > 256
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
    ):
        raise PersonalMcpHostError("invalid_tools_response", retryable=False)
    return {
        "count": min(count, len(names)),
        "names": names[:512],
        "fingerprint": fingerprint,
    }


async def _mark_runtime_error(
    runtime: PersonalMcpRuntime,
    db: AsyncSession,
    code: str,
) -> None:
    record_personal_mcp_failure(code)
    if runtime.actual_status in {"starting", "running", "verifying", "updating"}:
        transition_runtime(runtime, "error")
    runtime.last_error_code = code
    runtime.last_error_safe = _host_error(code, correlation_id=runtime.correlation_id)
    await record_event(
        db,
        runtime,
        step="failed",
        event_type="diagnostic",
        message=runtime.last_error_safe["message"],
        percent=100,
    )


async def _complete_host_verification(
    runtime: PersonalMcpRuntime,
    item: UserInstalledItem,
    db: AsyncSession,
    *,
    owner_id: UUID,
    secret_lease_token: str | None = None,
) -> None:
    """Start the operator host and atomically record its verified result."""

    verification_started = time.monotonic()
    await record_event(db, runtime, step="package_resolving", message="Checking the npm package version.", percent=35)
    await record_event(db, runtime, step="runner_starting", message="Starting the isolated runner.", percent=50)
    try:
        response = await host_client.start(
            owner_id,
            runtime.installed_item_id,
            runtime.launch_spec,
            memory_mb=runtime.memory_reservation_mb,
            storage_mb=runtime.storage_reservation_mb,
            revision=runtime.revision,
            correlation_id=runtime.correlation_id,
            secret_lease_token=secret_lease_token,
        )
        host_status = response.get("status")
        if host_status == "error":
            raise PersonalMcpHostError(
                str(response.get("error_code") or "runner_not_ready"),
                retryable=True,
            )
        endpoint = response.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise PersonalMcpHostError("runner_invalid_response", retryable=False)

        if runtime.actual_status == "starting":
            transition_runtime(runtime, "running")
        if runtime.actual_status == "running":
            await record_event(db, runtime, step="protocol_initializing", message="Checking MCP initialize.", percent=70)
            transition_runtime(runtime, "verifying")

        # mcp_host owns the private MCP network and has already performed the
        # readiness + initialize/tools-list probe before returning. Hub
        # validates and persists that bounded result instead of trying to
        # reach the private runner endpoint from the public Hub network.
        snapshot = _host_tools_snapshot(response)

        runtime.endpoint = endpoint
        runtime.tools_snapshot = snapshot
        runtime.tool_fingerprint = snapshot["fingerprint"]
        artifact = response.get("resolved_artifact")
        if isinstance(artifact, dict) and artifact.get("version") and artifact.get("integrity"):
            runtime.resolved_artifact = {
                "package": runtime.launch_spec.get("package"),
                "version": artifact["version"],
                "integrity": artifact["integrity"],
                "resolved_at": artifact.get("resolved_at"),
            }
            runtime.launch_spec = {
                **runtime.launch_spec,
                "resolved_artifact": {
                    "version": artifact["version"],
                    "integrity": artifact["integrity"],
                    "resolved_at": artifact.get("resolved_at"),
                },
            }
            await record_event(db, runtime, step="package_pinned", message="The npm package version is pinned.", percent=60)
        runtime.memory_usage_mb = int(float(response.get("memory_usage_mb") or 0))
        runtime.storage_used_mb = int(float(response.get("storage_used_mb") or 0))
        await record_event(
            db,
            runtime,
            step="tools_discovered",
            message="MCP tools discovered.",
            percent=90,
            safe_details=snapshot,
        )
        transition_runtime(runtime, "verified")
        runtime.desired_status = "running"
        runtime.last_error_code = None
        runtime.last_error_safe = None
        record_personal_mcp_verify(
            outcome="success",
            latency_seconds=time.monotonic() - verification_started,
        )
        await record_event(db, runtime, step="ready", message="MCP verified and ready to be assigned to a room.", percent=100)
    except PersonalMcpHostError as exc:
        record_personal_mcp_verify(
            outcome="failure",
            latency_seconds=time.monotonic() - verification_started,
            reason=exc.code,
        )
        await _mark_runtime_error(runtime, db, exc.code)
        try:
            await host_client.stop(owner_id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
        except Exception:
            pass


async def _issue_runtime_secret_lease(
    request: Request,
    runtime: PersonalMcpRuntime,
    owner_id: UUID,
    db: AsyncSession,
) -> str | None:
    """Issue the one-time host capability only when the profile declares envs."""

    env_names = runtime.launch_spec.get("env_names", []) if isinstance(runtime.launch_spec, dict) else []
    if not env_names:
        return None
    keyring = getattr(request.app.state, "secret_keyring", None)
    if keyring is None:
        await _mark_runtime_error(runtime, db, "secret_required")
        await db.commit()
        raise _error(
            "secret_required",
            "No Secret storage is configured for this MCP.",
            "Configure a Secret and run the check again.",
            status_code=409,
        )
    service = ManagedMcpSecretLeaseService(
        db,
        store=PostgresEncryptedSecretStore(db, keyring=keyring),
    )
    try:
        lease = await service.issue(
            owner_id=owner_id,
            resource_id=runtime.installed_item_id,
            runtime_revision=runtime.revision,
        )
        await db.commit()
        return lease.lease_token
    except EnvironmentLocked as exc:
        await db.rollback()
        raise _error(
            "environment_locked",
            f"Environment '{exc}' is protected — access confirmation is required.",
            "Open the confirmation window for this environment and run the check again.",
            status_code=423,
        ) from exc
    except SecretLeaseError as exc:
        await db.rollback()
        if hasattr(db, "refresh"):
            await db.refresh(runtime)
        await _mark_runtime_error(runtime, db, "secret_required")
        await db.commit()
        raise _error(
            "secret_required",
            "Could not grant Secret access for launching the MCP.",
            "Check the active binding and run the check again.",
            status_code=409,
        ) from exc
    except SecretStoreError as exc:
        await db.rollback()
        if hasattr(db, "refresh"):
            await db.refresh(runtime)
        await _mark_runtime_error(runtime, db, "secret_required")
        await db.commit()
        raise _error(
            "secret_required",
            "Secret storage is temporarily unavailable.",
            "Run the check again later.",
            status_code=503,
        ) from exc
    except Exception:
        await _mark_runtime_error(runtime, db, "runner_unavailable")
        try:
            await host_client.stop(owner_id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
        except Exception:
            pass


async def _preview_candidate(
    body: CreatePersonalMcpRequest,
    user_id: UUID,
    db: AsyncSession,
) -> tuple[PersonalMcpImportPreview, dict[str, Any]]:
    preview = await db.get(PersonalMcpImportPreview, body.preview_id)
    if preview is None or preview.owner_id != user_id:
        raise _error("preview_not_found", "Import preview not found.", "Run the configuration preview again.", status_code=404)
    expires_at = preview.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= _now():
        raise _error("preview_not_found", "Import preview expired.", "Run the configuration preview again.", status_code=404)
    candidate = next(
        (item for item in (preview.candidates or []) if str(item.get("candidate_id")) == str(body.candidate_id)),
        None,
    )
    if candidate is None or not candidate.get("supported", False):
        raise _error("candidate_not_found", "MCP candidate not found.", "Run the preview again and choose a supported server.", status_code=404)
    return preview, candidate


async def _validate_secret_references(
    env_names: list[str],
    bindings: list[dict[str, Any]],
    owner_id: UUID,
    db: AsyncSession,
) -> None:
    seen: set[str] = set()
    allowed = set(env_names)
    for binding in bindings:
        env_name = str(binding["env_name"])
        if env_name in seen or env_name not in allowed:
            raise _error(
                "secret_binding_invalid",
                "The Secret binding does not match the declared MCP variable.",
                "Choose an env name from the preview and an active Secret of the owner.",
            )
        seen.add(env_name)
        secret = await db.get(UserSecret, binding["secret_id"])
        if (
            secret is None
            or secret.owner_id != owner_id
            or _active_status(secret.state) != SecretState.ACTIVE.value
            or secret.kind != "mcp_env"
        ):
            raise _error(
                "secret_binding_invalid",
                "The Secret was not found, is disabled, or does not belong to the MCP owner.",
                "Choose an active Secret of the owner.",
            )


async def _secret_bindings_out(runtime: PersonalMcpRuntime, db: AsyncSession) -> list[dict[str, Any]]:
    result = await db.execute(
        select(SecretBinding, UserSecret)
        .join(UserSecret, UserSecret.id == SecretBinding.secret_id, isouter=True)
        .where(
            SecretBinding.owner_id == runtime.owner_id,
            SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
            SecretBinding.resource_id == runtime.installed_item_id,
            SecretBinding.purpose == SecretPurpose.MCP_ENV,
        )
        .order_by(SecretBinding.created_at)
    )
    output: list[dict[str, Any]] = []
    for binding, secret in result.all():
        config = binding.injection_config or {}
        env_name = config.get("name") or config.get("env_name")
        if not isinstance(env_name, str):
            continue
        if secret is None:
            state = "missing"
            label = "Deleted Secret"
        else:
            raw_state = _active_status(secret.state)
            state = raw_state if raw_state in {"active", "disabled"} else "missing"
            label = secret.label
        output.append({"env_name": env_name, "secret_id": binding.secret_id, "label": label, "state": state})
    return output


def _record_mcp_binding_audit(
    db: AsyncSession,
    *,
    owner_id: UUID,
    item_id: UUID,
    secret_id: UUID,
    env_name: str,
    revision: int,
    action: SecretAuditAction = SecretAuditAction.BIND,
) -> None:
    """Persist binding metadata only; never accept a value or token here."""

    db.add(
        SecretAudit(
            actor_type=SecretActorType.USER,
            actor_id=owner_id,
            action=action,
            outcome=SecretAuditOutcome.SUCCESS,
            secret_id=secret_id,
            resource_type=SecretResourceType.INSTALLED_MCP,
            resource_id=item_id,
            purpose=SecretPurpose.MCP_ENV,
            env_name=env_name,
            runtime_revision=revision,
        )
    )


async def _room_bindings_out(runtime: PersonalMcpRuntime, db: AsyncSession) -> list[dict[str, Any]]:
    result = await db.execute(
        select(RoomMcpServer)
        .where(RoomMcpServer.user_installed_item_id == runtime.installed_item_id)
        .order_by(RoomMcpServer.created_at)
    )
    available = runtime.actual_status == "verified"
    state = "available" if available else ("stopped" if runtime.actual_status == "stopped" else "unavailable")
    return [
        {"room_id": binding.room_id, "available": available, "status": state}
        for binding in result.scalars().all()
    ]


async def _as_out(runtime: PersonalMcpRuntime, item: UserInstalledItem, db: AsyncSession) -> dict[str, Any]:
    launch_spec_raw = runtime.launch_spec if isinstance(runtime.launch_spec, dict) else {}
    # Whitelist the normalized launch profile.  This remains safe even if an
    # operator repairs a legacy row containing an unexpected JSON key.
    launch_spec = {
        key: launch_spec_raw[key]
        for key in (
            "mode",
            "runtime_profile",
            "transport",
            "package",
            "command_args",
            "env_names",
            "normalized_argv",
            "shell",
            "resolved_artifact",
        )
        if key in launch_spec_raw
    }
    snapshot = runtime.tools_snapshot if isinstance(runtime.tools_snapshot, dict) else {}
    names = [name for name in snapshot.get("names", []) if isinstance(name, str)]
    tools = {
        "count": int(snapshot.get("count", len(names)) or 0),
        "names": names[:512],
        "fingerprint": runtime.tool_fingerprint or snapshot.get("fingerprint") or "",
    }
    return {
        "id": runtime.installed_item_id,
        "contract_version": CONTRACT_VERSION,
        "name": item.name,
        "mode": "managed_stdio",
        "runtime_profile": runtime.runtime_profile,
        "transport": "stdio",
        "desired_status": runtime.desired_status,
        "actual_status": runtime.actual_status,
        "ready_for_room": runtime.actual_status == "verified",
        "revision": runtime.revision,
        "launch_profile": launch_spec,
        "resolved_artifact": runtime.resolved_artifact,
        "resources": ResourceUsageOut(
            memory_reservation_mb=runtime.memory_reservation_mb if runtime.actual_status not in {"stopped", "deleted"} else 0,
            memory_usage_mb=runtime.memory_usage_mb if runtime.actual_status not in {"stopped", "deleted"} else 0,
            storage_reservation_mb=runtime.storage_reservation_mb,
            storage_used_mb=runtime.storage_used_mb,
        ),
        "tools": ToolsSnapshotOut(**tools),
        "room_bindings": await _room_bindings_out(runtime, db),
        "secret_bindings": await _secret_bindings_out(runtime, db),
        "last_error": runtime.last_error_safe,
        "correlation_id": runtime.correlation_id,
        "created_at": runtime.created_at,
        "updated_at": runtime.updated_at,
    }


def _personal_limits(limits: dict[str, Any]) -> dict[str, Any]:
    """Expose the v1 keys while old plan rows are being migrated."""

    keys = (
        "mcp_personal_max_installed",
        "mcp_personal_max_running",
        "mcp_personal_max_total_memory_mb",
        "mcp_personal_max_memory_per_server_mb",
        "mcp_personal_storage_quota_mb",
        "mcp_personal_max_storage_per_server_mb",
        "mcp_personal_max_room_bindings_per_server",
    )
    result = {key: limits.get(key, 0) for key in keys}
    # Existing installations only have the legacy key; it is an alias here,
    # never a global room count.
    if "mcp_personal_max_room_bindings_per_server" not in limits and "mcp_personal_max_rooms" in limits:
        result["mcp_personal_max_room_bindings_per_server"] = limits["mcp_personal_max_rooms"]
    result["mcp_personal_monthly_hours"] = limits.get("mcp_personal_monthly_hours")
    return result


async def _enforce_install_quota(
    owner_id: UUID,
    *,
    memory_mb: int,
    storage_mb: int,
    db: AsyncSession,
    exclude_item_id: UUID | None = None,
) -> None:
    """Check install/storage admission before creating or resizing a runtime."""

    personal_limits = _personal_limits(await get_effective_limits(owner_id, db))
    query = select(PersonalMcpRuntime).where(
        PersonalMcpRuntime.owner_id == owner_id,
        PersonalMcpRuntime.actual_status != "deleted",
    )
    if exclude_item_id is not None:
        query = query.where(PersonalMcpRuntime.installed_item_id != exclude_item_id)
    runtimes = list((await db.execute(query)).scalars().all())

    max_installed = personal_limits.get("mcp_personal_max_installed")
    if isinstance(max_installed, int) and len(runtimes) >= max_installed:
        record_personal_mcp_quota_denied("count")
        raise _error(
            "personal_mcp_count_limit",
            "The limit of installed personal MCPs has been reached.",
            "Delete a stopped MCP or change your plan.",
            status_code=409,
            details={"requested": 1, "available": 0},
        )
    max_memory = personal_limits.get("mcp_personal_max_memory_per_server_mb")
    if isinstance(max_memory, int) and memory_mb > max_memory:
        record_personal_mcp_quota_denied("memory")
        raise _error(
            "personal_mcp_memory_limit",
            "The requested personal MCP memory exceeds the server limit.",
            "Choose a smaller memory preset or change your plan.",
            status_code=409,
            details={"requested_mb": memory_mb, "available_mb": max_memory},
        )
    max_storage = personal_limits.get("mcp_personal_max_storage_per_server_mb")
    if isinstance(max_storage, int) and storage_mb > max_storage:
        record_personal_mcp_quota_denied("storage")
        raise _error(
            "personal_mcp_storage_limit",
            "The requested personal MCP storage exceeds the server limit.",
            "Choose a smaller storage preset or change your plan.",
            status_code=409,
            details={"requested_mb": storage_mb, "available_mb": max_storage},
        )
    storage_quota = personal_limits.get("mcp_personal_storage_quota_mb")
    allocated = sum(runtime.storage_reservation_mb for runtime in runtimes)
    if isinstance(storage_quota, int) and allocated + storage_mb > storage_quota:
        record_personal_mcp_quota_denied("storage")
        raise _error(
            "personal_mcp_storage_limit",
            "Not enough storage quota for the personal MCP.",
            "Delete or reduce the data of another MCP, or change your plan.",
            status_code=409,
            details={
                "requested_mb": storage_mb,
                "available_mb": max(0, storage_quota - allocated),
            },
        )


async def _enforce_start_quota(
    runtime: PersonalMcpRuntime,
    db: AsyncSession,
    *,
    memory_mb: int | None = None,
) -> None:
    """Reserve running count and memory budget before the lifecycle transition."""

    limits = _personal_limits(await get_effective_limits(runtime.owner_id, db))
    active_statuses = {"starting", "running", "verifying", "verified", "stopping"}
    result = await db.execute(
        select(PersonalMcpRuntime).where(
            PersonalMcpRuntime.owner_id == runtime.owner_id,
            PersonalMcpRuntime.actual_status != "deleted",
            PersonalMcpRuntime.installed_item_id != runtime.installed_item_id,
        )
    )
    active = [item for item in result.scalars().all() if item.actual_status in active_statuses]
    max_running = limits.get("mcp_personal_max_running")
    if isinstance(max_running, int) and len(active) >= max_running:
        record_personal_mcp_quota_denied("running")
        raise _error(
            "personal_mcp_running_limit",
            "The limit of simultaneously running personal MCPs has been reached.",
            "Stop another MCP or change your plan.",
            status_code=409,
            details={"requested": 1, "available": 0},
        )
    total_memory = limits.get("mcp_personal_max_total_memory_mb")
    reserved = sum(item.memory_reservation_mb for item in active)
    requested_memory = memory_mb if memory_mb is not None else runtime.memory_reservation_mb
    if isinstance(total_memory, int) and reserved + requested_memory > total_memory:
        record_personal_mcp_quota_denied("memory")
        raise _error(
            "personal_mcp_memory_limit",
            "Not enough total RAM quota to launch the personal MCP.",
            "Stop another MCP or choose a smaller memory preset.",
            status_code=409,
            details={
                "requested_mb": requested_memory,
                "available_mb": max(0, total_memory - reserved),
            },
        )
    # internal design notes stage 12.1: the machine's memory, all users together.
    try:
        async with host_capacity.host_lock():
            await host_capacity.check_personal_mcp_start(
                db, runtime.installed_item_id, requested_memory
            )
    except host_capacity.HostCapacityExceeded:
        record_personal_mcp_quota_denied("server_capacity")
        raise _error(
            host_capacity.SERVER_BUSY,
            "The server is at capacity right now: the personal MCP cannot start.",
            "Try again in a few minutes.",
            status_code=503,
        )


@router.post("/import-preview")
async def preview_personal_mcp_import(
    payload: dict[str, Any],
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id)
    try:
        parsed = parse_import_payload(payload)
    except PersonalMcpError:
        record_personal_mcp_import("rejected")
        raise
    record_personal_mcp_import("preview")
    preview_id = uuid4()
    expires_at = _now() + timedelta(minutes=10)
    db.add(
        PersonalMcpImportPreview(
            id=preview_id,
            owner_id=current_user.id,
            contract_version=CONTRACT_VERSION,
            candidates=parsed.candidates,
            rejected=parsed.rejected,
            expires_at=expires_at,
        )
    )
    await db.commit()
    return preview_response(parsed, preview_id=preview_id, expires_at=expires_at)


@router.get("/usage", response_model=PersonalMcpUsageOut)
async def personal_mcp_usage(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> PersonalMcpUsageOut:
    require_feature(current_user.id, operation="read")
    limits = await get_effective_limits(current_user.id, db)
    personal_limits = _personal_limits(limits)
    result = await db.execute(
        select(PersonalMcpRuntime).where(
            PersonalMcpRuntime.owner_id == current_user.id,
            PersonalMcpRuntime.actual_status != "deleted",
        )
    )
    runtimes = list(result.scalars().all())
    active = {"starting", "running", "verifying", "verified", "stopping"}
    memory_reserved_mb = sum(runtime.memory_reservation_mb for runtime in runtimes if runtime.actual_status in active)
    memory_used_mb = sum(runtime.memory_usage_mb for runtime in runtimes if runtime.actual_status in active)
    storage_allocated_mb = sum(runtime.storage_reservation_mb for runtime in runtimes)
    storage_used_mb = sum(runtime.storage_used_mb for runtime in runtimes)
    set_personal_mcp_usage(
        reserved_memory_mb=memory_reserved_mb,
        used_memory_mb=memory_used_mb,
        allocated_storage_mb=storage_allocated_mb,
        used_storage_mb=storage_used_mb,
    )
    return PersonalMcpUsageOut(
        limits=personal_limits,
        installed_count=len(runtimes),
        running_count=sum(runtime.actual_status in active for runtime in runtimes),
        memory_reserved_mb=memory_reserved_mb,
        memory_used_mb=memory_used_mb,
        storage_allocated_mb=storage_allocated_mb,
        storage_used_mb=storage_used_mb,
    )


@router.post("", response_model=PersonalMcpOut, status_code=status.HTTP_201_CREATED)
async def create_personal_mcp(
    body: CreatePersonalMcpRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id)
    preview, candidate = await _preview_candidate(body, current_user.id, db)
    bindings = [binding.model_dump() for binding in body.secret_bindings]
    await _validate_secret_references(candidate["env_names"], bindings, current_user.id, db)

    async with owner_lifecycle_lock(current_user.id):
        existing_result = await db.execute(
            select(PersonalMcpRuntime, UserInstalledItem)
            .join(UserInstalledItem, UserInstalledItem.id == PersonalMcpRuntime.installed_item_id)
            .where(
                PersonalMcpRuntime.owner_id == current_user.id,
                PersonalMcpRuntime.preview_id == body.preview_id,
                PersonalMcpRuntime.candidate_id == body.candidate_id,
            )
        )
        existing = existing_result.first()
        if existing is not None:
            return await _as_out(existing[0], existing[1], db)

        await require_rollout_capacity(
            current_user.id,
            launch_spec_from_candidate(candidate),
            db,
        )
        await _enforce_install_quota(
            current_user.id,
            memory_mb=body.memory_reservation_mb,
            storage_mb=body.storage_reservation_mb,
            db=db,
        )

        item = UserInstalledItem(
            user_id=current_user.id,
            type="mcp",
            is_custom=True,
            name=body.display_name or candidate["display_name"],
            isolation="stateless",
            custom_config={"mode": "managed_stdio"},
        )
        db.add(item)
        await db.flush()
        item.custom_config = {"mode": "managed_stdio", "runtime_id": str(item.id)}
        runtime = PersonalMcpRuntime(
            installed_item_id=item.id,
            owner_id=current_user.id,
            preview_id=preview.id,
            candidate_id=body.candidate_id,
            kind="managed_stdio",
            desired_status="stopped",
            actual_status="preparing",
            runtime_profile="node_npx_v1",
            launch_spec=launch_spec_from_candidate(candidate),
            memory_reservation_mb=body.memory_reservation_mb,
            storage_reservation_mb=body.storage_reservation_mb,
        )
        db.add(runtime)
        await record_event(db, runtime, step="config_accepted", message="MCP configuration accepted.", percent=10)
        for binding in bindings:
            db.add(
                SecretBinding(
                    secret_id=binding["secret_id"],
                    owner_id=current_user.id,
                    resource_type=SecretResourceType.INSTALLED_MCP,
                    resource_id=item.id,
                    purpose=SecretPurpose.MCP_ENV,
                    injection_config={"name": binding["env_name"]},
                )
            )
            _record_mcp_binding_audit(
                db,
                owner_id=current_user.id,
                item_id=item.id,
                secret_id=binding["secret_id"],
                env_name=binding["env_name"],
                revision=runtime.revision,
            )
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            existing_result = await db.execute(
                select(PersonalMcpRuntime, UserInstalledItem)
                .join(UserInstalledItem, UserInstalledItem.id == PersonalMcpRuntime.installed_item_id)
                .where(
                    PersonalMcpRuntime.owner_id == current_user.id,
                    PersonalMcpRuntime.preview_id == body.preview_id,
                    PersonalMcpRuntime.candidate_id == body.candidate_id,
                )
            )
            existing = existing_result.first()
            if existing is None:
                raise
            return await _as_out(existing[0], existing[1], db)
        await db.refresh(runtime)
        record_personal_mcp_import("created")
        return await _as_out(runtime, item, db)


@router.get("", response_model=list[PersonalMcpOut])
async def list_personal_mcps(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[dict[str, Any]]:
    require_feature(current_user.id, operation="read")
    result = await db.execute(
        select(PersonalMcpRuntime, UserInstalledItem)
        .join(UserInstalledItem, UserInstalledItem.id == PersonalMcpRuntime.installed_item_id)
        .where(
            PersonalMcpRuntime.owner_id == current_user.id,
            PersonalMcpRuntime.actual_status != "deleted",
        )
        .order_by(PersonalMcpRuntime.created_at.desc())
    )
    return [await _as_out(runtime, item, db) for runtime, item in result.all()]


@router.get("/{personal_mcp_id}", response_model=PersonalMcpOut)
async def get_personal_mcp(
    personal_mcp_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id, operation="read")
    runtime, item = await owned_runtime(personal_mcp_id, current_user.id, db)
    return await _as_out(runtime, item, db)


@router.get("/{personal_mcp_id}/events")
async def list_personal_mcp_events(
    personal_mcp_id: UUID,
    after: str | None = Query(default=None),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id, operation="read")
    runtime, _ = await owned_runtime(personal_mcp_id, current_user.id, db)
    query = select(PersonalMcpEvent).where(PersonalMcpEvent.installed_item_id == runtime.installed_item_id)
    cursor_event = None
    if after:
        try:
            cursor_id = UUID(after)
        except ValueError:
            raise _error("invalid_payload", "The event cursor has an invalid format.", "Use the cursor from the previous response.") from None
        cursor_event = await db.get(PersonalMcpEvent, cursor_id)
        if cursor_event is None or cursor_event.installed_item_id != runtime.installed_item_id:
            raise _error("invalid_payload", "The event cursor is no longer valid.", "Repeat the request without a cursor.")
        query = query.where(PersonalMcpEvent.occurred_at > cursor_event.occurred_at)
    query = query.order_by(PersonalMcpEvent.occurred_at, PersonalMcpEvent.id).limit(100)
    events = list((await db.execute(query)).scalars().all())
    return {
        "events": [
            PersonalMcpEventOut(
                event_id=event.id,
                type=event.type,
                step=event.step,
                occurred_at=event.occurred_at,
                correlation_id=event.correlation_id,
                message=event.message,
                percent=event.percent,
                safe_details=event.safe_details,
            )
            for event in events
        ],
        "next_cursor": str(events[-1].id) if events else None,
    }


async def _begin_start(runtime: PersonalMcpRuntime, db: AsyncSession) -> None:
    if runtime.actual_status in {"starting", "running", "verifying", "verified"}:
        return
    requested_memory = max(runtime.memory_reservation_mb, MIN_MANAGED_MCP_MEMORY_MB)
    requested_storage = max(runtime.storage_reservation_mb, MIN_MANAGED_MCP_STORAGE_MB)
    if requested_memory != runtime.memory_reservation_mb or requested_storage != runtime.storage_reservation_mb:
        await _enforce_install_quota(
            runtime.owner_id,
            memory_mb=requested_memory,
            storage_mb=requested_storage,
            db=db,
            exclude_item_id=runtime.installed_item_id,
        )
    await _enforce_start_quota(runtime, db, memory_mb=requested_memory)
    runtime.memory_reservation_mb = requested_memory
    runtime.storage_reservation_mb = requested_storage
    transition_runtime(runtime, "starting")
    runtime.desired_status = "running"
    runtime.correlation_id = str(uuid4())
    await record_event(db, runtime, step="runner_starting", message="Preparing the isolated runner.", percent=30)


@router.post("/{personal_mcp_id}/verify", response_model=PersonalMcpOut)
async def verify_personal_mcp(
    personal_mcp_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id)
    async with owner_lifecycle_lock(current_user.id):
        runtime, item = await owned_runtime(personal_mcp_id, current_user.id, db)
        if runtime.actual_status == "verified":
            return await _as_out(runtime, item, db)
        await _begin_start(runtime, db)
        await db.commit()
        lease_token = await _issue_runtime_secret_lease(request, runtime, current_user.id, db)
        await _complete_host_verification(
            runtime,
            item,
            db,
            owner_id=current_user.id,
            secret_lease_token=lease_token,
        )
        await db.commit()
        return await _as_out(runtime, item, db)


@router.post("/{personal_mcp_id}/start", response_model=PersonalMcpOut)
async def start_personal_mcp(
    personal_mcp_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    return await verify_personal_mcp(personal_mcp_id, request, db, current_user)


@router.post("/{personal_mcp_id}/stop", response_model=PersonalMcpOut)
async def stop_personal_mcp(
    personal_mcp_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id, operation="read")
    async with owner_lifecycle_lock(current_user.id):
        runtime, item = await owned_runtime(personal_mcp_id, current_user.id, db)
        if runtime.actual_status == "stopped":
            return await _as_out(runtime, item, db)
        if runtime.actual_status == "preparing":
            raise _error("invalid_state_transition", "The MCP is not yet registered as a launchable runtime.", "Finish preparing the configuration first.", status_code=409)
        if runtime.actual_status in {"starting", "running", "verifying", "verified", "stopping"}:
            try:
                await host_client.stop(current_user.id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
            except PersonalMcpHostError:
                await _mark_runtime_error(runtime, db, "runner_unavailable")
                await db.commit()
                return await _as_out(runtime, item, db)
        if runtime.actual_status == "error":
            transition_runtime(runtime, "stopped")
        elif runtime.actual_status == "stopping":
            transition_runtime(runtime, "stopped")
        elif runtime.actual_status not in {"stopping", "deleted"}:
            transition_runtime(runtime, "stopping")
            await record_event(db, runtime, step="runner_starting", message="Stopping the runner.", percent=80)
            if runtime.actual_status == "stopping":
                transition_runtime(runtime, "stopped")
        runtime.desired_status = "stopped"
        runtime.memory_usage_mb = 0
        await record_event(db, runtime, step="ready", message="MCP stopped; data and bindings are kept.", percent=100)
        await db.commit()
        return await _as_out(runtime, item, db)


@router.post("/{personal_mcp_id}/restart", response_model=PersonalMcpOut)
async def restart_personal_mcp(
    personal_mcp_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id)
    async with owner_lifecycle_lock(current_user.id):
        runtime, item = await owned_runtime(personal_mcp_id, current_user.id, db)
        if runtime.actual_status in {"running", "verified", "stopping"}:
            try:
                await host_client.stop(current_user.id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
            except PersonalMcpHostError:
                await _mark_runtime_error(runtime, db, "runner_unavailable")
                await db.commit()
                return await _as_out(runtime, item, db)
            if runtime.actual_status != "stopping":
                transition_runtime(runtime, "stopping")
            transition_runtime(runtime, "stopped")
        elif runtime.actual_status not in {"stopped", "error"}:
            raise _error("invalid_state_transition", "Restart is unavailable in the current MCP state.", "Wait for the operation to finish.", status_code=409)
        await _begin_start(runtime, db)
        await db.commit()
        lease_token = await _issue_runtime_secret_lease(request, runtime, current_user.id, db)
        await _complete_host_verification(
            runtime,
            item,
            db,
            owner_id=current_user.id,
            secret_lease_token=lease_token,
        )
        await db.commit()
        return await _as_out(runtime, item, db)


@router.post("/{personal_mcp_id}/update", response_model=PersonalMcpOut)
async def update_personal_mcp(
    personal_mcp_id: UUID,
    body: CreatePersonalMcpRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id)
    preview, candidate = await _preview_candidate(body, current_user.id, db)
    bindings = [binding.model_dump() for binding in body.secret_bindings]
    await _validate_secret_references(candidate["env_names"], bindings, current_user.id, db)
    async with owner_lifecycle_lock(current_user.id):
        runtime, item = await owned_runtime(personal_mcp_id, current_user.id, db)
        if runtime.actual_status not in {"verified", "stopped", "error"}:
            raise _error("invalid_state_transition", "Update is unavailable during the current operation.", "Wait for the operation to finish.", status_code=409)
        await _enforce_install_quota(
            current_user.id,
            memory_mb=body.memory_reservation_mb,
            storage_mb=body.storage_reservation_mb,
            db=db,
            exclude_item_id=personal_mcp_id,
        )
        old_state = {
            "actual_status": runtime.actual_status,
            "desired_status": runtime.desired_status,
            "revision": runtime.revision,
            "preview_id": runtime.preview_id,
            "candidate_id": runtime.candidate_id,
            "launch_spec": runtime.launch_spec,
            "resolved_artifact": runtime.resolved_artifact,
            "endpoint": runtime.endpoint,
            "tool_fingerprint": runtime.tool_fingerprint,
            "tools_snapshot": runtime.tools_snapshot,
            "memory_reservation_mb": runtime.memory_reservation_mb,
            "memory_usage_mb": runtime.memory_usage_mb,
            "storage_reservation_mb": runtime.storage_reservation_mb,
            "storage_used_mb": runtime.storage_used_mb,
            "last_error_code": runtime.last_error_code,
            "last_error_safe": runtime.last_error_safe,
            "name": item.name,
        }
        old_bindings_result = await db.execute(
            select(SecretBinding).where(
                SecretBinding.owner_id == current_user.id,
                SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
                SecretBinding.resource_id == runtime.installed_item_id,
                SecretBinding.purpose == SecretPurpose.MCP_ENV,
            )
        )
        old_bindings = [
            {
                "secret_id": binding.secret_id,
                "injection_config": dict(binding.injection_config or {}),
            }
            for binding in old_bindings_result.scalars().all()
        ]
        was_verified = old_state["actual_status"] == "verified"
        # The old revision is excluded from admission because it is replaced,
        # while the candidate's memory reservation is checked atomically
        # against every other running personal MCP.
        await _enforce_start_quota(runtime, db, memory_mb=body.memory_reservation_mb)
        if was_verified:
            try:
                await host_client.stop(current_user.id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
            except PersonalMcpHostError:
                raise _error(
                    "runner_unavailable",
                    "The current MCP revision could not be stopped safely.",
                    "Try the update again later.",
                    status_code=503,
                )
        # Keep the old verified revision in memory until the candidate has
        # completed its own start + protocol probe.  If it fails, every old
        # runtime field and binding is restored below.
        runtime.actual_status = "updating"
        runtime.updated_at = _now()
        runtime.revision = int(old_state["revision"]) + 1
        runtime.preview_id = preview.id
        runtime.candidate_id = body.candidate_id
        runtime.launch_spec = launch_spec_from_candidate(candidate)
        runtime.memory_reservation_mb = body.memory_reservation_mb
        runtime.storage_reservation_mb = body.storage_reservation_mb
        runtime.resolved_artifact = None
        runtime.tool_fingerprint = None
        runtime.tools_snapshot = None
        runtime.last_error_code = None
        runtime.last_error_safe = None
        if body.display_name:
            item.name = body.display_name
            item.updated_at = _now()
        await db.execute(
            delete(SecretBinding).where(
                SecretBinding.owner_id == current_user.id,
                SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
                SecretBinding.resource_id == runtime.installed_item_id,
                SecretBinding.purpose == SecretPurpose.MCP_ENV,
            )
        )
        for binding in bindings:
            db.add(
                SecretBinding(
                    secret_id=binding["secret_id"],
                    owner_id=current_user.id,
                    resource_type=SecretResourceType.INSTALLED_MCP,
                    resource_id=runtime.installed_item_id,
                    purpose=SecretPurpose.MCP_ENV,
                    injection_config={"name": binding["env_name"]},
                )
            )
            _record_mcp_binding_audit(
                db,
                owner_id=current_user.id,
                item_id=runtime.installed_item_id,
                secret_id=binding["secret_id"],
                env_name=binding["env_name"],
                revision=runtime.revision,
            )
        await record_event(db, runtime, step="config_accepted", message="The new MCP revision is accepted.", percent=20)
        await db.commit()
        lease_token = await _issue_runtime_secret_lease(request, runtime, current_user.id, db)
        await _complete_host_verification(
            runtime,
            item,
            db,
            owner_id=current_user.id,
            secret_lease_token=lease_token,
        )
        candidate_succeeded = runtime.actual_status == "verified"
        if not candidate_succeeded:
            await db.execute(
                delete(SecretBinding).where(
                    SecretBinding.owner_id == current_user.id,
                    SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
                    SecretBinding.resource_id == runtime.installed_item_id,
                    SecretBinding.purpose == SecretPurpose.MCP_ENV,
                )
            )
            for binding in old_bindings:
                db.add(
                    SecretBinding(
                        secret_id=binding["secret_id"],
                        owner_id=current_user.id,
                        resource_type=SecretResourceType.INSTALLED_MCP,
                        resource_id=runtime.installed_item_id,
                        purpose=SecretPurpose.MCP_ENV,
                        injection_config=binding["injection_config"],
                    )
                )
            for key, value in old_state.items():
                if key != "name":
                    setattr(runtime, key, value)
            item.name = old_state["name"]
            runtime.updated_at = _now()
            await record_event(db, runtime, step="failed", event_type="diagnostic", message="The new revision was rejected; the previous verified revision is kept.", percent=100)
            await db.commit()
            if was_verified:
                lease_token = await _issue_runtime_secret_lease(request, runtime, current_user.id, db)
                await _complete_host_verification(
                    runtime,
                    item,
                    db,
                    owner_id=current_user.id,
                    secret_lease_token=lease_token,
                )
                await db.commit()
            return await _as_out(runtime, item, db)

        if not was_verified:
            # Update may verify a stopped installation, but preserves its
            # explicit stopped desired state and storage semantics.
            try:
                await host_client.stop(current_user.id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
            except PersonalMcpHostError:
                await _mark_runtime_error(runtime, db, "runner_unavailable")
            else:
                transition_runtime(runtime, "stopping")
                transition_runtime(runtime, "stopped")
                runtime.desired_status = "stopped"
                runtime.memory_usage_mb = 0
        await db.commit()
        return await _as_out(runtime, item, db)


@router.delete("/{personal_mcp_id}", response_model=PersonalMcpOut, status_code=status.HTTP_202_ACCEPTED)
async def delete_personal_mcp(
    personal_mcp_id: UUID,
    confirm_data_delete: bool = Query(default=False),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id, operation="read")
    if not confirm_data_delete:
        raise _error("delete_confirmation_required", "Explicit confirmation is required to delete the data.", "Pass confirm_data_delete=true.", status_code=428)
    async with owner_lifecycle_lock(current_user.id):
        runtime, item = await owned_runtime(personal_mcp_id, current_user.id, db)
        if runtime.actual_status != "deleted":
            try:
                await host_client.delete(
                    current_user.id,
                    runtime.installed_item_id,
                    correlation_id=runtime.correlation_id,
                )
            except PersonalMcpHostError as exc:
                raise _error(
                    "delete_cleanup_failed",
                    "The runner or persistent data could not be deleted completely.",
                    "Try the deletion again later; the data remains saved.",
                    status_code=503,
                ) from exc
            if runtime.actual_status in {"running", "verified"}:
                transition_runtime(runtime, "stopping")
                transition_runtime(runtime, "stopped")
            if runtime.actual_status != "deleting":
                transition_runtime(runtime, "deleting")
            await _finish_runtime_delete(runtime, item, current_user.id, db)
            await record_event(db, runtime, step="failed", message="MCP deleted together with persistent data.", percent=100)
            await db.commit()
        return await _as_out(runtime, item, db)


async def _finish_runtime_delete(
    runtime: PersonalMcpRuntime, item: UserInstalledItem, owner_id: UUID, db: AsyncSession
) -> None:
    """The runner and its data are gone: mark the runtime deleted and drop
    its room and secret bindings (no commit)."""
    runtime.actual_status = "deleted"
    runtime.updated_at = _now()
    runtime.desired_status = "deleted"
    runtime.memory_usage_mb = 0
    runtime.last_error_code = None
    runtime.last_error_safe = None
    item.status = "disabled"
    await db.execute(delete(RoomMcpServer).where(RoomMcpServer.user_installed_item_id == runtime.installed_item_id))
    await db.execute(
        delete(SecretBinding).where(
            SecretBinding.owner_id == owner_id,
            SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
            SecretBinding.resource_id == runtime.installed_item_id,
            SecretBinding.purpose == SecretPurpose.MCP_ENV,
        )
    )


# ---------- Plan lifecycle (internal design notes §5.5, T11) ----------------------

_LIVE_RUNNER_STATES = frozenset({"starting", "running", "verifying", "verified", "stopping", "updating"})


async def suspend_runtime_for_plan(runtime: PersonalMcpRuntime, db: AsyncSession, message: str) -> bool:
    """Stop a runner the owner's plan no longer allows; its data and
    bindings stay. True when a runner was stopped. No commit."""
    if runtime.actual_status not in _LIVE_RUNNER_STATES and runtime.desired_status != "running":
        return False
    if runtime.actual_status in _LIVE_RUNNER_STATES:
        await host_client.stop(runtime.owner_id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
        # A system stop from any live state (the user's button refuses some).
        runtime.actual_status = "stopped"
        runtime.updated_at = _now()
    runtime.desired_status = "stopped"
    runtime.memory_usage_mb = 0
    await record_event(db, runtime, step="ready", message=message, percent=100)
    return True


async def purge_runtime_for_plan(runtime: PersonalMcpRuntime, db: AsyncSession, message: str) -> None:
    """Delete a runner and its data for good (30 days without a plan that
    allows it). No commit."""
    item = await db.get(UserInstalledItem, runtime.installed_item_id)
    await host_client.delete(runtime.owner_id, runtime.installed_item_id, correlation_id=runtime.correlation_id)
    if item is not None:
        await _finish_runtime_delete(runtime, item, runtime.owner_id, db)
    else:
        runtime.actual_status = "deleted"
        runtime.desired_status = "deleted"
        runtime.updated_at = _now()
    await record_event(db, runtime, step="failed", message=message, percent=100)


@router.get("/{personal_mcp_id}/secret-bindings", response_model=list[PersonalSecretBindingOut])
async def list_personal_mcp_secret_bindings(
    personal_mcp_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[dict[str, Any]]:
    require_feature(current_user.id, operation="read")
    runtime, _ = await owned_runtime(personal_mcp_id, current_user.id, db)
    return await _secret_bindings_out(runtime, db)


@router.put("/{personal_mcp_id}/secret-bindings", response_model=list[PersonalSecretBindingOut])
async def replace_personal_mcp_secret_bindings(
    personal_mcp_id: UUID,
    body: SecretBindingsRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[dict[str, Any]]:
    require_feature(current_user.id, operation="read")
    runtime, _ = await owned_runtime(personal_mcp_id, current_user.id, db)
    bindings = [binding.model_dump() for binding in body.bindings]
    await _validate_secret_references(runtime.launch_spec.get("env_names", []), bindings, current_user.id, db)
    await db.execute(
        delete(SecretBinding).where(
            SecretBinding.owner_id == current_user.id,
            SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
            SecretBinding.resource_id == runtime.installed_item_id,
            SecretBinding.purpose == SecretPurpose.MCP_ENV,
        )
    )
    for binding in bindings:
        db.add(
            SecretBinding(
                secret_id=binding["secret_id"],
                owner_id=current_user.id,
                resource_type=SecretResourceType.INSTALLED_MCP,
                resource_id=runtime.installed_item_id,
                purpose=SecretPurpose.MCP_ENV,
                injection_config={"name": binding["env_name"]},
            )
        )
        _record_mcp_binding_audit(
            db,
            owner_id=current_user.id,
            item_id=runtime.installed_item_id,
            secret_id=binding["secret_id"],
            env_name=binding["env_name"],
            revision=runtime.revision,
        )
    await db.commit()
    return await _secret_bindings_out(runtime, db)


def _room_binding_response(room_id: UUID, runtime: PersonalMcpRuntime) -> dict[str, Any]:
    available = runtime.actual_status == "verified"
    return {
        "room_id": room_id,
        "available": available,
        "status": "available" if available else ("stopped" if runtime.actual_status == "stopped" else "unavailable"),
    }


@router.get("/{personal_mcp_id}/room-bindings", response_model=list[RoomBindingOut])
async def list_personal_mcp_room_bindings(
    personal_mcp_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[dict[str, Any]]:
    require_feature(current_user.id, operation="read")
    runtime, _ = await owned_runtime(personal_mcp_id, current_user.id, db)
    result = await db.execute(select(RoomMcpServer).where(RoomMcpServer.user_installed_item_id == runtime.installed_item_id))
    return [_room_binding_response(binding.room_id, runtime) for binding in result.scalars().all()]


@router.post("/{personal_mcp_id}/room-bindings", response_model=RoomBindingOut, status_code=status.HTTP_201_CREATED)
async def attach_personal_mcp_to_room(
    personal_mcp_id: UUID,
    body: RoomBindingRequest,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> dict[str, Any]:
    require_feature(current_user.id, operation="read")
    runtime, _ = await owned_runtime(personal_mcp_id, current_user.id, db)
    if runtime.actual_status != "verified":
        raise _error("not_verified", "The MCP has not passed protocol verification yet.", "Start and verify the MCP before assigning it to a room.", status_code=409)
    room = await db.get(Room, body.room_id)
    if room is None or room.creator_id != current_user.id:
        raise _error("room_not_allowed", "The room is not available for this personal MCP.", "Choose a room you own.", status_code=403)
    existing = await db.get(RoomMcpServer, (body.room_id, runtime.installed_item_id))
    if existing is not None:
        return _room_binding_response(body.room_id, runtime)
    limits = await get_effective_limits(current_user.id, db)
    max_bindings = _personal_limits(limits).get("mcp_personal_max_room_bindings_per_server")
    if isinstance(max_bindings, int):
        count = await db.scalar(select(func.count(RoomMcpServer.room_id)).where(RoomMcpServer.user_installed_item_id == runtime.installed_item_id))
        if int(count or 0) >= max_bindings:
            raise _error("room_binding_limit", "The room limit for the personal MCP has been reached.", "Disconnect another room or change your plan.", status_code=409)
    db.add(RoomMcpServer(room_id=body.room_id, user_installed_item_id=runtime.installed_item_id, added_by=current_user.id))
    await db.commit()
    return _room_binding_response(body.room_id, runtime)


@router.delete("/{personal_mcp_id}/room-bindings/{room_id}", status_code=status.HTTP_204_NO_CONTENT)
async def detach_personal_mcp_from_room(
    personal_mcp_id: UUID,
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    require_feature(current_user.id, operation="read")
    runtime, _ = await owned_runtime(personal_mcp_id, current_user.id, db)
    room = await db.get(Room, room_id)
    if room is None or room.creator_id != current_user.id:
        raise _error("personal_mcp_not_found", "Room binding not found.", "Check the list of bindings.", status_code=404)
    result = await db.execute(
        delete(RoomMcpServer).where(
            RoomMcpServer.room_id == room_id,
            RoomMcpServer.user_installed_item_id == runtime.installed_item_id,
        )
    )
    if not result.rowcount:
        raise _error("personal_mcp_not_found", "Room binding not found.", "Check the list of bindings.", status_code=404)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


__all__ = ["router"]
