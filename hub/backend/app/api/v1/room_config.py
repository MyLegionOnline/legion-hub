"""Room "Secrets & variables" tab — variables API (Stage 1).

internal design notes Stage 1.  Mounted under ``/api/v1/rooms``.
Every endpoint requires the caller to be a member of the room; there is no
finer-grained gate on variables (they are not secret).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.core import (
    egress_posture,
    room_config,
    room_env_file,
    room_env_import,
    vps_bootstrap,
    vps_deploy_key,
    vps_inventory,
)
from app.core.mcp_gateway import VpsToolError
from app.core.request_id import request_id_from_request
from app.core.secrets import room_environment_gate, room_secrets
from app.core.secrets.service import SecretService
from app.core.secrets.store import PostgresEncryptedSecretStore
from app.db.session import get_session
from app.models.room import RoomUserLink
from app.models.secret import SecretPurpose
from app.models.user import User
from app.schemas.room_config import (
    RoomEnvImportRequest,
    RoomEnvironmentOut,
    RoomEnvironmentUpdate,
    RoomSecretCreate,
    RoomSecretOut,
    RoomSecretUpdate,
    RoomVariableCreate,
    RoomVariableOut,
    RoomVariableUpdate,
    RoomVpsDeployKeyOut,
    RoomVpsDeployKeyRequest,
    RoomVpsDeployKeyRevokedOut,
    RoomVpsFingerprintDecision,
    RoomVpsPrepareReportOut,
    RoomVpsPrepareRequest,
    RoomVpsServerCreate,
    RoomVpsServerOut,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _secret_service(request: Request, db: AsyncSession) -> SecretService:
    keyring = getattr(request.app.state, "secret_keyring", None)
    if keyring is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Secret storage is not configured",
        )
    return SecretService(db, store=PostgresEncryptedSecretStore(db, keyring=keyring))


def _resolve_purpose(raw: Optional[str]) -> SecretPurpose:
    if raw in (None, "", "room_env"):
        return SecretPurpose.ROOM_ENV
    try:
        purpose = SecretPurpose(raw)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"unknown purpose {raw!r}") from None
    if purpose not in room_secrets.ROOM_SECRET_PURPOSES:
        raise HTTPException(status_code=422, detail=f"purpose {raw!r} is not a room secret")
    return purpose


def _handle_room_secret_error(exc: room_secrets.RoomSecretError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail={"error_code": exc.code, "message": str(exc)})


def _handle_room_env_file_error(exc: room_env_file.RoomEnvFileError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail={"error_code": exc.code, "message": str(exc)})


def _handle_room_env_import_error(exc: room_env_import.RoomEnvImportError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail={"error_code": exc.code, "message": str(exc)})

_UNSET = room_config._UNSET


async def _autosync_env_file(
    room_id: UUID, actor_user_id: UUID, db: AsyncSession, request: Request
) -> None:
    """Best-effort ``.env`` re-sync after a variable changes (plan §5 Stage 8).

    Only runs if a Legion-managed section already exists in ``.env`` — a room
    that never ran the explicit sync should not get one created as a side
    effect of an unrelated variable edit. Never raises: a workspaced hiccup
    here must not fail the variable write that triggered it.
    """
    from app.core import workspaced_client as wsd

    try:
        existing = await wsd.read_workspace_file(
            str(room_id), room_env_file.ENV_FILE_PATH
        )
        if not room_env_file.has_managed_block(existing):
            return
        await room_env_file.sync_room_env_file(
            db,
            room_id=room_id,
            actor_user_id=actor_user_id,
            request_id=request_id_from_request(request),
        )
        await db.commit()
    except Exception:
        await db.rollback()
        logger.warning("env-file autosync failed for room %s", room_id, exc_info=True)


async def _verify_membership(room_id: UUID, user: User, db: AsyncSession) -> None:
    from fastapi import HTTPException

    res = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == user.id,
        )
    )
    if not res.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")


@router.get("/{room_id}/variables", response_model=List[RoomVariableOut])
async def list_room_variables(
    room_id: UUID,
    environment: Optional[str] = Query(
        default=None,
        description=(
            "Filter to one environment. Absent → all variables; "
            "empty string → only the shared (NULL) scope."
        ),
    ),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    # Distinguish "?environment" absent from "?environment=" (shared scope).
    passed = environment if environment is not None else _UNSET
    return await room_config.list_variables(db, room_id, environment=passed)


@router.post(
    "/{room_id}/variables", response_model=RoomVariableOut, status_code=201
)
async def create_room_variable(
    room_id: UUID,
    body: RoomVariableCreate,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    result = await room_config.create_variable(
        db,
        room_id,
        name=body.name,
        value=body.value,
        description=body.description,
        environment=body.environment,
        materialize_to_env=body.materialize_to_env,
        created_by=current_user.id,
    )
    await _autosync_env_file(room_id, current_user.id, db, request)
    return result


@router.patch(
    "/{room_id}/variables/{variable_id}", response_model=RoomVariableOut
)
async def update_room_variable(
    room_id: UUID,
    variable_id: UUID,
    body: RoomVariableUpdate,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    variable = await room_config.get_variable(db, room_id, variable_id)
    result = await room_config.update_variable(
        db, variable, fields=body.model_dump(exclude_unset=True)
    )
    await _autosync_env_file(room_id, current_user.id, db, request)
    return result


@router.delete(
    "/{room_id}/variables/{variable_id}", status_code=204
)
async def delete_room_variable(
    room_id: UUID,
    variable_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> None:
    await _verify_membership(room_id, current_user, db)
    variable = await room_config.get_variable(db, room_id, variable_id)
    await room_config.delete_variable(db, variable)
    await _autosync_env_file(room_id, current_user.id, db, request)


# ── Environments: protection + confirmation windows (Stage 12) ─────────────


@router.get("/{room_id}/environments", response_model=List[RoomEnvironmentOut])
async def list_room_environments(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Every environment in use by a variable or a secret, plus its ``protected``
    flag and current unlock countdown — backs the tab's environment switcher.

    An environment that has never been marked shows up here (``protected:
    false``) as soon as something is scoped to it; it does not need a
    ``RoomEnvironmentConfig`` row to appear.
    """
    await _verify_membership(room_id, current_user, db)
    configs = await room_environment_gate.list_configs(db, room_id)
    var_envs = await room_config.list_used_environments(db, room_id)
    secret_envs = await room_secrets.list_used_environments(db, room_id)
    names = sorted(set(configs) | set(var_envs) | set(secret_envs))
    unlocks = await room_environment_gate.list_active_unlocks(db, room_id)
    return [
        {
            "environment": name,
            "protected": configs.get(name, False),
            "unlocked_until": unlocks.get(name),
        }
        for name in names
    ]


@router.put(
    "/{room_id}/environments/{environment}", response_model=RoomEnvironmentOut
)
async def set_room_environment_protected(
    room_id: UUID,
    environment: str,
    body: RoomEnvironmentUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Mark one environment ``protected`` (or lift the flag). Room membership only —
    same access level as variables; this toggles a safety setting, not a secret.
    """
    await _verify_membership(room_id, current_user, db)
    normalized = room_config.normalize_environment(environment)
    if normalized is None:
        raise HTTPException(
            status_code=422,
            detail="the shared (empty) environment cannot be protected",
        )
    row = await room_environment_gate.set_protected(
        db, room_id=room_id, environment=normalized, protected=body.protected
    )
    unlocked_until = await room_environment_gate.active_unlock_expiry(
        db, room_id, normalized
    )
    return {
        "environment": row.environment,
        "protected": row.protected,
        "unlocked_until": unlocked_until,
    }


@router.post(
    "/{room_id}/environments/{environment}/unlock", response_model=RoomEnvironmentOut
)
async def unlock_room_environment(
    room_id: UUID,
    environment: str,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Manually open the confirmation window from the tab (plan §2.4 Stage 12).

    The same window an ask-user card grants mid-task from ``bash`` — a human
    can also open it ahead of time here, e.g. right before starting a chain of
    deploy commands.
    """
    await _verify_membership(room_id, current_user, db)
    normalized = room_config.normalize_environment(environment)
    if normalized is None:
        raise HTTPException(
            status_code=422,
            detail="the shared (empty) environment has no lock to open",
        )
    await room_environment_gate.grant_unlock(
        db,
        room_id=room_id,
        environment=normalized,
        actor_user_id=current_user.id,
        request_id=request_id_from_request(request),
    )
    protected = await room_environment_gate.is_protected(db, room_id, normalized)
    unlocked_until = await room_environment_gate.active_unlock_expiry(
        db, room_id, normalized
    )
    return {
        "environment": normalized,
        "protected": protected,
        "unlocked_until": unlocked_until,
    }


# ── Room secrets (Stage 2) ─────────────────────────────────────────────────


def _source_ip(request: Request) -> Optional[str]:
    return request.client.host if request.client else None


@router.get("/{room_id}/secrets", response_model=List[RoomSecretOut])
async def list_room_secrets(
    room_id: UUID,
    request: Request,
    environment: Optional[str] = Query(
        default=None,
        description="Filter to one environment; absent → every room secret.",
    ),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    passed = environment if environment is not None else room_secrets._UNSET
    try:
        return await room_secrets.list_room_secrets(
            db, room_id=room_id, actor_user_id=current_user.id, environment=passed
        )
    except room_secrets.RoomSecretError as exc:
        raise _handle_room_secret_error(exc) from None


@router.post("/{room_id}/secrets", response_model=RoomSecretOut, status_code=201)
async def create_room_secret(
    room_id: UUID,
    body: RoomSecretCreate,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    service = _secret_service(request, db)
    try:
        binding = await room_secrets.bind_room_secret(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=current_user.id,
            env_name=body.env_name,
            value=body.value,
            label=body.label,
            scope=body.scope,
            environment=body.environment,
            materialize_to_env=body.materialize_to_env,
            purpose=_resolve_purpose(body.purpose),
            mcp_item_id=body.mcp_item_id,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except room_secrets.RoomSecretError as exc:
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except Exception:
        await db.rollback()
        raise
    rows = await room_secrets.list_room_secrets(
        db, room_id=room_id, actor_user_id=current_user.id
    )
    return next(r for r in rows if r["binding_id"] == binding.id)


@router.patch("/{room_id}/secrets/{binding_id}", response_model=RoomSecretOut)
async def update_room_secret(
    room_id: UUID,
    binding_id: UUID,
    body: RoomSecretUpdate,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    service = _secret_service(request, db)
    fields = body.model_dump(exclude_unset=True)
    try:
        if "value" in fields:
            await room_secrets.replace_room_secret_value(
                db,
                service=service,
                room_id=room_id,
                actor_user_id=current_user.id,
                binding_id=binding_id,
                value=fields["value"],
                request_id=request_id_from_request(request),
                source_ip=_source_ip(request),
            )
        if "scope" in fields or "materialize_to_env" in fields:
            await room_secrets.update_room_secret_config(
                db,
                service=service,
                room_id=room_id,
                actor_user_id=current_user.id,
                binding_id=binding_id,
                scope=fields.get("scope", room_secrets._UNSET),
                materialize_to_env=fields.get("materialize_to_env"),
                request_id=request_id_from_request(request),
                source_ip=_source_ip(request),
            )
        await db.commit()
    except room_secrets.RoomSecretError as exc:
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except Exception:
        await db.rollback()
        raise
    rows = await room_secrets.list_room_secrets(
        db, room_id=room_id, actor_user_id=current_user.id
    )
    match = next((r for r in rows if r["binding_id"] == binding_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail="Secret not found")
    return match


@router.delete("/{room_id}/secrets/{binding_id}", status_code=204)
async def delete_room_secret(
    room_id: UUID,
    binding_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    service = _secret_service(request, db)
    try:
        await room_secrets.unbind_room_secret(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=current_user.id,
            binding_id=binding_id,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except room_secrets.RoomSecretError as exc:
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except Exception:
        await db.rollback()
        raise
    return Response(status_code=204)


# ── `.env` sync (Stage 8) ───────────────────────────────────────────────────


@router.post("/{room_id}/env-file/sync")
async def sync_room_env_file(
    room_id: UUID,
    request: Request,
    environment: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Write current Variables + opt-in Secrets into the room's ``.env``.

    Creates the Legion-managed section on first use; every later call
    (explicit, or the auto-sync a variable edit triggers once that section
    exists) refreshes only that section, leaving the developer's own lines
    untouched (plan §3.5).
    """
    await _verify_membership(room_id, current_user, db)
    try:
        result = await room_env_file.sync_room_env_file(
            db,
            room_id=room_id,
            actor_user_id=current_user.id,
            environment=environment,
            request_id=request_id_from_request(request),
        )
        await db.commit()
    except room_env_file.RoomEnvFileError as exc:
        await db.rollback()
        raise _handle_room_env_file_error(exc) from None
    except room_secrets.RoomSecretError as exc:
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except Exception:
        await db.rollback()
        raise
    return result


# ── VPS server inventory (internal design notes Stage 3) ─────────────────────────


def _handle_vps_error(exc: vps_inventory.VpsInventoryError) -> HTTPException:
    return HTTPException(status_code=exc.http_status, detail={"error_code": exc.code, "message": str(exc)})


@router.get("/{room_id}/vps/egress-posture")
async def room_vps_egress_posture(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """What this deployment actually enforces (internal design notes §3.13, Stage 9).

    Shown in the Room's "Servers" panel, prominently once `vps_access_mode` is
    `full`: turning write access on is exactly the moment a member deserves to
    know whether the service that will hold their production server can reach
    anything else. Membership is enough — the answer is a property of the
    deployment, not of the Room, and contains no address or credential.
    """

    await _verify_membership(room_id, current_user, db)
    posture = await egress_posture.get_posture()
    body = posture.as_dict()
    # A Room member is not an operator: the declared label and the raw service
    # report are Admin Center's business (and can name internal components).
    return {
        "verdict": body["verdict"],
        "enforced": body["enforced"],
        "summary": body["summary"],
    }


@router.get("/{room_id}/vps/servers", response_model=List[RoomVpsServerOut])
async def list_vps_servers(
    room_id: UUID,
    environment: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    return await vps_inventory.list_servers(db, room_id, environment=environment)


@router.post("/{room_id}/vps/servers", response_model=RoomVpsServerOut, status_code=201)
async def add_vps_server(
    room_id: UUID,
    body: RoomVpsServerCreate,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    service = _secret_service(request, db)
    try:
        result = await vps_inventory.add_server(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=current_user.id,
            name=body.name,
            host=body.host,
            port=body.port,
            user=body.user,
            root_password=body.root_password,
            root_key=body.root_key,
            environment=body.environment,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except vps_inventory.VpsInventoryError as exc:
        await db.rollback()
        raise _handle_vps_error(exc) from None
    except room_secrets.RoomSecretError as exc:
        # A credential-bind failure inside `add_server` (e.g. root secret
        # already exists, too many secrets for this environment) is not a
        # `VpsInventoryError` — without this arm it fell through to the
        # generic 500 below instead of the same typed error shape every
        # other VPS endpoint returns.
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except Exception:
        await db.rollback()
        raise
    return result


@router.delete("/{room_id}/vps/servers/{name}", status_code=204)
async def delete_vps_server(
    room_id: UUID,
    name: str,
    request: Request,
    environment: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    """Remove a server from the Room's inventory: coordinates and every credential."""
    await _verify_membership(room_id, current_user, db)
    service = _secret_service(request, db)
    # internal design notes Stage 13: the deploy key lives on the *forge*, so
    # dropping the Room's rows would leave a live read credential for the
    # repository attached to a machine nobody is tracking any more. Revoke
    # before the inventory that records its id is gone; best-effort, because
    # a forge that is down must not make a server undeletable.
    try:
        await vps_deploy_key.revoke(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=current_user.id,
            server=name,
            remote=None,
            environment=environment,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
    except Exception:
        logger.warning(
            "delete_vps_server: revoking the deploy credential of %s failed", name, exc_info=True
        )
        await db.rollback()
    try:
        await vps_inventory.remove_server(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=current_user.id,
            name=name,
            environment=environment,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except vps_inventory.VpsInventoryError as exc:
        await db.rollback()
        raise _handle_vps_error(exc) from None
    except room_secrets.RoomSecretError as exc:
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except Exception:
        await db.rollback()
        raise
    vps_bootstrap.forget_run(room_id, name, environment)
    return Response(status_code=204)


@router.post("/{room_id}/vps/servers/{name}/forget-root", status_code=204)
async def forget_vps_server_root_credentials(
    room_id: UUID,
    name: str,
    request: Request,
    environment: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    """Delete root credentials once a server no longer needs them (plan §3.6 step 10)."""
    await _verify_membership(room_id, current_user, db)
    service = _secret_service(request, db)
    try:
        await vps_inventory.forget_root_credentials(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=current_user.id,
            name=name,
            environment=environment,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except vps_inventory.VpsInventoryError as exc:
        await db.rollback()
        raise _handle_vps_error(exc) from None
    except room_secrets.RoomSecretError as exc:
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except Exception:
        await db.rollback()
        raise
    return Response(status_code=204)


# ── server preparation wizard (internal design notes §3.6, Stage 8) ──────────────

# There is deliberately no agent-facing counterpart to these three endpoints:
# the whole point of §3.6 is that root credentials are reachable only from a
# logged-in human pressing a button, never from a tool call or a session.


@router.post("/{room_id}/vps/servers/{name}/prepare", response_model=RoomVpsPrepareReportOut, status_code=202)
async def prepare_vps_server(
    room_id: UUID,
    name: str,
    body: RoomVpsPrepareRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Start preparing a server; the report is polled from the GET below."""
    await _verify_membership(room_id, current_user, db)
    try:
        return await vps_bootstrap.start(
            db,
            room_id=room_id,
            actor_user_id=current_user.id,
            server=name,
            environment=body.environment,
            profile=vps_bootstrap.BootstrapProfile(
                docker=body.docker, nginx=body.nginx, certbot=body.certbot, git=body.git,
                disable_root_password_login=body.disable_root_password_login,
            ),
            request_id=request_id_from_request(request),
        )
    except vps_bootstrap.BootstrapBusy as exc:
        raise HTTPException(status_code=409, detail={"error_code": exc.code, "message": str(exc)}) from None
    except vps_inventory.VpsInventoryError as exc:
        raise _handle_vps_error(exc) from None


@router.get("/{room_id}/vps/servers/{name}/prepare", response_model=RoomVpsPrepareReportOut)
async def vps_prepare_report(
    room_id: UUID,
    name: str,
    environment: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    await _verify_membership(room_id, current_user, db)
    try:
        return vps_bootstrap.get_run(room_id, name, environment)
    except vps_bootstrap.BootstrapNotFound as exc:
        raise HTTPException(status_code=404, detail={"error_code": exc.code, "message": str(exc)}) from None
    except vps_inventory.VpsInventoryError as exc:
        raise _handle_vps_error(exc) from None


@router.post("/{room_id}/vps/servers/{name}/prepare/fingerprint", response_model=RoomVpsPrepareReportOut)
async def vps_prepare_confirm_fingerprint(
    room_id: UUID,
    name: str,
    body: RoomVpsFingerprintDecision,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Answer the wizard's TOFU card — the §3.5 decision, made in the wizard."""
    await _verify_membership(room_id, current_user, db)
    try:
        return await vps_bootstrap.confirm_fingerprint(
            room_id, name, environment=body.environment, accept=body.accept
        )
    except vps_bootstrap.BootstrapNotFound as exc:
        raise HTTPException(status_code=404, detail={"error_code": exc.code, "message": str(exc)}) from None
    except vps_bootstrap.BootstrapBusy as exc:
        raise HTTPException(status_code=409, detail={"error_code": exc.code, "message": str(exc)}) from None
    except vps_inventory.VpsInventoryError as exc:
        raise _handle_vps_error(exc) from None


# ── repository credential for one server (internal design notes §16b, Stage 13) ───

# Same rule as the preparation wizard above: no agent-facing counterpart. An
# agent can be told *that* a server has a repository credential (Stage 14) and
# can use it through git, but issuing one mints a credential against the
# Room's repository, so it takes a logged-in member pressing a button.


def _handle_deploy_key_error(exc: vps_deploy_key.DeployKeyError) -> HTTPException:
    status = {
        "repo_not_configured": 409,
        "repo_credential_missing": 409,
        "server_not_prepared": 409,
        "forge_host_keys_unavailable": 422,
        "deploy_verify_failed": 502,
    }.get(exc.code, 400)
    return HTTPException(status_code=status, detail={"error_code": exc.code, "message": str(exc)})


@router.post("/{room_id}/vps/servers/{name}/deploy-key", response_model=RoomVpsDeployKeyOut)
async def issue_vps_deploy_key(
    room_id: UUID,
    name: str,
    body: RoomVpsDeployKeyRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Issue (or re-issue) this server's read-only access to the Room's repository."""
    await _verify_membership(room_id, current_user, db)
    service = _secret_service(request, db)
    try:
        remote = await vps_deploy_key.Remote.for_prepared_server(
            db, service=service, room_id=room_id, actor_user_id=current_user.id,
            server=name, environment=body.environment,
            request_id=request_id_from_request(request), source_ip=_source_ip(request),
        )
        return await vps_deploy_key.issue(
            db, service=service, room_id=room_id, actor_user_id=current_user.id,
            server=name, remote=remote, environment=body.environment,
            https_token=body.https_token, request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
    except vps_deploy_key.DeployKeyError as exc:
        await db.rollback()
        raise _handle_deploy_key_error(exc) from None
    except vps_inventory.VpsInventoryError as exc:
        await db.rollback()
        raise _handle_vps_error(exc) from None
    except room_secrets.RoomSecretError as exc:
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except VpsToolError as exc:
        await db.rollback()
        raise HTTPException(status_code=502, detail={"error_code": exc.code, "message": str(exc)}) from None


@router.post("/{room_id}/vps/servers/{name}/deploy-key/verify", response_model=RoomVpsDeployKeyOut)
async def verify_vps_deploy_key(
    room_id: UUID,
    name: str,
    body: RoomVpsDeployKeyRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Re-run ``git ls-remote`` from the server — the way out of ``status=manual``."""
    await _verify_membership(room_id, current_user, db)
    service = _secret_service(request, db)
    try:
        remote = await vps_deploy_key.Remote.for_prepared_server(
            db, service=service, room_id=room_id, actor_user_id=current_user.id,
            server=name, environment=body.environment,
            request_id=request_id_from_request(request), source_ip=_source_ip(request),
        )
        return await vps_deploy_key.verify(
            db, room_id=room_id, actor_user_id=current_user.id, server=name,
            remote=remote, environment=body.environment,
        )
    except vps_deploy_key.DeployKeyError as exc:
        await db.rollback()
        raise _handle_deploy_key_error(exc) from None
    except vps_inventory.VpsInventoryError as exc:
        await db.rollback()
        raise _handle_vps_error(exc) from None
    except VpsToolError as exc:
        await db.rollback()
        raise HTTPException(status_code=502, detail={"error_code": exc.code, "message": str(exc)}) from None


@router.delete("/{room_id}/vps/servers/{name}/deploy-key", response_model=RoomVpsDeployKeyRevokedOut)
async def revoke_vps_deploy_key(
    room_id: UUID,
    name: str,
    request: Request,
    environment: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Take the credential away: the forge key first, then the files on the server.

    A server that cannot be reached is not a reason to leave the key live, so
    the forge half runs regardless and the report says which halves happened.
    """
    await _verify_membership(room_id, current_user, db)
    service = _secret_service(request, db)
    remote = None
    try:
        remote = await vps_deploy_key.Remote.for_prepared_server(
            db, service=service, room_id=room_id, actor_user_id=current_user.id,
            server=name, environment=environment,
            request_id=request_id_from_request(request), source_ip=_source_ip(request),
        )
    except (vps_inventory.VpsInventoryError, room_secrets.RoomSecretError):
        logger.info("revoke_vps_deploy_key: %s unreachable, revoking at the forge only", name)
    try:
        return await vps_deploy_key.revoke(
            db, service=service, room_id=room_id, actor_user_id=current_user.id,
            server=name, remote=remote, environment=environment,
            request_id=request_id_from_request(request), source_ip=_source_ip(request),
        )
    except vps_deploy_key.DeployKeyError as exc:
        await db.rollback()
        raise _handle_deploy_key_error(exc) from None
    except vps_inventory.VpsInventoryError as exc:
        await db.rollback()
        raise _handle_vps_error(exc) from None


# ── `.env` import (Stage 10) ────────────────────────────────────────────────


@router.post("/{room_id}/config/import")
async def import_room_env_config(
    room_id: UUID,
    body: RoomEnvImportRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Preview (``dry_run``) or write a pasted/dropped ``.env`` file.

    Everything parsed becomes a Room **secret** by default; ``body.marks``
    names the entries the caller has explicitly checked "this is a variable"
    in the preview screen (plan §3.6). Never returns a value for anything it
    treats as a secret.
    """
    await _verify_membership(room_id, current_user, db)
    service = _secret_service(request, db)
    try:
        result = await room_env_import.import_room_env(
            db,
            service=service,
            room_id=room_id,
            actor_user_id=current_user.id,
            text=body.text,
            environment=body.environment,
            marks=body.marks,
            overwrite=body.overwrite,
            dry_run=body.dry_run,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        if not body.dry_run:
            await db.commit()
            await _autosync_env_file(room_id, current_user.id, db, request)
    except room_env_import.RoomEnvImportError as exc:
        await db.rollback()
        raise _handle_room_env_import_error(exc) from None
    except room_secrets.RoomSecretError as exc:
        await db.rollback()
        raise _handle_room_secret_error(exc) from None
    except Exception:
        await db.rollback()
        raise
    return result
