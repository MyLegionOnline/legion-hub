"""Authenticated API for user-owned encrypted secrets."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.core.request_id import request_id_from_request
from app.core.secrets.crypto import SecretValue
from app.core.secrets.service import SecretResourceAccessDenied, SecretService
from app.core.secrets.store import (
    BindingTarget,
    PostgresEncryptedSecretStore,
    SecretAccessDenied,
    SecretNotFound,
    SecretStoreError,
    SecretUnavailable,
    _UNSET,
)
from app.db.session import get_session
from app.models.user import User
from app.schemas.secrets import (
    SecretBindingCreate,
    SecretBindingOut,
    SecretCreate,
    SecretOut,
    SecretPatch,
    SecretValueReplace,
)
from app.models.secret import SecretBinding, SecretPurpose, SecretResourceType, UserSecret

router = APIRouter()


def _source_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


def _store(db: AsyncSession, request: Request) -> PostgresEncryptedSecretStore:
    keyring = getattr(request.app.state, "secret_keyring", None)
    # The runtime keyring is loaded once by startup.  Do not silently reload
    # it per request or fall back to an environment/default token.
    if keyring is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Secret storage is not configured",
        )
    return PostgresEncryptedSecretStore(db, keyring=keyring)


def _out(meta) -> SecretOut:
    return SecretOut(
        id=meta.id,
        kind=meta.kind,
        label=meta.label,
        provider=meta.provider,
        configured=meta.configured,
        version=meta.version,
        state=meta.state,
        expires_at=meta.expires_at,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
    )


def _binding_out(meta) -> SecretBindingOut:
    return SecretBindingOut(
        id=meta.id,
        secret_id=meta.secret_id,
        owner_id=meta.owner_id,
        resource_type=meta.resource_type,
        resource_id=meta.resource_id,
        purpose=meta.purpose,
        injection_config=meta.injection_config,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
    )


async def _invalidate_mcp_snapshots_for_secret(
    db: AsyncSession,
    secret_id: UUID,
    *,
    owner_id: UUID,
) -> None:
    """Keep link MCP tools hidden while a bound Secret is changing."""

    from app.core.mcp_discovery import invalidate_mcp_snapshots_for_secret

    await invalidate_mcp_snapshots_for_secret(db, secret_id, owner_id=owner_id)


def _handle_store_error(exc: Exception) -> HTTPException:
    if isinstance(exc, SecretNotFound):
        return HTTPException(status_code=404, detail="Secret not found")
    if isinstance(exc, (SecretAccessDenied, SecretResourceAccessDenied)):
        return HTTPException(status_code=403, detail="Secret operation is not permitted")
    if isinstance(exc, SecretUnavailable):
        return HTTPException(status_code=409, detail="Secret is not available")
    if isinstance(exc, SecretStoreError):
        return HTTPException(status_code=422, detail="Secret operation failed")
    return HTTPException(status_code=500, detail="Secret operation failed")


@router.post("", response_model=SecretOut, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=SecretOut, status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def create_secret(
    body: SecretCreate,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> SecretOut:
    if body.provider == "openai_chatgpt":
        raise HTTPException(422, detail="Connect ChatGPT from Settings → Providers")
    store = _store(db, request)
    try:
        meta = await store.create(
            owner_id=current_user.id,
            kind=body.kind,
            label=body.label,
            provider=body.provider,
            value=SecretValue(body.value),
            expires_at=body.expires_at,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise _handle_store_error(exc) from None
    return _out(meta)


@router.get("", response_model=list[SecretOut])
@router.get("/", response_model=list[SecretOut], include_in_schema=False)
async def list_secrets(
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[SecretOut]:
    store = _store(db, request)
    return [_out(meta) for meta in await store.list_metadata(owner_id=current_user.id)]


@router.get("/bindings", response_model=list[SecretBindingOut])
async def list_bindings_for_resource(
    resource_type: SecretResourceType,
    resource_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> list[SecretBindingOut]:
    """Return safe binding metadata for one owned resource."""

    try:
        authorization_purpose = {
            SecretResourceType.AGENT: SecretPurpose.LLM_PRIMARY,
            SecretResourceType.ROOM: SecretPurpose.GIT_HTTPS,
            SecretResourceType.INSTALLED_MCP: SecretPurpose.MCP_AUTH_HEADER,
        }[resource_type]
        await SecretService(db).authorize_target(
            current_user.id,
            BindingTarget(resource_type=resource_type, resource_id=resource_id, purpose=authorization_purpose),
        )
    except (SecretResourceAccessDenied, SecretAccessDenied):
        # Keep resource existence and membership opaque to callers.
        raise HTTPException(status_code=404, detail="Resource not found") from None

    result = await db.execute(
        select(SecretBinding).where(
            SecretBinding.owner_id == current_user.id,
            SecretBinding.resource_type == resource_type,
            SecretBinding.resource_id == resource_id,
        )
    )
    return [SecretBindingOut.model_validate(row) for row in result.scalars().all()]


@router.patch("/{secret_id}", response_model=SecretOut)
async def patch_secret(
    secret_id: UUID,
    body: SecretPatch,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> SecretOut:
    store = _store(db, request)
    try:
        await _invalidate_mcp_snapshots_for_secret(db, secret_id, owner_id=current_user.id)
        meta = await store.update_metadata(
            owner_id=current_user.id,
            secret_id=secret_id,
            label=body.label,
            expires_at=body.expires_at if "expires_at" in body.model_fields_set else _UNSET,
            enabled=body.enabled,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise _handle_store_error(exc) from None
    return _out(meta)


@router.put("/{secret_id}/value", response_model=SecretOut)
async def replace_secret_value(
    secret_id: UUID,
    body: SecretValueReplace,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> SecretOut:
    secret = await db.get(UserSecret, secret_id)
    if secret and secret.owner_id == current_user.id and secret.provider == "openai_chatgpt":
        raise HTTPException(422, detail="Renew ChatGPT authorization from Settings → Providers")
    store = _store(db, request)
    try:
        await _invalidate_mcp_snapshots_for_secret(db, secret_id, owner_id=current_user.id)
        meta = await store.replace_value(
            owner_id=current_user.id,
            secret_id=secret_id,
            value=SecretValue(body.value),
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise _handle_store_error(exc) from None
    return _out(meta)


@router.delete("/{secret_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_secret(
    secret_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    store = _store(db, request)
    remote_confirmed = True
    try:
        secret = (await db.execute(select(UserSecret).where(
            UserSecret.id == secret_id, UserSecret.owner_id == current_user.id).with_for_update())).scalar_one_or_none()
        if secret and secret.provider == "openai_chatgpt" and secret.state != "deleted":
            import json
            from app.core import chatgpt_oauth
            from app.api.v1.chatgpt_auth import registration
            bundle = registration(db, secret)
            if bundle.get("refresh_token"):
                remote_confirmed = await chatgpt_oauth.revoke(bundle)
            await store.replace_value(owner_id=current_user.id, secret_id=secret_id,
                                      value=SecretValue(json.dumps(chatgpt_oauth.registration_mapping(bundle))))
        await _invalidate_mcp_snapshots_for_secret(db, secret_id, owner_id=current_user.id)
        await store.revoke_or_delete(
            owner_id=current_user.id,
            secret_id=secret_id,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise _handle_store_error(exc) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT,
                    headers={"X-Legion-Remote-Revocation-Confirmed": str(remote_confirmed).lower()})


@router.post("/{secret_id}/bindings", response_model=SecretBindingOut, status_code=status.HTTP_201_CREATED)
async def bind_secret(
    secret_id: UUID,
    body: SecretBindingCreate,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> SecretBindingOut:
    store = _store(db, request)
    service = SecretService(db, store=store)
    target = BindingTarget(
        resource_type=body.resource_type,
        resource_id=body.resource_id,
        purpose=body.purpose,
        injection_config=body.injection_config,
    )
    try:
        meta = await service.bind(
            owner_id=current_user.id,
            secret_id=secret_id,
            target=target,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        if (
            target.resource_type == SecretResourceType.INSTALLED_MCP
            and target.purpose == SecretPurpose.MCP_AUTH_HEADER
        ):
            from app.core.mcp_discovery import invalidate_mcp_snapshot

            await invalidate_mcp_snapshot(db, target.resource_id, owner_id=current_user.id)
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise _handle_store_error(exc) from None
    return _binding_out(meta)


@router.delete("/{secret_id}/bindings/{binding_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unbind_secret(
    secret_id: UUID,
    binding_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Response:
    store = _store(db, request)
    try:
        bindings = await store.list_bindings(owner_id=current_user.id, secret_id=secret_id)
        if not any(binding.id == binding_id for binding in bindings):
            raise SecretNotFound("binding not found")
        binding = next(binding for binding in bindings if binding.id == binding_id)
        if (
            binding.resource_type == SecretResourceType.INSTALLED_MCP
            and binding.purpose == SecretPurpose.MCP_AUTH_HEADER
        ):
            await _invalidate_mcp_snapshots_for_secret(db, secret_id, owner_id=current_user.id)
        await store.unbind(
            owner_id=current_user.id,
            binding_id=binding_id,
            request_id=request_id_from_request(request),
            source_ip=_source_ip(request),
        )
        await db.commit()
    except Exception as exc:
        await db.rollback()
        raise _handle_store_error(exc) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)
