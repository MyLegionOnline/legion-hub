"""Service-to-service SecretStore endpoints for the managed MCP host.

These routes are deliberately outside the user-authenticated secrets router.
They accept only the mcp_host service credential and expose a lease token or
the one-time process environment mapping; browser/JWT callers cannot use them.
"""

from __future__ import annotations

import secrets as token_secrets
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.request_id import request_id_from_request
from app.core.secret_config import configured_secret
from app.core.secrets.lease import (
    ManagedMcpSecretLeaseService,
    SecretLeaseError,
)
from app.core.secrets.room_secrets import EnvironmentLocked
from app.core.secrets.store import PostgresEncryptedSecretStore, SecretStoreError
from app.db.session import get_session
from app.schemas.secrets import SecretLeaseConsumeRequest, SecretLeaseIssueRequest


router = APIRouter(prefix="/personal-mcps")


def _require_mcp_host(authorization: str | None = Header(default=None)) -> None:
    configured = configured_secret("MCP_HOST_TOKEN") or ""
    if not configured:
        # No token configured: the endpoint is off, not open with a known password.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service token is not configured")
    expected = f"Bearer {configured}"
    if not authorization or not token_secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service token")


def _store(request: Request, db: AsyncSession) -> PostgresEncryptedSecretStore:
    keyring = getattr(request.app.state, "secret_keyring", None)
    if keyring is None:
        raise HTTPException(status_code=503, detail="Secret storage is not configured")
    return PostgresEncryptedSecretStore(db, keyring=keyring)


def _lease_error(exc: SecretLeaseError) -> HTTPException:
    message = "Secret lease is not available"
    if exc.__class__.__name__ == "SecretLeaseExpired":
        message = "Secret lease has expired"
    return HTTPException(status_code=409, detail=message)


@router.post(
    "/{owner_id}/{installed_item_id}/secret-lease",
    dependencies=[Depends(_require_mcp_host)],
)
async def issue_secret_lease(
    owner_id: UUID,
    installed_item_id: UUID,
    body: SecretLeaseIssueRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Issue a short-lived one-time lease without exposing secret values."""

    service = ManagedMcpSecretLeaseService(db, store=_store(request, db))
    try:
        lease = await service.issue(
            owner_id=owner_id,
            resource_id=installed_item_id,
            runtime_revision=body.runtime_revision,
            room_id=body.room_id,
            environment=body.environment,
            request_id=request_id_from_request(request),
            source_ip=request.client.host if request.client else None,
        )
        await db.commit()
    except EnvironmentLocked as exc:
        await db.rollback()
        raise HTTPException(
            status_code=423, detail=f"Environment '{exc}' is locked"
        ) from None
    except SecretLeaseError as exc:
        await db.rollback()
        raise _lease_error(exc) from None
    except SecretStoreError:
        await db.rollback()
        raise HTTPException(status_code=503, detail="Secret storage is not available") from None
    return {
        "lease_id": lease.lease_id,
        "lease_token": lease.lease_token,
        "resource_id": lease.resource_id,
        "runtime_revision": lease.runtime_revision,
        "env_names": list(lease.env_names),
        "expires_at": lease.expires_at,
    }


@router.post(
    "/{owner_id}/{installed_item_id}/secret-lease/consume",
    dependencies=[Depends(_require_mcp_host)],
)
async def consume_secret_lease(
    owner_id: UUID,
    installed_item_id: UUID,
    body: SecretLeaseConsumeRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Consume the lease once and return only the process env mapping."""

    service = ManagedMcpSecretLeaseService(db, store=_store(request, db))
    try:
        lease = await service.consume(
            owner_id=owner_id,
            resource_id=installed_item_id,
            lease_token=body.lease_token,
            request_id=request_id_from_request(request),
            source_ip=request.client.host if request.client else None,
        )
        # ``reveal`` is intentionally kept at this final service boundary.
        values = {name: value.reveal() for name, value in lease.values.items()}
        await db.commit()
    except SecretLeaseError as exc:
        await db.rollback()
        raise _lease_error(exc) from None
    except SecretStoreError:
        await db.rollback()
        raise HTTPException(status_code=503, detail="Secret storage is not available") from None
    return {
        "lease_id": lease.lease_id,
        "resource_id": lease.resource_id,
        "runtime_revision": lease.runtime_revision,
        "expires_at": lease.expires_at,
        "env": values,
    }


__all__ = ["router"]
