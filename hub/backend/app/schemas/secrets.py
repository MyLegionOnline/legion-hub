"""Public SecretStore API schemas.

There is intentionally no output field for a value, ciphertext, nonce, key,
prefix, or value length.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.secrets.crypto import MAX_SECRET_BYTES, SecretValue
from app.core.secrets.redaction import assert_no_secret_fields
from app.models.secret import SecretKind, SecretPurpose, SecretResourceType, SecretState


def _validate_secret_text(value: str) -> str:
    try:
        SecretValue(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    return value


class SecretCreate(BaseModel):
    kind: SecretKind
    label: str = Field(min_length=1, max_length=120)
    provider: str | None = Field(default=None, max_length=64)
    value: str
    expires_at: datetime | None = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        return _validate_secret_text(value)


class SecretPatch(BaseModel):
    label: str | None = Field(default=None, min_length=1, max_length=120)
    expires_at: datetime | None = None
    enabled: bool | None = None


class SecretValueReplace(BaseModel):
    value: str

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        return _validate_secret_text(value)


class SecretOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kind: SecretKind
    label: str
    provider: str | None
    configured: bool = True
    version: int
    state: SecretState = SecretState.ACTIVE
    expires_at: datetime | None
    created_at: datetime
    updated_at: datetime


class SecretBindingCreate(BaseModel):
    resource_type: SecretResourceType
    resource_id: UUID
    purpose: SecretPurpose
    injection_config: dict[str, Any] | None = None

    @field_validator("injection_config")
    @classmethod
    def validate_injection_config(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is not None:
            assert_no_secret_fields(value, path="injection_config")
        return value


class SecretBindingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    secret_id: UUID
    owner_id: UUID
    resource_type: SecretResourceType
    resource_id: UUID
    purpose: SecretPurpose
    injection_config: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class SecretReference(BaseModel):
    """Safe reference used by resource configuration in later stages."""

    secret_binding_id: UUID


class SecretLeaseIssueRequest(BaseModel):
    """Internal mcp_host request; the token is returned only to the host."""

    runtime_revision: int = Field(ge=1)
    #: internal design notes §4/Stage 0 — when the host is (re)issuing a lease for a
    #: runtime invoked from inside a specific Room, it must say which Room
    #: and environment so `ManagedMcpSecretLeaseService.issue` can fold in
    #: that Room's `room_mcp_env` bindings and enforce the protected-
    #: environment gate on them. Omitted entirely, this call behaves exactly
    #: as before (no Room merge).
    room_id: UUID | None = None
    environment: str | None = None


class SecretLeaseConsumeRequest(BaseModel):
    """One-time capability submitted by the trusted host service."""

    lease_token: str = Field(min_length=32, max_length=256)


__all__ = [
    "SecretCreate",
    "SecretPatch",
    "SecretValueReplace",
    "SecretOut",
    "SecretBindingCreate",
    "SecretBindingOut",
    "SecretReference",
    "SecretLeaseIssueRequest",
    "SecretLeaseConsumeRequest",
    "MAX_SECRET_BYTES",
]
