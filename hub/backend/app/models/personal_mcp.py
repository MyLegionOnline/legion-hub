"""Durable state for managed stdio personal MCP installations.

The existing ``UserInstalledItem.custom_config`` column is intentionally kept
for legacy Market, link and Docker-image MCPs.  Managed stdio lifecycle state
belongs in these tables so that revisions, progress and safe diagnostics do
not get mixed with user supplied JSON.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from pydantic import field_validator
from sqlmodel import Column, Field, JSON, SQLModel, UniqueConstraint


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _validate_launch_spec_value(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("launch_spec must be an object")
    forbidden = {"env", "environment", "secret", "secret_id", "secret_bindings", "token", "api_key"}
    if forbidden.intersection(value):
        raise ValueError("launch_spec cannot contain secret values or secret bindings")
    env_names = value.get("env_names", [])
    if not isinstance(env_names, list) or any(
        not isinstance(name, str) or not name.isascii() or not name.isidentifier() or name.upper() != name
        for name in env_names
    ):
        raise ValueError("launch_spec env_names must contain safe names only")
    return value


class PersonalMcpImportPreview(SQLModel, table=True):
    """Short-lived, owner-scoped normalized import candidates."""

    __tablename__ = "personal_mcp_import_preview"
    __table_args__ = (
        sa.Index("ix_personal_mcp_preview_owner_expires", "owner_id", "expires_at"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    owner_id: UUID = Field(foreign_key="user.id", index=True)
    contract_version: str = Field(default="personal-mcp.v1", max_length=32)
    candidates: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    rejected: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    expires_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, index=True)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )


class PersonalMcpRuntime(SQLModel, table=True):
    """One managed runtime revision for one ``UserInstalledItem``."""

    __tablename__ = "personal_mcp_runtime"
    __table_args__ = (
        sa.UniqueConstraint(
            "installed_item_id", name="uq_personal_mcp_runtime_installed_item"
        ),
        sa.UniqueConstraint(
            "owner_id", "candidate_id", name="uq_personal_mcp_runtime_owner_candidate"
        ),
        sa.CheckConstraint(
            "kind IN ('managed_stdio', 'personal_image')",
            name="ck_personal_mcp_runtime_kind",
        ),
        sa.CheckConstraint(
            "desired_status IN ('stopped', 'running', 'deleted')",
            name="ck_personal_mcp_runtime_desired_status",
        ),
        sa.CheckConstraint(
            "actual_status IN ('preparing', 'starting', 'running', 'verifying', 'verified', 'stopping', 'stopped', 'updating', 'error', 'deleting', 'deleted')",
            name="ck_personal_mcp_runtime_actual_status",
        ),
        sa.Index("ix_personal_mcp_runtime_owner_status", "owner_id", "actual_status"),
    )

    def __init__(self, **data: Any) -> None:
        if "launch_spec" in data:
            _validate_launch_spec_value(data["launch_spec"])
        super().__init__(**data)

    installed_item_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user_installed_item.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    owner_id: UUID = Field(foreign_key="user.id", index=True)
    preview_id: UUID = Field(index=True)
    candidate_id: UUID = Field(index=True)
    kind: str = Field(default="managed_stdio", max_length=32)
    desired_status: str = Field(default="stopped", max_length=16)
    actual_status: str = Field(default="preparing", max_length=16)
    runtime_profile: str = Field(default="node_npx_v1", max_length=64)
    runner_version: Optional[str] = Field(default=None, max_length=128)
    launch_spec: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
    )

    @field_validator("launch_spec")
    @classmethod
    def validate_launch_spec(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_launch_spec_value(value)
    resolved_artifact: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    endpoint: Optional[str] = Field(default=None, max_length=2048)
    tool_fingerprint: Optional[str] = Field(default=None, max_length=256)
    tools_snapshot: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    memory_reservation_mb: int = Field(default=0)
    memory_usage_mb: int = Field(default=0)
    storage_reservation_mb: int = Field(default=0)
    storage_used_mb: int = Field(default=0)
    last_error_code: Optional[str] = Field(default=None, max_length=96)
    last_error_safe: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    correlation_id: str = Field(default_factory=lambda: str(uuid4()), max_length=128)
    revision: int = Field(default=1)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )


class PersonalMcpEvent(SQLModel, table=True):
    """Safe, value-free progress events exposed by the polling API."""

    __tablename__ = "personal_mcp_event"
    __table_args__ = (
        sa.Index(
            "ix_personal_mcp_event_item_occurred",
            "installed_item_id",
            "occurred_at",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    installed_item_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user_installed_item.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    owner_id: UUID = Field(foreign_key="user.id", index=True)
    type: str = Field(default="progress", max_length=32)
    step: str = Field(max_length=64)
    message: str = Field(max_length=500)
    percent: Optional[int] = Field(default=None)
    safe_details: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    correlation_id: str = Field(max_length=128)
    occurred_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )


__all__ = [
    "PersonalMcpImportPreview",
    "PersonalMcpRuntime",
    "PersonalMcpEvent",
]
