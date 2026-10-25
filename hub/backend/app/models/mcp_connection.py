"""Durable, value-free verification state for HTTP MCP links.

The endpoint and upstream bearer are deliberately not stored here.  The
``source_fingerprint`` lets Hub detect a changed endpoint or binding without
putting either value in the snapshot, while ``tools_snapshot`` contains only
the bounded, protocol-normalized tool schemas returned by Gateway.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

import sqlalchemy as sa
from sqlmodel import Column, Field, JSON, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class McpConnectionSnapshot(SQLModel, table=True):
    """Last safe verification result for one installed HTTP MCP item."""

    __tablename__ = "mcp_connection_snapshot"
    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('unknown', 'checking', 'verified', 'error')",
            name="ck_mcp_connection_snapshot_status",
        ),
        sa.Index("ix_mcp_connection_snapshot_owner_status", "owner_id", "status"),
        sa.Index("ix_mcp_connection_snapshot_expires", "expires_at"),
    )

    installed_item_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user_installed_item.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        )
    )
    owner_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    status: str = Field(default="unknown", max_length=16)
    tools_snapshot: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False),
    )
    tool_fingerprint: Optional[str] = Field(default=None, max_length=128)
    # Hash of the declarative endpoint/auth-binding identity.  It is never
    # exposed as a URL or secret and is used to reject stale verified rows.
    source_fingerprint: Optional[str] = Field(default=None, max_length=128)
    checked_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True, index=True),
    )
    last_error_code: Optional[str] = Field(default=None, max_length=64)
    last_error_safe: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
    )
    latency_ms: Optional[int] = Field(default=None)
    correlation_id: Optional[str] = Field(default=None, max_length=128)
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


__all__ = ["McpConnectionSnapshot"]
