"""Persistent metadata for external context observations.

The payload itself is deliberately kept out of ``session.context_snapshot``.
It is addressed by ``storage_key`` and can live in the configured artifact
store (filesystem for the default backend, or S3-compatible storage).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


class ContextArtifact(SQLModel, table=True):
    __tablename__ = "context_artifact"
    __table_args__ = (
        sa.Index(
            "ix_context_artifact_lookup",
            "room_id",
            "tool_name",
            "args_hash",
            "source_fingerprint",
        ),
        sa.Index("ix_context_artifact_session_created", "session_id", "created_at"),
        sa.Index("ix_context_artifact_expires", "expires_at"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: UUID = Field(foreign_key="room.id", index=True)
    task_id: Optional[UUID] = Field(default=None, foreign_key="task.id", index=True)
    session_id: UUID = Field(foreign_key="session.id", index=True)
    tool_name: str
    args_hash: str = Field(index=True, max_length=64)
    source_fingerprint: str = Field(index=True, max_length=128)
    workspace_revision: Optional[str] = Field(default=None, max_length=256)
    storage_key: str
    char_count: int = 0
    token_count: int = 0
    artifact_metadata: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=sa.Column("metadata", JSONB, nullable=False, server_default="{}"),
    )
    stale: bool = False
    pinned: bool = False
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False),
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
    )
