"""A user's configured LLM provider connection (Settings → Providers).

Until 2026-09-27 this list lived in the browser's localStorage, so a user who
opened the Hub on another device found Settings → Providers empty and could
not re-pick a model for an agent.  It is account data and lives here now.

The connection holds no key material: ``secret_id`` points at a
``user_secret`` row (``ON DELETE SET NULL``), and the cached ``models`` list is
the catalog the provider returned on the last validation — the agent form
draws its model picker from it without re-validating.

Several connections of one ``provider`` type are allowed (two
OpenAI-compatible gateways, each with its own key); the optional ``name``
tells them apart, and agents remember it as ``llm_config.provider_name``.
At most one connection per user is the default (partial unique index).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ProviderConnection(SQLModel, table=True):
    __tablename__ = "provider_connection"
    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('idle', 'ok', 'error')", name="ck_provider_connection_status"
        ),
        sa.Index("ix_provider_connection_user", "user_id"),
        sa.Index(
            "uq_provider_connection_default",
            "user_id",
            unique=True,
            postgresql_where=sa.text("is_default"),
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    user_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid, sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False
        )
    )
    provider: str = Field(max_length=64)
    name: Optional[str] = Field(default=None, max_length=60)
    base_url: Optional[str] = Field(default=None, max_length=512)
    secret_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(
            sa.Uuid, sa.ForeignKey("user_secret.id", ondelete="SET NULL"), nullable=True
        ),
    )
    selected_model: Optional[str] = Field(default=None, max_length=256)
    models: list[dict[str, Any]] = Field(
        default_factory=list,
        sa_column=sa.Column(JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    status: str = Field(
        default="idle",
        sa_column=sa.Column(sa.String(16), nullable=False, server_default="idle"),
    )
    error_msg: Optional[str] = Field(default=None, max_length=500)
    session_logging: bool = Field(
        default=False,
        sa_column=sa.Column(sa.Boolean, nullable=False, server_default=sa.text("false")),
    )
    is_default: bool = Field(
        default=False,
        sa_column=sa.Column(sa.Boolean, nullable=False, server_default=sa.text("false")),
    )
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
