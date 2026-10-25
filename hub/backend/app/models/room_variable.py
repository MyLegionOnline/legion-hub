"""Plaintext, non-secret configuration variables owned by a Room.

Room variables are the counterpart to Room secrets (``user_secret`` +
``secret_binding(resource_type=room)``): their values are safe to show in the
UI, safe to put in the agent's prompt, and are always materialised into the
env of every exec.  Secrets never are — see ``internal design notes``.

``ON DELETE CASCADE`` on ``room_id`` is declared in the SQL migration
(Stage 0) as well as here: SQLAlchemy only emits ``ondelete`` on
``create_all`` for a clean DB, so already-deployed databases need the cascade
from the migration's ``CREATE TABLE``.

The ``environment`` column is populated from Stage 0 but does not affect
behaviour until Stage 12 (environments UI + gate).  ``NULL`` means "a shared
value, good for any environment"; a non-null value scopes the variable to one
environment (``^[a-z][a-z0-9_-]{0,31}$``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RoomVariable(SQLModel, table=True):
    """One non-secret configuration variable for a Room."""

    __tablename__ = "room_variable"
    __table_args__ = (
        sa.CheckConstraint(
            "name ~ '^[A-Z_][A-Z0-9_]*$'",
            name="ck_room_variable_name",
        ),
        sa.CheckConstraint(
            "char_length(value) <= 4096",
            name="ck_room_variable_value_len",
        ),
        sa.CheckConstraint(
            "environment IS NULL OR environment ~ '^[a-z][a-z0-9_-]{0,31}$'",
            name="ck_room_variable_environment",
        ),
        sa.Index("ix_room_variable_room", "room_id"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    room_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("room.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    name: str = Field(max_length=128)
    value: str = Field(sa_column=sa.Column(sa.Text, nullable=False))
    description: Optional[str] = Field(default=None, max_length=256)
    environment: Optional[str] = Field(default=None, max_length=32)
    materialize_to_env: bool = Field(
        default=True,
        sa_column=sa.Column(sa.Boolean, nullable=False, server_default=sa.text("true")),
    )
    created_by: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(sa.Uuid, nullable=True),
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
