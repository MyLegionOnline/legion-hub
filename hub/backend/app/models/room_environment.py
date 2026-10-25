"""Environment protection state and confirmation windows (Stage 12).

``internal design notes`` §2.4 / Stage 12. Environments themselves
are still not a catalog table (Stage 0 kept them free-form strings on
``room_variable.environment`` / ``secret_binding.injection_config.environment``)
— these two tables only exist to hold the two pieces of state that stage adds
on top:

* :class:`RoomEnvironmentConfig` — whether one ``(room, environment)`` pair is
  ``protected``. Absence of a row means "not protected" (the default for
  every environment that has never been marked).
* :class:`RoomEnvironmentUnlock` — a short-lived human confirmation window
  that temporarily allows :func:`app.core.secrets.room_secrets.resolve_room_secret`
  to resolve secrets for a protected environment. Rows are cheap and
  disposable (no update-in-place); an expired one is simply ignored by every
  query and never deleted, mirroring ``secret_lease``.

Neither table is consulted for ``environment IS NULL`` (the shared bucket) —
it can never be marked protected (see ``room_environment_gate.is_protected``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RoomEnvironmentConfig(SQLModel, table=True):
    """Per-(room, environment) settings. Today, just the ``protected`` flag."""

    __tablename__ = "room_environment_config"
    __table_args__ = (
        sa.UniqueConstraint("room_id", "environment", name="uq_room_environment_config"),
        sa.CheckConstraint(
            "environment ~ '^[a-z][a-z0-9_-]{0,31}$'",
            name="ck_room_environment_config_environment",
        ),
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
    environment: str = Field(max_length=32)
    protected: bool = Field(
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


class RoomEnvironmentUnlock(SQLModel, table=True):
    """One human-granted confirmation window for ``(room, environment)``.

    Deliberately append-only: a grant never mutates an existing row, so two
    concurrent confirmations cannot race each other into a shorter window,
    and the audit trail of who unlocked what stays intact.
    """

    __tablename__ = "room_environment_unlock"
    __table_args__ = (
        sa.Index(
            "ix_room_environment_unlock_scope",
            "room_id", "environment", "expires_at",
        ),
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
    environment: str = Field(max_length=32)
    granted_by: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(sa.Uuid, nullable=True),
    )
    granted_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )
    expires_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False)
    )
