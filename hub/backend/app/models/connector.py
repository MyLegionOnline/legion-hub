"""Connector identity, pairing, and per-Room admission.

An external CLI agent (OpenCode) running on a user's own
machine registers as a :class:`Connector`. Pairing follows the one-time-code
pattern already used for managed-MCP capabilities
(``core/secrets/lease.py::ManagedMcpSecretLeaseService``): only a SHA-256
hash of the pairing code / refresh token is ever persisted, never plaintext.
:class:`ConnectorRoomGrant` is the explicit, revocable admission of
one Connector into one Room — Room membership alone never grants execution
on a Connector's machine (mirrors ``SecretBinding.owner_id``'s
separation of "sees the Room" from "can act on my behalf").

Keep the raw ``CREATE TABLE`` in ``db/session.py`` in sync with the columns
below (R4 convention: SQLModel class + explicit migration DDL, not just
``create_all``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import Column, Field, JSON, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


CONNECTOR_STATUSES = ("offline", "idle", "busy")


class Connector(SQLModel, table=True):
    """One external-agent installation, owned by a Hub user."""

    __tablename__ = "connector"
    __table_args__ = (
        sa.CheckConstraint(
            "status IN ('offline', 'idle', 'busy')", name="ck_connector_status"
        ),
        # NULL pairing_code_hash/refresh_token_hash are exempt from uniqueness
        # under standard SQL NULL semantics — a connector that never paired,
        # or whose code already expired and was cleared, does not collide
        # with any other row.
        sa.UniqueConstraint(
            "pairing_code_hash", name="uq_connector_pairing_code_hash"
        ),
        sa.UniqueConstraint(
            "refresh_token_hash", name="uq_connector_refresh_token_hash"
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    owner_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid, sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    label: str = Field(default="")
    status: str = Field(default="offline")
    version: Optional[str] = Field(default=None)
    capabilities: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    last_seen_at: Optional[datetime] = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    host_fingerprint: Optional[str] = Field(default=None)
    revoked_at: Optional[datetime] = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # ── Pairing ─────────────────────────────────────────────────────────
    # A pairing code is one-time: `pairing_code_consumed_at` distinguishes a
    # repeat-exchange attempt (409) from an unknown/expired code (404/410),
    # mirroring SecretLease.consumed_at.
    pairing_code_hash: Optional[str] = Field(default=None)
    pairing_code_expires_at: Optional[datetime] = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    pairing_code_consumed_at: Optional[datetime] = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
    # Long-lived refresh token, rotated only by a fresh /connectors/pair call
    # (rotation-on-refresh is out of scope). Revocation is
    # immediate via `revoked_at`, not by clearing this hash.
    refresh_token_hash: Optional[str] = Field(default=None)


class ConnectorOutbox(SQLModel, table=True):
    """Durable queue of downstream frames for one Connector.

    ``send_to_client`` (``websockets/manager.py``) is best-effort: it writes
    into a live socket and silently drops the frame if none is open. A
    connector on someone's VPS *will* drop off the network; without a durable
    queue a lost ``dispatch`` leaves the task ``DISPATCHED`` forever and the
    watchdog reaps it as stalled. Every downstream frame is written here
    first (``seq`` is a per-connector monotonic counter), delivered
    opportunistically to a live socket, and only removed from the "must
    redeliver" set once the connector's ``ack`` arrives (``acked_at``).
    ``resume(session_id, last_seq)`` replays exactly the rows this connector
    has not yet acked for that session.
    """

    __tablename__ = "connector_outbox"
    __table_args__ = (
        sa.UniqueConstraint("connector_id", "seq", name="uq_connector_outbox_seq"),
        sa.Index(
            "ix_connector_outbox_pending", "connector_id", "acked_at", "seq"
        ),
        sa.Index(
            "ix_connector_outbox_session", "connector_id", "session_id", "seq"
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    connector_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid, sa.ForeignKey("connector.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    # Nullable: connection-level frames (e.g. a future queued `ping`) are not
    # tied to any one Room session.
    session_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(sa.Uuid, sa.ForeignKey("session.id", ondelete="CASCADE"), nullable=True),
    )
    run_id: Optional[str] = Field(default=None)
    # Monotonic per connector_id (not per session): resume(session_id, last_seq)
    # filters this stream down to one session but never renumbers it, so gaps
    # from other sessions' frames are expected and harmless.
    seq: int = Field(nullable=False)
    kind: str = Field(nullable=False)
    payload: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    acked_at: Optional[datetime] = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )


class ConnectorRoomGrant(SQLModel, table=True):
    """One admission (or its revocation) of a Connector into a Room."""

    __tablename__ = "connector_room_grant"
    __table_args__ = (
        sa.Index(
            "ix_connector_room_grant_active", "connector_id", "room_id", "revoked_at"
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    connector_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid, sa.ForeignKey("connector.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    room_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid, sa.ForeignKey("room.id", ondelete="CASCADE"), nullable=False, index=True
        )
    )
    granted_by: UUID = Field(
        sa_column=sa.Column(sa.Uuid, sa.ForeignKey("user.id"), nullable=False)
    )
    granted_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    revoked_at: Optional[datetime] = Field(
        default=None, sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True)
    )
