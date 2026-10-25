"""Durable state for internal design notes's provider-login flow.

Nothing persisted here is a secret. A DCR ``client_id`` is not confidential
(RFC 7591 public clients have none of `client_secret`), and a PKCE
``code_verifier``/``state`` are one-shot values gone within ten minutes of
being minted. The access and refresh tokens a finished login produces live in
Vault instead, on the existing ``SecretKind.MCP_HEADER`` path (plan §3).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import Column, Field, JSON, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _flow_expiry() -> datetime:
    return _utcnow() + timedelta(minutes=10)


class McpOAuthClient(SQLModel, table=True):
    """The Hub's own client registration at one provider, reused across logins.

    Unique on (issuer, redirect_uri): the same Hub origin logging into the
    same provider a second time must find this row and skip registration
    (internal design notes Stage 1 acceptance), not create a duplicate client.
    """

    __tablename__ = "mcp_oauth_client"
    __table_args__ = (
        sa.UniqueConstraint(
            "issuer", "redirect_uri", name="uq_mcp_oauth_client_issuer_redirect"
        ),
        sa.CheckConstraint(
            "registration_source IN ('dcr', 'manual')",
            name="ck_mcp_oauth_client_registration_source",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    issuer: str = Field(max_length=512, index=True)
    redirect_uri: str = Field(max_length=512)
    client_id: str = Field(max_length=512)
    # Manual registration only (Stage 8) — a DCR public client has none.
    client_secret_binding_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("secret_binding.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    registration_source: str = Field(default="dcr", max_length=16)
    # A trimmed cache of the authorization-server metadata this registration
    # was made against — never the provider's raw registration response,
    # which is where a client_secret would appear for a confidential client.
    metadata_json: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    registered_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )


class McpOAuthFlow(SQLModel, table=True):
    """One not-yet-finished authorization-code login (Stage 3 start/callback).

    Gone as soon as it is used: the callback consumes and deletes/marks the
    row, and ``code_verifier`` never survives past the token exchange.
    """

    __tablename__ = "mcp_oauth_flow"
    __table_args__ = (
        sa.Index("ix_mcp_oauth_flow_expires", "expires_at"),
        sa.Index("ix_mcp_oauth_flow_item", "installed_item_id"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    state: str = Field(max_length=128, unique=True, index=True)
    code_verifier: str = Field(max_length=256)
    installed_item_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user_installed_item.id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    user_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid, sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False
        )
    )
    redirect_uri: str = Field(max_length=512)
    issuer: str = Field(max_length=512)
    client_id: str = Field(max_length=512)
    resource: Optional[str] = Field(default=None, max_length=512)
    scope: Optional[str] = Field(default=None, max_length=512)
    expires_at: datetime = Field(
        default_factory=_flow_expiry,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, index=True),
    )
    consumed_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
    )
    # Stage 3 — set once the callback has run, so GET .../oauth/status can
    # answer without re-deriving anything from the (already-gone) code.
    outcome: Optional[str] = Field(
        default=None,
        sa_column=sa.Column(sa.String(16), nullable=True),
    )
    error_code: Optional[str] = Field(default=None, max_length=64)
    created_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )


__all__ = ["McpOAuthClient", "McpOAuthFlow"]
