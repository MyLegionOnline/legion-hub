"""Database records for the Hub-owned encrypted secret store.

The database only stores authenticated ciphertext.  The keyring and all
encryption/decryption operations live in ``app.core.secrets``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import Column, Field, JSON, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SecretKind(str, Enum):
    LLM_API_KEY = "llm_api_key"
    GIT_ACCESS_TOKEN = "git_access_token"
    GIT_WEBHOOK_SECRET = "git_webhook_secret"
    MCP_HEADER = "mcp_header"
    MCP_ENV = "mcp_env"
    PAYMENT_PROVIDER_TOKEN = "payment_provider_token"
    CUSTOM = "custom"
    # internal design notes Stage 0
    SSH_PRIVATE_KEY = "ssh_private_key"
    ROOM_ENV = "room_env"


class SecretState(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    DELETED = "deleted"


class SecretResourceType(str, Enum):
    AGENT = "agent"
    ROOM = "room"
    INSTALLED_MCP = "installed_mcp"


class SecretPurpose(str, Enum):
    LLM_PRIMARY = "llm_primary"
    LLM_MEDIA = "llm_media"
    LLM_VISION = "llm_vision"
    GIT_HTTPS = "git_https"
    GIT_WEBHOOK = "git_webhook"
    MCP_AUTH_HEADER = "mcp_auth_header"
    MCP_ENV = "mcp_env"
    # internal design notes Stage 3/§3 — refresh token for an OAuth-authorized MCP.
    # Same SecretKind.MCP_HEADER as MCP_AUTH_HEADER; never resolved by
    # resolve_mcp_upstream_auth — only Stage 4's refresh path reads it.
    MCP_OAUTH_REFRESH = "mcp_oauth_refresh"
    # internal design notes Stage 8 — a manually registered OAuth client's
    # confidential secret (RFC 7591 clients created via DCR have none; this
    # purpose only ever appears for registration_source="manual"). Resolved
    # only from app.core.mcp_oauth, at authorization-code exchange and
    # refresh time — never by resolve_mcp_upstream_auth.
    MCP_OAUTH_CLIENT_SECRET = "mcp_oauth_client_secret"
    # internal design notes Stage 0
    ROOM_ENV = "room_env"
    DEPLOY_SSH_KEY = "deploy_ssh_key"
    ROOM_MCP_ENV = "room_mcp_env"
    ROOM_MCP_AUTH_HEADER = "room_mcp_auth_header"
    # internal design notes Stage 3 — server root/everyday credentials. Never
    # resolved by resolve_room_secrets_mixed or the .env sync; only
    # core/vps_inventory.py's connection resolver reads it.
    VPS_ACCESS = "vps_access"
    # internal design notes Stage 13 — the repository credential one prepared server
    # gets: a read-only HTTPS token a person pasted for a forge where the SSH
    # deploy-key path is not available. Like ``VPS_ACCESS`` it is resolved from
    # exactly one module (``core/vps_deploy_key.py``) and never by
    # ``resolve_room_secrets_mixed`` or the ``.env`` sync — otherwise an agent
    # could pull it out with ``bash(secrets=…)`` and carry it off the machine.
    GIT_DEPLOY = "git_deploy"
    # internal design notes Stage 1 — the Router agent's OpenRouter key for Jev
    # (Decisions API). Separate from LLM_PRIMARY because the Router's
    # escalation LLM may live at a different provider.
    LLM_ROUTER_JEV = "llm_router_jev"


def secret_binding_purpose_check() -> str:
    """The ck_secret_binding_purpose expression, generated from SecretPurpose.

    Used by the model (fresh databases) and by init_db's single rebuild
    (legacy databases) so a new purpose needs no hand-edited value list.
    """
    values = ", ".join(f"'{p.value}'" for p in SecretPurpose)
    return f"purpose IN ({values})"


class SecretActorType(str, Enum):
    USER = "user"
    SYSTEM = "system"
    MIGRATION = "migration"


class SecretAuditAction(str, Enum):
    CREATE = "create"
    UPDATE_VALUE = "update_value"
    BIND = "bind"
    UNBIND = "unbind"
    RESOLVE = "resolve"
    DECRYPT_FAILED = "decrypt_failed"
    DISABLE = "disable"
    DELETE = "delete"
    MIGRATE = "migrate"
    ROTATE = "rotate"
    ISSUE_LEASE = "issue_lease"
    CONSUME_LEASE = "consume_lease"
    REVOKE_LEASE = "revoke_lease"
    # internal design notes Stage 12 — human confirmation window for a
    # protected environment. Not a secret resolve; recorded so "who unlocked
    # production and when" is auditable the same way as any other secret op.
    UNLOCK_ENVIRONMENT = "unlock_environment"


class SecretAuditOutcome(str, Enum):
    SUCCESS = "success"
    DENIED = "denied"
    FAILED = "failed"


class UserSecret(SQLModel, table=True):
    """Metadata and ciphertext for one user-owned secret."""

    __tablename__ = "user_secret"
    __table_args__ = (
        sa.CheckConstraint("char_length(label) BETWEEN 1 AND 120", name="ck_user_secret_label"),
        sa.CheckConstraint("char_length(key_id) BETWEEN 1 AND 64", name="ck_user_secret_key_id"),
        sa.CheckConstraint("aad_version > 0", name="ck_user_secret_aad_version"),
        sa.CheckConstraint("version > 0", name="ck_user_secret_version"),
        sa.CheckConstraint(
            "kind IN ('llm_api_key', 'git_access_token', 'git_webhook_secret', 'mcp_header', 'mcp_env', 'payment_provider_token', 'custom', 'ssh_private_key', 'room_env')",
            name="ck_user_secret_kind",
        ),
        sa.CheckConstraint("state IN ('active', 'disabled', 'deleted')", name="ck_user_secret_state"),
        sa.Index("ix_user_secret_owner_state", "owner_id", "state"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    owner_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    kind: SecretKind = Field(sa_column=sa.Column(sa.String(64), nullable=False))
    label: str = Field(max_length=120)
    provider: Optional[str] = Field(default=None, max_length=64)
    ciphertext: bytes = Field(sa_column=sa.Column(sa.LargeBinary, nullable=False))
    key_id: str = Field(max_length=64)
    aad_version: int = Field(default=1)
    version: int = Field(default=1)
    state: SecretState = Field(
        default=SecretState.ACTIVE,
        sa_column=sa.Column(sa.String(16), nullable=False, index=True),
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
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
    deleted_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
    )


class SecretBinding(SQLModel, table=True):
    """A narrowly scoped use of a secret by one Legion resource."""

    __tablename__ = "secret_binding"
    __table_args__ = (
        sa.UniqueConstraint(
            "secret_id",
            "resource_type",
            "resource_id",
            "purpose",
            name="uq_secret_binding_target",
        ),
        sa.Index("ix_secret_binding_resource_purpose", "resource_type", "resource_id", "purpose"),
        sa.Index("ix_secret_binding_owner", "owner_id"),
        sa.CheckConstraint(
            "resource_type IN ('agent', 'room', 'installed_mcp')",
            name="ck_secret_binding_resource_type",
        ),
        sa.CheckConstraint(
            secret_binding_purpose_check(),
            name="ck_secret_binding_purpose",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    secret_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user_secret.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
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
    resource_type: SecretResourceType = Field(
        sa_column=sa.Column(sa.String(32), nullable=False)
    )
    resource_id: UUID = Field(sa_column=sa.Column(sa.Uuid, nullable=False))
    purpose: SecretPurpose = Field(sa_column=sa.Column(sa.String(32), nullable=False))
    injection_config: Optional[dict[str, Any]] = Field(
        default=None,
        sa_column=Column(JSON, nullable=True),
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


class SecretAudit(SQLModel, table=True):
    """Minimal, value-free audit trail for secret operations."""

    __tablename__ = "secret_audit"
    __table_args__ = (
        sa.Index("ix_secret_audit_occurred", "occurred_at"),
        sa.Index("ix_secret_audit_secret_occurred", "secret_id", "occurred_at"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    occurred_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )
    actor_type: SecretActorType = Field(sa_column=sa.Column(sa.String(16), nullable=False))
    actor_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(sa.Uuid, nullable=True),
    )
    action: SecretAuditAction = Field(sa_column=sa.Column(sa.String(32), nullable=False))
    secret_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(sa.Uuid, nullable=True),
    )
    resource_type: Optional[SecretResourceType] = Field(
        default=None,
        sa_column=sa.Column(sa.String(32), nullable=True),
    )
    resource_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(sa.Uuid, nullable=True),
    )
    purpose: Optional[SecretPurpose] = Field(
        default=None,
        sa_column=sa.Column(sa.String(32), nullable=True),
    )
    request_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(sa.Uuid, nullable=True),
    )
    outcome: SecretAuditOutcome = Field(sa_column=sa.Column(sa.String(16), nullable=False))
    error_code: Optional[str] = Field(default=None, max_length=64)
    source_ip: Optional[str] = Field(default=None, max_length=45)
    # Managed-stdio bindings are audited as metadata only.  Neither the
    # secret value nor a lease token is ever stored here.
    env_name: Optional[str] = Field(default=None, max_length=128)
    runtime_revision: Optional[int] = Field(default=None)


class SecretLease(SQLModel, table=True):
    """One-time, item-scoped capability for the managed stdio host.

    ``token_hash`` is the only persisted representation of the capability.
    The selected binding ids and env names are safe metadata; values are
    decrypted only while consuming the lease and are never written to this
    table or to a runtime profile.
    """

    __tablename__ = "secret_lease"
    __table_args__ = (
        sa.UniqueConstraint("token_hash", name="uq_secret_lease_token_hash"),
        sa.Index("ix_secret_lease_scope", "owner_id", "resource_id", "expires_at"),
        # internal design notes Stage 7 — a lease keeps the item owner's
        # own ``mcp_env`` bindings under ``installed_mcp`` (unchanged); ``room``
        # is reserved for a future lease issued directly against a Room scope.
        # Today's Stage 7 merge (`lease.py::issue(room_id=...)`) still stores
        # ``installed_mcp`` even when it folds in ``(room, room_mcp_env)``
        # bindings, so `consume_secret_lease`'s existing (owner_id,
        # installed_item_id) lookup keeps working unchanged.
        sa.CheckConstraint(
            "resource_type IN ('installed_mcp', 'room')",
            name="ck_secret_lease_resource_type",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    owner_id: UUID = Field(
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )
    )
    resource_type: SecretResourceType = Field(
        default=SecretResourceType.INSTALLED_MCP,
        sa_column=sa.Column(sa.String(32), nullable=False),
    )
    resource_id: UUID = Field(sa_column=sa.Column(sa.Uuid, nullable=False, index=True))
    runtime_revision: int = Field(default=1)
    binding_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    env_names: list[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    token_hash: str = Field(max_length=64, index=True)
    issued_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )
    expires_at: datetime = Field(
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=False, index=True)
    )
    consumed_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
    )
    revoked_at: Optional[datetime] = Field(
        default=None,
        sa_column=sa.Column(sa.DateTime(timezone=True), nullable=True),
    )


class SecretMigrationCheckpoint(SQLModel, table=True):
    """Idempotency ledger for one legacy secret field migration.

    ``source_hash`` is a SHA-256 fingerprint, never the legacy value itself.
    It lets a resumed migration distinguish the same source value from a later
    manual change without retaining plaintext in the database or audit log.
    """

    __tablename__ = "secret_migration_checkpoint"
    __table_args__ = (
        sa.UniqueConstraint(
            "source_table",
            "resource_id",
            "json_path",
            "source_hash",
            name="uq_secret_migration_checkpoint_source",
        ),
        sa.Index(
            "ix_secret_migration_checkpoint_resource",
            "source_table",
            "resource_id",
            "json_path",
        ),
        sa.CheckConstraint(
            "state IN ('completed', 'failed', 'skipped')",
            name="ck_secret_migration_checkpoint_state",
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    source_table: str = Field(max_length=64)
    resource_id: UUID = Field(sa_column=sa.Column(sa.Uuid, nullable=False))
    json_path: str = Field(max_length=256)
    source_hash: str = Field(max_length=64)
    state: str = Field(default="completed", max_length=16)
    secret_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("user_secret.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    binding_id: Optional[UUID] = Field(
        default=None,
        sa_column=sa.Column(
            sa.Uuid,
            sa.ForeignKey("secret_binding.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    error_code: Optional[str] = Field(default=None, max_length=64)
    processed_at: datetime = Field(
        default_factory=_utcnow,
        sa_column=sa.Column(
            sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")
        ),
    )
