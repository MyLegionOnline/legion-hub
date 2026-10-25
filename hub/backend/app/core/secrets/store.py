"""Owner-aware encrypted SecretStore implementation backed by PostgreSQL."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.secret_keyring import SecretKeyring, SecretKeyringError, load_keyring
from app.models.secret import (
    SecretActorType,
    SecretAudit,
    SecretAuditAction,
    SecretAuditOutcome,
    SecretBinding,
    SecretKind,
    SecretPurpose,
    SecretResourceType,
    SecretState,
    UserSecret,
)

from .crypto import SecretDecryptionError, SecretValue, decrypt_secret, encrypt_secret

logger = logging.getLogger(__name__)
_UNSET = object()


class SecretStoreError(RuntimeError):
    """Base class for errors safe to map at the API boundary."""


class SecretStoreConfigurationError(SecretStoreError):
    pass


class SecretNotFound(SecretStoreError):
    """The caller cannot distinguish missing from another user's record."""


class SecretAccessDenied(SecretStoreError):
    pass


class SecretUnavailable(SecretStoreError):
    pass


@dataclass(frozen=True)
class BindingTarget:
    resource_type: SecretResourceType | str
    resource_id: UUID
    purpose: SecretPurpose | str
    injection_config: dict[str, Any] | None = None
    expected_kind: SecretKind | str | None = None
    binding_id: UUID | None = None
    runtime_revision: int | None = None

    def normalized(self) -> "BindingTarget":
        try:
            resource_type = SecretResourceType(self.resource_type)
            purpose = SecretPurpose(self.purpose)
        except ValueError as exc:
            raise SecretAccessDenied("unsupported secret binding target") from exc
        if self.expected_kind is not None:
            try:
                expected_kind = SecretKind(self.expected_kind)
            except (TypeError, ValueError) as exc:
                raise SecretAccessDenied("unsupported secret kind") from exc
        else:
            expected_kind = None
        config = dict(self.injection_config) if self.injection_config is not None else None
        return BindingTarget(
            resource_type,
            self.resource_id,
            purpose,
            config,
            expected_kind,
            self.binding_id,
            self.runtime_revision,
        )


@dataclass(frozen=True)
class SecretMeta:
    id: UUID
    owner_id: UUID
    kind: SecretKind
    label: str
    provider: str | None
    configured: bool
    version: int
    expires_at: datetime | None
    state: SecretState
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class BindingMeta:
    id: UUID
    secret_id: UUID
    owner_id: UUID
    resource_type: SecretResourceType
    resource_id: UUID
    purpose: SecretPurpose
    injection_config: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class SecretStore(Protocol):
    async def create(
        self,
        *,
        owner_id: UUID,
        kind: SecretKind,
        label: str,
        provider: str | None,
        value: SecretValue,
        expires_at: datetime | None = None,
        request_id: UUID | None = None,
        source_ip: str | None = None,
        actor_type: SecretActorType = SecretActorType.USER,
    ) -> SecretMeta: ...

    async def replace_value(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        value: SecretValue,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> SecretMeta: ...

    async def bind(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        target: BindingTarget,
        request_id: UUID | None = None,
        source_ip: str | None = None,
        actor_type: SecretActorType = SecretActorType.USER,
    ) -> BindingMeta: ...

    async def resolve_for_use(
        self,
        *,
        owner_id: UUID,
        target: BindingTarget,
        request_id: UUID,
        source_ip: str | None = None,
    ) -> SecretValue: ...

    async def revoke_or_delete(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> None: ...


# Purposes where a resource has exactly one secret in the slot, so binding a new
# one replaces the previous binding instead of adding to it.  ``mcp_env`` and
# ``mcp_auth_header`` are deliberately absent: an MCP server can legitimately
# draw several secrets at once.
_SINGLE_SECRET_PURPOSES = frozenset({
    SecretPurpose.LLM_PRIMARY,
    SecretPurpose.LLM_MEDIA,
    SecretPurpose.LLM_VISION,
    SecretPurpose.LLM_ROUTER_JEV,
})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_ROOM_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_ROOM_ENVIRONMENT_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
# internal design notes §2.2 — where a Room secret may be injected.
ROOM_SECRET_SCOPES = frozenset({"deploy", "services", "checks", "mcp", "agent"})


def _validate_room_env_config(config: Any) -> dict[str, Any]:
    """Validate the ``injection_config`` of a Room-scoped secret binding.

    Shape (plan §2.2)::

        {"env_name": "DEPLOY_TOKEN", "scope": ["deploy"],
         "environment": null, "materialize_to_env": false,
         "mcp_item_id": "…"}

    Only ``env_name`` is required.  ``mcp_item_id`` is accepted (Stage 7) but
    not interpreted here.
    """

    if not isinstance(config, dict):
        raise SecretAccessDenied("room secret binding requires an injection config")
    allowed = {
        "env_name", "scope", "environment", "materialize_to_env", "mcp_item_id",
        # Stage 5 — deploy-key metadata. Never the private key itself; safe to
        # show in the UI and to keep in this plaintext column.
        "public_key", "fingerprint",
    }
    if not set(config).issubset(allowed):
        raise SecretAccessDenied("unexpected room secret injection config keys")

    name = config.get("env_name")
    if (
        not isinstance(name, str)
        or not name.isascii()
        or name.upper() != name
        or len(name) > 128
        or not _ROOM_ENV_NAME_RE.match(name)
    ):
        raise SecretAccessDenied("room secret env_name is invalid")

    scope = config.get("scope")
    if scope is not None:
        if not isinstance(scope, list) or not all(
            isinstance(s, str) and s in ROOM_SECRET_SCOPES for s in scope
        ):
            raise SecretAccessDenied("room secret scope is invalid")

    environment = config.get("environment")
    if environment is not None and not (
        isinstance(environment, str) and _ROOM_ENVIRONMENT_RE.match(environment)
    ):
        raise SecretAccessDenied("room secret environment is invalid")

    materialize = config.get("materialize_to_env")
    if materialize is not None and not isinstance(materialize, bool):
        raise SecretAccessDenied("room secret materialize_to_env must be a boolean")

    normalized: dict[str, Any] = {"env_name": name}
    if scope is not None:
        normalized["scope"] = list(scope)
    normalized["environment"] = environment if environment is not None else None
    if materialize is not None:
        normalized["materialize_to_env"] = materialize
    if config.get("mcp_item_id") is not None:
        normalized["mcp_item_id"] = str(config["mcp_item_id"])
    public_key = config.get("public_key")
    if public_key is not None:
        if not isinstance(public_key, str):
            raise SecretAccessDenied("room secret public_key must be a string")
        normalized["public_key"] = public_key
    fingerprint = config.get("fingerprint")
    if fingerprint is not None:
        if not isinstance(fingerprint, str):
            raise SecretAccessDenied("room secret fingerprint must be a string")
        normalized["fingerprint"] = fingerprint
    return normalized


def _enum(value: Any, enum_type):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise SecretAccessDenied("unsupported secret type") from exc


def _is_expired(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= _utcnow()


class PostgresEncryptedSecretStore:
    """The only component allowed to turn a ``SecretValue`` into plaintext."""

    def __init__(self, db: AsyncSession, keyring: SecretKeyring | None = None):
        self.db = db
        if keyring is None:
            try:
                keyring = load_keyring()
            except SecretKeyringError as exc:
                raise SecretStoreConfigurationError("secret keyring is unavailable") from exc
        self.keyring = keyring

    @staticmethod
    def _meta(row: UserSecret) -> SecretMeta:
        return SecretMeta(
            id=row.id,
            owner_id=row.owner_id,
            kind=SecretKind(row.kind),
            label=row.label,
            provider=row.provider,
            configured=True,
            version=row.version,
            expires_at=row.expires_at,
            state=SecretState(row.state),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _binding_meta(row: SecretBinding) -> BindingMeta:
        return BindingMeta(
            id=row.id,
            secret_id=row.secret_id,
            owner_id=row.owner_id,
            resource_type=SecretResourceType(row.resource_type),
            resource_id=row.resource_id,
            purpose=SecretPurpose(row.purpose),
            injection_config=dict(row.injection_config) if row.injection_config else None,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def _audit(
        self,
        *,
        owner_id: UUID | None,
        action: SecretAuditAction,
        outcome: SecretAuditOutcome,
        request_id: UUID | None = None,
        source_ip: str | None = None,
        secret_id: UUID | None = None,
        target: BindingTarget | None = None,
        error_code: str | None = None,
        actor_type: SecretActorType = SecretActorType.USER,
    ) -> None:
        normalized_target = target.normalized() if target is not None else None
        row = SecretAudit(
            actor_type=actor_type,
            actor_id=owner_id,
            action=action,
            outcome=outcome,
            secret_id=secret_id,
            resource_type=normalized_target.resource_type if normalized_target else None,
            resource_id=normalized_target.resource_id if normalized_target else None,
            purpose=normalized_target.purpose if normalized_target else None,
            request_id=request_id,
            error_code=error_code,
            source_ip=source_ip,
            env_name=(
                normalized_target.injection_config.get("name")
                or normalized_target.injection_config.get("env_name")
                if normalized_target and isinstance(normalized_target.injection_config, dict)
                else None
            ),
            runtime_revision=normalized_target.runtime_revision if normalized_target else None,
        )
        self.db.add(row)
        await self.db.flush()
        # Structured logs contain only identifiers and stable codes, never
        # request bodies, SecretValue, ciphertext, or exception text.
        logger.info(
            "secret_audit action=%s outcome=%s secret_id=%s resource_type=%s purpose=%s request_id=%s error_code=%s",
            action.value,
            outcome.value,
            secret_id,
            normalized_target.resource_type.value if normalized_target else None,
            normalized_target.purpose.value if normalized_target else None,
            request_id,
            error_code,
        )

    async def _owned_secret(self, owner_id: UUID, secret_id: UUID) -> UserSecret:
        row = (
            await self.db.execute(
                select(UserSecret).where(UserSecret.id == secret_id, UserSecret.owner_id == owner_id)
            )
        ).scalar_one_or_none()
        if row is None:
            raise SecretNotFound("secret not found")
        return row

    async def create(
        self,
        *,
        owner_id: UUID,
        kind: SecretKind,
        label: str,
        provider: str | None,
        value: SecretValue,
        expires_at: datetime | None = None,
        request_id: UUID | None = None,
        source_ip: str | None = None,
        actor_type: SecretActorType = SecretActorType.USER,
    ) -> SecretMeta:
        kind = _enum(kind, SecretKind)
        if not isinstance(label, str) or not 1 <= len(label) <= 120:
            raise SecretAccessDenied("secret label is invalid")
        if provider is not None and (not isinstance(provider, str) or len(provider) > 64):
            raise SecretAccessDenied("secret provider is invalid")
        secret_id = uuid4()
        ciphertext, key_id, aad_version = encrypt_secret(
            value,
            secret_id=secret_id,
            owner_id=owner_id,
            kind=kind,
            keyring=self.keyring,
        )
        row = UserSecret(
            id=secret_id,
            owner_id=owner_id,
            kind=kind,
            label=label,
            provider=provider,
            ciphertext=ciphertext,
            key_id=key_id,
            aad_version=aad_version,
            version=1,
            state=SecretState.ACTIVE,
            expires_at=expires_at,
        )
        self.db.add(row)
        await self.db.flush()
        await self._audit(
            owner_id=owner_id,
            action=SecretAuditAction.CREATE,
            outcome=SecretAuditOutcome.SUCCESS,
            request_id=request_id,
            source_ip=source_ip,
            actor_type=actor_type,
            secret_id=secret_id,
        )
        return self._meta(row)

    async def list_metadata(self, *, owner_id: UUID) -> list[SecretMeta]:
        result = await self.db.execute(
            select(UserSecret)
            .where(UserSecret.owner_id == owner_id, UserSecret.state != SecretState.DELETED)
            .order_by(UserSecret.updated_at.desc())
        )
        return [self._meta(row) for row in result.scalars().all()]

    async def update_metadata(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        label: str | None = None,
        expires_at: datetime | None | object = _UNSET,
        enabled: bool | None = None,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> SecretMeta:
        row = await self._owned_secret(owner_id, secret_id)
        if row.state == SecretState.DELETED:
            raise SecretNotFound("secret not found")
        if label is not None:
            if not 1 <= len(label) <= 120:
                raise SecretAccessDenied("secret label is invalid")
            row.label = label
        if expires_at is not _UNSET:
            row.expires_at = expires_at
        action = SecretAuditAction.DISABLE if enabled is False else SecretAuditAction.UPDATE_VALUE
        if enabled is not None:
            row.state = SecretState.ACTIVE if enabled else SecretState.DISABLED
            if enabled:
                row.deleted_at = None
        row.updated_at = _utcnow()
        self.db.add(row)
        await self.db.flush()
        await self._audit(
            owner_id=owner_id,
            action=action,
            outcome=SecretAuditOutcome.SUCCESS,
            request_id=request_id,
            source_ip=source_ip,
            secret_id=secret_id,
        )
        return self._meta(row)

    async def replace_value(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        value: SecretValue,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> SecretMeta:
        row = await self._owned_secret(owner_id, secret_id)
        if row.state == SecretState.DELETED:
            raise SecretNotFound("secret not found")
        ciphertext, key_id, aad_version = encrypt_secret(
            value,
            secret_id=row.id,
            owner_id=row.owner_id,
            kind=row.kind,
            keyring=self.keyring,
        )
        row.ciphertext = ciphertext
        row.key_id = key_id
        row.aad_version = aad_version
        row.version += 1
        row.updated_at = _utcnow()
        if row.state == SecretState.DISABLED:
            row.state = SecretState.ACTIVE
        self.db.add(row)
        await self.db.flush()
        await self._audit(
            owner_id=owner_id,
            action=SecretAuditAction.UPDATE_VALUE,
            outcome=SecretAuditOutcome.SUCCESS,
            request_id=request_id,
            source_ip=source_ip,
            secret_id=secret_id,
        )
        return self._meta(row)

    @staticmethod
    def _validate_target(target: BindingTarget) -> BindingTarget:
        target = target.normalized()
        valid: dict[tuple[SecretResourceType, SecretPurpose], SecretKind] = {
            (SecretResourceType.AGENT, SecretPurpose.LLM_PRIMARY): SecretKind.LLM_API_KEY,
            (SecretResourceType.AGENT, SecretPurpose.LLM_MEDIA): SecretKind.LLM_API_KEY,
            (SecretResourceType.AGENT, SecretPurpose.LLM_VISION): SecretKind.LLM_API_KEY,
            (SecretResourceType.AGENT, SecretPurpose.LLM_ROUTER_JEV): SecretKind.LLM_API_KEY,
            (SecretResourceType.ROOM, SecretPurpose.GIT_HTTPS): SecretKind.GIT_ACCESS_TOKEN,
            (SecretResourceType.ROOM, SecretPurpose.GIT_WEBHOOK): SecretKind.GIT_WEBHOOK_SECRET,
            (SecretResourceType.INSTALLED_MCP, SecretPurpose.MCP_AUTH_HEADER): SecretKind.MCP_HEADER,
            (SecretResourceType.INSTALLED_MCP, SecretPurpose.MCP_ENV): SecretKind.MCP_ENV,
            # internal design notes Stage 3/§3 — the refresh-token slot next to the
            # item's own access-token binding. Carries no injection_config: it
            # is never resolved as a header, only read by Stage 4's refresh.
            (SecretResourceType.INSTALLED_MCP, SecretPurpose.MCP_OAUTH_REFRESH): SecretKind.MCP_HEADER,
            # internal design notes §2.2 — Room-scoped secrets.
            (SecretResourceType.ROOM, SecretPurpose.ROOM_ENV): SecretKind.ROOM_ENV,
            (SecretResourceType.ROOM, SecretPurpose.DEPLOY_SSH_KEY): SecretKind.SSH_PRIVATE_KEY,
            (SecretResourceType.ROOM, SecretPurpose.ROOM_MCP_ENV): SecretKind.ROOM_ENV,
            # internal design notes §3.4 / Stage 7.
            (SecretResourceType.ROOM, SecretPurpose.ROOM_MCP_AUTH_HEADER): SecretKind.MCP_HEADER,
            # internal design notes Stage 13 — the read-only repository token one
            # prepared server holds. A plain opaque string, and deliberately
            # *not* `git_access_token`: that kind is the Room's own
            # full-account PAT, and the two must never be confusable.
            (SecretResourceType.ROOM, SecretPurpose.GIT_DEPLOY): SecretKind.ROOM_ENV,
        }
        # internal design notes Stage 3 — `vps_access` covers both a password and an
        # SSH key under one purpose (different env_name suffixes), so it has
        # no single valid kind; the caller (room_secrets.bind_room_secret's
        # ``kind_override``) picks one per call and it must be one of these.
        multi_kind = {
            (SecretResourceType.ROOM, SecretPurpose.VPS_ACCESS): frozenset(
                {SecretKind.ROOM_ENV, SecretKind.SSH_PRIVATE_KEY}
            ),
        }
        expected = valid.get((target.resource_type, target.purpose))
        if expected is None:
            allowed_kinds = multi_kind.get((target.resource_type, target.purpose))
            if allowed_kinds is not None and target.expected_kind in allowed_kinds:
                expected = target.expected_kind
        if expected is None or (target.expected_kind is not None and target.expected_kind != expected):
            raise SecretAccessDenied("secret kind and binding purpose are incompatible")
        config = target.injection_config
        if target.purpose in {
            SecretPurpose.LLM_PRIMARY,
            SecretPurpose.LLM_MEDIA,
            SecretPurpose.LLM_VISION,
            SecretPurpose.LLM_ROUTER_JEV,
            SecretPurpose.GIT_HTTPS,
            SecretPurpose.GIT_WEBHOOK,
        } and config not in (None, {}):
            raise SecretAccessDenied("injection configuration is not allowed for this purpose")
        if target.purpose == SecretPurpose.MCP_AUTH_HEADER and config not in (
            None,
            {"mode": "bearer"},
        ):
            raise SecretAccessDenied("only bearer MCP authorization is supported")
        if target.purpose == SecretPurpose.MCP_OAUTH_REFRESH and config not in (None, {}):
            raise SecretAccessDenied("injection configuration is not allowed for this purpose")
        if target.purpose == SecretPurpose.MCP_ENV:
            if not isinstance(config, dict) or set(config) != {"name"}:
                raise SecretAccessDenied("MCP environment binding requires one name")
            name = config.get("name")
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or len(name) > 128
                or not name.isascii()
                or not name.upper() == name
            ):
                raise SecretAccessDenied("MCP environment name is invalid")
        if target.purpose in {
            SecretPurpose.ROOM_ENV,
            SecretPurpose.DEPLOY_SSH_KEY,
            SecretPurpose.ROOM_MCP_ENV,
            SecretPurpose.ROOM_MCP_AUTH_HEADER,
            SecretPurpose.VPS_ACCESS,
        }:
            config = _validate_room_env_config(config)
        return BindingTarget(
            target.resource_type,
            target.resource_id,
            target.purpose,
            config,
            expected,
            target.binding_id,
            target.runtime_revision,
        )

    async def bind(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        target: BindingTarget,
        request_id: UUID | None = None,
        source_ip: str | None = None,
        actor_type: SecretActorType = SecretActorType.USER,
    ) -> BindingMeta:
        try:
            target = self._validate_target(target)
        except SecretAccessDenied:
            try:
                from app.core.metrics import record_secret_denied

                record_secret_denied()
            except Exception:
                pass
            raise
        secret = await self._owned_secret(owner_id, secret_id)
        if secret.state != SecretState.ACTIVE or _is_expired(secret.expires_at):
            raise SecretUnavailable("secret is not active")
        if SecretKind(secret.kind) != target.expected_kind:
            raise SecretAccessDenied("secret kind and binding purpose are incompatible")
        existing = (
            await self.db.execute(
                select(SecretBinding).where(
                    SecretBinding.secret_id == secret_id,
                    SecretBinding.owner_id == owner_id,
                    SecretBinding.resource_type == target.resource_type,
                    SecretBinding.resource_id == target.resource_id,
                    SecretBinding.purpose == target.purpose,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            existing.injection_config = target.injection_config
            existing.updated_at = _utcnow()
            self.db.add(existing)
            await self.db.flush()
            binding = existing
        else:
            binding = SecretBinding(
                secret_id=secret_id,
                owner_id=owner_id,
                resource_type=target.resource_type,
                resource_id=target.resource_id,
                purpose=target.purpose,
                injection_config=target.injection_config,
            )
            self.db.add(binding)
            await self.db.flush()
        await self._audit(
            owner_id=owner_id,
            action=SecretAuditAction.BIND,
            outcome=SecretAuditOutcome.SUCCESS,
            request_id=request_id,
            source_ip=source_ip,
            actor_type=actor_type,
            secret_id=secret_id,
            target=target,
        )
        await self._displace_single_bindings(
            owner_id=owner_id,
            keep=binding,
            target=target,
            request_id=request_id,
            source_ip=source_ip,
            actor_type=actor_type,
        )
        return self._binding_meta(binding)

    async def _displace_single_bindings(
        self,
        *,
        owner_id: UUID,
        keep: SecretBinding,
        target: BindingTarget,
        request_id: UUID | None,
        source_ip: str | None,
        actor_type: SecretActorType,
    ) -> None:
        """Drop other secrets bound to the same slot, for single-secret purposes.

        ``bind`` is keyed on (secret, resource, purpose), which is right for
        ``mcp_env``: one server legitimately draws several secrets, told apart by
        their injection_config.  An agent's LLM key is not like that — it has one
        primary, one media and one vision slot.  Without this, switching a key
        left the previous secret bound as well: the agent kept two ``llm_primary``
        bindings, the picker could prefill either of them, and the replaced
        secret silently kept its grant on the agent.
        """

        if target.purpose not in _SINGLE_SECRET_PURPOSES:
            return
        displaced = (
            await self.db.execute(
                select(SecretBinding).where(
                    SecretBinding.owner_id == owner_id,
                    SecretBinding.resource_type == target.resource_type,
                    SecretBinding.resource_id == target.resource_id,
                    SecretBinding.purpose == target.purpose,
                    SecretBinding.id != keep.id,
                )
            )
        ).scalars().all()
        for stale in displaced:
            stale_secret_id = stale.secret_id
            await self.db.delete(stale)
            await self._audit(
                owner_id=owner_id,
                action=SecretAuditAction.UNBIND,
                outcome=SecretAuditOutcome.SUCCESS,
                request_id=request_id,
                source_ip=source_ip,
                actor_type=actor_type,
                secret_id=stale_secret_id,
                target=target,
            )
        if displaced:
            await self.db.flush()

    async def list_bindings(self, *, owner_id: UUID, secret_id: UUID) -> list[BindingMeta]:
        await self._owned_secret(owner_id, secret_id)
        result = await self.db.execute(
            select(SecretBinding)
            .where(SecretBinding.owner_id == owner_id, SecretBinding.secret_id == secret_id)
            .order_by(SecretBinding.created_at)
        )
        return [self._binding_meta(row) for row in result.scalars().all()]

    async def unbind(
        self,
        *,
        owner_id: UUID,
        binding_id: UUID,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> None:
        binding = (
            await self.db.execute(
                select(SecretBinding).where(
                    SecretBinding.id == binding_id,
                    SecretBinding.owner_id == owner_id,
                )
            )
        ).scalar_one_or_none()
        if binding is None:
            raise SecretNotFound("binding not found")
        target = BindingTarget(binding.resource_type, binding.resource_id, binding.purpose)
        await self.db.delete(binding)
        await self.db.flush()
        await self._audit(
            owner_id=owner_id,
            action=SecretAuditAction.UNBIND,
            outcome=SecretAuditOutcome.SUCCESS,
            request_id=request_id,
            source_ip=source_ip,
            secret_id=binding.secret_id,
            target=target,
        )

    async def resolve_for_use(
        self,
        *,
        owner_id: UUID,
        target: BindingTarget,
        request_id: UUID,
        source_ip: str | None = None,
    ) -> SecretValue:
        try:
            from app.core.metrics import record_secret_resolve

            record_secret_resolve()
        except Exception:
            pass
        try:
            target = self._validate_target(target)
        except SecretAccessDenied:
            try:
                from app.core.metrics import record_secret_denied

                record_secret_denied()
            except Exception:
                pass
            raise
        binding = (
            await self.db.execute(
                select(SecretBinding).where(
                    SecretBinding.owner_id == owner_id,
                    SecretBinding.resource_type == target.resource_type,
                    SecretBinding.resource_id == target.resource_id,
                    SecretBinding.purpose == target.purpose,
                    *(
                        [SecretBinding.id == target.binding_id]
                        if target.binding_id is not None
                        else []
                    ),
                )
            )
        ).scalar_one_or_none()
        if binding is None:
            try:
                from app.core.metrics import record_secret_denied

                record_secret_denied()
            except Exception:
                pass
            await self._audit(
                owner_id=owner_id,
                action=SecretAuditAction.RESOLVE,
                outcome=SecretAuditOutcome.DENIED,
                request_id=request_id,
                source_ip=source_ip,
                target=target,
                error_code="binding_not_found",
            )
            raise SecretAccessDenied("secret binding is not available")
        secret = (
            await self.db.execute(
                select(UserSecret).where(
                    UserSecret.id == binding.secret_id,
                    UserSecret.owner_id == owner_id,
                )
            )
        ).scalar_one_or_none()
        if secret is None or SecretKind(secret.kind) != target.expected_kind:
            try:
                from app.core.metrics import record_secret_denied

                record_secret_denied()
            except Exception:
                pass
            await self._audit(
                owner_id=owner_id,
                action=SecretAuditAction.RESOLVE,
                outcome=SecretAuditOutcome.DENIED,
                request_id=request_id,
                source_ip=source_ip,
                secret_id=binding.secret_id,
                target=target,
                error_code="stale_binding",
            )
            raise SecretAccessDenied("secret binding is not available")
        if secret.state != SecretState.ACTIVE:
            try:
                from app.core.metrics import record_secret_denied

                record_secret_denied()
            except Exception:
                pass
            await self._audit(
                owner_id=owner_id,
                action=SecretAuditAction.RESOLVE,
                outcome=SecretAuditOutcome.DENIED,
                request_id=request_id,
                source_ip=source_ip,
                secret_id=secret.id,
                target=target,
                error_code="secret_disabled",
            )
            raise SecretUnavailable("secret is disabled")
        if _is_expired(secret.expires_at):
            try:
                from app.core.metrics import record_secret_denied

                record_secret_denied()
            except Exception:
                pass
            await self._audit(
                owner_id=owner_id,
                action=SecretAuditAction.RESOLVE,
                outcome=SecretAuditOutcome.DENIED,
                request_id=request_id,
                source_ip=source_ip,
                secret_id=secret.id,
                target=target,
                error_code="secret_expired",
            )
            raise SecretUnavailable("secret is expired")
        try:
            value = decrypt_secret(
                secret.ciphertext,
                secret_id=secret.id,
                owner_id=secret.owner_id,
                kind=secret.kind,
                key_id=secret.key_id,
                aad_version=secret.aad_version,
                keyring=self.keyring,
            )
        except SecretDecryptionError:
            try:
                from app.core.metrics import record_secret_decrypt_failed

                record_secret_decrypt_failed()
            except Exception:
                pass
            await self._audit(
                owner_id=owner_id,
                action=SecretAuditAction.DECRYPT_FAILED,
                outcome=SecretAuditOutcome.FAILED,
                request_id=request_id,
                source_ip=source_ip,
                secret_id=secret.id,
                target=target,
                error_code="decrypt_failed",
            )
            raise
        await self._audit(
            owner_id=owner_id,
            action=SecretAuditAction.RESOLVE,
            outcome=SecretAuditOutcome.SUCCESS,
            request_id=request_id,
            source_ip=source_ip,
            secret_id=secret.id,
            target=target,
        )
        if secret.provider == "openai_chatgpt" and SecretKind(secret.kind) == SecretKind.LLM_API_KEY:
            from app.core.chatgpt_credentials import runtime_reference
            return runtime_reference(secret.owner_id, secret.id)
        return value

    async def resolve_owned_llm_for_validation(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        request_id: UUID,
        source_ip: str | None = None,
    ) -> SecretValue:
        """Resolve one owner-selected LLM secret for a provider health check.

        This deliberately has no HTTP response carrying the value and is not a
        substitute for resource bindings at runtime.  It exists so that a user
        can validate a newly saved provider credential before assigning it to
        an agent.
        """

        try:
            from app.core.metrics import record_secret_denied, record_secret_resolve

            record_secret_resolve()
        except Exception:
            record_secret_denied = None

        try:
            secret = await self._owned_secret(owner_id, secret_id)
            if SecretKind(secret.kind) != SecretKind.LLM_API_KEY:
                raise SecretAccessDenied("provider validation requires an LLM secret")
            if secret.state != SecretState.ACTIVE:
                raise SecretUnavailable("secret is disabled")
            if _is_expired(secret.expires_at):
                raise SecretUnavailable("secret is expired")
        except (SecretNotFound, SecretAccessDenied, SecretUnavailable):
            if record_secret_denied is not None:
                record_secret_denied()
            await self._audit(
                owner_id=owner_id,
                action=SecretAuditAction.RESOLVE,
                outcome=SecretAuditOutcome.DENIED,
                request_id=request_id,
                source_ip=source_ip,
                secret_id=secret_id,
                error_code="provider_validation_unavailable",
            )
            raise

        try:
            value = decrypt_secret(
                secret.ciphertext,
                secret_id=secret.id,
                owner_id=secret.owner_id,
                kind=secret.kind,
                key_id=secret.key_id,
                aad_version=secret.aad_version,
                keyring=self.keyring,
            )
        except SecretDecryptionError:
            try:
                from app.core.metrics import record_secret_decrypt_failed

                record_secret_decrypt_failed()
            except Exception:
                pass
            await self._audit(
                owner_id=owner_id,
                action=SecretAuditAction.DECRYPT_FAILED,
                outcome=SecretAuditOutcome.FAILED,
                request_id=request_id,
                source_ip=source_ip,
                secret_id=secret_id,
                error_code="decrypt_failed",
            )
            raise
        await self._audit(
            owner_id=owner_id,
            action=SecretAuditAction.RESOLVE,
            outcome=SecretAuditOutcome.SUCCESS,
            request_id=request_id,
            source_ip=source_ip,
            secret_id=secret_id,
        )
        if secret.provider == "openai_chatgpt" and SecretKind(secret.kind) == SecretKind.LLM_API_KEY:
            from app.core.chatgpt_credentials import runtime_reference
            return runtime_reference(secret.owner_id, secret.id)
        return value

    async def revoke_or_delete(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> None:
        row = await self._owned_secret(owner_id, secret_id)
        if row.state == SecretState.DELETED:
            return
        await self.db.execute(delete(SecretBinding).where(SecretBinding.secret_id == secret_id))
        row.state = SecretState.DELETED
        row.deleted_at = _utcnow()
        row.updated_at = _utcnow()
        self.db.add(row)
        await self.db.flush()
        await self._audit(
            owner_id=owner_id,
            action=SecretAuditAction.DELETE,
            outcome=SecretAuditOutcome.SUCCESS,
            request_id=request_id,
            source_ip=source_ip,
            secret_id=secret_id,
        )
