"""Short-lived, item-scoped capabilities for managed stdio MCPs.

The public API deals only in ``secret_id`` references.  This module is the
service-to-service boundary used by mcp_host: it issues a one-time opaque
capability and resolves the selected values only when that capability is
consumed.  Plaintext exists only in the returned in-memory mapping at the
final host boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets as token_secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.market import UserInstalledItem
from app.models.personal_mcp import PersonalMcpRuntime
from app.models.secret import (
    SecretActorType,
    SecretAudit,
    SecretAuditAction,
    SecretAuditOutcome,
    SecretBinding,
    SecretKind,
    SecretLease,
    SecretPurpose,
    SecretResourceType,
    SecretState,
    UserSecret,
)

from .crypto import SecretValue
from .store import (
    BindingTarget,
    PostgresEncryptedSecretStore,
    SecretAccessDenied,
    SecretStoreError,
)


LEASE_TTL_SECONDS = 60
MAX_LEASE_TTL_SECONDS = 300


class SecretLeaseError(RuntimeError):
    """Base class for safe lease errors."""


class SecretLeaseDenied(SecretLeaseError):
    """The requested item, revision, or capability is not usable."""


class SecretLeaseExpired(SecretLeaseError):
    """The capability is no longer valid."""


@dataclass(frozen=True)
class IssuedSecretLease:
    lease_id: UUID
    lease_token: str
    resource_id: UUID
    runtime_revision: int
    env_names: tuple[str, ...]
    expires_at: datetime


@dataclass(frozen=True)
class ConsumedSecretLease:
    lease_id: UUID
    resource_id: UUID
    runtime_revision: int
    expires_at: datetime
    values: dict[str, SecretValue]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _env_name(binding: SecretBinding) -> str | None:
    config = binding.injection_config or {}
    value = config.get("name") or config.get("env_name")
    return value if isinstance(value, str) else None


def _active(secret: UserSecret | None) -> bool:
    if secret is None or SecretState(secret.state) != SecretState.ACTIVE:
        return False
    return secret.expires_at is None or _aware(secret.expires_at) > _now()


class ManagedMcpSecretLeaseService:
    """Issue and consume one lease for one managed-stdio runtime revision."""

    def __init__(
        self,
        db: AsyncSession,
        *,
        store: PostgresEncryptedSecretStore | None = None,
        ttl_seconds: int = LEASE_TTL_SECONDS,
    ) -> None:
        self.db = db
        self.store = store
        self.ttl_seconds = max(1, min(int(ttl_seconds), MAX_LEASE_TTL_SECONDS))

    async def _runtime(
        self, owner_id: UUID, resource_id: UUID, runtime_revision: int | None = None
    ) -> PersonalMcpRuntime:
        query = (
            select(PersonalMcpRuntime)
            .join(
                UserInstalledItem,
                UserInstalledItem.id == PersonalMcpRuntime.installed_item_id,
            )
            .where(
                PersonalMcpRuntime.installed_item_id == resource_id,
                PersonalMcpRuntime.owner_id == owner_id,
                UserInstalledItem.user_id == owner_id,
                UserInstalledItem.type == "mcp",
                PersonalMcpRuntime.kind == "managed_stdio",
                PersonalMcpRuntime.actual_status.not_in(("deleted", "deleting")),
            )
        )
        if runtime_revision is not None:
            query = query.where(PersonalMcpRuntime.revision == runtime_revision)
        runtime = (await self.db.execute(query)).scalar_one_or_none()
        if runtime is None:
            raise SecretLeaseDenied("managed MCP lease scope is not available")
        return runtime

    async def issue(
        self,
        *,
        owner_id: UUID,
        resource_id: UUID,
        runtime_revision: int,
        room_id: UUID | None = None,
        environment: str | None = None,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> IssuedSecretLease:
        runtime = await self._runtime(owner_id, resource_id, runtime_revision)
        result = await self.db.execute(
            select(SecretBinding, UserSecret)
            .join(UserSecret, UserSecret.id == SecretBinding.secret_id, isouter=True)
            .where(
                SecretBinding.owner_id == owner_id,
                SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
                SecretBinding.resource_id == resource_id,
                SecretBinding.purpose == SecretPurpose.MCP_ENV,
            )
            .order_by(SecretBinding.created_at, SecretBinding.id)
        )
        binding_ids: list[str] = []
        env_names: list[str] = []
        launch_spec = runtime.launch_spec if isinstance(runtime.launch_spec, dict) else {}
        launch_env_names = set(launch_spec.get("env_names", []))
        for binding, secret in result.all():
            name = _env_name(binding)
            # Fail closed: a lease is never issued with a malformed/stale
            # binding.  The UI can repair the mapping and retry the start.
            if (
                name is None
                or name in env_names
                or name not in launch_env_names
                or not _active(secret)
                or secret.owner_id != owner_id
                or SecretKind(secret.kind) != SecretKind.MCP_ENV
            ):
                raise SecretLeaseDenied("managed MCP secret binding is not available")
            binding_ids.append(str(binding.id))
            env_names.append(name)

        if room_id is not None:
            await self._merge_room_env(
                owner_id=owner_id,
                room_id=room_id,
                resource_id=resource_id,
                environment=environment,
                launch_env_names=launch_env_names,
                binding_ids=binding_ids,
                env_names=env_names,
            )

        token = token_secrets.token_urlsafe(32)
        lease = SecretLease(
            id=uuid4(),
            owner_id=owner_id,
            resource_type=SecretResourceType.INSTALLED_MCP,
            resource_id=resource_id,
            runtime_revision=runtime.revision,
            binding_ids=binding_ids,
            env_names=env_names,
            token_hash=_token_hash(token),
            expires_at=_now() + timedelta(seconds=self.ttl_seconds),
        )
        self.db.add(lease)
        self.db.add(
            SecretAudit(
                actor_type=SecretActorType.SYSTEM,
                actor_id=owner_id,
                action=SecretAuditAction.ISSUE_LEASE,
                outcome=SecretAuditOutcome.SUCCESS,
                resource_type=SecretResourceType.INSTALLED_MCP,
                resource_id=resource_id,
                request_id=request_id,
                source_ip=source_ip,
                runtime_revision=runtime.revision,
            )
        )
        await self.db.flush()
        return IssuedSecretLease(
            lease_id=lease.id,
            lease_token=token,
            resource_id=resource_id,
            runtime_revision=runtime.revision,
            env_names=tuple(env_names),
            expires_at=lease.expires_at,
        )

    async def _merge_room_env(
        self,
        *,
        owner_id: UUID,
        room_id: UUID,
        resource_id: UUID,
        environment: str | None,
        launch_env_names: set[str],
        binding_ids: list[str],
        env_names: list[str],
    ) -> None:
        """Fold Room-supplied env into an in-progress lease (plan §3.4, Stage 7).

        Best-effort and additive: unlike the item owner's own ``mcp_env``
        bindings above, a stale or misconfigured Room binding is silently
        skipped rather than failing the whole lease — the owner starting
        this MCP has no way to repair another member's binding, and a Room
        secret is optional supplementation, never the only source of a name
        the runtime actually declared.

        The protected-environment gate (internal design notes Stage 0) is the one
        exception to "best-effort": a locked ``environment`` raises
        ``EnvironmentLocked`` out of ``room_mcp_env_bindings_for_lease``
        rather than being swallowed here, so ``issue()`` fails the whole
        lease instead of quietly handing out a partial one.
        """

        from app.models.market import RoomMcpServer
        from app.models.room import Room, RoomUserLink

        from .room_secrets import room_mcp_env_bindings_for_lease

        attached = await self.db.scalar(
            select(RoomMcpServer.room_id).where(
                RoomMcpServer.room_id == room_id,
                RoomMcpServer.user_installed_item_id == resource_id,
            )
        )
        is_member = await self.db.scalar(
            select(Room.id)
            .outerjoin(RoomUserLink, RoomUserLink.room_id == Room.id)
            .where(
                Room.id == room_id,
                or_(Room.creator_id == owner_id, RoomUserLink.user_id == owner_id),
            )
        )
        if attached is None or is_member is None:
            return
        for binding, secret in await room_mcp_env_bindings_for_lease(
            self.db, room_id=room_id, mcp_item_id=resource_id, environment=environment
        ):
            name = _env_name(binding)
            if (
                name is None
                or name in env_names
                or name not in launch_env_names
                or not _active(secret)
                or SecretKind(secret.kind) != SecretKind.ROOM_ENV
            ):
                continue
            binding_ids.append(str(binding.id))
            env_names.append(name)

    async def consume(
        self,
        *,
        owner_id: UUID,
        resource_id: UUID,
        lease_token: str,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> ConsumedSecretLease:
        if not isinstance(lease_token, str) or not 32 <= len(lease_token) <= 256:
            raise SecretLeaseDenied("secret lease is invalid")
        lease = (
            await self.db.execute(
                select(SecretLease)
                .where(
                    SecretLease.owner_id == owner_id,
                    SecretLease.resource_id == resource_id,
                    SecretLease.token_hash == _token_hash(lease_token),
                )
                .with_for_update()
            )
        ).scalar_one_or_none()
        if lease is None or not hmac.compare_digest(lease.token_hash, _token_hash(lease_token)):
            raise SecretLeaseDenied("secret lease is invalid")
        if lease.consumed_at is not None or lease.revoked_at is not None:
            raise SecretLeaseDenied("secret lease is no longer available")
        if _aware(lease.expires_at) <= _now():
            raise SecretLeaseExpired("secret lease has expired")

        runtime = await self._runtime(owner_id, resource_id, lease.runtime_revision)
        if runtime.actual_status in {"stopped", "error"} or runtime.desired_status == "deleted":
            raise SecretLeaseDenied("managed MCP runtime is not running")

        expected_ids = [UUID(value) for value in lease.binding_ids]
        # No resource_type/purpose/owner filter here: a merged lease
        # (``issue(room_id=...)``, Stage 7) carries a mix of the item owner's
        # own ``(installed_mcp, mcp_env)`` bindings and other Room members'
        # ``(room, room_mcp_env)`` ones. Each id is re-checked against its own
        # expected shape below instead.
        result = (
            await self.db.execute(
                select(SecretBinding, UserSecret)
                .join(UserSecret, UserSecret.id == SecretBinding.secret_id, isouter=True)
                .where(SecretBinding.id.in_(expected_ids))
            )
            if expected_ids
            else None
        )
        rows = result.all() if result is not None else []
        by_id = {binding.id: (binding, secret) for binding, secret in rows}
        if len(by_id) != len(expected_ids):
            raise SecretLeaseDenied("secret binding changed; start again")

        values: dict[str, SecretValue] = {}
        store = self.store or PostgresEncryptedSecretStore(self.db)
        from .service import SecretService

        service = SecretService(self.db, store=store)
        for binding_id, expected_name in zip(expected_ids, lease.env_names):
            binding, secret = by_id[binding_id]
            name = _env_name(binding)
            resource_type = SecretResourceType(binding.resource_type)
            if resource_type == SecretResourceType.INSTALLED_MCP:
                if (
                    name != expected_name
                    or binding.owner_id != owner_id
                    or binding.resource_id != resource_id
                    or SecretPurpose(binding.purpose) != SecretPurpose.MCP_ENV
                    or not _active(secret)
                    or SecretKind(secret.kind) != SecretKind.MCP_ENV
                ):
                    raise SecretLeaseDenied("secret binding changed; start again")
                resolve_owner_id = owner_id
                target = BindingTarget(
                    resource_type=SecretResourceType.INSTALLED_MCP,
                    resource_id=resource_id,
                    purpose=SecretPurpose.MCP_ENV,
                    binding_id=binding.id,
                    injection_config={"name": name},
                    runtime_revision=lease.runtime_revision,
                )
            elif resource_type == SecretResourceType.ROOM:
                config = binding.injection_config or {}
                if (
                    name != expected_name
                    or config.get("mcp_item_id") != str(resource_id)
                    or SecretPurpose(binding.purpose) != SecretPurpose.ROOM_MCP_ENV
                    or not _active(secret)
                    or SecretKind(secret.kind) != SecretKind.ROOM_ENV
                ):
                    raise SecretLeaseDenied("secret binding changed; start again")
                # §2.3 — a Room secret always resolves under its own binder's
                # id, never the runtime owner who merely started this MCP.
                resolve_owner_id = binding.owner_id
                target = BindingTarget(
                    resource_type=SecretResourceType.ROOM,
                    resource_id=binding.resource_id,
                    purpose=SecretPurpose.ROOM_MCP_ENV,
                    binding_id=binding.id,
                    injection_config=dict(config),
                )
            else:
                raise SecretLeaseDenied("secret binding changed; start again")
            try:
                values[name] = await service.resolve_for_use(
                    owner_id=resolve_owner_id,
                    target=target,
                    request_id=request_id or uuid4(),
                    source_ip=source_ip,
                )
            except SecretStoreError:
                raise SecretLeaseDenied("secret binding is not available") from None

        lease.consumed_at = _now()
        self.db.add(lease)
        self.db.add(
            SecretAudit(
                actor_type=SecretActorType.SYSTEM,
                actor_id=owner_id,
                action=SecretAuditAction.CONSUME_LEASE,
                outcome=SecretAuditOutcome.SUCCESS,
                resource_type=SecretResourceType.INSTALLED_MCP,
                resource_id=resource_id,
                request_id=request_id,
                source_ip=source_ip,
                runtime_revision=lease.runtime_revision,
            )
        )
        await self.db.flush()
        return ConsumedSecretLease(
            lease_id=lease.id,
            resource_id=resource_id,
            runtime_revision=lease.runtime_revision,
            expires_at=lease.expires_at,
            values=values,
        )


# Short alias for callers that use the generic SecretStore terminology.
SecretLeaseService = ManagedMcpSecretLeaseService


__all__ = [
    "ConsumedSecretLease",
    "IssuedSecretLease",
    "LEASE_TTL_SECONDS",
    "ManagedMcpSecretLeaseService",
    "SecretLeaseService",
    "SecretLeaseDenied",
    "SecretLeaseError",
    "SecretLeaseExpired",
]
