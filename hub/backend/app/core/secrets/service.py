"""Resource authorization around the low-level SecretStore contract."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import Agent
from app.models.market import UserInstalledItem
from app.models.personal_mcp import PersonalMcpRuntime
from app.models.room import Room, RoomUserLink
from app.models.secret import SecretPurpose, SecretResourceType

from .crypto import SecretValue
from .store import (
    BindingMeta,
    BindingTarget,
    PostgresEncryptedSecretStore,
    SecretMeta,
    SecretStore,
)


class SecretResourceAccessDenied(RuntimeError):
    """The caller does not own or participate in the target resource."""


class SecretService:
    """Authorize agent, room, and installed-MCP bindings before persistence."""

    def __init__(self, db: AsyncSession, store: SecretStore | None = None):
        self.db = db
        self.store = store or PostgresEncryptedSecretStore(db)

    async def authorize_target(
        self, owner_id: UUID, target: BindingTarget, *, action: str = "bind"
    ) -> None:
        """``action`` distinguishes *creating* a binding from *using* one.

        Both ``bind`` and ``resolve_for_use`` funnel through here, but with a
        different meaning for ``owner_id``: at bind time it is the caller
        attaching a new secret; at resolve time it is the binding's own
        owner (§2.3 — a Room secret always resolves under its binder, who may
        not be the resource owner at all). The ``room_mcp_env`` /
        ``room_mcp_auth_header`` ownership check below only makes sense for
        the former — applying it to resolution would reject exactly the
        cross-member secrets Stage 7 exists to allow.
        """

        target = target.normalized()
        if target.resource_type.value == "agent":
            row = (
                await self.db.execute(
                    select(Agent.id).where(Agent.id == target.resource_id, Agent.owner_id == owner_id)
                )
            ).first()
        elif target.resource_type.value == "room":
            row = (
                await self.db.execute(
                    select(Room.id)
                    .outerjoin(RoomUserLink, RoomUserLink.room_id == Room.id)
                    .where(
                        Room.id == target.resource_id,
                        or_(Room.creator_id == owner_id, RoomUserLink.user_id == owner_id),
                    )
                )
            ).first()
            if (
                row is not None
                and action == "bind"
                and target.purpose.value in ("room_mcp_env", "room_mcp_auth_header")
            ):
                # internal design notes §3.4 / Stage 7 — membership alone
                # is not enough for these two purposes: the binder must also own
                # the MCP item this secret is meant to feed, or any Room member
                # could siphon another member's installed MCP's credentials by
                # naming their ``installed_item_id`` in ``mcp_item_id``.
                config = target.injection_config or {}
                item_row = None
                raw_item_id = config.get("mcp_item_id")
                if raw_item_id:
                    try:
                        item_uuid = UUID(str(raw_item_id))
                    except (TypeError, ValueError):
                        item_uuid = None
                    if item_uuid is not None:
                        item_row = (
                            await self.db.execute(
                                select(UserInstalledItem.id).where(
                                    UserInstalledItem.id == item_uuid,
                                    UserInstalledItem.user_id == owner_id,
                                    UserInstalledItem.type == "mcp",
                                )
                            )
                        ).first()
                if item_row is None:
                    row = None
        elif target.resource_type.value == "installed_mcp":
            query = select(UserInstalledItem.id).where(
                UserInstalledItem.id == target.resource_id,
                UserInstalledItem.user_id == owner_id,
                UserInstalledItem.type == "mcp",
            )
            # ``mcp_env`` is intentionally new-runtime-only.  Legacy
            # personal-image and remote HTTP MCPs retain their existing
            # header path but cannot opt into process environment injection.
            if target.purpose.value == "mcp_env":
                query = query.join(
                    PersonalMcpRuntime,
                    PersonalMcpRuntime.installed_item_id == UserInstalledItem.id,
                ).where(PersonalMcpRuntime.kind == "managed_stdio")
            row = (await self.db.execute(query)).first()
        else:  # pragma: no cover - BindingTarget.normalized guards this
            row = None
        if row is None:
            raise SecretResourceAccessDenied("resource is not available")

    async def bind(
        self,
        *,
        owner_id: UUID,
        secret_id: UUID,
        target: BindingTarget,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> BindingMeta:
        await self.authorize_target(owner_id, target)
        target = target.normalized()
        if (
            target.resource_type == SecretResourceType.INSTALLED_MCP
            and target.purpose == SecretPurpose.MCP_ENV
            and target.runtime_revision is None
        ):
            revision = await self.db.scalar(
                select(PersonalMcpRuntime.revision).where(
                    PersonalMcpRuntime.installed_item_id == target.resource_id,
                    PersonalMcpRuntime.owner_id == owner_id,
                    PersonalMcpRuntime.kind == "managed_stdio",
                )
            )
            target = BindingTarget(
                target.resource_type,
                target.resource_id,
                target.purpose,
                target.injection_config,
                target.expected_kind,
                target.binding_id,
                int(revision) if revision is not None else None,
            )
        return await self.store.bind(
            owner_id=owner_id,
            secret_id=secret_id,
            target=target,
            request_id=request_id,
            source_ip=source_ip,
        )

    async def resolve_for_use(
        self,
        *,
        owner_id: UUID,
        target: BindingTarget,
        request_id: UUID,
        source_ip: str | None = None,
    ) -> SecretValue:
        await self.authorize_target(owner_id, target, action="resolve")
        return await self.store.resolve_for_use(
            owner_id=owner_id,
            target=target,
            request_id=request_id,
            source_ip=source_ip,
        )

    async def create(
        self,
        *,
        owner_id: UUID,
        kind,
        label: str,
        provider: str | None,
        value: SecretValue,
        expires_at=None,
        request_id: UUID | None = None,
        source_ip: str | None = None,
    ) -> SecretMeta:
        return await self.store.create(
            owner_id=owner_id,
            kind=kind,
            label=label,
            provider=provider,
            value=value,
            expires_at=expires_at,
            request_id=request_id,
            source_ip=source_ip,
        )
