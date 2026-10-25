"""Protected-environment gate — internal design notes §2.4 / Stage 12.

An environment can be marked ``protected``. Resolving a Room secret for it
(``room_secrets.resolve_room_secret`` — the sole choke point every consumer
goes through: bash, workspace services, CI checks, ``.env`` sync) requires an
active human confirmation window scoped to exactly one ``(room_id,
environment)`` pair. Without one, the caller gets ``environment_locked``
instead of a resolved value.

Variables are never gated here: :func:`app.core.room_config.effective_variables`
does not call anything in this module, because a variable is not secret.

This module intentionally does not import ``app.core.secrets.room_secrets`` —
that module imports this one to raise ``EnvironmentLocked``, so the reverse
import would be circular. Everything below is either a pure DB read/write or
a plain audit row.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.room_environment import RoomEnvironmentConfig, RoomEnvironmentUnlock
from app.models.secret import (
    SecretActorType,
    SecretAudit,
    SecretAuditAction,
    SecretAuditOutcome,
)

# plan §2.4: "15 minutes by default".
DEFAULT_UNLOCK_MINUTES = 15


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def is_protected(
    db: AsyncSession, room_id: UUID, environment: Optional[str]
) -> bool:
    """The shared (``NULL``) bucket is never protected — nothing to unlock."""
    if not environment:
        return False
    row = (
        await db.execute(
            select(RoomEnvironmentConfig.protected).where(
                RoomEnvironmentConfig.room_id == room_id,
                RoomEnvironmentConfig.environment == environment,
            )
        )
    ).first()
    return bool(row and row[0])


async def has_active_unlock(
    db: AsyncSession, room_id: UUID, environment: str
) -> bool:
    row = (
        await db.execute(
            select(RoomEnvironmentUnlock.id).where(
                RoomEnvironmentUnlock.room_id == room_id,
                RoomEnvironmentUnlock.environment == environment,
                RoomEnvironmentUnlock.expires_at > _utcnow(),
            )
        )
    ).first()
    return row is not None


async def active_unlock_expiry(
    db: AsyncSession, room_id: UUID, environment: str
) -> Optional[datetime]:
    """Latest still-active window's expiry, for the UI countdown. ``None`` if locked."""
    row = (
        await db.execute(
            select(RoomEnvironmentUnlock.expires_at)
            .where(
                RoomEnvironmentUnlock.room_id == room_id,
                RoomEnvironmentUnlock.environment == environment,
                RoomEnvironmentUnlock.expires_at > _utcnow(),
            )
            .order_by(RoomEnvironmentUnlock.expires_at.desc())
        )
    ).first()
    return row[0] if row else None


async def grant_unlock(
    db: AsyncSession,
    *,
    room_id: UUID,
    environment: str,
    actor_user_id: Optional[UUID],
    minutes: int = DEFAULT_UNLOCK_MINUTES,
    request_id: Optional[UUID] = None,
) -> RoomEnvironmentUnlock:
    """Open a confirmation window. Always audited, even though it isn't a resolve."""
    now = _utcnow()
    row = RoomEnvironmentUnlock(
        room_id=room_id,
        environment=environment,
        granted_by=actor_user_id,
        granted_at=now,
        expires_at=now + timedelta(minutes=minutes),
    )
    db.add(row)
    db.add(
        SecretAudit(
            actor_type=SecretActorType.USER if actor_user_id else SecretActorType.SYSTEM,
            actor_id=actor_user_id,
            action=SecretAuditAction.UNLOCK_ENVIRONMENT,
            outcome=SecretAuditOutcome.SUCCESS,
            resource_type="room",
            resource_id=room_id,
            request_id=request_id or uuid4(),
            env_name=environment,
        )
    )
    await db.commit()
    await db.refresh(row)
    return row


async def get_config(
    db: AsyncSession, room_id: UUID, environment: str
) -> Optional[RoomEnvironmentConfig]:
    return (
        await db.execute(
            select(RoomEnvironmentConfig).where(
                RoomEnvironmentConfig.room_id == room_id,
                RoomEnvironmentConfig.environment == environment,
            )
        )
    ).scalars().first()


async def set_protected(
    db: AsyncSession, *, room_id: UUID, environment: str, protected: bool
) -> RoomEnvironmentConfig:
    row = await get_config(db, room_id, environment)
    if row is None:
        row = RoomEnvironmentConfig(room_id=room_id, environment=environment, protected=protected)
    else:
        row.protected = protected
        row.updated_at = _utcnow()
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def list_configs(db: AsyncSession, room_id: UUID) -> dict[str, bool]:
    """``{environment: protected}`` for every environment that has a config row."""
    rows = (
        await db.execute(
            select(
                RoomEnvironmentConfig.environment, RoomEnvironmentConfig.protected
            ).where(RoomEnvironmentConfig.room_id == room_id)
        )
    ).all()
    return {r[0]: bool(r[1]) for r in rows}


async def list_active_unlocks(db: AsyncSession, room_id: UUID) -> dict[str, datetime]:
    """Latest active-unlock expiry per environment, for the UI countdown."""
    rows = (
        await db.execute(
            select(
                RoomEnvironmentUnlock.environment, RoomEnvironmentUnlock.expires_at
            ).where(
                RoomEnvironmentUnlock.room_id == room_id,
                RoomEnvironmentUnlock.expires_at > _utcnow(),
            )
        )
    ).all()
    out: dict[str, datetime] = {}
    for env, expires_at in rows:
        if env not in out or expires_at > out[env]:
            out[env] = expires_at
    return out
