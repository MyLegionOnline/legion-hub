"""Connector identity, pairing, and room-grant service.

Mirrors ``core/secrets/lease.py``'s one-time-capability pattern: a pairing
code / refresh token is generated once, returned to the caller in memory, and
only its SHA-256 hash is ever persisted. Plaintext exists only in the
return value of :func:`create_connector` / :func:`pair` / :func:`issue_access_token`.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets as token_secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID, uuid4

from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import ALGORITHM, SECRET_KEY
from app.models.connector import Connector, ConnectorRoomGrant

PAIRING_CODE_TTL_SECONDS = 15 * 60
CONNECTOR_ACCESS_TOKEN_EXPIRE_MINUTES = 10
CONNECTOR_TOKEN_TYPE = "connector_access"


class ConnectorPairingError(RuntimeError):
    """Base class for safe, user-facing pairing errors."""

    http_status = 400


class ConnectorPairingInvalid(ConnectorPairingError):
    """The code/token does not match any connector."""

    http_status = 404


class ConnectorPairingAlreadyUsed(ConnectorPairingError):
    """The pairing code was already exchanged, or the connector is revoked."""

    http_status = 409


class ConnectorPairingExpired(ConnectorPairingError):
    """The pairing code's TTL has elapsed."""

    http_status = 410


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class IssuedPairing:
    connector_id: UUID
    pairing_code: str
    expires_at: datetime


@dataclass(frozen=True)
class PairedConnector:
    connector_id: UUID
    refresh_token: str


async def create_connector(
    db: AsyncSession, *, owner_id: UUID, label: str = ""
) -> IssuedPairing:
    """Create a Connector row and issue its first (and only live) pairing code.

    The code is one-time: a second exchange attempt fails with
    :class:`ConnectorPairingAlreadyUsed` (409), never silently re-issuing.
    """
    code = token_secrets.token_urlsafe(24)
    expires_at = _now() + timedelta(seconds=PAIRING_CODE_TTL_SECONDS)
    connector = Connector(
        id=uuid4(),
        owner_id=owner_id,
        label=label,
        status="offline",
        pairing_code_hash=_hash(code),
        pairing_code_expires_at=expires_at,
    )
    db.add(connector)
    await db.flush()
    return IssuedPairing(
        connector_id=connector.id, pairing_code=code, expires_at=expires_at
    )


async def pair(db: AsyncSession, *, pairing_code: str) -> PairedConnector:
    """Exchange a one-time pairing code for a long-lived refresh token.

    The code itself is the only credential — no user JWT is involved (the
    code authorizes itself).
    """
    if not isinstance(pairing_code, str) or not 16 <= len(pairing_code) <= 256:
        raise ConnectorPairingInvalid("pairing code is invalid")
    connector = (
        await db.execute(
            select(Connector)
            .where(Connector.pairing_code_hash == _hash(pairing_code))
            .with_for_update()
        )
    ).scalar_one_or_none()
    if connector is None or not hmac.compare_digest(
        connector.pairing_code_hash or "", _hash(pairing_code)
    ):
        raise ConnectorPairingInvalid("pairing code is invalid")
    if connector.pairing_code_consumed_at is not None or connector.revoked_at is not None:
        raise ConnectorPairingAlreadyUsed("pairing code was already used")
    if connector.pairing_code_expires_at is None or _aware(connector.pairing_code_expires_at) <= _now():
        raise ConnectorPairingExpired("pairing code has expired")

    refresh_token = token_secrets.token_urlsafe(32)
    connector.pairing_code_consumed_at = _now()
    connector.refresh_token_hash = _hash(refresh_token)
    db.add(connector)
    await db.flush()
    return PairedConnector(connector_id=connector.id, refresh_token=refresh_token)


async def issue_access_token(db: AsyncSession, *, refresh_token: str) -> str:
    """Refresh-token → short-lived access-JWT.

    Revocation is immediate: a revoked connector's refresh token stops minting
    access tokens on its very next call, well before any already-issued
    access-JWT would have expired on its own.
    """
    if not isinstance(refresh_token, str) or not 16 <= len(refresh_token) <= 256:
        raise ConnectorPairingInvalid("refresh token is invalid")
    connector = (
        await db.execute(
            select(Connector).where(Connector.refresh_token_hash == _hash(refresh_token))
        )
    ).scalar_one_or_none()
    if connector is None or not hmac.compare_digest(
        connector.refresh_token_hash or "", _hash(refresh_token)
    ):
        raise ConnectorPairingInvalid("refresh token is invalid")
    if connector.revoked_at is not None:
        raise ConnectorPairingAlreadyUsed("connector was revoked")

    expire = _now() + timedelta(minutes=CONNECTOR_ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(connector.id), "typ": CONNECTOR_TOKEN_TYPE, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[UUID]:
    """Return the connector_id encoded in an access-JWT, or ``None`` if invalid.

    Only accepts tokens minted by :func:`issue_access_token` — the ``typ``
    claim keeps this from ever being satisfied by an ordinary user JWT.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
    if payload.get("typ") != CONNECTOR_TOKEN_TYPE:
        return None
    try:
        return UUID(str(payload["sub"]))
    except (KeyError, ValueError):
        return None


async def revoke_connector(db: AsyncSession, *, connector: Connector) -> None:
    connector.revoked_at = _now()
    db.add(connector)
    await db.flush()


async def grant_room(
    db: AsyncSession, *, connector_id: UUID, room_id: UUID, granted_by: UUID
) -> ConnectorRoomGrant:
    """Idempotent: an already-active grant is returned as-is, not duplicated."""
    existing = (
        await db.execute(
            select(ConnectorRoomGrant).where(
                ConnectorRoomGrant.connector_id == connector_id,
                ConnectorRoomGrant.room_id == room_id,
                ConnectorRoomGrant.revoked_at.is_(None),
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    grant = ConnectorRoomGrant(
        connector_id=connector_id, room_id=room_id, granted_by=granted_by
    )
    db.add(grant)
    await db.flush()
    return grant


async def revoke_room_grant(db: AsyncSession, *, connector_id: UUID, room_id: UUID) -> None:
    grants = (
        await db.execute(
            select(ConnectorRoomGrant).where(
                ConnectorRoomGrant.connector_id == connector_id,
                ConnectorRoomGrant.room_id == room_id,
                ConnectorRoomGrant.revoked_at.is_(None),
            )
        )
    ).scalars().all()
    now = _now()
    for grant in grants:
        grant.revoked_at = now
        db.add(grant)
    await db.flush()


async def has_active_grant(db: AsyncSession, *, connector_id: UUID, room_id: UUID) -> bool:
    """Used by the dispatch preflight (proxy_session.py).

    A revoked connector never has an active grant, even if the grant row
    itself was never explicitly revoked.
    """
    row = (
        await db.execute(
            select(ConnectorRoomGrant.id)
            .join(Connector, Connector.id == ConnectorRoomGrant.connector_id)
            .where(
                ConnectorRoomGrant.connector_id == connector_id,
                ConnectorRoomGrant.room_id == room_id,
                ConnectorRoomGrant.revoked_at.is_(None),
                Connector.revoked_at.is_(None),
            )
            .limit(1)
        )
    ).first()
    return row is not None


__all__ = [
    "PAIRING_CODE_TTL_SECONDS",
    "CONNECTOR_ACCESS_TOKEN_EXPIRE_MINUTES",
    "IssuedPairing",
    "PairedConnector",
    "ConnectorPairingError",
    "ConnectorPairingInvalid",
    "ConnectorPairingAlreadyUsed",
    "ConnectorPairingExpired",
    "create_connector",
    "pair",
    "issue_access_token",
    "decode_access_token",
    "revoke_connector",
    "grant_room",
    "revoke_room_grant",
    "has_active_grant",
]
