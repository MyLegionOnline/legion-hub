"""Settings → Providers: the user's provider connections, stored per account.

Mounted under ``/api/v1/providers/connections``.  Every row belongs to the
caller; another user's id answers 404, never 403, so ids cannot be probed.

``POST /import`` is the one-time hand-over from the localStorage list the
Settings page kept before this table existed: each browser that still holds
such a list uploads it once and then forgets it.  The lists are merged — a
user may have set up different connections at home and at the office — and
an entry the account already has (same provider, name, base URL and key) is
skipped, so a second tab racing the first cannot create a duplicate set.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api import deps
from app.api.v1.providers import PROVIDER_CONFIGS
from app.db.session import get_session
from app.models.provider_connection import ProviderConnection
from app.models.secret import SecretKind, SecretState, UserSecret
from app.models.user import User

router = APIRouter()

# "openai_chatgpt" is paused until browser OAuth client/callback registration is configured.
KNOWN_PROVIDERS = frozenset(PROVIDER_CONFIGS) | {"anthropic"}
_MAX_CONNECTIONS = 50
_MAX_MODELS = 5000
_STATUSES = ("idle", "ok", "error")


def _clean_optional(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.strip()
    return value or None


class _ConnectionFields(BaseModel):
    name: Optional[str] = Field(default=None, max_length=60)
    base_url: Optional[str] = Field(default=None, max_length=512)
    secret_id: Optional[UUID] = None
    selected_model: Optional[str] = Field(default=None, max_length=256)
    models: Optional[list[dict[str, Any]]] = Field(default=None, max_length=_MAX_MODELS)
    status: Optional[str] = None
    error_msg: Optional[str] = None
    session_logging: Optional[bool] = None
    is_default: Optional[bool] = None

    @field_validator("name", "selected_model")
    @classmethod
    def _strip(cls, value: Optional[str]) -> Optional[str]:
        return _clean_optional(value)

    @field_validator("base_url")
    @classmethod
    def _url(cls, value: Optional[str]) -> Optional[str]:
        value = _clean_optional(value)
        if value is not None and not value.lower().startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return value

    @field_validator("status")
    @classmethod
    def _status(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in _STATUSES:
            raise ValueError(f"status must be one of {_STATUSES}")
        return value

    @field_validator("error_msg")
    @classmethod
    def _error(cls, value: Optional[str]) -> Optional[str]:
        value = _clean_optional(value)
        return value[:500] if value else None


class ConnectionCreate(_ConnectionFields):
    provider: str

    @field_validator("provider")
    @classmethod
    def _provider(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in KNOWN_PROVIDERS:
            raise ValueError(f"unknown provider: {value}")
        return value


class ConnectionUpdate(_ConnectionFields):
    pass


class ConnectionImport(BaseModel):
    connections: list[ConnectionCreate] = Field(max_length=_MAX_CONNECTIONS)


class ConnectionOut(BaseModel):
    id: UUID
    provider: str
    name: Optional[str]
    base_url: Optional[str]
    secret_id: Optional[UUID]
    selected_model: Optional[str]
    models: list[dict[str, Any]]
    status: str
    error_msg: Optional[str]
    session_logging: bool
    is_default: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


async def _list(db: AsyncSession, user_id: UUID) -> list[ProviderConnection]:
    rows = await db.execute(
        select(ProviderConnection)
        .where(ProviderConnection.user_id == user_id)
        .order_by(ProviderConnection.created_at, ProviderConnection.id)
    )
    return list(rows.scalars().all())


async def _get_owned(db: AsyncSession, user_id: UUID, connection_id: UUID) -> ProviderConnection:
    row = await db.get(ProviderConnection, connection_id)
    if row is None or row.user_id != user_id:
        raise HTTPException(status_code=404, detail="Provider connection not found")
    return row


async def _usable_secret(db: AsyncSession, user_id: UUID, secret_id: UUID | None) -> bool:
    if secret_id is None:
        return True
    secret = await db.get(UserSecret, secret_id)
    return (
        secret is not None
        and secret.owner_id == user_id
        and secret.kind == SecretKind.LLM_API_KEY
        and secret.state != SecretState.DELETED
    )


async def _clear_default(db: AsyncSession, user_id: UUID, keep: UUID | None = None) -> None:
    stmt = (
        update(ProviderConnection)
        .where(ProviderConnection.user_id == user_id, ProviderConnection.is_default.is_(True))
        .values(is_default=False)
    )
    if keep is not None:
        stmt = stmt.where(ProviderConnection.id != keep)
    await db.execute(stmt)


def _apply(row: ProviderConnection, fields: dict[str, Any]) -> None:
    for key, value in fields.items():
        if key == "models" and value is None:
            value = []
        if key in ("status", "session_logging", "is_default") and value is None:
            continue
        setattr(row, key, value)
    row.updated_at = datetime.now(timezone.utc)


def _identity(provider: str, name: str | None, base_url: str | None, secret_id: UUID | None) -> tuple:
    """What makes two connections "the same" when a browser's list is merged in."""
    return (provider, name or None, (base_url or "").rstrip("/") or None, secret_id)


async def _lock_user(db: AsyncSession, user_id: UUID) -> None:
    """Serialise writes that depend on the user's current set of rows."""
    await db.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
        {"key": f"provider_connection:{user_id}"},
    )


@router.get("", response_model=list[ConnectionOut])
async def list_connections(
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    return await _list(db, current_user.id)


@router.post("", response_model=ConnectionOut, status_code=201)
async def create_connection(
    body: ConnectionCreate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    await _lock_user(db, current_user.id)
    if len(await _list(db, current_user.id)) >= _MAX_CONNECTIONS:
        raise HTTPException(status_code=409, detail="Too many provider connections")
    if not await _usable_secret(db, current_user.id, body.secret_id):
        raise HTTPException(status_code=422, detail="Secret is not an active LLM key of this account")
    fields = body.model_dump(exclude_unset=True)
    if body.provider == "openai_chatgpt" and set(fields) - {"provider", "name", "is_default"}:
        raise HTTPException(422, detail="Connect ChatGPT through its authorization flow")
    row = ProviderConnection(user_id=current_user.id, provider=fields.pop("provider"))
    if fields.get("is_default"):
        await _clear_default(db, current_user.id)
    _apply(row, fields)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


@router.patch("/{connection_id}", response_model=ConnectionOut)
async def update_connection(
    connection_id: UUID,
    body: ConnectionUpdate,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    row = await _get_owned(db, current_user.id, connection_id)
    fields = body.model_dump(exclude_unset=True)
    if row.provider == "openai_chatgpt":
        if set(fields) - {"name", "is_default", "selected_model"}:
            raise HTTPException(422, detail="ChatGPT credentials and models are managed by authorization")
        if fields.get("selected_model") and fields["selected_model"] not in {m.get("id") for m in row.models}:
            raise HTTPException(422, detail="Select a model available to this ChatGPT account")
    if "secret_id" in fields and not await _usable_secret(db, current_user.id, fields["secret_id"]):
        raise HTTPException(status_code=422, detail="Secret is not an active LLM key of this account")
    if fields.get("is_default"):
        await _lock_user(db, current_user.id)
        await _clear_default(db, current_user.id, keep=row.id)
    _apply(row, fields)
    await db.commit()
    await db.refresh(row)
    return row


@router.delete("/{connection_id}", status_code=204)
async def delete_connection(
    connection_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    row = await _get_owned(db, current_user.id, connection_id)
    if row.provider == "openai_chatgpt":
        from app.api.v1.chatgpt_auth import disconnect_connection
        await disconnect_connection(db, row)
    await db.delete(row)
    await db.commit()
    return Response(status_code=204)


@router.post("/import", response_model=list[ConnectionOut])
async def import_connections(
    body: ConnectionImport,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    await _lock_user(db, current_user.id)
    existing = await _list(db, current_user.id)
    seen = {_identity(row.provider, row.name, row.base_url, row.secret_id) for row in existing}
    default_seen = any(row.is_default for row in existing)

    started = datetime.now(timezone.utc)
    for index, item in enumerate(body.connections):
        if item.provider == "openai_chatgpt":
            raise HTTPException(422, detail="ChatGPT connections cannot be imported from browser storage")
        fields = item.model_dump(exclude_unset=True)
        # A browser can hold a reference to a key deleted since; the
        # connection is still worth keeping, just without the dead link.
        if not await _usable_secret(db, current_user.id, fields.get("secret_id")):
            fields["secret_id"] = None
        identity = _identity(
            item.provider, fields.get("name"), fields.get("base_url"), fields.get("secret_id")
        )
        if identity in seen:
            continue
        seen.add(identity)
        if fields.get("is_default"):
            fields["is_default"] = not default_seen
            default_seen = True
        # Distinct created_at keeps the browser's order on the way back.
        row = ProviderConnection(
            user_id=current_user.id,
            provider=fields.pop("provider"),
            created_at=started + timedelta(microseconds=index),
        )
        _apply(row, fields)
        db.add(row)
    await db.commit()
    return await _list(db, current_user.id)
