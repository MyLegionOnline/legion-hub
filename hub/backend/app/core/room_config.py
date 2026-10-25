"""Room configuration variables — non-secret, plaintext (Stage 1).

``internal design notes`` §2.1 / Stage 1.  Secrets live elsewhere
(``user_secret`` + ``secret_binding(resource_type=room)``) and are handled by
later stages; this module only owns :class:`RoomVariable`.

Values here are deliberately visible: they show in the UI, they go into the
agent's prompt, and (from Stage 4) they are materialised into the env of every
exec.  Anything that must never surface in a session is a secret, not a
variable.

The ``environment`` column is stored but does not change behaviour until
Stage 12.  ``None`` means "a shared value, good for any environment"; a
non-null value scopes the variable to one environment.  :func:`effective_variables`
already implements the resolve rule the later stages rely on — an exact
environment match wins over the shared (``NULL``) value.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.room_variable import RoomVariable

# Keep these in lockstep with the CHECK constraints in
# ``models/room_variable.py`` and the Stage 0 migration.
NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
ENVIRONMENT_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

MAX_NAME_LEN = 128
MAX_VALUE_LEN = 4096
MAX_DESCRIPTION_LEN = 256

# Limits (plan §Stage 1). Per-environment count and a room-wide value budget.
MAX_VARIABLES_PER_ENVIRONMENT = 100
MAX_TOTAL_VALUE_BYTES = 64 * 1024

_UNSET = object()


# ── validation ──────────────────────────────────────────────────────────────


def normalize_environment(environment: Optional[str]) -> Optional[str]:
    """Collapse empty / whitespace-only to ``None``; validate the rest."""
    if environment is None:
        return None
    environment = environment.strip()
    if not environment:
        return None
    if not ENVIRONMENT_RE.match(environment):
        raise HTTPException(
            status_code=422,
            detail=(
                "environment must match ^[a-z][a-z0-9_-]{0,31}$ "
                f"(got {environment!r})"
            ),
        )
    return environment


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not NAME_RE.match(name) or len(name) > MAX_NAME_LEN:
        raise HTTPException(
            status_code=422,
            detail="name must match ^[A-Z_][A-Z0-9_]*$ and be ≤128 chars",
        )
    return name


def _validate_value(value: str) -> str:
    if value is None:
        raise HTTPException(status_code=422, detail="value is required")
    if len(value) > MAX_VALUE_LEN:
        raise HTTPException(
            status_code=422, detail=f"value must be ≤{MAX_VALUE_LEN} chars"
        )
    return value


def _validate_description(description: Optional[str]) -> Optional[str]:
    if description is None:
        return None
    if len(description) > MAX_DESCRIPTION_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"description must be ≤{MAX_DESCRIPTION_LEN} chars",
        )
    return description


# ── limits ──────────────────────────────────────────────────────────────────


async def _enforce_limits(
    db: AsyncSession,
    room_id: UUID,
    *,
    environment: Optional[str],
    incoming_value: str,
    exclude_id: Optional[UUID] = None,
) -> None:
    """Reject a write that would push the room over its variable budgets."""
    count_stmt = select(func.count()).select_from(RoomVariable).where(
        RoomVariable.room_id == room_id
    )
    bytes_stmt = select(
        func.coalesce(func.sum(func.char_length(RoomVariable.value)), 0)
    ).where(RoomVariable.room_id == room_id)
    if environment is None:
        count_stmt = count_stmt.where(RoomVariable.environment.is_(None))
    else:
        count_stmt = count_stmt.where(RoomVariable.environment == environment)
    if exclude_id is not None:
        count_stmt = count_stmt.where(RoomVariable.id != exclude_id)
        bytes_stmt = bytes_stmt.where(RoomVariable.id != exclude_id)

    existing_count = (await db.execute(count_stmt)).scalar_one()
    if existing_count >= MAX_VARIABLES_PER_ENVIRONMENT:
        raise HTTPException(
            status_code=422,
            detail=(
                f"too many variables for this environment "
                f"(limit {MAX_VARIABLES_PER_ENVIRONMENT})"
            ),
        )

    existing_bytes = (await db.execute(bytes_stmt)).scalar_one()
    if existing_bytes + len(incoming_value.encode("utf-8")) > MAX_TOTAL_VALUE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=(
                "room variable values exceed the total budget "
                f"({MAX_TOTAL_VALUE_BYTES} bytes)"
            ),
        )


# ── CRUD ────────────────────────────────────────────────────────────────────


async def list_variables(
    db: AsyncSession,
    room_id: UUID,
    *,
    environment=_UNSET,
) -> list[RoomVariable]:
    """All variables for a room, optionally filtered to one environment.

    ``environment`` unset → every variable.  ``environment=None`` or ``""`` →
    only the shared (``NULL``) bucket.  ``environment="staging"`` → only that
    environment.
    """
    stmt = select(RoomVariable).where(RoomVariable.room_id == room_id)
    if environment is not _UNSET:
        env = normalize_environment(environment)
        if env is None:
            stmt = stmt.where(RoomVariable.environment.is_(None))
        else:
            stmt = stmt.where(RoomVariable.environment == env)
    stmt = stmt.order_by(
        RoomVariable.environment.nulls_first(), RoomVariable.name
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_variable(
    db: AsyncSession, room_id: UUID, variable_id: UUID
) -> RoomVariable:
    row = (
        await db.execute(
            select(RoomVariable).where(
                RoomVariable.id == variable_id,
                RoomVariable.room_id == room_id,
            )
        )
    ).scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail="Variable not found")
    return row


async def _assert_unique(
    db: AsyncSession,
    room_id: UUID,
    name: str,
    environment: Optional[str],
    *,
    exclude_id: Optional[UUID] = None,
) -> None:
    stmt = select(RoomVariable.id).where(
        RoomVariable.room_id == room_id,
        RoomVariable.name == name,
    )
    stmt = (
        stmt.where(RoomVariable.environment.is_(None))
        if environment is None
        else stmt.where(RoomVariable.environment == environment)
    )
    if exclude_id is not None:
        stmt = stmt.where(RoomVariable.id != exclude_id)
    if (await db.execute(stmt)).first() is not None:
        where = f"environment {environment!r}" if environment else "the shared scope"
        raise HTTPException(
            status_code=409,
            detail=f"variable {name!r} already exists in {where}",
        )


async def create_variable(
    db: AsyncSession,
    room_id: UUID,
    *,
    name: str,
    value: str,
    description: Optional[str] = None,
    environment: Optional[str] = None,
    materialize_to_env: bool = True,
    created_by: Optional[UUID] = None,
) -> RoomVariable:
    name = _validate_name(name)
    value = _validate_value(value)
    description = _validate_description(description)
    environment = normalize_environment(environment)

    await _assert_unique(db, room_id, name, environment)
    await _enforce_limits(
        db, room_id, environment=environment, incoming_value=value
    )

    row = RoomVariable(
        room_id=room_id,
        name=name,
        value=value,
        description=description,
        environment=environment,
        materialize_to_env=materialize_to_env,
        created_by=created_by,
    )
    db.add(row)
    try:
        await db.commit()
    except IntegrityError:  # unique index race
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"variable {name!r} already exists in that environment",
        )
    await db.refresh(row)
    return row


async def update_variable(
    db: AsyncSession,
    variable: RoomVariable,
    *,
    fields: dict,
) -> RoomVariable:
    """Apply the subset of provided fields (``exclude_unset`` dump)."""
    name = variable.name
    environment = variable.environment
    value = variable.value

    if "name" in fields:
        name = _validate_name(fields["name"])
    if "environment" in fields:
        environment = normalize_environment(fields["environment"])
    if "value" in fields:
        value = _validate_value(fields["value"])
    if "description" in fields:
        variable.description = _validate_description(fields["description"])
    if "materialize_to_env" in fields:
        variable.materialize_to_env = bool(fields["materialize_to_env"])

    identity_changed = name != variable.name or environment != variable.environment
    if identity_changed:
        await _assert_unique(
            db, variable.room_id, name, environment, exclude_id=variable.id
        )
    if "value" in fields or identity_changed:
        await _enforce_limits(
            db,
            variable.room_id,
            environment=environment,
            incoming_value=value,
            exclude_id=variable.id,
        )

    variable.name = name
    variable.environment = environment
    variable.value = value
    variable.updated_at = datetime.now(timezone.utc)

    db.add(variable)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"variable {name!r} already exists in that environment",
        )
    await db.refresh(variable)
    return variable


async def delete_variable(db: AsyncSession, variable: RoomVariable) -> None:
    await db.delete(variable)
    await db.commit()


# ── resolve rule (used by Stage 4+; tested now) ─────────────────────────────


async def list_used_environments(db: AsyncSession, room_id: UUID) -> list[str]:
    """Distinct non-shared environments referenced by this room's variables.

    Powers the Stage 12 environment switcher together with
    ``room_secrets.list_used_environments`` — an environment shows up there as
    soon as anything uses it, not only once it has a protection setting.
    """
    rows = (
        await db.execute(
            select(RoomVariable.environment)
            .where(
                RoomVariable.room_id == room_id,
                RoomVariable.environment.is_not(None),
            )
            .distinct()
        )
    ).scalars().all()
    return sorted(set(rows))


async def effective_variables(
    db: AsyncSession,
    room_id: UUID,
    environment: Optional[str] = None,
) -> dict[str, str]:
    """``{name: value}`` for one environment.

    The shared (``NULL``) value is the base; an entry that matches
    ``environment`` exactly overrides it.  Only ``materialize_to_env`` rows are
    included — that is what an exec env cares about.
    """
    environment = normalize_environment(environment)
    rows = (
        await db.execute(
            select(RoomVariable).where(
                RoomVariable.room_id == room_id,
                RoomVariable.materialize_to_env.is_(True),
            )
        )
    ).scalars().all()

    shared = {r.name: r.value for r in rows if r.environment is None}
    scoped = (
        {r.name: r.value for r in rows if r.environment == environment}
        if environment is not None
        else {}
    )
    return {**shared, **scoped}
