"""Stable owner → XFS project-quota mapping for workspace storage.

The application-level disk accounting in :mod:`app.core.room_storage` stays
the user-facing source of truth.  This module supplies its kernel backstop:
all existing workspace directories of one room owner receive the same XFS
project id and hard byte limit.  The workspaced service is deliberately the
only process that runs ``xfs_quota``; the Hub only owns database identity and
authorization data.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.models.room import Room
from app.models.user_quota import UserQuota


logger = logging.getLogger(__name__)

# workspaced caps ``room_ids`` per request (schemas.StorageProjectRequest).
ROOM_IDS_PER_CALL = 500


@dataclass(frozen=True)
class StorageProject:
    """The wire representation that workspaced needs for one owner."""

    project_id: int
    quota_mb: int
    room_ids: list[str]


def hard_quota_sync_enabled() -> bool:
    """Whether this deployment must establish XFS limits before writes."""
    return os.getenv("WORKSPACE_PROJECT_QUOTA_SYNC_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def finite_positive_mb(value: Any) -> Optional[int]:
    """Return a finite positive MiB value, or ``None`` for an unlimited cap."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = int(value)
    return result if result > 0 and result == value else None


async def _lock_project_allocation(owner_id: UUID, db: AsyncSession) -> None:
    """Serialize a first allocation even before a ``UserQuota`` row exists.

    A row-level lock cannot protect a row that has not been inserted yet.  The
    Cloud database is PostgreSQL, so use a transaction-scoped advisory lock
    for that narrow first-write race.  Other dialects are only used by local
    unit tests and self-hosted installs have unlimited storage by default.
    """
    bind = db.get_bind()
    if bind.dialect.name != "postgresql":
        return
    await db.execute(
        text(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended(CAST(:owner_id AS text), 0))"
        ),
        {"owner_id": str(owner_id)},
    )


async def _locked_quota(owner_id: UUID, db: AsyncSession) -> UserQuota:
    row = (
        await db.execute(
            select(UserQuota)
            .where(UserQuota.user_id == owner_id)
            .with_for_update()
        )
    ).scalar_one_or_none()
    if row is not None:
        return row

    # Cloud billing normally creates this row first.  The nested transaction
    # also makes the lifecycle path safe for a freshly provisioned owner that
    # reaches workspace start concurrently with another quota-creating path.
    try:
        async with db.begin_nested():
            row = UserQuota(user_id=owner_id)
            db.add(row)
            await db.flush()
    except IntegrityError:
        row = (
            await db.execute(
                select(UserQuota)
                .where(UserQuota.user_id == owner_id)
                .with_for_update()
            )
        ).scalar_one()
    return row


async def ensure_xfs_project_id(owner_id: UUID, db: AsyncSession) -> int:
    """Return the owner's durable XFS project id, allocating it once."""
    await _lock_project_allocation(owner_id, db)
    quota = await _locked_quota(owner_id, db)
    existing = quota.xfs_project_id
    if existing is not None:
        return int(existing)

    # This deployment is PostgreSQL-backed.  Keeping the sequence in the DB
    # prevents two Hub processes from ever allocating the same project id.
    project_id = int(
        (await db.execute(text("SELECT nextval('workspace_xfs_project_id_seq')"))).scalar_one()
    )
    quota.xfs_project_id = project_id
    db.add(quota)
    await db.flush()
    return project_id


async def project_for_owner(
    owner_id: UUID,
    db: AsyncSession,
    *,
    limits: Optional[dict[str, Any]] = None,
) -> Optional[StorageProject]:
    """Build the quota assignment for an owner, or ``None`` if unlimited.

    Listing every live room is intentional: a project quota is per owner,
    not per running room.  ``workspaced`` skips a directory that has not been
    created yet and assigns it when that room is first opened.
    """
    if limits is None:
        from app.core.workspace_quota import get_effective_limits

        limits = await get_effective_limits(owner_id, db)
    quota_mb = finite_positive_mb(limits.get("room_storage_quota_mb"))
    if quota_mb is None:
        return None

    project_id = await ensure_xfs_project_id(owner_id, db)
    room_ids = [
        str(room_id)
        for room_id in (
            await db.execute(
                select(Room.id)
                .where(Room.creator_id == owner_id, Room.deletion_started_at.is_(None))
                .order_by(Room.created_at)
            )
        ).scalars().all()
    ]
    return StorageProject(project_id=project_id, quota_mb=quota_mb, room_ids=room_ids)


async def sync_owner_project(owner_id: UUID, db: AsyncSession) -> bool:
    """Push the owner's current quota to workspaced after a plan change.

    The caller decides whether an unavailable workspaced instance should be
    fatal.  Start always repeats this assignment, so a transient failure here
    cannot permanently leave a newly paid user without the hard limit.
    """
    project = await project_for_owner(owner_id, db)
    if project is None:
        return False
    from app.core import workspaced_client as wsd

    # workspaced accepts a bounded list per call; the limit itself is
    # idempotent, so an owner with more rooms is sent in several batches.
    room_ids = project.room_ids or []
    batches = [
        room_ids[i:i + ROOM_IDS_PER_CALL]
        for i in range(0, len(room_ids), ROOM_IDS_PER_CALL)
    ] or [[]]
    for batch in batches:
        await wsd.configure_storage_project(
            project_id=project.project_id,
            quota_mb=project.quota_mb,
            room_ids=batch,
        )
    return True


async def reconcile_storage_projects(engine: AsyncEngine) -> tuple[int, int]:
    """Establish hard quotas for every existing paid room.

    Returns ``(synchronized, failed)``.  One owner's failure (a huge tree,
    a transient workspaced error) does not stop the others.  File APIs can
    create a workspace directory without starting a container, so this runs
    right after Hub startup; room creation, workspace start and billing
    repeat the assignment for their own owner, which covers the gap.
    """
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as db:
        owner_ids = list(
            (
                await db.execute(
                    select(Room.creator_id)
                    .where(Room.deletion_started_at.is_(None))
                    .distinct()
                    .order_by(Room.creator_id)
                )
            ).scalars().all()
        )

    synchronized = failed = 0
    for owner_id in owner_ids:
        async with factory() as db:
            try:
                if await sync_owner_project(owner_id, db):
                    await db.commit()
                    synchronized += 1
                else:
                    await db.rollback()
            except Exception:
                await db.rollback()
                failed += 1
                logger.exception("T9: could not synchronize XFS quota for owner %s", owner_id)
    return synchronized, failed


async def run_reconcile_loop(
    engine: AsyncEngine, *, retry_sec: float = 60.0, max_attempts: int = 30
) -> None:
    """Background startup reconciliation, retried while workspaced catches up.

    A slow ``project -s`` over a big tree or a workspaced that is still
    starting must not keep the whole Hub from serving requests.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            synchronized, failed = await reconcile_storage_projects(engine)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("T9: storage quota reconciliation attempt %d crashed", attempt)
            synchronized, failed = 0, 1
        logger.info(
            "T9: attempt %d synchronized XFS quotas for %d storage owners, %d failed",
            attempt, synchronized, failed,
        )
        if failed == 0:
            return
        await asyncio.sleep(retry_sec)
    logger.error("T9: giving up on storage quota reconciliation after %d attempts", max_attempts)
