"""internal design notes §5.4 (T4) — retention for the fastest-growing tables.

* ``workspace_checkpoint`` — the last two per room. New checkpoints prune
  their room as they are written (``prune_room_checkpoints``); this sweep
  catches the backlog written before that and any room a failed write left
  behind.
* ``workspace_exec`` — the command journal, 30 days.

Messages and sessions are the product and are not touched here.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

EXEC_LOG_MAX_AGE_DAYS = 30
_FIRST_RUN_DELAY_SECONDS = 600
_INTERVAL_SECONDS = 24 * 3600


async def prune_checkpoints(db: AsyncSession) -> int:
    """Prune every room holding more than the kept number of checkpoints."""
    from app.core.workspace_checkpoint import (
        CHECKPOINTS_KEPT_PER_ROOM,
        prune_room_checkpoints,
    )
    from app.models.workspace import WorkspaceCheckpoint

    room_ids = (
        await db.execute(
            select(WorkspaceCheckpoint.room_id)
            .group_by(WorkspaceCheckpoint.room_id)
            .having(func.count(WorkspaceCheckpoint.id) > CHECKPOINTS_KEPT_PER_ROOM)
        )
    ).scalars().all()
    deleted = 0
    for room_id in room_ids:
        deleted += await prune_room_checkpoints(room_id, db)
        await db.commit()
    return deleted


async def prune_exec_log(db: AsyncSession, *, now: Optional[datetime] = None) -> int:
    """Delete command-journal rows older than EXEC_LOG_MAX_AGE_DAYS.

    No exec runs that long, so a row still marked ``running`` at that age
    is a leftover of a crash, not a live command.
    """
    from app.models.workspace import WorkspaceExec

    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=EXEC_LOG_MAX_AGE_DAYS)
    result = await db.execute(
        delete(WorkspaceExec)
        .where(WorkspaceExec.created_at < cutoff)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    return int(result.rowcount or 0)


async def sweep(db: AsyncSession) -> dict[str, int]:
    checkpoints = await prune_checkpoints(db)
    execs = await prune_exec_log(db)
    if checkpoints or execs:
        logger.info(
            "db_retention: removed %d checkpoint(s), %d exec log row(s)",
            checkpoints, execs,
        )
    return {"checkpoints": checkpoints, "execs": execs}


async def run_retention_loop(engine) -> None:
    """Infinite loop; runs as a background asyncio task from ``main.on_startup``."""
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    delay = _FIRST_RUN_DELAY_SECONDS
    while True:
        try:
            await asyncio.sleep(delay)
            delay = _INTERVAL_SECONDS
            async with async_session() as db:
                await sweep(db)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("db_retention loop error: %s", exc)
