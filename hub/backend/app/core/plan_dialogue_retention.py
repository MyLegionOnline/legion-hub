"""internal design notes Stage 8 — retention sweep for dialogue archives.

The Redis journal expires on its own (``JOURNAL_TTL_SEC``). The workspace files
do not, so a daily sweep removes:

* ``.legion/drafts/{session_id}.md`` older than ``DRAFT_MAX_AGE_DAYS`` or already
  folded into a later session (``plan_draft_merged:{sid}`` set in Redis);
* ``.legion/plans/goal-{id}.md`` (+ its ``…-attachments/`` files) whose Goal no
  longer exists;
* ``.legion/plans/task-{id}.md`` (+ attachments) whose Task no longer exists.

Everything is best-effort: a workspace that is down, a missing Redis, a delete
that 404s — none of it raises, the sweep just moves on and tries again tomorrow.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

DRAFT_MAX_AGE_DAYS = 14
_INTERVAL_SECONDS = 24 * 3600

_DRAFT_RE = re.compile(r"^\.legion/drafts/(?P<sid>[^/]+)\.md$")
_GOAL_RE = re.compile(r"^\.legion/plans/goal-(?P<gid>[0-9a-fA-F-]+)(?:\.md$|-attachments/)")
_TASK_RE = re.compile(r"^\.legion/plans/task-(?P<tid>[0-9a-fA-F-]+)(?:\.md$|-attachments/)")


async def _is_merged(session_id: str) -> bool:
    from app.core import plan_dialogue as pd

    r = pd._get_redis()
    if r is None:
        return False
    try:
        return bool(await r.get(pd._merged_key(session_id)))
    except Exception:  # pragma: no cover - defensive
        return False


async def sweep_room(room_id, db: AsyncSession) -> int:
    """Delete stale draft/goal/task dialogue files for one room. Returns count."""
    from app.core import workspaced_client as wsd
    from app.models.goal import RoomGoal
    from app.models.task import Task

    try:
        files = await wsd.list_dialogue_archive_files(str(room_id))
    except Exception as exc:
        logger.debug("plan_dialogue_retention: listing failed for %s: %s", room_id, exc)
        return 0
    if not files:
        return 0

    live_goals = {
        str(g) for g in (
            await db.execute(select(RoomGoal.id).where(RoomGoal.room_id == room_id))
        ).scalars().all()
    }
    live_tasks = {
        str(t) for t in (
            await db.execute(select(Task.id).where(Task.room_id == room_id))
        ).scalars().all()
    }

    cutoff = (datetime.now(timezone.utc) - timedelta(days=DRAFT_MAX_AGE_DAYS)).timestamp()
    to_delete: list[str] = []
    for entry in files:
        path = entry.get("path") or ""
        mtime = float(entry.get("mtime") or 0.0)

        m = _DRAFT_RE.match(path)
        if m:
            if mtime and mtime < cutoff:
                to_delete.append(path)
            elif await _is_merged(m.group("sid")):
                to_delete.append(path)
            continue

        m = _GOAL_RE.match(path)
        if m:
            if m.group("gid") not in live_goals:
                to_delete.append(path)
            continue

        m = _TASK_RE.match(path)
        if m:
            if m.group("tid") not in live_tasks:
                to_delete.append(path)
            continue

    deleted = 0
    for path in to_delete:
        try:
            if await wsd.delete_workspace_file(str(room_id), path):
                deleted += 1
        except Exception as exc:
            logger.debug("plan_dialogue_retention: delete failed (%s): %s", path, exc)
    if deleted:
        logger.info("plan_dialogue_retention: swept %d file(s) from room %s", deleted, room_id)
    return deleted


async def sweep_all(db: AsyncSession) -> int:
    """Sweep every room that has at least one goal or task."""
    from app.models.goal import RoomGoal
    from app.models.task import Task

    room_ids: set = set(
        (await db.execute(select(RoomGoal.room_id).distinct())).scalars().all()
    )
    room_ids |= set(
        (await db.execute(select(Task.room_id).distinct())).scalars().all()
    )
    total = 0
    for room_id in room_ids:
        try:
            total += await sweep_room(room_id, db)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("plan_dialogue_retention: room %s failed: %s", room_id, exc)
    return total


async def run_retention_loop(engine) -> None:
    """Infinite loop; runs as a background asyncio task from ``main.on_startup``."""
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    while True:
        try:
            await asyncio.sleep(_INTERVAL_SECONDS)
            async with async_session() as db:
                await sweep_all(db)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.warning("plan_dialogue_retention loop error: %s", exc)
