"""One-time backfill for stage 4 of internal design notes.

Before this stage a Skill row was the whole story: builtin (owner_id NULL)
or owned (owner_id = user). GET /skills/ now serves UserInstalledItem rows
instead, so two classes of pre-existing data need a matching installed item:

  - skills uploaded before this feature shipped (owned, source='upload', no
    UserInstalledItem yet) — each gets one, is_custom=True.
  - users who registered before market defaults existed — reconciled for
    everyone by the default-catalog sync in the background
    (core/default_catalog.py, internal design notes stage 3), no longer
    here: startup must not wait on per-user work or on Market.

Idempotent (skips rows that already have their installed item) and safe to
run on every startup. Never raises: a broken row must not block startup.
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

from app.models.market import UserInstalledItem
from app.models.skill import Skill

logger = logging.getLogger(__name__)


async def _backfill_uploaded_items(db: AsyncSession) -> int:
    res = await db.execute(
        select(Skill).where(Skill.owner_id.is_not(None), Skill.source == "upload")
    )
    created = 0
    for skill in res.scalars().all():
        existing = (
            await db.execute(
                select(UserInstalledItem.id).where(
                    UserInstalledItem.user_id == skill.owner_id,
                    UserInstalledItem.skill_id == skill.id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        db.add(
            UserInstalledItem(
                user_id=skill.owner_id,
                type="skill",
                is_custom=True,
                skill_id=skill.id,
                name=skill.display_name or skill.name,
            )
        )
        created += 1
    if created:
        await db.commit()
    return created


async def backfill_skill_installed_items(engine) -> None:
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with maker() as db:
        try:
            created = await _backfill_uploaded_items(db)
            if created:
                logger.info(
                    "Backfilled %d UserInstalledItem row(s) for pre-existing skill uploads", created
                )
        except Exception:
            await db.rollback()
            logger.exception("Failed to backfill uploaded skill installed items")
