"""Skills shipped inside the Hub image (internal design notes stage 3).

backend/builtin_skills/ is the only source of the starter skills: the Hub
seeds them from here on startup and ships them under its own license.
Market does not republish them, and core/default_catalog.py never asks
Market about them. manifest.json lists the shipped slugs; `default: true`
ones are installed for every new user.

Registration therefore never talks to Market: the shared skill rows
(owner_id NULL) already exist when the user is created. A new version
arrives with a new Hub image.

Seeding rules, per shipped skill:
  - no shared row with that name         → created (source='store')
  - row older than the shipped version   → content replaced in place (the
                                            skill id survives, so agent and
                                            install links keep working)
  - row same or newer (e.g. an earlier
    Market sync)                         → left alone

`checksum` stays NULL on a row seeded from the bundle.

Never raises: a broken skill folder must not block startup.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

from app.core.skill_install import delete_stored_files, store_package_files
from app.core.skill_package import SkillValidationError, build_package
from app.models.skill import Skill

logger = logging.getLogger(__name__)

# backend/builtin_skills/ (this file is backend/app/core/bundled_skills.py)
BUILTIN_SKILLS_DIR = Path(__file__).resolve().parents[2] / "builtin_skills"
MANIFEST_NAME = "manifest.json"

# Serialises writers of shared skill rows across Hub processes: owner_id is
# NULL on these rows, so uq_skill_owner_name does not stop two processes
# from inserting the same slug twice.
_SHARED_SKILL_LOCK_KEY = 0x5C111B0D


@dataclass(frozen=True)
class BundledSkill:
    slug: str
    default: bool


@lru_cache(maxsize=1)
def load_manifest() -> tuple[BundledSkill, ...]:
    path = BUILTIN_SKILLS_DIR / MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return tuple(
            BundledSkill(slug=str(entry["slug"]), default=bool(entry.get("default", False)))
            for entry in raw.get("skills", [])
        )
    except FileNotFoundError:
        logger.warning("Bundled skills manifest not found: %s", path)
    except (ValueError, KeyError, TypeError) as exc:
        logger.error("Bundled skills manifest %s is invalid: %s", path, exc)
    return ()


def bundled_slugs() -> list[str]:
    return [s.slug for s in load_manifest()]


def bundled_default_slugs() -> list[str]:
    return [s.slug for s in load_manifest() if s.default]


def version_key(version: Optional[str]) -> tuple:
    """Order "1.10.0" after "1.9.0". Non-numeric parts compare as text after
    every numeric part, which is enough for the x.y.z versions skills use."""
    parts = re.split(r"[.\-+]", (version or "").strip())
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts if p)


async def lock_shared_skills(db: AsyncSession) -> None:
    """Transaction-scoped lock around a shared-row upsert (released by the
    commit that follows). No-op off Postgres, e.g. in SQLite-backed tests."""
    try:
        dialect = db.get_bind().dialect.name
    except Exception:
        return
    if dialect == "postgresql":
        await db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _SHARED_SKILL_LOCK_KEY})


async def shared_skill(slug: str, db: AsyncSession) -> Optional[Skill]:
    """The shared row for `slug`. A DB from before legion_skills_centralization
    stage 6 can hold both a 'builtin' and a 'store' row of the same name; the
    store row is the live one (skill_legacy_migration moved agents onto it)."""
    res = await db.execute(select(Skill).where(Skill.owner_id.is_(None), Skill.name == slug))
    rows = res.scalars().all()
    return next((r for r in rows if r.source != "builtin"), rows[0] if rows else None)


def _read_skill_dir(skill_dir: Path) -> list[tuple[str, bytes]]:
    return [
        (p.relative_to(skill_dir).as_posix(), p.read_bytes())
        for p in sorted(skill_dir.rglob("*"))
        if p.is_file()
    ]


async def _seed_one(slug: str, db: AsyncSession) -> None:
    package = build_package(_read_skill_dir(BUILTIN_SKILLS_DIR / slug))
    if package.name != slug:
        raise SkillValidationError(f"SKILL.md name '{package.name}' does not match folder '{slug}'")

    await lock_shared_skills(db)
    existing = await shared_skill(slug, db)
    if existing is not None and version_key(existing.version) >= version_key(package.version):
        if existing.source == "builtin":
            # Pre-stage-3 builtin seed: same content, now a regular store row
            # that Market can update.
            existing.source = "store"
            existing.store_ref = f"{slug}@{existing.version}"
            db.add(existing)
            await db.commit()
        else:
            await db.rollback()  # release the lock
        return

    if existing is not None:
        await delete_stored_files(existing, db)
        skill = existing
        action = "updated"
    else:
        skill = Skill(owner_id=None, name=slug)
        db.add(skill)
        action = "created"
    skill.display_name = package.display_name
    skill.description = package.description
    skill.version = package.version
    skill.source = "store"
    skill.store_ref = f"{slug}@{package.version}"
    skill.checksum = None
    skill.size_bytes = package.size_bytes
    skill.skill_md = package.skill_md_body
    skill.updated_at = datetime.utcnow()
    db.add(skill)
    await db.flush()  # assign skill.id for SkillFile rows / S3 keys

    await store_package_files(skill, package, db)
    await db.commit()
    logger.info("Bundled skill %s %s (v%s)", slug, action, package.version)


async def seed_bundled_skills(engine) -> None:
    """Upsert every skill listed in the manifest. Never raises."""
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as db:
        for slug in bundled_slugs():
            if not (BUILTIN_SKILLS_DIR / slug / "SKILL.md").is_file():
                logger.error("Bundled skill %s listed in the manifest has no SKILL.md", slug)
                continue
            try:
                await _seed_one(slug, db)
            except SkillValidationError as exc:
                await db.rollback()
                logger.error("Bundled skill %s is invalid: %s", slug, exc.detail)
            except Exception:
                await db.rollback()
                logger.exception("Failed to seed bundled skill %s", slug)
