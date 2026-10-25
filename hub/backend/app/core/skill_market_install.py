"""Install skills sourced from Legion Market into Hub.

Shared content model (internal design notes section 1.1): one
shared `skill` row per slug@version with owner_id NULL, same as builtin
skills — every user's UserInstalledItem for that slug points at the same
row, so market/default skills never count toward the personal quota. The
package itself is downloaded from Market and validated with the same
skill_package parser used for uploads and builtin seeding.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import or_
from sqlalchemy.future import select

from app.core import market_client
from app.core.bundled_skills import shared_skill
from app.core.skill_install import delete_stored_files, store_package_files
from app.core.skill_package import SkillValidationError, parse_zip
from app.models.market import UserInstalledItem
from app.models.skill import Skill

logger = logging.getLogger(__name__)


class MarketInstallError(Exception):
    """A Market skill package could not be installed. status_code maps to HTTP."""

    def __init__(self, detail: str, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


async def _latest_version(slug: str) -> str:
    meta = await market_client.get_skill(slug)
    versions = meta.get("versions") or []
    if not versions:
        raise MarketInstallError(f"Skill '{slug}' has no published package", status_code=404)
    return versions[0]["version"]  # Market orders versions by created_at desc


async def ensure_shared_skill(slug: str, db: AsyncSession, version: Optional[str] = None) -> Skill:
    """Return the shared (owner_id NULL) Skill row for slug@version.

    Downloads and upserts the package from Market on first use, on a version
    change, or when the same version's content changed (checksum drift) —
    mirroring core/skill_seed.py's upsert-by-name behaviour so the skill id
    (referenced by AgentSkill/UserInstalledItem) survives an update.
    """
    if version is None:
        version = await _latest_version(slug)

    existing = await shared_skill(slug, db)
    if existing and existing.version == version and existing.source == "store":
        # Version alone isn't a reliable freshness signal: Market lets an
        # operator republish new package content (e.g. add the bundled files
        # that were missing the first time) under the *same* version string.
        # A metadata-only lookup (no package download) checks whether the
        # content actually changed before trusting the cached row.
        try:
            meta = await market_client.get_skill(slug)
        except market_client.MarketClientError:
            # Market unreachable — serve the last known-good install rather
            # than break skill resolution.
            return existing
        current_checksum = next(
            (v.get("checksum") for v in (meta.get("versions") or []) if v.get("version") == version),
            None,
        )
        if current_checksum is None or existing.checksum == current_checksum:
            return existing
        logger.info(
            "skill_market_install: %s@%s content changed under the same version "
            "(checksum drift) — resyncing", slug, version,
        )

    try:
        data, checksum = await market_client.fetch_package(slug, version)
    except market_client.MarketNotFoundError as exc:
        raise MarketInstallError(f"Skill package not found: {slug}@{version}", status_code=404) from exc
    except market_client.MarketAccessError as exc:
        raise MarketInstallError(
            f"Skill '{slug}' is a paid skill that this Hub is not entitled to download", status_code=403
        ) from exc
    except market_client.MarketClientError as exc:
        raise MarketInstallError(f"Marketplace is unavailable: {exc}", status_code=502) from exc

    actual_checksum = hashlib.sha256(data).hexdigest()
    if checksum and checksum.lower() != actual_checksum.lower():
        raise MarketInstallError(f"Checksum mismatch for {slug}@{version}", status_code=502)

    try:
        package = parse_zip(data)
    except SkillValidationError as exc:
        raise MarketInstallError(f"Invalid package for {slug}@{version}: {exc.detail}", status_code=502) from exc
    if package.name != slug:
        raise MarketInstallError(
            f"Package name '{package.name}' does not match slug '{slug}'", status_code=502
        )

    if existing:
        await delete_stored_files(existing, db)
        skill = existing
        skill.display_name = package.display_name
        skill.description = package.description
        skill.version = package.version
        skill.source = "store"
        skill.store_ref = f"{slug}@{package.version}"
        skill.checksum = actual_checksum
        skill.size_bytes = package.size_bytes
        skill.skill_md = package.skill_md_body
        skill.updated_at = datetime.utcnow()
        db.add(skill)
    else:
        skill = Skill(
            owner_id=None,
            name=package.name,
            display_name=package.display_name,
            description=package.description,
            version=package.version,
            source="store",
            store_ref=f"{slug}@{package.version}",
            checksum=actual_checksum,
            size_bytes=package.size_bytes,
            skill_md=package.skill_md_body,
        )
        db.add(skill)
        await db.flush()  # assign skill.id for SkillFile rows

    await store_package_files(skill, package, db)
    await db.commit()
    await db.refresh(skill)
    logger.info("Market skill %s installed at v%s", slug, package.version)
    return skill


async def install_market_skill_for_user(
    user_id: UUID,
    market_item: dict[str, Any],
    db: AsyncSession,
) -> UserInstalledItem:
    """Idempotent: a repeat call for the same user/market_item_id returns the
    existing installation without re-touching the shared skill content."""
    market_item_id = market_item["id"] if isinstance(market_item["id"], UUID) else UUID(str(market_item["id"]))
    slug = market_item["slug"]

    existing = (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == user_id,
                UserInstalledItem.market_item_id == market_item_id,
            )
        )
    ).scalar_one_or_none()
    if existing is None:
        existing = await _unlinked_bundled_install(user_id, slug, db)
        if existing is not None:
            existing.market_item_id = market_item_id
            db.add(existing)
            await db.commit()
            await db.refresh(existing)
    if existing is not None:
        return existing

    shared = await ensure_shared_skill(slug, db, version=market_item.get("version"))

    item = UserInstalledItem(
        user_id=user_id,
        type="skill",
        is_custom=False,
        market_item_id=market_item_id,
        market_slug=slug,
        name=market_item.get("name") or slug,
        skill_id=shared.id,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item


async def _unlinked_bundled_install(user_id: UUID, slug: str, db: AsyncSession) -> Optional[UserInstalledItem]:
    """A skill installed from the Hub's bundle before Market's item id was
    known (internal design notes stage 3). Matched by slug so a later
    install from Market adopts it instead of creating a duplicate."""
    return (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == user_id,
                UserInstalledItem.type == "skill",
                UserInstalledItem.is_custom.is_(False),
                UserInstalledItem.market_slug == slug,
                UserInstalledItem.market_item_id.is_(None),
            )
        )
    ).scalars().first()


async def install_shared_skill_for_user(
    user_id: UUID,
    slug: str,
    db: AsyncSession,
    market_item_id: Optional[UUID] = None,
) -> Optional[UserInstalledItem]:
    """Link the user to the shared skill row already in the Hub DB — no
    Market round-trip (internal design notes stage 3). Used for starter
    skills and starter agents at registration.

    Returns None when the Hub has no shared row for `slug` yet (a Market-only
    default that core/default_catalog.py has not downloaded). Idempotent: an
    existing install for the slug is returned, and gets `market_item_id`
    filled in if it was installed before the id was known.
    """
    conditions = [UserInstalledItem.market_slug == slug]
    if market_item_id is not None:
        conditions.append(UserInstalledItem.market_item_id == market_item_id)
    existing = (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == user_id,
                UserInstalledItem.type == "skill",
                UserInstalledItem.is_custom.is_(False),
                or_(*conditions),
            )
        )
    ).scalars().first()
    if existing is not None:
        if existing.market_item_id is None and market_item_id is not None:
            existing.market_item_id = market_item_id
            db.add(existing)
            await db.commit()
            await db.refresh(existing)
        return existing

    shared = await shared_skill(slug, db)
    if shared is None:
        return None

    item = UserInstalledItem(
        user_id=user_id,
        type="skill",
        is_custom=False,
        market_item_id=market_item_id,
        market_slug=slug,
        name=shared.display_name or slug,
        skill_id=shared.id,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return item
