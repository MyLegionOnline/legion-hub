"""What the Hub knows about Market's default items (internal design notes
stage 3).

Registration and the Skills tab read the default-item list from this
module's in-process snapshot and never call Market. One background loop per
Hub process refreshes it every DEFAULT_SKILLS_SYNC_HOURS:

  1. two catalogue calls — default skills and default MCPs;
  2. one metadata call per Market-only default skill — downloads a package
     only when Market has a newer version, or the same version with
     different content;
  3. on the first pass, and whenever the default set changes, reconciles
     every user once so a newly flagged default reaches existing accounts.

The skills shipped with the Hub (core/bundled_skills.py) are not Market's:
Market does not republish them, so the manifest's defaults are defaults
whether or not Market was reached, and Market is never asked about them.
No default MCPs are known until the first successful sync.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

from app.core import bundled_skills, market_client
from app.core.skill_market_install import MarketInstallError, ensure_shared_skill
from app.models.user import User

logger = logging.getLogger(__name__)

SYNC_INTERVAL_SEC = float(os.getenv("DEFAULT_SKILLS_SYNC_HOURS", "6")) * 3600
# Market unreachable: try again sooner than the full interval.
SYNC_RETRY_SEC = float(os.getenv("DEFAULT_SKILLS_SYNC_RETRY_SEC", "300"))

_state: dict[str, Any] = {
    "skills": None,   # list of Market default skill items, None = never synced
    "mcps": None,     # list of Market default MCP items
    "ids": {},        # slug -> Market item id, for every skill seen by the sync
}


def reset() -> None:
    _state["skills"] = None
    _state["mcps"] = None
    _state["ids"] = {}


def is_synced() -> bool:
    return _state["skills"] is not None


def default_skill_slugs() -> list[str]:
    """The shipped defaults, then Market's default skills once synced."""
    market = [str(item["slug"]) for item in _state["skills"] or []]
    return list(dict.fromkeys(bundled_skills.bundled_default_slugs() + market))


def default_mcp_items() -> list[dict[str, Any]]:
    return list(_state["mcps"] or [])


def market_item_id(slug: str) -> Optional[UUID]:
    value = _state["ids"].get(slug)
    return UUID(str(value)) if value is not None else None


def _checksum_of(meta: dict[str, Any], version: str) -> Optional[str]:
    return next(
        (v.get("checksum") for v in (meta.get("versions") or []) if v.get("version") == version),
        None,
    )


async def _sync_skill(slug: str, maker) -> Optional[UUID]:
    """Bring the shared row for `slug` up to Market's current version.
    Returns Market's item id, or None if Market does not have the skill."""
    try:
        meta = await market_client.get_skill(slug)
    except market_client.MarketNotFoundError:
        return None

    version = str(meta.get("version") or "")
    checksum = _checksum_of(meta, version)
    async with maker() as db:
        await bundled_skills.lock_shared_skills(db)
        row = await bundled_skills.shared_skill(slug, db)
        if row is not None and (
            bundled_skills.version_key(row.version) > bundled_skills.version_key(version)
            or (row.version == version and checksum is not None and row.checksum == checksum)
        ):
            # Up to date, or the image ships a newer version than Market has
            # published — never downgrade.
            await db.rollback()
        elif version:
            try:
                await ensure_shared_skill(slug, db, version=version)
            except MarketInstallError as exc:
                await db.rollback()
                logger.warning("default_catalog: cannot update %s@%s: %s", slug, version, exc.detail)
    return UUID(str(meta["id"]))


async def reconcile_all_users(maker) -> None:
    from app.core.skill_defaults import reconcile_default_mcps, reconcile_default_skills

    async with maker() as db:
        user_ids = (await db.execute(select(User.id))).scalars().all()
    for user_id in user_ids:
        async with maker() as db:
            try:
                await reconcile_default_skills(user_id, db)
                await reconcile_default_mcps(user_id, db)
            except Exception:
                logger.exception("default_catalog: failed to reconcile defaults for user %s", user_id)


def _default_set() -> tuple[frozenset[str], frozenset[str]]:
    return (
        frozenset(default_skill_slugs()),
        frozenset(str(item.get("slug")) for item in default_mcp_items()),
    )


async def sync_once(engine) -> bool:
    """One pass. Returns False when Market was unreachable (the snapshot and
    the shipped skills stay as they were)."""
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        skills = (await market_client.list_catalog(type="skill", is_default=True, page=1, page_size=100)).get("items", [])
        mcps = (await market_client.list_catalog(type="mcp", is_default=True, page=1, page_size=100)).get("items", [])
        # A Market from before it stopped republishing the shipped skills
        # still lists them; the Hub's own copy stays the only source.
        shipped = set(bundled_skills.bundled_slugs())
        skills = [item for item in skills if str(item["slug"]) not in shipped]
        ids: dict[str, UUID] = {}
        for slug in (str(item["slug"]) for item in skills):
            item_id = await _sync_skill(slug, maker)
            if item_id is not None:
                ids[slug] = item_id
    except market_client.MarketClientError as exc:
        logger.warning("default_catalog: Market unavailable, keeping the current skills: %s", exc)
        return False

    _state["skills"] = skills
    _state["mcps"] = mcps
    _state["ids"] = {slug: str(item_id) for slug, item_id in ids.items()}
    logger.info("default_catalog: synced %d default skill(s), %d default MCP(s)", len(skills), len(mcps))
    return True


async def run_sync_loop(engine) -> None:
    """Background task started from app startup. Never raises."""
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    reconciled: Optional[tuple[frozenset[str], frozenset[str]]] = None
    while True:
        ok = False
        try:
            ok = await sync_once(engine)
            current = _default_set()
            if current != reconciled:
                await reconcile_all_users(maker)
                reconciled = current
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("default_catalog: sync pass failed")
        await asyncio.sleep(SYNC_INTERVAL_SEC if ok else SYNC_RETRY_SEC)
