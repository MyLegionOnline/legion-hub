"""Auto-provision default skills and MCPs for a user (stage 3 of
internal design notes, made Market-free by stage 3 of
internal design notes).

reconcile_default_skills() is called from registration, as a cheap safety
net on every GET /api/v1/skills/, and for every user by the default-catalog
sync (core/default_catalog.py). It never calls Market: the default list
comes from that module's snapshot (or the shipped manifest before the first
sync) and the skills are linked to shared rows already in the Hub DB.

Idempotent via user_default_provision: a default is only ever installed once
per user — deleting it afterwards must not bring it back, and flagging a 5th
market item is_default must reach existing users exactly once. The ledger is
matched by slug as well as by Market item id, because a skill installed from
the bundle may not know its Market id yet.

Never raises: provisioning must not break registration or the skills list.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core import default_catalog
from app.core.skill_market_install import install_shared_skill_for_user
from app.models.market import UserDefaultProvision, UserInstalledItem

logger = logging.getLogger(__name__)

SKILL_DEFAULTS_PROVISION = os.getenv("SKILL_DEFAULTS_PROVISION", "1") == "1"

# Ledger key for a bundled default whose Market item id is not known yet.
# Stable per slug, so the slug-or-id ledger check stays exact.
_BUNDLED_LEDGER_NS = uuid.UUID("6b1f3a52-2f0e-4c1e-9a77-0c5d1a7e3b10")


def _ledger_item_id(slug: str) -> UUID:
    return default_catalog.market_item_id(slug) or uuid.uuid5(_BUNDLED_LEDGER_NS, slug)


async def _provisioned(user_id: UUID, slugs: list[str], ids: list[UUID], db: AsyncSession) -> tuple[set[str], set[UUID]]:
    conditions = [UserDefaultProvision.market_slug.in_(slugs)]
    if ids:
        conditions.append(UserDefaultProvision.market_item_id.in_(ids))
    res = await db.execute(
        select(UserDefaultProvision.market_slug, UserDefaultProvision.market_item_id).where(
            UserDefaultProvision.user_id == user_id, or_(*conditions)
        )
    )
    rows = res.all()
    return {row[0] for row in rows if row[0]}, {row[1] for row in rows}


def default_skill_slugs() -> set[str]:
    """Slugs shown as "default" on the Skills tab. Never calls Market."""
    return set(default_catalog.default_skill_slugs())


async def reconcile_default_skills(user_id: UUID, db: AsyncSession) -> None:
    if not SKILL_DEFAULTS_PROVISION:
        return

    slugs = default_catalog.default_skill_slugs()
    if not slugs:
        return

    ids = {slug: _ledger_item_id(slug) for slug in slugs}
    try:
        done_slugs, done_ids = await _provisioned(user_id, slugs, list(ids.values()), db)
    except Exception:
        logger.exception("reconcile_default_skills: failed to read ledger for user %s", user_id)
        return

    for slug in slugs:
        if slug in done_slugs or ids[slug] in done_ids:
            continue
        try:
            installed = await install_shared_skill_for_user(
                user_id, slug, db, market_item_id=default_catalog.market_item_id(slug)
            )
        except Exception:
            await db.rollback()
            logger.exception(
                "reconcile_default_skills: unexpected error installing default skill %s for user %s",
                slug, user_id,
            )
            continue
        if installed is None:
            # A Market-only default the sync has not downloaded yet; the next
            # reconcile after the sync picks it up.
            continue

        try:
            db.add(UserDefaultProvision(user_id=user_id, market_item_id=ids[slug], market_slug=slug))
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception(
                "reconcile_default_skills: failed to record ledger entry for %s/%s", user_id, slug
            )


async def install_market_mcp_for_user(
    user_id: UUID,
    market_item: dict[str, Any],
    db: AsyncSession,
) -> UserInstalledItem:
    """Install one Market MCP into a user's global registry.

    The catalog item is already validated and fetched by the caller. Keeping
    this helper beside default provisioning makes the automatic path use the
    same Hub-owned snapshot as the manual MCP install endpoint.
    """
    market_item_id = UUID(str(market_item["id"]))
    existing = (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == user_id,
                UserInstalledItem.market_item_id == market_item_id,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    slug = str(market_item["slug"])
    isolation = market_item.get("isolation")
    item = UserInstalledItem(
        user_id=user_id,
        type="mcp",
        is_custom=False,
        market_item_id=market_item_id,
        market_slug=slug,
        name=str(market_item.get("name") or slug),
        isolation=isolation if isolation in {"stateless", "tenant_aware"} else None,
    )
    db.add(item)

    # Cache the price already present in the catalog. A missing snapshot would
    # otherwise make the first room discovery perform an avoidable Market
    # round-trip and could hide the tool while Market is temporarily down.
    from app.core import mcp_credits

    await mcp_credits.store_price_snapshot(
        db,
        market_item_id=market_item_id,
        market_slug=slug,
        call_price_mcr=int(market_item.get("call_price_mcr") or 0),
        tool_price_overrides=market_item.get("tool_price_overrides") or {},
    )
    await db.commit()
    await db.refresh(item)
    return item


async def reconcile_default_mcps(user_id: UUID, db: AsyncSession) -> None:
    """Provision default Market MCPs once per user.

    The ledger intentionally survives removal of the installed item: a user
    who opted out must not have the MCP silently recreated on the next chat.
    The default list is default_catalog's snapshot: before the first
    successful Market sync no default MCP is known and this is a no-op.
    """
    if not SKILL_DEFAULTS_PROVISION or os.getenv("MCP_DEFAULTS_PROVISION", "1") != "1":
        return

    defaults = default_catalog.default_mcp_items()
    if not defaults:
        return

    try:
        market_item_ids = [UUID(str(item["id"])) for item in defaults]
        slugs = [str(item.get("slug")) for item in defaults]
        done_slugs, already_provisioned = await _provisioned(user_id, slugs, market_item_ids, db)
    except Exception:
        logger.exception("reconcile_default_mcps: failed to read ledger for user %s", user_id)
        return

    for item in defaults:
        try:
            item_id = UUID(str(item["id"]))
        except (KeyError, TypeError, ValueError):
            logger.warning("reconcile_default_mcps: skipping malformed catalog item for user %s", user_id)
            continue
        slug = item.get("slug")
        if item_id in already_provisioned or slug in done_slugs:
            continue

        try:
            await install_market_mcp_for_user(user_id, item, db)
        except Exception:
            await db.rollback()
            logger.exception(
                "reconcile_default_mcps: failed to install default MCP %s for user %s",
                slug, user_id,
            )
            continue

        try:
            db.add(UserDefaultProvision(user_id=user_id, market_item_id=item_id, market_slug=slug))
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception(
                "reconcile_default_mcps: failed to record ledger entry for %s/%s", user_id, slug
            )
