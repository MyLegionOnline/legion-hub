"""Retire the legacy agent skill fields.

Before the Market became the source of truth an agent's skills lived in
``agent.llm_config``: ``skill_ids`` (Skill ids) plus ``skills`` (free text).
The resolver used them only while the agent had no explicit AgentSkill rows.
The fallback is gone now, so this startup migration moves what it used to
render into the current model:

  - each ``skill_ids`` entry becomes an AgentSkill row pointing at the owner's
    registry item for that skill — the exact skill, else one with the same
    name (covers frozen builtin rows replaced by shared store copies), else a
    newly created registry item so the agent keeps the skill;
  - the ``skills`` text is appended to ``llm_config["system_prompt"]``.

An agent that already had AgentSkill rows never saw its legacy fields, so they
are dropped without being carried over.  Both keys are removed either way,
which makes the migration idempotent.  Safe to run on every startup — never
raises, mirroring skill_backfill.py.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.attributes import flag_modified

from app.models.agent import Agent
from app.models.market import AgentSkill, UserInstalledItem
from app.models.skill import Skill

logger = logging.getLogger(__name__)

LEGACY_KEYS = ("skill_ids", "skills")


def _parse_uuid(raw) -> Optional[UUID]:
    try:
        return raw if isinstance(raw, UUID) else UUID(str(raw))
    except (ValueError, TypeError):
        return None


def _pick(items: list[UserInstalledItem]) -> Optional[UserInstalledItem]:
    for item in items:
        if item.status == "active":
            return item
    return items[0] if items else None


async def _registry_item_for(
    db: AsyncSession, owner_id: UUID, skill_id: UUID
) -> Optional[UserInstalledItem]:
    """The owner's registry item for a legacy skill id, created if missing."""
    exact = (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == owner_id,
                UserInstalledItem.type == "skill",
                UserInstalledItem.skill_id == skill_id,
            )
        )
    ).scalars().all()
    if exact:
        return _pick(list(exact))

    skill = await db.get(Skill, skill_id)
    # The legacy resolver only rendered shared skills or the owner's own.
    if skill is None or skill.owner_id not in (None, owner_id):
        return None

    same_name = (
        await db.execute(
            select(UserInstalledItem)
            .join(Skill, Skill.id == UserInstalledItem.skill_id)
            .where(
                UserInstalledItem.user_id == owner_id,
                UserInstalledItem.type == "skill",
                Skill.name == skill.name,
            )
        )
    ).scalars().all()
    if same_name:
        return _pick(list(same_name))

    item = UserInstalledItem(
        user_id=owner_id,
        type="skill",
        is_custom=skill.source == "upload",
        skill_id=skill.id,
        name=skill.display_name or skill.name,
    )
    db.add(item)
    await db.flush()
    logger.info(
        "Created registry item for legacy skill %s (%s) of user %s",
        skill.name, skill.id, owner_id,
    )
    return item


async def _migrate_agent(db: AsyncSession, agent: Agent) -> bool:
    """Move one agent's legacy skill fields; False when there is nothing to do."""
    cfg = dict(agent.llm_config or {})
    if not any(key in cfg for key in LEGACY_KEYS):
        return False
    raw_ids = cfg.pop("skill_ids", None) or []
    raw_text = cfg.pop("skills", None)
    text = raw_text.strip() if isinstance(raw_text, str) else ""

    has_explicit = (
        await db.execute(
            select(AgentSkill.user_installed_item_id)
            .where(AgentSkill.agent_id == agent.id)
            .limit(1)
        )
    ).scalars().first() is not None

    if not has_explicit:
        assigned: set[UUID] = set()
        for raw in raw_ids if isinstance(raw_ids, list) else []:
            skill_id = _parse_uuid(raw)
            if skill_id is None:
                continue
            item = await _registry_item_for(db, agent.owner_id, skill_id)
            if item is None or item.id in assigned:
                continue
            assigned.add(item.id)
            db.add(AgentSkill(agent_id=agent.id, user_installed_item_id=item.id))
        if text:
            prompt = (cfg.get("system_prompt") or "").rstrip()
            cfg["system_prompt"] = f"{prompt}\n\n{text}" if prompt else text

    agent.llm_config = cfg
    flag_modified(agent, "llm_config")
    db.add(agent)
    return True


async def migrate_legacy_skill_ids(engine) -> None:
    """Move every agent's legacy ``skill_ids``/``skills`` into AgentSkill rows.

    Each agent runs in its own session, so one broken row cannot block the
    rest.  Never raises — failures are logged so they cannot block startup.
    """
    maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        async with maker() as db:
            rows = (await db.execute(select(Agent.id, Agent.llm_config))).all()
    except Exception:
        logger.exception("Failed to list agents for the legacy skill migration")
        return

    pending = [
        agent_id
        for agent_id, cfg in rows
        if isinstance(cfg, dict) and any(key in cfg for key in LEGACY_KEYS)
    ]
    migrated = 0
    for agent_id in pending:
        async with maker() as db:
            try:
                agent = await db.get(Agent, agent_id)
                if agent is not None and await _migrate_agent(db, agent):
                    await db.commit()
                    migrated += 1
            except Exception:
                await db.rollback()
                logger.exception("Failed to migrate legacy skills of agent %s", agent_id)
    if migrated:
        logger.info("Moved legacy skill fields of %d agent(s) into AgentSkill", migrated)
