"""Starter agents every new account gets.

A freshly registered user lands on an empty Agents list and has to rebuild
the same planner / coders / writer / devops set on every Hub deployment.
provision_default_agents() gives them that set pre-filled: nickname, role
summary and the Market skill each role relies on. Provider, model, key and
prompt are deliberately left empty — those are the user's choices.

Called once, from auth.register, right after reconcile_default_skills (so the
default skills are already installed and are just looked up here). Only new
users get the starters: existing accounts already built their own agents.

Never calls Market (internal design notes stage 3): every starter skill,
including the non-default skill-vps-deploy, ships with the Hub image
(core/bundled_skills.py). Never raises: a missing skill leaves the agent
without it (the user can attach it later), it never blocks registration.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core import default_catalog
from app.core.avatar_storage import (
    DatabaseAvatarBackend,
    avatar_key,
    get_avatar_storage_backend,
    normalize_avatar,
)
from app.core.skill_market_install import install_shared_skill_for_user
from app.models.agent import Agent
from app.models.agent_avatar import AgentAvatar
from app.models.market import AgentSkill, UserInstalledItem
from app.models.user_quota import UserQuota

logger = logging.getLogger(__name__)

AGENT_DEFAULTS_PROVISION = os.getenv("AGENT_DEFAULTS_PROVISION", "1") == "1"

# (nickname, role summary, Market skill slug)
DEFAULT_AGENTS: tuple[tuple[str, str, str], ...] = (
    (
        "Lily",
        "Planner: works out the solution plan, splits it into tasks and assigns "
        "a responsible agent to each one.",
        "skill-arch",
    ),
    (
        "Sara",
        "Backend developer: designs and writes the server side — APIs, business "
        "logic, data storage.",
        "skill-coder",
    ),
    (
        "Paul",
        "Frontend developer: designs and writes the user interface.",
        "skill-coder",
    ),
    (
        "Edgar",
        "Writer: produces texts to the given brief.",
        "skill-writer",
    ),
    (
        "Kevin",
        "DevOps: deploys and runs services on the Room's servers.",
        "skill-vps-deploy",
    ),
)


# Default portraits ship with the Hub image, like the starter skills:
# backend/builtin_avatars/<nickname lowercased>.jpg (this file is
# backend/app/core/agent_defaults.py). Stored downscaled — they are small
# round badges, and the 10 MiB per-owner avatar quota must stay free.
BUILTIN_AVATARS_DIR = Path(__file__).resolve().parents[2] / "builtin_avatars"
DEFAULT_AVATAR_MAX_PX = 256


def _uid(nickname: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", nickname.lower()).strip("-")[:24] or "agent"
    return f"{base}-{secrets.token_hex(3)}"


async def _installed_skill(user_id: UUID, slug: str, db: AsyncSession) -> Optional[UserInstalledItem]:
    """The user's installation of a starter skill, linking the shared row
    shipped with the Hub if missing (skill-vps-deploy is not a default, so
    register never put it there). None when the Hub has no such skill."""
    existing = (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == user_id,
                UserInstalledItem.type == "skill",
                UserInstalledItem.market_slug == slug,
                UserInstalledItem.status == "active",
            )
        )
    ).scalars().first()
    if existing is not None:
        return existing

    try:
        installed = await install_shared_skill_for_user(
            user_id, slug, db, market_item_id=default_catalog.market_item_id(slug)
        )
    except Exception:
        await db.rollback()
        logger.exception("provision_default_agents: failed to install skill %s for user %s", slug, user_id)
        return None
    if installed is None:
        logger.warning("provision_default_agents: skill %s is not shipped with this Hub", slug)
    return installed


async def _attach_default_avatar(agent: Agent, db: AsyncSession) -> None:
    """Give the starter agent its bundled portrait. Never raises: a missing
    or undecodable file just leaves the agent with the initials circle."""
    try:
        path = BUILTIN_AVATARS_DIR / f"{agent.nickname.lower()}.jpg"
        if not path.is_file():
            return
        data = normalize_avatar(path.read_bytes(), max_dim=DEFAULT_AVATAR_MAX_PX)
        backend = get_avatar_storage_backend()
        use_db = isinstance(backend, DatabaseAvatarBackend)
        if not use_db:
            await backend.put(avatar_key(agent.owner_id, agent.id), data)
        quota = await db.get(UserQuota, agent.owner_id)
        if quota is None:
            quota = UserQuota(user_id=agent.owner_id)
        quota.avatars_used_bytes = (quota.avatars_used_bytes or 0) + len(data)
        db.add(quota)
        now = datetime.utcnow()
        db.add(
            AgentAvatar(
                agent_id=agent.id,
                mime="image/png",
                content=base64.b64encode(data).decode("ascii") if use_db else "",
                size_bytes=len(data),
                updated_at=now,
            )
        )
        agent.has_avatar = True
        agent.avatar_updated_at = now
    except Exception:
        logger.exception("provision_default_agents: failed to attach avatar to %s", agent.nickname)


async def provision_default_agents(user_id: UUID, db: AsyncSession) -> None:
    if not AGENT_DEFAULTS_PROVISION:
        return

    try:
        taken = set(
            (await db.execute(select(Agent.nickname).where(Agent.owner_id == user_id))).scalars().all()
        )
    except Exception:
        logger.exception("provision_default_agents: failed to read agents of user %s", user_id)
        return

    skills: dict[str, Optional[UserInstalledItem]] = {}
    for nickname, role, slug in DEFAULT_AGENTS:
        if nickname in taken:
            continue
        if slug not in skills:
            skills[slug] = await _installed_skill(user_id, slug, db)
        try:
            agent = Agent(
                nickname=nickname,
                agent_uid=_uid(nickname),
                agent_type="internal",
                short_description=role,
                owner_id=user_id,
                token=secrets.token_urlsafe(32),
            )
            db.add(agent)
            await db.flush()
            await _attach_default_avatar(agent, db)
            installed = skills[slug]
            if installed is not None:
                db.add(AgentSkill(agent_id=agent.id, user_installed_item_id=installed.id))
            await db.commit()
        except Exception:
            await db.rollback()
            logger.exception("provision_default_agents: failed to create %s for user %s", nickname, user_id)
