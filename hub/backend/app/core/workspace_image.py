"""Resolve which Docker image a room's workspace container should run.

internal design notes Stage 5 (§7.1): most rooms get the
lightweight "base" workspace image. A room only needs the heavier "office"
image (LibreOffice headless + openpyxl/pandas/lxml/Pillow, see
hub/workspaced/workspace-image-office/Dockerfile) when at least one agent
assigned to one of its tasks has an executable bundled skill
(Skill.is_executable, stamped at install time by core/skill_package.py).
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

from sqlalchemy import or_, select
from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)

WORKSPACE_IMAGE_BASE = os.getenv("WORKSPACE_IMAGE_BASE", "legion-workspace:base")
WORKSPACE_IMAGE_OFFICE = os.getenv("WORKSPACE_IMAGE_OFFICE", "legion-workspace:office")


async def resolve_workspace_image(room_id: UUID, db: AsyncSession) -> str:
    """Base image, unless a task-assigned agent in this room has an executable skill.

    Never raises — image resolution is best-effort and falls back to the base
    image on any failure (same contract as skill_materialize.py).
    """
    from app.models.agent import Agent
    from app.models.market import AgentSkill, UserInstalledItem
    from app.models.skill import Skill
    from app.models.task import Task

    try:
        agent_ids = (
            await db.execute(
                select(Task.assigned_agent_id)
                .where(Task.room_id == room_id, Task.assigned_agent_id.is_not(None))
                .distinct()
            )
        ).scalars().all()
        if not agent_ids:
            return WORKSPACE_IMAGE_BASE

        has_office_skill = (
            await db.execute(
                select(Skill.id)
                .join(UserInstalledItem, UserInstalledItem.skill_id == Skill.id)
                .join(AgentSkill, AgentSkill.user_installed_item_id == UserInstalledItem.id)
                .join(Agent, Agent.id == AgentSkill.agent_id)
                .where(
                    AgentSkill.agent_id.in_(agent_ids),
                    UserInstalledItem.user_id == Agent.owner_id,
                    UserInstalledItem.type == "skill",
                    UserInstalledItem.status == "active",
                    Skill.is_executable.is_(True),
                    or_(Skill.owner_id.is_(None), Skill.owner_id == Agent.owner_id),
                )
                .limit(1)
            )
        ).first()
        return WORKSPACE_IMAGE_OFFICE if has_office_skill is not None else WORKSPACE_IMAGE_BASE
    except Exception:
        logger.exception("workspace_image: failed to resolve image for room %s", room_id)
        return WORKSPACE_IMAGE_BASE
