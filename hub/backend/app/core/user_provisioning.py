"""The starter set of a new Hub user.

Every way a user comes to exist (local registration, and in the cloud the
first paid plan, see ``legion_cloud.cloud_users``) ends in
``provision_new_user``: default skills, MCPs, agents and the router agent,
all local, from the Hub's own bundle (internal design notes stage 3).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.agent_defaults import provision_default_agents
from app.core.router_agent import ensure_router_agent
from app.core.skill_defaults import reconcile_default_mcps, reconcile_default_skills


async def provision_new_user(user_id: UUID, db: AsyncSession) -> None:
    """Starter skills, MCPs and agents of a new user."""
    await reconcile_default_skills(user_id, db)
    await reconcile_default_mcps(user_id, db)
    await provision_default_agents(user_id, db)
    await ensure_router_agent(user_id, db)
