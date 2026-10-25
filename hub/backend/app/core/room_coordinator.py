"""Shared helpers for resolving a room's coordinator (ChatMaster).

The triple of config keys (``agent_id`` / ``chat_master_agent_id`` /
``master_agent_id``) that names the ChatMaster used to be read independently in
four files (``api/v1/chat.py``, ``core/chat_master.py``,
``core/plan_session.py`` and the router).  Any drift between those copies
produces exactly the class of bugs that ``internal design notes``
§0.2 was written to close: the router silently borrowing a random agent's
model because it could not find a master its own copy of the keys knew about.

This module owns that triple.  Import it instead of re-reading the keys.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from app.models.room import Room

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.models.agent import Agent

logger = logging.getLogger(__name__)


def get_cm_agent_id(room: Room) -> UUID | None:
    """Extract the ChatMaster agent UUID from ``room.coordinator_config``.

    Reads the three historical key names in order of precedence.  Returns
    ``None`` when no master is configured or the value is not a valid UUID —
    callers decide what a missing master means (group rooms treat it as a
    misconfiguration, direct rooms never call this).
    """
    cfg = room.coordinator_config or {}
    raw_id = (
        cfg.get("agent_id")
        or cfg.get("chat_master_agent_id")
        or cfg.get("master_agent_id")
    )
    if raw_id is None:
        return None
    try:
        return UUID(str(raw_id))
    except (ValueError, AttributeError):
        return None


async def resolve_planner_llm_config(
    room: Room, db: "AsyncSession"
) -> "tuple[Agent | None, dict[str, Any]]":
    """Resolve the (agent, llm_config) the plan decomposer and the dialogue
    digest must run on — ``internal design notes`` §8.1.

    That is the room's ChatMaster agent's own ``llm_config``: decomposition is
    where model quality turns into requirement quality for every executor, so
    neither the executor's model nor a built-in cheap default is acceptable.
    An optional ``coordinator_config["planner_model"]`` override sits on top —
    a string replaces ``model``, a dict is merged — for when the planner should
    be smarter than the ChatMaster itself.

    Returns ``(agent, config)``. ``config`` carries a usable ``model`` key only
    when the ChatMaster is configured with one (or the override supplies it);
    the caller treats a missing ``model`` as ``PLANNER_MODEL_MISSING`` rather
    than falling back to a cheap default.
    """
    from app.models.agent import Agent

    agent_id = get_cm_agent_id(room)
    agent = await db.get(Agent, agent_id) if agent_id is not None else None
    raw = getattr(agent, "llm_config", None) if agent is not None else None
    config: dict[str, Any] = dict(raw) if isinstance(raw, dict) else {}

    override = (room.coordinator_config or {}).get("planner_model")
    if isinstance(override, str) and override.strip():
        config["model"] = override.strip()
    elif isinstance(override, dict) and override:
        config = {**config, **override}

    return agent, config
