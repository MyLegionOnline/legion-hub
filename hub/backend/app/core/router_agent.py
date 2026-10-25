"""The system Router agent — one per account (internal design notes Stage 1).

Routing used to borrow the ChatMaster's model (see core/router.py).  The
Router is a dedicated agent the user configures like any other, but it only
ever routes: it never joins a room, takes tasks, skills or @mentions, and it
cannot be deleted or renamed.

Its settings live in ``llm_config``:

* ``provider`` / ``model`` / ``primary_secret_binding_id`` — the ordinary agent
  LLM fields; in Jev mode this is the escalation LLM.  Empty means "use the
  ChatMaster's model".
* ``llm_config["router"]``::

      {"mode": null | "inherit" | "jev",
       "jev_model": "typesafe/jev-1.13",
       "jev_min_confidence": 0.6,
       "secret_binding_id": "<binding of purpose llm_router_jev>"}

  ``mode: null`` means the user never configured it and behaves exactly like
  ``"inherit"`` (route with the ChatMaster's model, as before).

Rows are created by the startup backfill in db/session.py for existing
accounts and by :func:`ensure_router_agent` on registration.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.agent import Agent

logger = logging.getLogger(__name__)

ROUTER_ROLE = "router"
ROUTER_NICKNAME = "Router"
ROUTER_DESCRIPTION = "Routes every user message in your rooms to the right place."

MODE_INHERIT = "inherit"
MODE_JEV = "jev"
ROUTER_MODES = (MODE_INHERIT, MODE_JEV)

DEFAULT_JEV_MODEL = "typesafe/jev-1.13"
DEFAULT_JEV_MIN_CONFIDENCE = 0.6

_ROUTER_KEYS = {"mode", "jev_model", "jev_min_confidence", "secret_binding_id"}


def is_router(agent: Agent | None) -> bool:
    return agent is not None and getattr(agent, "system_role", None) == ROUTER_ROLE


@dataclass(frozen=True)
class RouterSettings:
    mode: str                 # "inherit" | "jev" (null is folded into inherit)
    configured: bool          # False while the user never picked a mode
    jev_model: str
    jev_min_confidence: float
    has_escalation_model: bool


def router_settings(agent: Agent) -> RouterSettings:
    cfg = agent.llm_config or {}
    block = cfg.get("router") or {}
    mode = block.get("mode")
    return RouterSettings(
        mode=mode if mode in ROUTER_MODES else MODE_INHERIT,
        configured=mode in ROUTER_MODES,
        jev_model=block.get("jev_model") or DEFAULT_JEV_MODEL,
        jev_min_confidence=float(
            block.get("jev_min_confidence", DEFAULT_JEV_MIN_CONFIDENCE)
        ),
        has_escalation_model=bool(cfg.get("provider") and cfg.get("model")),
    )


def validate_router_block(llm_config: dict[str, Any]) -> dict[str, Any]:
    """Check ``llm_config["router"]`` of a Router update; raise ValueError."""
    block = llm_config.get("router")
    if block is None:
        return llm_config
    if not isinstance(block, dict):
        raise ValueError("llm_config.router must be an object")
    unknown = set(block) - _ROUTER_KEYS
    if unknown:
        raise ValueError(f"unknown llm_config.router fields: {sorted(unknown)}")
    mode = block.get("mode")
    if mode is not None and mode not in ROUTER_MODES:
        raise ValueError(f"llm_config.router.mode must be one of {ROUTER_MODES} or null")
    conf = block.get("jev_min_confidence")
    if conf is not None:
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            raise ValueError("llm_config.router.jev_min_confidence must be between 0 and 1")
    model = block.get("jev_model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("llm_config.router.jev_model must be a non-empty string")
    return llm_config


async def get_router_agent(owner_id: UUID, db: AsyncSession) -> Optional[Agent]:
    result = await db.execute(
        select(Agent).where(Agent.owner_id == owner_id, Agent.system_role == ROUTER_ROLE)
    )
    return result.scalars().first()


async def ensure_router_agent(owner_id: UUID, db: AsyncSession) -> Optional[Agent]:
    """Return the account's Router, creating it if missing.  Never raises."""
    try:
        existing = await get_router_agent(owner_id, db)
        if existing is not None:
            return existing
        agent = Agent(
            nickname=ROUTER_NICKNAME,
            agent_uid=f"router-{secrets.token_hex(3)}",
            agent_type="internal",
            short_description=ROUTER_DESCRIPTION,
            llm_config={"router": {"mode": None}},
            owner_id=owner_id,
            token=secrets.token_urlsafe(32),
            system_role=ROUTER_ROLE,
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        return agent
    except IntegrityError:
        # A concurrent call (or the startup backfill) created it first.
        await db.rollback()
        return await get_router_agent(owner_id, db)
    except Exception:
        await db.rollback()
        logger.exception("ensure_router_agent: failed for user %s", owner_id)
        return None
