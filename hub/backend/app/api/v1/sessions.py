"""Session-related endpoints.

GET /rooms/{room_id}/sessions
  List all Sessions (all kinds) for a room, with aggregate totals.
  Used by the Sessions tab.

GET /rooms/{room_id}/sessions/{session_id}/proxy-messages
  Returns all PROXY_MESSAGE events for a given session pair,
  ordered by turn. Used by the frontend accordion.
"""

from decimal import Decimal
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectin_polymorphic

from app.api import deps
from app.core.context_waterfall import context_waterfall
from app.db.session import get_session
from app.models.agent import Agent
from app.models.agent_session import AgentSession
from app.models.message import Message, PROXY_MESSAGE
from app.models.room import RoomUserLink
from app.models.session import (
    SESSION_SUBTYPES,
    AgentSubtaskSession,
    CmAgentProxySession,
    CmSelfPlanSession,
    Session,
    UserAgentSession,
    UserCmSession,
)
from app.models.task import Task
from app.models.user import User
from app.schemas.proxy import ProxyMessageRead

router = APIRouter()


# ── schemas ───────────────────────────────────────────────────────────────────

class SessionListItem(BaseModel):
    id: UUID
    kind: str
    chain_type: str = "other"
    task_id: Optional[UUID] = None
    session_title: Optional[str] = None
    task_title: Optional[str] = None
    agent_nickname: Optional[str] = None
    model_id: Optional[str] = None
    initiator_type: str = "user"
    started_at: Any
    ended_at: Any = None
    close_reason: Optional[str] = None
    token_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    context_waterfall: dict[str, int] = Field(default_factory=dict)
    max_context_window_tokens: int = 0
    last_context_window_tokens: int = 0
    # Stage 5.3/5.4/5.5: observable size + grouping for the Sessions tab tree.
    message_count: int = 0
    turn_count: int = 0
    peer_session_id: Optional[UUID] = None
    parent_session_id: Optional[UUID] = None

    class Config:
        from_attributes = True


class AgentBreakdownItem(BaseModel):
    agent_id: Optional[UUID] = None
    agent_nickname: Optional[str] = None
    session_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    max_context_window_tokens: int = 0


class ModelBreakdownItem(BaseModel):
    model_id: Optional[str] = None
    session_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    max_context_window_tokens: int = 0


class SessionListOut(BaseModel):
    sessions: List[SessionListItem]
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    agent_breakdown: List[AgentBreakdownItem] = Field(default_factory=list)
    model_breakdown: List[ModelBreakdownItem] = Field(default_factory=list)


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_agent_id(s: Session) -> Optional[UUID]:
    """Extract the agent_id from a polymorphic session subclass."""
    if isinstance(s, (CmAgentProxySession, AgentSubtaskSession)):
        return s.agent_id
    if isinstance(s, (UserCmSession, CmSelfPlanSession)):
        # cm_self_plan runs on the room's ChatMaster binding (plan_session.py
        # resolves it from room.coordinator_config), so the decomposition turn
        # belongs to that agent in both breakdowns. Without this branch the
        # planner fell into the "no agent" bucket and its model rendered as "—".
        return s.chat_master_agent_id
    if isinstance(s, UserAgentSession):
        return s.agent_id
    return None


def _resolve_task_id(s: Session) -> Optional[UUID]:
    """Extract task_id from a polymorphic subclass (proxy + subtask only)."""
    if isinstance(s, (CmAgentProxySession, AgentSubtaskSession)):
        return s.task_id
    return None


# Sessions-tab chain-type slugs (§5.5 of chatflow_session_lifecycle_repair_plan):
# the four chains the plan calls out, plus 'cm_self' for the internal
# plan-decomposition kind that fits none of them.
def _chain_type(s: Session) -> str:
    if isinstance(s, UserAgentSession):
        return "user_agent"
    if isinstance(s, UserCmSession):
        return "user_cm"
    if isinstance(s, CmAgentProxySession):
        return "cm_agent"
    if isinstance(s, AgentSubtaskSession):
        return "agent_subtask"
    if isinstance(s, CmSelfPlanSession):
        return "cm_self"
    return "other"


def _update_totals(bucket: dict, s: Session) -> None:
    """Accumulate token/cost/window metrics into a breakdown bucket."""
    bucket["session_count"] += 1
    bucket["input_tokens"] += s.tokens_in
    bucket["output_tokens"] += s.tokens_out
    bucket["cost_usd"] += float(s.cost_usd or Decimal("0"))
    bucket["max_context_window_tokens"] = max(
        bucket["max_context_window_tokens"], s.max_context_window_tokens
    )


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("/{room_id}/sessions", response_model=SessionListOut)
async def list_room_sessions(
    room_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """Return all sessions for a room with aggregate token/cost totals.

    Sessions live in the polymorphic `session` table (subclassed by kind).
    Model-id and initiator are enriched from the sibling AgentSession where possible.
    """
    membership = await db.execute(
        select(RoomUserLink).where(
            RoomUserLink.room_id == room_id,
            RoomUserLink.user_id == current_user.id,
        )
    )
    if not membership.scalars().first():
        raise HTTPException(status_code=403, detail="Not a member of this room")

    result = await db.execute(
        select(Session)
        .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
        .where(Session.room_id == room_id)
        .order_by(Session.started_at.desc())
    )
    sessions = result.scalars().all()

    # Prefetch agent names in one query
    agent_ids = {a for s in sessions if (a := _resolve_agent_id(s))}
    agents_by_id: dict[UUID, Agent] = {}
    if agent_ids:
        agents_res = await db.execute(select(Agent).where(Agent.id.in_(agent_ids)))
        agents_by_id = {a.id: a for a in agents_res.scalars().all()}

    # Prefetch task titles in one query
    task_ids = {t for s in sessions if (t := _resolve_task_id(s))}
    tasks_by_id: dict[UUID, Task] = {}
    if task_ids:
        tasks_res = await db.execute(select(Task).where(Task.id.in_(task_ids)))
        tasks_by_id = {t.id: t for t in tasks_res.scalars().all()}

    # Prefetch AgentSession siblings (model_id / initiator) if any
    agent_sessions_by_id: dict[UUID, AgentSession] = {}
    if sessions:
        ids = [s.id for s in sessions]
        as_res = await db.execute(select(AgentSession).where(AgentSession.id.in_(ids)))
        agent_sessions_by_id = {a.id: a for a in as_res.scalars().all()}

    items: list[dict] = []
    agent_totals: dict[str, dict] = {}
    model_totals: dict[str, dict] = {}

    for s in sessions:
        agent_id = _resolve_agent_id(s)
        task_id = _resolve_task_id(s)
        agent = agents_by_id.get(agent_id) if agent_id else None
        task = tasks_by_id.get(task_id) if task_id else None
        sibling = agent_sessions_by_id.get(s.id)

        # Session title: prefer subclass-specific hint via summary/task title
        session_title: Optional[str] = None
        if isinstance(s, AgentSubtaskSession):
            session_title = s.step_title or "Sub-step"
        elif task is not None:
            session_title = task.title
        elif isinstance(s, CmSelfPlanSession):
            session_title = "Plan (auto-plan)"
        elif isinstance(s, UserCmSession):
            session_title = "ChatMaster · Goal" if s.kind == "user_cm_p1" else "ChatMaster · New topic"
        elif isinstance(s, UserAgentSession):
            session_title = f"@{agent.nickname}" if agent else None

        # What the session actually ran on wins over what the agent is bound to
        # *now*: rebinding an agent must not retroactively relabel finished work.
        # The agent's current llm_config is a last resort for rows that closed
        # before session.model_id existed.
        model_id = None
        if getattr(s, "model_id", None):
            model_id = s.model_id
        elif sibling and sibling.model_id:
            model_id = sibling.model_id
        elif agent:
            model_id = (agent.llm_config or {}).get("model")

        initiator = "user"
        if isinstance(s, (CmSelfPlanSession, CmAgentProxySession)):
            initiator = "agent"

        items.append({
            "id": s.id,
            "kind": s.kind,
            "chain_type": _chain_type(s),
            "task_id": task_id,
            "session_title": session_title,
            "task_title": task.title if task else None,
            "agent_nickname": agent.nickname if agent else None,
            "model_id": model_id,
            "initiator_type": sibling.initiator_type if sibling else initiator,
            "started_at": s.started_at,
            "ended_at": s.closed_at,
            "close_reason": s.close_reason,
            "token_count": s.tokens_in + s.tokens_out,
            "input_tokens": s.tokens_in,
            "output_tokens": s.tokens_out,
            "cost_usd": float(s.cost_usd or Decimal("0")),
            "context_waterfall": context_waterfall(s.context_snapshot),
            "max_context_window_tokens": s.max_context_window_tokens,
            "last_context_window_tokens": s.last_context_window_tokens,
            "message_count": s.message_count,
            "turn_count": s.turn_count,
            "peer_session_id": s.peer_session_id,
            "parent_session_id": s.parent_session_id,
        })

        # Agent breakdown bucket
        aid_key = str(agent_id) if agent_id else ""
        if aid_key not in agent_totals:
            agent_totals[aid_key] = {
                "agent_id": agent_id,
                "agent_nickname": agent.nickname if agent else None,
                "session_count": 0, "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "max_context_window_tokens": 0,
            }
        _update_totals(agent_totals[aid_key], s)

        # Model breakdown bucket
        mid_key = model_id or ""
        if mid_key not in model_totals:
            model_totals[mid_key] = {
                "model_id": model_id,
                "session_count": 0, "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "max_context_window_tokens": 0,
            }
        _update_totals(model_totals[mid_key], s)

    total_in = sum(s.tokens_in for s in sessions)
    total_out = sum(s.tokens_out for s in sessions)
    total_cost = float(sum((s.cost_usd or Decimal("0")) for s in sessions))

    return {
        "sessions": items,
        "total_tokens": total_in + total_out,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": total_cost,
        "agent_breakdown": sorted(
            agent_totals.values(),
            key=lambda x: x["input_tokens"] + x["output_tokens"],
            reverse=True,
        ),
        "model_breakdown": sorted(
            model_totals.values(),
            key=lambda x: x["input_tokens"] + x["output_tokens"],
            reverse=True,
        ),
    }


@router.get("/{room_id}/sessions/{session_id}/proxy-messages", response_model=List[ProxyMessageRead])
async def get_session_proxy_messages(
    room_id: UUID,
    session_id: UUID,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
) -> List[ProxyMessageRead]:
    """Return all PROXY_MESSAGE events for a session (either side of the pair)."""
    result = await db.execute(
        select(Message)
        .where(
            Message.room_id == room_id,
            Message.event_type == PROXY_MESSAGE,
        )
        .order_by(Message.timestamp.asc())
    )
    all_msgs = result.scalars().all()

    sid_str = str(session_id)
    filtered = [
        m for m in all_msgs
        if m.event_metadata.get("session_id") == sid_str
        or m.event_metadata.get("peer_session_id") == sid_str
    ]

    return [
        ProxyMessageRead(
            id=m.id,
            room_id=m.room_id,
            content=m.content,
            task_id=m.task_id,
            timestamp=m.timestamp.isoformat(),
            metadata=m.event_metadata,
        )
        for m in filtered
    ]
