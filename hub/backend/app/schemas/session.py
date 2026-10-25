"""Pydantic schemas for creating new Session records.

Each Create schema validates that fields required for a specific kind
are present. The ORM models accept nullable columns because p1/p2 subtypes
share the same table — obligation is enforced here, not at the DB level.
"""

from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# UserAgentSession
# ---------------------------------------------------------------------------

class UserAgentSessionCreate(BaseModel):
    kind: Literal["user_agent_p1", "user_agent_p2"]
    room_id: UUID
    user_id: UUID
    agent_id: UUID
    task_id: Optional[UUID] = None
    goal_id: Optional[UUID] = None

    @model_validator(mode="after")
    def _required_by_kind(self) -> "UserAgentSessionCreate":
        if self.kind == "user_agent_p1":
            if not self.task_id or not self.goal_id:
                raise ValueError("user_agent_p1 requires task_id and goal_id")
        return self


# ---------------------------------------------------------------------------
# UserCmSession
# ---------------------------------------------------------------------------

class UserCmSessionCreate(BaseModel):
    kind: Literal["user_cm_p1", "user_cm_p2"]
    room_id: UUID
    user_id: UUID
    chat_master_agent_id: UUID
    goal_id: Optional[UUID] = None

    @model_validator(mode="after")
    def _required_by_kind(self) -> "UserCmSessionCreate":
        if self.kind == "user_cm_p1":
            if not self.goal_id:
                raise ValueError("user_cm_p1 requires goal_id")
        return self


# ---------------------------------------------------------------------------
# CmSelfPlanSession
# ---------------------------------------------------------------------------

class CmSelfPlanSessionCreate(BaseModel):
    kind: Literal["cm_self_plan"] = "cm_self_plan"
    room_id: UUID
    chat_master_agent_id: UUID
    goal_id: UUID
    plan_id: UUID


# ---------------------------------------------------------------------------
# CmAgentProxySession
# ---------------------------------------------------------------------------

class CmAgentProxySessionCreate(BaseModel):
    kind: Literal["cm_agent_proxy"] = "cm_agent_proxy"
    room_id: UUID
    side: Literal["agent", "chat_master"]
    task_id: UUID
    goal_id: Optional[UUID] = None
    plan_id: Optional[UUID] = None
    agent_id: UUID
    peer_session_id: Optional[UUID] = None


# ---------------------------------------------------------------------------
# AgentSubtaskSession (chatflow_session_lifecycle_repair_plan stage 5.4)
# ---------------------------------------------------------------------------

class AgentSubtaskSessionCreate(BaseModel):
    kind: Literal["agent_subtask"] = "agent_subtask"
    room_id: UUID
    agent_id: UUID
    task_id: Optional[UUID] = None
    parent_session_id: UUID
    step_title: str = ""


# ---------------------------------------------------------------------------
# Read schema (shared, returned from API)
# ---------------------------------------------------------------------------

class SessionRead(BaseModel):
    id: UUID
    room_id: UUID
    kind: str
    started_at: str
    closed_at: Optional[str] = None
    close_reason: Optional[str] = None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    summary: Optional[str] = None
    peer_session_id: Optional[UUID] = None

    model_config = {"from_attributes": True}
