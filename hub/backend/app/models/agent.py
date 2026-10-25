from datetime import datetime
from typing import List, Optional, TYPE_CHECKING, Dict, Any
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field, Relationship, Column, JSON

if TYPE_CHECKING:
    from .user import User

class AgentBase(SQLModel):
    nickname: str = Field(index=True)
    agent_uid: str = Field(unique=True, index=True)
    agent_type: str = Field(default="external")  # external | internal
    status: str = "offline"
    version: str = "0.1.0"
    update_policy: str = "manual"  # manual, auto, notify
    short_description: str = Field(default="")
    # internal design notes stage 1: lightweight avatar presence flag kept on
    # the hot `agent` row so the agents-list SELECT can decide whether to emit
    # avatar_url without an EXISTS subquery on agentavatar. Mirrors the DB
    # default set in the stage-1 migration (NOT NULL DEFAULT FALSE); never set
    # from AgentCreate/AgentUpdate — only the avatar upload endpoint flips it.
    has_avatar: bool = Field(default=False)
    # Bumped whenever an avatar is uploaded/replaced/deleted; drives the
    # frontend objectURL cache key (R6). NULL = never had an avatar.
    avatar_updated_at: Optional[datetime] = Field(default=None)
    config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    llm_config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    # null = default role tools; a list = explicit allowlist
    tools_allowed: Optional[List[str]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    # "local" runs the agent inside our own sandbox as always; "connector"
    # hands the same task off to an external CLI agent over the Connector
    # transport. An agent without a
    # linked connector_id stays "local" regardless of this flag.
    executor: str = Field(default="local")

class Agent(AgentBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    owner_id: UUID = Field(foreign_key="user.id")
    token: str = Field(unique=True, index=True)
    connector_id: Optional[UUID] = Field(default=None, foreign_key="connector.id")
    # internal design notes Stage 1: NULL for every user-built agent; "router"
    # marks the one system Router per account (core/router_agent.py). Never
    # set through the API — only provisioning writes it.
    system_role: Optional[str] = Field(default=None, max_length=32)

    owner: "User" = Relationship(back_populates="agents")
