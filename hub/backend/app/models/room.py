from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, TYPE_CHECKING, Dict, Any
from uuid import UUID, uuid4
from sqlalchemy import DateTime, func
from sqlmodel import SQLModel, Field, Relationship, Column, JSON

if TYPE_CHECKING:
    from .user import User
    from .agent import Agent
    from .message import Message


class RoomType(str, Enum):
    group = "group"    # multiple agents + chat-master coordinator
    direct = "direct"  # single agent, acts as its own coordinator


class RoomState(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"


class VpsAccessMode(str, Enum):
    """internal design notes §3.7 Stage 5 — sits above `core.mcp_policy` risk classes.

    ``readonly`` (default) denies anything but the ``read`` risk class,
    without an approval card and without a credit charge — a ChatMaster
    session can still inspect a server while gathering requirements
    (§3.7/§18.4). ``full`` defers to the per-call risk class instead, asking
    a person for every ``write_system``/``service``/``unknown`` call.
    ``trusted`` behaves exactly like ``full`` except those three risk
    classes are auto-approved instead of asking — an explicit, room-scoped
    opt-out of the approval card for people tired of confirming every VPS
    call in a session they already trust (§17 risk 7: approval fatigue).
    ``destructive`` calls are still always denied outright, in every mode —
    that boundary does not move with this setting.
    """

    readonly = "readonly"
    full = "full"
    trusted = "trusted"


ROOM_TRANSITIONS: dict[RoomState, list[RoomState]] = {
    RoomState.PLANNING: [RoomState.WAITING_USER, RoomState.EXECUTION],
    RoomState.EXECUTION: [RoomState.WAITING_USER, RoomState.COMPLETED],
    RoomState.WAITING_USER: [RoomState.EXECUTION, RoomState.PLANNING],
    RoomState.COMPLETED: [],
}


class RoomUserLink(SQLModel, table=True):
    room_id: UUID = Field(foreign_key="room.id", primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", primary_key=True)

class RoomAgentLink(SQLModel, table=True):
    room_id: UUID = Field(foreign_key="room.id", primary_key=True)
    agent_id: UUID = Field(foreign_key="agent.id", primary_key=True)

class RoomBase(SQLModel):
    name: str
    type: str = "group"  # "private", "group"
    room_type: RoomType = Field(default=RoomType.group)
    has_coordinator: bool = Field(default=False)
    coordinator_config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    git_config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    vps_access_mode: VpsAccessMode = Field(default=VpsAccessMode.readonly)

class Room(RoomBase, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    creator_id: UUID = Field(foreign_key="user.id")
    state: RoomState = Field(default=RoomState.PLANNING)
    # Fallback ordering key for the sidebar when a room has no messages yet
    # (see internal design notes Part 2, Stage 1).
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    # Durable marker for the two-system room deletion transaction.  It keeps
    # new workspace starts out while the external destroy is retried.
    deletion_started_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    # internal design notes §5.5 (T11): more rooms than the plan allows
    # (Pro -> Hobby) — this one is frozen: read and export only, no start,
    # no agent work. The owner picks the active ones (legion_cloud/data_lifecycle.py).
    frozen_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
    )

    creator: "User" = Relationship(back_populates="created_rooms")
    users: List["User"] = Relationship(link_model=RoomUserLink)
    agents: List["Agent"] = Relationship(link_model=RoomAgentLink)
    messages: List["Message"] = Relationship(back_populates="room")
