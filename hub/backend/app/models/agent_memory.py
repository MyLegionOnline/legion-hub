from datetime import datetime
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field, UniqueConstraint


DEFAULT_TEMPLATE = """# Agent Memory

## What I've done here

## Decisions I made

## Things to remember next time
"""


class AgentMemory(SQLModel, table=True):
    __tablename__ = "agentmemory"
    __table_args__ = (UniqueConstraint("room_id", "agent_id", name="uq_agentmemory_room_agent"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: UUID = Field(foreign_key="room.id", index=True)
    agent_id: UUID = Field(foreign_key="agent.id", index=True)
    content: str = Field(default=DEFAULT_TEMPLATE)
    token_count: int = Field(default=0)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
