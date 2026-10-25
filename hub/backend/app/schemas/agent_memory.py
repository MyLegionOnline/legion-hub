from datetime import datetime
from uuid import UUID
from pydantic import BaseModel


class AgentMemoryOut(BaseModel):
    id: UUID
    room_id: UUID
    agent_id: UUID
    content: str
    token_count: int
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentMemoryUpdate(BaseModel):
    content: str


class AgentMemoryMerge(BaseModel):
    session_summary: str
