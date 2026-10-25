from datetime import datetime
from uuid import UUID
from pydantic import BaseModel


class ProjectMemoryOut(BaseModel):
    id: UUID
    room_id: UUID
    content: str
    token_count: int
    updated_at: datetime

    class Config:
        from_attributes = True


class ProjectMemoryUpdate(BaseModel):
    content: str


class ProjectMemoryMerge(BaseModel):
    session_summary: str
