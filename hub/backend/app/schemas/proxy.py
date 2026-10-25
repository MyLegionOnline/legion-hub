"""Pydantic schemas for proxy messages (Step 5)."""

from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel


class ProxyMessageCreate(BaseModel):
    room_id: UUID
    sender_type: Literal["agent", "chat_master"]
    sender_id: UUID
    content: str
    task_id: UUID
    session_id: UUID
    peer_session_id: UUID
    kind: str = "cm_agent_proxy"


class ProxyMessageRead(BaseModel):
    id: UUID
    room_id: UUID
    content: str
    task_id: Optional[UUID] = None
    timestamp: str
    metadata: dict

    model_config = {"from_attributes": True}
