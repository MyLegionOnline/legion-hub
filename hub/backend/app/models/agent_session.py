from datetime import datetime
from typing import Optional, TYPE_CHECKING, Dict, Any
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field, Relationship, Column, JSON

if TYPE_CHECKING:
    from .task import Task


class AgentSession(SQLModel, table=True):
    __tablename__ = "agentsession"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    room_id: Optional[UUID] = Field(default=None, foreign_key="room.id", index=True)
    task_id: Optional[UUID] = Field(default=None, foreign_key="task.id", index=True)
    agent_id: UUID = Field(foreign_key="agent.id")

    context_snapshot: Dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )

    summary: Optional[str] = Field(default=None)
    session_title: Optional[str] = Field(default=None)
    initiator_type: str = Field(default="user")  # "user" or "agent"
    model_id: Optional[str] = Field(default=None)

    log_key: Optional[str] = Field(default=None)
    log_size_bytes: int = Field(default=0)
    log_status: str = Field(default="recording")  # recording | flushed | flush_failed | lost

    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = Field(default=None)

    token_count: int = Field(default=0)
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    # internal design notes stage 6 (§7.3): reasoning tokens are
    # already *inside* output_tokens, so cost and limits need no change — this
    # column only makes visible what share of the output went into thinking.
    # Never add it to any total; that would double-charge the user.
    reasoning_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)

    task: "Task" = Relationship(back_populates="sessions")
