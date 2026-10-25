from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID


class GoalPlanStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class GoalPlan(SQLModel, table=True):
    __tablename__ = "goal_plan"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    goal_id: UUID = Field(
        sa_column=Column(PGUUID(as_uuid=True), ForeignKey("room_goal.id", ondelete="CASCADE"), index=True, unique=True, nullable=False)
    )
    status: GoalPlanStatus = Field(default=GoalPlanStatus.DRAFT)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = Field(default=None)


class GoalPlanStep(SQLModel, table=True):
    __tablename__ = "goal_plan_step"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    plan_id: UUID = Field(
        sa_column=Column(PGUUID(as_uuid=True), ForeignKey("goal_plan.id", ondelete="CASCADE"), index=True, nullable=False)
    )
    title: str
    description: str = Field(default="")
    status: PlanStepStatus = Field(default=PlanStepStatus.PENDING)
    assigned_agent_id: Optional[UUID] = Field(default=None, foreign_key="agent.id")
    position: int = Field(default=0)
    is_parallel: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
