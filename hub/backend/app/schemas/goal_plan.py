from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel

from app.models.goal_plan import GoalPlanStatus, PlanStepStatus


class GoalPlanStepCreate(BaseModel):
    title: str
    description: str = ""
    position: Optional[int] = None
    is_parallel: bool = False
    assigned_agent_id: Optional[UUID] = None


class GoalPlanStepUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[PlanStepStatus] = None
    assigned_agent_id: Optional[UUID] = None
    position: Optional[int] = None
    is_parallel: Optional[bool] = None


class GoalPlanStepOut(BaseModel):
    id: UUID
    plan_id: UUID
    title: str
    description: str
    status: PlanStepStatus
    assigned_agent_id: Optional[UUID] = None
    assigned_agent_name: Optional[str] = None
    assigned_agent_in_room: bool = True
    position: int
    is_parallel: bool
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class GoalPlanCreate(BaseModel):
    steps: List[GoalPlanStepCreate] = []


class GoalPlanUpdate(BaseModel):
    status: Optional[GoalPlanStatus] = None


class GoalPlanOut(BaseModel):
    id: UUID
    goal_id: UUID
    status: GoalPlanStatus
    steps: List[GoalPlanStepOut]
    created_at: datetime
    updated_at: datetime
    approved_at: Optional[datetime] = None
    # How many Tasks this goal already carries. The panel keys its Approve
    # button off this: a plan that has already been decomposed must not offer
    # a button that would decompose it a second time.
    tasks_count: int = 0

    model_config = {"from_attributes": True}
