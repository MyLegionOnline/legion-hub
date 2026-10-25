from datetime import datetime
from typing import Optional
from uuid import UUID
from pydantic import BaseModel

from app.models.goal import GoalStatus


class GoalCreate(BaseModel):
    title: str
    description: str = ""
    position: Optional[int] = None  # None = append to end


class GoalUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[GoalStatus] = None
    position: Optional[int] = None


class GoalOut(BaseModel):
    id: UUID
    room_id: UUID
    title: str
    description: str
    status: GoalStatus
    position: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    # internal design notes Stage 5 — the goal's own git branch and the state of
    # its `goal/<id> → dev` human gate.
    branch_name: Optional[str] = None
    review_state: Optional[str] = None
    review_ci_status: Optional[str] = None
    # internal design notes Stage 4 — goal branch last pushed to remote.
    review_branch_pushed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class GoalReviewResolve(BaseModel):
    """Body of POST /rooms/{room_id}/goals/{goal_id}/review/resolve."""

    decision: str  # "approve" | "reject"
    note: Optional[str] = None
    # Merge into `dev` even though the full CI suite on the goal branch is red.
    force: bool = False


class RequirementsGapResolve(BaseModel):
    """Body of POST /rooms/{room_id}/goals/{goal_id}/requirements-gap-resolve.

    internal design notes Stage 6 — the soft coverage gate's card.
    """

    decision: str  # "accept" | "rework"
    note: Optional[str] = None
