from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from sqlalchemy import Column, String, Uuid
from sqlmodel import SQLModel, Field


class GoalStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class RoomGoal(SQLModel, table=True):
    __tablename__ = "room_goal"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    room_id: UUID = Field(foreign_key="room.id", index=True)
    title: str
    description: str = Field(default="")
    status: GoalStatus = Field(default=GoalStatus.ACTIVE)
    position: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(default=None)
    # internal design notes Stage 1: ``goal/<id>-<slug>`` — task branches for
    # this goal branch off it and merge back into it; the goal branch as a
    # whole merges into ``dev`` once every task is done (Stage 5).
    branch_name: Optional[str] = Field(default=None)
    # internal design notes Stage 5 — the human gate for ``goal/<id> → dev``.
    # None → not in review (goal has no isolated branch, or its tasks aren't
    # all done yet); "testing" → full CI running on the goal branch;
    # "waiting_review" → CHECKPOINT_REQUEST published, awaiting a human;
    # "approved" → merged into ``dev`` and the goal completed;
    # "rejected" → goal stays ACTIVE, agents iterate.
    review_state: Optional[str] = Field(
        default=None,
        sa_column=Column("review_state", String, nullable=True),
    )
    # Full-suite CI verdict on the goal branch captured when the gate opened:
    # null (not run / transport error), "passed", or "failed". Approve is
    # blocked unless "passed" or force=True.
    review_ci_status: Optional[str] = Field(
        default=None,
        sa_column=Column("review_ci_status", String, nullable=True),
    )
    # internal design notes Stage 4 — last time the goal branch was
    # pushed to the bound remote (``POST .../review/publish``) so the forge's
    # own CI (GitHub Actions / GitLab CI) can run on it. Informational only:
    # publishing never changes ``review_state``, and the goal still completes
    # only on approve → merge into ``dev``.
    review_branch_pushed_at: Optional[datetime] = Field(default=None)
    # internal design notes Stage 0: the user↔ChatMaster session this
    # goal was formed in (mirrors ``Task.origin_session_id``). Used to bind the
    # goal to its ``.legion/plans/goal-{id}.md`` dialogue archive without
    # relying on a title match. No FK: sessions are pruned on their own cycle
    # and must not take the goal with them.
    origin_session_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column("origin_session_id", Uuid, nullable=True, index=True),
    )
