from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, ForeignKey, String, Boolean
from sqlalchemy.dialects.postgresql import UUID as PGUUID


class RequirementKind(str, Enum):
    REQUIREMENT = "requirement"
    NON_GOAL = "non_goal"
    QUESTION = "question"


class RequirementPriority(str, Enum):
    MUST = "must"
    SHOULD = "should"
    COULD = "could"


class RequirementStatus(str, Enum):
    UNCOVERED = "uncovered"
    PARTIAL = "partial"
    COVERED = "covered"
    DEFERRED = "deferred"
    CONFLICT = "conflict"
    CLOSED = "closed"


class GoalRequirement(SQLModel, table=True):
    """internal design notes Stage 0 — an atomic requirement
    (R-NNN), non-goal (NG-NNN) or open question (Q-NNN) extracted verbatim
    from the plan dialogue that formed a ``RoomGoal``. ``quote`` is checked
    against the dialogue journal (Stage 3) so the model can't invent one; the
    result of that check is ``verified``, never silently assumed True."""

    __tablename__ = "goal_requirement"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    goal_id: UUID = Field(
        sa_column=Column(PGUUID(as_uuid=True), ForeignKey("room_goal.id", ondelete="CASCADE"), index=True, nullable=False)
    )
    code: str = Field(sa_column=Column(String, nullable=False))
    kind: RequirementKind = Field(default=RequirementKind.REQUIREMENT)
    quote: str = Field(default="")
    turn_ref: Optional[str] = Field(default=None, sa_column=Column(String, nullable=True))
    statement: str = Field(default="")
    priority: Optional[RequirementPriority] = Field(default=None)
    acceptance: str = Field(default="")
    status: RequirementStatus = Field(default=RequirementStatus.UNCOVERED)
    # None — the dialogue journal couldn't be checked (Redis expired, no
    # fallback archive found): absence of proof is not proof the model
    # invented the quote, so this must not collapse to False.
    verified: Optional[bool] = Field(
        default=None,
        sa_column=Column("verified", Boolean, nullable=True),
    )
    position: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TaskRequirement(SQLModel, table=True):
    """M2M: which requirements a task claims to cover (Stage 4's ``covers``)."""

    __tablename__ = "task_requirement"

    task_id: UUID = Field(
        sa_column=Column(PGUUID(as_uuid=True), ForeignKey("task.id", ondelete="CASCADE"), primary_key=True)
    )
    requirement_id: UUID = Field(
        sa_column=Column(PGUUID(as_uuid=True), ForeignKey("goal_requirement.id", ondelete="CASCADE"), primary_key=True)
    )
