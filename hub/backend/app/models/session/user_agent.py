from typing import Optional
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from .base import Session


class UserAgentSession(Session):
    """Direct session between a user and a specific agent.

    Shared table for both p1 (task-focused) and p2 (open conversation).
    Field obligation per kind is enforced in Pydantic schemas.
    """

    __tablename__ = "user_agent_session"

    __mapper_args__ = {"polymorphic_identity": "user_agent_p1"}

    # FK/PK to parent session; DB column is 'session_id'
    id: Mapped[UUID] = mapped_column(
        "session_id",
        sa.Uuid,
        sa.ForeignKey("session.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[UUID] = mapped_column(sa.Uuid, sa.ForeignKey("user.id"), nullable=False)
    agent_id: Mapped[UUID] = mapped_column(sa.Uuid, sa.ForeignKey("agent.id"), nullable=False)
    # REQUIRED for p1 — enforced by Pydantic schema. Optional for p2, which does
    # not put the task in its prompt but keeps the link so a task-level stop
    # (cancel/block) can find and close the conversation.
    task_id: Mapped[Optional[UUID]] = mapped_column(sa.Uuid, sa.ForeignKey("task.id"), nullable=True)
    goal_id: Mapped[Optional[UUID]] = mapped_column(
        sa.Uuid, sa.ForeignKey("room_goal.id"), nullable=True
    )


class UserAgentSessionP2(UserAgentSession):
    """Pattern-2 variant: open conversation, not tied to a specific task."""

    # No new table — shares user_agent_session via STI on top of JTI
    __mapper_args__ = {"polymorphic_identity": "user_agent_p2"}
