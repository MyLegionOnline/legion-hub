from typing import Optional
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from .base import Session


class UserCmSession(Session):
    """Direct session between a user and ChatMaster.

    Shared table for both p1 (goal-focused) and p2 (new topic).
    """

    __tablename__ = "user_cm_session"

    __mapper_args__ = {"polymorphic_identity": "user_cm_p1"}

    id: Mapped[UUID] = mapped_column(
        "session_id",
        sa.Uuid,
        sa.ForeignKey("session.id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id: Mapped[UUID] = mapped_column(sa.Uuid, sa.ForeignKey("user.id"), nullable=False)
    chat_master_agent_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("agent.id"), nullable=False
    )
    # REQUIRED for p1, NULL for p2 — enforced by Pydantic schema
    goal_id: Mapped[Optional[UUID]] = mapped_column(
        sa.Uuid, sa.ForeignKey("room_goal.id"), nullable=True
    )


class UserCmSessionP2(UserCmSession):
    """Pattern-2 variant: new topic, not tied to any existing goal."""

    __mapper_args__ = {"polymorphic_identity": "user_cm_p2"}
