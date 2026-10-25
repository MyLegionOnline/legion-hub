from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from .base import Session


class CmSelfPlanSession(Session):
    """ChatMaster reading a new plan and decomposing it into tasks.

    Triggered by PLAN_CREATED / PLAN_UPDATED room event.
    """

    __tablename__ = "cm_self_plan_session"

    __mapper_args__ = {"polymorphic_identity": "cm_self_plan"}

    id: Mapped[UUID] = mapped_column(
        "session_id",
        sa.Uuid,
        sa.ForeignKey("session.id", ondelete="CASCADE"),
        primary_key=True,
    )
    chat_master_agent_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("agent.id"), nullable=False
    )
    goal_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("room_goal.id"), nullable=False
    )
    plan_id: Mapped[UUID] = mapped_column(
        sa.Uuid, sa.ForeignKey("goal_plan.id"), nullable=False
    )
