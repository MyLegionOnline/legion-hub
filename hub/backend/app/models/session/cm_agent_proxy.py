from typing import Optional
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from .base import Session


class CmAgentProxySession(Session):
    """One side of a ChatMaster ↔ Agent proxy session pair.

    Two records are created per task handoff: side='agent' and side='chat_master'.
    They are linked via peer_session_id on the base Session.
    """

    __tablename__ = "cm_agent_proxy_session"

    __mapper_args__ = {"polymorphic_identity": "cm_agent_proxy"}

    __table_args__ = (
        sa.CheckConstraint("side IN ('agent', 'chat_master')", name="ck_proxy_side"),
    )

    id: Mapped[UUID] = mapped_column(
        "session_id",
        sa.Uuid,
        sa.ForeignKey("session.id", ondelete="CASCADE"),
        primary_key=True,
    )
    side: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    task_id: Mapped[UUID] = mapped_column(sa.Uuid, sa.ForeignKey("task.id"), nullable=False)
    # Direct tasks have a single executor and do not need a goal/plan tree.
    # These links remain populated for planned tasks to preserve their memory
    # context and reporting relationships.
    goal_id: Mapped[Optional[UUID]] = mapped_column(
        sa.Uuid, sa.ForeignKey("room_goal.id"), nullable=True
    )
    plan_id: Mapped[Optional[UUID]] = mapped_column(
        sa.Uuid, sa.ForeignKey("goal_plan.id"), nullable=True
    )
    # The chosen agent or ChatMaster's agent record
    agent_id: Mapped[UUID] = mapped_column(sa.Uuid, sa.ForeignKey("agent.id"), nullable=False)
