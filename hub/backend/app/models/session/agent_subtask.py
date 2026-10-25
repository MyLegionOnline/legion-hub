from typing import Optional
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from .base import Session


class AgentSubtaskSession(Session):
    """One run_subtask child loop, recorded for observability (stage 5.4).

    The parent proxy/user session still receives the compact tool report and
    still applies any close_session actions to itself — this row never
    receives close actions of its own. It exists so the child's token/cost
    spend stops being folded anonymously into the parent's totals.
    parent_session_id (on the base Session) links it back to its spawner.
    """

    __tablename__ = "agent_subtask_session"

    __mapper_args__ = {"polymorphic_identity": "agent_subtask"}

    id: Mapped[UUID] = mapped_column(
        "session_id",
        sa.Uuid,
        sa.ForeignKey("session.id", ondelete="CASCADE"),
        primary_key=True,
    )
    agent_id: Mapped[UUID] = mapped_column(sa.Uuid, sa.ForeignKey("agent.id"), nullable=False)
    task_id: Mapped[Optional[UUID]] = mapped_column(
        sa.Uuid, sa.ForeignKey("task.id"), nullable=True
    )
    step_title: Mapped[str] = mapped_column(sa.String(200), nullable=False, default="")
