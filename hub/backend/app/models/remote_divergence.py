from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, Column, DateTime, Index, func, text
from sqlmodel import JSON, Field, SQLModel


OPEN_REMOTE_DIVERGENCE = text("state <> 'resolved'")


class RemoteDivergence(SQLModel, table=True):
    """The last conflicting Pull into a local branch and its review state."""

    __tablename__ = "remote_divergence"
    __table_args__ = (
        CheckConstraint(
            "state IN ('detected', 'reviewing', 'reviewed', 'resolved', 'stale')",
            name="ck_remote_divergence_state",
        ),
        Index(
            "uq_remote_divergence_open_room_branch", "room_id", "branch",
            unique=True, postgresql_where=OPEN_REMOTE_DIVERGENCE,
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    room_id: UUID = Field(foreign_key="room.id", ondelete="CASCADE", index=True)
    branch: str
    remote_ref: str
    local_sha: str
    remote_sha: str
    conflict_files: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False, server_default="[]"),
    )
    state: str = Field(default="detected", sa_column_kwargs={"server_default": "detected"})
    analysis: dict | None = Field(default=None, sa_column=Column(JSON(none_as_null=True)))
    decision: str | None = None
    rework_task_id: UUID | None = Field(default=None, foreign_key="task.id", ondelete="SET NULL")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=func.now()),
    )
    resolved_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))
