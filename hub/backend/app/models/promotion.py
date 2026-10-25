from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, Column, String
from sqlmodel import SQLModel, Field, JSON


class Promotion(SQLModel, table=True):
    """internal design notes Stage 4 — Gate 2 (``dev → main``).

    One row = one human publication gate: the bundle of every task merged into
    ``dev`` since the last promotion, its ``git diff --stat`` against ``main``,
    the full-suite CI verdict, and the human's decision. ``main`` only ever
    moves when a ``Promotion`` reaches ``approved`` (§0.1 invariant).
    """

    __tablename__ = "promotion"

    id: UUID = Field(default_factory=uuid4, primary_key=True, index=True)
    room_id: UUID = Field(foreign_key="room.id", index=True)

    from_branch: str = Field(default="dev")
    to_branch: str = Field(default="main")

    # str(task_id) for every COMPLETED task whose merge into ``dev`` this
    # promotion carries to ``main``.
    task_ids: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))

    # ``git diff --stat main..dev`` at the moment the gate opened (truncated).
    diff_stat: str = Field(default="", sa_column=Column(String, nullable=False, server_default=""))

    # Full-suite CI on ``dev`` before the gate: null (not run / transport error),
    # "passed", or "failed". approve is blocked unless "passed" or force=True.
    ci_status: Optional[str] = Field(default=None, sa_column=Column("ci_status", String, nullable=True))
    ci_results: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False))

    # Populated by Stage 7 (variant A) when a GitHub PR backs the promotion.
    pr_url: Optional[str] = Field(default=None, sa_column=Column("pr_url", String, nullable=True))

    # Stage 6: the working branch (``dev``) was pushed to the remote when the
    # gate opened so an external CI (``ci_provider == "github_actions"``) can
    # run on it. False for the workspace-only flow.
    dev_pushed: bool = Field(
        default=False, sa_column=Column("dev_pushed", Boolean, nullable=False, server_default="false")
    )

    # waiting_review | approved | rejected | conflict
    state: str = Field(default="waiting_review", sa_column=Column(String, nullable=False, server_default="waiting_review"))
    note: Optional[str] = Field(default=None, sa_column=Column("note", String, nullable=True))

    # sha of the ``dev → main`` merge commit once approved.
    merge_commit: Optional[str] = Field(default=None, sa_column=Column("merge_commit", String, nullable=True))

    initiated_by: Optional[UUID] = Field(default=None)
    resolved_by: Optional[UUID] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = Field(default=None)
