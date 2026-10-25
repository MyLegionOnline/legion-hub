"""A request to erase everything a user left in this Hub (internal design notes stage 13, O18).

Made when Market reports that a Legion account was deleted
(``legion_auth.store_credentials``, status ``erased``) or when the operator
asks for it. ``app.core.user_erasure.erase_user`` does the work and the
reconciler repeats it with a back-off until ``completed_at`` is set. A row is
kept after it is done, as the record that the data is gone and when; it holds
counters only, nothing that says who the user was.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlmodel import Column, Field, JSON, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tz() -> sa.DateTime:
    return sa.DateTime(timezone=True)


class UserErasure(SQLModel, table=True):
    __tablename__ = "user_erasure"
    __table_args__ = (
        sa.CheckConstraint("source IN ('market', 'operator')", name="ck_user_erasure_source"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    # One request per user: a repeated push or click finds the same row.
    user_id: UUID = Field(sa_column=sa.Column(sa.Uuid, sa.ForeignKey("user.id"), nullable=False, unique=True))
    legion_account_id: Optional[UUID] = Field(default=None, sa_column=sa.Column(sa.Uuid, nullable=True))
    source: str = Field(default="market", max_length=16)
    requested_at: datetime = Field(default_factory=_now, sa_column=sa.Column(_tz(), nullable=False))
    next_attempt_at: datetime = Field(default_factory=_now, sa_column=sa.Column(_tz(), nullable=False, index=True))
    started_at: Optional[datetime] = Field(default=None, sa_column=sa.Column(_tz(), nullable=True))
    completed_at: Optional[datetime] = Field(default=None, sa_column=sa.Column(_tz(), nullable=True, index=True))
    attempts: int = Field(default=0)
    # The class and a short text of the last failure; never a user's data.
    last_error: Optional[str] = Field(default=None, max_length=500)
    # Counters per step (rooms, agents, ...).
    report: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
