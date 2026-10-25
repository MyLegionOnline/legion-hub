"""TaskInterjection — a user's message into a task that is already running.

internal design notes Stage 2.  The agent cannot read anything while a model
request is in flight, so the message is queued here and delivered at the next
tool-loop boundary (``core/task_interject.deliver_pending``).  The agent
answers with the ``reply_to_user`` tool, which fills ``reply_text``.

A row that is never delivered (the run ended first) stays pending and is
picked up by the next run of the same task.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TaskInterjection(SQLModel, table=True):
    __tablename__ = "task_interjection"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4),
    )
    room_id: UUID = Field(sa_column=Column(PG_UUID(as_uuid=True), nullable=False))
    task_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True),
    )
    author_user_id: Optional[UUID] = Field(
        default=None, sa_column=Column(PG_UUID(as_uuid=True), nullable=True),
    )
    text: str = Field(sa_column=Column(Text, nullable=False))
    # The chat bubble that shows the user's message.
    message_id: Optional[UUID] = Field(
        default=None, sa_column=Column(PG_UUID(as_uuid=True), nullable=True),
    )
    created_at: datetime = Field(
        default_factory=_now,
        sa_column=Column(DateTime(timezone=True), nullable=False, default=_now),
    )
    delivered_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    delivered_session_id: Optional[UUID] = Field(
        default=None, sa_column=Column(PG_UUID(as_uuid=True), nullable=True),
    )
    answered_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True),
    )
    reply_text: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True),
    )
