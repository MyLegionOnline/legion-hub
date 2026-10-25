from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import DateTime
from sqlmodel import SQLModel, Field, Column, JSON


class AdminAction(SQLModel, table=True):
    """Audit log of administrative actions performed via the Admin API."""

    __tablename__ = "admin_action"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    actor: str                                               # 'admin_center:<operator_login>'
    action: str                                              # set_plan | set_override | stop_workspace | ...
    target_user_id: Optional[UUID] = Field(default=None)
    payload: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
