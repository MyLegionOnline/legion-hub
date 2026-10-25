"""ModelCapability — cached reasoning-capability probe results.

Reasoning support is a per-(provider, model) fact that can only be measured
by a live probe (internal design notes stage 3 / R3): glm-5.3 and
deepseek-v4-flash-vision-exp live on the same gateway yet behave oppositely.
This table memoizes the probe answer with a TTL so no message-turn path ever
re-probes; the probe itself runs lazily at provider validation (stage 9) and
when a user first enables reasoning on an agent (stage 4).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

# Plan §4.3: probe results stay fresh for a week.  The constant lives here (not
# in app.core.reasoning) because reasoning.py imports this model for its cache
# helpers — importing it back would close a cycle.
CAPABILITY_TTL_DAYS = 7


class ModelCapability(SQLModel, table=True):
    __tablename__ = "model_capability"

    provider: str = Field(sa_column=Column(String(64), primary_key=True))
    model: str = Field(sa_column=Column(String(128), primary_key=True))
    supported: bool = Field(sa_column=Column(Boolean, nullable=False))
    controllable: bool = Field(sa_column=Column(Boolean, nullable=False))
    off_switch: Optional[dict[str, Any]] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )
    probed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=CAPABILITY_TTL_DAYS),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )
