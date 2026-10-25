from datetime import datetime
from typing import Any, Dict

from sqlmodel import SQLModel, Field, Column, JSON


class Plan(SQLModel, table=True):
    """Tariff plan. Source of truth for plan limits on the cloud (seeded by the
    legion_cloud plugin from legion_cloud.tariffs.PLAN_LIMITS, managed in the
    Admin Center). Unused, and empty, on a self-hosted Hub."""

    __tablename__ = "plan"

    tier: str = Field(primary_key=True)            # free | hobby | pro | team | ...
    title: str = Field(default="")
    # {"max_running": 1, "max_rooms": 3, "monthly_hours": 40, "mem_limit_mb": 1024}
    limits: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    is_default: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
