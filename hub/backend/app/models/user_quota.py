import os
import re
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import BigInteger, DateTime
from sqlmodel import SQLModel, Field, Column, JSON


def _env_default_tier() -> str:
    tier = os.getenv("DEFAULT_PLAN_TIER", "hobby")
    return tier if re.fullmatch(r"[a-z0-9_-]+", tier) else "hobby"


# Single default tier for the whole app: used both as the model default for
# new rows and as the fallback when a user has no userquota row at all.
# (These two used to disagree — "free" vs "hobby" — which caused surprise
# 402s on workspace/start; keep them unified via this constant.)
DEFAULT_PLAN_TIER = _env_default_tier()


class UserQuota(SQLModel, table=True):
    __tablename__ = "userquota"

    user_id: UUID = Field(primary_key=True, foreign_key="user.id")
    repo_quota_bytes: int = Field(default=524_288_000)    # 500 MB
    repo_used_bytes: int = Field(default=0)
    skills_quota_bytes: int = Field(default=52_428_800)   # 50 MB
    skills_used_bytes: int = Field(default=0)
    # internal design notes stage 1: total bytes of per-owner avatar images
    # (after normalization, see AVATAR_QUOTA_BYTES_PER_USER). Quota check is
    # wired in stage 3 against AVATAR_QUOTA_BYTES_PER_USER (default 10 MiB);
    # no dedicated avatars_quota_bytes column — the limit is an env-only cap,
    # unlike the legacy *_quota_bytes columns that are per-user overridable.
    avatars_used_bytes: int = Field(default=0)
    plan_tier: str = Field(default_factory=lambda: DEFAULT_PLAN_TIER)
    # Per-user limit overrides applied on top of the plan's limits
    # (e.g. {"monthly_hours": 100}); managed by the Admin Center.
    limits_override: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    # When set and in the past, plan_tier becomes "free" (EXPIRED_PLAN_TIER) and
    # limits_override is ignored (temporary upgrades for testing/promos).
    # TIMESTAMPTZ: aware UTC, like all workspace/admin timestamps.
    plan_expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # Start of the paid period (internal design notes §3.4): the monthly
    # hour windows of a paid plan run from it instead of the 1st of the month.
    # Set by billing entitlements only; NULL (free, or a plan an admin set by
    # hand) keeps the calendar month.
    plan_period_start: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # internal design notes §5.5 (T11, legion_cloud/data_lifecycle.py): when the
    # content of an ended plan was deleted on its 90th day (rooms, archives,
    # agents); cleared, with the starter agents back, by the next paid period.
    data_purged_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # Since when the plan allows no personal MCP runners while some are
    # installed (Pro -> Hobby, or ended): stopped at once, their data deleted
    # 30 days later. Cleared when the plan allows them again.
    personal_mcp_suspended_at: Optional[datetime] = Field(
        default=None, sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    # Stable XFS project id for all workspace directories of this owner.  It
    # is allocated lazily only when a finite room-storage plan is used; the
    # self-hosted default has no filesystem quota and therefore needs none.
    # Keep it in the Hub database rather than deriving it from a UUID: XFS
    # accepts a bounded numeric id and a persisted sequence avoids collisions
    # when data is restored or moved to a new worker host.  Cloud reserves
    # Docker's own overlay2 project-id range at a much higher value (see the
    # production runbook), leaving this low range for owners.
    xfs_project_id: Optional[int] = Field(
        default=None, sa_column=Column(BigInteger, nullable=True)
    )
    updated_at: datetime = Field(default_factory=datetime.utcnow)
