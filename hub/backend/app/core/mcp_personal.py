"""Quota and lifecycle helpers for hosted personal MCP installations."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from app.core.workspace_quota import get_effective_limits, get_hours_window
from app.models.market import UserInstalledItem

def _runtime(item: UserInstalledItem) -> dict:
    cfg = item.custom_config or {}
    value = cfg.get("runtime") if isinstance(cfg, dict) else None
    return value if isinstance(value, dict) else {}

async def personal_items(user_id: UUID, db: AsyncSession) -> list[UserInstalledItem]:
    result = await db.execute(select(UserInstalledItem).where(
        UserInstalledItem.user_id == user_id,
        UserInstalledItem.type == "mcp",
        UserInstalledItem.is_custom.is_(True),
    ))
    return list(result.scalars().all())

async def count_running_personal(user_id: UUID, db: AsyncSession) -> int:
    return sum(_runtime(item).get("status") in {"starting", "running"} for item in await personal_items(user_id, db))

async def effective_max_running(user_id: UUID, db: AsyncSession) -> int:
    """Return the validated personal-container concurrency limit."""
    limits = await get_effective_limits(user_id, db)
    value = limits.get("mcp_personal_max_running", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("effective mcp_personal_max_running must be a finite integer")
    return value

async def monthly_hours_consumed(user_id: UUID, db: AsyncSession) -> float:
    """Best-effort usage in the current hour window (the paid period or the
    calendar month, ``workspace_quota.hours_window``) from lifecycle timestamps in JSONB."""
    now = datetime.now(timezone.utc)
    month_start, _ = await get_hours_window(user_id, db, now)
    total = 0.0
    for item in await personal_items(user_id, db):
        runtime = _runtime(item)
        try:
            start = datetime.fromisoformat(str(runtime.get("started_at", "")).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(runtime.get("stopped_at", "")).replace("Z", "+00:00")) if runtime.get("stopped_at") else now
        except ValueError:
            continue
        if start.tzinfo is None: start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None: end = end.replace(tzinfo=timezone.utc)
        total += max(0.0, (end - max(start, month_start)).total_seconds()) / 3600.0
    return total

async def check_personal_start_allowed(user_id: UUID, db: AsyncSession) -> dict:
    limits = await get_effective_limits(user_id, db)
    try:
        raw = await effective_max_running(user_id, db)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if raw == 0:
        raise HTTPException(status_code=402, detail=f"Your plan '{limits['plan_tier']}' does not include personal MCP servers.")
    running = await count_running_personal(user_id, db)
    if running >= raw:
        raise HTTPException(status_code=429, detail=f"Personal MCP limit reached: {running}/{raw}. Stop another server or upgrade the plan.")
    return {"plan_tier": limits["plan_tier"], "max_running": raw, "running": running,
            "mem_limit_mb": limits.get("mcp_personal_mem_limit_mb", 0),
            "monthly_hours": limits.get("mcp_personal_monthly_hours")}

def update_runtime(item: UserInstalledItem, **values: object) -> dict:
    cfg = dict(item.custom_config or {})
    runtime = dict(cfg.get("runtime") or {})
    runtime.update(values)
    cfg["runtime"] = runtime
    item.custom_config = cfg
    return runtime

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Stage 1 parser/service exports live in their own module so the legacy hosted
# lifecycle helpers above remain stable.  Re-export the parser here as well:
# existing backend callers already use ``app.core.mcp_personal`` as the
# personal-MCP namespace.
from app.core.personal_mcp import (  # noqa: E402,F401
    PersonalMcpError,
    launch_spec_from_candidate,
    parse_import_payload,
    transition_runtime,
)
