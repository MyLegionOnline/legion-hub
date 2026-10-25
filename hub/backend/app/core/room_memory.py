"""Rooms memory pool (internal design notes §3a, T2a).

A room's memory limit is ``plan.mem_limit_mb`` unless the room's settings
say otherwise (``workspace.mem_setting_mb``), up to ``room_max_memory_mb``
in steps of 256 MiB. A room starts only while the limits of the owner's
running rooms, this one included, fit ``rooms_memory_pool_mb`` — checked in
``WorkspaceQuotaManager.reserve_and_run`` next to ``max_running``. A new
setting reaches a running room at once (``docker update``) when it fits
the pool, otherwise at the next start.

"Currently used" on the plan screen is the rooms' resident memory
from ``workspaced`` (``GET /workspaces/{id}/stats``, cached 15 s there).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

POOL_FULL = "memory_pool_full"
STEP_MB = 256
MIN_MB = 256
# workspaced's StartRequest / MemoryUpdateRequest ceiling: the cap of a
# room on a tier without room_max_memory_mb (selfhosted).
HARD_MAX_MB = 8192
DEFAULT_MB = 1024
_STATS_TIMEOUT_SEC = 5.0


def format_mb(mb: Optional[float]) -> str:
    """1024 → "1 GiB", 1536 → "1.5 GiB", 512 → "512 MiB"."""
    value = float(mb or 0)
    if value >= 1024:
        gib = value / 1024
        text = f"{gib:.1f}".rstrip("0").rstrip(".")
        return f"{text} GiB"
    return f"{int(round(value))} MiB"


def _positive(raw: Any) -> Optional[int]:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return None
    return raw


def default_mb(limits: dict) -> int:
    """The plan's room memory limit when the room has no setting."""
    value = _positive(limits.get("mem_limit_mb")) or DEFAULT_MB
    cap = max_mb(limits)
    return min(value, cap)


def max_mb(limits: dict) -> int:
    """The largest limit one room may take on this plan."""
    return min(_positive(limits.get("room_max_memory_mb")) or HARD_MAX_MB, HARD_MAX_MB)


def effective_mb(limits: dict, setting_mb: Optional[int], requested_mb: Optional[int] = None) -> int:
    """The limit the room's next container gets.

    *requested_mb* — an explicit value of ``POST /workspace/start``; the
    room's setting otherwise; the plan's default without either. Always
    clamped to the plan's room ceiling, so a setting made on Pro cannot
    outgrow a Hobby pool after a downgrade.
    """
    value = requested_mb or setting_mb or default_mb(limits)
    return max(128, min(int(value), max_mb(limits)))


def pool_full_text(pool_mb: int, allocated_mb: int) -> str:
    return (
        f"Memory pool is full: {format_mb(allocated_mb)} of {format_mb(pool_mb)} running. "
        "Stop a room or lower its limit."
    )


class RoomMemoryError(ValueError):
    """A memory setting the plan does not allow."""

    def __init__(self, error_code: str, message: str, *, status_code: int = 422):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.status_code = status_code


def validate_setting(limits: dict, value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    ceiling = max_mb(limits)
    if value < MIN_MB or value > ceiling:
        raise RoomMemoryError(
            "memory_out_of_range",
            f"The room memory limit is from {format_mb(MIN_MB)} to {format_mb(ceiling)}.",
        )
    if value % STEP_MB:
        raise RoomMemoryError(
            "memory_step", f"The memory limit is set in steps of {STEP_MB} MiB.",
        )
    return value


@dataclass
class _Context:
    room_id: UUID
    owner_id: UUID
    limits: dict
    workspace: Any


async def _context(room_id: UUID, db: AsyncSession) -> _Context:
    from app.core.workspace_lifecycle import _get_or_create_workspace
    from app.core.workspace_quota import get_effective_limits
    from app.models.room import Room

    room = await db.get(Room, room_id)
    if room is None:
        raise LookupError("room not found")
    ws = await _get_or_create_workspace(room_id, db)
    owner = ws.quota_owner_id or room.creator_id
    return _Context(room_id, owner, await get_effective_limits(owner, db), ws)


async def _view(ctx: _Context, db: AsyncSession) -> dict[str, Any]:
    from app.core.workspace_quota import memory_pool_mb, reserved_workspace_memory_mb

    ws = ctx.workspace
    running = ws.status in {"starting", "running"}
    others = await reserved_workspace_memory_mb(ctx.owner_id, db, exclude_room_id=ctx.room_id)
    effective = effective_mb(ctx.limits, ws.mem_setting_mb)
    return {
        "room_id": str(ctx.room_id),
        "mem_setting_mb": ws.mem_setting_mb,
        "effective_mb": effective,
        "default_mb": default_mb(ctx.limits),
        "max_mb": max_mb(ctx.limits),
        "min_mb": MIN_MB,
        "step_mb": STEP_MB,
        "status": ws.status,
        # The running container's limit; differs from effective_mb while a
        # new setting waits for the next start.
        "current_mb": ws.mem_limit_mb if running else None,
        "pending": running and ws.mem_limit_mb != effective,
        "pool_mb": memory_pool_mb(ctx.limits),
        "pool_used_by_others_mb": others,
    }


async def room_memory_view(room_id: UUID, db: AsyncSession) -> dict[str, Any]:
    return await _view(await _context(room_id, db), db)


async def set_room_memory(room_id: UUID, value: Optional[int], db: AsyncSession) -> dict[str, Any]:
    """Store the room's memory setting and apply it to a running room.

    The running container is updated in place when the new limit fits the
    pool and is not below what the room holds now (a lower cap than the
    resident memory would have the kernel kill a process in the room).
    Otherwise the setting waits for the next start; ``applied`` and
    ``message`` say which.
    """
    from app.core import workspaced_client as wsd
    from app.core.workspace_quota import (
        memory_pool_mb,
        quota_owner_lock,
        reserved_workspace_memory_mb,
    )

    from app.core.workspace_quota import EXPIRED_PLAN_TIER, PLAN_EXPIRED_READ_ONLY

    ctx = await _context(room_id, db)
    if ctx.limits.get("plan_tier") == EXPIRED_PLAN_TIER:
        # TR4: an expired plan's rooms are read-only.
        raise RoomMemoryError("plan_expired", PLAN_EXPIRED_READ_ONLY, status_code=402)
    setting = validate_setting(ctx.limits, value)
    async with quota_owner_lock(ctx.owner_id):
        ws = ctx.workspace
        await db.refresh(ws)
        ws.mem_setting_mb = setting
        db.add(ws)
        await db.commit()
        effective = effective_mb(ctx.limits, setting)
        applied = True
        message: Optional[str] = None
        if ws.status == "running" and ws.mem_limit_mb != effective:
            applied = False
            pool = memory_pool_mb(ctx.limits)
            others = await reserved_workspace_memory_mb(
                ctx.owner_id, db, exclude_room_id=room_id
            )
            resident = await _resident_mb(room_id)
            if pool is not None and others + effective > pool:
                message = (
                    f"Does not fit in the pool: other rooms hold {format_mb(others)} "
                    f"of {format_mb(pool)}. The limit will apply at the next room start."
                )
            elif effective < ws.mem_limit_mb and resident is not None and resident >= effective:
                message = (
                    f"The room currently occupies {format_mb(resident)}. "
                    "A smaller limit will apply at the next room start."
                )
            else:
                try:
                    await wsd.update_workspace_memory(str(room_id), effective)
                except Exception:
                    logger.warning("room memory: live update failed for %s", room_id, exc_info=True)
                    message = "Could not change the limit on the fly. It will apply at the next room start."
                else:
                    ws.mem_limit_mb = effective
                    db.add(ws)
                    await db.commit()
                    applied = True
        elif ws.status != "running":
            # Stopped (or still starting with the previous value): nothing
            # runs under the old limit to change.
            applied = ws.status != "starting" or ws.mem_limit_mb == effective
            if not applied:
                message = "The room is starting: the limit will apply at the next room start."
    view = await _view(ctx, db)
    return {**view, "applied": applied, "message": message}


async def _resident_mb(room_id: UUID) -> Optional[float]:
    from app.core import workspaced_client as wsd

    try:
        stats = await asyncio.wait_for(wsd.get_workspace_stats(str(room_id)), _STATS_TIMEOUT_SEC)
    except Exception:
        return None
    value = stats.get("memory_rss_mb")
    return float(value) if value is not None else None


async def resident_memory(room_ids: list[UUID]) -> dict[str, Optional[float]]:
    """Resident memory of each running room — "Currently used" (§3a).

    A room workspaced cannot answer for is None; the screen then shows the
    sum of the rest.
    """
    values = await asyncio.gather(*(_resident_mb(rid) for rid in room_ids))
    return {str(rid): value for rid, value in zip(room_ids, values)}
