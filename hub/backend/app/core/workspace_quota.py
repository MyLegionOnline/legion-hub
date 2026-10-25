from __future__ import annotations

import calendar
import logging
import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional, TypeVar
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import work_gates
from app.models.room import RoomUserLink
from app.models.user import User
from app.models.user_quota import DEFAULT_PLAN_TIER, UserQuota
from app.models.workspace import Workspace, WorkspaceUsage

logger = logging.getLogger(__name__)

COUNTED_WORKSPACE_STATES: frozenset[str] = frozenset(
    {"starting", "running", "draining", "stopping"}
)
_QUOTA_LOCKS: dict[UUID, asyncio.Lock] = {}
_ReservationResult = TypeVar("_ReservationResult")


class WorkspaceQuotaExceeded(RuntimeError):
    """Raised when an effective max_running quota has no free slot."""

    def __init__(self, owner_id: UUID, max_running: int, reserved: int):
        super().__init__("workspace quota exceeded")
        self.owner_id = owner_id
        self.max_running = max_running
        self.reserved = reserved


class WorkspaceMemoryPoolExceeded(RuntimeError):
    """Raised when a start would overflow the owner's rooms memory pool (§3a)."""

    def __init__(self, owner_id: UUID, pool_mb: int, allocated_mb: int, requested_mb: int):
        super().__init__("rooms memory pool exceeded")
        self.owner_id = owner_id
        self.pool_mb = pool_mb
        self.allocated_mb = allocated_mb
        self.requested_mb = requested_mb


def _quota_lock(owner_id: UUID) -> asyncio.Lock:
    lock = _QUOTA_LOCKS.get(owner_id)
    if lock is None:
        lock = asyncio.Lock()
        _QUOTA_LOCKS[owner_id] = lock
    return lock


def _counted_states() -> frozenset[str]:
    """Use the lifecycle policy while avoiding an import cycle at module load."""
    try:
        from app.core.workspace_lifecycle import CONFIG

        return frozenset(CONFIG.counted_states)
    except Exception:
        return COUNTED_WORKSPACE_STATES

# ---------- Built-in limits ----------------------------------------------------
# The self-hosted Hub's limits (ConfigLimitsPolicy starts from them): it is
# the owner's hardware, so no rooms/hours/storage caps; the counters the
# runtime needs as finite integers stay finite. MCP credits come from the
# Market account, so no weekly grant. Tariffs live in the cloud plugin.

SELFHOSTED_LIMITS: dict = {
    "max_running": 5, "max_rooms": None, "monthly_hours": None, "mem_limit_mb": 2048,
    "market_paid_tiers": [], "mcp_personal_max_running": 3,
    "mcp_personal_max_rooms": None, "mcp_personal_monthly_hours": None,
    "mcp_personal_mem_limit_mb": 2048, "mcp_personal_max_installed": None,
    "mcp_personal_max_total_memory_mb": None, "mcp_personal_max_memory_per_server_mb": None,
    "mcp_personal_storage_quota_mb": None, "mcp_personal_max_storage_per_server_mb": None,
    "mcp_personal_max_room_bindings_per_server": None,
    "mcp_weekly_credits": 0,
    "rooms_memory_pool_mb": None, "room_max_memory_mb": None,
    "room_storage_quota_mb": None, "container_layer_mb": None,
    "hub_data_soft_quota_mb": None, "link_mcp_max": None,
    # None: the format ceiling (skill_package.MAX_PACKAGE_BYTES) and the
    # per-user userquota.skills_quota_bytes.
    "skill_package_max_mb": None, "skills_storage_mb": None,
}

# A paid plan past plan_expires_at lands here, whatever DEFAULT_PLAN_TIER is.
EXPIRED_PLAN_TIER = "free"
# What an expired plan leaves (internal design notes TR4): the rooms can be
# read, nothing runs in them.
PLAN_EXPIRED_READ_ONLY = (
    "The paid plan has ended: the room is read-only. Agents do not "
    "work, the workspace does not start. Renew the plan to continue."
)
# Tiers billing never sells: free is a state, selfhosted lives on another Hub.
NON_BILLABLE_TIERS = frozenset({"free", "selfhosted"})

_LIMIT_KEYS = (
    "max_running", "max_rooms", "monthly_hours", "mem_limit_mb",
    "market_paid_tiers", "mcp_personal_max_running", "mcp_personal_max_rooms",
    "mcp_personal_monthly_hours", "mcp_personal_mem_limit_mb",
    "mcp_personal_max_installed", "mcp_personal_max_total_memory_mb",
    "mcp_personal_max_memory_per_server_mb", "mcp_personal_storage_quota_mb",
    "mcp_personal_max_storage_per_server_mb", "mcp_personal_max_room_bindings_per_server",
    "mcp_weekly_credits",
    # internal design notes §3: memory pool and storage. Declared here
    # first; enforced by T2a (pool), T3 (room files), T7 (link MCPs).
    "rooms_memory_pool_mb", "room_max_memory_mb", "room_storage_quota_mb",
    "container_layer_mb", "hub_data_soft_quota_mb", "link_mcp_max",
    # §5.4 (T6): one uploaded skill package, and all of a user's uploads.
    "skill_package_max_mb", "skills_storage_mb",
)

# ---------- Effective limits (pure helpers are unit-tested) --------------------

def resolve_tier(quota: Optional[UserQuota], now: Optional[datetime] = None) -> tuple[str, bool]:
    """Return (tier, expired) for a userquota row (or its absence).

    An expired plan becomes EXPIRED_PLAN_TIER (free: nothing runs) — not
    DEFAULT_PLAN_TIER, which is what a new user starts on. The caller must
    also ignore limits_override when expired is True.
    """
    if quota is None:
        return DEFAULT_PLAN_TIER, False
    exp = quota.plan_expires_at
    if exp is not None:
        now = now or datetime.now(timezone.utc)
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp <= now:
            return EXPIRED_PLAN_TIER, True
    return quota.plan_tier or DEFAULT_PLAN_TIER, False


async def get_effective_limits(user_id: UUID, db: AsyncSession) -> dict:
    """Effective limits for a user, as decided by the active limits policy.

    Returns the limit keys plus 'plan_tier' and 'override_active'.
    """
    from app.core.limits_policy import get_limits_policy

    return await get_limits_policy().effective_limits(user_id, db)


# ---------- DB helpers ---------------------------------------------------------

async def get_user_tier(user_id: UUID, db: AsyncSession) -> str:
    from app.core.limits_policy import get_limits_policy

    return await get_limits_policy().user_tier(user_id, db)


async def count_running_workspaces_for_user(user_id: UUID, db: AsyncSession) -> int:
    """Count all quota-reserving runtimes for the user's lease.

    ``quota_owner_id`` is authoritative for new rows.  The ``NULL`` branch is
    retained for legacy rows and counts a shared room only once for a member.
    """
    res = await db.execute(
        select(func.count(func.distinct(Workspace.id)))
        .join(RoomUserLink, RoomUserLink.room_id == Workspace.room_id)
        .where(RoomUserLink.user_id == user_id)
        .where(Workspace.status.in_(_counted_states()))
        .where(
            or_(
                Workspace.quota_owner_id == user_id,
                Workspace.quota_owner_id.is_(None),
            )
        )
    )
    return res.scalar_one() or 0


async def reserved_workspace_slots(
    user_id: UUID,
    db: AsyncSession,
    *,
    exclude_room_id: Optional[UUID] = None,
) -> int:
    """Return the number of slots reserved by *user_id* right now."""
    query = (
        select(func.count(func.distinct(Workspace.id)))
        .join(RoomUserLink, RoomUserLink.room_id == Workspace.room_id)
        .where(RoomUserLink.user_id == user_id)
        .where(Workspace.status.in_(_counted_states()))
        .where(
            or_(
                Workspace.quota_owner_id == user_id,
                Workspace.quota_owner_id.is_(None),
            )
        )
    )
    if exclude_room_id is not None:
        query = query.where(Workspace.room_id != exclude_room_id)
    result = await db.execute(query)
    return int(result.scalar_one() or 0)


async def reserved_workspace_memory_mb(
    user_id: UUID,
    db: AsyncSession,
    *,
    exclude_room_id: Optional[UUID] = None,
) -> int:
    """Memory limits of the runtimes holding *user_id*'s slots (§3a pool).

    The same runtimes ``reserved_workspace_slots`` counts, summed by limit.
    """
    query = (
        select(Workspace.id, Workspace.mem_limit_mb)
        .join(RoomUserLink, RoomUserLink.room_id == Workspace.room_id)
        .where(RoomUserLink.user_id == user_id)
        .where(Workspace.status.in_(_counted_states()))
        .where(
            or_(
                Workspace.quota_owner_id == user_id,
                Workspace.quota_owner_id.is_(None),
            )
        )
        .distinct()
    )
    if exclude_room_id is not None:
        query = query.where(Workspace.room_id != exclude_room_id)
    rows = (await db.execute(query)).all()
    return sum(int(mem or 0) for _, mem in rows)


def memory_pool_mb(limits: dict) -> Optional[int]:
    """``rooms_memory_pool_mb`` as a finite integer, or None — no pool."""
    raw = limits.get("rooms_memory_pool_mb")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw


class WorkspaceQuotaManager:
    """Single quota/lease facade used by lifecycle operations.

    The per-owner lock closes the race between two concurrent room opens in a
    backend process.  The row lock in callers protects existing workspace
    records; durable status and ``quota_owner_id`` make recovery after a
    process restart deterministic.
    """

    async def effective_max_running(self, owner_id: UUID, db: AsyncSession) -> int:
        limits = await get_effective_limits(owner_id, db)
        raw = limits.get("max_running")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError("effective max_running must be a finite integer")
        return raw

    async def count_reserved_slots(
        self,
        owner_id: UUID,
        db: AsyncSession,
        *,
        exclude_room_id: Optional[UUID] = None,
    ) -> int:
        return await reserved_workspace_slots(
            owner_id, db, exclude_room_id=exclude_room_id
        )

    async def reserve(
        self,
        owner_id: UUID,
        room_id: UUID,
        db: AsyncSession,
    ) -> dict[str, int]:
        async with _quota_lock(owner_id):
            max_running = await self.effective_max_running(owner_id, db)
            reserved = await reserved_workspace_slots(
                owner_id, db, exclude_room_id=room_id
            )
            if reserved >= max_running:
                raise WorkspaceQuotaExceeded(owner_id, max_running, reserved)
            return {"max_running": max_running, "reserved": reserved}

    async def reserve_and_run(
        self,
        owner_id: UUID,
        room_id: UUID,
        db: AsyncSession,
        operation: Callable[[], Awaitable[_ReservationResult]],
        *,
        mem_mb: Optional[int] = None,
    ) -> _ReservationResult:
        """Reserve and materialize a lease under one per-owner critical section.

        With *mem_mb* the start must also fit the owner's rooms memory pool
        (internal design notes §3a): the limits of the runtimes already
        holding slots plus this one. *operation* must record *mem_mb* on the
        workspace row, so the next start in this section counts it.

        The start must also fit the server-wide capacity (core/host_capacity.py,
        internal design notes stage 12.1); that check and *operation* run under the
        process-wide host lock, taken after the owner's lock.
        """
        from app.core import host_capacity

        async with _quota_lock(owner_id):
            max_running = await self.effective_max_running(owner_id, db)
            reserved = await reserved_workspace_slots(
                owner_id, db, exclude_room_id=room_id
            )
            if reserved >= max_running:
                raise WorkspaceQuotaExceeded(owner_id, max_running, reserved)
            if mem_mb is not None:
                pool = memory_pool_mb(await get_effective_limits(owner_id, db))
                if pool is not None:
                    allocated = await reserved_workspace_memory_mb(
                        owner_id, db, exclude_room_id=room_id
                    )
                    if allocated + mem_mb > pool:
                        raise WorkspaceMemoryPoolExceeded(owner_id, pool, allocated, mem_mb)
            async with host_capacity.host_lock():
                await host_capacity.check_workspace_start(db, room_id, mem_mb)
                return await operation()

    async def heartbeat(
        self,
        workspace: Workspace,
        db: AsyncSession,
        *,
        at: Optional[datetime] = None,
    ) -> None:
        workspace.last_heartbeat_at = at or datetime.now(timezone.utc)
        db.add(workspace)
        await db.flush()

    async def stale_workspaces(
        self,
        db: AsyncSession,
        *,
        older_than_sec: int,
        now: Optional[datetime] = None,
    ) -> list[Workspace]:
        now = now or datetime.now(timezone.utc)
        cutoff = now.timestamp() - max(1, int(older_than_sec))
        # Datetime comparisons are kept in Python for SQLite parity.
        result = await db.execute(
            select(Workspace).where(Workspace.status.in_(_counted_states()))
        )
        stale: list[Workspace] = []
        for workspace in result.scalars().all():
            heartbeat = workspace.last_heartbeat_at or workspace.started_at
            if heartbeat is None:
                continue
            if heartbeat.tzinfo is None:
                heartbeat = heartbeat.replace(tzinfo=timezone.utc)
            if heartbeat.timestamp() < cutoff:
                stale.append(workspace)
        return stale


quota_manager = WorkspaceQuotaManager()


def quota_owner_lock(owner_id: UUID) -> asyncio.Lock:
    """Expose the owner-scoped lock for compound lifecycle workflows."""
    return _quota_lock(owner_id)


# ---------- Hour window (internal design notes §3.4) ---------------------------

def add_months(dt: datetime, months: int) -> datetime:
    """``dt`` moved by whole calendar months, the day clamped to the month's end."""
    index = dt.year * 12 + dt.month - 1 + months
    year, month = divmod(index, 12)
    day = min(dt.day, calendar.monthrange(year, month + 1)[1])
    return dt.replace(year=year, month=month + 1, day=day)


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def hours_window(
    quota: Optional[UserQuota], now: Optional[datetime] = None
) -> tuple[datetime, datetime]:
    """[start, end) of the month the monthly hour limits count in.

    A paid plan (``plan_period_start`` set by billing, plan not expired)
    counts in months anchored at its period start: a client who paid on the
    25th gets the whole 40 h until the 25th of the next month, and a renewal
    starts a fresh window together with the payment. The anchor may be the
    start of an already-paid *next* period (a prepaid month starts where the
    current one ends); stepping back from it by whole months finds the
    current one. An expired period does not open a new window.

    Everything else (free, a plan an admin set by hand) counts in the
    calendar month, as before.
    """
    now = _aware(now or datetime.now(timezone.utc))
    tier_expired = resolve_tier(quota, now)[1]
    anchor = quota.plan_period_start if quota is not None else None
    if anchor is None or tier_expired:
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return start, add_months(start, 1)

    anchor = _aware(anchor)
    months = (now.year - anchor.year) * 12 + now.month - anchor.month
    if add_months(anchor, months) > now:
        months -= 1
    if quota.plan_expires_at is not None:
        period_end = _aware(quota.plan_expires_at)
        if add_months(anchor, months) >= period_end > add_months(anchor, months - 1):
            months -= 1
    return add_months(anchor, months), add_months(anchor, months + 1)


async def get_hours_window(
    user_id: UUID, db: AsyncSession, now: Optional[datetime] = None
) -> tuple[datetime, datetime]:
    res = await db.execute(select(UserQuota).where(UserQuota.user_id == user_id))
    return hours_window(res.scalar_one_or_none(), now)


async def monthly_hours_consumed(user_id: UUID, db: AsyncSession) -> float:
    """Total workspace hours used in the current hour window (:func:`hours_window`)."""
    now = datetime.now(timezone.utc)
    month_start, _ = await get_hours_window(user_id, db, now)

    res = await db.execute(
        select(WorkspaceUsage)
        .join(Workspace, Workspace.id == WorkspaceUsage.workspace_id)
        .join(RoomUserLink, RoomUserLink.room_id == Workspace.room_id)
        .where(RoomUserLink.user_id == user_id)
        .where(
            # sessions that overlap with current month
            WorkspaceUsage.started_at < now,
            (WorkspaceUsage.stopped_at.is_(None)) | (WorkspaceUsage.stopped_at >= month_start),
        )
    )
    usages = res.scalars().all()

    total_seconds = 0.0
    for u in usages:
        # Columns are TIMESTAMPTZ (aware UTC); the tzinfo checks below only
        # defend against stragglers (e.g. SQLite in tests returns naive).
        start = u.started_at
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        start = max(start, month_start)

        end = u.stopped_at
        if end is None:
            end = now
        elif end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        if end > start:
            total_seconds += (end - start).total_seconds()

    return total_seconds / 3600.0


# ---------- Enforcement --------------------------------------------------------

async def check_start_allowed(user: User, db: AsyncSession) -> None:
    """Raise HTTP 402/429 if the user's effective limits don't permit starting a workspace."""
    limits = await get_effective_limits(user.id, db)
    tier = limits["plan_tier"]

    if not limits["max_running"]:
        raise HTTPException(
            status_code=402,
            detail=(
                PLAN_EXPIRED_READ_ONLY
                if tier == EXPIRED_PLAN_TIER else
                f"Your plan '{tier}' does not include a workspace. "
                "Upgrade your plan to get code execution."
            ),
        )

    running = await count_running_workspaces_for_user(user.id, db)
    if running >= limits["max_running"]:
        raise HTTPException(
            status_code=429,
            detail=(
                f"The limit of running workspaces for the '{tier}' plan has been reached: "
                f"{running}/{limits['max_running']}. "
                "Stop another workspace or upgrade your plan."
            ),
        )

    if limits["monthly_hours"] is not None:
        hours = await monthly_hours_consumed(user.id, db)
        window_start, _ = await get_hours_window(user.id, db)
        # §3.4: month hours left + what live hour packs hold > 0.
        packs = await work_gates.extra_hours(user.id, window_start, db)
        extra = packs.extra if packs is not None else 0.0
        if hours >= limits["monthly_hours"] + extra:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"The monthly workspace hours limit is exhausted "
                    f"({hours:.1f} / {limits['monthly_hours'] + extra:g} h). "
                    "Renew the plan or buy more hours."
                ),
            )


async def hours_limit_exhausted(user_id: UUID, db: AsyncSession, monthly_limit: float) -> bool:
    """The owner used up ``monthly_limit`` plus the extra hours a plugin gate
    grants (bought packs, core/work_gates.py) in the current hour window."""
    hours = await monthly_hours_consumed(user_id, db)
    window_start, _ = await get_hours_window(user_id, db)
    packs = await work_gates.extra_hours(user_id, window_start, db)
    return hours >= monthly_limit + (packs.extra if packs is not None else 0.0)


# ---------- Usage summary ------------------------------------------------------

async def get_usage_summary(user: User, db: AsyncSession) -> dict:
    """Return workspace usage summary for the /me/usage endpoint."""
    limits = await get_effective_limits(user.id, db)
    running = await count_running_workspaces_for_user(user.id, db)

    hours = await monthly_hours_consumed(user.id, db)
    window_start, window_end = await get_hours_window(user.id, db)
    # Packs are spent only on a live plan: an expired one stops the work anyway.
    packs = (
        await work_gates.extra_hours(user.id, window_start, db)
        if limits["plan_tier"] != EXPIRED_PLAN_TIER else None
    )
    pack_extra = packs.extra if packs is not None else 0.0

    res = await db.execute(
        select(func.count(Workspace.id))
        .join(RoomUserLink, RoomUserLink.room_id == Workspace.room_id)
        .where(RoomUserLink.user_id == user.id)
    )
    rooms_with_workspace = res.scalar_one() or 0

    return {
        "plan_tier": limits["plan_tier"],
        "override_active": limits["override_active"],
        "workspaces_running": running,
        "max_running": limits["max_running"],
        "rooms_with_workspace": rooms_with_workspace,
        "max_rooms": limits["max_rooms"],
        "hours_this_month": round(hours, 2),
        "max_hours_per_month": limits["monthly_hours"],
        # T8: agents stopped and nothing billed until the window ends.
        "hours_exhausted": (
            limits["monthly_hours"] is not None and hours >= limits["monthly_hours"] + pack_extra
        ),
        # §3.6b: bought hours on top of the month — what is left in live packs,
        # what this window already took from them, the nearest expiry.
        "hour_packs": {
            "remaining_hours": round(packs.remaining, 2) if packs else 0.0,
            "used_this_window": round(packs.charged, 2) if packs else 0.0,
            "next_expires_at": packs.next_expires_at.isoformat() if packs and packs.next_expires_at else None,
            "packs": packs.packs if packs else 0,
        },

        # The month hours_this_month counts in: the paid period for a paid
        # plan, the calendar month otherwise (internal design notes §3.4).
        "hours_period_start": window_start.isoformat(),
        "hours_period_end": window_end.isoformat(),
        "mem_limit_mb": limits["mem_limit_mb"],
        # internal design notes §3a: rows of "Settings → Plan and limits".
        "rooms_memory_pool_mb": limits.get("rooms_memory_pool_mb"),
        "room_max_memory_mb": limits.get("room_max_memory_mb"),
        "room_storage_quota_mb": limits.get("room_storage_quota_mb"),
        "link_mcp_max": limits.get("link_mcp_max"),
        **(await _plan_dates(user.id, db)),
        **(await _memory_allocation(user.id, db)),
        "link_mcp_count": await _link_mcp_count(user.id, db),
        # T3 (§5.2): room files against room_storage_quota_mb.
        **(await _storage_block(user.id, db)),
        # T11 (§5.5): an ended plan's timeline, frozen and archived rooms.
        "data_lifecycle": await _lifecycle_block(user.id, db),
    }


async def _lifecycle_block(user_id: UUID, db: AsyncSession) -> Optional[dict]:
    from app.core.work_gates import lifecycle_payload

    try:
        return await lifecycle_payload(user_id, db)
    except Exception as exc:
        logger.warning("usage: data lifecycle block failed for %s: %s", user_id, exc)
        return None


async def _plan_dates(user_id: UUID, db: AsyncSession) -> dict:
    res = await db.execute(select(UserQuota).where(UserQuota.user_id == user_id))
    quota = res.scalar_one_or_none()

    def _iso(value: Optional[datetime]) -> Optional[str]:
        return _aware(value).isoformat() if value is not None else None

    return {
        "plan_expires_at": _iso(quota.plan_expires_at if quota else None),
        "plan_period_start": _iso(quota.plan_period_start if quota else None),
    }


async def _memory_allocation(user_id: UUID, db: AsyncSession) -> dict:
    """§3a "Room memory": the memory limits of the user's running rooms and
    what they hold now ("Currently used", resident memory per room)."""
    from app.core.room_memory import resident_memory
    from app.models.room import Room

    rows = (
        await db.execute(
            select(Room.id, Room.name, Workspace.mem_limit_mb, Workspace.status)
            .join(Workspace, Workspace.room_id == Room.id)
            .where(Room.creator_id == user_id)
            .where(Workspace.status.in_(_counted_states()))
        )
    ).all()
    resident = await resident_memory([room_id for room_id, _, _, status in rows if status == "running"])
    measured = [value for value in resident.values() if value is not None]
    return {
        "memory_allocated_mb": sum(int(mem or 0) for _, _, mem, _ in rows),
        # None when no running room could be measured.
        "memory_used_mb": round(sum(measured), 1) if measured else None,
        "memory_rooms": [
            {
                "room_id": str(room_id),
                "name": name or "",
                "mem_limit_mb": int(mem or 0),
                "rss_mb": resident.get(str(room_id)),
            }
            for room_id, name, mem, _ in rows
        ],
    }


async def _link_mcp_count(user_id: UUID, db: AsyncSession) -> int:
    """MCPs by link the user has installed — what ``link_mcp_max`` counts (T7)."""
    from app.core.mcp_discovery import is_http_link
    from app.models.market import UserInstalledItem

    items = (
        await db.execute(
            select(UserInstalledItem).where(
                UserInstalledItem.user_id == user_id,
                UserInstalledItem.type == "mcp",
                UserInstalledItem.is_custom.is_(True),
            )
        )
    ).scalars().all()
    return sum(1 for item in items if is_http_link(item))


async def _storage_block(user_id: UUID, db: AsyncSession) -> dict:
    from app.core.room_storage import storage_status, usage_payload

    return usage_payload(await storage_status(user_id, db))
