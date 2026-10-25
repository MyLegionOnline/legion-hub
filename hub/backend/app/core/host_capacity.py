"""Server-wide capacity: what all users together may run on this machine.

internal design notes stage 12.1. The per-user limits (``max_running``,
``rooms_memory_pool_mb``, ``mcp_personal_*``) say what one user may run; this
module says what the machine can hold, so many users starting at once get a
clear "server is busy" instead of an out-of-memory kill that also takes down
the services next to the Hub.

Configured in ``legion.toml`` ``[capacity]``, overridden by
``LEGION_CAPACITY_<KEY>``. Both keys default to no cap:

* ``max_workspaces`` — room workspaces running at the same time, all users;
* ``memory_mb`` — the memory limits of running room workspaces and personal
  MCP servers added up, all users.

The check is serialized by a process-wide lock, like the per-owner quota lock
in ``workspace_quota``: the Hub runs one backend process.
"""

from __future__ import annotations

import asyncio
import difflib
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

CONFIG_PATH_ENV = "LEGION_CONFIG"
DEFAULT_CONFIG_PATH = "/etc/legion/legion.toml"
ENV_PREFIX = "LEGION_CAPACITY_"
KEYS = ("max_workspaces", "memory_mb")
_UNLIMITED_WORDS = frozenset({"none", "null", "unlimited"})

# Personal MCP runtime states that hold memory (personal_mcps._enforce_start_quota).
ACTIVE_PERSONAL_MCP_STATES = frozenset({"starting", "running", "verifying", "verified", "stopping"})

SERVER_BUSY = "server_busy"
SERVER_BUSY_TEXT = (
    "The server is at capacity right now: no more workspaces can start. "
    "Try again in a few minutes."
)


class CapacityConfigError(RuntimeError):
    """legion.toml [capacity] or a LEGION_CAPACITY_* variable is wrong; startup must stop."""


class HostCapacityExceeded(RuntimeError):
    """A start would go over the server-wide capacity."""

    def __init__(self, kind: str, limit: int, used: int, requested: int):
        super().__init__(f"server capacity exceeded: {kind}")
        self.kind = kind
        self.limit = limit
        self.used = used
        self.requested = requested


@dataclass(frozen=True)
class HostCapacity:
    max_workspaces: Optional[int] = None
    memory_mb: Optional[int] = None

    @property
    def unlimited(self) -> bool:
        return self.max_workspaces is None and self.memory_mb is None


def _coerce(key: str, raw: Any, source: str) -> Optional[int]:
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in _UNLIMITED_WORDS:
            return None
        try:
            raw = int(text)
        except ValueError:
            raise CapacityConfigError(
                f"{source}: {key} must be a whole number or 'unlimited', got {raw!r}"
            ) from None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise CapacityConfigError(
            f"{source}: {key} must be a whole number or 'unlimited', got {raw!r}"
        )
    if raw < 0:
        raise CapacityConfigError(f"{source}: {key} must not be negative, got {raw}")
    return raw


def _check_key(key: str, source: str) -> str:
    if key in KEYS:
        return key
    hint = difflib.get_close_matches(key, KEYS, n=1)
    suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
    raise CapacityConfigError(
        f"{source}: unknown key {key!r}.{suggestion} Valid keys: {', '.join(KEYS)}"
    )


def load(path: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> HostCapacity:
    """No cap by default, then ``[capacity]``, then ``LEGION_CAPACITY_*``."""
    env = os.environ if environ is None else environ
    values: dict[str, Optional[int]] = {key: None for key in KEYS}

    explicit = path or env.get(CONFIG_PATH_ENV)
    config_file = Path(explicit or DEFAULT_CONFIG_PATH)
    if config_file.is_file():
        try:
            with config_file.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise CapacityConfigError(f"{config_file}: invalid TOML: {exc}") from exc
        table = data.get("capacity", {})
        if not isinstance(table, dict):
            raise CapacityConfigError(f"{config_file}: [capacity] must be a table")
        source = f"{config_file} [capacity]"
        for key, raw in table.items():
            values[_check_key(key, source)] = _coerce(key, raw, source)
    elif explicit:
        raise CapacityConfigError(f"{CONFIG_PATH_ENV} points to {explicit!r}, which is not a file")

    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX):
            continue
        key = name[len(ENV_PREFIX):].lower()
        _check_key(key, name)
        # Compose passes every key through, unset ones as "": the file decides.
        if str(raw).strip():
            values[key] = _coerce(key, raw, name)
    return HostCapacity(**values)


_capacity: Optional[HostCapacity] = None
_lock: Optional[asyncio.Lock] = None


def get_capacity() -> HostCapacity:
    """The configured capacity, read once; call it at startup to fail early."""
    global _capacity
    if _capacity is None:
        _capacity = load()
    return _capacity


def reset_capacity() -> None:
    global _capacity
    _capacity = None


def host_lock() -> asyncio.Lock:
    """One lock for every server-wide admission check in this process."""
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


def _counted_workspace_states() -> Iterable[str]:
    from app.core.workspace_quota import _counted_states

    return _counted_states()


async def workspace_usage(
    db: AsyncSession, *, exclude_room_id: Optional[UUID] = None
) -> tuple[int, int]:
    """(running workspaces, their memory limits in MB), all users."""
    from app.models.workspace import Workspace

    query = select(Workspace.mem_limit_mb).where(
        Workspace.status.in_(list(_counted_workspace_states()))
    )
    if exclude_room_id is not None:
        query = query.where(Workspace.room_id != exclude_room_id)
    rows = (await db.execute(query)).scalars().all()
    return len(rows), sum(int(mem or 0) for mem in rows)


async def personal_mcp_memory(
    db: AsyncSession, *, exclude_item_id: Optional[UUID] = None
) -> int:
    """Memory reserved by active personal MCP servers, all users, MB.

    Managed runtimes (``personalmcpruntime``) plus servers started through the
    older ``/user-installed-items/{id}/personal/start`` path, whose state lives
    in ``custom_config.runtime``.
    """
    from app.models.market import UserInstalledItem
    from app.models.personal_mcp import PersonalMcpRuntime

    query = select(PersonalMcpRuntime.memory_reservation_mb).where(
        PersonalMcpRuntime.actual_status.in_(list(ACTIVE_PERSONAL_MCP_STATES))
    )
    if exclude_item_id is not None:
        query = query.where(PersonalMcpRuntime.installed_item_id != exclude_item_id)
    total = sum(int(mem or 0) for mem in (await db.execute(query)).scalars().all())

    legacy = await db.execute(
        select(UserInstalledItem.id, UserInstalledItem.custom_config).where(
            UserInstalledItem.type == "mcp",
            UserInstalledItem.is_custom.is_(True),
        )
    )
    for item_id, cfg in legacy.all():
        if item_id == exclude_item_id or not isinstance(cfg, dict):
            continue
        runtime = cfg.get("runtime")
        if not isinstance(runtime, dict) or runtime.get("status") not in {"starting", "running"}:
            continue
        total += _int_or_zero(runtime.get("mem_limit_mb", cfg.get("mem_limit_mb")))
    return total


def _int_or_zero(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


async def check_workspace_start(
    db: AsyncSession, room_id: UUID, mem_mb: Optional[int]
) -> None:
    """Raise HostCapacityExceeded if the room's workspace does not fit.

    Call it under :func:`host_lock` together with the write that marks the
    workspace as starting, so two starts cannot both take the last place.
    """
    capacity = get_capacity()
    if capacity.unlimited:
        return
    running, workspace_mb = await workspace_usage(db, exclude_room_id=room_id)
    if capacity.max_workspaces is not None and running >= capacity.max_workspaces:
        raise HostCapacityExceeded("workspaces", capacity.max_workspaces, running, 1)
    if capacity.memory_mb is not None:
        used = workspace_mb + await personal_mcp_memory(db)
        requested = int(mem_mb or 0)
        if used + requested > capacity.memory_mb:
            raise HostCapacityExceeded("memory", capacity.memory_mb, used, requested)


async def check_personal_mcp_start(
    db: AsyncSession, installed_item_id: UUID, mem_mb: int
) -> None:
    """Raise HostCapacityExceeded if a personal MCP server does not fit in memory.

    Only memory counts: ``max_workspaces`` is about room workspaces. The caller
    commits the new reservation after this check, outside :func:`host_lock`, so
    a start that lands in the same instant as another one may not see it and
    the total can go over by that one server's memory. Keep some headroom
    between ``memory_mb`` and the machine's real memory.
    """
    capacity = get_capacity()
    if capacity.memory_mb is None:
        return
    _, workspace_mb = await workspace_usage(db)
    used = workspace_mb + await personal_mcp_memory(db, exclude_item_id=installed_item_id)
    if used + int(mem_mb or 0) > capacity.memory_mb:
        raise HostCapacityExceeded("memory", capacity.memory_mb, used, int(mem_mb or 0))


async def usage_summary(db: AsyncSession) -> dict:
    """What the server holds now against its capacity (operator view)."""
    capacity = get_capacity()
    running, workspace_mb = await workspace_usage(db)
    mcp_mb = await personal_mcp_memory(db)
    return {
        "max_workspaces": capacity.max_workspaces,
        "memory_mb": capacity.memory_mb,
        "workspaces_running": running,
        "memory_allocated_mb": workspace_mb + mcp_mb,
        "workspace_memory_mb": workspace_mb,
        "personal_mcp_memory_mb": mcp_mb,
    }
