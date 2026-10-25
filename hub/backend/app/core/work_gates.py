"""Gates that decide whether a room may work (internal design notes stage 3, module 3).

The core owns the *questions* — may this room take a new user message, may it
open, how many extra hours does the owner hold beyond ``monthly_hours`` — and
a plugin answers them through ``PluginHost.add_work_gate``. A Hub without
plugins has no gates: the only refusal left is the room-files quota, a limit
mechanism of the core (core/room_storage.py).

A gate is an object with any of these async methods (a missing one is skipped):

* ``work_refusal(room_id, db) -> WorkRefusal | None`` — no new agent work;
* ``open_refusal(room_id, db) -> WorkRefusal | None`` — the room does not open;
* ``extra_hours(user_id, window_start, db) -> HourAllowance | None`` — hours
  the owner may use beyond the ``monthly_hours`` limit;
* ``start_refusal(room_id, db) -> StartRefusal | None`` — the room's container
  does not start (files in the archive, room frozen);
* ``read_only(room_id, db) -> {"reason", "message"} | None`` — the room opens
  for reading only and takes no file writes;
* ``restore_on_open(room_id, db) -> {"reason", "message"} | None`` — bring an
  archived room's files back as the room opens; a read-only answer when they
  could not come back;
* ``files_refusal(room_id, db) -> {"reason", "message"} | None`` — the room's
  files are not on the disk (archived);
* ``lifecycle_payload(user_id, db) -> dict | None`` — the ``data_lifecycle``
  block of ``/me/usage``.

The first gate with an answer wins. A gate that raises is logged and skipped:
a guard that cannot decide must not stop the room (the reconciler re-checks).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.plugins import get_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkRefusal:
    text: str
    #: Event type of the notice posted to the room's history.
    event_type: str
    #: ``task.pause_reason`` / error code / ``room_closed.reason`` of the refusal.
    reason: str


@dataclass(frozen=True)
class StartRefusal:
    """Why a room's container does not start; ``reason`` is the error code."""

    reason: str
    text: str
    status_code: int = 402


@dataclass(frozen=True)
class HourAllowance:
    """Hours beyond the plan's month that an owner holds (bought packs)."""

    charged: float = 0.0           # this window's hours already taken from them
    remaining: float = 0.0         # what they still hold
    next_expires_at: Optional[datetime] = None
    packs: int = 0

    @property
    def extra(self) -> float:
        """Hours beyond the plan's month the owner may use in this window."""
        return self.charged + self.remaining


async def _first(method: str, *args: Any) -> Any:
    for gate in get_registry().work_gates:
        fn = getattr(gate, method, None)
        if fn is None:
            continue
        try:
            answer = await fn(*args)
        except Exception as exc:
            logger.warning(
                "work gate %s.%s failed (%s: %s); skipping",
                type(gate).__name__, method, type(exc).__name__, exc,
            )
            continue
        if answer is not None:
            return answer
    return None


async def work_refusal(room_id: UUID, db: AsyncSession) -> Optional[WorkRefusal]:
    """Why ``room_id`` may not start new agent work, or None."""
    return await _first("work_refusal", room_id, db)


async def open_refusal(room_id: UUID, db: AsyncSession) -> Optional[WorkRefusal]:
    """Why ``room_id`` may not even be opened, or None."""
    return await _first("open_refusal", room_id, db)


async def extra_hours(
    user_id: UUID, window_start: datetime, db: AsyncSession
) -> Optional[HourAllowance]:
    """Hours the owner may use beyond ``monthly_hours`` in this window, or None."""
    return await _first("extra_hours", user_id, window_start, db)


async def start_refusal(room_id: UUID, db: AsyncSession) -> Optional[StartRefusal]:
    """Why ``room_id``'s container may not start (beyond the quotas), or None."""
    return await _first("start_refusal", room_id, db)


async def room_read_only(room_id: UUID, db: AsyncSession) -> Optional[dict[str, str]]:
    """``{reason, message}`` when ``room_id`` opens for reading only, or None."""
    return await _first("read_only", room_id, db)


async def restore_on_open(room_id: UUID, db: AsyncSession) -> Optional[dict[str, str]]:
    """Bring an archived room's files back as it opens. None — nothing to
    restore, or restored; ``{reason, message}`` — they could not come back and
    the room opens for reading only."""
    return await _first("restore_on_open", room_id, db)


async def lifecycle_payload(user_id: UUID, db: AsyncSession) -> Optional[dict[str, Any]]:
    """The ``data_lifecycle`` block of ``/me/usage`` (None without a plugin)."""
    return await _first("lifecycle_payload", user_id, db)


async def require_room_writable(room_id: UUID, db: AsyncSession) -> None:
    """File API gate: a read-only room takes no writes or uploads (409)."""
    from fastapi import HTTPException

    read_only = await room_read_only(room_id, db)
    if read_only is not None:
        raise HTTPException(
            status_code=409,
            detail={"error_code": read_only["reason"], "message": read_only["message"]},
        )


async def require_room_files(room_id: UUID, db: AsyncSession) -> None:
    """File API gate for reads: an archived room has no files on the disk (409)."""
    from fastapi import HTTPException

    refusal = await _first("files_refusal", room_id, db)
    if refusal is not None:
        raise HTTPException(
            status_code=409,
            detail={"error_code": refusal["reason"], "message": refusal["message"]},
        )


async def refuse_room_work(room_id: UUID, db: AsyncSession) -> bool:
    """Gate for a user message: True, with a notice in the room, when the room
    may not work now — a plugin gate says no, or the owner's room files are at
    the quota (core/room_storage.py)."""
    from app.core import room_storage
    from app.core.events import emit_event

    refusal = await work_refusal(room_id, db)
    if refusal is not None:
        await emit_event(
            room_id, refusal.event_type, refusal.text, db,
            metadata={"reason": refusal.reason},
        )
        logger.info("work gate: refused new work in room %s (%s)", room_id, refusal.reason)
        return True

    text = await room_storage.room_storage_refusal(room_id, db)
    if text is not None:
        await emit_event(
            room_id, room_storage.STORAGE_FULL_EVENT, text, db,
            metadata={"reason": room_storage.PAUSE_REASON},
        )
        logger.info("work gate: refused new work in room %s (files quota)", room_id)
        return True
    return False


async def run_reconciler_steps(db: AsyncSession) -> None:
    """Run the plugins' reconciler steps, each in its own try/except.

    A step is an async callable ``step(db)``. A failing step is rolled back and
    logged without stopping the others or the tick.
    """
    for step in get_registry().reconciler_steps:
        name = getattr(step, "__name__", None) or type(step).__name__
        try:
            await step(db)
        except Exception as exc:
            await db.rollback()
            logger.warning("reconciler: plugin step %s failed: %s", name, exc)
