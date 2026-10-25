"""Persist and broadcast the room-private-files Git policy state.

The workspaced service only knows about a volume and returns bounded Git
metadata. This module owns the backend-side room marker and the persistent
observability events, keeping their payloads deliberately free of file bytes.
"""
from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.events import emit_event
from app.models.room import Room


USERFILES_POLICY_KEY = "userfiles_policy"
USERFILES_POLICY_READY = "ready"
USERFILES_POLICY_CONFLICT = "conflict"
USERFILES_POLICY_PATHS_KEY = "userfiles_policy_paths"
USERFILES_POLICY_SOURCE_KEY = "userfiles_policy_source"


class UserfilesProtectionUnavailable(RuntimeError):
    """workspaced cannot vouch that ``.userfiles`` is kept out of Git."""


async def ensure_userfiles_protected(room_id: UUID | str) -> dict[str, Any]:
    """Refuse to write a private file unless workspaced confirms Git exclusion.

    The exclusion lives in workspaced (``.git/info/exclude`` plus a staging
    pathspec). The backend and workspaced are deployed separately and can run
    different versions — a backend that already routes uploads to
    ``/.userfiles`` talking to a workspaced that predates the rule commits the
    file on the next ``git add -A`` and pushes it. So the check is made on the
    write path itself, not only at startup: the idempotent backfill call writes
    the rule and reports ``userfiles_protected`` from the file it just read.
    A build that does not know the field omits it, which fails closed here.
    """
    from app.core import workspaced_client as wsd

    result = await wsd.backfill_local_excludes(str(room_id))
    if result.get("userfiles_protected") is not True:
        raise UserfilesProtectionUnavailable(
            "workspaced did not confirm that .userfiles is excluded from Git "
            "(restart workspaced so it runs the current code)"
        )
    return result


def room_userfiles_policy(room: Room) -> str:
    """Return the durable policy state, defaulting old rooms to ``ready``."""
    git_config = getattr(room, "git_config", None)
    config = git_config if isinstance(git_config, dict) else {}
    return str(config.get(USERFILES_POLICY_KEY) or USERFILES_POLICY_READY)


def room_userfiles_policy_conflict(room: Room) -> bool:
    return room_userfiles_policy(room) == USERFILES_POLICY_CONFLICT


def _entries(result: dict[str, Any], *, source: str) -> list[dict[str, Any]]:
    raw = result.get("userfiles_tracked_details") or []
    entries: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            try:
                size = max(0, int(item.get("size") or 0))
            except (TypeError, ValueError):
                size = 0
            entries.append(
                {
                    "path": path,
                    "size": size,
                    "source": str(item.get("source") or source),
                    "error": str(item.get("error") or "userfiles_policy_conflict"),
                }
            )
    if entries:
        return entries[:20]

    # Older workspaced responses carry only path strings. Preserve support for
    # those deployments while keeping the event shape stable.
    for path in result.get("userfiles_tracked") or []:
        path = str(path or "").strip()
        if path:
            entries.append(
                {
                    "path": path,
                    "size": 0,
                    "source": source,
                    "error": "userfiles_policy_conflict",
                }
            )
    return entries[:20]


async def apply_userfiles_policy(
    room: Room,
    result: dict[str, Any],
    db: AsyncSession,
    *,
    source: str,
) -> dict[str, Any]:
    """Apply a workspaced policy result and emit only bounded metadata.

    A conflict is persisted in ``Room.git_config`` so every upload surface can
    refuse the private target until the remote is fixed and a later Pull or
    backfill rechecks it. The helper is idempotent: repeated startup/backfill
    observations do not create duplicate chat events.
    """
    entries = _entries(result, source=source)
    conflict = bool(result.get("userfiles_policy_conflict") or entries)
    config = dict(room.git_config or {})
    previous_policy = room_userfiles_policy(room)
    previous_paths = list(config.get(USERFILES_POLICY_PATHS_KEY) or [])
    paths = [entry["path"] for entry in entries]

    if conflict:
        changed = (
            previous_policy != USERFILES_POLICY_CONFLICT
            or previous_paths != paths
            or config.get(USERFILES_POLICY_SOURCE_KEY) != source
        )
        config[USERFILES_POLICY_KEY] = USERFILES_POLICY_CONFLICT
        config[USERFILES_POLICY_PATHS_KEY] = paths
        config[USERFILES_POLICY_SOURCE_KEY] = source
    else:
        changed = previous_policy == USERFILES_POLICY_CONFLICT
        config.pop(USERFILES_POLICY_KEY, None)
        config.pop(USERFILES_POLICY_PATHS_KEY, None)
        config.pop(USERFILES_POLICY_SOURCE_KEY, None)

    if changed:
        room.git_config = config
        flag_modified(room, "git_config")
        await db.commit()

    if conflict and changed:
        for entry in entries:
            await emit_event(
                room_id=room.id,
                event_type="USERFILES_TRACKED_BLOCKED",
                content="The external Git repository tracks the private-files path; upload paused",
                db=db,
                metadata=entry,
            )
    elif not conflict and previous_policy == USERFILES_POLICY_CONFLICT:
        await emit_userfiles_ready(
            room.id,
            db,
            source="policy_recheck",
            size=0,
            error=None,
        )

    return {
        "conflict": conflict,
        "paths": paths,
        "changed": changed,
    }


async def emit_userfiles_ready(
    room_id: UUID,
    db: AsyncSession,
    *,
    source: str,
    size: int = 0,
    error: str | None = None,
) -> None:
    """Record a readiness/repair observation without exposing file content."""
    await emit_event(
        room_id=room_id,
        event_type="USERFILES_READY",
        content="The room's private-files directory is ready",
        db=db,
        metadata={
            "path": "/.userfiles",
            "size": max(0, int(size)),
            "source": source,
            "error": error,
        },
    )


async def emit_userfile_uploaded(
    room_id: UUID,
    db: AsyncSession,
    *,
    path: str,
    size: int,
    source: str,
    error: str | None = None,
) -> None:
    """Record an upload with path/size metadata only."""
    await emit_event(
        room_id=room_id,
        event_type="USERFILE_UPLOADED",
        content="Room private file uploaded",
        db=db,
        metadata={
            "path": path,
            "size": max(0, int(size)),
            "source": source,
            "error": error,
        },
    )
