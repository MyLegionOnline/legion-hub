"""Facade for file I/O over the workspace volume.

Stage 6 of ``internal design notes`` collapsed the dual
``RepoFile`` + volume storage into a single source of truth: the workspace
volume (driven by ``workspaced``). The ``repofile`` table, the
``WORKSPACE_FS_BACKEND`` flag, and the legacy branches that read from
``RepoFile.content`` / ``StorageBackend`` are gone. Every public function
here now reads/writes the volume directly through ``workspaced_client`` —
the same client the path-based ``/rooms/{id}/workspace/fs/*`` API
(``app/api/v1/workspace_files.py``) uses.

Callers (chat_master, memory_layer, session_close, tools) import from here
for historical reasons; the implementations are thin wrappers over
``workspaced_client`` so that the empty-overwrite guard and the volume-snapshot
shaping stay in one place.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.path_contract import (
    normalize_workspace_path,
)

logger = logging.getLogger(__name__)


class EmptyOverwriteBlockedError(RuntimeError):
    """Raised by fs_write_file when an empty write would replace known
    non-empty content — the exact failure mode that zeroed ~26 files in the
    Kesi/Loki room incident. See internal design notes Stage 5.

    Callers that fetch content from a source which already validates
    emptiness against an expected size (e.g. git_sync.sync_branch) should
    pass fs_write_file(..., allow_empty=True) instead of relying on this
    guard, since it only has metadata/heuristic visibility into "was this
    file non-empty before".
    """


# ---------------------------------------------------------------------------
# Public facade API
# ---------------------------------------------------------------------------


async def fs_list_files(room_id: UUID, db: AsyncSession) -> list[dict[str, Any]]:
    """Return lightweight file list from the volume snapshot.

    Returns list of {path, version, updated_at, line_count, size, mtime,
    is_binary, fingerprint, generation}.
    """
    from app.core import workspaced_client as wsd

    snapshot = await wsd.get_workspace_snapshot(str(room_id), max_previews=0)
    generation = snapshot.get("generation") or {}
    revision = int(generation.get("revision", 0))
    return [
        {
            "path": normalize_ws_path(item.get("path", "")),
            "version": revision,
            "updated_at": datetime.utcfromtimestamp(
                float(item.get("mtime", 0))
            ).isoformat(),
            "line_count": 0,
            "size": int(item.get("size", 0)),
            "mtime": float(item.get("mtime", 0)),
            "is_binary": bool(item.get("binary", False)),
            "fingerprint": item.get("fingerprint"),
            "generation": generation,
        }
        for item in snapshot.get("files", [])
    ]


async def fs_list_files_short(room_id: UUID, db: AsyncSession) -> list[dict[str, Any]]:
    """Return [{path, version, fingerprint, generation}] — cheaper variant
    for system-prompt snapshots."""
    from app.core import workspaced_client as wsd

    snapshot = await wsd.get_workspace_snapshot(str(room_id), max_previews=0)
    return [
        {
            "path": normalize_ws_path(item.get("path", "")),
            "version": int((snapshot.get("generation") or {}).get("revision", 0)),
            "fingerprint": item.get("fingerprint"),
            "generation": snapshot.get("generation") or {},
        }
        for item in snapshot.get("files", [])
    ]


async def fs_read_file(room_id: UUID, path: str, db: AsyncSession) -> Optional[str]:
    """Return file content from the workspace volume.

    Stage 6.3: the legacy fallback to ``RepoFile.content`` / S3 is gone —
    a missing file is now an honest ``None`` instead of a silent lift of a
    stale DB copy. Callers that need to distinguish "missing" from "empty"
    get exactly that.
    """
    from app.core import workspaced_client as wsd

    normalized = normalize_ws_path(path)
    return await wsd.read_workspace_file(str(room_id), normalized)


async def fs_write_file(
    room_id: UUID,
    path: str,
    content: str,
    db: AsyncSession,
    agent_id: Optional[UUID] = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    """Write file to the workspace volume (auto-committed by workspaced).

    Refuses (EmptyOverwriteBlockedError) to replace known non-empty content
    with an empty string unless ``allow_empty=True`` — see Stage 5 of
    internal design notes. The existing-content probe reads the volume
    directly (there is no longer a metadata cache to consult).
    """
    from app.core import workspaced_client as wsd

    normalized = normalize_ws_path(path)
    is_binary = content.startswith("data:")

    if content == "" and not is_binary and not allow_empty:
        existing_size = await _existing_content_size(room_id, normalized)
        if existing_size:
            logger.warning(
                "fs_write_file: blocked empty overwrite of %s/%s (had %d bytes of content)",
                room_id, normalized, existing_size,
            )
            try:
                from app.core.metrics import record_repo_file_empty_write_blocked
                record_repo_file_empty_write_blocked()
            except Exception:
                pass
            try:
                from app.core.events import emit_event
                await emit_event(
                    room_id=room_id,
                    event_type="FILE_WRITE_BLOCKED",
                    content=(
                        f"Writing empty content to {normalized} was blocked — "
                        f"the file contained {existing_size} bytes"
                    ),
                    db=db,
                    metadata={
                        "path": normalized,
                        "reason": "empty_overwrite",
                        "previous_size": existing_size,
                    },
                )
            except Exception:
                logger.warning(
                    "fs_write_file: failed to emit empty-write-blocked event for %s/%s",
                    room_id, normalized, exc_info=True,
                )
            raise EmptyOverwriteBlockedError(
                f"refusing to overwrite non-empty file {normalized} with empty content"
            )

    source = f"agent:{agent_id}" if agent_id else "user"
    commit_msg = f"{source}: write {normalized}"
    try:
        await wsd.write_workspace_file(
            str(room_id), normalized, content, commit_message=commit_msg
        )
    except Exception as exc:
        logger.warning("workspace write failed for %s/%s: %s", room_id, path, exc)
        raise

    return {"path": normalized, "content": content, "is_binary": is_binary}


async def fs_delete_file(
    room_id: UUID,
    path: str,
    db: AsyncSession,
) -> bool:
    """Delete file from the workspace volume. Returns True if deleted."""
    from app.core import workspaced_client as wsd

    normalized = normalize_ws_path(path)
    try:
        await wsd.delete_workspace_file(
            str(room_id), normalized, commit_message=f"user: delete {normalized}"
        )
    except Exception as exc:
        logger.warning("workspace delete failed for %s/%s: %s", room_id, path, exc)
        return False
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BINARY_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".svgz",
    ".pdf", ".zip", ".tar", ".gz", ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm",
})


def normalize_ws_path(path: str) -> str:
    """Return the compatible ``/x/y`` form used by tree APIs.

    The canonical agent form is relative; this wrapper is retained for the
    existing database/tree contract and is the only place that adds a slash.
    """
    relative = normalize_workspace_path(path, allow_empty=True)
    return f"/{relative}" if relative else "/"


def _guess_binary(path: str) -> bool:
    dot = path.rfind(".")
    return dot != -1 and path[dot:].lower() in _BINARY_EXTS


async def _existing_content_size(room_id: UUID, normalized: str) -> int:
    """Best-effort size of the live content an empty write is about to replace.

    Only called when the incoming write is already known to be empty, so the
    extra volume read never runs on the hot path of a normal (non-empty) write.
    """
    from app.core import workspaced_client as wsd

    try:
        current = await wsd.read_workspace_file(str(room_id), normalized)
    except Exception:
        return 0
    return len(current) if current else 0
