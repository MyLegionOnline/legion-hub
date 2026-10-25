"""External observation storage and persistent range-dedupe helpers.

This module is intentionally provider-neutral.  Tool implementations keep
their bounded response in the active conversation, while the complete payload
is addressed by an observation reference and can be recalled by page.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.token_counter import TokenCounter
from app.models.context_artifact import ContextArtifact
from app.models.session import Session


_TOKENS = TokenCounter()


async def _db_add(db: AsyncSession, value: Any) -> None:
    result = db.add(value)
    if inspect.isawaitable(result):
        await result


async def _db_flush(db: AsyncSession) -> None:
    flush = getattr(db, "flush", None)
    if flush is None:
        return
    result = flush()
    if inspect.isawaitable(result):
        await result


async def _db_delete(db: AsyncSession, value: Any) -> None:
    result = db.delete(value)
    if inspect.isawaitable(result):
        await result


def normalize_path(path: str) -> str:
    from app.core.path_contract import normalize_workspace_path

    return normalize_workspace_path(path, allow_empty=True)


def canonicalize_args(value: Any) -> Any:
    """Canonicalize tool args so equivalent requests share the same key."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            key_text = str(key)
            child = value[key]
            result[key_text] = (
                normalize_path(child)
                if key_text in {"path", "cwd", "log_path"} and isinstance(child, str)
                else canonicalize_args(child)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [canonicalize_args(item) for item in value]
    return value


def args_hash(value: Any) -> str:
    encoded = json.dumps(
        canonicalize_args(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint(payload: str | bytes) -> str:
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(raw).hexdigest()


def merge_intervals(intervals: Iterable[Iterable[int]]) -> list[list[int]]:
    """Merge inclusive integer intervals, including adjacent ranges."""
    prepared = sorted(
        (int(pair[0]), int(pair[1]))
        for pair in intervals
        if pair and len(pair) >= 2 and int(pair[0]) <= int(pair[1])
    )
    merged: list[list[int]] = []
    for start, end in prepared:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def subtract_interval(
    start: int, end: int, covered: Iterable[Iterable[int]]
) -> list[list[int]]:
    """Return the portions of inclusive ``start..end`` not in ``covered``."""
    if start > end:
        return []
    cursor = start
    result: list[list[int]] = []
    for left, right in merge_intervals(covered):
        if right < cursor:
            continue
        if left > end:
            break
        if left > cursor:
            result.append([cursor, min(end, left - 1)])
        cursor = max(cursor, right + 1)
        if cursor > end:
            break
    if cursor <= end:
        result.append([cursor, end])
    return result


def intersect_interval(
    start: int, end: int, covered: Iterable[Iterable[int]]
) -> list[list[int]]:
    """Return the portions of inclusive ``start..end`` that *are* in ``covered``."""
    if start > end:
        return []
    result: list[list[int]] = []
    for left, right in merge_intervals(covered):
        low, high = max(start, left), min(end, right)
        if low <= high:
            result.append([low, high])
    return merge_intervals(result)


def subtract_interval_list(
    intervals: Iterable[Iterable[int]], removed: Iterable[Iterable[int]]
) -> list[list[int]]:
    """Return ``intervals`` with every part of ``removed`` taken out."""
    removed = merge_intervals(removed)
    result: list[list[int]] = []
    for start, end in merge_intervals(intervals):
        result.extend(subtract_interval(start, end, removed))
    return merge_intervals(result)


def _state_copy(value: Any) -> Any:
    return json.loads(json.dumps({} if value is None else value, default=str))


async def load_context_state(db: AsyncSession, session_id: UUID) -> dict[str, Any]:
    row = await db.get(Session, session_id)
    if not isinstance(row, Session):
        return {}
    return _state_copy(getattr(row, "context_state", None))


async def save_context_state(
    db: AsyncSession, session_id: UUID, state: dict[str, Any]
) -> None:
    row = await db.get(Session, session_id)
    if row is None:
        return
    row.context_state = state
    await _db_add(db, row)
    await _db_flush(db)


async def build_task_read_manifest(
    db: AsyncSession, *, task_id: UUID, side: str = "agent"
) -> list[dict[str, Any]]:
    """Merge the durable read ledger across every session of a task.

    ``context_state.read_ledger`` is written by ``record_read_interval`` on
    every ``read_file`` call and lives on the DB row independently of the
    chat-message history. History-based manifests (derived by re-scanning raw
    provider messages) get silently dropped once a resumed session is large
    enough to be compacted a second time, because the compact note that
    carried the earlier file list is plain text, not structured tool blocks.
    This function sidesteps that by reading the ledger straight from every
    past session of the task, so the list of already-read paths survives any
    number of tool_limit/token_limit resume cycles.
    """
    from app.models.session import CmAgentProxySession

    rows = (
        await db.execute(
            sa.select(CmAgentProxySession)
            .where(
                CmAgentProxySession.task_id == task_id,
                CmAgentProxySession.side == side,
            )
            .order_by(CmAgentProxySession.started_at.asc())
        )
    ).scalars().all()

    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        state = getattr(row, "context_state", None) or {}
        ledger = state.get("read_ledger") if isinstance(state, dict) else None
        if not isinstance(ledger, dict):
            continue
        for path, entry in ledger.items():
            if not isinstance(entry, dict):
                continue
            target = merged.setdefault(
                path,
                {"path": path, "fingerprint": entry.get("fingerprint"), "ranges": [], "observation_refs": []},
            )
            if entry.get("fingerprint") != target["fingerprint"]:
                # A later session re-read the file after it changed on disk —
                # the old ranges no longer describe the current content.
                target["fingerprint"] = entry.get("fingerprint")
                target["ranges"] = []
            target["ranges"] = merge_intervals([*target["ranges"], *(entry.get("ranges") or [])])
            refs = list(dict.fromkeys([*target["observation_refs"], *(entry.get("observation_refs") or [])]))
            target["observation_refs"] = refs[-50:]

    return [merged[path] for path in sorted(merged)]


def format_read_manifest_note(manifest: list[dict[str, Any]]) -> str:
    """Render a durable read manifest as a resume-note block for the agent."""
    if not manifest:
        return ""
    lines = [
        "[Files already read in this task across all prior sessions — durable, "
        "survives resume/compaction. Their text is NOT in your context: recall "
        "an exact section via the recall hint below, or read the file again "
        "when you actually need its contents. Do not re-read a file you can "
        "already see, and do not re-read the same range twice.]"
    ]
    for i, entry in enumerate(manifest, 1):
        ranges = ", ".join(f"{a}-{b}" for a, b in entry.get("ranges") or [])
        refs = entry.get("observation_refs") or []
        ref_hint = f" — recall exact text via observation_read(ref={refs[-1]!r})" if refs else ""
        lines.append(f"{i}. {entry['path']} (lines {ranges or 'whole file'}){ref_hint}")
    return "\n".join(lines)


async def build_task_todo(
    db: AsyncSession, *, task_id: UUID, side: str = "agent"
) -> list[dict[str, Any]]:
    """Return the latest durable todo list recorded for a task."""
    from app.models.session import CmAgentProxySession

    rows = (
        await db.execute(
            sa.select(CmAgentProxySession)
            .where(
                CmAgentProxySession.task_id == task_id,
                CmAgentProxySession.side == side,
            )
            .order_by(CmAgentProxySession.started_at.asc())
        )
    ).scalars().all()

    for row in reversed(rows):
        state = getattr(row, "context_state", None)
        if not isinstance(state, dict) or "todo" not in state:
            continue
        todo = state.get("todo")
        if isinstance(todo, list):
            return _state_copy(todo)
    return []


def format_todo_note(todo: list[dict[str, Any]] | None) -> str:
    """Render the durable todo list for resume and compaction notes."""
    if not isinstance(todo, list) or not todo:
        return ""

    lines = [
        "[Current todo list for this task — durable across resume/compaction. "
        "Continue from these statuses; do not recreate completed items.]"
    ]
    for index, item in enumerate(todo, 1):
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        title = str(item.get("title") or item_id or f"item {index}")
        status = str(item.get("status") or "pending")
        suffix = f" (id: {item_id})" if item_id else ""
        lines.append(f"{index}. [{status}] {title}{suffix}")
    return "\n".join(lines) if len(lines) > 1 else ""


async def record_read_interval(
    db: AsyncSession,
    *,
    session_id: UUID,
    path: str,
    source_fingerprint: str,
    requested_start: int,
    requested_end: int,
    observation_ref: str,
) -> list[list[int]]:
    """Return unseen portions and persist the merged read ledger."""
    norm = normalize_path(path)
    state = await load_context_state(db, session_id)
    ledger = state.setdefault("read_ledger", {})
    entry = ledger.get(norm) or {}
    if entry.get("fingerprint") != source_fingerprint:
        entry = {
            "fingerprint": source_fingerprint,
            "ranges": [],
            "evicted": [],
            "observation_refs": [],
        }
    covered = entry.get("ranges") or []
    evicted = entry.get("evicted") or []
    # A line counts as unseen when it was never read *or* when the tool result
    # that carried it has since been compacted out of the active context.
    # Without the second half the agent is refused content it cannot see.
    unseen = merge_intervals([
        *subtract_interval(requested_start, requested_end, covered),
        *intersect_interval(requested_start, requested_end, evicted),
    ])
    entry["ranges"] = merge_intervals([*covered, [requested_start, requested_end]])
    # Whatever is about to be served is live again.
    entry["evicted"] = subtract_interval_list(
        evicted, [[requested_start, requested_end]]
    )
    refs = list(entry.get("observation_refs") or [])
    if observation_ref not in refs:
        refs.append(observation_ref)
    entry["observation_refs"] = refs[-50:]
    ledger[norm] = entry
    state["read_ledger"] = ledger
    await save_context_state(db, session_id, state)
    return unseen

async def mark_reads_evicted(
    db: AsyncSession,
    *,
    session_id: UUID,
    reads: Iterable[dict[str, Any]],
) -> int:
    """Record that compaction dropped the text of these reads from the context.

    ``read_ledger`` answers "what can the agent currently see", and the tools
    refuse to resend anything it covers.  Compaction happens outside the tools,
    so without this the ledger keeps claiming coverage for text that is no
    longer in the prompt, and every re-read comes back as a bare receipt for
    content the agent has no way to look at.

    ``start``/``end`` of ``None`` mean the whole path.
    """
    entries = [item for item in reads if isinstance(item, dict) and item.get("path")]
    if not entries:
        return 0
    state = await load_context_state(db, session_id)
    ledger = state.get("read_ledger")
    if not isinstance(ledger, dict) or not ledger:
        return 0

    changed = 0
    for item in entries:
        norm = normalize_path(str(item["path"]))
        entry = ledger.get(norm)
        if not isinstance(entry, dict):
            continue
        ranges = entry.get("ranges") or []
        if not ranges:
            continue
        start, end = item.get("start"), item.get("end")
        if start is None or end is None:
            gone = merge_intervals(ranges)
        else:
            gone = intersect_interval(int(start), int(end), ranges)
        if not gone:
            continue
        merged = merge_intervals([*(entry.get("evicted") or []), *gone])
        if merged == merge_intervals(entry.get("evicted") or []):
            continue
        entry["evicted"] = merged
        ledger[norm] = entry
        changed += 1

    if changed:
        state["read_ledger"] = ledger
        await save_context_state(db, session_id, state)
    return changed


# Bounded: only recent refs can still be the subject of a repeat call, and the
# list rides along in the session row.
_EVICTED_OBSERVATIONS_MAX = 500


async def mark_observations_evicted(
    db: AsyncSession, *, session_id: UUID, refs: Iterable[str]
) -> int:
    """Record observation payloads that compaction removed from the context.

    The range ledger covers file reads.  Searches and outlines have no line
    ranges, so their repeat-call suppression keys off the stored artifact
    instead, and needs the same signal: once the text is compacted away, the
    next identical call has to return it rather than a receipt.
    """
    wanted = [str(ref) for ref in refs if ref]
    if not wanted:
        return 0
    state = await load_context_state(db, session_id)
    known = state.get("evicted_observations")
    known = [str(ref) for ref in known] if isinstance(known, list) else []
    merged = list(dict.fromkeys([*known, *wanted]))[-_EVICTED_OBSERVATIONS_MAX:]
    if merged == known:
        return 0
    state["evicted_observations"] = merged
    await save_context_state(db, session_id, state)
    return len(merged) - len(known)


async def take_evicted_observation(
    db: AsyncSession, *, session_id: UUID, ref: str
) -> bool:
    """Consume *ref*'s eviction mark, reporting whether one was set.

    The mark is one-shot on purpose: the caller only asks when it is about to
    send the payload again, which puts it back in the context and makes the
    ordinary repeat-call suppression correct once more.
    """
    try:
        state = await load_context_state(db, session_id)
    except Exception:
        # Fail open: resending content is always safe, refusing it is not.
        return True
    known = state.get("evicted_observations")
    if not isinstance(known, list):
        return False
    remaining = [item for item in known if str(item) != str(ref)]
    if len(remaining) == len(known):
        return False
    state["evicted_observations"] = remaining
    try:
        await save_context_state(db, session_id, state)
    except Exception:
        pass
    return True


def _payload_store():
    # Reuse S3 when it is already configured for repository storage; otherwise
    # use a durable local artifact directory.  DatabaseBackend intentionally
    # remains a no-op for legacy RepoFile callers.
    if os.getenv("REPO_STORAGE_BACKEND", "db").lower() == "s3":
        from app.core.storage import S3Backend
        return S3Backend()
    from app.core.storage import FilesystemBackend
    return FilesystemBackend()


@dataclass
class ObservationResult:
    artifact: ContextArtifact
    duplicate: bool = False
    #: Whether the deduplicated artifact was produced by the *calling* session.
    same_session: bool = True

    @property
    def ref(self) -> str:
        return str(self.artifact.id)

    @property
    def seen_in_session(self) -> bool:
        """Whether the caller already has this payload in its own context.

        Artifacts are stored per room, so an identical payload recorded by an
        earlier session — a previous attempt of the same task, for instance —
        deduplicates *storage* but says nothing about what the current session
        can see.  Withholding content on that basis leaves a retried session
        unable to read a single file it has not read itself.
        """
        return self.duplicate and self.same_session

    def receipt(self) -> str:
        status = "stale" if self.artifact.stale else "available"
        return (
            f"[observation {status}: {self.ref}; tool={self.artifact.tool_name}; "
            f"tokens={self.artifact.token_count}; use observation_read(ref={self.ref!r}, "
            "offset=..., limit=...) for exact text]"
        )


async def record_observation(
    db: AsyncSession,
    *,
    room_id: UUID,
    session_id: UUID,
    task_id: Optional[UUID],
    tool_name: str,
    canonical_args: Any,
    payload: str,
    source_fingerprint: Optional[str] = None,
    workspace_revision: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    pinned: bool = False,
    expires_at: Optional[datetime] = None,
) -> ObservationResult:
    """Persist metadata and payload, returning a duplicate receipt when exact."""
    fp = source_fingerprint or fingerprint(payload)
    key_hash = args_hash(canonical_args)
    filters = [
        ContextArtifact.room_id == room_id,
        ContextArtifact.tool_name == tool_name,
        ContextArtifact.args_hash == key_hash,
        ContextArtifact.source_fingerprint == fp,
        ContextArtifact.stale.is_(False),
    ]
    filters.append(
        ContextArtifact.task_id == task_id
        if task_id is not None
        else ContextArtifact.task_id.is_(None)
    )
    existing = (
        await db.execute(
            sa.select(ContextArtifact)
            .where(*filters)
            .order_by(ContextArtifact.created_at.desc())
            .limit(1)
        )
    ).scalars().first()
    if not isinstance(existing, ContextArtifact):
        # AsyncMock-backed unit tests and degraded DB adapters should not turn
        # an arbitrary mock value into a false duplicate.
        existing = None
    if existing is not None:
        try:
            from app.core.metrics import record_context_artifact_cache_hit

            record_context_artifact_cache_hit(
                tool_name,
                "task" if task_id is not None else "room",
            )
        except Exception:
            pass
        return ObservationResult(
            existing,
            duplicate=True,
            same_session=str(getattr(existing, "session_id", "")) == str(session_id),
        )

    artifact_id = uuid4()
    storage_key = (
        f"context-observations/{room_id}/{artifact_id}.txt"
    )
    await _payload_store().put(storage_key, payload)
    artifact = ContextArtifact(
        id=artifact_id,
        room_id=room_id,
        task_id=task_id,
        session_id=session_id,
        tool_name=tool_name,
        args_hash=key_hash,
        source_fingerprint=fp,
        workspace_revision=workspace_revision,
        storage_key=storage_key,
        char_count=len(payload),
        token_count=_TOKENS.count(payload),
        artifact_metadata=metadata or {},
        pinned=pinned,
        expires_at=expires_at,
    )
    await _db_add(db, artifact)
    await _db_flush(db)
    return ObservationResult(artifact)


async def get_observation(
    db: AsyncSession, ref: str, *, room_id: Optional[UUID] = None
) -> Optional[ContextArtifact]:
    try:
        artifact_id = UUID(str(ref))
    except (ValueError, TypeError, AttributeError):
        return None
    filters = [ContextArtifact.id == artifact_id]
    if room_id is not None:
        filters.append(ContextArtifact.room_id == room_id)
    result = (await db.execute(sa.select(ContextArtifact).where(*filters))).scalars().first()
    return result if isinstance(result, ContextArtifact) else None


async def read_observation_payload(
    db: AsyncSession,
    ref: str,
    *,
    room_id: Optional[UUID] = None,
    offset: int = 1,
    limit: int = 100,
) -> tuple[Optional[ContextArtifact], Optional[str], int, int]:
    artifact = await get_observation(db, ref, room_id=room_id)
    if artifact is None:
        return None, None, 0, 0
    payload = await _payload_store().get(artifact.storage_key)
    if payload is None:
        return artifact, None, 0, 0
    lines = payload.splitlines(keepends=True)
    offset = max(1, int(offset))
    limit = max(1, min(int(limit), 200))
    selected = lines[offset - 1 : offset - 1 + limit]
    return artifact, "".join(selected), offset, len(lines)


async def invalidate_path(
    db: AsyncSession, room_id: UUID, path: str, *, new_fingerprint: Optional[str] = None
) -> int:
    """Mark file observations stale and reset the corresponding read ledger."""
    norm = normalize_path(path)
    result = await db.execute(
        sa.select(ContextArtifact).where(
            ContextArtifact.room_id == room_id,
            ContextArtifact.artifact_metadata["path"].as_string() == norm,
            ContextArtifact.stale.is_(False),
        )
    )
    rows = [row for row in result.scalars().all() if isinstance(row, ContextArtifact)]
    for row in rows:
        if new_fingerprint and row.source_fingerprint == new_fingerprint:
            continue
        row.stale = True
        await _db_add(db, row)
    # Reset ledger entries for this path in all active sessions in the room.
    sessions = (
        await db.execute(sa.select(Session).where(Session.room_id == room_id))
    ).scalars().all()
    sessions = [row for row in sessions if isinstance(row, Session)]
    for session in sessions:
        state = _state_copy(getattr(session, "context_state", None))
        ledger = state.get("read_ledger") or {}
        if norm in ledger:
            ledger.pop(norm, None)
            state["read_ledger"] = ledger
            session.context_state = state
            await _db_add(db, session)
    await _db_flush(db)
    return len(rows)


async def garbage_collect(
    db: AsyncSession,
    *,
    room_id: Optional[UUID] = None,
    now: Optional[datetime] = None,
    retention_days: Optional[int] = None,
) -> int:
    """Delete expired/retained, unpinned artifact metadata and payloads.

    ``retention_days`` is optional so deployments can apply a room-specific
    retention policy without changing the default expiry-only behavior.
    """
    now = now or datetime.now(timezone.utc)
    filters = [
        ContextArtifact.pinned.is_(False),
    ]
    expiry_filters = [
        sa.and_(
            ContextArtifact.expires_at.is_not(None),
            ContextArtifact.expires_at < now,
        )
    ]
    if retention_days is not None:
        cutoff = now - timedelta(days=max(0, int(retention_days)))
        expiry_filters.append(ContextArtifact.created_at < cutoff)
    filters.append(sa.or_(*expiry_filters))
    if room_id is not None:
        filters.append(ContextArtifact.room_id == room_id)
    rows = (await db.execute(sa.select(ContextArtifact).where(*filters))).scalars().all()
    rows = [row for row in rows if isinstance(row, ContextArtifact)]
    store = _payload_store()
    for row in rows:
        await store.delete(row.storage_key)
        await _db_delete(db, row)
    await _db_flush(db)
    return len(rows)
