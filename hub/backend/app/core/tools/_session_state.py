"""Per-session in-memory state shared across tool calls within one session.

Tracks:
- read_paths and content fingerprints for files read (for edit_file's
  read-before-edit and stale-content checks)

Keyed by session_id (str). State persists until the process restarts; sessions are
short-lived enough that this dict stays small.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

_session_reads: dict[str, dict[str, str | None]] = {}
_tool_failures: dict[str, dict[str, Any]] = {}
_duplicate_reads: dict[str, dict[str, int]] = {}

MAX_IDENTICAL_TOOL_ERRORS = 3

# How many times the exact same (unchanged) content may be re-served in one
# session before the receipt escalates from an informational note to a hard
# refusal. Kept separate from MAX_IDENTICAL_TOOL_ERRORS: a repeated duplicate
# read is not a tool error (record_tool_result ignores it), so without this
# counter an agent can loop on cheap receipts indefinitely without ever
# tripping the existing repeated-error circuit breaker.
DUPLICATE_READ_HARD_LIMIT = 3


def _norm(path: str) -> str:
    """Normalize a workspace path to a stable comparison key."""
    from app.core.path_contract import normalize_workspace_path

    try:
        return normalize_workspace_path(path)
    except ValueError:
        return str(path).strip().replace("\\", "/").lstrip("/")


def mark_read(session_id: str, path: str, fingerprint: str | None = None) -> None:
    """Record that *path* was successfully read in *session_id*."""
    bucket = _session_reads.setdefault(session_id, {})
    bucket[_norm(path)] = fingerprint


def was_read(session_id: str, path: str) -> bool:
    """Return True if *path* was read in *session_id*."""
    return _norm(path) in _session_reads.get(session_id, {})


def read_fingerprint(session_id: str, path: str) -> str | None:
    """Return the content fingerprint captured by the last successful read."""
    return _session_reads.get(session_id, {}).get(_norm(path))


def export_reads(session_id: str) -> dict[str, str | None]:
    """Return a serialisable copy of the read-before-edit state."""
    return dict(_session_reads.get(session_id, {}))


def restore_reads(
    session_id: str,
    reads: dict[str, str | None],
    *,
    replace: bool = False,
) -> None:
    """Restore validated reads into a new or restarted provider session."""
    bucket = {} if replace else dict(_session_reads.get(session_id, {}))
    for path, source_fingerprint in (reads or {}).items():
        bucket[_norm(path)] = source_fingerprint
    if bucket:
        _session_reads[session_id] = bucket


def tool_call_signature(tool_name: str, tool_input: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"tool": tool_name, "input": tool_input},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def classify_tool_error(result: str) -> str:
    text = (result or "").lower()
    if "not running" in text or "unreachable" in text or "timed out" in text:
        return "workspace_transient"
    if "permission" in text or "denied" in text:
        return "permission"
    if "not found" in text or "path" in text or "traversal" in text:
        return "path"
    if "required" in text or "invalid" in text or "expected_" in text:
        return "invalid_input"
    return "tool_error"


def repeated_tool_error(
    session_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
) -> dict[str, Any] | None:
    """Return diagnostic state when an identical failed call is already blocked."""
    state = _tool_failures.get(session_id) or {}
    signature = tool_call_signature(tool_name, tool_input)
    if state.get("signature") != signature:
        return None
    if int(state.get("count", 0)) < MAX_IDENTICAL_TOOL_ERRORS:
        return None
    return dict(state)


def record_tool_result(
    session_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    result: str,
) -> dict[str, Any] | None:
    """Track identical deterministic errors and clear the circuit on success."""
    if not str(result).lstrip().lower().startswith("error:"):
        _tool_failures.pop(session_id, None)
        return None
    signature = tool_call_signature(tool_name, tool_input)
    previous = _tool_failures.get(session_id) or {}
    count = int(previous.get("count", 0)) + 1 if previous.get("signature") == signature else 1
    state = {
        "signature": signature,
        "tool": tool_name,
        "count": count,
        "category": classify_tool_error(result),
        "last_error": str(result)[:2_000],
    }
    _tool_failures[session_id] = state
    return dict(state)


def note_duplicate_read(session_id: str, key: str) -> int:
    """Record a repeat-serve of *key* (already-seen content) and return its count."""
    bucket = _duplicate_reads.setdefault(session_id, {})
    bucket[key] = bucket.get(key, 0) + 1
    return bucket[key]


def duplicate_escalation_notice(subject: str, count: int, limit: int = DUPLICATE_READ_HARD_LIMIT) -> str:
    """Prefix for a duplicate-read receipt once *count* reaches *limit*.

    Empty string below the threshold — the plain receipt already says enough
    the first couple of times. At/above it, be blunt: soft "already read"
    notes are easy for a model to skim past and retry anyway.
    """
    if count < limit:
        return ""
    return (
        f"[BLOCKED: {subject} has been requested {count} times this session with unchanged "
        "content — it will not be served again. Stop re-reading/re-checking it. Use grep for "
        "a specific section, or move on to your next real action (write/edit/verify something "
        "else).]\n"
    )


def export_tool_failure(session_id: str) -> dict[str, Any] | None:
    state = _tool_failures.get(session_id)
    return dict(state) if state else None


def restore_tool_failure(session_id: str, state: dict[str, Any] | None) -> None:
    if state:
        _tool_failures[session_id] = dict(state)


def clear_session(session_id: str) -> None:
    """Drop the read-set for *session_id* (call on session close to free memory)."""
    _session_reads.pop(session_id, None)
    _tool_failures.pop(session_id, None)
    _duplicate_reads.pop(session_id, None)
