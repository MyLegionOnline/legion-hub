"""Token/context measurements and offline replay primitives.

Stage 0 deliberately keeps this module independent from provider SDKs.  Live
requests use provider-reported input tokens for the authoritative total, while
individual prompt blocks are estimated locally.  Replay corpora contain only
counts, tool names, ranges and hashes; message/tool-result text is never stored.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import posixpath
from dataclasses import dataclass, field
from typing import Any, Iterable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_TRACE_KEY = "context_baseline"
_TRACE_LIMIT = max(10, int(os.getenv("CONTEXT_BASELINE_TRACE_LIMIT", "200")))
_READ_TOOLS = frozenset({"read_file", "observation_read"})


def _compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def estimate_tokens(value: Any) -> int:
    """Conservative, tokenizer-free token estimate for an immutable block.

    UTF-8 bytes are a better fallback than Python character count for Cyrillic
    and other non-ASCII text.  The 20% margin follows the context plan and the
    result is intentionally labelled as an estimate everywhere it is exposed.
    """
    text = value if isinstance(value, str) else _compact_json(value)
    if not text:
        return 0
    return max(1, math.ceil((len(text.encode("utf-8")) / 4) * 1.2))


def _block_size(value: Any) -> dict[str, int]:
    text = value if isinstance(value, str) else _compact_json(value)
    return {"chars": len(text), "estimated_tokens": estimate_tokens(text)}


def _split_messages(messages: Iterable[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """Split request messages into conversation and raw tool-result blocks."""
    conversation: list[Any] = []
    tool_results: list[Any] = []

    for message in messages:
        if not isinstance(message, dict):
            conversation.append(message)
            continue

        if message.get("role") == "tool":
            tool_results.append(message)
            continue

        content = message.get("content")
        if not isinstance(content, list):
            conversation.append(message)
            continue

        normal_blocks: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_results.append(block)
            else:
                normal_blocks.append(block)
        if normal_blocks or len(message) > 2:
            compact = dict(message)
            compact["content"] = normal_blocks
            conversation.append(compact)

    return conversation, tool_results


def prompt_components(
    system_prompt: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return non-overlapping sizes for the four Stage-0 prompt blocks."""
    conversation, tool_results = _split_messages(messages)
    return {
        "system": _block_size(system_prompt),
        "schemas": _block_size(tools),
        "conversation": _block_size(conversation),
        "tool_results": _block_size(tool_results),
    }


def _canonical_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    normalized = posixpath.normpath("/" + raw.lstrip("/"))
    return normalized.casefold()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _merge_interval(intervals: list[tuple[int, int]], start: int, end: int) -> None:
    merged: list[tuple[int, int]] = []
    for left, right in sorted([*intervals, (start, end)]):
        if not merged or left > merged[-1][1] + 1:
            merged.append((left, right))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
    intervals[:] = merged


@dataclass
class ToolCallStats:
    """Incremental duplicate/overlap tracker shared by live and replay paths."""

    tool_calls: int = 0
    repeated_reads: int = 0
    read_requested_lines: int = 0
    read_overlap_lines: int = 0
    _seen_reads: set[tuple[str, int, int]] = field(default_factory=set)
    _read_intervals: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def observe(self, calls: Iterable[dict[str, Any]]) -> dict[str, int | float]:
        before = (
            self.tool_calls,
            self.repeated_reads,
            self.read_requested_lines,
            self.read_overlap_lines,
        )
        for call in calls:
            name = str(call.get("name") or "")
            args = call.get("input") if isinstance(call.get("input"), dict) else {}
            self.tool_calls += 1
            if name not in _READ_TOOLS:
                continue
            path = _canonical_path(args.get("path") or args.get("ref"))
            if not path:
                continue
            try:
                start = max(1, int(args.get("offset") or 1))
                limit = max(1, int(args.get("limit") or 100))
            except (TypeError, ValueError):
                start, limit = 1, 100
            end = start + limit - 1
            key = (path, start, end)
            if key in self._seen_reads:
                self.repeated_reads += 1
            self._seen_reads.add(key)

            intervals = self._read_intervals.setdefault(path, [])
            overlap = sum(max(0, min(end, right) - max(start, left) + 1) for left, right in intervals)
            self.read_requested_lines += limit
            self.read_overlap_lines += min(limit, overlap)
            _merge_interval(intervals, start, end)

        delta_calls = self.tool_calls - before[0]
        delta_repeats = self.repeated_reads - before[1]
        delta_requested = self.read_requested_lines - before[2]
        delta_overlap = self.read_overlap_lines - before[3]
        return {
            "tool_calls": delta_calls,
            "repeated_reads": delta_repeats,
            "read_requested_lines": delta_requested,
            "read_overlap_lines": delta_overlap,
            "read_overlap_ratio": round(delta_overlap / delta_requested, 4) if delta_requested else 0.0,
        }


def build_context_measurement(
    *,
    provider: str,
    model: str,
    role: str,
    step: int,
    system_prompt: str,
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    actual_input_tokens: int,
    actual_output_tokens: int,
    response_calls: list[dict[str, Any]],
    tracker: ToolCallStats,
    request_kind: str = "tool_loop",
) -> dict[str, Any]:
    components = prompt_components(system_prompt, tools, messages)
    estimated_input = sum(block["estimated_tokens"] for block in components.values())
    return {
        "version": 1,
        "kind": request_kind,
        "step": step,
        "provider": provider,
        "model": model,
        "role": role,
        "blocks": components,
        "estimated_input_tokens": estimated_input,
        "actual_input_tokens": max(0, int(actual_input_tokens or 0)),
        "actual_output_tokens": max(0, int(actual_output_tokens or 0)),
        **tracker.observe(response_calls),
    }


async def record_context_measurement(
    measurement: dict[str, Any],
    *,
    session_id: UUID,
    db: AsyncSession,
) -> None:
    """Write a compact trace to logs/metrics and the session snapshot.

    Observability is best-effort by design: a metrics or persistence failure may
    never interrupt a paid model request or corrupt the agent loop.
    """
    logger.info("context_baseline %s", _compact_json(measurement))
    try:
        from app.core.metrics import record_context_request

        record_context_request(measurement)
    except Exception as exc:  # pragma: no cover - defensive isolation
        logger.warning("context_baseline: metrics failed for %s: %s", session_id, exc)

    try:
        from app.models.session.base import Session

        row = await db.get(Session, session_id)
        if row is None:
            return
        current = getattr(row, "context_snapshot", None)
        if current is not None and not isinstance(current, dict):
            return
        snapshot = dict(current or {})
        trace = list(snapshot.get(_TRACE_KEY) or [])
        trace.append(measurement)
        snapshot[_TRACE_KEY] = trace[-_TRACE_LIMIT:]
        row.context_snapshot = snapshot
        db.add(row)
    except Exception as exc:  # pragma: no cover - defensive isolation
        logger.warning("context_baseline: snapshot trace failed for %s: %s", session_id, exc)


def tool_calls_from_anthropic(content: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"name": str(block.get("name") or ""), "input": block.get("input") or {}}
        for block in content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]


def tool_calls_from_openai(tool_calls: Iterable[Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for call in tool_calls:
        function = getattr(call, "function", None)
        name = getattr(function, "name", "")
        raw_args = getattr(function, "arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        except (TypeError, ValueError, json.JSONDecodeError):
            args = {}
        calls.append({"name": str(name or ""), "input": args})
    return calls


def anonymized_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    """Return only replay-safe call metadata (no paths, patterns or commands)."""
    args = call.get("input") if isinstance(call.get("input"), dict) else {}
    result: dict[str, Any] = {"name": str(call.get("name") or "")}
    target = args.get("path") or args.get("ref")
    if target:
        result["target_hash"] = _digest(_canonical_path(target))
    for key in ("offset", "limit"):
        if key in args:
            try:
                result[key] = int(args[key])
            except (TypeError, ValueError):
                pass
    return result
