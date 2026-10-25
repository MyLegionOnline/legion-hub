"""Build anonymized context trajectories and compare current/proposed packing.

The proposed policy here is a simulator, not the Stage-1 ContextAssembler: it
keeps the newest K raw tool results and charges a small receipt for every older
result.  It lets us measure the likely gain before changing production prompts.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Iterable

from app.core.context_baseline import ToolCallStats, estimate_tokens

_TRACE_KEY = "context_baseline"
_MUTATING_TOOLS = frozenset({"edit_file", "write_file", "bash"})
_RESEARCH_TOOLS = frozenset({"grep", "glob", "find_file", "web_search", "web_fetch", "read_file"})
_GENERATED_SUFFIXES = (".md", ".txt", ".html", ".docx", ".pdf", ".pptx", ".csv", ".xlsx")


def _hash_id(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:20]


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _assistant_calls(message: dict[str, Any]) -> list[tuple[str, dict[str, Any], str]]:
    calls: list[tuple[str, dict[str, Any], str]] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append((
                    str(block.get("name") or ""),
                    block.get("input") if isinstance(block.get("input"), dict) else {},
                    str(block.get("id") or ""),
                ))
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        calls.append((
            str(function.get("name") or ""),
            _parse_args(function.get("arguments")),
            str(call.get("id") or ""),
        ))
    return calls


def _message_parts(
    message: dict[str, Any],
    calls_by_id: dict[str, tuple[str, dict[str, Any]]],
) -> tuple[int, list[dict[str, Any]]]:
    """Return conversation tokens and replay-safe tool-result descriptors."""
    results: list[dict[str, Any]] = []
    if message.get("role") == "tool":
        call_id = str(message.get("tool_call_id") or "")
        name, args = calls_by_id.get(call_id, ("unknown", {}))
        content = message.get("content") or ""
        results.append({
            "tool": name,
            "tokens": estimate_tokens(message),
            "call": _safe_call(name, args),
            "verification": _verification(name, content),
        })
        return 0, results

    content = message.get("content")
    if not isinstance(content, list):
        return estimate_tokens(message), results

    normal: list[Any] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            normal.append(block)
            continue
        call_id = str(block.get("tool_use_id") or "")
        name, args = calls_by_id.get(call_id, ("unknown", {}))
        payload = block.get("content") or ""
        results.append({
            "tool": name,
            "tokens": estimate_tokens(block),
            "call": _safe_call(name, args),
            "verification": _verification(name, payload),
        })
    conversation_tokens = 0
    if normal or len(message) > 2:
        compact = dict(message)
        compact["content"] = normal
        conversation_tokens = estimate_tokens(compact)
    return conversation_tokens, results


def _safe_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {"name": name}
    target = args.get("path") or args.get("ref")
    if target:
        normalized = str(target).replace("\\", "/").casefold()
        safe["target_hash"] = _hash_id(normalized)
    for key, default in (("offset", 1), ("limit", 100)):
        if name in {"read_file", "observation_read"}:
            try:
                safe[key] = max(1, int(args.get(key) or default))
            except (TypeError, ValueError):
                safe[key] = default
    return safe


def _verification(tool: str, content: Any) -> str | None:
    if tool != "bash":
        return None
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    match = re.search(r"\bexit_code\s*:\s*(-?\d+)", text)
    if not match:
        return "unknown"
    return "passed" if int(match.group(1)) == 0 else "failed"


def _request_points(messages: list[dict[str, Any]]) -> list[int]:
    points: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") not in {"user", "tool"}:
            continue
        next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
        if next_role == "assistant" or next_role is None:
            points.append(index)
    return points or ([len(messages) - 1] if messages else [])


def _classify(
    calls: list[tuple[str, dict[str, Any], str]],
    *,
    title: str,
    close_reason: str | None,
    max_tool_result_tokens: int,
) -> str:
    names = {name for name, _, _ in calls}
    title_l = title.casefold()
    if close_reason in {"token_limit", "context_limit", "tool_limit", "changes_requested"}:
        return "resume"
    if max_tool_result_tokens >= 2500 or (
        "bash" in names and any(word in title_l for word in ("test", "log"))
    ):
        return "long_test_log"
    generated_hint = any(
        word in title_l
        for word in ("report", "article", "document", "content", "presentation")
    )
    generated_path = any(
        str(args.get("path") or "").casefold().endswith(_GENERATED_SUFFIXES)
        for name, args, _ in calls
        if name == "write_file"
    )
    if generated_hint or (generated_path and "edit_file" not in names):
        return "generated_content"
    if names & _MUTATING_TOOLS:
        return "code_edit"
    if names & _RESEARCH_TOOLS:
        return "research_docs"
    return "research_docs"


def build_anonymized_trajectory(
    *,
    session_id: Any,
    snapshot: dict[str, Any] | None,
    close_reason: str | None = None,
    task_state: str | None = None,
    task_title: str = "",
) -> dict[str, Any]:
    """Convert a session snapshot to a text-free, replayable trajectory."""
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    messages = [m for m in snapshot.get("messages") or [] if isinstance(m, dict)]
    traces = [t for t in snapshot.get(_TRACE_KEY) or [] if isinstance(t, dict)]

    calls_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    all_calls: list[tuple[str, dict[str, Any], str]] = []
    message_conversation: list[int] = []
    message_results: list[list[dict[str, Any]]] = []
    verification_results: list[str] = []

    for message in messages:
        if message.get("role") == "assistant":
            for name, args, call_id in _assistant_calls(message):
                all_calls.append((name, args, call_id))
                if call_id:
                    calls_by_id[call_id] = (name, args)
        conversation_tokens, results = _message_parts(message, calls_by_id)
        message_conversation.append(conversation_tokens)
        message_results.append(results)
        verification_results.extend(
            result["verification"] for result in results if result.get("verification")
        )

    requests: list[dict[str, Any]] = []
    for sequence, point in enumerate(_request_points(messages)):
        results = [
            result
            for group in message_results[: point + 1]
            for result in group
        ]
        trace = traces[sequence] if sequence < len(traces) else {}
        blocks = trace.get("blocks") or {}
        system_tokens = int((blocks.get("system") or {}).get("estimated_tokens") or 0)
        schema_tokens = int((blocks.get("schemas") or {}).get("estimated_tokens") or 0)
        conversation_tokens = sum(message_conversation[: point + 1])
        # Prefer the live request decomposition where available; it includes
        # provider-format details that cannot be reconstructed from compacted history.
        live_conversation = int((blocks.get("conversation") or {}).get("estimated_tokens") or 0)
        if live_conversation:
            conversation_tokens = live_conversation
        requests.append({
            "sequence": sequence,
            "system_tokens": system_tokens,
            "schema_tokens": schema_tokens,
            "conversation_tokens": conversation_tokens,
            "tool_results": [
                {"tool": result["tool"], "tokens": result["tokens"]}
                for result in results
            ],
            "actual_input_tokens": int(trace.get("actual_input_tokens") or 0),
        })

    if traces and not requests:
        for sequence, trace in enumerate(traces):
            blocks = trace.get("blocks") or {}
            requests.append({
                "sequence": sequence,
                "system_tokens": int((blocks.get("system") or {}).get("estimated_tokens") or 0),
                "schema_tokens": int((blocks.get("schemas") or {}).get("estimated_tokens") or 0),
                "conversation_tokens": int((blocks.get("conversation") or {}).get("estimated_tokens") or 0),
                "tool_results": [],
                "actual_input_tokens": int(trace.get("actual_input_tokens") or 0),
            })

    tracker = ToolCallStats()
    tracker_calls = [
        {"name": name, "input": args}
        for name, args, _ in all_calls
    ]
    tracker.observe(tracker_calls)
    max_result = max(
        (result["tokens"] for group in message_results for result in group),
        default=0,
    )
    trajectory_type = _classify(
        all_calls,
        title=task_title,
        close_reason=close_reason,
        max_tool_result_tokens=max_result,
    )
    if "failed" in verification_results:
        verification = "failed"
    elif "passed" in verification_results:
        verification = "passed"
    else:
        verification = "unknown"

    normalized_state = str(task_state or "").casefold()
    success = normalized_state in {"completed", "approved", "taskstate.completed"} or (
        close_reason == "task_done" and normalized_state not in {"rejected", "cancelled"}
    )
    return {
        "version": 1,
        "session_hash": _hash_id(session_id),
        "trajectory_type": trajectory_type,
        "success": success,
        "verification_result": verification,
        "requests": requests,
        "baseline": {
            "tool_calls": tracker.tool_calls,
            "repeated_reads": tracker.repeated_reads,
            "read_requested_lines": tracker.read_requested_lines,
            "read_overlap_lines": tracker.read_overlap_lines,
            "read_overlap_ratio": round(
                tracker.read_overlap_lines / tracker.read_requested_lines, 4
            ) if tracker.read_requested_lines else 0.0,
        },
    }


def replay_trajectory(
    trajectory: dict[str, Any],
    *,
    keep_raw_results: int = 3,
    receipt_tokens: int = 32,
) -> dict[str, Any]:
    current_total = proposed_total = 0
    current_max = proposed_max = 0
    steps: list[dict[str, int]] = []
    keep_raw_results = max(0, int(keep_raw_results))
    receipt_tokens = max(1, int(receipt_tokens))

    for request in trajectory.get("requests") or []:
        result_tokens = [max(0, int(r.get("tokens") or 0)) for r in request.get("tool_results") or []]
        base = sum(max(0, int(request.get(k) or 0)) for k in (
            "system_tokens", "schema_tokens", "conversation_tokens"
        ))
        estimated_current = base + sum(result_tokens)
        actual_current = max(0, int(request.get("actual_input_tokens") or 0))
        current = actual_current or estimated_current
        old_count = max(0, len(result_tokens) - keep_raw_results)
        proposed_tools = sum(result_tokens[-keep_raw_results:]) if keep_raw_results else 0
        proposed_tools += old_count * receipt_tokens
        # Preserve the provider serialization/tokenizer delta measured live.
        provider_delta = max(0, current - estimated_current)
        proposed = base + proposed_tools + provider_delta
        current_total += current
        proposed_total += proposed
        current_max = max(current_max, current)
        proposed_max = max(proposed_max, proposed)
        steps.append({"current_input_tokens": current, "proposed_input_tokens": proposed})

    return {
        "session_hash": trajectory.get("session_hash"),
        "trajectory_type": trajectory.get("trajectory_type"),
        "current_total_input_tokens": current_total,
        "proposed_total_input_tokens": proposed_total,
        "saved_input_tokens": max(0, current_total - proposed_total),
        "saved_percent": round(
            ((current_total - proposed_total) / current_total) * 100, 2
        ) if current_total else 0.0,
        "current_max_prompt_tokens": current_max,
        "proposed_max_prompt_tokens": proposed_max,
        "steps": steps,
        **(trajectory.get("baseline") or {}),
        "success": bool(trajectory.get("success")),
        "verification_result": trajectory.get("verification_result", "unknown"),
    }


def replay_corpus(
    trajectories: Iterable[dict[str, Any]],
    *,
    keep_raw_results: int = 3,
    receipt_tokens: int = 32,
) -> dict[str, Any]:
    rows = [
        replay_trajectory(
            trajectory,
            keep_raw_results=keep_raw_results,
            receipt_tokens=receipt_tokens,
        )
        for trajectory in trajectories
    ]

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        current = sum(r["current_total_input_tokens"] for r in group)
        proposed = sum(r["proposed_total_input_tokens"] for r in group)
        requested = sum(r.get("read_requested_lines", 0) for r in group)
        overlap = sum(r.get("read_overlap_lines", 0) for r in group)
        return {
            "trajectories": len(group),
            "total_input_tokens": current,
            "proposed_total_input_tokens": proposed,
            "saved_input_tokens": max(0, current - proposed),
            "saved_percent": round(((current - proposed) / current) * 100, 2) if current else 0.0,
            "max_prompt_tokens": max((r["current_max_prompt_tokens"] for r in group), default=0),
            "proposed_max_prompt_tokens": max((r["proposed_max_prompt_tokens"] for r in group), default=0),
            "tool_calls": sum(r.get("tool_calls", 0) for r in group),
            "repeated_reads": sum(r.get("repeated_reads", 0) for r in group),
            "read_overlap_ratio": round(overlap / requested, 4) if requested else 0.0,
            "successes": sum(1 for r in group if r.get("success")),
            "verification": dict(_counts(r.get("verification_result", "unknown") for r in group)),
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("trajectory_type") or "unknown")].append(row)
    return {
        "policy": {"keep_raw_results": keep_raw_results, "receipt_tokens": receipt_tokens},
        "baseline": summarize(rows),
        "by_trajectory_type": {name: summarize(group) for name, group in sorted(grouped.items())},
        "trajectories": rows,
    }


def _counts(values: Iterable[str]) -> Iterable[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return sorted(counts.items())
