"""Compact session-level Context Gateway waterfall projection."""
from __future__ import annotations

from typing import Any


def context_waterfall(snapshot: Any) -> dict[str, int]:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    gateway = snapshot.get("context_gateway")
    gateway = gateway if isinstance(gateway, dict) else {}
    traces = snapshot.get("context_baseline")
    traces = traces if isinstance(traces, list) else []
    latest = next((item for item in reversed(traces) if isinstance(item, dict)), {})
    blocks = latest.get("blocks")
    blocks = blocks if isinstance(blocks, dict) else {}

    def block_tokens(name: str) -> int:
        value = blocks.get(name)
        value = value if isinstance(value, dict) else {}
        return max(0, int(value.get("estimated_tokens") or 0))

    active_tools = max(0, int(gateway.get("last_active_tool_tokens") or 0))
    if not active_tools:
        active_tools = block_tokens("tool_results")
    return {
        "pinned": block_tokens("system") + block_tokens("schemas"),
        "conversation": block_tokens("conversation"),
        "active_tools": active_tools,
        "evicted": max(0, int(gateway.get("tokens_evicted_total") or 0)),
        "provider_cached": max(
            0, int(gateway.get("provider_cache_read_tokens") or 0)
        ),
    }
