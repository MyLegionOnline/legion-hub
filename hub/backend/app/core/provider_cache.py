"""Provider-neutral helpers for stable prompts and provider cache telemetry.

The context gateway remains correct when provider caching is unavailable.  The
helpers in this module only make the stable prefix explicit and normalize the
different usage shapes returned by OpenAI-compatible and Anthropic APIs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


def _dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _tool_name(tool: dict[str, Any]) -> str:
    if not isinstance(tool, dict):
        return ""
    if isinstance(tool.get("function"), dict):
        return str(tool["function"].get("name") or "")
    return str(tool.get("name") or "")


def stable_tool_schemas(tools: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return a deterministic, deep-copied schema order.

    Tool names are the primary key and the complete canonical JSON is a
    tie-breaker.  This makes equivalent registry/configuration orderings share
    the same provider cache prefix without mutating caller-owned lists.
    """
    values = [tool for tool in (tools or []) if isinstance(tool, dict)]
    return sorted(
        (json.loads(_dump(tool)) for tool in values),
        key=lambda tool: (_tool_name(tool), _dump(tool)),
    )


def tool_schema_version(tools: Iterable[dict[str, Any]] | None) -> str:
    digest = hashlib.sha256(_dump(stable_tool_schemas(tools)).encode("utf-8")).hexdigest()
    return digest[:16]


def system_prompt_version(system_prompt: str, configured: Any = None) -> str:
    if configured is not None and str(configured).strip():
        return str(configured).strip()
    digest = hashlib.sha256(str(system_prompt or "").encode("utf-8")).hexdigest()
    return digest[:16]


def provider_cache_key(
    *,
    room_id: Any,
    agent_role: str,
    system_version: str,
    tool_schema_version_value: str,
) -> str:
    """Build a bounded, opaque key for providers supporting prompt cache keys."""
    payload = {
        "room": str(room_id),
        "role": str(agent_role or "unknown"),
        "system": str(system_version),
        "tools": str(tool_schema_version_value),
    }
    digest = hashlib.sha256(_dump(payload).encode("utf-8")).hexdigest()
    return f"legion-{digest[:32]}"


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        try:
            dumped = to_dict()
            return dumped if isinstance(dumped, dict) else {}
        except Exception:
            return {}
    return {}


def _first_int(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        raw = mapping.get(key)
        try:
            if raw is not None:
                return max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return 0


@dataclass(frozen=True)
class ProviderCacheUsage:
    read_tokens: int = 0
    write_tokens: int = 0
    # OpenAI-compatible usage counts cached tokens inside prompt/input tokens;
    # the Anthropic Messages API reports them next to ``input_tokens``.
    included_in_input: bool = True


def extract_cache_usage(usage: Any, provider: str = "") -> ProviderCacheUsage:
    """Normalize cache read/write counters across provider response formats."""
    data = _mapping(usage)
    provider = str(provider or "").casefold()

    # Anthropic Messages API.
    read = _first_int(data, "cache_read_input_tokens", "cached_input_tokens")
    write = _first_int(data, "cache_creation_input_tokens", "cache_write_input_tokens")

    # OpenAI and OpenAI-compatible APIs expose nested prompt token details.
    details = _mapping(data.get("prompt_tokens_details"))
    input_details = _mapping(data.get("input_tokens_details"))
    read = max(
        read,
        _first_int(
            data,
            "cached_tokens",
            "cache_read_input_tokens",
            "prompt_cache_hit_tokens",
        ),
        _first_int(details, "cached_tokens", "cache_read_tokens"),
        _first_int(input_details, "cached_tokens", "cache_read_tokens"),
    )
    write = max(
        write,
        _first_int(data, "cache_write_input_tokens", "cache_creation_input_tokens"),
        _first_int(details, "cache_write_tokens"),
        _first_int(input_details, "cache_write_tokens"),
    )
    anthropic_shape = (
        "prompt_tokens" not in data
        and "input_tokens" in data
        and ("cache_read_input_tokens" in data or "cache_creation_input_tokens" in data)
    )
    return ProviderCacheUsage(
        read_tokens=read, write_tokens=write, included_in_input=not anthropic_shape
    )


def openai_cache_request_extra(
    *,
    enabled: bool,
    cache_key: str,
) -> dict[str, Any]:
    """Return optional Chat Completions parameters, never enabled implicitly."""
    if not enabled or not cache_key:
        return {}
    return {"prompt_cache_key": cache_key}


def anthropic_system_payload(
    system_prompt: str,
    *,
    enabled: bool,
) -> str | list[dict[str, Any]]:
    """Mark the stable system prefix cacheable when explicitly requested."""
    if not enabled:
        return system_prompt
    return [{
        "type": "text",
        "text": str(system_prompt or ""),
        "cache_control": {"type": "ephemeral"},
    }]
