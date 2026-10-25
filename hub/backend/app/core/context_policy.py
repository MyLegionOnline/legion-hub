"""Configuration and override resolution for active context compaction."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on", "enabled"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Module-level config derived from env
# ---------------------------------------------------------------------------

# "window"    — enforce per-call context-window size (default).
# "cumulative" — legacy behaviour (token sum per session), kept for rollback only.
ContextLimitMode = Literal["window", "cumulative"]
_VALID_CONTEXT_LIMIT_MODES = frozenset({"window", "cumulative"})


def _read_context_limit_mode(value: str | None = None) -> ContextLimitMode:
    """Read and validate the only supported context-limit modes.

    Invalid configuration is a startup error, rather than a silent fallback to
    a mode with different session-closing semantics.
    """
    raw = os.getenv("CONTEXT_LIMIT_MODE", "window") if value is None else value
    mode = str(raw).strip().casefold()
    if mode not in _VALID_CONTEXT_LIMIT_MODES:
        allowed = ", ".join(sorted(_VALID_CONTEXT_LIMIT_MODES))
        raise ValueError(
            f"Invalid CONTEXT_LIMIT_MODE={raw!r}; expected one of: {allowed}"
        )
    if mode == "window":
        return "window"
    return "cumulative"


CONTEXT_LIMIT_MODE: ContextLimitMode = _read_context_limit_mode()


def context_limit_mode() -> ContextLimitMode:
    """Return the validated, process-wide context-limit mode."""
    return CONTEXT_LIMIT_MODE


def uses_window_limit() -> bool:
    """Whether cumulative session-token guards must be bypassed."""
    return context_limit_mode() == "window"


def uses_cumulative_limit() -> bool:
    """Whether the temporary legacy cumulative guard is enabled."""
    return context_limit_mode() == "cumulative"

# Default enforced context-window budget (tokens). Matches TOKEN_LIMIT so that
# agents without an explicit setting behave exactly as before.
_CONTEXT_WINDOW_LIMIT_DEFAULT: int = _env_int("CONTEXT_WINDOW_LIMIT", 100_000)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
    return None


# "on_demand" — keep every tool result verbatim while the assembled prompt
#               fits the budget; evict the oldest only when it does not.
# "eager"     — legacy behaviour: a fixed recent-result window is enforced on
#               every request, even when the context window is nearly empty.
EvictionMode = Literal["on_demand", "eager"]
_VALID_EVICTION_MODES = frozenset({"on_demand", "eager"})


def _read_eviction_mode(value: Any, default: EvictionMode) -> EvictionMode:
    """Read an eviction mode, falling back to ``default`` on junk input."""
    mode = str(default if value is None else value).strip().casefold()
    return mode if mode in _VALID_EVICTION_MODES else default


@dataclass(frozen=True)
class ContextPolicy:
    enabled: bool
    input_cap: int
    active_tool_result_cap: int
    keep_raw_results: int
    output_reserve: int
    model_context_limit: int
    checkpoint_cap: int
    # Enforced per-call context-window budget set by the application.
    # Distinct from model_context_limit (provider capacity, read-only).
    context_window_limit: int = 100_000
    # When tool results are demoted to receipts.  ``active_tool_result_cap``
    # and ``keep_raw_results`` are only applied in "eager" mode.
    eviction_mode: EvictionMode = "on_demand"


def policy_for_role(role: str) -> ContextPolicy:
    is_chat_master = str(role or "").casefold() in {
        "cm",
        "chat_master",
        "chatmaster",
    }
    prefix = "CONTEXT_CHAT_MASTER" if is_chat_master else "CONTEXT_AGENT"
    cwl = max(1024, _CONTEXT_WINDOW_LIMIT_DEFAULT)
    # input_cap defaults to context_window_limit — no hidden secondary cap.
    # Administrators can still narrow it explicitly via the ENV variable.
    return ContextPolicy(
        enabled=_env_bool("CONTEXT_GATEWAY_ENABLED", True),
        input_cap=max(512, _env_int(f"{prefix}_INPUT_CAP", cwl)),
        active_tool_result_cap=max(
            128,
            int(os.getenv(f"{prefix}_TOOL_RESULT_CAP", "2000" if is_chat_master else "6000")),
        ),
        keep_raw_results=max(
            0,
            int(os.getenv(f"{prefix}_KEEP_RAW_RESULTS", "2" if is_chat_master else "3")),
        ),
        output_reserve=max(
            256,
            int(os.getenv(f"{prefix}_OUTPUT_RESERVE", "2048" if is_chat_master else "4096")),
        ),
        model_context_limit=max(
            1024,
            int(os.getenv("CONTEXT_MODEL_LIMIT", "200000")),
        ),
        checkpoint_cap=max(
            128,
            int(os.getenv(f"{prefix}_CHECKPOINT_CAP", "400" if is_chat_master else "600")),
        ),
        context_window_limit=cwl,
        eviction_mode=_read_eviction_mode(
            os.getenv(f"{prefix}_EVICTION_MODE") or os.getenv("CONTEXT_EVICTION_MODE"),
            "on_demand",
        ),
    )


_OVERRIDE_FIELDS = {
    "enabled": ("enabled", "context_gateway_enabled"),
    "input_cap": ("input_cap", "context_input_cap"),
    "active_tool_result_cap": (
        "active_tool_result_cap",
        "context_tool_result_cap",
    ),
    "keep_raw_results": ("keep_raw_results", "context_keep_raw_results"),
    "output_reserve": ("output_reserve", "context_output_reserve"),
    # ``context_length`` is the existing provider metadata field on agents.
    # Treat it as a fallback so provider capacity follows the selected model
    # even before all stored agents have the explicit runtime-named field.
    "model_context_limit": (
        "model_context_limit",
        "context_model_limit",
        "context_length",
    ),
    "checkpoint_cap": ("checkpoint_cap", "context_checkpoint_cap"),
    "context_window_limit": ("context_window_limit", "context_window", "context_limit"),
    "eviction_mode": ("eviction_mode", "context_eviction_mode"),
}


def _gateway_mapping(config: Any, role: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    merged = dict(config)
    gateway = config.get("context_gateway")
    if isinstance(gateway, bool):
        merged["enabled"] = gateway
    elif isinstance(gateway, dict):
        merged.update(gateway)
        role_config = gateway.get(role)
        if isinstance(role_config, dict):
            merged.update(role_config)
    return merged


def apply_policy_overrides(
    policy: ContextPolicy,
    *configs: dict[str, Any] | None,
    role: str,
) -> ContextPolicy:
    """Apply room then agent then LLM overrides (later configs win)."""
    values = dict(policy.__dict__)
    for config in configs:
        mapping = _gateway_mapping(config, role)
        for field, aliases in _OVERRIDE_FIELDS.items():
            raw = next((mapping[name] for name in aliases if name in mapping), None)
            if raw is None:
                continue
            if field == "enabled":
                parsed = _as_bool(raw)
                if parsed is not None:
                    values[field] = parsed
                continue
            if field == "eviction_mode":
                values[field] = _read_eviction_mode(raw, values[field])
                continue
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                continue
            # Provider metadata uses 0 when its context capacity is unknown.
            # It must not replace the configured model ceiling with the
            # minimum clamp of 1,024 tokens.
            if field == "model_context_limit" and parsed <= 0:
                continue
            values[field] = parsed
    values["input_cap"] = max(512, values["input_cap"])
    values["active_tool_result_cap"] = max(128, values["active_tool_result_cap"])
    values["keep_raw_results"] = max(0, values["keep_raw_results"])
    values["output_reserve"] = max(256, values["output_reserve"])
    values["model_context_limit"] = max(1024, values["model_context_limit"])
    values["checkpoint_cap"] = max(128, values["checkpoint_cap"])
    values["context_window_limit"] = max(1024, values["context_window_limit"])
    values["eviction_mode"] = _read_eviction_mode(values["eviction_mode"], "on_demand")
    # input_cap must never silently exceed context_window_limit: the window
    # boundary is the authoritative user-visible cap.
    values["input_cap"] = min(values["input_cap"], values["context_window_limit"])
    return replace(policy, **values)


async def resolve_context_policy(
    *,
    role: str,
    db: AsyncSession,
    room_id: UUID,
    agent_id: UUID | None,
    llm_config: dict[str, Any] | None,
) -> ContextPolicy:
    """Resolve global → room → agent → request configuration."""
    from app.models.agent import Agent
    from app.models.room import Room

    room = await db.get(Room, room_id)
    agent = await db.get(Agent, agent_id) if agent_id is not None else None
    return apply_policy_overrides(
        policy_for_role(role),
        getattr(room, "coordinator_config", None),
        getattr(agent, "config", None),
        getattr(agent, "llm_config", None),
        llm_config,
        role=role,
    )
