"""Tool registry — core types for workspace tools."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

_MAX_OUTPUT = 16_384


def _truncate(text: str) -> str:
    if len(text) <= _MAX_OUTPUT:
        return text
    suffix = f"\n[...output truncated at {_MAX_OUTPUT} chars]"
    return text[: _MAX_OUTPUT - len(suffix)] + suffix


def canonicalize_tool_input(tool_name: str, tool_input: dict[str, Any] | None) -> dict[str, Any]:
    """Return stable tool arguments before execution and event persistence.

    The workspace path contract is deliberately enforced at the registry
    boundary as well as inside individual tools.  This makes equivalent
    legacy inputs (``/workspace/src/a.py`` and ``src/a.py``) share the same
    retry/error identity and prevents a model from spending turns correcting
    its own path spelling.
    """
    value = dict(tool_input or {})
    try:
        from app.core.path_contract import normalize_workspace_glob, normalize_workspace_path

        # screenshot.path is a URL path on the declared service, not a
        # workspace file path. Keep its leading slash and query string so the
        # preview gateway receives the same path the Browser tab would open.
        path_keys = ("file_path", "path_prefix", "dir") if tool_name == "screenshot" else (
            "path", "file_path", "path_prefix", "dir"
        )
        for key in path_keys:
            if key not in value or value[key] in (None, ""):
                continue
            value[key] = normalize_workspace_path(str(value[key]), allow_empty=key in {"path_prefix", "dir"})
        if value.get("glob"):
            value["glob"] = normalize_workspace_glob(str(value["glob"]))
    except Exception:
        # Individual tools still return their deterministic validation error.
        # Canonicalization is an optimization and must never hide it.
        pass
    return value


def tool_error_key(tool_name: str, tool_input: dict[str, Any] | None, result: Any) -> str | None:
    """Build a stable key for a repeatable tool failure, if *result* is one."""
    text = str(result or "").strip()
    first_line = text.splitlines()[0].casefold() if text else ""
    is_error = (
        first_line.startswith(("error:", "file not found:", "invalid path:", "invalid pattern:"))
        or "not available in this context" in text.casefold()
        or "tool call budget exhausted" in text.casefold()
        or bool(re.search(r"(?m)^exit_code:\s*-?[1-9]\d*\b", text))
    )
    if not is_error:
        return None
    payload = {
        "tool": str(tool_name),
        "args": canonicalize_tool_input(tool_name, tool_input),
        "error": text[:2000],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def safe_tool_error_text(result: Any, *, max_chars: int = 200) -> str | None:
    """User-safe one-line summary of a tool error result.

    Only the first line survives (multi-line tool output can include command
    output, stack traces, or file contents), long enough to explain *what*
    failed but capped well short of a full payload, and run through the same
    sensitive-key vocabulary as ``redact_secrets`` so an accidentally leaked
    token/password in a tool's own error text doesn't reach the chat UI.
    """
    text = str(result or "").strip()
    if not text:
        return None
    first_line = text.splitlines()[0].strip()
    from app.core.secrets.redaction import redact_secrets_in_text

    redacted = redact_secrets_in_text(first_line)
    if len(redacted) > max_chars:
        redacted = redacted[: max_chars - 1].rstrip() + "…"
    return redacted or None


@dataclass
class ToolErrorGuard:
    """Bound automatic retries of an identical tool error in one LLM loop."""

    max_repeats: int = 2
    counts: dict[str, int] = field(default_factory=dict)

    def observe(self, tool_name: str, tool_input: dict[str, Any] | None, result: Any) -> int:
        key = tool_error_key(tool_name, tool_input, result)
        if key is None:
            return 0
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]


@dataclass
class ToolContext:
    room_id: UUID
    session_id: UUID
    task_id: Optional[UUID]
    db: AsyncSession
    # The fields below are populated by the proxy tool loop when a tool needs
    # to start a provider call of its own (currently ``run_subtask``).  They
    # are intentionally optional so direct workspace-tool callers and tests
    # keep the small, existing context contract.
    llm_config: Optional[dict[str, Any]] = None
    secret_value: Any = None
    actor_agent_id: Optional[UUID] = None
    actor_label: Optional[str] = None
    workspace_ant_tools: Optional[list[dict[str, Any]]] = None
    workspace_oai_tools: Optional[list[dict[str, Any]]] = None
    mcp_ant_tools: Optional[list[dict[str, Any]]] = None
    mcp_oai_tools: Optional[list[dict[str, Any]]] = None
    mcp_bindings: Optional[dict[str, Any]] = None
    mcp_user_id: Optional[UUID] = None
    # Correlates a child tool loop with the parent ``run_subtask`` call for
    # durable chat observability.  It is absent for normal tool calls.
    subtask_activity: Optional[dict[str, Any]] = None
    # Generation this tool call belongs to.  A child tool loop must carry it so
    # the stop fence (``task_runtime.is_run_current``) reaches the child too —
    # without it a cancelled task kept running inside run_subtask.
    run_id: Optional[str] = None

    async def release_db(self, *, reason: str = "tool") -> bool:
        """Commit before a long external wait (stage 1.2.3, internal design notes §8.2).

        A tool that reads from the database and then waits on ``workspaced``,
        MCP or a provider must not hold the loop's pool connection while it
        waits.  Read what you need first, release, wait, then write.  First
        step towards the per-operation ``ctx.db_session()`` of stage 1.3.
        """
        return await release_tool_db(self, reason=reason)


async def release_tool_db(ctx: Any, *, reason: str = "tool") -> bool:
    """``ctx.release_db()`` for any context shape — tests pass plain namespaces."""
    from app.db.release import release_db

    return await release_db(getattr(ctx, "db", None), reason=reason)


class Tool(ABC):
    name: str
    description: str
    input_schema: dict[str, Any]
    roles: frozenset[str]          # subset of {"agent", "cm"}
    role_descriptions: dict[str, str] = {}  # per-role description overrides
    requires_container: bool = False
    mutating: bool = False
    # Offered even when the agent has a tools_allowed list: a channel back to
    # the user, not a capability (internal design notes Stage 2).
    always_offered: bool = False

    def ant_schema(self, role: str = "agent") -> dict[str, Any]:
        desc = self.role_descriptions.get(role, self.description)
        return {"name": self.name, "description": desc, "input_schema": self.input_schema}

    def oai_schema(self, role: str = "agent") -> dict[str, Any]:
        desc = self.role_descriptions.get(role, self.description)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": self.input_schema,
            },
        }

    @abstractmethod
    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def tools_for(self, role: str, allowed: list[str] | None = None) -> list[Tool]:
        tools = [t for t in self._tools.values() if role in t.roles]
        if allowed is not None:
            tools = [t for t in tools if t.name in allowed or t.always_offered]
        # Registry import order is an implementation detail.  A stable schema
        # order keeps the provider-cacheable prefix identical across restarts.
        return sorted(tools, key=lambda tool: tool.name)

    def ant_schemas_for(self, role: str, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        return [t.ant_schema(role) for t in self.tools_for(role, allowed)]

    def oai_schemas_for(self, role: str, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        return [t.oai_schema(role) for t in self.tools_for(role, allowed)]


REGISTRY = ToolRegistry()

# Tools that only make sense inside a direct user↔CM / user↔agent dialogue
# session (they mutate that session's own pending history). They carry the
# "agent"/"cm" roles for the direct-session loop but must never be offered to
# proxy sessions, cm_self_plan decomposition, or run_subtask children.
DIALOGUE_ONLY_TOOLS = frozenset({"merge_draft"})


def strip_dialogue_only(schemas: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Drop DIALOGUE_ONLY_TOOLS from an anthropic/openai schema list."""
    out: list[dict[str, Any]] = []
    for schema in schemas or []:
        name = schema.get("name") or (schema.get("function") or {}).get("name")
        if name in DIALOGUE_ONLY_TOOLS:
            continue
        out.append(schema)
    return out
