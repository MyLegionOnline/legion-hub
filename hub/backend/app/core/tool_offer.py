"""Which tools an agent session is offered, decided once per session.

internal design notes S2/S3.  Every LLM step carries the
schemas of every offered tool, so rarely needed tools are *deferred*: the
agent sees them only as one catalogue line inside the ``load_tools`` tool, and
their full schemas join the request from the step after it loads them.  They
are then called natively, like any other tool.

Invariants (they keep the provider prompt cache intact):

* The decision is made from inputs that are fixed for a session (role, the
  agent's own ``tools_allowed``, its MCP bindings, the task's archive), so the
  tool list is byte-identical from step to step.
* The loaded set only grows.  It is rebuilt from the session history on every
  call, so it survives turns, pauses and restarts without separate storage.
* Core tools are never deferred.  Nothing is stored in the room repository.

The schemas of deferred tools are not kept here: callers keep passing the full
workspace and MCP lists, and :meth:`ToolOffer.offered` filters them per step.
Dispatch is unchanged, so a deferred tool the model calls by name still runs
and counts as loaded from then on.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Optional

logger = logging.getLogger(__name__)

LOAD_TOOLS = "load_tools"

#: MCP tools are deferred only when their schemas together are at least this
#: big.  Below it, one load step would cost more than the schemas save.
MCP_DEFER_MIN_CHARS = int(os.getenv("TOOL_OFFER_MCP_DEFER_MIN_CHARS", "3000"))

#: Set to "0" to offer every tool in full (the pre-S3 behaviour).
DEFERRED_TOOLS_ENABLED = os.getenv("AGENT_DEFERRED_TOOLS_ENABLED", "1").lower() not in (
    "0",
    "false",
    "no",
)

#: One catalogue line per deferred tool is cut to this many characters.
_SUMMARY_MAX = 110

# Direct-session tool notes (user_sessions._tool_note_text / ToolOffer.load).
_NOTE_PREFIX = "[tools executed: "
#: A short name that matches up to this many deferred tools loads them all.
_MAX_FUZZY_MATCHES = 3
_LOADED_RE = re.compile(r"(?:^|\s)(?:Already loaded|Loaded): ([^.\n]+)\.")

Fmt = Literal["anthropic", "openai"]


def schema_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(schema.get("name") or "")


def _schema_description(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if isinstance(function, Mapping):
        return str(function.get("description") or "")
    return str(schema.get("description") or "")


def summarize(description: str) -> str:
    """First sentence of a tool description, cut to one catalogue line."""
    text = " ".join(str(description or "").split())
    for stop in (". ", "; ", " — "):
        cut = text.find(stop)
        if 0 < cut < _SUMMARY_MAX:
            text = text[:cut]
            break
    text = text.rstrip(".")
    if len(text) > _SUMMARY_MAX:
        text = text[: _SUMMARY_MAX - 1].rstrip() + "…"
    return text


def load_tools_schema(catalogue: list[tuple[str, str]], fmt: Fmt) -> dict[str, Any]:
    """The ``load_tools`` schema; its description carries the catalogue."""
    lines = "\n".join(f"- {name}: {summary}" if summary else f"- {name}" for name, summary in catalogue)
    description = (
        "These tools are not loaded yet. Load the ones the task needs by name — all you "
        "expect to need in ONE call; they stay loaded and you call them normally from your "
        "next step.\n" + lines
    )
    parameters = {
        "type": "object",
        "properties": {
            "names": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "required": ["names"],
    }
    if fmt == "anthropic":
        return {"name": LOAD_TOOLS, "description": description, "input_schema": parameters}
    return {
        "type": "function",
        "function": {"name": LOAD_TOOLS, "description": description, "parameters": parameters},
    }


@dataclass
class ToolOffer:
    """Deferred tool names of one session and the ones it has loaded."""

    deferred: frozenset[str]
    loaded: set[str] = field(default_factory=set)
    #: Bumped whenever ``loaded`` grows; callers rebuild their tool list then.
    version: int = 0

    @property
    def active(self) -> bool:
        return bool(self.deferred)

    def pending(self) -> frozenset[str]:
        return frozenset(self.deferred - self.loaded)

    def offered(self, schemas: Iterable[dict[str, Any]], fmt: Fmt) -> list[dict[str, Any]]:
        """Filter a full schema list down to what this step may see.

        ``load_tools`` is present whenever any tool of *schemas* is deferred
        (loaded or not), so its catalogue never changes within a session.
        """
        values = [schema for schema in schemas if isinstance(schema, dict)]
        catalogue = sorted(
            (schema_name(schema), summarize(_schema_description(schema)))
            for schema in values
            if schema_name(schema) in self.deferred
        )
        out = [
            schema
            for schema in values
            if schema_name(schema) not in self.deferred or schema_name(schema) in self.loaded
        ]
        if catalogue:
            out.append(load_tools_schema(catalogue, fmt))
        return out

    def _mark(self, names: Iterable[str]) -> list[str]:
        added = [name for name in names if name in self.deferred and name not in self.loaded]
        if added:
            self.loaded.update(added)
            self.version += 1
        return added

    def absorb(self, messages: Iterable[Mapping[str, Any]] | None) -> bool:
        """Mark as loaded what the history already loaded or called.

        Reads ``load_tools`` calls and direct calls of deferred tools in both
        wire formats.  Returns True when the loaded set grew.
        """
        if not self.deferred:
            return False
        found: list[str] = []
        for message in messages or []:
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            for name, args in _calls_in(message):
                if name == LOAD_TOOLS:
                    found.extend(self._resolve(_names_arg(args))[0])
                elif name in self.deferred:
                    found.append(name)
        return bool(self._mark(found))

    def absorb_notes(self, messages: Iterable[Mapping[str, Any]] | None) -> bool:
        """Same as :meth:`absorb` for direct sessions, which keep tool calls as text.

        ``user_sessions`` stores a finished tool exchange as an assistant note
        "[tools executed: a, b — …]" followed by "### name {args}\\nresult"
        lines; ``load_tools`` results there read "Loaded: a, b." / "Already
        loaded: a, b.".  Returns True when the loaded set grew.
        """
        if not self.deferred:
            return False
        found: list[str] = []
        for message in messages or []:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            if message.get("role") == "assistant" and content.startswith(_NOTE_PREFIX):
                listed = content[len(_NOTE_PREFIX):].split(" — ", 1)[0]
                found.extend(part.strip() for part in listed.split(","))
            elif f"### {LOAD_TOOLS} " in content:
                for match in _LOADED_RE.finditer(content):
                    found.extend(part.strip() for part in match.group(1).split(","))
        return bool(self._mark(found))

    def load(self, tool_input: Mapping[str, Any] | None) -> str:
        """Execute ``load_tools``; returns the tool result text."""
        names = _names_arg(tool_input)
        if not names:
            return (
                "Error: pass names=[...] with tool names from the load_tools list: "
                + ", ".join(sorted(self.deferred))
            )
        # Weak models ask for a short name ("web_search") instead of the full
        # "mcp__mcp-searxng__searxng_web_search" (dev live check, 2026-11-15):
        # resolve a short name that points at a few tools instead of a dead end.
        resolved, ambiguous, unknown = self._resolve(names)
        already = [n for n in resolved if n in self.loaded]
        added = self._mark(resolved)
        parts: list[str] = []
        if added:
            parts.append(
                f"Loaded: {', '.join(added)}. Their full schemas are available from your next "
                "step; call them by these exact names."
            )
        if already:
            parts.append(f"Already loaded: {', '.join(already)}.")
        for name, matches in ambiguous.items():
            parts.append(f"'{name}' matches several tools, pick exact names: {', '.join(matches)}.")
        if unknown:
            parts.append(
                f"Not deferred tools: {', '.join(unknown)} (already in your tool list or do not "
                "exist). Deferred tools are: " + ", ".join(sorted(self.pending())) + "."
            )
        return " ".join(parts)

    def _resolve(self, names: Iterable[str]) -> tuple[list[str], dict[str, list[str]], list[str]]:
        """Split requested names into (deferred names to load, ambiguous, unknown)."""
        resolved: list[str] = []
        unknown: list[str] = []
        ambiguous: dict[str, list[str]] = {}
        for name in names:
            if name in self.deferred:
                matches = [name]
            else:
                matches = self._matches(name)
            if 1 <= len(matches) <= _MAX_FUZZY_MATCHES:
                resolved.extend(m for m in matches if m not in resolved)
            elif matches:
                ambiguous[name] = matches
            else:
                unknown.append(name)
        return resolved, ambiguous, unknown

    def _matches(self, name: str) -> list[str]:
        key = name.strip().lower()
        if not key:
            return []
        exact_tail = sorted(
            d for d in self.deferred
            if d.lower().endswith("__" + key) or d.lower().rsplit("__", 1)[-1] == key
        )
        if exact_tail:
            return exact_tail
        return sorted(d for d in self.deferred if key in d.lower())

    def for_child(self, *, drop: Iterable[str] = ()) -> "ToolOffer":
        """A copy for a run_subtask child: same deferral, its own loaded set."""
        dropped = set(drop)
        return ToolOffer(
            deferred=frozenset(self.deferred - dropped),
            loaded=set(self.loaded - dropped),
        )


def _calls_in(message: Mapping[str, Any]) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                calls.append((str(block.get("name") or ""), block.get("input")))
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, Mapping) else getattr(call, "function", None)
        if function is None:
            continue
        name = function.get("name") if isinstance(function, Mapping) else getattr(function, "name", None)
        args = function.get("arguments") if isinstance(function, Mapping) else getattr(function, "arguments", None)
        calls.append((str(name or ""), args))
    return calls


def _names_arg(args: Any) -> list[str]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return []
    if not isinstance(args, Mapping):
        return []
    raw = args.get("names")
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for name in raw:
        text = str(name or "").strip()
        if text and text not in out:
            out.append(text)
    return out


# ── the per-session decision (S2) ───────────────────────────────────────────


@dataclass(frozen=True)
class TaskToolHints:
    """Universal facts about a task that switch on rarely needed tool groups."""

    #: The agent can deploy to a server (the Legion VPS MCP is bound).
    can_deploy: bool = False
    #: The task was set in a dialogue whose archive exists (plan_dialogue).
    has_dialogue_archive: bool = False


def enabled_groups(hints: TaskToolHints) -> frozenset[str]:
    groups: set[str] = set()
    if hints.can_deploy:
        groups.add("deploy")
    if hints.has_dialogue_archive:
        groups.add("dialogue_archive")
    return frozenset(groups)


def mcp_schema_chars(schemas: Iterable[dict[str, Any]] | None) -> int:
    return sum(len(json.dumps(schema, ensure_ascii=False)) for schema in schemas or [])


def plan_agent_offer(
    *,
    registry: Any,
    allowed: Optional[list[str]],
    hints: TaskToolHints,
    mcp_schemas: Iterable[dict[str, Any]] | None = None,
    role: str = "agent",
) -> Optional[ToolOffer]:
    """Decide the deferred tools of one agent session, or None for "offer all".

    Built-in tools carry ``offer_group``; a group the task does not need is
    deferred.  A tool the agent's settings name explicitly (``tools_allowed``)
    is always offered in full — that toggle is the owner's choice.  MCP tools
    are deferred together once their schemas reach ``MCP_DEFER_MIN_CHARS``.
    """
    if not DEFERRED_TOOLS_ENABLED:
        return None
    try:
        return _plan(registry=registry, allowed=allowed, hints=hints, mcp_schemas=mcp_schemas, role=role)
    except Exception:  # noqa: BLE001 — never fail a session over a schema diet
        logger.warning("tool_offer: could not plan deferred tools; offering all", exc_info=True)
        return None


def _plan(
    *,
    registry: Any,
    allowed: Optional[list[str]],
    hints: TaskToolHints,
    mcp_schemas: Iterable[dict[str, Any]] | None,
    role: str,
) -> Optional[ToolOffer]:
    groups = enabled_groups(hints)
    explicit = set(allowed or [])
    deferred = {
        tool.name
        for tool in registry.tools_for(role, allowed)
        if getattr(tool, "offer_group", None)
        and tool.offer_group not in groups
        and tool.name not in explicit
    }
    mcp_list = [schema for schema in (mcp_schemas or []) if isinstance(schema, dict)]
    if mcp_list and mcp_schema_chars(mcp_list) >= MCP_DEFER_MIN_CHARS:
        deferred.update(schema_name(schema) for schema in mcp_list if schema_name(schema))
    if not deferred:
        return None
    return ToolOffer(deferred=frozenset(deferred))


def can_deploy(mcp_bindings: Mapping[str, Any] | None) -> bool:
    """True when the Legion VPS MCP is among the agent's bindings."""
    from app.core.mcp_gateway import _VPS_MCP_MARKET_SLUG

    return any(
        getattr(binding, "market_slug", None) == _VPS_MCP_MARKET_SLUG
        for binding in (mcp_bindings or {}).values()
    )


async def task_hints(
    *,
    room_id: Any,
    task_id: Any,
    goal_id: Any,
    mcp_bindings: Mapping[str, Any] | None,
) -> TaskToolHints:
    """Collect :class:`TaskToolHints` for a task session (best effort)."""
    has_archive = False
    if task_id is not None or goal_id is not None:
        try:
            from app.core.plan_dialogue import plan_dialogue_ref

            has_archive = bool(await plan_dialogue_ref(room_id, task_id, goal_id))
        except Exception:  # noqa: BLE001 — a missing hint only defers a tool
            logger.debug("tool_offer: plan_dialogue_ref failed", exc_info=True)
    return TaskToolHints(can_deploy=can_deploy(mcp_bindings), has_dialogue_archive=has_archive)
