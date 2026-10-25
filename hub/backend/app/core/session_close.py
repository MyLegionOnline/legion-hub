"""
Unified session close contract (Step 2 / §8.2, §8.3).

Public API:
    SessionCloseActions      — root Pydantic model for all close actions
    filter_by_kind           — strip fields not allowed for the session's kind
    ANTHROPIC_CLOSE_TOOL     — tool definition for Anthropic SDK
    OPENAI_CLOSE_TOOL        — tool definition for OpenAI SDK
    extract_close_actions_from_tool_use   — parse Anthropic/OpenAI tool-call input
    extract_close_actions_from_json_envelope — parse raw JSON with retry-repair
    SESSION_CLOSE_ACTIONS_INVALID         — sentinel returned on parse failure
    strip_tool_call_markup   — drop leaked tool-call template markup from text
    clean_reply_text         — strip_tool_call_markup + logging for final replies
    resource_lock            — acquire a Postgres advisory transaction lock
    apply_close_actions      — apply filtered actions in one transaction
    flush_session_memory     — deterministic agent-memory flush, any close path (§4.1)
    maybe_refresh_room_memory — deterministic room-memory refresh on goal/task events (§4.2)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.models.session import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models (§8.2)
# ---------------------------------------------------------------------------


class MemoryPatch(BaseModel):
    """Full replacement for a memory block (project or agent)."""

    content: str


class PlanStepChange(BaseModel):
    step_id: UUID
    status: Optional[Literal["pending", "in_progress", "completed", "blocked"]] = None
    title: Optional[str] = None
    description: Optional[str] = None
    assigned_agent_id: Optional[UUID] = None


class PlanUpdate(BaseModel):
    plan_id: UUID
    status: Optional[Literal["draft", "approved"]] = None
    steps: list[PlanStepChange] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _drop_steps_with_invalid_ids(cls, values: Any) -> Any:
        """LLMs sometimes invent step ids like 'step-1' — drop those steps
        instead of failing the whole close-actions payload."""
        if not isinstance(values, dict) or not isinstance(values.get("steps"), list):
            return values
        kept: list[Any] = []
        for step in values["steps"]:
            sid = step.get("step_id") if isinstance(step, dict) else None
            if isinstance(sid, str):
                try:
                    UUID(sid)
                except ValueError:
                    logger.warning(
                        "session_close: dropping plan step with non-UUID step_id %r", sid
                    )
                    continue
            kept.append(step)
        values["steps"] = kept
        return values


class GoalCreate(BaseModel):
    title: str
    description: str = ""


class TaskCreate(BaseModel):
    ref: Optional[str] = None          # short name within the batch, e.g. "build-api"
    title: str
    description: str = ""
    goal_id: Optional[UUID] = None
    assigned_agent_id: Optional[UUID] = None
    depends_on: list[str] = Field(default_factory=list)  # ref-names or UUID strings
    # Generation checkpoint plan, stage 2: when set, the task is a human-decision
    # gate, not work for an agent. `checkpoint_kind` selects the interaction
    # (confirm | select_variant) and `checkpoint_payload` carries the options.
    # assigned_agent_id must be null for a checkpoint task.
    checkpoint_kind: Optional[Literal["confirm", "select_variant"]] = None
    checkpoint_payload: dict = Field(default_factory=dict)
    media_spec: Optional[dict] = None
    # internal design notes Stage 4: requirement codes (R-NNN)
    # this task claims to cover, from the goal it's created under. Strictly
    # optional — an empty list just leaves those requirements uncovered
    # (Stage 6's soft gate), it never blocks task creation.
    covers: list[str] = Field(default_factory=list)

    model_config = {"coerce_numbers_to_str": False}

    @model_validator(mode="before")
    @classmethod
    def _coerce_uuids(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        for key in ("goal_id", "assigned_agent_id"):
            v = values.get(key)
            if isinstance(v, str) and v.strip():
                try:
                    values[key] = UUID(v.strip())
                except ValueError:
                    values[key] = None
        # Coerce depends_on: accept a single string as a one-element list
        dep = values.get("depends_on")
        if isinstance(dep, str):
            values["depends_on"] = [dep] if dep.strip() else []
        # Same tolerance for covers (single code instead of a list)
        covers = values.get("covers")
        if isinstance(covers, str):
            values["covers"] = [covers] if covers.strip() else []
        return values

    @model_validator(mode="after")
    def _checkpoint_agent_exclusive(self) -> "TaskCreate":
        # A checkpoint task is a human-decision gate, not agent work — drop a
        # stray assigned_agent_id rather than silently creating an agent-bound
        # task that the dispatcher would skip via _NON_DISPATCHABLE_STATES.
        if self.checkpoint_kind is not None and self.assigned_agent_id is not None:
            self.assigned_agent_id = None
        return self


class TaskUpdate(BaseModel):
    task_id: UUID
    title: Optional[str] = None
    description: Optional[str] = None
    state: Optional[str] = None
    assigned_agent_id: Optional[UUID] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_state(cls, values: Any) -> Any:
        # LLMs frequently send the enum repr ("TaskState.COMPLETED") or an
        # upper-cased member name instead of the value ("completed").
        if isinstance(values, dict) and isinstance(values.get("state"), str):
            raw = values["state"].strip()
            if "." in raw:
                raw = raw.rsplit(".", 1)[-1]
            values = {**values, "state": raw.lower()}
        return values


class FileWrite(BaseModel):
    path: str
    content: str
    encoding: Literal["utf-8", "base64"] = "utf-8"

    @model_validator(mode="before")
    @classmethod
    def _default_null_encoding(cls, values: Any) -> Any:
        # LLMs frequently send an explicit null for optional fields
        if isinstance(values, dict) and values.get("encoding") in (None, ""):
            values = {**values, "encoding": "utf-8"}
        return values


# Common LLM field-name variations → canonical field names
_FIELD_ALIASES: dict[str, str] = {
    "tasks": "tasks_create",
    "new_tasks": "tasks_create",
    "create_tasks": "tasks_create",
    "update_tasks": "tasks_update",
    "cancel_tasks": "tasks_cancel",
    "goals": "goals_create",
    "new_goals": "goals_create",
    "create_goals": "goals_create",
    "files": "file_writes",
    "repo_files": "file_writes",
    "update_repo": "file_writes",
    "memory": "agent_memory_patch",
    "agent_memory": "agent_memory_patch",
    "room_memory": "room_memory_patch",
    "plan": "plan_update",
}


class SessionCloseActions(BaseModel):
    """All possible close actions.

    Per-kind permission filtering happens in filter_by_kind() — do not
    validate permissions here so the model can carry the full LLM response.
    """

    room_memory_patch: Optional[MemoryPatch] = None
    agent_memory_patch: Optional[MemoryPatch] = None
    goals_create: list[GoalCreate] = Field(default_factory=list)
    plan_update: Optional[PlanUpdate] = None
    tasks_create: list[TaskCreate] = Field(default_factory=list)
    tasks_update: list[TaskUpdate] = Field(default_factory=list)
    tasks_cancel: list[UUID] = Field(default_factory=list)
    file_writes: list[FileWrite] = Field(default_factory=list)
    summary: Optional[str] = None
    # Review ruling of a ChatMaster proxy side. Carried so the close row keeps
    # what was decided; the dispatch decision itself is taken from the raw tool
    # input (proxy_session.classify_review_verdict) before the session closes.
    verdict: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_and_normalize(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        # Coerce string memory patches to {"content": ...}
        for key in ("room_memory_patch", "agent_memory_patch"):
            v = values.get(key)
            if isinstance(v, str):
                values[key] = {"content": v}

        # Normalize common LLM field-name variations
        for alias, canonical in _FIELD_ALIASES.items():
            if alias in values and canonical not in values:
                values[canonical] = values.pop(alias)
            elif alias in values:
                values.pop(alias)

        return values


# ---------------------------------------------------------------------------
# Kind → allowed fields matrix (§8.2)
# ---------------------------------------------------------------------------

# List-typed fields whose "empty" value is []
_LIST_FIELDS: frozenset[str] = frozenset(
    {"goals_create", "tasks_create", "tasks_update", "tasks_cancel", "file_writes"}
)

_KIND_ALLOWED: dict[str, frozenset[str]] = {
    "user_agent_p1": frozenset({"agent_memory_patch", "summary"}),
    "user_agent_p2": frozenset({"agent_memory_patch", "summary"}),
    "user_cm_p1": frozenset(
        {
            "agent_memory_patch",
            "room_memory_patch",
            "plan_update",
            "tasks_create",
            "tasks_cancel",
            "summary",
        }
    ),
    "user_cm_p2": frozenset(
        {
            "agent_memory_patch",
            "room_memory_patch",
            "goals_create",
            "plan_update",
            "summary",
        }
    ),
    "cm_self_plan": frozenset({"tasks_create", "summary"}),
    # cm_agent_proxy is split by .side (values mirror CmAgentProxySession.side)
    "cm_agent_proxy_agent": frozenset(
        {"agent_memory_patch", "tasks_update", "file_writes", "summary"}
    ),
    "cm_agent_proxy_chat_master": frozenset(
        {"agent_memory_patch", "room_memory_patch", "plan_update", "summary", "verdict"}
    ),
    # chatflow_session_lifecycle_repair_plan stage 5.4: a run_subtask child
    # never receives its own close actions — those are always applied to the
    # parent session — but the matrix entry keeps filter_by_kind/flush_session_memory
    # well-defined if this row is ever closed directly instead of only through
    # RunSubtaskTool's own bookkeeping.
    "agent_subtask": frozenset({"agent_memory_patch", "summary"}),
}


def _matrix_key(session: "Session") -> str:
    """Return the _KIND_ALLOWED key for a session, splitting cm_agent_proxy by side."""
    if session.kind == "cm_agent_proxy":
        return f"cm_agent_proxy_{session.side}"
    return session.kind


def filter_by_kind(
    actions: SessionCloseActions, session: "Session"
) -> tuple[SessionCloseActions, list[str]]:
    """Return a copy of *actions* with disallowed fields cleared, plus a list of dropped field names.

    Disallowed non-empty fields are logged at WARNING level and included in the returned list
    so callers can decide whether to retry or emit a notification.
    """
    allowed = _KIND_ALLOWED.get(_matrix_key(session), frozenset())
    data = actions.model_dump()
    dropped: list[str] = []

    for field, value in list(data.items()):
        if field in allowed:
            continue
        is_empty = value is None or (field in _LIST_FIELDS and value == [])
        if not is_empty:
            logger.warning(
                "session_close: dropping disallowed field '%s' for kind '%s' (session=%s)",
                field,
                session.kind,
                session.id,
            )
            dropped.append(field)
            data[field] = [] if field in _LIST_FIELDS else None

    return SessionCloseActions.model_validate(data), dropped


# ---------------------------------------------------------------------------
# Tool definitions (Anthropic / OpenAI)
# ---------------------------------------------------------------------------

_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "room_memory_patch": {
            "type": "object",
            "description": "Full replacement content for the project/room memory.",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
        "agent_memory_patch": {
            "type": "object",
            "description": "Full replacement content for this agent's memory in the room.",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
        "goals_create": {
            "type": "array",
            "description": "New goals to add to the room.",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title"],
            },
        },
        "plan_update": {
            "type": "object",
            "description": "Changes to a goal plan.",
            "properties": {
                "plan_id": {"type": "string", "format": "uuid"},
                "status": {"type": "string", "enum": ["draft", "approved"]},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step_id": {"type": "string", "format": "uuid"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "blocked"],
                            },
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "assigned_agent_id": {"type": "string", "format": "uuid"},
                        },
                        "required": ["step_id"],
                    },
                },
            },
            "required": ["plan_id"],
        },
        "tasks_create": {
            "type": "array",
            "description": "New tasks to create.",
            "items": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": (
                            "Unique short name for this task within the batch "
                            "(e.g. 'build-api'). Required if other tasks reference "
                            "this task in their depends_on."
                        ),
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "goal_id": {"type": "string", "format": "uuid"},
                    "assigned_agent_id": {"type": "string", "format": "uuid"},
                    "depends_on": {
                        "type": "array",
                        "description": (
                            "Refs of tasks that must complete before this one starts. "
                            "Use the ref value of tasks in this same batch, or the UUID "
                            "string of an already-existing task."
                        ),
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "checkpoint_kind": {
                        "type": "string",
                        "enum": ["confirm", "select_variant"],
                        "description": (
                            "When set, this task is a human-decision gate, not agent "
                            "work. 'confirm' = yes/no approval; 'select_variant' = pick "
                            "ONE of several outputs produced by a depended-on task. "
                            "assigned_agent_id must be null when this is set. Only use "
                            "this when the plan itself names an explicit approval/choice "
                            "point — never insert a checkpoint by default."
                        ),
                    },
                    "checkpoint_payload": {
                        "type": "object",
                        "description": (
                            "Options for the checkpoint. {prompt: str} for confirm; "
                            "{prompt: str, variants: [{id, label, paths: [str], note?}]} "
                            "for select_variant. `paths` reference files produced by the "
                            "depended-on task."
                        ),
                        "default": {},
                    },
                    "media_spec": {
                        "type": "object",
                        "description": (
                            "Structured parameters for image-generation tasks. "
                            "references: list of {path, role, note?} — workspace-relative "
                            "paths to files (logos, photos, layouts, portraits) that must "
                            "guide generation; role is one of logo|style|subject|layout. "
                            "size: {size: 'WxH'} for exact pixels (e.g. '1000x100') or "
                            "{aspect_ratio: '16:9'} when exact dimensions are not critical. "
                            "Only set on tasks assigned to [image_generation] agents."
                        ),
                    },
                    "covers": {
                        "type": "array",
                        "description": (
                            "Requirement codes (R-NNN/NG-NNN/Q-NNN) from the plan's "
                            "'## Requirements' section that this task satisfies. Optional — "
                            "omit when the task doesn't map to a listed requirement."
                        ),
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["title"],
            },
        },
        "tasks_update": {
            "type": "array",
            "description": "Updates to existing tasks (agent side of proxy session).",
            "items": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "format": "uuid"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "state": {"type": "string"},
                    "assigned_agent_id": {"type": "string", "format": "uuid"},
                },
                "required": ["task_id"],
            },
        },
        "tasks_cancel": {
            "type": "array",
            "description": "UUIDs of tasks to cancel.",
            "items": {"type": "string", "format": "uuid"},
        },
        "file_writes": {
            "type": "array",
            "description": "Files to write/update in the repository.",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "encoding": {"type": "string", "enum": ["utf-8", "base64"]},
                },
                "required": ["path", "content"],
            },
        },
        "summary": {
            "type": "string",
            "description": "Human-readable summary of what was accomplished.",
        },
        "verdict": {
            "type": "string",
            "enum": ["approve", "request_changes", "reject"],
            "description": (
                "ChatMaster reviewing a submitted task: the ruling on the work. "
                "Set it on every review close — it is read literally, while the "
                "summary is prose that may quote a previous verdict. Ignored for "
                "every other kind of session."
            ),
        },
    },
}

_TOOL_DESCRIPTION = (
    "Signal that the current work session is complete and record all actions "
    "to apply: memory updates, plan changes, task creation, file writes, etc."
)

# Anthropic SDK format — full schema (used by proxy/agent sessions and for backward compat)
ANTHROPIC_CLOSE_TOOL: dict[str, Any] = {
    "name": "close_session",
    "description": _TOOL_DESCRIPTION,
    "input_schema": _TOOL_INPUT_SCHEMA,
}

# OpenAI SDK format — full schema (used by proxy/agent sessions and for backward compat)
OPENAI_CLOSE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "close_session",
        "description": _TOOL_DESCRIPTION,
        "parameters": _TOOL_INPUT_SCHEMA,
    },
}


def close_tool_for_kind(matrix_key: str, fmt: Literal["anthropic", "openai"]) -> dict[str, Any]:
    """Return a close_session tool schema restricted to fields allowed for *matrix_key*.

    LLMs can only request actions listed in the schema — this prevents user_cm_*
    sessions from even attempting file_writes, which is not in their allowed set.
    Falls back to the full schema for unknown matrix_key values.
    """
    allowed = _KIND_ALLOWED.get(matrix_key, frozenset())
    if not allowed:
        # Unknown kind — return full schema so nothing breaks silently
        return ANTHROPIC_CLOSE_TOOL if fmt == "anthropic" else OPENAI_CLOSE_TOOL
    props = {k: v for k, v in _TOOL_INPUT_SCHEMA["properties"].items() if k in allowed}
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if fmt == "anthropic":
        return {
            "name": "close_session",
            "description": _TOOL_DESCRIPTION,
            "input_schema": schema,
        }
    return {
        "type": "function",
        "function": {
            "name": "close_session",
            "description": _TOOL_DESCRIPTION,
            "parameters": schema,
        },
    }


def _keys_for_field(field: str) -> set[str]:
    """Canonical field name plus every alias that normalizes to it."""
    return {field} | {a for a, c in _FIELD_ALIASES.items() if c == field}


def extract_close_actions_from_tool_use(
    tool_call_input: dict[str, Any],
) -> SessionCloseActions:
    """Parse close actions from an Anthropic/OpenAI tool-call input dict.

    One malformed field must not discard the whole payload (a bad
    file_writes entry used to silently drop goals_create — no goal, no
    plan, no tasks).  On validation failure the offending top-level
    fields are pruned and validation retried; only if that also fails
    does the original pydantic.ValidationError propagate.
    """
    from pydantic import ValidationError

    try:
        return SessionCloseActions.model_validate(tool_call_input)
    except ValidationError as exc:
        if not isinstance(tool_call_input, dict):
            raise
        bad_fields = {
            err["loc"][0]
            for err in exc.errors()
            if err["loc"] and isinstance(err["loc"][0], str)
        }
        if not bad_fields or not bad_fields <= set(SessionCloseActions.model_fields):
            raise
        drop_keys = {key for f in bad_fields for key in _keys_for_field(f)}
        pruned = {k: v for k, v in tool_call_input.items() if k not in drop_keys}
        if not pruned:
            # Nothing salvageable — let the caller report the failure
            raise
        logger.warning(
            "session_close: dropping invalid close_session fields %s and retrying: %s",
            sorted(bad_fields),
            exc,
        )
        try:
            return SessionCloseActions.model_validate(pruned)
        except ValidationError:
            raise exc


# ---------------------------------------------------------------------------
# JSON-envelope adapter (deepseek / groq / ollama)
# ---------------------------------------------------------------------------

SESSION_CLOSE_ACTIONS_INVALID = "SESSION_CLOSE_ACTIONS_INVALID"

_REPAIR_HINT = (
    "The previous response could not be parsed as SessionCloseActions. "
    "Return ONLY a valid JSON object matching the schema. "
    "Do not include markdown fences or any text outside the JSON."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


class ToolArgumentsParseError(Exception):
    """Tool-call arguments returned by the LLM are not valid JSON."""


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse tool-call arguments, tolerating markdown fences around the JSON.

    Raises ToolArgumentsParseError (with a logged snippet around the failure
    position) so callers can retry the LLM call instead of failing the session.
    """
    raw = raw or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        snippet = raw[max(0, exc.pos - 150) : exc.pos + 150]
        logger.warning(
            "session_close: malformed tool-call arguments (%s); around failure: %r",
            exc,
            snippet,
        )
        raise ToolArgumentsParseError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Raw tool-call template markup leaking into plain text
# ---------------------------------------------------------------------------

# A model that still wants to call a tool, but whose request carried no tool
# schema for the provider's parser to latch onto, emits its native tool-call
# template as ordinary content.  Publishing that blob into a room is always
# wrong, so final replies are stripped of it.
_TOOL_MARKUP_BLOCK_RES: tuple[re.Pattern[str], ...] = (
    # DeepSeek DSML: <｜DSML｜tool_calls> … </｜DSML｜tool_calls>
    re.compile(r"<[|｜]DSML[|｜]tool_calls\b[^>]*>.*?</[|｜]DSML[|｜]tool_calls>", re.DOTALL),
    # DeepSeek V3: <｜tool▁calls▁begin｜> … <｜tool▁calls▁end｜>
    re.compile(
        r"<[|｜]tool[_▁]calls[_▁]begin[|｜]>.*?<[|｜]tool[_▁]calls[_▁]end[|｜]>", re.DOTALL
    ),
    # Qwen / Hermes: <tool_call> … </tool_call>
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    # XML-style function calling: <function_calls> … </function_calls>
    re.compile(r"<function_calls>.*?</function_calls>", re.DOTALL),
)

# An unterminated leak always sits at the tail of the reply (the model was cut
# off mid-call), so drop everything from the first orphan marker onward rather
# than leaving half a tool call in the text.  Only unambiguous special-token
# markers qualify: this rule truncates, so a false positive costs a whole
# answer.  A bare <invoke>/<parameter> never appears outside a wrapper that the
# block rules above already matched.
_TOOL_MARKUP_TAIL_RE = re.compile(
    r"</?[|｜](?:DSML[|｜])?[^>]*>"
    r"|</?(?:tool_call|function_calls)\b[^>]*>"
)


def strip_tool_call_markup(text: str) -> tuple[str, bool]:
    """Strip raw tool-call template markup that leaked into plain content.

    Returns ``(cleaned_text, found_markup)``.  ``found_markup`` lets callers
    distinguish "the model wrote a normal short reply" from "the model tried to
    call a tool in prose" — the latter is worth a retry or an explicit error,
    never a verbatim blob in the room.
    """
    if not text:
        return "", False
    cleaned = text
    for pattern in _TOOL_MARKUP_BLOCK_RES:
        cleaned = pattern.sub("", cleaned)
    tail = _TOOL_MARKUP_TAIL_RE.search(cleaned)
    if tail is not None:
        cleaned = cleaned[: tail.start()]
    cleaned = cleaned.strip()
    return cleaned, cleaned != text.strip()


TOOL_MARKUP_FALLBACK_TEXT = (
    "Could not produce a final reply: the model returned service markup "
    "of a tool call instead of text. Repeat the request."
)


def clean_reply_text(text: str | None, *, where: str, session_id: Any = None) -> str:
    """Final-reply guard: never publish leaked tool-call template markup.

    Returns the cleaned text, or a user-facing error string when the reply was
    nothing but markup.
    """
    cleaned, leaked = strip_tool_call_markup(text or "")
    if leaked:
        logger.warning(
            "session_close: stripped raw tool-call markup from %s reply "
            "(session %s) — %d chars left",
            where, session_id, len(cleaned),
        )
        if not cleaned:
            return TOOL_MARKUP_FALLBACK_TEXT
    return cleaned


def _try_parse(raw: str) -> SessionCloseActions | None:
    try:
        data = json.loads(_strip_fences(raw))
        return extract_close_actions_from_tool_use(data)
    except Exception:
        return None


async def extract_close_actions_from_json_envelope(
    raw_response: str,
    repair_fn: "Callable[[str, str], Awaitable[str]] | None" = None,
) -> "SessionCloseActions | str":
    """Parse close actions from a raw LLM JSON string.

    On parse failure, calls *repair_fn(broken_text, hint)* up to 2 times.
    If all attempts fail, returns the sentinel SESSION_CLOSE_ACTIONS_INVALID.

    Args:
        raw_response: The raw string from the LLM response.
        repair_fn:    Async callable(broken_text, hint) → repaired_text.
                      If None, failures immediately return the sentinel.
    """
    result = _try_parse(raw_response)
    if result is not None:
        return result

    if repair_fn is None:
        logger.warning("session_close: JSON parse failed, no repair_fn — flagging invalid")
        return SESSION_CLOSE_ACTIONS_INVALID

    current = raw_response
    for attempt in range(1, 3):  # attempts 1 and 2
        try:
            current = await repair_fn(current, _REPAIR_HINT)
        except Exception as exc:
            logger.warning("session_close: repair_fn raised on attempt %d: %s", attempt, exc)
            break
        result = _try_parse(current)
        if result is not None:
            logger.info("session_close: JSON repaired on attempt %d", attempt)
            return result
        logger.warning(
            "session_close: JSON still invalid after repair attempt %d", attempt
        )

    logger.error("session_close: all parse/repair attempts exhausted — returning INVALID sentinel")
    from app.core.metrics import record_session_close_invalid
    record_session_close_invalid(kind="unknown")
    return SESSION_CLOSE_ACTIONS_INVALID


# ---------------------------------------------------------------------------
# Postgres advisory lock
# ---------------------------------------------------------------------------

_LOCK_NS = 0xC4A75E55  # namespace for session-close advisory locks


def _advisory_key(resource_id: UUID) -> int:
    """Derive a signed int64 advisory-lock key from a UUID.

    Strategy: blake2b(uuid_bytes, digest_size=8) XOR namespace → int64.
    """
    raw = int.from_bytes(
        hashlib.blake2b(resource_id.bytes, digest_size=8).digest(), "big"
    )
    key = raw ^ _LOCK_NS
    # Postgres bigint is signed — wrap if necessary
    if key >= (1 << 63):
        key -= 1 << 64
    return key


async def resource_lock(db: AsyncSession, resource_id: UUID) -> None:
    """Acquire a Postgres transaction-scoped advisory lock for *resource_id*.

    The lock is released automatically when the surrounding transaction ends.
    Raises sqlalchemy.exc.OperationalError (pgcode 55P03) if lock_timeout fires.
    """
    key = _advisory_key(resource_id)
    await db.execute(text(f"SELECT pg_advisory_xact_lock({key})"))
    # Stage 1.2: release_db() must not commit this transaction before a long
    # await — that would drop the lock silently.
    from app.db.release import mark_xact_lock

    mark_xact_lock(db)


# ---------------------------------------------------------------------------
# apply_close_actions
# ---------------------------------------------------------------------------

_STATEMENT_TIMEOUT_MS = 30_000   # 30 s for the whole apply
_LOCK_TIMEOUT_MS = 5_000         # 5 s to acquire any advisory lock


def _is_lock_timeout(exc: Exception) -> bool:
    orig = getattr(exc, "orig", None)
    pgcode = getattr(orig, "pgcode", "") or ""
    if pgcode == "55P03":  # lock_not_available
        return True
    msg = str(exc).lower()
    return "lock" in msg and ("not available" in msg or "timeout" in msg)


async def _refetch_session(db: AsyncSession, session_id: UUID) -> "Session | None":
    from sqlalchemy import select
    from sqlalchemy.orm import selectin_polymorphic

    from app.models.session import SESSION_SUBTYPES, Session

    result = await db.execute(
        select(Session)
        .where(Session.id == session_id)
        .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
    )
    return result.scalars().first()


async def _do_apply(
    session: "Session",
    actions: SessionCloseActions,
    db: AsyncSession,
) -> dict[str, Any]:
    """Execute all permitted changes in the current transaction.

    Returns an outcome dict for post-commit side effects:
      file_changes     — list of {path, created, [version]} for FILE_CHANGED events
                         (workspace mode: path-only, no file_id; legacy: +version)
      touched_goal_ids — goals whose tasks changed state (candidates for auto-completion)
    """
    import base64 as b64
    from datetime import datetime

    from sqlalchemy import select

    from app.core.storage import DatabaseBackend, get_storage_backend
    from app.models.agent_memory import AgentMemory
    from app.models.goal import GoalStatus, RoomGoal
    from app.models.goal_plan import GoalPlan, GoalPlanStatus, GoalPlanStep, PlanStepStatus
    from app.models.project_memory import ProjectMemory
    from app.core.workspace_fs import normalize_ws_path
    from app.models.task import (
        TASK_FAILED_STATES, TASK_TERMINAL_STATES, Task, TaskState, is_checkpoint_approval,
    )

    now = datetime.utcnow()
    outcome: dict[str, Any] = {
        "file_changes": [],
        "touched_goal_ids": set(),
        "created_tasks": [],     # list[dict] populated after flush
        "reopened_goals": {},    # UUID -> str title
        "terminal_task_ids": set(),  # tasks that just reached a terminal state
        "failed_task_ids": set(),    # tasks that just reached a failed/cancelled state
    }

    # Set per-transaction timeouts
    await db.execute(text(f"SET LOCAL lock_timeout = {_LOCK_TIMEOUT_MS}"))
    await db.execute(text(f"SET LOCAL statement_timeout = {_STATEMENT_TIMEOUT_MS}"))

    # Acquire advisory lock on this room's project memory (always)
    await resource_lock(db, session.room_id)

    # Acquire advisory lock on the plan if we're about to update it
    plan_id: UUID | None = getattr(session, "plan_id", None)
    if plan_id and actions.plan_update:
        await resource_lock(db, plan_id)

    # 1. Room memory patch
    if actions.room_memory_patch:
        row = (
            await db.execute(
                select(ProjectMemory).where(ProjectMemory.room_id == session.room_id)
            )
        ).scalars().first()
        if row:
            row.content = actions.room_memory_patch.content
            row.updated_at = now
            db.add(row)

    # 2. Agent memory patch
    if actions.agent_memory_patch:
        agent_id: UUID | None = (
            getattr(session, "agent_id", None)
            or getattr(session, "chat_master_agent_id", None)
        )
        if agent_id:
            row = (
                await db.execute(
                    select(AgentMemory).where(
                        AgentMemory.room_id == session.room_id,
                        AgentMemory.agent_id == agent_id,
                    )
                )
            ).scalars().first()
            if row is None:
                row = AgentMemory(
                    room_id=session.room_id,
                    agent_id=agent_id,
                    content=actions.agent_memory_patch.content,
                )
            else:
                row.content = actions.agent_memory_patch.content
                row.updated_at = now
            db.add(row)

    # 3. Create goals
    for spec in actions.goals_create:
        db.add(RoomGoal(room_id=session.room_id, title=spec.title, description=spec.description))

    # 4. Plan update
    if actions.plan_update:
        upd = actions.plan_update
        gp = (
            await db.execute(select(GoalPlan).where(GoalPlan.id == upd.plan_id))
        ).scalars().first()
        if gp:
            if upd.status:
                gp.status = GoalPlanStatus(upd.status)
                if upd.status == "approved":
                    gp.approved_at = now
            gp.updated_at = now
            db.add(gp)

            for sc in upd.steps:
                step = (
                    await db.execute(
                        select(GoalPlanStep).where(GoalPlanStep.id == sc.step_id)
                    )
                ).scalars().first()
                if step:
                    if sc.status:
                        step.status = PlanStepStatus(sc.status)
                    if sc.title is not None:
                        step.title = sc.title
                    if sc.description is not None:
                        step.description = sc.description
                    if sc.assigned_agent_id is not None:
                        step.assigned_agent_id = sc.assigned_agent_id
                    step.updated_at = now
                    db.add(step)

    # 5. Create tasks
    _new_tasks: list[Task] = []
    if actions.tasks_create:
        session_goal_id: UUID | None = getattr(session, "goal_id", None)
        # LLM-supplied goal_id can be hallucinated (e.g. the nil UUID) — only
        # accept ids that actually exist in this room, else fall back to the
        # session's goal.
        valid_goal_ids: set[UUID] = set(
            (
                await db.execute(
                    select(RoomGoal.id).where(RoomGoal.room_id == session.room_id)
                )
            ).scalars().all()
        )
        fallback_goal_id = session_goal_id if session_goal_id in valid_goal_ids else None
        _reopened_in_apply: set[UUID] = set()  # avoid duplicate reopen per goal
        _task_triples: list[tuple[str | None, list[str], Task]] = []
        for spec in actions.tasks_create:
            goal_id = spec.goal_id
            if goal_id not in valid_goal_ids:
                if goal_id is not None:
                    logger.warning(
                        "apply_close_actions: task %r references goal_id %s not found "
                        "in room %s — using session goal %s instead",
                        spec.title,
                        goal_id,
                        session.room_id,
                        fallback_goal_id,
                    )
                goal_id = fallback_goal_id
            # Reopen COMPLETED goal when new task is assigned to it
            if goal_id and goal_id not in _reopened_in_apply:
                _goal_row = (
                    await db.execute(select(RoomGoal).where(RoomGoal.id == goal_id))
                ).scalars().first()
                if _goal_row is not None and _goal_row.status == GoalStatus.COMPLETED:
                    _goal_row.status = GoalStatus.ACTIVE
                    _goal_row.completed_at = None
                    _goal_row.updated_at = now
                    db.add(_goal_row)
                    outcome["reopened_goals"][goal_id] = _goal_row.title
                _reopened_in_apply.add(goal_id)
            task = Task(
                room_id=session.room_id,
                goal_id=goal_id,
                title=spec.title,
                description=spec.description,
                assigned_agent_id=spec.assigned_agent_id,
                # internal design notes Stage 6: bind the task to the
                # formation dialogue so archive_for_task / plan_dialogue can
                # find it later (mirrors direct-session tasks).
                origin_session_id=getattr(session, "id", None),
                # Generation checkpoint plan, stage 2: a checkpoint task is a
                # human-decision gate, not agent work — start it in
                # WAITING_APPROVAL so the dispatcher skips it (it is already in
                # _NON_DISPATCHABLE_STATES) until the human resolves it via the
                # checkpoint/resolve endpoint (stage 3). On APPROVED,
                # try_unblock_tasks unblocks dependents normally because
                # APPROVED is already a terminal state.
                state=(
                    TaskState.WAITING_APPROVAL
                    if spec.checkpoint_kind is not None
                    else TaskState.ASSIGNED
                ),
                checkpoint_kind=spec.checkpoint_kind,
                checkpoint_payload=spec.checkpoint_payload,
                media_spec=spec.media_spec or {},
            )
            db.add(task)
            _task_triples.append((spec.ref, spec.depends_on, task))

        from app.core.task_deps import TaskDependencyCycleError, create_tasks_with_deps
        try:
            _new_tasks = await create_tasks_with_deps(_task_triples, db)
        except TaskDependencyCycleError as exc:
            logger.error(
                "apply_close_actions: dependency cycle detected in task batch for room %s: %s",
                session.room_id,
                exc,
            )
            # Propagate so the transaction rolls back; caller emits TASK_DEPS_CYCLE event
            raise

        # internal design notes Stage 4: link each task to the
        # requirement codes it claims to cover, then mark those requirements
        # covered. An unknown code is dropped with a log line — a garbled
        # `covers` must not fail the decomposition. `zip` is safe here because
        # create_tasks_with_deps returns tasks in the same order it received
        # them (task_deps.py's own contract).
        _covers_by_goal: dict[UUID, dict[str, list[Task]]] = {}
        for spec, task in zip(actions.tasks_create, _new_tasks):
            if not spec.covers or not task.goal_id:
                continue
            goal_codes = _covers_by_goal.setdefault(task.goal_id, {})
            for code in dict.fromkeys(spec.covers):  # de-dupe, preserve order
                code = str(code or "").strip()
                if code:
                    tasks_for_code = goal_codes.setdefault(code, [])
                    if task not in tasks_for_code:
                        tasks_for_code.append(task)

        if _covers_by_goal:
            from app.models.requirement import GoalRequirement, RequirementStatus, TaskRequirement

            for goal_id, code_map in _covers_by_goal.items():
                req_rows = (
                    await db.execute(
                        select(GoalRequirement).where(
                            GoalRequirement.goal_id == goal_id,
                            GoalRequirement.code.in_(list(code_map.keys())),
                        )
                    )
                ).scalars().all()
                found_codes = {r.code for r in req_rows}
                for missing in set(code_map.keys()) - found_codes:
                    logger.warning(
                        "apply_close_actions: task covers unknown requirement code %r "
                        "for goal %s — ignoring",
                        missing, goal_id,
                    )
                for req in req_rows:
                    for covering_task in code_map[req.code]:
                        db.add(TaskRequirement(task_id=covering_task.id, requirement_id=req.id))
                    if req.status not in (RequirementStatus.CONFLICT, RequirementStatus.CLOSED):
                        req.status = RequirementStatus.COVERED
                        req.updated_at = now
                        db.add(req)

        # internal design notes Stage 5: append the deterministic
        # requirements block to every new task of a goal — not just the ones
        # with `covers`, since non-goals are shared by the whole goal. Needs a
        # flush first so render_task_requirements_block's TaskRequirement
        # query sees the links just added above.
        if any(t.goal_id for t in _new_tasks):
            from app.core.requirements import render_task_requirements_block

            await db.flush()
            for task in _new_tasks:
                if not task.goal_id:
                    continue
                block = await render_task_requirements_block(task, db)
                if block:
                    task.description = (task.description or "") + block
                    db.add(task)

    # 6. Update tasks
    for upd in actions.tasks_update:
        task = (
            await db.execute(select(Task).where(Task.id == upd.task_id))
        ).scalars().first()
        if task:
            if upd.title is not None:
                task.title = upd.title
            if upd.description is not None:
                task.description = upd.description
            if upd.state is not None:
                try:
                    new_state = TaskState(upd.state)
                except ValueError:
                    logger.warning(
                        "apply_close_actions: invalid task state %r for task %s — skipping",
                        upd.state, upd.task_id,
                    )
                    new_state = None
            else:
                new_state = None
            if new_state is not None and is_checkpoint_approval(task, new_state):
                # ChatMaster relaying the user's "ok" on a checkpoint finishes
                # the gate, same as the chat card does.
                new_state = TaskState.COMPLETED
                task.checkpoint_choice = task.checkpoint_choice or {
                    "approved": True,
                    "resolved_at": now.isoformat(),
                }
            if new_state is not None:
                task.state = new_state
                if task.goal_id:
                    outcome["touched_goal_ids"].add(task.goal_id)
                if new_state in TASK_TERMINAL_STATES:
                    outcome["terminal_task_ids"].add(task.id)
                elif new_state in TASK_FAILED_STATES:
                    outcome["failed_task_ids"].add(task.id)
            if upd.assigned_agent_id is not None:
                task.assigned_agent_id = upd.assigned_agent_id
            task.updated_at = now
            db.add(task)

    # 7. Cancel tasks
    for task_id in actions.tasks_cancel:
        task = (
            await db.execute(select(Task).where(Task.id == task_id))
        ).scalars().first()
        if task:
            task.state = TaskState.CANCELLED
            task.updated_at = now
            db.add(task)
            if task.goal_id:
                outcome["touched_goal_ids"].add(task.goal_id)
            outcome["failed_task_ids"].add(task.id)

    # 8. File writes → workspace volume
    #    Stage 6: the volume is the single source of truth. Writes go through
    #    the facade post-commit (apply_close_actions); no DB rows here.
    if actions.file_writes:
        for fw in actions.file_writes:
            path = normalize_ws_path(fw.path)
            outcome["file_changes"].append(
                {"path": path, "created": True}
            )

    # 9. Summary
    if actions.summary:
        session.summary = actions.summary
        db.add(session)

    await db.flush()

    # Populate created_tasks: IDs are now assigned; look up plan_id for each goal
    if _new_tasks:
        _task_goal_ids = {t.goal_id for t in _new_tasks if t.goal_id}
        _plan_by_goal: dict[UUID, UUID] = {}
        for _gid in _task_goal_ids:
            _plan_row = (
                await db.execute(
                    select(GoalPlan.id).where(GoalPlan.goal_id == _gid).limit(1)
                )
            ).first()
            if _plan_row:
                _plan_by_goal[_gid] = _plan_row[0]
        for _t in _new_tasks:
            outcome["created_tasks"].append({
                "task_id": _t.id,
                "title": _t.title,
                "goal_id": _t.goal_id,
                "assigned_agent_id": _t.assigned_agent_id,
                "plan_id": _plan_by_goal.get(_t.goal_id) if _t.goal_id else None,
                "dispatch_status": _t.dispatch_status,
                "checkpoint_kind": _t.checkpoint_kind,
            })

    return outcome


async def apply_close_actions(
    session: "Session",
    actions: SessionCloseActions,
    db: AsyncSession,
) -> None:
    """Apply filtered close actions inside a single Postgres transaction.

    Flow:
      1. filter_by_kind strips disallowed fields.
      2. Attempt _do_apply (sets statement_timeout=30s, lock_timeout=5s,
         acquires advisory locks on room_id and plan_id).
      3. On lock-timeout: rollback, sleep 1 s, retry once.
      4. On second lock-timeout: rollback, save session.pending_actions,
         emit SESSION_CLOSE_LOCK_TIMEOUT event.
    """
    from app.core import workspace_fs as wfs
    from app.core.events import emit_event
    from app.models.task import Task

    filtered, _ = filter_by_kind(actions, session)

    # Capture identity fields before any potential rollback expires the object
    session_id = session.id
    room_id = session.room_id
    kind = session.kind

    for attempt in range(2):
        try:
            outcome = await _do_apply(session, filtered, db)

            # Write agent file_writes to the workspace volume (post-commit).
            # The volume is the primary truth; the write auto-commits to git
            # via workspaced. Errors are logged but do not interrupt the
            # other side effects (events, task dispatch), since the DB
            # transaction has already committed at this point.
            if filtered.file_writes:
                import base64 as _b64
                from app.core import workspaced_client as _wsd

                _agent_id = getattr(session, "agent_id", None)
                _source = f"agent:{_agent_id}" if _agent_id else "agent"
                for fw in filtered.file_writes:
                    try:
                        content_ws = fw.content
                        if fw.encoding == "base64":
                            try:
                                content_ws = _b64.b64decode(fw.content).decode("utf-8")
                            except Exception:
                                pass
                        _norm = wfs.normalize_ws_path(fw.path)
                        await _wsd.write_workspace_file(
                            str(room_id),
                            _norm,
                            content_ws,
                            commit_message=f"{_source}: write {_norm}",
                        )
                    except Exception as _e:
                        logger.warning(
                            "session_close: workspace write failed for %s: %s",
                            fw.path, _e,
                        )

            # Post-apply side effects (each emit_event commits the transaction,
            # so they must run only after _do_apply has fully succeeded).
            for fc in outcome["file_changes"]:
                await emit_event(
                    room_id=room_id,
                    event_type="FILE_CHANGED",
                    content=f"File {fc['path']} {'created' if fc['created'] else 'updated'}",
                    db=db,
                    metadata={**fc, "session_id": str(session_id)},
                )
            # Notify about goals that were reopened by new task creation
            for _goal_id, _goal_title in outcome["reopened_goals"].items():
                await emit_event(
                    room_id=room_id,
                    event_type="GOAL_REOPENED",
                    content=f"Goal '{_goal_title}' reopened — new tasks added.",
                    db=db,
                    metadata={
                        "goal_id": str(_goal_id),
                        "goal_title": _goal_title,
                        "session_id": str(session_id),
                    },
                )
            # internal design notes Stage 6: a user_cm_p1 dialogue just
            # created tasks in an existing Goal — these are the "rework it like
            # this, here's the new config" dialogues, the most frequent source
            # of executor drift. user_cm_p2 closes with goal-less tasks, which
            # have no goal archive to fall back on at all. Archive the
            # transcript per task. Runs BEFORE the dispatch below: the
            # executor's system prompt links "Spec-agreement dialogue" only if
            # the archive file already exists when its session starts.
            # Best-effort: the DB transaction has already committed.
            if kind in ("user_cm_p1", "user_cm_p2") and outcome["created_tasks"]:
                try:
                    from app.core import plan_dialogue
                    from app.core.room_coordinator import resolve_planner_llm_config
                    from app.models.room import Room as _Room

                    _room = await db.get(_Room, room_id)
                    try:
                        _cm_agent, _llm_cfg = (
                            await resolve_planner_llm_config(_room, db)
                            if _room is not None else (None, {})
                        )
                        _llm_cfg = _llm_cfg or None
                    except Exception:
                        _llm_cfg = None
                    for _tc in outcome["created_tasks"]:
                        await plan_dialogue.archive_for_task(
                            room_id,
                            _tc["task_id"],
                            session,
                            db,
                            goal_id=_tc.get("goal_id"),
                            task_title=_tc.get("title") or "",
                            llm_config=_llm_cfg,
                        )
                    await plan_dialogue.clear_draft_heads(
                        room_id, getattr(session, "user_id", None)
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "session_close: plan-dialogue task archive failed: %s", exc
                    )
            # Emit TASK_CREATED and dispatch proxy sessions for each created task
            for _tc in outcome["created_tasks"]:
                await emit_event(
                    room_id=room_id,
                    event_type="TASK_CREATED",
                    content=f"Task '{_tc['title']}' created (id={_tc['task_id']})",
                    db=db,
                    metadata={
                        "task_id": str(_tc["task_id"]),
                        "title": _tc["title"],
                        # ChatWindow reads task_title; keep "title" for old consumers.
                        "task_title": _tc["title"],
                        "goal_id": str(_tc["goal_id"]) if _tc["goal_id"] else None,
                        "assigned_agent_id": (
                            str(_tc["assigned_agent_id"]) if _tc["assigned_agent_id"] else None
                        ),
                        "session_id": str(session_id),
                    },
                )
                if (
                    _tc["assigned_agent_id"]
                    and _tc.get("dispatch_status", "ready") == "ready"
                ):
                    from app.core.proxy_session import (
                        TaskDispatchPreflightError,
                        request_task_dispatch,
                    )

                    dispatch_task = await db.get(Task, _tc["task_id"])
                    if dispatch_task is not None:
                        try:
                            dispatch_outcome = await request_task_dispatch(
                                dispatch_task, room_id, db,
                                source="session_close_created_task",
                            )
                        except TaskDispatchPreflightError as exc:
                            logger.info(
                                "session_close: task %s not dispatched: %s",
                                dispatch_task.id,
                                exc.code,
                            )
                        else:
                            logger.info(
                                "session_close: dispatching newly created task %s "
                                "(outcome=%s, closing_session=%s)",
                                dispatch_task.id,
                                dispatch_outcome,
                                session_id,
                            )
                elif (
                    _tc.get("checkpoint_kind")
                    and _tc.get("dispatch_status", "ready") == "ready"
                ):
                    # Generation checkpoint plan, stage 3: a checkpoint born
                    # READY has no predecessor whose unblock would publish it —
                    # publish the request right away. Checkpoints with deps are
                    # published by publish_unblocked_tasks when the last
                    # predecessor finishes (the variant files exist by then).
                    from app.core.task_deps import emit_checkpoint_request

                    _cp_task = await db.get(Task, _tc["task_id"])
                    if _cp_task is not None:
                        await emit_checkpoint_request(
                            room_id, _cp_task, db, session_id=session_id,
                        )

            if outcome["touched_goal_ids"]:
                from app.core.goal_progress import maybe_complete_goal

                for goal_id in outcome["touched_goal_ids"]:
                    await maybe_complete_goal(goal_id, room_id, db)

                # §4.2: a task with a goal just changed state (completed, blocked,
                # cancelled, ...). maybe_complete_goal above only refreshes Room
                # memory when the goal fully completes; this covers the other
                # terminal transitions too. Debounced, so calling it here even
                # when the goal above already refreshed is cheap.
                await maybe_refresh_room_memory(room_id, db, trigger="task_state_changed")

            # Unblock tasks whose last predecessor just reached a terminal state (§3.3).
            # Checkpoint tasks among them are published as CHECKPOINT_REQUEST
            # instead of being dispatched (generation checkpoint plan, stage 3).
            if outcome.get("terminal_task_ids"):
                from app.core.task_deps import publish_unblocked_tasks, try_unblock_tasks

                for _tid in outcome["terminal_task_ids"]:
                    _newly_ready = await try_unblock_tasks(_tid, db)
                    await publish_unblocked_tasks(
                        _newly_ready, room_id, db,
                        source="session_close_unblock",
                        unblocked_by=_tid,
                        session_id=session_id,
                    )

            # Emit TASK_DEPENDENCY_FAILED for dependents of failed/cancelled tasks (§3.4).
            if outcome.get("failed_task_ids"):
                from app.core.task_deps import get_blocked_dependents

                for _fid in outcome["failed_task_ids"]:
                    _blocked = await get_blocked_dependents(_fid, db)
                    for _bt in _blocked:
                        from app.core.metrics import record_task_dependency_failure
                        record_task_dependency_failure()
                        await emit_event(
                            room_id=room_id,
                            event_type="TASK_DEPENDENCY_FAILED",
                            content=(
                                f"Task '{_bt.title}' is blocked: "
                                f"its predecessor was cancelled or rejected."
                            ),
                            db=db,
                            metadata={
                                "task_id": str(_bt.id),
                                "blocked_because": str(_fid),
                                "failed_status": "failed",
                                "session_id": str(session_id),
                            },
                        )

            # Free per-session in-memory state (read-path cache for edit_file guard).
            from app.core.tools._session_state import clear_session as _cs
            _cs(str(session_id))

            return  # success
        except OperationalError as exc:
            if not _is_lock_timeout(exc):
                raise
            await db.rollback()
            if attempt == 0:
                logger.warning(
                    "session_close: advisory lock timeout (attempt 1) for session %s — retrying in 1s",
                    session_id,
                )
                await asyncio.sleep(1.0)
                # Refresh session reference after rollback
                session = await _refetch_session(db, session_id) or session
                continue
            # Second failure
            break

    # Save pending_actions and notify
    logger.error(
        "session_close: advisory lock timeout after retry for session %s — saving pending_actions",
        session_id,
    )
    # Re-fetch to get a clean ORM object in the new implicit transaction
    fresh = await _refetch_session(db, session_id)
    if fresh is not None:
        fresh.pending_actions = filtered.model_dump(mode="json")
        db.add(fresh)
        await db.flush()

    from app.core.metrics import record_advisory_lock_timeout
    record_advisory_lock_timeout(kind=kind)

    await emit_event(
        room_id=room_id,
        event_type="SESSION_CLOSE_LOCK_TIMEOUT",
        content=f"Session {session_id} close actions could not be applied due to a lock timeout.",
        db=db,
        metadata={"session_id": str(session_id), "kind": kind},
    )


# ---------------------------------------------------------------------------
# Deterministic memory flush (internal design notes §4)
#
# Before this, only the "normal" close path — the model calling close_session
# with an agent_memory_patch — ever wrote Agent memory.  Every involuntary
# close (tool_limit, timeout, orphaned_restart, idle, and the token_limit/
# rotated paths when the model's last call didn't include a memory patch)
# left Agent memory untouched.  On the room studied for the repair plan, 57%
# of proxy sessions closed via tool_limit — more than half of all completed
# work left no trace in memory.
#
# flush_session_memory (§4.1) is the single point every close path calls.
# maybe_refresh_room_memory (§4.2) is a separate, cheaper trigger for Room
# memory: it fires on task/goal terminal events, not on every session close,
# and never makes an LLM call (a debounced deterministic digest instead).
# ---------------------------------------------------------------------------

_TRUTHY = frozenset({"1", "true", "yes", "on", "enabled"})


def _memory_flush_forced() -> bool:
    """Kill switch for §4 (both flush_session_memory and maybe_refresh_room_memory)."""
    raw = os.getenv("MEMORY_FLUSH_FORCED", "true")
    return str(raw).strip().casefold() in _TRUTHY


# ---------------------------------------------------------------------------
# §4.1 — flush_session_memory
# ---------------------------------------------------------------------------

_FLUSH_APPEND_CHAR_LIMIT = 8000  # ~2000 tokens — matches AgentMemory's own budget


async def _flush_resolve_agent(session: "Session", db: AsyncSession) -> tuple[Optional[UUID], Any]:
    from sqlalchemy import select as _select

    from app.models.agent import Agent

    agent_id: UUID | None = getattr(session, "agent_id", None) or getattr(
        session, "chat_master_agent_id", None
    )
    if agent_id is None:
        return None, None
    agent = (
        await db.execute(_select(Agent).where(Agent.id == agent_id))
    ).scalars().first()
    return agent_id, agent


async def _flush_cheap_summary(
    session: "Session", agent: Any, llm_config: dict[str, Any] | None, db: AsyncSession
) -> str | None:
    """One cheap one-shot LLM call over the session's own history (§4.1 tier 2).

    Reuses proxy_session's _call_compact_summary / _build_summary_request rather
    than inventing a second summarizer — same model choice (Haiku for Anthropic),
    same prompt shape. Returns None on any failure or missing prerequisite; the
    caller falls back to a deterministic summary.
    """
    if agent is None or not llm_config or not llm_config.get("provider"):
        return None
    history = [m for m in (session.context_snapshot or {}).get("messages", []) if isinstance(m, dict)]
    if not any(m.get("role") == "assistant" for m in history):
        return None
    try:
        from app.core.proxy_session import _build_summary_request, _call_compact_summary
        from app.core.secrets.runtime import resolve_agent_secret

        secret_value = await resolve_agent_secret(
            db=db,
            owner_id=agent.owner_id,
            agent_id=agent.id,
            config=llm_config,
            request_id=session.id,
        )
        # Stage 1.2.2: commit before the summary's model call.  Under a
        # resource_lock release_db keeps the transaction (and logs) instead.
        from app.db.release import release_db

        await release_db(db, reason="session_close/cheap_summary")
        try:
            text = await _call_compact_summary(
                llm_config, _build_summary_request(history), secret_value=secret_value,
                purpose="session_summary", log_session=session,
            )
        finally:
            secret_value = None
        text = (text or "").strip()
        return text or None
    except Exception as exc:
        logger.warning(
            "session_close: flush cheap-summary call failed for session %s: %s",
            session.id, exc,
        )
        return None


def _flush_deterministic_summary(
    session: "Session",
    *,
    reason: str,
    task_title: str | None,
    changed_paths: list[str],
) -> str:
    """Last-resort summary (§4.1 tier 3) — no provider reachable, no text to reuse.

    Empty memory is never acceptable, so this always returns something, built
    only from facts already on hand: no LLM call, no chance of failing again.
    """
    bits = [f"Session closed automatically (kind={session.kind}, reason={reason})."]
    if task_title:
        bits.append(f"Task: {task_title}.")
    if changed_paths:
        shown = ", ".join(changed_paths[:15])
        extra = len(changed_paths) - 15
        if extra > 0:
            shown += f" (+{extra})"
        bits.append(f"Changed files: {shown}.")
    if not task_title and not changed_paths:
        bits.append("Session content is unavailable for the summary.")
    return " ".join(bits)


async def flush_session_memory(
    session: "Session",
    db: AsyncSession,
    *,
    reason: str,
    applied_actions: "SessionCloseActions | None" = None,
    llm_config: dict[str, Any] | None = None,
) -> bool:
    """Ensure this session's close leaves non-empty Agent memory (§4.1).

    Call this from every close path — the normal one (the model called
    close_session with tool_input) and every involuntary one (tool_limit,
    timeout, orphaned_restart, idle, rotated, ...). Pass whatever
    SessionCloseActions the normal path already applied as *applied_actions*
    so a session that wrote its own agent_memory_patch is left alone.

    Behavior:
      1. No-op if MEMORY_FLUSH_FORCED=false, if *applied_actions* already carries
         an agent_memory_patch, or if this session kind never gets agent memory
         (cm_self_plan — no per-agent memory in its close-actions matrix).
      2. Otherwise append a summary to AgentMemory (mode='append', trimmed to the
         same budget as a manual edit): reuse applied_actions.summary or
         session.summary if either is set (never re-derive one for free);
         else make one cheap LLM call; else a deterministic fallback.

    Never raises — the session is closing regardless of what this does.
    Returns True iff it actually wrote to AgentMemory.
    """
    if not _memory_flush_forced():
        return False
    if applied_actions is not None and applied_actions.agent_memory_patch is not None:
        return False
    if "agent_memory_patch" not in _KIND_ALLOWED.get(_matrix_key(session), frozenset()):
        return False

    try:
        agent_id, agent = await _flush_resolve_agent(session, db)
        if agent_id is None:
            return False

        summary_text: str | None = None
        if applied_actions is not None and (applied_actions.summary or "").strip():
            summary_text = applied_actions.summary.strip()
        elif (session.summary or "").strip():
            summary_text = session.summary.strip()
        tier = "reused_summary"

        if not summary_text:
            cfg = llm_config or (agent.llm_config if agent else None) or {}
            cheap = await _flush_cheap_summary(session, agent, cfg, db)
            if cheap:
                summary_text = cheap
                tier = "cheap_llm"

        task_title: str | None = None
        task_id = getattr(session, "task_id", None)
        if task_id is not None:
            from app.models.task import Task

            task_row = await db.get(Task, task_id)
            if task_row is not None:
                task_title = task_row.title

        changed_paths: list[str] = []
        if applied_actions is not None and applied_actions.file_writes:
            changed_paths = [fw.path for fw in applied_actions.file_writes]

        if not summary_text:
            summary_text = _flush_deterministic_summary(
                session, reason=reason, task_title=task_title, changed_paths=changed_paths,
            )
            tier = "deterministic"

        from sqlalchemy import select as _select

        from app.models.agent_memory import AgentMemory

        row = (
            await db.execute(
                _select(AgentMemory).where(
                    AgentMemory.room_id == session.room_id,
                    AgentMemory.agent_id == agent_id,
                )
            )
        ).scalars().first()
        now = datetime.utcnow()
        if row is None:
            row = AgentMemory(room_id=session.room_id, agent_id=agent_id)
        stamp = now.strftime("%Y-%m-%d %H:%M UTC")
        appended = (
            f"{(row.content or '').rstrip()}\n\n---\n"
            f"[{stamp} · auto-flush · {reason}]\n{summary_text.strip()}"
        )
        if len(appended) > _FLUSH_APPEND_CHAR_LIMIT:
            appended = appended[-_FLUSH_APPEND_CHAR_LIMIT:]
        row.content = appended
        row.token_count = len(appended) // 4
        row.updated_at = now
        db.add(row)
        await db.flush()

        from app.core.metrics import record_session_close_without_memory
        record_session_close_without_memory(kind=session.kind, reason=reason, tier=tier)

        from app.core.events import emit_event

        await emit_event(
            room_id=session.room_id,
            event_type="MEMORY_FLUSHED",
            content=f"Agent memory was updated automatically when the session closed (reason={reason}).",
            db=db,
            metadata={
                "session_id": str(session.id),
                "kind": session.kind,
                "agent_id": str(agent_id),
                "close_reason": reason,
                "tier": tier,
            },
        )
        return True
    except Exception:
        logger.exception(
            "session_close: flush_session_memory failed for session %s (reason=%s) — "
            "continuing without it",
            session.id, reason,
        )
        return False


# ---------------------------------------------------------------------------
# §4.2 — maybe_refresh_room_memory
# ---------------------------------------------------------------------------

_ROOM_MEMORY_MAX_CHARS = 8000  # keep in sync with api/v1/memory.py MAX_CHARS
_ROOM_MEMORY_AUTO_START = "<!-- room-memory:auto-status:start -->"
_ROOM_MEMORY_AUTO_END = "<!-- room-memory:auto-status:end -->"


def _room_memory_refresh_debounce() -> timedelta:
    raw = os.getenv("ROOM_MEMORY_AUTO_REFRESH_MINUTES", "15")
    try:
        minutes = max(1.0, float(raw))
    except (TypeError, ValueError):
        minutes = 15.0
    return timedelta(minutes=minutes)


def _render_room_status_digest(snapshot: dict[str, Any]) -> str:
    """Deterministic goal → status → done/blocked digest for Room memory (§4.2).

    Deliberately not the Task Board brick (board_snapshot.py): that one is a
    per-turn work list the model reads every session; this is a short standing
    paragraph inside Room memory, refreshed on goal/task terminal events so
    Room memory stops depending on the model remembering room_memory_patch.
    """
    counters = snapshot.get("counters") or {}
    lines = [
        f"Total tasks: {counters.get('total', 0)} · "
        f"completed: {counters.get('completed', 0)} · "
        f"blocked: {counters.get('blocked', 0)}."
    ]
    for goal in snapshot.get("goals", []):
        collapsed = goal.get("collapsed") or {}
        done = collapsed.get("completed", 0)
        total = goal.get("total", 0)
        blocked_titles = [
            t["title"]
            for bucket in (goal.get("columns") or {}).values()
            for t in bucket
            if t.get("blocked_reason")
        ][:5]
        line = f"- **{goal['title']}** ({goal['status']}): done {done}/{total}"
        if blocked_titles:
            line += "; blocked: " + ", ".join(blocked_titles)
        lines.append(line)
    without_goal = snapshot.get("without_goal") or {}
    if without_goal.get("total"):
        lines.append(f"- No goal: {without_goal['total']} tasks")
    return "\n".join(lines)


def _upsert_room_memory_auto_section(content: str, digest: str) -> str:
    """Replace the auto-generated section in place; leave the rest untouched.

    Keeps the digest from competing with (or repeatedly duplicating under) any
    narrative content the model itself wrote via room_memory_patch.
    """
    block = (
        f"{_ROOM_MEMORY_AUTO_START}\n## Status (updated automatically)\n\n"
        f"{digest}\n{_ROOM_MEMORY_AUTO_END}"
    )
    if _ROOM_MEMORY_AUTO_START in content and _ROOM_MEMORY_AUTO_END in content:
        pre, _, rest = content.partition(_ROOM_MEMORY_AUTO_START)
        _, _, post = rest.partition(_ROOM_MEMORY_AUTO_END)
        return (pre.rstrip() + "\n\n" + block + post).strip()
    return (content.rstrip() + "\n\n" + block).strip()


async def maybe_refresh_room_memory(room_id: UUID, db: AsyncSession, *, trigger: str) -> bool:
    """Deterministic Room memory refresh on task/goal terminal events (§4.2).

    Debounced per room (ROOM_MEMORY_AUTO_REFRESH_MINUTES, default 15) so a burst
    of task transitions doesn't spam the memory row or the room feed. No LLM
    call — this can fire on every task completion, far too often to spend a
    model call on; the digest is built straight from the Board snapshot
    projector (board_snapshot.py), already deterministic and cheap.
    Never raises.
    """
    if not _memory_flush_forced():
        return False
    try:
        from sqlalchemy import select as _select

        from app.models.project_memory import ProjectMemory

        now = datetime.utcnow()
        debounce = _room_memory_refresh_debounce()

        # Cheap pre-check outside the lock — the common case (recently refreshed)
        # never needs to take the advisory lock at all.
        row = (
            await db.execute(_select(ProjectMemory).where(ProjectMemory.room_id == room_id))
        ).scalars().first()
        if row is not None and now - row.updated_at < debounce:
            return False

        await resource_lock(db, room_id)
        # Re-check under lock: a concurrent close may have just refreshed it.
        row = (
            await db.execute(_select(ProjectMemory).where(ProjectMemory.room_id == room_id))
        ).scalars().first()
        if row is not None and now - row.updated_at < debounce:
            return False

        from app.core.board_snapshot import build_board_snapshot

        snapshot = await build_board_snapshot(room_id, db)
        digest = _render_room_status_digest(snapshot)

        if row is None:
            row = ProjectMemory(room_id=room_id)
        updated = _upsert_room_memory_auto_section(row.content or "", digest)
        if len(updated) > _ROOM_MEMORY_MAX_CHARS:
            updated = updated[:_ROOM_MEMORY_MAX_CHARS]
        row.content = updated
        row.token_count = len(updated) // 4
        row.updated_at = now
        db.add(row)
        await db.flush()

        from app.core.events import emit_event

        await emit_event(
            room_id=room_id,
            event_type="MEMORY_FLUSHED",
            content=f"Room memory updated automatically (trigger={trigger}).",
            db=db,
            metadata={"trigger": trigger, "token_count": row.token_count},
        )
        return True
    except Exception:
        logger.exception(
            "session_close: maybe_refresh_room_memory failed for room %s (trigger=%s)",
            room_id, trigger,
        )
        return False
