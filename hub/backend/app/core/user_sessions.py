"""
Direct-session lifecycle for user-facing sessions (Step 7 / §3).

Session kinds handled:
  user_agent_p1 — user ↔ specific agent, task-focused
  user_agent_p2 — user ↔ specific agent, open conversation
  user_cm_p1    — user ↔ ChatMaster, existing goal
  user_cm_p2    — user ↔ ChatMaster, new topic

Public API:
  start_user_session  — factory: create the right Session record from RouteDecision
  process_user_turn   — handle one user message turn; returns True if session stays open
"""
from __future__ import annotations


import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Awaitable, Callable
from uuid import UUID

import httpx
from openai import AsyncOpenAI, BadRequestError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.core.context_baseline import (
    ToolCallStats,
    build_context_measurement,
    record_context_measurement,
    tool_calls_from_anthropic,
    tool_calls_from_openai,
)
from app.core.context_assembler import (
    ContextAssembly,
    ContextAssembler,
    log_preflight_observation,
    persist_context_assembly,
    persist_provider_cache_usage,
)
from app.core.context_policy import resolve_context_policy, uses_cumulative_limit
from app.core.dialogue_scope import (
    quick_tasks_block_plan_message,
    quick_tasks_for_session,
)
from app.core.step_profile import (
    SIDE_CHAT_MASTER,
    step_tick,
    timed_llm,
)
from app.db.release import release_db
from app.core.chatgpt_responses import llm_client
from app.core.llm_providers import (
    messages_headers,
    messages_text_and_thinking,
    messages_url,
    openai_base_url,
    openai_client_extras,
    uses_anthropic_env_key,
    uses_messages_wire,
)
from app.core.reasoning import (
    apply_reasoning_body,
    apply_reasoning_chat,
    messages_reasoning_tokens,
    reasoning_from_chat_choice,
    show_reasoning_in_chat,
)
from app.core import session_interject
from app.core.events import (
    AgentStatus,
    emit_event,
    emit_interim_text_event,
    emit_reasoning_event,
    emit_status,
)
from app.core.subtask_activity import activity_metadata, new_subtask_activity
from app.core.memory_layer import build_memory_layer, build_system_prompt
from app.core.provider_cache import (
    anthropic_system_payload,
    extract_cache_usage,
    openai_cache_request_extra,
    provider_cache_key,
    stable_tool_schemas,
    system_prompt_version,
    tool_schema_version,
)
from app.core.router import OpenSessionContext, RouteDecision
from app.core.session_close import (
    ANTHROPIC_CLOSE_TOOL,
    OPENAI_CLOSE_TOOL,
    TOOL_MARKUP_FALLBACK_TEXT,
    SessionCloseActions,
    ToolArgumentsParseError,
    apply_close_actions,
    clean_reply_text,
    close_tool_for_kind,
    extract_close_actions_from_tool_use,
    filter_by_kind,
    flush_session_memory,
    parse_tool_arguments,
    strip_tool_call_markup,
    _matrix_key,
)
from app.core.token_budget import (
    TOKEN_LIMIT,
    check_token_budget,
    track_llm_usage,
    update_context_window_metrics,
)
from app.core.user_session_limits import (
    USER_SESSION_KINDS,
    lifecycle_enforced,
    rotation_reason,
    window_soft_ratio,
)
from app.core.secrets.crypto import SecretValue
from app.core.secrets.runtime import binding_id_from_config, legacy_secret_value, resolve_for_use
from app.models.secret import SecretPurpose, SecretResourceType
from app.models.agent import Agent
from app.models.message import Message
from app.models.room import Room
from app.models.room import RoomAgentLink
from app.models.session.base import Session
from app.models.task import Task, TaskDependency
from app.models.user import User
from app.repositories.session import SessionRepository
from app.schemas.session import UserAgentSessionCreate, UserCmSessionCreate
from app.schemas.task_change import (
    TaskChangeProposal,
    TaskChangeValidationContext,
    validate_task_change_proposal,
)
from app.websockets.manager import manager

logger = logging.getLogger(__name__)


def _record_provider_cache(
    config: dict[str, Any],
    usage: Any,
    *,
    role: str,
) -> None:
    cache = extract_cache_usage(usage, str(config.get("provider") or ""))
    if not cache.read_tokens and not cache.write_tokens:
        return
    try:
        from app.core.metrics import record_provider_cache_usage

        record_provider_cache_usage(
            provider=str(config.get("provider") or "unknown"),
            model=str(config.get("model") or "unknown"),
            role=role,
            read_tokens=cache.read_tokens,
            write_tokens=cache.write_tokens,
        )
    except Exception:
        logger.debug("user_sessions: provider cache metric recording failed", exc_info=True)


def _parse_xml_plan_steps(inner: str) -> list[dict[str, Any]]:
    """Parse a ``<plan_steps><step>…</step></plan_steps>`` block from an XML
    ``<propose_plan>`` body into the normalized step list."""
    import re as _re

    block_m = _re.search(
        r"<plan_steps>(.*?)</plan_steps>", inner, _re.DOTALL | _re.IGNORECASE
    )
    if not block_m:
        return []
    steps: list[dict[str, Any]] = []
    for step_m in _re.finditer(
        r"<step>(.*?)</step>", block_m.group(1), _re.DOTALL | _re.IGNORECASE
    ):
        body = step_m.group(1)

        def _tag(tag: str) -> str:
            m = _re.search(rf"<{tag}>(.*?)</{tag}>", body, _re.DOTALL | _re.IGNORECASE)
            return m.group(1).strip() if m else ""

        steps.append(
            {
                "title": _tag("title"),
                "description": _tag("description"),
                "assignee": _tag("assignee") or _tag("agent"),
            }
        )
    return steps


def _plan_steps_from_proposal(plan_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a propose_plan ``plan_steps`` payload into a clean list.

    Accepts the tool-call shape and the looser shapes weaker models emit — a
    step as a bare string, or the assignee under ``agent`` /
    ``assigned_agent_id`` instead of ``assignee``. Returns
    ``[{"title", "description", "assignee"}]``; entries without a usable title
    are dropped. The ``assignee`` string (an ``@nickname`` or a UUID) is
    resolved to an agent id later, in ``_trigger_plan_pipeline``.
    """
    raw = plan_data.get("plan_steps")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            title, desc, assignee = item.strip(), "", ""
        elif isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or "").strip()
            desc = str(item.get("description") or item.get("desc") or "").strip()
            assignee = str(
                item.get("assignee")
                or item.get("agent")
                or item.get("assigned_agent_id")
                or item.get("assigned_agent")
                or ""
            ).strip()
        else:
            continue
        if not title:
            continue
        out.append({"title": title, "description": desc, "assignee": assignee})
    return out


def _normalize_requirements(raw: Any) -> list[dict[str, Any]]:
    """Normalize a ``requirements[]`` payload: ``{quote, turn, statement,
    priority, acceptance}``. Tolerates a bare string (becomes the quote) and
    ``text`` instead of ``quote``, same as ``_plan_steps_from_proposal``.
    Entries without a usable quote are dropped — an unquoted "requirement" is
    exactly the unverifiable prose this stage exists to stop."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            quote, turn, statement, priority, acceptance = item.strip(), "", "", "", ""
        elif isinstance(item, dict):
            quote = str(item.get("quote") or item.get("text") or "").strip()
            turn = str(item.get("turn") or item.get("turn_ref") or "").strip()
            statement = str(item.get("statement") or "").strip()
            priority = str(item.get("priority") or "").strip().lower()
            acceptance = str(item.get("acceptance") or "").strip()
        else:
            continue
        if not quote:
            continue
        out.append({
            "quote": quote,
            "turn": turn,
            "statement": statement,
            "priority": priority,
            "acceptance": acceptance,
        })
    return out


def _normalize_non_goals(raw: Any) -> list[dict[str, Any]]:
    """Normalize a ``non_goals[]`` payload: ``{quote, statement}``."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            quote, statement = item.strip(), ""
        elif isinstance(item, dict):
            quote = str(item.get("quote") or item.get("text") or "").strip()
            statement = str(item.get("statement") or "").strip()
        else:
            continue
        if not quote:
            continue
        out.append({"quote": quote, "statement": statement})
    return out


def _normalize_open_questions(raw: Any) -> list[dict[str, Any]]:
    """Normalize an ``open_questions[]`` payload: ``{question, assumption}``."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            question, assumption = item.strip(), ""
        elif isinstance(item, dict):
            question = str(item.get("question") or item.get("text") or "").strip()
            assumption = str(item.get("assumption") or "").strip()
        else:
            continue
        if not question:
            continue
        out.append({"question": question, "assumption": assumption})
    return out


def _requirements_from_proposal(plan_data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Normalize propose_plan's optional ``requirements``/``non_goals``/
    ``open_questions`` arrays (internal design notes Stage 1) —
    the sibling of ``_plan_steps_from_proposal`` for the traceability fields.
    """
    return {
        "requirements": _normalize_requirements(plan_data.get("requirements")),
        "non_goals": _normalize_non_goals(plan_data.get("non_goals")),
        "open_questions": _normalize_open_questions(plan_data.get("open_questions")),
    }


def _parse_xml_requirements(inner: str) -> list[dict[str, Any]]:
    """Parse a ``<requirements><requirement>…</requirement></requirements>``
    block from an XML ``<propose_plan>`` body."""
    import re as _re

    block_m = _re.search(
        r"<requirements>(.*?)</requirements>", inner, _re.DOTALL | _re.IGNORECASE
    )
    if not block_m:
        return []
    items: list[dict[str, Any]] = []
    for item_m in _re.finditer(
        r"<requirement>(.*?)</requirement>", block_m.group(1), _re.DOTALL | _re.IGNORECASE
    ):
        body = item_m.group(1)

        def _tag(tag: str) -> str:
            m = _re.search(rf"<{tag}>(.*?)</{tag}>", body, _re.DOTALL | _re.IGNORECASE)
            return m.group(1).strip() if m else ""

        items.append({
            "quote": _tag("quote"),
            "turn": _tag("turn"),
            "statement": _tag("statement"),
            "priority": _tag("priority"),
            "acceptance": _tag("acceptance"),
        })
    return items


def _parse_xml_non_goals(inner: str) -> list[dict[str, Any]]:
    """Parse a ``<non_goals><non_goal>…</non_goal></non_goals>`` block from an
    XML ``<propose_plan>`` body."""
    import re as _re

    block_m = _re.search(
        r"<non_goals>(.*?)</non_goals>", inner, _re.DOTALL | _re.IGNORECASE
    )
    if not block_m:
        return []
    items: list[dict[str, Any]] = []
    for item_m in _re.finditer(
        r"<non_goal>(.*?)</non_goal>", block_m.group(1), _re.DOTALL | _re.IGNORECASE
    ):
        body = item_m.group(1)

        def _tag(tag: str) -> str:
            m = _re.search(rf"<{tag}>(.*?)</{tag}>", body, _re.DOTALL | _re.IGNORECASE)
            return m.group(1).strip() if m else ""

        items.append({"quote": _tag("quote"), "statement": _tag("statement")})
    return items


def _parse_xml_open_questions(inner: str) -> list[dict[str, Any]]:
    """Parse an ``<open_questions><open_question>…</open_question></open_questions>``
    block from an XML ``<propose_plan>`` body."""
    import re as _re

    block_m = _re.search(
        r"<open_questions>(.*?)</open_questions>", inner, _re.DOTALL | _re.IGNORECASE
    )
    if not block_m:
        return []
    items: list[dict[str, Any]] = []
    for item_m in _re.finditer(
        r"<open_question>(.*?)</open_question>", block_m.group(1), _re.DOTALL | _re.IGNORECASE
    ):
        body = item_m.group(1)

        def _tag(tag: str) -> str:
            m = _re.search(rf"<{tag}>(.*?)</{tag}>", body, _re.DOTALL | _re.IGNORECASE)
            return m.group(1).strip() if m else ""

        items.append({"question": _tag("question"), "assumption": _tag("assumption")})
    return items


def _extract_propose_plan_from_text(text: str) -> dict[str, Any] | None:
    """Detect a text-based propose_plan call from models that write JSON/XML instead of calling the tool.

    Handles multiple output formats:
      1. XML tags: <propose_plan><goal_title>...</goal_title>...</propose_plan>
      2. JSON block: {"goal_title": "...", "goal_description": "..."}
      3. Markdown code block containing JSON with goal_title

    Returns the parsed dict or None.
    """
    import re as _re

    # --- Format 1: XML tags <propose_plan>...</propose_plan> ---
    xml_m = _re.search(r'<propose_plan>(.*?)</propose_plan>', text, _re.DOTALL | _re.IGNORECASE)
    if xml_m:
        inner = xml_m.group(1)
        def _xtag(tag: str) -> str:
            m = _re.search(rf'<{tag}>(.*?)</{tag}>', inner, _re.DOTALL | _re.IGNORECASE)
            return m.group(1).strip() if m else ""
        title = _xtag("goal_title")
        desc = _xtag("goal_description")
        if title and desc:
            return {
                "goal_title": title,
                "goal_description": desc,
                "plan_summary": _xtag("plan_summary"),
                "plan_steps": _parse_xml_plan_steps(inner),
                "requirements": _normalize_requirements(_parse_xml_requirements(inner)),
                "non_goals": _normalize_non_goals(_parse_xml_non_goals(inner)),
                "open_questions": _normalize_open_questions(_parse_xml_open_questions(inner)),
            }

    # --- Format 2: JSON object with goal_title anywhere in text ---
    # Try greedy multi-line JSON blocks first (code fences or bare)
    for pattern in (
        r'```(?:json)?\s*(\{.*?\})\s*```',   # fenced code block
        r'(\{[^{}]{20,}"goal_title"[^{}]*\})', # bare JSON, min 20 chars to avoid short matches
    ):
        for m in _re.finditer(pattern, text, _re.DOTALL):
            try:
                data = json.loads(m.group(1))
                if data.get("goal_title") and data.get("goal_description"):
                    return {
                        "goal_title": str(data["goal_title"]),
                        "goal_description": str(data["goal_description"]),
                        "plan_summary": str(data.get("plan_summary", "")),
                        "plan_steps": _plan_steps_from_proposal(data),
                        **_requirements_from_proposal(data),
                    }
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    return None


from app.core.model_pricing import cache_cost_adjustment, compute_cost as _compute_cost


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def start_user_session(
    room: Room,
    user: User,
    route_decision: RouteDecision,
    db: AsyncSession,
    *,
    cm_agent_id: UUID | None = None,
) -> Session:
    """Create the right Session record for a user-facing direct session.

    Args:
        room:          The room where the session lives.
        user:          The user initiating the session.
        route_decision: Classification result from route_user_message().
        db:            AsyncSession.
        cm_agent_id:   Required for chatmaster_* routes (the ChatMaster agent UUID).
                       Callers obtain it from room.coordinator_config or a room-level lookup.
    """
    route = route_decision.route
    repo = SessionRepository(db)

    payload: UserAgentSessionCreate | UserCmSessionCreate

    if route == "single_agent_p1":
        agent_id = route_decision.mentioned_agent_ids[0]
        # user_agent_p1 requires both task_id and goal_id, but the router
        # routinely names the task and leaves matched_goal_id null (observed in
        # production: single_agent_p1 with matched_goal_id=None → the
        # UserAgentSessionCreate ValidationError that the caller only logs, so
        # the turn silently vanishes). Backfill the goal from the task row; if
        # the task is unknown or goal-less, degrade to p2 instead of raising.
        task_id = route_decision.matched_task_id
        goal_id = route_decision.matched_goal_id
        if task_id is not None and goal_id is None:
            task = (
                await db.execute(
                    select(Task).where(
                        Task.id == task_id, Task.room_id == room.id
                    )
                )
            ).scalars().first()
            if task is not None:
                goal_id = task.goal_id
        if task_id is None or goal_id is None:
            logger.warning(
                "start_user_session: single_agent_p1 without a usable task/goal "
                "(task_id=%s goal_id=%s) — falling back to p2",
                task_id, route_decision.matched_goal_id,
            )
            route = "single_agent_p2"
            # p2 ignores task_id when it builds its memory layer, but the row
            # keeps the link on purpose: it is what lets Cancel/Block on the task
            # reach this dialogue. Without it the conversation is just "a chat
            # with an agent" and kept working on a task the user had stopped.
            payload = UserAgentSessionCreate(
                kind="user_agent_p2",
                room_id=room.id,
                user_id=user.id,
                agent_id=agent_id,
                task_id=task_id,
            )
        else:
            payload = UserAgentSessionCreate(
                kind="user_agent_p1",
                room_id=room.id,
                user_id=user.id,
                agent_id=agent_id,
                task_id=task_id,
                goal_id=goal_id,
            )
    elif route == "single_agent_p2":
        agent_id = route_decision.mentioned_agent_ids[0]
        payload = UserAgentSessionCreate(
            kind="user_agent_p2",
            room_id=room.id,
            user_id=user.id,
            agent_id=agent_id,
        )
    elif route == "chatmaster_p1":
        if cm_agent_id is None:
            raise ValueError("cm_agent_id is required for chatmaster_p1 sessions")
        if route_decision.matched_goal_id is None:
            logger.warning("chatmaster_p1 without goal_id — falling back to p2")
            route = "chatmaster_p2"
            payload = UserCmSessionCreate(
                kind="user_cm_p2",
                room_id=room.id,
                user_id=user.id,
                chat_master_agent_id=cm_agent_id,
            )
        else:
            payload = UserCmSessionCreate(
                kind="user_cm_p1",
                room_id=room.id,
                user_id=user.id,
                chat_master_agent_id=cm_agent_id,
                goal_id=route_decision.matched_goal_id,
            )
    elif route == "chatmaster_p2":
        if cm_agent_id is None:
            raise ValueError("cm_agent_id is required for chatmaster_p2 sessions")
        payload = UserCmSessionCreate(
            kind="user_cm_p2",
            room_id=room.id,
            user_id=user.id,
            chat_master_agent_id=cm_agent_id,
        )
    else:
        raise ValueError(
            f"start_user_session: unsupported route {route!r}. "
            "Expected single_agent_p1/p2 or chatmaster_p1/p2."
        )

    session = await repo.create(payload)

    await emit_event(
        room_id=room.id,
        event_type="SESSION_STARTED",
        content=f"Session {session.id} started (kind={session.kind})",
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": session.kind,
            "route": route,
        },
    )

    if route in ("single_agent_p1", "single_agent_p2"):
        # internal design notes stage 4.3: a previous dialogue in this
        # room may have left dialogue work unfinalized (a crash, or workspaced
        # unreachable when its own close tried Gate 1). Sweeping here — lazily,
        # once per new direct session — needs no background loop of its own.
        try:
            from app.core.session_work import sweep_orphan_session_work

            await sweep_orphan_session_work(room.id, db)
        except Exception:
            logger.warning(
                "user_sessions: orphan session-work sweep failed for room %s",
                room.id, exc_info=True,
            )

    return session


# ---------------------------------------------------------------------------
# Conversation history helpers (stored in session.context_snapshot)
# ---------------------------------------------------------------------------


def _get_history(session: Session) -> list[dict[str, str]]:
    snap = session.context_snapshot or {}
    return list(snap.get("messages", []))


def _adopt_merged_draft_prefix(session: Session, history: list[dict[str, str]]) -> None:
    """Prepend turns ``merge_draft`` attached during this turn to *history*.

    ``merge_draft_into_session`` writes the carried turns straight into
    ``context_snapshot``, but the running turn holds its own ``history`` list
    and saves it back at the end — which silently dropped the merge, so the
    next turn had no memory of the dialogue it had just "continued". The merge
    leaves the prefix under ``PENDING_MERGED_PREFIX_KEY``; take it here, once.
    """
    from app.core.plan_dialogue import PENDING_MERGED_PREFIX_KEY

    snap = session.context_snapshot or {}
    pending = snap.get(PENDING_MERGED_PREFIX_KEY)
    if not pending:
        return
    head = str(pending[0].get("content") or "") if isinstance(pending[0], dict) else ""
    if not (history and str(history[0].get("content") or "") == head):
        history[:0] = [dict(m) for m in pending if isinstance(m, dict)]
    snap = dict(snap)
    snap.pop(PENDING_MERGED_PREFIX_KEY, None)
    session.context_snapshot = snap
    _save_history(session, history)


def _flag_snapshot_dirty(session: Session) -> None:
    """Force ``context_snapshot`` into the next UPDATE.

    It is a plain JSONB column (no MutableDict): SQLAlchemy compares the new
    value against its committed state at flush time, and an aliased-and-mutated
    ``messages`` list can make that comparison say "unchanged", silently
    dropping the column from the UPDATE. A no-op on the plain stand-ins used in
    unit tests, which is fine — there the assignment alone is the whole story.
    """
    try:
        flag_modified(session, "context_snapshot")
    except (AttributeError, KeyError):
        pass


async def _journal_turn(session: Session, role: str, content: str) -> None:
    """Best-effort append of one real dialogue turn to the L0 journal.

    internal design notes Stage 1: the journal is written as the turn
    happens so Stage 2's archive never depends on ``context_snapshot`` alone.
    Never raises; a Redis outage is invisible here.
    """
    try:
        from app.core.plan_dialogue import journal_append

        await journal_append(session.id, role, content)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("_journal_turn skipped (session %s): %s", session.id, exc)


_DRAFT_SESSION_KINDS = frozenset(
    {"user_cm_p1", "user_cm_p2", "user_agent_p1", "user_agent_p2"}
)


async def _maybe_save_dialogue_draft(
    session: Session, db: AsyncSession, *, close_reason: str
) -> None:
    """L1: materialize a draft when a formation dialogue closes without a Goal/Task.

    internal design notes Stage 5. Best-effort — the session is already
    closing, so a draft-write failure must stay invisible here.
    """
    if session.kind not in _DRAFT_SESSION_KINDS:
        return
    try:
        from app.core import plan_dialogue

        await plan_dialogue.save_draft(session, db, close_reason=close_reason)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("_maybe_save_dialogue_draft skipped (session %s): %s", session.id, exc)


def _save_history(session: Session, history: list[dict[str, str]]) -> None:
    # context_snapshot is a plain JSONB column (no MutableDict wrapper). Two
    # traps follow from that and this function has to defuse both, or a turn's
    # reply silently never reaches the DB (incident 2026-09-08#3: the CM's
    # clarifying questions were published to the room but lost from the session,
    # so the next turn's router saw no open question and cold-rotated the
    # session mid requirement-gathering):
    #   1. Storing the caller's live `history` list into the snapshot aliases it.
    #      Every later `history.append(...)` in the same turn then also mutates
    #      the value SQLAlchemy holds as the *committed* state, so at flush time
    #      `compare_values(committed, new)` sees them equal and drops the column
    #      from the UPDATE. Store a detached deep-ish copy instead.
    #   2. Even with the copy, be explicit — flag_modified forces the column into
    #      the UPDATE regardless of value comparison (matches the flag_modified
    #      pattern used for the other JSON columns in this codebase).
    snap = dict(session.context_snapshot or {})
    snap["messages"] = [dict(m) for m in history]
    session.context_snapshot = snap
    _flag_snapshot_dirty(session)
    # Stage 5.3: observable size of the conversation without decoding
    # context_snapshot — message_count mirrors len(history) (already the
    # number rotation_reason() and is_continuable() rotate on); turn_count is
    # the coarser "how many times did the user speak" count for the tab badge.
    session.message_count = len(history)
    session.turn_count = sum(1 for m in history if m.get("role") == "user")


# ---------------------------------------------------------------------------
# LLM call with close_session tool support
# ---------------------------------------------------------------------------


async def _get_agent_llm_config(session: Session, db: AsyncSession) -> dict[str, Any]:
    agent_id: UUID | None = getattr(session, "agent_id", None) or getattr(
        session, "chat_master_agent_id", None
    )
    if agent_id is None:
        return {}
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalars().first()
    return (agent.llm_config or {}) if agent else {}


# ---------------------------------------------------------------------------
# propose_plan tool (user_cm_p2 only) — presents plan for button confirmation
# ---------------------------------------------------------------------------

_PROPOSE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal_title": {
            "type": "string",
            "description": "Concise goal name (becomes the goal title if accepted).",
        },
        "goal_description": {
            "type": "string",
            "description": "Full scope including all agreed requirements.",
        },
        "plan_summary": {
            "type": "string",
            "description": "Brief human-readable summary shown in the confirmation card.",
        },
        "plan_steps": {
            "type": "array",
            "description": (
                "The plan broken into concrete steps — one per unit of work an "
                "agent can pick up. Each step becomes a task and its `assignee` "
                "becomes that task's responsible agent, so name an assignee for "
                "every step you can. Provide at least one step whenever the goal "
                "needs delegated work."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short imperative name of this step.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Scope and acceptance criteria for this step.",
                    },
                    "assignee": {
                        "type": "string",
                        "description": (
                            "Which room agent should do this step: their "
                            "@nickname from the Agents List (a bare agent UUID "
                            "also works). Leave empty only for a human-decision / "
                            "approval step."
                        ),
                    },
                },
                "required": ["title"],
            },
        },
        "requirements": {
            "type": "array",
            "description": (
                "Atomic requirements gathered from the user, one per item. "
                "`quote` must be the user's own words verbatim (not your "
                "paraphrase) — it is checked against the dialogue log, so a "
                "made-up quote is flagged unverified. `turn` is the dialogue "
                "turn number (e.g. \"#7\") the quote came from, if known."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "quote": {
                        "type": "string",
                        "description": "The user's exact words, unparaphrased.",
                    },
                    "turn": {
                        "type": "string",
                        "description": "Dialogue turn the quote is from, e.g. \"#7\".",
                    },
                    "statement": {
                        "type": "string",
                        "description": "Normalized one-line restatement of the requirement.",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["must", "should", "could"],
                        "description": "MUST needs an observable acceptance criterion.",
                    },
                    "acceptance": {
                        "type": "string",
                        "description": "Observable evidence this requirement is met.",
                    },
                },
                "required": ["quote"],
            },
        },
        "non_goals": {
            "type": "array",
            "description": (
                "Things the user explicitly excluded from scope (e.g. "
                "\"no authorization\", \"we are not touching the frontend\") — quoted verbatim, "
                "same as requirements."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "quote": {
                        "type": "string",
                        "description": "The user's exact words excluding this from scope.",
                    },
                    "statement": {
                        "type": "string",
                        "description": "Normalized one-line restatement of the exclusion.",
                    },
                },
                "required": ["quote"],
            },
        },
        "open_questions": {
            "type": "array",
            "description": (
                "Anything left unresolved by the dialogue. State the "
                "assumption you're proceeding on instead of silently folding "
                "it into goal_description."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "What remains unclear.",
                    },
                    "assumption": {
                        "type": "string",
                        "description": "The assumption used in the plan pending an answer.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    "required": ["goal_title", "goal_description"],
}

_PROPOSE_PLAN_DESCRIPTION = (
    "Present a finalized goal plan to the user for button-based confirmation. "
    "Call this when you have gathered all requirements and are ready to create the goal. "
    "The user will see a card with 'Accept' and 'Edit' buttons — do NOT ask them to type 'ok'. "
    "The session remains open until the user responds."
)

PROPOSE_PLAN_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": "propose_plan",
    "description": _PROPOSE_PLAN_DESCRIPTION,
    "input_schema": _PROPOSE_PLAN_SCHEMA,
}

PROPOSE_PLAN_TOOL_OPENAI: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "propose_plan",
        "description": _PROPOSE_PLAN_DESCRIPTION,
        "parameters": _PROPOSE_PLAN_SCHEMA,
    },
}

# The turn text of a "Form a plan" button press. It is a user message on
# purpose: the history must show why the session suddenly stopped asking
# questions, and after an Edit the model can see this is where the requirement
# gathering was cut off.
FORCE_PLAN_USER_MESSAGE = (
    "[The user pressed the 'Form a plan' button.] "
    "Requirement gathering is over. Do not ask new clarifying questions, do not read files "
    "and do not continue discussing the spec — propose a plan right now for everything "
    "collected in this dialogue so far. If some detail is missing, record a reasonable "
    "assumption right in goal_description instead of asking a question."
)

# offer_plan tool (user_cm_p2 only) — the "Form a plan" button is not on screen
# for the whole discussion: ChatMaster raises it with this tool once it has
# gathered enough to plan, and the click is the user's one-step confirmation.
# The turn ends as an ordinary text reply (`message`); the session remembers
# the offer in context_snapshot["plan_offered"] until a plan is proposed.
OFFER_PLAN_TOOL_NAME = "offer_plan"

_OFFER_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string",
            "description": (
                "Your reply to the user, shown in the chat above the button: a short "
                "recap of what you are about to plan and, if anything, what you will "
                "assume. Do not ask a new question here."
            ),
        },
    },
    "required": ["message"],
}

_OFFER_PLAN_DESCRIPTION = (
    "Show the user the 'Form a plan' button. Call this — instead of asking "
    "'shall I form the plan?' — once the requirements are clear enough to plan. "
    "The user confirms with one click (you then get a turn that asks for the plan) "
    "or keeps refining in the chat. Never call it while you still have open questions."
)

OFFER_PLAN_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": OFFER_PLAN_TOOL_NAME,
    "description": _OFFER_PLAN_DESCRIPTION,
    "input_schema": _OFFER_PLAN_SCHEMA,
}

OFFER_PLAN_TOOL_OPENAI: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": OFFER_PLAN_TOOL_NAME,
        "description": _OFFER_PLAN_DESCRIPTION,
        "parameters": _OFFER_PLAN_SCHEMA,
    },
}

_OFFER_PLAN_FALLBACK_MESSAGE = (
    "I have what I need to form the plan. Press 'Form a plan' when you are ready, "
    "or keep refining the task here."
)


# ---------------------------------------------------------------------------
# propose_task_change tool (user_cm_p1 only) — stores a reviewable draft
# ---------------------------------------------------------------------------

_TASK_CHANGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "version": {"type": "integer", "const": 1},
        "proposal_id": {"type": "string", "format": "uuid"},
        "rationale": {
            "type": "string",
            "description": "Why the existing task must be replaced.",
        },
        "sources": {
            "type": "array",
            "minItems": 1,
            "description": "Existing non-terminal tasks to cancel after confirmation.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task_id": {"type": "string", "format": "uuid"},
                    "expected_updated_at": {"type": "string", "format": "date-time"},
                    "expected_state": {"type": "string"},
                },
                "required": ["task_id", "expected_updated_at", "expected_state"],
            },
        },
        "create_tasks": {
            "type": "array",
            "minItems": 1,
            "description": (
                "Replacement tasks. Each must include acceptance criteria in description, "
                "an existing room agent UUID, and depends_on refs or existing task UUIDs."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ref": {"type": "string", "description": "Unique proposal-local slug."},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "assigned_agent_id": {"type": "string", "format": "uuid"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                    "checkpoint_kind": {
                        "type": "string",
                        "enum": ["confirm", "select_variant"],
                        "description": (
                            "When set, this task is a human-decision gate, not agent "
                            "work. 'confirm' = yes/no approval; 'select_variant' = pick "
                            "ONE of several outputs of a depended-on task. "
                            "assigned_agent_id must be null. Only use when the plan "
                            "itself names an explicit approval/choice point."
                        ),
                    },
                    "checkpoint_payload": {
                        "type": "object",
                        "description": (
                            "{prompt: str} for confirm; "
                            "{prompt: str, variants: [{id, label, paths: [str], note?}]} "
                            "for select_variant."
                        ),
                        "default": {},
                    },
                },
                "required": ["ref", "title", "assigned_agent_id"],
            },
        },
        "dependency_rewrites": {
            "type": "array",
            "description": (
                "Explicit rewrites for every existing downstream task that depended on a source."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task_id": {"type": "string", "format": "uuid"},
                    "remove_depends_on": {
                        "type": "array",
                        "items": {"type": "string", "format": "uuid"},
                        "default": [],
                    },
                    "add_depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["task_id"],
            },
        },
        "created_at": {"type": "string", "format": "date-time"},
    },
    "required": ["rationale", "sources", "create_tasks"],
}

_TASK_CHANGE_DESCRIPTION = (
    "Prepare an exact, reviewable replacement of one or more existing tasks in the current goal. "
    "Use only after you can name every source task, replacement task, assignee, dependency, and "
    "downstream rewrite. This tool only saves a draft and shows a confirmation card; it never "
    "changes tasks. Do not ask the user to type 'ok' and do not use task_create, task_update, "
    "or close_session to perform this replacement."
)

PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC: dict[str, Any] = {
    "name": "propose_task_change",
    "description": _TASK_CHANGE_DESCRIPTION,
    "input_schema": _TASK_CHANGE_SCHEMA,
}

PROPOSE_TASK_CHANGE_TOOL_OPENAI: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "propose_task_change",
        "description": _TASK_CHANGE_DESCRIPTION,
        "parameters": _TASK_CHANGE_SCHEMA,
    },
}

_USER_CM_P1_BLOCKED_WORKSPACE_TOOLS = frozenset({"task_create", "task_update"})


def _workspace_tool_names_for_session(
    session: Session,
    role: str,
    registry: Any,
) -> list[str]:
    """Return the workspace capability boundary for a direct user session."""
    names = [tool.name for tool in registry.tools_for(role)]
    if session.kind == "user_cm_p1":
        names = [name for name in names if name not in _USER_CM_P1_BLOCKED_WORKSPACE_TOOLS]
    return names


def _compact_task_change_error(exc: Exception) -> str:
    """Turn schema/live-board errors into a short repair instruction for the LLM."""
    message = str(exc).strip() or "proposal does not match the task-change contract"
    try:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError) and exc.errors():
            error = exc.errors()[0]
            location = ".".join(str(part) for part in error.get("loc", ())) or "proposal"
            message = f"{location}: {error.get('msg', message)}"
    except Exception:
        pass
    message = " ".join(message.split())
    if len(message) > 360:
        message = message[:357].rstrip() + "..."
    return (
        "Error: propose_task_change was rejected: "
        f"{message}. Fix the IDs, refs, assignees, or dependency rewrites and call "
        "propose_task_change again. No tasks were changed."
    )


def _record_task_change_validation_failure(exc: Exception, *, stage: str) -> None:
    """Record a bounded validation reason without leaking arbitrary labels."""

    from app.core.metrics import log_task_change_event, record_task_change_validation_error

    message = str(exc).lower()
    if "cycle" in message:
        reason = "graph_cycle"
    elif "depend" in message or "rewir" in message:
        reason = "dependency"
    elif stage == "schema":
        reason = "schema"
    elif stage == "scope":
        reason = "scope"
    elif stage == "board":
        reason = "board_unavailable"
    else:
        reason = "semantic"
    record_task_change_validation_error(reason)
    log_task_change_event(
        "task_change_validation_failed",
        stage=stage,
        reason=reason,
        error_type=type(exc).__name__,
    )


async def _parse_and_validate_task_change(
    session: Session,
    tool_input: dict[str, Any],
    db: AsyncSession,
) -> tuple[TaskChangeProposal | None, str | None, list[Task]]:
    """Parse a CM proposal and validate it against the current board snapshot."""
    try:
        proposal = TaskChangeProposal.model_validate(tool_input)
    except Exception as exc:
        _record_task_change_validation_failure(exc, stage="schema")
        return None, _compact_task_change_error(exc), []

    if session.kind != "user_cm_p1":
        exc = ValueError("propose_task_change is available only for user_cm_p1")
        _record_task_change_validation_failure(exc, stage="scope")
        return None, _compact_task_change_error(exc), []
    goal_id = getattr(session, "goal_id", None)
    if goal_id is None:
        exc = ValueError("user_cm_p1 session has no current goal")
        _record_task_change_validation_failure(exc, stage="scope")
        return None, _compact_task_change_error(exc), []

    try:
        tasks = list(
            (
                await db.execute(select(Task).where(Task.room_id == session.room_id))
            ).scalars().all()
        )
        room_agent_ids = set(
            (
                await db.execute(
                    select(RoomAgentLink.agent_id).where(RoomAgentLink.room_id == session.room_id)
                )
            ).scalars().all()
        )
        dependency_edges = [
            (row[0], row[1])
            for row in (
                await db.execute(
                    select(TaskDependency.task_id, TaskDependency.depends_on_id).where(
                        TaskDependency.task_id.in_([task.id for task in tasks])
                    )
                )
            ).all()
        ]
    except Exception as exc:
        _record_task_change_validation_failure(exc, stage="board")
        return None, _compact_task_change_error(
            RuntimeError(f"current board snapshot is unavailable: {exc}")
        ), []
    try:
        validate_task_change_proposal(
            proposal,
            TaskChangeValidationContext(
                room_id=session.room_id,
                goal_id=goal_id,
                tasks=tasks,
                room_agent_ids=room_agent_ids,
                dependency_edges=dependency_edges,
            ),
        )
    except Exception as exc:
        _record_task_change_validation_failure(exc, stage="semantic")
        return None, _compact_task_change_error(exc), tasks
    return proposal, None, tasks


def _task_change_event_metadata(
    session: Session,
    proposal: TaskChangeProposal,
    tasks: list[Task],
) -> dict[str, Any]:
    """Build a self-contained event payload for restoring the confirmation card."""
    task_by_id = {task.id: task for task in tasks}
    sources: list[dict[str, Any]] = []
    for source in proposal.sources:
        task = task_by_id.get(source.task_id)
        task_state = getattr(task, "state", None) if task is not None else None
        task_state_value = getattr(task_state, "value", task_state)
        sources.append(
            {
                **source.model_dump(mode="json"),
                "title": task.title if task is not None else "",
                "state": task_state_value or source.expected_state.value,
                "assigned_agent_id": (
                    str(task.assigned_agent_id) if task is not None and task.assigned_agent_id else None
                ),
                "is_running": bool(
                    task is not None
                    and (getattr(task, "runtime_state", None) == "running" or task_state_value in {
                        "in_progress", "pausing", "resuming"
                    })
                ),
            }
        )
    proposal_json = proposal.model_dump(mode="json")
    return {
        "session_id": str(session.id),
        "proposal_id": str(proposal.proposal_id),
        "version": proposal.version,
        "rationale": proposal.rationale,
        "created_at": proposal_json["created_at"],
        "sources": sources,
        "create_tasks": proposal_json["create_tasks"],
        "dependency_rewrites": proposal_json["dependency_rewrites"],
        "has_running_source": any(source["is_running"] for source in sources),
        "proposal": proposal_json,
    }


class _LLMTurnResult:
    __slots__ = (
        "text", "tool_input", "tool_name", "tokens_in", "tokens_out", "model",
        "tool_notes", "budget_exhausted", "preflight_rejected_reason",
        "max_window_in", "last_window_in", "externally_stopped",
        "reasoning_text", "reasoning_tokens", "forced_text_fallback",
        "injected_user_messages",
    )

    def __init__(
        self,
        text: str | None,
        tool_input: dict[str, Any] | None,
        tokens_in: int,
        tokens_out: int,
        model: str,
        tool_name: str | None = None,
        tool_notes: list[dict[str, str]] | None = None,
        budget_exhausted: bool = False,
        preflight_rejected_reason: str | None = None,
        externally_stopped: bool = False,
        max_window_in: int = 0,
        last_window_in: int = 0,
        reasoning_text: str = "",
        reasoning_tokens: int = 0,
        forced_text_fallback: bool = False,
    ) -> None:
        self.text = text
        self.tool_input = tool_input
        self.tool_name = tool_name
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model
        # Plain-text mirror of the workspace tool exchange — persisted into the
        # session history so the next turn does not re-read the same files.
        self.tool_notes = tool_notes or []
        # True when the loop stopped because the session token budget ran out.
        self.budget_exhausted = budget_exhausted
        # ``session_budget`` is legacy-only.  Window failures are carried
        # separately so callers never misreport them as a token budget limit.
        self.preflight_rejected_reason = preflight_rejected_reason
        # A workspace pause or a user stop can close the row from another DB
        # session while this coroutine is still awaiting an LLM/tool call.
        self.externally_stopped = externally_stopped
        # Per-call prompt size (stage 5.2): tokens_in is the SUM over a tool
        # loop's calls, which is not the context window.  These two carry the
        # largest and the most recent single-request prompt instead.
        self.max_window_in = max_window_in
        self.last_window_in = last_window_in
        # Stage 6 (§7.1–7.3): the model's own thinking, carried out of the
        # provider call so the *session* decides whether to show it.  The call
        # itself has no room or db in scope, and giving it one just to emit a
        # chat block would put visibility inside the transport.  Empty text
        # with non-zero tokens is a real, measured state (§4A.5 q3): closed
        # reasoners spend tokens and return nothing to display.
        self.reasoning_text = reasoning_text
        self.reasoning_tokens = reasoning_tokens
        # waiting_reply_safeguard S4: the turn only produced a reply because the
        # tool-call budget ran out and a text answer was forced. The agent did
        # not choose to stop, so a task adopted from this turn must not be
        # treated as a finished submission.
        self.forced_text_fallback = forced_text_fallback
        # Messages the user sent while the loop ran, taken in before a model
        # request (session_interject.py). The loop's history copy is discarded,
        # so the caller records them in the session history from here.
        self.injected_user_messages: list[str] = []


async def _session_was_closed_externally(session_id: UUID, db: AsyncSession) -> bool:
    """Return whether another request closed this session while a turn ran.

    A workspace pause uses a different database session.  Selecting the column
    (rather than looking at the ORM object held by this coroutine) deliberately
    bypasses the identity-map snapshot and makes Stop a live cancellation fence.
    """
    row = (
        await db.execute(
            select(Session.id)
            .where(Session.id == session_id, Session.closed_at.isnot(None))
            .limit(1)
        )
    ).scalar_one_or_none()
    # Production PostgreSQL returns UUID.  Keeping this precise also lets
    # lightweight unit-test mocks omit unrelated database behavior.
    return isinstance(row, UUID)


def _openai_tool_choice(
    force_text: bool, force_tool: str | None
) -> str | dict[str, Any]:
    """tool_choice for the OpenAI-compatible wire: none / a named tool / auto."""
    if force_text:
        return "none"
    if force_tool:
        return {"type": "function", "function": {"name": force_tool}}
    return "auto"


async def _call_llm_with_tools_impl(
    llm_config: dict[str, Any],
    system_prompt: str,
    history: list[dict[str, str]],
    matrix_key: str | None = None,
    force_text: bool = False,
    force_tool: str | None = None,
    secret_value: SecretValue | None = None,
) -> _LLMTurnResult:
    """Call the LLM with session tools for the current direct-session kind.

    force_text=True forbids tool calls (``tool_choice=none``) so the model can
    only return plain text.  Use this when a tool call would be premature (e.g.
    first CM turn).  The tool schemas stay in the request on purpose: dropping
    them leaves the provider without a tool-call parser, and a model that still
    wants a tool then emits its raw template markup as prose.

    force_tool="propose_plan" is the opposite pressure and the one the
    "Form a plan" button uses: the model must answer with that tool call
    on this turn instead of asking one more clarifying question.  Endpoints that
    reject a named tool_choice fall back to the unconstrained call below, where
    the XML/JSON text fallback still catches the plan.

    Returns a _LLMTurnResult with either text (normal reply) or tool_input
    (the agent called a tool) — never both. tool_name identifies which tool.
    """
    history = _modernize_tool_notes(history)
    provider = (llm_config.get("provider") or "anthropic").lower()
    model = llm_config.get("model") or "claude-haiku-4-5-20251001"
    api_key = secret_value.reveal() if secret_value is not None else None
    if not api_key:
        legacy = legacy_secret_value(llm_config, "api_key")
        try:
            api_key = legacy.reveal() if legacy is not None else None
        finally:
            legacy = None
    base_url = llm_config.get("base_url")

    logger.info(
        "_call_llm_with_tools: provider=%s model=%s base_url=%s has_key=%s",
        provider, model, base_url, bool(api_key),
    )

    anthropic_tool = (
        close_tool_for_kind(matrix_key, "anthropic") if matrix_key else ANTHROPIC_CLOSE_TOOL
    )
    openai_tool = (
        close_tool_for_kind(matrix_key, "openai") if matrix_key else OPENAI_CLOSE_TOOL
    )

    # Each confirmation flow is deliberately scoped to its session kind.
    include_propose_plan = matrix_key == "user_cm_p2" or force_tool == "propose_plan"
    include_offer_plan = matrix_key == "user_cm_p2" and not force_tool
    include_task_change = matrix_key == "user_cm_p1"

    if uses_messages_wire(provider, model):
        if uses_anthropic_env_key(provider):
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        tools_payload: list[dict[str, Any]] = [anthropic_tool]
        if include_propose_plan:
            tools_payload.append(PROPOSE_PLAN_TOOL_ANTHROPIC)
        if include_offer_plan:
            tools_payload.append(OFFER_PLAN_TOOL_ANTHROPIC)
        if include_task_change:
            tools_payload.append(PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC)
        body: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": history,
            "tools": tools_payload,
            "max_tokens": 4096,
        }
        if force_text:
            body["tool_choice"] = {"type": "none"}
        elif force_tool:
            body["tool_choice"] = {"type": "tool", "name": force_tool}
        apply_reasoning_body(body, llm_config, provider, model)
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = messages_headers(provider, api_key)
            response = await timed_llm(
                client.post(messages_url(provider, base_url), headers=headers, json=body),
                provider=provider,
                side=SIDE_CHAT_MASTER,
            )
            if response.status_code == 400 and "tool_choice" in response.text:
                # Endpoint (or proxy) does not know this tool_choice form.  Retry
                # without it rather than dropping `tools`: keeping the schemas
                # is what stops the model from answering in raw template
                # markup, and the only tool on offer here is close_session,
                # which callers already handle.
                logger.warning(
                    "user_sessions: anthropic endpoint rejected tool_choice %s — "
                    "retrying with tools but no tool_choice",
                    body.get("tool_choice"),
                )
                body.pop("tool_choice", None)
                response = await timed_llm(
                    client.post(messages_url(provider, base_url), headers=headers, json=body),
                    provider=provider,
                    side=SIDE_CHAT_MASTER,
                )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            _record_provider_cache(
                llm_config,
                usage,
                role="agent" if str(matrix_key or "").startswith("user_agent") else "chat_master",
            )
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            text_all, thinking = messages_text_and_thinking(data)
            reasoning = {
                "reasoning_text": thinking,
                "reasoning_tokens": messages_reasoning_tokens(data),
            }

            if data.get("stop_reason") == "tool_use":
                tool_block = next(
                    (b for b in data.get("content", []) if b.get("type") == "tool_use"),
                    None,
                )
                return _LLMTurnResult(
                    None,
                    tool_block["input"] if tool_block else {},
                    tokens_in,
                    tokens_out,
                    model,
                    tool_name=tool_block.get("name") if tool_block else None,
                    **reasoning,
                )

            return _LLMTurnResult(
                clean_reply_text(text_all, where="_call_llm_with_tools/anthropic"),
                None, tokens_in, tokens_out, model, **reasoning,
            )

    # OpenAI-compatible (openai, groq, deepseek, ollama, openrouter, …)
    api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-no-key-required"
    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
    client_kwargs["base_url"] = openai_base_url(provider, base_url)
    client_kwargs.update(openai_client_extras(client_kwargs["base_url"]))

    logger.info("_call_llm_with_tools: effective base_url=%s", client_kwargs.get("base_url"))
    oai = llm_client(provider, chat_client_factory=AsyncOpenAI, **client_kwargs)
    messages = [{"role": "system", "content": system_prompt}] + list(history)
    oai_tools: list[dict[str, Any]] = [openai_tool]
    if include_propose_plan:
        oai_tools.append(PROPOSE_PLAN_TOOL_OPENAI)
    if include_offer_plan:
        oai_tools.append(OFFER_PLAN_TOOL_OPENAI)
    if include_task_change:
        oai_tools.append(PROPOSE_TASK_CHANGE_TOOL_OPENAI)
    # Malformed tool-call JSON is a transient model failure — retry once.
    for attempt in range(2):
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": oai_tools,
            "tool_choice": _openai_tool_choice(force_text, force_tool),
        }
        # Called at the assembly, so the tool_choice-less retry below inherits
        # the same reasoning fields instead of quietly losing them.
        apply_reasoning_chat(create_kwargs, llm_config, provider, model)
        try:
            resp = await timed_llm(
                oai.chat.completions.create(**create_kwargs),
                provider=provider,
                side=SIDE_CHAT_MASTER,
            )
        except BadRequestError as exc:
            if not (force_text or force_tool) or "tool_choice" not in str(exc):
                raise
            # Endpoint does not know tool_choice=none.  Retry without it rather
            # than dropping `tools`: keeping the schemas is what stops the model
            # from answering in raw template markup, and the only tool on offer
            # here is close_session, which callers already handle.
            logger.warning(
                "_call_llm_with_tools: endpoint rejected tool_choice %r (%s) — "
                "retrying with tools but no tool_choice",
                create_kwargs.get("tool_choice"), exc,
            )
            create_kwargs.pop("tool_choice")
            resp = await timed_llm(
                oai.chat.completions.create(**create_kwargs),
                provider=provider,
                side=SIDE_CHAT_MASTER,
            )
        choice = resp.choices[0]
        usage = resp.usage
        _record_provider_cache(
            llm_config,
            usage,
            role="agent" if str(matrix_key or "").startswith("user_agent") else "chat_master",
        )
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        _thinking, _reasoning_tokens = reasoning_from_chat_choice(choice.message, usage)
        reasoning = {"reasoning_text": _thinking, "reasoning_tokens": _reasoning_tokens}

        logger.info(
            "_call_llm_with_tools: finish_reason=%s has_tool_calls=%s content_len=%d",
            choice.finish_reason,
            bool(choice.message.tool_calls),
            len(choice.message.content or ""),
        )
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            try:
                tool_input = parse_tool_arguments(tool_call.function.arguments)
            except ToolArgumentsParseError:
                if attempt == 0:
                    logger.warning(
                        "user_sessions: malformed tool JSON — retrying LLM call"
                    )
                    continue
                raise
            return _LLMTurnResult(
                None, tool_input, tokens_in, tokens_out, model,
                tool_name=tool_call.function.name,
                **reasoning,
            )

        cleaned, leaked = strip_tool_call_markup(choice.message.content or "")
        if leaked:
            logger.warning(
                "user_sessions: stripped raw tool-call markup from "
                "_call_llm_with_tools/openai reply (force_text=%s) — %d chars left",
                force_text, len(cleaned),
            )
            # Nothing but markup: the model answered in template instead of
            # prose.  One retry is cheaper than shipping a broken reply.
            if not cleaned and attempt == 0:
                continue
            if not cleaned:
                cleaned = TOOL_MARKUP_FALLBACK_TEXT
        return _LLMTurnResult(cleaned, None, tokens_in, tokens_out, model, **reasoning)


# ---------------------------------------------------------------------------
# Post-close: Goal → Plan → cm_self_plan pipeline (§4)
# ---------------------------------------------------------------------------


async def _resolve_room_agent(
    token: str, room_id: UUID, db: AsyncSession
) -> UUID | None:
    """Resolve a propose_plan assignee token to an agent id scoped to the room.

    Accepts a UUID string or an ``@nickname`` / bare nickname (case-insensitive).
    Returns None for an empty token or one that matches no agent in the room —
    the step is then created unassigned and cm_self_plan picks the agent.
    """
    from sqlalchemy import select as sa_select

    from app.models.agent import Agent
    from app.models.room import RoomAgentLink

    token = (token or "").strip().lstrip("@").strip()
    if not token:
        return None

    try:
        as_uuid: UUID | None = UUID(token)
    except (ValueError, AttributeError):
        as_uuid = None

    rows = (
        await db.execute(
            sa_select(Agent.id, Agent.nickname)
            .join(RoomAgentLink, Agent.id == RoomAgentLink.agent_id)
            .where(RoomAgentLink.room_id == room_id)
        )
    ).all()
    if as_uuid is not None:
        return next((aid for aid, _ in rows if aid == as_uuid), None)
    folded = token.casefold()
    return next((aid for aid, nick in rows if (nick or "").casefold() == folded), None)


async def _trigger_plan_pipeline(
    room_id: UUID,
    actions: SessionCloseActions,
    db: AsyncSession,
    *,
    plan_steps: list[dict[str, Any]] | None = None,
    requirements_payload: dict[str, list[dict[str, Any]]] | None = None,
    session: Session | None = None,
) -> None:
    """After a user_cm_p2 session creates Goals, auto-create Plans and dispatch cm_self_plan.

    For each newly created Goal:
      1. Find the RoomGoal record (by title match on the just-flushed rows).
      2. Create GoalPlan steps: one per entry in *plan_steps* (with the
         assignee resolved to a room agent), or a single step derived from the
         Goal description when the proposal carried no steps.
      3. Emit PLAN_CREATED event.
      4. Dispatch plan_session background task to decompose into tasks.

    *plan_steps* comes from the accepted propose_plan payload (normalized by
    ``_plan_steps_from_proposal``). It is only applied when exactly one goal is
    being created — the steps describe that one goal. *requirements_payload*
    (normalized by ``_requirements_from_proposal``) is persisted the same way,
    via ``requirements.persist_requirements`` (Stage 3).
    """
    from sqlalchemy import select as sa_select

    from app.core.plan_session import dispatch_plan_event, render_plan_text_with_agents
    from app.models.goal import RoomGoal
    from app.models.goal_plan import GoalPlan, GoalPlanStatus, GoalPlanStep

    use_steps = bool(plan_steps) and len(actions.goals_create) == 1

    for spec in actions.goals_create:
        goal_result = await db.execute(
            sa_select(RoomGoal)
            .where(RoomGoal.room_id == room_id, RoomGoal.title == spec.title)
            .order_by(RoomGoal.created_at.desc())
            .limit(1)
        )
        goal = goal_result.scalars().first()
        if goal is None:
            logger.warning(
                "user_sessions: could not find just-created goal '%s' in room %s",
                spec.title, room_id,
            )
            continue

        if session is not None and getattr(goal, "origin_session_id", None) is None:
            goal.origin_session_id = session.id
            db.add(goal)

        # internal design notes Stage 3: persist requirements before
        # PLAN_CREATED so cm_self_plan's decomposition never starts without them.
        if requirements_payload and len(actions.goals_create) == 1:
            try:
                from app.core.requirements import persist_requirements

                await persist_requirements(goal, requirements_payload, db, session=session)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "user_sessions: persist_requirements failed for goal %s: %s",
                    goal.id, exc,
                )

        # The human already approved this plan — that acceptance is what got
        # us here — and decomposition is dispatched at the bottom of this
        # loop. Storing it as a draft left the Goals panel showing an Approve
        # button for a plan whose tasks were already on the board; pressing it
        # dispatched a second decomposition and duplicated every task.
        plan = GoalPlan(
            goal_id=goal.id,
            status=GoalPlanStatus.APPROVED,
            approved_at=datetime.utcnow(),
        )
        db.add(plan)
        await db.flush()

        steps: list[GoalPlanStep] = []
        if use_steps:
            for i, st in enumerate(plan_steps or []):
                agent_id = await _resolve_room_agent(
                    st.get("assignee", ""), room_id, db
                )
                if st.get("assignee") and agent_id is None:
                    logger.info(
                        "user_sessions: plan step %r names assignee %r with no "
                        "matching room agent — leaving it unassigned",
                        st.get("title"), st.get("assignee"),
                    )
                steps.append(
                    GoalPlanStep(
                        plan_id=plan.id,
                        title=st["title"],
                        description=st.get("description") or "",
                        position=i,
                        assigned_agent_id=agent_id,
                    )
                )
        else:
            steps.append(
                GoalPlanStep(
                    plan_id=plan.id,
                    title=spec.title,
                    description=spec.description or goal.description or "",
                    position=0,
                )
            )
        for s in steps:
            db.add(s)
        await db.flush()

        plan_text = await render_plan_text_with_agents(plan, steps, db)

        await emit_event(
            room_id=room_id,
            event_type="PLAN_CREATED",
            content=f"Plan {plan.id} created for goal '{goal.title}'",
            db=db,
            metadata={
                "plan_id": str(plan.id),
                "goal_id": str(goal.id),
                "goal_title": goal.title,
            },
        )
        await db.commit()

        # internal design notes Stage 2: after the Goal is durably
        # committed, materialize the task-formation dialogue to
        # ``.legion/plans/goal-{id}.md``. Best-effort — a failed archive must
        # not undo a committed Goal.
        if session is not None:
            try:
                from app.core import plan_dialogue

                snap = getattr(session, "context_snapshot", None) or {}
                pending_plan = snap.get("pending_plan") if isinstance(snap, dict) else None
                # §8.1: the digest runs on the same model as decomposition — the
                # ChatMaster's own, plus the planner_model override.
                try:
                    from app.core.room_coordinator import resolve_planner_llm_config
                    from app.models.room import Room as _Room

                    _room = await db.get(_Room, room_id)
                    _cm_agent, llm_config = (
                        await resolve_planner_llm_config(_room, db)
                        if _room is not None
                        else (None, {})
                    )
                    llm_config = llm_config or None
                except Exception:
                    llm_config = None
                await plan_dialogue.archive_for_goal(
                    room_id,
                    goal.id,
                    goal.title,
                    session,
                    db,
                    pending_plan=pending_plan,
                    llm_config=llm_config,
                )
                # L1: a Goal landed — drop any lingering draft head pointers for
                # this user so the successor session stops being offered them.
                await plan_dialogue.clear_draft_heads(
                    room_id, getattr(session, "user_id", None)
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "user_sessions: plan-dialogue archive failed for goal %s: %s",
                    goal.id, exc,
                )

        dispatch_plan_event(room_id, goal.id, plan.id, plan_text)
        logger.info(
            "user_sessions: dispatched plan_session for goal %s, plan %s",
            goal.id, plan.id,
        )


# ---------------------------------------------------------------------------
# Close flow
# ---------------------------------------------------------------------------


_OPEN_QUESTION_RE = re.compile(
    r"(?:\?|would\s+you\s+like\b|do\s+you\s+want\b|(?:please\s+)?confirm|"
    r"(?:shall|should)\s+i\s+continue|as\s+soon\s+as\s+the\s+tool|"
    r"(?:still\s+)?left\s+to\s+(?:finish|do)|shall\s+i\b|should\s+i\b|let\s+me\s+know)",
    re.IGNORECASE,
)


def _looks_like_open_question(text: str) -> bool:
    """True when the agent's closing reply reads as waiting for a user answer."""
    tail = (text or "").strip()
    if not tail:
        return False
    # Look only at the last ~2 sentences — a mid-reply "?" in an explanation is
    # not the agent handing the turn back.
    tail = tail[-500:]
    return tail.rstrip().endswith("?") or bool(_OPEN_QUESTION_RE.search(tail))


# project_cm_p2_close_midgathering (incident 2026-09-08): a user_cm_p2 turn that
# fetched a commit, formulated two clarifying questions and called
# close_session(task_done) with the questions buried in the summary. The session
# was gone, so the user's next message cold-routed to a fresh ChatMaster that saw
# a finished board. These two helpers let the close flow tell "the work is done"
# apart from "my analysis turn is done".
_CM_TOPIC_DROP_RE = re.compile(
    r"(?:no\s+need\b|not\s+needed\b|cancel\b|skip\s+(?:it|this)|"
    r"doesn'?t\s+matter|not\s+important|changed\s+my\s+mind|"
    r"let'?s\s+(?:do\s+(?:it\s+)?)?(?:later|another\s+time)|(?:another|different|other)\s+topic|"
    r"change\s+(?:the\s+)?(?:topic|subject)|never\s*mind|forget\s+(?:it|this|that)|drop\s+(?:it|this)|"
    r"scratch\s+that)",
    re.IGNORECASE,
)


def _user_dropped_topic(text: str) -> bool:
    """True when the user's message reads as 'stop, drop this' — not an answer."""
    return bool(_CM_TOPIC_DROP_RE.search((text or "")[:400]))


def _cm_p2_close_is_premature(
    session: Session,
    tool_input: dict[str, Any] | None,
    history: list[dict[str, str]],
) -> bool:
    """A ``user_cm_p2`` ``close_session`` that abandons requirement-gathering.

    Legitimate closes for this kind carry a plan (proposed earlier in the
    session, or ``goals_create`` in the close itself) or follow the user
    dropping the topic. A close that has asked the user something, proposed no
    plan, creates no goals and is not answering a "forget it" is the model
    mistaking a finished analysis turn for a finished task — and the session is
    the only place the pending questions live.
    """
    if session.kind != "user_cm_p2" or tool_input is None:
        return False

    snap = session.context_snapshot or {}
    if snap.get("pending_plan"):
        return False
    if any(
        m.get("role") == "assistant"
        and "Proposing a plan" in str(m.get("content") or "")
        for m in history
    ):
        return False

    # Only a session that actually put a question to the user can be abandoning
    # one. A plain Q&A that ends with the user saying "thanks" closes fine.
    cm_asked = any(
        m.get("role") == "assistant" and "?" in str(m.get("content") or "")
        for m in history
    )
    if not cm_asked:
        return False

    try:
        actions = extract_close_actions_from_tool_use(tool_input)
    except Exception:
        return True
    if actions.goals_create:
        return False

    last_user = next(
        (
            str(m.get("content") or "")
            for m in reversed(history)
            if m.get("role") == "user"
        ),
        "",
    )
    return not _user_dropped_topic(last_user)


async def _finalize_session_work(
    session: Session,
    db: AsyncSession,
    *,
    reason: str,
    user_request: str | None = None,
    agent_reply: str | None = None,
    incomplete: bool = False,
) -> None:
    """Run the dialogue's Gate 1 (internal design notes), never fatally.

    The turn is already answered by the time this runs, so a git or workspace
    problem here costs the branch its gate — not the conversation.

    ``incomplete`` (waiting_reply_safeguard S4): the turn ended on a forced
    text reply after the tool budget ran out — the work is not a finished
    submission, so the task is parked BLOCKED for the user instead of being
    sent through the review gate.
    """
    try:
        from app.core.session_work import finalize_session_work

        await finalize_session_work(
            session, db, reason=reason, user_request=user_request,
            agent_reply=agent_reply, incomplete=incomplete,
        )
    except Exception as exc:
        logger.warning(
            "user_sessions: could not finalize dialogue work for session %s (%s): %s",
            session.id, reason, exc, exc_info=True,
        )


USER_STOP_REASON = "user_stop"

# History marker for a turn the user cut short — the next turn must know why the
# last one has no reply.
_USER_STOP_NOTE = "[The user stopped this turn. The work done so far was saved.]"


async def _end_turn_on_user_stop(
    session: Session,
    user_request: str | None,
    result: "_LLMTurnResult",
    history: list[dict[str, Any]],
    db: AsyncSession,
) -> bool:
    """Handle the Stop button for a turn that has just unwound; True if it was one.

    The user presses Stop because they already have what they wanted (or enough
    of it), so the stop must not lose the files the agent wrote. They go through
    the same road as a turn that ended on its own: committed on the session
    branch, recorded as a task, then Gate 1. The session stays open — this ends
    the turn, not the conversation.
    """
    if not session_interject.consume_stop(session.id):
        return False
    notes = getattr(result, "tool_notes", None)
    if isinstance(notes, list):
        history.extend(notes)
    history.append({"role": "assistant", "content": _USER_STOP_NOTE})
    _save_history(session, history)
    session.last_activity_at = datetime.now(timezone.utc)
    db.add(session)
    await db.flush()
    await _finalize_session_work(
        session, db, reason=USER_STOP_REASON, user_request=user_request,
    )
    return True


async def stop_dialogue_turn(room_id: UUID, user_id: UUID, db: AsyncSession) -> dict[str, Any]:
    """The Stop button: end the user's running dialogue turn in this room.

    ``stopped`` — a turn was running and has been told to wind down at its next
    boundary (a tool call already in flight finishes first). ``finalized`` — the
    session was idle but still held unrecorded work (a turn that died without its
    gate), which is recorded now. ``idle`` — nothing to stop.
    """
    from app.repositories.session import SessionRepository

    session = await SessionRepository(db).find_open_user_session(room_id, user_id)
    if session is None:
        return {"status": "idle"}
    if turn_in_progress(session.id):
        session_interject.request_stop(session.id)
        logger.info("stop_dialogue_turn: stop requested for session %s", session.id)
        return {"status": "stopped", "session_id": str(session.id)}
    await _finalize_session_work(session, db, reason=USER_STOP_REASON)
    await db.commit()
    return {"status": "finalized", "session_id": str(session.id)}


async def _run_close_flow(
    session: Session,
    tool_input: dict[str, Any],
    db: AsyncSession,
    *,
    llm_config: dict[str, Any],
    system_prompt: str,
    history: list[dict[str, str]],
) -> None:
    """Parse close actions, retry once if disallowed fields detected, then close.

    If the LLM returned fields forbidden for this session kind:
    - First occurrence: inject an error turn and call the LLM again (one retry).
    - If retry also returns forbidden fields, or returns text instead of a tool call:
      emit SESSION_CLOSE_ACTIONS_DROPPED and close with whatever filtered actions remain.
    After closing, publish the summary as a visible MESSAGE if non-empty (3.4).
    """
    try:
        actions = extract_close_actions_from_tool_use(tool_input)
    except Exception as exc:
        logger.error(
            "user_sessions: invalid close_session input for session %s: %s\ntool_input: %s",
            session.id,
            exc,
            json.dumps(tool_input, default=str, ensure_ascii=False)[:1000],
        )
        from app.core.metrics import record_session_close_invalid
        record_session_close_invalid(kind=session.kind)
        await emit_event(
            room_id=session.room_id,
            event_type="SESSION_CLOSE_ACTIONS_INVALID",
            content=f"Session {session.id} returned invalid close_session input.",
            db=db,
            metadata={"session_id": str(session.id), "error": str(exc)},
        )
        actions = SessionCloseActions()

    _, dropped = filter_by_kind(actions, session)

    if dropped:
        snap = dict(session.context_snapshot or {})
        retry_count = snap.get("close_retry_count", 0)

        if retry_count == 0:
            snap["close_retry_count"] = 1
            session.context_snapshot = snap
            _flag_snapshot_dirty(session)
            db.add(session)
            await db.flush()

            fields_str = ", ".join(dropped)
            error_text = (
                f"Your actions {{{fields_str}}} are not allowed for this type of session and were rejected. "
                "Formulate this work as tasks via tasks_create (allowed) and call close_session again."
            )
            retry_history = list(history) + [
                {
                    "role": "assistant",
                    "content": f"[close_session attempt with dropped fields: {fields_str}]",
                },
                {"role": "user", "content": error_text},
            ]

            mk = _matrix_key(session)
            await release_db(db, reason="user_session/close_retry")
            retry_result = await _call_llm_with_tools(
                llm_config, system_prompt, retry_history, matrix_key=mk
            )

            if retry_result.tool_input is not None:
                try:
                    actions = extract_close_actions_from_tool_use(retry_result.tool_input)
                except Exception:
                    actions = SessionCloseActions()
                _, retry_dropped = filter_by_kind(actions, session)
                if retry_dropped:
                    await emit_event(
                        room_id=session.room_id,
                        event_type="SESSION_CLOSE_ACTIONS_DROPPED",
                        content=(
                            f"Some actions of session {session.id} were rejected: {', '.join(retry_dropped)}"
                        ),
                        db=db,
                        metadata={
                            "session_id": str(session.id),
                            "kind": session.kind,
                            "dropped": retry_dropped,
                        },
                    )
            else:
                # LLM returned text on retry: keep original actions (disallowed fields will
                # be stripped by apply_close_actions) and notify the room.
                await emit_event(
                    room_id=session.room_id,
                    event_type="SESSION_CLOSE_ACTIONS_DROPPED",
                    content=(
                        f"Some actions of session {session.id} were rejected: {', '.join(dropped)}"
                    ),
                    db=db,
                    metadata={
                        "session_id": str(session.id),
                        "kind": session.kind,
                        "dropped": dropped,
                    },
                )
        else:
            # Retry counter already set: this is the re-entrant path after a retry (should
            # not normally happen since we don't recurse), or a bug guard.
            await emit_event(
                room_id=session.room_id,
                event_type="SESSION_CLOSE_ACTIONS_DROPPED",
                content=f"Some actions of session {session.id} were rejected: {', '.join(dropped)}",
                db=db,
                metadata={
                    "session_id": str(session.id),
                    "kind": session.kind,
                    "dropped": dropped,
                },
            )

    # One dialogue, one kind of work: the close tool is the other door to goal
    # creation (the Accept button is guarded in api/v1/chat.py). A dialogue that
    # already dispatched quick goal-less tasks must not walk a goal through it.
    if actions.goals_create:
        _quick_tasks = await quick_tasks_for_session(session.id, db)
        if _quick_tasks:
            logger.warning(
                "_run_close_flow: dropped %d goal(s) from the close of session %s — "
                "%d quick task(s) from this dialogue are still running",
                len(actions.goals_create), session.id, len(_quick_tasks),
            )
            actions.goals_create = []
            await emit_event(
                room_id=session.room_id,
                event_type="SESSION_CLOSE_ACTIONS_DROPPED",
                content=(
                    f"The goal from session {session.id} was not created: this discussion already "
                    "has quick tasks running without a goal."
                ),
                db=db,
                metadata={
                    "session_id": str(session.id),
                    "kind": session.kind,
                    "dropped": ["goals_create"],
                    "reason": "quick_tasks_in_dialogue",
                    "quick_task_ids": [str(t.id) for t in _quick_tasks],
                },
            )

    await apply_close_actions(session, actions, db)

    history.append({"role": "assistant", "content": "[session closed]"})
    _save_history(session, history)
    session.closed_at = datetime.now(timezone.utc)
    session.close_reason = "task_done"
    db.add(session)
    await db.flush()

    # §4.1: no-op if `actions` already carries an agent_memory_patch — this is
    # the voluntary close path, so that is the common case.
    await flush_session_memory(session, db, reason="task_done", applied_actions=actions)

    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_CLOSED",
        content=f"Session {session.id} closed (kind={session.kind}, reason=task_done)",
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": session.kind,
            "close_reason": "task_done",
            "summary": actions.summary,
        },
    )

    # 3.4: Publish summary as a visible message so users see what was accomplished
    if actions.summary:
        agent_id: UUID | None = getattr(session, "agent_id", None) or getattr(
            session, "chat_master_agent_id", None
        )
        summary_msg = Message(
            room_id=session.room_id,
            sender_agent_id=agent_id,
            event_type="MESSAGE",
            content=actions.summary,
            event_metadata={
                "session_id": str(session.id),
                "kind": session.kind,
                "is_summary": True,
            },
        )
        db.add(summary_msg)
        await db.flush()
        await manager.broadcast(
            {
                "type": "message",
                "event_type": "MESSAGE",
                "id": str(summary_msg.id),
                "room_id": str(session.room_id),
                "sender_id": str(agent_id) if agent_id else None,
                "sender_type": "agent",
                "content": actions.summary,
                "timestamp": summary_msg.timestamp.isoformat(),
                "status": summary_msg.status,
                "session_id": str(session.id),
            },
            session.room_id,
        )

    # The agent closed the conversation itself — the last turn may have written
    # something that never reached a turn-end finalization.
    await _finalize_session_work(
        session, db, reason="session_closed", agent_reply=actions.summary
    )

    # L1: a voluntary close that produced neither a Goal nor a Task is still a
    # formation dialogue worth keeping — most notably a premature
    # close_session(task_done) mid-gathering (project_cm_p2_close_midgathering).
    if not actions.goals_create and not actions.tasks_create:
        await _maybe_save_dialogue_draft(session, db, close_reason="task_done")

    # Post-close: if user_cm_p2 created goals, auto-create Plans and trigger pipeline
    if session.kind == "user_cm_p2" and actions.goals_create:
        await _trigger_plan_pipeline(session.room_id, actions, db, session=session)


# ---------------------------------------------------------------------------
# Token-limit close: save memory so the next session can pick up the context
# ---------------------------------------------------------------------------

_TOKEN_LIMIT_CLOSE_PROMPT = (
    "The session token budget is exhausted — the session is closing RIGHT NOW. "
    "Call close_session immediately and save the context for the next session:\n"
    "- agent_memory_patch: all accepted user decisions, the current state "
    "of the work, key facts about the project and a concrete next step;\n"
    "- room_memory_patch (if allowed): key project decisions;\n"
    "- summary: a short report on the state of the dialogue.\n"
    "Do NOT ask questions and do NOT write a regular reply — only close_session."
)


async def _final_memory_save(
    session: Session,
    llm_config: dict[str, Any],
    system_prompt: str,
    history: list[dict[str, Any]],
    db: AsyncSession,
    *,
    instruction: str,
    where: str,
) -> SessionCloseActions:
    """One last close_session call so an involuntary close keeps its memory.

    Shared by the legacy token-limit close and the stage-1 rotation: in both
    cases the model did not choose to end the session, so nothing else would
    write agent memory, room memory or a summary. Never raises — the session is
    closing regardless of what the provider does.
    """
    mk = _matrix_key(session)
    actions = SessionCloseActions()

    # Only worth an LLM call if the conversation actually happened.
    if any(m.get("role") == "assistant" for m in history):
        close_history = [dict(m) for m in history]
        # Keep user/assistant alternation: merge the instruction into a trailing
        # user message if there is one, else append a new user turn.
        if close_history and close_history[-1].get("role") == "user":
            close_history[-1]["content"] = (
                str(close_history[-1]["content"]) + "\n\n" + instruction
            )
        else:
            close_history.append({"role": "user", "content": instruction})
        try:
            await release_db(db, reason="user_session/final_memory_save")
            res = await _call_llm_with_tools(
                llm_config, system_prompt, close_history, matrix_key=mk
            )
            track_llm_usage(
                session, res.tokens_in, res.tokens_out,
                _compute_cost(res.model, res.tokens_in, res.tokens_out, llm_config),
                model=res.model,
            )
            update_context_window_metrics(session, res.max_window_in, res.last_window_in)
            if res.tool_input is not None and res.tool_name in (None, "close_session"):
                try:
                    actions = extract_close_actions_from_tool_use(res.tool_input)
                except Exception as exc:
                    logger.warning(
                        "%s close: invalid close actions for session %s: %s",
                        where, session.id, exc,
                    )
            elif (res.text or "").strip():
                actions = SessionCloseActions(summary=res.text.strip()[:4000])
        except Exception as exc:
            logger.error(
                "%s close: memory-save LLM call failed for session %s: %s",
                where, session.id, exc,
            )

    try:
        await apply_close_actions(session, actions, db)
    except Exception as exc:
        logger.error(
            "%s close: apply_close_actions failed for session %s: %s",
            where, session.id, exc,
        )
    return actions


async def _handle_token_limit_close(
    session: Session,
    llm_config: dict[str, Any],
    system_prompt: str,
    history: list[dict[str, Any]],
    db: AsyncSession,
) -> None:
    """Close a session that ran out of token budget WITHOUT losing context.

    Makes one final LLM call asking for close_session (agent/room memory patch +
    summary), applies the actions, and posts a user-visible message explaining
    that the next message will continue in a fresh session with saved memory.
    All failures are logged, never raised — the session is already closed.
    """
    actions = await _final_memory_save(
        session, llm_config, system_prompt, history, db,
        instruction=_TOKEN_LIMIT_CLOSE_PROMPT,
        where="token_limit",
    )
    # §4.1: no-op if the final close_session call above already wrote
    # agent_memory_patch; otherwise reuses actions.summary / falls back.
    await flush_session_memory(session, db, reason="token_limit", applied_actions=actions)

    if not actions.goals_create and not actions.tasks_create:
        await _maybe_save_dialogue_draft(session, db, close_reason="token_limit")

    history.append({"role": "assistant", "content": "[session closed: token limit]"})
    _save_history(session, history)
    db.add(session)
    await db.flush()

    agent_id: UUID | None = getattr(session, "agent_id", None) or getattr(
        session, "chat_master_agent_id", None
    )
    notice = (
        "⚠️ The session reached the token limit and was closed. "
        "Key context was saved to memory — just continue the dialogue, "
        "a new session will pick it up."
    )
    if actions.summary:
        notice += "\n\n**State at the time of closing:**\n" + actions.summary
    try:
        msg = Message(
            room_id=session.room_id,
            sender_agent_id=agent_id,
            event_type="MESSAGE",
            content=notice,
            event_metadata={
                "session_id": str(session.id),
                "kind": session.kind,
                "is_summary": True,
                "close_reason": "token_limit",
            },
        )
        db.add(msg)
        await db.flush()
        await manager.broadcast(
            {
                "type": "message",
                "event_type": "MESSAGE",
                "id": str(msg.id),
                "room_id": str(session.room_id),
                "sender_id": str(agent_id) if agent_id else None,
                "sender_type": "agent",
                "content": notice,
                "timestamp": msg.timestamp.isoformat(),
                "status": msg.status,
                "session_id": str(session.id),
            },
            session.room_id,
        )
    except Exception as exc:
        logger.error(
            "token_limit close: failed to publish notice for session %s: %s",
            session.id, exc,
        )


# ---------------------------------------------------------------------------
# Session rotation (chatflow_session_lifecycle_repair_plan stage 1.3)
# ---------------------------------------------------------------------------

_ROTATION_CLOSE_PROMPT = (
    "This session reached the length limit and is closing right now; the conversation with "
    "the user will continue in a new session.\n"
    "Call close_session immediately and save the context for it:\n"
    "- agent_memory_patch: accepted user decisions, the current state "
    "of the work, key facts about the project and a concrete next step;\n"
    "- room_memory_patch (if allowed): key project decisions;\n"
    "- summary: a short report on the state of the dialogue — the next session will read it.\n"
    "Do NOT ask questions and do NOT write a regular reply — only close_session."
)

_CARRYOVER_SUMMARY_MAX_CHARS = 2_000


def _deterministic_summary(history: list[dict[str, Any]]) -> str:
    """Last-resort summary when the provider gave us nothing.

    Rotation must never hand the successor session an empty context, and the
    graceful close is exactly the moment a provider outage would do that.
    """
    replies = [
        str(m.get("content") or "").strip()
        for m in history
        if m.get("role") == "assistant" and str(m.get("content") or "").strip()
    ]
    if not replies:
        return f"Session closed automatically. Messages in the feed: {len(history)}."
    tail = "\n\n---\n\n".join(replies[-3:])
    header = (
        f"Session closed automatically. Messages in the feed: {len(history)}. "
        "Last assistant replies:\n\n"
    )
    return (header + tail)[:_CARRYOVER_SUMMARY_MAX_CHARS]


def _carryover_prefix(previous: Session) -> str:
    """Bridge text prepended to the first user message of a successor session.

    The successor gets memory-layer bricks plus this summary — never the raw
    transcript of the session it replaced (§6.3 of the plan).
    """
    summary = (previous.summary or "").strip()[:_CARRYOVER_SUMMARY_MAX_CHARS]
    if not summary:
        return ""
    return (
        "[Conversation continued: the previous session was closed at the length limit. "
        f"Its result:\n{summary}]\n\n"
    )


async def _context_window_limit(session: Session, db: AsyncSession) -> int:
    """Enforced per-call window budget for this session's role and agent."""
    agent_id: UUID | None = getattr(session, "agent_id", None) or getattr(
        session, "chat_master_agent_id", None
    )
    role = "agent" if str(session.kind).startswith("user_agent") else "chat_master"
    policy = await resolve_context_policy(
        role=role,
        db=db,
        room_id=session.room_id,
        agent_id=agent_id,
        llm_config=None,
    )
    return policy.context_window_limit


async def _successor_session(session: Session, db: AsyncSession) -> Session | None:
    """Create a fresh session row of the same kind for the same participants."""
    user_id: UUID | None = getattr(session, "user_id", None)
    if user_id is None:
        return None

    kind = session.kind
    payload: UserAgentSessionCreate | UserCmSessionCreate
    if kind in ("user_agent_p1", "user_agent_p2"):
        payload = UserAgentSessionCreate(
            kind=kind,
            room_id=session.room_id,
            user_id=user_id,
            agent_id=session.agent_id,  # type: ignore[attr-defined]
            task_id=getattr(session, "task_id", None),
            goal_id=getattr(session, "goal_id", None),
        )
    elif kind in ("user_cm_p1", "user_cm_p2"):
        payload = UserCmSessionCreate(
            kind=kind,
            room_id=session.room_id,
            user_id=user_id,
            chat_master_agent_id=session.chat_master_agent_id,  # type: ignore[attr-defined]
            goal_id=getattr(session, "goal_id", None),
        )
    else:
        return None

    return await SessionRepository(db).create(payload)


async def close_user_session_gracefully(
    session: Session,
    db: AsyncSession,
    *,
    rotation_reason: str,
    close_reason: str = "rotated",
) -> str:
    """Close a user session with one final memory-saving LLM call. Returns the summary.

    Used by the stage-1 rotation and by the stage-2 router path (the user
    changed topic, so the open session ends here). The session is closed either
    way: if the provider is unavailable, the summary is built deterministically
    from the last assistant replies rather than left empty — an empty summary is
    a lost conversation.
    """
    llm_config = await _get_agent_llm_config(session, db)
    memory_layer = await build_memory_layer(session, db)
    system_prompt = build_system_prompt(memory_layer, session.kind)
    history = _get_history(session)

    actions = await _final_memory_save(
        session, llm_config, system_prompt, history, db,
        instruction=_ROTATION_CLOSE_PROMPT,
        where=rotation_reason,
    )

    summary = (actions.summary or "").strip() or _deterministic_summary(history)

    history.append({"role": "assistant", "content": f"[session closed: {close_reason}]"})
    _save_history(session, history)
    session.closed_at = datetime.now(timezone.utc)
    session.close_reason = close_reason
    if not session.summary:
        session.summary = summary
    db.add(session)
    await db.flush()

    # §4.1: no-op if the final close_session call above already wrote
    # agent_memory_patch; otherwise reuses actions.summary/session.summary
    # (both set above — session.summary carries the deterministic fallback
    # when the provider gave nothing) before falling back to a fresh call.
    await flush_session_memory(session, db, reason=close_reason, applied_actions=actions)

    # L1: the router path (topic_change / not_continuable) and rotation both end
    # a formation dialogue without a Goal — save its turns as a draft so the
    # next session can pick them back up (plan_dialogue_archive_plan Stage 5).
    if not actions.goals_create and not actions.tasks_create:
        await _maybe_save_dialogue_draft(session, db, close_reason=close_reason)

    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_CLOSED",
        content=(
            f"Session {session.id} closed (kind={session.kind}, "
            f"reason={close_reason}, limit={rotation_reason})"
        ),
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": session.kind,
            "close_reason": close_reason,
            "rotation_reason": rotation_reason,
            "summary": session.summary,
        },
    )

    # internal design notes stage 4.1: rotation/topic-change also ends
    # the dialogue without going through _run_close_flow's close_session tool
    # call — the last turn's work still needs its Gate 1.
    await _finalize_session_work(
        session, db, reason="session_closed", agent_reply=session.summary or summary
    )

    return session.summary or summary


async def _rotate_user_session(
    session: Session, db: AsyncSession, *, reason: str
) -> Session | None:
    """Graceful close of an over-long session plus its immediate successor.

    The user never sees an interruption: the current turn is replayed into the
    successor by the caller. The successor row is created FIRST — if that fails
    the original session stays open and untouched, so a rotation problem can
    never swallow the user's message.
    """
    logger.info(
        "user_sessions: rotating session %s (kind=%s, reason=%s)",
        session.id, session.kind, reason,
    )
    try:
        successor = await _successor_session(session, db)
    except Exception:
        logger.exception(
            "user_sessions: successor session could not be created for %s — "
            "continuing in the current session",
            session.id,
        )
        return None
    if successor is None:
        return None

    # project_cm_p2_close_midgathering: if the session being rotated out ended on
    # a question to the user, the message about to be replayed into the successor
    # is very likely the answer. Carry the flag so the router still sees an open
    # question on the successor's first turn (size-limit waiver + the
    # continue_session coercion in router.py) instead of treating a fresh row as
    # a clean slate. Self-expires — build_open_session_context only reads it while
    # the successor's history is still short.
    _pre_history = _get_history(session)
    _last_assistant = next(
        (
            str(m.get("content") or "")
            for m in reversed(_pre_history)
            if m.get("role") == "assistant"
        ),
        "",
    )
    if _looks_like_open_question(_last_assistant):
        snap = dict(successor.context_snapshot or {})
        snap["carried_open_question"] = True
        successor.context_snapshot = snap
        db.add(successor)
        await db.flush()

    await close_user_session_gracefully(session, db, rotation_reason=reason)

    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_STARTED",
        content=f"Session {successor.id} started (kind={successor.kind}, rotated)",
        db=db,
        metadata={
            "session_id": str(successor.id),
            "kind": successor.kind,
            "route": "rotated",
            "rotated_from": str(session.id),
            "rotation_reason": reason,
        },
    )
    return successor


async def _maybe_rotate_user_session(
    session: Session, db: AsyncSession
) -> Session | None:
    """Return a successor session if *session* has outlived its limits."""
    if not lifecycle_enforced() or session.kind not in USER_SESSION_KINDS:
        return None

    history = _get_history(session)
    window_limit = 0
    if session.last_context_window_tokens > 0:
        try:
            window_limit = await _context_window_limit(session, db)
        except Exception:
            logger.debug(
                "user_sessions: context policy unavailable for session %s",
                session.id, exc_info=True,
            )

    reason = rotation_reason(
        message_count=len(history),
        started_at=session.started_at,
        last_window_tokens=session.last_context_window_tokens,
        window_limit=window_limit,
    )
    if reason is None:
        return None
    return await _rotate_user_session(session, db, reason=reason)


# ---------------------------------------------------------------------------
# Growth warning (chatflow_session_lifecycle_repair_plan stage 5.6)
# ---------------------------------------------------------------------------


async def _maybe_warn_growth(session: Session, db: AsyncSession) -> None:
    """Post a one-time banner when this session crosses the rotation threshold.

    Fires at the same window_soft_ratio() share of the window that
    _maybe_rotate_user_session() rotates on — so the user is told a session is
    growing at the exact turn where the NEXT turn would silently start a
    successor. Debounced via context_snapshot so it fires once per session,
    not on every turn once the threshold is crossed. Never raises: a missed
    warning is not worth failing the turn over.
    """
    if session.kind not in USER_SESSION_KINDS or not session.is_open:
        return
    snap = session.context_snapshot or {}
    if snap.get("growth_warned"):
        return
    if session.last_context_window_tokens <= 0:
        return
    try:
        window_limit = await _context_window_limit(session, db)
    except Exception:
        logger.debug(
            "user_sessions: context policy unavailable for growth warning (session %s)",
            session.id, exc_info=True,
        )
        return
    if window_limit <= 0 or session.last_context_window_tokens < window_soft_ratio() * window_limit:
        return

    new_snap = dict(session.context_snapshot or {})
    new_snap["growth_warned"] = True
    session.context_snapshot = new_snap
    _flag_snapshot_dirty(session)
    db.add(session)
    await db.flush()

    pct = int(round(100 * session.last_context_window_tokens / window_limit))
    try:
        await emit_event(
            room_id=session.room_id,
            event_type="SESSION_GROWING",
            content=(
                f"The session has grown (~{pct}% of the window limit) — a new one will start soon, "
                "the context will be saved automatically."
            ),
            db=db,
            metadata={
                "session_id": str(session.id),
                "kind": session.kind,
                "window_ratio_pct": pct,
            },
        )
    except Exception:
        logger.warning(
            "user_sessions: failed to emit SESSION_GROWING for session %s",
            session.id, exc_info=True,
        )


# ---------------------------------------------------------------------------
# Router hand-off (chatflow_session_lifecycle_repair_plan stage 2.2)
# ---------------------------------------------------------------------------

_OPEN_SESSION_SUMMARY_CHARS = 600
# Keep in sync with router._OPEN_SESSION_EXCERPT_CHARS. The tail of the last
# assistant turn is what matters (that is where the question is), so the router
# gets the *end* of the reply, not its opening.
_OPEN_SESSION_EXCERPT_CHARS = 800


async def open_session_context(
    session: Session, db: AsyncSession
) -> OpenSessionContext | None:
    """Describe an open session for the router, or None if it must not continue.

    None means "this session is no longer a candidate": it is not user-facing,
    the continue route is switched off, or it already outlived the limits that
    would rotate it on the next turn. The caller then routes normally and closes
    this session through ``close_user_session_gracefully``.
    """
    from app.core.user_session_limits import is_continuable, router_continue_enabled

    if session.kind not in USER_SESSION_KINDS or not session.is_open:
        return None
    if not router_continue_enabled():
        return None

    window_limit = 0
    if session.last_context_window_tokens > 0:
        try:
            window_limit = await _context_window_limit(session, db)
        except Exception:
            logger.debug(
                "user_sessions: context policy unavailable for session %s",
                session.id, exc_info=True,
            )

    history = _get_history(session)

    first_user = next(
        (str(m.get("content") or "") for m in history if m.get("role") == "user"), ""
    )
    last_assistant = next(
        (
            str(m.get("content") or "")
            for m in reversed(history)
            if m.get("role") == "assistant"
        ),
        "",
    )

    # waiting_reply_safeguard S3: the last assistant turn put a question to the
    # user, an ask_user is pending, or the session's dialogue task is parked on
    # a question. The answer belongs in this session — waive the size limits
    # (not age / idle) for the one turn that carries it.
    has_open_question = _looks_like_open_question(last_assistant)
    if not has_open_question:
        try:
            from app.core.ask_user_registry import is_session_waiting

            has_open_question = is_session_waiting(session.id)
        except Exception:
            pass
    if not has_open_question:
        try:
            from app.core.session_work import open_dialogue_task

            _dtask = await open_dialogue_task(session, db)
            has_open_question = _dtask is not None and _dtask.awaiting_user_kind in (
                "question", "continuation_offer",
            )
        except Exception:
            pass
    # project_cm_p2_close_midgathering: the predecessor session rotated out while
    # its last turn was a question; this reply is the answer. Honoured only until
    # the successor has an assistant turn of its own.
    if (
        not has_open_question
        and (session.context_snapshot or {}).get("carried_open_question")
        and not any(m.get("role") == "assistant" for m in history)
    ):
        has_open_question = True
    # Fallback to the room feed (incident 2026-09-08#3): the CM asked its
    # clarifying questions through the force_text retry after a premature
    # close_session on turn 1. That reply reached the room as a MESSAGE row but
    # was not in this session's persisted `context_snapshot["messages"]` when the
    # next turn's router read it (the two _bg_group_flow turns run in separate
    # transactions). Without this the answer cold-routed to a fresh ChatMaster.
    # The published feed is the source of truth for "what did the user last see".
    _cm_agent_id = getattr(session, "chat_master_agent_id", None) or getattr(
        session, "agent_id", None
    )
    if not has_open_question and _cm_agent_id is not None:
        try:
            last_msg = (
                await db.execute(
                    select(Message.content)
                    .where(
                        Message.room_id == session.room_id,
                        Message.sender_agent_id == _cm_agent_id,
                        Message.event_type == "MESSAGE",
                        Message.timestamp >= session.started_at,
                    )
                    .order_by(Message.timestamp.desc())
                    .limit(1)
                )
            ).scalars().first()
            if last_msg and _looks_like_open_question(str(last_msg)):
                has_open_question = True
        except Exception:
            logger.debug(
                "open_session_context: room-feed open-question fallback failed "
                "for session %s", session.id, exc_info=True,
            )

    if not is_continuable(
        message_count=len(history),
        started_at=session.started_at,
        last_activity_at=session.last_activity_at,
        last_window_tokens=session.last_context_window_tokens,
        window_limit=window_limit,
        has_open_question=has_open_question,
    ):
        return None
    return OpenSessionContext(
        session_id=session.id,
        kind=session.kind,
        turns=len(history),
        summary_so_far=(session.summary or first_user)[:_OPEN_SESSION_SUMMARY_CHARS],
        last_assistant_excerpt=last_assistant[-_OPEN_SESSION_EXCERPT_CHARS:],
        goal_id=getattr(session, "goal_id", None),
        task_id=getattr(session, "task_id", None),
        has_open_question=has_open_question,
        peer_agent_id=getattr(session, "chat_master_agent_id", None)
        or getattr(session, "agent_id", None),
        peer_is_chatmaster=str(session.kind).startswith("user_cm"),
    )


# ---------------------------------------------------------------------------
# Workspace tool loop for direct sessions (read-only: glob, grep, read_file, git_diff)
# ---------------------------------------------------------------------------

# Workspace tool calls allowed per user turn.  Data work (profiling a
# spreadsheet, walking a repo) routinely spends 6-8 calls just on discovery, so
# a ceiling of 10 cut real answers short; the agent-side proxy loop allows 60
# (PROXY_AGENT_MAX_TOOL_CALLS).
_WS_MAX_TOOL_CALLS = int(os.getenv("USER_SESSION_MAX_TOOL_CALLS", "150"))
# ``bash_wait`` blocks server-side until a background process exits, so a turn
# that installs packages or runs a long build must not pay a budget step for it.
# Each wait refunds its step, up to this many per turn, so a hung process
# cannot spin the loop forever.
_WS_WAIT_TOOL = "bash_wait"
_WS_MAX_WAIT_REFUNDS = int(os.getenv("USER_SESSION_MAX_WAIT_REFUNDS", "200"))

# Max chars of a single tool result kept in the persisted session history.
# Full results are still fed to the model within the current turn; history
# keeps a truncated copy so later turns remember what was read without
# re-paying for entire files.
_WS_NOTE_MAX_CHARS = 4_000
_MCP_VISION_TOOL_NAMES = frozenset({"describe_image", "extract_design_spec", "critique_design"})
_VISION_OBS_HEAD_CHARS = 400

_WS_TOOL_LABELS: dict[str, str] = {
    "read_file": "reads a file",
    "grep": "searches the code",
    "repo_map": "builds a repository map",
    "get_outline": "gets a file outline",
    "glob": "looks at files",
    "git_diff": "views the diff",
    "run_subtask": "starts a sub-step",
}


_FORCED_TEXT_INSTRUCTION = (
    "[The tool-call limit for this turn is reached — tools are no longer "
    "available.] Give the user a final answer in plain text, based on the tool "
    "results above, in the language the user writes in. Do not try to call a "
    "tool and do not output tool-call markup. If the work is unfinished, say "
    "briefly what is done and what remains."
)


def _forced_limit_text() -> str:
    # Fallback shown verbatim to the user, whatever their language: keep it
    # plain and free of any "magic word" they would have to type.
    return (
        f"Tool-call limit for this turn ({_WS_MAX_TOOL_CALLS}) reached — the work "
        "is not finished. Send any message (e.g. \"continue\") and I will pick up "
        "where I stopped."
    )


def _guard_forced_reply(forced: "_LLMTurnResult", session_id: Any) -> None:
    """Never publish a tool call written as prose from the forced-text reply.

    The forced call runs with the workspace tools gone, so a model mid-way
    through (say) paging a file answers with the *next* call as text — which
    the room then shows as if it were a reply.  There is nothing to retry
    into here, so keep any real prose and otherwise say plainly that the
    turn's tool budget ran out.
    """
    if not _looks_like_unexecuted_tool_call(forced.text):
        return
    logger.warning(
        "user_sessions: forced-text reply for session %s was a tool call "
        "written as text — replacing it",
        session_id,
    )
    cleaned = _strip_tool_call_mimic(forced.text)
    forced.text = (
        _forced_limit_text() if cleaned == _TOOL_MIMIC_FALLBACK_TEXT
        else cleaned + "\n\n" + _forced_limit_text()
    )


def _forced_text_history(
    history: list[dict[str, Any]],
    tool_notes: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """History for the force_text fallback after tool-call exhaustion.

    The tool exchange of *this* turn must be visible: ``tool_notes`` only reach
    the persisted history after the turn ends, so a forced call given the bare
    pre-loop history answers with amnesia and restarts from its first
    exploratory tool call.
    """
    messages: list[dict[str, Any]] = [*history, *tool_notes]
    last = messages[-1] if messages else None
    if last and last.get("role") == "user" and isinstance(last.get("content"), str):
        # Keep user/assistant strictly alternating — Anthropic rejects two
        # consecutive user messages.
        messages[-1] = {**last, "content": last["content"] + "\n\n" + _FORCED_TEXT_INSTRUCTION}
    else:
        messages.append({"role": "user", "content": _FORCED_TEXT_INSTRUCTION})
    return messages


# A model that meant to call a tool but wrote it out as prose instead —
# observed with the compacted "[tool call] name(args)" tool_notes
# convention (§_note_iteration) apparently teaching a model, via its own
# history, that this text is how a tool call looks. Caught once per ws_loop
# turn and retried with the wire's own tool_choice forced, which removes the
# "answer in prose" option at the API level instead of just asking nicely —
# a plain text nudge does not reliably work, the model tends to repeat the
# same mistake.
_TOOL_CALL_MIMIC_RE = re.compile(
    r"\[tool call\]|\[tools executed\b|\bmcp__[\w-]+__\w+\s*\(",
    re.IGNORECASE,
)

# Retries per ws_loop turn, shared by both provider branches: a long turn can
# slip more than once, but a model that keeps mimicking must not loop forever.
_TOOL_MIMIC_MAX_RETRIES = 3


def _looks_like_unexecuted_tool_call(text: str | None) -> bool:
    return bool(text) and bool(_TOOL_CALL_MIMIC_RE.search(text))


_TOOL_MIMIC_FALLBACK_TEXT = (
    "Could not continue: the model described a tool call as text instead of "
    "making a real call. Send any message (e.g. \"continue\") to retry the step."
)


def _strip_tool_call_mimic(text: str | None) -> str:
    """Drop the lines of a reply that are a tool call written out as prose.

    A mimicked call never ran, so publishing it tells the user work happened
    that did not.  Prose around it (a model often explains, then "calls") is
    kept; a reply that was nothing but the mimic gets an honest fallback.
    """
    kept = [
        line for line in (text or "").splitlines()
        if not _TOOL_CALL_MIMIC_RE.search(line)
    ]
    cleaned = "\n".join(kept).strip()
    return cleaned or _TOOL_MIMIC_FALLBACK_TEXT


# The assistant half of a compacted tool exchange (§_note_iteration).  It used
# to read "[tool call] name({args})" — i.e. exactly what a tool call
# looks like when written as text, so models learned from their own history to
# "call" tools in prose.  It now only names what ran, in the past tense; the
# calls' arguments live in the results message that follows it.
_TOOL_NOTE_PREFIX = "[tools executed: "
_TOOL_NOTE_SUFFIX = " — arguments and results in the next message]"
_LEGACY_TOOL_NOTE_PREFIX = "[tool call] "
# Calls were joined with "; " after a closing ")" — requiring the ")" keeps a
# "; x(" inside the JSON arguments from reading as another tool.
_LEGACY_NOTE_NAME_RE = re.compile(r"(?:^\[tool call\] |\); )([A-Za-z_][\w.-]*)\(")


def _tool_note_text(tool_names: list[str]) -> str:
    unique = list(dict.fromkeys(n for n in tool_names if n))
    return _TOOL_NOTE_PREFIX + (", ".join(unique) or "—") + _TOOL_NOTE_SUFFIX


def _modernize_tool_notes(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rewrite legacy call-shaped tool notes in a stored history.

    Sessions persisted before the note format changed still carry
    "[tool call] name(args)" assistant messages; left as they are,
    they keep teaching the model the prose-call shape on every later turn.
    Returns a new list only when something changed.
    """
    out: list[dict[str, Any]] | None = None
    for i, msg in enumerate(history):
        content = msg.get("content")
        if (
            msg.get("role") == "assistant"
            and isinstance(content, str)
            and content.startswith(_LEGACY_TOOL_NOTE_PREFIX)
        ):
            if out is None:
                out = list(history)
            names = _LEGACY_NOTE_NAME_RE.findall(content)
            out[i] = {**msg, "content": _tool_note_text(names)}
    return out if out is not None else history


def _note_truncate(text: str) -> str:
    if len(text) <= _WS_NOTE_MAX_CHARS:
        return text
    return (
        text[:_WS_NOTE_MAX_CHARS]
        + f"\n[...first {_WS_NOTE_MAX_CHARS} characters saved; re-read the needed section if necessary]"
    )


async def _vision_obs_note(
    binding: Any,
    tool_name: str,
    tool_input: dict[str, Any],
    result_text: str,
    db: Any,
    room_id: UUID,
    session_id: UUID,
) -> str:
    """Record a vision observation and return receipt + short head.

    Falls back to _note_truncate when the binding is not mcp-vision, db is
    unavailable, or the storage write fails — so the note is always non-empty.
    """
    if (
        getattr(binding, "market_slug", None) != "mcp-vision"
        or getattr(binding, "tool_name", None) not in _MCP_VISION_TOOL_NAMES
        or db is None
    ):
        return _note_truncate(result_text)
    try:
        from app.core import context_observations as co
        obs = await co.record_observation(
            db,
            room_id=room_id,
            session_id=session_id,
            task_id=None,
            tool_name=tool_name,
            canonical_args=co.canonicalize_args(tool_input),
            payload=result_text,
            metadata={"source": "mcp-vision"},
        )
        head = result_text[:_VISION_OBS_HEAD_CHARS]
        tail = "…" if len(result_text) > _VISION_OBS_HEAD_CHARS else ""
        return f"{obs.receipt()}\n{head}{tail}"
    except Exception:
        return _note_truncate(result_text)


async def _call_llm_with_tools(*args, **kwargs) -> _LLMTurnResult:
    """Resolve an agent binding for one provider call and clear it afterward."""

    db = kwargs.pop("db", None)
    agent_id = kwargs.pop("agent_id", None)
    request_id = kwargs.pop("request_id", None)
    secret_value = kwargs.get("secret_value")
    owned_secret = secret_value
    if owned_secret is None and db is not None and agent_id is not None:
        llm_config = args[0] if args else kwargs.get("llm_config", {})
        owned_secret = await _resolve_session_secret(
            llm_config, db, agent_id, request_id or UUID(int=0)
        )
        kwargs["secret_value"] = owned_secret
    # Stage 1.2.2 (internal design notes §8.2): the provider call must not be
    # waited out holding the caller's pool connection.  Callers that do not
    # hand their session in release it themselves right before the call.
    await release_db(db, reason="user_session/single_turn")
    try:
        result = await _call_llm_with_tools_impl(*args, **kwargs)
    finally:
        if secret_value is None:
            owned_secret = None
    # A single provider call: its prompt tokens ARE the context window (5.2).
    if result.max_window_in == 0:
        result.max_window_in = result.tokens_in
    if result.last_window_in == 0:
        result.last_window_in = result.tokens_in
    return result


async def _resolve_session_secret(
    llm_config: dict[str, Any], db: AsyncSession, agent_id: UUID | None, request_id: UUID
) -> SecretValue | None:
    binding_id = binding_id_from_config(
        llm_config, "primary_secret_binding_id", "secret_binding_id"
    )
    if binding_id is None:
        return legacy_secret_value(llm_config, "api_key")
    if agent_id is None:
        raise ValueError("LLM secret binding has no owning agent")
    agent = await db.get(Agent, agent_id)
    if agent is None:
        raise ValueError("LLM agent is not available")
    return await resolve_for_use(
        db=db,
        owner_id=agent.owner_id,
        resource_type=SecretResourceType.AGENT,
        resource_id=agent.id,
        purpose=SecretPurpose.LLM_PRIMARY,
        binding_id=binding_id,
        request_id=request_id,
    )


async def _call_with_ws_loop(*args, **kwargs) -> _LLMTurnResult:
    """Resolve the agent binding once, then clear it after the tool loop."""

    llm_config = args[0] if args else kwargs["llm_config"]
    session_id = args[5] if len(args) > 5 else kwargs["session_id"]
    agent_id = args[6] if len(args) > 6 else kwargs.get("agent_id")
    db = args[7] if len(args) > 7 else kwargs["db"]
    secret_value = await _resolve_session_secret(llm_config, db, agent_id, session_id)
    try:
        return await _call_with_ws_loop_impl(*args, secret_value=secret_value, **kwargs)
    finally:
        secret_value = None


async def _call_with_ws_loop_impl(
    llm_config: dict[str, Any],
    system_prompt: str,
    history: list[dict[str, Any]],
    matrix_key: str | None,
    room_id: UUID,
    session_id: UUID,
    agent_id: UUID | None,
    db: AsyncSession,
    *,
    ant_ws_tools: list[dict[str, Any]],
    oai_ws_tools: list[dict[str, Any]],
    mcp_ant_tools: list[dict[str, Any]] | None = None,
    mcp_oai_tools: list[dict[str, Any]] | None = None,
    mcp_bindings: dict[str, Any] | None = None,
    mcp_user_id: UUID | None = None,
    goal_id: UUID | None = None,
    budget_used: int = 0,
    secret_value: SecretValue | None = None,
    before_tool: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
) -> _LLMTurnResult:
    """Call LLM in a loop until text or a validated session-level tool call.

    In the explicit legacy cumulative mode, stops early
    (``budget_exhausted=True``) when cumulative session tokens reach
    ``TOKEN_LIMIT``.  In window mode, each assembled request is independently
    preflighted by ``ContextAssembler``.
    On tool-call exhaustion falls back to a force_text call so the session
    always continues.
    """
    from app.core.tools import REGISTRY as _ws_registry, ToolContext

    history = _modernize_tool_notes(history)
    mcp_ant_tools = mcp_ant_tools or []
    mcp_oai_tools = mcp_oai_tools or []
    mcp_bindings = mcp_bindings or {}
    allowed_workspace_tool_names = {
        schema.get("name")
        or (schema.get("function") or {}).get("name")
        for schema in [*ant_ws_tools, *oai_ws_tools]
    }
    mcp_failure_counts: dict[str, int] = {}
    mcp_circuit_limit = max(1, int(os.getenv("MCP_FAILURE_CIRCUIT_LIMIT", "2")))

    provider = (llm_config.get("provider") or "anthropic").lower()
    model = llm_config.get("model") or "claude-haiku-4-5-20251001"
    api_key = secret_value.reveal() if secret_value is not None else None
    if not api_key:
        legacy = legacy_secret_value(llm_config, "api_key")
        try:
            api_key = legacy.reveal() if legacy is not None else None
        finally:
            legacy = None
    base_url = llm_config.get("base_url")
    include_propose_plan = matrix_key == "user_cm_p2"
    include_task_change = matrix_key == "user_cm_p1"

    ant_close = close_tool_for_kind(matrix_key, "anthropic") if matrix_key else ANTHROPIC_CLOSE_TOOL
    oai_close = close_tool_for_kind(matrix_key, "openai") if matrix_key else OPENAI_CLOSE_TOOL
    _CLOSE_NAMES = frozenset(
        {
            "close_session", "propose_plan", OFFER_PLAN_TOOL_NAME,
            "propose_task_change", "request_file_read",
        }
    )

    def _observe_mcp_result(tool_name: str, result_text: str) -> tuple[str, bool]:
        if not isinstance(result_text, str) or not result_text.startswith("Error: MCP"):
            return result_text, False
        mcp_failure_counts[tool_name] = mcp_failure_counts.get(tool_name, 0) + 1
        safe_result = (
            f"{result_text}\nMCP capability is unavailable. Do not use raw "
            "HTTP/curl/requests or guess credentials."
        )
        return safe_result, mcp_failure_counts[tool_name] >= mcp_circuit_limit

    total_in = total_out = 0
    reasoning_tokens_total = 0
    # Per-request prompt sizes across the loop (5.2). total_in accumulates every
    # call, so it must never be reported as the session's context window.
    max_window_in = last_window_in = 0
    context_step = 0
    context_tracker = ToolCallStats()
    context_role = "agent" if str(matrix_key or "").startswith("user_agent") else "chat_master"
    # internal design notes Stage 5: tool-call attribution. A dialogue
    # session's tool calls used to be logged as "ChatMaster ..." unconditionally,
    # which misattributed an agent's own work in a direct session (§0 incident).
    tool_call_actor = "ChatMaster"
    if context_role == "agent" and agent_id is not None:
        try:
            _actor_result = await db.execute(select(Agent).where(Agent.id == agent_id))
            _actor_agent = _actor_result.scalars().first()
            if _actor_agent is not None and getattr(_actor_agent, "nickname", None):
                tool_call_actor = f"@{_actor_agent.nickname}"
        except Exception as exc:
            logger.warning("user_sessions: could not resolve agent nickname for tool-call attribution: %s", exc)
    context_policy = await resolve_context_policy(
        role=context_role,
        db=db,
        room_id=room_id,
        agent_id=agent_id,
        llm_config=llm_config,
    )
    context_assembler = ContextAssembler(context_policy)
    cache_enabled = bool(
        llm_config.get("provider_cache_enabled")
        or llm_config.get("prompt_cache_enabled")
        or llm_config.get("prompt_cache_key")
    )
    cache_system_version = system_prompt_version(
        system_prompt,
        llm_config.get("system_prompt_version"),
    )

    def _cache_key(tools: list[dict[str, Any]]) -> str:
        return provider_cache_key(
            room_id=room_id,
            agent_role=context_role,
            system_version=cache_system_version,
            tool_schema_version_value=tool_schema_version(tools),
        )

    async def _persist_provider_cache(usage: Any) -> None:
        cache_usage = extract_cache_usage(usage, provider)
        await persist_provider_cache_usage(
            session_id=session_id,
            db=db,
            read_tokens=cache_usage.read_tokens,
            write_tokens=cache_usage.write_tokens,
            cost_usd=cache_cost_adjustment(
                cache_usage.read_tokens,
                cache_usage.write_tokens,
                included_in_input=cache_usage.included_in_input,
                model=model,
                llm_config=llm_config,
            ),
        )

    async def _assemble_request(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        request_kind: str,
    ):
        session_remaining = (
            max(0, TOKEN_LIMIT - budget_used - total_in - total_out)
            if uses_cumulative_limit()
            else None
        )
        assembly = context_assembler.build(
            system_prompt=system_prompt,
            tool_schemas=tools,
            messages=messages,
            session_remaining_tokens=session_remaining,
        )
        try:
            from app.core.metrics import record_context_assembly

            record_context_assembly(assembly)
        except Exception as exc:
            logger.warning(
                "user_sessions: context assembly metrics failed for %s: %s",
                session_id,
                exc,
            )
        await persist_context_assembly(
            assembly,
            session_id=session_id,
            db=db,
            role=context_role,
            request_kind=request_kind,
        )
        log_preflight_observation(
            assembly,
            session_id=session_id,
            role=context_role,
            request_kind=request_kind,
            policy=context_policy,
            session_remaining_tokens=session_remaining,
            cumulative_tokens=budget_used + total_in + total_out,
        )
        if assembly.rejected_reason:
            logger.warning(
                "user_sessions: %s preflight rejected for session %s "
                "(reason=%s estimated_input=%d remaining=%d)",
                request_kind,
                session_id,
                assembly.rejected_reason,
                assembly.estimated_input_tokens,
                session_remaining if session_remaining is not None else 0,
            )
        return assembly

    async def _record_request(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
        response_calls: list[dict[str, Any]],
    ) -> None:
        nonlocal context_step
        measurement = build_context_measurement(
            provider=provider,
            model=model,
            role=context_role,
            step=context_step,
            system_prompt=system_prompt,
            tools=tools,
            messages=messages,
            actual_input_tokens=input_tokens,
            actual_output_tokens=output_tokens,
            response_calls=response_calls,
            tracker=context_tracker,
            request_kind="workspace_tool_loop",
        )
        await record_context_measurement(measurement, session_id=session_id, db=db)
        context_step += 1

    async def _emit_tool_call(
        tool_name: str,
        tool_input_data: dict[str, Any],
        subtask_activity: dict[str, Any] | None = None,
    ) -> None:
        label = _WS_TOOL_LABELS.get(tool_name, f"calls {tool_name}")
        target = (
            tool_input_data.get("path")
            or tool_input_data.get("pattern")
            or tool_input_data.get("cmd")
            or tool_input_data.get("step_title")
        )
        content = tool_call_actor + " " + label + (f": {str(target)[:120]}" if target else "")
        try:
            metadata: dict[str, Any] = {
                "session_id": str(session_id),
                "tool": tool_name,
                "target": str(target) if target else None,
                "agent_id": str(agent_id) if agent_id else None,
            }
            subtask = activity_metadata(
                subtask_activity,
                phase="started" if tool_name == "run_subtask" else "activity",
            )
            if subtask is not None:
                metadata["subtask"] = subtask
            await emit_event(
                room_id=room_id,
                event_type="AGENT_TOOL_CALL",
                content=content,
                db=db,
                metadata=metadata,
            )
        except Exception as exc:
            logger.warning("user_sessions: emit AGENT_TOOL_CALL failed: %s", exc)

    async def _exec_ws_tool(tool_name: str, tool_input_data: dict[str, Any]) -> str:
        # internal design notes Stage 1: last chance to put HEAD on the
        # branch this dialogue's work belongs on — a write auto-commits onto
        # whatever is checked out, so the pin has to happen before the tool
        # runs, not after. Never fatal: a git problem must not cost the turn.
        if before_tool is not None:
            try:
                await before_tool(tool_name, tool_input_data)
            except Exception as exc:
                logger.warning(
                    "user_sessions: pre-tool branch pin failed for session %s (%s): %s",
                    session_id, tool_name, exc,
                )

        subtask_activity = new_subtask_activity(tool_name, tool_input_data)
        await _emit_tool_call(tool_name, tool_input_data, subtask_activity)
        # Keep nested tools in direct chat on the same capability boundary as
        # this loop.  In particular, run_subtask must be able to forward the
        # room MCP schemas/bindings instead of falling back to workspace-only
        # registry tools.
        ctx = ToolContext(
            room_id=room_id,
            session_id=session_id,
            task_id=None,
            db=db,
            llm_config=llm_config,
            secret_value=secret_value,
            actor_agent_id=agent_id,
            workspace_ant_tools=ant_ws_tools,
            workspace_oai_tools=oai_ws_tools,
            mcp_ant_tools=mcp_ant_tools,
            mcp_oai_tools=mcp_oai_tools,
            mcp_bindings=mcp_bindings,
            mcp_user_id=mcp_user_id,
            subtask_activity=subtask_activity,
        )
        return await _ws_registry.get(tool_name).execute(ctx, tool_input_data)

    tool_notes: list[dict[str, str]] = []

    def _note_iteration(calls: list[str], results: list[str]) -> None:
        if not calls:
            return
        # Names only — see _TOOL_NOTE_PREFIX for why the call shape is gone.
        tool_notes.append({
            "role": "assistant",
            "content": _tool_note_text([c.split("(", 1)[0] for c in calls]),
        })
        tool_notes.append({
            "role": "user",
            "content": "[tool results]\n\n" + "\n\n".join(results),
        })

    injected_messages: list[str] = []

    def _result(*args: Any, **kwargs: Any) -> _LLMTurnResult:
        """Build a turn result carrying this loop's context-window metrics."""
        res = _LLMTurnResult(*args, **kwargs)
        res.max_window_in = max_window_in
        res.last_window_in = last_window_in
        res.reasoning_tokens = reasoning_tokens_total
        res.injected_user_messages = list(injected_messages)
        return res

    def _take_interjections() -> str | None:
        """The user's messages sent since the last model request, as one note.

        Called at the top of every iteration, right after the stop fence: a
        "stop" typed into the chat reaches the model before its next step
        instead of starting a second turn beside this one.
        """
        texts = session_interject.drain(session_id)
        if not texts:
            return None
        injected_messages.extend(texts)
        return session_interject.format_note(texts)

    # Stage 6 (§7.2): this loop can make several provider calls in one user
    # turn, so the block is emitted per call rather than once from the result —
    # otherwise everything the agent thought before its first tool call would
    # be dropped.  Tokens are summed for the session row (§7.3); they are
    # already part of output_tokens and are never added to the totals.
    show_reasoning = show_reasoning_in_chat(llm_config)

    async def _emit_reasoning(
        thinking: str, tokens: int, *, started_at: float
    ) -> None:
        nonlocal reasoning_tokens_total
        reasoning_tokens_total += max(0, int(tokens or 0))
        if not show_reasoning:
            return
        await emit_reasoning_event(
            room_id=room_id,
            session_id=session_id,
            agent_id=agent_id,
            actor_label=None,
            task_id=None,
            thinking=thinking,
            reasoning_tokens=int(tokens or 0),
            duration_ms=round((time.monotonic() - started_at) * 1000),
            db=db,
        )

    async def _emit_interim_text(text: str | None) -> None:
        await emit_interim_text_event(
            room_id=room_id,
            session_id=session_id,
            agent_id=agent_id,
            actor_label=tool_call_actor,
            task_id=None,
            text=text,
            db=db,
        )

    async def _stopped_result() -> _LLMTurnResult | None:
        if not (
            session_interject.stop_requested(session_id)
            or await _session_was_closed_externally(session_id, db)
        ):
            return None
        logger.info("user_sessions: stopping externally closed session %s", session_id)
        return _result(
            None,
            None,
            total_in,
            total_out,
            model,
            tool_notes=tool_notes,
            externally_stopped=True,
        )

    def _budget_left() -> bool:
        if not uses_cumulative_limit():
            return True
        if budget_used + total_in + total_out < TOKEN_LIMIT:
            return True
        logger.warning(
            "user_sessions: token budget exhausted mid ws_loop for session %s "
            "(budget_used=%d loop_in=%d loop_out=%d limit=%d)",
            session_id, budget_used, total_in, total_out, TOKEN_LIMIT,
        )
        return False

    # Retry budget per turn across either provider branch — see
    # _looks_like_unexecuted_tool_call / _TOOL_MIMIC_MAX_RETRIES.
    tool_mimic_retries = 0

    # ---- Anthropic path ----
    if uses_messages_wire(provider, model):
        if uses_anthropic_env_key(provider):
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        all_tools = stable_tool_schemas([*ant_ws_tools, *mcp_ant_tools, ant_close])
        if include_propose_plan:
            all_tools = stable_tool_schemas(
                [*all_tools, PROPOSE_PLAN_TOOL_ANTHROPIC, OFFER_PLAN_TOOL_ANTHROPIC]
            )
        if include_task_change:
            all_tools = stable_tool_schemas([*all_tools, PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC])
        working: list[dict[str, Any]] = list(history)

        _wait_refunds = 0
        _iter = -1
        while _iter < _WS_MAX_TOOL_CALLS + _wait_refunds:
            _iter += 1
            # One iteration is one step in the stage-0.3 time profile
            # (internal design notes §7, app/core/step_profile.py).
            step_tick(SIDE_CHAT_MASTER)
            stopped = await _stopped_result()
            if stopped is not None:
                return stopped
            _note = _take_interjections()
            if _note is not None:
                session_interject.append_user_note(working, _note)
            if not _budget_left():
                return _result(
                    None, None, total_in, total_out, model,
                    tool_notes=tool_notes, budget_exhausted=True,
                )
            assembly = await _assemble_request(
                working,
                all_tools,
                request_kind="workspace_tool_loop/anthropic",
            )
            if not assembly.allowed:
                return _result(
                    None, None, total_in, total_out, model,
                    tool_notes=tool_notes,
                    budget_exhausted=assembly.rejected_reason == "session_budget",
                    preflight_rejected_reason=assembly.rejected_reason,
                )
            request_messages = assembly.messages
            body: dict[str, Any] = {
                "model": model,
                "system": anthropic_system_payload(
                    system_prompt,
                    enabled=bool(llm_config.get("anthropic_prompt_caching")),
                ),
                "messages": request_messages,
                "tools": all_tools,
                "max_tokens": 4096,
                **(
                    {"context_management": llm_config["anthropic_context_management"]}
                    if isinstance(llm_config.get("anthropic_context_management"), dict)
                    else {}
                ),
            }
            apply_reasoning_body(body, llm_config, provider, model)
            _reasoning_t0 = time.monotonic()
            await release_db(db, reason="ws_loop/anthropic")
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await timed_llm(
                    client.post(
                        messages_url(provider, base_url),
                        headers=messages_headers(provider, api_key),
                        json=body,
                    ),
                    provider=provider,
                    side=SIDE_CHAT_MASTER,
                )
                resp.raise_for_status()
                data = resp.json()
            _text_all, _thinking = messages_text_and_thinking(data)
            await _emit_reasoning(
                _thinking, messages_reasoning_tokens(data), started_at=_reasoning_t0
            )

            usage = data.get("usage", {})
            _record_provider_cache(llm_config, usage, role=context_role)
            await _persist_provider_cache(usage)
            request_in = usage.get("input_tokens", 0)
            request_out = usage.get("output_tokens", 0)
            total_in += request_in
            total_out += request_out
            last_window_in = request_in
            max_window_in = max(max_window_in, request_in)
            content_blocks = data.get("content", [])
            await _record_request(
                request_messages,
                all_tools,
                request_in,
                request_out,
                tool_calls_from_anthropic(content_blocks),
            )

            if data.get("stop_reason") != "tool_use":
                text = next(
                    (b["text"] for b in content_blocks if b.get("type") == "text"), ""
                )
                retried_into_tool_use = False
                while (
                    tool_mimic_retries < _TOOL_MIMIC_MAX_RETRIES
                    and _looks_like_unexecuted_tool_call(text)
                ):
                    tool_mimic_retries += 1
                    logger.warning(
                        "user_sessions: ws_loop/anthropic model wrote a tool call as "
                        "text for session %s — retry %d/%d with tool_choice=any",
                        session_id, tool_mimic_retries, _TOOL_MIMIC_MAX_RETRIES,
                    )
                    retry_body = {**body, "tool_choice": {"type": "any"}}
                    await release_db(db, reason="ws_loop/anthropic_mimic_retry")
                    async with httpx.AsyncClient(timeout=120.0) as retry_client:
                        retry_resp = await timed_llm(
                            retry_client.post(
                                messages_url(provider, base_url),
                                headers=messages_headers(provider, api_key),
                                json=retry_body,
                            ),
                            provider=provider,
                            side=SIDE_CHAT_MASTER,
                        )
                    if retry_resp.status_code >= 400:
                        break
                    retry_data = retry_resp.json()
                    retry_usage = retry_data.get("usage", {})
                    total_in += retry_usage.get("input_tokens", 0)
                    total_out += retry_usage.get("output_tokens", 0)
                    if retry_data.get("stop_reason") == "tool_use":
                        data = retry_data
                        content_blocks = data.get("content", [])
                        retried_into_tool_use = True
                        break
                    # The endpoint ignored tool_choice; its new text may be a
                    # real answer this time — re-check it.
                    text = next(
                        (
                            b["text"] for b in retry_data.get("content", [])
                            if b.get("type") == "text"
                        ),
                        "",
                    ) or text
                if not retried_into_tool_use:
                    reply = clean_reply_text(
                        text, where="ws_loop/anthropic", session_id=session_id
                    )
                    if _looks_like_unexecuted_tool_call(reply):
                        reply = _strip_tool_call_mimic(reply)
                    return _result(
                        reply, None, total_in, total_out, model, tool_notes=tool_notes,
                    )

            working.append({"role": "assistant", "content": content_blocks})

            # Emit any reasoning text the LLM produced before the tool calls
            reasoning_text = " ".join(
                b.get("text", "").strip()
                for b in content_blocks
                if b.get("type") == "text" and b.get("text", "").strip()
            )
            if reasoning_text:
                try:
                    await emit_status(
                        room_id,
                        "cm_thinking",
                        session_id=session_id,
                        agent_id=agent_id,
                        detail=reasoning_text[:300],
                    )
                except Exception:
                    pass
            await _emit_interim_text("\n\n".join(
                b.get("text", "").strip()
                for b in content_blocks
                if b.get("type") == "text" and b.get("text", "").strip()
            ))

            tool_results: list[dict[str, Any]] = []
            close_block: dict[str, Any] | None = None
            mcp_circuit_open = False
            note_calls: list[str] = []
            note_results: list[str] = []

            for block in content_blocks:
                if block.get("type") != "tool_use":
                    continue
                stopped = await _stopped_result()
                if stopped is not None:
                    return stopped
                tool_id = block["id"]
                tool_name = block["name"]
                tool_input_data = block.get("input", {})

                if tool_name == "propose_task_change":
                    validation_session = SimpleNamespace(
                        id=session_id,
                        room_id=room_id,
                        kind=matrix_key,
                        goal_id=goal_id,
                    )
                    _, validation_error, _ = await _parse_and_validate_task_change(
                        validation_session, tool_input_data, db
                    )
                    if validation_error:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": validation_error,
                            "is_error": True,
                        })
                        args_str = json.dumps(
                            tool_input_data, ensure_ascii=False, default=str
                        )[:300]
                        note_calls.append(f"propose_task_change({args_str})")
                        note_results.append(
                            f"### propose_task_change {args_str}\n{validation_error}"
                        )
                    else:
                        close_block = block
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": "Draft validated. Waiting for confirmation card action.",
                        })
                elif tool_name in _CLOSE_NAMES:
                    close_block = block
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "Ok.",
                    })
                elif _ws_registry.is_registered(tool_name):
                    if tool_name not in allowed_workspace_tool_names:
                        result_text = f"Error: tool {tool_name!r} is not available in this context"
                    else:
                        result_text = await _exec_ws_tool(tool_name, tool_input_data)
                        if tool_name == _WS_WAIT_TOOL and _wait_refunds < _WS_MAX_WAIT_REFUNDS:
                            _wait_refunds += 1
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                    })
                    args_str = json.dumps(tool_input_data, ensure_ascii=False, default=str)[:300]
                    note_calls.append(f"{tool_name}({args_str})")
                    note_results.append(
                        f"### {tool_name} {args_str}\n{_note_truncate(result_text)}"
                    )
                elif tool_name in mcp_bindings and mcp_user_id is not None:
                    await _emit_tool_call(tool_name, tool_input_data)
                    binding = mcp_bindings[tool_name]
                    from app.core.vision_dispatch import (
                        execute_vision_override,
                        should_dispatch_vision_override,
                    )
                    if (
                        agent_id is not None
                        and should_dispatch_vision_override(llm_config, binding)
                    ):
                        result_text = await execute_vision_override(
                            llm_config=llm_config,
                            arguments=tool_input_data,
                            room_id=room_id,
                            db=db,
                            owner_id=mcp_user_id,
                            agent_id=agent_id,
                            request_id=session_id,
                            tool_name=binding.tool_name,
                        )
                    else:
                        from app.core.mcp_gateway import execute_mcp_tool
                        result_text = await execute_mcp_tool(
                            binding, mcp_user_id, room_id, tool_input_data,
                            db=db,
                            request_id=session_id,
                        )
                        result_text, circuit_open = _observe_mcp_result(tool_name, result_text)
                        mcp_circuit_open = mcp_circuit_open or circuit_open
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                    })
                    args_str = json.dumps(tool_input_data, ensure_ascii=False, default=str)[:300]
                    note_calls.append(f"{tool_name}({args_str})")
                    note_results.append(
                        f"### {tool_name} {args_str}\n"
                        + await _vision_obs_note(binding, tool_name, tool_input_data, result_text, db, room_id, session_id)
                    )
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Error: tool {tool_name!r} is not available",
                        "is_error": True,
                    })

            _note_iteration(note_calls, note_results)
            if close_block is not None:
                return _result(
                    None, close_block["input"], total_in, total_out, model,
                    tool_name=close_block["name"], tool_notes=tool_notes,
                )
            working.append({"role": "user", "content": tool_results})
            if mcp_circuit_open:
                return _result(
                    "MCP capability is unavailable after a repeated error. "
                    "Check the connection and repeat the task via Retry.",
                    None,
                    total_in,
                    total_out,
                    model,
                    tool_notes=tool_notes,
                )

        logger.warning(
            "user_sessions: ws_loop exhausted (%d calls) for session %s — forcing text reply",
            _WS_MAX_TOOL_CALLS, session_id,
        )
        await release_db(db, reason="ws_loop/forced_text")
        forced = await _call_llm_with_tools(
            llm_config,
            system_prompt,
            _forced_text_history(history, tool_notes),
            matrix_key=matrix_key,
            secret_value=secret_value,
            force_text=True,
        )
        forced.tool_notes = tool_notes
        forced.forced_text_fallback = True
        forced.injected_user_messages = list(injected_messages)
        _guard_forced_reply(forced, session_id)
        forced.tokens_in += total_in
        forced.tokens_out += total_out
        # The forced call carries its own window; the loop's peak still counts.
        forced.max_window_in = max(forced.max_window_in, max_window_in)
        return forced

    # ---- OpenAI-compatible path ----
    api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-no-key-required"
    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 120.0}
    client_kwargs["base_url"] = openai_base_url(provider, base_url)
    client_kwargs.update(openai_client_extras(client_kwargs["base_url"]))

    oai = llm_client(provider, chat_client_factory=AsyncOpenAI, **client_kwargs)
    all_oai_tools = stable_tool_schemas([*oai_ws_tools, *mcp_oai_tools, oai_close])
    if include_propose_plan:
        all_oai_tools = stable_tool_schemas(
            [*all_oai_tools, PROPOSE_PLAN_TOOL_OPENAI, OFFER_PLAN_TOOL_OPENAI]
        )
    if include_task_change:
        all_oai_tools = stable_tool_schemas([*all_oai_tools, PROPOSE_TASK_CHANGE_TOOL_OPENAI])
    working_oai: list[dict[str, Any]] = list(history)

    _wait_refunds = 0
    _iter = -1
    while _iter < _WS_MAX_TOOL_CALLS + _wait_refunds:
        _iter += 1
        # See the anthropic branch above: one iteration is one profiled step.
        step_tick(SIDE_CHAT_MASTER)
        stopped = await _stopped_result()
        if stopped is not None:
            return stopped
        _note = _take_interjections()
        if _note is not None:
            session_interject.append_user_note(working_oai, _note)
        if not _budget_left():
            return _result(
                None, None, total_in, total_out, model,
                tool_notes=tool_notes, budget_exhausted=True,
            )
        assembly_oai = await _assemble_request(
            working_oai,
            all_oai_tools,
            request_kind="workspace_tool_loop/openai",
        )
        if not assembly_oai.allowed:
            return _result(
                None, None, total_in, total_out, model,
                tool_notes=tool_notes,
                budget_exhausted=assembly_oai.rejected_reason == "session_budget",
                preflight_rejected_reason=assembly_oai.rejected_reason,
            )
        request_messages_oai = [
            {"role": "system", "content": system_prompt},
            *assembly_oai.messages,
        ]
        create_extra_oai: dict[str, Any] = dict(
            openai_cache_request_extra(
                enabled=cache_enabled,
                cache_key=str(llm_config.get("prompt_cache_key") or _cache_key(all_oai_tools)),
            )
        )
        apply_reasoning_chat(create_extra_oai, llm_config, provider, model)
        _reasoning_t0 = time.monotonic()
        await release_db(db, reason="ws_loop/openai")
        resp_oai = await timed_llm(
            oai.chat.completions.create(
                model=model,
                messages=request_messages_oai,
                tools=all_oai_tools,
                tool_choice="auto",
                **create_extra_oai,
            ),
            provider=provider,
            side=SIDE_CHAT_MASTER,
        )
        choice = resp_oai.choices[0]
        usage_oai = resp_oai.usage
        await _emit_reasoning(
            *reasoning_from_chat_choice(choice.message, usage_oai),
            started_at=_reasoning_t0,
        )
        _record_provider_cache(llm_config, usage_oai, role=context_role)
        await _persist_provider_cache(usage_oai)
        request_in = usage_oai.prompt_tokens if usage_oai else 0
        request_out = usage_oai.completion_tokens if usage_oai else 0
        total_in += request_in
        total_out += request_out
        last_window_in = request_in
        max_window_in = max(max_window_in, request_in)
        await _record_request(
            assembly_oai.messages,
            all_oai_tools,
            request_in,
            request_out,
            tool_calls_from_openai(choice.message.tool_calls or []),
        )

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            text = clean_reply_text(
                choice.message.content, where="ws_loop/openai", session_id=session_id
            )
            retried_into_tool_calls = False
            while (
                tool_mimic_retries < _TOOL_MIMIC_MAX_RETRIES
                and _looks_like_unexecuted_tool_call(text)
            ):
                tool_mimic_retries += 1
                logger.warning(
                    "user_sessions: ws_loop/openai model wrote a tool call as "
                    "text for session %s — retry %d/%d with tool_choice=required",
                    session_id, tool_mimic_retries, _TOOL_MIMIC_MAX_RETRIES,
                )
                await release_db(db, reason="ws_loop/openai_mimic_retry")
                retry_resp_oai = await timed_llm(
                    oai.chat.completions.create(
                        model=model,
                        messages=request_messages_oai,
                        tools=all_oai_tools,
                        tool_choice="required",
                        **create_extra_oai,
                    ),
                    provider=provider,
                    side=SIDE_CHAT_MASTER,
                )
                retry_choice = retry_resp_oai.choices[0]
                retry_usage_oai = retry_resp_oai.usage
                total_in += retry_usage_oai.prompt_tokens if retry_usage_oai else 0
                total_out += retry_usage_oai.completion_tokens if retry_usage_oai else 0
                if retry_choice.finish_reason == "tool_calls" and retry_choice.message.tool_calls:
                    choice = retry_choice
                    retried_into_tool_calls = True
                    break
                # The endpoint ignored tool_choice; re-check its new text.
                text = clean_reply_text(
                    retry_choice.message.content,
                    where="ws_loop/openai", session_id=session_id,
                ) or text
            if not retried_into_tool_calls:
                if _looks_like_unexecuted_tool_call(text):
                    text = _strip_tool_call_mimic(text)
                return _result(
                    text, None, total_in, total_out, model,
                    tool_notes=tool_notes,
                )

        working_oai.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ],
        })
        await _emit_interim_text(choice.message.content)

        close_tc: tuple | None = None
        mcp_circuit_open = False
        note_calls_oai: list[str] = []
        note_results_oai: list[str] = []
        for tc in choice.message.tool_calls:
            stopped = await _stopped_result()
            if stopped is not None:
                return stopped
            try:
                tool_args = parse_tool_arguments(tc.function.arguments)
            except ToolArgumentsParseError:
                tool_args = {}

            if tc.function.name == "propose_task_change":
                validation_session = SimpleNamespace(
                    id=session_id,
                    room_id=room_id,
                    kind=matrix_key,
                    goal_id=goal_id,
                )
                _, validation_error, _ = await _parse_and_validate_task_change(
                    validation_session, tool_args, db
                )
                if validation_error:
                    working_oai.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": validation_error,
                    })
                    args_str = json.dumps(tool_args, ensure_ascii=False, default=str)[:300]
                    note_calls_oai.append(f"propose_task_change({args_str})")
                    note_results_oai.append(
                        f"### propose_task_change {args_str}\n{validation_error}"
                    )
                else:
                    close_tc = (tc, tool_args)
                    working_oai.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "Draft validated. Waiting for confirmation card action.",
                    })
            elif tc.function.name in _CLOSE_NAMES:
                close_tc = (tc, tool_args)
                working_oai.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Ok.",
                })
            elif _ws_registry.is_registered(tc.function.name):
                if tc.function.name not in allowed_workspace_tool_names:
                    result_text = (
                        f"Error: tool {tc.function.name!r} is not available in this context"
                    )
                else:
                    result_text = await _exec_ws_tool(tc.function.name, tool_args)
                    if tc.function.name == _WS_WAIT_TOOL and _wait_refunds < _WS_MAX_WAIT_REFUNDS:
                        _wait_refunds += 1
                working_oai.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
                args_str = json.dumps(tool_args, ensure_ascii=False, default=str)[:300]
                note_calls_oai.append(f"{tc.function.name}({args_str})")
                note_results_oai.append(
                    f"### {tc.function.name} {args_str}\n{_note_truncate(result_text)}"
                )
            elif tc.function.name in mcp_bindings and mcp_user_id is not None:
                await _emit_tool_call(tc.function.name, tool_args)
                binding = mcp_bindings[tc.function.name]
                from app.core.vision_dispatch import (
                    execute_vision_override,
                    should_dispatch_vision_override,
                )
                if (
                    agent_id is not None
                    and should_dispatch_vision_override(llm_config, binding)
                ):
                    result_text = await execute_vision_override(
                        llm_config=llm_config,
                        arguments=tool_args,
                        room_id=room_id,
                        db=db,
                        owner_id=mcp_user_id,
                        agent_id=agent_id,
                        request_id=session_id,
                        tool_name=binding.tool_name,
                    )
                else:
                    from app.core.mcp_gateway import execute_mcp_tool
                    result_text = await execute_mcp_tool(
                        binding, mcp_user_id, room_id, tool_args,
                        db=db,
                        request_id=session_id,
                    )
                    result_text, circuit_open = _observe_mcp_result(tc.function.name, result_text)
                    mcp_circuit_open = mcp_circuit_open or circuit_open
                working_oai.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
                args_str = json.dumps(tool_args, ensure_ascii=False, default=str)[:300]
                note_calls_oai.append(f"{tc.function.name}({args_str})")
                note_results_oai.append(
                    f"### {tc.function.name} {args_str}\n"
                    + await _vision_obs_note(binding, tc.function.name, tool_args, result_text, db, room_id, session_id)
                )
            else:
                working_oai.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: tool {tc.function.name!r} is not available",
                })

        _note_iteration(note_calls_oai, note_results_oai)
        if close_tc is not None:
            tc, tool_args = close_tc
            return _result(
                None, tool_args, total_in, total_out, model,
                tool_name=tc.function.name, tool_notes=tool_notes,
            )
        if mcp_circuit_open:
            return _result(
                "MCP capability is unavailable after a repeated error. "
                "Check the connection and repeat the task via Retry.",
                None,
                total_in,
                total_out,
                model,
                tool_notes=tool_notes,
            )

    logger.warning(
        "user_sessions: ws_loop exhausted (%d calls, oai) for session %s — forcing text reply",
        _WS_MAX_TOOL_CALLS, session_id,
    )
    await release_db(db, reason="ws_loop/forced_text")
    forced = await _call_llm_with_tools(
        llm_config,
        system_prompt,
        _forced_text_history(history, tool_notes),
        matrix_key=matrix_key,
        secret_value=secret_value,
        force_text=True,
    )
    forced.tool_notes = tool_notes
    forced.forced_text_fallback = True
    forced.injected_user_messages = list(injected_messages)
    _guard_forced_reply(forced, session_id)
    forced.tokens_in += total_in
    forced.tokens_out += total_out
    # The forced call carries its own window; the loop's peak still counts.
    forced.max_window_in = max(forced.max_window_in, max_window_in)
    return forced


# ---------------------------------------------------------------------------
# Direct-request preflight and close handling
# ---------------------------------------------------------------------------


async def _preflight_direct_user_request(
    *,
    session: Session,
    llm_config: dict[str, Any],
    system_prompt: str,
    history: list[dict[str, Any]],
    matrix_key: str | None,
    agent_id: UUID | None,
    db: AsyncSession,
) -> ContextAssembly:
    """Preflight the non-workspace user request with its actual tool schema.

    Workspace turns do this inside ``_call_with_ws_loop`` because every tool
    iteration has a different history.  Direct turns need the same check
    before their sole provider request.
    """
    provider = (llm_config.get("provider") or "anthropic").lower()
    context_role = "agent" if str(matrix_key or "").startswith("user_agent") else "chat_master"
    policy = await resolve_context_policy(
        role=context_role,
        db=db,
        room_id=session.room_id,
        agent_id=agent_id,
        llm_config=llm_config,
    )
    if uses_messages_wire(provider, llm_config.get("model")):
        tools: list[dict[str, Any]] = [
            close_tool_for_kind(matrix_key, "anthropic") if matrix_key else ANTHROPIC_CLOSE_TOOL
        ]
        if matrix_key == "user_cm_p2":
            tools.append(PROPOSE_PLAN_TOOL_ANTHROPIC)
        if matrix_key == "user_cm_p1":
            tools.append(PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC)
    else:
        tools = [
            close_tool_for_kind(matrix_key, "openai") if matrix_key else OPENAI_CLOSE_TOOL
        ]
        if matrix_key == "user_cm_p2":
            tools.append(PROPOSE_PLAN_TOOL_OPENAI)
        if matrix_key == "user_cm_p1":
            tools.append(PROPOSE_TASK_CHANGE_TOOL_OPENAI)

    session_remaining = (
        max(0, TOKEN_LIMIT - session.tokens_in - session.tokens_out)
        if uses_cumulative_limit()
        else None
    )
    assembly = ContextAssembler(policy).build(
        system_prompt=system_prompt,
        tool_schemas=tools,
        messages=history,
        session_remaining_tokens=session_remaining,
    )
    try:
        from app.core.metrics import record_context_assembly

        record_context_assembly(assembly)
    except Exception:
        logger.debug("user_sessions: direct context assembly metric failed", exc_info=True)
    await persist_context_assembly(
        assembly,
        session_id=session.id,
        db=db,
        role=context_role,
        request_kind="direct_user_turn",
    )
    log_preflight_observation(
        assembly,
        session_id=session.id,
        role=context_role,
        request_kind="direct_user_turn",
        policy=policy,
        session_remaining_tokens=session_remaining,
        cumulative_tokens=session.tokens_in + session.tokens_out,
    )
    return assembly


async def _close_for_context_preflight(
    session: Session,
    *,
    rejected_reason: str,
    db: AsyncSession,
) -> None:
    """Close without a provider call when no valid single-turn request fits."""
    session.closed_at = datetime.now(timezone.utc)
    # ``context_limit`` remains the user-facing close reason.  The metadata
    # preserves the more specific administrative/provider-cap diagnosis.
    session.close_reason = "context_limit"
    db.add(session)
    await db.flush()

    # §4.1: no provider call was made at all here — nothing else would ever
    # write agent memory for this close, so the deterministic fallback tier
    # is the expected (not exceptional) outcome.
    await flush_session_memory(session, db, reason="context_limit")

    await _maybe_save_dialogue_draft(session, db, close_reason="context_limit")

    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_CLOSED",
        content=(
            f"Session {session.id} closed (kind={session.kind}, "
            f"reason=context_limit, preflight={rejected_reason})"
        ),
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": session.kind,
            "close_reason": "context_limit",
            "preflight_reason": rejected_reason,
        },
    )


# ---------------------------------------------------------------------------
# Public: one user message turn
# ---------------------------------------------------------------------------


# Sessions with a turn currently running in this process.  Two turns on one
# session would race on context_snapshot, and the loser's history — including a
# just-stored pending_plan — would be silently overwritten.  Read by the
# "Form a plan" endpoint, which refuses rather than joins the race.
_ACTIVE_TURNS: set[UUID] = set()


def turn_in_progress(session_id: UUID) -> bool:
    """True while this process is running a turn for that session."""
    return session_id in _ACTIVE_TURNS


async def process_user_turn(
    session: Session,
    user_message_text: str,
    db: AsyncSession,
    *,
    inject_user_message: bool = True,
    allow_rotation: bool = True,
    force_plan: bool = False,
) -> bool:
    """Run one turn, marking the session busy for its duration (_ACTIVE_TURNS).

    A message that arrives while the session's turn is still running is not a
    second turn: it is parked for the running loop (session_interject.py), which
    hands it to the model before its next step.  Without this, "stop" typed into
    a busy direct room started a parallel agent loop beside the one it meant to
    stop.
    """
    if inject_user_message and session.id in _ACTIVE_TURNS:
        session_interject.queue(session.id, user_message_text)
        logger.info(
            "process_user_turn: session %s is mid-turn — message queued for the loop",
            session.id,
        )
        return True
    _ACTIVE_TURNS.add(session.id)
    try:
        still_open = await _process_user_turn(
            session,
            user_message_text,
            db,
            inject_user_message=inject_user_message,
            allow_rotation=allow_rotation,
            force_plan=force_plan,
        )
        # A message that landed after the loop's last model request (or on a
        # turn that never entered the loop) is answered as a follow-up turn —
        # still inside the busy window, so a third message queues behind it.
        while still_open and session.is_open:
            pending = session_interject.drain(session.id)
            if not pending:
                break
            still_open = await _process_user_turn(
                session, "\n\n".join(pending), db, allow_rotation=allow_rotation,
            )
        return still_open
    finally:
        _ACTIVE_TURNS.discard(session.id)
        session_interject.clear(session.id)


async def _process_user_turn(
    session: Session,
    user_message_text: str,
    db: AsyncSession,
    *,
    inject_user_message: bool = True,
    allow_rotation: bool = True,
    force_plan: bool = False,
) -> bool:
    """Handle one incoming user message inside a direct session.

    Flow:
      0. Rotate the session first if it outlived its lifecycle limits (1.3).
      1. Append user message to persisted conversation history.
      2. Build memory layer + system prompt (§3).
      3. In legacy cumulative mode, check_token_budget.
      4. Call LLM with close_session tool.
      5. track_llm_usage + flush.
      6. LLM called close_session → run close flow → return False.
         LLM returned text → publish assistant reply, persist history → return True.

    Args:
        allow_rotation: internal. False on the replayed turn of a rotation, so a
            successor session can never trigger another rotation immediately.
        force_plan: the user pressed "Form a plan". The turn skips the
            workspace tool loop entirely and forces ``propose_plan`` as the only
            allowed answer, so the request cannot be spent on one more round of
            reading files or clarifying the spec. user_cm_p2 only — no other
            session kind has the tool.

    Returns:
        True  — session open, waiting for next user message.
        False — session closed (context_limit, legacy token_limit, or task_done).
    """
    if not session.is_open or await _session_was_closed_externally(session.id, db):
        return False

    # Tell the UI right away that the message was picked up: the prep below
    # (memory, workspace wake, MCP discovery) can take a while before the first
    # model request reports llm_request_sent.
    if inject_user_message:
        await emit_status(
            session.room_id,
            AgentStatus.TURN_ACCEPTED,
            session_id=session.id,
            agent_id=getattr(session, "agent_id", None)
            or getattr(session, "chat_master_agent_id", None),
        )

    # 0. Lifecycle limits: close this session and continue the same turn in its
    # successor.  Runs before the user message is appended, so the closing
    # session summarizes the conversation it actually had.
    if inject_user_message and allow_rotation:
        successor = await _maybe_rotate_user_session(session, db)
        if successor is not None:
            return await process_user_turn(
                successor,
                _carryover_prefix(session) + user_message_text,
                db,
                allow_rotation=False,
                force_plan=force_plan,
            )
        if not session.is_open:
            # Rotation closed this session without producing a successor; the
            # next user message starts a fresh one through the router.
            logger.warning(
                "process_user_turn: session %s closed during rotation with no "
                "successor — turn not answered",
                session.id,
            )
            return False

    # 1. Append user turn
    history = _get_history(session)
    if inject_user_message:
        history.append({"role": "user", "content": user_message_text})
        # Persist user message immediately — if the LLM call below raises (e.g. 402),
        # the next process_user_turn will still have the correct history.
        _save_history(session, history)
        await _journal_turn(session, "user", user_message_text)
        # Idle detection in the watchdog reads last_activity_at; without this it
        # would fall back to started_at and close live conversations (1.4).
        session.last_activity_at = datetime.now(timezone.utc)
        db.add(session)
        await db.flush()

    # 2. Build context
    memory_layer = await build_memory_layer(session, db)
    system_prompt = build_system_prompt(memory_layer, session.kind)

    llm_config = await _get_agent_llm_config(session, db)
    mk = _matrix_key(session)
    # propose_plan exists only in the p2 ChatMaster dialogue; a forced plan in
    # any other kind would force a tool the model was never given.
    plan_forced = force_plan and mk == "user_cm_p2"
    if force_plan and not plan_forced:
        logger.warning(
            "process_user_turn: force_plan ignored for session %s (kind=%s)",
            session.id, session.kind,
        )
    agent_id: UUID | None = getattr(session, "agent_id", None) or getattr(
        session, "chat_master_agent_id", None
    )

    # 3. Cumulative-token closure exists only for the explicit rollback mode.
    # In window mode these counters are observability data, never an admission
    # condition for the next request.
    if uses_cumulative_limit() and not await check_token_budget(session, user_message_text, db):
        await _handle_token_limit_close(session, llm_config, system_prompt, history, db)
        return False

    # 4. LLM call (with retry on empty reply)

    # Use workspace tool loop when workspace is available — lets CM read files
    # (glob/grep/read_file/git_diff) before forming a reply or proposing a plan.
    from app.core.tools import REGISTRY as _ws_registry
    from app.core.workspace_tools import workspace_tools_available
    _ws_available = await workspace_tools_available(session.room_id, db)

    # internal design notes §0.1: a direct session writes through the same
    # branch model as a dispatched task. Without this pin the agent's tool
    # writes auto-commit onto whatever branch the last dispatch left checked
    # out — a stale `task/<id>`, not `dev`. Best-effort: a room without a repo
    # just keeps writing where it was.
    if _ws_available:
        try:
            from app.core.room_git_flow import ensure_session_branch

            pinned = await ensure_session_branch(session.room_id, session, db)
            await db.flush()
            logger.debug(
                "process_user_turn: session %s pinned to branch %s", session.id, pinned
            )
        except Exception as _exc:
            logger.warning(
                "process_user_turn: could not pin branch for session %s: %s",
                session.id, _exc,
            )

    # Room MCP servers are available to direct agent and ChatMaster sessions,
    # as well as the task proxy loop.  Resolve them once per turn; a Gateway
    # outage must not make the ordinary chat path fail.
    mcp_ant_tools: list[dict[str, Any]] = []
    mcp_oai_tools: list[dict[str, Any]] = []
    mcp_bindings: dict[str, Any] = {}
    mcp_user_id: UUID | None = None
    # Both direct user-agent sessions and user ↔ ChatMaster sessions have an
    # owning Agent record.  The owner is the MCP billing/access principal;
    # ChatMaster must see the same room capabilities it can coordinate for
    # other agents.
    if str(mk or "").startswith(("user_agent", "user_cm")) and agent_id:
        agent_row = await db.get(Agent, agent_id)
        mcp_user_id = agent_row.owner_id if agent_row else None
        if agent_row is not None:
            try:
                from app.core.skill_materialize import materialize_agent_skills

                await materialize_agent_skills(session.room_id, agent_row, db)
            except Exception as _exc:
                logger.warning(
                    "user_sessions: skill materialization failed for agent %s in room %s: %s",
                    agent_row.id, session.room_id, _exc,
                )
        if mcp_user_id:
            try:
                from app.core.mcp_gateway import room_mcp_tools

                mcp_ant_tools, mcp_oai_tools, mcp_bindings = await room_mcp_tools(
                    session.room_id, mcp_user_id, db
                )
            except Exception:
                logger.warning(
                    "user_sessions: MCP room tool discovery failed for room %s",
                    session.room_id,
                    exc_info=True,
                )
            from app.core.vision_dispatch import with_vision_override_tools

            mcp_ant_tools, mcp_oai_tools, mcp_bindings = with_vision_override_tools(
                llm_config, mcp_ant_tools, mcp_oai_tools, mcp_bindings
            )
            # This guard remains present even when the snapshot is unavailable:
            # the model must not replace a missing capability with raw HTTP or
            # credential guessing.
            system_prompt = (
                f"{system_prompt}\n\n[MCP capability policy]\n"
                "Use only explicitly provided MCP tools. "
                "If the needed tool is missing, do not use raw HTTP/curl/"
                "requests, do not scan the network and do not guess Secrets; report "
                "that the capability is unavailable and ask for a Retry."
            )

    # internal design notes Stage 1: the turn-start pin above puts HEAD
    # on the working branch (or on the work this dialogue already has open).
    # This hook is the second half — the moment a tool that could touch the
    # volume is about to run, the dialogue gets a branch of its own. Read-only
    # turns never reach it, so a conversation that only looks at the code
    # leaves no branch behind.
    async def _pin_before_tool(tool_name: str, _tool_input: dict[str, Any]) -> None:
        from app.core.session_work import ensure_worktree_branch, tool_touches_worktree

        if not tool_touches_worktree(tool_name):
            return
        await ensure_worktree_branch(session, db, user_request=user_message_text)

    request_history = history
    if (_ws_available or mcp_bindings) and not plan_forced:
        # MCP ownership is independent of the workspace role.  In particular,
        # ChatMaster receives MCP bindings but must retain its CM workspace
        # allowlist rather than inheriting the direct-agent allowlist.
        workspace_role = "agent" if str(mk or "").startswith("user_agent") else "cm"
        workspace_allowed = _workspace_tool_names_for_session(
            session, workspace_role, _ws_registry
        )
        result = await _call_with_ws_loop(
            llm_config, system_prompt, history, mk,
            session.room_id, session.id, agent_id, db,
            ant_ws_tools=(
                _ws_registry.ant_schemas_for(workspace_role, allowed=workspace_allowed)
                if _ws_available else []
            ),
            oai_ws_tools=(
                _ws_registry.oai_schemas_for(workspace_role, allowed=workspace_allowed)
                if _ws_available else []
            ),
            mcp_ant_tools=mcp_ant_tools,
            mcp_oai_tools=mcp_oai_tools,
            mcp_bindings=mcp_bindings,
            mcp_user_id=mcp_user_id,
            goal_id=getattr(session, "goal_id", None),
            budget_used=session.tokens_in + session.tokens_out,
            before_tool=_pin_before_tool if _ws_available else None,
        )
    else:
        assembly = await _preflight_direct_user_request(
            session=session,
            llm_config=llm_config,
            system_prompt=system_prompt,
            history=history,
            matrix_key=mk,
            agent_id=agent_id,
            db=db,
        )
        if assembly.rejected_reason:
            if assembly.rejected_reason == "session_budget" and uses_cumulative_limit():
                await _handle_token_limit_close(session, llm_config, system_prompt, history, db)
            else:
                await _close_for_context_preflight(
                    session, rejected_reason=assembly.rejected_reason, db=db
                )
            return False
        # Send the compacted assembly, not the original history that may have
        # exceeded the cap before receipts/checkpoint compaction.
        request_history = assembly.messages
        result = await _call_llm_with_tools(
            llm_config,
            system_prompt,
            request_history,
            matrix_key=mk,
            force_tool="propose_plan" if plan_forced else None,
            db=db,
            agent_id=agent_id,
            request_id=session.id,
        )

    # Do not persist a late result or publish a reply after Stop/Pause closed
    # this row in another request.  This check covers non-workspace turns too.
    if result.externally_stopped or await _session_was_closed_externally(session.id, db):
        if await _end_turn_on_user_stop(session, user_message_text, result, history, db):
            return True
        logger.info("process_user_turn: session %s was stopped during its turn", session.id)
        return False

    # Stage 6 (§7.2): the workspace loop emits its own blocks per provider call
    # (it makes several per turn); the plain path has exactly one, carried out
    # on the result.  `reasoning_text` is empty there whenever the loop already
    # emitted, so this never doubles a block.
    if result.reasoning_text and show_reasoning_in_chat(llm_config):
        await emit_reasoning_event(
            room_id=session.room_id,
            session_id=session.id,
            agent_id=agent_id,
            actor_label=None,
            task_id=None,
            thinking=result.reasoning_text,
            reasoning_tokens=result.reasoning_tokens,
            duration_ms=0,
            db=db,
        )
    if result.reasoning_tokens:
        session.reasoning_tokens = (session.reasoning_tokens or 0) + int(result.reasoning_tokens)

    # merge_draft ran inside this turn's tool loop: fold the carried turns into
    # the local history before any of the saves below overwrite them.
    _adopt_merged_draft_prefix(session, history)

    # Persist the tool exchange into history so the NEXT turn remembers what
    # was read and does not re-read the same files from scratch.
    if result.tool_notes:
        history.extend(result.tool_notes)
        _save_history(session, history)
        db.add(session)
        await db.flush()

    # Messages the user sent while the loop ran were handed to the model inside
    # it; the history must show them too, after the work they interrupted and
    # ahead of the reply they shaped.
    _injected_msgs = getattr(result, "injected_user_messages", None)
    if isinstance(_injected_msgs, list) and _injected_msgs:
        for _injected in _injected_msgs:
            history.append({"role": "user", "content": _injected})
            await _journal_turn(session, "user", _injected)
        _save_history(session, history)
        db.add(session)
        await db.flush()

    if result.preflight_rejected_reason:
        if result.preflight_rejected_reason == "session_budget" and uses_cumulative_limit():
            await _handle_token_limit_close(session, llm_config, system_prompt, history, db)
        else:
            await _close_for_context_preflight(
                session,
                rejected_reason=result.preflight_rejected_reason,
                db=db,
            )
        return False

    # Budget ran out mid tool-loop in the explicit legacy mode.
    if result.budget_exhausted:
        cost = _compute_cost(result.model, result.tokens_in, result.tokens_out, llm_config)
        track_llm_usage(
            session, result.tokens_in, result.tokens_out, cost, model=result.model
        )
        update_context_window_metrics(session, result.max_window_in, result.last_window_in)
        db.add(session)
        await db.flush()
        await check_token_budget(session, "", db)  # marks closed + emits event
        await _handle_token_limit_close(session, llm_config, system_prompt, history, db)
        return False

    # 4a-offer: ChatMaster says it is ready to plan. That is a plain reply plus
    # the composer's "Form a plan" button — remember the offer on the session
    # (GET .../plan_request reads it) and let the turn finish as text. Done
    # before the guards below, which would read any tool call as a premature close.
    plan_offered_now = False
    if result.tool_input is not None and result.tool_name == OFFER_PLAN_TOOL_NAME:
        offer_text = ""
        if isinstance(result.tool_input, dict):
            offer_text = str(result.tool_input.get("message") or "").strip()
        result.tool_input = None
        result.tool_name = None
        result.text = offer_text or (result.text or "").strip() or _OFFER_PLAN_FALLBACK_MESSAGE
        if mk == "user_cm_p2":
            snap = dict(session.context_snapshot or {})
            snap["plan_offered"] = True
            session.context_snapshot = snap
            _flag_snapshot_dirty(session)
            plan_offered_now = True

    # 4b-guard: user_cm_p2 must not call close_session before giving a single
    # text reply, and must not close mid requirement-gathering (incident
    # 2026-09-08, project_cm_p2_close_midgathering). Retry with force_text so the
    # CM responds conversationally and the session stays open for the answer.
    _cm_first_turn = not any(m["role"] == "assistant" for m in history)
    if (
        result.tool_input is not None
        and session.kind == "user_cm_p2"
        and not plan_forced
        and (
            _cm_first_turn
            or _cm_p2_close_is_premature(session, result.tool_input, history)
        )
    ):
        # waiting_reply_safeguard S5: on a COLD ChatMaster (first turn), a task in
        # this room parked waiting for the user means the message is almost
        # certainly an answer to it. Hand it back to waiting_reply rather than
        # letting the CM summarise-and-close. Only on the first turn — mid-dialogue
        # the answer belongs in this session, not a foreign task.
        if _cm_first_turn:
            try:
                from app.core.waiting_reply import (
                    classify_reply,
                    find_waiting_task,
                    handle_waiting_reply,
                )

                if classify_reply(user_message_text or "") is not None:
                    _wtask = await find_waiting_task(session.room_id, db)
                    if _wtask is not None:
                        _u = await db.get(User, session.user_id)
                        _room = await db.get(Room, session.room_id)
                        if _u is not None and _room is not None and await handle_waiting_reply(
                            _room, _u, user_message_text or "", db, task=_wtask,
                        ):
                            session.closed_at = datetime.now(timezone.utc)
                            session.close_reason = "waiting_reply_handoff"
                            db.add(session)
                            await db.flush()
                            await emit_event(
                                room_id=session.room_id,
                                event_type="SESSION_CLOSED",
                                content=(
                                    f"Session {session.id} closed (kind={session.kind}, "
                                    "reason=waiting_reply_handoff)"
                                ),
                                db=db,
                                metadata={
                                    "session_id": str(session.id),
                                    "kind": session.kind,
                                    "close_reason": "waiting_reply_handoff",
                                },
                            )
                            await db.commit()
                            return False
            except Exception as exc:
                logger.warning(
                    "process_user_turn: S5 waiting_reply handoff failed (session %s): %s",
                    session.id, exc,
                )
        logger.warning(
            "process_user_turn: user_cm_p2 called close_session %s (session %s) — "
            "retrying with force_text",
            "on first turn" if _cm_first_turn else "mid requirement-gathering",
            session.id,
        )
        forced = await _call_llm_with_tools(
            llm_config,
            system_prompt,
            request_history,
            matrix_key=mk,
            force_text=True,
            db=db,
            agent_id=agent_id,
            request_id=session.id,
        )
        if (forced.text or "").strip():
            result = forced

    # 4c-plan-guard: a forced plan must never end the conversation. If the
    # endpoint refused the named tool_choice (it is dropped on a 400 above) and
    # the model reached for close_session instead, retry as plain text: the
    # <propose_plan> XML fallback below still turns that answer into a plan
    # card, whereas the close flow would have thrown the dialogue away.
    if (
        plan_forced
        and result.tool_input is not None
        and result.tool_name != "propose_plan"
    ):
        logger.warning(
            "process_user_turn: forced plan turn answered with %s in session %s — "
            "retrying with force_text",
            result.tool_name, session.id,
        )
        forced_plan_text = await _call_llm_with_tools(
            llm_config,
            system_prompt,
            request_history,
            matrix_key=mk,
            force_text=True,
            db=db,
            agent_id=agent_id,
            request_id=session.id,
        )
        if (forced_plan_text.text or "").strip():
            result = forced_plan_text

    # 4d. Retry once if LLM returned empty text (no tool call)
    if result.tool_input is None and not (result.text or "").strip():
        logger.warning(
            "process_user_turn: LLM returned empty text for session %s "
            "(provider=%s model=%s) — retrying once",
            session.id,
            llm_config.get("provider", "?"),
            llm_config.get("model", "?"),
        )
        retry_result = await _call_llm_with_tools(
            llm_config,
            system_prompt,
            request_history,
            matrix_key=mk,
            db=db,
            agent_id=agent_id,
            request_id=session.id,
        )
        # Only use retry result if it produced something useful
        if retry_result.tool_input is not None or (retry_result.text or "").strip():
            result = retry_result
        else:
            logger.error(
                "process_user_turn: retry also returned empty for session %s — "
                "injecting fallback reply",
                session.id,
            )

    # A direct (non-workspace) provider can return an invalid structured
    # proposal before it gets a tool-result round-trip. Give it one repair
    # turn in the same dialogue; the workspace loop already performs this
    # exchange itself above.
    task_change_proposal: TaskChangeProposal | None = None
    task_change_tasks: list[Task] = []
    if result.tool_input is not None and result.tool_name == "propose_task_change":
        for repair_attempt in range(2):
            proposal, validation_error, proposal_tasks = await _parse_and_validate_task_change(
                session, result.tool_input, db
            )
            if proposal is not None:
                task_change_proposal = proposal
                task_change_tasks = proposal_tasks
                break
            if repair_attempt == 1 or _ws_available or mcp_bindings:
                result.tool_input = None
                result.tool_name = None
                result.text = (
                    "Proposal not saved: "
                    + (validation_error or "unknown validation error")
                )
                break
            repair_history = list(request_history) + [
                {
                    "role": "assistant",
                    "content": "[propose_task_change rejected]",
                },
                {
                    "role": "user",
                    "content": validation_error or "Fix the proposal and call propose_task_change again.",
                },
            ]
            repaired = await _call_llm_with_tools(
                llm_config,
                system_prompt,
                repair_history,
                matrix_key=mk,
                db=db,
                agent_id=agent_id,
                request_id=session.id,
            )
            if repaired.tool_input is None and not (repaired.text or "").strip():
                result.tool_input = None
                result.tool_name = None
                result.text = validation_error or "Proposal not saved."
                break
            result = repaired

    # 5. Track usage (window metrics feed the Sessions tab and the rotation check)
    cost = _compute_cost(result.model, result.tokens_in, result.tokens_out, llm_config)
    track_llm_usage(
        session, result.tokens_in, result.tokens_out, cost, model=result.model
    )
    update_context_window_metrics(session, result.max_window_in, result.last_window_in)
    session.last_activity_at = datetime.now(timezone.utc)
    db.add(session)
    await db.flush()

    # 6a-file: CM wants to read a file — pause and ask the user first
    if result.tool_input is not None and result.tool_name == "request_file_read":
        path = result.tool_input.get("path", "")
        reason = result.tool_input.get("reason", "")
        history.append({
            "role": "assistant",
            "content": (
                f"[Requesting permission to read the file: `{path}`"
                + (f". Reason: {reason}" if reason else "")
                + "]"
            ),
        })
        snap = dict(session.context_snapshot or {})
        snap["pending_file_read"] = {"path": path, "reason": reason}
        snap["messages"] = [dict(m) for m in history]
        session.context_snapshot = snap
        _flag_snapshot_dirty(session)
        db.add(session)
        await db.flush()

        from app.models.message import FILE_READ_REQUESTED
        await emit_event(
            room_id=session.room_id,
            event_type=FILE_READ_REQUESTED,
            content=f"[File read requested: {path}]",
            db=db,
            metadata={
                "session_id": str(session.id),
                "path": path,
                "reason": reason,
            },
        )
        await db.commit()
        await _maybe_warn_growth(session, db)
        return True

    # 6a-task-change: CM wants button confirmation — store only the normalized
    # draft, emit a self-contained proposal event, and stay open. No Task row is
    # mutated here; confirmation API code owns application of the draft.
    if result.tool_input is not None and result.tool_name == "propose_task_change":
        if task_change_proposal is None:
            task_change_proposal, validation_error, task_change_tasks = (
                await _parse_and_validate_task_change(session, result.tool_input, db)
            )
            if task_change_proposal is None:
                history.append({
                    "role": "assistant",
                    "content": validation_error or "[Proposal not saved.]",
                })
                _save_history(session, history)
                db.add(session)
                await db.flush()
                await db.commit()
                await _maybe_warn_growth(session, db)
                return True

        history.append({
            "role": "assistant",
            "content": "[Proposing a confirmable task replacement. Awaiting action on the card.]",
        })
        snap = dict(session.context_snapshot or {})
        snap["pending_task_change"] = task_change_proposal.model_dump(mode="json")
        snap["messages"] = [dict(m) for m in history]
        session.context_snapshot = snap
        _flag_snapshot_dirty(session)
        db.add(session)
        await db.flush()

        from app.core.metrics import log_task_change_event, record_task_change_proposal

        record_task_change_proposal()
        log_task_change_event(
            "task_change_proposed",
            room_id=session.room_id,
            session_id=session.id,
            proposal_id=task_change_proposal.proposal_id,
            source_count=len(task_change_proposal.sources),
            replacement_count=len(task_change_proposal.create_tasks),
        )

        from app.models.message import TASK_CHANGE_PROPOSED

        await emit_event(
            room_id=session.room_id,
            event_type=TASK_CHANGE_PROPOSED,
            content="[Confirmable task replacement proposed]",
            db=db,
            metadata=_task_change_event_metadata(
                session, task_change_proposal, task_change_tasks
            ),
        )
        await db.commit()
        await _maybe_warn_growth(session, db)
        return True

    # One dialogue, one kind of work (core/dialogue_scope). If this dialogue has
    # already dispatched quick goal-less tasks, a goal formed from the same
    # conversation would plan work that is concurrently being done from another
    # base — the divergence incident of 2026-09-18. Turn the plan into an
    # ordinary reply that says so, so the proposal card never appears.
    if result.tool_input is not None and result.tool_name == "propose_plan":
        _quick_tasks = await quick_tasks_for_session(session.id, db)
        if _quick_tasks:
            logger.warning(
                "process_user_turn: refused propose_plan in session %s — "
                "%d quick task(s) already running from this dialogue",
                session.id, len(_quick_tasks),
            )
            result.tool_input = None
            result.tool_name = None
            result.text = quick_tasks_block_plan_message(_quick_tasks)

    # 6a-propose: CM wants button confirmation — store plan, emit event, stay open
    if result.tool_input is not None and result.tool_name == "propose_plan":
        plan_data = result.tool_input
        # Add synthetic assistant message so history ends on assistant turn;
        # the next user message (after Edit) won't create two consecutive user roles.
        history.append({
            "role": "assistant",
            "content": (
                f"[Proposing a plan: '{plan_data.get('goal_title', '…')}'. "
                "Awaiting confirmation.]"
            ),
        })
        snap = dict(session.context_snapshot or {})
        snap["pending_plan"] = plan_data
        snap.pop("plan_offered", None)
        snap["messages"] = [dict(m) for m in history]
        session.context_snapshot = snap
        _flag_snapshot_dirty(session)
        db.add(session)
        await db.flush()

        from app.models.message import PLAN_PROPOSED
        await emit_event(
            room_id=session.room_id,
            event_type=PLAN_PROPOSED,
            content=f"[Plan proposed: {plan_data.get('goal_title', '…')}]",
            db=db,
            metadata={
                "session_id": str(session.id),
                "goal_title": plan_data.get("goal_title", ""),
                "goal_description": plan_data.get("goal_description", ""),
                "plan_summary": plan_data.get("plan_summary", ""),
                "plan_steps": _plan_steps_from_proposal(plan_data),
                **_requirements_from_proposal(plan_data),
            },
        )
        await db.commit()
        await _maybe_warn_growth(session, db)
        return True

    # 4b-guard (fallback): a user_cm_p2 close that still reads as premature — the
    # forced-text retry above came back empty, or this was a plan_forced turn.
    # Publish the close summary as an ordinary reply and keep the session open,
    # so the user's answer lands here instead of cold-routing to a fresh
    # ChatMaster (incident 2026-09-08, project_cm_p2_close_midgathering).
    if (
        result.tool_input is not None
        and _cm_p2_close_is_premature(session, result.tool_input, history)
    ):
        try:
            _summary = (
                extract_close_actions_from_tool_use(result.tool_input).summary or ""
            ).strip()
        except Exception:
            _summary = ""
        logger.warning(
            "process_user_turn: suppressed premature user_cm_p2 close for session "
            "%s — keeping session open, publishing summary as reply",
            session.id,
        )
        result.tool_input = None
        result.tool_name = None
        result.text = _summary or (
            "I still need some details to put the plan together — please answer "
            "the questions above."
        )

    # 6a. Close flow — LLM requested session end
    if result.tool_input is not None:
        await _run_close_flow(
            session, result.tool_input, db,
            llm_config=llm_config,
            system_prompt=system_prompt,
            history=history,
        )
        return False

    # 6b. Normal reply — publish to room and persist history
    reply_text = (result.text or "").strip()
    if not reply_text:
        reply_text = (
            "Sorry, an error occurred while generating the reply. "
            "Please repeat your request."
        )

    # 6b-text-fallback: some models (openrouter) write propose_plan as text instead of
    # calling the tool. Detect JSON with goal_title/goal_description in CM's reply.
    if session.kind == "user_cm_p2":
        plan_data = _extract_propose_plan_from_text(reply_text)
        if plan_data:
            # Same guard as the tool path above: a dialogue that already started
            # quick goal-less tasks does not also form a goal, whichever way the
            # model chose to write the plan.
            _quick_tasks = await quick_tasks_for_session(session.id, db)
            if _quick_tasks:
                logger.warning(
                    "process_user_turn: refused text-based propose_plan in session %s — "
                    "%d quick task(s) already running from this dialogue",
                    session.id, len(_quick_tasks),
                )
                plan_data = None
                reply_text = quick_tasks_block_plan_message(_quick_tasks)
        if plan_data:
            logger.info(
                "process_user_turn: detected text-based propose_plan in session %s — "
                "treating as tool call (goal_title=%r)",
                session.id, plan_data.get("goal_title"),
            )
            history.append({"role": "assistant", "content": reply_text})
            await _journal_turn(session, "assistant", reply_text)
            snap = dict(session.context_snapshot or {})
            snap["pending_plan"] = plan_data
            snap.pop("plan_offered", None)
            snap["messages"] = [dict(m) for m in history]
            session.context_snapshot = snap
            _flag_snapshot_dirty(session)
            db.add(session)
            await db.flush()

            from app.models.message import PLAN_PROPOSED
            await emit_event(
                room_id=session.room_id,
                event_type=PLAN_PROPOSED,
                content=f"[Plan proposed: {plan_data.get('goal_title', '…')}]",
                db=db,
                metadata={
                    "session_id": str(session.id),
                    "goal_title": plan_data.get("goal_title", ""),
                    "goal_description": plan_data.get("goal_description", ""),
                    "plan_summary": plan_data.get("plan_summary", ""),
                    "plan_steps": _plan_steps_from_proposal(plan_data),
                },
            )
            await db.commit()
            await _maybe_warn_growth(session, db)
            return True

    history.append({"role": "assistant", "content": reply_text})
    _save_history(session, history)
    await _journal_turn(session, "assistant", reply_text)

    msg = Message(
        room_id=session.room_id,
        sender_agent_id=agent_id,
        event_type="MESSAGE",
        content=reply_text,
        event_metadata={
            "session_id": str(session.id),
            "kind": session.kind,
            **({"plan_offered": True} if plan_offered_now else {}),
        },
    )
    db.add(msg)
    db.add(session)
    await db.flush()

    await manager.broadcast(
        {
            "type": "message",
            "event_type": "MESSAGE",
            "id": str(msg.id),
            "room_id": str(session.room_id),
            "sender_id": str(agent_id) if agent_id else None,
            "sender_type": "agent",
            "content": reply_text,
            "timestamp": msg.timestamp.isoformat(),
            "status": msg.status,
            "session_id": str(session.id),
            # The composer re-reads GET .../plan_request on this flag: the
            # "Form a plan" button appears only after ChatMaster offered it.
            **({"plan_offered": True} if plan_offered_now else {}),
        },
        session.room_id,
    )

    # internal design notes Stage 3: the reply is already on its way, so
    # the gate runs on the way out — the dialogue keeps its speed and Gate 1
    # announces itself with its own events. No-op unless this turn actually
    # wrote something.
    # waiting_reply_safeguard S4: a turn that only answered because the tool
    # budget ran out did not finish its work — it must not be gated as a
    # completed submission.
    await _finalize_session_work(
        session, db, reason="turn_end",
        user_request=user_message_text, agent_reply=reply_text,
        incomplete=bool(getattr(result, "forced_text_fallback", False)),
    )

    # waiting_reply_safeguard S1: the agent ended its turn on a question and
    # made no tool call — mark the dialogue's task so a reply like "finish"
    # routes back to it instead of to a fresh ChatMaster session.
    if (
        str(session.kind or "").startswith("user_agent")
        and result.tool_input is None
        and _looks_like_open_question(reply_text)
    ):
        try:
            from app.core.session_work import open_dialogue_task
            from app.core.waiting_reply import mark_task_awaiting

            _q_task = await open_dialogue_task(session, db)
            if _q_task is not None:
                await mark_task_awaiting(
                    _q_task, "question", db, prompt=reply_text[-2000:]
                )
        except Exception as exc:
            logger.warning(
                "user_sessions: could not mark question on dialogue task "
                "(session %s): %s", session.id, exc,
            )

    await _maybe_warn_growth(session, db)
    return True
