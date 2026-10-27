"""
Step 9: TASK_CREATED → CmAgentProxySession pair (§5.1, §5.2).

Two sessions are created per task handoff:
  - side='agent'        — talks with the assigned task agent
  - side='chat_master'  — the ChatMaster that supervises / approves

Conversation loop (Variant A):
  task_data → agent LLM → ProxyBroker.publish → CM receives → CM LLM → publish → ...
  until CM calls close_session (approve or reject) or either side hits the token limit.

Token budgets are independent per side (spec §2): exhausting one does not close the other.

Public API:
    handle_task_created   — main handler (Variant A: TASK_CREATED trigger)
    dispatch_task_event   — asyncio background dispatch
"""
from __future__ import annotations


import asyncio
import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import httpx
from openai import AsyncOpenAI
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.context_baseline import (
    ToolCallStats,
    build_context_measurement,
    record_context_measurement,
    tool_calls_from_anthropic,
    tool_calls_from_openai,
)
from app.core.context_assembler import (
    ContextAssembler,
    log_preflight_observation,
    persist_context_assembly,
    persist_provider_cache_usage,
)
from app.core.context_policy import (
    resolve_context_policy,
    uses_cumulative_limit,
    uses_window_limit,
)
from app.core.events import (
    AgentStatus,
    emit_event,
    emit_interim_text_event,
    emit_reasoning_event,
    emit_status,
)
from app.core import llm_call_state
from app.core import task_interject as _task_interject
from app.core.subtask_activity import activity_metadata, new_subtask_activity
from app.core.memory_layer import build_memory_layer, build_system_prompt
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
from app.core.llm_errors import (
    OPT_IN_REASONS,
    classify_gateway_error,
    model_opt_in_content,
)
from app.core.reasoning import (
    apply_reasoning_body,
    apply_reasoning_chat,
    messages_reasoning_tokens,
    reasoning_from_chat_choice,
    show_reasoning_in_chat,
)
from app.core.provider_cache import (
    anthropic_system_payload,
    extract_cache_usage,
    openai_cache_request_extra,
    provider_cache_key,
    stable_tool_schemas,
    system_prompt_version,
    tool_schema_version,
)
from app.core.proxy_broker import ProxyBroker
from app.core.step_profile import (
    SIDE_AGENT,
    SIDE_CHAT_MASTER,
    llm_span,
    note_llm_call,
    record_llm_wait,
    step_tick,
    timed_llm,
)
from app.db.release import release_db
from app.core.secrets.crypto import SecretValue
from app.core.secrets.runtime import legacy_secret_value, resolve_agent_secret
from app.models.secret import SecretPurpose
from app.core.session_close import (
    ANTHROPIC_CLOSE_TOOL,
    OPENAI_CLOSE_TOOL,
    SessionCloseActions,
    ToolArgumentsParseError,
    apply_close_actions,
    clean_reply_text,
    extract_close_actions_from_tool_use,
    flush_session_memory,
    parse_tool_arguments,
)
from app.core.media_refs import (
    MEDIA_REF_MAX_PER_IMAGE,
    ReferenceManifestEntry,
    ResolvedRef,
    build_reference_manifest,
    build_vision_blocks,
    composite_logo_overlay,
    resolve_task_references,
    to_data_url,
)
from app.core.media_size import (
    build_provider_params,
    chat_model_supports_image_input,
    fit_exact,
    model_capabilities,
    resolve_size,
    supports_input_references,
)
from app.core.media_size import parse_size as parse_media_size
from app.core.token_budget import TOKEN_LIMIT, check_token_budget, track_llm_usage, update_context_window_metrics
from app.models.agent import Agent
from app.models.message import TASK_DISPATCH_FAILED, TASK_DISPATCH_STARTED, TOOL_LIMIT_REACHED
from app.models.room import Room
from app.models.session.base import Session
from app.models.task import DispatchStatus, Task, TaskState
from app.core import run_clock
from app.repositories.session import SessionRepository
from app.schemas.proxy import ProxyMessageCreate
from app.schemas.session import CmAgentProxySessionCreate

logger = logging.getLogger(__name__)

# Safety cap: if neither side closes after this many turns the loop forcibly stops.
# One turn = one agent LLM call (with its whole tool loop) plus one CM call, so a
# mid-task question and its answer consume a full turn. The dialogue is unbounded
# by design, bounded in practice by the token budget and PROXY_MAX_DURATION_SECONDS;
# this cap is only the last-resort circuit breaker.
_MAX_PROXY_TURNS = int(os.getenv("PROXY_MAX_TURNS", "40"))
# After this many question/answer round trips the agent is told to stop asking and
# decide. Not a hard stop — the next answer simply carries a "this is final" note.
_QA_SOFT_LIMIT = int(os.getenv("PROXY_QA_SOFT_LIMIT", "6"))
# Turns left before the cap at which the agent is told to wrap up and close.
_TURN_WARNING_MARGIN = 2
# A task may use many short LLM/tool turns, but it may not keep a worker alive
# indefinitely.  The watchdog remains a second, durable safety net.
_PROXY_MAX_DURATION_SECONDS = float(os.getenv("PROXY_MAX_DURATION_SECONDS", "2400"))
_MCP_FAILURE_CIRCUIT_LIMIT = int(os.getenv("MCP_FAILURE_CIRCUIT_LIMIT", "2"))
# Calls that never spend the tool-call budget: closing the session, and
# ``bash_wait`` (blocks on a background process; its time is credited back to
# the deadline through app.core.run_clock, so waiting is not work).
_FREE_TOOLS = frozenset({"close_session", "bash_wait"})

# While an LLM call is in flight, an ephemeral llm_heartbeat status is broadcast
# every N seconds so the frontend can show "thinking for Ns…".
_HEARTBEAT_INTERVAL_S = int(os.getenv("LLM_HEARTBEAT_SECONDS", "15"))

# --- Mid-task dialogue framing -------------------------------------------
# A text turn (no close_session) is a message between the pair, not a delivery.
# Both sides get an explicit marker so the CM never mistakes a question for a
# completion report and the agent never mistakes an answer for a verdict.
_MID_TASK_NOTE = (
    "[MID-TASK MESSAGE FROM THE AGENT — the task is NOT finished and this is NOT a "
    "completion report. There is no [SYSTEM VERIFICATION] block because nothing has "
    "been delivered yet. Answer it in plain text and do NOT call close_session: "
    "closing now ends the agent's session and destroys unfinished work. Decide the "
    "question — you own this task's wording — and reply with one concrete "
    "instruction. If your ruling makes the task's own text wrong, call "
    "task_update(task_id=<id from Task Data>, description=<full corrected text>) in "
    "THIS turn, before replying — your turn ends with the reply, so a promise to fix "
    "it later is never kept.]\n\n"
)
_CM_ANSWER_NOTE = (
    "[ANSWER FROM CHATMASTER — this is a reply to your message, not a review and not "
    "a verdict. Your session is still open and the task is still yours to finish. "
    "Act on this answer, continue the work, and call close_session only when it is "
    "actually done. You may ask again if something else in the task blocks you.]\n\n"
)
_CM_CLOSE_CONVERTED_NOTE = (
    "[SYSTEM: ChatMaster tried to close the session while your work is still in "
    "progress. The close was refused and its message is handed to you as the "
    "answer you asked for. Wording like CHANGES_REQUESTED inside it is NOT a "
    "verdict — nothing has been reviewed, your session is open and the task is "
    "still yours. Act on the content below.]\n\n"
)
_CM_NO_GUIDANCE_FALLBACK = (
    "ChatMaster could not formulate a reply this turn. Act on your own "
    "best understanding of the task: pick an approach you have not tried yet, take a "
    "minimal verifiable step and record the result in a file."
)
_DIALOGUE_FOOTER_MARKER = "is on the other end of this session"
_TASK_DIALOGUE_FOOTER = (
    "\n\n---\n"
    "**{cm} is on the other end of this session and will answer you.** If the task "
    "above contradicts itself, omits something you cannot look up with a tool, or "
    "conflicts with what you find in the workspace — reply with a plain-text question "
    "instead of calling close_session. Your session stays open, the answer comes back "
    "as your next message, and you may ask again as many times as you need. Use "
    "close_session only to deliver finished work."
)
_QA_FINAL_NOTE = (
    "\n\n[SYSTEM: you have already exchanged {rounds} question/answer rounds with "
    "ChatMaster on this task. Treat this answer as final — do not ask again. Take the "
    "best-supported reading, note the remaining ambiguity in your close_session "
    "summary, and finish the work.]"
)
_TURNS_LOW_NOTE = (
    "\n\n[SYSTEM: {left} exchange(s) remain before this session is force-closed. "
    "Stop asking questions, finish what you can and call close_session now — work "
    "that is not delivered through close_session is lost.]"
)
# Prefixed to the automatic stall report when it reaches the ChatMaster. Unlike
# _MID_TASK_NOTE this is not the agent's own question — the loop guard stopped a
# non-productive agent (app/core/loop_guard.py) and the CM must actively steer it.
_STALL_INTERVENTION_NOTE = (
    "[SYSTEM — STALL INTERVENTION. The message below was generated automatically "
    "because the agent working on this task is looping without progress, NOT by "
    "the agent. The agent is stopped and waiting for your reply; its session is "
    "still open and the task is still in progress. Do NOT call close_session. "
    "Your job now: look at the task and, if useful, the workspace, then reply "
    "with ONE concrete way forward — a specific next step, or a pick between "
    "2-3 named approaches. If the task's own wording is what's blocking progress "
    "(ambiguous, or it points at something that isn't in the workspace), call "
    "task_update(task_id=<id from Task Data>, description=<full corrected text>) "
    "in THIS turn before replying. Be directive: the agent already tried the "
    "obvious path.]\n\n"
)
# Injected into a resumed agent's history when Task.stall_streak is already over
# the restart threshold (Signal B) — the previous attempts died without a single
# file change, so the resumed session must re-plan out loud before touching a tool.
_STALL_RESTART_DIRECTIVE = (
    "\n\n[SYSTEM — this task has now been restarted {streak} times and every "
    "attempt ended without changing a single file. Do not resume the same "
    "discovery. Your FIRST action this session must be a plain-text message to "
    "{cm} (no tool call): state what is actually blocking you and lay out 2-3 "
    "genuinely different approaches, then ask which to take. Only start tool "
    "calls after they answer.]"
)

# CM puts this sentinel in SessionCloseActions.summary to signal rejection.
_REJECT_SENTINEL = "TASK_REJECTED"
# CM puts this sentinel in summary to return the task for rework instead of rejecting.
_CHANGES_SENTINEL = "CHANGES_REQUESTED"
# The approval marker. Approval is still "no other marker", but a summary that
# opens with this one outranks a sentinel mentioned later in prose.
_APPROVE_SENTINEL = "APPROVED"
MAX_TASK_REWORKS = int(os.getenv("MAX_TASK_REWORKS", "2"))

VERDICT_APPROVE = "approve"
VERDICT_CHANGES = "request_changes"
VERDICT_REJECT = "reject"
_VERDICT_BY_SENTINEL = {
    _APPROVE_SENTINEL: VERDICT_APPROVE,
    _CHANGES_SENTINEL: VERDICT_CHANGES,
    _REJECT_SENTINEL: VERDICT_REJECT,
}
_VERDICT_ALIASES = {
    "approve": VERDICT_APPROVE,
    "approved": VERDICT_APPROVE,
    "request_changes": VERDICT_CHANGES,
    "changes_requested": VERDICT_CHANGES,
    "reject": VERDICT_REJECT,
    "rejected": VERDICT_REJECT,
}
# A marker is *directive* when it opens the summary or one of its lines, allowing
# for markdown/quoting punctuation ("**CHANGES_REQUESTED** — ..."). Anything else
# is prose: "Both CHANGES_REQUESTED fixes are verified" is a sentence about an
# earlier verdict, not a new one. Reading that as a verdict is exactly how an
# approval came back as a rework request and then, once the rework budget was
# spent, as a rejection whose stated reason read "APPROVED".
_DIRECTIVE_MARKER_RE = re.compile(
    r"""^[\s>*_#\-\[("'`]*(?:verdict\s*[:\-—]\s*)?"""
    r"(APPROVED|CHANGES_REQUESTED|TASK_REJECTED)\b",
    re.MULTILINE,
)


def classify_review_verdict(tool_input: dict[str, Any] | None) -> str:
    """Return the CM's verdict for a close_session call on a submitted task.

    Priority:
      1. the explicit ``verdict`` field of close_session — unambiguous, and what
         the review prompt now asks for;
      2. the first marker in a directive position inside the summary;
      3. a bare sentinel anywhere in the summary, kept so a CM that writes
         "...please fix X. CHANGES_REQUESTED" is still understood.
    """
    data = tool_input or {}
    raw_verdict = str(data.get("verdict") or "").strip().lower()
    if raw_verdict in _VERDICT_ALIASES:
        return _VERDICT_ALIASES[raw_verdict]

    summary = str(data.get("summary") or "")
    match = _DIRECTIVE_MARKER_RE.search(summary)
    if match is not None:
        return _VERDICT_BY_SENTINEL[match.group(1)]

    if _REJECT_SENTINEL in summary:
        return VERDICT_REJECT
    if _CHANGES_SENTINEL in summary:
        return VERDICT_CHANGES
    return VERDICT_APPROVE

# Tool-call budget for the agent side of a proxy pair. Coder tasks routinely
# need dozens of read/write/exec calls, so this must be far above the CM's cap.
#
# This used to be the primary defense against ballooning "context baggage" —
# every resumed session replays its full history, so more calls meant a bigger
# prompt on every subsequent turn. That job now belongs to the window-mode
# context preflight (context_policy.uses_window_limit(), default-on since
# CONTEXT_LIMIT_MODE=window): it caps and progressively compacts the assembled
# prompt on every single request, independent of how many tool calls happened.
# A duplicate read that costs ~50 tokens (see ReadFileTool's re-read receipt)
# no longer needs to be treated the same as a full raw file dump.
#
# Raised from 75 to 1000: the tight budget forced ordinary coding tasks through
# repeated tool_limit resumes, and each resume reset every loop-guard counter —
# turning the safety valve into an infinite loop with no circuit breaker. The
# real defence against a non-productive loop is now app/core/loop_guard.py
# (repeated-call + low-diversity detection → ChatMaster intervention), plus the
# unchanged wall-clock (_PROXY_MAX_DURATION_SECONDS) and turn-cap
# (_MAX_PROXY_TURNS) backstops. This number stays only as a last-resort ceiling.
_AGENT_MAX_TOOL_CALLS = int(os.getenv("PROXY_AGENT_MAX_TOOL_CALLS", "1000"))

# Tool-call budget for the ChatMaster side of a proxy pair.
#
# A review turn reads the [SYSTEM VERIFICATION] block and rules on it, so a
# handful of calls is enough. An *answer* turn is different: a mid-task question
# and, above all, a loop-guard stall intervention explicitly ask the CM to look
# at the workspace before naming a way forward. Five calls ran out in the middle
# of that inspection, and exhaustion then forced a close_session
# (tool_choice=close_session) — turning the answer into a CHANGES_REQUESTED
# verdict, tearing the pair down and restarting the stalled agent from zero with
# every loop-guard counter reset. That is why the stall detector looked like it
# "did nothing". Answer turns get their own budget and, on exhaustion, are made
# to produce plain text instead of a verdict (on_exhaust="answer").
_CM_MAX_TOOL_CALLS = int(os.getenv("PROXY_CM_MAX_TOOL_CALLS", "5"))
_CM_ANSWER_MAX_TOOL_CALLS = int(os.getenv("PROXY_CM_ANSWER_MAX_TOOL_CALLS", "20"))

# Final instruction of an exhausted answer turn (on_exhaust="answer").
_NO_TOOLS_LEFT_ANSWER_PROMPT = (
    "The tool call limit for this turn is exhausted — there are no more tools, "
    "and close_session must not be called: the counterpart's session continues and is waiting "
    "for your reply. Answer in plain text right now, based on what you have already "
    "found out: one concrete next step or a choice among 2-3 approaches."
)


def _preflight_limit_display(reason: str | None, policy: Any) -> int:
    """Return the threshold that actually caused a preflight rejection.

    Used in log messages so the printed limit matches the enforcement point,
    not just the outermost context_window_limit.
    """
    if reason == "prompt_cap":
        return min(policy.input_cap, policy.model_context_limit)
    if reason == "model_context_limit":
        return policy.model_context_limit
    return policy.context_window_limit


class ToolLimitReached(Exception):
    """The agent exhausted the workspace tool-call budget."""

    reason = "tool_calls"

    def __init__(
        self,
        working: list[dict[str, Any]],
        tokens_in: int,
        tokens_out: int,
        model: str,
        *,
        tool_calls_used: int = 0,
        tool_calls_limit: int = 0,
        tokens_used: int | None = None,
        tokens_limit: int = TOKEN_LIMIT,
        max_window_in: int = 0,
        last_window_in: int = 0,
    ):
        super().__init__("tool call limit reached")
        self.working = working
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model
        self.tool_calls_used = tool_calls_used
        self.tool_calls_limit = tool_calls_limit
        self.tokens_used = tokens_in + tokens_out if tokens_used is None else tokens_used
        self.tokens_limit = tokens_limit
        self.max_window_in = max_window_in
        self.last_window_in = last_window_in


class RunStopped(RuntimeError):
    """A user/watchdog/retry fenced this proxy generation while it was running."""

    def __init__(self, task_id: UUID | None, run_id: str | None, reason: str = "stopped"):
        super().__init__(f"proxy run stopped (task={task_id}, run={run_id})")
        self.task_id = task_id
        self.run_id = run_id
        self.reason = reason
        # Populated by _call_llm_with_tools before the exception leaves the
        # provider loop, so the caller can persist the in-flight tool-loop
        # transcript instead of discarding a partially-completed turn (a pause
        # mid-turn otherwise loses every tool call made since the last handoff).
        self.working: list[dict[str, Any]] | None = None
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.model: str = ""
        self.max_window_in: int = 0
        self.last_window_in: int = 0


class ProxyRunTimeout(RunStopped):
    """The proxy exceeded its total wall-clock budget."""

    def __init__(self, task_id: UUID | None, run_id: str | None):
        super().__init__(task_id, run_id, reason="proxy_timeout")


class McpCapabilityCircuitOpen(RuntimeError):
    """Repeated MCP execution failures stop the task before more tool calls."""

    def __init__(self, tool_name: str):
        super().__init__(f"MCP capability circuit open for {tool_name}")
        self.tool_name = tool_name


class TokenBudgetLimitReached(ToolLimitReached):
    """The session's cumulative token budget was exhausted.

    Reserved for the accumulated-usage case checked outside the assembler
    (see ``token_budget.check_token_budget``).  For preflight rejections that
    have nothing to do with cumulative usage, use ``PromptCapLimitReached`` or
    ``ContextWindowLimitReached``.  Callers must branch on ``reason`` rather
    than the base type.
    """

    reason = "token_budget"

    def __init__(
        self,
        working: list[dict[str, Any]],
        tokens_in: int,
        tokens_out: int,
        model: str,
        *,
        tool_calls_used: int = 0,
        tool_calls_limit: int = 0,
        tokens_used: int = 0,
        tokens_limit: int = TOKEN_LIMIT,
        max_window_in: int = 0,
        last_window_in: int = 0,
    ):
        super().__init__(
            working,
            tokens_in,
            tokens_out,
            model,
            tool_calls_used=tool_calls_used,
            tool_calls_limit=tool_calls_limit,
            tokens_used=tokens_used,
            tokens_limit=tokens_limit,
            max_window_in=max_window_in,
            last_window_in=last_window_in,
        )


class PromptCapLimitReached(ToolLimitReached):
    """The assembled prompt was rejected by ``prompt_cap`` or ``model_context_limit``.

    Distinct from ``TokenBudgetLimitReached``: not caused by cumulative usage,
    but by a single request whose non-compactable parts (system prompt +
    tool schemas + protected messages) alone exceed the input-cap policy.
    In practice this signals a bloated system prompt (e.g. an oversized task
    handoff) rather than a session that "ran out of tokens" — the UI must not
    show "used 0 of N tokens" for this.  Distinguish via ``exc.reason``:
    ``prompt_cap`` is the administrative input cap and
    ``model_context_limit`` is the provider capacity.
    """

    reason = "prompt_cap"

    def __init__(
        self,
        working: list[dict[str, Any]],
        tokens_in: int,
        tokens_out: int,
        model: str,
        *,
        tool_calls_used: int = 0,
        tool_calls_limit: int = 0,
        tokens_used: int = 0,
        tokens_limit: int = TOKEN_LIMIT,
        max_window_in: int = 0,
        last_window_in: int = 0,
        rejected_reason: str = "prompt_cap",
        estimated_input_tokens: int = 0,
        cap: int = 0,
        cap_env_var: str = "CONTEXT_AGENT_INPUT_CAP",
    ):
        super().__init__(
            working,
            tokens_in,
            tokens_out,
            model,
            tool_calls_used=tool_calls_used,
            tool_calls_limit=tool_calls_limit,
            tokens_used=tokens_used,
            tokens_limit=tokens_limit,
            max_window_in=max_window_in,
            last_window_in=last_window_in,
        )
        self.rejected_reason = rejected_reason
        # Keep provider-cap rejections distinguishable in events and UI
        # payloads; both remain this exception type for backwards-compatible
        # callers that already catch it.
        self.reason = rejected_reason
        self.estimated_input_tokens = estimated_input_tokens
        self.cap = cap
        self.cap_env_var = cap_env_var


class ContextWindowLimitReached(ToolLimitReached):
    """The assembled prompt exceeded the per-call context-window limit (window mode).

    Subclasses ToolLimitReached so existing ``except ToolLimitReached`` callers
    catch it automatically.  Distinguish via ``exc.reason == "context_window"``.
    """

    reason = "context_window"

    def __init__(
        self,
        working: list[dict[str, Any]],
        tokens_in: int,
        tokens_out: int,
        model: str,
        *,
        tool_calls_used: int = 0,
        tool_calls_limit: int = 0,
        tokens_used: int = 0,
        context_window_limit: int = TOKEN_LIMIT,
        max_window_in: int = 0,
        last_window_in: int = 0,
    ):
        super().__init__(
            working,
            tokens_in,
            tokens_out,
            model,
            tool_calls_used=tool_calls_used,
            tool_calls_limit=tool_calls_limit,
            tokens_used=tokens_used,
            tokens_limit=context_window_limit,
            max_window_in=max_window_in,
            last_window_in=last_window_in,
        )
        self.context_window_limit = context_window_limit


class RepeatedToolErrorLimitReached(ToolLimitReached):
    """The model repeated the same deterministic tool error."""

    reason = "repeated_tool_error"

    def __init__(
        self,
        *args,
        error_key: str | None = None,
        trigger_tool: str | None = None,
        repeat_count: int = 0,
        safe_error: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.error_key = error_key
        # Which tool/error actually repeated — without these the UI can only
        # say "some tool failed twice", never which one (internal design notes,
        # part 4, stage 1).
        self.trigger_tool = trigger_tool
        self.repeat_count = repeat_count
        self.safe_error = safe_error


# reason -> primary suggested next action for SessionInterruptionCard-style UI.
# Kept next to the reason enum itself so a new ToolLimitReached subclass and
# its recommended action are added together, not in two different files.
_RECOMMENDED_ACTION_BY_REASON: dict[str, str] = {
    "tool_calls": "resume",
    "token_budget": "handoff",
    "context_window": "resume",
    "prompt_cap": "restart",
    "model_context_limit": "restart",
    "repeated_tool_error": "resume_with_correction",
    # Infrastructure failures: nothing about the task is wrong, run it again.
    "provider_error": "retry",
    "timeout": "retry",
}

# Interruption reasons that are persisted as TASK_STALLED (via
# _block_proxy_for_runtime_failure) rather than TOOL_LIMIT_REACHED — the
# readback in api/v1/tasks.py filters on this so the watchdog's own stall
# reasons never surface as a "last interruption".
INFRASTRUCTURE_INTERRUPTION_REASONS = ("provider_error", "timeout")


def _recommended_action_for_reason(reason: str) -> str:
    return _RECOMMENDED_ACTION_BY_REASON.get(reason, "resume")


# Human-readable labels for the persistent AGENT_TOOL_CALL chat events.
_TOOL_LABELS: dict[str, str] = {
    "read_file": "reads a file",
    "write_file": "writes a file",
    "edit_file": "edits a file",
    "bash": "runs a command",
    "grep": "searches the code",
    "glob": "looks at files",
    "find_file": "looks for a file",
    "git_diff": "views the diff",
    "ask_user": "asks the user a question",
    "task_list": "views the task list",
    "task_update": "updates a task",
    "task_create": "creates a task",
    "plan_read": "reads the plan",
    "run_subtask": "starts a sub-step",
    "todo_write": "updates the plan",
}

_TODO_STATUS_ICONS: dict[str, str] = {
    "pending": "○",
    "in_progress": "▶",
    "done": "✅",
}


def _todo_write_summary(args: dict[str, Any]) -> str | None:
    items = args.get("items")
    if not isinstance(items, list) or not items:
        return None
    parts = []
    for item in items:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        icon = _TODO_STATUS_ICONS.get(str(item.get("status") or ""), "•")
        parts.append(f"{icon} {title}")
    return "; ".join(parts) if parts else None


def _tool_call_target(tool_name: str, args: dict[str, Any]) -> str | None:
    """Short human-readable target of a tool call (path, command, pattern)."""
    if tool_name == "todo_write":
        target = _todo_write_summary(args)
    else:
        target = (
            args.get("path") or args.get("cmd") or args.get("command")
            or args.get("pattern") or args.get("step_title")
        )
    if target is None:
        return None
    target = str(target).strip().replace("\n", " ")
    return target[:120] + "…" if len(target) > 120 else target


async def _resolve_observation_path(db: Any, room_id: Any, ref: Any) -> str | None:
    """Map an observation ref back to its workspace path for loop_guard Signal C.

    ``observation_read`` addresses a stored payload by ref; the read-loop guard
    needs the underlying file path so a ``read_file`` and an ``observation_read``
    of the same file collapse onto one target. Best-effort — any failure just
    means the guard can't tie this call to a path.
    """
    if not ref:
        return None
    try:
        from app.core.context_observations import get_observation

        artifact = await get_observation(db, str(ref), room_id=room_id)
    except Exception:
        return None
    if artifact is None:
        return None
    meta = getattr(artifact, "artifact_metadata", None) or {}
    path = meta.get("path")
    return str(path) if path else None


def _tool_error_escalation(error_repeats: int, max_repeats: int) -> tuple[str | None, str | None]:
    """Two-stage response to an identical tool error repeating.

    Stage 1 (``error_repeats == max_repeats``): inject a warning into the
    tool result and let the loop continue — the model gets one chance to
    read it, explain its goal in plain text, and try something different
    before anything is fatal.

    Stage 2 (``error_repeats > max_repeats``): the same error happened again
    even after the warning, so give up automatically the same way this
    always worked — stop the loop and let the human decide.

    Returns ``(warning, fatal)``; exactly one is non-None, or both are None
    below the threshold.
    """
    if error_repeats == max_repeats:
        return (
            "This exact tool call has now failed the same way "
            f"{error_repeats} times in a row. Do not repeat it as-is. Before your next "
            "tool call, explain in plain text what you are trying to accomplish and why "
            "this approach keeps failing, then try a genuinely different tool, argument, "
            "or strategy. Repeating this same call again will end the session.",
            None,
        )
    if error_repeats > max_repeats:
        return (
            None,
            "The same deterministic tool error was repeated even after a warning. "
            "Automatic retries are stopped; apply the corrective action from the error "
            "message or choose a different tool call.",
        )
    return None, None


async def _emit_tool_call_event(
    *,
    room_id: UUID,
    session_id: UUID,
    task_id: UUID | None,
    agent_id: UUID | None,
    actor_label: str | None,
    tool_name: str,
    args: dict[str, Any],
    db: AsyncSession,
    subtask_activity: dict[str, Any] | None = None,
) -> None:
    """Persist a user-visible AGENT_TOOL_CALL message for every tool invocation.

    Unlike emit_status (ephemeral WS), these survive reloads so the user can
    always see which files/commands an agent touched during a task.
    Failures are swallowed — chat visibility must never break the tool loop.
    """
    try:
        label = _TOOL_LABELS.get(tool_name, f"calls {tool_name}")
        target = _tool_call_target(tool_name, args)
        who = actor_label or "Agent"
        content = f"{who} {label}" + (f": {target}" if target else "")
        metadata: dict[str, Any] = {
            "session_id": str(session_id),
            "tool": tool_name,
            "target": target,
            "nickname": actor_label,
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
            task_id=task_id,
            metadata=metadata,
        )
    except Exception as exc:
        logger.warning("proxy_session: emit AGENT_TOOL_CALL failed (session=%s): %s", session_id, exc)


async def _emit_model_opt_in_required(
    exc: Exception,
    *,
    room_id: UUID,
    task_id: UUID | None,
    llm_config: dict[str, Any],
    agent_nickname: str | None,
    db: AsyncSession,
) -> bool:
    """Persist MODEL_OPT_IN_REQUIRED when the exception is an opt-in 403.

    Reasoning/router plan stage 9 (§10.3): the key is fine, the model is locked
    behind a browser-side workspace toggle, and the gateway's error body
    carries the working link.  Without this event the turn just looks dead.
    Returns True when the event was emitted; failures are swallowed — chat
    visibility must never break the error cleanup path that calls this.
    """
    try:
        category, opt_in_url = classify_gateway_error(exc)
        if category not in OPT_IN_REASONS:
            return False
        model = llm_config.get("model")
        await emit_event(
            room_id=room_id,
            event_type="MODEL_OPT_IN_REQUIRED",
            content=model_opt_in_content(model, category),
            db=db,
            task_id=task_id,
            metadata={
                "level": "warning",
                "model": model,
                "reason": category,
                "opt_in_url": opt_in_url,
                "agent_nickname": agent_nickname,
                "provider": (llm_config.get("provider") or "").lower() or None,
            },
        )
        return True
    except Exception as emit_exc:
        logger.warning(
            "proxy_session: emit MODEL_OPT_IN_REQUIRED failed (room=%s): %s",
            room_id, emit_exc,
        )
        return False

from app.core.model_pricing import cache_cost_adjustment, compute_cost as _compute_cost


def _extract_subtask_usage(result_text: Any) -> tuple[int, int, int]:
    """Extract child-loop usage from a compact ``run_subtask`` result.

    ``run_subtask`` deliberately returns JSON instead of the child transcript.
    Keep parsing defensive because malformed/legacy tool results must not break
    the parent loop or make usage accounting go backwards.
    """
    if not isinstance(result_text, str):
        return 0, 0, 0
    try:
        report = json.loads(result_text)
    except (TypeError, ValueError):
        return 0, 0, 0
    if not isinstance(report, dict):
        return 0, 0, 0

    def _non_negative_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    tokens_in = _non_negative_int(report.get("tokens_in"))
    tokens_out = _non_negative_int(report.get("tokens_out"))
    tool_calls_used = _non_negative_int(report.get("tool_calls_used"))

    # Accept reports produced before the split fields were added.  Charging
    # their aggregate usage as input preserves the total budget accounting;
    # current reports always take the precise branch above.
    if "tokens_in" not in report and "tokens_out" not in report:
        tokens_in = _non_negative_int(report.get("tokens_used"))

    return tokens_in, tokens_out, tool_calls_used


def _record_provider_cache_metric(
    *,
    provider: str,
    model: str,
    role: str,
    usage: Any,
) -> None:
    cache = extract_cache_usage(usage, provider)
    if not cache.read_tokens and not cache.write_tokens:
        return
    try:
        from app.core.metrics import record_provider_cache_usage

        record_provider_cache_usage(
            provider=provider,
            model=model,
            role=role,
            read_tokens=cache.read_tokens,
            write_tokens=cache.write_tokens,
        )
    except Exception:
        logger.debug("proxy_session: provider cache metric recording failed", exc_info=True)


# ---------------------------------------------------------------------------
# Conversation history helpers
# ---------------------------------------------------------------------------


def _get_history(session: Session) -> list[dict[str, str]]:
    snap = session.context_snapshot or {}
    return list(snap.get("messages", []))


def _save_history(session: Session, history: list[dict[str, str]]) -> None:
    snap = dict(session.context_snapshot or {})
    snap["messages"] = history
    session.context_snapshot = snap
    # Stage 5.3: observable size of the conversation without decoding
    # context_snapshot. turn_count counts user-authored turns; the proxy loop's
    # history is seeded with a synthetic "user" kickoff message, so this is the
    # number of handoffs/replies exchanged, not literally human messages.
    session.message_count = len(history)
    session.turn_count = sum(1 for m in history if m.get("role") == "user")


_STOP_INTERRUPT_NOTE = "Execution interrupted: the session was paused."


def _seal_dangling_tool_calls(working: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Close tool calls the stop fence interrupted before their results landed.

    A pause between the assistant's tool_use turn and its tool_result turn
    leaves an unanswered call; replaying that to either provider is a hard
    error. Synthesise error results so the transcript stays valid and the
    resumed agent sees the calls were cut off.
    """
    if not working:
        return working
    last = working[-1]
    if last.get("role") != "assistant":
        return working

    content = last.get("content")
    # Anthropic wire: content is a list that may hold tool_use blocks.
    if isinstance(content, list):
        pending = [
            b.get("id")
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        ]
        if pending:
            working = working + [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tid,
                        "content": _STOP_INTERRUPT_NOTE,
                        "is_error": True,
                    }
                    for tid in pending
                ],
            }]
        return working

    # OpenAI wire: assistant message carries a tool_calls array.
    tool_calls = last.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        working = working + [
            {
                "role": "tool",
                "tool_call_id": (tc.get("id") if isinstance(tc, dict) else None),
                "content": _STOP_INTERRUPT_NOTE,
            }
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id")
        ]
    return working


async def _persist_stopped_progress(
    session: Session,
    exc: "RunStopped",
    db: AsyncSession,
    llm_config: dict[str, Any] | None = None,
) -> bool:
    """Save the in-flight tool-loop transcript a stop fence interrupted.

    A user/workspace pause trips the fence mid-turn; without this the partial
    turn (every tool call since the last handoff) is lost and Resume restarts
    the agent from almost nothing. Mirrors the ToolLimitReached handler, minus
    the auto-summary. Safe on an already-closed session — Resume reads
    ``context_snapshot`` from the row, not from whoever closed it first.
    """
    working = getattr(exc, "working", None)
    if not working or not isinstance(working, list):
        return False
    working = _seal_dangling_tool_calls(working)
    _save_history(session, working)
    tokens_in = int(getattr(exc, "tokens_in", 0) or 0)
    tokens_out = int(getattr(exc, "tokens_out", 0) or 0)
    if tokens_in or tokens_out:
        track_llm_usage(
            session, tokens_in, tokens_out,
            _compute_cost(
                getattr(exc, "model", "") or "", tokens_in, tokens_out, llm_config
            ),
            model=getattr(exc, "model", "") or None,
        )
    update_context_window_metrics(
        session,
        int(getattr(exc, "max_window_in", 0) or 0),
        int(getattr(exc, "last_window_in", 0) or 0),
    )
    db.add(session)
    await db.flush()
    return True


# ---------------------------------------------------------------------------
# Image-prompt tool definitions (Phase A of media tasks)
# ---------------------------------------------------------------------------

class ImagePromptValidationError(Exception):
    """Raised when Phase A returns an invalid image prompts list."""


MEDIA_PROMPTGEN_SUFFIX = (
    "\n\n---\n\n"
    "You are generating a set of image prompts for the task above.\n"
    "Apply your artistic skills (see Skills section) to define a consistent\n"
    "visual language for the WHOLE series: style, lighting, palette, mood.\n"
    "Rules:\n"
    "- one entry per image, each prompt describes exactly ONE image, in English;\n"
    "- do NOT include acceptance criteria, file formats or task metadata in prompts;\n"
    "- filenames: short kebab-case slugs (e.g. 'cafe-interior', 'eggs-benedict');\n"
    "- if the image is a UI mockup or app screen meant for a front-end engineer\n"
    "  to implement, prompt for a flat, straight-on screenshot-style mockup —\n"
    "  no camera perspective, no photographic lighting or shadows — with crisp,\n"
    "  legible text, and pick a fixed aspect_ratio suited to the target device;\n"
    "- if the task lists 'Reference files' above, attach the relevant ones to\n"
    "  each image via reference_paths (copy the exact path shown) — 'logo'/'style'\n"
    "  keep a look consistent, 'subject' keeps a specific person/object recognizable,\n"
    "  'layout' keeps a composition; a diffusion model reinterprets a referenced\n"
    "  logo's exact pixels/letters rather than reproducing them, so still describe\n"
    "  the logo in the prompt (position, rough shape) instead of relying on the\n"
    "  reference alone; never invent a reference path that isn't listed;\n"
    "- a reference with role 'logo_overlay' will be pasted on the final image\n"
    "  deterministically after generation (exact logo pixels, not a reimagined\n"
    "  version); describe a clean, uncluttered area at the target anchor in your\n"
    "  prompt so the generated content does not clash with the overlay (e.g.\n"
    "  'top-left corner kept clear for logo placement, subtle dark background');\n"
    "  you may still include the path in reference_paths to let the model see it\n"
    "  as composition context, but the overlay happens regardless;\n"
    "- if the image must fit a specific layout slot (a banner, an avatar, an\n"
    "  og-image) set an exact 'size' in pixels (e.g. '1000x100') instead of\n"
    "  aspect_ratio — size always wins; for a very wide/narrow size, say so\n"
    "  explicitly in the prompt ('wide horizontal banner, subject centered,\n"
    "  empty margins left and right') so the composition survives the crop;\n"
    "- leave aspect_ratio/size/fit/reference_paths out entirely to inherit the\n"
    "  agent's configured defaults — they are optional on every image;\n"
    "- return the full list via the submit_image_prompts tool in a single call."
)

_IMAGE_PROMPTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "kebab-case slug, no extension",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "English, single focused image",
                    },
                    "aspect_ratio": {
                        "type": "string",
                        "description": (
                            "e.g. 1:1, 3:2, 4:3, 16:9, 9:16, 21:9. "
                            "Omit to use the agent default."
                        ),
                    },
                    "size": {
                        "type": "string",
                        "description": (
                            "exact pixels WxH (e.g. 1000x100) when the image must fit "
                            "a specific layout block; overrides aspect_ratio"
                        ),
                    },
                    "fit": {
                        "type": "string",
                        "enum": ["cover", "contain", "stretch"],
                    },
                    "reference_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": MEDIA_REF_MAX_PER_IMAGE,
                        "description": (
                            "workspace paths from the Reference files list to guide "
                            "this image"
                        ),
                    },
                },
                "required": ["filename", "prompt"],
            },
        }
    },
    "required": ["images"],
}

ANTHROPIC_IMAGE_PROMPTS_TOOL: dict[str, Any] = {
    "name": "submit_image_prompts",
    "description": "Submit the final list of image generation prompts for this task.",
    "input_schema": _IMAGE_PROMPTS_SCHEMA,
}

OPENAI_IMAGE_PROMPTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_image_prompts",
        "description": "Submit the final list of image generation prompts for this task.",
        "parameters": _IMAGE_PROMPTS_SCHEMA,
    },
}


# ---------------------------------------------------------------------------
# Transient-error retry for individual LLM requests
#
# A single dropped connection to the provider used to kill the whole proxy
# session (close_reason=error). Retry ONLY the failing HTTP request — never
# the surrounding tool loop, which may already have executed mutating tools.
# ---------------------------------------------------------------------------

_LLM_RETRY_BACKOFF_S: tuple[float, ...] = (2.0, 5.0, 10.0)
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 529})


# internal design notes Stage 0.1: a ceiling on ONE model request.  The
# clients' ``timeout=120`` is a per-read timeout, and OpenRouter keeps a
# non-streamed request alive with whitespace, so a request the upstream never
# answers used to hang until PROXY_MAX_DURATION_SECONDS (40 min).  Generous on
# purpose: high-effort reasoning legitimately runs for minutes.
_LLM_ATTEMPT_TIMEOUT_S = float(os.getenv("LLM_ATTEMPT_TIMEOUT_S", "600"))


class EmptyLLMChoicesError(RuntimeError):
    """An OpenAI-compatible provider returned a successful but empty reply."""


class LLMAttemptTimeout(RuntimeError):
    """One model request outlived ``_LLM_ATTEMPT_TIMEOUT_S`` — retried."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__(f"no answer in {int(timeout_s)}s")
        self.timeout_s = timeout_s


class LLMCallRestarted(RuntimeError):
    """The user pressed "Abort the request and retry" (internal design notes
    Stage 3).  ``_llm_request_with_retry`` sends the same request again at once;
    the attempt is not counted against the retry budget."""

    def __init__(self) -> None:
        super().__init__("request aborted by the user, sent again")


async def _await_attempt(make_request, restart_event: asyncio.Event | None):
    """One model request, capped at ``_LLM_ATTEMPT_TIMEOUT_S`` and abortable
    through ``restart_event``.  A response that is already in wins over a
    restart that raced it — a finished answer is never thrown away."""
    if restart_event is None:
        try:
            return await asyncio.wait_for(make_request(), timeout=_LLM_ATTEMPT_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            # Only the attempt's own ceiling lands here: the proxy deadline
            # cancels this coroutine from outside instead.
            raise LLMAttemptTimeout(_LLM_ATTEMPT_TIMEOUT_S) from exc

    request = asyncio.ensure_future(make_request())
    restart = asyncio.ensure_future(restart_event.wait())
    try:
        await asyncio.wait(
            {request, restart},
            timeout=_LLM_ATTEMPT_TIMEOUT_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        restart.cancel()
        if not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
    if request.done() and not request.cancelled():
        return request.result()
    if restart_event.is_set():
        raise LLMCallRestarted()
    raise LLMAttemptTimeout(_LLM_ATTEMPT_TIMEOUT_S)


def _require_openai_choices(response: Any) -> None:
    """Raise a retryable error instead of indexing an empty ``choices`` list."""
    if not getattr(response, "choices", None):
        raise EmptyLLMChoicesError("provider returned an empty choices list")


def _is_retryable_llm_error(exc: BaseException) -> bool:
    import openai as _openai

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRYABLE_HTTP_STATUSES
    if isinstance(exc, httpx.TransportError):
        # ConnectError, ReadError, timeouts, dropped connections, …
        return True
    if isinstance(exc, EmptyLLMChoicesError):
        # Some OpenAI-compatible upstreams intermittently return HTTP 200 with
        # no choices. Retrying the request is safe: no tool has run yet.
        return True
    if isinstance(exc, LLMAttemptTimeout):
        return True
    if isinstance(exc, _openai.APIConnectionError):
        # includes APITimeoutError
        return True
    if isinstance(exc, _openai.APIStatusError):
        # includes RateLimitError (429) and InternalServerError (5xx)
        return exc.status_code in _RETRYABLE_HTTP_STATUSES
    return False


async def _execute_registry_tool(tool_name: str, ctx: Any, tool_input: dict[str, Any]) -> str:
    """Run one workspace-registry tool and turn its failure into a tool result.

    A tool that raises (a workspaced 409/5xx, a bug in the tool) must not end
    the session: unhandled, the exception reached the turn's outer handler and
    was shown as a model-provider error with the tool's HTTP status, and Retry
    hit it again.  The agent gets the error text instead, the way
    ``workspace_tools.execute_workspace_tool`` already does.  The run's own
    control-flow exceptions still propagate.
    """
    from app.core.tools import REGISTRY as _REGISTRY
    from app.core.tools.registry import safe_tool_error_text

    try:
        return await _REGISTRY.get(tool_name).execute(ctx, tool_input)
    except (RunStopped, ToolLimitReached, McpCapabilityCircuitOpen, LLMCallRestarted):
        raise
    except Exception as exc:
        logger.warning(
            "proxy_session: tool %s failed (session=%s): %s",
            tool_name, getattr(ctx, "session_id", None), exc, exc_info=True,
        )
        detail = safe_tool_error_text(str(exc))
        return f"Error: {detail}" if detail else f"Error: {type(exc).__name__}"


def _classify_llm_failure(exc: BaseException) -> tuple[str, str, int | None]:
    """Map an LLM call failure that survived the retry budget to
    ``(reason, safe_error, http_status)`` for the user-facing interruption card.

    ``reason`` is ``"timeout"`` when the request never got an answer and
    ``"provider_error"`` otherwise.  ``safe_error`` is one short line that is
    safe to persist in the chat: the exception text of an httpx error embeds
    the request URL (some providers carry the API key in its query string), so
    for those only the status line is used.
    """
    import openai as _openai

    status: int | None = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    elif isinstance(exc, _openai.APIStatusError):
        status = exc.status_code

    if isinstance(
        exc,
        (LLMAttemptTimeout, asyncio.TimeoutError, httpx.TimeoutException, _openai.APITimeoutError),
    ):
        return "timeout", "the model provider did not answer in time", None

    if isinstance(exc, httpx.HTTPStatusError):
        phrase = exc.response.reason_phrase or ""
        return "provider_error", f"HTTP {status} {phrase}".strip(), status
    if isinstance(exc, (httpx.TransportError, _openai.APIConnectionError)):
        return "provider_error", "the model provider is unreachable", None

    from app.core.tools.registry import safe_tool_error_text

    detail = safe_tool_error_text(str(exc))
    label = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
    return "provider_error", label, status


async def _llm_request_with_retry(
    make_request,
    *,
    session_id: Any,
    label: str,
    validate_response=None,
    provider: str | None = None,
    db: AsyncSession | None = None,
):
    """Await make_request(), retrying transient network / 429 / 5xx failures
    with backoff. Non-retryable errors and the final failure propagate.

    ``db`` (stage 1.2.1, internal design notes §8.2): the caller's session is
    committed before every attempt, so neither the model's answer nor the
    backoff sleep between retries is waited out holding a pool connection.
    All of the proxy loop's model calls, both sides, come through here — this
    one release is what lifts the ``pool_size + max_overflow`` room ceiling
    measured by the 0.4 ladder.

    Every LLM call on the agent side funnels through here, so this is also the
    one place that needs stage-0.3 timing (internal design notes §7).  Each
    attempt is timed on its own: the backoff sleeps between attempts are the
    Hub waiting, not the provider answering, and folding them into
    ``llm_wait`` would understate ``hub_other`` exactly when the provider is
    misbehaving.

    Each attempt is also capped at ``_LLM_ATTEMPT_TIMEOUT_S`` and recorded in
    ``llm_call_state`` (attempt N of M, the previous error), which is what the
    heartbeat shows instead of the length of the whole turn.

    A manual restart (``LLMCallRestarted``, Stage 3) sends the request again at
    once and keeps the attempt number; ``llm_call_state`` caps those at
    ``RESTART_LIMIT`` per call.
    """
    max_attempts = len(_LLM_RETRY_BACKOFF_S) + 1
    attempt = 0
    restarts = 0
    while True:
        try:
            await release_db(db, reason=label)
            note_llm_call(SIDE_AGENT)
            restart_event = llm_call_state.begin_attempt(
                session_id,
                attempt=attempt + 1,
                max_attempts=max_attempts,
                label=label,
                provider=provider,
                restarts=restarts,
            )
            _attempt_t0 = time.perf_counter()
            try:
                response = await _await_attempt(make_request, restart_event)
            finally:
                record_llm_wait(provider, SIDE_AGENT, time.perf_counter() - _attempt_t0)
            if validate_response is not None:
                validate_response(response)
            llm_call_state.end_attempt(session_id)
            return response
        except LLMCallRestarted as exc:
            llm_call_state.end_attempt(session_id, error=exc)
            restarts += 1
            logger.warning(
                "proxy_session: %s aborted by the user (session=%s, attempt %d/%d, "
                "restart %d/%d) — sending again",
                label, session_id, attempt + 1, max_attempts,
                restarts, llm_call_state.RESTART_LIMIT,
            )
            continue
        except Exception as exc:
            llm_call_state.end_attempt(session_id, error=exc)
            if attempt >= len(_LLM_RETRY_BACKOFF_S) or not _is_retryable_llm_error(exc):
                raise
            delay = _LLM_RETRY_BACKOFF_S[attempt]
            logger.warning(
                "proxy_session: transient LLM error in %s (session=%s, "
                "attempt %d/%d): %s — retrying in %.0fs",
                label, session_id, attempt + 1,
                len(_LLM_RETRY_BACKOFF_S) + 1, exc, delay,
            )
            await asyncio.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# LLM call (close_session tool, multi-provider)
# ---------------------------------------------------------------------------


async def _get_agent_llm_config(agent_id: UUID, db: AsyncSession) -> dict[str, Any]:
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalars().first()
    return (agent.llm_config or {}) if agent else {}


async def _call_llm(
    llm_config: dict[str, Any],
    system_prompt: str,
    history: list[dict[str, str]],
    or_session_id: str | None = None,
    *,
    anthropic_tool: dict[str, Any] | None = None,
    openai_tool: dict[str, Any] | None = None,
    tool_required: bool = False,
    secret_value: SecretValue | None = None,
    db: AsyncSession | None = None,
) -> tuple[str | None, dict[str, Any] | None, int, int, str]:
    """Call LLM with a tool (defaults to close_session).

    Returns (text, tool_input, tokens_in, tokens_out, model).
    Exactly one of text / tool_input is non-None.

    anthropic_tool / openai_tool: override the default close_session tool.
    tool_required: force the model to call the tool (no text-only reply).
    or_session_id: groups this call under an OpenRouter session for debugging.
    """
    _ant_tool = anthropic_tool if anthropic_tool is not None else ANTHROPIC_CLOSE_TOOL
    _oai_tool = openai_tool if openai_tool is not None else OPENAI_CLOSE_TOOL

    provider = llm_config.get("provider", "anthropic").lower()
    model = llm_config.get("model") or "claude-haiku-4-5-20251001"
    api_key = secret_value.reveal() if secret_value is not None else None
    if not api_key:
        legacy = legacy_secret_value(llm_config, "api_key")
        try:
            api_key = legacy.reveal() if legacy is not None else None
        finally:
            legacy = None
    base_url = llm_config.get("base_url")

    if uses_messages_wire(provider, model):
        if uses_anthropic_env_key(provider):
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        body: dict[str, Any] = {
            "model": model,
            "system": anthropic_system_payload(
                system_prompt,
                enabled=bool(llm_config.get("anthropic_prompt_caching")),
            ),
            "messages": history,
            "tools": [_ant_tool],
            "max_tokens": 4096,
        }
        if isinstance(llm_config.get("anthropic_context_management"), dict):
            body["context_management"] = llm_config["anthropic_context_management"]
        if tool_required:
            body["tool_choice"] = {"type": "any"}
        apply_reasoning_body(body, llm_config, provider, model)

        async def _anthropic_request():
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    messages_url(provider, base_url),
                    headers=messages_headers(provider, api_key),
                    json=body,
                )
                response.raise_for_status()
                return response.json()

        data = await _llm_request_with_retry(
            _anthropic_request,
            session_id=or_session_id,
            label="_call_llm/anthropic",
            provider=provider,
            db=db,
        )
        usage = data.get("usage", {})
        _record_provider_cache_metric(
            provider=provider,
            model=model,
            role="proxy",
            usage=usage,
        )
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)

        if data.get("stop_reason") == "tool_use":
            tool_block = next(
                (b for b in data.get("content", []) if b.get("type") == "tool_use"),
                None,
            )
            return None, (tool_block["input"] if tool_block else {}), tokens_in, tokens_out, model

        text = next(
            (b["text"] for b in data.get("content", []) if b.get("type") == "text"),
            "",
        )
        return text, None, tokens_in, tokens_out, model

    # OpenAI-compatible (openai, groq, deepseek, ollama, openrouter, …)
    api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-no-key-required"
    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 120.0}
    client_kwargs["base_url"] = openai_base_url(provider, base_url)
    client_kwargs.update(openai_client_extras(client_kwargs["base_url"]))

    oai = llm_client(provider, chat_client_factory=AsyncOpenAI, **client_kwargs)
    messages = [{"role": "system", "content": system_prompt}] + list(history)

    create_extra: dict[str, Any] = {}
    if provider == "openrouter" and llm_config.get("session_logging") and or_session_id:
        create_extra["extra_body"] = {"session_id": or_session_id}
    if llm_config.get("prompt_cache_key"):
        create_extra["prompt_cache_key"] = str(llm_config["prompt_cache_key"])
    apply_reasoning_chat(create_extra, llm_config, provider, model)

    _oai_tool_choice = "required" if tool_required else "auto"

    # Malformed tool-call JSON is a transient model failure — retry once.
    for attempt in range(2):
        resp = await _llm_request_with_retry(
            lambda: oai.chat.completions.create(
                model=model,
                messages=messages,
                tools=[_oai_tool],
                tool_choice=_oai_tool_choice,
                **create_extra,
            ),
            session_id=or_session_id,
            label="_call_llm/openai",
            provider=provider,
            db=db,
            validate_response=_require_openai_choices,
        )
        choice = resp.choices[0]
        usage = resp.usage
        _record_provider_cache_metric(
            provider=provider,
            model=model,
            role="proxy",
            usage=usage,
        )
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0

        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            try:
                tool_input = parse_tool_arguments(tool_call.function.arguments)
            except ToolArgumentsParseError:
                if attempt == 0:
                    logger.warning(
                        "proxy_session: malformed tool JSON — retrying LLM call"
                    )
                    continue
                raise
            return None, tool_input, tokens_in, tokens_out, model

        text = choice.message.content or ""
        return text, None, tokens_in, tokens_out, model


# ---------------------------------------------------------------------------
# Tool-loop LLM call (Stage 4: workspace tools)
# ---------------------------------------------------------------------------


async def _apply_db_restart_request(session_id: UUID) -> None:
    """Stage 3: pick up "Abort the request and retry" left in the DB.

    The restart endpoint signals ``llm_call_state`` directly when the loop runs
    in its own process; when it does not (scale-out), it stamps
    ``session.llm_call_restart_requested_at`` and this, on a heartbeat tick,
    consumes the stamp.  Only while a model request is in flight — one short
    indexed UPDATE per tick, on its own connection, released at once.
    """
    if llm_call_state.snapshot(session_id).get("phase") != llm_call_state.PHASE_LLM:
        return
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.db.session import engine

    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as short_db:
            requested_at = (
                await short_db.execute(
                    update(Session)
                    .where(
                        Session.id == session_id,
                        Session.llm_call_restart_requested_at.is_not(None),
                    )
                    .values(llm_call_restart_requested_at=None)
                    .returning(Session.llm_call_restart_requested_at)
                    .execution_options(synchronize_session=False)
                )
            ).scalar_one_or_none()
            await short_db.commit()
    except Exception:
        logger.debug("proxy_session: restart-flag check failed (session=%s)", session_id, exc_info=True)
        return
    if requested_at is None:
        return
    if requested_at.tzinfo is None:
        requested_at = requested_at.replace(tzinfo=timezone.utc)
    outcome = llm_call_state.request_restart(session_id, requested_at_wall=requested_at.timestamp())
    logger.info("proxy_session: DB restart request for session %s → %s", session_id, outcome)


async def _call_with_heartbeat(
    coro,
    *,
    room_id: UUID,
    agent_id: UUID | None,
    session_id: UUID,
    task_id: UUID | None,
    interval: float = _HEARTBEAT_INTERVAL_S,
    deadline_monotonic: float | None = None,
):
    """Await an LLM coroutine, broadcasting llm_heartbeat while it is in flight.

    Exceptions from the coroutine propagate unchanged, so callers keep their
    existing error handling.
    """
    llm_task = asyncio.ensure_future(coro)
    started = time.monotonic()
    try:
        while True:
            wait_timeout = interval
            # Re-read every pass: ``bash_wait`` inside this turn credits its
            # waiting time to the deadline while we are in flight.
            deadline = run_clock.effective_deadline(deadline_monotonic, task_id)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProxyRunTimeout(task_id, None)
                wait_timeout = min(wait_timeout, remaining)
            done, _ = await asyncio.wait({llm_task}, timeout=wait_timeout)
            if done:
                return llm_task.result()
            deadline = run_clock.effective_deadline(deadline_monotonic, task_id)
            if deadline is not None and time.monotonic() >= deadline:
                raise ProxyRunTimeout(task_id, None)
            # Stage 3: the restart aborts only the request in flight (inside
            # _llm_request_with_retry), not this whole turn — tool results the
            # turn already produced are kept.
            await _apply_db_restart_request(session_id)
            # ``elapsed_s`` is the whole turn (model requests AND tools); the
            # llm_call_state snapshot says what is actually in flight.
            await emit_status(
                room_id,
                AgentStatus.LLM_HEARTBEAT,
                agent_id=agent_id,
                session_id=session_id,
                task_id=task_id,
                extra={
                    "elapsed_s": int(time.monotonic() - started),
                    **llm_call_state.snapshot(session_id),
                },
            )
    finally:
        # ``wait_for`` cancels this heartbeat coroutine on a proxy deadline.
        # Do not leave the provider request running after the worker has been
        # fenced; otherwise it can publish a late tool result.
        if not llm_task.done():
            llm_task.cancel()
            await asyncio.gather(llm_task, return_exceptions=True)


async def _call_llm_with_tools(
    llm_config: dict[str, Any],
    system_prompt: str,
    history: list[dict[str, Any]],
    room_id: UUID,
    session_id: UUID,
    task_id: UUID | None,
    db: AsyncSession,
    *,
    workspace_ant_tools: list[dict[str, Any]] | None = None,
    workspace_oai_tools: list[dict[str, Any]] | None = None,
    mcp_ant_tools: list[dict[str, Any]] | None = None,
    mcp_oai_tools: list[dict[str, Any]] | None = None,
    mcp_bindings: dict[str, Any] | None = None,
    mcp_user_id: UUID | None = None,
    max_tool_calls: int = 15,
    or_session_id: str | None = None,
    actor_agent_id: UUID | None = None,
    actor_label: str | None = None,
    on_exhaust: str = "force_close",
    budget_used: int = 0,
    context_role: str = "unknown",
    secret_value: SecretValue | None = None,
    usage_stats: dict[str, int] | None = None,
    subtask_activity: dict[str, Any] | None = None,
    run_id: str | None = None,
    deadline_monotonic: float | None = None,
    mcp_failure_counts: dict[str, int] | None = None,
    loop_signals: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any] | None, int, int, str, int, int]:
    """Call LLM with workspace tools + close_session, looping on tool calls.

    Workspace tools are called in a local working list (history is read-only).
    The loop exits when the model returns text, calls close_session, or
    max_tool_calls is reached. What happens on exhaustion depends on on_exhaust:
      * "force_close" — a final forced-close request (tool_choice=close_session)
        summarizes the work done so far; (None, None, ...) only if that fails.
      * "answer" — a final request with no tools at all, so the turn ends in
        plain text. For turns that must produce a reply and not a verdict (the
        ChatMaster answering a mid-task question or a stall intervention):
        forcing close_session there would end a session that is still working.
      * "raise" — ToolLimitReached is raised carrying the working history so
        the caller can pause the session and let the user decide whether to
        resume it or restart the task.

    Every workspace tool call is also persisted to the room chat as an
    AGENT_TOOL_CALL message (see _emit_tool_call_event).

    Returns (text, close_tool_input, tokens_in, tokens_out, model, max_window_in, last_window_in).
    max_window_in / last_window_in track the maximum and last per-call prompt size
    (input_tokens from the provider) across all LLM iterations in this loop.
    Usage reported by an isolated ``run_subtask`` is rolled into the returned
    token totals and the parent tool-call counter before the next iteration.
    """
    from app.core.tools import REGISTRY as _REGISTRY, ToolContext as _ToolContext
    from app.core.tools.registry import (
        ToolErrorGuard,
        canonicalize_tool_input,
        safe_tool_error_text,
    )

    # The active provider loop registers its working transcript here so a stop
    # fence tripped mid-turn can carry the partial history out for the caller to
    # persist (see RunStopped.working / _run_proxy_loop's RunStopped handlers).
    _working_ref: dict[str, Any] = {"messages": None}
    # The ChatMaster's turns run through this same loop; label its steps as its
    # own so the profile and hub_llm_wait_open_tx_total{side} can tell the two
    # sides apart (§8.2: the target is zero for both, reported separately).
    _step_side = SIDE_CHAT_MASTER if context_role == "chat_master" else SIDE_AGENT

    def _attach_progress(exc: RunStopped) -> RunStopped:
        exc.working = _working_ref["messages"]
        exc.tokens_in = total_in
        exc.tokens_out = total_out
        exc.model = model
        exc.max_window_in = max_window_in
        exc.last_window_in = last_window_in
        return exc

    async def _ensure_run_active() -> None:
        _deadline = run_clock.effective_deadline(deadline_monotonic, task_id)
        if _deadline is not None and time.monotonic() >= _deadline:
            raise _attach_progress(ProxyRunTimeout(task_id, run_id))
        if task_id is None and context_role == "agent_subtask":
            # A run_subtask child of a direct dialogue has no task to fence, so
            # the Stop button (and a closed session) reaches it through the
            # parent session — otherwise the child ran on to its tool budget
            # while the parent was already told to stop.
            from app.core import session_interject
            from app.core.user_sessions import _session_was_closed_externally

            if session_interject.stop_requested(
                session_id
            ) or await _session_was_closed_externally(session_id, db):
                raise _attach_progress(RunStopped(None, run_id, reason="session_stopped"))
            return
        if task_id is None or run_id is None:
            return
        from app.core.task_runtime import is_run_current

        if not await is_run_current(task_id, run_id, db):
            raise _attach_progress(RunStopped(task_id, run_id))

    async def _touch_run_boundary() -> None:
        if task_id is None or run_id is None:
            return
        from app.core.task_runtime import touch_run_activity

        if not await touch_run_activity(session_id, task_id, run_id, db):
            raise _attach_progress(RunStopped(task_id, run_id))

    async def _deliver_interjections(inject) -> None:
        """internal design notes Stage 2: hand the executor the messages the
        user queued into this task, before the next model request.  The note
        is also left in ``loop_signals`` for the proxy loop to keep in the
        agent's history after the turn (this ``working`` list is discarded).
        """
        if context_role != "agent" or task_id is None:
            return
        from app.core import task_interject

        texts = await task_interject.deliver_pending(task_id, session_id, db)
        if not texts:
            return
        note = task_interject.format_note(texts)
        inject(note)
        if loop_signals is not None:
            loop_signals.setdefault("interjections", []).append(note)

    provider = llm_config.get("provider", "anthropic").lower()
    model = llm_config.get("model") or "claude-haiku-4-5-20251001"
    api_key = secret_value.reveal() if secret_value is not None else None
    if not api_key:
        legacy = legacy_secret_value(llm_config, "api_key")
        try:
            api_key = legacy.reveal() if legacy is not None else None
        finally:
            legacy = None
    base_url = llm_config.get("base_url")
    total_in = total_out = 0
    max_window_in = 0
    last_window_in = 0
    tool_calls_used = 0

    # Stage 6 (§7.3): reasoning tokens are already inside `output_tokens`, so
    # they are counted here only to show what share of the turn went into
    # thinking — never added to the totals, which would double-charge.
    reasoning_tokens_total = 0
    show_reasoning = show_reasoning_in_chat(llm_config)

    def _messages_reasoning(data: dict[str, Any]) -> tuple[str, int]:
        _text, thinking = messages_text_and_thinking(data)
        return thinking, messages_reasoning_tokens(data)

    async def _emit_reasoning(
        thinking: str, tokens: int, *, started_at: float
    ) -> None:
        nonlocal reasoning_tokens_total
        reasoning_tokens_total += max(0, int(tokens or 0))
        if usage_stats is not None:
            usage_stats["reasoning_tokens"] = reasoning_tokens_total
        if not show_reasoning:
            return
        await emit_reasoning_event(
            room_id=room_id,
            session_id=session_id,
            agent_id=actor_agent_id,
            actor_label=actor_label,
            task_id=task_id,
            thinking=thinking,
            reasoning_tokens=int(tokens or 0),
            duration_ms=round((time.monotonic() - started_at) * 1000),
            db=db,
        )

    async def _emit_interim_text(text: str | None) -> None:
        # A run_subtask child's steps are folded into the parent's subtask
        # card; its words stay out of the top-level chat with them.
        if subtask_activity is not None:
            return
        await emit_interim_text_event(
            room_id=room_id,
            session_id=session_id,
            agent_id=actor_agent_id,
            actor_label=actor_label,
            task_id=task_id,
            text=text,
            db=db,
        )

    def _publish_usage_stats() -> None:
        if usage_stats is not None:
            usage_stats.update({
                "tokens_in": max(0, int(total_in)),
                "tokens_out": max(0, int(total_out)),
                "tool_calls_used": max(0, int(tool_calls_used)),
                "reasoning_tokens": max(0, int(reasoning_tokens_total)),
            })
        # loop_guard: let the proxy loop reset Task.stall_streak when this turn
        # actually produced file changes (see _run_proxy_loop, agent turn).
        if loop_signals is not None and stall_guard is not None:
            loop_signals["productive_calls"] = stall_guard.productive_calls
            loop_signals["tool_calls"] = stall_guard.total_calls

    def _roll_up_subtask_usage(tool_name: str, result_text: Any) -> None:
        """Charge an isolated child loop to this parent tool loop."""
        nonlocal total_in, total_out, tool_calls_used
        if tool_name != "run_subtask":
            return
        child_in, child_out, child_tool_calls = _extract_subtask_usage(result_text)
        total_in += child_in
        total_out += child_out
        tool_calls_used += child_tool_calls
        if child_in or child_out or child_tool_calls:
            logger.debug(
                "proxy_session: rolled up run_subtask usage for session %s "
                "(in=%d out=%d tool_calls=%d)",
                session_id,
                child_in,
                child_out,
                child_tool_calls,
            )

    token_budget_exhausted = False
    context_window_exhausted = False
    prompt_cap_exhausted = False
    prompt_cap_reason: str | None = None
    prompt_cap_estimated: int = 0
    prompt_cap_cap: int = 0
    tool_limit_exhausted = False
    repeated_tool_error: str | None = None
    # Which tool/error actually repeated, for RepeatedToolErrorLimitReached
    # below — set alongside repeated_tool_error in both the anthropic and
    # openai tool-call branches (internal design notes, part 4, stage 1).
    repeated_tool_error_tool: str | None = None
    repeated_tool_error_repeats: int = 0
    repeated_tool_error_safe: str | None = None
    mcp_failure_counts = mcp_failure_counts if mcp_failure_counts is not None else {}
    tool_error_guard = ToolErrorGuard(
        max_repeats=max(1, int(llm_config.get("max_identical_tool_errors", 2) or 2))
    )
    # Non-productivity detector — only on the real agent side (not the CM, not
    # an isolated run_subtask child). Catches a loop that raw error-repeat
    # detection misses: successful reads/greps run over and over with no writes.
    from app.core.loop_guard import (
        READ_TARGET_UNSET,
        RepeatedToolCallGuard,
        build_stall_report,
        mcp_call_productive,
        mcp_read_target,
    )

    stall_guard = RepeatedToolCallGuard() if context_role == "agent" else None
    stall_signal: dict[str, Any] | None = None

    def _guard_mcp_call(
        tool_name: str, tool_args: dict[str, Any], result_text: str, *, paid: bool
    ) -> str:
        """Run both loop guards over a finished MCP call (Stage 10 of internal design notes).

        The MCP branch used to skip them entirely — the only ceiling on a
        repeating paid call was ``PROXY_AGENT_MAX_TOOL_CALLS``, hundreds of
        billed calls away. `paid` lowers Signal A's dup threshold by one and
        adds the cost to the warning; a successful VPS write or service action
        counts as progress so a real deploy is never read as thrashing.
        """
        nonlocal repeated_tool_error, repeated_tool_error_tool
        nonlocal repeated_tool_error_repeats, repeated_tool_error_safe, stall_signal

        if not isinstance(result_text, str):
            return result_text

        error_repeats = tool_error_guard.observe(tool_name, tool_args, result_text)
        _warning, _fatal = _tool_error_escalation(error_repeats, tool_error_guard.max_repeats)
        if _fatal:
            repeated_tool_error = _fatal
            repeated_tool_error_tool = tool_name
            repeated_tool_error_repeats = error_repeats
            repeated_tool_error_safe = safe_tool_error_text(result_text)
            result_text = f"{result_text}\n[{repeated_tool_error}]"
        elif _warning:
            result_text = f"{result_text}\n[{_warning}]"

        if stall_guard is None:
            return result_text
        _sv = stall_guard.observe(
            tool_name,
            tool_args,
            productive=mcp_call_productive(tool_name, result_text),
            result_text=result_text,
            read_target=mcp_read_target(tool_name, tool_args),
            paid=paid,
        )
        if _sv.warn:
            result_text = f"{result_text}\n[{_sv.warn}]"
        elif _sv.stall:
            stall_signal = {
                "reason": _sv.reason,
                "tool_name": _sv.tool_name,
                "repeat_count": _sv.repeat_count,
                "target": _sv.target,
            }
        return result_text

    def _stall_return() -> tuple[str | None, dict[str, Any] | None, int, int, str, int, int]:
        """Graceful exit that hands the loop to the ChatMaster via loop_signals."""
        signal = dict(stall_signal or {})
        signal["summary"] = stall_guard.summary_line() if stall_guard else ""
        if loop_signals is not None:
            loop_signals["stall"] = signal
        logger.warning(
            "proxy_session: stall detected in session %s (%s) — %s",
            session_id, signal.get("reason"), signal["summary"],
        )
        _publish_usage_stats()
        return (
            build_stall_report(signal), None,
            total_in, total_out, model, max_window_in, last_window_in,
        )

    has_ws_tools = bool(workspace_ant_tools or workspace_oai_tools or mcp_ant_tools or mcp_oai_tools)
    context_step = 0
    context_tracker = ToolCallStats()
    context_policy = await resolve_context_policy(
        role=context_role,
        db=db,
        room_id=room_id,
        agent_id=actor_agent_id,
        llm_config=llm_config,
    )
    context_assembler = ContextAssembler(context_policy)
    # Tool order and cache identity are deterministic for the whole loop.
    cache_tools_enabled = bool(
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

    def _observe_mcp_result(tool_name: str, result_text: str) -> str:
        """Make one MCP failure explicit and open a small capability breaker."""

        if not isinstance(result_text, str) or not result_text.startswith("Error: MCP"):
            return result_text
        failures = mcp_failure_counts.get(tool_name, 0) + 1
        mcp_failure_counts[tool_name] = failures
        safe_result = (
            f"{result_text}\nMCP capability is unavailable. Do not use raw "
            "HTTP/curl/requests or guess credentials."
        )
        if failures >= max(1, _MCP_FAILURE_CIRCUIT_LIMIT):
            raise McpCapabilityCircuitOpen(tool_name)
        return safe_result

    async def _record_provider_cache(usage: Any) -> None:
        cache_usage = extract_cache_usage(usage, provider)
        if not cache_usage.read_tokens and not cache_usage.write_tokens:
            return
        try:
            from app.core.metrics import record_provider_cache_usage

            record_provider_cache_usage(
                provider=provider,
                model=model,
                role=context_role,
                read_tokens=cache_usage.read_tokens,
                write_tokens=cache_usage.write_tokens,
            )
        except Exception:
            logger.debug("provider cache metric recording failed", exc_info=True)
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
        # ``context_window_limit`` constrains this assembled request, not the
        # sum of previous requests in the session.  The cumulative remainder
        # is retained solely for the explicit legacy rollback mode.
        session_remaining = (
            max(0, context_policy.context_window_limit - (budget_used + total_in + total_out))
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
                "proxy_session: context assembly metrics failed for %s: %s",
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
                "proxy_session: %s preflight rejected for session %s "
                "(reason=%s estimated_input=%d limit=%d)",
                request_kind,
                session_id,
                assembly.rejected_reason,
                assembly.estimated_input_tokens,
                _preflight_limit_display(assembly.rejected_reason, context_policy),
            )
        return assembly

    def _record_preflight_rejection(assembly: Any) -> None:
        """Map assembly failures to explicit loop states.

        ``session_budget`` is a legacy-only reason.  If it ever reaches a
        window-mode proxy loop, fail loudly instead of turning it into the
        misleading ``token_limit`` close reason.
        """
        nonlocal context_window_exhausted, prompt_cap_exhausted
        nonlocal prompt_cap_reason, prompt_cap_estimated, prompt_cap_cap
        nonlocal token_budget_exhausted

        reason = assembly.rejected_reason
        if reason == "context_window_limit":
            context_window_exhausted = True
        elif reason in ("prompt_cap", "model_context_limit"):
            prompt_cap_exhausted = True
            prompt_cap_reason = reason
            prompt_cap_estimated = assembly.estimated_input_tokens
            prompt_cap_cap = _preflight_limit_display(reason, context_policy)
        elif reason == "session_budget":
            if uses_window_limit():
                logger.error(
                    "proxy_session: legacy session_budget rejection reached window mode "
                    "(session=%s role=%s)",
                    session_id,
                    context_role,
                )
                raise RuntimeError("session_budget rejection is invalid in window mode")
            token_budget_exhausted = True
        else:
            raise RuntimeError(f"unknown context preflight rejection: {reason!r}")

    def _log_loop_stop(provider_name: str) -> None:
        if context_window_exhausted:
            logger.warning(
                "proxy_session: context-window preflight rejected session %s (%s)",
                session_id,
                provider_name,
            )
        elif prompt_cap_exhausted:
            logger.warning(
                "proxy_session: %s preflight rejected session %s (%s)",
                prompt_cap_reason,
                session_id,
                provider_name,
            )
        elif token_budget_exhausted:
            logger.warning(
                "proxy_session: legacy cumulative token budget exhausted in session %s (%s)",
                session_id,
                provider_name,
            )
        elif repeated_tool_error:
            logger.warning(
                "proxy_session: repeated tool error stopped session %s (%s)",
                session_id,
                provider_name,
            )
        elif stall_signal:
            # Not an exhausted budget — the non-productivity guard cut the loop
            # short. Reporting it as "max_tool_calls reached" made the live logs
            # claim a 1000-call limit was hit after 55 calls.
            logger.warning(
                "proxy_session: non-productivity guard stopped session %s (%s, reason=%s)",
                session_id,
                provider_name,
                stall_signal.get("reason"),
            )
        else:
            logger.warning(
                "proxy_session: max_tool_calls=%d reached in session %s (%s, on_exhaust=%s)",
                max_tool_calls,
                session_id,
                provider_name,
                on_exhaust,
            )

    async def _record_request(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
        response_calls: list[dict[str, Any]],
        *,
        request_kind: str = "tool_loop",
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
            request_kind=request_kind,
        )
        await record_context_measurement(measurement, session_id=session_id, db=db)
        context_step += 1

    # -----------------------------------------------------------------------
    # Anthropic path
    # -----------------------------------------------------------------------
    if uses_messages_wire(provider, model):
        if uses_anthropic_env_key(provider):
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        all_ant_tools = stable_tool_schemas(
            [*(workspace_ant_tools or []), *(mcp_ant_tools or []), ANTHROPIC_CLOSE_TOOL]
        )
        working: list[dict[str, Any]] = list(history)
        _working_ref["messages"] = working

        while True:
            # One loop iteration is one "step" in the stage-0.3 time profile
            # (internal design notes §7): this closes the previous iteration's
            # measurement and opens the next.  The final, partial iteration is
            # deliberately never recorded — see app/core/step_profile.py.
            step_tick(_step_side)
            await _ensure_run_active()
            await _deliver_interjections(
                lambda note: _task_interject.inject_anthropic(working, note)
            )
            assembly = await _assemble_request(
                working,
                all_ant_tools,
                request_kind="tool_loop/anthropic",
            )
            if not assembly.allowed:
                _record_preflight_rejection(assembly)
                break
            request_messages = assembly.messages
            body: dict[str, Any] = {
                "model": model,
                "system": anthropic_system_payload(
                    system_prompt,
                    enabled=bool(llm_config.get("anthropic_prompt_caching")),
                ),
                "messages": request_messages,
                "tools": all_ant_tools,
                "max_tokens": 4096,
            }
            if isinstance(llm_config.get("anthropic_context_management"), dict):
                body["context_management"] = llm_config["anthropic_context_management"]
            apply_reasoning_body(body, llm_config, provider, model)

            async def _tool_loop_request(body=body):
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        messages_url(provider, base_url),
                        headers=messages_headers(provider, api_key),
                        json=body,
                    )
                    response.raise_for_status()
                    return response.json()

            _reasoning_t0 = time.monotonic()
            data = await _llm_request_with_retry(
                _tool_loop_request,
                session_id=session_id,
                label="tool_loop/anthropic",
                db=db,
                provider=provider,
            )
            await _ensure_run_active()
            await _emit_reasoning(
                *_messages_reasoning(data),
                started_at=_reasoning_t0,
            )

            usage = data.get("usage", {})
            await _record_provider_cache(usage)
            request_in = usage.get("input_tokens", 0)
            request_out = usage.get("output_tokens", 0)
            total_in += request_in
            total_out += request_out
            if request_in > 0:
                max_window_in = max(max_window_in, request_in)
                last_window_in = request_in

            content_blocks = data.get("content", [])
            stop_reason = data.get("stop_reason")
            response_calls = tool_calls_from_anthropic(content_blocks)
            await _record_request(
                request_messages, all_ant_tools, request_in, request_out, response_calls
            )

            if stop_reason != "tool_use":
                text = next(
                    (b["text"] for b in content_blocks if b.get("type") == "text"), ""
                )
                text = clean_reply_text(text, where="tool_loop/anthropic", session_id=session_id)
                _publish_usage_stats()
                return text, None, total_in, total_out, model, max_window_in, last_window_in

            # Append full assistant turn (may include text + tool_use blocks)
            working.append({"role": "assistant", "content": content_blocks})
            await _emit_interim_text("\n\n".join(
                b.get("text", "").strip()
                for b in content_blocks
                if b.get("type") == "text" and b.get("text", "").strip()
            ))

            # Execute each tool_use block and collect results
            tool_results: list[dict[str, Any]] = []
            close_input: dict[str, Any] | None = None

            for block in content_blocks:
                if block.get("type") != "tool_use":
                    continue
                await _ensure_run_active()
                tool_id: str = block["id"]
                tool_name: str = block["name"]
                tool_input_data: dict[str, Any] = canonicalize_tool_input(
                    tool_name, block.get("input", {})
                )
                call_subtask_activity = new_subtask_activity(tool_name, tool_input_data) or subtask_activity

                if tool_name not in _FREE_TOOLS and tool_calls_used >= max_tool_calls:
                    tool_limit_exhausted = True
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "Tool call budget exhausted; the call was not executed.",
                        "is_error": True,
                    })
                    continue
                if tool_name not in _FREE_TOOLS:
                    tool_calls_used += 1

                if tool_name == "close_session":
                    close_input = tool_input_data
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "Session closing.",
                    })
                elif mcp_bindings and tool_name in mcp_bindings:
                    await _touch_run_boundary()
                    await emit_status(
                        room_id, AgentStatus.TOOL_CALL_STARTED, session_id=session_id,
                        task_id=task_id, detail=tool_name,
                    )
                    await _emit_tool_call_event(
                        room_id=room_id, session_id=session_id, task_id=task_id,
                        agent_id=actor_agent_id, actor_label=actor_label,
                        tool_name=tool_name, args=tool_input_data, db=db,
                        subtask_activity=call_subtask_activity,
                    )
                    binding = mcp_bindings[tool_name]
                    from app.core.vision_dispatch import (
                        execute_vision_override,
                        should_dispatch_vision_override,
                    )
                    if (
                        mcp_user_id is not None
                        and actor_agent_id is not None
                        and should_dispatch_vision_override(llm_config, binding)
                    ):
                        # BYOK vision is deliberately local to Hub: it uses
                        # the agent-scoped SecretStore binding and never makes
                        # an MCP Gateway/credit reservation call.
                        result_text = await execute_vision_override(
                            llm_config=llm_config,
                            arguments=tool_input_data,
                            room_id=room_id,
                            db=db,
                            owner_id=mcp_user_id,
                            agent_id=actor_agent_id,
                            request_id=session_id,
                            tool_name=binding.tool_name,
                        )
                        # BYOK vision costs the owner's own provider key, not
                        # credits, so `paid=False` — but a repeat is still a
                        # repeat and the guards should see it.
                        result_text = _guard_mcp_call(
                            tool_name, tool_input_data, result_text, paid=False
                        )
                    else:
                        from app.core.mcp_gateway import execute_mcp_tool
                        result_text = await execute_mcp_tool(
                            binding, mcp_user_id, room_id, tool_input_data,
                            db=db,
                            request_id=session_id,
                        )
                        result_text = _observe_mcp_result(tool_name, result_text)
                        result_text = _guard_mcp_call(
                            tool_name, tool_input_data, result_text,
                            paid=binding.market_item_id is not None,
                        )
                    await _touch_run_boundary()
                    await emit_status(
                        room_id, AgentStatus.TOOL_CALL_FINISHED, session_id=session_id,
                        task_id=task_id, detail=tool_name,
                    )
                    tool_results.append({"type": "tool_result", "tool_use_id": tool_id, "content": result_text})
                elif has_ws_tools and _REGISTRY.is_registered(tool_name):
                    await _touch_run_boundary()
                    await emit_status(
                        room_id,
                        AgentStatus.TOOL_CALL_STARTED,
                        session_id=session_id,
                        task_id=task_id,
                        detail=tool_name,
                        extra={"path": tool_input_data.get("path")},
                    )
                    await _emit_tool_call_event(
                        room_id=room_id,
                        session_id=session_id,
                        task_id=task_id,
                        agent_id=actor_agent_id,
                        actor_label=actor_label,
                        tool_name=tool_name,
                        args=tool_input_data,
                        db=db,
                        subtask_activity=call_subtask_activity,
                    )
                    _ctx = _ToolContext(
                        room_id=room_id,
                        session_id=session_id,
                        task_id=task_id,
                        db=db,
                        llm_config=llm_config,
                        secret_value=secret_value,
                        actor_agent_id=actor_agent_id,
                        actor_label=actor_label,
                        workspace_ant_tools=workspace_ant_tools,
                        workspace_oai_tools=workspace_oai_tools,
                        mcp_ant_tools=mcp_ant_tools,
                        mcp_oai_tools=mcp_oai_tools,
                        mcp_bindings=mcp_bindings,
                        mcp_user_id=mcp_user_id,
                        subtask_activity=call_subtask_activity,
                        run_id=run_id,
                    )
                    result_text = await _execute_registry_tool(tool_name, _ctx, tool_input_data)
                    await _touch_run_boundary()
                    _roll_up_subtask_usage(tool_name, result_text)
                    error_repeats = tool_error_guard.observe(
                        tool_name, tool_input_data, result_text
                    )
                    _warning, _fatal = _tool_error_escalation(
                        error_repeats, tool_error_guard.max_repeats
                    )
                    if _fatal:
                        repeated_tool_error = _fatal
                        repeated_tool_error_tool = tool_name
                        repeated_tool_error_repeats = error_repeats
                        repeated_tool_error_safe = safe_tool_error_text(result_text)
                        result_text = f"{result_text}\n[{repeated_tool_error}]"
                    elif _warning:
                        result_text = f"{result_text}\n[{_warning}]"
                    if stall_guard is not None:
                        _read_target: Any = READ_TARGET_UNSET
                        if tool_name == "observation_read":
                            _read_target = await _resolve_observation_path(
                                db, room_id, tool_input_data.get("ref")
                            )
                        _sv = stall_guard.observe(
                            tool_name,
                            tool_input_data,
                            result_text=result_text if isinstance(result_text, str) else None,
                            read_target=_read_target,
                        )
                        if _sv.warn:
                            result_text = f"{result_text}\n[{_sv.warn}]"
                        elif _sv.stall:
                            stall_signal = {
                                "reason": _sv.reason,
                                "tool_name": _sv.tool_name,
                                "repeat_count": _sv.repeat_count,
                                "target": _sv.target,
                            }
                    await emit_status(
                        room_id,
                        AgentStatus.TOOL_CALL_FINISHED,
                        session_id=session_id,
                        task_id=task_id,
                        detail=tool_name,
                        extra={"path": tool_input_data.get("path")},
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_text,
                    })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": f"Error: tool {tool_name!r} is not available in this context",
                        "is_error": True,
                    })

            if close_input is not None:
                _publish_usage_stats()
                return None, close_input, total_in, total_out, model, max_window_in, last_window_in

            working.append({"role": "user", "content": tool_results})
            if tool_limit_exhausted or repeated_tool_error or stall_signal:
                break

        _log_loop_stop("anthropic")
        if stall_signal and on_exhaust == "raise":
            return _stall_return()
        if context_window_exhausted and on_exhaust == "raise":
            raise ContextWindowLimitReached(
                working,
                total_in,
                total_out,
                model,
                tool_calls_used=tool_calls_used,
                tool_calls_limit=max_tool_calls,
                tokens_used=budget_used + total_in + total_out,
                context_window_limit=context_policy.context_window_limit,
                max_window_in=max_window_in,
                last_window_in=last_window_in,
            )
        if prompt_cap_exhausted and on_exhaust == "raise":
            _cap_env_var = (
                "CONTEXT_CHAT_MASTER_INPUT_CAP"
                if context_role == "chat_master"
                else "CONTEXT_AGENT_INPUT_CAP"
            )
            raise PromptCapLimitReached(
                working,
                total_in,
                total_out,
                model,
                tool_calls_used=tool_calls_used,
                tool_calls_limit=max_tool_calls,
                tokens_used=budget_used + total_in + total_out,
                max_window_in=max_window_in,
                last_window_in=last_window_in,
                rejected_reason=prompt_cap_reason or "prompt_cap",
                estimated_input_tokens=prompt_cap_estimated,
                cap=prompt_cap_cap,
                cap_env_var=_cap_env_var,
            )
        if token_budget_exhausted and on_exhaust == "raise":
            raise TokenBudgetLimitReached(
                working,
                total_in,
                total_out,
                model,
                tool_calls_used=tool_calls_used,
                tool_calls_limit=max_tool_calls,
                tokens_used=budget_used + total_in + total_out,
                max_window_in=max_window_in,
                last_window_in=last_window_in,
            )
        if context_window_exhausted or prompt_cap_exhausted or token_budget_exhausted:
            # A real preflight overflow must not result in a follow-up provider
            # request merely to force-close the session.
            _publish_usage_stats()
            return None, None, total_in, total_out, model, max_window_in, last_window_in
        if on_exhaust == "raise":
            # This anthropic-specific tail must distinguish repeated_tool_error
            # from a real tool_calls exhaustion the same way the shared/openai
            # tail below does — otherwise the default provider never raises
            # RepeatedToolErrorLimitReached at all and every repeated-error
            # stop is misreported as "reached the tool-call limit"
            # (internal design notes, part 4, stage 1 — this was found by
            # the regression test for this stage, not by the original
            # incident report, which only observed the frontend symptom).
            if repeated_tool_error:
                raise RepeatedToolErrorLimitReached(
                    working,
                    total_in,
                    total_out,
                    model,
                    tool_calls_used=tool_calls_used,
                    tool_calls_limit=max_tool_calls,
                    tokens_used=budget_used + total_in + total_out,
                    max_window_in=max_window_in,
                    last_window_in=last_window_in,
                    trigger_tool=repeated_tool_error_tool,
                    repeat_count=repeated_tool_error_repeats,
                    safe_error=repeated_tool_error_safe,
                )
            raise ToolLimitReached(
                working,
                total_in,
                total_out,
                model,
                tool_calls_used=tool_calls_used,
                tool_calls_limit=max_tool_calls,
                tokens_used=budget_used + total_in + total_out,
                max_window_in=max_window_in,
                last_window_in=last_window_in,
            )
        if on_exhaust == "answer":
            # Exhausted turn that owes a reply, not a verdict: one final request
            # with no tools at all, so the only thing the model can produce is
            # text. (A forced close_session here would close a peer session that
            # is still mid-task.)
            working.append({"role": "user", "content": _NO_TOOLS_LEFT_ANSWER_PROMPT})
            try:
                answer_assembly = await _assemble_request(
                    working, [], request_kind="forced_answer/anthropic",
                )
                if not answer_assembly.allowed:
                    _publish_usage_stats()
                    return None, None, total_in, total_out, model, max_window_in, last_window_in
                answer_messages = answer_assembly.messages
                body_answer: dict[str, Any] = {
                    "model": model,
                    "system": anthropic_system_payload(
                        system_prompt,
                        enabled=bool(llm_config.get("anthropic_prompt_caching")),
                    ),
                    "messages": answer_messages,
                    "max_tokens": 4096,
                }
                if isinstance(llm_config.get("anthropic_context_management"), dict):
                    body_answer["context_management"] = llm_config["anthropic_context_management"]
                apply_reasoning_body(body_answer, llm_config, provider, model)

                async def _forced_answer_request():
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(
                            messages_url(provider, base_url),
                            headers=messages_headers(provider, api_key),
                            json=body_answer,
                        )
                        response.raise_for_status()
                        return response.json()

                data = await _llm_request_with_retry(
                    _forced_answer_request,
                    session_id=session_id,
                    label="forced_answer/anthropic",
                    db=db,
                    provider=provider,
                )
                usage = data.get("usage", {})
                await _record_provider_cache(usage)
                request_in = usage.get("input_tokens", 0)
                request_out = usage.get("output_tokens", 0)
                total_in += request_in
                total_out += request_out
                if request_in > 0:
                    max_window_in = max(max_window_in, request_in)
                    last_window_in = request_in
                await _record_request(
                    answer_messages, [], request_in, request_out, [],
                    request_kind="forced_answer",
                )
                text = next(
                    (b["text"] for b in data.get("content", []) if b.get("type") == "text"), ""
                )
                text = clean_reply_text(
                    text, where="forced_answer/anthropic", session_id=session_id
                )
                if text.strip():
                    _publish_usage_stats()
                    return text, None, total_in, total_out, model, max_window_in, last_window_in
            except Exception as exc:
                logger.error(
                    "proxy_session: forced answer failed for session %s: %s", session_id, exc
                )
            _publish_usage_stats()
            return None, None, total_in, total_out, model, max_window_in, last_window_in

        # Graceful exhaustion: one final request with only the close tool,
        # forced, so the model persists a summary of the work done so far.
        working.append({
            "role": "user",
            "content": (
                f"The session limit has been reached (tool calls or token budget). "
                "Call close_session immediately: describe in summary what has already been done, "
                "which files were created/changed and what is left to finish."
            ),
        })
        try:
            close_assembly = await _assemble_request(
                working,
                [ANTHROPIC_CLOSE_TOOL],
                request_kind="forced_close/anthropic",
            )
            if not close_assembly.allowed:
                _publish_usage_stats()
                return None, None, total_in, total_out, model, max_window_in, last_window_in
            close_messages = close_assembly.messages
            body_close: dict[str, Any] = {
                "model": model,
                "system": anthropic_system_payload(
                    system_prompt,
                    enabled=bool(llm_config.get("anthropic_prompt_caching")),
                ),
                "messages": close_messages,
                "tools": [ANTHROPIC_CLOSE_TOOL],
                "tool_choice": {"type": "tool", "name": "close_session"},
                "max_tokens": 4096,
            }
            if isinstance(llm_config.get("anthropic_context_management"), dict):
                body_close["context_management"] = llm_config["anthropic_context_management"]
            apply_reasoning_body(body_close, llm_config, provider, model)

            async def _forced_close_request():
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        messages_url(provider, base_url),
                        headers=messages_headers(provider, api_key),
                        json=body_close,
                    )
                    response.raise_for_status()
                    return response.json()

            data = await _llm_request_with_retry(
                _forced_close_request,
                session_id=session_id,
                label="forced_close/anthropic",
                db=db,
                provider=provider,
            )
            usage = data.get("usage", {})
            await _record_provider_cache(usage)
            request_in = usage.get("input_tokens", 0)
            request_out = usage.get("output_tokens", 0)
            total_in += request_in
            total_out += request_out
            if request_in > 0:
                max_window_in = max(max_window_in, request_in)
                last_window_in = request_in
            await _record_request(
                close_messages,
                [ANTHROPIC_CLOSE_TOOL],
                request_in,
                request_out,
                tool_calls_from_anthropic(data.get("content", [])),
                request_kind="forced_close",
            )
            for block in data.get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "close_session":
                    _publish_usage_stats()
                    return None, block.get("input", {}), total_in, total_out, model, max_window_in, last_window_in
        except Exception as exc:
            logger.error(
                "proxy_session: forced close_session failed for session %s: %s", session_id, exc
            )
        _publish_usage_stats()
        return None, None, total_in, total_out, model, max_window_in, last_window_in

    # -----------------------------------------------------------------------
    # OpenAI-compatible path
    # -----------------------------------------------------------------------
    api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-no-key-required"
    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 120.0}
    client_kwargs["base_url"] = openai_base_url(provider, base_url)
    client_kwargs.update(openai_client_extras(client_kwargs["base_url"]))

    all_oai_tools = stable_tool_schemas(
        [*(workspace_oai_tools or []), *(mcp_oai_tools or []), OPENAI_CLOSE_TOOL]
    )
    create_extra: dict[str, Any] = {}
    if provider == "openrouter" and llm_config.get("session_logging") and or_session_id:
        create_extra["extra_body"] = {"session_id": or_session_id}
    create_extra.update(
        openai_cache_request_extra(
            enabled=cache_tools_enabled,
            cache_key=str(llm_config.get("prompt_cache_key") or _cache_key(all_oai_tools)),
        )
    )
    # One dict feeds three requests below — tool_loop, forced_answer and
    # forced_close — so the helper is called at the assembly, not at each
    # create(): eight lines instead of fourteen, and no later request forgotten.
    apply_reasoning_chat(create_extra, llm_config, provider, model)

    oai = llm_client(provider, chat_client_factory=AsyncOpenAI, **client_kwargs)
    working_oai: list[dict[str, Any]] = list(history)
    _working_ref["messages"] = working_oai

    while True:
        # See the anthropic branch above: one iteration is one profiled step.
        step_tick(_step_side)
        await _ensure_run_active()
        await _deliver_interjections(
            lambda note: _task_interject.inject_openai(working_oai, note)
        )
        assembly_oai = await _assemble_request(
            working_oai,
            all_oai_tools,
            request_kind="tool_loop/openai",
        )
        if not assembly_oai.allowed:
            _record_preflight_rejection(assembly_oai)
            break
        request_messages_oai = [
            {"role": "system", "content": system_prompt},
            *assembly_oai.messages,
        ]
        _reasoning_t0 = time.monotonic()
        resp = await _llm_request_with_retry(
            lambda: oai.chat.completions.create(
                model=model,
                messages=request_messages_oai,
                tools=all_oai_tools,
                tool_choice="auto",
                **create_extra,
            ),
            session_id=session_id,
            label="tool_loop/openai",
            db=db,
            provider=provider,
            validate_response=_require_openai_choices,
        )

        await _ensure_run_active()
        choice = resp.choices[0]
        usage = resp.usage
        await _emit_reasoning(
            *reasoning_from_chat_choice(choice.message, usage),
            started_at=_reasoning_t0,
        )
        await _record_provider_cache(usage)
        request_in = usage.prompt_tokens if usage else 0
        request_out = usage.completion_tokens if usage else 0
        total_in += request_in
        total_out += request_out
        if request_in > 0:
            max_window_in = max(max_window_in, request_in)
            last_window_in = request_in
        await _record_request(
            assembly_oai.messages,
            all_oai_tools,
            request_in,
            request_out,
            tool_calls_from_openai(choice.message.tool_calls or []),
        )

        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            text = clean_reply_text(
                choice.message.content, where="tool_loop/openai", session_id=session_id
            )
            _publish_usage_stats()
            return text, None, total_in, total_out, model, max_window_in, last_window_in

        # Append assistant turn with tool_calls
        working_oai.append({
            "role": "assistant",
            "content": choice.message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ],
        })
        await _emit_interim_text(choice.message.content)

        close_input_oai: dict[str, Any] | None = None

        for tc in choice.message.tool_calls:
            await _ensure_run_active()
            try:
                tool_args = parse_tool_arguments(tc.function.arguments)
            except ToolArgumentsParseError as exc:
                logger.warning(
                    "proxy_session: malformed tool JSON for %s in session %s: %s",
                    tc.function.name,
                    session_id,
                    exc,
                )
                tool_args = {}
            tool_args = canonicalize_tool_input(tc.function.name, tool_args)
            call_subtask_activity = new_subtask_activity(tc.function.name, tool_args) or subtask_activity

            if tc.function.name not in _FREE_TOOLS and tool_calls_used >= max_tool_calls:
                tool_limit_exhausted = True
                working_oai.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Tool call budget exhausted; the call was not executed.",
                })
                continue
            if tc.function.name not in _FREE_TOOLS:
                tool_calls_used += 1

            if tc.function.name == "close_session":
                close_input_oai = tool_args
                working_oai.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Session closing.",
                })
            elif mcp_bindings and tc.function.name in mcp_bindings:
                await _touch_run_boundary()
                await emit_status(
                    room_id, AgentStatus.TOOL_CALL_STARTED, session_id=session_id,
                    task_id=task_id, detail=tc.function.name,
                )
                await _emit_tool_call_event(
                    room_id=room_id, session_id=session_id, task_id=task_id,
                    agent_id=actor_agent_id, actor_label=actor_label,
                    tool_name=tc.function.name, args=tool_args, db=db,
                    subtask_activity=call_subtask_activity,
                )
                binding = mcp_bindings[tc.function.name]
                from app.core.vision_dispatch import (
                    execute_vision_override,
                    should_dispatch_vision_override,
                )
                if (
                    mcp_user_id is not None
                    and actor_agent_id is not None
                    and should_dispatch_vision_override(llm_config, binding)
                ):
                    result_text = await execute_vision_override(
                        llm_config=llm_config,
                        arguments=tool_args,
                        room_id=room_id,
                        db=db,
                        owner_id=mcp_user_id,
                        agent_id=actor_agent_id,
                        request_id=session_id,
                        tool_name=binding.tool_name,
                    )
                    result_text = _guard_mcp_call(
                        tc.function.name, tool_args, result_text, paid=False
                    )
                else:
                    from app.core.mcp_gateway import execute_mcp_tool
                    result_text = await execute_mcp_tool(
                        binding, mcp_user_id, room_id, tool_args,
                        db=db,
                        request_id=session_id,
                    )
                    result_text = _observe_mcp_result(tc.function.name, result_text)
                    result_text = _guard_mcp_call(
                        tc.function.name, tool_args, result_text,
                        paid=binding.market_item_id is not None,
                    )
                await _touch_run_boundary()
                await emit_status(
                    room_id, AgentStatus.TOOL_CALL_FINISHED, session_id=session_id,
                    task_id=task_id, detail=tc.function.name,
                )
                working_oai.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
            elif has_ws_tools and _REGISTRY.is_registered(tc.function.name):
                await _touch_run_boundary()
                await emit_status(
                    room_id,
                    AgentStatus.TOOL_CALL_STARTED,
                    session_id=session_id,
                    task_id=task_id,
                    detail=tc.function.name,
                    extra={"path": tool_args.get("path")},
                )
                await _emit_tool_call_event(
                    room_id=room_id,
                    session_id=session_id,
                    task_id=task_id,
                    agent_id=actor_agent_id,
                    actor_label=actor_label,
                    tool_name=tc.function.name,
                    args=tool_args,
                    db=db,
                    subtask_activity=call_subtask_activity,
                )
                _ctx = _ToolContext(
                    room_id=room_id,
                    session_id=session_id,
                    task_id=task_id,
                    db=db,
                    llm_config=llm_config,
                    secret_value=secret_value,
                    actor_agent_id=actor_agent_id,
                    actor_label=actor_label,
                    workspace_ant_tools=workspace_ant_tools,
                    workspace_oai_tools=workspace_oai_tools,
                    mcp_ant_tools=mcp_ant_tools,
                    mcp_oai_tools=mcp_oai_tools,
                    mcp_bindings=mcp_bindings,
                    mcp_user_id=mcp_user_id,
                    subtask_activity=call_subtask_activity,
                    run_id=run_id,
                )
                result_text = await _execute_registry_tool(tc.function.name, _ctx, tool_args)
                await _touch_run_boundary()
                _roll_up_subtask_usage(tc.function.name, result_text)
                error_repeats = tool_error_guard.observe(
                    tc.function.name, tool_args, result_text
                )
                _warning, _fatal = _tool_error_escalation(
                    error_repeats, tool_error_guard.max_repeats
                )
                if _fatal:
                    repeated_tool_error = _fatal
                    repeated_tool_error_tool = tc.function.name
                    repeated_tool_error_repeats = error_repeats
                    repeated_tool_error_safe = safe_tool_error_text(result_text)
                    result_text = f"{result_text}\n[{repeated_tool_error}]"
                elif _warning:
                    result_text = f"{result_text}\n[{_warning}]"
                if stall_guard is not None:
                    _read_target: Any = READ_TARGET_UNSET
                    if tc.function.name == "observation_read":
                        _read_target = await _resolve_observation_path(
                            db, room_id, tool_args.get("ref")
                        )
                    _sv = stall_guard.observe(
                        tc.function.name,
                        tool_args,
                        result_text=result_text if isinstance(result_text, str) else None,
                        read_target=_read_target,
                    )
                    if _sv.warn:
                        result_text = f"{result_text}\n[{_sv.warn}]"
                    elif _sv.stall:
                        stall_signal = {
                            "reason": _sv.reason,
                            "tool_name": _sv.tool_name,
                            "repeat_count": _sv.repeat_count,
                            "target": _sv.target,
                        }
                await emit_status(
                    room_id,
                    AgentStatus.TOOL_CALL_FINISHED,
                    session_id=session_id,
                    task_id=task_id,
                    detail=tc.function.name,
                    extra={"path": tool_args.get("path")},
                )
                working_oai.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_text,
                })
            else:
                working_oai.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: tool {tc.function.name!r} is not available",
                })

        if close_input_oai is not None:
            _publish_usage_stats()
            return None, close_input_oai, total_in, total_out, model, max_window_in, last_window_in
        if tool_limit_exhausted or repeated_tool_error or stall_signal:
            break

    _log_loop_stop("openai")
    if stall_signal and on_exhaust == "raise":
        return _stall_return()
    if repeated_tool_error and on_exhaust == "raise":
        raise RepeatedToolErrorLimitReached(
            working if uses_messages_wire(provider, model) else working_oai,
            total_in,
            total_out,
            model,
            tool_calls_used=tool_calls_used,
            tool_calls_limit=max_tool_calls,
            tokens_limit=TOKEN_LIMIT,
            max_window_in=max_window_in,
            last_window_in=last_window_in,
            trigger_tool=repeated_tool_error_tool,
            repeat_count=repeated_tool_error_repeats,
            safe_error=repeated_tool_error_safe,
        )
    if context_window_exhausted and on_exhaust == "raise":
        raise ContextWindowLimitReached(
            working_oai,
            total_in,
            total_out,
            model,
            tool_calls_used=tool_calls_used,
            tool_calls_limit=max_tool_calls,
            tokens_used=budget_used + total_in + total_out,
            context_window_limit=context_policy.context_window_limit,
            max_window_in=max_window_in,
            last_window_in=last_window_in,
        )
    if prompt_cap_exhausted and on_exhaust == "raise":
        _cap_env_var = (
            "CONTEXT_CHAT_MASTER_INPUT_CAP"
            if context_role == "chat_master"
            else "CONTEXT_AGENT_INPUT_CAP"
        )
        raise PromptCapLimitReached(
            working_oai,
            total_in,
            total_out,
            model,
            tool_calls_used=tool_calls_used,
            tool_calls_limit=max_tool_calls,
            tokens_used=budget_used + total_in + total_out,
            max_window_in=max_window_in,
            last_window_in=last_window_in,
            rejected_reason=prompt_cap_reason or "prompt_cap",
            estimated_input_tokens=prompt_cap_estimated,
            cap=prompt_cap_cap,
            cap_env_var=_cap_env_var,
        )
    if token_budget_exhausted and on_exhaust == "raise":
        raise TokenBudgetLimitReached(
            working_oai,
            total_in,
            total_out,
            model,
            tool_calls_used=tool_calls_used,
            tool_calls_limit=max_tool_calls,
            tokens_used=budget_used + total_in + total_out,
            max_window_in=max_window_in,
            last_window_in=last_window_in,
        )
    if context_window_exhausted or prompt_cap_exhausted or token_budget_exhausted:
        # A real preflight overflow must not result in a follow-up provider
        # request merely to force-close the session.
        _publish_usage_stats()
        return None, None, total_in, total_out, model, max_window_in, last_window_in
    if on_exhaust == "raise":
        raise ToolLimitReached(
            working_oai,
            total_in,
            total_out,
            model,
            tool_calls_used=tool_calls_used,
            tool_calls_limit=max_tool_calls,
            tokens_used=budget_used + total_in + total_out,
            max_window_in=max_window_in,
            last_window_in=last_window_in,
        )
    if on_exhaust == "answer":
        # See the anthropic counterpart: a turn that owes a reply must end in
        # text, never in a forced close_session.
        working_oai.append({"role": "user", "content": _NO_TOOLS_LEFT_ANSWER_PROMPT})
        try:
            answer_assembly_oai = await _assemble_request(
                working_oai, [], request_kind="forced_answer/openai",
            )
            if not answer_assembly_oai.allowed:
                _publish_usage_stats()
                return None, None, total_in, total_out, model, max_window_in, last_window_in
            answer_messages_oai = [
                {"role": "system", "content": system_prompt},
                *answer_assembly_oai.messages,
            ]
            resp = await _llm_request_with_retry(
                lambda: oai.chat.completions.create(
                    model=model,
                    messages=answer_messages_oai,
                    **create_extra,
                ),
                session_id=session_id,
                label="forced_answer/openai",
                db=db,
                provider=provider,
                validate_response=_require_openai_choices,
            )
            usage = resp.usage
            await _record_provider_cache(usage)
            request_in = usage.prompt_tokens if usage else 0
            request_out = usage.completion_tokens if usage else 0
            total_in += request_in
            total_out += request_out
            if request_in > 0:
                max_window_in = max(max_window_in, request_in)
                last_window_in = request_in
            await _record_request(
                answer_assembly_oai.messages, [], request_in, request_out, [],
                request_kind="forced_answer",
            )
            text = clean_reply_text(
                resp.choices[0].message.content or "",
                where="forced_answer/openai",
                session_id=session_id,
            )
            if text.strip():
                _publish_usage_stats()
                return text, None, total_in, total_out, model, max_window_in, last_window_in
        except Exception as exc:
            logger.error(
                "proxy_session: forced answer failed for session %s: %s", session_id, exc
            )
        _publish_usage_stats()
        return None, None, total_in, total_out, model, max_window_in, last_window_in

    # Graceful exhaustion: one final request with only the close tool, forced,
    # so the model persists a summary of the work done so far.
    working_oai.append({
        "role": "user",
        "content": (
            f"The session limit has been reached (tool calls or token budget). "
            "Call close_session immediately: describe in summary what has already been done, "
            "which files were created/changed and what is left to finish."
        ),
    })
    try:
        close_assembly_oai = await _assemble_request(
            working_oai,
            [OPENAI_CLOSE_TOOL],
            request_kind="forced_close/openai",
        )
        if not close_assembly_oai.allowed:
            _publish_usage_stats()
            return None, None, total_in, total_out, model, max_window_in, last_window_in
        close_messages_oai = [
            {"role": "system", "content": system_prompt},
            *close_assembly_oai.messages,
        ]
        resp = await _llm_request_with_retry(
            lambda: oai.chat.completions.create(
                model=model,
                messages=close_messages_oai,
                tools=[OPENAI_CLOSE_TOOL],
                tool_choice={"type": "function", "function": {"name": "close_session"}},
                **create_extra,
            ),
            session_id=session_id,
            label="forced_close/openai",
            db=db,
            provider=provider,
            validate_response=_require_openai_choices,
        )
        usage = resp.usage
        await _record_provider_cache(usage)
        request_in = usage.prompt_tokens if usage else 0
        request_out = usage.completion_tokens if usage else 0
        total_in += request_in
        total_out += request_out
        if request_in > 0:
            max_window_in = max(max_window_in, request_in)
            last_window_in = request_in
        await _record_request(
            close_assembly_oai.messages,
            [OPENAI_CLOSE_TOOL],
            request_in,
            request_out,
            tool_calls_from_openai(resp.choices[0].message.tool_calls or []),
            request_kind="forced_close",
        )
        for tc in resp.choices[0].message.tool_calls or []:
            if tc.function.name == "close_session":
                try:
                    _publish_usage_stats()
                    return None, parse_tool_arguments(tc.function.arguments), total_in, total_out, model, max_window_in, last_window_in
                except ToolArgumentsParseError as exc:
                    logger.error(
                        "proxy_session: forced close_session returned malformed JSON "
                        "for session %s: %s", session_id, exc,
                    )
    except Exception as exc:
        logger.error(
            "proxy_session: forced close_session failed for session %s (oai): %s", session_id, exc
        )
    _publish_usage_stats()
    return None, None, total_in, total_out, model, max_window_in, last_window_in


# ---------------------------------------------------------------------------
# Phase A — image prompt authoring
# ---------------------------------------------------------------------------


def _validate_image_prompts(
    images: list[dict[str, Any]],
    known_reference_paths: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate the prompts list, returning a possibly-filtered copy.

    Raises ImagePromptValidationError for whole-batch problems (empty list,
    duplicate filenames, a too-short prompt) — those are authoring mistakes
    that make the whole submission unusable.

    A per-image problem is softer: an entry naming a ``reference_paths``
    value outside ``known_reference_paths`` (an invented path — stage 5.4)
    or an unparseable ``size`` is dropped from the result instead of failing
    every other image in the batch, since the rest of the series is still
    usable. ``known_reference_paths=None`` skips the reference_paths check
    entirely (no manifest was built — e.g. the task declared no references).
    """
    if not images:
        raise ImagePromptValidationError(
            "submit_image_prompts returned an empty images list"
        )
    filenames = [img.get("filename", "") for img in images]
    if len(filenames) != len(set(filenames)):
        raise ImagePromptValidationError("Duplicate filenames in image prompts")
    for img in images:
        if len((img.get("prompt") or "").strip()) < 10:
            raise ImagePromptValidationError(
                f"Prompt too short for '{img.get('filename')}': {img.get('prompt')!r}"
            )

    valid: list[dict[str, Any]] = []
    for img in images:
        filename = img.get("filename")

        raw_size = img.get("size")
        if raw_size:
            try:
                parse_media_size(raw_size)
            except ValueError as exc:
                logger.warning(
                    "proxy_session: dropping image '%s' — invalid size %r: %s",
                    filename, raw_size, exc,
                )
                continue

        ref_paths = img.get("reference_paths") or []
        if known_reference_paths is not None:
            unknown = [p for p in ref_paths if p not in known_reference_paths]
            if unknown:
                logger.warning(
                    "proxy_session: dropping image '%s' — unknown reference_paths %s",
                    filename, unknown,
                )
                continue
        if len(ref_paths) > MEDIA_REF_MAX_PER_IMAGE:
            logger.warning(
                "proxy_session: truncating reference_paths for image '%s' to %d entries",
                filename, MEDIA_REF_MAX_PER_IMAGE,
            )
            img = {**img, "reference_paths": ref_paths[:MEDIA_REF_MAX_PER_IMAGE]}

        valid.append(img)

    if not valid:
        raise ImagePromptValidationError(
            "no valid images remained after size/reference validation"
        )
    return valid


async def _generate_image_prompts(
    agent_session: Session,
    agent: Agent,
    task: Task,
    db: AsyncSession,
) -> list[dict[str, Any]]:
    """Phase A: use the agent's TEXT model (+ skills) to author image prompts.

    Returns a validated list of {filename, prompt, aspect_ratio?} dicts.
    Raises ImagePromptValidationError on unusable output.
    """
    agent_llm_config = agent.llm_config or {}
    memory = await build_memory_layer(agent_session, db)
    sys_prompt = (
        build_system_prompt(memory, "cm_agent_proxy_agent") + MEDIA_PROMPTGEN_SUFFIX
    )

    # Stage 5.1-5.3: resolve declared references (if any) into a text manifest
    # + an optional multimodal vision turn, so Phase A can see what it must
    # not redraw from scratch instead of only reading a path string.
    media_spec: dict[str, Any] = task.media_spec or {}
    ref_specs: list[dict[str, Any]] = media_spec.get("references") or []
    reference_entries: list[ReferenceManifestEntry] = []
    if ref_specs:
        try:
            reference_entries = await resolve_task_references(task.room_id, ref_specs, db)
        except Exception:
            logger.exception(
                "proxy_session: failed to resolve media references for task %s", task.id
            )
            reference_entries = []

    task_data_text = render_task_data(task) + build_reference_manifest(reference_entries)
    known_reference_paths = frozenset(entry.path for entry in reference_entries) or None

    media_cfg: dict[str, Any] = agent_llm_config.get("media") or {}
    reference_vision = str(media_cfg.get("reference_vision") or "auto").strip().lower()
    provider = str(agent_llm_config.get("provider", "anthropic")).lower()

    resolved_refs = [entry.ref for entry in reference_entries if entry.ref is not None]
    content: Any = task_data_text
    if resolved_refs and reference_vision != "off":
        # The primary text model — not the media/image-generation model — is
        # what actually receives these blocks here. Check it can accept image
        # content before attaching any: an OpenRouter chat model that can't
        # (e.g. a text-only model like deepseek-v4-flash) answers with a 404
        # "No endpoints found that support image input", failing Phase A
        # outright. `reference_vision: auto` means "use images when the model
        # can take them" — the text manifest already baked into
        # `task_data_text` is the graceful degrade when it can't.
        vision_capable = True
        if provider == "openrouter":
            primary_model = str(agent_llm_config.get("model") or "").strip()
            primary_base_url = str(agent_llm_config.get("base_url") or "").strip() or "https://openrouter.ai/api/v1"
            if primary_model:
                supports = await chat_model_supports_image_input(primary_base_url, primary_model)
                vision_capable = supports is not False
        if vision_capable:
            content = [{"type": "text", "text": task_data_text}] + build_vision_blocks(
                provider, resolved_refs, str(agent_llm_config.get("model") or "")
            )

    history: list[dict[str, Any]] = [{"role": "user", "content": content}]

    secret_value = await resolve_agent_secret(
        db=db,
        owner_id=agent.owner_id,
        agent_id=agent.id,
        config=agent_llm_config,
        request_id=agent_session.id,
    )
    try:
        _, tool_input, t_in, t_out, model = await _call_llm(
            agent_llm_config,
            sys_prompt,
            history,
            or_session_id=str(agent_session.id),
            anthropic_tool=ANTHROPIC_IMAGE_PROMPTS_TOOL,
            openai_tool=OPENAI_IMAGE_PROMPTS_TOOL,
            tool_required=True,
            secret_value=secret_value,
            db=db,
        )
    finally:
        secret_value = None

    track_llm_usage(
        agent_session, t_in, t_out, _compute_cost(model, t_in, t_out, agent_llm_config), model=model
    )
    update_context_window_metrics(agent_session, t_in, t_in)
    db.add(agent_session)
    await db.flush()

    images: list[dict[str, Any]] = (tool_input or {}).get("images", [])
    return _validate_image_prompts(images, known_reference_paths)


# ---------------------------------------------------------------------------
# Close helpers
# ---------------------------------------------------------------------------


async def _close_side(
    session: Session,
    tool_input: dict[str, Any] | None,
    close_reason: str,
    db: AsyncSession,
) -> SessionCloseActions:
    """Parse and apply close actions for one proxy session side."""
    actions = SessionCloseActions()
    if tool_input is not None:
        try:
            actions = extract_close_actions_from_tool_use(tool_input)
        except Exception as exc:
            logger.error(
                "proxy_session: invalid close_session input for session %s: %s\ntool_input: %s",
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

    await apply_close_actions(session, actions, db)
    session.closed_at = datetime.now(timezone.utc)
    session.close_reason = close_reason
    db.add(session)
    await db.flush()
    # Single choke point for every proxy-side close (§4.1/§4.4): a no-op when
    # `actions` already carries an agent_memory_patch, otherwise reuses
    # session.summary (e.g. the ToolLimitReached auto-summary) or falls back
    # to one cheap call / a deterministic note — never leaves memory untouched.
    await flush_session_memory(session, db, reason=close_reason, applied_actions=actions)
    return actions


async def _emit_closed(
    session: Session,
    close_reason: str,
    actions: SessionCloseActions,
    db: AsyncSession,
) -> None:
    side: str = getattr(session, "side", "unknown")
    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_CLOSED",
        content=(
            f"Session {session.id} closed "
            f"(kind=cm_agent_proxy, side={side}, reason={close_reason})"
        ),
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": "cm_agent_proxy",
            "side": side,
            "close_reason": close_reason,
            "summary": actions.summary,
        },
    )


_REVIEW_PREVIEW_CHARS = 1200
_REVIEW_FILE_WRITES_MAX = 10
_REVIEW_CHANGED_FILES_MAX = 40
_REVIEW_GIT_STAT_CHARS = 6000
_REVIEW_TOTAL_MAX_CHARS = 12000
_REVIEW_FILE_WRITES_SECTION_MAX_CHARS = 5500
_REVIEW_TRUNCATION_MARKER = "\n… [truncated]"

_REVIEW_IGNORED_DIRS = frozenset({
    "node_modules", "dist", "build", ".next", ".nuxt", ".svelte-kit",
    ".parcel-cache", ".turbo", ".output", ".cache", ".git", ".venv", "venv",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", ".gradle",
    "target", ".idea", ".vscode",
})
_REVIEW_IGNORED_EXTS = frozenset({".log", ".tmp", ".pyc"})
_REVIEW_IGNORED_NAMES = frozenset({
    ".DS_Store",
    "Thumbs.db",
    "package-lock.json",
})


def _truncate_review(value: str, limit: int) -> str:
    """Truncate *value* without letting the marker exceed the hard limit."""
    if len(value) <= limit:
        return value
    if limit <= len(_REVIEW_TRUNCATION_MARKER):
        return _REVIEW_TRUNCATION_MARKER[:limit]
    return value[:limit - len(_REVIEW_TRUNCATION_MARKER)] + _REVIEW_TRUNCATION_MARKER


def _is_review_path_ignored(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    for part in parts[:-1]:
        if part in _REVIEW_IGNORED_DIRS:
            return True
    name = parts[-1] if parts else ""
    if name in _REVIEW_IGNORED_NAMES:
        return True
    _, ext = os.path.splitext(name)
    if ext in _REVIEW_IGNORED_EXTS:
        return True
    if name.endswith(".lock") and name != "poetry.lock":
        return True
    return False


def _render_review_file_writes(file_writes: list[Any]) -> str:
    """Render persisted writes while reserving room for omitted-file lines.

    Ten previews at the legacy 1,200-character maximum already exceed the
    review block's 12,000-character hard cap once headings and file metadata
    are included. Reduce the per-file preview only when necessary so all ten
    preview entries and the compact lines for remaining writes stay visible.
    """
    shown = file_writes[:_REVIEW_FILE_WRITES_MAX]
    omitted = file_writes[_REVIEW_FILE_WRITES_MAX:]

    def _render(preview_chars: int) -> str:
        section_lines = [
            f"The system has ALREADY saved {len(file_writes)} file(s) "
            "from the agent's file_writes to the repository:"
        ]
        for fw in shown:
            content = fw.content or ""
            preview = content[:preview_chars]
            suffix = (
                f"\n… [{len(content) - preview_chars} more chars]"
                if len(content) > preview_chars else ""
            )
            section_lines.append(
                f"- `{fw.path}` ({len(content)} chars). Preview:\n"
                f"```\n{preview}{suffix}\n```"
            )
        for fw in omitted:
            content = fw.content or ""
            section_lines.append(
                f"- `{fw.path}` ({len(content)} chars, preview omitted)"
            )
        if omitted:
            section_lines.append("Full file listing available via `git_diff` tool.")
        return "\n".join(section_lines)

    preview_chars = _REVIEW_PREVIEW_CHARS
    rendered = _render(preview_chars)
    while len(rendered) > _REVIEW_FILE_WRITES_SECTION_MAX_CHARS and preview_chars > 0:
        excess = len(rendered) - _REVIEW_FILE_WRITES_SECTION_MAX_CHARS
        preview_chars = max(0, preview_chars - max(1, (excess + len(shown) - 1) // len(shown)))
        rendered = _render(preview_chars)
    return _truncate_review(rendered, _REVIEW_FILE_WRITES_SECTION_MAX_CHARS)


async def _build_review_verification(
    actions: SessionCloseActions,
    start_snapshot: list[dict[str, Any]],
    room_id: UUID,
    db: AsyncSession,
    task_id: Optional[UUID] = None,
) -> str:
    """System-generated verification block for the CM review notification.

    The CM's system prompt holds a repo snapshot taken at pair start, so it
    cannot see files the agent wrote during the session — reviews based on it
    produced false rejects ("repository snapshot shows no changes"). Give the
    CM trusted, fresh facts instead: which file_writes the system has already
    persisted (with content previews) and how the repo changed since start.
    """
    lines: list[str] = [
        "[SYSTEM VERIFICATION — generated by the system, not by the agent. "
        "Trust this block over the agent's summary and over the Repository "
        "Snapshot in your system prompt (that snapshot predates the agent's work).]"
    ]

    if actions.file_writes:
        lines.append(_render_review_file_writes(actions.file_writes))
    else:
        lines.append("The agent's close_session contained NO file_writes.")

    # Fresh `git diff --stat` of the workspace vs main (merge-base scoped):
    # a compact, complete list of every changed file — immune to snapshot
    # staleness and to path-filter mistakes in the CM's own git_diff calls.
    try:
        from app.core import workspaced_client as wsd

        data = await wsd.git_diff_workspace(
            str(room_id), from_ref="main", stat=True, merge_base=True
        )
        stat = (data.get("diff") or "").strip()
        lines.append("Workspace git changes vs main (diff --stat, fresh):")
        lines.append(f"```\n{stat[:_REVIEW_GIT_STAT_CHARS]}\n```" if stat else "*(none)*")
    except Exception as exc:
        logger.warning(
            "proxy_session: git --stat for CM review failed: %s", exc
        )

    # Files changed since the TASK started (not since this pair started — a
    # timeout/tool_limit restart rebuilds the pair, and a pair-start snapshot
    # would hide everything written before the restart).
    changes = None
    if task_id is not None:
        try:
            changes = await _task_scoped_file_changes(task_id, room_id, db)
        except Exception as exc:
            logger.warning(
                "proxy_session: task-scoped file changes for CM review failed: %s", exc
            )
    if changes is not None:
        lines.append(
            "Repository files changed since the task started (fresh check at review time):"
        )
        lines.append("\n".join(changes) if changes else "*(no repository changes)*")
    else:
        try:
            from app.core.memory_layer import _repo_snapshot

            fresh = await _repo_snapshot(room_id, db)
            before = {f["path"]: f.get("version") for f in (start_snapshot or [])}
            diff_lines: list[str] = []
            ignored_snap = 0
            omitted_snap = 0
            for f in fresh:
                old_v = before.get(f["path"])
                if old_v is not None and old_v == f.get("version"):
                    continue
                if _is_review_path_ignored(f["path"]):
                    ignored_snap += 1
                    continue
                if len(diff_lines) >= _REVIEW_CHANGED_FILES_MAX:
                    omitted_snap += 1
                    continue
                change = (
                    f"created (v{f.get('version')})"
                    if old_v is None
                    else f"v{old_v} → v{f.get('version')}"
                )
                diff_lines.append(f"- `{f['path']}`: {change}")
            if omitted_snap or ignored_snap:
                note_parts: list[str] = []
                if omitted_snap:
                    note_parts.append(f"… and {omitted_snap} more files")
                if ignored_snap:
                    note_parts.append(f"+{ignored_snap} generated/build files ignored")
                diff_lines.append(f"- {'; '.join(note_parts)}")
            lines.append(
                "Repository changes during this session (fresh check at review time):"
            )
            lines.append("\n".join(diff_lines) if diff_lines else "*(no repository changes)*")
        except Exception as exc:
            logger.warning(
                "proxy_session: fresh repo snapshot for CM review failed: %s", exc
            )

    return _truncate_review("\n".join(lines), _REVIEW_TOTAL_MAX_CHARS)


async def _task_scoped_file_changes(
    task_id: UUID, room_id: UUID, db: AsyncSession
) -> Optional[list[str]]:
    """Files touched since the first proxy pair of *task_id* started.

    Workspace mode: reads the volume snapshot (mtime-based filtering) —
    no longer relies on ``RepoFile.updated_at`` (stage 4 of
    internal design notes).
    Legacy mode: queries ``RepoFile.updated_at`` as before.

    Returns None when the task start can't be established (no sessions yet) so
    the caller falls back to the pair-start snapshot comparison.
    """
    import sqlalchemy as sa

    from app.models.session.cm_agent_proxy import CmAgentProxySession

    started = (
        await db.execute(
            sa.select(sa.func.min(CmAgentProxySession.started_at)).where(
                CmAgentProxySession.task_id == task_id
            )
        )
    ).scalar()
    if started is None:
        return None
    # Timestamps from the DB are tz-aware; volume mtime is a POSIX timestamp.
    if started.tzinfo is not None:
        started_naive = started.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        started_naive = started
    started_ts = started_naive.replace(tzinfo=timezone.utc).timestamp()

    from app.core import workspace_fs as wfs

    from app.core import workspaced_client as wsd

    try:
        snapshot = await wsd.get_workspace_snapshot(str(room_id), max_previews=0)
    except Exception:
        return None  # fall back to pair-start snapshot comparison

    changed: list[tuple[str, float]] = []
    ignored_count = 0
    for item in snapshot.get("files", []):
        mtime = float(item.get("mtime", 0))
        if mtime < started_ts:
            continue
        path = wfs.normalize_ws_path(item.get("path", ""))
        if _is_review_path_ignored(path):
            ignored_count += 1
            continue
        changed.append((path, mtime))

    changed.sort(key=lambda x: x[1], reverse=True)
    truncated_count = max(0, len(changed) - _REVIEW_CHANGED_FILES_MAX)
    shown = changed[:_REVIEW_CHANGED_FILES_MAX]

    lines: list[str] = []
    for path, _mtime in shown:
        lines.append(f"- `{path}`: changed")

    if truncated_count > 0 or ignored_count > 0:
        note_parts: list[str] = []
        if truncated_count > 0:
            note_parts.append(f"… and {truncated_count} more files")
        if ignored_count > 0:
            note_parts.append(f"+{ignored_count} generated/build files ignored")
        lines.append(f"- {'; '.join(note_parts)}")

    return lines


async def _pair_closed_externally(
    agent_session: Session,
    cm_session: Session,
    db: AsyncSession,
    *,
    task_id: UUID | None = None,
    run_id: str | None = None,
) -> bool:
    """True if a side we still consider open is already closed in the DB.

    Retry, manual Start and the watchdog close sessions from other DB sessions;
    this loop's in-memory objects never see that. A superseded loop must stop
    immediately — without publishing results, closing sessions again (duplicate
    SESSION_CLOSED) or flipping task state under the replacement loop's feet.
    Column-select bypasses the identity map, so the read is fresh.
    """
    if task_id is not None and run_id is not None:
        runtime = (
            await db.execute(
                select(Task.run_id, Task.runtime_state).where(Task.id == task_id)
            )
        ).first()
        if runtime is None or runtime[0] != run_id or runtime[1] != "running":
            return True

    ids = [s.id for s in (agent_session, cm_session) if s.is_open]
    if not ids:
        return False
    row = (
        await db.execute(
            select(Session.id).where(Session.id.in_(ids), Session.closed_at.isnot(None)).limit(1)
        )
    ).first()
    return row is not None


async def _run_fence_self_orphaned(
    task_id: UUID, run_id: str | None, db: AsyncSession
) -> bool:
    """True when this run still owns the task but its fence says it is not running.

    Distinguishes a genuine supersede (another generation claimed the task, or
    the user stopped/paused it — both legitimate reasons to bail out quietly)
    from a lost fence, where nothing else took over and the task would simply
    hang with no coroutine behind it.
    """
    if not run_id:
        return False
    row = (
        await db.execute(
            select(Task.run_id, Task.runtime_state).where(Task.id == task_id)
        )
    ).first()
    if row is None:
        return False
    current_run, state = row[0], row[1]
    if not isinstance(current_run, str) or not isinstance(state, str):
        return False
    if current_run != run_id:
        return False  # somebody else owns this task now — an ordinary supersede
    return state not in ("running", "stop_requested", "paused")


async def _block_proxy_for_mcp_capability(
    agent_session: Session,
    cm_session: Session,
    task: Task,
    room_id: UUID,
    db: AsyncSession,
    note: str,
    failures: list[Any],
) -> None:
    """Close a proxy pair when its explicitly requested MCP is unavailable."""

    for session in (agent_session, cm_session):
        if session.is_open:
            actions = await _close_side(session, None, "error", db)
            await _emit_closed(session, "error", actions, db)

    code = str(getattr(failures[0], "code", "mcp_unreachable")) if failures else "mcp_unreachable"
    correlation_id = getattr(failures[0], "correlation_id", None) if failures else None
    from app.core.watchdog import block_stalled_task

    await block_stalled_task(
        task.id,
        "mcp_capability_unavailable",
        (
            f"Task '{task.title}' is blocked: the required MCP capability "
            "is unavailable. Fix the connection and press Retry."
        ),
        db,
        extra_meta={
            "capability_error_code": code,
            "correlation_id": correlation_id,
            "remedy": "retry_after_mcp_recovery",
        },
    )
    # The note is intentionally only supplied to the LLM in the non-blocking
    # path.  Keep it out of the durable task/event payload so a Secret label or
    # endpoint can never be persisted through this diagnostic path.
    del note, room_id


async def _block_proxy_for_runtime_failure(
    agent_session: Session,
    cm_session: Session,
    task: Task,
    db: AsyncSession,
    *,
    reason: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Close both sides and persist a retryable runtime diagnostic."""

    for session in (agent_session, cm_session):
        if session.is_open:
            actions = await _close_side(session, None, "error", db)
            await _emit_closed(session, "error", actions, db)
    from app.core.watchdog import block_stalled_task

    await block_stalled_task(
        task.id,
        reason,
        content,
        db,
        extra_meta=metadata,
    )


# ---------------------------------------------------------------------------
# Task data renderer (first message sent to agent)
# ---------------------------------------------------------------------------


def _render_close_note(actions: SessionCloseActions) -> str:
    """Human-readable summary of a close_session call for the room chat."""
    parts: list[str] = []
    if actions.summary:
        parts.append(actions.summary)
    if actions.file_writes:
        files = ", ".join(f"`{fw.path}`" for fw in actions.file_writes)
        parts.append(f"📁 Files: {files}")
    if actions.tasks_update:
        parts.append(f"🔄 Tasks updated: {len(actions.tasks_update)}")
    return "\n\n".join(parts) or "[Work finished without details]"


def _append_user_note(history: list[dict[str, Any]], note: str) -> None:
    """Add a system note to a seeded history without breaking role alternation.

    Anthropic requires alternating roles, so a note that lands on a trailing
    user turn is merged into it (its content may be a tool_result block list).
    """
    if not note:
        return
    last = history[-1] if history else None
    if last is not None and last.get("role") == "user":
        content = last.get("content")
        if isinstance(content, list):
            history[-1] = {**last, "content": content + [{"type": "text", "text": note}]}
        else:
            history[-1] = {**last, "content": f"{content}\n\n{note}"}
    else:
        history.append({"role": "user", "content": note})


async def _durable_context_notes(db: AsyncSession, task_id: UUID) -> str:
    """Read manifest + todo accumulated across previous sessions of this task.

    Every reseed needs these, not just a limit resume: a rework respawns the
    same task against the same workspace, and without the manifest the fresh
    session re-reads every file the previous one already read — the discovery
    loop the non-productivity guard keeps catching.
    """
    from app.core.context_observations import (
        build_task_read_manifest,
        build_task_todo,
        format_read_manifest_note,
        format_todo_note,
    )

    notes: list[str] = []
    try:
        durable_manifest = await build_task_read_manifest(db, task_id=task_id)
    except Exception:
        logger.exception(
            "proxy_session: failed to build durable read manifest for task %s", task_id
        )
        durable_manifest = []
    manifest_note = format_read_manifest_note(durable_manifest)
    if manifest_note:
        notes.append(manifest_note)

    try:
        durable_todo = await build_task_todo(db, task_id=task_id)
    except Exception:
        logger.exception("proxy_session: failed to build durable todo for task %s", task_id)
        durable_todo = []
    todo_note = format_todo_note(durable_todo)
    if todo_note:
        notes.append(todo_note)

    return "\n\n".join(notes)


def _ensure_dialogue_footer(
    history: list[dict[str, Any]], cm_label: str
) -> list[dict[str, Any]]:
    """Make sure the agent's first user turn names ChatMaster as an answerable peer.

    Applied to resumed and reworked histories too: a session seeded before the
    dialogue channel existed carries no footer, and neither does the checkpoint
    resume path that builds its own first message.
    """
    footer = _TASK_DIALOGUE_FOOTER.format(cm=cm_label)
    for msg in history:
        content = msg.get("content")
        if isinstance(content, str):
            if _DIALOGUE_FOOTER_MARKER in content:
                return history
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and _DIALOGUE_FOOTER_MARKER in (block.get("text") or "")
                ):
                    return history
    for idx, msg in enumerate(history):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            history[idx] = {**msg, "content": content + footer}
            return history
        if isinstance(content, list):
            history[idx] = {
                **msg,
                "content": list(content) + [{"type": "text", "text": footer}],
            }
            return history
    return history


def render_task_data(task: Task) -> str:
    """Render task fields as a plain-text message for the initial agent turn."""
    lines = [f"# Task: {task.title}"]
    if task.description:
        lines.append(f"\n{task.description}")
    if task.goal_id:
        lines.append(f"\n**Goal ID:** {task.goal_id}")
    if task.assigned_agent_id:
        lines.append(f"**Assigned agent:** {task.assigned_agent_id}")
    lines.append(f"**State:** {task.state}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compact-summary helpers (used by 8.2 auto-summarization on limit reached)
# ---------------------------------------------------------------------------


def _build_summary_request(working: list[dict[str, Any]]) -> str:
    """Build a one-shot summarization prompt from the last 10 turns of history."""
    recent = working[-10:] if len(working) > 10 else working

    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content[:500]
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", "")[:300])
                elif btype == "tool_use":
                    parts.append(f"[tool:{block.get('name', '?')}]")
                elif btype == "tool_result":
                    inner = block.get("content", "")
                    snippet = (
                        inner[:200] if isinstance(inner, str)
                        else next(
                            (b.get("text", "")[:200] for b in inner
                             if isinstance(b, dict) and b.get("type") == "text"),
                            "",
                        )
                    )
                    if snippet:
                        parts.append(f"[result: {snippet}]")
            return " ".join(parts)[:500]
        return str(content)[:500]

    turns_text = "\n".join(
        f"[{m['role']}]: {_text(m.get('content', ''))}"
        for m in recent
        if isinstance(m, dict)
    )
    return (
        "Summarize in 3-5 sentences what was accomplished in this coding session. "
        "Focus on: files created/modified, decisions made, what's left to do.\n\n"
        f"Recent turns:\n{turns_text}"
    )


async def _call_compact_summary(
    llm_config: dict[str, Any],
    prompt: str,
    *,
    secret_value: SecretValue | None = None,
    downgrade_model: bool = True,
    purpose: str = "session_summary",
    log_session: Any = None,
) -> str:
    """One-shot LLM call for a session summary — no tools, no history.

    Uses the same provider/credentials as the agent. By default it switches to a
    fast model (Haiku for Anthropic, or the agent's own model for other
    providers); pass ``downgrade_model=False`` to run on ``llm_config["model"]``
    as-is — the plan-dialogue digest (§8.1) needs the ChatMaster's real model,
    not a cheap stand-in. Returns a trimmed string ≤500 chars, or raises on
    network/API failure.
    """
    provider = llm_config.get("provider", "anthropic").lower()
    api_key = secret_value.reveal() if secret_value is not None else None
    if not api_key:
        legacy = legacy_secret_value(llm_config, "api_key")
        try:
            api_key = legacy.reveal() if legacy is not None else None
        finally:
            legacy = None
    base_url = llm_config.get("base_url")
    system = "You are a concise technical summarizer. Reply with plain text only, no markdown."

    # Class B of the reasoning plan (§5.3.2): the Hub's own utility call, so no
    # reasoning fields are attached here even when the agent has reasoning on.
    # `max_tokens: 512` below is why this is not a matter of taste — a
    # reasoning model spends the whole cap on thinking and returns an empty
    # `content` (§0.7, reproduced on OpenRouter in §4A.5 q7), which would break
    # summarisation silently.  Do not add apply_reasoning_* to this function.
    model = llm_config.get("model") or "gpt-4o-mini"

    if uses_messages_wire(provider, model):
        if not api_key and uses_anthropic_env_key(provider):
            api_key = os.getenv("ANTHROPIC_API_KEY")
        # Anthropic has a cheap first-party model for summaries; a gateway on
        # the same wire only knows the agent's own model.
        summary_model = (
            "claude-haiku-4-5-20251001"
            if (downgrade_model and provider == "anthropic")
            else model
        )
        with llm_span(provider, SIDE_AGENT):
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    messages_url(provider, base_url),
                    headers=messages_headers(provider, api_key),
                    json={
                        "model": summary_model,
                        "system": system,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        text = next(
            (b["text"] for b in data.get("content", []) if b.get("type") == "text"),
            "",
        ).strip()

    else:
        # OpenAI-compatible providers — keep their credentials and model
        oai_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-no-key-required"
        client_kwargs: dict[str, Any] = {"api_key": oai_key, "timeout": 30.0}
        client_kwargs["base_url"] = openai_base_url(provider, base_url)
        client_kwargs.update(openai_client_extras(client_kwargs["base_url"]))
        oai = llm_client(provider, chat_client_factory=AsyncOpenAI, **client_kwargs)
        oai_resp = await timed_llm(
            oai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
            ),
            provider=provider,
            side=SIDE_AGENT,
        )
        _require_openai_choices(oai_resp)
        text = (oai_resp.choices[0].message.content or "").strip()

    try:
        from app.core.token_budget import log_llm_call

        log_llm_call(
            purpose=purpose,
            model=(summary_model if uses_messages_wire(provider, model) else model),
            session=log_session,
        )
    except Exception:  # pragma: no cover - logging must never break a summary
        pass

    return text[:500] if text else ""


async def _call_with_agent_secret(
    *,
    owner_id: UUID | None,
    agent_id: UUID,
    db: AsyncSession,
    request_id: UUID,
    config: dict[str, Any],
    call_factory: Any,
):
    """Resolve one agent binding for exactly one outbound proxy call."""

    secret_value = (
        await resolve_agent_secret(
            db=db,
            owner_id=owner_id,
            agent_id=agent_id,
            config=config,
            request_id=request_id,
        )
        if owner_id is not None
        else None
    )
    try:
        return await call_factory(secret_value)
    finally:
        secret_value = None


# ---------------------------------------------------------------------------
# Lock order: task before session (stage 1.2.5, internal design notes §8.2)
# ---------------------------------------------------------------------------


async def _flush_task_first(db: AsyncSession, task: Task) -> None:
    """Flush pending changes with the ``task`` row written first.

    After a turn the loop changes both ``session.last_activity_at`` and
    ``task.updated_at`` and flushes them together; the unit of work was free to
    issue ``UPDATE session`` before ``UPDATE task``, while the watchdog, Cancel
    and close paths take the task row first.  Two transactions locking the same
    pair in opposite orders is the residual deadlock of ladder run B2.  Once
    1.2.1 commits before every model call the window shrinks from seconds to
    milliseconds, but only a fixed order removes it.
    """
    await db.flush([task])
    await db.flush()


# ---------------------------------------------------------------------------
# Proxy conversation loop
# ---------------------------------------------------------------------------


async def _run_proxy_loop(
    agent_session: Session,
    cm_session: Session,
    task: Task,
    task_data_text: str,
    room_id: UUID,
    db: AsyncSession,
    agent_history_seed: list[dict[str, Any]] | None = None,
    run_id: str | None = None,
    pending_dispatch: dict[str, Any] | None = None,
) -> bool | None:
    """Run the coordinated agent ↔ CM conversation.

    Each iteration:
      1. Agent processes its latest "user" message and replies (or calls close_session).
      2. If agent replied → ProxyBroker publishes → CM receives the copy as a new user message.
      3. CM processes and replies (or calls close_session == approve/reject).
      4. If CM replied → ProxyBroker publishes → agent receives the copy → loop continues.

    ``pending_dispatch`` is the caller's slot for a follow-up generation of this
    same task (rework). The loop fills it instead of dispatching directly so the
    successor starts only after this run has been finalized — dispatching from
    here races this run's own terminal runtime_state write.
    """
    broker = ProxyBroker()

    agent_llm_config = await _get_agent_llm_config(agent_session.agent_id, db)
    cm_llm_config = await _get_agent_llm_config(cm_session.agent_id, db)
    for session, config, side in (
        (agent_session, agent_llm_config, "agent"),
        (cm_session, cm_llm_config, "chat_master"),
    ):
        session.provider_session_metadata = {
            "provider": config.get("provider"),
            "model": config.get("model"),
            "side": side,
            "resumable": True,
        }
        db.add(session)
    await _flush_task_first(db, task)

    # Nicknames for the persistent AGENT_TOOL_CALL chat events.
    _agent_row = (
        await db.execute(select(Agent).where(Agent.id == agent_session.agent_id))
    ).scalars().first()
    _cm_row = (
        await db.execute(select(Agent).where(Agent.id == cm_session.agent_id))
    ).scalars().first()
    agent_label = _agent_row.nickname if _agent_row else "Agent"
    cm_label = _cm_row.nickname if _cm_row else "ChatMaster"

    agent_memory = await build_memory_layer(agent_session, db)
    agent_sys = build_system_prompt(agent_memory, "cm_agent_proxy_agent")

    cm_memory = await build_memory_layer(cm_session, db)
    cm_sys = build_system_prompt(cm_memory, "cm_agent_proxy_cm")

    # Seed the agent's history with the task data as the first user turn —
    # or, on resume, with the saved history of the paused tool_limit session.
    # The footer names the counterpart: without it the task data reads as a
    # one-way brief and the agent has no reason to believe anyone will answer.
    if agent_history_seed is not None:
        agent_history: list[dict[str, Any]] = list(agent_history_seed)
    else:
        agent_history = [{"role": "user", "content": task_data_text}]
    agent_history = _ensure_dialogue_footer(agent_history, cm_label)

    # loop_guard Signal B: this task has already been restarted repeatedly with
    # nothing to show for it. Force the resumed session to re-plan out loud to
    # the ChatMaster before it grinds the same discovery again.
    from app.core.loop_guard import STALL_RESTART_THRESHOLD as _STALL_RESTART_THRESHOLD

    if agent_history_seed is not None and (task.stall_streak or 0) >= _STALL_RESTART_THRESHOLD:
        agent_history.append({
            "role": "user",
            "content": _STALL_RESTART_DIRECTIVE.format(
                streak=task.stall_streak, cm=cm_label
            ),
        })
        try:
            await emit_event(
                room_id=room_id,
                event_type="AGENT_STALL_INTERVENTION",
                content=(
                    f"Task '{task.title}' was restarted {task.stall_streak} time(s) "
                    "without progress — the agent must agree a new plan with ChatMaster."
                ),
                db=db,
                task_id=task.id,
                metadata={
                    "task_id": str(task.id),
                    "reason": "restart_streak",
                    "stall_streak": task.stall_streak,
                },
            )
        except Exception:
            logger.exception("proxy_session: failed to emit AGENT_STALL_INTERVENTION (restart)")

    _save_history(agent_session, agent_history)
    cm_history: list[dict[str, str]] = []

    # Check workspace tool availability once for the whole conversation
    from app.core.tools import REGISTRY as _ws_registry
    from app.core.tools.registry import strip_dialogue_only as _strip_dialogue_only
    from app.core.workspace_tools import workspace_tools_available
    _ws_available = await workspace_tools_available(room_id, db)

    # MCP tools are dynamic: discover only servers explicitly attached to this
    # room and owned by the agent's user. HTTP links use the verified snapshot
    # preflight; a stale snapshot gets at most one bounded probe.
    _mcp_ant_tools: list[dict[str, Any]] = []
    _mcp_oai_tools: list[dict[str, Any]] = []
    _mcp_bindings: dict[str, Any] = {}
    _mcp_failures: list[Any] = []
    _mcp_user_id = _agent_row.owner_id if _agent_row else None
    if _mcp_user_id:
        try:
            from app.core.mcp_gateway import room_mcp_capabilities

            (
                _mcp_ant_tools,
                _mcp_oai_tools,
                _mcp_bindings,
                _mcp_failures,
            ) = await room_mcp_capabilities(
                room_id, _mcp_user_id, db
            )
        except Exception:
            logger.warning("MCP room tool discovery failed for room %s", room_id, exc_info=True)
            from app.core.mcp_gateway import McpCapabilityFailure

            _mcp_failures = [
                McpCapabilityFailure(
                    installed_item_id=UUID(int=0),
                    display_name="MCP capability",
                    code="mcp_unreachable",
                    message="The MCP check did not complete.",
                    remedy="Run the MCP check again before using it.",
                )
            ]

    _mcp_failure_note = ""
    if _mcp_failures:
        from app.core.mcp_gateway import format_mcp_failure_note, task_requires_mcp_capability

        _mcp_failure_note = format_mcp_failure_note(_mcp_failures)
        if task_requires_mcp_capability(
            f"{task.title}\n{task.description}",
            _mcp_failures,
        ):
            await _block_proxy_for_mcp_capability(
                agent_session,
                cm_session,
                task,
                room_id,
                db,
                _mcp_failure_note,
                _mcp_failures,
            )
            return True
        agent_sys = f"{agent_sys}\n\n{_mcp_failure_note}"
        cm_sys = f"{cm_sys}\n\n{_mcp_failure_note}"

    from app.core.vision_dispatch import agent_can_see_images, no_vision_note

    if not await agent_can_see_images(agent_llm_config, _mcp_bindings):
        agent_sys = f"{agent_sys}\n\n{no_vision_note(task.id)}"

    # Per-agent tool allowlists (null = unrestricted within the role)
    _agent_allowed = _agent_row.tools_allowed if _agent_row else None
    _cm_allowed = _cm_row.tools_allowed if _cm_row else None

    _start_ts = datetime.now(timezone.utc)
    run_clock.reset(task.id)  # a retry starts with a clean wait credit
    _deadline_monotonic = (
        time.monotonic() + max(1.0, _PROXY_MAX_DURATION_SECONDS)
        if _PROXY_MAX_DURATION_SECONDS > 0
        else None
    )
    # internal design notes Stage 1: the status card reads turn/side/deadline
    # from here; the caller (handle_task_created) ends the entry.
    llm_call_state.begin_run(
        task.id,
        run_id=run_id,
        agent_session_id=agent_session.id,
        cm_session_id=cm_session.id,
        max_turns=_MAX_PROXY_TURNS,
        deadline_monotonic=_deadline_monotonic,
    )
    _mcp_failure_counts: dict[str, int] = {}
    # Mid-task question/answer round trips between the agent and CM.
    _qa_rounds = 0
    # internal design notes Stage 6: the last unanswered mid-task
    # question from the agent, so the CM's answer turn can archive the pair.
    _last_agent_question = ""
    # True while the CM owes the agent an answer (mid-task question or stall
    # intervention) rather than a verdict. An answer turn gets a bigger tool
    # budget, is never forced into close_session, and — if the model calls
    # close_session anyway — has that call converted into the answer instead of
    # being applied as a review outcome. A review attempt is only ever spent on
    # work the agent actually submitted via close_session.
    _agent_awaiting_answer = False
    # loop_guard: per-agent-turn scratch dict filled by _call_llm_with_tools —
    # "stall" when the non-productivity detector fired, "productive_calls" so a
    # turn that changed files clears Task.stall_streak.
    _loop_signals: dict[str, Any] = {}
    agent_session.last_activity_at = _start_ts
    cm_session.last_activity_at = _start_ts
    db.add(agent_session)
    db.add(cm_session)
    await _flush_task_first(db, task)

    for _turn in range(_MAX_PROXY_TURNS):
        _turn_deadline = run_clock.effective_deadline(_deadline_monotonic, task.id)
        if _turn_deadline is not None and time.monotonic() >= _turn_deadline:
            await _block_proxy_for_runtime_failure(
                agent_session,
                cm_session,
                task,
                db,
                reason="proxy_timeout",
                content=(
                    f"Task '{task.title}' was stopped by the overall "
                    "execution time limit. Press Retry to try again."
                ),
                metadata={"capability_error_code": "proxy_timeout"},
            )
            return True
        if await _pair_closed_externally(
            agent_session, cm_session, db, task_id=task.id, run_id=run_id
        ):
            # A pair that never got to speak, whose sessions nobody closed and
            # whose run is still the task's current one, was not superseded —
            # its own runtime fence is broken. Returning quietly here leaves the
            # task IN_PROGRESS/DISPATCHED with two open sessions and no living
            # coroutine: invisible in the UI ("starting the session…") and immune to
            # redispatch until the watchdog kills it minutes later.
            if (
                _turn == 0
                and agent_session.is_open
                and cm_session.is_open
                and await _run_fence_self_orphaned(task.id, run_id, db)
            ):
                logger.warning(
                    "proxy_session: pair %s/%s never started — run %s owns task %s but "
                    "its runtime fence is not 'running'; blocking instead of hanging",
                    agent_session.id, cm_session.id, run_id, task.id,
                )
                await _block_proxy_for_runtime_failure(
                    agent_session,
                    cm_session,
                    task,
                    db,
                    reason="runtime_fence_lost",
                    content=(
                        f"Task '{task.title}' could not start: the execution state "
                        "was reset by a previous run. "
                        "Press Retry to restart it."
                    ),
                    metadata={"capability_error_code": "runtime_fence_lost", "run_id": run_id},
                )
                return True
            logger.info(
                "proxy_session: pair %s/%s closed externally — stopping stale loop (turn %d)",
                agent_session.id, cm_session.id, _turn,
            )
            return True

        # ---- Agent turn (skipped once the agent side has closed) ----
        if agent_session.is_open:
            _loop_signals.clear()
            # Stage 6 (§7.3): carries this turn's reasoning tokens out of the
            # tool loop so they land on the session row.  Cleared per turn, so
            # a turn that reasons nothing writes nothing.
            _agent_usage: dict[str, int] = {}
            llm_call_state.set_turn(task.id, _turn, llm_call_state.SIDE_AGENT)
            await emit_status(
                room_id,
                AgentStatus.LLM_REQUEST_SENT,
                agent_id=agent_session.agent_id,
                session_id=agent_session.id,
                task_id=task.id,
                extra={"side": "agent", "turn": _turn},
            )
            try:
                a_text, a_tool, a_in, a_out, a_model, a_max_w, a_last_w = await _call_with_agent_secret(
                    owner_id=_agent_row.owner_id if _agent_row else None,
                    agent_id=agent_session.agent_id,
                    db=db,
                    request_id=task.id,
                    config=agent_llm_config,
                    call_factory=lambda secret_value: _call_with_heartbeat(
                    _call_llm_with_tools(
                        agent_llm_config,
                        agent_sys,
                        agent_history,
                        room_id=room_id,
                        session_id=agent_session.id,
                        task_id=task.id,
                        db=db,
                        workspace_ant_tools=_strip_dialogue_only(_ws_registry.ant_schemas_for("agent", _agent_allowed)) if _ws_available else None,
                        workspace_oai_tools=_strip_dialogue_only(_ws_registry.oai_schemas_for("agent", _agent_allowed)) if _ws_available else None,
                        mcp_ant_tools=_mcp_ant_tools,
                        mcp_oai_tools=_mcp_oai_tools,
                        mcp_bindings=_mcp_bindings,
                        mcp_user_id=_mcp_user_id,
                        or_session_id=str(agent_session.id),
                        max_tool_calls=_AGENT_MAX_TOOL_CALLS,
                        actor_agent_id=agent_session.agent_id,
                        actor_label=agent_label,
                        on_exhaust="raise",
                        budget_used=agent_session.tokens_in + agent_session.tokens_out,
                        context_role="agent",
                        secret_value=secret_value,
                        run_id=run_id,
                        deadline_monotonic=_deadline_monotonic,
                        mcp_failure_counts=_mcp_failure_counts,
                        loop_signals=_loop_signals,
                        usage_stats=_agent_usage,
                    ),
                    room_id=room_id,
                    agent_id=agent_session.agent_id,
                    session_id=agent_session.id,
                    task_id=task.id,
                    deadline_monotonic=_deadline_monotonic,
                ))
            except McpCapabilityCircuitOpen as exc:
                await _block_proxy_for_runtime_failure(
                    agent_session,
                    cm_session,
                    task,
                    db,
                    reason="mcp_capability_circuit_open",
                    content=(
                        f"Task '{task.title}' stopped: MCP tool "
                        f"{exc.tool_name} failed several times. "
                        "Check the connection and press Retry."
                    ),
                    metadata={
                        "capability_error_code": "mcp_capability_circuit_open",
                        "tool_name": exc.tool_name,
                    },
                )
                return True
            except ProxyRunTimeout:
                await _block_proxy_for_runtime_failure(
                    agent_session,
                    cm_session,
                    task,
                    db,
                    reason="proxy_timeout",
                    content=(
                        f"Task '{task.title}' was stopped by the overall "
                        "execution time limit. Press Retry to try again."
                    ),
                    metadata={"capability_error_code": "proxy_timeout"},
                )
                return True
            except RunStopped as exc:
                saved = await _persist_stopped_progress(agent_session, exc, db, agent_llm_config)
                logger.info(
                    "proxy_session: agent run %s for task %s stopped during tool loop "
                    "(history_saved=%s)",
                    run_id, task.id, saved,
                )
                return True
            except ToolLimitReached as exc:
                # Pause instead of failing: persist the working history, close
                # the pair with the actual reason, block the task and let the
                # user choose how to continue.
                limit_reason = getattr(exc, "reason", "tool_calls")
                close_reason = (
                    "context_limit"
                    if limit_reason in ("context_window", "prompt_cap", "model_context_limit")
                    else "token_limit"
                    if limit_reason == "token_budget"
                    else "tool_limit"
                )
                _save_history(agent_session, exc.working)
                track_llm_usage(
                    agent_session, exc.tokens_in, exc.tokens_out,
                    _compute_cost(exc.model, exc.tokens_in, exc.tokens_out, agent_llm_config),
                    model=exc.model,
                )
                update_context_window_metrics(
                    agent_session,
                    getattr(exc, "max_window_in", 0),
                    getattr(exc, "last_window_in", 0),
                )
                # Auto-summarize so 8.1 compaction has a meaningful description
                # instead of just a turn count.
                if not agent_session.summary:
                    try:
                        summary_secret = await resolve_agent_secret(
                            db=db,
                            owner_id=_agent_row.owner_id if _agent_row else None,
                            agent_id=agent_session.agent_id,
                            config=agent_llm_config,
                            request_id=agent_session.id,
                        )
                        # Stage 1.2.2: the summary is a model call too.
                        await release_db(db, reason="compact_summary/tool_limit")
                        try:
                            agent_session.summary = await _call_compact_summary(
                                agent_llm_config,
                                _build_summary_request(exc.working),
                                secret_value=summary_secret,
                            )
                        finally:
                            summary_secret = None
                        logger.info(
                            "proxy_session: auto-summary generated for session %s (%d chars)",
                            agent_session.id, len(agent_session.summary or ""),
                        )
                    except Exception:
                        logger.warning(
                            "proxy_session: auto-summary failed for session %s — "
                            "compaction will use turn count instead",
                            agent_session.id, exc_info=True,
                        )
                db.add(agent_session)
                await _flush_task_first(db, task)
                agent_act = await _close_side(agent_session, None, close_reason, db)
                cm_act = await _close_side(cm_session, None, close_reason, db)
                await _emit_closed(agent_session, close_reason, agent_act, db)
                await _emit_closed(cm_session, close_reason, cm_act, db)
                task.state = TaskState.BLOCKED
                task.updated_at = datetime.utcnow()
                # loop_guard Signal B: a session that ran out of budget/context
                # without finishing is a non-productive restart. Reset happens on
                # the next session that makes real progress or closes cleanly.
                if limit_reason in ("tool_calls", "token_budget", "context_window"):
                    task.stall_streak = (task.stall_streak or 0) + 1
                db.add(task)
                await _flush_task_first(db, task)
                preflight_detail = (
                    f"cannot send the request: the prompt is estimated at "
                    f"{getattr(exc, 'estimated_input_tokens', 0):,} tokens and exceeds "
                    f"the model's physical limit ({getattr(exc, 'cap', 0):,} tokens)."
                    if limit_reason == "model_context_limit"
                    else f"cannot send the request: the prompt is estimated at "
                    f"{getattr(exc, 'estimated_input_tokens', 0):,} tokens, "
                    f"the limit of {getattr(exc, 'cap', 0):,} tokens is exceeded "
                    f"(adjustable: {getattr(exc, 'cap_env_var', 'CONTEXT_AGENT_INPUT_CAP')})."
                    if limit_reason == "prompt_cap"
                    else None
                )
                try:
                    await emit_event(
                        room_id=room_id,
                        event_type=TOOL_LIMIT_REACHED,
                        content=(
                            f"While executing task '{task.title}', agent {agent_label} "
                            + (
                                f"exhausted the token budget ({getattr(exc, 'tokens_used', agent_session.tokens_in + agent_session.tokens_out):,} / {TOKEN_LIMIT:,} tokens)."
                                if limit_reason == "token_budget"
                                else f"filled the context window (limit {getattr(exc, 'context_window_limit', TOKEN_LIMIT):,} tokens)."
                                if limit_reason == "context_window"
                                else f"{preflight_detail} Reduce the task context size or simplify the handoff."
                                if preflight_detail is not None
                                else "repeated the same tool error; automatic retries were stopped."
                                if limit_reason == "repeated_tool_error"
                                else f"reached the tool call limit ({getattr(exc, 'tool_calls_used', _AGENT_MAX_TOOL_CALLS)} / {getattr(exc, 'tool_calls_limit', _AGENT_MAX_TOOL_CALLS)})."
                            )
                            + " Choose how to continue."
                        ),
                        db=db,
                        task_id=task.id,
                        metadata={
                            "task_id": str(task.id),
                            "task_title": task.title,
                            "agent_nickname": agent_label,
                            # ``limit`` remains for old clients; new clients use
                            # the explicit reason-specific fields below.
                            "limit": getattr(exc, "tool_calls_limit", _AGENT_MAX_TOOL_CALLS),
                            "reason": limit_reason,
                            "tool_calls_used": getattr(exc, "tool_calls_used", 0),
                            "tool_calls_limit": getattr(exc, "tool_calls_limit", _AGENT_MAX_TOOL_CALLS),
                            "tokens_used": getattr(exc, "tokens_used", agent_session.tokens_in + agent_session.tokens_out),
                            "tokens_limit": getattr(exc, "tokens_limit", TOKEN_LIMIT),
                            "rejected_reason": getattr(exc, "rejected_reason", None),
                            "estimated_input_tokens": getattr(exc, "estimated_input_tokens", None),
                            "context_cap": getattr(exc, "cap", None),
                            "agent_session_id": str(agent_session.id),
                            # Only RepeatedToolErrorLimitReached sets these —
                            # every other reason leaves them None.
                            "trigger_tool": getattr(exc, "trigger_tool", None),
                            "repeat_count": getattr(exc, "repeat_count", None),
                            "safe_error": getattr(exc, "safe_error", None),
                            "recommended_action": _recommended_action_for_reason(limit_reason),
                        },
                    )
                except Exception:
                    logger.exception("proxy_session: failed to emit TOOL_LIMIT_REACHED")
                break
            except Exception as exc:
                # Preserve the traceback: provider errors alone are often not enough to
                # distinguish a malformed model response from a failure in tool handling.
                logger.exception("proxy_session: agent LLM failed (session=%s): %s", agent_session.id, exc)
                await emit_status(
                    room_id,
                    AgentStatus.LLM_ERROR,
                    agent_id=agent_session.agent_id,
                    session_id=agent_session.id,
                    task_id=task.id,
                    detail=type(exc).__name__,
                    extra={"side": "agent"},
                )
                # The ephemeral LLM_ERROR above vanishes on reload.  Persist the
                # cause and move the task to BLOCKED so the user sees why it
                # stopped and gets a Retry, instead of a silently dead session.
                failure_reason, safe_error, http_status = _classify_llm_failure(exc)
                failure_hint = (
                    " Check the provider API key in the agent settings."
                    if http_status in (401, 403)
                    else ""
                )
                await _block_proxy_for_runtime_failure(
                    agent_session,
                    cm_session,
                    task,
                    db,
                    reason=failure_reason,
                    content=(
                        f"Task '{task.title}' stopped: "
                        + (
                            "the model provider did not answer in time."
                            if failure_reason == "timeout"
                            else f"the model provider returned an error ({safe_error})."
                        )
                        + f"{failure_hint} Press Retry to try again."
                    ),
                    metadata={
                        "task_title": task.title,
                        "agent_nickname": agent_label,
                        "origin_session_id": str(agent_session.id),
                        "origin_side": "agent",
                        "pair_closed": True,
                        "error_class": type(exc).__name__,
                        "safe_error": safe_error,
                        "http_status": http_status,
                        "recommended_action": _recommended_action_for_reason(failure_reason),
                    },
                )
                return True

            # internal design notes Stage 2: user messages delivered during
            # this turn lived only in the tool loop's working list; keep them
            # in the agent's history so later turns still know about them.
            _task_interject.persist_in_history(
                agent_history, _loop_signals.pop("interjections", None)
            )

            if a_text is None and a_tool is None:
                # max_tool_calls exceeded AND the forced close_session failed too
                logger.error(
                    "proxy_session: agent tool loop exhausted for session %s", agent_session.id
                )
                try:
                    await emit_event(
                        room_id=room_id,
                        event_type="TASK_STALLED",
                        content=(
                            f"Agent {agent_label} exhausted the tool call limit "
                            f"({_AGENT_MAX_TOOL_CALLS}) and could not close the session properly. "
                            "Files written up to this point are saved in the workspace."
                        ),
                        db=db,
                        task_id=task.id,
                        metadata={"task_id": str(task.id), "reason": "tool_loop_exhausted"},
                    )
                except Exception:
                    logger.exception("proxy_session: failed to emit tool-loop-exhausted event")
                agent_act = await _close_side(agent_session, None, "error", db)
                cm_act = await _close_side(cm_session, None, "error", db)
                await _emit_closed(agent_session, "error", agent_act, db)
                await _emit_closed(cm_session, "error", cm_act, db)
                break

            # The LLM call is the long window where retry/Start/watchdog may
            # have closed this pair — discard the result instead of publishing.
            if await _pair_closed_externally(
                agent_session, cm_session, db, task_id=task.id, run_id=run_id
            ):
                logger.info(
                    "proxy_session: pair %s/%s closed externally during agent LLM call — "
                    "discarding result and stopping stale loop",
                    agent_session.id, cm_session.id,
                )
                return True

            await emit_status(
                room_id,
                AgentStatus.LLM_RESPONSE_RECEIVED,
                agent_id=agent_session.agent_id,
                session_id=agent_session.id,
                task_id=task.id,
                extra={
                    "side": "agent",
                    "turn": _turn,
                    "tokens_in": a_in,
                    "tokens_out": a_out,
                    "model": a_model,
                },
            )
            if a_tool is None and not (a_text or "").strip():
                await emit_status(
                    room_id,
                    AgentStatus.LLM_EMPTY_RESPONSE,
                    agent_id=agent_session.agent_id,
                    session_id=agent_session.id,
                    task_id=task.id,
                    extra={"side": "agent", "turn": _turn},
                )

            agent_session.last_activity_at = datetime.now(timezone.utc)
            # Task.updated_at is a naive TIMESTAMP column — an aware datetime here
            # makes asyncpg blow up the whole proxy loop (DataError on encode).
            task.updated_at = datetime.utcnow()
            track_llm_usage(
                agent_session, a_in, a_out, _compute_cost(a_model, a_in, a_out, agent_llm_config),
                reasoning_tokens=int(_agent_usage.get("reasoning_tokens", 0) or 0),
                model=a_model,
            )
            update_context_window_metrics(agent_session, a_max_w, a_last_w)
            db.add(agent_session)
            db.add(task)
            await _flush_task_first(db, task)
        else:
            a_text, a_tool = None, None

        if agent_session.is_open and a_tool is not None:
            # Agent called close_session — its work is done; inform CM.
            # This is the only path that turns the next CM turn into a review.
            _agent_awaiting_answer = False
            # A completed session is progress: clear any restart streak.
            if task.stall_streak or 0:
                task.stall_streak = 0
                task.stall_last_progress_at = datetime.now(timezone.utc)
                db.add(task)
            agent_history.append({"role": "assistant", "content": "[session closed]"})
            _save_history(agent_session, agent_history)
            agent_actions = await _close_side(agent_session, a_tool, "task_done", db)
            await _emit_closed(agent_session, "task_done", agent_actions, db)

            # Publish the result so the negotiation is visible in the room chat.
            close_note = _render_close_note(agent_actions)
            try:
                await broker.publish(
                    ProxyMessageCreate(
                        room_id=room_id,
                        sender_type="agent",
                        sender_id=agent_session.agent_id,
                        content=close_note,
                        task_id=task.id,
                        session_id=agent_session.id,
                        peer_session_id=cm_session.id,
                    ),
                    db,
                )
            except Exception as exc:
                logger.warning("proxy_session: publish agent close note failed: %s", exc)

            # A bash call that hit its tool timeout leaves its process running
            # in the container rather than killing it, so a snapshot taken
            # right now could catch a file mid-write and produce a false
            # "missing" in the review. Give it a bounded grace period to
            # finish before the snapshot below is treated as final.
            idle_note = ""
            try:
                from app.core import workspaced_client as wsd

                idle_result = await wsd.wait_workspace_idle(str(room_id))
                if idle_result.get("pending"):
                    idle_note = (
                        "\n[SYSTEM NOTE: at least one shell command from the agent's "
                        "session was still running in the workspace after its tool "
                        f"timeout, even after waiting {idle_result.get('waited_sec')}s. "
                        "The verification below may understate what the agent produced — "
                        "if you're about to flag missing files, consider re-checking "
                        "before requesting changes.]"
                    )
            except Exception as exc:
                logger.warning("proxy_session: wait_workspace_idle failed: %s", exc)

            # internal design notes Stage 1/2: the agent may have produced files
            # only through bash (which, unlike write_file, does not auto-commit).
            # Fold them into the task branch now, while HEAD is still reliably
            # on it, so the review verification, Gate 1 and the eventual merge
            # all see the real result.
            try:
                from app.core.room_git_flow import commit_working_tree

                await commit_working_tree(
                    str(room_id), task, db, reason="finalize task branch after agent session"
                )
            except Exception as exc:
                logger.warning("proxy_session: commit_working_tree failed for task %s: %s", task.id, exc)

            # Push the result to CM so it can review and decide. Include the
            # system verification block: the CM's own repo snapshot is stale
            # (taken at pair start) and must not drive the verdict.
            verification = await _build_review_verification(
                agent_actions,
                cm_memory.get("repo_snapshot") or [],
                room_id,
                db,
                task_id=task.id,
            )
            notification = (
                "[Agent has completed their work and closed the session.]\n\n"
                f"{close_note}\n\n"
                f"{verification}"
                f"{idle_note}\n\n"
                "[Review the result and call close_session with verdict set to "
                "'approve', 'request_changes' (summary must carry concrete "
                "feedback: what is wrong → what to fix, with file paths) or "
                "'reject'. The verdict field is what decides the task's fate — "
                "the summary is prose and may safely mention an earlier ruling.]"
            )
            cm_history.append({"role": "user", "content": notification})
            _save_history(cm_session, cm_history)
        elif agent_session.is_open:
            # A text turn without close_session is a mid-task message to CM: the
            # task is still in progress and this is the agent's question channel.
            # Nothing has been submitted, so the CM's next turn owes an answer,
            # not a verdict — and must not consume a review attempt.
            _agent_awaiting_answer = True
            _stall = _loop_signals.get("stall")
            # A turn that changed files is real progress — clear the restart streak.
            if _loop_signals.get("productive_calls") and (task.stall_streak or 0):
                task.stall_streak = 0
                task.stall_last_progress_at = datetime.now(timezone.utc)
                db.add(task)
            agent_reply = a_text or ""
            agent_history.append({"role": "assistant", "content": agent_reply})
            _save_history(agent_session, agent_history)
            db.add(agent_session)
            await _flush_task_first(db, task)
            _qa_rounds += 1
            if not _stall:
                _last_agent_question = agent_reply
            logger.info(
                "proxy_session: %s from agent for task %s (round %d, turn %d)",
                "STALL intervention" if _stall else "mid-task message",
                task.id, _qa_rounds, _turn,
            )

            try:
                await broker.publish(
                    ProxyMessageCreate(
                        room_id=room_id,
                        sender_type="agent",
                        sender_id=agent_session.agent_id,
                        content=agent_reply,
                        task_id=task.id,
                        session_id=agent_session.id,
                        peer_session_id=cm_session.id,
                        kind="question",
                    ),
                    db,
                )
                agent_session.last_activity_at = datetime.now(timezone.utc)
            except Exception as exc:
                logger.warning("proxy_session: publish agent→CM failed: %s", exc)

            if _stall:
                task.stall_streak = (task.stall_streak or 0) + 1
                task.updated_at = datetime.utcnow()
                db.add(task)
                await _flush_task_first(db, task)
                cm_history.append({
                    "role": "user",
                    "content": _STALL_INTERVENTION_NOTE + agent_reply,
                })
                try:
                    await emit_event(
                        room_id=room_id,
                        event_type="AGENT_STALL_INTERVENTION",
                        content=(
                            f"Agent {agent_label} is stuck on task '{task.title}' "
                            f"({_stall.get('summary') or _stall.get('reason')}). "
                            f"ChatMaster is stepping in (attempt {task.stall_streak})."
                        ),
                        db=db,
                        task_id=task.id,
                        metadata={
                            "task_id": str(task.id),
                            "agent_nickname": agent_label,
                            "reason": _stall.get("reason"),
                            "tool_name": _stall.get("tool_name"),
                            "repeat_count": _stall.get("repeat_count"),
                            "target": _stall.get("target"),
                            "summary": _stall.get("summary"),
                            "stall_streak": task.stall_streak,
                        },
                    )
                except Exception:
                    logger.exception("proxy_session: failed to emit AGENT_STALL_INTERVENTION")
            else:
                cm_history.append({"role": "user", "content": _MID_TASK_NOTE + agent_reply})
            _save_history(cm_session, cm_history)
            await emit_status(
                room_id,
                AgentStatus.WAITING_PEER,
                agent_id=agent_session.agent_id,
                session_id=agent_session.id,
                task_id=task.id,
                extra={"side": "agent", "waiting_for": "chat_master"},
            )

        # ---- CM turn ----
        if not cm_session.is_open:
            break

        _cm_usage: dict[str, int] = {}
        llm_call_state.set_turn(task.id, _turn, llm_call_state.SIDE_CHAT_MASTER)
        await emit_status(
            room_id,
            AgentStatus.LLM_REQUEST_SENT,
            agent_id=cm_session.agent_id,
            session_id=cm_session.id,
            task_id=task.id,
            extra={"side": "chat_master", "turn": _turn},
        )
        try:
            c_text, c_tool, c_in, c_out, c_model, c_max_w, c_last_w = await _call_with_agent_secret(
                owner_id=_cm_row.owner_id if _cm_row else None,
                agent_id=cm_session.agent_id,
                db=db,
                request_id=task.id,
                config=cm_llm_config,
                call_factory=lambda secret_value: _call_with_heartbeat(
                _call_llm_with_tools(
                    cm_llm_config,
                    cm_sys,
                    cm_history,
                    room_id=room_id,
                    session_id=cm_session.id,
                    task_id=task.id,
                    db=db,
                    workspace_ant_tools=_strip_dialogue_only(_ws_registry.ant_schemas_for("cm", _cm_allowed)) if _ws_available else None,
                    workspace_oai_tools=_strip_dialogue_only(_ws_registry.oai_schemas_for("cm", _cm_allowed)) if _ws_available else None,
                    mcp_ant_tools=_mcp_ant_tools,
                    mcp_oai_tools=_mcp_oai_tools,
                    mcp_bindings=_mcp_bindings,
                    mcp_user_id=_mcp_user_id,
                    or_session_id=str(cm_session.id),
                    # An answer turn inspects the workspace before it can name a
                    # way forward, and must never be forced into a verdict.
                    max_tool_calls=(
                        _CM_ANSWER_MAX_TOOL_CALLS if _agent_awaiting_answer
                        else _CM_MAX_TOOL_CALLS
                    ),
                    on_exhaust="answer" if _agent_awaiting_answer else "force_close",
                    actor_agent_id=cm_session.agent_id,
                    actor_label=cm_label,
                    budget_used=cm_session.tokens_in + cm_session.tokens_out,
                    context_role="chat_master",
                    secret_value=secret_value,
                    run_id=run_id,
                    deadline_monotonic=_deadline_monotonic,
                    mcp_failure_counts=_mcp_failure_counts,
                    usage_stats=_cm_usage,
                ),
                room_id=room_id,
                agent_id=cm_session.agent_id,
                session_id=cm_session.id,
                task_id=task.id,
                deadline_monotonic=_deadline_monotonic,
            ))
        except McpCapabilityCircuitOpen as exc:
            await _block_proxy_for_runtime_failure(
                agent_session,
                cm_session,
                task,
                db,
                reason="mcp_capability_circuit_open",
                content=(
                    f"Task '{task.title}' stopped: MCP tool "
                    f"{exc.tool_name} failed several times. "
                    "Check the connection and press Retry."
                ),
                metadata={
                    "capability_error_code": "mcp_capability_circuit_open",
                    "tool_name": exc.tool_name,
                },
            )
            return True
        except ProxyRunTimeout:
            await _block_proxy_for_runtime_failure(
                agent_session,
                cm_session,
                task,
                db,
                reason="proxy_timeout",
                content=(
                    f"Task '{task.title}' was stopped by the overall "
                    "execution time limit. Press Retry to try again."
                ),
                metadata={"capability_error_code": "proxy_timeout"},
            )
            return True
        except RunStopped as exc:
            saved = await _persist_stopped_progress(cm_session, exc, db, cm_llm_config)
            logger.info(
                "proxy_session: ChatMaster run %s for task %s stopped during tool loop "
                "(history_saved=%s)",
                run_id, task.id, saved,
            )
            return True
        except Exception as exc:
            logger.error("proxy_session: CM LLM failed (session=%s): %s", cm_session.id, exc)
            await emit_status(
                room_id,
                AgentStatus.LLM_ERROR,
                agent_id=cm_session.agent_id,
                session_id=cm_session.id,
                task_id=task.id,
                detail=type(exc).__name__,
                extra={"side": "chat_master"},
            )
            cm_act = await _close_side(cm_session, None, "error", db)
            await _emit_closed(cm_session, "error", cm_act, db)
            if agent_session.is_open:
                # The agent is mid-task and waiting for an answer that will now
                # never arrive. Breaking out here would leave its session open
                # and the task IN_PROGRESS with no coroutine behind it.
                await _block_proxy_for_runtime_failure(
                    agent_session,
                    cm_session,
                    task,
                    db,
                    reason="cm_answer_failed",
                    content=(
                        f"Task '{task.title}' stopped: ChatMaster could not "
                        "answer the agent (LLM error). Press Retry to "
                        "restart it."
                    ),
                    metadata={"capability_error_code": "cm_answer_failed"},
                )
                return True
            # Agent already finished but CM failed — move task to WAITING_REVIEW
            # so the user can manually approve or reject instead of it getting stuck.
            if not agent_session.is_open and task.state not in (
                TaskState.COMPLETED, TaskState.REJECTED, TaskState.CANCELLED
            ):
                task.state = TaskState.WAITING_REVIEW
                db.add(task)
                await _flush_task_first(db, task)
                await emit_event(
                    room_id=room_id,
                    event_type="TASK_WAITING_REVIEW",
                    content=(
                        f"Task '{task.title}' moved to manual review "
                        f"— CM session failed (error). Agent work is complete."
                    ),
                    db=db,
                    metadata={"task_id": str(task.id), "cm_session_id": str(cm_session.id)},
                )
            break

        if c_text is None and c_tool is None:
            # CM tool loop exhausted — fall back to WAITING_REVIEW
            logger.error(
                "proxy_session: CM tool loop exhausted for session %s", cm_session.id
            )
            cm_act = await _close_side(cm_session, None, "error", db)
            await _emit_closed(cm_session, "error", cm_act, db)
            if agent_session.is_open:
                # Same wedge as the CM-error branch above: an open agent session
                # with nobody left to answer it.
                await _block_proxy_for_runtime_failure(
                    agent_session,
                    cm_session,
                    task,
                    db,
                    reason="cm_no_answer",
                    content=(
                        f"Task '{task.title}' stopped: ChatMaster did not return "
                        "a reply to the agent. Press Retry to restart it."
                    ),
                    metadata={"capability_error_code": "cm_no_answer"},
                )
                return True
            if not agent_session.is_open and task.state not in (
                TaskState.COMPLETED, TaskState.REJECTED, TaskState.CANCELLED
            ):
                task.state = TaskState.WAITING_REVIEW
                db.add(task)
                await _flush_task_first(db, task)
                await emit_event(
                    room_id=room_id,
                    event_type="TASK_WAITING_REVIEW",
                    content=(
                        f"Task '{task.title}' moved to manual review "
                        "— CM tool loop exhausted."
                    ),
                    db=db,
                    metadata={"task_id": str(task.id), "cm_session_id": str(cm_session.id)},
                )
            break

        # Same guard as the agent turn: don't apply a verdict from a stale loop.
        if await _pair_closed_externally(
            agent_session, cm_session, db, task_id=task.id, run_id=run_id
        ):
            logger.info(
                "proxy_session: pair %s/%s closed externally during CM LLM call — "
                "discarding verdict and stopping stale loop",
                agent_session.id, cm_session.id,
            )
            return True

        await emit_status(
            room_id,
            AgentStatus.LLM_RESPONSE_RECEIVED,
            agent_id=cm_session.agent_id,
            session_id=cm_session.id,
            task_id=task.id,
            extra={
                "side": "chat_master",
                "turn": _turn,
                "tokens_in": c_in,
                "tokens_out": c_out,
                "model": c_model,
            },
        )
        if c_tool is None and not (c_text or "").strip():
            await emit_status(
                room_id,
                AgentStatus.LLM_EMPTY_RESPONSE,
                agent_id=cm_session.agent_id,
                session_id=cm_session.id,
                task_id=task.id,
                extra={"side": "chat_master", "turn": _turn},
            )

        cm_session.last_activity_at = datetime.now(timezone.utc)
        # Naive on purpose — see the agent-turn counterpart above.
        task.updated_at = datetime.utcnow()
        track_llm_usage(
            cm_session, c_in, c_out, _compute_cost(c_model, c_in, c_out, cm_llm_config),
            reasoning_tokens=int(_cm_usage.get("reasoning_tokens", 0) or 0),
            model=c_model,
        )
        update_context_window_metrics(cm_session, c_max_w, c_last_w)
        db.add(cm_session)
        db.add(task)
        await _flush_task_first(db, task)

        # Mechanical gate: a verdict is only meaningful about work that was
        # actually submitted. While the agent's session is open and waiting for
        # an answer, a close_session from the CM cannot end the task — the
        # summary is delivered as the answer instead, and no review attempt is
        # spent. Prompt wording alone did not hold here: an exhausted tool loop
        # used to force the call with tool_choice=close_session.
        _cm_close_converted = False
        if c_tool is not None and _agent_awaiting_answer and agent_session.is_open:
            _forced_summary = str((c_tool or {}).get("summary") or "").strip()
            logger.warning(
                "proxy_session: CM called close_session while agent session %s is "
                "still open and awaiting an answer (task %s) — delivering it as an "
                "answer instead of a verdict",
                agent_session.id, task.id,
            )
            c_text = (c_text or "").strip() or _forced_summary or _CM_NO_GUIDANCE_FALLBACK
            c_tool = None
            _cm_close_converted = True

        if c_tool is not None:
            # CM calling close_session: approve, request_changes, or reject.
            cm_history.append({"role": "assistant", "content": "[session closed]"})
            _save_history(cm_session, cm_history)

            # Read the verdict off the raw tool input before closing, so we can
            # choose the correct close_reason upfront.
            _verdict = classify_review_verdict(c_tool)
            is_rejected = _verdict == VERDICT_REJECT
            is_changes = _verdict == VERDICT_CHANGES

            # Degrade changes to reject when the rework limit is exhausted.
            _degraded_from_changes = False
            if is_changes and task.rework_count >= MAX_TASK_REWORKS:
                logger.info(
                    "proxy_session: task %s hit rework limit (%d/%d) "
                    "— degrading CHANGES_REQUESTED to reject",
                    task.id, task.rework_count, MAX_TASK_REWORKS,
                )
                _degraded_from_changes = True
                is_changes = False
                is_rejected = True

            cm_close_reason = "changes_requested" if is_changes else "task_done"
            cm_actions = await _close_side(cm_session, c_tool, cm_close_reason, db)
            await _emit_closed(cm_session, cm_close_reason, cm_actions, db)

            # Close the agent side if still open; ensure history is saved first for rework.
            if agent_session.is_open:
                if is_changes:
                    _save_history(agent_session, agent_history)
                agent_close_reason = "changes_requested" if is_changes else "task_done"
                agent_act = await _close_side(agent_session, None, agent_close_reason, db)
                await _emit_closed(agent_session, agent_close_reason, agent_act, db)

            if is_changes:
                # === request_changes path ===
                _rework_feedback = cm_actions.summary or ""
                try:
                    await broker.publish(
                        ProxyMessageCreate(
                            room_id=room_id,
                            sender_type="chat_master",
                            sender_id=cm_session.agent_id,
                            content=_render_close_note(cm_actions),
                            task_id=task.id,
                            session_id=cm_session.id,
                            peer_session_id=agent_session.id,
                            kind="request_changes",
                        ),
                        db,
                    )
                except Exception as exc:
                    logger.warning("proxy_session: publish request_changes failed: %s", exc)

                task.rework_count += 1
                task.state = TaskState.ASSIGNED
                task.dispatch_status = DispatchStatus.READY
                db.add(task)
                await _flush_task_first(db, task)

                # emit_event commits — the task state is persisted before respawn.
                await emit_event(
                    room_id=room_id,
                    event_type="TASK_CHANGES_REQUESTED",
                    content=(
                        f"Task '{task.title}' returned for rework "
                        f"(attempt {task.rework_count} of {MAX_TASK_REWORKS})."
                    ),
                    db=db,
                    metadata={
                        "task_id": str(task.id),
                        "task_title": task.title,
                        "cm_session_id": str(cm_session.id),
                        "rework_count": task.rework_count,
                        "max_reworks": MAX_TASK_REWORKS,
                        "feedback": _rework_feedback,
                    },
                )
                # Hand the redispatch to the caller: it must happen after this
                # run is finalized, or the two generations race over
                # Task.runtime_state and the replacement pair dies at turn 0.
                if pending_dispatch is not None:
                    pending_dispatch.update({
                        "resume": True,
                        "rework_feedback": _rework_feedback,
                        "source": "rework",
                    })
                else:
                    # Direct callers (focused tests) without a dispatch slot.
                    try:
                        _rework_outcome = await request_task_dispatch(
                            task, room_id, db,
                            resume=True,
                            rework_feedback=_rework_feedback,
                            source="rework",
                        )
                    except TaskDispatchPreflightError as exc:
                        logger.warning(
                            "proxy_session: rework redispatch for task %s skipped: %s",
                            task.id, exc.code,
                        )
                    else:
                        logger.info(
                            "proxy_session: rework dispatch for task %s outcome=%s",
                            task.id, _rework_outcome,
                        )
                break

            # === approve / reject path ===
            # Publish the CM verdict so the negotiation is visible in the room chat.
            try:
                await broker.publish(
                    ProxyMessageCreate(
                        room_id=room_id,
                        sender_type="chat_master",
                        sender_id=cm_session.agent_id,
                        content=_render_close_note(cm_actions),
                        task_id=task.id,
                        session_id=cm_session.id,
                        peer_session_id=agent_session.id,
                        kind="reject" if is_rejected else "approve",
                    ),
                    db,
                )
            except Exception as exc:
                logger.warning("proxy_session: publish CM verdict failed: %s", exc)
            if is_rejected:
                # Return the task to a reviewable state and emit TASK_REJECTED.
                task.state = TaskState.REJECTED
                db.add(task)
                await _flush_task_first(db, task)
                _reject_content = (
                    f"Task '{task.title}' rejected by ChatMaster "
                    f"(rework limit {MAX_TASK_REWORKS} exhausted). "
                    f"Reason: {cm_actions.summary}"
                ) if _degraded_from_changes else (
                    f"Task '{task.title}' rejected by ChatMaster. "
                    f"Reason: {cm_actions.summary}"
                )
                await emit_event(
                    room_id=room_id,
                    event_type="TASK_REJECTED",
                    content=_reject_content,
                    db=db,
                    metadata={
                        "task_id": str(task.id),
                        "cm_session_id": str(cm_session.id),
                        "reason": cm_actions.summary,
                        **({"limit_exhausted": True} if _degraded_from_changes else {}),
                    },
                )
                # Notify dependents that their predecessor failed (§3.4).
                from app.core.task_deps import get_blocked_dependents
                _blocked = await get_blocked_dependents(task.id, db)
                for _bt in _blocked:
                    await emit_event(
                        room_id=room_id,
                        event_type="TASK_DEPENDENCY_FAILED",
                        content=(
                            f"Task '{_bt.title}' is blocked: "
                            f"predecessor '{task.title}' was rejected."
                        ),
                        db=db,
                        metadata={
                            "task_id": str(_bt.id),
                            "blocked_because": str(task.id),
                            "failed_status": "rejected",
                        },
                    )
            else:
                task.state = TaskState.COMPLETED
                db.add(task)
                await _flush_task_first(db, task)
                # internal design notes Stage 1: fold the task branch into its
                # base (`dev` or the goal branch). Not `main` — that gate is a
                # later stage. GIT_CONFLICT is emitted by merge_task_branch.
                try:
                    from app.core.room_git_flow import merge_task_branch
                    _merge = await merge_task_branch(str(room_id), task, db)
                except Exception as _exc:
                    logger.warning(
                        "proxy_session: task branch merge failed for task %s: %s",
                        task.id, _exc,
                    )
                    _merge = {"ok": False}

                if _merge.get("conflict"):
                    # The task branch diverged from its base (the pre-merge
                    # refresh hit a real conflict) — hold it for a human instead
                    # of reporting it approved. GIT_CONFLICT is already emitted
                    # by merge_task_branch.
                    task.state = TaskState.WAITING_REVIEW
                    db.add(task)
                    await _flush_task_first(db, task)
                    break

                if _merge.get("ok"):
                    logger.info(
                        "proxy_session: merged %s into %s for task %s",
                        _merge.get("branch"), _merge.get("target"), task.id,
                    )
                await emit_event(
                    room_id=room_id,
                    event_type="TASK_APPROVED",
                    content=f"Task '{task.title}' approved by ChatMaster.",
                    db=db,
                    metadata={
                        "task_id": str(task.id),
                        "cm_session_id": str(cm_session.id),
                    },
                )
                if task.goal_id:
                    from app.core.goal_progress import maybe_complete_goal

                    await maybe_complete_goal(task.goal_id, room_id, db)
                    # §4.2: task completed — refresh even if the goal itself
                    # isn't fully done yet (maybe_complete_goal only refreshes
                    # on full completion). Debounced, so this is cheap.
                    from app.core.session_close import maybe_refresh_room_memory

                    await maybe_refresh_room_memory(room_id, db, trigger="task_approved")
                # Unblock and dispatch tasks that were waiting on this one (§3.2).
                # Checkpoint tasks among them are published as CHECKPOINT_REQUEST
                # instead of being dispatched (generation checkpoint plan, stage 3).
                from app.core.task_deps import publish_unblocked_tasks, try_unblock_tasks
                _newly_ready = await try_unblock_tasks(task.id, db)
                await publish_unblocked_tasks(
                    _newly_ready, room_id, db,
                    source="media_task_unblock",
                    unblocked_by=task.id,
                    predecessor_title=task.title,
                )
            break

        # CM replied with text — this is an answer to a mid-task message, not a
        # verdict. Publish it back to the agent so it can continue working.
        cm_reply = c_text or ""
        cm_history.append({"role": "assistant", "content": cm_reply})
        _save_history(cm_session, cm_history)
        db.add(cm_session)
        await _flush_task_first(db, task)

        try:
            await broker.publish(
                ProxyMessageCreate(
                    room_id=room_id,
                    sender_type="chat_master",
                    sender_id=cm_session.agent_id,
                    content=cm_reply,
                    task_id=task.id,
                    session_id=cm_session.id,
                    peer_session_id=agent_session.id,
                    kind="answer" if agent_session.is_open else "cm_agent_proxy",
                ),
                db,
            )
            cm_session.last_activity_at = datetime.now(timezone.utc)
        except Exception as exc:
            logger.warning("proxy_session: publish CM→agent failed: %s", exc)

        if agent_session.is_open and _last_agent_question:
            # internal design notes Stage 6: persist the mid-task Q&A
            # into the task archive so a later rework / sibling task sees the
            # clarification. Best-effort.
            try:
                from app.core import plan_dialogue

                await plan_dialogue.append_to_task_archive(
                    task.id, room_id,
                    [
                        {"role": "agent", "content": _last_agent_question},
                        {"role": "cm", "content": cm_reply},
                    ],
                    db,
                    goal_id=getattr(task, "goal_id", None),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "proxy_session: mid-task archive append failed for task %s: %s",
                    task.id, exc,
                )
            _last_agent_question = ""

        if agent_session.is_open:
            # The dialogue is unbounded by design, but a pair that only talks
            # never delivers: past the soft limit the answer carries "this is
            # final", and near the turn cap it carries "close now".
            _answer = _CM_ANSWER_NOTE + (
                _CM_CLOSE_CONVERTED_NOTE if _cm_close_converted else ""
            ) + cm_reply
            if _qa_rounds >= _QA_SOFT_LIMIT:
                _answer += _QA_FINAL_NOTE.format(rounds=_qa_rounds)
            _turns_left = _MAX_PROXY_TURNS - _turn - 1
            if _turns_left <= _TURN_WARNING_MARGIN:
                _answer += _TURNS_LOW_NOTE.format(left=max(_turns_left, 0))
            agent_history.append({"role": "user", "content": _answer})
            _save_history(agent_session, agent_history)
        else:
            # Agent already closed — nobody will answer. Nudge CM to decide
            # instead of silently leaving its session open forever.
            cm_history.append({
                "role": "user",
                "content": (
                    "[The agent's session is closed — no one will reply to your "
                    "message. Call close_session now with verdict set to "
                    "'approve', 'request_changes' or 'reject'.]"
                ),
            })
            _save_history(cm_session, cm_history)

    else:
        # Safety: max turns exceeded — force-close both open sides.
        logger.warning(
            "proxy_session: max turns (%d) reached for task %s after %d Q&A round(s) "
            "— force-closing",
            _MAX_PROXY_TURNS,
            task.id,
            _qa_rounds,
        )
        if agent_session.is_open:
            act = await _close_side(agent_session, None, "cancelled", db)
            await _emit_closed(agent_session, "cancelled", act, db)
        if cm_session.is_open:
            act = await _close_side(cm_session, None, "cancelled", db)
            await _emit_closed(cm_session, "cancelled", act, db)
        # The pair talked itself out of budget without ever delivering. Leaving
        # the task IN_PROGRESS/DISPATCHED would make it a silent dead end, so
        # block it retryably the same way a stall does.
        try:
            from app.core.watchdog import block_stalled_task

            await block_stalled_task(
                task.id,
                "proxy_max_turns",
                (
                    f"Task '{task.title}' stopped: the agent and ChatMaster exchanged "
                    f"{_MAX_PROXY_TURNS} turns ({_qa_rounds} question/answer rounds), but the task "
                    "was never delivered. Clarify the task wording and press Retry."
                ),
                db,
                extra_meta={
                    "reason_detail": "max_turns",
                    "turns": _MAX_PROXY_TURNS,
                    "qa_rounds": _qa_rounds,
                },
            )
        except Exception:
            logger.exception(
                "proxy_session: failed to block task %s after max turns", task.id
            )


# ---------------------------------------------------------------------------
# Media task handler (image_generation capability)
# ---------------------------------------------------------------------------


async def _close_media_session(
    session: Session,
    task: Task,
    ok: list[str],
    failed: list[tuple[str, str]],
    error: str | None,
    db: AsyncSession,
    with_references: int = 0,
    postprocessed_count: int = 0,
    composited_count: int = 0,
) -> None:
    """Close the media agent session and set task state per the completion policy.

    Policy:
      ≥1 image saved          → COMPLETED / task_done   (partial failures logged per-event)
      phase A error / 0 saved → ASSIGNED + READY / error (dispatch retry)

    ``with_references``/``postprocessed_count`` (stage 6) and
    ``composited_count`` (stage 8) surface in the summary and SESSION_CLOSED
    metadata how many images used each post-generation step.
    """
    room_id: UUID = session.room_id

    if ok:
        task.state = TaskState.COMPLETED
        close_reason = "task_done"
        parts = [f"Generated {len(ok)} image(s): " + ", ".join(ok)]
        extras = []
        if with_references:
            extras.append(f"{with_references} with reference(s)")
        if postprocessed_count:
            extras.append(f"{postprocessed_count} resized to an exact pixel size")
        if composited_count:
            extras.append(f"{composited_count} with logo overlay")
        if extras:
            parts.append("; ".join(extras).capitalize() + ".")
        if failed:
            parts.append("Failed: " + ", ".join(p for p, _ in failed))
        summary = "\n".join(parts)
    else:
        task.state = TaskState.ASSIGNED
        task.dispatch_status = DispatchStatus.READY
        close_reason = "error"
        summary = (
            f"Image generation failed: {error}"
            if error
            else "All images failed: " + "; ".join(f"{p}: {e}" for p, e in failed)
        )

    task.runtime_state = "stopped" if ok else "failed"
    db.add(task)
    session.closed_at = datetime.now(timezone.utc)
    session.close_reason = close_reason
    db.add(session)
    await db.flush()

    await flush_session_memory(
        session, db, reason=close_reason,
        applied_actions=SessionCloseActions(summary=summary),
    )

    await emit_event(
        room_id=room_id,
        event_type="SESSION_CLOSED",
        content=f"Media session {session.id} closed (reason={close_reason})",
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": "cm_agent_proxy",
            "side": "agent",
            "close_reason": close_reason,
            "summary": summary,
            "capability": "image_generation",
            "ok": ok,
            "failed": [{"path": p, "error": e} for p, e in failed],
            "with_references": with_references,
            "postprocessed_count": postprocessed_count,
            "composited_count": composited_count,
        },
    )

    if task.state == TaskState.COMPLETED:
        if task.goal_id:
            from app.core.goal_progress import maybe_complete_goal
            await maybe_complete_goal(task.goal_id, room_id, db)
        # Checkpoint tasks among the unblocked ones are published as
        # CHECKPOINT_REQUEST instead of being dispatched (generation
        # checkpoint plan, stage 3).
        from app.core.task_deps import publish_unblocked_tasks, try_unblock_tasks
        newly_ready = await try_unblock_tasks(task.id, db)
        await publish_unblocked_tasks(
            newly_ready, room_id, db,
            source="media_task_unblock",
            unblocked_by=task.id,
            predecessor_title=task.title,
        )


async def _resolve_phase_b_references(
    room_id: UUID,
    media_spec: dict[str, Any],
    images: list[dict[str, Any]],
    db: AsyncSession,
) -> dict[str, ResolvedRef]:
    """Stage 6, R11: resolve every reference path this task can use, once.

    Covers both ``media_spec.references`` (the task-level list) and every
    ``reference_paths`` entry phase A attached to any image — a path phase A
    named that isn't in ``media_spec.references`` is still worth resolving,
    since ``_validate_image_prompts`` only rejects invented paths, not ones
    that are simply missing from the task-level list. Resolution failures are
    per-path (``resolve_task_references``), never fatal to the task: a path
    that fails to resolve is just absent from the returned mapping, and the
    image it was meant for still generates without it.
    """
    specs_by_path: dict[str, dict[str, Any]] = {}
    for spec in media_spec.get("references") or []:
        raw_path = str((spec or {}).get("path") or "").strip()
        if raw_path:
            specs_by_path.setdefault(raw_path, spec)
    for item in images:
        for raw_path in item.get("reference_paths") or []:
            raw_path = str(raw_path or "").strip()
            if raw_path:
                specs_by_path.setdefault(raw_path, {"path": raw_path})

    if not specs_by_path:
        return {}

    entries = await resolve_task_references(room_id, list(specs_by_path.values()), db)
    return {entry.path: entry.ref for entry in entries if entry.ref is not None}


#: A generation provider returns whatever raster format it likes (OpenRouter +
#: Google models often answer with JPEG, not PNG). The file extension and the
#: ``data:`` MIME must follow the *actual bytes*, never a hardcoded ``.png`` —
#: a mismatch there silently breaks every downstream consumer that trusts the
#: declared type (browser preview, the vision MCP validator, media references).
_IMAGE_EXT_FOR_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def _read_png_dimensions(png: bytes) -> tuple[int, int] | None:
    """Best-effort pixel size of a generated image, for the ``size_produced`` event field."""
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(png)) as img:
            return img.width, img.height
    except Exception:
        return None


def _build_overlay_params(media_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a {normalized_path: spec} dict for every logo_overlay reference in ``media_spec``.

    Used by Phase B to look up anchor/margin_pct/width_pct/opacity without
    re-reading the spec on every image. Paths are normalized so they match the
    keys returned by ``_resolve_phase_b_references`` / ``resolve_task_references``.
    """
    from app.core.path_contract import WorkspacePathError, normalize_workspace_path

    result: dict[str, dict[str, Any]] = {}
    for spec in (media_spec or {}).get("references") or []:
        if (spec or {}).get("role") != "logo_overlay":
            continue
        raw_path = str((spec or {}).get("path") or "").strip()
        if not raw_path:
            continue
        try:
            norm = normalize_workspace_path(raw_path)
        except (WorkspacePathError, Exception):
            norm = raw_path
        result[norm] = spec
    return result


async def _handle_image_generation_task(
    task: Task,
    agent: Agent,
    goal_id: UUID | None,
    plan_id: UUID | None,
    room_id: UUID,
    db: AsyncSession,
) -> None:
    """Two-phase media handler: phase A authors prompts, phase B generates images."""
    from app.core.image_bytes import sniff_image_mime
    from app.core.image_gen import ImageGenHTTPError, generate_image
    from app.core.workspace_fs import fs_write_file

    repo = SessionRepository(db)

    # 1. One agent-side session (no CM side needed for media tasks)
    agent_session = await repo.create(
        CmAgentProxySessionCreate(
            kind="cm_agent_proxy",
            side="agent",
            room_id=room_id,
            task_id=task.id,
            goal_id=goal_id,
            plan_id=plan_id,
            agent_id=agent.id,
            peer_session_id=None,
        )
    )
    await emit_event(
        room_id=room_id,
        event_type="SESSION_STARTED",
        content=f"Media session {agent_session.id} started (image_generation)",
        db=db,
        metadata={
            "session_id": str(agent_session.id),
            "kind": "cm_agent_proxy",
            "side": "agent",
            "capability": "image_generation",
            "task_id": str(task.id),
            "goal_id": str(goal_id) if goal_id else None,
            "plan_id": str(plan_id) if plan_id else None,
        },
    )

    task.state = TaskState.IN_PROGRESS
    db.add(task)
    await db.flush()

    # 2. Phase A — text LLM authors the prompt list (uses agent's skills)
    await emit_status(
        room_id,
        AgentStatus.LLM_REQUEST_SENT,
        agent_id=agent.id,
        session_id=agent_session.id,
        task_id=task.id,
        extra={"side": "agent", "phase": "image_prompts"},
    )
    try:
        images = await _call_with_heartbeat(
            _generate_image_prompts(agent_session, agent, task, db),
            room_id=room_id,
            agent_id=agent.id,
            session_id=agent_session.id,
            task_id=task.id,
        )
    except Exception as exc:
        logger.error(
            "proxy_session: phase A failed for task %s: %s", task.id, exc
        )
        await emit_status(
            room_id,
            AgentStatus.LLM_ERROR,
            agent_id=agent.id,
            session_id=agent_session.id,
            task_id=task.id,
            detail=type(exc).__name__,
            extra={"side": "agent", "phase": "image_prompts"},
        )
        await _close_media_session(
            agent_session, task, ok=[], failed=[], error=str(exc), db=db
        )
        return

    await emit_event(
        room_id=room_id,
        event_type="IMAGE_PROMPTS_GENERATED",
        content=f"Phase A complete: {len(images)} prompt(s) authored",
        db=db,
        metadata={
            "session_id": str(agent_session.id),
            "task_id": str(task.id),
            "count": len(images),
            "filenames": [img["filename"] for img in images],
        },
    )

    # 3. Phase B — deterministic image generation loop
    media_cfg: dict[str, Any] = (agent.llm_config or {}).get("media") or {}
    image_model: str = media_cfg.get("model", "google/gemini-3.1-flash-lite-image")
    base_url: str = media_cfg.get("base_url") or "https://openrouter.ai/api/v1"
    media_spec: dict[str, Any] = task.media_spec or {}

    # Stage 6, R11: one resolve for the whole task — every media_spec.references
    # path plus every reference_paths value phase A attached to any image,
    # deduped by path. A path that fails to resolve is simply absent below;
    # per §6 the image it was meant for still generates, just without it.
    resolved_refs = await _resolve_phase_b_references(room_id, media_spec, images, db)

    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    with_references = 0
    postprocessed_count = 0
    composited_count = 0

    # Stage 8: build path→spec lookup for logo_overlay references so overlay
    # params (anchor, margin_pct, width_pct, opacity) are available per-image.
    _overlay_params: dict[str, dict[str, Any]] = _build_overlay_params(media_spec)

    for _img_idx, item in enumerate(images, start=1):
        # Provisional — the extension is corrected from the generated bytes
        # right before the file is written (a provider may return JPEG/WebP).
        path_stem = f"/project/images/{item['filename']}"
        path = f"{path_stem}.png"
        await emit_status(
            room_id,
            AgentStatus.IMAGE_GENERATION_PROGRESS,
            agent_id=agent.id,
            session_id=agent_session.id,
            task_id=task.id,
            detail=path,
            extra={"current": _img_idx, "total": len(images)},
        )

        size_spec = resolve_size(item, media_spec, media_cfg)
        size_requested = {
            "aspect_ratio": size_spec.aspect_ratio,
            "size": size_spec.size,
            "resolution": size_spec.resolution,
            "fit": size_spec.fit,
        }
        # logo_overlay references are for stage 8's deterministic composite,
        # not for guiding the generation model.
        item_refs: list[tuple[str, ResolvedRef]] = [
            (p, resolved_refs[p])
            for p in (item.get("reference_paths") or [])
            if p in resolved_refs and resolved_refs[p].role != "logo_overlay"
        ]
        reference_meta = [{"path": p, "role": ref.role} for p, ref in item_refs]

        await emit_event(
            room_id=room_id,
            event_type="IMAGE_GENERATION_STARTED",
            content=f"Generating {path}",
            db=db,
            metadata={
                "session_id": str(agent_session.id),
                "path": path,
                "prompt": item["prompt"],
                "references": reference_meta,
                "size_requested": size_requested,
            },
        )

        degraded: list[str] = []
        provider_params: dict[str, Any] = {}
        try:
            media_secret = await resolve_agent_secret(
                db=db,
                owner_id=agent.owner_id,
                agent_id=agent.id,
                config=media_cfg,
                request_id=agent_session.id,
                purpose=SecretPurpose.LLM_MEDIA,
                binding_keys=("secret_binding_id", "media_secret_binding_id"),
            )
            try:
                api_key = (
                    media_secret.reveal() if media_secret is not None else None
                ) or os.getenv("OPENROUTER_API_KEY", "")
                caps = await model_capabilities(base_url, image_model, api_key)
                provider_params, exact_target, degraded = build_provider_params(size_spec, caps)
                input_refs = [to_data_url(ref) for _, ref in item_refs]
                if input_refs and not supports_input_references(caps):
                    # Known-incompatible model (R7): drop references up front rather
                    # than making a doomed request — OpenRouter answers these with a
                    # 404 "No endpoints found that support image input", not the 400
                    # the retry-on-error path below is built to catch.
                    degraded = [*degraded, "input_references"]
                    input_refs = []

                # Stage 1.2.3: image generation is a long provider wait.
                await release_db(db, reason="image_gen")
                try:
                    png = await generate_image(
                        api_key=api_key,
                        prompt=item["prompt"],
                        model=image_model,
                        aspect_ratio=provider_params.get("aspect_ratio"),
                        base_url=base_url,
                        size=provider_params.get("size"),
                        resolution=provider_params.get("resolution"),
                        input_references=input_refs or None,
                    )
                except ImageGenHTTPError as exc:
                    # R8: one retry, dropping the disputed parameter in
                    # priority order size → input_references. A 400 with
                    # neither present has no retry to try — re-raise as-is.
                    # A 404 is only retried for OpenRouter's specific
                    # "no endpoints found that support image input" case (caps
                    # was unknown, so the preventive check above couldn't catch
                    # it) — any other 404 (e.g. an unknown model id) re-raises,
                    # since blindly retrying those would mask a real config error.
                    is_no_image_input_404 = (
                        exc.status == 404
                        and "image input" in exc.body_snippet.lower()
                    )
                    if exc.status != 400 and not is_no_image_input_404:
                        raise
                    if is_no_image_input_404 and not input_refs:
                        raise
                    retry_params = dict(provider_params)
                    if is_no_image_input_404:
                        # We know exactly what the provider objected to — drop
                        # references, leave size/aspect_ratio untouched.
                        degraded = [*degraded, "input_references"]
                        input_refs = []
                    elif retry_params.pop("size", None) is not None:
                        degraded = [*degraded, "size"]
                        retry_params.setdefault(
                            "aspect_ratio", size_spec.aspect_ratio or "4:3"
                        )
                        # R7: the provider rejected exact pixels, but the
                        # Hub-side guarantee doesn't — force the final file to
                        # the originally requested size once generation
                        # succeeds, even though we optimistically didn't know
                        # ahead of time that size was unsupported.
                        if exact_target is None and size_spec.size:
                            try:
                                parsed = parse_media_size(size_spec.size)
                            except ValueError:
                                parsed = None
                            if isinstance(parsed, tuple):
                                exact_target = parsed
                    elif input_refs:
                        degraded = [*degraded, "input_references"]
                        input_refs = []
                    else:
                        raise
                    provider_params = retry_params
                    await release_db(db, reason="image_gen/retry")
                    png = await generate_image(
                        api_key=api_key,
                        prompt=item["prompt"],
                        model=image_model,
                        aspect_ratio=provider_params.get("aspect_ratio"),
                        base_url=base_url,
                        size=provider_params.get("size"),
                        resolution=provider_params.get("resolution"),
                        input_references=input_refs or None,
                    )
            finally:
                media_secret = None
        except Exception as exc:
            failed.append((path, str(exc)))
            await emit_event(
                room_id=room_id,
                event_type="IMAGE_GENERATION_FAILED",
                content=f"Failed to generate {path}: {exc}",
                db=db,
                metadata={
                    "session_id": str(agent_session.id),
                    "path": path,
                    "error": str(exc),
                    "references": reference_meta,
                    "size_requested": size_requested,
                    "degraded": degraded,
                },
            )
            continue

        postprocessed = False
        if exact_target is not None:
            try:
                png = fit_exact(png, exact_target[0], exact_target[1], size_spec.fit)
                postprocessed = True
            except Exception as exc:
                failed.append((path, f"post-process failed: {exc}"))
                await emit_event(
                    room_id=room_id,
                    event_type="IMAGE_GENERATION_FAILED",
                    content=f"Failed to generate {path}: {exc}",
                    db=db,
                    metadata={
                        "session_id": str(agent_session.id),
                        "path": path,
                        "error": str(exc),
                        "references": reference_meta,
                        "size_requested": size_requested,
                        "degraded": degraded,
                    },
                )
                continue

        if postprocessed:
            postprocessed_count += 1
        if item_refs:
            with_references += 1
        size_produced = exact_target if postprocessed else _read_png_dimensions(png)

        # Stage 8: deterministic logo overlay.  Collect logo_overlay refs
        # for this image — per-image reference_paths first, then task-level
        # refs (apply to every image in the series) not already covered.
        per_image_overlay_paths = {
            p for p in (item.get("reference_paths") or [])
            if p in resolved_refs and resolved_refs[p].role == "logo_overlay"
        }
        all_overlay_pairs: list[tuple[str, ResolvedRef]] = [
            (p, resolved_refs[p]) for p in per_image_overlay_paths
        ] + [
            (p, ref)
            for p, ref in resolved_refs.items()
            if ref.role == "logo_overlay" and p not in per_image_overlay_paths
        ]

        for overlay_path, overlay_ref in all_overlay_pairs:
            params = _overlay_params.get(overlay_path, {})
            try:
                png = composite_logo_overlay(
                    png,
                    overlay_ref.data,
                    anchor=str(params.get("anchor") or "top-left"),
                    margin_pct=float(params.get("margin_pct") or 3.0),
                    width_pct=float(params.get("width_pct") or 20.0),
                    opacity=float(params.get("opacity") if params.get("opacity") is not None else 1.0),
                )
                try:
                    from app.core.metrics import record_media_logo_composited
                    record_media_logo_composited()
                except Exception:
                    pass
                await emit_event(
                    room_id=room_id,
                    event_type="IMAGE_COMPOSITED",
                    content=f"Logo overlay applied to {path}",
                    db=db,
                    metadata={
                        "session_id": str(agent_session.id),
                        "path": path,
                        "logo_path": overlay_path,
                        "anchor": str(params.get("anchor") or "top-left"),
                        "margin_pct": float(params.get("margin_pct") or 3.0),
                        "width_pct": float(params.get("width_pct") or 20.0),
                        "opacity": float(params.get("opacity") if params.get("opacity") is not None else 1.0),
                    },
                )
                composited_count += 1
            except Exception as exc:
                logger.warning(
                    "proxy_session: logo overlay failed for %s (logo=%s): %s",
                    path, overlay_path, exc,
                )
                degraded.append("logo_overlay")

        # The extension and MIME follow the real bytes. `fit_exact` and
        # `composite_logo_overlay` always emit PNG, so a post-processed image
        # is genuinely PNG regardless of what the provider first returned.
        mime = sniff_image_mime(png) or "image/png"
        ext = _IMAGE_EXT_FOR_MIME.get(mime, ".png")
        if not path.endswith(ext):
            path = f"{path_stem}{ext}"
        data_uri = f"data:{mime};base64," + base64.b64encode(png).decode()
        await fs_write_file(room_id, path, data_uri, db, agent_id=agent.id)
        ok.append(path)
        await emit_event(
            room_id=room_id,
            event_type="IMAGE_GENERATION_COMPLETED",
            content=f"Saved {path}",
            db=db,
            metadata={
                "session_id": str(agent_session.id),
                "path": path,
                "agent_nickname": agent.nickname,
                "references": reference_meta,
                "size_requested": size_requested,
                "provider_params": provider_params,
                "size_produced": list(size_produced) if size_produced else None,
                "postprocessed": postprocessed,
                "degraded": degraded,
            },
        )

    # 4. Close with outcome report
    await _close_media_session(
        agent_session, task, ok=ok, failed=failed, error=None, db=db,
        with_references=with_references, postprocessed_count=postprocessed_count,
        composited_count=composited_count,
    )


# ---------------------------------------------------------------------------
# Main handler (Variant A — TASK_CREATED)
# ---------------------------------------------------------------------------

# History compaction thresholds for load_paused_history.
# When the saved history exceeds _COMPACT_THRESHOLD_CHARS we keep only the
# first _COMPACT_KEEP_HEAD messages (task description turn) and the last
# _COMPACT_KEEP_TAIL messages, bridging them with a single compact_note so
# the agent knows context was trimmed.
_COMPACT_THRESHOLD_CHARS: int = 40_000  # ~10k tokens; env-overridable at startup
_COMPACT_KEEP_HEAD: int = 1   # first message(s) — carries the task description
_COMPACT_KEEP_TAIL: int = 15  # recent turns verbatim

# Manifest of files already read — surfaced in the compact_note so the resumed
# session does not blindly re-open the same files (main cause of resume loops
# on content-heavy tasks).
_MANIFEST_PREVIEW_CHARS: int = 400   # per file, in the manifest text
_MANIFEST_MAX_ENTRIES: int = 40      # cap to keep the note itself bounded

_READ_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "grep", "glob", "list_dir", "list_directory",
})
# bash commands treated as reads when the leading token matches.
_READ_BASH_CMDS: frozenset[str] = frozenset({
    "cat", "head", "tail", "less", "more", "ls", "find", "grep", "rg",
})


def _preview(text: str, limit: int = _MANIFEST_PREVIEW_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"… [+{len(text) - limit} chars]"


def _classify_tool_use(name: str | None, tool_input: dict | None) -> tuple[str, str] | None:
    """Return (display_name, target) if the call reads a file/dir, else None."""
    if not name:
        return None
    tool_input = tool_input or {}
    if name in _READ_TOOL_NAMES:
        target = (
            tool_input.get("path")
            or tool_input.get("file_path")
            or tool_input.get("pattern")
            or tool_input.get("query")
            or ""
        )
        return name, str(target)
    if name in ("bash", "shell", "run", "exec"):
        cmd = str(tool_input.get("command") or tool_input.get("cmd") or "").strip()
        if not cmd:
            return None
        head = cmd.split(None, 1)[0]
        if head in _READ_BASH_CMDS:
            return f"bash:{head}", cmd
    return None


def _extract_read_manifest(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Walk history and pair read-like tool_use blocks with their tool_result.

    Supports both provider shapes:
      • Anthropic — assistant.content = [{"type":"tool_use", "id", "name", "input"}]
                    user.content      = [{"type":"tool_result", "tool_use_id", "content"}]
      • OpenAI    — assistant.tool_calls = [{"id","function":{"name","arguments"}}]
                    tool message         = {"role":"tool","tool_call_id","content"}

    Duplicates by (name, target) are collapsed to the LAST observed preview.
    """
    import json as _json

    pending: dict[str, dict[str, str]] = {}   # tool_use_id -> {name, target}
    results: dict[str, str] = {}              # tool_use_id -> preview
    order: list[str] = []                     # id order-of-first-appearance

    def _remember(call_id: str, name: str, target: str) -> None:
        if call_id in pending:
            return
        pending[call_id] = {"name": name, "target": target}
        order.append(call_id)

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        # Anthropic-style content blocks.
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "tool_use":
                    hit = _classify_tool_use(block.get("name"), block.get("input"))
                    if hit:
                        _remember(block.get("id") or "", hit[0], hit[1])
                elif btype == "tool_result":
                    call_id = block.get("tool_use_id") or ""
                    if call_id not in pending:
                        continue
                    inner = block.get("content")
                    if isinstance(inner, list):
                        parts = [
                            str(p.get("text", "")) for p in inner
                            if isinstance(p, dict) and p.get("type") == "text"
                        ]
                        results[call_id] = _preview("\n".join(parts))
                    else:
                        results[call_id] = _preview(str(inner or ""))

        # OpenAI-style tool_calls on the assistant turn.
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments")
                try:
                    parsed_args = _json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    parsed_args = {}
                hit = _classify_tool_use(fn.get("name"), parsed_args)
                if hit:
                    _remember(tc.get("id") or "", hit[0], hit[1])

        # OpenAI-style tool result message.
        if role == "tool":
            call_id = msg.get("tool_call_id") or ""
            if call_id in pending:
                results[call_id] = _preview(str(content or ""))

    manifest: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for call_id in order:
        meta = pending[call_id]
        key = (meta["name"], meta["target"])
        if key in seen:
            # Keep only the latest occurrence — drop the earlier one already added.
            manifest = [m for m in manifest if (m["name"], m["target"]) != key]
        seen.add(key)
        manifest.append({
            "name": meta["name"],
            "target": meta["target"],
            "preview": results.get(call_id, "[no result recorded]"),
        })
    return manifest[-_MANIFEST_MAX_ENTRIES:]


def _format_manifest(manifest: list[dict[str, str]]) -> str:
    if not manifest:
        return ""
    lines = ["[Files already inspected in this task — do NOT re-read unless the content is stale.]"]
    for i, entry in enumerate(manifest, 1):
        lines.append(f"\n{i}. {entry['name']} → {entry['target']}")
        preview = entry["preview"]
        if preview:
            lines.append(f"   preview: {preview}")
    return "\n".join(lines)


async def load_paused_history(
    task_id: UUID,
    db: AsyncSession,
    close_reasons: tuple[str, ...] = ("tool_limit", "token_limit", "workspace_paused"),
) -> list[dict[str, Any]] | None:
    """Saved conversation of the latest matching agent session, or None.

    Used by the resume endpoint (tool_limit/token_limit) and by the rework flow
    (changes_requested / task_done) to seed a new proxy pair.

    Large histories are compacted: head (task description) + compact_note +
    tail (recent turns) so the new session does not immediately exhaust the
    token budget replaying 100k+ tokens of prior work.
    """
    import json as _json

    from app.models.session import CmAgentProxySession

    row = (
        await db.execute(
            select(CmAgentProxySession)
            .where(
                CmAgentProxySession.task_id == task_id,
                CmAgentProxySession.side == "agent",
                CmAgentProxySession.close_reason.in_(close_reasons),
            )
            .order_by(CmAgentProxySession.started_at.desc())
            .limit(1)
        )
    ).scalars().first()
    if row is None:
        return None
    messages: list[dict[str, Any]] = list((row.context_snapshot or {}).get("messages") or [])
    if not messages:
        return None

    # --- Compaction ---
    raw_chars = sum(len(_json.dumps(m, ensure_ascii=False)) for m in messages)
    if raw_chars <= _COMPACT_THRESHOLD_CHARS:
        return messages

    min_needed = _COMPACT_KEEP_HEAD + 1 + _COMPACT_KEEP_TAIL  # head + note + tail
    if len(messages) <= min_needed:
        # History is large in bytes but too short to split usefully — return as-is.
        return messages

    head = messages[:_COMPACT_KEEP_HEAD]
    tail = messages[-_COMPACT_KEEP_TAIL:]
    middle_count = len(messages) - _COMPACT_KEEP_HEAD - _COMPACT_KEEP_TAIL
    summary_text = row.summary or f"[{middle_count} turns omitted — token budget reached]"

    # Manifest is derived from the FULL pre-compaction history so nothing gets
    # dropped just because it happens to sit in the omitted middle.
    manifest = _extract_read_manifest(messages)
    manifest_block = _format_manifest(manifest)

    try:
        from app.core.context_observations import build_task_todo, format_todo_note

        durable_todo = await build_task_todo(db, task_id=task_id)
    except Exception:
        logger.exception("proxy_session: failed to build durable todo for task %s", task_id)
        durable_todo = []
    todo_block = format_todo_note(durable_todo)

    note_parts = [
        "[CONTEXT COMPACTED — previous history was too large to replay in full.]",
        f"Summary of omitted work ({middle_count} turns):\n{summary_text}",
    ]
    if manifest_block:
        note_parts.append(manifest_block)
    if todo_block:
        note_parts.append(todo_block)
    note_parts.append(
        "The most recent turns follow below. Continue from the last saved state."
    )
    compact_note: dict[str, Any] = {
        "role": "user",
        "content": "\n\n".join(note_parts),
    }
    logger.info(
        "proxy_session: compacting history for task %s: %d msgs / %d chars → "
        "head=%d + note (manifest=%d) + tail=%d",
        task_id, len(messages), raw_chars, len(head), len(manifest), len(tail),
    )
    return head + [compact_note] + tail


class TaskDispatchPreflightError(ValueError):
    """A task cannot be safely handed to the proxy dispatcher yet."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


# States from which no current call site ever legitimately dispatches: a DRAFT
# or WAITING_APPROVAL task has not been through the human approval step yet
# (internal design notes, part 3, stage 3 — POST /tasks defaults new tasks
# to DRAFT precisely so they do NOT enter the dispatch conveyor on creation),
# WAITING_REVIEW is between execution and a human verdict, and the terminal
# states must never restart silently. dispatch_status alone (READY/BLOCKED) is
# not a safe proxy for "this task is on the execution track" — it only tracks
# dependency readiness, not workflow approval. Every legitimate caller of
# prepare_task_dispatch operates on APPROVED, ASSIGNED, PAUSING, PAUSED,
# RESUMING, BLOCKED, IN_PROGRESS or REJECTED (retry, before it resets the
# state to ASSIGNED) — none of those are in this set.
_NON_DISPATCHABLE_STATES: frozenset[TaskState] = frozenset({
    TaskState.DRAFT,
    TaskState.WAITING_APPROVAL,
    TaskState.WAITING_REVIEW,
    TaskState.COMPLETED,
    TaskState.CANCELLED,
    TaskState.CONTINUED,
})


@dataclass(frozen=True)
class TaskDispatchContext:
    """Resolved, validated data needed to start one task execution."""

    goal_id: UUID | None
    plan_id: UUID | None
    agent: Agent
    chat_master_agent_id: UUID | None
    is_media_task: bool


async def prepare_task_dispatch(
    task: Task,
    db: AsyncSession,
    *,
    require_ready: bool = True,
    target_state: TaskState | None = None,
) -> TaskDispatchContext:
    """Validate that *task* can actually be dispatched and resolve its context.

    A direct single-executor task may have no goal or plan.  Those links
    enrich memory when available; they are not prerequisites for opening the
    agent session. Text tasks still require a configured ChatMaster because
    their proxy conversation has an agent and coordinator side.

    *target_state* is for callers that preflight a transition **before** they
    write it: what must be dispatchable is the state the task is moving into,
    not the one it is leaving.  Without it "Back to Work" out of
    WAITING_REVIEW — an allowed transition per TASK_TRANSITIONS — could never
    succeed: the preflight read the not-yet-changed WAITING_REVIEW and refused
    with invalid_task_state, so the button 422'd on every click.
    """
    if task.assigned_agent_id is None:
        raise TaskDispatchPreflightError(
            "no_assigned_agent", "no executor is assigned"
        )
    effective_state = target_state or task.state
    if effective_state in _NON_DISPATCHABLE_STATES:
        raise TaskDispatchPreflightError(
            "invalid_task_state",
            f"the task cannot be started from state {effective_state.value}",
        )
    # internal design notes T8: no task runs once the room owner's hours are
    # spent — with or without a container.
    from app.core import work_gates

    gate_refusal = await work_gates.work_refusal(task.room_id, db)
    if gate_refusal is not None:
        raise TaskDispatchPreflightError(gate_refusal.reason, gate_refusal.text)
    # T3 (§5.3): nor once the owner's room files are at the quota.
    from app.core.room_storage import PAUSE_REASON as STORAGE_FULL
    from app.core.room_storage import room_storage_refusal

    storage_refusal = await room_storage_refusal(task.room_id, db)
    if storage_refusal is not None:
        raise TaskDispatchPreflightError(STORAGE_FULL, storage_refusal)
    if require_ready and task.dispatch_status == DispatchStatus.BLOCKED:
        # A task whose predecessors had already finished when the edge was
        # written never gets an unblocking transition (try_unblock_tasks is
        # edge-triggered), so re-derive readiness from their states before
        # refusing to start it.
        from app.core.task_deps import refresh_blocked_dispatch_status

        await refresh_blocked_dispatch_status(task, db)
    if require_ready and task.dispatch_status != DispatchStatus.READY:
        code = (
            "blocked_by_dependencies"
            if task.dispatch_status == DispatchStatus.BLOCKED
            else "already_dispatched"
        )
        raise TaskDispatchPreflightError(
            code,
            f"the task is not ready to start (dispatch_status={task.dispatch_status.value})",
        )

    agent_result = await db.execute(select(Agent).where(Agent.id == task.assigned_agent_id))
    agent = agent_result.scalars().first()
    if agent is None:
        raise TaskDispatchPreflightError(
            "assigned_agent_not_found", "the assigned executor was not found"
        )
    from app.core.room_members import agent_in_room

    if not await agent_in_room(task.room_id, agent.id, db):
        raise TaskDispatchPreflightError(
            "assigned_agent_not_in_room",
            f"executor @{agent.nickname} is no longer in the room — assign another agent",
        )

    plan_id: UUID | None = None
    if task.goal_id is not None:
        from app.models.goal_plan import GoalPlan

        plan_result = await db.execute(
            select(GoalPlan.id).where(GoalPlan.goal_id == task.goal_id).limit(1)
        )
        plan_row = plan_result.first()
        plan_id = plan_row[0] if plan_row else None

    media = (agent.llm_config or {}).get("media") or {}
    is_media_task = media.get("capability") == "image_generation"
    if is_media_task:
        return TaskDispatchContext(task.goal_id, plan_id, agent, None, True)

    room_result = await db.execute(select(Room).where(Room.id == task.room_id))
    room = room_result.scalars().first()
    if room is None:
        raise TaskDispatchPreflightError("room_not_found", "room not found")

    from app.core.connector_direct import is_direct_connector

    if is_direct_connector(room, agent):
        # A direct room with a connector agent has no ChatMaster: the result
        # goes straight back to the chat (core/connector_direct.py).
        return TaskDispatchContext(task.goal_id, plan_id, agent, None, False)

    coord_cfg = room.coordinator_config or {}
    cm_agent_id_raw = (
        coord_cfg.get("agent_id")
        or coord_cfg.get("chat_master_agent_id")
        or coord_cfg.get("master_agent_id")
    )
    if not cm_agent_id_raw:
        raise TaskDispatchPreflightError(
            "no_chat_master", "no ChatMaster coordinator is configured in the room"
        )
    try:
        cm_agent_id = UUID(str(cm_agent_id_raw))
    except (ValueError, AttributeError):
        raise TaskDispatchPreflightError(
            "invalid_chat_master_id", "the ChatMaster identifier is invalid"
        ) from None

    cm_result = await db.execute(select(Agent).where(Agent.id == cm_agent_id))
    if cm_result.scalars().first() is None:
        raise TaskDispatchPreflightError(
            "chat_master_not_found", "the configured ChatMaster was not found"
        )

    return TaskDispatchContext(task.goal_id, plan_id, agent, cm_agent_id, False)


async def _emit_dispatch_failed(
    task: Task,
    room_id: UUID,
    reason: str,
    db,
    run_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    """Persistent, user-visible marker that a claimed task will NOT get a session.

    The claim (READY→DISPATCHED) already happened in handle_task_created before
    this runs, so leaving dispatch_status untouched here would strand the task
    as "assigned + dispatched" forever — indistinguishable from a genuinely
    running session and invisible to request_task_dispatch's idempotency
    check. Reset it back to READY (and drop a state stuck at IN_PROGRESS to
    BLOCKED) so the task reads as a normal retryable failure.
    """
    logger.warning(
        "proxy_session: dispatch failed for task %s (reason=%s, run_id=%s, room=%s)",
        task.id,
        reason,
        run_id,
        room_id,
    )
    task.dispatch_status = DispatchStatus.READY
    if task.state == TaskState.IN_PROGRESS:
        task.state = TaskState.BLOCKED
    task.updated_at = datetime.utcnow()
    db.add(task)
    try:
        await emit_event(
            room_id=room_id,
            event_type=TASK_DISPATCH_FAILED,
            content=(
                f"Task \"{task.title}\": the agent session did not start ({reason}). "
                "Check the room settings and press Retry."
            ),
            db=db,
            task_id=task.id,
            metadata={"task_id": str(task.id), "reason": reason, **(extra_metadata or {})},
        )
        await db.commit()
    except Exception:
        logger.exception("proxy_session: failed to emit TASK_DISPATCH_FAILED for %s", task.id)


def _sync_committed_claim(task: Any, **values: Any) -> None:
    """Align in-memory attributes with an UPDATE that is already committed.

    ``set_committed_value`` writes the attribute without marking it dirty, so
    the value is visible to the rest of the run but never re-flushed on top of
    a fence another worker may have written since the commit. Focused tests
    pass lightweight task stand-ins that SQLAlchemy cannot instrument — plain
    assignment is the correct behaviour there.
    """
    from sqlalchemy.orm.attributes import set_committed_value

    for field, value in values.items():
        try:
            set_committed_value(task, field, value)
        except Exception:
            setattr(task, field, value)


async def handle_task_created(
    task_id: UUID,
    room_id: UUID,
    goal_id: UUID | None,
    plan_id: UUID | None,
    db: AsyncSession,
    resume: bool = False,
    rework_feedback: str | None = None,
) -> None:
    """React to TASK_CREATED by creating a proxy session pair and running the conversation.

    Silently skips if the task has no assigned agent or the room has no ChatMaster.
    With resume=True the agent history is seeded from the latest paused session:
    - tool_limit (default): agent continues where it stopped after user increased the limit.
    - rework (rework_feedback is not None): agent history from the previous submission is
      loaded and the CM's feedback is appended so the agent knows what to fix.
    """
    task_result = await db.execute(select(Task).where(Task.id == task_id))
    task = task_result.scalars().first()
    if task is None:
        logger.error("proxy_session: task %s not found — aborting", task_id)
        return
    if task.assigned_agent_id is None:
        logger.info("proxy_session: task %s has no assigned agent — skipping", task_id)
        return
    if task.state in _NON_DISPATCHABLE_STATES:
        logger.info(
            "proxy_session: task %s is in non-dispatchable state %s — skipping",
            task_id,
            task.state,
        )
        return
    if task.dispatch_status != DispatchStatus.READY:
        logger.info(
            "proxy_session: task %s has dispatch_status=%s — skipping",
            task_id,
            task.dispatch_status,
        )
        return

    # A run id is the durable ownership fence for this background coroutine.
    # Retry/Cancel/Pause can revoke it from another worker without relying on
    # this process's in-memory Session identity map.
    run_id = str(uuid4())

    # Atomically claim the task (READY → DISPATCHED) and commit immediately.
    # Concurrent dispatchers run in separate DB sessions, so a read-then-flush
    # guard races: the conditional UPDATE + rowcount check is the real gate —
    # exactly one transaction sees rowcount=1, the rest see 0 and bail out.
    #
    # The whole runtime identity of the generation (run_id + runtime_state +
    # a cleared stop fence) is written by this one statement rather than by a
    # follow-up ORM flush. An ORM flush only emits the columns it believes
    # changed, and this row is routinely loaded *before* the outgoing run
    # finalizes itself: with a stale "running" in the identity map the
    # assignment below is a no-op, the UPDATE carries run_id alone, and the
    # replacement generation starts life with the previous run's terminal
    # runtime_state="stopped" — which the first _pair_closed_externally check
    # reads as "superseded" and the new pair dies at turn 0 without ever
    # calling the model.
    claim = await db.execute(
        update(Task)
        .where(
            Task.id == task_id,
            Task.dispatch_status == DispatchStatus.READY,
            Task.state.notin_(_NON_DISPATCHABLE_STATES),
        )
        .values(
            dispatch_status=DispatchStatus.DISPATCHED,
            run_id=run_id,
            runtime_state="running",
            stop_requested_at=None,
            stop_reason=None,
        )
    )
    if claim.rowcount == 0:
        await db.rollback()
        logger.info(
            "proxy_session: task %s already claimed by another dispatcher — skipping",
            task_id,
        )
        return
    await db.commit()
    prev_dispatch_status = task.dispatch_status
    prev_state = task.state

    # Mirror the claim onto the in-memory row without marking it dirty: the
    # values are already durable, and a dirty attribute would be re-flushed
    # later, potentially overwriting a stop fence set in the meantime.
    _sync_committed_claim(
        task,
        dispatch_status=DispatchStatus.DISPATCHED,
        run_id=run_id,
        runtime_state="running",
        stop_requested_at=None,
        stop_reason=None,
    )

    logger.info(
        "proxy_session: claimed task %s for dispatch (run_id=%s, prev_state=%s, "
        "prev_dispatch_status=%s, room=%s, resume=%s)",
        task.id,
        run_id,
        prev_state,
        prev_dispatch_status,
        room_id,
        resume,
    )

    # Persisted, unconditional claim marker (part 3, stage 4). Unlike the
    # TASK_STATE_CHANGED event below — which only fires when task.state
    # actually flips to in_progress — this always fires on a successful claim,
    # so TaskOut.dispatch_started_at has a reliable timestamp even when the
    # task was already in_progress (resume/rework) or when dispatch later
    # fails before a session pair exists.
    await emit_event(
        room_id=room_id,
        event_type=TASK_DISPATCH_STARTED,
        content=f"The dispatcher took task '{task.title}' into work.",
        db=db,
        task_id=task.id,
        metadata={"task_id": str(task.id), "run_id": run_id, "resume": resume},
    )

    # The dispatcher has now claimed the task, so this is the first point at
    # which an automatic transition to IN_PROGRESS is truthful. It covers a
    # direct ChatMaster task even if no separate task_update was called.
    if task.state != TaskState.IN_PROGRESS:
        previous_state = task.state
        task.state = TaskState.IN_PROGRESS
        task.updated_at = datetime.utcnow()
        db.add(task)
        await db.flush()
        await emit_event(
            room_id=room_id,
            event_type="TASK_STATE_CHANGED",
            content=f"Task \"{task.title}\": {previous_state.value} → in_progress",
            db=db,
            task_id=task.id,
            metadata={
                "task_id": str(task.id),
                "task_title": task.title,
                "prev_state": previous_state.value,
                "task_state": TaskState.IN_PROGRESS.value,
                "dispatch": "claimed",
            },
        )

    # Immediate user feedback: the dispatch really started (ephemeral, WS-only).
    await emit_status(
        room_id,
        AgentStatus.DISPATCHING,
        agent_id=task.assigned_agent_id,
        task_id=task_id,
    )

    # Route media-capable agents before the text-proxy flow.
    agent_result = await db.execute(select(Agent).where(Agent.id == task.assigned_agent_id))
    agent = agent_result.scalars().first()
    if agent is not None:
        agent_media = (agent.llm_config or {}).get("media") or {}
        if agent_media.get("capability") == "image_generation":
            await _handle_image_generation_task(task, agent, goal_id, plan_id, room_id, db)
            return

    # An agent handed off
    # to a connector cannot run at all without an active grant into *this*
    # Room, even though the connector transport itself does not exist yet
    # (Stage 4 wires `_run_connector_loop`). Same failure shape as the
    # room/ChatMaster checks below — claim already happened, so this reverts
    # it via `_emit_dispatch_failed` rather than raising a preflight error.
    if agent is not None and agent.executor == "connector":
        from app.core.connector_pairing import has_active_grant

        granted = (
            agent.connector_id is not None
            and await has_active_grant(db, connector_id=agent.connector_id, room_id=room_id)
        )
        if not granted:
            await _emit_dispatch_failed(
                task, room_id, "connector_not_granted", db, run_id=run_id,
                extra_metadata={
                    "connector_id": str(agent.connector_id) if agent.connector_id else None,
                    "agent_id": str(agent.id),
                },
            )
            from app.core.task_runtime import finalize_run
            await finalize_run(task.id, run_id, db, failed=True, task=task)
            return

        # The connector ships separately from the Hub: refuse a protocol
        # outside this Hub's range, or an agent mode the connector does not
        # offer, now rather than after CONNECTOR_DISPATCH_TIMEOUT.
        from app.core.connector_session import connector_block_reason

        block_reason = await connector_block_reason(db, agent)
        if block_reason is not None:
            await _emit_dispatch_failed(task, room_id, block_reason, db, run_id=run_id)
            from app.core.task_runtime import finalize_run
            await finalize_run(task.id, run_id, db, failed=True, task=task)
            return

    room_result = await db.execute(select(Room).where(Room.id == room_id))
    room = room_result.scalars().first()
    if room is None:
        logger.error("proxy_session: room %s not found — aborting", room_id)
        await _emit_dispatch_failed(task, room_id, "room_not_found", db, run_id=run_id)
        from app.core.task_runtime import finalize_run
        await finalize_run(task.id, run_id, db, failed=True, task=task)
        return

    from app.core.connector_direct import is_direct_connector

    direct_connector = is_direct_connector(room, agent)
    coord_cfg = room.coordinator_config or {}
    cm_agent_id_raw = (
        coord_cfg.get("agent_id")
        or coord_cfg.get("chat_master_agent_id")
        or coord_cfg.get("master_agent_id")
    )
    if direct_connector:
        # No ChatMaster side: the connector's answer goes straight to the chat.
        cm_agent_id_raw = None
    elif not cm_agent_id_raw:
        logger.info("proxy_session: room %s has no master_agent_id — skipping", room_id)
        await _emit_dispatch_failed(task, room_id, "no_chat_master", db, run_id=run_id)
        from app.core.task_runtime import finalize_run
        await finalize_run(task.id, run_id, db, failed=True, task=task)
        return

    try:
        cm_agent_id = UUID(str(cm_agent_id_raw)) if cm_agent_id_raw else None
    except (ValueError, AttributeError):
        logger.error(
            "proxy_session: invalid master_agent_id %r in room %s — aborting",
            cm_agent_id_raw,
            room_id,
        )
        await _emit_dispatch_failed(task, room_id, "invalid_chat_master_id", db, run_id=run_id)
        from app.core.task_runtime import finalize_run
        await finalize_run(task.id, run_id, db, failed=True, task=task)
        return

    repo = SessionRepository(db)

    # Create agent-side first (no peer yet)
    agent_session = await repo.create(
        CmAgentProxySessionCreate(
            kind="cm_agent_proxy",
            room_id=room_id,
            side="agent",
            task_id=task_id,
            goal_id=goal_id,
            plan_id=plan_id,
            agent_id=task.assigned_agent_id,
            peer_session_id=None,
        )
    )

    # Create CM-side linked to agent
    cm_session = None
    if cm_agent_id is not None:
        cm_session = await repo.create(
            CmAgentProxySessionCreate(
                kind="cm_agent_proxy",
                room_id=room_id,
                side="chat_master",
                task_id=task_id,
                goal_id=goal_id,
                plan_id=plan_id,
                agent_id=cm_agent_id,
                peer_session_id=agent_session.id,
            )
        )

        # Back-link: agent now points to CM
        agent_session.peer_session_id = cm_session.id
        db.add(agent_session)
        await db.flush()

    if resume:
        try:
            from app.core.workspace_checkpoint import apply_checkpoint_to_session

            await apply_checkpoint_to_session(
                room_id,
                agent_session.id,
                db,
                task_id=task.id,
            )
        except Exception:
            logger.exception(
                "proxy_session: failed to restore checkpoint state for session %s",
                agent_session.id,
            )

    pair = [(agent_session, "agent")]
    if cm_session is not None:
        pair.append((cm_session, "chat_master"))
    for session, side in pair:
        peer_id = (cm_session.id if cm_session else None) if side == "agent" else agent_session.id
        await emit_event(
            room_id=room_id,
            event_type="SESSION_STARTED",
            content=f"Session {session.id} started (kind=cm_agent_proxy, side={side})",
            db=db,
            metadata={
                "session_id": str(session.id),
                "kind": "cm_agent_proxy",
                "side": side,
                "task_id": str(task_id),
                "goal_id": str(goal_id) if goal_id else None,
                "plan_id": str(plan_id) if plan_id else None,
                "peer_session_id": str(peer_id) if peer_id else None,
            },
        )

    logger.info(
        "proxy_session: started session pair for task %s (run_id=%s, agent_session=%s, cm_session=%s)",
        task_id,
        run_id,
        agent_session.id,
        cm_session.id if cm_session else None,
    )

    # A direct-room message for an operator-mode connector runs on the owner's
    # machine and never touches the Room's repository: no task branch, no
    # skill files, and no reason to wake the Room's workspace for either.
    needs_workspace = not (
        direct_connector
        and str((agent.llm_config or {}).get("connector_mode") or "operator") == "operator"
    )

    # Create the task branch in the workspace (non-blocking). internal design notes
    # Stage 1: branched off `dev` or the goal branch, not off whatever HEAD is —
    # `main` never receives agent work.
    try:
        if needs_workspace:
            from app.core.room_git_flow import ensure_task_branch
            branch_name = await ensure_task_branch(str(room_id), task, db)
            await db.flush()
            logger.info("proxy_session: task branch %s ready for task %s", branch_name, task_id)
    except Exception as _exc:
        logger.warning(
            "proxy_session: could not prepare task branch for task %s: %s",
            task_id, _exc,
        )

    # Materialize the dispatched agent's bundled skill files onto the workspace
    # volume (internal design notes Stage 3), non-blocking.
    if agent is not None and needs_workspace:
        try:
            from app.core.skill_materialize import materialize_agent_skills

            await materialize_agent_skills(room_id, agent, db)
        except Exception as _exc:
            logger.warning(
                "proxy_session: skill materialization failed for agent %s in room %s: %s",
                agent.id, room_id, _exc,
            )

    task_data_text = render_task_data(task)

    agent_history_seed: list[dict[str, Any]] | None = None
    if resume:
        if rework_feedback is not None:
            # Rework: load agent history from the previous submission (task_done) or
            # from a prior rework that was also changes_requested.
            agent_history_seed = await load_paused_history(
                task_id, db, close_reasons=("changes_requested", "task_done")
            )
            if agent_history_seed:
                feedback_note = (
                    "[ChatMaster reviewed your work and requested changes — "
                    "the task is NOT done yet]\n"
                    f"{rework_feedback}\n"
                    "[Continue in the same workspace: fix the issues and call "
                    "close_session again with an updated summary.]"
                )
                # Same durable manifest/todo the limit resume gets: a rework is
                # a restart of the same task, and without it the new session
                # re-discovers the workspace from scratch.
                context_note = await _durable_context_notes(db, task_id)
                if context_note:
                    feedback_note = f"{feedback_note}\n\n{context_note}"
                _append_user_note(agent_history_seed, feedback_note)
            else:
                logger.warning(
                    "proxy_session: rework requested for task %s but no prior agent history found "
                    "— starting fresh", task_id,
                )
        else:
            # Limit resume: continue with the complete saved provider history.
            try:
                from app.core.metrics import record_workspace_token_limit_event

                record_workspace_token_limit_event("continued")
            except Exception:
                pass
            agent_history_seed = await load_paused_history(task_id, db)

            if agent_history_seed:
                resume_note = (
                    "The user decided to continue after the session limit was reached. "
                    "Continue the task exactly from where you stopped — "
                    "do not start over and do not repeat steps already done."
                )
                context_note = await _durable_context_notes(db, task_id)
                if context_note:
                    resume_note = f"{resume_note}\n\n{context_note}"
                _append_user_note(agent_history_seed, resume_note)
            else:
                logger.warning(
                    "proxy_session: resume requested for task %s but no paused history found "
                    "— starting fresh", task_id,
                )
        try:
            from app.core.workspace_checkpoint import checkpoint_resume_note

            checkpoint_note = await checkpoint_resume_note(room_id, db)
            if checkpoint_note:
                if agent_history_seed:
                    last = agent_history_seed[-1]
                    if last.get("role") == "user":
                        content = last.get("content")
                        if isinstance(content, list):
                            agent_history_seed[-1] = {
                                **last,
                                "content": content
                                + [{"type": "text", "text": checkpoint_note}],
                            }
                        else:
                            agent_history_seed[-1] = {
                                **last,
                                "content": f"{content}\n\n{checkpoint_note}",
                            }
                    else:
                        agent_history_seed.append(
                            {"role": "user", "content": checkpoint_note}
                        )
                else:
                    agent_history_seed = [
                        {
                            "role": "user",
                            "content": f"{task_data_text}\n\n{checkpoint_note}",
                        },
                    ]
        except Exception:
            logger.exception(
                "proxy_session: failed to build checkpoint note for task %s",
                task_id,
            )

    # Filled by the loop when a verdict schedules another generation of this
    # same task (rework). Dispatching from inside the loop would race this
    # run's own finalization, so the loop only records the intent here.
    pending_dispatch: dict[str, Any] = {}

    # The only branch point between the
    # local and remote executors. Everything above (claim, session-pair
    # creation, task_data_text/history) and below (finalize_run, rework
    # redispatch) is shared — only who runs the loop changes.
    from app.core.connector_session import _run_connector_loop, resolve_executor

    executor = await resolve_executor(agent, db)
    if executor == "connector":
        await _run_connector_loop(
            agent_session=agent_session,
            cm_session=cm_session,
            task=task,
            task_data_text=task_data_text,
            room_id=room_id,
            db=db,
            agent_history_seed=agent_history_seed,
            run_id=run_id,
            pending_dispatch=pending_dispatch,
        )
    else:
        try:
            await _run_proxy_loop(
                agent_session=agent_session,
                cm_session=cm_session,
                task=task,
                task_data_text=task_data_text,
                room_id=room_id,
                db=db,
                agent_history_seed=agent_history_seed,
                run_id=run_id,
                pending_dispatch=pending_dispatch,
            )
        finally:
            llm_call_state.end_run(task.id, run_id)
    # Every terminal path settles through the same run_id-fenced update. The
    # ORM shortcut this replaces had no fence: a rework redispatched from
    # inside the loop could already own the row, and the outgoing run would
    # then stamp runtime_state="stopped" onto the *replacement* generation.
    from app.core.task_runtime import finalize_run

    await finalize_run(task.id, run_id, db, task=task)
    # finalize_run only flushes; a room without a workspace row (a direct room
    # with a connector agent) has no later commit, and the task would read as
    # still running.
    await db.commit()

    # Only now, with this generation settled, hand the task to its successor.
    if pending_dispatch:
        try:
            outcome = await request_task_dispatch(
                task, room_id, db,
                resume=pending_dispatch.get("resume", True),
                rework_feedback=pending_dispatch.get("rework_feedback"),
                source=pending_dispatch.get("source", "rework"),
            )
        except TaskDispatchPreflightError as exc:
            # Leave the task ASSIGNED + READY (the loop already set that) — the
            # feedback and closed sessions are persisted, so the task is
            # visibly queued and the user can retry it.
            logger.warning(
                "proxy_session: redispatch for task %s skipped: %s", task.id, exc.code,
            )
        else:
            logger.info(
                "proxy_session: redispatch for task %s outcome=%s", task.id, outcome,
            )


# ---------------------------------------------------------------------------
# Background dispatch
# ---------------------------------------------------------------------------


async def _run_proxy_bg(
    task_id: UUID,
    room_id: UUID,
    goal_id: UUID | None,
    plan_id: UUID | None,
    resume: bool = False,
    rework_feedback: str | None = None,
) -> None:
    """Open an independent DB session and run handle_task_created."""
    from app.db.session import engine

    async_session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session_factory() as db:
        try:
            await handle_task_created(
                task_id, room_id, goal_id, plan_id, db,
                resume=resume, rework_feedback=rework_feedback,
            )
        except asyncio.CancelledError:
            # A same-worker Cancel interrupts the coroutine immediately.  The
            # API has already written the durable stop fence and closed the
            # proxy rows; settle the visible runtime state before propagating
            # cancellation to asyncio.
            try:
                current = (
                    await db.execute(select(Task).where(Task.id == task_id))
                ).scalars().first()
                if current is not None:
                    from app.core.task_runtime import finalize_run

                    await finalize_run(task_id, current.run_id, db)
                    await db.commit()
            except Exception:
                logger.exception(
                    "proxy_session: failed to settle cancelled runtime for task %s",
                    task_id,
                )
                await db.rollback()
            raise
        except Exception:
            logger.exception(
                "proxy_session: unhandled error in background handler for task %s", task_id
            )
            await db.rollback()
            # Safety net: the rollback may have wiped session activity and left
            # open sessions / an IN_PROGRESS task with no living coroutine.
            # Close them and block the task so the crash is visible in the chat
            # and the task gets a Retry button instead of hanging forever.
            try:
                from app.core.watchdog import block_stalled_task, close_open_task_sessions

                await close_open_task_sessions(task_id, "error", db)
                await block_stalled_task(
                    task_id,
                    "proxy_error",
                    "The task's proxy session crashed with an error — the task is blocked. "
                    "Press Retry to restart it.",
                    db,
                )
                await db.commit()
            except Exception:
                logger.exception(
                    "proxy_session: post-crash cleanup failed for task %s", task_id
                )
                await db.rollback()
        finally:
            # A completed (or failed) proxy conversation is the exact point
            # from which the workspace idle window should be measured.  The
            # reconciler refreshes lease heartbeats, but must not move this
            # activity timestamp on every liveness probe.
            try:
                from app.models.workspace import Workspace

                result = await db.execute(
                    select(Workspace).where(Workspace.room_id == room_id)
                )
                workspace = result.scalar_one_or_none()
                if workspace is not None:
                    finished_at = datetime.now(timezone.utc)
                    workspace.last_activity_at = finished_at
                    workspace.last_heartbeat_at = finished_at
                    db.add(workspace)
                    await db.commit()
            except Exception:
                logger.exception(
                    "proxy_session: failed to record workspace activity for room %s",
                    room_id,
                )


def dispatch_task_event(
    task_id: UUID,
    room_id: UUID,
    goal_id: UUID | None,
    plan_id: UUID | None,
    resume: bool = False,
    rework_feedback: str | None = None,
) -> None:
    """Schedule proxy session handling as an asyncio background task.

    Must be called from within a running event loop.
    The TASK_CREATED event should be emitted by the caller before this call.
    """
    runtime_task = asyncio.create_task(
        _run_proxy_bg(task_id, room_id, goal_id, plan_id, resume=resume, rework_feedback=rework_feedback),
        name=f"proxy_session_{task_id}",
    )
    from app.core.task_runtime import register_local_proxy_task

    register_local_proxy_task(task_id, runtime_task)


async def request_task_dispatch(
    task: Task,
    room_id: UUID,
    db: AsyncSession,
    *,
    resume: bool = False,
    rework_feedback: str | None = None,
    require_ready: bool = True,
    source: str = "unknown",
) -> str:
    """Single owner for scheduling a task's proxy execution.

    Every call site that used to resolve a TaskDispatchContext and call
    dispatch_task_event directly should go through here instead (internal design notes,
    part 3, stage 2). The claim itself stays atomic inside handle_task_created
    (conditional READY→DISPATCHED update); this function adds the idempotency
    check that was missing around it: a task with dispatch_status=DISPATCHED
    or an open CmAgentProxySession is already active and must not be
    re-dispatched or torn down as a side effect of an ordinary request.

    Callers that need an explicit restart (Retry/Resume endpoints) close the
    stale sessions and force dispatch_status back to READY themselves BEFORE
    calling this — that reset is the "explicit Retry/Restart" intent from the
    plan, not something this function does on a caller's behalf.

    Raises TaskDispatchPreflightError if the task cannot be dispatched at all
    (no assigned agent, room has no ChatMaster, etc). Callers decide how to
    surface that: an HTTPException for direct user actions, or a log line for
    automatic/background triggers.
    """
    from app.models.session.cm_agent_proxy import CmAgentProxySession

    open_session = (
        await db.execute(
            select(CmAgentProxySession.id)
            .where(
                CmAgentProxySession.task_id == task.id,
                CmAgentProxySession.closed_at.is_(None),
            )
            .limit(1)
        )
    ).first()
    is_active = task.dispatch_status == DispatchStatus.DISPATCHED or open_session is not None
    if is_active:
        logger.info(
            "proxy_session: dispatch request for task %s ignored — already active "
            "(dispatch_status=%s, has_open_session=%s, source=%s)",
            task.id, task.dispatch_status.value, open_session is not None, source,
        )
        return "already_active"

    dispatch = await prepare_task_dispatch(task, db, require_ready=require_ready)
    dispatch_task_event(
        task_id=task.id,
        room_id=room_id,
        goal_id=dispatch.goal_id,
        plan_id=dispatch.plan_id,
        resume=resume,
        rework_feedback=rework_feedback,
    )
    logger.info(
        "proxy_session: dispatch requested for task %s (source=%s, resume=%s)",
        task.id, source, resume,
    )
    return "started"
