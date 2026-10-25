"""
Step 8: PLAN_CREATED / PLAN_UPDATED → CmSelfPlanSession.

Public API:
    render_plan_text       — render GoalPlan steps as plain text for the LLM message
    render_plan_text_with_agents — render_plan_text + resolve assignee @nicknames
    handle_plan_created    — core handler: create session → LLM → tasks → close
    dispatch_plan_event    — schedule handle_plan_created as an asyncio background task

Flow triggered by PLAN_CREATED / PLAN_UPDATED room event:
  1. Find ChatMaster agent from room.coordinator_config["master_agent_id"].
  2. Create CmSelfPlanSession + emit SESSION_STARTED.
  3. Build minimal memory layer (room_memory, agents_list, repo_snapshot) + system prompt.
  4. check_token_budget on plan_text.
  5. Single LLM call: plan_text as user message, close_session tool mandatory.
  6. track_llm_usage.
  7. LLM calls close_session → apply_close_actions (creates tasks) →
     emit TASK_CREATED per task → mark session closed → emit SESSION_CLOSED.
"""
from __future__ import annotations


import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import httpx
from openai import AsyncOpenAI, BadRequestError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from app.core.context_assembler import (
    ContextAssembler,
    log_preflight_observation,
    persist_context_assembly,
)
from app.core.context_policy import resolve_context_policy, uses_cumulative_limit
from app.core.events import emit_event
from app.core.memory_layer import build_memory_layer, build_system_prompt
from app.core.step_profile import SIDE_PLAN, timed_llm
from app.db.release import release_db
from app.core.chatgpt_responses import llm_client
from app.core.llm_providers import (
    messages_headers,
    messages_url,
    openai_base_url,
    openai_client_extras,
    uses_anthropic_env_key,
    uses_messages_wire,
)
from app.core.reasoning import apply_reasoning_body, apply_reasoning_chat
from app.core.secrets.crypto import SecretValue
from app.core.secrets.runtime import legacy_secret_value, resolve_agent_secret
from app.core.session_close import (
    ANTHROPIC_CLOSE_TOOL,
    OPENAI_CLOSE_TOOL,
    SessionCloseActions,
    ToolArgumentsParseError,
    apply_close_actions,
    extract_close_actions_from_tool_use,
    parse_tool_arguments,
)
from app.core.room_coordinator import get_cm_agent_id, resolve_planner_llm_config
from app.core.token_budget import check_token_budget, log_llm_call, track_llm_usage
from app.models.agent import Agent
from app.models.goal_plan import GoalPlan, GoalPlanStep
from app.models.room import Room
from app.models.session.base import Session
from app.repositories.session import SessionRepository
from app.schemas.session import CmSelfPlanSessionCreate

logger = logging.getLogger(__name__)

# Output budget for the single decomposition turn. Planning is deliberately a
# reasoning task (a thinking ChatMaster keeps its budget here), so the cap has
# to cover both the thinking block and a multi-task structured close.
_PLAN_MAX_TOKENS = 16384

# LLM attempts before the deterministic plan-step fallback: first shot, then up
# to two nudged retries (malformed tool JSON, prose instead of a tool call, or a
# close_session whose tasks_create is empty — weak models describe the tasks in
# `summary` and leave the array blank).
_MAX_LLM_ATTEMPTS = 3

_PROSE_NUDGE = (
    "\n\n---\n"
    "You replied with plain text. That is not allowed at this step: immediately "
    "call the close_session tool and pass the breakdown in the "
    "tasks_create field (one task per logical step, with title and description). "
    "Do not write prose, do not ask questions."
)

_EMPTY_TASKS_NUDGE = (
    "\n\n---\n"
    "You called close_session, but the tasks_create field is EMPTY. A description of tasks in "
    "summary does not count — each task MUST be a separate object in the "
    "tasks_create array with title and description fields (and, where appropriate, "
    "assigned_agent_id, ref, depends_on). Call close_session again with "
    "tasks_create filled in."
)

_JSON_MODE_INSTRUCTION = (
    "\n\n---\n"
    "OUTPUT ONLY a valid JSON object (no markdown, no text around it) of the form:\n"
    '{"tasks_create": [{"ref": "build-x", "title": "...", "description": "...", '
    '"assigned_agent_id": "<uuid from Agents List>", "depends_on": ["ref", ...]}], '
    '"summary": "one line"}\n'
    "Each task is a separate object of the tasks_create array. assigned_agent_id is "
    "a UUID from the agents list above (NOT a nickname). depends_on refers to the ref "
    "of another task from the same array. Write nothing except JSON."
)

# Canonical + alias keys a model might use for the task list (see
# session_close._FIELD_ALIASES).
_TASK_LIST_KEYS = ("tasks_create", "tasks", "new_tasks", "create_tasks")

# internal design notes Stage 4: the task-formation dialogue digest is
# prepended to the decomposition turn so the split is driven by the user's own
# agreed requirements and the real attachment paths, not by the lossy plan
# wording. Best-effort — no archive means the block is simply absent.
_DIALOGUE_DIGEST_BLOCK = (
    "## Context: spec-agreement dialogue archive\n\n"
    "Below are the digest of the agreed requirements, the list of attachments (these are real "
    "files in the repository under `.legion/plans/…-attachments/`) and the reference links "
    "from the user's original dialogue with the coordinator. This is the primary source — "
    "more precise than the plan's wording.\n\n"
    "When breaking down into tasks:\n"
    "- carry the requirements relevant to a task from the digest into its `description` "
    "verbatim, together with the source turn number `[#N]`;\n"
    "- if a task relies on an attachment — **quote the exact file path** "
    "(for example `.legion/plans/goal-<id>-attachments/01-nginx.conf`) right in "
    "`description`, so the executor opens exactly that file rather than reinventing it;\n"
    "- carry the reference links into the descriptions of the tasks they relate to.\n\n"
    "{digest}\n\n---\n\n"
)


async def _augment_plan_text_with_dialogue(
    plan_text: str, room_id: UUID, goal_id: UUID
) -> str:
    """Prepend the task-formation dialogue digest to the decomposition turn.

    Best-effort: any failure (no archive, workspace unreachable) leaves the
    plan text untouched.
    """
    try:
        from app.core.plan_dialogue import digest_for_decomposition

        digest = await digest_for_decomposition(room_id, goal_id=goal_id)
    except Exception:
        logger.warning(
            "plan_session: dialogue digest injection failed for goal %s",
            goal_id,
            exc_info=True,
        )
        return plan_text
    if not digest:
        return plan_text
    return _DIALOGUE_DIGEST_BLOCK.format(digest=digest.strip()) + plan_text


def _tool_input_task_count(tool_input: dict[str, Any] | None) -> int:
    """How many task entries a raw close_session payload carries (any alias)."""
    if not isinstance(tool_input, dict):
        return 0
    for key in _TASK_LIST_KEYS:
        val = tool_input.get(key)
        if isinstance(val, list) and val:
            return len(val)
    return 0

from app.core.model_pricing import compute_cost as _compute_cost


# ---------------------------------------------------------------------------
# Plan rendering helper
# ---------------------------------------------------------------------------


def render_plan_text(
    plan: GoalPlan,
    steps: list[GoalPlanStep],
    *,
    agent_names: dict[UUID, str] | None = None,
) -> str:
    """Render a GoalPlan and its steps as a plain-text string.

    Used as the first (user) message in the cm_self_plan LLM turn. When
    *agent_names* maps a step's ``assigned_agent_id`` to a nickname, the step is
    annotated ``— agent: @nickname`` (clearer for the decomposition model than a
    bare UUID); otherwise the raw id is shown.
    """
    agent_names = agent_names or {}
    lines = [f"# Plan (id: {plan.id})"]
    if not steps:
        lines.append("*(no steps defined)*")
        return "\n".join(lines)

    for i, step in enumerate(steps, 1):
        agent_note = ""
        if step.assigned_agent_id:
            nick = agent_names.get(step.assigned_agent_id)
            agent_note = (
                f" — agent: @{nick}" if nick else f" — agent: {step.assigned_agent_id}"
            )
        lines.append(f"\n{i}. **{step.title}**{agent_note}")
        if step.description:
            lines.append(f"   {step.description}")
    return "\n".join(lines)


async def render_plan_text_with_agents(
    plan: GoalPlan, steps: list[GoalPlanStep], db: AsyncSession
) -> str:
    """``render_plan_text`` with each assigned step annotated by @nickname.

    Looks the nicknames up in one query; falls back to the bare id per step
    when an agent row is missing. Appends the goal's requirements digest
    (internal design notes Stage 4), if any were collected, so
    cm_self_plan's decomposition turn can set each task's ``covers``.
    """
    ids = {s.assigned_agent_id for s in steps if s.assigned_agent_id}
    names: dict[UUID, str] = {}
    if ids:
        rows = (
            await db.execute(select(Agent.id, Agent.nickname).where(Agent.id.in_(ids)))
        ).all()
        names = {aid: nick for aid, nick in rows}
    text = render_plan_text(plan, steps, agent_names=names)

    from app.core.requirements import load_goal_requirements, render_requirements_digest

    req_rows = await load_goal_requirements(plan.goal_id, db)
    digest = render_requirements_digest(req_rows)
    if digest:
        text = f"{text}\n\n{digest}"
    return text


# ---------------------------------------------------------------------------
# LLM call (single turn, close_session tool)
# ---------------------------------------------------------------------------


async def _get_agent_llm_config(agent_id: UUID, db: AsyncSession) -> dict[str, Any]:
    result = await db.execute(select(Agent).where(Agent.id == agent_id))
    agent = result.scalars().first()
    return (agent.llm_config or {}) if agent else {}


async def _call_llm_single_turn(
    llm_config: dict[str, Any],
    system_prompt: str,
    user_message: str,
    *,
    secret_value: SecretValue | None = None,
) -> tuple[str | None, dict[str, Any] | None, int, int, str]:
    """Call the LLM once and return (text, tool_input, tokens_in, tokens_out, model).

    Exactly one of text / tool_input is non-None.
    """
    provider = llm_config.get("provider", "anthropic").lower()
    # No silent cheap fallback (§8.1): handle_plan_created guarantees a model is
    # set before we get here — a missing one is PLANNER_MODEL_MISSING, not Haiku.
    model = llm_config.get("model")
    if not model:
        raise ValueError("plan_session: planner llm_config has no model")
    api_key = secret_value.reveal() if secret_value is not None else None
    if not api_key:
        legacy = legacy_secret_value(llm_config, "api_key")
        try:
            api_key = legacy.reveal() if legacy is not None else None
        finally:
            legacy = None
    base_url = llm_config.get("base_url")

    history = [{"role": "user", "content": user_message}]

    if uses_messages_wire(provider, model):
        if uses_anthropic_env_key(provider):
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        body: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": history,
            "tools": [ANTHROPIC_CLOSE_TOOL],
            # Decomposing a plan into several tasks with full acceptance
            # criteria is real reasoning work; 4096 left a thinking model with
            # no room to both deliberate and emit the structured close, so it
            # hit the cap mid-thought and returned no tool call at all.
            "max_tokens": _PLAN_MAX_TOKENS,
        }
        apply_reasoning_body(body, llm_config, provider, model)
        # Force close_session so the planner cannot answer in prose — but
        # Anthropic rejects a forced tool_choice together with extended
        # thinking, and planning wants thinking ON. Only force when this call
        # is not sending an enabled thinking block; the endpoint-rejects-it
        # retry below is the second safety net.
        _thinking_on = body.get("thinking", {}).get("type") == "enabled"
        if not _thinking_on:
            body["tool_choice"] = {"type": "tool", "name": "close_session"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = messages_headers(provider, api_key)
            url = messages_url(provider, base_url)
            response = await timed_llm(
                client.post(url, headers=headers, json=body),
                provider=provider,
                side=SIDE_PLAN,
            )
            if (
                response.status_code == 400
                and "tool_choice" in response.text
                and "tool_choice" in body
            ):
                logger.warning(
                    "plan_session: endpoint rejected tool_choice %s — retrying "
                    "with tools but no tool_choice",
                    body.get("tool_choice"),
                )
                body.pop("tool_choice", None)
                response = await timed_llm(
                    client.post(url, headers=headers, json=body),
                    provider=provider,
                    side=SIDE_PLAN,
                )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
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
    messages = [{"role": "system", "content": system_prompt}] + history
    create_extra: dict[str, Any] = {}
    apply_reasoning_chat(create_extra, llm_config, provider, model)
    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": [OPENAI_CLOSE_TOOL],
        # Force the decomposition into the structured close, same reason as the
        # messages wire above. Some reasoning models on OpenRouter answer 400
        # to a named tool_choice — fall back to "auto" rather than fail.
        "tool_choice": {"type": "function", "function": {"name": "close_session"}},
        **create_extra,
    }
    try:
        resp = await timed_llm(
            oai.chat.completions.create(**create_kwargs),
            provider=provider,
            side=SIDE_PLAN,
        )
    except BadRequestError as exc:
        if "tool_choice" not in str(exc):
            raise
        logger.warning(
            "plan_session: endpoint rejected tool_choice %r (%s) — retrying with "
            "tool_choice='auto'",
            create_kwargs.get("tool_choice"), exc,
        )
        create_kwargs["tool_choice"] = "auto"
        resp = await timed_llm(
            oai.chat.completions.create(**create_kwargs),
            provider=provider,
            side=SIDE_PLAN,
        )
    choice = resp.choices[0]
    usage = resp.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0

    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        tool_call = choice.message.tool_calls[0]
        tool_input = parse_tool_arguments(tool_call.function.arguments)
        return None, tool_input, tokens_in, tokens_out, model

    text = choice.message.content or ""
    return text, None, tokens_in, tokens_out, model


def _parse_json_actions(raw: str) -> dict[str, Any] | None:
    """Tolerant parse of a JSON-mode decomposition reply (fences, prose around)."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-z]*\n?", "", s).rstrip("`").strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


async def _call_llm_json_mode(
    llm_config: dict[str, Any],
    system_prompt: str,
    user_message: str,
    *,
    secret_value: SecretValue | None = None,
) -> tuple[str, int, int]:
    """One decomposition call in JSON-object mode — no tools.

    The escape hatch for models that return an empty forced tool call but a
    clean structured object under ``response_format={"type": "json_object"}``.
    Returns ``(raw_text, tokens_in, tokens_out)``.
    """
    provider = llm_config.get("provider", "anthropic").lower()
    model = llm_config.get("model")
    if not model:
        raise ValueError("plan_session: planner llm_config has no model")
    api_key = secret_value.reveal() if secret_value is not None else None
    if not api_key:
        legacy = legacy_secret_value(llm_config, "api_key")
        try:
            api_key = legacy.reveal() if legacy is not None else None
        finally:
            legacy = None
    base_url = llm_config.get("base_url")
    history = [{"role": "user", "content": user_message}]

    if uses_messages_wire(provider, model):
        if uses_anthropic_env_key(provider):
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        body: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "messages": history,
            "max_tokens": _PLAN_MAX_TOKENS,
        }
        apply_reasoning_body(body, llm_config, provider, model)
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await timed_llm(
                client.post(
                    messages_url(provider, base_url),
                    headers=messages_headers(provider, api_key),
                    json=body,
                ),
                provider=provider,
                side=SIDE_PLAN,
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            text = next(
                (b["text"] for b in data.get("content", []) if b.get("type") == "text"),
                "",
            )
            return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    api_key = api_key or os.getenv("OPENAI_API_KEY") or "sk-no-key-required"
    oai = llm_client(provider, chat_client_factory=AsyncOpenAI,
        api_key=api_key, base_url=openai_base_url(provider, base_url), timeout=120.0,
        **openai_client_extras(openai_base_url(provider, base_url)),
    )
    messages = [{"role": "system", "content": system_prompt}] + history
    create_extra: dict[str, Any] = {}
    apply_reasoning_chat(create_extra, llm_config, provider, model)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        **create_extra,
    }
    try:
        resp = await timed_llm(
            oai.chat.completions.create(**kwargs),
            provider=provider,
            side=SIDE_PLAN,
        )
    except BadRequestError as exc:
        if "response_format" not in str(exc):
            raise
        kwargs.pop("response_format", None)
        resp = await timed_llm(
            oai.chat.completions.create(**kwargs),
            provider=provider,
            side=SIDE_PLAN,
        )
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


async def _preflight_plan_request(
    *,
    session: Session,
    llm_config: dict[str, Any],
    system_prompt: str,
    plan_text: str,
    agent_id: UUID,
    db: AsyncSession,
) -> str | None:
    """Admit the planner's actual single provider request before sending it."""
    policy = await resolve_context_policy(
        role="chat_master",
        db=db,
        room_id=session.room_id,
        agent_id=agent_id,
        llm_config=llm_config,
    )
    provider = (llm_config.get("provider") or "anthropic").lower()
    tools = [
        ANTHROPIC_CLOSE_TOOL
        if uses_messages_wire(provider, llm_config.get("model"))
        else OPENAI_CLOSE_TOOL
    ]
    assembly = ContextAssembler(policy).build(
        system_prompt=system_prompt,
        tool_schemas=tools,
        messages=[{"role": "user", "content": plan_text}],
        session_remaining_tokens=(
            max(0, policy.context_window_limit - session.tokens_in - session.tokens_out)
            if uses_cumulative_limit()
            else None
        ),
    )
    try:
        from app.core.metrics import record_context_assembly

        record_context_assembly(assembly)
    except Exception:
        logger.debug("plan_session: context assembly metric failed", exc_info=True)
    await persist_context_assembly(
        assembly,
        session_id=session.id,
        db=db,
        role="chat_master",
        request_kind="cm_self_plan",
    )
    log_preflight_observation(
        assembly,
        session_id=session.id,
        role="chat_master",
        request_kind="cm_self_plan",
        policy=policy,
        session_remaining_tokens=(
            max(0, policy.context_window_limit - session.tokens_in - session.tokens_out)
            if uses_cumulative_limit()
            else None
        ),
        cumulative_tokens=session.tokens_in + session.tokens_out,
    )
    return assembly.rejected_reason


async def _close_for_context_preflight(
    session: Session,
    *,
    rejected_reason: str,
    db: AsyncSession,
) -> None:
    """Close a planner session when its minimum valid turn exceeds a cap."""
    session.closed_at = datetime.now(timezone.utc)
    session.close_reason = "context_limit"
    db.add(session)
    await db.flush()
    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_CLOSED",
        content=(
            f"Session {session.id} closed (kind=cm_self_plan, "
            f"reason=context_limit, preflight={rejected_reason})"
        ),
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": "cm_self_plan",
            "close_reason": "context_limit",
            "preflight_reason": rejected_reason,
        },
    )


# ---------------------------------------------------------------------------
# Close flow
# ---------------------------------------------------------------------------


async def _run_close_flow(
    session: Session,
    tool_input: dict[str, Any],
    db: AsyncSession,
) -> None:
    """Parse close actions, apply them (incl. TASK_CREATED + dispatch), close session."""
    try:
        actions = extract_close_actions_from_tool_use(tool_input)
    except Exception as exc:
        logger.error(
            "plan_session: invalid close_session input for session %s: %s\ntool_input: %s",
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

    # apply_close_actions flushes task inserts but does not commit
    await apply_close_actions(session, actions, db)

    # Mark session closed (flushed into the same uncommitted transaction)
    session.closed_at = datetime.now(timezone.utc)
    session.close_reason = "task_done"
    db.add(session)
    await db.flush()

    # TASK_CREATED events and proxy dispatch are handled inside apply_close_actions —
    # re-emitting/re-dispatching here duplicated every task (and double-ran ready ones).
    tasks_created = len(actions.tasks_create)

    await emit_event(
        room_id=session.room_id,
        event_type="SESSION_CLOSED",
        content=(
            f"Session {session.id} closed "
            f"(kind=cm_self_plan, reason=task_done, tasks_created={tasks_created})"
        ),
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": "cm_self_plan",
            "close_reason": "task_done",
            "tasks_created": tasks_created,
            "summary": actions.summary,
        },
    )

    # internal design notes Stage 6: the soft coverage gate, run
    # right after decomposition — covers both this path and the
    # _decompose_from_plan_steps fallback, since that one also ends by
    # calling this function. Never blocks: emit_requirements_gap only raises
    # a REQUIREMENTS_GAP card when there's actually something uncovered.
    goal_id = getattr(session, "goal_id", None)
    if goal_id is not None:
        from app.core.requirements import emit_requirements_gap
        from app.models.goal import RoomGoal

        goal_row = (
            await db.execute(select(RoomGoal).where(RoomGoal.id == goal_id))
        ).scalars().first()
        if goal_row is not None:
            await emit_requirements_gap(session.room_id, goal_row.id, goal_row.title, db)


async def _decompose_from_plan_steps(
    session: Session,
    plan_id: UUID,
    llm_text: str | None,
    db: AsyncSession,
    *,
    model_summary: str | None = None,
) -> None:
    """Fallback: the planner produced no usable close_session (prose, or an
    empty ``tasks_create``).

    Build one task per plan step so the goal is not stranded with zero tasks,
    run the normal close flow (TASK_CREATED + dispatch), and emit a
    ``PLAN_DECOMPOSITION_INCOMPLETE`` warning so the operator knows the
    automatic decomposition did not run. ``model_summary`` (the model's own
    prose breakdown, when it gave one) is folded into the single-step task so
    the executing side still sees the intended split.
    """
    steps_result = await db.execute(
        select(GoalPlanStep)
        .where(GoalPlanStep.plan_id == plan_id)
        .order_by(GoalPlanStep.position.asc(), GoalPlanStep.created_at.asc())
    )
    steps = list(steps_result.scalars().all())
    _extra = (
        f"\n\n---\nAutomatic decomposition did not work. The breakdown "
        f"proposed by ChatMaster:\n{model_summary.strip()}"
        if model_summary and model_summary.strip()
        else ""
    )
    tasks: list[dict[str, Any]] = []
    for s in steps:
        if not (s.title or "").strip():
            continue
        entry: dict[str, Any] = {
            "title": s.title,
            "description": (s.description or "") + _extra,
        }
        if getattr(s, "assigned_agent_id", None):
            entry["assigned_agent_id"] = str(s.assigned_agent_id)
        tasks.append(entry)

    if tasks:
        salvaged = {
            "tasks_create": tasks,
            "summary": (
                "Fallback decomposition: ChatMaster did not call close_session; "
                f"{len(tasks)} task(s) created directly from the plan steps. "
                "Review the split and the assignees."
            ),
        }
        await _run_close_flow(session, salvaged, db)
    else:
        session.closed_at = datetime.now(timezone.utc)
        session.close_reason = "task_done"
        db.add(session)
        await db.flush()
        await emit_event(
            room_id=session.room_id,
            event_type="SESSION_CLOSED",
            content=(
                f"Session {session.id} closed (kind=cm_self_plan, reason=task_done, "
                "no tasks created — LLM did not call close_session and the plan "
                "has no steps to fall back on)"
            ),
            db=db,
            metadata={
                "session_id": str(session.id),
                "kind": "cm_self_plan",
                "close_reason": "task_done",
                "tasks_created": 0,
            },
        )

    await emit_event(
        room_id=session.room_id,
        event_type="PLAN_DECOMPOSITION_INCOMPLETE",
        content=(
            "Could not automatically break the plan into tasks "
            f"(created directly from the plan steps: {len(tasks)}). "
            "Check the goal's tasks and refine the plan if necessary."
        ),
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": "cm_self_plan",
            "level": "warning",
            "plan_id": str(plan_id),
            "tasks_created": len(tasks),
            "fallback": "plan_steps",
            "llm_text_snippet": (llm_text or "")[:200],
        },
    )


async def _touch_activity(session: Session, db: AsyncSession) -> None:
    """Stamp ``last_activity_at`` after an LLM step and make it visible.

    The watchdog's stale-plan sweep reads this column from its own connection,
    so the stamp is committed, not just flushed — like emit_event does for
    every event of this turn.
    """
    session.last_activity_at = datetime.now(timezone.utc)
    db.add(session)
    await db.commit()


async def _close_after_failure(
    session: Session,
    db: AsyncSession,
    *,
    reason: str,
    detail: str,
) -> None:
    """Close a decomposition session whose turn died mid-way.

    The transaction may be aborted (DB error) or hold a half-applied close
    flow, so it is rolled back first and the close is a conditional UPDATE —
    a session that did reach its own close before the failure keeps its
    reason. Never raises: the caller re-raises the original error.
    """
    try:
        await db.rollback()
        now = datetime.now(timezone.utc)
        result = await db.execute(
            update(Session)
            .where(Session.id == session.id, Session.closed_at.is_(None))
            .values(closed_at=now, close_reason=reason, last_activity_at=now)
            .returning(Session.id)
        )
        if result.first() is None:
            await db.commit()
            return
        session.closed_at = now
        session.close_reason = reason
        await emit_event(
            room_id=session.room_id,
            event_type="SESSION_CLOSED",
            content=(
                f"Session {session.id} closed (kind=cm_self_plan, "
                f"reason={reason}): {detail[:300]}"
            ),
            db=db,
            metadata={
                "session_id": str(session.id),
                "kind": "cm_self_plan",
                "close_reason": reason,
            },
        )
    except BaseException:
        logger.exception(
            "plan_session: could not close session %s after failure (%s) — "
            "the watchdog stale-plan sweep will close it",
            session.id, reason,
        )


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


async def handle_plan_created(
    room_id: UUID,
    goal_id: UUID,
    plan_id: UUID,
    plan_text: str,
    db: AsyncSession,
) -> None:
    """React to PLAN_CREATED / PLAN_UPDATED by decomposing the plan into tasks.

    Silently skips if the room has no ChatMaster (master_agent_id not set).
    """
    # Resolve ChatMaster agent from room coordinator config
    room_result = await db.execute(select(Room).where(Room.id == room_id))
    room = room_result.scalars().first()
    if room is None:
        logger.error("plan_session: room %s not found — aborting", room_id)
        return

    cm_agent_id = get_cm_agent_id(room)
    if cm_agent_id is None:
        logger.info(
            "plan_session: room %s has no (valid) ChatMaster in coordinator_config "
            "— skipping",
            room_id,
        )
        return

    # Create CmSelfPlanSession
    repo = SessionRepository(db)
    session = await repo.create(
        CmSelfPlanSessionCreate(
            kind="cm_self_plan",
            room_id=room_id,
            chat_master_agent_id=cm_agent_id,
            goal_id=goal_id,
            plan_id=plan_id,
        )
    )

    # Stamped from the start: the watchdog's stale-plan sweep measures idle
    # time from here, then from every LLM step (_touch_activity).
    session.last_activity_at = datetime.now(timezone.utc)
    db.add(session)

    try:
        await _run_plan_turn(
            session, room, cm_agent_id, room_id, goal_id, plan_id, plan_text, db
        )
    except BaseException as exc:
        # Every exit must close the row: an exception here (DB error in the
        # close flow, a bug in a helper) or a cancelled task used to roll the
        # uncommitted close back and leave the session open forever.
        reason = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        await _close_after_failure(session, db, reason=reason, detail=repr(exc))
        raise
    if session.closed_at is None:
        # Defensive: an early return that forgot to close.
        logger.error(
            "plan_session: session %s returned without closing — closing as error",
            session.id,
        )
        await _close_after_failure(
            session, db, reason="error", detail="decomposition returned without closing"
        )


async def _run_plan_turn(
    session: Session,
    room: Room,
    cm_agent_id: UUID,
    room_id: UUID,
    goal_id: UUID,
    plan_id: UUID,
    plan_text: str,
    db: AsyncSession,
) -> None:
    """The decomposition turn proper; every path closes ``session``."""
    await emit_event(
        room_id=room_id,
        event_type="SESSION_STARTED",
        content=f"Session {session.id} started (kind=cm_self_plan)",
        db=db,
        metadata={
            "session_id": str(session.id),
            "kind": "cm_self_plan",
            "goal_id": str(goal_id),
            "plan_id": str(plan_id),
        },
    )

    # Build memory layer (minimal for cm_self_plan: room_memory, agents_list, repo_snapshot)
    memory_layer = await build_memory_layer(session, db)
    system_prompt = build_system_prompt(memory_layer, "cm_self_plan")

    # Stage 4: fold the task-formation dialogue digest into the turn before any
    # budget / preflight check so the admitted request matches what is sent.
    plan_text = await _augment_plan_text_with_dialogue(plan_text, room_id, goal_id)

    # §8.1: the decomposer runs on the ChatMaster's own model (plus an optional
    # coordinator_config["planner_model"] override) — never the executor's model
    # and never a silent cheap default.
    cm_agent, llm_config = await resolve_planner_llm_config(room, db)
    if not isinstance(cm_agent, Agent):
        cm_agent = None

    if not (llm_config or {}).get("model"):
        # No usable planner model. Zero LLM calls beats a blind decomposition on
        # a cheap default — salvage the goal deterministically from its plan
        # steps and make the misconfiguration visible in the room feed.
        cm_nick = getattr(cm_agent, "nickname", None)
        logger.warning(
            "plan_session: room %s ChatMaster (%s) has no model in llm_config — "
            "PLANNER_MODEL_MISSING, decomposing from plan steps",
            room_id, cm_nick or cm_agent_id,
        )
        await emit_event(
            room_id=room_id,
            event_type="PLANNER_MODEL_MISSING",
            content=(
                "The coordinator "
                f"{('@' + cm_nick) if cm_nick else cm_agent_id} has no model set "
                "in llm_config — the automatic plan decomposition was done "
                "directly from the plan steps, without an LLM. Set a model for ChatMaster "
                "or coordinator_config[\"planner_model\"]."
            ),
            db=db,
            metadata={
                "session_id": str(session.id),
                "kind": "cm_self_plan",
                "level": "warning",
                "chat_master_agent_id": str(cm_agent_id),
                "chat_master_nickname": cm_nick,
            },
        )
        await _decompose_from_plan_steps(session, plan_id, None, db)
        return

    # The legacy guard is an explicit rollback-only path.  In window mode,
    # counters accumulated by other turns must never prevent this plan turn.
    if uses_cumulative_limit() and not await check_token_budget(session, plan_text, db):
        return  # limit reached; session already closed + event emitted by guard

    rejected_reason = await _preflight_plan_request(
        session=session,
        llm_config=llm_config,
        system_prompt=system_prompt,
        plan_text=plan_text,
        agent_id=cm_agent_id,
        db=db,
    )
    if rejected_reason:
        if rejected_reason == "session_budget" and uses_cumulative_limit():
            # The guard above normally catches this first.  Retain an explicit
            # branch so a legacy preflight cannot be relabelled as context loss.
            await check_token_budget(session, plan_text, db)
        else:
            await _close_for_context_preflight(
                session, rejected_reason=rejected_reason, db=db
            )
        return

    # LLM turn: plan_text as user message. Three failure classes get a nudged
    # retry before we give up:
    #   * malformed tool-call JSON (ToolArgumentsParseError);
    #   * prose instead of a close_session call;
    #   * close_session called but tasks_create empty (weak models put the plan
    #     into `summary`).
    # If all attempts fail, _decompose_from_plan_steps salvages the goal from
    # the plan rather than leaving it with zero tasks.
    text: str | None = None
    tool_input: dict[str, Any] | None = None
    last_summary: str | None = None
    tokens_in = tokens_out = 0
    model = llm_config.get("model")  # guaranteed set by the §8.1 guard above
    try:
        secret_value = (
            await resolve_agent_secret(
                db=db,
                owner_id=cm_agent.owner_id,
                agent_id=cm_agent.id,
                config=llm_config,
                request_id=session.id,
            )
            if cm_agent is not None
            else None
        )
        try:
            turn_message = plan_text
            for attempt in range(_MAX_LLM_ATTEMPTS):
                try:
                    await release_db(db, reason="plan_session/decompose")
                    text, tool_input, _ti, _to, model = await _call_llm_single_turn(
                        llm_config,
                        system_prompt,
                        turn_message,
                        secret_value=secret_value,
                    )
                except ToolArgumentsParseError:
                    await _touch_activity(session, db)
                    if attempt < _MAX_LLM_ATTEMPTS - 1:
                        logger.warning(
                            "plan_session: malformed tool JSON for session %s — retrying LLM call",
                            session.id,
                        )
                        continue
                    raise
                await _touch_activity(session, db)
                tokens_in += _ti
                tokens_out += _to
                if tool_input is not None and _tool_input_task_count(tool_input) > 0:
                    break
                last = attempt >= _MAX_LLM_ATTEMPTS - 1
                if tool_input is None:
                    reason, nudge = "answered in prose", _PROSE_NUDGE
                else:
                    # close_session called but tasks_create empty — weak models
                    # put the plan into `summary` and leave the array blank.
                    reason, nudge = "left tasks_create empty", _EMPTY_TASKS_NUDGE
                    _s = tool_input.get("summary")
                    if isinstance(_s, str) and _s.strip():
                        last_summary = _s.strip()
                    tool_input = None  # do not run the close flow on an empty payload
                if last:
                    break
                logger.warning(
                    "plan_session: session %s %s instead of a filled close_session "
                    "— nudging (attempt %d/%d)",
                    session.id, reason, attempt + 1, _MAX_LLM_ATTEMPTS,
                )
                turn_message = plan_text + nudge

            # Tool-use gave nothing usable. Many open models (measured: qwen3-27b
            # on OpenRouter) return `{}` to a forced tool_choice but produce a
            # full, well-formed decomposition under response_format=json_object.
            # One JSON-mode call before the deterministic plan-step fallback.
            if tool_input is None:
                try:
                    await release_db(db, reason="plan_session/json_mode")
                    raw, _ji, _jo = await _call_llm_json_mode(
                        llm_config,
                        system_prompt,
                        plan_text + _JSON_MODE_INSTRUCTION,
                        secret_value=secret_value,
                    )
                    await _touch_activity(session, db)
                    tokens_in += _ji
                    tokens_out += _jo
                    parsed = _parse_json_actions(raw)
                    if _tool_input_task_count(parsed) > 0:
                        logger.info(
                            "plan_session: recovered %d task(s) via JSON-mode for "
                            "session %s (tool-use produced none)",
                            _tool_input_task_count(parsed), session.id,
                        )
                        tool_input = parsed
                    elif isinstance(parsed, dict) and isinstance(
                        parsed.get("summary"), str
                    ) and parsed["summary"].strip():
                        last_summary = parsed["summary"].strip()
                except Exception:
                    logger.warning(
                        "plan_session: JSON-mode fallback failed for session %s",
                        session.id, exc_info=True,
                    )
        finally:
            secret_value = None
    except Exception as exc:
        logger.error("plan_session: LLM call failed for session %s: %s", session.id, exc)
        session.closed_at = datetime.now(timezone.utc)
        session.close_reason = "error"
        db.add(session)
        await db.flush()
        await emit_event(
            room_id=room_id,
            event_type="SESSION_CLOSED",
            content=f"Session {session.id} closed (kind=cm_self_plan, reason=error): {exc}",
            db=db,
            metadata={
                "session_id": str(session.id),
                "kind": "cm_self_plan",
                "close_reason": "error",
            },
        )
        return

    # Track usage
    cost = _compute_cost(model, tokens_in, tokens_out, llm_config)
    track_llm_usage(session, tokens_in, tokens_out, cost, model=model)
    log_llm_call(
        purpose="plan_decompose",
        model=model,
        session=session,
        agent_nick=getattr(cm_agent, "nickname", None),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )
    db.add(session)
    await db.flush()

    if tool_input is not None:
        await _run_close_flow(session, tool_input, db)
    else:
        # LLM never called close_session, even after the nudge. Salvage the
        # goal from its plan steps so it is not left with zero tasks and no
        # work, and emit a visible warning so the gap is not silent.
        logger.warning(
            "plan_session: session %s produced no usable close_session after %d "
            "attempts — falling back to plan-step decomposition. text=%s summary=%s",
            session.id, _MAX_LLM_ATTEMPTS, (text or "")[:200], (last_summary or "")[:200],
        )
        await _decompose_from_plan_steps(
            session, plan_id, text or last_summary, db, model_summary=last_summary
        )


# ---------------------------------------------------------------------------
# Background dispatch
# ---------------------------------------------------------------------------


async def _run_plan_session_bg(
    room_id: UUID,
    goal_id: UUID,
    plan_id: UUID,
    plan_text: str,
) -> None:
    """Open an independent DB session and run handle_plan_created."""
    from app.db.session import engine

    async_session_factory = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session_factory() as db:
        try:
            await handle_plan_created(room_id, goal_id, plan_id, plan_text, db)
            # Anything flushed after the last emit_event (e.g. a requirements
            # gap with nothing to report) would otherwise be dropped on exit.
            await db.commit()
        except Exception:
            logger.exception(
                "plan_session: unhandled error in background handler for plan %s", plan_id
            )
            await db.rollback()


def dispatch_plan_event(
    room_id: UUID,
    goal_id: UUID,
    plan_id: UUID,
    plan_text: str,
) -> None:
    """Schedule background plan-session handling via asyncio.create_task.

    Must be called from within a running event loop (i.e. from an async FastAPI endpoint).
    The PLAN_CREATED / PLAN_UPDATED event should be emitted by the caller before this call.
    """
    asyncio.create_task(
        _run_plan_session_bg(room_id, goal_id, plan_id, plan_text),
        name=f"plan_session_{plan_id}",
    )
