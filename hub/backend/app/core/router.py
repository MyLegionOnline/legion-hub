"""
LLM-router for user messages (Step 3 / §1, §8.1).

Public API:
    SplitPart           — one entry in a multi-agent split
    RouteDecision       — classification result returned by the router
    parse_mentions      — deterministic @mention extractor (no LLM)
    route_user_message  — main entry point; call before any other routing
"""

from __future__ import annotations


import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel, field_validator
from sqlalchemy import case
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.step_profile import SIDE_ROUTER, timed_llm
from app.db.release import release_db
from app.core.chatgpt_responses import llm_client
from app.core.llm_providers import (
    messages_headers,
    messages_text_and_thinking,
    messages_url,
    openai_base_url,
    openai_client_extras,
    uses_messages_wire,
)
from app.core.reasoning import (
    apply_reasoning_body,
    apply_reasoning_chat,
    messages_reasoning_tokens,
    reasoning_from_chat_choice,
)
from app.core.events import emit_event
from app.core.llm_errors import (
    OPT_IN_REASONS,
    classify_gateway_error,
    model_opt_in_content,
)
from app.core.secrets.crypto import SecretValue
from app.core.secrets.runtime import resolve_agent_secret
from app.models.agent import Agent
from app.models.goal import GoalStatus, RoomGoal
from app.models.room import Room, RoomAgentLink, RoomType
from app.models.task import Task
from app.core.room_coordinator import get_cm_agent_id as _get_cm_agent_id
from app.core.router_agent import (
    MODE_JEV,
    RouterSettings,
    get_router_agent,
    is_router,
    router_settings,
)

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.6
_ROUTER_MODEL_ANTHROPIC = "claude-haiku-4-5-20251001"
_ROUTER_MODEL_OPENAI = "gpt-4o-mini"

# The router calls the same model as ChatMaster, and that is often a reasoning model.
# Measured glm-5.3 on the router's real system prompt: median 15-26 s,
# individual runs up to 27 s.  The old 30 s with the default max_retries=2 gave
# 93.8 s per failure (three full attempts) — the user waited a minute and a half
# to get the fallback rule.  One honest attempt with a wide window.
_ROUTER_TIMEOUT_S = 90.0


# ---------------------------------------------------------------------------
# Pydantic schemas (§8.1)
# ---------------------------------------------------------------------------


class SplitPart(BaseModel):
    agent_id: UUID
    text: str


class RouteDecision(BaseModel):
    route: Literal[
        "multi_agent_split",
        "multi_agent_ask_detail",
        "single_agent_p1",
        "single_agent_p2",
        "chatmaster_p1",
        "chatmaster_p2",
        # Stage 2: continuing the user's open session is a routing decision of
        # its own, not a way of skipping the router (defect C1).
        "continue_session",
        # waiting_reply_safeguard S2: the user is answering a task that is
        # parked waiting for them (a review, or a question the agent ended its
        # turn on). Only offered when such a task exists; carries
        # matched_task_id.
        "resume_waiting_task",
        # internal design notes Stage 4: "@agent …" to an agent busy with a
        # running task goes INTO that task (the queue of core/task_interject),
        # not into a new task or session. Only offered when the mentioned agent
        # has a running task; carries matched_task_id.
        "interject_task",
    ]
    mentioned_agent_ids: list[UUID] = []
    can_split: bool | None = None
    split_messages: list[SplitPart] | None = None
    matched_task_id: UUID | None = None
    matched_goal_id: UUID | None = None
    continue_session_id: UUID | None = None
    # internal design notes Stage 3: an optional LLM hint about how risky the
    # work looks. Only ever *raises* the deterministic risk class computed from
    # the diff (see room_git_flow.classify_task_risk) — it can never lower it.
    risk_hint: Literal["trivial", "normal", "risky"] | None = None
    confidence: float
    reasoning: str

    @field_validator("mentioned_agent_ids", mode="before")
    @classmethod
    def _null_mentions_mean_empty(cls, v: Any) -> Any:
        """Accept `null` where the schema asks for a list of mentions.

        Every other optional field in the system prompt is spelled
        ``... | null``, so a router model with nothing to report writes
        ``"mentioned_agent_ids": null`` — valid JSON that used to fail
        validation and drop the whole reply into the fallback route
        (observed 2026-09-04 on deepseek-v4-pro).  The list is overwritten
        from the deterministic @-parse right after the call anyway, so a null
        here never carried information worth losing the decision over.
        """
        return [] if v is None else v


class OpenSessionContext(BaseModel):
    """The user's currently open session, as the router sees it (§2.2).

    Only built for a session that is still young — see
    ``user_session_limits.is_continuable``.  Carries a summary and an excerpt,
    never the transcript: the whole point of stage 2 is that a transcript stops
    being the way context travels between turns.
    """

    session_id: UUID
    kind: str
    turns: int = 0
    summary_so_far: str = ""
    last_assistant_excerpt: str = ""
    goal_id: UUID | None = None
    task_id: UUID | None = None
    # waiting_reply_safeguard S3 / bug-fix 2026-09-07: the last assistant turn
    # put a question to the user (or an ask_user / a parked dialogue task is
    # pending). A mention-free reply here is almost always the answer, not a new
    # topic — surfaced to the model in the prompt and enforced post-decision.
    has_open_question: bool = False
    # Who the conversation is with (router_agent_plan §1, finding 1): Jev needs
    # it to tell a reply inside a dialogue with @X from a new task for @X.
    peer_agent_id: UUID | None = None
    peer_is_chatmaster: bool = False


class _GatheredContext:
    """Four-value compatibility view plus the internal router agent row."""

    def __init__(
        self, agents_ctx, goals_ctx, mentioned_ids, router_llm_config, router_agent,
        waiting_ctx=None, room_agents=None,
    ):
        self.agents_ctx = agents_ctx
        self.goals_ctx = goals_ctx
        self.mentioned_ids = mentioned_ids
        self.router_llm_config = router_llm_config
        self.router_agent = router_agent
        # waiting_reply_safeguard S2: tasks parked waiting for the user, if any.
        self.waiting_ctx = waiting_ctx or []
        # The room's agents — route_user_message resolves the Router from them.
        self.room_agents = room_agents or []

    def __iter__(self):
        yield self.agents_ctx
        yield self.goals_ctx
        yield self.mentioned_ids
        yield self.router_llm_config


# ---------------------------------------------------------------------------
# Router prompt template (§8.1)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the routing module of a chat room. You classify the user's message —
you never answer it. The message may be in any language. You are given:
- the user's message
- the @-mentioned agents (id + short description)
- for each mentioned agent, their 10 most recent tasks (id, title, state, short description)
- the room's **active** Goals (id, title) — only goals that are not finished yet

Reply with STRICT JSON only — no explanations, no code fences:
{
  "route": "multi_agent_split" | "multi_agent_ask_detail"
         | "single_agent_p1"  | "single_agent_p2"
         | "chatmaster_p1"    | "chatmaster_p2",
  "mentioned_agent_ids": [uuid, ...],   // always an array; [] when nobody is mentioned
  "can_split": bool | null,
  "split_messages": [{"agent_id": uuid, "text": "..."}] | null,
  "matched_task_id": uuid | null,
  "matched_goal_id": uuid | null,
  "risk_hint": "trivial" | "normal" | "risky" | null,
  "confidence": 0.0..1.0,
  "reasoning": "1-2 short sentences for the log"
}

risk_hint (optional): how risky the requested work looks from the text —
"risky" if it touches DB migrations, dependencies, CI/deploy, schemas, payments
or security; "trivial" for a small text/style edit; otherwise null.
The hint can only raise the final risk class, never lower it.

Route selection rules:
- ≥2 @mentions → multi_agent_split (if the request can be divided between the
  agents; write each agent's part in split_messages, in the user's language)
  or multi_agent_ask_detail (if it cannot)
- exactly 1 @mention, the text semantically matches one of that agent's 10
  recent tasks → single_agent_p1 + matched_task_id
- exactly 1 @mention, none of the 10 tasks matches → single_agent_p2
- 0 @mentions, the text relates to one of the active Goals → chatmaster_p1
  + matched_goal_id
- 0 @mentions, no active Goal matches (or the list is empty) → chatmaster_p2
  (the Goals list holds active goals only; finished goals are never in it —
   text about finished work or a new topic always → chatmaster_p2)

If confidence < 0.6, prefer the more general variant of your branch (p2).\
"""

# Appended to the system prompt only when the user has a young open session.
# Without it the model must not know that "continue_session" exists — a route it
# cannot fulfil (there is no session to continue) is worse than no route at all.
_CONTINUE_SESSION_RULES = """

ADDITIONALLY: the user has an OPEN session (its short state is given in the
message). One more route is available to you:
  "continue_session" — continue this open session without creating anything.

Rules:
- a short confirmation ("ok", "yes", "go ahead", "thanks" in any language),
  a clarification of the same question, an answer to a
  question the assistant asked, a follow-up on the same topic
  → "continue_session";
- a new topic, a new goal, "now let's talk about something else", a request
  unrelated to the current dialogue, an @mention of another agent → a regular
  route. The system will close the open session properly and carry its
  outcome into the new one.
- Torn between continuing and a new topic while the task is clearly new —
  choose the regular route: the new session gets the project context from
  memory and the Board.
With "continue_session" the fields matched_task_id / matched_goal_id /
split_messages are not needed."""

# Appended only when the room has a task parked waiting for the user
# (waiting_reply_safeguard S2). Same rule as continue_session: a route the
# system cannot fulfil is worse than none, so the model only learns about
# "resume_waiting_task" when there is a task for it to resume.
_WAITING_TASK_RULES = """

ADDITIONALLY: the room has task(s) waiting for the user's reply (see the list
"Tasks waiting for the user's reply" in the message — each is either a review
or a question the agent ended its turn on). One more route is available:
  "resume_waiting_task" — the user is answering such a task; return that
  task's id as matched_task_id.

Rules:
- 0 @mentions and the text reads as a confirmation ("finish it", "ok", "yes",
  "approved"), a request to
  continue/complete the work, or a direct answer to an "agent question" from
  the list → "resume_waiting_task" + matched_task_id;
- the user clearly starts a new topic or talks about a different task →
  a regular route.
Torn between "resume_waiting_task" and "chatmaster_p2" on a short confirmation
while a waiting task exists — choose "resume_waiting_task"."""

# Appended only when the mentioned agent is executing a task right now
# (internal design notes Stage 4) — same rule: no route the system cannot serve.
_RUNNING_TASK_RULES = """

ADDITIONALLY: the mentioned agent is executing a task RIGHT NOW (marked
"RUNNING NOW" in its task list). One more route is available:
  "interject_task" — deliver the message INTO that running task: the agent
  reads it before its next step and answers without stopping; return that
  task's id as matched_task_id.

Rules:
- a status question ("status?", "are you alive?", "what are you doing?", "how
  long?", "stuck?"),
  a correction, a narrowing or widening of the scope, "stop after
  this step", extra details for the work in progress → "interject_task";
- clearly separate new work unrelated to the running task → single_agent_p2.
Torn between "interject_task" and "single_agent_p1" — choose "interject_task":
the agent can still say it is a separate task, a new branch cannot be undone."""

# ---------------------------------------------------------------------------
# Deterministic @mention parser
# ---------------------------------------------------------------------------

_MENTION_RE = re.compile(r"@([\w-]+)")


def parse_mentions(text: str) -> list[str]:
    """Return lowercased nicknames found as @mentions in text (no LLM)."""
    return [m.lower() for m in _MENTION_RE.findall(text)]


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------


def _resolve_router_agent(room: Room, room_agents: list[Agent]) -> Agent | None:
    """The room's own routing model: the ChatMaster (inherit mode).

    A group room without a ChatMaster is a misconfiguration, not a routing
    scenario: the room contract requires one at creation time (schemas/room.py).
    Returning None here makes route_user_message emit ROOM_CHATMASTER_MISSING
    and fall back deterministically, instead of silently borrowing a random
    agent's model — which is what produced alternating RegionError/timeout
    failures in room 15261a56 before this change.
    """
    if room.room_type == RoomType.direct:
        return room_agents[0] if room_agents else None

    cm_id = _get_cm_agent_id(room)
    if cm_id is None:
        return None
    return next((a for a in room_agents if a.id == cm_id), None)


@dataclass(frozen=True)
class RouterBinding:
    """Who routes a room's messages (internal design notes §3.1).

    ``engine`` is ``"llm"`` (inherit: the room's ChatMaster model) or ``"jev"``.
    ``llm_agent`` is the agent whose ``llm_config`` and secret binding power the
    LLM call — the ChatMaster in inherit mode, the escalation LLM in Jev mode
    (the Router itself when it has its own provider+model, else the
    ChatMaster).  None only for a group room without a ChatMaster whose Router
    has no model either: route_user_message reports ROOM_CHATMASTER_MISSING.
    ``router`` is the account's Router row, None when it is missing.
    """

    engine: Literal["llm", "jev"]
    llm_agent: Agent | None
    router: Agent | None = None
    settings: RouterSettings | None = None


async def _resolve_router(
    room: Room, room_agents: list[Agent], db: AsyncSession
) -> RouterBinding:
    """Resolve the room owner's Router into a :class:`RouterBinding`.

    An unconfigured Router (mode null), inherit mode, a missing Router or a
    failed lookup all route exactly as before — with the ChatMaster's model.
    """
    room_llm = _resolve_router_agent(room, room_agents)
    try:
        router = await get_router_agent(room.creator_id, db)
    except Exception as exc:  # noqa: BLE001 — routing must survive a bad lookup
        logger.debug("router: Router lookup failed for room %s: %s", room.id, exc)
        router = None
    if not is_router(router):
        return RouterBinding(engine="llm", llm_agent=room_llm)

    settings = router_settings(router)
    if settings.mode != MODE_JEV:
        return RouterBinding(
            engine="llm", llm_agent=room_llm, router=router, settings=settings
        )
    return RouterBinding(
        engine="jev",
        llm_agent=router if settings.has_escalation_model else room_llm,
        router=router,
        settings=settings,
    )


async def _gather_context(
    room: Room,
    message_text: str,
    db: AsyncSession,
) -> _GatheredContext:
    """Return context plus the agent whose exact binding powers the router.

    router_llm_config is the llm_config of the ChatMaster (in a group room) or
    the sole agent (in a direct room) — used by the router to make its
    classification call.  The router speaks the same model as the ChatMaster:
    a route is a decision about what the ChatMaster will do, and making it
    with a different model means diverging from the ChatMaster's reading of
    the context.
    """
    nicknames = parse_mentions(message_text)

    result = await db.execute(
        select(Agent)
        .join(RoomAgentLink, RoomAgentLink.agent_id == Agent.id)
        .where(RoomAgentLink.room_id == room.id)
        .order_by(Agent.id)   # determinism for direct rooms
    )
    room_agents: list[Agent] = result.scalars().all()

    router_agent = _resolve_router_agent(room, room_agents)
    router_llm_config = (router_agent.llm_config or {}) if router_agent else {}
    agent_by_nick: dict[str, Agent] = {a.nickname.lower(): a for a in room_agents}

    mentioned_agents = [agent_by_nick[n] for n in nicknames if n in agent_by_nick]
    mentioned_ids = [a.id for a in mentioned_agents]

    agents_ctx: list[dict[str, Any]] = []
    for agent in mentioned_agents:
        # Running tasks first (internal design notes Stage 4): an old task that
        # is still running must not fall out of the 10 most recent ones.
        tasks_result = await db.execute(
            select(Task)
            .where(Task.room_id == room.id, Task.assigned_agent_id == agent.id)
            .order_by(
                case((Task.runtime_state == "running", 0), else_=1),
                Task.updated_at.desc(),
            )
            .limit(10)
        )
        tasks: list[Task] = tasks_result.scalars().all()
        task_rows = [
            {
                "id": str(t.id),
                "title": t.title,
                "state": t.state.value,
                "description": (t.description or "")[:200],
                "running": _is_running(t),
            }
            for t in tasks
        ]
        agents_ctx.append(
            {
                "id": str(agent.id),
                "nickname": agent.nickname,
                "short_description": agent.short_description or "",
                "tasks": task_rows,
                "running_tasks": [
                    {"id": t["id"], "title": t["title"]} for t in task_rows if t["running"]
                ],
            }
        )

    # waiting_reply_safeguard S2: tasks parked waiting for a word from the user.
    # Best-effort — a router decision must not fail over this extra query.
    # Runs before the goals query so ``goals`` stays the last db.execute call
    # (asserted by tests that inspect the compiled SQL).
    waiting_ctx: list[dict[str, Any]] = []
    try:
        from datetime import datetime, timedelta, timezone

        from app.models.task import TaskState as _TaskState

        _cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        waiting_result = await db.execute(
            select(Task)
            .where(
                Task.room_id == room.id,
                Task.awaiting_user_since.isnot(None),
                Task.awaiting_user_since >= _cutoff,
                Task.state.in_((_TaskState.WAITING_REVIEW, _TaskState.BLOCKED)),
            )
            .order_by(Task.awaiting_user_since.desc())
            .limit(5)
        )
        _agent_nick = {a.id: a.nickname for a in room_agents}
        waiting_ctx = [
            {
                "id": str(t.id),
                "title": t.title,
                "agent": _agent_nick.get(t.assigned_agent_id),
                "kind": t.awaiting_user_kind or "review",
                "prompt": (t.awaiting_user_prompt or "")[:300],
            }
            for t in waiting_result.scalars().all()
        ]
    except Exception as exc:  # noqa: BLE001 — routing must survive a bad query
        logger.debug("router: waiting-task context unavailable: %s", exc)

    goals_result = await db.execute(
        select(RoomGoal).where(
            RoomGoal.room_id == room.id,
            RoomGoal.status == GoalStatus.ACTIVE,
        )
    )
    goals_ctx = [
        {"id": str(g.id), "title": g.title}
        for g in goals_result.scalars().all()
    ]

    return _GatheredContext(
        agents_ctx, goals_ctx, mentioned_ids, router_llm_config, router_agent,
        waiting_ctx=waiting_ctx, room_agents=room_agents,
    )


# Wide enough for a multi-question assistant turn to reach the model intact —
# a truncated "…and 3) which package manager?" read as a new feature request.
_OPEN_SESSION_EXCERPT_CHARS = 800
_OPEN_SESSION_SUMMARY_CHARS = 600


def _build_user_message(
    text: str,
    agents_ctx: list[dict[str, Any]],
    goals_ctx: list[dict[str, Any]],
    open_session: "OpenSessionContext | None" = None,
    waiting_ctx: list[dict[str, Any]] | None = None,
) -> str:
    parts: list[str] = [f"User message:\n{text}\n"]

    if waiting_ctx:
        parts.append("Tasks waiting for the user's reply:")
        for w in waiting_ctx:
            agent = f" @{w['agent']}" if w.get("agent") else ""
            parts.append(
                f"  task id={w['id']}{agent} | {w['title']} | waiting for: {w['kind']}"
            )
            if w.get("prompt"):
                parts.append(f"    agent question: {w['prompt']}")
        parts.append("")

    if open_session is not None:
        parts.append("User's open session:")
        parts.append(f"  id={open_session.session_id}  kind={open_session.kind}"
                     f"  messages={open_session.turns}")
        if open_session.goal_id:
            parts.append(f"  goal_id={open_session.goal_id}")
        if open_session.task_id:
            parts.append(f"  task_id={open_session.task_id}")
        if open_session.summary_so_far:
            parts.append(
                "  what it was about: "
                + open_session.summary_so_far[:_OPEN_SESSION_SUMMARY_CHARS]
            )
        if open_session.last_assistant_excerpt:
            parts.append(
                "  assistant's last turn: "
                + open_session.last_assistant_excerpt[:_OPEN_SESSION_EXCERPT_CHARS]
            )
        if open_session.has_open_question:
            parts.append(
                "  ⚠️ the assistant's last turn asked the user a QUESTION. "
                "This message is almost certainly the answer to it → continue_session, "
                "unless the user clearly changed the topic or set a new task."
            )
        parts.append("")

    if agents_ctx:
        parts.append("Mentioned agents:")
        for a in agents_ctx:
            parts.append(
                f"  id={a['id']}  nickname=@{a['nickname']}  desc={a['short_description']}"
            )
            for t in a["tasks"]:
                running = " | RUNNING NOW" if t.get("running") else ""
                parts.append(
                    f"    task id={t['id']} | {t['title']} | {t['state']}{running} | {t['description']}"
                )
    else:
        parts.append("Mentioned agents: none")

    parts.append("\nActive room Goals:")
    if goals_ctx:
        for g in goals_ctx:
            parts.append(f"  id={g['id']}  title={g['title']}")
    else:
        parts.append("  (none)")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call (isolated so it can be mocked in tests)
# ---------------------------------------------------------------------------


async def _call_router_llm(
    user_msg: str,
    llm_config: dict[str, Any],
    or_session_id: str | None = None,
    *,
    secret_value: SecretValue | None = None,
    system_prompt: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> tuple[str, int, int, str]:
    """
    Call the router LLM using credentials from an agent's llm_config.
    Returns (raw_text, in_tokens, out_tokens, model).

    or_session_id: if set and provider is openrouter with session_logging enabled,
    groups routing calls for a room under one OpenRouter session.

    metrics: optional out-parameter filled with ``{"reasoning_tokens": N}``.
    An out-parameter rather than a fifth tuple element on purpose — the return
    shape is mocked in half a dozen test files, and stage 5 is about
    observability, not about a signature churn those mocks would have to
    absorb.  A caller that passes nothing simply records nothing, and the
    column stays NULL, which reads as "not measured" rather than as zero.
    """
    provider = (llm_config.get("provider") or "").lower()
    system_text = system_prompt or _SYSTEM_PROMPT
    api_key = secret_value.reveal() if secret_value is not None else ""
    base_url = llm_config.get("base_url") or ""
    model = llm_config.get("model") or _ROUTER_MODEL_OPENAI

    if not api_key:
        raise RuntimeError("No configured LLM secret")

    if uses_messages_wire(provider, model):
        body: dict[str, Any] = {
            "model": model,
            "system": system_text,
            "messages": [{"role": "user", "content": user_msg}],
            "max_tokens": 512,
        }
        # Stage 5 (§6): the router classifies — "are agents mentioned, is this
        # the same topic" — and gains nothing from thinking, while losing
        # seconds on every user message.  The off-switch goes out ALWAYS, over
        # the agent's own setting.  It is also what makes `max_tokens: 512`
        # here safe: a reasoning model spends the whole cap on thought and
        # returns an empty `content`, which used to surface as a
        # JSONDecodeError fallback (§0.7).
        apply_reasoning_body(body, llm_config, provider, model, override="off")
        async with httpx.AsyncClient(timeout=_ROUTER_TIMEOUT_S) as client:
            response = await timed_llm(
                client.post(
                    messages_url(provider, base_url),
                    headers=messages_headers(provider, api_key),
                    json=body,
                ),
                provider=provider,
                side=SIDE_ROUTER,
            )
            response.raise_for_status()
            data = response.json()
            text, _thinking = messages_text_and_thinking(data)  # the router does not show its thinking
            usage = data.get("usage", {})
            if metrics is not None:
                metrics["reasoning_tokens"] = messages_reasoning_tokens(data)
            return (
                text,
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                model,
            )

    # OpenAI-compatible: openrouter, openai, groq, together, ollama, etc.
    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "timeout": _ROUTER_TIMEOUT_S,
        "max_retries": 0,
        "base_url": openai_base_url(provider, base_url),
    }
    client_kwargs.update(openai_client_extras(client_kwargs["base_url"]))

    create_extra: dict[str, Any] = {}
    if provider == "openrouter" and llm_config.get("session_logging") and or_session_id:
        create_extra["extra_body"] = {"session_id": or_session_id}
    # Same forced off-switch as the messages branch above — merged into
    # extra_body, never assigned over it, so session logging survives.
    apply_reasoning_chat(create_extra, llm_config, provider, model, override="off")

    client = llm_client(provider, chat_client_factory=AsyncOpenAI, **client_kwargs)
    resp = await timed_llm(
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            **create_extra,
        ),
        provider=provider,
        side=SIDE_ROUTER,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    if metrics is not None:
        _thinking, metrics["reasoning_tokens"] = reasoning_from_chat_choice(
            resp.choices[0].message, usage
        )
    return (
        text,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
        model,
    )


def _parse_router_reply(raw: str, reasoning_tokens: int | None = None) -> RouteDecision:
    """Turn the router model's raw reply into a decision.

    Separate from `route_user_message` so the empty-reply case can be tested
    without a room: it is not hypothetical.  Measured on OpenRouter
    (plan §4A.5 q7, and originally §0.7 on the other gateway): a reasoning
    model under a `max_tokens` cap can spend the entire cap on thought and
    return `content=""` with `finish_reason="length"`.  `json.loads("")` then
    raises "Expecting value: line 1 column 1", which sends whoever reads
    ROUTE_LLM_FAILURE hunting for a malformed payload that was never sent —
    so the empty case gets its own sentence, with the token count that
    explains it.
    """
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    if not raw:
        raise ValueError(
            "router LLM returned an empty message"
            + (
                f" after spending {reasoning_tokens} tokens on reasoning"
                if reasoning_tokens
                else ""
            )
            + " — the reply cap was consumed by thinking"
        )
    return RouteDecision(**json.loads(raw))


# ---------------------------------------------------------------------------
# Visibility: is the router still thinking? (§6.1–6.2)
# ---------------------------------------------------------------------------

# The last 20 reasoning-token readings per model, and when we last complained
# about it.  Deliberately *not* a model substitution: inheriting the
# ChatMaster's model is a user decision (0.10 p.2), so a model that refuses to
# stop thinking is reported, never replaced.
_ROUTER_REASONING_WINDOW = 20
_ROUTER_REASONING_LOG_INTERVAL_S = 3600.0
_router_reasoning_history: dict[str, list[int]] = {}
_router_reasoning_logged_at: dict[str, float] = {}


def _note_router_reasoning(model: str | None, tokens: int | None) -> None:
    """Record one reading and complain, at most hourly, if reasoning persists.

    The off-switch is a best effort, not a guarantee (§6.1): a model can ignore
    it, and on gateways that reject it the Hub does not send it at all.  Either
    way the symptom is the same — a slower router — and the operator deserves
    to be told which model is doing it rather than left to wonder.
    """
    if not model or tokens is None:
        return
    history = _router_reasoning_history.setdefault(model, [])
    history.append(int(tokens))
    if len(history) > _ROUTER_REASONING_WINDOW:
        del history[:-_ROUTER_REASONING_WINDOW]
    if len(history) < _ROUTER_REASONING_WINDOW:
        return
    ordered = sorted(history)
    median = ordered[len(ordered) // 2]
    if median <= 0:
        return
    now = time.monotonic()
    if now - _router_reasoning_logged_at.get(model, 0.0) < _ROUTER_REASONING_LOG_INTERVAL_S:
        return
    _router_reasoning_logged_at[model] = now
    logger.info(
        "router: ChatMaster model %s keeps reasoning despite the off-switch "
        "(median %d reasoning tokens over the last %d decisions) — routing is "
        "slower than it needs to be; changing the ChatMaster's model is the "
        "user's call, not ours",
        model, median, _ROUTER_REASONING_WINDOW,
    )


# ---------------------------------------------------------------------------
# Confidence-based route fallback (§8.1, §7.1)
# ---------------------------------------------------------------------------

_P2_FALLBACK: dict[str, str] = {
    "single_agent_p1": "single_agent_p2",
    "chatmaster_p1": "chatmaster_p2",
    "multi_agent_split": "multi_agent_ask_detail",
}


def _apply_confidence_fallback(decision: RouteDecision) -> RouteDecision:
    """If confidence < threshold, force to the p2-variant of the same branch."""
    if decision.confidence >= CONFIDENCE_THRESHOLD:
        return decision
    new_route = _P2_FALLBACK.get(decision.route)
    if new_route is None:
        return decision
    return decision.model_copy(
        update={
            "route": new_route,
            "matched_task_id": None,
            "matched_goal_id": None,
            "can_split": None,
            "split_messages": None,
        }
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_FALLBACK_DECISION = RouteDecision(
    route="chatmaster_p2",
    confidence=0.0,
    reasoning="fallback: router LLM call failed",
)

_SHORTCUT_MODEL = "shortcut"


def _trivial_continue(
    message_text: str, open_session: OpenSessionContext | None
) -> RouteDecision | None:
    """Deterministic ``continue_session`` for a reply not worth an LLM call (§2.3).

    This is what the old unconditional bypass was really trying to protect:
    a bare "ok" must not be classified as a new topic. Bounded to trivially
    short, mention-free replies inside a young session, everything else goes to
    the router.
    """
    if open_session is None:
        return None
    from app.core.user_session_limits import trivial_reply_chars

    text = (message_text or "").strip()
    limit = trivial_reply_chars()
    if not text or limit <= 0 or len(text) > limit or parse_mentions(text):
        return None
    return RouteDecision(
        route="continue_session",
        continue_session_id=open_session.session_id,
        confidence=1.0,
        reasoning=f"trivial reply ({len(text)} chars) in an open session — no LLM call",
    )


# ---------------------------------------------------------------------------
# interject_task (internal design notes Stage 4)
# ---------------------------------------------------------------------------


def _is_running(task: Any) -> bool:
    """Executing right now, and not parked on a question to the user (that one
    is answered through resume_waiting_task)."""
    return (
        getattr(task, "runtime_state", None) == "running"
        and not getattr(task, "awaiting_user_kind", None)
    )


# A question about the work in progress: always for the running task.
_STATUS_RE = re.compile(
    r"\b(hung|frozen|what\s+now|what\s+(are\s+you\s+)?(currently\s+)?doing|"
    r"how\s+(is\s+it\s+going|are\s+things|much\s+(more|time|is\s+left))|"
    r"are\s+you\s+(there|here)|which\s+(step|stage)|"
    r"status|alive|progress|how\s+long|what\s+are\s+you\s+doing|stuck|eta)\b",
    re.IGNORECASE,
)
# An order for new work: never shortcut, let Jev/LLM weigh interject vs p1/p2.
_NEW_WORK_RE = re.compile(
    r"\b(redo\w*|rework\w*|develop\w*|prepare\w*|launch\w*|run|rewrite\w*|"
    r"set\s*up|configure\w*|lay\s*out|code\s+up|"
    r"repair\w*|upgrade\w*|"
    r"new\s+task|one\s+more|another\s+task|"
    r"implement|create|add|build|fix|write|make|generate|draw|deploy|refactor|"
    r"update|remove|delete|new\s+task)\b",
    re.IGNORECASE,
)


def _single_running_target(agents_ctx: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
    """``(agent id, running task ids)`` when exactly one agent is addressed."""
    from app.core.router_jev import distinct_mentioned

    agents = distinct_mentioned({"agents_ctx": agents_ctx or []})
    if len(agents) != 1:
        return None, []
    return agents[0]["id"], [t["id"] for t in agents[0].get("running_tasks") or []]


def _interject_shortcut(
    message_text: str,
    agents_ctx: list[dict[str, Any]],
    open_session: OpenSessionContext | None,
) -> RouteDecision | None:
    """Deterministic ``interject_task`` before Jev/LLM (Stage 4.2).

    Exactly one addressee with exactly one running task in this room, and the
    text is either a status question (always) or carries no order for new
    work.  A message to the agent the user is already in a dialogue with is
    the dialogue's — unless it asks for the status.
    """
    agent_id, running = _single_running_target(agents_ctx)
    if len(running) != 1:
        return None
    text = message_text or ""
    is_status = bool(_STATUS_RE.search(text))
    if not is_status:
        if open_session is not None and str(open_session.peer_agent_id or "") == agent_id:
            return None
        if _NEW_WORK_RE.search(text):
            return None
    return RouteDecision(
        route="interject_task",
        matched_task_id=UUID(running[0]),
        confidence=1.0,
        reasoning=(
            "status question to an agent with a running task — no model call"
            if is_status
            else "message to an agent with a running task, no new work asked — no model call"
        ),
    )


def _apply_interject_guard(
    decision: RouteDecision, running: list[str]
) -> RouteDecision:
    """Make ``interject_task`` serviceable and prefer it where it is safe (4.3).

    - without a running task it cannot be served → single_agent_p2;
    - its task id must be a running one (the model's id is replaced);
    - single_agent_p1 onto the running task itself means "write into it";
    - a single-agent route below CONFIDENCE_THRESHOLD while a task runs →
      interject: the agent can say "this is a separate task", a new branch
      cannot be taken back.
    """
    if not running:
        if decision.route == "interject_task":
            return decision.model_copy(update={
                "route": "single_agent_p2",
                "matched_task_id": None,
                "reasoning": decision.reasoning + " [no running task — coerced]",
            })
        return decision
    matched = str(decision.matched_task_id) if decision.matched_task_id else None
    note = None
    if decision.route == "interject_task":
        if matched in running:
            return decision
        note = " [running task id fixed]"
    elif decision.route == "single_agent_p1" and matched in running:
        note = " [p1 onto the running task → interject]"
    elif (
        decision.route in ("single_agent_p1", "single_agent_p2")
        and decision.confidence < CONFIDENCE_THRESHOLD
    ):
        note = " [low confidence with a running task → interject]"
    if note is None:
        return decision
    return decision.model_copy(update={
        "route": "interject_task",
        "matched_task_id": UUID(matched if matched in running else running[0]),
        "reasoning": decision.reasoning + note,
    })


# Why a Jev-mode message went to the LLM instead (router_agent_plan §3.3).
ESCALATE_MULTI_MENTION = "multi_mention"
ESCALATE_LOW_CONFIDENCE = "low_confidence"
ESCALATE_WAITING_GUARD = "waiting_guard"
ESCALATE_JEV_ERROR = "jev_error"


def _jev_context(
    message_text: str,
    gathered: "_GatheredContext",
    open_session: OpenSessionContext | None,
) -> dict[str, Any]:
    """The plain context dict core/router_jev.py works on."""
    os_: dict[str, Any] | None = None
    if open_session is not None:
        peer_id = open_session.peer_agent_id
        peer = next(
            (a for a in getattr(gathered, "room_agents", []) if a.id == peer_id), None
        )
        os_ = {
            "summary_so_far": open_session.summary_so_far[:_OPEN_SESSION_SUMMARY_CHARS],
            "last_assistant_excerpt":
                open_session.last_assistant_excerpt[:_OPEN_SESSION_EXCERPT_CHARS],
            "has_open_question": open_session.has_open_question,
            "turns": open_session.turns,
            "with_agent_id": str(peer_id) if peer_id else None,
            "with_agent": peer.nickname if peer is not None else None,
            "with_chatmaster": open_session.peer_is_chatmaster,
        }
    return {
        "text": message_text,
        "agents_ctx": gathered.agents_ctx,
        "goals_ctx": gathered.goals_ctx,
        "waiting_ctx": getattr(gathered, "waiting_ctx", []),
        "open_session": os_,
        "mentioned_ids": [str(i) for i in gathered.mentioned_ids],
    }


async def _route_with_jev(
    binding: RouterBinding,
    room: Room,
    message_text: str,
    gathered: "_GatheredContext",
    open_session: OpenSessionContext | None,
    db: AsyncSession,
) -> tuple[RouteDecision | None, str | None, dict[str, Any]]:
    """Ask Jev; return ``(decision, escalation_reason, info)``.

    ``decision`` is None when the message must go to the escalation LLM, and
    ``escalation_reason`` says why.  ``info`` carries what the log needs:
    ``confidence``, ``probabilities``, ``latency_ms``, ``cost_usd``, ``model``.
    Never raises — a Jev failure is an escalation, not a routing failure.
    """
    from app.core import router_jev
    from app.models.secret import SecretPurpose

    settings = binding.settings
    ctx = _jev_context(message_text, gathered, open_session)
    body, maps = router_jev.build_jev_request(ctx, settings.jev_model)
    info: dict[str, Any] = {"model": settings.jev_model}
    if body is None:
        return None, ESCALATE_MULTI_MENTION, info

    try:
        router = binding.router
        secret = await resolve_agent_secret(
            db=db,
            owner_id=router.owner_id,
            agent_id=router.id,
            config=(router.llm_config or {}).get("router") or {},
            request_id=room.id,
            purpose=SecretPurpose.LLM_ROUTER_JEV,
            binding_keys=("secret_binding_id",),
        )
        if secret is None:
            raise RuntimeError("the Router has no OpenRouter key for Jev")
        # Stage 1.2 point 6: the route decision is seconds of waiting.
        await release_db(db, reason="router/jev")
        try:
            data, latency_s = await timed_llm(
                router_jev.call_jev(body, secret.reveal()),
                provider="openrouter",
                side=SIDE_ROUTER,
            )
        finally:
            secret = None
        answer = router_jev.decide(data["answers"], maps)
    except Exception as exc:  # noqa: BLE001 — any Jev failure escalates
        logger.warning("router: Jev failed for room %s: %r", room.id, exc)
        info["error"] = repr(exc)[:300]
        return None, ESCALATE_JEV_ERROR, info

    usage = data.get("usage") or {}
    info.update(
        confidence=answer["confidence"],
        probabilities=answer["probs"],
        latency_ms=round(latency_s * 1000),
        cost_usd=float(usage.get("cost") or 0.0),
        route=answer["route"],
    )
    if answer["confidence"] < settings.jev_min_confidence:
        return None, ESCALATE_LOW_CONFIDENCE, info
    # Stage 0: Jev confidently (0.91) chose continue_session over a waiting
    # task the user was answering — this pair is the LLM's call.
    if answer["route"] == "continue_session" and ctx["waiting_ctx"]:
        return None, ESCALATE_WAITING_GUARD, info

    decision = RouteDecision(
        route=answer["route"],
        matched_goal_id=UUID(answer["goal_id"]) if answer["goal_id"] else None,
        matched_task_id=UUID(answer["task_id"]) if answer["task_id"] else None,
        risk_hint=answer["risk"],
        confidence=answer["confidence"],
        reasoning=f"jev: {answer['route']} p={answer['confidence']:.2f}",
    )
    return _apply_confidence_fallback(decision), None, info


async def route_user_message(
    room: Room,
    message_text: str,
    db: AsyncSession,
    *,
    open_session: OpenSessionContext | None = None,
    dry_run: bool = False,
    report: dict[str, Any] | None = None,
) -> RouteDecision:
    """
    Classify a user message into one of the routes.

    Always emits ROUTE_DECIDED with latency/cost metadata and writes exactly one
    router_decision_log row — including the deterministic shortcut, so "how was
    this message routed?" always has an answer.
    On LLM failure emits ROUTE_LLM_FAILURE and falls back to continuing the open
    session when there is one, else to chatmaster_p2.
    Confidence < 0.6 forces the more-general (p2) variant of the branch.

    Args:
        open_session: the user's open session, when it is young enough to be a
            candidate for ``continue_session`` (stage 2.2). None means the router
            must pick a real route; any open session is closed by the caller.
        dry_run: "Test a route" (router_agent_plan §5) — decide exactly as for a
            real message, but emit no events, record no metrics and write no
            router_decision_log row.
        report: filled with the ROUTE_DECIDED metadata plus the model, Jev's
            probabilities and any failure — what the test panel shows.
    """
    t0 = time.monotonic()

    async def _emit(**kwargs: Any) -> None:
        if not dry_run:
            await emit_event(**kwargs)

    gathered = await _gather_context(room, message_text, db)
    agents_ctx, goals_ctx, mentioned_ids, router_llm_config = gathered
    waiting_ctx = getattr(gathered, "waiting_ctx", [])
    # router_agent_plan §3.1: the account's Router decides whose model routes.
    # Inherit (and an unconfigured or missing Router) keeps the ChatMaster;
    # Jev mode asks Jev first and escalates to this LLM (§3.3).
    binding = await _resolve_router(
        room, getattr(gathered, "room_agents", []), db
    )
    # Inherit resolves to the same ChatMaster _gather_context already picked.
    router_agent = getattr(gathered, "router_agent", None)
    if binding.engine == "jev":
        router_agent = binding.llm_agent
        router_llm_config = (router_agent.llm_config or {}) if router_agent else {}
    user_msg = _build_user_message(
        message_text, agents_ctx, goals_ctx, open_session, waiting_ctx
    )

    decision: RouteDecision | None = None
    fail_reason: str | None = None
    # Stage 3 (§3.3): what Jev did for this message; logged below (stage 4).
    escalation_reason: str | None = None
    jev_info: dict[str, Any] = {}
    # Stage 9 (plan §10.3): when the gateway refused the ChatMaster's model
    # with an opt-in 403, the failure is actionable — the user gets the link
    # instead of a generic "router failed" line.
    opt_in_reason: str | None = None
    opt_in_url: str | None = None
    in_tok = out_tok = 0
    model_used = router_llm_config.get("model", _ROUTER_MODEL_OPENAI)
    # Filled by _call_router_llm when it actually reaches a reply; stays empty
    # for the shortcut and for a failed call, and an empty dict writes NULL.
    llm_metrics: dict[str, Any] = {}

    _, running_ids = _single_running_target(agents_ctx)
    shortcut = _trivial_continue(message_text, open_session) or _interject_shortcut(
        message_text, agents_ctx, open_session
    )
    if shortcut is not None:
        decision = shortcut
        model_used = _SHORTCUT_MODEL
    elif binding.engine == "jev":
        decision, escalation_reason, jev_info = await _route_with_jev(
            binding, room, message_text, gathered, open_session, db
        )
        if decision is not None:
            model_used = jev_info["model"]
        else:
            logger.info(
                "router: Jev escalated room %s to the LLM (%s)",
                room.id, escalation_reason,
            )
    if decision is None:
        if router_agent is None:
            # A group room without a ChatMaster is a misconfiguration, not a
            # routing scenario.  Surface it instead of silently borrowing a
            # random agent's model — see _resolve_router_agent and plan §0.2.
            await _emit(
                room_id=room.id,
                event_type="ROOM_CHATMASTER_MISSING",
                content="No ChatMaster is assigned in the room — routing works by the fallback rule",
                db=db,
                metadata={
                    "level": "warning",
                    "room_id": str(room.id),
                    "hint": "Assign a ChatMaster in the room settings",
                },
            )
        try:
            secret_value = (
                await resolve_agent_secret(
                    db=db,
                    owner_id=router_agent.owner_id,
                    agent_id=router_agent.id,
                    config=router_llm_config,
                    request_id=room.id,
                )
                if router_agent is not None
                else None
            )
            # Stage 1.2 point 6: the route decision is seconds of waiting.
            await release_db(db, reason="router/llm")
            try:
                raw, in_tok, out_tok, model_used = await _call_router_llm(
                    user_msg,
                    router_llm_config,
                    or_session_id=f"router-{room.id}",
                    secret_value=secret_value,
                    system_prompt=(
                        _SYSTEM_PROMPT
                        + (_CONTINUE_SESSION_RULES if open_session is not None else "")
                        + (_WAITING_TASK_RULES if waiting_ctx else "")
                        + (_RUNNING_TASK_RULES if running_ids else "")
                    ),
                    metrics=llm_metrics,
                )
            finally:
                secret_value = None
            decision = _parse_router_reply(raw, llm_metrics.get("reasoning_tokens"))
            decision = _apply_confidence_fallback(decision)
            if escalation_reason:
                decision = decision.model_copy(update={
                    "reasoning": f"jev→llm ({escalation_reason}): {decision.reasoning}"
                })
        except Exception as exc:
            fail_reason = repr(exc)[:300]
            category, classified_url = classify_gateway_error(exc)
            if category in OPT_IN_REASONS:
                opt_in_reason = category
                opt_in_url = classified_url
            logger.warning("ROUTE_LLM_FAILURE room=%s: %s", room.id, fail_reason)
            # A router outage must not silently discard a live conversation:
            # with a young open session, continuing it loses nothing, while
            # chatmaster_p2 would close it and start from bricks alone.
            decision = (
                RouteDecision(
                    route="continue_session",
                    continue_session_id=open_session.session_id,
                    confidence=0.0,
                    reasoning="fallback: router LLM call failed — continuing the open session",
                )
                if open_session is not None
                else _FALLBACK_DECISION
            )

    # Always override mentioned_agent_ids from deterministic parse
    decision = decision.model_copy(update={"mentioned_agent_ids": mentioned_ids})
    # Stage 4: interject_task only onto a running task of the one addressee.
    decision = _apply_interject_guard(decision, running_ids)

    # Open-question override (bug-fix 2026-09-07): the assistant's last turn put
    # a question to the user and this reply carries no @mention. A fresh
    # chatmaster_* route here almost always means the router misread an answer as
    # a new topic (observed: a 3-part answer full of tech nouns rotated the
    # session mid-dialogue). Coerce it back to continuing the open session.
    #
    # History of the guard on this branch:
    #   2026-09-07  coerce only when confidence <= 0.9
    #   2026-09-08  widen to include exactly 0.9 ("write an article for the commit
    #               <url>" right after the CM's questions landed at 0.90)
    #   2026-09-08#2  drop the confidence ceiling entirely. An answer that quotes
    #               the assistant's question back and ends with an imperative
    #               ("...describe X, draw up the plan") scored 0.95 as a fresh task and
    #               cold-rotated a user_cm_p2 session mid requirement-gathering;
    #               the new ChatMaster then planned only the last sub-question and
    #               lost the whole deployment task being specified. The router's
    #               confidence is simply not a usable signal here. The one
    #               reliable "this really is a new topic" marker is the user
    #               saying so outright ("forget it", "let's talk about something else") — deferred
    #               to _user_dropped_topic. Everything else stays in the session;
    #               a genuine pivot the user did not flag is handled inside the
    #               session (ChatMaster keeps the summary and can still branch).
    _open_q_reply = (
        open_session is not None
        and getattr(open_session, "has_open_question", False)
        and not mentioned_ids
        and decision.route in ("chatmaster_p1", "chatmaster_p2")
    )
    if _open_q_reply:
        from app.core.user_sessions import _user_dropped_topic

        if _user_dropped_topic(message_text):
            logger.info(
                "router: keeping %s for room %s despite an open question — the "
                "reply explicitly drops the topic",
                decision.route, room.id,
            )
        else:
            logger.info(
                "router: coercing %s→continue_session for room %s — the assistant "
                "had an open question and the reply has no @mention (confidence "
                "%.2f)",
                decision.route, room.id, decision.confidence,
            )
            decision = decision.model_copy(
                update={
                    "route": "continue_session",
                    "continue_session_id": open_session.session_id,
                    "matched_goal_id": None,
                    "reasoning": decision.reasoning
                    + " [coerced: reply to an open question]",
                }
            )

    # continue_session is only answerable when a session was actually offered,
    # and the id is ours to set — never the model's.
    if decision.route == "continue_session":
        if open_session is None:
            logger.warning(
                "router: continue_session returned for room %s without an open "
                "session — falling back to chatmaster_p2", room.id,
            )
            decision = decision.model_copy(
                update={
                    "route": "chatmaster_p2",
                    "continue_session_id": None,
                    "reasoning": decision.reasoning + " [no open session — coerced]",
                }
            )
        else:
            decision = decision.model_copy(
                update={"continue_session_id": open_session.session_id}
            )
    elif decision.continue_session_id is not None:
        decision = decision.model_copy(update={"continue_session_id": None})

    # resume_waiting_task (S2) is only answerable when a task is actually
    # waiting; the id is validated against the gathered list.
    if decision.route == "resume_waiting_task":
        _waiting_ids = {w["id"] for w in waiting_ctx}
        _picked = (
            str(decision.matched_task_id)
            if decision.matched_task_id is not None
            and str(decision.matched_task_id) in _waiting_ids
            else (waiting_ctx[0]["id"] if waiting_ctx else None)
        )
        if _picked is None:
            logger.warning(
                "router: resume_waiting_task for room %s with no waiting task — "
                "falling back to chatmaster_p2", room.id,
            )
            decision = decision.model_copy(
                update={
                    "route": "chatmaster_p2",
                    "matched_task_id": None,
                    "reasoning": decision.reasoning + " [no waiting task — coerced]",
                }
            )
        else:
            decision = decision.model_copy(
                update={"matched_task_id": UUID(_picked)}
            )

    latency_ms = round((time.monotonic() - t0) * 1000)
    # router_agent_plan §4: which engine made the final call.  An escalated
    # message is "llm" with escalation_reason set; a failed LLM call keeps
    # "llm" too — outcome="fallback" already tells that story.
    if model_used == _SHORTCUT_MODEL:
        engine = "shortcut"
    elif binding.engine == "jev" and escalation_reason is None:
        engine = "jev"
    else:
        engine = "llm"
    from app.core.model_pricing import compute_cost
    cost_usd = compute_cost(model_used, in_tok, out_tok, router_llm_config)
    # Jev reports its own cost; an escalated message pays for both calls.
    cost_usd = round(cost_usd + jev_info.get("cost_usd", 0.0), 8)
    open_session_id = open_session.session_id if open_session is not None else None

    if fail_reason:
        # Until stage 2.4 this degraded silently: 4 of the 9 decisions in the
        # production log were fallbacks nobody ever saw (defect R1).
        # Stage 9 (§10.3): an opt-in 403 gets its own event with the working
        # link — one event for both reasons, distinguished by metadata.
        if opt_in_reason is not None:
            await _emit(
                room_id=room.id,
                event_type="MODEL_OPT_IN_REQUIRED",
                content=model_opt_in_content(model_used, opt_in_reason),
                db=db,
                metadata={
                    "level": "warning",
                    "model": model_used,
                    "reason": opt_in_reason,
                    "opt_in_url": opt_in_url,
                    "agent_nickname": router_agent.nickname if router_agent else None,
                    "provider": (router_llm_config.get("provider") or "").lower() or None,
                    "fallback_route": decision.route,
                },
            )
        else:
            await _emit(
                room_id=room.id,
                event_type="ROUTE_LLM_FAILURE",
                content="Router LLM call failed — route chosen by the fallback rule",
                db=db,
                metadata={
                    "reason": fail_reason,
                    "level": "warning",
                    "fallback_route": decision.route,
                    "open_session_id": str(open_session_id) if open_session_id else None,
                },
            )

    route_meta = {
            "route": decision.route,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "open_session_id": str(open_session_id) if open_session_id else None,
            "continue_session_id": (
                str(decision.continue_session_id) if decision.continue_session_id else None
            ),
            "shortcut": model_used == _SHORTCUT_MODEL,
            "engine": engine,
            "escalation_reason": escalation_reason,
            "jev_confidence": jev_info.get("confidence"),
            "jev_latency_ms": jev_info.get("latency_ms"),
    }
    if report is not None:
        report.update(
            route_meta,
            model=model_used,
            matched_goal_id=str(decision.matched_goal_id) if decision.matched_goal_id else None,
            matched_task_id=str(decision.matched_task_id) if decision.matched_task_id else None,
            jev_route=jev_info.get("route"),
            jev_probabilities=jev_info.get("probabilities"),
            jev_error=jev_info.get("error"),
            fail_reason=fail_reason,
        )
    if dry_run:
        return decision

    await _emit(
        room_id=room.id,
        event_type="ROUTE_DECIDED",
        content=f"route={decision.route}",
        db=db,
        metadata=route_meta,
    )

    # Prometheus metrics
    from app.core.metrics import record_route_decided, record_route_fallback
    record_route_decided(
        route=decision.route,
        confidence=decision.confidence,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        model=model_used,
    )
    if fail_reason:
        record_route_fallback(fallback_route=decision.route)

    # Persist reasoning to router_decision_log for offline accuracy analysis
    try:
        from app.models.router_log import RouterDecisionLog
        log_entry = RouterDecisionLog(
            room_id=room.id,
            route=decision.route,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model_used=model_used,
            mentioned_agents=[str(aid) for aid in decision.mentioned_agent_ids],
            matched_task_id=decision.matched_task_id,
            matched_goal_id=decision.matched_goal_id,
            open_session_id=open_session_id,
            outcome="fallback" if fail_reason else (
                "shortcut" if model_used == _SHORTCUT_MODEL else "llm"
            ),
            reasoning_tokens=llm_metrics.get("reasoning_tokens"),
            engine=engine,
            escalation_reason=escalation_reason,
            jev_probabilities=jev_info.get("probabilities"),
        )
        db.add(log_entry)
        await db.commit()
        _note_router_reasoning(model_used, llm_metrics.get("reasoning_tokens"))
    except Exception as exc:
        logger.warning("router: failed to persist decision log: %s", exc)

    return decision
