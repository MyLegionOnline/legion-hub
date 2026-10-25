"""Jev — the Router's decision model (internal design notes §3.2).

Jev (TypeSafe, via OpenRouter's Decisions API) answers multiple-choice
questions about a JSON ``state``.  Routing maps onto that directly: the
``route`` question offers only the routes that are possible for this message
(chosen deterministically from the @mention count, goals, waiting tasks and
the open session); ``goal`` / ``task`` pick the matched ids by short keys
(g1, t1, …) that :func:`decide` maps back to UUIDs.

The same code serves production (core/router.py) and the stage-0 replay
(scripts/router_jev_eval.py), so the measurement and the product cannot drift.

This module knows nothing about RouteDecision or the DB — it takes a plain
context dict::

    {"text": str,
     "agents_ctx": [{"id", "nickname", "short_description",
                     "tasks": [{"id", "title", "state", "description"}]}],
     "goals_ctx": [{"id", "title"}],
     "waiting_ctx": [{"id", "title", "agent", "kind", "prompt"}],
     "open_session": None | {"summary_so_far", "last_assistant_excerpt",
                             "has_open_question", "turns",
                             "with_agent_id", "with_agent", "with_chatmaster"},
     "mentioned_ids": [str]}
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.llm_providers import openrouter_attribution_headers

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
# A routing decision is on the user's critical path; Jev answers in ~0.3 s
# (p95 0.7 s), so anything past this is an outage — escalate, don't retry.
JEV_TIMEOUT_S = 10.0

ROUTE_CRITERIA = {
    "chatmaster_p1": "No agent is mentioned and the message continues or relates to "
                     "one of the room's active goals.",
    "chatmaster_p2": "No agent is mentioned and the message is a new topic, a new "
                     "request or about finished work — none of the active goals fits.",
    "single_agent_p1": "The message is addressed to the mentioned agent and continues "
                       "or relates to one of that agent's recent tasks.",
    "single_agent_p2": "The message is addressed to the mentioned agent and asks for "
                       "new work unrelated to any of that agent's recent tasks.",
    "continue_session": "The message continues the user's open conversation: a short "
                        "confirmation, an answer to the assistant's question, a "
                        "clarification, a report of results or a follow-up on the "
                        "same topic — including when it @mentions the same agent the "
                        "conversation is already with.",
    "resume_waiting_task": "The message answers or confirms one of the tasks waiting "
                           "for the user's reply (approve, finish it, continue, or "
                           "answers the agent's question).",
    # internal design notes Stage 4: offered only when the mentioned agent is
    # executing a task right now.
    "interject_task": "The message is for the task the mentioned agent is executing "
                      "right now: a status question, a correction, a change of scope, "
                      "\"stop after this step\" or extra details for the work in "
                      "progress — not separate new work.",
}


def distinct_mentioned(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Mentioned room agents, each once — "@a … @a" is one addressee."""
    seen: set[str] = set()
    out = []
    for a in ctx["agents_ctx"]:
        if a["id"] not in seen:
            seen.add(a["id"])
            out.append(a)
    return out


def build_jev_request(
    ctx: dict[str, Any], model: str
) -> tuple[dict[str, Any] | None, dict[str, dict[str, str]]]:
    """Return ``(request body, key maps)``; the body is None when Jev cannot
    decide — two or more addressees need a split with generated text."""
    agents = distinct_mentioned(ctx)
    n = len(agents)
    maps: dict[str, dict[str, str]] = {"goal": {}, "task": {}}
    if n >= 2:
        return None, maps
    options: list[str] = []
    running = (agents[0].get("running_tasks") or []) if n == 1 else []
    maps["running"] = {"ids": [t["id"] for t in running]}
    if n == 1:
        options = ["single_agent_p1", "single_agent_p2"]
        if running:
            options.insert(0, "interject_task")
    else:
        if ctx["goals_ctx"]:
            options.append("chatmaster_p1")
        options.append("chatmaster_p2")
        if ctx["waiting_ctx"]:
            options.append("resume_waiting_task")
    if ctx["open_session"]:
        options.append("continue_session")

    state: dict[str, Any] = {"user_message": ctx["text"]}
    os_ = ctx["open_session"]
    if os_:
        peer_id = os_.get("with_agent_id")
        state["open_conversation"] = {
            "with": (
                "the ChatMaster (room coordinator)" if os_.get("with_chatmaster")
                else f"agent @{os_.get('with_agent')}" if os_.get("with_agent")
                else "an agent"
            ),
            # Without this Jev read a reply to the agent inside its own
            # dialogue as a new task for that agent (stage 0: 4 errors → 0).
            "message_mentions_this_same_agent": bool(peer_id)
            and peer_id in [str(i) for i in ctx["mentioned_ids"]],
            "started_with": os_.get("summary_so_far") or "",
            "assistant_last_turn": os_.get("last_assistant_excerpt") or "",
            "assistant_asked_a_question": bool(os_.get("has_open_question")),
            "messages_so_far": os_.get("turns") or 0,
        }
    if ctx["waiting_ctx"]:
        state["tasks_waiting_for_user"] = [
            {"title": w["title"], "agent": w.get("agent"), "waiting_for": w["kind"],
             "agent_question": w.get("prompt") or ""}
            for w in ctx["waiting_ctx"]
        ]
    if n == 1:
        a = agents[0]
        state["mentioned_agent"] = {
            "nickname": a["nickname"], "role": a.get("short_description") or "",
            "recent_tasks": [
                {"title": t["title"], "state": t["state"], "description": t["description"],
                 **({"running_now": True} if t.get("running") else {})}
                for t in a["tasks"]
            ],
        }
        if running:
            state["mentioned_agent"]["executing_now"] = [t["title"] for t in running]
    if ctx["goals_ctx"]:
        state["active_goals"] = [g["title"] for g in ctx["goals_ctx"]]

    questions: dict[str, Any] = {
        "route": {
            "type": "choice",
            "instructions": "Route the user's message in a multi-agent chat room. "
                            "The message may be in Russian or English. Pick where it "
                            "should go.",
            "criteria": {k: ROUTE_CRITERIA[k] for k in options},
        }
    }
    if n == 0 and ctx["goals_ctx"]:
        crit = {}
        for i, g in enumerate(ctx["goals_ctx"], 1):
            crit[f"g{i}"] = f"The message is about the goal: {g['title']}"
            maps["goal"][f"g{i}"] = g["id"]
        crit["none"] = "The message is not about any of these goals."
        questions["goal"] = {"type": "choice",
                             "instructions": "Which active goal is the message about?",
                             "criteria": crit}
    task_pool = (
        agents[0]["tasks"] if n == 1
        else [{"id": w["id"], "title": w["title"]} for w in ctx["waiting_ctx"]]
    )
    if task_pool:
        crit = {}
        for i, tk in enumerate(task_pool, 1):
            crit[f"t{i}"] = f"The message is about the task: {tk['title']}"
            maps["task"][f"t{i}"] = tk["id"]
        crit["none"] = "The message is not about any of these tasks."
        questions["task"] = {"type": "choice",
                             "instructions": "Which task is the message about?",
                             "criteria": crit}
    questions["risk"] = {
        "type": "choice",
        "instructions": "How risky is the requested work?",
        "criteria": {
            "risky": "Touches DB migrations, dependencies, CI/deploy, schemas, "
                     "payments or security.",
            "trivial": "A small text or style edit.",
            "normal": "Anything else, or no work is requested.",
        },
    }
    return {"model": model, "state": state, "questions": questions}, maps


def decide(answers: dict[str, Any], maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Turn Jev's answers into ``{route, confidence, goal_id, task_id, risk, probs}``.

    Ids are strings from the key maps.  A p1 route without a match cannot be
    served, so it falls to its p2 sibling (resume without a task → chatmaster_p2).
    """
    r = answers["route"]
    route, conf = r["choice"], float(r.get("confidence") or 0.0)
    goal = (answers.get("goal") or {}).get("choice")
    task = (answers.get("task") or {}).get("choice")
    goal_id = maps["goal"].get(goal) if goal else None
    task_id = maps["task"].get(task) if task else None
    if route == "chatmaster_p1" and not goal_id:
        route = "chatmaster_p2"
    if route == "single_agent_p1" and not task_id:
        route = "single_agent_p2"
    if route == "resume_waiting_task" and not task_id:
        route = "chatmaster_p2"
    running = ((maps.get("running") or {}).get("ids")) or []
    if route == "interject_task" and task_id not in running:
        # Only a running task can take the message; the most recent one wins.
        task_id = running[0] if running else None
        if task_id is None:
            route = "single_agent_p2"
    risk = (answers.get("risk") or {}).get("choice")
    return {
        "route": route,
        "confidence": conf,
        "goal_id": goal_id,
        "task_id": task_id,
        "risk": risk if risk in ("trivial", "risky") else None,
        "probs": r.get("probabilities"),
    }


async def call_jev(
    body: dict[str, Any], api_key: str, *, client: httpx.AsyncClient | None = None
) -> tuple[dict[str, Any], float]:
    """POST one decision request; return ``(response json, latency s)``.

    No retries: any failure is the caller's cue to escalate to the LLM.
    """
    t0 = time.monotonic()
    headers = {"Authorization": f"Bearer {api_key}", **openrouter_attribution_headers(DECISIONS_URL)}
    if client is not None:
        resp = await client.post(DECISIONS_URL, json=body, headers=headers)
    else:
        async with httpx.AsyncClient(timeout=JEV_TIMEOUT_S) as c:
            resp = await c.post(DECISIONS_URL, json=body, headers=headers)
    dt = time.monotonic() - t0
    if resp.status_code != 200:
        raise RuntimeError(f"Jev HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    if not isinstance(data, dict) or "route" not in (data.get("answers") or {}):
        raise RuntimeError(f"Jev reply without a route answer: {str(data)[:300]}")
    return data, dt
