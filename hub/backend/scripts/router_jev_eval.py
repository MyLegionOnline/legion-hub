"""Stage 0 of the Router-agent plan: replay routing history through Jev.

Run inside deployment-backend-1:
    docker exec deployment-backend-1 python scripts/router_jev_eval.py [--llm] [--limit N]

For every LLM-made row in ``router_decision_log`` the script rebuilds what the
router saw at that moment (the user's message, mentioned agents and their
tasks, goals active then, the open session, tasks waiting for the user) and
asks Jev (OpenRouter Decisions API) to route it.  With ``--llm`` it also
replays the same rebuilt context through the room's current router LLM, which
separates "Jev disagrees" from "the rebuilt context differs from the original".

The historical route is not ground truth — it is what the ChatMaster's model
decided.  Disagreements are written to a CSV for a human to judge.

Rebuilt context is approximate: task states are today's, the session summary
is replaced by the session's first user message (the stored summary is written
at close and would leak the future), agent membership is today's.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import sys
import time
import uuid
from collections import Counter
from datetime import timedelta
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from sqlalchemy import text

OUT_DIR = "/tmp/router_jev_eval"

MENTION_RE = re.compile(r"@([\w-]+)")


# ---------------------------------------------------------------------------
# Context reconstruction
# ---------------------------------------------------------------------------


async def rebuild(db, row: dict[str, Any]) -> dict[str, Any] | None:
    t = row["created_at"].replace(tzinfo=None)
    room_id = row["room_id"]
    msg = (await db.execute(text("""
        select id, content, timestamp from message
        where room_id=:r and sender_user_id is not null and event_type='MESSAGE'
          and timestamp <= :t
        order by timestamp desc limit 1"""), {"r": room_id, "t": t})).mappings().first()
    if not msg:
        return None
    msg_t = msg["timestamp"]

    agents = (await db.execute(text("""
        select a.id, a.nickname, a.short_description from agent a
        join roomagentlink l on l.agent_id=a.id where l.room_id=:r"""),
        {"r": room_id})).mappings().all()
    by_nick = {a["nickname"].lower(): a for a in agents}
    mentioned = [by_nick[n.lower()] for n in MENTION_RE.findall(msg["content"])
                 if n.lower() in by_nick]
    # dedupe, keep order
    seen: set = set()
    mentioned = [a for a in mentioned if not (a["id"] in seen or seen.add(a["id"]))]

    agents_ctx = []
    for a in mentioned:
        tasks = (await db.execute(text("""
            select id, title, state, left(coalesce(description,''),200) d from task
            where room_id=:r and assigned_agent_id=:a and created_at <= :t
            order by updated_at desc limit 10"""),
            {"r": room_id, "a": a["id"], "t": msg_t})).mappings().all()
        agents_ctx.append({
            "id": str(a["id"]), "nickname": a["nickname"],
            "short_description": a["short_description"] or "",
            "tasks": [{"id": str(x["id"]), "title": x["title"], "state": x["state"],
                       "description": x["d"]} for x in tasks],
        })

    goals = (await db.execute(text("""
        select id, title from room_goal
        where room_id=:r and created_at <= :t
          and (completed_at is null or completed_at > :t)
          and status::text not in ('CANCELLED','cancelled','ARCHIVED','archived')
        order by position"""), {"r": room_id, "t": msg_t})).mappings().all()
    goals_ctx = [{"id": str(g["id"]), "title": g["title"]} for g in goals]

    waiting = (await db.execute(text("""
        select t.id, t.title, a.nickname, t.awaiting_user_kind k,
               left(coalesce(t.awaiting_user_prompt,''),300) p
        from task t left join agent a on a.id=t.assigned_agent_id
        where t.room_id=:r and t.awaiting_user_since is not null
          and t.awaiting_user_since between :lo and :t
        order by t.awaiting_user_since desc limit 5"""),
        {"r": room_id, "t": msg_t, "lo": msg_t - timedelta(hours=24)})).mappings().all()
    waiting_ctx = [{"id": str(w["id"]), "title": w["title"], "agent": w["nickname"],
                    "kind": w["k"] or "review", "prompt": w["p"]} for w in waiting]

    open_session = None
    if row["open_session_id"]:
        s = (await db.execute(text(
            "select id, kind, started_at from session where id=:i"),
            {"i": row["open_session_id"]})).mappings().first()
        if s:
            start = s["started_at"].replace(tzinfo=None)
            first_user = (await db.execute(text("""
                select content from message where room_id=:r and sender_user_id is not null
                  and event_type='MESSAGE' and timestamp >= :s and timestamp < :t
                order by timestamp limit 1"""),
                {"r": room_id, "s": start, "t": msg_t})).scalar() or ""
            last_asst = (await db.execute(text("""
                select content from message where room_id=:r and sender_agent_id is not null
                  and event_type='MESSAGE' and timestamp >= :s and timestamp < :t
                order by timestamp desc limit 1"""),
                {"r": room_id, "s": start, "t": msg_t})).scalar() or ""
            turns = (await db.execute(text("""
                select count(*) from message where room_id=:r and event_type='MESSAGE'
                  and timestamp >= :s and timestamp < :t"""),
                {"r": room_id, "s": start, "t": msg_t})).scalar() or 0
            peer = (await db.execute(text("""
                select a.id, a.nickname, (ucs.session_id is not null) is_cm
                from session s
                left join user_agent_session uas on uas.session_id=s.id
                left join user_cm_session ucs on ucs.session_id=s.id
                left join agent a on a.id=coalesce(uas.agent_id, ucs.chat_master_agent_id)
                where s.id=:i"""), {"i": s["id"]})).mappings().first()
            from app.core.user_sessions import _looks_like_open_question
            open_session = {
                # script-only fields, dropped before OpenSessionContext(**...)
                "with_agent_id": str(peer["id"]) if peer and peer["id"] else None,
                "with_agent": peer["nickname"] if peer else None,
                "with_chatmaster": bool(peer and peer["is_cm"]),
                "session_id": str(s["id"]), "kind": s["kind"], "turns": int(turns),
                "summary_so_far": first_user[:600],
                "last_assistant_excerpt": last_asst[-800:],
                "goal_id": str(s["goal_id"]) if s.get("goal_id") else None,
                "task_id": str(s["task_id"]) if s.get("task_id") else None,
                "has_open_question": bool(last_asst) and _looks_like_open_question(last_asst),
            }

    return {
        "text": msg["content"], "agents_ctx": agents_ctx, "goals_ctx": goals_ctx,
        "waiting_ctx": waiting_ctx, "open_session": open_session,
        "mentioned_ids": [str(a["id"]) for a in mentioned],
    }


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------

# The request/decision logic lives in app/core/router_jev.py — the replay and
# production share it, so this measurement always tests what actually runs.
from app.core.router_jev import build_jev_request, decide  # noqa: E402


def postprocess(route: str, conf: float, ctx: dict) -> str:
    """The deterministic overrides route_user_message applies after the model."""
    if conf < 0.6:
        route = {"chatmaster_p1": "chatmaster_p2", "single_agent_p1": "single_agent_p2"}.get(route, route)
    os_ = ctx["open_session"]
    if (os_ and os_["has_open_question"] and not ctx["mentioned_ids"]
            and route in ("chatmaster_p1", "chatmaster_p2")):
        route = "continue_session"
    return route


async def call_jev(client, key, body) -> tuple[dict, float]:
    from app.core.router_jev import call_jev as _call
    return await _call(body, key, client=client)


# ---------------------------------------------------------------------------
# Baseline: the room's current router LLM on the same rebuilt context
# ---------------------------------------------------------------------------


async def call_llm(db, room_id, ctx) -> dict[str, Any]:
    from app.core import router as R
    from app.core.secrets.runtime import resolve_agent_secret
    from app.models.room import Room
    room = await db.get(Room, room_id)
    g = await R._gather_context(room, "", db)
    agent = getattr(g, "router_agent", None)
    cfg = g.router_llm_config
    if agent is None:
        raise RuntimeError("no router agent")
    os_ = ctx["open_session"]
    _osc_fields = set(R.OpenSessionContext.model_fields)
    osc = R.OpenSessionContext(**{**{k: v for k, v in os_.items() if k in _osc_fields}, "session_id": uuid.UUID(os_["session_id"])}) if os_ else None
    user_msg = R._build_user_message(ctx["text"], ctx["agents_ctx"], ctx["goals_ctx"],
                                     osc, ctx["waiting_ctx"])
    sv = await resolve_agent_secret(db=db, owner_id=agent.owner_id, agent_id=agent.id,
                                    config=cfg, request_id=uuid.uuid4())
    system = (R._SYSTEM_PROMPT + (R._CONTINUE_SESSION_RULES if osc else "")
              + (R._WAITING_TASK_RULES if ctx["waiting_ctx"] else ""))
    t0 = time.monotonic()
    raw, i, o, model = await R._call_router_llm(user_msg, cfg, secret_value=sv,
                                                system_prompt=system, metrics={})
    dt = time.monotonic() - t0
    d = R._parse_router_reply(raw)
    from app.core.model_pricing import compute_cost
    return {"route": d.route, "confidence": d.confidence, "latency": dt, "model": model,
            "cost": compute_cost(model, i, o, cfg),
            "goal_id": str(d.matched_goal_id) if d.matched_goal_id else None,
            "task_id": str(d.matched_task_id) if d.matched_task_id else None}


# ---------------------------------------------------------------------------


async def openrouter_key(db) -> str:
    from app.core.secrets.runtime import resolve_agent_secret
    from app.models.agent import Agent
    from sqlmodel import select
    for a in (await db.execute(select(Agent))).scalars().all():
        cfg = a.llm_config or {}
        if (cfg.get("provider") or "").lower() != "openrouter":
            continue
        try:
            sv = await resolve_agent_secret(db=db, owner_id=a.owner_id, agent_id=a.id,
                                            config=cfg, request_id=uuid.uuid4())
        except Exception:
            continue
        if sv:
            print(f"OpenRouter key: from agent @{a.nickname}")
            return sv.reveal()
    raise SystemExit("no OpenRouter key found on any agent")


def pct(a, b):
    return f"{100*a/b:.0f}%" if b else "-"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="typesafe/jev-1.13")
    ap.add_argument("--llm", action="store_true", help="also replay through the router LLM")
    ap.add_argument("--llm-from", default="", help="reuse llm_replay routes from an earlier rows.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--min-confidence", type=float, default=0.6,
                    help="Jev threshold for the hybrid metric (the Router's jev_min_confidence)")
    args = ap.parse_args()

    from app.core.secret_keyring import keyring_required, load_keyring
    from app.core.secrets.runtime import configure_runtime_keyring
    from app.db.session import get_session
    configure_runtime_keyring(load_keyring() if keyring_required() else None)

    agen = get_session()
    db = await agen.__anext__()
    try:
        key = await openrouter_key(db)
        rows = (await db.execute(text(f"""
            select id, room_id, route, confidence, latency_ms, cost_usd, model_used,
                   matched_goal_id, matched_task_id, open_session_id, created_at
            from router_decision_log where outcome='llm'
            order by created_at desc {f'limit {args.limit}' if args.limit else ''}"""
        ))).mappings().all()
        print(f"{len(rows)} LLM-routed rows")

        ctxs = []
        for row in rows:
            ctx = await rebuild(db, dict(row))
            if ctx:
                ctxs.append((dict(row), ctx))
        print(f"{len(ctxs)} rebuilt")

        sem = asyncio.Semaphore(args.concurrency)
        results: list[dict[str, Any]] = []

        async with httpx.AsyncClient(timeout=60) as client:
            async def one(row, ctx):
                rec: dict[str, Any] = {"row": row, "ctx": ctx}
                body, maps = build_jev_request(ctx, args.model)
                if body is None:
                    rec["jev"] = {"route": "ESCALATE_LLM"}
                else:
                    async with sem:
                        try:
                            data, dt = await call_jev(client, key, body)
                            d = decide(data["answers"], maps)
                            d["route"] = postprocess(d["route"], d["confidence"], ctx)
                            d["latency"] = dt
                            d["cost"] = (data.get("usage") or {}).get("cost")
                            d["in_tokens"] = (data.get("usage") or {}).get("input_tokens")
                            rec["jev"] = d
                        except Exception as e:  # noqa: BLE001
                            rec["jev"] = {"route": "ERROR", "error": str(e)[:300]}
                results.append(rec)

            await asyncio.gather(*(one(r, c) for r, c in ctxs))

        if args.llm_from:
            prev = {r["created_at"]: r["llm_replay"] for r in csv.DictReader(open(args.llm_from, encoding="utf-8"))}
            for rec in results:
                k = rec["row"]["created_at"].isoformat(timespec="seconds")
                if prev.get(k):
                    rec["llm"] = {"route": prev[k], "latency": 0, "cost": 0}
            args.llm = True
        elif args.llm:
            print("replaying through the router LLM (sequential)…")
            for rec in results:
                try:
                    d = await call_llm(db, rec["row"]["room_id"], rec["ctx"])
                    d["route"] = postprocess(d["route"], d["confidence"], rec["ctx"])
                    rec["llm"] = d
                except Exception as e:  # noqa: BLE001
                    rec["llm"] = {"route": "ERROR", "error": str(e)[:300]}
    finally:
        await agen.aclose()

    report(results, args)


def report(results, args) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    judged = [r for r in results if r["jev"]["route"] not in ("ERROR", "ESCALATE_LLM")]
    errors = [r for r in results if r["jev"]["route"] == "ERROR"]
    esc = [r for r in results if r["jev"]["route"] == "ESCALATE_LLM"]
    agree = [r for r in judged if r["jev"]["route"] == r["row"]["route"]]

    lines = [f"# Router replay: {args.model}", ""]
    lines.append(f"rows: {len(results)}  judged by Jev: {len(judged)}  "
                 f"escalated (≥2 mentions): {len(esc)}  errors: {len(errors)}")
    lines.append(f"**Jev vs historical route: {len(agree)}/{len(judged)} = {pct(len(agree), len(judged))}**")
    if args.llm:
        both = [r for r in judged if r.get("llm", {}).get("route") not in (None, "ERROR")]
        la = sum(r["llm"]["route"] == r["row"]["route"] for r in both)
        jl = sum(r["llm"]["route"] == r["jev"]["route"] for r in both)
        lines.append(f"LLM replay vs historical: {la}/{len(both)} = {pct(la, len(both))}  "
                     f"(ceiling set by the rebuilt context)")
        lines.append(f"Jev vs LLM replay (same context): {jl}/{len(both)} = {pct(jl, len(both))}")
        # Production rules (core/router.py §3.3): Jev decides unless it
        # escalates — ≥2 mentions, confidence below the threshold, or
        # continue_session while a task waits; the LLM decides the rest.
        hyb = [r for r in results if r.get("llm", {}).get("route") not in (None, "ERROR")]
        def _hybrid(r):
            j = r["jev"]
            if j["route"] in ("ERROR", "ESCALATE_LLM") or j["confidence"] < args.min_confidence:
                return r["llm"]["route"], True
            if j["route"] == "continue_session" and r["ctx"]["waiting_ctx"]:
                return r["llm"]["route"], True
            return j["route"], False
        decided = [(_hybrid(r), r) for r in hyb]
        ha = sum(route == r["llm"]["route"] for (route, _), r in decided)
        esc_n = sum(esc for (_, esc), _ in decided)
        lines.append(f"**Hybrid (prod rules, Jev ≥ {args.min_confidence}) vs LLM replay: "
                     f"{ha}/{len(decided)} = {pct(ha, len(decided))}**  "
                     f"(escalated to LLM: {esc_n} = {pct(esc_n, len(decided))})")

    # match accuracy where both picked p1
    g_both = [r for r in agree if r["row"]["route"] == "chatmaster_p1"]
    t_both = [r for r in agree if r["row"]["route"] in ("single_agent_p1", "resume_waiting_task")]
    if g_both:
        ok = sum(str(r["row"]["matched_goal_id"]) == r["jev"].get("goal_id") for r in g_both)
        lines.append(f"goal id match on agreed chatmaster_p1: {ok}/{len(g_both)}")
    if t_both:
        ok = sum(str(r["row"]["matched_task_id"]) == r["jev"].get("task_id") for r in t_both)
        lines.append(f"task id match on agreed p1/resume: {ok}/{len(t_both)}")

    lines += ["", "## Per historical route", "", "| historical | n | Jev agrees | Jev said instead |", "|---|---|---|---|"]
    by = Counter(r["row"]["route"] for r in judged)
    for route, n in by.most_common():
        rs = [r for r in judged if r["row"]["route"] == route]
        a = sum(r["jev"]["route"] == route for r in rs)
        other = Counter(r["jev"]["route"] for r in rs if r["jev"]["route"] != route)
        lines.append(f"| {route} | {n} | {pct(a, n)} | {dict(other)} |")

    jl_ = [r["jev"]["latency"] for r in judged]
    jc_ = [r["jev"]["cost"] or 0 for r in judged]
    hl_ = [r["row"]["latency_ms"] / 1000 for r in judged if r["row"]["latency_ms"]]
    hc_ = [r["row"]["cost_usd"] or 0 for r in judged]
    def q(xs, p):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0
    lines += ["", "## Latency and cost", "", "| | p50 s | p95 s | avg $ / call | total $ |", "|---|---|---|---|---|"]
    lines.append(f"| historical LLM | {q(hl_, .5):.2f} | {q(hl_, .95):.2f} | {statistics.mean(hc_) if hc_ else 0:.5f} | {sum(hc_):.4f} |")
    lines.append(f"| Jev | {q(jl_, .5):.2f} | {q(jl_, .95):.2f} | {statistics.mean(jc_) if jc_ else 0:.6f} | {sum(jc_):.4f} |")
    if args.llm:
        ll = [r["llm"] for r in results if r.get("llm", {}).get("route") not in (None, "ERROR")]
        if ll:
            lines.append(f"| LLM replay | {q([x['latency'] for x in ll], .5):.2f} | {q([x['latency'] for x in ll], .95):.2f} | "
                         f"{statistics.mean([x['cost'] or 0 for x in ll]):.5f} | {sum(x['cost'] or 0 for x in ll):.4f} |")
    confs = [r["jev"]["confidence"] for r in judged]
    if confs:
        lo = [r for r in judged if r["jev"]["confidence"] < 0.6]
        lines.append(f"\nJev confidence: median {statistics.median(confs):.2f}; "
                     f"below 0.6 on {len(lo)} rows (would escalate to LLM)")
        hi = [r for r in judged if r["jev"]["confidence"] >= 0.6]
        ha = sum(r["jev"]["route"] == r["row"]["route"] for r in hi)
        lines.append(f"agreement when Jev confidence ≥ 0.6: {ha}/{len(hi)} = {pct(ha, len(hi))}")
    if errors:
        lines += ["", "## Errors", ""] + [f"- {e['jev']['error']}" for e in errors[:5]]

    md = "\n".join(lines)
    open(f"{OUT_DIR}/report.md", "w", encoding="utf-8").write(md)
    with open(f"{OUT_DIR}/rows.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["created_at", "historical", "hist_model", "jev", "jev_conf", "llm_replay",
                    "agree", "mentions", "open_session", "open_question", "goals", "message", "jev_probs"])
        for r in sorted(results, key=lambda r: r["row"]["created_at"]):
            c = r["ctx"]
            w.writerow([r["row"]["created_at"].isoformat(timespec="seconds"), r["row"]["route"],
                        r["row"]["model_used"], r["jev"]["route"],
                        f"{r['jev'].get('confidence', 0):.2f}", r.get("llm", {}).get("route", ""),
                        r["jev"]["route"] == r["row"]["route"], len(c["mentioned_ids"]),
                        bool(c["open_session"]),
                        bool(c["open_session"] and c["open_session"]["has_open_question"]),
                        len(c["goals_ctx"]), c["text"][:300].replace("\n", " "),
                        json.dumps(r["jev"].get("probs"), ensure_ascii=False)])
    print(md)
    print(f"\nwritten: {OUT_DIR}/report.md, {OUT_DIR}/rows.csv")


if __name__ == "__main__":
    asyncio.run(main())
