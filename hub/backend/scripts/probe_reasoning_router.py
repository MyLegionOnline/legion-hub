"""Live probe for reasoning_and_router_model_plan stages 0-2.

Run inside deployment-backend-1:
    docker exec deployment-backend-1 python scripts/probe_reasoning_router.py

Verifies against the real OpenCode Go gateway with the user's configured key:
  * Stage 0 / defect B  — /messages authenticates with x-api-key, not Bearer
  * Stage 0 / defect C  — messages_text_and_thinking survives a thinking block
  * Stage 0 / catalog   — GET /models count vs OPENCODE_GO_CATALOG
  * chat wire           — the configured model (glm-5.2) answers, reasoning_content
  * defect / RegionError — deepseek-v4-flash still 403 (known, not fixed here)
  * Stage 1             — router inherits the ChatMaster's model for room 15261a56
  * Stage 1+2 e2e       — route_user_message writes one log row, no ROUTE_LLM_FAILURE
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

ROOM_ID = uuid.UUID("15261a56-f837-4188-88d1-8e19fd6a72e5")
ARCHITECT_ID = uuid.UUID("dfdd1501-e033-4a63-af53-c3332ed8f20a")
GW = "https://opencode.ai/zen/go/v1"


def hdr(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")


async def main() -> None:
    from app.core.secret_keyring import keyring_required, load_keyring
    from app.core.secrets.runtime import configure_runtime_keyring, resolve_agent_secret
    from app.models.secret import SecretPurpose
    from app.db.session import get_session
    from app.core.llm_providers import (
        OPENCODE_GO,
        OPENCODE_GO_CATALOG,
        messages_headers,
        messages_text_and_thinking,
    )
    from app.models.agent import Agent
    from app.models.room import Room
    from app.core import router as router_mod

    if keyring_required():
        configure_runtime_keyring(load_keyring())
    else:
        configure_runtime_keyring(None)

    agen = get_session()
    db = await agen.__anext__()
    try:
        agent = await db.get(Agent, ARCHITECT_ID)
        room = await db.get(Room, ROOM_ID)
        print(f"agent={agent.nickname!r} llm_config.provider={agent.llm_config.get('provider')} "
              f"model={agent.llm_config.get('model')}")

        sv = await resolve_agent_secret(
            db=db,
            owner_id=agent.owner_id,
            agent_id=agent.id,
            config=agent.llm_config,
            request_id=uuid.uuid4(),
            purpose=SecretPurpose.LLM_PRIMARY,
        )
        key = sv.reveal() if sv else None
        print(f"secret resolved: {'yes' if key else 'NO'} (len={len(key) if key else 0})")
        if not key:
            print("!! cannot continue live probes without a key")
            return

        async with httpx.AsyncClient(timeout=90.0) as c:
            # --- Stage 0 catalog ---
            hdr("Stage 0 — GET /models")
            r = await c.get(f"{GW}/models", headers={"x-api-key": key})
            ids = []
            if r.status_code == 200:
                body = r.json()
                ids = [m.get("id") for m in (body.get("data") or body.get("models") or [])]
            print(f"HTTP {r.status_code}  gateway lists {len(ids)} models; catalog has {len(OPENCODE_GO_CATALOG)}")
            cat_ids = {e.id for e in OPENCODE_GO_CATALOG}
            missing = sorted(set(ids) - cat_ids)
            extra = sorted(cat_ids - set(ids))
            print(f"  in gateway, not in catalog: {missing}")
            print(f"  in catalog, not in gateway: {extra}")

            # --- Stage 0 defect B: x-api-key vs Bearer on /messages ---
            hdr("Stage 0 / defect B — /messages auth scheme")
            probe_model = "qwen3.7-plus"
            payload = {
                "model": probe_model,
                "system": "reply with the single word: ok",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 512,
            }
            r_key = await c.post(f"{GW}/messages",
                                 headers=messages_headers(OPENCODE_GO, key), json=payload)
            r_bearer = await c.post(f"{GW}/messages",
                                    headers={"anthropic-version": "2023-06-01",
                                             "content-type": "application/json",
                                             "Authorization": f"Bearer {key}"},
                                    json=payload)
            print(f"messages_headers() (x-api-key) -> HTTP {r_key.status_code}   [expect 200]")
            print(f"Authorization: Bearer         -> HTTP {r_bearer.status_code}   [expect 401 -> proves the fix]")

            # --- Stage 0 defect C: thinking-block parsing ---
            hdr("Stage 0 / defect C — messages_text_and_thinking on a real reply")
            if r_key.status_code == 200:
                data = r_key.json()
                c0 = (data.get("content") or [{}])[0]
                text, thinking = messages_text_and_thinking(data)
                print(f"content[0].type = {c0.get('type')!r}  (raw content[0]['text'] would "
                      f"{'KeyError' if c0.get('type') != 'text' else 'be safe'})")
                print(f"parsed visible text : {text[:120]!r}")
                print(f"parsed thinking     : {thinking[:120]!r}")
            else:
                print(f"skipped — /messages returned {r_key.status_code}: {r_key.text[:300]}")

            # --- chat wire: the actually-configured model ---
            hdr("chat wire — configured model glm-5.2 via /chat/completions")
            r_chat = await c.post(
                f"{GW}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
                json={"model": "glm-5.2",
                      "messages": [{"role": "user", "content": "Say 'ok' and nothing else."}]},
            )
            print(f"HTTP {r_chat.status_code}")
            if r_chat.status_code == 200:
                j = r_chat.json()
                msg = j["choices"][0]["message"]
                det = (j.get("usage") or {}).get("completion_tokens_details") or {}
                print(f"  message.content        : {(msg.get('content') or '')[:80]!r}")
                print(f"  message.reasoning_content present: {'reasoning_content' in msg and bool(msg['reasoning_content'])}")
                print(f"  usage.reasoning_tokens : {det.get('reasoning_tokens')}")
            else:
                print(f"  {r_chat.text[:300]}")

            # --- RegionError model ---
            hdr("known defect — deepseek-v4-flash RegionError (not fixed in stages 0-2)")
            r_ds = await c.post(f"{GW}/messages",
                                headers=messages_headers(OPENCODE_GO, key),
                                json={"model": "deepseek-v4-flash",
                                      "messages": [{"role": "user", "content": "ping"}],
                                      "max_tokens": 64})
            print(f"HTTP {r_ds.status_code}: {r_ds.text[:220]}")

        # --- Stage 1: router inherits ChatMaster's model ---
        hdr("Stage 1 — _resolve_router_agent / _gather_context for room 15261a56")
        gathered = await router_mod._gather_context(room, "routing test", db)
        ra = getattr(gathered, "router_agent", None)
        cfg = gathered.router_llm_config
        print(f"router_agent      : {ra.nickname if ra else None}  (id={ra.id if ra else None})")
        print(f"router model_used : {cfg.get('model')}   [expect glm-5.2 = ChatMaster's model]")
        print(f"ChatMaster id     : {router_mod._get_cm_agent_id(room)}")

        # --- Stage 1 + 2 end to end ---
        hdr("Stage 1+2 e2e — route_user_message (writes one router_decision_log row)")
        from app.models.router_log import RouterDecisionLog  # noqa
        from sqlalchemy import select, func
        before = (await db.execute(
            select(func.count()).select_from(RouterDecisionLog)
            .where(RouterDecisionLog.room_id == ROOM_ID))).scalar()
        import time as _t
        t0 = _t.monotonic()
        decision = await router_mod.route_user_message(
            room, "let's discuss the blog structure", db)
        dt = _t.monotonic() - t0
        await db.commit()
        after_rows = (await db.execute(
            select(RouterDecisionLog).where(RouterDecisionLog.room_id == ROOM_ID)
            .order_by(RouterDecisionLog.created_at.desc()).limit(1))).scalars().all()
        after = (await db.execute(
            select(func.count()).select_from(RouterDecisionLog)
            .where(RouterDecisionLog.room_id == ROOM_ID))).scalar()
        print(f"route={decision.route!r} confidence={decision.confidence} latency={dt:.1f}s")
        print(f"log rows: {before} -> {after}  (expect +1)")
        if after_rows:
            row = after_rows[0]
            print(f"last log row: model_used={row.model_used!r} route={row.route!r} "
                  f"outcome={row.outcome!r} latency_ms={row.latency_ms}")
    finally:
        await agen.aclose()


if __name__ == "__main__":
    asyncio.run(main())
