"""Live probe for the two messages-wire OpenCode Go agents.

    Writer     -> qwen3.7-max   (WIRE_MESSAGES)
    Frontender -> minimax-m3    (WIRE_MESSAGES)

Run inside deployment-backend-1:
    docker exec deployment-backend-1 python scripts/probe_messages_wire_agents.py

Checks:
  * raw POST /zen/go/v1/messages with messages_headers -> 200 (defect B)
  * same with a proxy-shaped body (system + tools + max_tokens 4096)
  * messages_text_and_thinking survives the thinking block (defect C)
  * proxy_session._call_llm end-to-end with the real agent config + secret
  * router: does the ChatMaster stay glm-5.2 when these agents are @mentioned
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx

GW = "https://opencode.ai/zen/go/v1"
AGENTS = {
    "Writer": uuid.UUID("e91dc455-8bbc-48aa-b971-23a7c60b4808"),
    "Frontender": uuid.UUID("cb51a97b-0cff-4fe7-bf61-80a602776e3b"),
}
ROOM_BLOG = uuid.UUID("15261a56-f837-4188-88d1-8e19fd6a72e5")


def hdr(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")


async def main() -> None:
    from app.core.secret_keyring import keyring_required, load_keyring
    from app.core.secrets.runtime import configure_runtime_keyring, resolve_agent_secret
    from app.models.secret import SecretPurpose
    from app.db.session import get_session
    from app.core.llm_providers import (
        OPENCODE_GO,
        messages_headers,
        messages_text_and_thinking,
        messages_url,
        provider_wire,
        uses_messages_wire,
    )
    from app.models.agent import Agent
    from app.models.room import Room
    from app.core import router as router_mod
    from app.core import proxy_session

    configure_runtime_keyring(load_keyring() if keyring_required() else None)
    agen = get_session()
    db = await agen.__anext__()
    try:
        for name, aid in AGENTS.items():
            agent = await db.get(Agent, aid)
            cfg = agent.llm_config
            model = cfg.get("model")
            prov = cfg.get("provider")
            hdr(f"{name} — {prov}/{model}  wire={provider_wire(prov, model)}  "
                f"messages_wire={uses_messages_wire(prov, model)}")

            sv = await resolve_agent_secret(
                db=db, owner_id=agent.owner_id, agent_id=agent.id, config=cfg,
                request_id=uuid.uuid4(), purpose=SecretPurpose.LLM_PRIMARY,
            )
            key = sv.reveal() if sv else None
            print(f"secret: {'ok len=%d' % len(key) if key else 'MISSING'}")
            if not key:
                continue

            async with httpx.AsyncClient(timeout=90.0) as c:
                # 1. bare /messages
                r = await c.post(
                    messages_url(OPENCODE_GO, None),
                    headers=messages_headers(OPENCODE_GO, key),
                    json={"model": model,
                          "messages": [{"role": "user", "content": "Answer in one word: hello"}],
                          "max_tokens": 512},
                )
                print(f"[1] bare /messages          -> HTTP {r.status_code}")
                if r.status_code == 200:
                    data = r.json()
                    types = [b.get("type") for b in data.get("content", [])]
                    text, think = messages_text_and_thinking(data)
                    print(f"    content block types = {types}")
                    print(f"    text={text[:80]!r}  thinking_len={len(think)}")
                else:
                    print(f"    {r.text[:300]}")

                # 2. proxy-shaped body: system + tools + max_tokens 4096
                r2 = await c.post(
                    messages_url(OPENCODE_GO, None),
                    headers=messages_headers(OPENCODE_GO, key),
                    json={
                        "model": model,
                        "system": "You are an agent. Answer briefly.",
                        "messages": [{"role": "user", "content": "Say 'done'."}],
                        "tools": [proxy_session.ANTHROPIC_CLOSE_TOOL],
                        "max_tokens": 4096,
                    },
                )
                print(f"[2] proxy-shaped (tools+4096)-> HTTP {r2.status_code}")
                if r2.status_code != 200:
                    print(f"    {r2.text[:400]}")
                else:
                    d2 = r2.json()
                    print(f"    stop_reason={d2.get('stop_reason')}  "
                          f"types={[b.get('type') for b in d2.get('content', [])]}")

            # 3. real code path: proxy_session._call_llm
            try:
                text, tool_input, tin, tout, mdl = await proxy_session._call_llm(
                    cfg,
                    "You are an assistant. Answer briefly.",
                    [{"role": "user", "content": "Name one word: test"}],
                    secret_value=sv,
                )
                print(f"[3] proxy_session._call_llm -> OK  model={mdl}  "
                      f"text={ (text or '')[:80]!r}  tool_input={tool_input}  "
                      f"tokens={tin}/{tout}")
            except Exception as exc:
                print(f"[3] proxy_session._call_llm -> FAIL  {type(exc).__name__}: {repr(exc)[:400]}")

        # 4. router: mention both messages-wire agents, ChatMaster model must stay glm-5.2
        hdr("router — @mention messages-wire agents, ChatMaster model must hold")
        room = await db.get(Room, ROOM_BLOG)
        for msg in ["@Writer write an introduction", "@Frontender lay out a card", "@Writer @Frontender put together a landing page"]:
            g = await router_mod._gather_context(room, msg, db)
            ra = getattr(g, "router_agent", None)
            print(f"  {msg!r:45} -> router_agent={ra.nickname if ra else None} "
                  f"model={g.router_llm_config.get('model')}")
    finally:
        await agen.aclose()


if __name__ == "__main__":
    asyncio.run(main())
