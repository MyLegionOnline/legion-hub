"""Live reasoning-capability probe — reasoning_and_router_model_plan stage 3 (R3).

Run inside deployment-backend-1:

    docker exec deployment-backend-1 python scripts/probe_reasoning_support.py
    docker exec deployment-backend-1 python scripts/probe_reasoning_support.py \
        --model glm-5.3 --model qwen3.7-plus --cache

For every OpenCode Go catalog model on the chat/messages wires it runs
`probe_reasoning_support` (2-3 minimal calls each) and prints the capability
table.  `--cache` additionally persists each result into `model_capability`
with the 7-day TTL, so stages 4 and 9 read it without re-probing.

The key is resolved the same way as in probe_reasoning_router.py: through
resolve_agent_secret for the Architect agent, never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOM_ID = uuid.UUID("15261a56-f837-4188-88d1-8e19fd6a72e5")
ARCHITECT_ID = uuid.UUID("dfdd1501-e033-4a63-af53-c3332ed8f20a")

CONCURRENCY = 6


def hdr(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[],
                        help="probe only these model ids (repeatable); default: all chat/messages catalog models")
    parser.add_argument("--cache", action="store_true",
                        help="persist each result into model_capability (7-day TTL)")
    args = parser.parse_args()

    from app.core.secret_keyring import keyring_required, load_keyring
    from app.core.secrets.runtime import configure_runtime_keyring, resolve_agent_secret
    from app.models.secret import SecretPurpose
    from app.db.session import get_session
    from app.core.llm_providers import OPENCODE_GO, OPENCODE_GO_CATALOG, WIRE_CHAT, WIRE_MESSAGES
    from app.core.reasoning import ReasoningProbeError, probe_reasoning_support, save_reasoning_support
    from app.models.agent import Agent

    if keyring_required():
        configure_runtime_keyring(load_keyring())
    else:
        configure_runtime_keyring(None)

    agen = get_session()
    db = await agen.__anext__()
    try:
        agent = await db.get(Agent, ARCHITECT_ID)
        if agent is None or not (agent.llm_config or {}).get("provider"):
            print("!! architect agent not found / has no llm_config — cannot resolve a key")
            return
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
            print("!! cannot probe without a key")
            return

        if args.model:
            models = [(m, None) for m in args.model]
        else:
            models = [
                (entry.id, entry.wire)
                for entry in OPENCODE_GO_CATALOG
                if entry.wire in (WIRE_CHAT, WIRE_MESSAGES)
            ]
        print(f"probing {len(models)} models on {OPENCODE_GO}, concurrency={CONCURRENCY}")

        sem = asyncio.Semaphore(CONCURRENCY)
        results: dict[str, tuple] = {}
        failures: dict[str, str] = {}

        async def one(model_id: str, wire) -> None:
            label = f"{model_id}({wire})" if wire else model_id
            async with sem:
                t0 = time.monotonic()
                try:
                    support = await probe_reasoning_support(
                        provider=OPENCODE_GO, model=model_id, api_key=key,
                    )
                except ReasoningProbeError as exc:
                    failures[label] = str(exc)
                    print(f"  {label:<44} PROBE FAILED — {exc}")
                    return
                dt = time.monotonic() - t0
                results[model_id] = (support, dt)
                print(f"  {label:<44} supported={support.supported!s:<5} "
                      f"controllable={support.controllable!s:<5} "
                      f"off_switch={support.off_switch}  [{dt:.1f}s]")
                if args.cache:
                    await save_reasoning_support(
                        db, provider=OPENCODE_GO, model=model_id, support=support,
                    )
                    print(f"    cached (TTL 7d)")

        hdr(f"Stage 3 (R3) — reasoning capability probe")
        await asyncio.gather(*(one(m, w) for m, w in models))

        hdr("Summary")
        controllable = sorted(m for m, (s, _) in results.items() if s.controllable)
        stubborn = sorted(m for m, (s, _) in results.items() if s.supported and not s.controllable)
        silent = sorted(m for m, (s, _) in results.items() if not s.supported)
        print(f"probed ok      : {len(results)}")
        print(f"probe failures : {len(failures)}")
        print(f"controllable   : {controllable}")
        print(f"always-reasons : {stubborn}   (router stage 5 can only log these, 0.6)")
        print(f"never-reasons  : {silent}")
        if args.cache:
            print("results persisted to model_capability (7-day TTL)")
    finally:
        await agen.aclose()


if __name__ == "__main__":
    asyncio.run(main())
