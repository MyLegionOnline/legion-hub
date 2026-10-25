"""Live reasoning probe for OpenRouter — plan stage OR1 (§4A).

Run inside deployment-backend-1:

    docker exec deployment-backend-1 python scripts/probe_reasoning_openrouter.py --questions
    docker exec deployment-backend-1 python scripts/probe_reasoning_openrouter.py \
        --model z-ai/glm-5.3 --model qwen/qwen3.8-flash

Answers the seven questions of §4A.2 by *measuring*, never by extrapolating
from the OpenCode Go numbers in §0.6: which request field OpenRouter honours,
what "off" looks like there, whether `reasoning_tokens` and the reasoning text
come back, and how a non-reasoning model reacts to reasoning fields.

The variant matrix answers questions 1-3 and 6; `--questions` adds the three
follow-ups a single-turn matrix cannot reach — the tool loop (q4), a model
that declares no reasoning at all (q5) and the `max_tokens` trap (q7).

The key is resolved through `resolve_agent_secret` for an OpenRouter agent and
never printed — same contract as `probe_reasoning_support.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# An OpenRouter agent of the plan's account; used only to resolve a working key.
ARCHITECT_ID = uuid.UUID("dfdd1501-e033-4a63-af53-c3332ed8f20a")

BASE_URL = "https://openrouter.ai/api/v1"
CONCURRENCY = 4
TIMEOUT_S = 180.0

# One model per class of the catalog's own `reasoning` declaration, so the
# table of §4A.5 covers every shape the annotator will meet.
DEFAULT_MODELS = [
    "deepseek/deepseek-v4-flash",             # mandatory=false, efforts [xhigh, high] — agents' own model
    "deepseek/deepseek-v4-flash-vision-exp",  # mandatory=false, efforts [max, high, low]
    "qwen/qwen3.8-flash",                     # mandatory=false, no efforts, supports_max_tokens
    "z-ai/glm-5.3",                           # mandatory=true  — the "cannot be disabled" class
    "openai/gpt-6-astra",                     # mandatory=true, OpenAI-shaped
    "anthropic/claude-fable-5.1",             # mandatory=true, Anthropic via OpenRouter (question 4)
    "google/gemini-3.8-flash",                # mandatory=true, Google
]

# The variants under test.  `label` is what goes into the result table.
VARIANTS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ('reasoning_effort:"none"', {"reasoning_effort": "none"}),
    ("reasoning:{enabled:false}", {"reasoning": {"enabled": False}}),
    ('reasoning:{effort:"none"}', {"reasoning": {"effort": "none"}}),
    ("reasoning:{exclude:true}", {"reasoning": {"exclude": True}}),
    ('reasoning_effort:"low"', {"reasoning_effort": "low"}),
    ('reasoning:{effort:"low"}', {"reasoning": {"effort": "low"}}),
    ('reasoning:{effort:"medium"}', {"reasoning": {"effort": "medium"}}),
    ("reasoning:{max_tokens:1024}", {"reasoning": {"max_tokens": 1024}}),
]

PROMPT = "Reply with the single word: ok"

# A prompt heavy enough that reasoning alone can overrun a small cap — the
# §0.7 trap, re-checked on this gateway rather than assumed from the other one.
HARD_PROMPT = (
    "A train leaves A at 60 km/h and another leaves B at 90 km/h, 400 km apart, "
    "moving toward each other. A bird flies at 120 km/h between them until they "
    "meet. How far does the bird fly? Think it through carefully step by step."
)

# Models that declare no `reasoning` block at all — question 5 of §4A.2.
PLAIN_MODELS = [
    "openai/gpt-chat-latest",
    "qwen/qwen3-coder-next",
    "writer/palmyra-x5",
    "minimax/minimax-m2-her",
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def hdr(s: str) -> None:
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}", flush=True)


async def call(
    client, key: str, model: str, fields: dict, *,
    max_tokens=None, prompt: str = PROMPT, tools=None, history=None, raw: bool = False,
) -> dict:
    """One chat/completions call; returns a flat reading, never raises on HTTP."""
    import httpx

    body: dict = {
        "model": model,
        "messages": history or [{"role": "user", "content": prompt}],
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if tools is not None:
        body["tools"] = tools
    body.update(fields)
    t0 = time.monotonic()
    try:
        r = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
            json=body,
        )
    except httpx.HTTPError as exc:
        return {"status": "ERR", "error": f"{type(exc).__name__}: {exc}", "dt": time.monotonic() - t0}
    dt = time.monotonic() - t0
    if r.status_code != 200:
        return {"status": r.status_code, "error": r.text[:220], "dt": dt}
    data = r.json()
    choice = (data.get("choices") or [{}])[0] or {}
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    thinking = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return {
        "status": 200,
        "dt": dt,
        "message": msg if raw else None,
        "tool_calls": len(msg.get("tool_calls") or []),
        "text": (msg.get("content") or "")[:60],
        "content_chars": len(msg.get("content") or ""),
        "thinking_chars": len(thinking or ""),
        "reasoning_details": bool(msg.get("reasoning_details")),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": choice.get("finish_reason"),
    }


def render(label: str, reading: dict) -> str:
    if reading["status"] != 200:
        return f"    {label:<44} {reading['status']}  {reading.get('error', '')[:110]}"
    rt = reading["reasoning_tokens"]
    return (
        f"    {label:<44} 200 {reading['dt']:5.1f}s "
        f"rt={str(rt):<5} think_chars={reading['thinking_chars']:<5} "
        f"details={str(reading['reasoning_details']):<5} "
        f"out={reading['completion_tokens']:<5} content_chars={reading['content_chars']:<5} "
        f"finish={reading['finish']} text={reading['text']!r}"
    )


async def follow_ups(client, key: str, models: list[str]) -> None:
    """Questions 4, 5 and 7 of §4A.2 — the ones the variant matrix cannot answer."""
    hdr("Q5 — a model that declares no reasoning: 400, or silent ignore?")
    for mid in PLAIN_MODELS:
        for label, fields in (
            ("baseline", {}),
            ('reasoning_effort:"none"', {"reasoning_effort": "none"}),
            ('reasoning:{effort:"high"}', {"reasoning": {"effort": "high"}}),
        ):
            print(render(f"{mid} {label}", await call(client, key, mid, fields)), flush=True)

    hdr("Q7 — max_tokens trap: does reasoning eat the whole cap and empty `content`?")
    for mid in models:
        for label, extra in (("cap=200", {}), ("cap=200 effort:high", {"reasoning": {"effort": "high"}})):
            reading = await call(client, key, mid, extra, max_tokens=200, prompt=HARD_PROMPT)
            print(render(f"{mid} {label}", reading), flush=True)

    hdr("Q4 — tool loop: must the reasoning blocks be echoed back in history?")
    for mid in models:
        msgs = [{"role": "user", "content": "What's the weather in Paris? Use the tool, then answer."}]
        first = await call(
            client, key, mid, {"reasoning": {"effort": "high"}},
            tools=TOOLS, history=msgs, raw=True,
        )
        if first.get("status") != 200:
            print(f"  {mid}: turn 1 -> {first['status']} {first.get('error', '')[:110]}", flush=True)
            continue
        message = first["message"]
        calls = message.get("tool_calls") or []
        print(f"  {mid}: turn 1 tool_calls={len(calls)} "
              f"reasoning_chars={len(message.get('reasoning') or '')} "
              f"details={bool(message.get('reasoning_details'))}", flush=True)
        if not calls:
            print("     (no tool call — the echo requirement cannot be tested here)", flush=True)
            continue
        tool_reply = {"role": "tool", "tool_call_id": calls[0]["id"], "content": "18C, clear"}
        # (a) history without any reasoning — what every Hub call site builds today
        stripped = {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
        # (b) history with the reasoning echoed back
        echoed = dict(stripped)
        for field in ("reasoning", "reasoning_details"):
            if message.get(field):
                echoed[field] = message[field]
        for label, assistant_turn in (("stripped", stripped), ("echoed", echoed)):
            reading = await call(
                client, key, mid, {"reasoning": {"effort": "high"}},
                tools=TOOLS, history=msgs + [assistant_turn, tool_reply],
            )
            print(render(f"{mid} turn 2 / {label}", reading), flush=True)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--cap-trap", action="store_true",
                        help="also run the §0.7 max_tokens trap check on the trivial prompt")
    parser.add_argument("--questions", action="store_true",
                        help="run the §4A.2 follow-ups the variant matrix cannot answer: "
                             "q4 (tool loop must echo reasoning back?), q5 (model with no "
                             "reasoning block), q7 (max_tokens trap on a hard prompt)")
    args = parser.parse_args()

    import httpx

    from app.core.secret_keyring import keyring_required, load_keyring
    from app.core.secrets.runtime import configure_runtime_keyring, resolve_agent_secret
    from app.db.session import get_session
    from app.models.agent import Agent
    from app.models.secret import SecretPurpose

    if keyring_required():
        configure_runtime_keyring(load_keyring())
    else:
        configure_runtime_keyring(None)

    agen = get_session()
    db = await agen.__anext__()
    try:
        agent = await db.get(Agent, ARCHITECT_ID)
        if agent is None:
            print("!! agent not found — cannot resolve a key")
            return
        sv = await resolve_agent_secret(
            db=db, owner_id=agent.owner_id, agent_id=agent.id,
            config=agent.llm_config, request_id=uuid.uuid4(),
            purpose=SecretPurpose.LLM_PRIMARY,
        )
        key = sv.reveal() if sv else None
        print(f"secret resolved: {'yes' if key else 'NO'} (len={len(key) if key else 0})")
        if not key:
            return

        models = args.model or DEFAULT_MODELS
        async with httpx.AsyncClient(timeout=60) as c:
            cat = await c.get(f"{BASE_URL}/models")
            catalog = {m["id"]: m for m in cat.json().get("data", [])}

        sem = asyncio.Semaphore(CONCURRENCY)

        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            async def one(model: str) -> None:
                declared = (catalog.get(model) or {}).get("reasoning")
                lines = [f"\n  {model}   catalog.reasoning={json.dumps(declared)}"]
                for label, fields in VARIANTS:
                    async with sem:
                        reading = await call(client, key, model, fields)
                    lines.append(render(label, reading))
                if args.cap_trap:
                    async with sem:
                        reading = await call(client, key, model, {}, max_tokens=512)
                    lines.append(render("baseline + max_tokens:512", reading))
                print("\n".join(lines), flush=True)

            hdr("Stage OR1 (§4A) — OpenRouter reasoning probe")
            await asyncio.gather(*(one(m) for m in models))

            if args.questions:
                await follow_ups(client, key, models)
    finally:
        await agen.aclose()


if __name__ == "__main__":
    asyncio.run(main())
