"""Reasoning (thinking) support: what a model can do and how to ask for it.

Reasoning is a per-MODEL capability, not a per-provider one.  Measured on the
OpenCode Go gateway 27.08.2026: `deepseek-v4-flash-vision-exp` honours
`thinking:{type:"disabled"}` (reasoning_tokens 862→0, 13.6s→2.2s), while
`glm-5.3` on the same gateway ignores every known off-switch — eight variants
tried, reasoning_tokens stayed at 600-2000 in all of them.  Anything that
hardcodes "provider X takes parameter Y" is therefore wrong by construction.

This module is the single place that answers the two questions the rest of the
plan builds on — "can this (provider, model) pair reason" and "which request
parameter turns that reasoning on/off on the given wire".  The answer is
measured, not assumed: `probe_reasoning_support` calls the model with and
without the off-switch and the result is cached in the `model_capability`
table (7-day TTL).  The probe is lazy by contract — it runs at provider
validation (stage 9) and when a user first enables reasoning on an agent
(stage 4), NEVER in the hot path of a message turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import httpx

from app.core.llm_providers import (
    WIRE_CHAT,
    WIRE_MESSAGES,
    WIRE_RESPONSES,
    messages_headers,
    messages_text_and_thinking,
    messages_url,
    normalize_provider,
    openai_base_url,
    openrouter_attribution_headers,
    provider_wire,
)
from app.models.model_capability import CAPABILITY_TTL_DAYS, ModelCapability

logger = logging.getLogger(__name__)

Effort = Literal["off", "low", "medium", "high"]

OPENROUTER = "openrouter"

# What "off" and "level" are called on each wire.  The values were checked by
# requests; models that silently ignore the parameter are caught by a probe (below).
_CHAT_EFFORT = {"off": "none", "low": "low", "medium": "medium", "high": "high"}

# OpenRouter's own disable body (§4A.5 q1).  Measured 05.09.2026: this,
# `reasoning_effort:"none"` and `reasoning:{effort:"none"}` are the same flag
# after the gateway normalises them — the unified block is preferred because
# 302 catalog models declare `reasoning` against 159 declaring
# `reasoning_effort`.
OPENROUTER_OFF_SWITCH: dict[str, Any] = {"reasoning": {"enabled": False}}

_MESSAGES_BUDGET = {"low": 1024, "medium": 4096, "high": 16384}

# On the messages wire `thinking.budget_tokens` must be strictly less than
# the request's `max_tokens`, otherwise the gateway answers 400.  The formula for all
# messages calls: budget = min(budget, max_tokens - 512), and if the result
# is less than 1024 — reasoning is not enabled at all.
_BUDGET_HEADROOM = 512
_MIN_BUDGET = 1024


@dataclass(frozen=True)
class ReasoningSupport:
    """What the Hub believes one (provider, model) pair can do."""

    supported: bool          # the model reasons at all
    controllable: bool       # the switch actually works (confirmed by a probe)
    off_switch: dict[str, Any] | None   # request body that turns reasoning off


def reasoning_request_fields(
    provider: str | None,
    model: str | None,
    wire: str,
    effort: Effort | None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Extra request fields that put the model at the requested effort level.

    ``effort=None`` means "provider default" — returns {} so that an agent
    configured before this feature existed keeps behaving exactly as it did
    (decision 0.10 p.4: nothing is sent unless the user asked for it).

    ``max_tokens`` (messages wire only) clamps ``thinking.budget_tokens``
    under the call's own cap: the gateway answers 400 when the budget reaches
    max_tokens, and a budget squeezed below the floor is not requested at all
    — the call proceeds at the provider default instead of failing.
    """
    if effort is None:
        return {}
    if wire == WIRE_MESSAGES:
        if effort == "off":
            return {"thinking": {"type": "disabled"}}
        budget = _MESSAGES_BUDGET[effort]
        if max_tokens is not None:
            budget = min(budget, max_tokens - _BUDGET_HEADROOM)
            if budget < _MIN_BUDGET:
                return {}
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    if wire == WIRE_RESPONSES:
        # No Responses transport exists in the Hub yet (internal design notes);
        # guessing a chat-shaped knob onto that wire would be exactly the
        # hardcoding this module exists to prevent.  Stage A4 measures it.
        return {}
    # chat wire.  The knob is per-GATEWAY here: OpenCode Go was measured on
    # `reasoning_effort` (§4.6), OpenRouter on its own unified `reasoning: {…}`
    # block (§4A.5 q1).  Which models *honour* it stays a per-model question —
    # this branch only decides what to send.  Providers nobody measured
    # (openai, groq, together, deepseek, ollama) keep the old branch untouched.
    if normalize_provider(provider) == OPENROUTER:
        if effort == "off":
            return dict(OPENROUTER_OFF_SWITCH)
        # Effort names outside the model's declared `supported_efforts` are
        # normalised by the gateway — measured on deepseek-v4-flash, which
        # declares only xhigh/high yet answered 200 to "low" and "medium".
        return {"reasoning": {"effort": effort}}
    return {"reasoning_effort": _CHAT_EFFORT[effort]}


def openrouter_catalog_support(entry: dict) -> ReasoningSupport:
    """Reasoning capability of one OpenRouter catalog entry — no network call.

    OpenRouter publishes per model a ``reasoning`` object
    ``{mandatory, default_enabled, supported_efforts, default_effort}``; a
    model without it does not reason.  ``mandatory`` is the same fact the
    gateway enforces at request time — it answers
    ``400 "Reasoning is mandatory for this endpoint and cannot be disabled."``
    to every off-switch — so ``controllable = not mandatory`` is not an
    inference but the gateway's own contract.  Measured 05.09.2026 on seven
    models across four vendors, 7/7 agreement (plan §4A.5).

    This is why OpenRouter needs no per-model probe: the two flags stages 8
    and 5 read are already in a listing `/validate` fetches anyway.
    """
    declared = entry.get("reasoning")
    if not isinstance(declared, dict):
        supported = "reasoning" in (entry.get("supported_parameters") or [])
        # A model that lists the parameter but publishes no reasoning object
        # is unmeasured, not uncontrollable — say so honestly.
        return ReasoningSupport(supported=supported, controllable=False, off_switch=None)
    controllable = not bool(declared.get("mandatory"))
    return ReasoningSupport(
        supported=True,
        controllable=controllable,
        off_switch=dict(OPENROUTER_OFF_SWITCH) if controllable else None,
    )


# ------------------------------------------------------- per-request plumbing

# Sentinel: "take whatever the agent itself is configured for".  Distinct from
# None, which means "send nothing regardless of the agent's setting".
_UNSET: Any = "unset"


def resolve_effort(
    llm_config: dict | None, override: Effort | None | Any = _UNSET
) -> Effort | None:
    """Which effort level this particular call must ask for.

    Three modes, and the gap between the first two is deliberate:

    * ``override=_UNSET``  — the agent's own ``llm_config.reasoning.effort``.
      Absent key means None, which means nothing is sent (decision 0.10 p.4).
    * ``override=None``    — send nothing regardless of the agent's setting.
      This is the mode for the Hub's *own* utility calls (summariser, plan
      JSON): the user configured a thinking budget for the agent's work, not
      for a 512-token summary the agent never sees.
    * ``override="off"``   — force the off-switch on top of the setting.
      Exactly one caller: the router (stage 5).
    """
    if override is not _UNSET:
        return override
    block = (llm_config or {}).get("reasoning")
    if not isinstance(block, dict):
        return None
    effort = block.get("effort")
    return effort if effort in _CHAT_EFFORT else None


def show_reasoning_in_chat(llm_config: dict | None) -> bool:
    """Whether this agent's thinking is rendered as a chat block (0.10 p.3).

    Default is on: the key is absent for every agent created before the
    feature existed, and the decision was "shown unless turned off".
    """
    block = (llm_config or {}).get("reasoning")
    if not isinstance(block, dict):
        return True
    return block.get("show_in_chat") is not False


def apply_reasoning_body(
    body: dict,
    llm_config: dict | None,
    provider: str | None,
    model: str | None,
    *,
    override: Effort | None | Any = _UNSET,
    fields: dict[str, Any] | None = None,
) -> None:
    """Merge reasoning fields into an Anthropic-shaped /messages request body.

    Reads ``max_tokens`` out of ``body`` itself for the budget clamp, so no
    call site can forget to pass it: the gateway answers 400 when
    ``thinking.budget_tokens`` reaches ``max_tokens``, and §0.7 is the same
    trap seen from the other side.

    ``fields`` lets a caller that resolved them itself pass them in rather
    than have them recomputed here.
    """
    if fields is None:
        fields = gated_reasoning_fields(
            provider, model, WIRE_MESSAGES,
            resolve_effort(llm_config, override),
            max_tokens=body.get("max_tokens"),
        )
    if fields:
        body.update(fields)


def apply_reasoning_chat(
    create_extra: dict,
    llm_config: dict | None,
    provider: str | None,
    model: str | None,
    *,
    override: Effort | None | Any = _UNSET,
    fields: dict[str, Any] | None = None,
) -> None:
    """Merge reasoning fields into the ``extra_body`` of a Chat Completions call.

    Two rules, both learned the hard way:

    1. **Merge, never assign.**  ``create_extra["extra_body"]`` already carries
       ``{"session_id": …}`` for OpenRouter session logging at three sites
       (§0.11).  ``create_extra["extra_body"] = fields`` would silently switch
       that logging off — for OpenRouter agents specifically, i.e. exactly the
       provider this stage exists for.
    2. **Everything goes through ``extra_body``, never as a named kwarg** —
       even ``reasoning_effort``, which recent SDK versions do know.  One rule
       survives an SDK upgrade in either direction, and OpenRouter's own
       ``reasoning: {...}`` block is not an SDK parameter at all.

    Never touches ``max_tokens``: on the chat wire a cap together with
    reasoning is banned outright (§0.7, reproduced on OpenRouter in §4A.5 q7).
    """
    if fields is None:
        fields = gated_reasoning_fields(
            provider, model, WIRE_CHAT, resolve_effort(llm_config, override)
        )
    if not fields:
        return
    extra = create_extra.get("extra_body")
    if not isinstance(extra, dict):
        extra = {}
        create_extra["extra_body"] = extra
    extra.update(fields)


# --------------------------------------------------- off-switch gate (§4B.5)
#
# In-process mirror of `model_capability`, so a request-building helper can ask
# "may I disable this model?" without an await and without a database round
# trip on every turn.  Written by the cache helpers below and warmed once at
# startup; an empty snapshot is a safe state, not a broken one — see
# `off_switch_allowed`.
_CAPABILITY_SNAPSHOT: dict[tuple[str, str], ReasoningSupport] = {}

# Gateways measured to *reject* an off-switch the model cannot honour, rather
# than ignore it.  OpenRouter answers `400 "Reasoning is mandatory for this
# endpoint and cannot be disabled."` (§4A.5 q1); OpenCode Go silently ignores
# the field (§0.6), so gating it there would remove a switch that costs
# nothing and sometimes works.
_GATED_PROVIDERS = {OPENROUTER}

# Pairs whose off-switch was dropped, so the log line is written once per
# (provider, model) rather than on every turn.
_OFF_SWITCH_DENIED: set[tuple[str, str]] = set()


def remember_reasoning_support(provider: str | None, model: str, support: ReasoningSupport) -> None:
    """Publish one capability answer to the in-process snapshot."""
    _CAPABILITY_SNAPSHOT[(normalize_provider(provider), model)] = support


def known_reasoning_support(provider: str | None, model: str | None) -> ReasoningSupport | None:
    """Capability of (provider, model) if this process already learned it."""
    return _CAPABILITY_SNAPSHOT.get((normalize_provider(provider), model or ""))


def off_switch_allowed(provider: str | None, model: str | None) -> bool:
    """May this call ask the model to stop reasoning?

    Only OpenRouter is gated, and only because it was measured to answer 400
    instead of ignoring an impossible request (§4A.5 q1): on that gateway an
    unknown or mandatory-reasoning model must be left at its default, or every
    turn of that agent fails.  Everywhere else the switch is free to send —
    the worst case there is that the model ignores it, which is the state we
    are already in.

    An empty snapshot therefore degrades to "do not disable on OpenRouter",
    i.e. exactly today's behaviour, never to a broken request.
    """
    if normalize_provider(provider) not in _GATED_PROVIDERS:
        return True
    support = known_reasoning_support(provider, model)
    return bool(support and support.controllable)


def gated_reasoning_fields(
    provider: str | None,
    model: str | None,
    wire: str,
    effort: Effort | None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """`reasoning_request_fields` with the off-switch gated by §4B.5.

    Only ``off`` is gated: levels are accepted by every reasoning model, and a
    model that does not reason ignores the fields silently (§4A.5 q5).
    """
    if effort == "off" and not off_switch_allowed(provider, model):
        key = (normalize_provider(provider), model or "")
        if key not in _OFF_SWITCH_DENIED:
            _OFF_SWITCH_DENIED.add(key)
            logger.info(
                "reasoning off-switch not sent for %s/%s: this model reasons "
                "unconditionally (or its capability is not measured yet) — the "
                "call proceeds at the provider default",
                key[0] or "?", key[1] or "?",
            )
        return {}
    return reasoning_request_fields(provider, model, wire, effort, max_tokens=max_tokens)


async def warm_reasoning_capability_snapshot(db) -> int:
    """Load `model_capability` into the in-process snapshot at startup.

    The table holds one row per measured or catalogued (provider, model) pair —
    hundreds, not millions — so this is one query, once per worker.  Without
    it the router's forced off-switch (stage 5) would stay inert until
    something else happened to read the same row.
    """
    from sqlalchemy import select

    rows = (await db.execute(select(ModelCapability))).scalars().all()
    now = datetime.now(timezone.utc)
    loaded = 0
    for row in rows:
        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= now:
            continue
        remember_reasoning_support(
            row.provider,
            row.model,
            ReasoningSupport(
                supported=row.supported,
                controllable=row.controllable,
                off_switch=row.off_switch,
            ),
        )
        loaded += 1
    logger.info("reasoning capability snapshot warmed: %d fresh rows", loaded)
    return loaded


def messages_reasoning_tokens(payload: dict) -> int:
    """Reasoning tokens reported by an Anthropic-shaped reply, 0 when absent.

    The messages wire carries no standard counter — Anthropic itself reports
    none, and gateways that do put it under either
    ``completion_tokens_details`` or ``output_tokens_details``.  Zero here
    means "the reply did not say", which for the router's log is the same
    thing as "nothing measurable was spent on thinking".
    """
    usage = payload.get("usage") or {}
    details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    try:
        return int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _pick(source: Any, *names: str) -> Any:
    """Read the first present field from an SDK object or a plain dict."""
    for name in names:
        if isinstance(source, dict):
            if name in source:
                return source[name]
        else:
            value = getattr(source, name, None)
            if value is not None:
                return value
    return None


def reasoning_from_chat_choice(message: Any, usage: Any) -> tuple[str, int]:
    """(thinking text, reasoning tokens) from an OpenAI-shaped reply.

    OpenCode Go returns the text in `message.reasoning_content`; OpenRouter
    uses `message.reasoning`.  Token count lives in
    `usage.completion_tokens_details.reasoning_tokens` on both.  Accepts both
    SDK objects and plain dicts — the probe reads raw JSON, call sites read
    the SDK.
    """
    text = _pick(message, "reasoning_content", "reasoning") or ""
    details = _pick(usage, "completion_tokens_details") if usage is not None else None
    tokens = _pick(details, "reasoning_tokens") or 0
    try:
        return str(text), int(tokens)
    except (TypeError, ValueError):
        return str(text), 0


# ------------------------------------------------------------------- probing


class ReasoningProbeError(RuntimeError):
    """The probe call itself failed — no conclusion can be drawn.

    Raised only when the *baseline* call fails: a model that cannot be called
    at all (403 RegionError, bad key, dead endpoint) has no measurable
    reasoning capability, and the caller must not cache a guess as an answer.
    Failures of the off-switch call are not errors — they are the measurement
    (a switch the model rejects or ignores means controllable=False).
    """

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(f"reasoning probe failed: HTTP {status_code}: {body[:300]}")


@dataclass(frozen=True)
class _ProbeReading:
    """What one probe call showed: did the model reason, and how many tokens."""

    reasoning: bool
    tokens: int


_PROBE_PROMPT = "Reply with the single word: ok"
# Anthropic-shaped /messages requires max_tokens; 2048 leaves room for a
# low-effort budget (1024) under the clamp formula plus a visible answer.
# The chat wire sends no max_tokens at all — 0.7: a cap on a reasoning chat
# model can swallow the whole budget into thinking, and the probe reads usage,
# not content, so an uncapped reply is both safe and honest.
_PROBE_MAX_TOKENS = 2048
_PROBE_TIMEOUT_S = 90.0


def _chat_reading(payload: dict) -> _ProbeReading:
    choices = payload.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    text, tokens = reasoning_from_chat_choice(message, payload.get("usage"))
    return _ProbeReading(reasoning=bool(text.strip()) or tokens > 0, tokens=tokens)


def _messages_reading(payload: dict) -> _ProbeReading:
    _text, thinking = messages_text_and_thinking(payload)
    blocks = payload.get("content") or []
    has_block = any(
        isinstance(b, dict) and b.get("type") == "thinking" for b in blocks
    )
    # The messages wire carries no cross-gateway reasoning_tokens counter; the
    # thinking block itself is the ground truth (measured 0.4: qwen3.7-plus
    # leads with a thinking block, {"thinking":{"type":"disabled"}} removes it).
    usage = payload.get("usage") or {}
    details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    try:
        tokens = int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        tokens = 0
    return _ProbeReading(reasoning=has_block or bool(thinking.strip()), tokens=tokens)


def _wire_reading(wire: str, payload: dict) -> _ProbeReading:
    if wire == WIRE_MESSAGES:
        return _messages_reading(payload)
    return _chat_reading(payload)


def _probe_request(
    *,
    provider: str | None,
    model: str,
    wire: str,
    api_key: str | None,
    base_url: str | None,
    fields: dict[str, Any],
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """(url, headers, body) of one minimal probe call on the model's wire."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _PROBE_PROMPT}],
    }
    if wire == WIRE_MESSAGES:
        body["max_tokens"] = _PROBE_MAX_TOKENS
        body.update(fields)
        return messages_url(provider, base_url), messages_headers(provider, api_key), body
    body.update(fields)
    url = f"{openai_base_url(provider, base_url)}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key or ''}",
        "content-type": "application/json",
        **openrouter_attribution_headers(url),
    }
    return url, headers, body


async def _run_probe_call(
    client: httpx.AsyncClient,
    *,
    provider: str | None,
    model: str,
    wire: str,
    api_key: str | None,
    base_url: str | None,
    fields: dict[str, Any],
) -> dict[str, Any]:
    url, headers, body = _probe_request(
        provider=provider, model=model, wire=wire,
        api_key=api_key, base_url=base_url, fields=fields,
    )
    response = await client.post(url, headers=headers, json=body)
    if not 200 <= response.status_code < 300:
        raise ReasoningProbeError(response.status_code, response.text)
    return response.json()


async def probe_reasoning_support(
    *,
    provider: str | None,
    model: str,
    api_key: str | None,
    wire: str | None = None,
    base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> ReasoningSupport:
    """Measure whether (provider, model) reasons and honours the off-switch.

    The same minimal call, up to three times: baseline (no reasoning fields),
    with the off-switch, and — only when the baseline shows no reasoning —
    with thinking explicitly requested, so `supported` is True for opt-in
    reasoners too (measured 31.08.2026: minimax-m3 stays silent by default
    while qwen3.7-plus thinks by default).  The model is `controllable` when
    it reasons in some mode yet the off-switch call shows none (glm-5.3
    ignores every known switch — its probe returns controllable=False).

    Never call this from the hot path of a message turn: it costs two to
    three LLM round-trips.  It exists for provider validation (stage 9) and
    for the moment a user first enables reasoning on an agent (stage 4).
    """
    resolved_wire = provider_wire(provider, model) if wire is None else wire
    if resolved_wire == WIRE_RESPONSES:
        # No Responses transport in the Hub (internal design notes) — probing
        # that wire is stage A4's work, not a guess made here.
        return ReasoningSupport(supported=False, controllable=False, off_switch=None)

    call = {
        "provider": provider, "model": model, "wire": resolved_wire,
        "api_key": api_key, "base_url": base_url,
    }
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S)
    try:
        baseline = _wire_reading(
            resolved_wire, await _run_probe_call(client, fields={}, **call)
        )

        off_fields = reasoning_request_fields(provider, model, resolved_wire, "off")
        off: _ProbeReading | None = None
        try:
            off = _wire_reading(
                resolved_wire, await _run_probe_call(client, fields=off_fields, **call)
            )
        except ReasoningProbeError as exc:
            # A model that rejects the off-switch outright (400) is exactly as
            # uncontrollable as one that ignores it — measured, not assumed.
            logger.info(
                "reasoning probe: %s/%s rejected the off-switch (HTTP %s) — "
                "treating as uncontrollable", provider, model, exc.status_code,
            )

        reasons_when_asked = False
        if not baseline.reasoning:
            on_fields = reasoning_request_fields(
                provider, model, resolved_wire, "low",
                max_tokens=_PROBE_MAX_TOKENS if resolved_wire == WIRE_MESSAGES else None,
            )
            if on_fields:
                try:
                    asked = _wire_reading(
                        resolved_wire, await _run_probe_call(client, fields=on_fields, **call)
                    )
                    reasons_when_asked = asked.reasoning
                except ReasoningProbeError as exc:
                    logger.info(
                        "reasoning probe: %s/%s rejected thinking-on (HTTP %s) — "
                        "treating as not supported on request", provider, model, exc.status_code,
                    )

        supported = baseline.reasoning or reasons_when_asked
        controllable = supported and off is not None and not off.reasoning
        return ReasoningSupport(
            supported=supported,
            controllable=controllable,
            off_switch=off_fields if controllable else None,
        )
    finally:
        if own_client:
            await client.aclose()


# ---------------------------------------------------------------------- cache


async def get_cached_reasoning_support(
    db, *, provider: str, model: str
) -> ReasoningSupport | None:
    """Fresh cached probe result for (provider, model), or None."""
    provider = (provider or "").lower()
    row = await db.get(ModelCapability, (provider, model))
    if row is None:
        return None
    expires_at = row.expires_at
    if expires_at.tzinfo is None:  # legacy naive columns read back naive
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at <= datetime.now(timezone.utc):
        return None
    support = ReasoningSupport(
        supported=row.supported,
        controllable=row.controllable,
        off_switch=row.off_switch,
    )
    remember_reasoning_support(provider, model, support)
    return support


async def save_reasoning_support(
    db, *, provider: str, model: str, support: ReasoningSupport
) -> None:
    """Upsert a probe result with the 7-day TTL (plan 4.3)."""
    provider = (provider or "").lower()
    now = datetime.now(timezone.utc)
    row = await db.get(ModelCapability, (provider, model))
    if row is None:
        row = ModelCapability(provider=provider, model=model)
        db.add(row)
    row.supported = support.supported
    row.controllable = support.controllable
    row.off_switch = support.off_switch
    row.probed_at = now
    row.expires_at = now + timedelta(days=CAPABILITY_TTL_DAYS)
    await db.commit()
    remember_reasoning_support(provider, model, support)


async def save_reasoning_support_bulk(
    db, *, provider: str, entries: dict[str, ReasoningSupport]
) -> int:
    """Upsert many (provider, model) capability rows in one statement.

    Used by OpenRouter validation, where the answers come from the catalog
    rather than from probes (§4B.2): there are hundreds of models and one
    round-trip is the difference between "free" and "unusable".  The rows are
    what the hot-path gate (`reasoning_fields_for_call`) reads, which is why
    they are written eagerly instead of on first use — the router's forced
    off-switch (stage 5) must work without the user having opened any form.
    """
    if not entries:
        return 0
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    provider = (provider or "").lower()
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=CAPABILITY_TTL_DAYS)
    rows = [
        {
            "provider": provider,
            "model": model,
            "supported": support.supported,
            "controllable": support.controllable,
            "off_switch": support.off_switch,
            "probed_at": now,
            "expires_at": expires_at,
        }
        for model, support in entries.items()
        if len(model) <= 128
    ]
    if not rows:
        return 0
    statement = pg_insert(ModelCapability.__table__).values(rows)
    statement = statement.on_conflict_do_update(
        index_elements=[ModelCapability.__table__.c.provider, ModelCapability.__table__.c.model],
        set_={
            "supported": statement.excluded.supported,
            "controllable": statement.excluded.controllable,
            "off_switch": statement.excluded.off_switch,
            "probed_at": statement.excluded.probed_at,
            "expires_at": statement.excluded.expires_at,
        },
    )
    await db.execute(statement)
    await db.commit()
    for model, support in entries.items():
        if len(model) <= 128:
            remember_reasoning_support(provider, model, support)
    return len(rows)


async def reasoning_support(
    db,
    *,
    provider: str | None,
    model: str,
    api_key: str | None,
    base_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> ReasoningSupport:
    """Cache-first reasoning support: fresh row, else probe and store.

    For the lazy paths only (provider validation, first reasoning enablement).
    Raises ReasoningProbeError when the model cannot be called at all — the
    caller shows that error instead of caching a guess.
    """
    provider_key = (provider or "").lower()
    cached = await get_cached_reasoning_support(db, provider=provider_key, model=model)
    if cached is not None:
        return cached
    support = await probe_reasoning_support(
        provider=provider, model=model, api_key=api_key,
        base_url=base_url, client=client,
    )
    await save_reasoning_support(
        db, provider=provider_key, model=model, support=support
    )
    return support
