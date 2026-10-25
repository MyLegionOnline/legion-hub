import asyncio
import hashlib
import logging
import time
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.api import deps
from app.core.llm_errors import extract_opt_in_url
from app.core.llm_providers import (
    OPENCODE_GO,
    OPENCODE_GO_BASE_URL,
    OPENCODE_GO_CATALOG,
    WIRE_CHAT,
    WIRE_MESSAGES,
    WIRE_RESPONSES,
    messages_headers,
    opencode_go_wire,
    openrouter_attribution_headers,
)
from app.core.reasoning import (
    get_cached_reasoning_support,
    openrouter_catalog_support,
    probe_reasoning_support,
    save_reasoning_support,
    save_reasoning_support_bulk,
)
from app.core.request_id import request_id_from_request
from app.core.secrets.crypto import SecretDecryptionError
from app.core.secrets.service import SecretResourceAccessDenied
from app.core.secrets.runtime import resolve_for_use, resolve_owned_llm_for_validation
from app.core.secrets.store import SecretStoreError
from app.db.session import get_session
from app.models.secret import SecretPurpose, SecretResourceType
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()

def _bearer(models_url: str) -> dict[str, str]:
    return {"models_url": models_url, "auth_header": "Authorization", "auth_prefix": "Bearer "}


# Where each provider lists its models. A connection with its own base URL
# lists `<base_url>/models` instead; for "openai_compatible" (any server that
# speaks the OpenAI API: vLLM, LM Studio, llama.cpp...) that is the normal case.
PROVIDER_CONFIGS = {
    "openrouter": _bearer("https://openrouter.ai/api/v1/models"),
    "openai": _bearer("https://api.openai.com/v1/models"),
    "openai_compatible": _bearer("http://127.0.0.1:8000/v1/models"),
    "together": _bearer("https://api.together.xyz/v1/models"),
    "groq": _bearer("https://api.groq.com/openai/v1/models"),
    "ollama": {"models_url": "http://localhost:11434/api/tags", "auth_header": None, "auth_prefix": None},
    OPENCODE_GO: _bearer(f"{OPENCODE_GO_BASE_URL}/models"),
}


class ValidateRequest(BaseModel):
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    secret_binding_id: UUID | None = None
    secret_id: UUID | None = None
    resource_id: UUID | None = None
    # Stage 9 (plan §10.2/§10.4): bypass the 24h availability-probe cache —
    # the "I enabled it — check again" button on the MODEL_OPT_IN_REQUIRED
    # card sends this after the user flipped the workspace toggle.
    refresh_probe: bool = False


class ModelInfo(BaseModel):
    id: str
    name: str
    context_length: int | None = None
    input_price: float | None = None   # USD per 1M tokens
    output_price: float | None = None  # USD per 1M tokens
    cache_read_price: float | None = None   # USD per 1M cached prompt tokens read
    cache_write_price: float | None = None  # USD per 1M prompt tokens written to cache
    # Stage 9 (plan §10.1) — per-model availability, measured by a 1-token
    # probe, not trusted from the catalog listing (§0.8: GET /v1/models lists
    # deepseek-v4-flash as healthy while every call returns 403 RegionError).
    available: bool = True              # False = the gateway refused (region/data/plan/broken model)
    unavailable_reason: str | None = None   # "region_opt_in" | "data_policy_opt_in" | "plan" | "unsupported"
    opt_in_url: str | None = None       # link from the RegionError body, if it had one
    # Stage 9 (plan §10.1) — reasoning capability flags for the picker UI
    # (stage 8 reads them from this response).  Sourced from the R3
    # `model_capability` cache; never probed live inside /validate.
    supports_reasoning: bool = False
    reasoning_controllable: bool = False


class ValidateResponse(BaseModel):
    valid: bool
    authenticated: bool = False
    models: list[ModelInfo]
    error: str | None = None
    # Set when the key is accepted but the provider still refused something —
    # the model list is usable, and the user needs to know what was refused.
    warning: str | None = None


class MediaModelInfo(BaseModel):
    id: str
    name: str
    supports_references: bool = False
    supports_size: bool = False
    supports_resolution: bool = False
    supports_aspect_ratio: bool = False


class MediaModelsResponse(BaseModel):
    valid: bool
    models: list[MediaModelInfo] = []
    error: str | None = None


ANTHROPIC_MODELS = [
    ModelInfo(id="claude-opus-4-8",          name="Claude Opus 4.8",    context_length=200000, input_price=15.0,  output_price=75.0,  cache_read_price=1.5,  cache_write_price=18.75),
    ModelInfo(id="claude-sonnet-4-6",        name="Claude Sonnet 4.6",  context_length=200000, input_price=3.0,   output_price=15.0,  cache_read_price=0.3,  cache_write_price=3.75),
    ModelInfo(id="claude-haiku-4-5-20251001",name="Claude Haiku 4.5",   context_length=200000, input_price=0.8,   output_price=4.0,   cache_read_price=0.08, cache_write_price=1.0),
    ModelInfo(id="claude-3-5-sonnet-20241022",name="Claude 3.5 Sonnet", context_length=200000, input_price=3.0,   output_price=15.0,  cache_read_price=0.3,  cache_write_price=3.75),
    ModelInfo(id="claude-3-opus-20240229",   name="Claude 3 Opus",      context_length=200000, input_price=15.0,  output_price=75.0,  cache_read_price=1.5,  cache_write_price=18.75),
]


def _to_price_per_million(v) -> float | None:
    """OpenRouter pricing is per token; convert to per-million."""
    try:
        f = float(v)
        return round(f * 1_000_000, 6) if f else None
    except (TypeError, ValueError):
        return None


_OPENCODE_GO_NAMES = {entry.id: entry.name for entry in OPENCODE_GO_CATALOG}


def _opencode_go_models(data: dict) -> list[ModelInfo]:
    """Live availability from /v1/models, display names from the catalog.

    Models the gateway serves only over the OpenAI Responses API are dropped:
    the Hub has no transport for that endpoint, so offering them would produce
    agents that fail on their first turn.
    """
    models = [
        ModelInfo(id=mid, name=_OPENCODE_GO_NAMES.get(mid, mid))
        for mid in (entry.get("id", "") for entry in data.get("data", []))
        if mid and opencode_go_wire(mid) != WIRE_RESPONSES
    ]
    return sorted(models, key=lambda m: m.id)


# Stage 9 (plan §10.2): every chat/messages model gets one 1-token probe —
# the listing is not availability (§0.8: models the account cannot call are
# listed as healthy).  Concurrency is capped so validating a key does not
# look like a burst of traffic to the gateway.
_OPENCODE_GO_PROBE_CONCURRENCY = 6
_OPENCODE_GO_PROBE_TIMEOUT_S = 30.0
_OPENCODE_GO_CATALOG_TIMEOUT_S = 15.0

# Probe outcomes are cached for 24 hours (models are unblocked rarely).  The
# cache is in-process and keyed by a hash of the api key: different keys see
# different plans/regions, and the raw key never sits in a long-lived dict.
# `refresh_probe` on the request bypasses it (the opt-in card's button).
_PROBE_CACHE_TTL_S = 24 * 3600
_PROBE_CACHE: dict[tuple[str, str, str], tuple[float, dict]] = {}


def _probe_cache_key(root: str, api_key: str | None, model_id: str) -> tuple[str, str, str]:
    digest = hashlib.sha256((api_key or "").encode()).hexdigest()
    return (root, digest, model_id)


def _cached_probe_outcome(root: str, api_key: str | None, model_id: str) -> dict | None:
    entry = _PROBE_CACHE.get(_probe_cache_key(root, api_key, model_id))
    if entry is None:
        return None
    stored_at, outcome = entry
    if time.monotonic() - stored_at > _PROBE_CACHE_TTL_S:
        _PROBE_CACHE.pop(_probe_cache_key(root, api_key, model_id), None)
        return None
    return outcome


def _store_probe_outcome(
    root: str, api_key: str | None, model_id: str, outcome: dict
) -> None:
    # Opportunistic pruning: keep the dict bounded without a background task.
    if len(_PROBE_CACHE) > 512:
        now = time.monotonic()
        for key, (stored_at, _outcome) in list(_PROBE_CACHE.items()):
            if now - stored_at > _PROBE_CACHE_TTL_S:
                _PROBE_CACHE.pop(key, None)
    _PROBE_CACHE[_probe_cache_key(root, api_key, model_id)] = (time.monotonic(), outcome)


# Two 403 error types mean the same thing: the key was accepted but the model
# is locked behind a browser-side workspace toggle.  Both carry the working
# link (with the user's wrk_… id) inside the message — extracted, never
# invented (plan §10.2).
_OPT_IN_ERRORS = {"RegionError": "region_opt_in", "DataPolicyError": "data_policy_opt_in"}


async def _probe_model(
    client: httpx.AsyncClient,
    *,
    root: str,
    api_key: str | None,
    model_id: str,
) -> dict:
    """One 1-token call per model: the only honest source of availability.

    Returns the ``model_copy(update=…)`` fields for `ModelInfo`, or
    ``{"auth_failed": True}`` on 401 — a dead key is a provider-level verdict,
    not a per-model outcome, and must not be cached as one.

    Transport-level failures (timeout, connection reset) return an *optimistic*
    outcome and are not cached by the caller: a transient blip must not lock a
    model out of the picker for 24 hours.
    """
    body = {"model": model_id, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
    wire = opencode_go_wire(model_id)
    if wire == WIRE_RESPONSES:
        # Filtered out of the catalog list already; defensive if the gateway
        # starts listing new Responses-only families.
        return {"available": False, "unavailable_reason": "unsupported", "opt_in_url": None}
    try:
        if wire == WIRE_CHAT:
            r = await client.post(
                f"{root}/chat/completions",
                headers={"Authorization": f"Bearer {api_key or ''}", "content-type": "application/json"},
                json=body,
            )
        else:
            r = await client.post(
                f"{root}/messages",
                headers=messages_headers(OPENCODE_GO, api_key),
                json=body,
            )
    except httpx.HTTPError:
        # Unknown, not unavailable — the model stays selectable.
        return {"available": True, "unavailable_reason": None, "opt_in_url": None, "transient": True}

    if 200 <= r.status_code < 300:
        return {"available": True, "unavailable_reason": None, "opt_in_url": None}
    if r.status_code == 401:
        # Key-level verdict: reported by the caller, never cached per-model.
        return {"auth_failed": True, "message": _gateway_error_message(r)}

    payload = {}
    if r.headers.get("content-type", "").startswith("application/json"):
        try:
            parsed = r.json()
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    err = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    if err.get("type") in _OPT_IN_ERRORS:
        return {
            "available": False,
            "unavailable_reason": _OPT_IN_ERRORS[err["type"]],
            "opt_in_url": extract_opt_in_url(err.get("message", "")),
        }
    if r.status_code == 400:
        # The gateway lists models it cannot serve ("400 Unsupported model
        # mimo-v2-omni", §0.9) — re-badged so the probe can drop them without
        # a hardcoded catalog exclusion.
        return {"available": False, "unavailable_reason": "unsupported", "opt_in_url": None}
    return {"available": False, "unavailable_reason": "plan", "opt_in_url": None}


def _gateway_error_message(response: httpx.Response) -> str:
    """The provider's own explanation, if it sent a JSON error body."""
    try:
        payload = response.json()
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"].strip()[:200]
    if isinstance(error, str):
        return error.strip()[:200]
    if isinstance(payload.get("message"), str):
        return payload["message"].strip()[:200]
    return ""


# Bound live probing so one /providers/validate call can never sit longer
# than the frontend's proxy read timeout (nginx default 60s): each probe is
# up to 3 sequential calls at up to _PROBE_TIMEOUT_S (90s) apiece, so an
# unbounded loop over a provider with a large catalog (Together.ai, Groq, a
# self-hosted vLLM listing) would hang the request well past that. The cap
# and the overall budget are deliberately conservative — models past either
# limit simply keep the False/False defaults and get probed on a later
# validation or by the lazy re-fill path, same as an ordinary cache miss.
_MAX_LIVE_PROBES = 8
_PROBE_CONCURRENCY = 4
_PROBE_BATCH_BUDGET_S = 40.0


async def _probe_and_save(
    *, provider: str, model_id: str, api_key: str, base_url: str | None,
    db: AsyncSession, sem: asyncio.Semaphore,
):
    async with sem:
        support = await probe_reasoning_support(
            provider=provider, model=model_id, api_key=api_key, base_url=base_url,
        )
    await save_reasoning_support(db, provider=provider, model=model_id, support=support)
    return support


async def _annotate_reasoning_support(
    models: list[ModelInfo], db: AsyncSession | None, *, provider: str,
    api_key: str | None = None, base_url: str | None = None,
) -> None:
    """Fill supports_reasoning / reasoning_controllable from the R3 cache.

    Cache-first by design (plan §10.1 + §4.7): the `model_capability` table is
    filled by the stage-3 probe (7-day TTL) and its lazy re-fill paths.  Up to
    ``_MAX_LIVE_PROBES`` models missing from the cache are probed live (2–3
    minimal calls each, ``_PROBE_CONCURRENCY`` at a time, within an overall
    ``_PROBE_BATCH_BUDGET_S`` budget) so the stage-8 selector is visible on
    the first validation rather than after a second one.  Probing failures —
    including a 401 that means the key itself is rejected — are logged and
    leave the False/False defaults, the honest answer for "cannot measure
    right now".

    Provider-neutral since stage OR2 (§4B.2): it used to hardcode
    ``provider=OPENCODE_GO`` and was reachable only from the OpenCode Go
    branch, so every OpenRouter model came back False/False and stage 8's
    selector would have been invisible for all of them.
    """
    if db is None:
        return
    to_probe: list[ModelInfo] = []
    for m in models:
        try:
            cached = await get_cached_reasoning_support(db, provider=provider, model=m.id)
        except Exception:
            cached = None
        if cached is not None:
            m.supports_reasoning = cached.supported
            m.reasoning_controllable = cached.controllable
        elif api_key:
            to_probe.append(m)

    if not to_probe:
        return

    batch = to_probe[:_MAX_LIVE_PROBES]
    sem = asyncio.Semaphore(_PROBE_CONCURRENCY)
    try:
        results = await asyncio.wait_for(
            asyncio.gather(
                *(
                    _probe_and_save(
                        provider=provider, model_id=m.id, api_key=api_key,
                        base_url=base_url, db=db, sem=sem,
                    )
                    for m in batch
                ),
                return_exceptions=True,
            ),
            timeout=_PROBE_BATCH_BUDGET_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "reasoning probe batch for %s timed out after %.0fs — "
            "remaining models stay unmeasured for this validation",
            provider, _PROBE_BATCH_BUDGET_S,
        )
        return

    auth_failed = False
    for m, result in zip(batch, results):
        if isinstance(result, BaseException):
            if getattr(result, "status_code", None) == 401:
                auth_failed = True
            else:
                logger.warning(
                    "reasoning probe failed for %s/%s during validation: %s",
                    provider, m.id, result,
                )
            continue
        m.supports_reasoning = result.supported
        m.reasoning_controllable = result.controllable
    if auth_failed:
        logger.warning(
            "reasoning probe: %s rejected the API key (401) on at least one model",
            provider,
        )


async def _annotate_openrouter_reasoning(
    models: list[ModelInfo], entries: list, db: AsyncSession | None
) -> None:
    """Reasoning flags for OpenRouter — straight out of the catalog (§4B.2).

    Measured in stage OR1 (§4A.5 q6): ``/api/v1/models`` publishes a
    ``reasoning`` object per model whose ``mandatory`` flag is the same fact
    the gateway enforces at request time, so both flags come for free from a
    listing this endpoint already fetches — no probe, no extra round-trip.

    The same answers are mirrored into `model_capability` in one statement so
    the hot-path gate (`reasoning_fields_for_call`) can read them without a
    network call: the router's forced off-switch (stage 5) must not depend on
    the user having opened the agent form first.
    """
    support_by_id = {
        entry.get("id"): openrouter_catalog_support(entry)
        for entry in entries
        if isinstance(entry, dict) and entry.get("id")
    }
    for m in models:
        support = support_by_id.get(m.id)
        if support is not None:
            m.supports_reasoning = support.supported
            m.reasoning_controllable = support.controllable
    if db is None:
        return
    try:
        await save_reasoning_support_bulk(
            db,
            provider="openrouter",
            entries={mid: s for mid, s in support_by_id.items() if s.supported},
        )
    except Exception:
        # Caching is an optimisation for later calls; a validation that shows
        # the right flags must not fail because the mirror write did.
        logger.warning("openrouter reasoning capability cache write failed", exc_info=True)


async def _validate_opencode_go(
    api_key: str | None,
    base_url: str | None,
    db: AsyncSession | None = None,
    *,
    refresh_probe: bool = False,
) -> ValidateResponse:
    """Validate an OpenCode Go key and list the models the Hub can drive.

    The catalog at ``/v1/models`` is public, so a 200 there says nothing about
    the key.  Stage 9 (plan §10.2) probes **every** chat/messages model with
    one 1-token call instead of extrapolating from three: the listing lies
    about availability (§0.8), so per-model truth is the only truth.

    The gateway separates the two failures cleanly, and so must this: a bad or
    missing key is **401** (``AuthError``) on any model, while **403** means
    the key itself was accepted and the account was refused *this model* — a
    plan, billing or region limit.  Reporting a 403 as a dead key sends the
    user to rotate a credential that was never the problem.  403/400 on
    individual models therefore never invalidate the key; they are recorded
    per-model as ``available=False`` with the reason (and the working opt-in
    link when the gateway sent one).
    """
    root = (base_url or OPENCODE_GO_BASE_URL).rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # One client for the whole validation: the catalog keeps its own tighter
    # per-request timeout, the probes get the longer one.
    async with httpx.AsyncClient(timeout=_OPENCODE_GO_PROBE_TIMEOUT_S) as client:
        catalog = await client.get(
            f"{root}/models", headers=headers, timeout=_OPENCODE_GO_CATALOG_TIMEOUT_S
        )
        if catalog.status_code != 200:
            return ValidateResponse(valid=False, models=[], error="Provider validation failed")
        models = _opencode_go_models(catalog.json())
        if not models:
            return ValidateResponse(
                valid=False, models=[],
                error="OpenCode Go returned no model that the Hub knows how to call.",
            )

        auth_failed = False
        auth_message: list[str] = []
        semaphore = asyncio.Semaphore(_OPENCODE_GO_PROBE_CONCURRENCY)

        async def probe_one(model: ModelInfo) -> None:
            nonlocal auth_failed
            outcome = None if refresh_probe else _cached_probe_outcome(root, api_key, model.id)
            if outcome is None:
                async with semaphore:
                    outcome = await _probe_model(
                        client, root=root, api_key=api_key, model_id=model.id
                    )
                if not (outcome.get("auth_failed") or outcome.get("transient")):
                    _store_probe_outcome(root, api_key, model.id, outcome)
            if outcome.get("auth_failed"):
                auth_failed = True
                if outcome.get("message"):
                    auth_message.append(str(outcome["message"]))
                return
            model.available = outcome.get("available", True)
            model.unavailable_reason = outcome.get("unavailable_reason")
            model.opt_in_url = outcome.get("opt_in_url")

        await asyncio.gather(*(probe_one(m) for m in models))

        if auth_failed:
            return ValidateResponse(
                valid=False, models=[],
                error=(
                    "OpenCode Go rejected the key: "
                    f"{auth_message[0] if auth_message else 'the key is invalid or revoked'} (401)."
                ),
            )
        await _annotate_reasoning_support(
            models, db, provider=OPENCODE_GO, api_key=api_key, base_url=base_url,
        )

        if not any(m.available for m in models):
            # The key authenticated (a 401 would have fired otherwise).  Offer
            # the catalog so the user can see what was refused and why.
            return ValidateResponse(
                valid=True, authenticated=True, models=models,
                warning=(
                    "The key was accepted, but no model responds — this is a "
                    "limitation of the account's plan, balance or region, not of the key."
                ),
            )
        return ValidateResponse(valid=True, authenticated=True, models=models)


def _parse_models(provider: str, data: dict) -> list[ModelInfo]:
    models = []
    if provider == OPENCODE_GO:
        return _opencode_go_models(data)
    if provider == "ollama":
        for m in data.get("models", []):
            mid = m.get("name", "")
            models.append(ModelInfo(id=mid, name=mid))
    elif provider == "openrouter":
        for m in data.get("data", []):
            mid = m.get("id", "")
            name = m.get("name") or mid
            pricing = m.get("pricing") or {}
            models.append(ModelInfo(
                id=mid,
                name=name,
                context_length=m.get("context_length"),
                input_price=_to_price_per_million(pricing.get("prompt")),
                output_price=_to_price_per_million(pricing.get("completion")),
                cache_read_price=_to_price_per_million(pricing.get("input_cache_read")),
                cache_write_price=_to_price_per_million(pricing.get("input_cache_write")),
            ))
    else:
        for m in data.get("data", []):
            mid = m.get("id", "")
            name = m.get("name") or mid
            models.append(ModelInfo(id=mid, name=name))
    return sorted(models, key=lambda m: m.id)


async def _resolve_validation_api_key(
    req: ValidateRequest,
    request: Request,
    db: AsyncSession,
    current_user: User,
    purpose: SecretPurpose = SecretPurpose.LLM_PRIMARY,
) -> str | None:
    """Resolve the API key a validation/introspection call should use.

    Shared by ``/validate`` and ``/image-models`` — same binding/secret_id
    contract, only the binding purpose differs by caller.
    """
    secret_value = None
    if req.api_key and req.secret_binding_id is None:
        raise HTTPException(
            status_code=422,
            detail="Provider validation requires a SecretStore binding",
        )
    if req.secret_binding_id is not None and req.secret_id is not None:
        raise HTTPException(
            status_code=422,
            detail="Provider validation accepts either a binding or one saved secret",
        )
    if req.secret_binding_id is not None:
        if req.resource_id is None:
            raise HTTPException(status_code=422, detail="resource_id is required for secret validation")
        try:
            secret_value = await resolve_for_use(
                db=db,
                owner_id=current_user.id,
                resource_type=SecretResourceType.AGENT,
                resource_id=req.resource_id,
                purpose=purpose,
                binding_id=req.secret_binding_id,
                request_id=request_id_from_request(request),
                source_ip=request.client.host if request.client else None,
            )
        except (SecretStoreError, SecretResourceAccessDenied, SecretDecryptionError):
            raise HTTPException(status_code=409, detail="Provider secret is not available") from None
    elif req.secret_id is not None:
        try:
            secret_value = await resolve_owned_llm_for_validation(
                db=db,
                owner_id=current_user.id,
                secret_id=req.secret_id,
                request_id=request_id_from_request(request),
                source_ip=request.client.host if request.client else None,
            )
        except (SecretStoreError, SecretResourceAccessDenied, SecretDecryptionError):
            raise HTTPException(status_code=409, detail="Provider secret is not available") from None

    return secret_value.reveal() if secret_value is not None else None


@router.post("/validate", response_model=ValidateResponse)
async def validate_provider(
    req: ValidateRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    provider = req.provider.lower()
    api_key = await _resolve_validation_api_key(req, request, db, current_user)
    from app.core.chatgpt_credentials import PREFIX
    if provider == "openai_chatgpt":
        raise HTTPException(422, detail="Validate ChatGPT through its account-scoped model endpoint")
    if str(api_key or "").startswith(PREFIX):
        raise HTTPException(422, detail="A ChatGPT connection is not an API key")

    try:
        if provider == "anthropic":
            if not api_key or not api_key.startswith("sk-ant-"):
                return ValidateResponse(valid=False, models=[], error="Invalid Anthropic API key format")
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]},
                )
                if resp.status_code in (200, 400):
                    return ValidateResponse(valid=True, models=list(ANTHROPIC_MODELS))
                return ValidateResponse(valid=False, models=[], error="Provider authentication failed")

        if provider == OPENCODE_GO:
            return await _validate_opencode_go(
                api_key, req.base_url, db, refresh_probe=req.refresh_probe
            )

        cfg = PROVIDER_CONFIGS.get(provider)
        if not cfg:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

        models_url = req.base_url.rstrip("/") + "/models" if req.base_url else cfg["models_url"]
        headers = {}
        if cfg["auth_header"] and api_key:
            headers[cfg["auth_header"]] = cfg["auth_prefix"] + api_key

        async with httpx.AsyncClient(timeout=15) as client:
            # OpenRouter publishes its model catalog without authentication.
            # Validate the key first with a minimal authenticated completion;
            # only then is a 200 model list meaningful to the user.
            if provider == "openrouter":
                auth_response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        **headers,
                        **openrouter_attribution_headers("https://openrouter.ai"),
                    },
                    json={
                        "model": "openai/gpt-4o-mini",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "ping"}],
                    },
                )
                if auth_response.status_code == 401:
                    return ValidateResponse(
                        valid=False,
                        models=[],
                        error="The key is invalid or revoked (OpenRouter returned 401).",
                    )
                if auth_response.status_code < 200 or auth_response.status_code >= 300:
                    return ValidateResponse(
                        valid=False,
                        models=[],
                        error=f"OpenRouter did not confirm access (HTTP {auth_response.status_code}).",
                    )
            resp = await client.get(models_url, headers=headers)
            if resp.status_code != 200:
                return ValidateResponse(valid=False, models=[], error="Provider validation failed")
            data = resp.json()
            models = _parse_models(provider, data)
            # Stage OR2 (§4B.2): reasoning flags for the stage-8 selector.
            # OpenRouter answers them from the listing just fetched; every
            # other provider answers them from the R3 probe cache, if it has
            # anything to say.
            if provider == "openrouter":
                await _annotate_openrouter_reasoning(models, data.get("data") or [], db)
            else:
                await _annotate_reasoning_support(
                    models, db, provider=provider,
                    api_key=api_key, base_url=req.base_url,
                )
            return ValidateResponse(valid=True, authenticated=True, models=models)
    except HTTPException:
        raise
    except httpx.ConnectError:
        return ValidateResponse(valid=False, models=[], error="Connection refused — is the server running?")
    except Exception:
        return ValidateResponse(valid=False, models=[], error="Provider validation failed")


def _parse_image_model_caps(entries: list) -> list[MediaModelInfo]:
    """Parse OpenRouter's ``/images/models`` response into capability-tagged entries."""
    models: list[MediaModelInfo] = []
    for entry in entries:
        mid = entry.get("id", "")
        if not mid:
            continue
        caps = entry.get("supported_parameters") or entry.get("capabilities") or {}
        models.append(MediaModelInfo(
            id=mid,
            name=entry.get("name") or mid,
            supports_references="input_references" in caps,
            supports_size="size" in caps,
            supports_resolution="resolution" in caps,
            supports_aspect_ratio="aspect_ratio" in caps,
        ))
    return sorted(models, key=lambda m: m.id)


@router.post("/image-models", response_model=MediaModelsResponse)
async def list_image_models(
    req: ValidateRequest,
    request: Request,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(deps.get_current_user),
):
    """List OpenRouter image-generation models with their capability flags.

    Powers the Media agent configurator's model picker (media_references_and_size_plan
    follow-up): lets the UI restrict/badge model choices by whether they actually
    support ``input_references`` before a task tries to use them, instead of only
    discovering the mismatch from a failed generation.
    """
    provider = req.provider.lower()
    if provider != "openrouter":
        return MediaModelsResponse(
            valid=False, models=[],
            error="Image generation is only available via OpenRouter",
        )

    api_key = await _resolve_validation_api_key(
        req, request, db, current_user, purpose=SecretPurpose.LLM_MEDIA
    )
    base_url = (req.base_url or "https://openrouter.ai/api/v1").rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{base_url}/images/models",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            )
            if resp.status_code != 200:
                return MediaModelsResponse(valid=False, models=[], error="Provider validation failed")
            data = resp.json()
            entries = data if isinstance(data, list) else data.get("data", [])
            return MediaModelsResponse(valid=True, models=_parse_image_model_caps(entries))
    except HTTPException:
        raise
    except httpx.ConnectError:
        return MediaModelsResponse(valid=False, models=[], error="Connection refused — is the server running?")
    except Exception:
        return MediaModelsResponse(valid=False, models=[], error="Provider validation failed")
