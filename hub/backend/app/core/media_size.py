"""Size resolution and post-processing for the media agent.

internal design notes, stage 3.  Responsibilities:
  - parse the three size syntaxes (exact pixels, tier, aspect ratio);
  - resolve per-image / task-level / agent-default priority (R6);
  - fetch and cache OpenRouter capability descriptors so we know which
    parameters a model actually accepts (R7/R9);
  - build the provider payload without illegal parameter combinations
    (size never sent together with aspect_ratio/resolution — API returns 400);
  - post-process the generated PNG to an exact pixel target when the provider
    cannot honour it natively (R7, always guaranteed by Hub);
  - record postprocessed / degraded metrics (R8).

Stage 4 extends ``image_gen.generate_image`` signature; stages 5–6 wire this
module into the proxy_session media-task loop.  Nothing calls this module yet —
behaviour of the running system is unchanged.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (internal design notes stage 0)
# ---------------------------------------------------------------------------

_MEDIA_IMAGE_MODELS_TTL_S = int(os.getenv("MEDIA_IMAGE_MODELS_TTL_S", "21600"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KNOWN_TIERS: frozenset[str] = frozenset({"512", "1K", "2K", "4K"})
_TIER_MAX_DIM: dict[str, int] = {"512": 512, "1K": 1024, "2K": 2048, "4K": 4096}
_SIZE_RE = re.compile(r"^(\d+)[xX](\d+)$")
_MIN_SIDE = 16
_MAX_SIDE = 8192
_MAX_AREA = 16 * 1024 * 1024

# Fallback aspect ratio when no size information is available at any priority level.
_DEFAULT_ASPECT_RATIO = "4:3"

# Tier chosen when we need to send resolution alongside a nearest-ratio fallback
# and no finer preference is available.
_DEFAULT_FALLBACK_TIER = "2K"

# Known fit modes.
_VALID_FIT_MODES: frozenset[str] = frozenset({"cover", "contain", "stretch"})

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SizeSpec:
    """Requested size at one priority level (per-image / task / agent / default)."""

    aspect_ratio: str | None = None   # e.g. "16:9"
    size: str | None = None           # "1000x100" exact or "2K" tier
    resolution: str | None = None     # "1K" | "2K" | "4K"
    fit: str = "cover"                # cover | contain | stretch


@dataclass
class SizeOutcome:
    """What actually happened when building and applying a size request."""

    requested: SizeSpec
    provider_params: dict             # what was sent in the API payload
    produced: tuple[int, int]         # actual pixel dimensions of the saved file
    postprocessed: bool               # fit_exact was applied after generation
    degraded: list[str] = field(default_factory=list)  # params that had to be dropped


# ---------------------------------------------------------------------------
# Size parsing
# ---------------------------------------------------------------------------


def parse_size(raw: str) -> tuple[int, int] | str:
    """Parse a size string into either an exact pixel tuple or a tier string.

    Accepts:
      - ``"1000x100"`` / ``"1000X100"`` → ``(1000, 100)``  (exact pixels)
      - ``"2K"`` / ``"4K"`` / ``"1K"`` / ``"512"``         → tier string passed through
    Raises:
      - ``ValueError`` for anything else, or for pixel dimensions outside the
        valid range (16–8192 per side, area ≤ 16 Mpx).
    """
    raw = str(raw or "").strip()
    if raw in _KNOWN_TIERS:
        return raw
    m = _SIZE_RE.match(raw)
    if m:
        w, h = int(m.group(1)), int(m.group(2))
        if w < _MIN_SIDE or h < _MIN_SIDE:
            raise ValueError(
                f"size sides must be at least {_MIN_SIDE} px: {raw!r}"
            )
        if w > _MAX_SIDE or h > _MAX_SIDE:
            raise ValueError(
                f"size sides must be at most {_MAX_SIDE} px: {raw!r}"
            )
        if w * h > _MAX_AREA:
            raise ValueError(
                f"size area exceeds 16 Mpx limit: {raw!r}"
            )
        return w, h
    raise ValueError(
        f"invalid size {raw!r}: expected WxH pixels (e.g. '1000x100') or a tier (e.g. '2K')"
    )


# ---------------------------------------------------------------------------
# Size resolution (priority chain)
# ---------------------------------------------------------------------------


def _extract_spec(source: dict[str, Any] | None) -> SizeSpec | None:
    """Try to build a SizeSpec from a generic config dict.

    Accepts both ``size``/``aspect_ratio`` keys (per-image item and agent
    config) and the ``default_size`` alias added by stage 7 frontend.
    Returns ``None`` if the source carries no size information at all.
    """
    if not source:
        return None
    size = source.get("size") or source.get("default_size") or None
    aspect_ratio = source.get("aspect_ratio") or None
    resolution = source.get("resolution") or None
    fit = str(source.get("fit") or "cover").strip() or "cover"
    if fit not in _VALID_FIT_MODES:
        fit = "cover"
    if size or aspect_ratio or resolution:
        return SizeSpec(
            aspect_ratio=str(aspect_ratio).strip() if aspect_ratio else None,
            size=str(size).strip() if size else None,
            resolution=str(resolution).strip() if resolution else None,
            fit=fit,
        )
    return None


def resolve_size(
    per_image: dict[str, Any] | None,
    task_spec: dict[str, Any] | None,
    agent_default: dict[str, Any] | None,
) -> SizeSpec:
    """Resolve the effective SizeSpec using the three-level priority chain (R6).

    Priority (highest first):
      1. ``per_image``   — fields from the ``submit_image_prompts`` tool item
                           (``aspect_ratio``, ``size``, ``fit``).
      2. ``task_spec``   — the whole ``Task.media_spec`` JSONB; the ``size``
                           sub-object is extracted automatically.
      3. ``agent_default`` — ``llm_config.media`` dict of the agent
                             (``aspect_ratio`` / ``default_size`` / ``fit``).
      4. Hardcoded ``"4:3"`` if none of the above provides anything.
    """
    # task_spec is the whole media_spec dict; size info lives in its "size" sub-object.
    task_size = (task_spec or {}).get("size") if task_spec else None
    sources: list[dict[str, Any] | None] = [per_image, task_size, agent_default]
    for source in sources:
        spec = _extract_spec(source)
        if spec is not None:
            return spec
    return SizeSpec(aspect_ratio=_DEFAULT_ASPECT_RATIO, size=None, resolution=None)


# ---------------------------------------------------------------------------
# Capability registry
# ---------------------------------------------------------------------------

# Cache: (base_url, model) → (monotonic_timestamp, capabilities_dict | None)
# API key is explicitly NOT stored.
_cap_cache: dict[tuple[str, str], tuple[float, dict | None]] = {}
_cap_lock: asyncio.Lock | None = None


def _get_cap_lock() -> asyncio.Lock:
    global _cap_lock
    if _cap_lock is None:
        _cap_lock = asyncio.Lock()
    return _cap_lock


async def model_capabilities(
    base_url: str,
    model: str,
    api_key: str,
) -> dict | None:
    """Fetch and cache the capability descriptor for one model.

    Returns a dict whose keys are parameter names (``"aspect_ratio"``,
    ``"size"``, ``"resolution"``, ``"input_references"``, …) and whose
    values are capability descriptors as returned by OpenRouter:
    ``{"type":"enum","values":[…]}``, ``{"type":"range",…}``,
    ``{"type":"boolean"}``.

    Returns ``None`` on any error or if the model is not found in the
    /images/models response — the caller must work optimistically and handle
    a provider-level 400 if the sent params happen to be unsupported.

    The API key is used only during the HTTP fetch and is never stored in the
    cache (R6 in the security checklist).
    """
    key = (str(base_url), str(model))
    now = time.monotonic()

    async with _get_cap_lock():
        cached = _cap_cache.get(key)
        if cached is not None:
            ts, data = cached
            if now - ts < _MEDIA_IMAGE_MODELS_TTL_S:
                return data

    # Fetch outside the lock so we don't serialise other lookups.
    caps: dict | None = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{str(base_url).rstrip('/')}/images/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            body = resp.json()
            entries = body if isinstance(body, list) else body.get("data", [])
            for entry in entries:
                if entry.get("id") == model:
                    caps = (
                        entry.get("supported_parameters")
                        or entry.get("capabilities")
                        or {}
                    )
                    break
    except Exception as exc:
        logger.warning(
            "media_size: capability fetch failed for %s (%s: %s)",
            model, type(exc).__name__, exc,
        )
        caps = None

    async with _get_cap_lock():
        _cap_cache[key] = (time.monotonic(), caps)

    return caps


def _clear_cap_cache() -> None:
    """Test helper — wipe the in-process capability cache."""
    _cap_cache.clear()


# ---------------------------------------------------------------------------
# Chat-model vision capability (Phase A of media tasks: does the *primary*
# text model accept image content in a chat completion, before reference
# images get attached to it — a separate concern from the /images/models
# capability cache above, which only covers the image-generation model).
# ---------------------------------------------------------------------------

_chat_vision_cache: dict[tuple[str, str], tuple[float, bool | None]] = {}
_chat_vision_lock: asyncio.Lock | None = None


def _get_chat_vision_lock() -> asyncio.Lock:
    global _chat_vision_lock
    if _chat_vision_lock is None:
        _chat_vision_lock = asyncio.Lock()
    return _chat_vision_lock


async def chat_model_supports_image_input(base_url: str, model: str) -> bool | None:
    """True/False per OpenRouter's public model catalog; ``None`` if unknown.

    OpenRouter's ``/models`` catalog is public (no API key needed) and lists
    each model's ``architecture.input_modalities``. Callers should stay
    optimistic on ``None`` (fetch failed or model absent), matching the
    convention of ``model_capabilities`` above.
    """
    key = (str(base_url), str(model))
    now = time.monotonic()

    async with _get_chat_vision_lock():
        cached = _chat_vision_cache.get(key)
        if cached is not None:
            ts, data = cached
            if now - ts < _MEDIA_IMAGE_MODELS_TTL_S:
                return data

    supports: bool | None = None
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{str(base_url).rstrip('/')}/models")
            resp.raise_for_status()
            body = resp.json()
            entries = body if isinstance(body, list) else body.get("data", [])
            for entry in entries:
                if entry.get("id") == model:
                    modalities = (entry.get("architecture") or {}).get("input_modalities") or []
                    supports = "image" in modalities
                    break
    except Exception as exc:
        logger.warning(
            "media_size: chat vision capability fetch failed for %s (%s: %s)",
            model, type(exc).__name__, exc,
        )
        supports = None

    async with _get_chat_vision_lock():
        _chat_vision_cache[key] = (time.monotonic(), supports)

    return supports


# ---------------------------------------------------------------------------
# Capability introspection helpers
# ---------------------------------------------------------------------------


def _supports_size(caps: dict | None) -> bool:
    """True if the model accepts the ``size`` parameter (exact pixels or tier)."""
    if caps is None:
        return True  # optimistic when unknown
    cap = caps.get("size")
    if cap is None:
        return False
    if isinstance(cap, dict):
        t = cap.get("type", "")
        return t == "boolean" or t == "enum" or bool(cap)
    return bool(cap)


def supports_input_references(caps: dict | None) -> bool:
    """True if the model accepts ``input_references`` (image-input/reference mode).

    Unlike ``_supports_size``, a *known* capability dict lacking this key means
    the provider positively does not support it (OpenRouter's ``/images/models``
    always lists every parameter the model accepts). Only truly unknown
    capabilities (``caps is None`` — fetch failed or model absent from the
    catalog) stay optimistic, matching the module's existing convention.
    """
    if caps is None:
        return True  # optimistic when unknown
    return "input_references" in caps


def _supported_aspect_ratios(caps: dict | None) -> list[str] | None:
    """Return the list of aspect-ratio values the model supports, or None if unknown."""
    if caps is None:
        return None
    cap = caps.get("aspect_ratio")
    if isinstance(cap, dict) and cap.get("type") == "enum":
        return cap.get("values") or None
    return None


def _supported_tiers(caps: dict | None) -> list[str] | None:
    """Return the list of resolution/size tiers the model supports, or None if unknown."""
    if caps is None:
        return None
    for key in ("resolution", "size"):
        cap = caps.get(key)
        if isinstance(cap, dict) and cap.get("type") == "enum":
            values = cap.get("values") or []
            tiers = [v for v in values if v in _KNOWN_TIERS]
            if tiers:
                return tiers
    return None


# ---------------------------------------------------------------------------
# Aspect-ratio arithmetic
# ---------------------------------------------------------------------------


def _ratio_to_float(ratio: str) -> float:
    """Convert ``"W:H"`` to ``W/H``; returns 1.0 on parse failure."""
    parts = ratio.split(":")
    if len(parts) == 2:
        try:
            w, h = float(parts[0]), float(parts[1])
            if h > 0:
                return w / h
        except ValueError:
            pass
    return 1.0


def _nearest_ratio(target: float, supported: list[str]) -> str:
    """Return the supported aspect-ratio string closest to ``target`` (W/H).

    Uses the absolute difference of natural logarithms, which is
    scale-invariant and treats both narrow and wide mismatches symmetrically.
    """
    target_log = math.log(max(target, 1e-9))
    return min(supported, key=lambda r: abs(math.log(max(_ratio_to_float(r), 1e-9)) - target_log))


def _pixels_to_ratio(w: int, h: int) -> float:
    return w / h if h else 1.0


def _pixels_to_tier(w: int, h: int) -> str:
    """Return the smallest tier whose max dimension covers max(w, h)."""
    dim = max(w, h)
    for tier in ("512", "1K", "2K", "4K"):
        if dim <= _TIER_MAX_DIM[tier]:
            return tier
    return "4K"


# ---------------------------------------------------------------------------
# Provider parameter builder
# ---------------------------------------------------------------------------


def build_provider_params(
    spec: SizeSpec,
    caps: dict | None,
) -> tuple[dict, tuple[int, int] | None, list[str]]:
    """Build the ``image_gen`` payload slice for size, and decide post-processing.

    Returns ``(provider_params, exact_target, degraded)``:
    - ``provider_params``: keys to merge into the generation payload.
      The caller must never send ``size`` together with ``aspect_ratio`` or
      ``resolution`` — this function enforces that invariant.
    - ``exact_target``: if not None, a ``(w, h)`` tuple for ``fit_exact``
      after generation (model cannot produce the exact requested pixels natively).
    - ``degraded``: parameter names that could not be honoured — recorded in
      the generation event for observability (R8).
    """
    params: dict[str, Any] = {}
    exact_target: tuple[int, int] | None = None
    degraded: list[str] = []

    size_raw = spec.size

    if size_raw:
        try:
            parsed = parse_size(size_raw)
        except ValueError:
            parsed = None

        if isinstance(parsed, tuple):
            # Exact pixels requested.
            w, h = parsed
            if _supports_size(caps):
                # Provider can handle size directly — send only size.
                params["size"] = size_raw
            else:
                # Degrade: find nearest supported aspect ratio + a usable tier.
                degraded.append("size")
                target_ratio = _pixels_to_ratio(w, h)
                supported_ratios = _supported_aspect_ratios(caps)
                if supported_ratios:
                    nearest = _nearest_ratio(target_ratio, supported_ratios)
                    params["aspect_ratio"] = nearest
                else:
                    # No capability info — send aspect_ratio optimistically.
                    params["aspect_ratio"] = _format_ratio(w, h)
                # Add tier so we get at least the resolution we need.
                tier = _pixels_to_tier(w, h)
                supported_tiers = _supported_tiers(caps)
                if supported_tiers:
                    if tier not in supported_tiers:
                        tier = max(supported_tiers, key=lambda t: _TIER_MAX_DIM.get(t, 0))
                    params["resolution"] = tier
                elif _supports_resolution(caps):
                    params["resolution"] = tier
                # Always post-process to the exact requested size.
                exact_target = (w, h)
        elif isinstance(parsed, str):
            # Tier requested (e.g. "2K").
            if _supports_size(caps):
                params["size"] = parsed
            else:
                # Try resolution param as fallback.
                supported_tiers = _supported_tiers(caps)
                if supported_tiers and parsed not in supported_tiers:
                    # Use largest available tier.
                    parsed = max(supported_tiers, key=lambda t: _TIER_MAX_DIM.get(t, 0))
                    degraded.append("resolution")
                params["resolution"] = parsed
        # size_raw handled — return early.
        return params, exact_target, degraded

    # No exact size — use aspect_ratio / resolution.
    if spec.aspect_ratio:
        supported_ratios = _supported_aspect_ratios(caps)
        if supported_ratios and spec.aspect_ratio not in supported_ratios:
            nearest = _nearest_ratio(_ratio_to_float(spec.aspect_ratio), supported_ratios)
            params["aspect_ratio"] = nearest
            degraded.append("aspect_ratio")
        else:
            params["aspect_ratio"] = spec.aspect_ratio

    if spec.resolution and not params.get("size"):
        params["resolution"] = spec.resolution

    if not params:
        # Nothing from the spec — send the hardcoded default.
        params["aspect_ratio"] = _DEFAULT_ASPECT_RATIO

    return params, exact_target, degraded


def _format_ratio(w: int, h: int) -> str:
    """Reduce ``w/h`` to a simple ratio string like ``"16:9"``."""
    from math import gcd
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def _supports_resolution(caps: dict | None) -> bool:
    if caps is None:
        return True
    return "resolution" in caps


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def fit_exact(png: bytes, w: int, h: int, mode: str = "cover") -> bytes:
    """Resize / crop / pad a PNG so the output is exactly ``w × h`` pixels.

    ``cover``   — scale so the image fills the box, then center-crop (default).
                  Best for banners / headers where the whole area must be used.
    ``contain`` — scale so the image fits inside the box, then pad with
                  transparent pixels. Output is RGBA.
    ``stretch`` — scale the image to exactly ``w × h`` regardless of aspect.

    Always returns a PNG byte string.
    """
    from PIL import Image

    mode = mode if mode in _VALID_FIT_MODES else "cover"

    with Image.open(BytesIO(png)) as img:
        src_w, src_h = img.size

        if mode == "stretch":
            result = img.resize((w, h), Image.LANCZOS)

        elif mode == "cover":
            # Scale so the smaller ratio side fills the target box.
            scale = max(w / src_w, h / src_h)
            scaled_w = round(src_w * scale)
            scaled_h = round(src_h * scale)
            scaled = img.resize((scaled_w, scaled_h), Image.LANCZOS)
            left = (scaled_w - w) // 2
            top = (scaled_h - h) // 2
            result = scaled.crop((left, top, left + w, top + h))

        else:  # contain
            scale = min(w / src_w, h / src_h)
            scaled_w = round(src_w * scale)
            scaled_h = round(src_h * scale)
            scaled = img.resize((scaled_w, scaled_h), Image.LANCZOS).convert("RGBA")
            canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            paste_x = (w - scaled_w) // 2
            paste_y = (h - scaled_h) // 2
            canvas.paste(scaled, (paste_x, paste_y))
            result = canvas

        out = BytesIO()
        result.save(out, format="PNG")
        return out.getvalue()


# ---------------------------------------------------------------------------
# Metrics wrappers (imported lazily to avoid import cycles)
# ---------------------------------------------------------------------------


def _record_postprocessed(mode: str) -> None:
    try:
        from app.core.metrics import record_media_size_postprocessed
        record_media_size_postprocessed(mode)
    except Exception:
        pass


def _record_degraded(param: str) -> None:
    try:
        from app.core.metrics import record_media_generation_degraded
        record_media_generation_degraded(param)
    except Exception:
        pass
