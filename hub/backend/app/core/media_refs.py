"""Reference-image resolver for the media agent.

internal design notes, stages 1-2 (resolver + SVG rasterization),
5 (Phase A wiring), 6 (Phase B wiring), 8 (deterministic logo overlay).

Public API:
  resolve_reference / resolve_references / resolve_task_references — validate a
    workspace path and return decoded, dimension-aware ``ResolvedRef`` bytes.
  rasterize_svg — render an SVG to PNG via the shared Chromium pool.
  to_data_url — encode a ``ResolvedRef`` as a data URL for providers.
  build_reference_manifest / build_vision_blocks — Phase A content helpers.
  composite_logo_overlay — deterministically paste a logo on a generated image
    (stage 8); the logo is resized with Pillow, positioned by anchor+margin,
    and blended with the requested opacity.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.image_bytes import sniff_image_mime, split_image_data_uri
from app.core.llm_providers import uses_messages_wire
from app.core.path_contract import WorkspacePathError, normalize_workspace_path

logger = logging.getLogger(__name__)

#: Ceiling on a single decoded reference file (env: internal design notes stage 0).
MEDIA_REF_MAX_BYTES = int(os.getenv("MEDIA_REF_MAX_BYTES", str(8 * 1024 * 1024)))
#: How many references may be attached to one generated image (enforced by callers, e.g. the
#: ``submit_image_prompts`` tool schema added in stage 5); exposed here so both sides agree.
MEDIA_REF_MAX_PER_IMAGE = int(os.getenv("MEDIA_REF_MAX_PER_IMAGE", "4"))
#: Ceiling on the combined decoded size of every reference resolved for one task.
MEDIA_REF_MAX_TOTAL_BYTES = int(os.getenv("MEDIA_REF_MAX_TOTAL_BYTES", str(12 * 1024 * 1024)))
#: Kill switch for vector rasterization (env: internal design notes stage 0).
_MEDIA_SVG_RASTER_ENABLED_RAW = "MEDIA_SVG_RASTER_ENABLED"
#: Ceiling on a source .svg file, before rendering.
MEDIA_SVG_MAX_BYTES = int(os.getenv("MEDIA_SVG_MAX_BYTES", str(2 * 1024 * 1024)))

#: Directory the rasterization cache lives in — ``.legion`` is already excluded
#: from git sync and workspace snapshots everywhere else (skill_materialize.py),
#: which is exactly the property a scratch render cache needs.
_RASTER_CACHE_DIR = ".legion/media-refs"
#: Longer side of the rendered PNG when the SVG gives no usable intrinsic size.
_DEFAULT_RASTER_PX = 1024
_MAX_RASTER_PX = 4096
_RASTER_TIMEOUT_S = 15
#: Chromium is shared with workspace-preview screenshots (screenshot.py); cap
#: concurrent rasterizations so a 9-image series doesn't open 9 contexts at once.
_RASTER_SEMAPHORE = asyncio.Semaphore(3)

#: Roles a reference can be tagged with (internal design notes §4, R4/R8).
#: ``logo_overlay`` is accepted here too even though only stage 8 (optional) acts on it,
#: so an agent that names it early is not rejected outright.
VALID_REFERENCE_ROLES = frozenset({"style", "subject", "logo", "layout", "logo_overlay"})

_DEFAULT_ROLE = "style"


class MediaReferenceError(RuntimeError):
    """Safe to show an agent as-is: never carries file bytes or content."""


@dataclass(frozen=True)
class ResolvedRef:
    path: str
    mime: str
    data: bytes
    width: int | None
    height: int | None
    role: str
    rasterized_from: str | None = None


def _record_resolve_failed(reason: str) -> None:
    try:
        from app.core.metrics import record_media_ref_resolve_failed

        record_media_ref_resolve_failed(reason)
    except Exception:
        pass


def _record_svg_rasterized(cache: str) -> None:
    try:
        from app.core.metrics import record_media_svg_rasterized

        record_media_svg_rasterized(cache)
    except Exception:
        pass


def svg_raster_enabled() -> bool:
    return os.getenv(_MEDIA_SVG_RASTER_ENABLED_RAW, "true").strip().lower() in {"1", "true", "yes", "on"}


# Regex-only parsing on purpose (internal design notes §R5): the
# SVG is untrusted input, and a full XML parser opens an XXE surface for a
# value we only need the *root tag's* width/height/viewBox out of.
_SVG_OPEN_TAG_RE = re.compile(rb"<svg\b[^>]{0,4096}>", re.IGNORECASE | re.DOTALL)
_SVG_ATTR_RE = re.compile(rb'([a-zA-Z_:][\w:.-]*)\s*=\s*"([^"]*)"')
_LENGTH_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z%]*)\s*$")
_UNIT_TO_PX = {"": 1.0, "px": 1.0, "pt": 96 / 72, "pc": 16.0, "in": 96.0, "cm": 96 / 2.54, "mm": 96 / 25.4}


def _svg_root_attrs(svg: bytes) -> dict[str, str]:
    match = _SVG_OPEN_TAG_RE.search(svg[:65536])
    if match is None:
        return {}
    attrs: dict[str, str] = {}
    for attr_match in _SVG_ATTR_RE.finditer(match.group(0)):
        key = attr_match.group(1).decode("ascii", "ignore").lower()
        attrs[key] = attr_match.group(2).decode("utf-8", "ignore")
    return attrs


def _parse_svg_length(raw: str) -> float | None:
    match = _LENGTH_RE.match(raw or "")
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "%":
        return None
    scale = _UNIT_TO_PX.get(unit)
    return value * scale if scale is not None else None


def _svg_intrinsic_size(svg: bytes) -> tuple[float, float] | None:
    """Return an aspect-ratio source ``(w, h)`` — viewBox wins over width/height.

    ``viewBox`` numbers are the SVG's user-space units, not necessarily
    pixels, but the ratio between them is exactly what render sizing needs;
    ``width``/``height`` (possibly in physical units like ``mm``) are only a
    fallback for SVGs with no viewBox at all.
    """

    attrs = _svg_root_attrs(svg)
    view_box = attrs.get("viewbox", "").strip()
    if view_box:
        parts = re.split(r"[\s,]+", view_box)
        if len(parts) == 4:
            try:
                _, _, vb_w, vb_h = (float(p) for p in parts)
            except ValueError:
                vb_w = vb_h = 0.0
            if vb_w > 0 and vb_h > 0:
                return vb_w, vb_h
    width = _parse_svg_length(attrs.get("width", ""))
    height = _parse_svg_length(attrs.get("height", ""))
    if width and height and width > 0 and height > 0:
        return width, height
    return None


def _render_size(svg: bytes, target_px: int | None) -> tuple[int, int]:
    base = int(target_px) if target_px else _DEFAULT_RASTER_PX
    base = max(1, min(base, _MAX_RASTER_PX))
    dims = _svg_intrinsic_size(svg)
    if dims is None:
        return base, base
    width, height = dims
    if width >= height:
        render_w, render_h = base, max(1, round(base * height / width))
    else:
        render_h, render_w = base, max(1, round(base * width / height))
    return min(render_w, _MAX_RASTER_PX), min(render_h, _MAX_RASTER_PX)


def _normalize_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        raise MediaReferenceError("reference path is required")
    try:
        return normalize_workspace_path(raw)
    except WorkspacePathError as exc:
        _record_resolve_failed("invalid_path")
        raise MediaReferenceError(f"invalid reference path: {raw}") from exc


def _decode_image_content(content: str, normalized_path: str) -> tuple[str, bytes]:
    parsed = split_image_data_uri(content)
    if parsed is None:
        _record_resolve_failed("not_an_image")
        raise MediaReferenceError(f"reference is not a supported image: {normalized_path}")
    declared_mime, encoded = parsed
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        _record_resolve_failed("corrupt_data")
        raise MediaReferenceError(f"reference file is corrupt: {normalized_path}") from None
    if not decoded:
        _record_resolve_failed("corrupt_data")
        raise MediaReferenceError(f"reference file is corrupt: {normalized_path}")
    if len(decoded) > MEDIA_REF_MAX_BYTES:
        _record_resolve_failed("too_large")
        raise MediaReferenceError(f"reference file is too large: {normalized_path}")
    inferred_mime = sniff_image_mime(decoded)
    if inferred_mime is None or inferred_mime != declared_mime:
        _record_resolve_failed("not_an_image")
        raise MediaReferenceError(f"reference is not a supported image: {normalized_path}")
    return inferred_mime, decoded


def _read_dimensions(decoded: bytes, normalized_path: str) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(BytesIO(decoded)) as img:
            return img.width, img.height
    except Exception:
        _record_resolve_failed("unreadable_dimensions")
        return None, None


async def rasterize_svg(svg: bytes, *, target_px: int | None = None) -> tuple[bytes, int, int]:
    """Render untrusted SVG bytes to an opaque-safe PNG via the shared Chromium pool.

    No new native dependency: reuses ``screenshot._browser()`` (already in the
    image for preview screenshots). The SVG is treated as hostile input — JS
    disabled, every network request aborted, ``set_content`` instead of
    navigating to a URL — so embedded ``<image href="https://…">`` or
    ``@import`` cannot exfiltrate anything or hang the render.
    """

    if not svg_raster_enabled():
        raise MediaReferenceError("vector references are disabled")
    if len(svg) > MEDIA_SVG_MAX_BYTES:
        _record_resolve_failed("too_large")
        raise MediaReferenceError("reference SVG is too large")

    render_w, render_h = _render_size(svg, target_px)
    encoded = base64.b64encode(svg).decode("ascii")
    html = (
        "<!doctype html><html><head><style>"
        "html,body{margin:0;padding:0;background:transparent}"
        f"img{{display:block;width:{render_w}px;height:{render_h}px}}"
        "</style></head><body>"
        f'<img src="data:image/svg+xml;base64,{encoded}"></body></html>'
    )

    from app.core import screenshot as screenshot_mod

    async with _RASTER_SEMAPHORE:
        try:
            browser = await screenshot_mod._browser()
        except screenshot_mod.ScreenshotError as exc:
            _record_resolve_failed("rasterize_unavailable")
            raise MediaReferenceError("vector rendering is unavailable") from exc

        context = None
        try:
            context = await browser.new_context(
                viewport={"width": render_w, "height": render_h},
                java_script_enabled=False,
                device_scale_factor=1,
            )
            await context.route("**/*", lambda route: route.abort())
            page = await context.new_page()
            await asyncio.wait_for(
                page.set_content(html, wait_until="load"), timeout=_RASTER_TIMEOUT_S
            )
            png = await asyncio.wait_for(
                page.screenshot(type="png", omit_background=True), timeout=_RASTER_TIMEOUT_S
            )
        except MediaReferenceError:
            raise
        except asyncio.TimeoutError as exc:
            _record_resolve_failed("rasterize_timeout")
            raise MediaReferenceError("vector reference took too long to render") from exc
        except Exception as exc:
            logger.warning("media_refs: SVG rasterization failed (%s)", type(exc).__name__)
            _record_resolve_failed("rasterize_failed")
            raise MediaReferenceError("failed to render vector reference") from exc
        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass

    return png, render_w, render_h


async def _rasterize_with_cache(room_id: UUID, normalized_path: str, svg: bytes) -> tuple[bytes, int, int]:
    """Rasterize once per distinct SVG content; cache under ``.legion/media-refs``.

    Cached like ``skill_materialize``'s ``.legion/skills`` writes: straight
    through ``workspaced_client``, not ``workspace_fs.fs_write_file`` — the
    latter also upserts a ``RepoFile`` row, which would surface a scratch
    render cache in the Repository panel and git sync (internal design notes §6).
    """

    digest = hashlib.sha256(svg).hexdigest()
    cache_path = f"{_RASTER_CACHE_DIR}/{digest}.png"

    from app.core import workspaced_client as wsd

    try:
        cached = await wsd.read_workspace_file(str(room_id), cache_path)
    except Exception as exc:
        logger.warning("media_refs: rasterization cache read failed for %s (%s)", cache_path, type(exc).__name__)
        cached = None

    if cached:
        parsed = split_image_data_uri(cached)
        if parsed is not None:
            try:
                png = base64.b64decode(parsed[1], validate=True)
            except (binascii.Error, ValueError):
                png = b""
            if png:
                width, height = _read_dimensions(png, normalized_path)
                if width and height:
                    _record_svg_rasterized("hit")
                    return png, width, height

    png, width, height = await rasterize_svg(svg)
    _record_svg_rasterized("miss")

    try:
        encoded = base64.b64encode(png).decode("ascii")
        await wsd.write_workspace_file(str(room_id), cache_path, f"data:image/png;base64,{encoded}")
    except Exception as exc:
        logger.warning("media_refs: rasterization cache write failed for %s (%s)", cache_path, type(exc).__name__)

    return png, width, height


async def resolve_reference(
    room_id: UUID,
    path: str,
    db: AsyncSession,
    *,
    role: str = _DEFAULT_ROLE,
) -> ResolvedRef:
    """Resolve one workspace path into a validated raster reference.

    Raises ``MediaReferenceError`` (safe to show the agent verbatim) for any
    failure — missing file, non-image content, oversized file, or invalid
    path. ``.svg`` paths are rasterized to PNG (stage 2, cached per task —
    see ``_rasterize_with_cache``) instead of being rejected.
    """

    role = str(role or _DEFAULT_ROLE).strip() or _DEFAULT_ROLE
    if role not in VALID_REFERENCE_ROLES:
        raise MediaReferenceError(f"unknown reference role: {role}")

    normalized = _normalize_path(path)

    from app.core import workspace_fs

    content = await workspace_fs.fs_read_file(room_id, normalized, db)
    if content is None:
        _record_resolve_failed("not_found")
        raise MediaReferenceError(f"reference not found: {normalized}")

    if normalized.lower().endswith(".svg"):
        if not svg_raster_enabled():
            raise MediaReferenceError("vector references are disabled")
        try:
            svg_bytes = content.encode("utf-8")
        except UnicodeEncodeError:
            _record_resolve_failed("corrupt_data")
            raise MediaReferenceError(f"reference file is corrupt: {normalized}") from None
        if not svg_bytes:
            _record_resolve_failed("corrupt_data")
            raise MediaReferenceError(f"reference file is corrupt: {normalized}")
        if len(svg_bytes) > MEDIA_SVG_MAX_BYTES:
            _record_resolve_failed("too_large")
            raise MediaReferenceError(f"reference file is too large: {normalized}")

        png, width, height = await _rasterize_with_cache(room_id, normalized, svg_bytes)
        if len(png) > MEDIA_REF_MAX_BYTES:
            _record_resolve_failed("too_large")
            raise MediaReferenceError(f"reference file is too large: {normalized}")

        return ResolvedRef(
            path=normalized,
            mime="image/png",
            data=png,
            width=width,
            height=height,
            role=role,
            rasterized_from=normalized,
        )

    mime, decoded = _decode_image_content(content, normalized)
    width, height = _read_dimensions(decoded, normalized)

    return ResolvedRef(
        path=normalized,
        mime=mime,
        data=decoded,
        width=width,
        height=height,
        role=role,
        rasterized_from=None,
    )


async def resolve_references(
    room_id: UUID,
    specs: list[dict[str, Any]],
    db: AsyncSession,
) -> list[ResolvedRef]:
    """Resolve a task's ``media_spec.references`` list, deduped by path.

    A resolution failure for one entry aborts the whole call — callers that
    want partial success (stage 6: "an unresolved reference is dropped, not
    fatal") catch ``MediaReferenceError`` per spec themselves.
    """

    resolved: dict[str, ResolvedRef] = {}
    total_bytes = 0
    for spec in specs or []:
        raw_path = str((spec or {}).get("path") or "").strip()
        if not raw_path:
            continue
        role = str((spec or {}).get("role") or _DEFAULT_ROLE).strip() or _DEFAULT_ROLE
        normalized_key = _normalize_path(raw_path)
        if normalized_key in resolved:
            continue
        ref = await resolve_reference(room_id, raw_path, db, role=role)
        total_bytes += len(ref.data)
        if total_bytes > MEDIA_REF_MAX_TOTAL_BYTES:
            _record_resolve_failed("total_too_large")
            raise MediaReferenceError("reference files exceed the total size limit for this task")
        resolved[normalized_key] = ref
    return list(resolved.values())


def to_data_url(ref: ResolvedRef) -> str:
    encoded = base64.b64encode(ref.data).decode("ascii")
    return f"data:{ref.mime};base64,{encoded}"


#: Cap on the megapixel count of a reference embedded in the Phase A vision
#: turn (stage 5.3) — references are shown so the prompt author can *describe*
#: them, not to reproduce them pixel-for-pixel, so full resolution would only
#: spend context budget for no benefit.
MEDIA_REF_VISION_MAX_PIXELS = int(os.getenv("MEDIA_REF_VISION_MAX_PIXELS", str(1_000_000)))


@dataclass(frozen=True)
class ReferenceManifestEntry:
    """One line of the Phase A ``## Reference files`` manifest.

    ``ref`` is ``None`` when resolution failed — the entry still carries
    ``path``/``role``/``note`` so the failure is a visible line item in the
    manifest (and a legal value for ``reference_paths[i]`` in
    ``submit_image_prompts``) instead of silently vanishing.
    """

    path: str
    role: str
    note: str
    ref: ResolvedRef | None
    error: str | None


async def resolve_task_references(
    room_id: UUID,
    specs: list[dict[str, Any]],
    db: AsyncSession,
) -> list[ReferenceManifestEntry]:
    """Resolve a task's ``media_spec.references`` with per-entry failure isolation.

    Unlike ``resolve_references`` (which aborts the whole call on the first
    bad path — the right policy for Phase B, where a task-level media_spec is
    assumed already validated), Phase A is the first place these paths meet
    reality: a mistyped or stale path must not crash prompt authoring for the
    whole task, it must show up as one broken line in the manifest.
    """

    entries: list[ReferenceManifestEntry] = []
    seen: set[str] = set()
    total_bytes = 0
    for spec in specs or []:
        raw_path = str((spec or {}).get("path") or "").strip()
        if not raw_path:
            continue
        role = str((spec or {}).get("role") or _DEFAULT_ROLE).strip() or _DEFAULT_ROLE
        note = str((spec or {}).get("note") or "").strip()
        try:
            normalized = _normalize_path(raw_path)
        except MediaReferenceError as exc:
            entries.append(ReferenceManifestEntry(raw_path, role, note, None, str(exc)))
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            ref = await resolve_reference(room_id, raw_path, db, role=role)
        except MediaReferenceError as exc:
            entries.append(ReferenceManifestEntry(normalized, role, note, None, str(exc)))
            continue
        total_bytes += len(ref.data)
        if total_bytes > MEDIA_REF_MAX_TOTAL_BYTES:
            _record_resolve_failed("total_too_large")
            entries.append(ReferenceManifestEntry(
                normalized, role, note, None,
                "reference files exceed the total size limit for this task",
            ))
            continue
        entries.append(ReferenceManifestEntry(normalized, role, note, ref, None))
    return entries


def build_reference_manifest(entries: list[ReferenceManifestEntry]) -> str:
    """Render the ``## Reference files`` section of the Phase A user turn.

    Appended to ``render_task_data(task)``'s plain-text output (proxy_session.py)
    — this works even for providers where the multimodal vision turn
    (``build_vision_blocks``) is skipped, since it names real pixel dimensions
    for every resolved reference.
    """

    if not entries:
        return ""
    lines = ["\n\n## Reference files"]
    for entry in entries:
        if entry.ref is not None:
            dims = (
                f"{entry.ref.width}x{entry.ref.height}px"
                if entry.ref.width and entry.ref.height
                else "size unknown"
            )
            line = f"- `{entry.path}` (role: {entry.role}, {dims})"
        else:
            line = f"- `{entry.path}` (role: {entry.role}, UNAVAILABLE: {entry.error or 'could not be resolved'})"
        if entry.note:
            line += f" — {entry.note}"
        lines.append(line)
    return "\n".join(lines)


def _shrink_for_vision(data: bytes, mime: str) -> tuple[bytes, str]:
    """Downscale reference bytes to ``MEDIA_REF_VISION_MAX_PIXELS`` if needed.

    Falls back to the original bytes/mime on any Pillow failure — the vision
    embedding is a bonus on top of the always-present text manifest, never
    the only way the agent learns about a reference, so it must not raise.
    """

    try:
        from PIL import Image
    except Exception:
        return data, mime
    try:
        with Image.open(BytesIO(data)) as img:
            width, height = img.width, img.height
            pixels = width * height
            if pixels <= 0 or pixels <= MEDIA_REF_VISION_MAX_PIXELS:
                return data, mime
            scale = (MEDIA_REF_VISION_MAX_PIXELS / pixels) ** 0.5
            new_w = max(1, round(width * scale))
            new_h = max(1, round(height * scale))
            mode = "RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB"
            resized = img.convert(mode).resize((new_w, new_h), Image.LANCZOS)
            buf = BytesIO()
            resized.save(buf, format="PNG")
            return buf.getvalue(), "image/png"
    except Exception:
        logger.warning("media_refs: vision downscale failed, using original bytes", exc_info=True)
        return data, mime


_LOGO_ANCHORS = frozenset({
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
})


def composite_logo_overlay(
    base_png: bytes,
    logo_data: bytes,
    *,
    anchor: str = "top-left",
    margin_pct: float = 3.0,
    width_pct: float = 20.0,
    opacity: float = 1.0,
) -> bytes:
    """Deterministically paste a logo on a generated image (stage 8).

    The logo is resized proportionally so its width equals ``width_pct`` % of
    the base image width, then placed at the anchor corner with a gap of
    ``margin_pct`` % of the base width from the edge(s). ``opacity`` is
    multiplied into the logo's alpha channel (1.0 = fully opaque).

    Accepted anchors: "top-left", "top-center", "top-right", "center-left",
    "center", "center-right", "bottom-left", "bottom-center", "bottom-right".

    Raises ``ValueError`` on invalid parameters; raises ``MediaReferenceError``
    if Pillow cannot open either image. Both are safe to log and surface to the
    caller — neither leaks file content.
    """

    if anchor not in _LOGO_ANCHORS:
        raise ValueError(
            f"unknown logo overlay anchor: {anchor!r}; expected one of "
            + ", ".join(sorted(_LOGO_ANCHORS))
        )
    width_pct = float(width_pct)
    margin_pct = float(margin_pct)
    opacity = max(0.0, min(1.0, float(opacity)))
    if not (0 < width_pct <= 100):
        raise ValueError(f"width_pct must be in (0, 100], got {width_pct}")
    if not (0 <= margin_pct < 50):
        raise ValueError(f"margin_pct must be in [0, 50), got {margin_pct}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise MediaReferenceError("Pillow is not available for logo overlay") from exc

    try:
        with Image.open(BytesIO(base_png)) as _base:
            base_w, base_h = _base.size
            base_rgba = _base.convert("RGBA")
    except Exception as exc:
        raise MediaReferenceError("failed to open base image for logo overlay") from exc

    logo_w = max(1, round(base_w * width_pct / 100))
    try:
        with Image.open(BytesIO(logo_data)) as _logo:
            orig_w, orig_h = _logo.size
            logo_h = max(1, round(logo_w * orig_h / orig_w)) if orig_w > 0 else logo_w
            logo_rgba = _logo.convert("RGBA").resize((logo_w, logo_h), Image.LANCZOS)
    except Exception as exc:
        raise MediaReferenceError("failed to open logo image for overlay") from exc

    if opacity < 1.0:
        r, g, b, a = logo_rgba.split()
        a = a.point(lambda v: round(v * opacity))
        logo_rgba = Image.merge("RGBA", (r, g, b, a))

    margin_px = round(base_w * margin_pct / 100)
    parts = anchor.split("-", 1)
    vert = parts[0]
    horiz = parts[1] if len(parts) > 1 else "center"

    if horiz == "left":
        paste_x = margin_px
    elif horiz == "right":
        paste_x = base_w - logo_w - margin_px
    else:
        paste_x = (base_w - logo_w) // 2

    if vert == "top":
        paste_y = margin_px
    elif vert == "bottom":
        paste_y = base_h - logo_h - margin_px
    else:
        paste_y = (base_h - logo_h) // 2

    paste_x = max(0, min(paste_x, max(0, base_w - logo_w)))
    paste_y = max(0, min(paste_y, max(0, base_h - logo_h)))

    base_rgba.paste(logo_rgba, (paste_x, paste_y), mask=logo_rgba)

    buf = BytesIO()
    base_rgba.save(buf, format="PNG")
    return buf.getvalue()


def build_vision_blocks(
    provider: str, refs: list[ResolvedRef], model: str | None = None
) -> list[dict[str, Any]]:
    """Build wire-specific image content blocks for the Phase A first turn.

    Two shapes only, matching the two branches ``proxy_session._call_llm``
    actually has: Anthropic's ``{"type": "image", "source": {...}}`` and the
    OpenAI-compatible ``{"type": "image_url", "image_url": {...}}`` used by
    every other provider (openrouter, groq, deepseek, ollama, together, ...).

    ``model`` matters for gateways such as OpenCode Go that put some of their
    models on the Anthropic wire and the rest on the OpenAI one.
    """

    provider = (provider or "").lower()
    anthropic_shape = uses_messages_wire(provider, model)
    blocks: list[dict[str, Any]] = []
    for ref in refs:
        data, mime = _shrink_for_vision(ref.data, ref.mime)
        encoded = base64.b64encode(data).decode("ascii")
        if anthropic_shape:
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": encoded},
            })
        else:
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
    return blocks
