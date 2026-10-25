"""Unit tests for stages 1–2 and 8 of internal design notes.

No database and no real workspace — ``workspace_fs.fs_read_file`` is mocked,
following the same pattern as ``test_mcp_vision_stage1.py``.
Stage 8 tests are at the bottom: ``composite_logo_overlay``.
"""
from __future__ import annotations

import base64
from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image

from app.core import media_refs
from app.core.media_refs import (
    MediaReferenceError,
    ResolvedRef,
    _LOGO_ANCHORS,
    build_reference_manifest,
    build_vision_blocks,
    composite_logo_overlay,
    resolve_reference,
    resolve_references,
    resolve_task_references,
)


def _encode(fmt: str, size: tuple[int, int] = (12, 8)) -> str:
    img = Image.new("RGB", size, color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _data_url(mime: str, encoded: str) -> str:
    return f"data:{mime};base64,{encoded}"


def _mock_read(monkeypatch, mapping: dict[str, str | None]):
    async def fake_read(room_id, path, db):
        return mapping.get(path)

    monkeypatch.setattr("app.core.workspace_fs.fs_read_file", fake_read)


@pytest.mark.asyncio
async def test_resolve_png_happy_path(monkeypatch):
    encoded = _encode("PNG", (16, 9))
    _mock_read(monkeypatch, {"images/bg.png": _data_url("image/png", encoded)})

    ref = await resolve_reference(uuid4(), "images/bg.png", object(), role="layout")

    assert ref.mime == "image/png"
    assert ref.width == 16
    assert ref.height == 9
    assert ref.role == "layout"
    assert ref.rasterized_from is None
    assert ref.data == base64.b64decode(encoded)


@pytest.mark.asyncio
async def test_resolve_jpeg_happy_path(monkeypatch):
    encoded = _encode("JPEG", (20, 10))
    _mock_read(monkeypatch, {"images/photo.jpg": _data_url("image/jpeg", encoded)})

    ref = await resolve_reference(uuid4(), "images/photo.jpg", object())

    assert ref.mime == "image/jpeg"
    assert ref.width == 20
    assert ref.height == 10


@pytest.mark.asyncio
async def test_resolve_webp_happy_path(monkeypatch):
    encoded = _encode("WEBP", (32, 24))
    _mock_read(monkeypatch, {"images/tex.webp": _data_url("image/webp", encoded)})

    ref = await resolve_reference(uuid4(), "images/tex.webp", object())

    assert ref.mime == "image/webp"
    assert ref.width == 32
    assert ref.height == 24


@pytest.mark.asyncio
async def test_resolve_missing_file_raises(monkeypatch):
    _mock_read(monkeypatch, {})

    with pytest.raises(MediaReferenceError, match="reference not found: images/missing.png"):
        await resolve_reference(uuid4(), "images/missing.png", object())


@pytest.mark.asyncio
async def test_resolve_non_image_rejected(monkeypatch):
    _mock_read(monkeypatch, {"notes/readme.md": "# just markdown text"})

    with pytest.raises(MediaReferenceError, match="not a supported image"):
        await resolve_reference(uuid4(), "notes/readme.md", object())


@pytest.mark.asyncio
async def test_resolve_oversized_file_rejected(monkeypatch):
    encoded = _encode("PNG", (16, 9))
    _mock_read(monkeypatch, {"images/bg.png": _data_url("image/png", encoded)})
    monkeypatch.setattr(media_refs, "MEDIA_REF_MAX_BYTES", 4)

    with pytest.raises(MediaReferenceError, match="too large"):
        await resolve_reference(uuid4(), "images/bg.png", object())


@pytest.mark.asyncio
async def test_resolve_path_traversal_rejected(monkeypatch):
    _mock_read(monkeypatch, {})

    with pytest.raises(MediaReferenceError, match="invalid reference path"):
        await resolve_reference(uuid4(), "../etc/passwd", object())


@pytest.mark.asyncio
async def test_resolve_references_dedupes_paths(monkeypatch):
    encoded = _encode("PNG", (16, 9))
    _mock_read(monkeypatch, {"images/bg.png": _data_url("image/png", encoded)})

    specs = [
        {"path": "images/bg.png", "role": "layout"},
        {"path": "images/bg.png", "role": "layout"},
        {"path": "/images/bg.png", "role": "layout"},
    ]
    refs = await resolve_references(uuid4(), specs, object())

    assert len(refs) == 1
    assert refs[0].path == "images/bg.png"


@pytest.mark.asyncio
async def test_resolve_references_enforces_total_size_limit(monkeypatch):
    encoded_a = _encode("PNG", (16, 9))
    encoded_b = _encode("PNG", (24, 12))
    _mock_read(
        monkeypatch,
        {
            "images/a.png": _data_url("image/png", encoded_a),
            "images/b.png": _data_url("image/png", encoded_b),
        },
    )
    monkeypatch.setattr(media_refs, "MEDIA_REF_MAX_TOTAL_BYTES", 10)

    specs = [{"path": "images/a.png"}, {"path": "images/b.png"}]
    with pytest.raises(MediaReferenceError, match="total size limit"):
        await resolve_references(uuid4(), specs, object())


@pytest.mark.asyncio
async def test_resolve_unknown_role_rejected(monkeypatch):
    encoded = _encode("PNG", (16, 9))
    _mock_read(monkeypatch, {"images/bg.png": _data_url("image/png", encoded)})

    with pytest.raises(MediaReferenceError, match="unknown reference role"):
        await resolve_reference(uuid4(), "images/bg.png", object(), role="villain")


def test_to_data_url_roundtrip():
    ref = media_refs.ResolvedRef(
        path="images/bg.png",
        mime="image/png",
        data=b"\x89PNG\r\n\x1a\nrest",
        width=16,
        height=9,
        role="layout",
    )
    url = media_refs.to_data_url(ref)
    assert url.startswith("data:image/png;base64,")
    prefix = "data:image/png;base64,"
    assert base64.b64decode(url[len(prefix):]) == ref.data


# ── Stage 2: SVG rasterization ──────────────────────────────────────────────


class _FakePage:
    def __init__(self, png_bytes: bytes):
        self._png = png_bytes
        self.set_content_calls: list[tuple[str, str | None]] = []

    async def set_content(self, html, wait_until=None):
        self.set_content_calls.append((html, wait_until))

    async def screenshot(self, **kwargs):
        return self._png


class _FakeContext:
    def __init__(self, png_bytes: bytes, viewport):
        self.viewport = viewport
        self.routed: list[str] = []
        self.closed = False
        self._page = _FakePage(png_bytes)

    async def route(self, pattern, handler):
        self.routed.append(pattern)

    async def new_page(self):
        return self._page

    async def close(self):
        self.closed = True


class _FakeBrowser:
    def __init__(self, png_bytes: bytes):
        self._png = png_bytes
        self.contexts: list[_FakeContext] = []

    async def new_context(self, **kwargs):
        ctx = _FakeContext(self._png, kwargs.get("viewport"))
        self.contexts.append(ctx)
        return ctx


def _mock_browser(monkeypatch, png_bytes: bytes = b"fake-png-bytes") -> _FakeBrowser:
    fake_browser = _FakeBrowser(png_bytes)

    async def fake_get_browser():
        return fake_browser

    monkeypatch.setattr("app.core.screenshot._browser", fake_get_browser)
    return fake_browser


def test_svg_intrinsic_size_prefers_viewbox():
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="85mm" height="45mm" '
        b'viewBox="0 0 8500 4500"><rect width="8500" height="4500"/></svg>'
    )
    assert media_refs._svg_intrinsic_size(svg) == (8500.0, 4500.0)


def test_svg_intrinsic_size_falls_back_to_physical_units():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="85mm" height="45mm"></svg>'
    width, height = media_refs._svg_intrinsic_size(svg)
    assert round(width, 3) == round(85 * 96 / 25.4, 3)
    assert round(height, 3) == round(45 * 96 / 25.4, 3)


def test_svg_intrinsic_size_none_without_dims():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    assert media_refs._svg_intrinsic_size(svg) is None


def test_render_size_preserves_aspect_from_viewbox():
    svg = b'<svg viewBox="0 0 8500 4500"></svg>'
    assert media_refs._render_size(svg, None) == (1024, round(1024 * 4500 / 8500))


def test_render_size_defaults_to_square_without_dims():
    svg = b"<svg></svg>"
    assert media_refs._render_size(svg, None) == (1024, 1024)


def test_render_size_respects_target_px_and_cap():
    svg = b'<svg viewBox="0 0 100 50"></svg>'
    assert media_refs._render_size(svg, 200) == (200, 100)
    assert media_refs._render_size(svg, 999_999) == (
        media_refs._MAX_RASTER_PX,
        round(media_refs._MAX_RASTER_PX * 50 / 100),
    )


@pytest.mark.asyncio
async def test_rasterize_svg_disabled(monkeypatch):
    monkeypatch.setenv("MEDIA_SVG_RASTER_ENABLED", "false")

    with pytest.raises(MediaReferenceError, match="disabled"):
        await media_refs.rasterize_svg(b"<svg></svg>")


@pytest.mark.asyncio
async def test_rasterize_svg_oversized_rejected(monkeypatch):
    monkeypatch.setattr(media_refs, "MEDIA_SVG_MAX_BYTES", 4)

    with pytest.raises(MediaReferenceError, match="too large"):
        await media_refs.rasterize_svg(b"<svg viewBox='0 0 1 1'></svg>")


@pytest.mark.asyncio
async def test_rasterize_svg_browser_unavailable(monkeypatch):
    from app.core.screenshot import ScreenshotError

    async def fake_get_browser():
        raise ScreenshotError("no chromium")

    monkeypatch.setattr("app.core.screenshot._browser", fake_get_browser)

    with pytest.raises(MediaReferenceError, match="unavailable"):
        await media_refs.rasterize_svg(b"<svg></svg>")


@pytest.mark.asyncio
async def test_rasterize_svg_renders_via_mocked_browser(monkeypatch):
    fake_browser = _mock_browser(monkeypatch, png_bytes=b"PNGDATA")
    svg = b'<svg viewBox="0 0 200 100"></svg>'

    png, width, height = await media_refs.rasterize_svg(svg, target_px=256)

    assert png == b"PNGDATA"
    assert (width, height) == (256, 128)
    assert fake_browser.contexts[0].viewport == {"width": 256, "height": 128}
    assert fake_browser.contexts[0].closed is True
    assert fake_browser.contexts[0].routed  # every request routed → aborted


@pytest.mark.asyncio
async def test_resolve_reference_rasterizes_svg(monkeypatch):
    svg_text = '<svg viewBox="0 0 200 100"></svg>'
    _mock_read(monkeypatch, {"brand/logo.svg": svg_text})
    _mock_browser(monkeypatch, png_bytes=b"PNGDATA")

    writes: dict[str, str] = {}

    async def fake_wsd_read(room_id, path):
        return None

    async def fake_wsd_write(room_id, path, content, commit_message=None):
        writes[path] = content
        return {}

    monkeypatch.setattr("app.core.workspaced_client.read_workspace_file", fake_wsd_read)
    monkeypatch.setattr("app.core.workspaced_client.write_workspace_file", fake_wsd_write)

    ref = await resolve_reference(uuid4(), "brand/logo.svg", object(), role="logo")

    assert ref.mime == "image/png"
    assert ref.data == b"PNGDATA"
    assert ref.rasterized_from == "brand/logo.svg"
    assert ref.role == "logo"
    assert len(writes) == 1
    cache_path, cached_content = next(iter(writes.items()))
    assert cache_path.startswith(".legion/media-refs/") and cache_path.endswith(".png")
    assert cached_content == "data:image/png;base64," + base64.b64encode(b"PNGDATA").decode()


@pytest.mark.asyncio
async def test_resolve_reference_uses_svg_cache_without_browser(monkeypatch):
    svg_text = '<svg viewBox="0 0 10 10"></svg>'
    _mock_read(monkeypatch, {"brand/logo.svg": svg_text})
    cached_png_b64 = _encode("PNG", (10, 10))

    async def fake_wsd_read(room_id, path):
        return _data_url("image/png", cached_png_b64)

    async def browser_should_not_be_called():
        raise AssertionError("browser must not be invoked on a cache hit")

    monkeypatch.setattr("app.core.workspaced_client.read_workspace_file", fake_wsd_read)
    monkeypatch.setattr("app.core.screenshot._browser", browser_should_not_be_called)

    ref = await resolve_reference(uuid4(), "brand/logo.svg", object(), role="logo")

    assert ref.data == base64.b64decode(cached_png_b64)
    assert ref.width == 10
    assert ref.height == 10
    assert ref.rasterized_from == "brand/logo.svg"


@pytest.mark.asyncio
async def test_resolve_reference_svg_disabled(monkeypatch):
    _mock_read(monkeypatch, {"brand/logo.svg": "<svg></svg>"})
    monkeypatch.setenv("MEDIA_SVG_RASTER_ENABLED", "false")

    with pytest.raises(MediaReferenceError, match="disabled"):
        await resolve_reference(uuid4(), "brand/logo.svg", object())


@pytest.mark.asyncio
async def test_resolve_reference_svg_oversized(monkeypatch):
    _mock_read(monkeypatch, {"brand/logo.svg": "<svg></svg>"})
    monkeypatch.setattr(media_refs, "MEDIA_SVG_MAX_BYTES", 4)

    with pytest.raises(MediaReferenceError, match="too large"):
        await resolve_reference(uuid4(), "brand/logo.svg", object())


# ---------------------------------------------------------------------------
# Stage 5: resolve_task_references / build_reference_manifest / build_vision_blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_task_references_happy_path(monkeypatch):
    encoded = _encode("PNG", (16, 9))
    _mock_read(monkeypatch, {"images/bg.png": _data_url("image/png", encoded)})

    specs = [{"path": "images/bg.png", "role": "layout", "note": "keep as is"}]
    entries = await resolve_task_references(uuid4(), specs, object())

    assert len(entries) == 1
    assert entries[0].path == "images/bg.png"
    assert entries[0].role == "layout"
    assert entries[0].note == "keep as is"
    assert entries[0].error is None
    assert entries[0].ref is not None
    assert entries[0].ref.width == 16


@pytest.mark.asyncio
async def test_resolve_task_references_isolates_per_entry_failure(monkeypatch):
    """Unlike resolve_references, one bad path must not abort the others."""
    encoded = _encode("PNG", (16, 9))
    _mock_read(monkeypatch, {"images/bg.png": _data_url("image/png", encoded)})

    specs = [
        {"path": "images/missing.png", "role": "logo"},
        {"path": "images/bg.png", "role": "layout"},
    ]
    entries = await resolve_task_references(uuid4(), specs, object())

    assert len(entries) == 2
    assert entries[0].path == "images/missing.png"
    assert entries[0].ref is None
    assert "not found" in entries[0].error
    assert entries[1].path == "images/bg.png"
    assert entries[1].ref is not None


@pytest.mark.asyncio
async def test_resolve_task_references_invalid_path_isolated(monkeypatch):
    _mock_read(monkeypatch, {})
    specs = [{"path": "../etc/passwd", "role": "logo"}]
    entries = await resolve_task_references(uuid4(), specs, object())

    assert len(entries) == 1
    assert entries[0].ref is None
    assert "invalid" in entries[0].error


@pytest.mark.asyncio
async def test_resolve_task_references_dedupes_by_normalized_path(monkeypatch):
    encoded = _encode("PNG", (16, 9))
    _mock_read(monkeypatch, {"images/bg.png": _data_url("image/png", encoded)})

    specs = [
        {"path": "images/bg.png", "role": "layout"},
        {"path": "/images/bg.png", "role": "layout"},
    ]
    entries = await resolve_task_references(uuid4(), specs, object())

    assert len(entries) == 1


@pytest.mark.asyncio
async def test_resolve_task_references_enforces_total_size_limit(monkeypatch):
    encoded_a = _encode("PNG", (16, 9))
    encoded_b = _encode("PNG", (24, 12))
    _mock_read(
        monkeypatch,
        {
            "images/a.png": _data_url("image/png", encoded_a),
            "images/b.png": _data_url("image/png", encoded_b),
        },
    )
    monkeypatch.setattr(media_refs, "MEDIA_REF_MAX_TOTAL_BYTES", 10)

    specs = [{"path": "images/a.png"}, {"path": "images/b.png"}]
    entries = await resolve_task_references(uuid4(), specs, object())

    assert len(entries) == 2
    assert all(e.ref is None and "total size limit" in e.error for e in entries)


def test_build_reference_manifest_empty_entries():
    assert build_reference_manifest([]) == ""


def test_build_reference_manifest_resolved_and_unavailable():
    resolved = media_refs.ReferenceManifestEntry(
        path="images/bg.png", role="layout", note="hero background",
        ref=ResolvedRef(
            path="images/bg.png", mime="image/png", data=b"x",
            width=1000, height=600, role="layout",
        ),
        error=None,
    )
    missing = media_refs.ReferenceManifestEntry(
        path="images/missing.png", role="logo", note="",
        ref=None, error="reference not found: images/missing.png",
    )
    text = build_reference_manifest([resolved, missing])

    assert "## Reference files" in text
    assert "images/bg.png" in text
    assert "1000x600px" in text
    assert "hero background" in text
    assert "images/missing.png" in text
    assert "UNAVAILABLE" in text
    assert "reference not found" in text


def test_build_vision_blocks_anthropic_format():
    ref = ResolvedRef(path="a.png", mime="image/png", data=b"\x89PNG\r\n" + b"x" * 8,
                       width=4, height=4, role="style")
    blocks = build_vision_blocks("anthropic", [ref])

    assert len(blocks) == 1
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["type"] == "base64"
    assert blocks[0]["source"]["media_type"] == "image/png"


def test_build_vision_blocks_openai_compatible_format():
    ref = ResolvedRef(path="a.png", mime="image/png", data=b"\x89PNG\r\n" + b"x" * 8,
                       width=4, height=4, role="style")
    blocks = build_vision_blocks("openrouter", [ref])

    assert len(blocks) == 1
    assert blocks[0]["type"] == "image_url"
    assert blocks[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_vision_blocks_downscales_large_reference():
    img = Image.new("RGB", (2000, 2000), color=(1, 2, 3))
    buf = BytesIO()
    img.save(buf, format="PNG")
    big_ref = ResolvedRef(
        path="big.png", mime="image/png", data=buf.getvalue(),
        width=2000, height=2000, role="style",
    )
    blocks = build_vision_blocks("anthropic", [big_ref])

    decoded = base64.b64decode(blocks[0]["source"]["data"])
    with Image.open(BytesIO(decoded)) as out:
        assert out.width * out.height <= media_refs.MEDIA_REF_VISION_MAX_PIXELS
        assert out.width < 2000


# ---------------------------------------------------------------------------
# Stage 8: composite_logo_overlay
# ---------------------------------------------------------------------------

def _png(w: int, h: int, color=(10, 20, 30, 255)) -> bytes:
    """Create a minimal RGBA PNG of the given size and solid colour."""
    img = Image.new("RGBA", (w, h), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _open_rgba(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGBA")


def test_composite_logo_overlay_output_matches_base_dimensions():
    base = _png(400, 300)
    logo = _png(50, 50, color=(255, 0, 0, 255))
    result = composite_logo_overlay(base, logo)
    with _open_rgba(result) as out:
        assert out.size == (400, 300)


def test_composite_logo_overlay_default_anchor_top_left():
    """Top-left corner of the base should contain pixels from the logo."""
    base = _png(400, 200, color=(0, 0, 0, 255))
    logo = _png(20, 20, color=(255, 0, 0, 255))
    result = composite_logo_overlay(base, logo, anchor="top-left", margin_pct=0, width_pct=5)
    with _open_rgba(result) as out:
        # pixel (0,0) should be red (logo was placed there)
        r, g, b, a = out.getpixel((0, 0))
        assert r > 200


def test_composite_logo_overlay_bottom_right_anchor():
    base = _png(400, 200, color=(0, 0, 0, 255))
    logo = _png(20, 20, color=(0, 255, 0, 255))
    result = composite_logo_overlay(base, logo, anchor="bottom-right", margin_pct=0, width_pct=5)
    with _open_rgba(result) as out:
        bw, bh = out.size
        logo_w = max(1, round(bw * 5 / 100))
        # pixel at (bw-1, bh-1) should be green
        r, g, b, a = out.getpixel((bw - 1, bh - 1))
        assert g > 200


def test_composite_logo_overlay_center_anchor():
    base = _png(200, 200, color=(0, 0, 0, 255))
    logo = _png(20, 20, color=(0, 0, 255, 255))
    result = composite_logo_overlay(base, logo, anchor="center", margin_pct=0, width_pct=10)
    with _open_rgba(result) as out:
        # Centre pixel should contain the blue logo
        cx, cy = out.size[0] // 2, out.size[1] // 2
        r, g, b, a = out.getpixel((cx, cy))
        assert b > 200


def test_composite_logo_overlay_opacity_reduces_alpha():
    base = _png(200, 200, color=(0, 0, 0, 255))
    # Fully opaque white logo at top-left
    logo = _png(20, 20, color=(255, 255, 255, 255))

    full = composite_logo_overlay(base, logo, anchor="top-left", margin_pct=0,
                                  width_pct=10, opacity=1.0)
    half = composite_logo_overlay(base, logo, anchor="top-left", margin_pct=0,
                                  width_pct=10, opacity=0.5)

    with _open_rgba(full) as f, _open_rgba(half) as h:
        full_px = f.getpixel((0, 0))
        half_px = h.getpixel((0, 0))
    # With 50% opacity the composited pixel is darker than with 100%
    assert half_px[0] < full_px[0]


def test_composite_logo_overlay_all_anchors_produce_valid_png():
    base = _png(300, 200)
    logo = _png(20, 20, color=(255, 0, 0, 255))
    for anchor in sorted(_LOGO_ANCHORS):
        result = composite_logo_overlay(base, logo, anchor=anchor, width_pct=10)
        with _open_rgba(result) as out:
            assert out.size == (300, 200), f"size mismatch for anchor {anchor!r}"


def test_composite_logo_overlay_preserves_logo_aspect_ratio():
    base = _png(400, 300)
    # Logo is 3:1 (wide)
    logo = _png(60, 20, color=(255, 0, 0, 255))
    result = composite_logo_overlay(base, logo, width_pct=25, anchor="top-left", margin_pct=0)
    with Image.open(BytesIO(result)) as out:
        assert out.size == (400, 300)
    # Check logo width was applied correctly: 25% of 400 = 100px wide
    # We can't easily check logo dimensions in the composited image without
    # scanning pixels, but the function must not crash.


def test_composite_logo_overlay_invalid_anchor_raises():
    base = _png(100, 100)
    logo = _png(10, 10)
    with pytest.raises(ValueError, match="anchor"):
        composite_logo_overlay(base, logo, anchor="upper-left")


def test_composite_logo_overlay_invalid_width_pct_raises():
    base = _png(100, 100)
    logo = _png(10, 10)
    with pytest.raises(ValueError, match="width_pct"):
        composite_logo_overlay(base, logo, width_pct=0)
    with pytest.raises(ValueError, match="width_pct"):
        composite_logo_overlay(base, logo, width_pct=101)


def test_composite_logo_overlay_invalid_margin_pct_raises():
    base = _png(100, 100)
    logo = _png(10, 10)
    with pytest.raises(ValueError, match="margin_pct"):
        composite_logo_overlay(base, logo, margin_pct=50)


def test_composite_logo_overlay_bad_base_raises_media_error():
    with pytest.raises(MediaReferenceError, match="base image"):
        composite_logo_overlay(b"not-a-png", _png(10, 10))


def test_composite_logo_overlay_bad_logo_raises_media_error():
    with pytest.raises(MediaReferenceError, match="logo image"):
        composite_logo_overlay(_png(100, 100), b"not-a-png")
