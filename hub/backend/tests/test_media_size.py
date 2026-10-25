"""Unit tests for Stage 3: media_size.py (internal design notes).

Tests cover:
  - parse_size: valid pixel strings, tier strings, invalid inputs
  - resolve_size: three-level priority chain
  - build_provider_params: size/aspect_ratio mutual exclusion, nearest-ratio
    selection for extreme ratios (1000x100 → 21:9 or 16:9), degraded list
  - fit_exact: cover / contain / stretch all produce exactly W×H
  - model_capabilities: in-memory cache does not hit the network on second call
"""
from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from app.core.media_size import (
    SizeSpec,
    _clear_cap_cache,
    _nearest_ratio,
    build_provider_params,
    fit_exact,
    model_capabilities,
    parse_size,
    resolve_size,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(w: int = 160, h: int = 90) -> bytes:
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _dimensions(png: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(png)) as img:
        return img.size


# ---------------------------------------------------------------------------
# parse_size
# ---------------------------------------------------------------------------


class TestParseSize:
    def test_exact_pixels_basic(self):
        assert parse_size("1000x100") == (1000, 100)

    def test_exact_pixels_uppercase_x(self):
        assert parse_size("800X600") == (800, 600)

    def test_exact_pixels_square(self):
        assert parse_size("256x256") == (256, 256)

    def test_exact_pixels_min_valid(self):
        assert parse_size("16x16") == (16, 16)

    def test_exact_pixels_max_valid(self):
        # 8192×8192 would exceed the 16 Mpx area limit; use a side at the max but area within limit.
        assert parse_size("8192x16") == (8192, 16)

    def test_tier_1k(self):
        assert parse_size("1K") == "1K"

    def test_tier_2k(self):
        assert parse_size("2K") == "2K"

    def test_tier_4k(self):
        assert parse_size("4K") == "4K"

    def test_tier_512(self):
        assert parse_size("512") == "512"

    def test_side_too_small_raises(self):
        with pytest.raises(ValueError, match="at least"):
            parse_size("8x256")

    def test_zero_side_raises(self):
        with pytest.raises(ValueError):
            parse_size("0x100")

    def test_side_too_large_raises(self):
        with pytest.raises(ValueError, match="at most"):
            parse_size("10000x100")

    def test_area_too_large_raises(self):
        with pytest.raises(ValueError, match="16 Mpx"):
            parse_size("5000x4000")  # 20 Mpx > 16 Mpx limit

    def test_area_limit_enforced(self):
        # 5000x4000 = 20 Mpx → over 16 Mpx
        with pytest.raises(ValueError, match="16 Mpx"):
            parse_size("5000x4000")

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="invalid size"):
            parse_size("large")

    def test_aspect_ratio_string_raises(self):
        with pytest.raises(ValueError):
            parse_size("16:9")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_size("")


# ---------------------------------------------------------------------------
# resolve_size
# ---------------------------------------------------------------------------


class TestResolveSize:
    def test_per_image_wins_over_task(self):
        per_image = {"aspect_ratio": "1:1"}
        task_spec = {"size": {"aspect_ratio": "16:9"}}
        spec = resolve_size(per_image, task_spec, None)
        assert spec.aspect_ratio == "1:1"

    def test_task_wins_over_agent(self):
        task_spec = {"size": {"aspect_ratio": "9:16"}}
        agent = {"aspect_ratio": "4:3"}
        spec = resolve_size(None, task_spec, agent)
        assert spec.aspect_ratio == "9:16"

    def test_agent_used_when_no_per_image_or_task(self):
        agent = {"aspect_ratio": "21:9"}
        spec = resolve_size(None, None, agent)
        assert spec.aspect_ratio == "21:9"

    def test_default_when_all_none(self):
        spec = resolve_size(None, None, None)
        assert spec.aspect_ratio == "4:3"
        assert spec.size is None

    def test_per_image_size_wins(self):
        per_image = {"size": "1000x100"}
        task_spec = {"size": {"aspect_ratio": "16:9"}}
        spec = resolve_size(per_image, task_spec, None)
        assert spec.size == "1000x100"

    def test_fit_propagates(self):
        per_image = {"aspect_ratio": "1:1", "fit": "contain"}
        spec = resolve_size(per_image, None, None)
        assert spec.fit == "contain"

    def test_invalid_fit_normalised_to_cover(self):
        per_image = {"aspect_ratio": "1:1", "fit": "BOGUS"}
        spec = resolve_size(per_image, None, None)
        assert spec.fit == "cover"

    def test_task_spec_without_size_subkey_is_skipped(self):
        # task_spec with no "size" sub-object → falls through to agent
        agent = {"aspect_ratio": "3:2"}
        spec = resolve_size(None, {"references": []}, agent)
        assert spec.aspect_ratio == "3:2"

    def test_agent_default_size_alias(self):
        # stage 7 adds default_size alias
        agent = {"default_size": "2K"}
        spec = resolve_size(None, None, agent)
        assert spec.size == "2K"


# ---------------------------------------------------------------------------
# build_provider_params
# ---------------------------------------------------------------------------


def _caps_with_ratios(ratios: list[str], *, has_size: bool = True) -> dict:
    caps: dict = {"aspect_ratio": {"type": "enum", "values": ratios}}
    if has_size:
        caps["size"] = {"type": "boolean"}
    return caps


class TestBuildProviderParams:
    def test_exact_pixels_with_size_support(self):
        spec = SizeSpec(size="1000x100")
        caps = _caps_with_ratios(["16:9", "21:9"], has_size=True)
        params, target, degraded = build_provider_params(spec, caps)
        # Must send ONLY size, never together with aspect_ratio
        assert "size" in params
        assert "aspect_ratio" not in params
        assert "resolution" not in params
        assert target is None
        assert degraded == []

    def test_exact_pixels_without_size_support_uses_nearest_ratio(self):
        spec = SizeSpec(size="1000x100")  # ratio ≈ 10:1, widest → 21:9 closest
        caps = _caps_with_ratios(["1:1", "4:3", "16:9", "9:16", "21:9"], has_size=False)
        params, target, degraded = build_provider_params(spec, caps)
        assert "size" not in params
        assert params.get("aspect_ratio") == "21:9"
        assert target == (1000, 100)   # post-process required
        assert "size" in degraded

    def test_exact_pixels_without_size_or_21_9(self):
        spec = SizeSpec(size="1000x100")
        caps = _caps_with_ratios(["1:1", "4:3", "16:9", "9:16"], has_size=False)
        params, target, degraded = build_provider_params(spec, caps)
        assert params.get("aspect_ratio") == "16:9"   # widest available
        assert target == (1000, 100)

    def test_aspect_ratio_sent_as_is_when_supported(self):
        spec = SizeSpec(aspect_ratio="16:9")
        caps = _caps_with_ratios(["16:9", "4:3"])
        params, target, degraded = build_provider_params(spec, caps)
        assert params["aspect_ratio"] == "16:9"
        assert target is None
        assert degraded == []

    def test_aspect_ratio_nearest_when_unsupported(self):
        spec = SizeSpec(aspect_ratio="3:2")   # not in list
        caps = _caps_with_ratios(["1:1", "4:3", "16:9"])
        params, target, degraded = build_provider_params(spec, caps)
        # 3:2 = 1.5; 4:3 ≈ 1.33; 16:9 ≈ 1.78 → 4:3 is closer to 1.5
        assert params["aspect_ratio"] == "4:3"
        assert "aspect_ratio" in degraded

    def test_no_size_info_sends_default_ratio(self):
        spec = SizeSpec()
        params, target, degraded = build_provider_params(spec, None)
        assert params.get("aspect_ratio") == "4:3"
        assert target is None

    def test_size_never_sent_with_aspect_ratio_together(self):
        spec = SizeSpec(size="512x512", aspect_ratio="1:1")
        caps = {"size": {"type": "boolean"}}
        params, _, _ = build_provider_params(spec, caps)
        # size wins; aspect_ratio must NOT appear
        assert "size" in params
        assert "aspect_ratio" not in params

    def test_tier_size_sent_via_size_param(self):
        spec = SizeSpec(size="2K")
        caps = {"size": {"type": "boolean"}}
        params, target, degraded = build_provider_params(spec, caps)
        assert params["size"] == "2K"
        assert target is None
        assert degraded == []

    def test_optimistic_when_caps_none(self):
        spec = SizeSpec(aspect_ratio="16:9")
        params, _, degraded = build_provider_params(spec, None)
        assert params["aspect_ratio"] == "16:9"
        assert degraded == []

    def test_resolution_included_when_no_size(self):
        spec = SizeSpec(aspect_ratio="4:3", resolution="2K")
        caps = _caps_with_ratios(["4:3"])
        params, _, _ = build_provider_params(spec, caps)
        assert params.get("aspect_ratio") == "4:3"
        assert params.get("resolution") == "2K"


# ---------------------------------------------------------------------------
# _nearest_ratio
# ---------------------------------------------------------------------------


class TestNearestRatio:
    def test_exact_match(self):
        assert _nearest_ratio(16 / 9, ["1:1", "4:3", "16:9"]) == "16:9"

    def test_wide_banner_prefers_21_9(self):
        # 1000/100 = 10.0; log closest to log(21/9) = log(2.33)
        assert _nearest_ratio(10.0, ["1:1", "4:3", "16:9", "9:16", "21:9"]) == "21:9"

    def test_portrait_prefers_9_16(self):
        assert _nearest_ratio(9 / 16, ["1:1", "4:3", "16:9", "9:16"]) == "9:16"

    def test_square_prefers_1_1(self):
        assert _nearest_ratio(1.0, ["1:1", "4:3", "16:9"]) == "1:1"


# ---------------------------------------------------------------------------
# fit_exact
# ---------------------------------------------------------------------------


class TestFitExact:
    def test_cover_exact_dimensions(self):
        png = _make_png(160, 90)   # 16:9
        result = fit_exact(png, 1000, 100, "cover")
        assert _dimensions(result) == (1000, 100)

    def test_contain_exact_dimensions(self):
        png = _make_png(160, 90)
        result = fit_exact(png, 1000, 100, "contain")
        assert _dimensions(result) == (1000, 100)

    def test_stretch_exact_dimensions(self):
        png = _make_png(160, 90)
        result = fit_exact(png, 1000, 100, "stretch")
        assert _dimensions(result) == (1000, 100)

    def test_cover_square_from_wide(self):
        png = _make_png(200, 100)
        result = fit_exact(png, 64, 64, "cover")
        assert _dimensions(result) == (64, 64)

    def test_contain_adds_transparent_padding(self):
        png = _make_png(100, 100)
        result = fit_exact(png, 200, 100, "contain")
        w, h = _dimensions(result)
        assert (w, h) == (200, 100)
        # Padding columns should be transparent
        with Image.open(BytesIO(result)) as img:
            arr = img.convert("RGBA")
            # leftmost pixel should be transparent (padding zone)
            left_px = arr.getpixel((0, 0))
            assert left_px[3] == 0, "Expected transparent padding"

    def test_output_is_always_png(self):
        png = _make_png()
        for mode in ("cover", "contain", "stretch"):
            result = fit_exact(png, 64, 64, mode)
            assert result[:4] == b"\x89PNG", f"Expected PNG header for mode={mode}"

    def test_unknown_mode_defaults_to_cover(self):
        png = _make_png(160, 90)
        result = fit_exact(png, 64, 64, "bogus")
        assert _dimensions(result) == (64, 64)


# ---------------------------------------------------------------------------
# model_capabilities (caching)
# ---------------------------------------------------------------------------


class TestModelCapabilities:
    @pytest.mark.asyncio
    async def test_caches_result_second_call_skips_network(self):
        _clear_cap_cache()
        fake_caps = {"aspect_ratio": {"type": "enum", "values": ["1:1", "4:3"]}}
        fake_body = [{"id": "my-model", "supported_parameters": fake_caps}]

        call_count = 0

        async def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=fake_body)
            return resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            result1 = await model_capabilities(
                "https://openrouter.ai/api/v1", "my-model", "sk-test"
            )
            result2 = await model_capabilities(
                "https://openrouter.ai/api/v1", "my-model", "sk-test"
            )

        assert result1 == fake_caps
        assert result2 == fake_caps
        assert call_count == 1, "Second call should have been served from cache"

    @pytest.mark.asyncio
    async def test_returns_none_on_http_error(self):
        _clear_cap_cache()

        async def fake_get(*args, **kwargs):
            raise httpx.ConnectError("unreachable")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            result = await model_capabilities(
                "https://openrouter.ai/api/v1", "unknown-model", "sk-test"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_model_not_in_list(self):
        _clear_cap_cache()
        fake_body = [{"id": "other-model", "supported_parameters": {}}]

        async def fake_get(*args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=fake_body)
            return resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            result = await model_capabilities(
                "https://openrouter.ai/api/v1", "my-model", "sk-test"
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_api_key_not_stored_in_cache(self):
        """Verify the cache key contains only (base_url, model), no api_key."""
        from app.core.media_size import _cap_cache

        _clear_cap_cache()

        async def fake_get(*args, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json = MagicMock(return_value=[{"id": "mdl", "supported_parameters": {}}])
            return resp

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = fake_get
            mock_client_cls.return_value = mock_client

            await model_capabilities("https://api.example.com/v1", "mdl", "sk-secret")

        for key, (ts, val) in _cap_cache.items():
            assert "sk-secret" not in str(key), "API key must not appear in cache key"
            assert "sk-secret" not in str(val), "API key must not appear in cache value"
