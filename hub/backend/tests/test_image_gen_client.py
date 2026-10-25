"""Unit tests for core.image_gen.generate_image.

No real HTTP — httpx.AsyncClient.post is mocked via unittest.mock.
Tests: mocked HTTP response, b64 decoding, retry on 429/503/529, non-retriable
errors raised, new fields (size, input_references), payload hygiene, redaction.
"""

from __future__ import annotations

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.image_gen import ImageGenHTTPError, generate_image


def _png_b64() -> str:
    return base64.b64encode(b"FAKEPNG").decode()


def _mock_response(
    status_code: int,
    b64: str | None = None,
    text: str = "",
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    if b64 is not None:
        resp.json.return_value = {"data": [{"b64_json": b64, "media_type": "image/png"}]}
    return resp


def _make_client(mock_post: AsyncMock) -> MagicMock:
    """Wire up the httpx.AsyncClient context-manager mock."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=MagicMock(post=mock_post))
    client.__aexit__ = AsyncMock(return_value=False)
    return client


# ---------------------------------------------------------------------------
# Basic happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_successful_response_returns_decoded_bytes():
    b64 = _png_b64()
    mock_post = AsyncMock(return_value=_mock_response(200, b64))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        result = await generate_image(
            api_key="sk-test",
            prompt="a red square",
            model="test-model",
            aspect_ratio="1:1",
            base_url="https://api.example.com/v1",
        )

    assert result == b"FAKEPNG"
    mock_post.assert_awaited_once()
    sent = mock_post.await_args.kwargs["json"]
    assert sent["prompt"] == "a red square"
    assert sent["aspect_ratio"] == "1:1"
    assert "Bearer sk-test" in mock_post.await_args.kwargs["headers"]["Authorization"]


# ---------------------------------------------------------------------------
# Payload hygiene
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_none_keys_in_payload():
    """Keys for absent optional fields must not appear in the payload at all."""
    mock_post = AsyncMock(return_value=_mock_response(200, _png_b64()))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        await generate_image(api_key="k", prompt="p", aspect_ratio=None)

    sent = mock_post.await_args.kwargs["json"]
    assert "aspect_ratio" not in sent
    assert "size" not in sent
    assert "resolution" not in sent
    assert "input_references" not in sent


@pytest.mark.asyncio
async def test_size_overrides_aspect_ratio():
    """When size is given, aspect_ratio must NOT appear in the payload."""
    mock_post = AsyncMock(return_value=_mock_response(200, _png_b64()))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        await generate_image(
            api_key="k",
            prompt="p",
            aspect_ratio="16:9",
            size="1000x100",
        )

    sent = mock_post.await_args.kwargs["json"]
    assert sent["size"] == "1000x100"
    assert "aspect_ratio" not in sent


@pytest.mark.asyncio
async def test_resolution_excluded_when_size_present():
    """resolution must not be sent alongside size (causes 400 from provider)."""
    mock_post = AsyncMock(return_value=_mock_response(200, _png_b64()))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        await generate_image(
            api_key="k",
            prompt="p",
            size="2048x2048",
            resolution="2K",
        )

    sent = mock_post.await_args.kwargs["json"]
    assert "size" in sent
    assert "resolution" not in sent


@pytest.mark.asyncio
async def test_resolution_included_without_size():
    mock_post = AsyncMock(return_value=_mock_response(200, _png_b64()))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        await generate_image(
            api_key="k",
            prompt="p",
            aspect_ratio="4:3",
            resolution="2K",
        )

    sent = mock_post.await_args.kwargs["json"]
    assert sent["resolution"] == "2K"
    assert "size" not in sent


# ---------------------------------------------------------------------------
# input_references
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_input_references_wrapped_correctly():
    mock_post = AsyncMock(return_value=_mock_response(200, _png_b64()))
    refs = ["data:image/png;base64,ABC", "https://example.com/img.png"]

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        await generate_image(api_key="k", prompt="p", input_references=refs)

    sent = mock_post.await_args.kwargs["json"]
    assert sent["input_references"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABC"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
    ]


@pytest.mark.asyncio
async def test_empty_input_references_not_sent():
    mock_post = AsyncMock(return_value=_mock_response(200, _png_b64()))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        await generate_image(api_key="k", prompt="p", input_references=[])

    sent = mock_post.await_args.kwargs["json"]
    assert "input_references" not in sent


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_on_429_then_success():
    b64 = _png_b64()
    mock_post = AsyncMock(side_effect=[_mock_response(429), _mock_response(200, b64)])

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await generate_image(api_key="sk-test", prompt="retry me")

    assert result == b"FAKEPNG"
    assert mock_post.await_count == 2
    mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_retry_on_503_then_success():
    b64 = _png_b64()
    mock_post = AsyncMock(side_effect=[_mock_response(503), _mock_response(200, b64)])

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await generate_image(api_key="sk-test", prompt="service unavailable")

    assert result == b"FAKEPNG"
    assert mock_post.await_count == 2
    mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_retry_on_529_then_success():
    b64 = _png_b64()
    mock_post = AsyncMock(side_effect=[_mock_response(529), _mock_response(200, b64)])

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await generate_image(api_key="sk-test", prompt="overloaded")

    assert result == b"FAKEPNG"
    mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_double_429_raises_image_gen_http_error():
    """Second retry-eligible response is not retried — raises ImageGenHTTPError."""
    mock_post = AsyncMock(
        side_effect=[_mock_response(429), _mock_response(429, text="rate limited")]
    )

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ImageGenHTTPError) as exc_info:
            await generate_image(api_key="sk-test", prompt="double rate limit")

    assert exc_info.value.status == 429
    assert mock_post.await_count == 2


# ---------------------------------------------------------------------------
# ImageGenHTTPError — 4xx body exposed, secrets redacted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_400_raises_image_gen_http_error_with_body():
    body = '{"error": "size parameter not supported by this model"}'
    mock_post = AsyncMock(return_value=_mock_response(400, text=body))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        with pytest.raises(ImageGenHTTPError) as exc_info:
            await generate_image(api_key="sk-test", prompt="bad request", size="9999x9999")

    err = exc_info.value
    assert err.status == 400
    assert "size parameter" in err.body_snippet
    assert mock_post.await_count == 1  # 400 is not retried


@pytest.mark.asyncio
async def test_4xx_body_redacts_api_key_and_base_url():
    api_key = "sk-supersecret-key"
    base_url = "https://my-private-proxy.example.com/v1"
    body = f'auth failed for {api_key} at {base_url}/images'
    mock_post = AsyncMock(return_value=_mock_response(401, text=body))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        with pytest.raises(ImageGenHTTPError) as exc_info:
            await generate_image(api_key=api_key, prompt="p", base_url=base_url)

    snippet = exc_info.value.body_snippet
    assert api_key not in snippet
    assert "my-private-proxy.example.com" not in snippet
    assert "[redacted]" in snippet


@pytest.mark.asyncio
async def test_500_raises_image_gen_http_error():
    mock_post = AsyncMock(return_value=_mock_response(500, text="internal error"))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        with pytest.raises(ImageGenHTTPError) as exc_info:
            await generate_image(api_key="k", prompt="p")

    assert exc_info.value.status == 500


# ---------------------------------------------------------------------------
# Back-compat: old positional call style still works
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_back_compat_positional_args():
    """Old callers using generate_image(key, prompt, model, aspect_ratio, base_url)
    must continue to work without modification."""
    mock_post = AsyncMock(return_value=_mock_response(200, _png_b64()))

    with patch("httpx.AsyncClient", return_value=_make_client(mock_post)):
        result = await generate_image(
            "sk-compat",
            "legacy call",
            "some-model",
            "4:3",
            "https://api.example.com/v1",
        )

    assert result == b"FAKEPNG"
    sent = mock_post.await_args.kwargs["json"]
    assert sent["aspect_ratio"] == "4:3"
    assert sent["model"] == "some-model"
