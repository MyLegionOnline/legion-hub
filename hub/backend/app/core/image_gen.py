"""Image generation client for OpenRouter POST /images endpoint.

Returns the raw image bytes exactly as the provider encoded them — this is
often PNG but not always (OpenRouter + Google models frequently answer with
JPEG). The caller is responsible for sniffing the real format (magic bytes)
before choosing a file extension / ``data:`` MIME — never assume PNG.
"""

import asyncio
import base64
import logging
import re

import httpx

from app.core.llm_providers import openrouter_attribution_headers

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "google/gemini-3.1-flash-lite-image"
_DEFAULT_ASPECT_RATIO = "1:1"
_TIMEOUT = 120.0

# HTTP status codes that warrant a single retry with backoff.
_RETRY_STATUSES = {429, 503, 529}


class ImageGenHTTPError(Exception):
    """4xx/5xx from the image provider, with a redacted body snippet."""

    def __init__(self, status: int, body_snippet: str) -> None:
        self.status = status
        self.body_snippet = body_snippet
        super().__init__(f"image_gen: HTTP {status} — {body_snippet}")


def _redact_secrets(text: str, api_key: str, base_url: str) -> str:
    """Remove api_key and base_url from error text before surfacing to callers."""
    for secret in (api_key, base_url):
        if len(secret) >= 4:
            text = re.sub(re.escape(secret), "[redacted]", text, flags=re.IGNORECASE)
    return text


async def generate_image(
    api_key: str,
    prompt: str,
    model: str = _DEFAULT_MODEL,
    aspect_ratio: str | None = _DEFAULT_ASPECT_RATIO,   # back-compat positional arg
    base_url: str = _DEFAULT_BASE_URL,
    *,
    size: str | None = None,
    resolution: str | None = None,
    input_references: list[str] | None = None,           # data URLs or https URLs
) -> bytes:
    """POST /images and return the decoded image bytes (format as sent by the provider).

    Payload rules (mirrors OpenRouter contract):
    - ``size`` (exact pixels or tier) is mutually exclusive with ``aspect_ratio``
      and takes priority; sending both causes a 400.
    - ``resolution`` is only included when ``size`` is absent.
    - ``input_references`` are wrapped in the image_url object format the API expects.
    - None-valued keys are never serialised into the payload.

    Retries once on 429/503/529 with a 2-second backoff.
    On any 4xx (including a retriable status after exhausting retries), raises
    ``ImageGenHTTPError`` with a truncated, redacted body snippet so callers can
    distinguish "model doesn't support size" from "out of credits".
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **openrouter_attribution_headers(base_url),
    }

    payload: dict = {"model": model, "prompt": prompt}
    if size:
        payload["size"] = size
    elif aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio
    if resolution and not size:
        payload["resolution"] = resolution
    if input_references:
        payload["input_references"] = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in input_references
        ]

    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(2):
            resp = await client.post(
                f"{base_url}/images",
                headers=headers,
                json=payload,
            )
            if resp.status_code in _RETRY_STATUSES and attempt == 0:
                logger.warning(
                    "image_gen: %s (attempt %d), retrying in 2s",
                    resp.status_code,
                    attempt,
                )
                await asyncio.sleep(2.0)
                continue
            if resp.status_code >= 400:
                snippet = _redact_secrets(
                    resp.text[:500], api_key=api_key, base_url=base_url
                )
                raise ImageGenHTTPError(resp.status_code, snippet)
            data = resp.json()
            b64 = data["data"][0]["b64_json"]
            return base64.b64decode(b64)

    # Should not be reached — loop always returns or raises on the second attempt
    raise RuntimeError("image_gen: unexpected exit from retry loop")
