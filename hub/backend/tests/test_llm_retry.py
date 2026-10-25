"""
Tests for transient-error retry around individual LLM requests
(proxy_session._is_retryable_llm_error / _llm_request_with_retry).

Bug context: a single dropped connection to OpenRouter killed the whole
proxy session with close_reason=error.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import openai
import pytest

from app.core.proxy_session import (
    _LLM_RETRY_BACKOFF_S,
    _is_retryable_llm_error,
    _llm_request_with_retry,
)


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://api.example.com/v1/messages")
    resp = httpx.Response(status, request=req)
    return httpx.HTTPStatusError(f"HTTP {status}", request=req, response=resp)


def _openai_status_error(status: int) -> openai.APIStatusError:
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    resp = httpx.Response(status, request=req)
    return openai.APIStatusError("boom", response=resp, body=None)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_httpx_transport_errors_are_retryable():
    assert _is_retryable_llm_error(httpx.ConnectError("refused"))
    assert _is_retryable_llm_error(httpx.ReadTimeout("timeout"))
    assert _is_retryable_llm_error(httpx.RemoteProtocolError("server disconnected"))


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 529])
def test_retryable_http_statuses(status):
    assert _is_retryable_llm_error(_http_status_error(status))
    assert _is_retryable_llm_error(_openai_status_error(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_non_retryable_http_statuses(status):
    assert not _is_retryable_llm_error(_http_status_error(status))
    assert not _is_retryable_llm_error(_openai_status_error(status))


def test_openai_connection_error_is_retryable():
    req = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    exc = openai.APIConnectionError(request=req)
    assert _is_retryable_llm_error(exc)
    # APITimeoutError subclasses APIConnectionError
    assert _is_retryable_llm_error(openai.APITimeoutError(request=req))


def test_generic_errors_are_not_retryable():
    assert not _is_retryable_llm_error(ValueError("bad json"))
    assert not _is_retryable_llm_error(KeyError("choices"))


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_succeeds_after_transient_failures():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ConnectError("refused")
        return {"ok": True}

    with patch("app.core.proxy_session.asyncio.sleep", new_callable=AsyncMock):
        result = await _llm_request_with_retry(flaky, session_id=uuid4(), label="test")
    assert result == {"ok": True}
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_non_retryable_raises_immediately():
    calls = {"n": 0}

    async def bad_request():
        calls["n"] += 1
        raise _http_status_error(401)

    with patch("app.core.proxy_session.asyncio.sleep", new_callable=AsyncMock) as ms:
        with pytest.raises(httpx.HTTPStatusError):
            await _llm_request_with_retry(bad_request, session_id=uuid4(), label="test")
    assert calls["n"] == 1
    ms.assert_not_awaited()


@pytest.mark.asyncio
async def test_gives_up_after_all_attempts():
    calls = {"n": 0}

    async def always_down():
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    with patch("app.core.proxy_session.asyncio.sleep", new_callable=AsyncMock) as ms:
        with pytest.raises(httpx.ConnectError):
            await _llm_request_with_retry(always_down, session_id=uuid4(), label="test")
    assert calls["n"] == len(_LLM_RETRY_BACKOFF_S) + 1
    assert ms.await_count == len(_LLM_RETRY_BACKOFF_S)


@pytest.mark.asyncio
async def test_backoff_delays_are_used():
    async def always_down():
        raise httpx.ConnectError("refused")

    with patch("app.core.proxy_session.asyncio.sleep", new_callable=AsyncMock) as ms:
        with pytest.raises(httpx.ConnectError):
            await _llm_request_with_retry(always_down, session_id=uuid4(), label="test")
    delays = [c.args[0] for c in ms.await_args_list]
    assert delays == list(_LLM_RETRY_BACKOFF_S)
