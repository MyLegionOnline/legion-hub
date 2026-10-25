"""
Tests for Stage 3 web tools: web_fetch, web_search, preview_fetch.

All external HTTP calls are mocked — no real network access needed.
"""
from __future__ import annotations

import base64
import ipaddress
import socket
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from app.core.tools import REGISTRY, ToolContext
from app.core.tools._web_fetch import _ssrf_check, _html_to_markdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx() -> ToolContext:
    return ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=None, db=AsyncMock())


def _httpx_response(status: int, content_type: str, body: str | bytes) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.headers = {"content-type": content_type}
    if isinstance(body, str):
        body = body.encode()
    resp.content = body
    resp.is_redirect = False
    resp.url = httpx.URL("http://example.com/")
    resp.reason_phrase = httpx.codes.get_reason_phrase(status)
    return resp


def _httpx_redirect(status: int, location: str) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.headers = {"location": location}
    resp.is_redirect = True
    resp.url = httpx.URL("http://example.com/old")
    return resp


# ---------------------------------------------------------------------------
# _ssrf_check — unit tests (synchronous)
# ---------------------------------------------------------------------------

class TestSsrfCheck:
    def test_blocks_loopback(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("127.0.0.1", 80))]):
            err = _ssrf_check("http://localhost/")
        assert err is not None
        assert "127.0.0.1" in err

    def test_blocks_private_10(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("10.0.0.1", 80))]):
            err = _ssrf_check("http://internal/")
        assert err is not None
        assert "10.0.0.1" in err

    def test_blocks_private_192_168(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("192.168.1.100", 80))]):
            err = _ssrf_check("http://router/")
        assert err is not None

    def test_blocks_link_local_metadata(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("169.254.169.254", 80))]):
            err = _ssrf_check("http://metadata/")
        assert err is not None
        assert "169.254.169.254" in err

    def test_blocks_172_16(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("172.20.0.5", 80))]):
            err = _ssrf_check("http://docker-internal/")
        assert err is not None

    def test_allows_public_ip(self):
        with patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 80))]):
            err = _ssrf_check("https://example.com/")
        assert err is None

    def test_blocks_non_http_scheme(self):
        err = _ssrf_check("ftp://example.com/file")
        assert err is not None
        assert "http" in err.lower()

    def test_blocks_missing_host(self):
        err = _ssrf_check("http:///path")
        assert err is not None

    def test_dns_failure(self):
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name not resolved")):
            err = _ssrf_check("http://doesnotexist.invalid/")
        assert err is not None
        assert "DNS" in err


# ---------------------------------------------------------------------------
# _html_to_markdown — unit tests
# ---------------------------------------------------------------------------

class TestHtmlToMarkdown:
    def test_basic_html(self):
        html = "<h1>Title</h1><p>Hello <b>world</b></p>"
        md = _html_to_markdown(html)
        assert "Title" in md
        assert "Hello" in md
        assert "world" in md

    def test_strips_scripts(self):
        html = "<script>alert('xss')</script><p>Safe content</p>"
        md = _html_to_markdown(html)
        assert "alert" not in md
        assert "Safe content" in md

    def test_strips_style(self):
        html = "<style>.foo { color: red }</style><p>Text</p>"
        md = _html_to_markdown(html)
        assert ".foo" not in md
        assert "Text" in md


# ---------------------------------------------------------------------------
# web_fetch — integration tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestWebFetch:
    @pytest.fixture
    def tool(self):
        t = REGISTRY.get("web_fetch")
        if t is None:
            pytest.skip("web_fetch not registered (WEB_ACCESS_ENABLED not set)")
        return t

    @pytest.mark.asyncio
    async def test_ssrf_blocked(self, tool):
        with patch("app.core.tools._web_fetch._ssrf_check", return_value="Blocked: 127.0.0.1"):
            result = await tool.execute(_ctx(), {"url": "http://localhost/"})
        assert "Error" in result
        assert "127.0.0.1" in result or "Blocked" in result

    @pytest.mark.asyncio
    async def test_fetches_html_as_markdown(self, tool):
        html_body = b"<h1>Hello</h1><p>World</p>"
        mock_resp = _httpx_response(200, "text/html; charset=utf-8", html_body)

        with patch("app.core.tools._web_fetch._ssrf_check", return_value=None), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"url": "https://example.com/"})

        assert "Hello" in result
        assert "World" in result
        assert "Status: 200" in result

    @pytest.mark.asyncio
    async def test_fetches_json_raw(self, tool):
        json_body = b'{"key": "value", "n": 42}'
        mock_resp = _httpx_response(200, "application/json", json_body)

        with patch("app.core.tools._web_fetch._ssrf_check", return_value=None), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"url": "https://api.example.com/data"})

        assert '"key"' in result
        assert '"value"' in result

    @pytest.mark.asyncio
    async def test_http_error_returns_status(self, tool):
        mock_resp = _httpx_response(404, "text/plain", b"Not Found")
        mock_resp.is_redirect = False

        with patch("app.core.tools._web_fetch._ssrf_check", return_value=None), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"url": "https://example.com/missing"})

        assert "404" in result

    @pytest.mark.asyncio
    async def test_truncation_at_max_chars(self, tool):
        large_body = ("A" * 200_000).encode()
        mock_resp = _httpx_response(200, "text/plain", large_body)

        with patch("app.core.tools._web_fetch._ssrf_check", return_value=None), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"url": "https://example.com/big", "max_chars": 1000})

        assert "truncated" in result.lower()
        assert len(result) < 5000   # well under the 200KB original

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, tool):
        with patch("app.core.tools._web_fetch._ssrf_check", return_value=None), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"url": "https://slow.example.com/"})

        assert "Error" in result
        assert "timed out" in result.lower() or "timeout" in result.lower()

    @pytest.mark.asyncio
    async def test_redirect_ssrf_check_on_new_host(self, tool):
        """A redirect to a private IP must be blocked even if the original host was public."""
        redirect_resp = _httpx_redirect(301, "http://169.254.169.254/metadata")

        call_count = 0

        def _ssrf_side_effect(url):
            nonlocal call_count
            call_count += 1
            if "169.254" in url:
                return "Blocked: 169.254.169.254 is in private range"
            return None

        with patch("app.core.tools._web_fetch._ssrf_check", side_effect=_ssrf_side_effect), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=redirect_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"url": "https://public.example.com/"})

        # Should have been called at least twice (original + redirect)
        assert call_count >= 2
        assert "Error" in result or "Blocked" in result

    @pytest.mark.asyncio
    async def test_missing_url(self, tool):
        result = await tool.execute(_ctx(), {})
        assert "Error" in result
        assert "url" in result.lower()


# ---------------------------------------------------------------------------
# web_search — integration tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestWebSearch:
    @pytest.fixture
    def tool(self):
        t = REGISTRY.get("web_search")
        # web_search may not be registered without an API key — force it for tests
        if t is None:
            from app.core.tools._web_search import WebSearchTool
            t = WebSearchTool()
        return t

    @pytest.mark.asyncio
    async def test_no_api_key_returns_error(self, tool):
        with patch.dict("os.environ", {"WEB_SEARCH_API_KEY": ""}):
            result = await tool.execute(_ctx(), {"query": "python asyncio"})
        assert "Error" in result
        assert "WEB_SEARCH_API_KEY" in result

    @pytest.mark.asyncio
    async def test_brave_search_ok(self, tool):
        brave_payload = {
            "web": {
                "results": [
                    {"title": "AsyncIO Guide", "url": "https://docs.python.org/asyncio", "description": "Official asyncio docs"},
                    {"title": "Real Python", "url": "https://realpython.com/asyncio", "description": "Tutorial"},
                ]
            }
        }
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = brave_payload
        mock_resp.raise_for_status = MagicMock()

        with patch.dict("os.environ", {"WEB_SEARCH_API_KEY": "test-key", "WEB_SEARCH_PROVIDER": "brave"}), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"query": "python asyncio", "max_results": 2})

        assert "AsyncIO Guide" in result
        assert "docs.python.org" in result
        assert "Official asyncio docs" in result

    @pytest.mark.asyncio
    async def test_tavily_search_ok(self, tool):
        tavily_payload = {
            "results": [
                {"title": "Tavily Result", "url": "https://tavily.com/result", "content": "Some content here"},
            ]
        }
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200
        mock_resp.json.return_value = tavily_payload
        mock_resp.raise_for_status = MagicMock()

        with patch.dict("os.environ", {"WEB_SEARCH_API_KEY": "test-key", "WEB_SEARCH_PROVIDER": "tavily"}), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"query": "tavily test"})

        assert "Tavily Result" in result
        assert "tavily.com" in result

    @pytest.mark.asyncio
    async def test_brave_no_results(self, tool):
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.json.return_value = {"web": {"results": []}}
        mock_resp.raise_for_status = MagicMock()

        with patch.dict("os.environ", {"WEB_SEARCH_API_KEY": "test-key", "WEB_SEARCH_PROVIDER": "brave"}), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"query": "xyzzy12345"})

        assert "No results" in result

    @pytest.mark.asyncio
    async def test_api_error_returns_error_message(self, tool):
        http_error = httpx.HTTPStatusError("Unauthorized", request=MagicMock(), response=MagicMock(status_code=401))

        with patch.dict("os.environ", {"WEB_SEARCH_API_KEY": "bad-key", "WEB_SEARCH_PROVIDER": "brave"}), \
             patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock(side_effect=http_error)
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client_cls.return_value = mock_client

            result = await tool.execute(_ctx(), {"query": "test"})

        assert "Error" in result
        assert "401" in result

    @pytest.mark.asyncio
    async def test_missing_query(self, tool):
        with patch.dict("os.environ", {"WEB_SEARCH_API_KEY": "test-key"}):
            result = await tool.execute(_ctx(), {})
        assert "Error" in result
        assert "query" in result.lower()


# ---------------------------------------------------------------------------
# preview_fetch — integration tests (mocked workspaced_client)
# ---------------------------------------------------------------------------

class TestPreviewFetch:
    @pytest.fixture
    def tool(self):
        t = REGISTRY.get("preview_fetch")
        assert t is not None, "preview_fetch should always be registered"
        return t

    @pytest.mark.asyncio
    async def test_ok_response(self, tool):
        body = "Hello from dev server!"
        body_b64 = base64.b64encode(body.encode()).decode()
        proxy_result = {"status_code": 200, "content_type": "text/plain", "body_b64": body_b64}

        with patch("app.core.workspaced_client.proxy_fetch", new_callable=AsyncMock, return_value=proxy_result):
            result = await tool.execute(_ctx(), {"port": 3000, "path": "/"})

        assert "200" in result
        assert "Hello from dev server!" in result

    @pytest.mark.asyncio
    async def test_json_response(self, tool):
        body = '{"status": "ok", "version": "1.0"}'
        body_b64 = base64.b64encode(body.encode()).decode()
        proxy_result = {"status_code": 200, "content_type": "application/json", "body_b64": body_b64}

        with patch("app.core.workspaced_client.proxy_fetch", new_callable=AsyncMock, return_value=proxy_result):
            result = await tool.execute(_ctx(), {"port": 8080, "path": "/api/status"})

        assert "status" in result
        assert "ok" in result

    @pytest.mark.asyncio
    async def test_error_from_workspaced(self, tool):
        proxy_result = {"error": "connection refused on port 3000"}

        with patch("app.core.workspaced_client.proxy_fetch", new_callable=AsyncMock, return_value=proxy_result):
            result = await tool.execute(_ctx(), {"port": 3000})

        assert "Error" in result
        assert "connection refused" in result

    @pytest.mark.asyncio
    async def test_workspaced_exception(self, tool):
        with patch("app.core.workspaced_client.proxy_fetch", new_callable=AsyncMock,
                   side_effect=Exception("workspaced unreachable")):
            result = await tool.execute(_ctx(), {"port": 3000})

        assert "Error" in result
        assert "3000" in result

    @pytest.mark.asyncio
    async def test_body_truncated(self, tool):
        body = "X" * 50_000
        body_b64 = base64.b64encode(body.encode()).decode()
        proxy_result = {"status_code": 200, "content_type": "text/html", "body_b64": body_b64}

        with patch("app.core.workspaced_client.proxy_fetch", new_callable=AsyncMock, return_value=proxy_result):
            result = await tool.execute(_ctx(), {"port": 5000})

        assert "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_invalid_port(self, tool):
        result = await tool.execute(_ctx(), {"port": 0})
        assert "Error" in result
        assert "port" in result.lower()

    @pytest.mark.asyncio
    async def test_empty_body(self, tool):
        proxy_result = {"status_code": 204, "content_type": "", "body_b64": ""}

        with patch("app.core.workspaced_client.proxy_fetch", new_callable=AsyncMock, return_value=proxy_result):
            result = await tool.execute(_ctx(), {"port": 3000, "path": "/health"})

        assert "204" in result
        assert "empty body" in result.lower()

    @pytest.mark.asyncio
    async def test_query_string_passed(self, tool):
        body_b64 = base64.b64encode(b"ok").decode()
        proxy_result = {"status_code": 200, "content_type": "text/plain", "body_b64": body_b64}

        with patch("app.core.workspaced_client.proxy_fetch", new_callable=AsyncMock, return_value=proxy_result) as mock_pf:
            await tool.execute(_ctx(), {"port": 3000, "path": "/search", "query": "q=hello"})

        mock_pf.assert_called_once()
        call_kwargs = mock_pf.call_args
        assert "q=hello" in str(call_kwargs)
