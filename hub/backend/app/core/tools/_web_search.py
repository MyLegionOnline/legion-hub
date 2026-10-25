"""web_search tool — search the web via Brave Search or Tavily API.

Environment variables:
  WEB_SEARCH_API_KEY      — API key (required; if absent, the tool is not registered)
  WEB_SEARCH_PROVIDER     — 'brave' (default) or 'tavily'
"""
from __future__ import annotations

import os
from typing import Any

import httpx

from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_API_KEY: str = os.getenv("WEB_SEARCH_API_KEY", "")
_PROVIDER: str = os.getenv("WEB_SEARCH_PROVIDER", "brave").lower().strip()
_TIMEOUT = 15.0
_MAX_N = 10


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web and return a ranked list of results (title, URL, snippet). "
        "Use to find documentation, library versions, error explanations, Stack Overflow answers, "
        "or anything not available in the workspace. "
        "Follow up with web_fetch to read the full content of a result URL."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": f"Number of results to return (1–{_MAX_N}, default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = False
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        query = str(tool_input.get("query") or "").strip()
        if not query:
            return "Error: query is required"

        n = max(1, min(int(tool_input.get("max_results") or 5), _MAX_N))

        api_key = os.getenv("WEB_SEARCH_API_KEY", "")
        if not api_key:
            return "Error: web_search is not configured (WEB_SEARCH_API_KEY not set)"

        provider = os.getenv("WEB_SEARCH_PROVIDER", "brave").lower().strip()
        try:
            if provider == "tavily":
                return await _search_tavily(query, n, api_key)
            else:
                return await _search_brave(query, n, api_key)
        except httpx.HTTPStatusError as exc:
            return f"Error: search API returned HTTP {exc.response.status_code}"
        except httpx.TimeoutException:
            return "Error: search request timed out"
        except Exception as exc:
            return f"Error: {exc}"


async def _search_brave(query: str, n: int, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": n, "text_decorations": "false", "safesearch": "off"},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("web", {}).get("results", [])
    if not results:
        return f"No results found for: {query!r}"

    lines: list[str] = [f"Search results for: {query!r}", ""]
    for i, r in enumerate(results[:n], 1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        snippet = (r.get("description") or "").replace("\n", " ")
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    return _truncate("\n".join(lines))


async def _search_tavily(query: str, n: int, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": n,
                "search_depth": "basic",
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results:
        return f"No results found for: {query!r}"

    lines: list[str] = [f"Search results for: {query!r}", ""]
    for i, r in enumerate(results[:n], 1):
        title = r.get("title", "(no title)")
        url = r.get("url", "")
        snippet = (r.get("content") or "").replace("\n", " ")[:400]
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    return _truncate("\n".join(lines))


# Only register if an API key is configured.
if _API_KEY:
    REGISTRY.register(WebSearchTool())
