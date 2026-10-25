"""preview_fetch tool — fetch from a dev server running inside the workspace container.

Routes through the workspaced proxy endpoint, so this is not an external HTTP
request and carries no SSRF risk. Use it only to diagnose an existing legacy
server; new user-visible applications must use ``workspace_service``.
"""
from __future__ import annotations

import base64
from typing import Any

from app.core.tools.registry import REGISTRY, Tool, ToolContext, _truncate

_MAX_BODY_CHARS = 32_000


class PreviewFetchTool(Tool):
    name = "preview_fetch"
    description = (
        "Fetch a URL from an existing legacy dev server inside the workspace container. "
        "For a new user-visible application, use workspace_service declare/start/request instead. "
        "Example: preview_fetch(port=3000, path='/api/health')"
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "port": {
                "type": "integer",
                "description": "Port the dev server is listening on (inside the container)",
            },
            "path": {
                "type": "string",
                "description": "URL path to request (default: '/')",
                "default": "/",
            },
            "query": {
                "type": "string",
                "description": "Query string without the leading '?' (e.g. 'foo=bar&baz=1')",
            },
        },
        "required": ["port"],
    }
    roles: frozenset[str] = frozenset({"agent"})
    requires_container: bool = True
    mutating: bool = False

    async def execute(self, ctx: ToolContext, tool_input: dict[str, Any]) -> str:
        from app.core import workspaced_client as wsd

        port = int(tool_input.get("port") or 0)
        if not (1 <= port <= 65535):
            return "Error: invalid port number"

        path = str(tool_input.get("path") or "/").strip() or "/"
        query = str(tool_input.get("query") or "").strip()
        rid = str(ctx.room_id)

        try:
            result = await wsd.proxy_fetch(rid, port, path, query)
        except Exception as exc:
            return f"Error: could not reach port {port} — {exc}"

        if result.get("error"):
            return f"Error: {result['error']}"

        status = result.get("status_code", "?")
        ct = result.get("content_type", "")
        body_b64 = result.get("body_b64", "")

        if body_b64:
            try:
                body = base64.b64decode(body_b64).decode("utf-8", errors="replace")
            except Exception:
                body = "(binary or undecodable response)"
        else:
            body = "(empty body)"

        if len(body) > _MAX_BODY_CHARS:
            body = body[:_MAX_BODY_CHARS] + f"\n[...body truncated at {_MAX_BODY_CHARS} chars]"

        header = f"Status: {status}\nContent-Type: {ct}\n"
        return _truncate(header + "\n" + body)


REGISTRY.register(PreviewFetchTool())
