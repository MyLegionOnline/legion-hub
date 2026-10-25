from __future__ import annotations

import json
import hashlib
import re
from datetime import date
from typing import Any

import httpx
from fastapi import HTTPException

from app import config


_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,255}$")
_MAX_TOOLS = 128
_MAX_DESCRIPTION_BYTES = 4096
_MAX_SCHEMA_BYTES = 32 * 1024
_MAX_SNAPSHOT_BYTES = 256 * 1024
_CLIENT_PROTOCOL_VERSION = "2025-03-26"
_STREAMABLE_PROTOCOL_START = date(2025, 3, 26)
_STATELESS_PROTOCOL_CUTOFF = date(2026, 7, 28)
_SAFE_MESSAGES = {
    "mcp_auth_failed": "MCP server authentication failed",
    "mcp_unreachable": "MCP server is unavailable",
    "mcp_protocol_error": "MCP server protocol validation failed",
    "mcp_bad_response": "MCP server returned an invalid response",
    "mcp_http_error": "MCP server returned an HTTP error",
    "mcp_secret_unavailable": "MCP authorization is not configured",
    "mcp_bad_request": "MCP request was invalid",
    "mcp_rpc_error": "MCP tool call failed",
}


def _mcp_headers(token: str | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def _protocol_headers(version: str | None) -> dict[str, str]:
    """Return the Streamable HTTP protocol header for session requests."""

    if not isinstance(version, str) or not version:
        return {}
    try:
        if date.fromisoformat(version) < _STREAMABLE_PROTOCOL_START:
            return {}
    except ValueError:
        return {}
    return {"MCP-Protocol-Version": version}


def _uses_initialized_notification(version: str) -> bool:
    """Keep the 2025 session lifecycle while supporting the 2026 stateless one."""

    try:
        return date.fromisoformat(version) < _STATELESS_PROTOCOL_CUTOFF
    except ValueError:
        # Unknown versions should retain the established session lifecycle;
        # this is the safer compatibility choice for older upstreams.
        return True


#: Structured ``detail`` so Hub's credit ledger (internal design notes
#: Stage 4) can tell "our side never reached the MCP, refund" apart from
#: "the MCP itself returned an error, settle" without parsing prose. Every
#: raise in this module goes through here — no bare string details.
def _mcp_error(code: str, message: str, *, reached_upstream: bool, status_code: int = 502) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "reached_upstream": reached_upstream,
            "message": _SAFE_MESSAGES.get(code, message),
        },
    )


def _auth_error(response: httpx.Response, *, reached_upstream: bool) -> HTTPException:
    """A 401/403 that carries its ``WWW-Authenticate`` challenge out of band.

    The challenge is upstream-controlled data, so it is attached as an
    attribute rather than added to ``detail``: discovery reads it to tell an
    OAuth login apart from a wrong token (internal design notes Stage 0), and it
    never reaches a client through the response body.
    """

    error = _mcp_error(
        "mcp_auth_failed", "MCP server authentication failed", reached_upstream=reached_upstream
    )
    error.www_authenticate = response.headers.get("www-authenticate")
    return error


def _decode_response(response: httpx.Response) -> dict[str, Any] | list[Any]:
    if len(response.content) > config.MAX_RESPONSE_BYTES:
        raise _mcp_error(
            "mcp_bad_response", "MCP response exceeds the configured size limit", reached_upstream=True
        )
    try:
        return response.json()
    except ValueError:
        # Streamable HTTP servers may return one JSON-RPC object in an SSE
        # data frame. Keep the transport adapter tolerant of both forms.
        for line in response.text.splitlines():
            if line.startswith("data:"):
                try:
                    return json.loads(line[5:].strip())
                except ValueError:
                    continue
    raise _mcp_error("mcp_bad_response", "MCP returned an invalid JSON response", reached_upstream=True)


def _raise_for_rpc_error(result: dict[str, Any] | list[Any], *, phase: str = "call") -> None:
    """Turn a JSON-RPC ``error`` object into the matching structured code.

    ``-32601``/``-32602`` (unknown method / invalid params) mean the request
    we built was malformed — our fault, refunded. Any other JSON-RPC error is
    the upstream tool genuinely failing after accepting the call — settled.
    """
    if not isinstance(result, dict) or not result.get("error"):
        return
    error = result["error"]
    rpc_code = error.get("code") if isinstance(error, dict) else None
    if phase == "handshake":
        raise _mcp_error("mcp_protocol_error", "MCP handshake failed", reached_upstream=True)
    if rpc_code in (-32601, -32602):
        raise _mcp_error("mcp_bad_request", "MCP request was invalid", reached_upstream=True)
    raise _mcp_error("mcp_rpc_error", "MCP tool call failed", reached_upstream=True)


def _json_depth(value: Any, *, current: int = 0) -> int:
    if isinstance(value, dict):
        return max([current, *(_json_depth(child, current=current + 1) for child in value.values())])
    if isinstance(value, list):
        return max([current, *(_json_depth(child, current=current + 1) for child in value)])
    return current


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    """Validate external tool metadata before it can cross the Gateway boundary."""

    if not isinstance(tools, list):
        raise _mcp_error("mcp_bad_response", "MCP tools payload is invalid", reached_upstream=True)
    if len(tools) > _MAX_TOOLS:
        raise _mcp_error("mcp_bad_response", "MCP tools payload is too large", reached_upstream=True)
    if not tools:
        raise _mcp_error("mcp_protocol_error", "MCP server returned no tools", reached_upstream=True)

    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True)
        name = tool["name"]
        if len(name.encode("utf-8")) > 255 or not _TOOL_NAME.fullmatch(name) or name in names:
            raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True)
        names.add(name)

        description = tool.get("description", "")
        if not isinstance(description, str):
            raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True)
        description = "".join(
            char
            for char in description
            if (ord(char) >= 0x20 and ord(char) != 0x7F) or char in {"\n", "\r", "\t"}
        )
        if len(description.encode("utf-8")) > _MAX_DESCRIPTION_BYTES:
            raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True)

        input_schema = tool.get("inputSchema", tool.get("input_schema", {}))
        if not isinstance(input_schema, dict) or _json_depth(input_schema) > 20:
            raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True)
        try:
            schema_json = json.dumps(input_schema, ensure_ascii=False, separators=(",", ":"))
            json.dumps(input_schema, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True) from exc
        if len(schema_json.encode("utf-8")) > _MAX_SCHEMA_BYTES:
            raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True)
        normalized.append({"name": name, "description": description, "inputSchema": input_schema})

    normalized.sort(key=lambda tool: tool["name"])
    try:
        snapshot_json = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True) from exc
    if len(snapshot_json.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise _mcp_error("mcp_bad_response", "MCP tool metadata is invalid", reached_upstream=True)
    return normalized


def tools_fingerprint(tools: list[dict[str, Any]]) -> str:
    canonical = json.dumps(tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


class McpHttpClient:
    def __init__(
        self,
        endpoint: str,
        *,
        token: str | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout_sec = timeout_sec or config.REQUEST_TIMEOUT_SEC

    async def request(self, method: str, params: dict[str, Any] | None = None, *, meta: dict[str, str] | None = None) -> dict[str, Any] | list[Any]:
        request_params = dict(params or {})
        if meta:
            request_params["_meta"] = dict(meta)
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": request_params}
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_sec,
                follow_redirects=False,
            ) as client:
                response = await client.post(self.endpoint, json=payload, headers=_mcp_headers(self.token))
        except httpx.RequestError as exc:
            raise _mcp_error("mcp_unreachable", "MCP server is unavailable", reached_upstream=False) from exc
        if response.status_code in {401, 403}:
            raise _auth_error(response, reached_upstream=False)
        if 300 <= response.status_code < 400:
            raise _mcp_error("mcp_http_error", "MCP server returned an HTTP error", reached_upstream=True)
        if response.status_code >= 400:
            raise _mcp_error(
                "mcp_http_error", f"MCP server returned HTTP {response.status_code}", reached_upstream=True
            )
        result = _decode_response(response)
        _raise_for_rpc_error(result)
        return result

    async def _session_request(self, method: str, params: dict[str, Any] | None = None, *, meta: dict[str, str] | None = None) -> dict[str, Any] | list[Any]:
        """Initialize a streamable-HTTP MCP session, then issue one request."""
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "legion-mcp-gateway", "version": "0.1.0"},
            },
        }
        request_params = dict(params or {})
        if meta:
            request_params["_meta"] = dict(meta)
        payload = {"jsonrpc": "2.0", "id": 2, "method": method, "params": request_params}
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_sec,
                follow_redirects=False,
            ) as client:
                init_response = await client.post(
                    self.endpoint,
                    json=init_payload,
                    headers=_mcp_headers(
                        self.token,
                        _protocol_headers(_CLIENT_PROTOCOL_VERSION),
                    ),
                )
                if init_response.status_code >= 400:
                    # The tool-call frame was never sent — this failed during
                    # session setup, so it never reached the upstream tool.
                    if init_response.status_code in {401, 403}:
                        raise _auth_error(init_response, reached_upstream=False)
                    raise _mcp_error(
                        "mcp_http_error",
                        "MCP server returned an HTTP error",
                        reached_upstream=False,
                    )
                if 300 <= init_response.status_code < 400:
                    raise _mcp_error(
                        "mcp_http_error",
                        "MCP server returned an HTTP error",
                        reached_upstream=True,
                    )
                init_result = _decode_response(init_response)
                # Keep direct tools/call callers' existing RPC error
                # classification; discovery maps these non-contract codes to
                # the safe mcp_protocol_error response.
                _raise_for_rpc_error(init_result, phase="initialize")
                if not isinstance(init_result, dict) or not isinstance(init_result.get("result"), dict):
                    raise _mcp_error(
                        "mcp_protocol_error", "MCP initialize response is invalid", reached_upstream=True
                    )
                negotiated_version = init_result["result"].get(
                    "protocolVersion", _CLIENT_PROTOCOL_VERSION
                )
                if not isinstance(negotiated_version, str) or not negotiated_version:
                    raise _mcp_error(
                        "mcp_protocol_error", "MCP initialize response is invalid", reached_upstream=True
                    )
                session_id = init_response.headers.get("mcp-session-id")
                session_headers = _protocol_headers(negotiated_version)
                if session_id:
                    session_headers["mcp-session-id"] = session_id
                headers = _mcp_headers(self.token, session_headers)
                if _uses_initialized_notification(negotiated_version):
                    initialized_response = await client.post(
                        self.endpoint,
                        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                        headers=headers,
                    )
                    if initialized_response.status_code in {401, 403}:
                        raise _auth_error(initialized_response, reached_upstream=True)
                    if initialized_response.status_code >= 400:
                        raise _mcp_error(
                            "mcp_http_error",
                            "MCP server returned an HTTP error",
                            reached_upstream=True,
                        )
                response = await client.post(self.endpoint, json=payload, headers=headers)
        except httpx.RequestError as exc:
            raise _mcp_error("mcp_unreachable", "MCP server is unavailable", reached_upstream=False) from exc
        if response.status_code in {401, 403}:
            raise _auth_error(response, reached_upstream=True)
        if 300 <= response.status_code < 400:
            raise _mcp_error("mcp_http_error", "MCP server returned an HTTP error", reached_upstream=True)
        if response.status_code >= 400:
            raise _mcp_error(
                "mcp_http_error", "MCP server returned an HTTP error", reached_upstream=True
            )
        result = _decode_response(response)
        _raise_for_rpc_error(result, phase="handshake" if method == "tools/list" else "call")
        return result

    async def tools(self) -> list[dict[str, Any]]:
        result = await self._session_request("tools/list")
        if not isinstance(result, dict) or not isinstance(result.get("result"), dict):
            raise _mcp_error("mcp_bad_response", "MCP tools response is invalid", reached_upstream=True)
        raw_tools = result["result"].get("tools")
        return normalize_tools(raw_tools)

    async def call(self, tool: str, arguments: dict[str, Any], *, meta: dict[str, str] | None = None) -> dict[str, Any] | list[Any]:
        params: dict[str, Any] = {"name": tool, "arguments": arguments}
        return await self._session_request("tools/call", params, meta=meta)
