"""Hub client for the operator-owned personal MCP host.

The Hub owns authorization and durable state; this module only transports a
normalized launch profile to ``mcp_host`` and returns bounded status data.  It
never logs or serializes secret values outside the one-time lease capability.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import httpx

from app.core.secret_config import configured_secret

MCP_HOST_URL = os.getenv("MCP_HOST_URL", "http://mcp-host:8320").rstrip("/")
MCP_HOST_TOKEN = configured_secret("MCP_HOST_TOKEN") or ""
MCP_HOST_TIMEOUT = float(os.getenv("MCP_HOST_TIMEOUT_SEC", "45"))


class PersonalMcpHostError(RuntimeError):
    """Safe host failure with a stable Hub-facing error code."""

    def __init__(self, code: str, *, retryable: bool = True) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class ProtocolProbe:
    count: int
    names: list[str]
    fingerprint: str


def tool_fingerprint(tools: list[dict[str, Any]]) -> str:
    """Fingerprint the bounded public tool catalog, including schemas."""

    safe_tools: list[dict[str, Any]] = []
    for raw in tools[:512]:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        if not isinstance(name, str) or not name or len(name) > 128:
            continue
        safe_tools.append(
            {
                "name": name,
                "description": str(raw.get("description") or "")[:2000],
                "inputSchema": raw.get("inputSchema") or raw.get("input_schema") or {},
            }
        )
    # The verify-time probe (mcp_host, direct) and the per-turn live fetch
    # (mcp_gateway, proxied) each list the same running server's tools
    # independently and are not guaranteed to return them in the same order.
    # Sort by name so the fingerprint reflects content, not call-path order.
    safe_tools.sort(key=lambda tool: tool["name"])
    canonical = json.dumps(safe_tools, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _protocol_probe(raw_tools: list[Any]) -> ProtocolProbe:
    tools = [tool for tool in raw_tools if isinstance(tool, dict)]
    names = [str(tool["name"]) for tool in tools if isinstance(tool.get("name"), str) and tool["name"]]
    normalized = [tool for tool in tools if isinstance(tool.get("name"), str) and tool["name"]]
    return ProtocolProbe(len(names), names[:512], tool_fingerprint(normalized))


class PersonalMcpHostClient:
    def __init__(self, *, base_url: str = MCP_HOST_URL, token: str = MCP_HOST_TOKEN, timeout: float = MCP_HOST_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _headers(self, correlation_id: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(
                    method,
                    f"{self.base_url}{path}",
                    json=json_body,
                    headers=self._headers(correlation_id),
                )
        except httpx.RequestError as exc:
            raise PersonalMcpHostError("runner_unavailable") from exc
        if response.status_code >= 400:
            raise PersonalMcpHostError(
                "runner_unavailable" if response.status_code >= 500 else "runner_request_rejected",
                retryable=response.status_code >= 500,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise PersonalMcpHostError("runner_invalid_response") from exc
        if not isinstance(body, dict):
            raise PersonalMcpHostError("runner_invalid_response")
        return body

    async def start(
        self,
        owner_id: UUID,
        installed_item_id: UUID,
        launch_spec: dict[str, Any],
        *,
        memory_mb: int,
        storage_mb: int,
        revision: int,
        correlation_id: str,
        secret_lease_token: str | None = None,
    ) -> dict[str, Any]:
        package = launch_spec.get("package")
        if not isinstance(package, str) or not package:
            raise PersonalMcpHostError("runner_request_rejected", retryable=False)
        artifact = launch_spec.get("resolved_artifact")
        if not isinstance(artifact, dict) or not artifact.get("version") or not artifact.get("integrity"):
            artifact = None
        body = {
            "mode": "managed_stdio",
            "runtime_profile": "node_npx_v1",
            "transport": "stdio",
            "package": package,
            "command_args": [value for value in launch_spec.get("command_args", []) if isinstance(value, str)],
            "env_names": [value for value in launch_spec.get("env_names", []) if isinstance(value, str)],
            "shell": False,
            "resolved_artifact": artifact,
            "mem_limit_mb": memory_mb,
            "storage_limit_mb": storage_mb,
            "correlation_id": correlation_id,
            "secret_lease_token": secret_lease_token,
        }
        return await self._request(
            "POST",
            f"/v1/personal/{owner_id}/{installed_item_id}/start",
            json_body=body,
            correlation_id=correlation_id,
        )

    async def stop(self, owner_id: UUID, installed_item_id: UUID, *, correlation_id: str | None = None) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/v1/personal/{owner_id}/{installed_item_id}/stop",
            correlation_id=correlation_id,
        )

    async def delete(self, owner_id: UUID, installed_item_id: UUID, *, correlation_id: str | None = None) -> dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/v1/personal/{owner_id}/{installed_item_id}",
            json_body={"confirm": True},
            correlation_id=correlation_id,
        )

    async def status(self, owner_id: UUID, installed_item_id: UUID, *, correlation_id: str | None = None) -> dict[str, Any]:
        return await self._request(
            "GET",
            f"/v1/personal/{owner_id}/{installed_item_id}/status",
            correlation_id=correlation_id,
        )

    async def probe(self, endpoint: str, *, correlation_id: str) -> ProtocolProbe:
        if not isinstance(endpoint, str) or not endpoint.startswith(("http://", "https://")):
            raise PersonalMcpHostError("runner_invalid_response", retryable=False)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Correlation-ID": correlation_id,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                initialize = await client.post(
                    endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": "legion-initialize",
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "legion-hub", "version": "1.0.0"},
                        },
                    },
                    headers=headers,
                )
                init_payload = _decode_json(initialize)
                if initialize.status_code >= 400 or not isinstance(init_payload.get("result"), dict):
                    raise PersonalMcpHostError("not_mcp_server", retryable=False)
                session_id = initialize.headers.get("mcp-session-id")
                if session_id:
                    headers = {**headers, "mcp-session-id": session_id}
                tools_response = await client.post(
                    endpoint,
                    json={
                        "jsonrpc": "2.0",
                        "id": "legion-tools-list",
                        "method": "tools/list",
                        "params": {},
                    },
                    headers=headers,
                )
                tools_payload = _decode_json(tools_response)
            except PersonalMcpHostError:
                raise
            except (httpx.RequestError, ValueError) as exc:
                raise PersonalMcpHostError("protocol_handshake_timeout") from exc
        if tools_response.status_code >= 400:
            raise PersonalMcpHostError("invalid_tools_response", retryable=False)
        result = tools_payload.get("result")
        raw_tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(raw_tools, list):
            raise PersonalMcpHostError("invalid_tools_response", retryable=False)
        return _protocol_probe(raw_tools)


def _decode_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
        if isinstance(body, dict):
            return body
    except ValueError:
        for line in response.text.splitlines():
            if line.startswith("data:"):
                try:
                    body = json.loads(line[5:].strip())
                except ValueError:
                    continue
                if isinstance(body, dict):
                    return body
    raise PersonalMcpHostError("invalid_jsonrpc", retryable=False)


host_client = PersonalMcpHostClient()


__all__ = ["PersonalMcpHostClient", "PersonalMcpHostError", "ProtocolProbe", "host_client", "tool_fingerprint"]
