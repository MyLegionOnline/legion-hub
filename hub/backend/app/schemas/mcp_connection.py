"""Safe status DTOs for HTTP MCP connection verification."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class McpToolSnapshotOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    inputSchema: dict[str, Any] = Field(default_factory=dict)


class McpSafeErrorOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    remedy: str
    retryable: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None


class McpConnectionStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    installed_item_id: UUID
    status: Literal["unknown", "checking", "verified", "error", "expired"]
    available: bool = False
    ready_for_room: bool = False
    tools: list[McpToolSnapshotOut] = Field(default_factory=list)
    tool_count: int = 0
    # ``tools_count`` is retained as a descriptive alias for clients that use
    # the wording from the room status contract.
    tools_count: int = 0
    tool_fingerprint: str | None = None
    checked_at: datetime | None = None
    expires_at: datetime | None = None
    ttl_seconds: int = 0
    last_error_code: str | None = None
    last_error: McpSafeErrorOut | None = None
    latency_ms: int | None = None
    correlation_id: str | None = None


class McpRoomBindingOut(BaseModel):
    room_id: UUID
    available: bool
    status: Literal["available", "unavailable"]


class McpAgentScopeOut(BaseModel):
    """Which agents of a room get one room MCP server's tools.

    internal design notes §3.5. ``all`` → every agent
    of the room (``agent_ids`` is empty); ``selected`` → only ``agent_ids``
    (empty means nobody).
    """

    mode: Literal["all", "selected"] = "all"
    agent_ids: list[UUID] = Field(default_factory=list)


class McpAgentScopeIn(BaseModel):
    mode: Literal["all", "selected"]
    agent_ids: list[UUID] = Field(default_factory=list, max_length=200)


__all__ = [
    "McpAgentScopeIn",
    "McpAgentScopeOut",
    "McpConnectionStatusOut",
    "McpRoomBindingOut",
    "McpSafeErrorOut",
    "McpToolSnapshotOut",
]
