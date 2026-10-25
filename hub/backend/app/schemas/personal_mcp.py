"""Pydantic schemas for the personal-mcp.v1 Hub API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class SecretBindingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    env_name: str = Field(pattern=r"^[A-Z_][A-Z0-9_]*$", max_length=128)
    secret_id: UUID


class CreatePersonalMcpRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["personal-mcp.v1"]
    preview_id: UUID
    candidate_id: UUID
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    # Node/npm managed runners need headroom for the package loader and the
    # first MCP parse. 256 MB can still OOM during a cold depwire start, so it
    # is not a viable managed_stdio runtime limit.
    memory_reservation_mb: int = Field(ge=512)
    storage_reservation_mb: int = Field(ge=512)
    secret_bindings: list[SecretBindingInput] = Field(default_factory=list, max_length=64)


class SecretBindingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bindings: list[SecretBindingInput] = Field(default_factory=list, max_length=64)


class RoomBindingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_id: UUID


class ResourceUsageOut(BaseModel):
    memory_reservation_mb: int = 0
    memory_usage_mb: int = 0
    storage_reservation_mb: int = 0
    storage_used_mb: int = 0


class ToolsSnapshotOut(BaseModel):
    count: int = 0
    names: list[str] = Field(default_factory=list)
    fingerprint: str = ""


class RoomBindingOut(BaseModel):
    room_id: UUID
    available: bool
    status: Literal["available", "stopped", "unavailable"]


class PersonalSecretBindingOut(BaseModel):
    env_name: str
    secret_id: UUID
    label: str
    state: Literal["active", "disabled", "missing"]


class PersonalMcpOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    contract_version: Literal["personal-mcp.v1"] = "personal-mcp.v1"
    name: str
    mode: Literal["managed_stdio"] = "managed_stdio"
    runtime_profile: Literal["node_npx_v1"] = "node_npx_v1"
    transport: Literal["stdio"] = "stdio"
    desired_status: Literal["stopped", "running", "deleted"]
    actual_status: str
    ready_for_room: bool
    revision: int
    launch_profile: dict[str, Any]
    resolved_artifact: dict[str, Any] | None = None
    resources: ResourceUsageOut
    tools: ToolsSnapshotOut
    room_bindings: list[RoomBindingOut] = Field(default_factory=list)
    secret_bindings: list[PersonalSecretBindingOut] = Field(default_factory=list)
    last_error: dict[str, Any] | None = None
    correlation_id: str
    created_at: datetime
    updated_at: datetime


class PersonalMcpUsageOut(BaseModel):
    limits: dict[str, Any]
    installed_count: int
    running_count: int
    memory_reserved_mb: int
    memory_used_mb: int
    storage_allocated_mb: int
    storage_used_mb: int


class PersonalMcpEventOut(BaseModel):
    event_id: UUID
    type: str
    step: str
    occurred_at: datetime
    correlation_id: str
    message: str
    percent: int | None = None
    safe_details: dict[str, Any] | None = None


__all__ = [
    "CreatePersonalMcpRequest",
    "PersonalMcpEventOut",
    "PersonalMcpOut",
    "PersonalMcpUsageOut",
    "PersonalSecretBindingOut",
    "ResourceUsageOut",
    "RoomBindingOut",
    "RoomBindingRequest",
    "SecretBindingInput",
    "SecretBindingsRequest",
    "ToolsSnapshotOut",
]
