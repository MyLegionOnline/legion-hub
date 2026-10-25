import re
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ResolvedArtifact(BaseModel):
    version: str = Field(min_length=5, max_length=128)
    integrity: str = Field(min_length=20, max_length=512)
    resolved_at: str | None = None


class ToolSnapshot(BaseModel):
    count: int = Field(default=0, ge=0, le=512)
    names: list[str] = Field(default_factory=list, max_length=512)
    fingerprint: str = Field(default="", max_length=256)


class StartRequest(BaseModel):
    """Existing user-image path. Kept compatible with the legacy API."""

    mode: Literal["personal_image"] = "personal_image"
    image: str = Field(min_length=1, max_length=512)
    port: int = Field(default=8000, ge=1, le=65535)
    mem_limit_mb: int = Field(default=1024, ge=128, le=8192)
    cpu_limit_milli: int = Field(default=500, ge=100, le=4000)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


class ManagedStdioStartRequest(BaseModel):
    """Normalized profile sent by Hub to the internal mcp_host service."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["managed_stdio"]
    runtime_profile: Literal["node_npx_v1"]
    transport: Literal["stdio"] = "stdio"
    package: str = Field(min_length=1, max_length=214)
    command_args: list[str] = Field(default_factory=list, max_length=64)
    env_names: list[str] = Field(default_factory=list, max_length=64)
    shell: Literal[False] = False
    resolved_artifact: ResolvedArtifact | None = None
    mem_limit_mb: int = Field(default=1024, ge=512, le=8192)
    cpu_limit_milli: int = Field(default=500, ge=100, le=4000)
    storage_limit_mb: int = Field(default=1024, ge=512, le=102400)
    correlation_id: str | None = Field(default=None, max_length=128)
    # Opaque Hub capability.  It is consumed in host memory and is never
    # copied into Docker configuration or the runner launch profile.
    secret_lease_token: str | None = Field(default=None, min_length=32, max_length=256)

    @field_validator("env_names")
    @classmethod
    def validate_env_names(cls, names: list[str]) -> list[str]:
        pattern = re.compile(r"^[A-Z_][A-Z0-9_]{0,127}$")
        if len(set(names)) != len(names) or any(
            not pattern.fullmatch(name) or name in {"PATH", "NODE_OPTIONS"}
            for name in names
        ):
            raise ValueError("only explicitly bound environment names are allowed")
        return names


# A non-discriminated union is intentional: the legacy request predates the
# `mode` field and must continue to validate when it is omitted.
StartBody = Union[StartRequest, ManagedStdioStartRequest]


class DeleteRequest(BaseModel):
    confirm: bool = False


class StatusResponse(BaseModel):
    installed_item_id: str
    user_id: str
    status: Literal[
        "preparing",
        "starting",
        "running",
        "verifying",
        "verified",
        "stopping",
        "stopped",
        "updating",
        "error",
        "deleting",
        "deleted",
    ]
    mode: Literal["managed_stdio", "personal_image"] | None = None
    container_id: str | None = None
    endpoint: str | None = None
    # `port` is populated only for the legacy public-image path. Managed MCPs
    # use an internal endpoint and never publish a host port.
    port: int | None = None
    internal_port: int | None = None
    uptime_sec: float | None = None
    memory_usage_mb: float | None = None
    memory_limit_mb: int | None = None
    storage_used_mb: float | None = None
    storage_limit_mb: int | None = None
    volume_name: str | None = None
    ready: bool | None = None
    healthy: bool | None = None
    tool_count: int | None = None
    tools: ToolSnapshot | None = None
    resolved_artifact: ResolvedArtifact | None = None
    correlation_id: str | None = None
    error_code: str | None = None
    message: str | None = None
    log_tail: list[str] = Field(default_factory=list)
