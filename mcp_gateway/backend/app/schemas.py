from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MAX_ENDPOINT_BYTES = 2048


class McpAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["bearer", "oauth", "none"] = "none"


class McpCustomConfig(BaseModel):
    """Non-secret configuration accepted by the Gateway discovery boundary."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, max_length=MAX_ENDPOINT_BYTES)
    auth: McpAuthConfig = Field(default_factory=McpAuthConfig)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_ENDPOINT_BYTES:
            raise ValueError("custom_config.url is too long")
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
            raise ValueError("custom_config.url contains control characters")
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("custom_config.url must be an absolute HTTP(S) URL")
        if parts.username is not None or parts.password is not None:
            raise ValueError("custom_config.url must not contain userinfo")
        if parts.query:
            raise ValueError("custom_config.url must not contain a query string")
        if parts.fragment:
            raise ValueError("custom_config.url must not contain a fragment")
        return value


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    installed_item_id: UUID
    market_item_id: UUID | None = None
    market_slug: str | None = Field(default=None, max_length=100)
    custom_config: McpCustomConfig | None = None


McpErrorCode = Literal[
    "mcp_auth_failed",
    "mcp_unreachable",
    "mcp_protocol_error",
    "mcp_bad_response",
    "mcp_http_error",
    "mcp_secret_unavailable",
    # The server answered the handshake with an OAuth challenge: it wants a
    # login, not a better token (internal design notes Stage 0).
    "mcp_oauth_required",
]


class McpError(BaseModel):
    code: McpErrorCode
    message: str
    retryable: bool
    retry_after_seconds: int | None = None
    #: Who is asking to be logged into, for ``mcp_oauth_required`` only.  A
    #: short display label derived from provider metadata — never a URL.
    provider: str | None = Field(default=None, max_length=80)


class DiscoveryResponse(BaseModel):
    installed_item_id: UUID
    status: Literal["verified", "error"]
    tools: list["ToolInfo"] = Field(default_factory=list)
    fingerprint: str | None = None
    checked_at: datetime
    expires_at: datetime | None = None
    ttl_seconds: int = 60
    error: McpError | None = None
    correlation_id: str


class ExecuteRequest(BaseModel):
    installed_item_id: UUID
    market_item_id: UUID | None = None
    market_slug: str | None = Field(default=None, max_length=100)
    tool: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9_.:-]+$")
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Custom configurations are deliberately request-scoped. Gateway never
    # persists them (this also keeps user secrets out of its cache).
    custom_config: dict[str, Any] | None = None
    isolation: Literal["stateless", "tenant_aware"] | None = None
    # Set by Hub when it reserved MCP call credits for this call
    # (internal design notes Stage 7). Gateway hands it back to Hub's
    # verify endpoint so a direct call bypassing Hub's reservation step can be
    # told apart from a legitimate one.
    call_id: UUID | None = None

    @field_validator("custom_config")
    @classmethod
    def reject_secret_bearing_config(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None

        forbidden = {
            "token",
            "api_key",
            "secret",
            "secret_id",
            "headers",
            "authorization",
            "runtime_auth",
            "env",
        }

        def walk(candidate: Any) -> bool:
            if isinstance(candidate, dict):
                if any(str(key).lower() in forbidden for key in candidate):
                    return True
                return any(walk(child) for child in candidate.values())
            if isinstance(candidate, list):
                return any(walk(child) for child in candidate)
            return False

        if walk(value):
            raise ValueError("custom_config contains a forbidden credential field")
        return value



class OAuthMetadataRequest(BaseModel):
    """internal design notes Stage 2 — ``POST /v1/oauth/metadata``."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["authorization_server", "protected_resource"]
    issuer: str | None = Field(default=None, max_length=MAX_ENDPOINT_BYTES)
    resource: str | None = Field(default=None, max_length=MAX_ENDPOINT_BYTES)

    @model_validator(mode="after")
    def _one_target_for_the_kind(self) -> "OAuthMetadataRequest":
        target = self.issuer if self.kind == "authorization_server" else self.resource
        if not target:
            raise ValueError("issuer is required for authorization_server, resource for protected_resource")
        return self


class OAuthRegisterRequest(BaseModel):
    """internal design notes Stage 2 — ``POST /v1/oauth/register`` (RFC 7591 DCR)."""

    model_config = ConfigDict(extra="forbid")

    issuer: str = Field(min_length=1, max_length=MAX_ENDPOINT_BYTES)
    registration_endpoint: str = Field(min_length=1, max_length=MAX_ENDPOINT_BYTES)
    redirect_uris: list[str] = Field(min_length=1, max_length=4)
    client_name: str = Field(default="Legion Hub", max_length=128)
    token_endpoint_auth_method: str = Field(default="none", max_length=32)
    grant_types: list[str] = Field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = Field(default_factory=lambda: ["code"])

    @field_validator("redirect_uris")
    @classmethod
    def _validate_redirect_uris(cls, value: list[str]) -> list[str]:
        for uri in value:
            if not uri or len(uri) > MAX_ENDPOINT_BYTES:
                raise ValueError("redirect_uri is invalid")
        return value


class OAuthTokenRequest(BaseModel):
    """internal design notes Stage 2 — ``POST /v1/oauth/token``.

    Covers both grants the plan needs: ``authorization_code`` (Stage 3's
    callback) and ``refresh_token`` (Stage 4's just-in-time refresh).
    """

    model_config = ConfigDict(extra="forbid")

    token_endpoint: str = Field(min_length=1, max_length=MAX_ENDPOINT_BYTES)
    grant_type: Literal["authorization_code", "refresh_token"]
    client_id: str = Field(min_length=1, max_length=512)
    code: str | None = Field(default=None, max_length=2048)
    redirect_uri: str | None = Field(default=None, max_length=MAX_ENDPOINT_BYTES)
    code_verifier: str | None = Field(default=None, max_length=256)
    refresh_token: str | None = Field(default=None, max_length=4096)
    resource: str | None = Field(default=None, max_length=MAX_ENDPOINT_BYTES)
    # internal design notes Stage 8 — set only for a manually registered
    # confidential client; a DCR public client never has one.
    client_secret: str | None = Field(default=None, max_length=4096)

    @model_validator(mode="after")
    def _grant_carries_its_fields(self) -> "OAuthTokenRequest":
        if self.grant_type == "authorization_code":
            if not self.code or not self.redirect_uri or not self.code_verifier:
                raise ValueError(
                    "authorization_code grant requires code, redirect_uri and code_verifier"
                )
        elif not self.refresh_token:
            raise ValueError("refresh_token grant requires refresh_token")
        return self


class ToolInfo(BaseModel):
    name: str
    description: str = ""
    inputSchema: dict[str, Any] = Field(default_factory=dict)


class ToolsResponse(BaseModel):
    installed_item_id: UUID
    tools: list[ToolInfo]
