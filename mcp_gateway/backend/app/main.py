from __future__ import annotations

import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import account_keys, config, oauth_client, oauth_probe, server_terms, urlguard
from app.account_keys import Caller
from app.cache import CatalogCache
from app.market_client import MarketClient
from app.mcp_client import McpHttpClient, normalize_tools, tools_fingerprint
from app.quota import verify_call_credits
from app.schemas import (
    DiscoveryRequest,
    DiscoveryResponse,
    ExecuteRequest,
    McpCustomConfig,
    McpError,
    OAuthMetadataRequest,
    OAuthRegisterRequest,
    OAuthTokenRequest,
    ToolInfo,
    ToolsResponse,
)
from app.security import (
    caller_correlation_id,
    caller_upstream_authorization,
    gateway_caller,
    request_correlation_id,
    required_identity,
    require_gateway_admin_token,
    require_gateway_token,
    upstream_authorization,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("mcp_gateway")
cache = CatalogCache()
market = MarketClient(cache)

#: The first-party VPS transport (internal design notes §3.2). Like mcp-searxng and
#: mcp-vision it is a shared internal service whose bearer token belongs to
#: Gateway configuration, not to the public Market catalog record.
VPS_MCP_SLUG = "legion-vps-mcp"
_EGRESS_MODES = {"broker", "host_firewall", "off"}


@asynccontextmanager
async def lifespan(_: FastAPI):
    await cache.start()
    yield
    await cache.close()


app = FastAPI(title="Legion MCP Gateway", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def request_validation_error(_: Request, __: RequestValidationError) -> JSONResponse:
    """Never echo invalid request values; they may contain a submitted secret."""

    return JSONResponse(status_code=422, content={"detail": "Invalid Gateway request"})


#: The SSRF perimeter lives in ``urlguard`` so the OAuth metadata probe
#: (internal design notes Stage 0) crosses the same barrier as the MCP endpoint
#: itself.  These aliases keep the existing call sites unchanged.
_validate_endpoint = urlguard.validate_endpoint
_is_blocked_address = urlguard.is_blocked_address
_validate_public_endpoint = urlguard.validate_public_endpoint


def _validate_secret_free_config(config_value: dict[str, Any]) -> None:
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

    def contains_forbidden(value: Any) -> bool:
        if isinstance(value, dict):
            if any(str(key).lower() in forbidden for key in value):
                return True
            return any(contains_forbidden(child) for child in value.values())
        if isinstance(value, list):
            return any(contains_forbidden(child) for child in value)
        return False

    if contains_forbidden(config_value):
        raise HTTPException(status_code=422, detail="custom_config contains a forbidden credential field")


def _endpoint_config(
    item: dict[str, Any],
    custom_config: McpCustomConfig | dict[str, Any] | None,
    *,
    upstream_authorization_value: str | None = None,
) -> tuple[str, str | None]:
    trusted_runtime = False
    if custom_config is not None:
        if isinstance(custom_config, McpCustomConfig):
            config_value = custom_config.model_dump()
        else:
            _validate_secret_free_config(custom_config)
            config_value = custom_config
        runtime = config_value.get("runtime") if isinstance(config_value.get("runtime"), dict) else {}
        if runtime and runtime.get("verified") is not True:
            raise HTTPException(status_code=409, detail="MCP runtime is not verified")
        trusted_runtime = bool(runtime and runtime.get("verified") is True)
        endpoint = config_value.get("url") or runtime.get("endpoint")
        auth_config = config_value.get("auth") if isinstance(config_value.get("auth"), dict) else {}
        auth_mode = auth_config.get("mode", "none")
        if auth_mode not in {"bearer", "oauth", "none"}:
            raise HTTPException(status_code=422, detail="MCP auth mode is invalid")
        if auth_mode in {"bearer", "oauth"} and not upstream_authorization_value:
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "mcp_auth_failed",
                    "message": "MCP server authentication failed",
                    "reached_upstream": False,
                },
            )
        token = upstream_authorization_value if auth_mode in {"bearer", "oauth"} else None
    else:
        endpoint = item.get("internal_endpoint")
        token = None
        if item.get("slug") == "mcp-searxng":
            token = config.MCP_SEARXNG_AUTH_TOKEN or None
        elif item.get("slug") == "mcp-vision":
            token = config.MCP_VISION_AUTH_TOKEN or None
        elif item.get("slug") == VPS_MCP_SLUG:
            token = config.MCP_VPS_AUTH_TOKEN or None
    endpoint = _validate_endpoint(endpoint, reject_query=custom_config is not None)
    if custom_config is not None and not trusted_runtime:
        _validate_public_endpoint(endpoint)
    return endpoint, token


def _effective_isolation(item: dict[str, Any], requested: str | None, *, custom: bool) -> str:
    """Choose isolation without allowing a caller to downgrade a catalog item."""
    catalog_isolation = item.get("isolation") or "stateless"
    if catalog_isolation not in {"stateless", "tenant_aware"}:
        raise HTTPException(status_code=502, detail="MCP catalog returned invalid isolation")
    if custom:
        return requested or "stateless"
    if requested and requested != catalog_isolation:
        raise HTTPException(status_code=400, detail="isolation does not match the catalog item")
    return catalog_isolation


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _catalog_token(item: dict[str, Any]) -> str | None:
    # Mirrors _endpoint_config(): the pilot's single shared instance keeps its
    # bearer token in Gateway config, never in the Market catalog record.
    if item.get("slug") == "mcp-searxng":
        return config.MCP_SEARXNG_AUTH_TOKEN or None
    if item.get("slug") == "mcp-vision":
        return config.MCP_VISION_AUTH_TOKEN or None
    if item.get("slug") == VPS_MCP_SLUG:
        return config.MCP_VPS_AUTH_TOKEN or None
    return None


class VisionConfigUpdate(BaseModel):
    provider: Literal["gemini", "openrouter"]
    model: str = Field(min_length=1, max_length=200)
    api_key: str | None = Field(default=None, min_length=8, max_length=8192)
    clear_api_key: bool = False
    # internal design notes Stage 3A.1: split from the old
    # max_image_bytes, which was always the transport gate — a straight
    # rename, range widened to match mcp-vision's new default (24 MiB) and
    # its own upper bound (32 MiB, the Express body-limit ceiling).
    max_transport_image_bytes: int | None = Field(default=None, ge=1024 * 1024, le=32 * 1024 * 1024)
    max_processing_pixels: int | None = Field(default=None, ge=100_000, le=40_000_000)
    site_url: str | None = Field(default=None, max_length=500)
    app_name: str | None = Field(default=None, max_length=200)


class VisionModelsRequest(BaseModel):
    provider: Literal["gemini", "openrouter"]
    api_key: str | None = Field(default=None, min_length=8, max_length=8192)
    site_url: str | None = Field(default=None, max_length=500)
    app_name: str | None = Field(default=None, max_length=200)


def _vision_admin_endpoint(resource: Literal["config", "models"]) -> str:
    root = config.MCP_VISION_ADMIN_URL.rsplit("/", 1)[0]
    return f"{root}/{resource}"


async def _vision_admin_request(
    method: str,
    resource: Literal["config", "models"],
    payload: dict[str, Any] | None = None,
) -> Any:
    if not config.MCP_VISION_ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="Vision administration is unavailable")
    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SEC) as client:
            response = await client.request(
                method,
                _vision_admin_endpoint(resource),
                json=payload,
                headers={"Authorization": f"Bearer {config.MCP_VISION_ADMIN_TOKEN}"},
            )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Vision administration is unavailable") from None
    if response.status_code >= 400:
        # Never proxy upstream bodies: validation errors can be derived from
        # submitted credentials and the service deliberately keeps them local.
        status_code = 422 if response.status_code < 500 else 503
        raise HTTPException(status_code=status_code, detail="Vision configuration was rejected")
    try:
        return response.json()
    except ValueError:
        raise HTTPException(status_code=503, detail="Vision administration is unavailable") from None


@app.get("/v1/admin/vision-config", dependencies=[Depends(require_gateway_admin_token)])
async def get_vision_config() -> Any:
    return await _vision_admin_request("GET", "config")


@app.put("/v1/admin/vision-config", dependencies=[Depends(require_gateway_admin_token)])
async def update_vision_config(body: VisionConfigUpdate) -> Any:
    return await _vision_admin_request("PUT", "config", body.model_dump(exclude_none=True))


@app.post("/v1/admin/vision-models", dependencies=[Depends(require_gateway_admin_token)])
async def validate_vision_models(body: VisionModelsRequest) -> Any:
    return await _vision_admin_request("POST", "models", body.model_dump(exclude_none=True))


async def _vps_service_posture() -> dict[str, Any]:
    """Ask legion-vps-mcp what it actually does, via its own /health.

    internal design notes §3.13, Stage 9. The service is the only party that knows
    whether it has an egress broker; everything else is an operator's label.
    An unreachable service yields ``reachable: false`` — never an assumed
    posture.
    """

    endpoint = config.MCP_VPS_INTERNAL_ENDPOINT
    try:
        item = await market.resolve_by_slug(VPS_MCP_SLUG)
    except HTTPException:
        item = None
    if isinstance(item, dict) and item.get("internal_endpoint"):
        endpoint = str(item["internal_endpoint"])
    try:
        health_url = _adapter_health_url(_validate_endpoint(endpoint))
    except HTTPException:
        return {"reachable": False, "reason": "endpoint_invalid"}
    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SEC) as client:
            response = await client.get(health_url)
    except httpx.RequestError:
        return {"reachable": False, "reason": "service_unreachable"}
    if response.status_code >= 400:
        return {"reachable": False, "reason": f"http_{response.status_code}"}
    try:
        body = response.json()
    except ValueError:
        return {"reachable": False, "reason": "bad_health_response"}
    egress = body.get("egress") if isinstance(body, dict) else None
    observed = egress.get("mode") if isinstance(egress, dict) else None
    return {
        "reachable": True,
        "version": str(body.get("version")) if isinstance(body, dict) and body.get("version") else None,
        # Older builds of the service predate the `egress` block; "unknown" is
        # the honest answer there, not "direct".
        "egress_mode": observed if observed in {"broker", "direct"} else "unknown",
    }


async def _egress_posture() -> dict[str, Any]:
    declared = config.MCP_EGRESS_MODE
    service = await _vps_service_posture()
    observed = service.get("egress_mode") if service.get("reachable") else None
    if declared not in _EGRESS_MODES:
        # A typo in MCP_EGRESS_MODE must not read as a configured posture.
        verdict = "misconfigured"
    elif observed is None or observed == "unknown":
        verdict = "unverified"
    elif declared == "broker":
        verdict = "enforced" if observed == "broker" else "mismatch"
    elif declared == "host_firewall":
        # Layer C lives in the host's kernel; Legion can neither confirm nor
        # deny it. What we *can* say is that we are not enforcing it ourselves.
        verdict = "claimed_by_operator" if observed == "direct" else "mismatch"
    else:  # "off"
        verdict = "not_enforced" if observed == "direct" else "mismatch"
    return {
        "declared_mode": declared,
        "declared_mode_valid": declared in _EGRESS_MODES,
        "service": service,
        "verdict": verdict,
    }


@app.get("/v1/egress-posture", dependencies=[Depends(require_gateway_token)])
async def egress_posture() -> dict[str, Any]:
    """Hub-facing posture read (internal design notes Stage 9)."""

    return await _egress_posture()


@app.get("/v1/admin/egress-posture", dependencies=[Depends(require_gateway_admin_token)])
async def admin_egress_posture() -> dict[str, Any]:
    """Admin Center-facing posture read — same answer, operator credential."""

    return await _egress_posture()


def _adapter_health_url(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


async def _diag_market_catalog(slug: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        item = await market.resolve_by_slug(slug)
    except HTTPException as exc:
        reason = "market_catalog_unavailable" if exc.status_code == 502 else "item_not_found"
        return {"status": "error", "reason": reason, "detail": exc.detail}, None
    return {"status": "ok"}, item


async def _diag_mcp_transport(endpoint: str) -> dict[str, Any]:
    health_urls = [_adapter_health_url(endpoint)]
    # MCP Toolbox exposes /healthz while the SearXNG adapter exposes /health.
    health_urls.append(urlunsplit((*urlsplit(health_urls[0])[:2], "/healthz", "", "")))
    last_error = "no health response"
    try:
        async with httpx.AsyncClient(timeout=config.REQUEST_TIMEOUT_SEC) as client:
            for health_url in health_urls:
                response = await client.get(health_url)
                if response.status_code < 400:
                    return {"status": "ok"}
                last_error = f"HTTP {response.status_code}"
    except httpx.RequestError as exc:
        last_error = str(exc)
    return {"status": "error", "reason": "mcp_transport_unreachable", "detail": last_error}


async def _diag_tools_list(endpoint: str, token: str | None) -> tuple[dict[str, Any], list[str]]:
    try:
        tools = await McpHttpClient(endpoint, token=token).tools()
    except HTTPException as exc:
        reason = "mcp_auth_failed" if "authentication" in str(exc.detail).lower() else "mcp_transport_unreachable"
        return {"status": "error", "reason": reason, "detail": exc.detail}, []
    names = [str(tool.get("name")) for tool in tools if tool.get("name")]
    return {"status": "ok", "tools": names}, names


async def _diag_upstream_search(endpoint: str, token: str | None, tool_names: list[str]) -> dict[str, Any]:
    tool_name = next((name for name in tool_names if "search" in name.lower()), None)
    if not tool_name:
        return {"status": "skipped", "reason": "no_search_tool_published"}
    try:
        response = await McpHttpClient(endpoint, token=token).call(
            tool_name, {"query": "legion mcp gateway diagnostic"}
        )
    except HTTPException as exc:
        return {"status": "error", "reason": "mcp_transport_unreachable", "detail": exc.detail}
    payload = response.get("result") if isinstance(response, dict) else None
    if isinstance(payload, dict) and payload.get("isError"):
        text = " ".join(
            str(part.get("text", "")) for part in payload.get("content", []) if isinstance(part, dict)
        )
        lowered = text.lower()
        if "not set" in lowered or "configuration error" in lowered:
            reason = "upstream_not_configured"
        elif "connection error" in lowered or "dns error" in lowered or "timeout error" in lowered:
            reason = "upstream_unreachable"
        else:
            reason = "upstream_error"
        return {"status": "error", "reason": reason, "detail": text[:300]}
    return {"status": "ok"}


@app.get("/v1/diagnostics", dependencies=[Depends(require_gateway_token)])
async def diagnostics(slug: str = Query(default="mcp-searxng", max_length=100)) -> dict[str, Any]:
    """One-shot deep health check: catalog -> adapter -> upstream search.

    Intended for operator/CI smoke-testing (§9.2/§10.3 of the market plan),
    not for a tight orchestrator healthcheck loop — the final probe issues one
    real search call and must not be polled at second-level intervals.
    """
    market_status, item = await _diag_market_catalog(slug)
    report: dict[str, Any] = {
        "market_catalog": market_status,
        "gateway": {"status": "ok"},
        "mcp_transport": {"status": "skipped", "reason": "market_catalog_unavailable"},
        "tools_list": {"status": "skipped", "reason": "market_catalog_unavailable"},
        "upstream_search": {"status": "skipped", "reason": "market_catalog_unavailable"},
    }
    if item is None:
        return report
    endpoint = str(item.get("internal_endpoint") or "")
    token = _catalog_token(item)
    report["mcp_transport"] = await _diag_mcp_transport(endpoint)
    tools_status, tool_names = await _diag_tools_list(endpoint, token)
    report["tools_list"] = tools_status
    if tools_status["status"] == "ok":
        report["upstream_search"] = await _diag_upstream_search(endpoint, token, tool_names)
    else:
        report["upstream_search"] = {"status": "skipped", "reason": "tools_list_failed"}
    return report


_DISCOVERY_MESSAGES = {
    "mcp_auth_failed": "MCP server authentication failed",
    "mcp_unreachable": "MCP server is unavailable",
    "mcp_protocol_error": "MCP server protocol validation failed",
    "mcp_bad_response": "MCP server returned an invalid response",
    "mcp_http_error": "MCP server returned an HTTP error",
    "mcp_secret_unavailable": "MCP authorization is not configured",
    "mcp_oauth_required": "MCP server requires an OAuth login",
}
#: Not retryable: pressing "check again" cannot produce a login.
_DISCOVERY_RETRYABLE = {"mcp_unreachable", "mcp_http_error"}


def _discovery_error_code(exc: HTTPException) -> str:
    detail = exc.detail
    if isinstance(detail, dict) and detail.get("code") in _DISCOVERY_MESSAGES:
        return str(detail["code"])
    if exc.status_code >= 500:
        return "mcp_unreachable"
    return "mcp_protocol_error"


def _discovery_response_error(
    installed_item_id: UUID,
    *,
    checked_at: datetime,
    correlation_id: str,
    code: str,
    provider: str | None = None,
) -> JSONResponse:
    error = McpError(
        code=code,
        message=_DISCOVERY_MESSAGES[code],
        retryable=code in _DISCOVERY_RETRYABLE,
        retry_after_seconds=None,
        provider=provider if code == "mcp_oauth_required" else None,
    )
    body = DiscoveryResponse(
        installed_item_id=installed_item_id,
        status="error",
        tools=[],
        fingerprint=None,
        checked_at=checked_at,
        expires_at=None,
        ttl_seconds=60,
        error=error,
        correlation_id=correlation_id,
    )
    return JSONResponse(status_code=502, content=body.model_dump(mode="json"))


@app.post("/v1/tools/discover", response_model=DiscoveryResponse, dependencies=[Depends(require_gateway_token)])
async def discover_tools(
    body: DiscoveryRequest,
    _: tuple[str, str | None] = Depends(required_identity),
    upstream_token: str | None = Depends(upstream_authorization),
    request_correlation: str = Depends(request_correlation_id),
) -> DiscoveryResponse | JSONResponse:
    checked_at = datetime.now(timezone.utc)
    correlation_id = request_correlation
    endpoint: str | None = None
    try:
        item = (
            await market.resolve(body.installed_item_id, market_item_id=body.market_item_id, slug=body.market_slug)
            if body.custom_config is None
            else {"slug": f"custom-{body.installed_item_id}", "isolation": "stateless"}
        )
        endpoint, token = _endpoint_config(
            item,
            body.custom_config,
            upstream_authorization_value=upstream_token,
        )
        raw_tools = await McpHttpClient(endpoint, token=token).tools()
        tools = normalize_tools(raw_tools)
        fingerprint = tools_fingerprint(tools)
        expires_at = checked_at + timedelta(seconds=60)
        logger.info("mcp_discovery outcome=success code=verified correlation_id=%s", correlation_id)
        return DiscoveryResponse(
            installed_item_id=body.installed_item_id,
            status="verified",
            tools=[ToolInfo.model_validate(tool) for tool in tools],
            fingerprint=fingerprint,
            checked_at=checked_at,
            expires_at=expires_at,
            ttl_seconds=60,
            error=None,
            correlation_id=correlation_id,
        )
    except HTTPException as exc:
        code = _discovery_error_code(exc)
        provider: str | None = None
        if code == "mcp_auth_failed":
            # internal design notes Stage 0 — a 401 carrying an OAuth challenge is
            # not a wrong token: the server wants a login, and the user needs
            # to be told by whom.  The probe never changes any other outcome.
            provider = await oauth_probe.detect_provider(
                endpoint, getattr(exc, "www_authenticate", None)
            )
            if provider is not None:
                code = "mcp_oauth_required"
        logger.info(
            "mcp_discovery outcome=error code=%s correlation_id=%s",
            code,
            correlation_id,
        )
        return _discovery_response_error(
            body.installed_item_id,
            checked_at=checked_at,
            correlation_id=correlation_id,
            code=code,
            provider=provider,
        )


#: internal design notes Stage 2 — the only three routes allowed to reach a
#: provider's OAuth endpoints. Hub's app.core.mcp_oauth (Stage 1) calls these
#: instead of ever making the hop itself; each one crosses oauth_client's
#: urlguard-guarded perimeter, the same one /v1/execute crosses.
@app.post("/v1/oauth/metadata", dependencies=[Depends(require_gateway_token)])
async def oauth_metadata(body: OAuthMetadataRequest) -> dict[str, Any]:
    target = body.issuer if body.kind == "authorization_server" else body.resource
    if not target:
        raise HTTPException(status_code=422, detail="Invalid Gateway request")
    return await oauth_client.fetch_metadata(kind=body.kind, target=target)


@app.post("/v1/oauth/register", dependencies=[Depends(require_gateway_token)])
async def oauth_register(body: OAuthRegisterRequest) -> dict[str, Any]:
    payload = {
        "client_name": body.client_name,
        "redirect_uris": body.redirect_uris,
        "token_endpoint_auth_method": body.token_endpoint_auth_method,
        "grant_types": body.grant_types,
        "response_types": body.response_types,
    }
    return await oauth_client.register_client(
        registration_endpoint=body.registration_endpoint, payload=payload
    )


@app.post("/v1/oauth/token", dependencies=[Depends(require_gateway_token)])
async def oauth_token(body: OAuthTokenRequest) -> dict[str, Any]:
    payload = {"grant_type": body.grant_type, "client_id": body.client_id}
    if body.grant_type == "authorization_code":
        payload["code"] = body.code
        payload["redirect_uri"] = body.redirect_uri
        payload["code_verifier"] = body.code_verifier
    else:
        payload["refresh_token"] = body.refresh_token
    if body.resource:
        payload["resource"] = body.resource
    if body.client_secret:
        payload["client_secret"] = body.client_secret
    return await oauth_client.exchange_token(token_endpoint=body.token_endpoint, payload=payload)


def _catalog_only(caller: Caller, custom_config: Any) -> None:
    """A Legion account key reaches catalogue items only (payments plan §3.10):
    a custom link through our Gateway would make it an open proxy."""
    if caller.is_account and custom_config:
        raise HTTPException(
            status_code=403,
            detail={"code": "custom_not_allowed",
                    "message": "A Legion API key reaches Legion catalogue MCP servers only"},
        )


@app.get("/v1/tools", response_model=ToolsResponse)
async def list_tools(
    installed_item_id: UUID = Query(...),
    market_item_id: UUID | None = Query(default=None),
    market_slug: str | None = Query(default=None, max_length=100),
    custom_config: str | None = Query(default=None),
    caller: Caller = Depends(gateway_caller),
    identity: tuple[str, str | None] = Depends(required_identity),
    upstream_token: str | None = Depends(caller_upstream_authorization),
) -> ToolsResponse:
    _catalog_only(caller, custom_config)
    custom: dict[str, Any] | None = None
    if custom_config:
        try:
            parsed = json.loads(custom_config)
            if isinstance(parsed, dict):
                custom = parsed
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="custom_config must be valid JSON")
    item = await market.resolve(installed_item_id, market_item_id=market_item_id, slug=market_slug) if custom is None else {"slug": f"custom-{installed_item_id}", "isolation": "stateless"}
    endpoint, token = _endpoint_config(item, custom, upstream_authorization_value=upstream_token)
    tools = await McpHttpClient(endpoint, token=token).tools()
    normalized = [
        ToolInfo(
            name=str(tool.get("name", "")),
            description=str(tool.get("description", "")),
            inputSchema=tool.get("inputSchema") or tool.get("input_schema") or {},
        )
        for tool in tools
        if tool.get("name")
    ]
    return ToolsResponse(installed_item_id=installed_item_id, tools=normalized)


def _mcp_result(response: Any) -> Any:
    # Expose the MCP result to Hub, rather than transport bookkeeping
    # (jsonrpc/id), while preserving non-standard list responses.
    if isinstance(response, dict) and "result" in response:
        return response["result"]
    return response


#: Upstream errors after which the call is still charged: the MCP accepted it
#: and the tool itself failed. Mirrors the Hub's own ``_GATEWAY_SETTLE_CODES``.
_SETTLE_CODES = {"mcp_rpc_error"}


async def _execute_for_account(body: ExecuteRequest, caller: Caller, room_id: str | None,
                               correlation_id: str) -> Any:
    """``/v1/execute`` for a self-hosted Hub: paid from the central wallet
    (payments plan §3.10). Reserve in Market → call the MCP → settle, or
    refund when the call never reached it or the tool reported an error."""
    _catalog_only(caller, body.custom_config)
    item = await market.resolve(body.installed_item_id, market_item_id=body.market_item_id, slug=body.market_slug)
    endpoint, token = _endpoint_config(item, None)
    isolation = _effective_isolation(item, body.isolation, custom=False)
    call_id = body.call_id or uuid4()
    # The item reserved is the item that will be called: its id from the
    # catalogue, not whatever slug the caller named.
    reservation = await account_keys.reserve(
        caller, call_id=call_id, market_item_id=str(item.get("id") or body.market_item_id or "") or None,
        market_slug=item.get("slug") or body.market_slug, tool=body.tool,
    )
    reserved = bool(reservation.get("reserved"))
    meta = {"user_id": caller.tenant_id()}
    if room_id:
        meta["room_id"] = room_id
    vision_timeout = config.VISION_REQUEST_TIMEOUT_SEC if item.get("slug") == "mcp-vision" else None
    try:
        response = await McpHttpClient(endpoint, token=token, timeout_sec=vision_timeout).call(
            body.tool, body.arguments, meta=meta if isolation == "tenant_aware" else None,
        )
    except HTTPException as exc:
        if reserved:
            code = exc.detail.get("code") if isinstance(exc.detail, dict) else None
            if code in _SETTLE_CODES:
                await account_keys.settle(call_id)
            else:
                await account_keys.refund(call_id, f"gateway: {code or exc.status_code}")
        raise
    except Exception:
        if reserved:
            await account_keys.refund(call_id, "gateway: upstream call failed")
        raise
    result = _mcp_result(response)
    if reserved:
        if isinstance(result, dict) and result.get("isError"):
            await account_keys.refund(call_id, "tool returned isError")
        else:
            await account_keys.settle(call_id)
    logger.info("mcp_call outcome=success caller=account key=%s correlation_id=%s", caller.key_prefix,
                correlation_id)
    return result


@app.post("/v1/execute")
async def execute(
    body: ExecuteRequest,
    request: Request,
    caller: Caller = Depends(gateway_caller),
    identity: tuple[str, str | None] = Depends(required_identity),
    upstream_token: str | None = Depends(caller_upstream_authorization),
    request_correlation: str = Depends(caller_correlation_id),
) -> Any:
    user_id, room_id = identity
    if caller.is_account:
        return await _execute_for_account(body, caller, room_id, request_correlation)
    # Custom configs are supplied by Hub only after Hub has verified ownership
    # and room assignment; they are never resolved from a client-side catalog.
    item = await market.resolve(
        body.installed_item_id,
        market_item_id=body.market_item_id,
        slug=body.market_slug,
    ) if body.custom_config is None else {
        "slug": f"custom-{body.installed_item_id}",
        "isolation": body.isolation or "stateless",
    }
    endpoint, token = _endpoint_config(
        item,
        body.custom_config,
        upstream_authorization_value=upstream_token,
    )
    isolation = _effective_isolation(item, body.isolation, custom=body.custom_config is not None)
    if body.custom_config is None:
        # Our Hub names the caller's Legion account; a self-hosted Hub's key is
        # checked at the reservation instead (``_execute_for_account``).
        await server_terms.verify(item, request.headers.get(server_terms.ACCOUNT_HEADER))
    await verify_call_credits(
        user_id=user_id,
        call_id=body.call_id,
        market_item_id=str(body.market_item_id) if body.market_item_id else None,
        market_slug=body.market_slug,
        tool=body.tool,
    )
    meta = {"user_id": user_id}
    if room_id:
        meta["room_id"] = room_id
    vision_timeout = (
        config.VISION_REQUEST_TIMEOUT_SEC
        if item.get("slug") == "mcp-vision"
        else None
    )
    response = await McpHttpClient(endpoint, token=token, timeout_sec=vision_timeout).call(
        body.tool,
        body.arguments,
        meta=meta if isolation == "tenant_aware" else None,
    )
    logger.info("mcp_call outcome=success correlation_id=%s", request_correlation)
    return _mcp_result(response)
