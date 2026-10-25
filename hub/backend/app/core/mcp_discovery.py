"""Hub-owned HTTP MCP verification and snapshot lifecycle.

Only the Gateway performs the MCP handshake.  This module resolves a Secret
just in time, sends the resolved bearer through the stage-2 header boundary,
and persists a bounded, value-free result for room/API consumers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import mcp_gateway
from app.core.http_mcp_rollout import enabled_for
from app.core.metrics import record_mcp_discovery
from app.core.secrets import runtime as secret_runtime
from app.models.market import UserInstalledItem
from app.models.mcp_connection import McpConnectionSnapshot
from app.models.secret import SecretBinding, SecretPurpose, SecretResourceType
from app.schemas.mcp_connection import McpConnectionStatusOut

logger = logging.getLogger(__name__)

SNAPSHOT_TTL_SECONDS = 60
MAX_TOOLS = 128
MAX_TOOL_SNAPSHOT_BYTES = 256 * 1024
MAX_OPERATION_SECONDS = 10
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,255}$")
_SAFE_ERROR_CODES = {
    "mcp_auth_failed",
    "mcp_unreachable",
    "mcp_protocol_error",
    "mcp_bad_response",
    "mcp_http_error",
    "mcp_secret_unavailable",
    "mcp_oauth_required",
}

#: A provider label is upstream-derived data that reached us through Gateway.
#: Hub re-checks it instead of trusting the hop: display-safe characters only,
#: never a URL, never long enough to smuggle a sentence into the card.
_MAX_PROVIDER_CHARS = 64

_ERRORS: dict[str, dict[str, Any]] = {
    "mcp_auth_failed": {
        "message": "HTTP MCP authorization failed.",
        "remedy": "Check the Secret binding and run the check again.",
        "retryable": True,
    },
    "mcp_unreachable": {
        "message": "HTTP MCP is unavailable.",
        "remedy": "Check the endpoint and the network, then run the check again.",
        "retryable": True,
    },
    "mcp_protocol_error": {
        "message": "HTTP MCP returned an incompatible MCP protocol response.",
        "remedy": "Check initialize/tools/list on the endpoint and run the check again.",
        "retryable": False,
    },
    "mcp_bad_response": {
        "message": "HTTP MCP returned an invalid response.",
        "remedy": "Check the MCP tools/list format and run the check again.",
        "retryable": False,
    },
    "mcp_http_error": {
        "message": "HTTP MCP returned a transport error.",
        "remedy": "Check the endpoint and run the check again later.",
        "retryable": True,
    },
    "mcp_secret_unavailable": {
        "message": "The Secret for MCP authorization is unavailable.",
        "remedy": "Assign an active mcp_header Secret and run the check again.",
        "retryable": True,
    },
    # internal design notes Stage 0.  Re-checking changes nothing until
    # someone signs in, hence retryable=False — unlike mcp_auth_failed.
    "mcp_oauth_required": {
        "message": "The server requires a sign-in at the provider.",
        "remedy": "Press \"Sign in\" and confirm access at the provider.",
        "retryable": False,
    },
}

_locks: dict[UUID, asyncio.Lock] = {}
_last_attempt: dict[UUID, float] = {}
_ZERO_ROOM_ID = UUID(int=0)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def is_http_link(item: Any) -> bool:
    config = item.custom_config if isinstance(getattr(item, "custom_config", None), dict) else {}
    return bool(
        getattr(item, "type", None) == "mcp"
        and getattr(item, "is_custom", False)
        and config.get("mode") not in {"hosted", "managed_stdio"}
    )


def _safe_auth(item: Any) -> tuple[str, UUID | None]:
    config = item.custom_config if isinstance(getattr(item, "custom_config", None), dict) else {}
    auth = config.get("auth")
    if not isinstance(auth, dict):
        return "none", None
    mode = auth.get("mode", "none")
    if mode not in {"none", "bearer", "oauth"}:
        raise ValueError("unsupported auth mode")
    raw_binding = auth.get("secret_binding_id")
    if raw_binding in (None, ""):
        return str(mode), None
    try:
        return str(mode), UUID(str(raw_binding))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid auth binding") from exc


def _source_fingerprint(item: Any) -> str:
    """Hash only declarative endpoint identity; never persist the endpoint."""

    config = item.custom_config if isinstance(getattr(item, "custom_config", None), dict) else {}
    mode, binding_id = _safe_auth(item)
    identity = {
        "url": config.get("url"),
        "auth_mode": mode,
        "binding_id": str(binding_id) if binding_id is not None else ("auto" if mode in {"bearer", "oauth"} else None),
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _safe_provider(value: Any) -> str | None:
    """Re-sanitize the provider label Gateway derived from upstream metadata."""

    if not isinstance(value, str):
        return None
    cleaned = "".join(
        char for char in value.strip() if ord(char) >= 0x20 and ord(char) != 0x7F
    ).strip()
    if not cleaned or "://" in cleaned or len(cleaned) > _MAX_PROVIDER_CHARS:
        return None
    return cleaned


def _safe_error(code: str, correlation_id: str, provider: str | None = None) -> dict[str, Any]:
    details = _ERRORS.get(code, _ERRORS["mcp_http_error"])
    code = code if code in _SAFE_ERROR_CODES else "mcp_http_error"
    message = details["message"]
    extra: dict[str, Any] = {}
    provider = _safe_provider(provider) if code == "mcp_oauth_required" else None
    if provider:
        # The provider name is the only thing Stage 0 adds to the card:
        # "who exactly is asking for a sign-in" instead of a silent check failure.
        message = f"The server requires a sign-in at the provider ({provider})."
        extra["provider"] = provider
    return {
        "code": code,
        "message": message,
        "remedy": details["remedy"],
        "retryable": bool(details["retryable"]),
        "details": extra,
        "correlation_id": correlation_id,
    }


def _safe_code(value: Any, *, status_code: int | None = None) -> str:
    if isinstance(value, str) and value in _SAFE_ERROR_CODES:
        return value
    if status_code in {401, 403}:
        return "mcp_auth_failed"
    if status_code is not None and status_code >= 500:
        return "mcp_unreachable"
    return "mcp_http_error"


def _normalize_tools(raw_tools: Iterable[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw in raw_tools:
        if not isinstance(raw, dict):
            raise ValueError("tool is not an object")
        name = raw.get("name")
        if not isinstance(name, str) or not _TOOL_NAME_RE.fullmatch(name):
            raise ValueError("tool name is invalid")
        description = raw.get("description", "")
        if description is None:
            description = ""
        if not isinstance(description, str) or len(description) > 4096:
            raise ValueError("tool description is invalid")
        schema = raw.get("inputSchema", raw.get("input_schema", {"type": "object"}))
        if not isinstance(schema, dict):
            raise ValueError("tool input schema is invalid")
        normalized.append({"name": name, "description": description, "inputSchema": schema})
        if len(normalized) > MAX_TOOLS:
            raise ValueError("too many tools")
    if not normalized:
        raise ValueError("tools list is empty")
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_TOOL_SNAPSHOT_BYTES:
        raise ValueError("tools snapshot is too large")
    return normalized


def _tools_fingerprint(tools: list[dict[str, Any]]) -> str:
    encoded = json.dumps(tools, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _discovery_config(item: Any) -> dict[str, Any]:
    config = item.custom_config if isinstance(getattr(item, "custom_config", None), dict) else {}
    url = config.get("url")
    mode, _ = _safe_auth(item)
    if not isinstance(url, str) or not url:
        raise ValueError("endpoint is missing")
    return {"url": url, "auth": {"mode": mode}}


async def _gateway_post(*args: Any, **kwargs: Any) -> Any:
    """Indirection keeps the service easy to contract-test and patches Gateway dynamically."""

    return await mcp_gateway._gateway_post(*args, **kwargs)


async def resolve_mcp_upstream_auth(**kwargs: Any) -> dict[str, str]:
    """Patchable wrapper around the stage-2 Secret boundary."""

    # Keep discovery and execution on the exact same Hub-owned resolver.  The
    # indirection is also useful for contract tests that replace the shared
    # Gateway boundary without ever materializing a Secret value here.
    return await mcp_gateway.resolve_mcp_upstream_auth(**kwargs)


async def ensure_snapshot(
    db: AsyncSession,
    item: Any,
    *,
    status: str = "unknown",
) -> McpConnectionSnapshot:
    snapshot = await db.get(McpConnectionSnapshot, item.id)
    if snapshot is None:
        snapshot = McpConnectionSnapshot(
            installed_item_id=item.id,
            owner_id=item.user_id,
            status=status,
        )
        add = getattr(db, "add", None)
        if callable(add):
            add(snapshot)
    return snapshot


async def invalidate_mcp_snapshot(
    db: AsyncSession,
    installed_item_id: UUID,
    *,
    owner_id: UUID | None = None,
) -> McpConnectionSnapshot | None:
    """Hide tools immediately after endpoint/binding/secret state changes."""

    snapshot = await db.get(McpConnectionSnapshot, installed_item_id)
    if snapshot is None:
        if owner_id is None:
            return None
        snapshot = McpConnectionSnapshot(
            installed_item_id=installed_item_id,
            owner_id=owner_id,
        )
        db.add(snapshot)
    snapshot.status = "unknown"
    snapshot.tools_snapshot = []
    snapshot.tool_fingerprint = None
    snapshot.source_fingerprint = None
    snapshot.checked_at = None
    snapshot.expires_at = None
    snapshot.last_error_code = None
    snapshot.last_error_safe = None
    snapshot.latency_ms = None
    snapshot.correlation_id = None
    snapshot.updated_at = _utcnow()
    add = getattr(db, "add", None)
    if callable(add):
        add(snapshot)
    return snapshot


async def invalidate_mcp_snapshots_for_secret(
    db: AsyncSession,
    secret_id: UUID,
    *,
    owner_id: UUID | None = None,
) -> list[UUID]:
    conditions = [
        SecretBinding.secret_id == secret_id,
        SecretBinding.resource_type == SecretResourceType.INSTALLED_MCP,
        SecretBinding.purpose == SecretPurpose.MCP_AUTH_HEADER,
    ]
    if owner_id is not None:
        conditions.append(SecretBinding.owner_id == owner_id)
    bindings = await db.execute(select(SecretBinding.resource_id).where(*conditions))
    item_ids = [row[0] for row in bindings.all()]
    for item_id in item_ids:
        await invalidate_mcp_snapshot(db, item_id)
    return item_ids


async def _status_from_snapshot(
    db: AsyncSession,
    item: Any,
    snapshot: McpConnectionSnapshot | None,
) -> McpConnectionStatusOut:
    # Rollback is fail-closed: keep the durable row and bindings intact, but
    # never expose a previously verified link while the flag is off or the
    # owner is outside the active internal/canary cohort.
    if not enabled_for(getattr(item, "user_id", None)):
        return McpConnectionStatusOut(installed_item_id=item.id, status="unknown")
    if snapshot is None:
        return McpConnectionStatusOut(installed_item_id=item.id, status="unknown")

    owner_matches = getattr(snapshot, "owner_id", None) == getattr(item, "user_id", None)
    status = snapshot.status if snapshot.status in {"unknown", "checking", "verified", "error"} else "unknown"
    if not owner_matches:
        status = "unknown"
    available = False
    if status == "verified" and owner_matches:
        expires_at = _aware(snapshot.expires_at)
        source_matches = False
        try:
            source_matches = bool(snapshot.source_fingerprint) and snapshot.source_fingerprint == _source_fingerprint(item)
        except (TypeError, ValueError):
            source_matches = False
        if expires_at is not None and expires_at <= _utcnow():
            status = "expired"
        elif source_matches and isinstance(snapshot.tools_snapshot, list) and snapshot.tools_snapshot:
            available = True

    tools = snapshot.tools_snapshot if isinstance(snapshot.tools_snapshot, list) else []
    safe_tools: list[dict[str, Any]] = []
    for tool in tools[:MAX_TOOLS]:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str) and isinstance(tool.get("inputSchema", {}), dict):
            safe_tools.append({
                "name": tool["name"],
                "description": str(tool.get("description") or ""),
                "inputSchema": tool.get("inputSchema") or {},
            })
    expires_at = _aware(snapshot.expires_at)
    ttl = max(0, int((expires_at - _utcnow()).total_seconds())) if expires_at else 0
    last_error = (
        snapshot.last_error_safe
        if owner_matches and isinstance(snapshot.last_error_safe, dict)
        else None
    )
    return McpConnectionStatusOut(
        installed_item_id=item.id,
        status=status,
        available=available and owner_matches and getattr(item, "status", "active") == "active",
        ready_for_room=available and owner_matches and getattr(item, "status", "active") == "active",
        tools=safe_tools if available and owner_matches else [],
        tool_count=len(safe_tools) if available and owner_matches else 0,
        tools_count=len(safe_tools) if available and owner_matches else 0,
        tool_fingerprint=snapshot.tool_fingerprint if available and owner_matches else None,
        checked_at=snapshot.checked_at,
        expires_at=expires_at,
        ttl_seconds=ttl,
        last_error_code=snapshot.last_error_code,
        last_error=last_error,
        latency_ms=snapshot.latency_ms,
        correlation_id=snapshot.correlation_id,
    )


async def mcp_connection_status(
    db: AsyncSession,
    item: Any,
    *,
    snapshot: McpConnectionSnapshot | None = None,
) -> McpConnectionStatusOut | None:
    if not is_http_link(item):
        return None
    if snapshot is None:
        snapshot = await db.get(McpConnectionSnapshot, item.id)
    if not isinstance(snapshot, McpConnectionSnapshot):
        # Lightweight unit-test doubles often return an AsyncMock for an
        # unconfigured ``db.get``; it represents a missing row, not a status.
        snapshot = None
    return await _status_from_snapshot(db, item, snapshot)


async def fresh_mcp_connection_status(
    db: AsyncSession,
    item: UserInstalledItem,
) -> McpConnectionStatusOut:
    """Return a verified HTTP MCP status, probing at most once per cooldown.

    A valid snapshot is the runtime source of truth.  Unknown, expired, or
    failed snapshots get one bounded verification attempt; the per-item lock
    and cooldown in :func:`verify_mcp_connection` prevent a room with many
    sessions from turning an unavailable endpoint into a request storm.
    """

    snapshot = await db.get(McpConnectionSnapshot, item.id)
    status = await _status_from_snapshot(db, item, snapshot)
    if not enabled_for(getattr(item, "user_id", None)):
        return status
    if status.available:
        return status
    return await verify_mcp_connection(db, item, force=False)


def _response_error(body: Any, *, status_code: int | None = None) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return _safe_code(error.get("code"), status_code=status_code)
        detail = body.get("detail")
        if isinstance(detail, dict):
            return _safe_code(detail.get("code"), status_code=status_code)
    return _safe_code(None, status_code=status_code)


def _response_provider(body: Any) -> str | None:
    """Pull the OAuth provider label out of a Gateway error body, if any."""

    if not isinstance(body, dict):
        return None
    for key in ("error", "detail"):
        section = body.get(key)
        if isinstance(section, dict):
            return _safe_provider(section.get("provider"))
    return None


async def verify_mcp_connection(
    db: AsyncSession,
    item: UserInstalledItem,
    *,
    force: bool = False,
) -> McpConnectionStatusOut:
    """Run one bounded initialize/tools-list probe and persist its result."""

    if not is_http_link(item):
        raise ValueError("only custom HTTP MCP links have connection snapshots")
    if not enabled_for(getattr(item, "user_id", None)):
        record_mcp_discovery(
            origin="http_link",
            outcome="disabled",
            code="rollout_disabled",
            duration_seconds=0,
        )
        snapshot = await db.get(McpConnectionSnapshot, item.id)
        return await _status_from_snapshot(db, item, snapshot)
    lock = _locks.setdefault(item.id, asyncio.Lock())
    async with lock:
        now_monotonic = time.monotonic()
        if not force and now_monotonic - _last_attempt.get(item.id, 0) < 10:
            existing = await db.get(McpConnectionSnapshot, item.id)
            return await _status_from_snapshot(db, item, existing)
        _last_attempt[item.id] = now_monotonic

        snapshot = await ensure_snapshot(db, item)
        if snapshot.status == "checking" and not force:
            return await _status_from_snapshot(db, item, snapshot)
        snapshot.status = "checking"
        snapshot.tools_snapshot = []
        snapshot.tool_fingerprint = None
        snapshot.expires_at = None
        snapshot.last_error_code = None
        snapshot.last_error_safe = None
        snapshot.correlation_id = f"mcpd_hub_{uuid4().hex}"
        snapshot.updated_at = _utcnow()
        add = getattr(db, "add", None)
        if callable(add):
            add(snapshot)
        commit = getattr(db, "commit", None)
        if callable(commit):
            await commit()

        started = time.monotonic()
        error_code: str | None = None
        error_provider: str | None = None
        tools: list[dict[str, Any]] = []
        source_fingerprint: str | None = None
        try:
            source_fingerprint = _source_fingerprint(item)
            mode, binding_id = _safe_auth(item)
            headers: dict[str, str] = {}
            if mode in {"bearer", "oauth"}:
                headers = await resolve_mcp_upstream_auth(
                    db=db,
                    owner_id=item.user_id,
                    installed_item_id=item.id,
                    binding_id=binding_id,
                    request_id=uuid4(),
                )
            # Correlation is safe metadata and is intentionally separate from
            # the credential-bearing upstream header.
            gateway_headers = dict(headers)
            gateway_headers["X-Legion-Correlation-ID"] = snapshot.correlation_id or ""
            body = await asyncio.wait_for(
                _gateway_post(
                    "/v1/tools/discover",
                    item.user_id,
                    _ZERO_ROOM_ID,
                    {
                        "installed_item_id": str(item.id),
                        "market_item_id": None,
                        "market_slug": None,
                        "custom_config": _discovery_config(item),
                    },
                    extra_headers=gateway_headers,
                ),
                timeout=MAX_OPERATION_SECONDS,
            )
            if not isinstance(body, dict) or body.get("status") != "verified":
                error_code = _response_error(body)
                error_provider = _response_provider(body)
            else:
                tools = _normalize_tools(body.get("tools"))
                calculated = _tools_fingerprint(tools)
                if body.get("fingerprint") not in {None, calculated}:
                    error_code = "mcp_bad_response"
        except secret_runtime.McpSecretUnavailable:
            # For an OAuth item the "secret" is the provider's access token:
            # missing or expired with no working refresh token, only a new
            # login fixes it — not the mcp_header binding the generic remedy
            # tells the person to assign.
            error_code = "mcp_oauth_required" if mode == "oauth" else "mcp_secret_unavailable"
        except httpx.HTTPStatusError as exc:
            body: Any = None
            try:
                body = exc.response.json()
            except Exception:
                body = None
            error_code = _response_error(body, status_code=exc.response.status_code)
            error_provider = _response_provider(body)
        except (httpx.TimeoutException, TimeoutError):
            error_code = "mcp_unreachable"
        except (ValueError, TypeError, KeyError):
            error_code = "mcp_bad_response"
        except Exception:
            # Do not log exception text: an HTTP client exception may include
            # a URL or an upstream response body.
            error_code = "mcp_unreachable"

        snapshot.checked_at = _utcnow()
        snapshot.latency_ms = min(2**31 - 1, max(0, int((time.monotonic() - started) * 1000)))
        if error_code is None:
            snapshot.status = "verified"
            snapshot.tools_snapshot = tools
            snapshot.tool_fingerprint = _tools_fingerprint(tools)
            snapshot.source_fingerprint = source_fingerprint
            snapshot.expires_at = snapshot.checked_at + timedelta(seconds=SNAPSHOT_TTL_SECONDS)
            snapshot.last_error_code = None
            snapshot.last_error_safe = None
        else:
            snapshot.status = "error"
            snapshot.tools_snapshot = []
            snapshot.tool_fingerprint = None
            snapshot.source_fingerprint = source_fingerprint
            snapshot.expires_at = None
            snapshot.last_error_code = error_code
            snapshot.last_error_safe = _safe_error(
                error_code,
                snapshot.correlation_id or f"mcpd_hub_{uuid4().hex}",
                error_provider,
            )
        record_mcp_discovery(
            origin="http_link",
            outcome="success" if error_code is None else "error",
            code=error_code or "verified",
            duration_seconds=(time.monotonic() - started),
        )
        logger.info(
            "mcp_discovery outcome=%s code=%s correlation_id=%s",
            "success" if error_code is None else "error",
            error_code or "verified",
            snapshot.correlation_id,
        )
        snapshot.updated_at = _utcnow()
        if callable(add):
            add(snapshot)
        if callable(commit):
            await commit()
        return await _status_from_snapshot(db, item, snapshot)


__all__ = [
    "MAX_TOOL_SNAPSHOT_BYTES",
    "MAX_TOOLS",
    "SNAPSHOT_TTL_SECONDS",
    "ensure_snapshot",
    "fresh_mcp_connection_status",
    "invalidate_mcp_snapshot",
    "invalidate_mcp_snapshots_for_secret",
    "is_http_link",
    "mcp_connection_status",
    "verify_mcp_connection",
]
