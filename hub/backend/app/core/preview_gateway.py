"""Authenticated, origin-isolated gateway for declared HTTP services.

The legacy ``workspace_preview`` endpoint is intentionally not used here.  It
accepts a port and returns a base64 encoded, complete response, neither of
which is safe or sufficient for a browser application.  This module resolves
the service from its host name and streams the original HTTP wire response
through workspaced's service-only relay.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import AsyncIterator
from urllib.parse import parse_qsl, quote, unquote_plus, urlsplit
from uuid import UUID, uuid4

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core import security, workspaced_client as wsd
from app.models.workspace import Workspace, WorkspaceManagedProcess

logger = logging.getLogger(__name__)
router = APIRouter()

_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}
_ALLOWED_METHODS = {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}
_USED_TICKETS: OrderedDict[str, int] = OrderedDict()
_TICKET_LOCK = asyncio.Lock()
_SESSION_QUERY_PARAM = "_legion_preview_session"
_INTERNAL_QUERY_PARAMS = {"ticket", _SESSION_QUERY_PARAM}
_ROOT_URL_IN_TEXT = re.compile(r"(?P<quote>[\"'])(?P<url>/(?!/)[^\"'\s<>]*)\1")
_REWRITABLE_CONTENT_TYPES = ("text/html", "text/javascript", "application/javascript", "text/css")
_PREVIEW_RETRY_DELAYS = (0.0, 0.25, 0.75, 1.5, 2.0)
_TRANSIENT_UPSTREAM_STATUSES = {409, 502, 503, 504}


class PreviewAuthorizationError(ValueError):
    """The ticket/session is not valid for the requested service origin."""


def _float_env(name: str, default: float, *, low: float, high: float) -> float:
    try:
        return max(low, min(high, float(os.getenv(name, str(default)))))
    except ValueError:
        return default


class _TokenBuckets:
    """Per-key token buckets held in this process.

    The backend runs as a single uvicorn process, so in-memory state is
    enough; a second worker would multiply every limit by the worker count.
    Memory is bounded: once ``max_keys`` is exceeded, the least recently used
    keys are dropped, which at worst grants the dropped key a fresh burst.
    """

    def __init__(self, max_keys: int = 10_000) -> None:
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self._max_keys = max_keys

    def _level(self, key: str, rate: float, burst: float, now: float) -> float:
        tokens, stamp = self._buckets.get(key, (burst, now))
        return min(burst, tokens + max(0.0, now - stamp) * rate)

    def take(self, key: str, *, rate: float, burst: float) -> float:
        """Spend one token; return 0.0 when allowed, else seconds until one exists."""
        now = time.monotonic()
        tokens = self._level(key, rate, burst, now)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0
        self._buckets[key] = (tokens, now)
        self._buckets.move_to_end(key)
        while len(self._buckets) > self._max_keys:
            self._buckets.popitem(last=False)
        return 0.0 if allowed else (1.0 - tokens) / rate

    def retry_after(self, key: str, *, rate: float, burst: float) -> float:
        """Seconds until ``key`` has a token again, without spending one."""
        tokens = self._level(key, rate, burst, time.monotonic())
        return 0.0 if tokens >= 1.0 else (1.0 - tokens) / rate

    def clear(self) -> None:
        self._buckets.clear()


# Authorized traffic: one bucket per service.  A Vite page loads hundreds of
# unbundled modules at once, hence the large burst; the sustained rate is what
# protects the workspace container and the relay from a runaway client.
_SERVICE_BUCKETS = _TokenBuckets()
# Failed credentials / unknown services: one bucket per client address.  This
# is checked before the database lookup, so probing random service ids cannot
# turn every request into a query.
_FAILURE_BUCKETS = _TokenBuckets()


def _service_rate() -> tuple[float, float]:
    """``(rate, burst)``; a rate of 0 disables the limit."""
    rate = _float_env("WORKSPACE_PREVIEW_RATE_PER_SEC", 100.0, low=0.0, high=100_000.0)
    burst = _float_env("WORKSPACE_PREVIEW_RATE_BURST", 400.0, low=1.0, high=1_000_000.0)
    return rate, max(burst, 1.0)


def _failure_rate() -> tuple[float, float]:
    rate = _float_env("WORKSPACE_PREVIEW_FAIL_RATE_PER_SEC", 2.0, low=0.0, high=10_000.0)
    burst = _float_env("WORKSPACE_PREVIEW_FAIL_BURST", 120.0, low=1.0, high=1_000_000.0)
    return rate, max(burst, 1.0)


def _client_key(headers, client) -> str:
    """Identify the caller for failure limiting.

    Behind nginx/Cloudflare ``client.host`` is the proxy, so a single bucket
    would throttle every user at once.  The forwarded headers are honoured
    only when ``WORKSPACE_PREVIEW_TRUST_PROXY_HEADERS`` is set, because a
    directly reachable backend would let a caller pick its own bucket.
    """
    if os.getenv("WORKSPACE_PREVIEW_TRUST_PROXY_HEADERS", "false").strip().lower() in {"1", "true", "yes", "on"}:
        forwarded = headers.get("cf-connecting-ip") or headers.get("x-forwarded-for", "").split(",")[0]
        if forwarded.strip():
            return forwarded.strip()
    return client.host if client else "unknown"


# Open relayed WebSockets per service.  Each one holds a browser socket, an
# upstream socket and two tasks for as long as the page lives (HMR sockets
# live for hours), which the request rate limit does not bound.
_OPEN_WEBSOCKETS: dict[str, int] = {}


def _max_websockets() -> int:
    """Per-service cap; 0 disables."""
    return int(_float_env("WORKSPACE_PREVIEW_MAX_WEBSOCKETS", 32.0, low=0.0, high=100_000.0))


def _acquire_websocket_slot(service_id: UUID) -> bool:
    key, limit = str(service_id), _max_websockets()
    if limit and _OPEN_WEBSOCKETS.get(key, 0) >= limit:
        return False
    _OPEN_WEBSOCKETS[key] = _OPEN_WEBSOCKETS.get(key, 0) + 1
    return True


def _release_websocket_slot(service_id: UUID) -> None:
    key = str(service_id)
    remaining = _OPEN_WEBSOCKETS.get(key, 0) - 1
    if remaining > 0:
        _OPEN_WEBSOCKETS[key] = remaining
    else:
        _OPEN_WEBSOCKETS.pop(key, None)


def _record_rate_limited() -> None:
    from app.core.metrics import record_workspace_service_failure
    record_workspace_service_failure("preview_rate_limited")


def _service_retry_after(service_id: UUID) -> float:
    """Spend one request from the service's bucket; seconds to wait if empty."""
    rate, burst = _service_rate()
    if rate <= 0:
        return 0.0
    return _SERVICE_BUCKETS.take(str(service_id), rate=rate, burst=burst)


def _failure_blocked(client: str) -> float:
    """Seconds the client must wait after too many failed credentials, else 0."""
    rate, burst = _failure_rate()
    if rate <= 0:
        return 0.0
    return _FAILURE_BUCKETS.retry_after(client, rate=rate, burst=burst)


def _note_failure(client: str) -> None:
    rate, burst = _failure_rate()
    if rate > 0:
        _FAILURE_BUCKETS.take(client, rate=rate, burst=burst)


def _too_many_requests(wait: float) -> Response:
    return Response(
        "Too many requests to this preview, slow down.", status_code=429,
        headers={"Retry-After": str(max(1, int(wait) + 1)), "Cache-Control": "no-store"},
        media_type="text/plain",
    )


@dataclass(frozen=True)
class PreviewTarget:
    room_id: UUID
    workspace_id: UUID
    service_id: UUID
    title: str
    runtime_state: str
    last_error: str | None
    port: int


def preview_enabled() -> bool:
    return os.getenv("WORKSPACE_PREVIEW_GATEWAY_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def _ttl() -> int:
    try:
        return max(30, min(3600, int(os.getenv("WORKSPACE_PREVIEW_TICKET_TTL_SEC", "300"))))
    except ValueError:
        return 300


def ticket_ttl_sec() -> int:
    return _ttl()


def _max_body_bytes() -> int:
    try:
        return max(0, min(128 * 1024 * 1024, int(os.getenv("WORKSPACE_PREVIEW_MAX_BODY_BYTES", str(16 * 1024 * 1024)))))
    except ValueError:
        return 16 * 1024 * 1024


def _domain() -> str:
    return os.getenv("WORKSPACE_PREVIEW_DOMAIN", "preview.localhost").strip().lower().rstrip(".")


def _scheme() -> str:
    return os.getenv("WORKSPACE_PREVIEW_SCHEME", "http").strip().lower() or "http"


def _port_suffix() -> str:
    value = os.getenv("WORKSPACE_PREVIEW_PORT", "").strip()
    return f":{value}" if value and value not in {"80", "443"} else ""


def preview_cookie_name() -> str:
    """Use the hardened Host prefix only where its Secure requirement holds."""
    return "__Host-legion-preview" if _scheme() == "https" else "legion-preview"


def preview_cookie_policy() -> tuple[bool, str]:
    """Return cookie flags compatible with an embedded service origin.

    Production preview origins must be HTTPS.  Chromium also treats the
    ``*.localhost`` development namespace as trustworthy, so a Secure,
    SameSite=None cookie can be used there for the cross-site iframe from the
    local frontend.  A Lax cookie is otherwise intentionally retained for an
    explicitly insecure non-local configuration.
    """
    secure = _scheme() == "https" or _domain() == "localhost" or _domain().endswith(".localhost")
    return secure, "none" if secure else "lax"


def service_origin(service_id: UUID | str) -> str:
    return f"{_scheme()}://svc-{service_id}.{_domain()}{_port_suffix()}"


def service_id_from_host(host: str) -> UUID | None:
    hostname = host.split(":", 1)[0].lower().rstrip(".")
    # Default preview.localhost yields svc-<id>.preview.localhost.  A custom
    # domain is matched exactly; accepting a shorter suffix would make an
    # unrelated Host header routable to a workspace service.
    configured_suffix = f".{_domain()}"
    for candidate in (configured_suffix,):
        if hostname.endswith(candidate):
            raw = hostname[: -len(candidate)]
            if raw.startswith("svc-"):
                try:
                    return UUID(raw[4:])
                except ValueError:
                    return None
    return None


def host_matches_origin(host: str, origin: str) -> bool:
    return host.lower().rstrip(".") == origin.split("://", 1)[1].lower().rstrip(".")


def _encode(payload: dict) -> str:
    return jwt.encode(payload, security.SECRET_KEY, algorithm=security.ALGORITHM)


def issue_ticket(*, user_id: UUID, room_id: UUID, service_id: UUID, origin: str) -> str:
    now = int(time.time())
    return _encode({
        "typ": "workspace_preview_ticket", "sub": str(user_id), "room_id": str(room_id),
        "service_id": str(service_id), "origin": origin, "iat": now, "exp": now + _ttl(),
        "jti": str(uuid4()),
    })


def _decode(value: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(value, security.SECRET_KEY, algorithms=[security.ALGORITHM])
    except JWTError as exc:
        raise PreviewAuthorizationError("invalid preview credential") from exc
    if payload.get("typ") != expected_type:
        raise PreviewAuthorizationError("invalid preview credential")
    return payload


async def redeem_ticket(value: str, *, service_id: UUID, origin: str) -> dict:
    payload = _decode(value, "workspace_preview_ticket")
    if payload.get("service_id") != str(service_id) or payload.get("origin") != origin:
        raise PreviewAuthorizationError("invalid preview credential")
    jti = str(payload.get("jti") or "")
    exp = int(payload.get("exp") or 0)
    if not jti or exp <= int(time.time()):
        raise PreviewAuthorizationError("expired preview credential")
    async with _TICKET_LOCK:
        now = int(time.time())
        while _USED_TICKETS and next(iter(_USED_TICKETS.values())) < now:
            _USED_TICKETS.popitem(last=False)
        if jti in _USED_TICKETS:
            raise PreviewAuthorizationError("preview ticket already used")
        _USED_TICKETS[jti] = exp
    return payload


def issue_session(ticket_payload: dict) -> str:
    now = int(time.time())
    return _encode({
        "typ": "workspace_preview_session", "sub": ticket_payload["sub"],
        "room_id": ticket_payload["room_id"], "service_id": ticket_payload["service_id"],
        "origin": ticket_payload["origin"], "iat": now, "exp": now + _ttl(), "jti": str(uuid4()),
    })


def validate_session(value: str, *, service_id: UUID, origin: str) -> dict:
    payload = _decode(value, "workspace_preview_session")
    if payload.get("service_id") != str(service_id) or payload.get("origin") != origin:
        raise PreviewAuthorizationError("invalid preview credential")
    return payload


def _clean_preview_query(query: str) -> str:
    """Keep gateway credentials out of the request forwarded to the app.

    This must preserve every other segment byte-for-byte. Vite (and other
    dev servers) use bare, value-less query flags as routing markers, e.g.
    ``?vue&type=style&index=0&lang.css`` for a CSS-as-JS-module import: the
    trailing ``lang.css`` has no ``=``. Round-tripping through
    ``parse_qsl``/``urlencode`` cannot tell "no value, no equals sign" apart
    from "empty value" and always re-adds ``=`` to every key, which silently
    turns that request into a different, unrecognized one — the app then
    serves a mismatched response (observed: raw CSS bytes labeled as a JS
    module, which crashes the importing script with a syntax error).
    """
    if not query:
        return query
    kept = []
    for segment in query.split("&"):
        if not segment:
            continue
        key = unquote_plus(segment.split("=", 1)[0])
        if key in _INTERNAL_QUERY_PARAMS:
            continue
        kept.append(segment)
    return "&".join(kept)


def _session_from_request(request: Request, origin: str) -> str:
    """Read a short-lived bootstrap session without trusting another origin.

    Cookies can be blocked in an embedded ``localhost`` preview as third-party
    storage.  Browser subresources retain a same-origin Referer, allowing the
    signed, short-lived bootstrap token to authorize their first load.
    """
    direct = request.query_params.get(_SESSION_QUERY_PARAM)
    if direct:
        return direct
    referer = request.headers.get("referer", "")
    parsed = urlsplit(referer)
    if f"{parsed.scheme}://{parsed.netloc}" != origin:
        return ""
    return dict(parse_qsl(parsed.query, keep_blank_values=True)).get(_SESSION_QUERY_PARAM, "")


def _session_from_websocket(websocket: WebSocket, origin: str) -> str:
    direct = websocket.query_params.get(_SESSION_QUERY_PARAM)
    if direct:
        return direct
    referer = websocket.headers.get("referer", "")
    parsed = urlsplit(referer)
    if f"{parsed.scheme}://{parsed.netloc}" != origin:
        return ""
    return dict(parse_qsl(parsed.query, keep_blank_values=True)).get(_SESSION_QUERY_PARAM, "")


def _url_with_session(url: str, session: str) -> str:
    """Append the bootstrap credential while leaving external URLs untouched."""
    if _SESSION_QUERY_PARAM in dict(parse_qsl(urlsplit(url).query, keep_blank_values=True)):
        return url
    before_fragment, separator, fragment = url.partition("#")
    joiner = "&" if "?" in before_fragment else "?"
    result = f"{before_fragment}{joiner}{_SESSION_QUERY_PARAM}={quote(session, safe='')}"
    return f"{result}{separator}{fragment}" if separator else result


def _rewrite_bootstrap_urls(text: str, session: str) -> str:
    """Propagate local bootstrap auth through Vite's root-relative imports."""
    def replace(match: re.Match[str]) -> str:
        quote_mark = match.group("quote")
        return f"{quote_mark}{_url_with_session(match.group('url'), session)}{quote_mark}"
    return _ROOT_URL_IN_TEXT.sub(replace, text)


def _content_type(headers: list[tuple[bytes, bytes]]) -> str:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value.decode("latin-1").split(";", 1)[0].strip().lower()
    return ""


def _preview_response_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    """Keep service pages embeddable in the authenticated preview iframe."""
    forwarded: list[tuple[bytes, bytes]] = []
    for key, value in headers:
        lowered = key.lower()
        if lowered == b"x-frame-options":
            continue
        if lowered == b"content-security-policy":
            directives = [
                directive.strip()
                for directive in value.decode("latin-1").split(";")
                if directive.strip() and not directive.lstrip().lower().startswith("frame-ancestors")
            ]
            if not directives:
                continue
            value = "; ".join(directives).encode("latin-1")
        forwarded.append((key, value))
    return forwarded


async def _rewrite_bootstrap_body(source: AsyncIterator[bytes], session: str) -> AsyncIterator[bytes]:
    """Rewrite small text dev assets; preserve stream behavior for other data."""
    buffered = bytearray()
    async for chunk in source:
        buffered.extend(chunk)
        if len(buffered) > 8 * 1024 * 1024:
            # Vite modules are small.  A larger or binary response must stay
            # unmodified instead of risking memory pressure in the gateway.
            yield bytes(buffered)
            async for rest in source:
                yield rest
            return
    try:
        yield _rewrite_bootstrap_urls(bytes(buffered).decode("utf-8"), session).encode("utf-8")
    except UnicodeDecodeError:
        yield bytes(buffered)


async def resolve_target(db: AsyncSession, service_id: UUID) -> PreviewTarget | None:
    row = await db.execute(
        select(WorkspaceManagedProcess, Workspace)
        .join(Workspace, Workspace.id == WorkspaceManagedProcess.workspace_id)
        .where(WorkspaceManagedProcess.id == service_id)
    )
    pair = row.first()
    if pair is None:
        return None
    service, workspace = pair
    listen = service.listen or {}
    port = listen.get("assigned_port")
    if service.kind not in {"web", "api"} or listen.get("protocol", "http") != "http" or not isinstance(port, int):
        return None
    return PreviewTarget(
        room_id=workspace.room_id, workspace_id=workspace.id, service_id=service.id,
        title=service.title, runtime_state=service.runtime_state, last_error=service.last_error,
        port=port,
    )


def _error_page(target: PreviewTarget | None, error_id: str, message: str) -> HTMLResponse:
    title = html.escape(target.title if target else "Service")
    health = target.runtime_state if target else "unknown"
    return HTMLResponse(
        "<!doctype html><html><body style='margin:0;display:grid;place-items:center;height:100vh;"
        "font-family:system-ui;background:#0d1117;color:#c9d1d9'><main style='max-width:560px;padding:28px'>"
        f"<h2>{title} is unavailable</h2><p>{message}</p><p>State: {health}. Code: {error_id}</p>"
        "<p>Open the safe log tail in the services directory.</p></main></body></html>",
        status_code=502,
    )


def _filtered_request_headers(request: Request, origin: str) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = [("Host", origin.removeprefix("http://").removeprefix("https://"))]
    for key, value in request.headers.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP or lower in {"host", "cookie", "authorization", "content-length"}:
            continue
        headers.append((key, value))
    return headers


async def _raw_request(request: Request, origin: str) -> AsyncIterator[bytes]:
    path = request.url.path or "/"
    query = _clean_preview_query(request.url.query)
    if query:
        path += f"?{query}"
    headers = _filtered_request_headers(request, origin)
    body_length = request.headers.get("content-length")
    max_body = _max_body_bytes()
    if body_length and int(body_length) > max_body:
        raise ValueError("preview request body exceeds policy")
    if body_length:
        headers.append(("Content-Length", body_length))
    elif request.method not in {"GET", "HEAD"}:
        headers.append(("Transfer-Encoding", "chunked"))
    head = f"{request.method} {path} HTTP/1.1\r\n".encode("latin-1")
    head += b"".join(f"{key}: {value}\r\n".encode("latin-1") for key, value in headers) + b"\r\n"
    yield head
    chunked = not body_length and request.method not in {"GET", "HEAD"}
    streamed = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        streamed += len(chunk)
        if streamed > max_body:
            raise ValueError("preview request body exceeds policy")
        if chunked:
            yield f"{len(chunk):X}\r\n".encode("ascii")
            yield chunk + b"\r\n"
        else:
            yield chunk
    if chunked:
        yield b"0\r\n\r\n"


async def _open_preview_relay(
    request: Request,
    service_id: UUID,
    origin: str,
    db: AsyncSession,
    target: PreviewTarget,
) -> tuple[PreviewTarget, wsd.ServiceRelayStream, int, list[tuple[bytes, bytes]], AsyncIterator[bytes]]:
    """Open the upstream relay, tolerating a service restart in flight.

    The service catalog and the supervisor are separate state machines.  A
    browser can therefore receive a ticket while a reconcile/start operation
    is replacing the supervisor runtime.  GET/HEAD requests are safe to
    replay, so refresh the durable target and retry briefly before exposing a
    transient 502 page to the user.  Request bodies are deliberately never
    replayed.
    """
    # A user-facing restart takes a little over three seconds in the
    # workspace supervisor. Keep the preview request alive long enough for
    # the service to become ready instead of returning a transient 502.
    delays = _PREVIEW_RETRY_DELAYS if request.method in {"GET", "HEAD"} else (0.0,)
    if len(delays) > 1:
        # Buffer the (normally empty) GET/HEAD body once: Starlette replays a
        # cached body on every ``stream()`` call, whereas a second read of the
        # raw stream raises "Stream consumed" and would defeat every retry.
        await request.body()
    last_error: Exception | None = None
    readopted = False

    for delay in delays:
        if delay:
            await asyncio.sleep(delay)
            refreshed = await resolve_target(db, service_id)
            if refreshed is None or refreshed.runtime_state != "ready":
                last_error = RuntimeError("service is changing state")
                continue
            target = refreshed

        relay: wsd.ServiceRelayStream | None = None
        try:
            relay = await wsd.open_workspace_service_relay(
                str(target.room_id), str(target.service_id), _raw_request(request, origin)
            )
            status, upstream_headers, body = await _read_response_head(
                relay.response.aiter_raw(), head_request=request.method == "HEAD",
            )
            if status in _TRANSIENT_UPSTREAM_STATUSES and delay != delays[-1]:
                last_error = RuntimeError(f"transient upstream status {status}")
                await relay.aclose()
                continue
            return target, relay, status, upstream_headers, body
        except Exception as exc:
            last_error = exc
            if relay is not None:
                await relay.aclose()
            if not readopted and _is_supervisor_not_ready(exc):
                # workspaced keeps its runtime map in memory, so after a
                # workspaced restart it answers 409 for a service that is
                # still running.  Re-adopt it (or let reconcile demote it to
                # ``error``) before the next attempt instead of failing until
                # someone happens to list the catalog or presses Restart.
                readopted = True
                await _readopt_service_runtime(db, target)

    assert last_error is not None
    raise last_error


def _is_supervisor_not_ready(exc: Exception) -> bool:
    import httpx
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 409


async def _readopt_service_runtime(db: AsyncSession, target: PreviewTarget) -> None:
    from app.core.workspace_services import WorkspaceServiceRegistry, reconcile_service_runtime
    try:
        workspace = await db.get(Workspace, target.workspace_id)
        if workspace is not None:
            await reconcile_service_runtime(target.room_id, workspace, WorkspaceServiceRegistry(db), db)
    except Exception:
        logger.warning("preview gateway could not reconcile service=%s", target.service_id, exc_info=True)


async def _read_response_head(
    chunks: AsyncIterator[bytes], *, head_request: bool = False,
) -> tuple[int, list[tuple[bytes, bytes]], AsyncIterator[bytes]]:
    buffer = bytearray()
    async for chunk in chunks:
        buffer.extend(chunk)
        marker = buffer.find(b"\r\n\r\n")
        if marker >= 0:
            raw_head, first_body = bytes(buffer[:marker]), bytes(buffer[marker + 4:])
            lines = raw_head.split(b"\r\n")
            try:
                status = int(lines[0].split(b" ", 2)[1])
            except (IndexError, ValueError):
                raise ValueError("upstream returned malformed HTTP status")
            headers: list[tuple[bytes, bytes]] = []
            for line in lines[1:]:
                if b":" not in line:
                    continue
                key, value = line.split(b":", 1)
                headers.append((key.strip(), value.strip()))

            is_chunked = any(
                key.lower() == b"transfer-encoding" and b"chunked" in value.lower()
                for key, value in headers
            )
            content_length: int | None = None
            if head_request or status in (204, 304) or 100 <= status < 200:
                # These responses never carry a body (RFC 9110 §6.4.1), and
                # Vite answers every cached module on a reload with a 304 that
                # has neither Content-Length nor chunked framing. Draining it
                # "until close" stalls each module for the upstream keep-alive
                # timeout, and the page never mounts (a white preview).
                is_chunked = False
                content_length = 0
            elif not is_chunked:
                for key, value in headers:
                    if key.lower() == b"content-length":
                        try:
                            content_length = int(value)
                        except ValueError:
                            content_length = None
                        break

            async def body() -> AsyncIterator[bytes]:
                # The tunnel is a raw, potentially keep-alive TCP stream: Vite
                # does not close it after one response (HTTP/1.1 default), so
                # reading until the connection closes would either block for
                # the upstream's idle keep-alive timeout or, worse, start
                # consuming bytes that belong to a later response reusing the
                # same tunnel. A response that declares its own length must be
                # read exactly that far and no further; only a response with
                # neither Content-Length nor chunked framing (which HTTP/1.1
                # allows solely alongside an explicit connection close) falls
                # back to draining until the stream ends.
                remaining = content_length
                if remaining is not None:
                    take = first_body[:remaining]
                    if take:
                        yield take
                    remaining -= len(take)
                    if remaining <= 0:
                        return
                elif first_body:
                    yield first_body
                async for rest in chunks:
                    if remaining is not None:
                        take = rest[:remaining]
                        if take:
                            yield take
                        remaining -= len(take)
                        if remaining <= 0:
                            return
                    else:
                        yield rest
            return status, headers, body()
        if len(buffer) > 64 * 1024:
            break
    raise ValueError("upstream did not return HTTP headers")


async def _wait_for_ready_preview_target(
    db: AsyncSession, service_id: UUID, target: PreviewTarget,
) -> PreviewTarget:
    """Wait through a supervisor restart before rendering a preview error."""
    for delay in _PREVIEW_RETRY_DELAYS[1:]:
        await asyncio.sleep(delay)
        refreshed = await resolve_target(db, service_id)
        if refreshed is not None and refreshed.runtime_state == "ready":
            return refreshed
    return target


async def _dechunk(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Decode a chunked upstream body while retaining streaming semantics."""
    buffer = bytearray()
    expected: int | None = None
    finished = False
    async for chunk in source:
        buffer.extend(chunk)
        while not finished:
            if expected is None:
                marker = buffer.find(b"\r\n")
                if marker < 0:
                    break
                raw_size = bytes(buffer[:marker]).split(b";", 1)[0]
                del buffer[:marker + 2]
                try:
                    expected = int(raw_size, 16)
                except ValueError as exc:
                    raise ValueError("malformed chunked upstream response") from exc
                if expected == 0:
                    finished = True
                    break
            if expected is not None and len(buffer) >= expected + 2:
                if buffer[expected:expected + 2] != b"\r\n":
                    raise ValueError("malformed chunked upstream response")
                if expected:
                    yield bytes(buffer[:expected])
                del buffer[:expected + 2]
                expected = None
            else:
                break
        if finished:
            # The tunnel's underlying connection is a keep-alive stream that
            # Vite does not close on its own; without this, the outer loop
            # would ask the source for one more chunk and block until the
            # upstream's idle timeout, even though the chunked body already
            # ended with its terminating 0-length chunk.
            break
    if not finished and (expected is not None or buffer):
        raise ValueError("truncated chunked upstream response")


@router.api_route("/{path:path}", methods=list(_ALLOWED_METHODS), include_in_schema=False)
async def service_origin_router(path: str, request: Request) -> Response:
    if not preview_enabled():
        return Response(status_code=404)
    service_id = service_id_from_host(request.headers.get("host", ""))
    if service_id is None:
        return Response(status_code=404)
    origin = service_origin(service_id)
    if not host_matches_origin(request.headers.get("host", ""), origin):
        return Response(status_code=404)
    db: AsyncSession = request.state.db if hasattr(request.state, "db") else None  # type: ignore[assignment]
    # Router dependencies are intentionally avoided here: this route is mounted
    # at the application root, after all authenticated API routers.
    if db is None:
        from app.db.session import engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.ext.asyncio import AsyncSession as SAAsyncSession
        async_session = sessionmaker(engine, class_=SAAsyncSession, expire_on_commit=False)
        async with async_session() as opened:
            return await _serve(request, service_id, origin, opened)
    return await _serve(request, service_id, origin, db)


async def _serve(request: Request, service_id: UUID, origin: str, db: AsyncSession) -> Response:
    client = _client_key(request.headers, request.client)
    blocked_for = _failure_blocked(client)
    if blocked_for:
        _record_rate_limited()
        return _too_many_requests(blocked_for)
    target = await resolve_target(db, service_id)
    error_id = str(uuid4())[:8]
    if target is None:
        _note_failure(client)
    if target is None or target.runtime_state != "ready":
        if target is not None and request.method in {"GET", "HEAD"}:
            target = await _wait_for_ready_preview_target(db, service_id, target)
        if target is None or target.runtime_state != "ready":
            return _error_page(target, error_id, "The service is not ready yet or is no longer published.")
    session = request.cookies.get(preview_cookie_name(), "")
    bootstrap_session = ""
    ticket = request.query_params.get("ticket", "")
    if ticket:
        try:
            ticket_payload = await redeem_ticket(ticket, service_id=service_id, origin=origin)
        except PreviewAuthorizationError:
            from app.core.metrics import record_workspace_service_failure
            record_workspace_service_failure("preview_ticket_failure")
            _note_failure(client)
            return Response(status_code=404)
        query = _clean_preview_query(request.url.query)
        session_token = issue_session(ticket_payload)
        # Append rather than round-trip through parse_qsl/urlencode: that
        # would re-add "=" to bare query flags (see _clean_preview_query)
        # for the initial deep-linked path too.
        session_param = f"{_SESSION_QUERY_PARAM}={quote(session_token, safe='')}"
        location_query = f"{query}&{session_param}" if query else session_param
        location = request.url.path + f"?{location_query}"
        # A cookie remains the preferred transport.  Some embedded browsers
        # deny third-party cookie storage between localhost and *.localhost,
        # however.  The location also carries an origin-bound, five-minute
        # signed bootstrap session; it is stripped before reaching the app and
        # is only accepted from this same service origin (including Referer).
        target = json.dumps(location)
        response = HTMLResponse(
            "<!doctype html><meta charset='utf-8'><title>Opening preview…</title>"
            f"<script>location.replace({target})</script>"
            f"<a href={html.escape(location, quote=True)}>Open preview</a>",
            status_code=200,
            headers={"Cache-Control": "no-store"},
        )
        secure, samesite = preview_cookie_policy()
        response.set_cookie(
            preview_cookie_name(), session_token, httponly=True,
            secure=secure, samesite=samesite, path="/",
            max_age=_ttl(),
        )
        return response
    try:
        validate_session(session, service_id=service_id, origin=origin)
    except PreviewAuthorizationError:
        try:
            bootstrap_session = _session_from_request(request, origin)
            validate_session(bootstrap_session, service_id=service_id, origin=origin)
        except PreviewAuthorizationError:
            from app.core.metrics import record_workspace_service_failure
            record_workspace_service_failure("preview_ticket_failure")
            _note_failure(client)
            return Response(status_code=404)
    # Only authorized requests spend the service's budget: an outsider must not
    # be able to drain it and lock the owner out of their own preview.
    throttled_for = _service_retry_after(service_id)
    if throttled_for:
        _record_rate_limited()
        return _too_many_requests(throttled_for)
    try:
        target, relay, status, upstream_headers, body = await _open_preview_relay(
            request, service_id, origin, db, target
        )
    except Exception:
        from app.core.metrics import record_workspace_service_failure
        record_workspace_service_failure("gateway_upstream_error")
        logger.exception("preview gateway upstream error id=%s service=%s method=%s path=%s", error_id, service_id, request.method, request.url.path)
        return _error_page(target, error_id, "Could not establish a connection to the service.")

    chunked = any(key.lower() == b"transfer-encoding" and b"chunked" in value.lower() for key, value in upstream_headers)
    forward_headers = _preview_response_headers([
        (key, value) for key, value in upstream_headers
        if key.lower().decode("latin-1") not in _HOP_BY_HOP
    ])
    source_body = _dechunk(body) if chunked else body
    should_rewrite = bootstrap_session and _content_type(upstream_headers) in _REWRITABLE_CONTENT_TYPES
    if should_rewrite:
        # Rewriting changes byte length.  The preview credential is still
        # removed by _raw_request before the service itself sees a request.
        forward_headers = [
            (key, value) for key, value in forward_headers
            if key.lower() not in {b"content-length", b"content-encoding"}
        ]
        source_body = _rewrite_bootstrap_body(source_body, bootstrap_session)

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in source_body:
                yield chunk
        finally:
            await relay.aclose()
            target.last_error  # keep the target immutable and logs metadata-only

    # Preserve duplicate Set-Cookie headers while removing hop-by-hop headers.
    response = StreamingResponse(stream_body(), status_code=status)
    response.raw_headers = [(key.lower(), value) for key, value in forward_headers]
    logger.info("preview request service=%s method=%s path=%s status=%s error_id=%s", service_id, request.method, request.url.path, status, error_id)
    service = await db.get(WorkspaceManagedProcess, service_id)
    workspace = await db.get(Workspace, target.workspace_id)
    if service is not None:
        from datetime import datetime, timezone
        service.last_accessed_at = datetime.now(timezone.utc)
    if workspace is not None:
        from datetime import datetime, timezone
        workspace.last_activity_at = datetime.now(timezone.utc)
    await db.commit()
    return response


@router.websocket("/{path:path}")
async def service_origin_websocket(path: str, websocket: WebSocket) -> None:
    """Relay application/HMR frames after the origin-scoped cookie check."""
    if not preview_enabled():
        await websocket.close(code=1008)
        return
    service_id = service_id_from_host(websocket.headers.get("host", ""))
    if service_id is None:
        await websocket.close(code=1008)
        return
    origin = service_origin(service_id)
    if not host_matches_origin(websocket.headers.get("host", ""), origin):
        await websocket.close(code=1008)
        return
    client = _client_key(websocket.headers, websocket.client)
    if _failure_blocked(client):
        _record_rate_limited()
        await websocket.close(code=1013)
        return
    try:
        validate_session(websocket.cookies.get(preview_cookie_name(), ""), service_id=service_id, origin=origin)
    except PreviewAuthorizationError:
        try:
            validate_session(_session_from_websocket(websocket, origin), service_id=service_id, origin=origin)
        except PreviewAuthorizationError:
            from app.core.metrics import record_workspace_service_failure
            record_workspace_service_failure("preview_ticket_failure")
            _note_failure(client)
            await websocket.close(code=1008)
            return
    if _service_retry_after(service_id):
        _record_rate_limited()
        await websocket.close(code=1013)
        return
    if not _acquire_websocket_slot(service_id):
        _record_rate_limited()
        await websocket.close(code=1013)
        return
    try:
        await _relay_websocket(websocket, service_id, path)
    finally:
        _release_websocket_slot(service_id)


async def _relay_websocket(websocket: WebSocket, service_id: UUID, path: str) -> None:
    from app.db.session import engine
    from sqlalchemy.ext.asyncio import AsyncSession as SAAsyncSession
    from sqlalchemy.orm import sessionmaker
    factory = sessionmaker(engine, class_=SAAsyncSession, expire_on_commit=False)
    async with factory() as db:
        target = await resolve_target(db, service_id)
    if target is None or target.runtime_state != "ready":
        await websocket.close(code=1013)
        return
    try:
        import websockets
        upstream_url = wsd.workspace_service_relay_ws_url(
            str(target.room_id), str(service_id), websocket.url.path, websocket.url.query
        )
        offered = list(websocket.scope.get("subprotocols", []))
        async with websockets.connect(
            upstream_url, max_size=16 * 1024 * 1024, subprotocols=offered or None,
        ) as upstream:
            await websocket.accept(subprotocol=upstream.subprotocol)

            async def browser_to_upstream() -> None:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("text") is not None:
                        await upstream.send(message["text"])
                    elif message.get("bytes") is not None:
                        await upstream.send(message["bytes"])

            async def upstream_to_browser() -> None:
                async for message in upstream:
                    if isinstance(message, bytes):
                        await websocket.send_bytes(message)
                    else:
                        await websocket.send_text(message)

            left = asyncio.create_task(browser_to_upstream())
            right = asyncio.create_task(upstream_to_browser())
            done, pending = await asyncio.wait({left, right}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
    except WebSocketDisconnect:
        logger.debug("preview WebSocket relay closed service=%s path=%s", service_id, path)
    except Exception:
        from app.core.metrics import record_workspace_service_failure
        record_workspace_service_failure("gateway_upstream_error")
        logger.debug("preview WebSocket relay ended service=%s path=%s", service_id, path, exc_info=True)


def build_preview_app():
    """The gateway as a standalone app: no Hub routes, docs, CORS or metrics."""
    from fastapi import FastAPI

    sub = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    sub.include_router(router)
    return sub


class PreviewHostDispatch:
    """Send ``svc-<id>.<preview domain>`` traffic to the gateway, nothing else.

    Mounted as a catch-all route behind the Hub's own routers, the gateway was
    shadowed on a service origin: the user's ``/docs`` and ``/openapi.json``
    answered with the Hub's, ``/api/v1/*`` and ``/internal/*`` reached the Hub
    API, and the Hub's CORS middleware rewrote the service's headers.  Choosing
    by Host before any routing keeps the two apart.  Outermost on purpose.
    """

    def __init__(self, app) -> None:
        self.app = app
        self.preview = build_preview_app()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] in {"http", "websocket"}:
            host = next((v.decode("latin-1") for k, v in scope["headers"] if k == b"host"), "")
            if service_id_from_host(host) is not None:
                await self.preview(scope, receive, send)
                return
        await self.app(scope, receive, send)
