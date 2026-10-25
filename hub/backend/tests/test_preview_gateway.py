"""Contract tests for browser-services stage 3 preview gateway."""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from starlette.requests import Request

from app.core import preview_gateway as gateway


def test_service_origin_is_exact_and_host_isolated(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_DOMAIN", "preview.localhost")
    monkeypatch.setenv("WORKSPACE_PREVIEW_SCHEME", "http")
    monkeypatch.setenv("WORKSPACE_PREVIEW_PORT", "8001")
    service_id = uuid4()
    origin = gateway.service_origin(service_id)

    assert origin == f"http://svc-{service_id}.preview.localhost:8001"
    assert gateway.service_id_from_host(f"svc-{service_id}.preview.localhost:8001") == service_id
    assert gateway.service_id_from_host(f"svc-{service_id}.other.localhost:8001") is None
    assert gateway.preview_cookie_name() == "legion-preview"
    assert gateway.preview_cookie_policy() == (True, "none")
    monkeypatch.setenv("WORKSPACE_PREVIEW_SCHEME", "https")
    assert gateway.preview_cookie_name() == "__Host-legion-preview"
    assert gateway.preview_cookie_policy() == (True, "none")


def test_preview_response_headers_allow_iframed_service_pages():
    headers = gateway._preview_response_headers([
        (b"x-frame-options", b"DENY"),
        (b"content-security-policy", b"default-src 'self'; frame-ancestors 'none'; script-src 'self'"),
        (b"content-type", b"text/html"),
    ])

    assert (b"x-frame-options", b"DENY") not in headers
    assert (b"content-security-policy", b"default-src 'self'; script-src 'self'") in headers
    assert (b"content-type", b"text/html") in headers


@pytest.mark.asyncio
async def test_ticket_is_bound_to_origin_and_single_use(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_TICKET_TTL_SEC", "300")
    gateway._USED_TICKETS.clear()
    user_id, room_id, service_id = uuid4(), uuid4(), uuid4()
    origin = gateway.service_origin(service_id)
    ticket = gateway.issue_ticket(user_id=user_id, room_id=room_id, service_id=service_id, origin=origin)

    payload = await gateway.redeem_ticket(ticket, service_id=service_id, origin=origin)
    session = gateway.issue_session(payload)
    assert gateway.validate_session(session, service_id=service_id, origin=origin)["sub"] == str(user_id)
    with pytest.raises(gateway.PreviewAuthorizationError):
        await gateway.redeem_ticket(ticket, service_id=service_id, origin=origin)
    with pytest.raises(gateway.PreviewAuthorizationError):
        gateway.validate_session(session, service_id=uuid4(), origin=origin)


@pytest.mark.asyncio
async def test_ticket_redemption_bootstraps_session_and_sets_preview_cookie(monkeypatch):
    """The bootstrap URL remains usable when embedded cookies are denied."""
    monkeypatch.setenv("WORKSPACE_PREVIEW_TICKET_TTL_SEC", "300")
    gateway._USED_TICKETS.clear()
    user_id, room_id, service_id = uuid4(), uuid4(), uuid4()
    origin = gateway.service_origin(service_id)
    ticket = gateway.issue_ticket(
        user_id=user_id, room_id=room_id, service_id=service_id, origin=origin,
    )
    target = gateway.PreviewTarget(
        room_id=room_id,
        workspace_id=uuid4(),
        service_id=service_id,
        title="Preview",
        runtime_state="ready",
        last_error=None,
        port=3000,
    )

    async def resolve_target(_db, _service_id):
        return target

    monkeypatch.setattr(gateway, "resolve_target", resolve_target)
    host = origin.removeprefix("http://")
    request = Request({
        "type": "http", "method": "GET", "scheme": "http",
        "path": "/", "raw_path": b"/", "query_string": f"ticket={ticket}".encode(),
        "headers": [(b"host", host.encode())], "server": ("testserver", 80),
    })

    response = await gateway._serve(request, service_id, origin, object())

    assert response.status_code == 200
    assert gateway._SESSION_QUERY_PARAM in response.body.decode()
    assert f"{gateway.preview_cookie_name()}=" in response.headers["set-cookie"]


async def _empty_body():
    return {"type": "http.request", "body": b"", "more_body": False}


@pytest.mark.asyncio
async def test_get_preview_retries_when_supervisor_restarts_service(monkeypatch):
    service_id = uuid4()
    origin = gateway.service_origin(service_id)
    target = gateway.PreviewTarget(
        room_id=uuid4(), workspace_id=uuid4(), service_id=service_id,
        title="Preview", runtime_state="ready", last_error=None, port=3000,
    )
    request = Request({
        "type": "http", "method": "GET", "scheme": "http", "path": "/",
        "raw_path": b"/", "query_string": b"",
        "headers": [(b"host", origin.removeprefix("http://").encode())],
        "server": ("testserver", 80),
    }, receive=_empty_body)
    calls = 0

    class FakeResponse:
        def aiter_raw(self):
            async def chunks():
                yield b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nok"
            return chunks()

    class FakeRelay:
        response = FakeResponse()

        async def aclose(self):
            return None

    async def open_relay(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("service runtime is restarting")
        return FakeRelay()

    async def resolve(_db, _service_id):
        return target

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(gateway.wsd, "open_workspace_service_relay", open_relay)
    monkeypatch.setattr(gateway, "resolve_target", resolve)
    monkeypatch.setattr(gateway.asyncio, "sleep", no_sleep)

    resolved, _relay, status, _headers, body = await gateway._open_preview_relay(
        request, service_id, origin, object(), target,
    )

    assert resolved is target
    assert status == 200
    assert b"".join([part async for part in body]) == b"ok"
    assert calls == 2


@pytest.mark.asyncio
async def test_preview_readopts_service_after_workspaced_restart(monkeypatch):
    """A 409 from workspaced's empty runtime map triggers one reconcile."""
    import httpx
    from app.core import workspace_services

    service_id = uuid4()
    origin = gateway.service_origin(service_id)
    target = gateway.PreviewTarget(
        room_id=uuid4(), workspace_id=uuid4(), service_id=service_id,
        title="Preview", runtime_state="ready", last_error=None, port=3000,
    )
    received = False

    async def receive():
        # A real ASGI receive channel: the request body is delivered once.
        nonlocal received
        assert not received, "request stream read twice"
        received = True
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request({
        "type": "http", "method": "GET", "scheme": "http", "path": "/",
        "raw_path": b"/", "query_string": b"",
        "headers": [(b"host", origin.removeprefix("http://").encode())],
        "server": ("testserver", 80),
    }, receive=receive)
    workspace = object()
    adopted: list[tuple] = []

    class FakeResponse:
        def aiter_raw(self):
            async def chunks():
                yield b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nok"
            return chunks()

    class FakeRelay:
        response = FakeResponse()

        async def aclose(self):
            return None

    async def open_relay(_room_id, _service_id, content):
        # Like httpx, send the whole request before the status is known.
        assert [chunk async for chunk in content][0].startswith(b"GET / HTTP/1.1")
        if not adopted:
            response = httpx.Response(409, request=httpx.Request("POST", "http://workspaced/relay"))
            raise httpx.HTTPStatusError("not ready", request=response.request, response=response)
        return FakeRelay()

    class FakeDb:
        async def get(self, _model, workspace_id):
            assert workspace_id == target.workspace_id
            return workspace

    async def reconcile(room_id, ws, _registry, _db):
        adopted.append((room_id, ws))

    async def resolve(_db, _service_id):
        return target

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(gateway.wsd, "open_workspace_service_relay", open_relay)
    monkeypatch.setattr(workspace_services, "reconcile_service_runtime", reconcile)
    monkeypatch.setattr(gateway, "resolve_target", resolve)
    monkeypatch.setattr(gateway.asyncio, "sleep", no_sleep)

    _target, _relay, status, _headers, body = await gateway._open_preview_relay(
        request, service_id, origin, FakeDb(), target,
    )

    assert status == 200
    assert b"".join([part async for part in body]) == b"ok"
    assert adopted == [(target.room_id, workspace)]


@pytest.mark.asyncio
async def test_preview_waits_for_service_to_become_ready(monkeypatch):
    service_id = uuid4()
    starting = gateway.PreviewTarget(
        room_id=uuid4(), workspace_id=uuid4(), service_id=service_id,
        title="Preview", runtime_state="starting", last_error=None, port=3000,
    )
    ready = gateway.PreviewTarget(
        room_id=starting.room_id, workspace_id=starting.workspace_id,
        service_id=service_id, title="Preview", runtime_state="ready",
        last_error=None, port=3000,
    )
    calls = 0

    async def resolve(_db, _service_id):
        nonlocal calls
        calls += 1
        return starting if calls == 1 else ready

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(gateway, "resolve_target", resolve)
    monkeypatch.setattr(gateway.asyncio, "sleep", no_sleep)

    resolved = await gateway._wait_for_ready_preview_target(object(), service_id, starting)

    assert resolved is ready
    assert calls == 2


def test_session_can_be_recovered_from_a_same_origin_referer(monkeypatch):
    service_id = uuid4()
    origin = gateway.service_origin(service_id)
    token = gateway.issue_session({
        "sub": str(uuid4()), "room_id": str(uuid4()), "service_id": str(service_id), "origin": origin,
    })
    request = Request({
        "type": "http", "method": "GET", "scheme": "http", "path": "/src/main.js",
        "raw_path": b"/src/main.js", "query_string": b"",
        "headers": [(b"host", origin.removeprefix("http://").encode()),
                    (b"referer", f"{origin}/?{gateway._SESSION_QUERY_PARAM}={token}".encode())],
        "server": ("testserver", 80),
    })

    assert gateway._session_from_request(request, origin) == token
    assert gateway._clean_preview_query(f"x=1&{gateway._SESSION_QUERY_PARAM}=secret&ticket=nope") == "x=1"


def test_bootstrap_session_is_propagated_through_vite_import_urls():
    rewritten = gateway._rewrite_bootstrap_urls(
        'import Vue from "/node_modules/vue.js?v=1"; const image = \'/assets/a.png\';', "signed-token",
    )

    assert f"/node_modules/vue.js?v=1&{gateway._SESSION_QUERY_PARAM}=signed-token" in rewritten
    assert f"/assets/a.png?{gateway._SESSION_QUERY_PARAM}=signed-token" in rewritten
    assert gateway._url_with_session(f"/x?{gateway._SESSION_QUERY_PARAM}=kept", "new") == f"/x?{gateway._SESSION_QUERY_PARAM}=kept"


@pytest.mark.asyncio
async def test_response_head_preserves_status_headers_and_binary_body():
    async def chunks():
        yield b"HTTP/1.1 201 Created\r\nContent-Type: image/png\r\nSet-Cookie: a=b\r\n"
        yield b"Location: /next\r\n\r\n\x89PNG\x00"
        yield b"\xffdata"

    status, headers, body = await gateway._read_response_head(chunks())
    assert status == 201
    assert (b"Set-Cookie", b"a=b") in headers
    assert b"".join([part async for part in body]) == b"\x89PNG\x00\xffdata"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_line,head_request", [
    (b"HTTP/1.1 304 Not Modified", False),
    (b"HTTP/1.1 204 No Content", False),
    (b"HTTP/1.1 200 OK", True),
])
async def test_bodiless_response_ends_without_waiting_for_upstream_close(status_line, head_request):
    upstream_closed = False

    async def chunks():
        nonlocal upstream_closed
        yield status_line + b"\r\nETag: W/\"1\"\r\n\r\n"
        # A keep-alive upstream sends nothing more until its idle timeout.
        await asyncio.sleep(30)
        upstream_closed = True
        yield b"next-response"

    _status, _headers, body = await gateway._read_response_head(chunks(), head_request=head_request)
    assert await asyncio.wait_for(_collect(body), timeout=1) == b""
    assert not upstream_closed


async def _collect(body):
    return b"".join([part async for part in body])


@pytest.mark.asyncio
async def test_chunked_response_is_decoded_without_buffering():
    async def chunks():
        yield b"4\r\nWiki\r\n5\r\npedia\r\n"
        yield b"0\r\n\r\n"

    assert b"".join([part async for part in gateway._dechunk(chunks())]) == b"Wikipedia"


# ── rate limiting ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _fresh_rate_limits(monkeypatch):
    for name in (
        "WORKSPACE_PREVIEW_RATE_PER_SEC", "WORKSPACE_PREVIEW_RATE_BURST",
        "WORKSPACE_PREVIEW_FAIL_RATE_PER_SEC", "WORKSPACE_PREVIEW_FAIL_BURST",
        "WORKSPACE_PREVIEW_TRUST_PROXY_HEADERS", "WORKSPACE_PREVIEW_MAX_WEBSOCKETS",
    ):
        monkeypatch.delenv(name, raising=False)
    gateway._SERVICE_BUCKETS.clear()
    gateway._FAILURE_BUCKETS.clear()
    gateway._OPEN_WEBSOCKETS.clear()
    yield
    gateway._SERVICE_BUCKETS.clear()
    gateway._FAILURE_BUCKETS.clear()
    gateway._OPEN_WEBSOCKETS.clear()


def _preview_request(service_id, *, session="", client=("203.0.113.7", 5000)):
    origin = gateway.service_origin(service_id)
    headers = [(b"host", origin.removeprefix("http://").encode())]
    if session:
        headers.append((b"cookie", f"{gateway.preview_cookie_name()}={session}".encode()))
    return origin, Request({
        "type": "http", "method": "GET", "scheme": "http", "path": "/",
        "raw_path": b"/", "query_string": b"", "headers": headers,
        "server": ("testserver", 80), "client": client,
    })


def _ready_target(service_id):
    return gateway.PreviewTarget(
        room_id=uuid4(), workspace_id=uuid4(), service_id=service_id,
        title="Preview", runtime_state="ready", last_error=None, port=3000,
    )


def _session_for(service_id):
    return gateway.issue_session({
        "sub": str(uuid4()), "room_id": str(uuid4()),
        "service_id": str(service_id), "origin": gateway.service_origin(service_id),
    })


def _patch_gateway(monkeypatch, resolve):
    async def open_relay(*_args, **_kwargs):
        raise RuntimeError("upstream is down")

    monkeypatch.setattr(gateway, "resolve_target", resolve)
    monkeypatch.setattr(gateway, "_open_preview_relay", open_relay)


def test_token_bucket_spends_burst_then_refills(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(gateway.time, "monotonic", lambda: clock[0])
    buckets = gateway._TokenBuckets()

    assert [buckets.take("k", rate=2.0, burst=3.0) for _ in range(3)] == [0.0, 0.0, 0.0]
    assert buckets.take("k", rate=2.0, burst=3.0) == pytest.approx(0.5)
    assert buckets.take("other", rate=2.0, burst=3.0) == 0.0
    clock[0] += 0.5
    assert buckets.take("k", rate=2.0, burst=3.0) == 0.0
    clock[0] += 60
    assert buckets.retry_after("k", rate=2.0, burst=3.0) == 0.0


def test_token_bucket_memory_is_bounded():
    buckets = gateway._TokenBuckets(max_keys=3)
    for key in "abcde":
        buckets.take(key, rate=1.0, burst=1.0)
    assert len(buckets._buckets) == 3


@pytest.mark.asyncio
async def test_authorized_preview_is_throttled_once_the_service_budget_is_spent(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_RATE_PER_SEC", "0.01")
    monkeypatch.setenv("WORKSPACE_PREVIEW_RATE_BURST", "2")
    service_id = uuid4()
    target = _ready_target(service_id)
    session = _session_for(service_id)

    async def resolve(_db, _service_id):
        return target

    _patch_gateway(monkeypatch, resolve)

    statuses = []
    for _ in range(4):
        origin, request = _preview_request(service_id, session=session)
        response = await gateway._serve(request, service_id, origin, object())
        statuses.append(response.status_code)

    assert statuses == [502, 502, 429, 429]
    assert int(response.headers["retry-after"]) >= 1


@pytest.mark.asyncio
async def test_rate_limit_is_per_service(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_RATE_PER_SEC", "0.01")
    monkeypatch.setenv("WORKSPACE_PREVIEW_RATE_BURST", "1")
    first, second = uuid4(), uuid4()
    targets = {first: _ready_target(first), second: _ready_target(second)}

    async def resolve(_db, service_id):
        return targets[service_id]

    _patch_gateway(monkeypatch, resolve)

    async def hit(service_id):
        origin, request = _preview_request(service_id, session=_session_for(service_id))
        return (await gateway._serve(request, service_id, origin, object())).status_code

    assert [await hit(first), await hit(first), await hit(second)] == [502, 429, 502]


@pytest.mark.asyncio
async def test_rate_zero_disables_the_service_limit(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_RATE_PER_SEC", "0")
    monkeypatch.setenv("WORKSPACE_PREVIEW_RATE_BURST", "1")
    service_id = uuid4()
    target = _ready_target(service_id)
    session = _session_for(service_id)

    async def resolve(_db, _service_id):
        return target

    _patch_gateway(monkeypatch, resolve)

    for _ in range(5):
        origin, request = _preview_request(service_id, session=session)
        assert (await gateway._serve(request, service_id, origin, object())).status_code == 502


@pytest.mark.asyncio
async def test_probing_client_is_blocked_before_the_lookup_and_cannot_starve_the_owner(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_FAIL_RATE_PER_SEC", "0.01")
    monkeypatch.setenv("WORKSPACE_PREVIEW_FAIL_BURST", "3")
    monkeypatch.setenv("WORKSPACE_PREVIEW_RATE_PER_SEC", "0.01")
    monkeypatch.setenv("WORKSPACE_PREVIEW_RATE_BURST", "1")
    service_id = uuid4()
    target = _ready_target(service_id)
    lookups = 0

    async def resolve(_db, _service_id):
        nonlocal lookups
        lookups += 1
        return target

    _patch_gateway(monkeypatch, resolve)

    statuses = []
    for _ in range(5):
        origin, request = _preview_request(service_id, client=("198.51.100.9", 4444))
        statuses.append((await gateway._serve(request, service_id, origin, object())).status_code)

    assert statuses == [404, 404, 404, 429, 429]
    assert lookups == 3  # the blocked requests never reached the lookup

    # The outsider's failures did not spend the service budget (burst is 1).
    origin, request = _preview_request(
        service_id, session=_session_for(service_id), client=("203.0.113.7", 1),
    )
    assert (await gateway._serve(request, service_id, origin, object())).status_code == 502


@pytest.mark.asyncio
async def test_unknown_service_ids_count_as_failures(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_FAIL_RATE_PER_SEC", "0.01")
    monkeypatch.setenv("WORKSPACE_PREVIEW_FAIL_BURST", "2")

    async def resolve(_db, _service_id):
        return None

    monkeypatch.setattr(gateway, "resolve_target", resolve)

    statuses = []
    for _ in range(4):
        service_id = uuid4()
        origin, request = _preview_request(service_id)
        statuses.append((await gateway._serve(request, service_id, origin, object())).status_code)

    assert statuses == [502, 502, 429, 429]


def test_client_key_honours_forwarded_headers_only_when_trusted(monkeypatch):
    class Client:
        host = "10.0.0.2"

    headers = {"x-forwarded-for": "198.51.100.9, 10.0.0.2"}
    assert gateway._client_key(headers, Client()) == "10.0.0.2"
    monkeypatch.setenv("WORKSPACE_PREVIEW_TRUST_PROXY_HEADERS", "true")
    assert gateway._client_key(headers, Client()) == "198.51.100.9"
    assert gateway._client_key({"cf-connecting-ip": "192.0.2.1", **headers}, Client()) == "192.0.2.1"
    assert gateway._client_key({}, Client()) == "10.0.0.2"
    assert gateway._client_key({}, None) == "unknown"


class _FakeWebSocket:
    def __init__(self, service_id, session, client="203.0.113.7"):
        origin = gateway.service_origin(service_id)
        self.headers = {"host": origin.removeprefix("http://")}
        self.cookies = {gateway.preview_cookie_name(): session}
        self.query_params = {}
        self.client = type("C", (), {"host": client})()
        self.closed = None

    async def close(self, code=1000):
        self.closed = code


@pytest.mark.asyncio
async def test_websocket_cap_is_per_service_and_slots_are_released(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_MAX_WEBSOCKETS", "2")
    gateway._OPEN_WEBSOCKETS.clear()
    first, second = uuid4(), uuid4()
    release = asyncio.Event()
    relayed = []

    async def relay(_ws, service_id, _path):
        relayed.append(service_id)
        await release.wait()

    monkeypatch.setattr(gateway, "_relay_websocket", relay)

    def connect(service_id):
        ws = _FakeWebSocket(service_id, _session_for(service_id))
        return ws, asyncio.create_task(gateway.service_origin_websocket("", ws))

    held = [connect(first), connect(first)]
    await asyncio.sleep(0)
    extra_ws, extra_task = connect(first)
    await extra_task
    other_ws, other_task = connect(second)
    await asyncio.sleep(0)

    assert extra_ws.closed == 1013
    assert other_ws.closed is None
    assert relayed.count(first) == 2 and relayed.count(second) == 1

    release.set()
    await asyncio.gather(*(task for _ws, task in [*held, (other_ws, other_task)]))
    assert gateway._OPEN_WEBSOCKETS == {}

    # A freed slot is reusable.
    release.clear()
    again_ws, again_task = connect(first)
    await asyncio.sleep(0)
    assert again_ws.closed is None
    release.set()
    await again_task


@pytest.mark.asyncio
async def test_websocket_slot_is_released_when_the_relay_fails(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_MAX_WEBSOCKETS", "1")
    gateway._OPEN_WEBSOCKETS.clear()
    service_id = uuid4()

    async def relay(*_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(gateway, "_relay_websocket", relay)
    for _ in range(3):
        ws = _FakeWebSocket(service_id, _session_for(service_id))
        with pytest.raises(RuntimeError):
            await gateway.service_origin_websocket("", ws)
    assert gateway._OPEN_WEBSOCKETS == {}


def test_websocket_cap_zero_disables(monkeypatch):
    monkeypatch.setenv("WORKSPACE_PREVIEW_MAX_WEBSOCKETS", "0")
    gateway._OPEN_WEBSOCKETS.clear()
    service_id = uuid4()
    assert all(gateway._acquire_websocket_slot(service_id) for _ in range(100))
    for _ in range(100):
        gateway._release_websocket_slot(service_id)
    assert gateway._OPEN_WEBSOCKETS == {}


def _hub_behind_dispatch():
    from fastapi import FastAPI

    hub = FastAPI()

    @hub.get("/api/v1/ping")
    async def ping():
        return {"who": "hub"}

    return gateway.PreviewHostDispatch(hub)


@pytest.mark.asyncio
async def test_service_host_never_reaches_hub_routes(monkeypatch):
    import httpx

    monkeypatch.setenv("WORKSPACE_PREVIEW_GATEWAY_ENABLED", "true")
    monkeypatch.setenv("WORKSPACE_PREVIEW_FAIL_RATE_PER_SEC", "0")
    host = gateway.service_origin(uuid4()).removeprefix("http://")

    async def resolve(_db, _service_id):
        return None

    monkeypatch.setattr(gateway, "resolve_target", resolve)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_hub_behind_dispatch()), base_url="http://hub") as client:
        for path in ("/docs", "/openapi.json", "/api/v1/ping"):
            reply = await client.get(path, headers={"host": host})
            # The gateway's own answer for a service that does not exist.
            assert reply.status_code == 502 and "unavailable" in reply.text, path

        # The Hub's own host still reaches the Hub.
        own = await client.get("/api/v1/ping", headers={"host": "hub.example.com"})
        assert own.json() == {"who": "hub"}
        assert (await client.get("/docs", headers={"host": "hub.example.com"})).status_code == 200


@pytest.mark.asyncio
async def test_service_host_is_closed_while_the_gateway_is_disabled(monkeypatch):
    import httpx

    monkeypatch.setenv("WORKSPACE_PREVIEW_GATEWAY_ENABLED", "false")
    host = gateway.service_origin(uuid4()).removeprefix("http://")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=_hub_behind_dispatch()), base_url="http://hub") as client:
        for path in ("/docs", "/api/v1/ping"):
            assert (await client.get(path, headers={"host": host})).status_code == 404
