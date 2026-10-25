from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import config
from app.main import app, market
from app.mcp_client import McpHttpClient


def headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.GATEWAY_TOKEN}",
        "X-Legion-User-Id": str(uuid4()),
        "X-Legion-Room-Id": str(uuid4()),
    }


def test_gateway_requires_service_auth():
    with TestClient(app) as client:
        response = client.get("/v1/tools", params={"installed_item_id": str(uuid4())}, headers={"X-Legion-User-Id": "u"})
    assert response.status_code == 401


def test_discovery_protected_fixture_uses_bearer_for_both_handshake_requests(monkeypatch):
    import app.mcp_client as mcp_client_module

    requests: list[dict] = []

    class Response:
        status_code = 200
        headers = {"mcp-session-id": "fixture-session"}

        def __init__(self, body: dict):
            self.content = json.dumps(body).encode()
            self.text = self.content.decode()

        def json(self):
            return json.loads(self.text)

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, *, json, headers):
            requests.append({"endpoint": endpoint, "json": json, "headers": headers})
            if json["method"] == "initialize":
                return Response({"result": {"protocolVersion": "2025-03-26"}})
            if json["method"] == "notifications/initialized":
                return Response({})
            return Response(
                {
                    "result": {
                        "tools": [
                            {
                                "name": "get_commit",
                                "description": "Read a commit.",
                                "inputSchema": {"type": "object"},
                            }
                        ]
                    }
                }
            )

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: Client())
    upstream = "fixture-upstream-token"
    with TestClient(app) as client:
        response = client.post(
            "/v1/tools/discover",
            headers={**headers(), "X-Legion-MCP-Upstream-Authorization": f"Bearer {upstream}"},
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {"url": "https://fixture.example/mcp", "auth": {"mode": "bearer"}},
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified"
    assert body["tools"][0]["name"] == "get_commit"
    assert body["fingerprint"].startswith("sha256:")
    assert [request["json"]["method"] for request in requests] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
    ]
    assert all(request["headers"]["authorization"] == f"Bearer {upstream}" for request in requests)
    assert all(request["headers"]["MCP-Protocol-Version"] == "2025-03-26" for request in requests)
    assert requests[1]["headers"]["mcp-session-id"] == "fixture-session"
    assert requests[2]["headers"]["mcp-session-id"] == "fixture-session"


def test_discovery_auth_failure_is_structured_and_does_not_send_tools_list(monkeypatch):
    import app.mcp_client as mcp_client_module

    methods: list[str] = []

    class Response:
        status_code = 401
        headers: dict[str, str] = {}
        content = b'{"error":"unauthorized"}'
        text = content.decode()

        def json(self):
            return json.loads(self.text)

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, *, json, headers):
            methods.append(json["method"])
            return Response()

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: Client())
    with TestClient(app) as client:
        response = client.post(
            "/v1/tools/discover",
            headers={**headers(), "X-Legion-MCP-Upstream-Authorization": "Bearer wrong-token"},
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {"url": "https://fixture.example/mcp", "auth": {"mode": "bearer"}},
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "mcp_auth_failed"
    assert methods == ["initialize"]
    assert "unauthorized" not in response.text
    assert "wrong-token" not in response.text


def test_discovery_public_fixture_does_not_forward_upstream_header(monkeypatch):
    import app.mcp_client as mcp_client_module

    request_headers: list[dict[str, str]] = []

    class Response:
        status_code = 200
        headers: dict[str, str] = {}
        content = b'{"result":{"protocolVersion":"2025-03-26","tools":[{"name":"ping","inputSchema":{}}]}}'
        text = content.decode()

        def json(self):
            return json.loads(self.text)

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, *, json, headers):
            request_headers.append(headers)
            return Response()

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: Client())
    with TestClient(app) as client:
        response = client.post(
            "/v1/tools/discover",
            headers={**headers(), "X-Legion-MCP-Upstream-Authorization": "Bearer should-not-forward"},
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {"url": "https://fixture.example/mcp", "auth": {"mode": "none"}},
            },
        )

    assert response.status_code == 200
    assert all("authorization" not in request for request in request_headers)


def test_execute_uses_the_same_upstream_bearer_path(monkeypatch):
    captured: dict[str, object] = {}

    async def call(self, tool, arguments, *, meta=None):
        captured.update(token=self.token, tool=tool, arguments=arguments)
        return {"result": {"ok": True}}

    async def no_quota(*args, **kwargs):
        return None

    monkeypatch.setattr("app.main.McpHttpClient.call", call)
    monkeypatch.setattr("app.main.verify_call_credits", no_quota)
    upstream = "fixture-upstream-token"
    with TestClient(app) as client:
        response = client.post(
            "/v1/execute",
            headers={**headers(), "X-Legion-MCP-Upstream-Authorization": f"Bearer {upstream}"},
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {"url": "https://fixture.example/mcp", "auth": {"mode": "bearer"}},
                "tool": "get_commit",
                "arguments": {"repository": "legion"},
            },
        )

    assert response.status_code == 200
    assert captured == {
        "token": upstream,
        "tool": "get_commit",
        "arguments": {"repository": "legion"},
    }


def test_execute_vision_uses_its_extended_timeout(monkeypatch):
    captured: dict[str, object] = {}

    async def resolve(*args, **kwargs):
        return {
            "type": "mcp",
            "slug": "mcp-vision",
            "isolation": "stateless",
            "internal_endpoint": "http://mcp-vision:3000/mcp",
            "is_active": True,
        }

    class VisionClient:
        def __init__(self, endpoint, *, token=None, timeout_sec=None):
            captured.update(endpoint=endpoint, token=token, timeout_sec=timeout_sec)

        async def call(self, tool, arguments, *, meta=None):
            captured.update(tool=tool, arguments=arguments, meta=meta)
            return {"result": {"ok": True}}

    async def no_quota(*args, **kwargs):
        return None

    monkeypatch.setattr(market, "resolve", resolve)
    monkeypatch.setattr("app.main.McpHttpClient", VisionClient)
    monkeypatch.setattr("app.main.verify_call_credits", no_quota)
    with TestClient(app) as client:
        response = client.post(
            "/v1/execute",
            headers=headers(),
            json={
                "installed_item_id": str(uuid4()),
                "market_slug": "mcp-vision",
                "tool": "describe_image",
                "arguments": {"image_base64": "fixture"},
            },
        )

    assert response.status_code == 200
    assert captured["timeout_sec"] == config.VISION_REQUEST_TIMEOUT_SEC
    assert captured["tool"] == "describe_image"


def test_gateway_rejects_secret_bearing_discovery_and_execute_config_without_echoing_value():
    secret = "fixture-secret-value"
    with TestClient(app) as client:
        discovery = client.post(
            "/v1/tools/discover",
            headers=headers(),
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {
                    "url": "https://fixture.example/mcp",
                    "auth": {"mode": "bearer"},
                    "token": secret,
                },
            },
        )
        execute = client.post(
            "/v1/execute",
            headers=headers(),
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {"url": "https://fixture.example/mcp", "runtime_auth": {"value": secret}},
                "tool": "ping",
            },
        )

    assert discovery.status_code == 422
    assert execute.status_code == 422
    assert secret not in discovery.text
    assert secret not in execute.text


def test_execute_schema_accepts_market_snapshot(monkeypatch):
    async def resolve(*args, **kwargs):
        return {
            "type": "mcp",
            "slug": "mcp-searxng",
            "isolation": "stateless",
            "internal_endpoint": "http://mcp.internal/mcp",
            "is_active": True,
        }

    monkeypatch.setattr(market, "resolve", resolve)
    with TestClient(app) as client:
        response = client.post(
            "/v1/execute",
            headers=headers(),
            json={
                "installed_item_id": str(uuid4()),
                "market_item_id": str(uuid4()),
                "market_slug": "mcp-searxng",
                "tool": "search",
                "arguments": {"query": "Legion"},
            },
        )
    # The endpoint is intentionally not contacted in this schema test; the
    # invalid private endpoint produces the expected proxy-level error rather
    # than a validation error.
    assert response.status_code == 502
    # Structured detail (internal design notes Stage 4): Hub's credit
    # ledger decides settle-vs-refund from `code`, not by parsing prose.
    detail = response.json()["detail"]
    assert detail == {
        "code": "mcp_unreachable",
        "reached_upstream": False,
        "message": "MCP server is unavailable",
    }


def test_tenant_aware_execute_adds_trusted_meta(monkeypatch):
    user_id = str(uuid4())
    room_id = str(uuid4())
    captured: dict[str, object] = {}

    async def resolve(*args, **kwargs):
        return {
            "type": "mcp",
            "slug": "tenant-notes",
            "isolation": "tenant_aware",
            "internal_endpoint": "http://mcp.internal/mcp",
            "is_active": True,
        }

    async def call(self, tool, arguments, *, meta=None):
        captured.update(tool=tool, arguments=arguments, meta=meta)
        return {"result": {"ok": True}}

    async def no_quota(*args, **kwargs):
        return None

    monkeypatch.setattr(market, "resolve", resolve)
    monkeypatch.setattr("app.main.McpHttpClient.call", call)
    monkeypatch.setattr("app.main.verify_call_credits", no_quota)
    with TestClient(app) as client:
        response = client.post(
            "/v1/execute",
            headers={
                "Authorization": f"Bearer {config.GATEWAY_TOKEN}",
                "X-Legion-User-Id": user_id,
                "X-Legion-Room-Id": room_id,
            },
            json={
                "installed_item_id": str(uuid4()),
                "market_slug": "tenant-notes",
                "tool": "save_note",
                "arguments": {"body": "hello"},
            },
        )
    assert response.status_code == 200
    assert captured == {
        "tool": "save_note",
        "arguments": {"body": "hello"},
        "meta": {"user_id": user_id, "room_id": room_id},
    }


def test_catalog_isolation_cannot_be_downgraded_by_request(monkeypatch):
    async def resolve(*args, **kwargs):
        return {
            "type": "mcp",
            "slug": "tenant-notes",
            "isolation": "tenant_aware",
            "internal_endpoint": "http://mcp.internal/mcp",
            "is_active": True,
        }

    monkeypatch.setattr(market, "resolve", resolve)
    with TestClient(app) as client:
        response = client.post(
            "/v1/execute",
            headers=headers(),
            json={
                "installed_item_id": str(uuid4()),
                "market_slug": "tenant-notes",
                "isolation": "stateless",
                "tool": "save_note",
            },
        )
    assert response.status_code == 400


def test_diagnostics_requires_service_auth():
    with TestClient(app) as client:
        response = client.get("/v1/diagnostics")
    assert response.status_code == 401


def test_diagnostics_reports_market_unavailable(monkeypatch):
    async def resolve_by_slug(*args, **kwargs):
        raise HTTPException(status_code=502, detail="Market catalog is unavailable")

    monkeypatch.setattr(market, "resolve_by_slug", resolve_by_slug)
    with TestClient(app) as client:
        response = client.get("/v1/diagnostics", headers={"Authorization": f"Bearer {config.GATEWAY_TOKEN}"})
    assert response.status_code == 200
    body = response.json()
    assert body["market_catalog"] == {
        "status": "error",
        "reason": "market_catalog_unavailable",
        "detail": "Market catalog is unavailable",
    }
    assert body["upstream_search"]["status"] == "skipped"


def test_diagnostics_full_pipeline_ok(monkeypatch):
    async def resolve_by_slug(slug):
        return {
            "slug": slug,
            "isolation": "stateless",
            "internal_endpoint": "http://mcp-searxng:3000/mcp",
            "is_active": True,
        }

    async def fake_health(self, url, **kwargs):
        class Response:
            status_code = 200

        return Response()

    async def fake_tools(self):
        return [{"name": "searxng_web_search", "description": "", "inputSchema": {}}]

    async def fake_call(self, tool, arguments, *, meta=None):
        return {"result": {"isError": False, "content": [{"type": "text", "text": "3 results"}]}}

    monkeypatch.setattr(market, "resolve_by_slug", resolve_by_slug)
    monkeypatch.setattr("app.main.httpx.AsyncClient.get", fake_health)
    monkeypatch.setattr("app.main.McpHttpClient.tools", fake_tools)
    monkeypatch.setattr("app.main.McpHttpClient.call", fake_call)
    with TestClient(app) as client:
        response = client.get(
            "/v1/diagnostics",
            params={"slug": "mcp-searxng"},
            headers={"Authorization": f"Bearer {config.GATEWAY_TOKEN}"},
        )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "market_catalog": {"status": "ok"},
        "gateway": {"status": "ok"},
        "mcp_transport": {"status": "ok"},
        "tools_list": {"status": "ok", "tools": ["searxng_web_search"]},
        "upstream_search": {"status": "ok"},
    }


def test_diagnostics_classifies_upstream_not_configured(monkeypatch):
    async def resolve_by_slug(slug):
        return {
            "slug": slug,
            "isolation": "stateless",
            "internal_endpoint": "http://mcp-searxng:3000/mcp",
            "is_active": True,
        }

    async def fake_health(self, url, **kwargs):
        class Response:
            status_code = 200

        return Response()

    async def fake_tools(self):
        return [{"name": "searxng_web_search", "description": "", "inputSchema": {}}]

    async def fake_call(self, tool, arguments, *, meta=None):
        return {
            "result": {
                "isError": True,
                "content": [{"type": "text", "text": "\U0001f527 Configuration Error: SEARXNG_URL is not set"}],
            }
        }

    monkeypatch.setattr(market, "resolve_by_slug", resolve_by_slug)
    monkeypatch.setattr("app.main.httpx.AsyncClient.get", fake_health)
    monkeypatch.setattr("app.main.McpHttpClient.tools", fake_tools)
    monkeypatch.setattr("app.main.McpHttpClient.call", fake_call)
    with TestClient(app) as client:
        response = client.get(
            "/v1/diagnostics",
            headers={"Authorization": f"Bearer {config.GATEWAY_TOKEN}"},
        )
    assert response.json()["upstream_search"]["reason"] == "upstream_not_configured"


def test_mcp_client_rpc_error_codes_split_bad_request_vs_upstream(monkeypatch):
    """-32601/-32602 are our malformed request (refund); any other JSON-RPC
    error code is the upstream tool genuinely failing (settle) — Stage 4's
    outcome table in internal design notes."""
    import asyncio

    import app.mcp_client as mcp_client_module

    def make_client(rpc_code: int):
        class Response:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self):
                body = json.dumps({"error": {"code": rpc_code, "message": "boom"}}).encode()
                self.content = body
                self.text = body.decode()

            def json(self):
                return json.loads(self.text)

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def post(self, endpoint, *, json, headers):
                return Response()

        return Client()

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: make_client(-32601))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(McpHttpClient("http://mcp.internal/mcp").call("save_note", {}))
    assert exc_info.value.detail["code"] == "mcp_bad_request"
    assert exc_info.value.detail["reached_upstream"] is True

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: make_client(-32000))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(McpHttpClient("http://mcp.internal/mcp").call("save_note", {}))
    assert exc_info.value.detail["code"] == "mcp_rpc_error"
    assert exc_info.value.detail["reached_upstream"] is True


def test_mcp_client_places_meta_inside_tools_call_params(monkeypatch):
    import app.mcp_client as mcp_client_module

    requests: list[dict] = []

    class Response:
        status_code = 200
        headers: dict[str, str] = {}
        content = b'{"result": {}}'
        text = content.decode()

        def json(self):
            return {"result": {}}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, *, json, headers):
            requests.append(json)
            return Response()

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: Client())

    import asyncio

    asyncio.run(
        McpHttpClient("http://mcp.internal/mcp").call(
            "save_note", {"body": "hello"}, meta={"user_id": "u", "room_id": "r"}
        )
    )
    assert requests[2]["params"]["_meta"] == {"user_id": "u", "room_id": "r"}


def test_decode_response_accepts_body_between_the_old_and_new_ceiling():
    """Design Vision Toolkit plan, Stage 3.1: MAX_RESPONSE_BYTES 1 MiB -> 8 MiB.

    A body that the old per-slug-less 1 MiB limit would have rejected must
    now pass through unchanged for every MCP, not just mcp-vision.
    """
    import httpx

    from app import mcp_client as mcp_client_module

    body = {"result": {"data": "x" * (6 * 1024 * 1024)}}
    payload = json.dumps(body).encode()
    assert 1024 * 1024 < len(payload) <= config.MAX_RESPONSE_BYTES

    response = httpx.Response(200, content=payload)
    assert mcp_client_module._decode_response(response) == body


def test_decode_response_still_rejects_bodies_over_the_configured_ceiling(monkeypatch):
    import httpx

    from app import mcp_client as mcp_client_module

    monkeypatch.setattr(config, "MAX_RESPONSE_BYTES", 1024)
    response = httpx.Response(200, content=b"x" * 2048)

    with pytest.raises(HTTPException) as exc_info:
        mcp_client_module._decode_response(response)
    assert exc_info.value.detail["code"] == "mcp_bad_response"


def test_gateway_without_configured_tokens_refuses_everything(monkeypatch):
    """No built-in default token: an unset token must not become a way in."""
    from app import security

    monkeypatch.setattr(security, "GATEWAY_TOKEN", "")
    monkeypatch.setattr(security, "GATEWAY_ADMIN_TOKEN", "")
    for sent in (None, "", "anything"):
        with pytest.raises(HTTPException) as exc:
            security.require_gateway_admin_token(authorization=None, x_mcp_gateway_admin_token=sent)
        assert exc.value.status_code == 401
        with pytest.raises(HTTPException) as exc:
            security.require_gateway_token(authorization=None, x_mcp_gateway_token=sent)
        assert exc.value.status_code == 401
