"""Stage 6: protected fixture, correlation, redirect and SSRF gates."""

from __future__ import annotations

import json
from uuid import uuid4

from fastapi.testclient import TestClient

from app import config
from app.main import app


def _headers(*, correlation: str | None = None) -> dict[str, str]:
    result = {
        "Authorization": f"Bearer {config.GATEWAY_TOKEN}",
        "X-Legion-User-Id": str(uuid4()),
        "X-Legion-Room-Id": str(uuid4()),
        "X-Legion-MCP-Upstream-Authorization": "Bearer fixture-upstream-token",
    }
    if correlation:
        result["X-Legion-Correlation-ID"] = correlation
    return result


def test_required_bearer_fixture_survives_discovery_and_call(monkeypatch):
    """The deterministic fixture uses one bearer for both MCP sessions."""

    import app.mcp_client as mcp_client_module

    requests: list[dict] = []

    class Response:
        headers = {"mcp-session-id": "fixture-stage6"}

        def __init__(self, body: dict, status_code: int = 200):
            self.status_code = status_code
            self.content = json.dumps(body).encode()
            self.text = self.content.decode()

        def json(self):
            return json.loads(self.text)

    class FixtureClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, *, json, headers):
            requests.append({"endpoint": endpoint, "json": json, "headers": headers})
            assert headers["authorization"] == "Bearer fixture-upstream-token"
            if json["method"] == "initialize":
                return Response({"result": {"protocolVersion": "2025-03-26"}})
            if json["method"] == "notifications/initialized":
                return Response({})
            if json["method"] == "tools/list":
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
            return Response({"result": {"content": [{"type": "text", "text": "fixture commit"}]}})

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: FixtureClient())
    correlation = "mcpd_stage6_fixture"
    with TestClient(app) as client:
        discovery = client.post(
            "/v1/tools/discover",
            headers=_headers(correlation=correlation),
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {
                    "url": "https://fixture.example/mcp",
                    "auth": {"mode": "bearer"},
                },
            },
        )
        execute = client.post(
            "/v1/execute",
            headers=_headers(correlation=correlation),
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {
                    "url": "https://fixture.example/mcp",
                    "auth": {"mode": "bearer"},
                },
                "tool": "get_commit",
                "arguments": {"repository": "legion/legion"},
            },
        )

    assert discovery.status_code == 200
    assert discovery.json()["status"] == "verified"
    assert discovery.json()["correlation_id"] == correlation
    assert execute.status_code == 200
    assert execute.json()["content"][0]["text"] == "fixture commit"
    assert [request["json"]["method"] for request in requests] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    assert all(
        request["headers"]["MCP-Protocol-Version"] == "2025-03-26"
        for request in requests
    )
    assert requests[1]["headers"]["mcp-session-id"] == "fixture-stage6"
    assert requests[2]["headers"]["mcp-session-id"] == "fixture-stage6"
    assert requests[4]["headers"]["mcp-session-id"] == "fixture-stage6"
    assert requests[5]["headers"]["mcp-session-id"] == "fixture-stage6"
    assert all("fixture-upstream-token" not in request["json"].__repr__() for request in requests)


def test_redirects_are_not_followed(monkeypatch):
    import app.mcp_client as mcp_client_module

    class Response:
        status_code = 302
        headers = {"location": "https://private.example/mcp"}
        content = b"redirect body"
        text = content.decode()

        def json(self):
            return {"detail": "redirect body"}

    class FixtureClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, *, json, headers):
            return Response()

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: FixtureClient())
    with TestClient(app) as client:
        response = client.post(
            "/v1/tools/discover",
            headers=_headers(),
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {
                    "url": "https://fixture.example/mcp",
                    "auth": {"mode": "bearer"},
                },
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "mcp_http_error"
    assert "private.example" not in response.text


def test_private_endpoint_is_rejected_before_upstream_request():
    with TestClient(app) as client:
        response = client.post(
            "/v1/tools/discover",
            headers=_headers(),
            json={
                "installed_item_id": str(uuid4()),
                "custom_config": {
                    "url": "https://127.0.0.1/mcp",
                    "auth": {"mode": "bearer"},
                },
            },
        )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "mcp_unreachable"
    assert "127.0.0.1" not in response.text
