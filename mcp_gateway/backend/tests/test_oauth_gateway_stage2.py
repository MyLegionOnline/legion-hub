"""internal design notes Stage 2 — Gateway is the only hop to a provider's OAuth endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import config, oauth_client
from app.main import app


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.GATEWAY_TOKEN}"}


class _Response:
    def __init__(self, body, status_code: int = 200):
        self.status_code = status_code
        self.content = json.dumps(body).encode() if body is not None else b""
        self.text = self.content.decode()

    def json(self):
        if not self.text:
            raise ValueError("no body")
        return json.loads(self.text)


def _fake_client(monkeypatch, *, get=None, post=None):
    """Patch the ``httpx`` module ``oauth_client`` itself imported."""

    requested: dict[str, list] = {"get": [], "post": []}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, *, headers=None):
            requested["get"].append(url)
            if get is None:
                return _Response(None, status_code=404)
            return get(url)

        async def post(self, url, *, json=None, data=None, headers=None):
            requested["post"].append((url, json if json is not None else data))
            if post is None:
                return _Response(None, status_code=404)
            return post(url, json if json is not None else data)

    monkeypatch.setattr(oauth_client.httpx, "AsyncClient", lambda **kwargs: Client())
    return requested


@pytest.fixture(autouse=True)
def _no_dns_wait(monkeypatch):
    """Fixture hostnames don't resolve; skip the rebind lookup like test_oauth_probe.py."""

    monkeypatch.setattr(config, "MCP_GATEWAY_DNS_REBIND_PROTECTION", False)


def test_metadata_route_requires_the_gateway_token():
    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/metadata",
            json={"kind": "authorization_server", "issuer": "https://provider.example"},
        )
    assert response.status_code == 401


def test_metadata_route_reads_the_authorization_server_document(monkeypatch):
    asm = {"issuer": "https://provider.example", "registration_endpoint": "https://provider.example/register"}
    requested = _fake_client(monkeypatch, get=lambda url: _Response(asm))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/metadata",
            headers=_headers(),
            json={"kind": "authorization_server", "issuer": "https://provider.example"},
        )

    assert response.status_code == 200
    assert response.json() == asm
    assert requested["get"] == ["https://provider.example/.well-known/oauth-authorization-server"]


def test_metadata_route_reads_the_protected_resource_document(monkeypatch):
    prm = {"resource": "https://provider.example/mcp", "authorization_servers": ["https://provider.example"]}
    requested = _fake_client(monkeypatch, get=lambda url: _Response(prm))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/metadata",
            headers=_headers(),
            json={"kind": "protected_resource", "resource": "https://provider.example/mcp"},
        )

    assert response.status_code == 200
    assert response.json() == prm
    assert requested["get"] == ["https://provider.example/.well-known/oauth-protected-resource/mcp"]


def test_metadata_route_requires_a_target_for_the_declared_kind():
    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/metadata",
            headers=_headers(),
            json={"kind": "authorization_server"},
        )
    assert response.status_code == 422


def test_metadata_route_never_reaches_a_private_address(monkeypatch):
    requested = _fake_client(monkeypatch, get=lambda url: _Response({"issuer": "x"}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/metadata",
            headers=_headers(),
            json={"kind": "authorization_server", "issuer": "https://127.0.0.1:9999"},
        )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "mcp_unreachable"
    assert requested["get"] == []


def test_metadata_route_rejects_a_metadata_provider_link_address(monkeypatch):
    """169.254.169.254 (cloud metadata) is blocked the same way a private IP is."""

    requested = _fake_client(monkeypatch, get=lambda url: _Response({"issuer": "x"}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/metadata",
            headers=_headers(),
            json={"kind": "authorization_server", "issuer": "https://169.254.169.254"},
        )

    assert response.status_code == 502
    assert requested["get"] == []


def test_register_route_forwards_the_dcr_payload_and_returns_the_client_id(monkeypatch):
    requested = _fake_client(
        monkeypatch, post=lambda url, body: _Response({"client_id": "abc123"})
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/register",
            headers=_headers(),
            json={
                "issuer": "https://provider.example",
                "registration_endpoint": "https://provider.example/register",
                "redirect_uris": ["https://legion.example/callback"],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"client_id": "abc123"}
    (url, body), = requested["post"]
    assert url == "https://provider.example/register"
    assert body["redirect_uris"] == ["https://legion.example/callback"]
    assert body["token_endpoint_auth_method"] == "none"


def test_register_route_surfaces_a_provider_rejection_without_the_body(monkeypatch):
    _fake_client(monkeypatch, post=lambda url, body: _Response({"error": "invalid_client_metadata"}, status_code=400))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/register",
            headers=_headers(),
            json={
                "issuer": "https://provider.example",
                "registration_endpoint": "https://provider.example/register",
                "redirect_uris": ["https://legion.example/callback"],
            },
        )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "mcp_oauth_registration_rejected"
    assert "invalid_client_metadata" not in response.text


def test_register_route_never_reaches_a_private_registration_endpoint(monkeypatch):
    requested = _fake_client(monkeypatch, post=lambda url, body: _Response({"client_id": "x"}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/register",
            headers=_headers(),
            json={
                "issuer": "https://provider.example",
                "registration_endpoint": "http://10.0.0.5/register",
                "redirect_uris": ["https://legion.example/callback"],
            },
        )

    assert response.status_code == 502
    assert requested["post"] == []


def test_token_route_authorization_code_grant_forwards_pkce_fields(monkeypatch):
    token_response = {"access_token": "at", "refresh_token": "rt", "expires_in": 3600, "token_type": "Bearer"}
    requested = _fake_client(monkeypatch, post=lambda url, body: _Response(token_response))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/token",
            headers=_headers(),
            json={
                "token_endpoint": "https://provider.example/token",
                "grant_type": "authorization_code",
                "client_id": "abc123",
                "code": "one-time-code",
                "redirect_uri": "https://legion.example/callback",
                "code_verifier": "verifier-value",
            },
        )

    assert response.status_code == 200
    assert response.json() == token_response
    (url, body), = requested["post"]
    assert url == "https://provider.example/token"
    assert body["code"] == "one-time-code"
    assert body["code_verifier"] == "verifier-value"
    assert body["grant_type"] == "authorization_code"


def test_token_route_refresh_grant_requires_the_refresh_token():
    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/token",
            headers=_headers(),
            json={
                "token_endpoint": "https://provider.example/token",
                "grant_type": "refresh_token",
                "client_id": "abc123",
            },
        )
    assert response.status_code == 422


def test_token_route_forwards_a_manual_clients_client_secret(monkeypatch):
    """internal design notes Stage 8 — a manually registered confidential client's
    ``client_secret`` rides along in the token-endpoint form body."""

    token_response = {"access_token": "at", "expires_in": 3600}
    requested = _fake_client(monkeypatch, post=lambda url, body: _Response(token_response))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/token",
            headers=_headers(),
            json={
                "token_endpoint": "https://provider.example/token",
                "grant_type": "refresh_token",
                "client_id": "manual-client",
                "refresh_token": "the-refresh-token",
                "client_secret": "the-client-secret",
            },
        )

    assert response.status_code == 200
    (_url, body), = requested["post"]
    assert body["client_secret"] == "the-client-secret"


def test_token_route_omits_client_secret_for_a_public_client(monkeypatch):
    token_response = {"access_token": "at", "expires_in": 3600}
    requested = _fake_client(monkeypatch, post=lambda url, body: _Response(token_response))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/token",
            headers=_headers(),
            json={
                "token_endpoint": "https://provider.example/token",
                "grant_type": "refresh_token",
                "client_id": "dcr-client",
                "refresh_token": "the-refresh-token",
            },
        )

    assert response.status_code == 200
    (_url, body), = requested["post"]
    assert "client_secret" not in body


def test_token_route_rejection_never_echoes_the_code_or_token(monkeypatch):
    _fake_client(monkeypatch, post=lambda url, body: _Response({"error": "invalid_grant"}, status_code=400))

    with TestClient(app) as client:
        response = client.post(
            "/v1/oauth/token",
            headers=_headers(),
            json={
                "token_endpoint": "https://provider.example/token",
                "grant_type": "authorization_code",
                "client_id": "abc123",
                "code": "super-secret-code",
                "redirect_uri": "https://legion.example/callback",
                "code_verifier": "super-secret-verifier",
            },
        )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "mcp_oauth_token_rejected"
    assert "super-secret-code" not in response.text
    assert "super-secret-verifier" not in response.text
