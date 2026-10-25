from __future__ import annotations

from fastapi.testclient import TestClient

from app import config
from app import main as gateway_main


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.GATEWAY_ADMIN_TOKEN}"}


def test_catalog_vision_uses_its_private_bearer(monkeypatch):
    monkeypatch.setattr(config, "MCP_VISION_AUTH_TOKEN", "vision-mcp-token")
    endpoint, token = gateway_main._endpoint_config(
        {"slug": "mcp-vision", "internal_endpoint": "http://mcp-vision:3000/mcp"},
        None,
    )
    assert endpoint == "http://mcp-vision:3000/mcp"
    assert token == "vision-mcp-token"
    assert gateway_main._catalog_token({"slug": "mcp-vision"}) == "vision-mcp-token"


def test_vision_admin_requires_separate_admin_token():
    with TestClient(gateway_main.app) as client:
        response = client.get("/v1/admin/vision-config")
    assert response.status_code == 401


def test_vision_admin_forwards_configuration_without_returning_key(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {
                "configured": True,
                "provider": "gemini",
                "model": "vision-model",
                "api_key_configured": True,
                "max_transport_image_bytes": 25165824,
                "max_processing_pixels": 8000000,
                "source": "runtime",
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request(self, method, url, *, json, headers):
            captured.update({"method": method, "url": url, "json": json, "headers": headers})
            return Response()

    monkeypatch.setattr(config, "MCP_VISION_ADMIN_TOKEN", "vision-service-admin")
    monkeypatch.setattr(gateway_main.httpx, "AsyncClient", lambda **kwargs: Client())
    secret = "provider-secret-key"
    with TestClient(gateway_main.app) as client:
        response = client.put(
            "/v1/admin/vision-config",
            headers=_admin_headers(),
            json={
                "provider": "gemini",
                "model": "vision-model",
                "api_key": secret,
                "max_transport_image_bytes": 25165824,
                "max_processing_pixels": 8000000,
            },
        )
    assert response.status_code == 200, response.text
    assert captured["headers"]["Authorization"] == "Bearer vision-service-admin"
    assert captured["json"]["api_key"] == secret
    assert secret not in response.text
    assert "api_key" not in response.json()


def test_vision_admin_does_not_proxy_upstream_error_body(monkeypatch):
    secret = "provider-secret-key"

    class Response:
        status_code = 422

        def json(self):
            return {"detail": f"bad key {secret}"}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(config, "MCP_VISION_ADMIN_TOKEN", "vision-service-admin")
    monkeypatch.setattr(gateway_main.httpx, "AsyncClient", lambda **kwargs: Client())
    with TestClient(gateway_main.app) as client:
        response = client.put(
            "/v1/admin/vision-config",
            headers=_admin_headers(),
            json={"provider": "gemini", "model": "vision-model", "api_key": secret},
        )
    assert response.status_code == 422
    assert secret not in response.text


def test_vision_model_validation_forwards_candidate_key_without_returning_it(monkeypatch):
    captured = {}

    class Response:
        status_code = 200

        def json(self):
            return {
                "valid": True,
                "authenticated": True,
                "provider": "openrouter",
                "models": [{"id": "vendor/vision", "name": "Vision"}],
            }

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request(self, method, url, *, json, headers):
            captured.update({"method": method, "url": url, "json": json, "headers": headers})
            return Response()

    monkeypatch.setattr(config, "MCP_VISION_ADMIN_TOKEN", "vision-service-admin")
    monkeypatch.setattr(gateway_main.httpx, "AsyncClient", lambda **kwargs: Client())
    secret = "candidate-provider-secret"
    with TestClient(gateway_main.app) as client:
        response = client.post(
            "/v1/admin/vision-models",
            headers=_admin_headers(),
            json={"provider": "openrouter", "api_key": secret},
        )
    assert response.status_code == 200, response.text
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/admin/models")
    assert captured["json"]["api_key"] == secret
    assert secret not in response.text
    assert response.json()["models"][0]["id"] == "vendor/vision"
