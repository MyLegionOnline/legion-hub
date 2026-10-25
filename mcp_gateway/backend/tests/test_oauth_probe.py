"""internal design notes Stage 0 — "login required" is told apart from "bad token"."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import config, oauth_probe
from app.main import app


@pytest.fixture(autouse=True)
def _no_dns_wait(monkeypatch):
    """Skip the rebind lookup: these fixtures' hostnames do not resolve.

    Without this every guarded URL waits out a DNS timeout in the container.
    The literal-IP branch of the perimeter is what this file exercises; the
    resolved-address branch stays covered by test_http_mcp_stage6.py.
    """

    monkeypatch.setattr(config, "MCP_GATEWAY_DNS_REBIND_PROTECTION", False)


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.GATEWAY_TOKEN}",
        "X-Legion-User-Id": str(uuid4()),
        "X-Legion-Room-Id": str(uuid4()),
    }


class _Response:
    def __init__(self, body, status_code: int = 200, headers: dict | None = None):
        self.status_code = status_code
        self.content = json.dumps(body).encode() if body is not None else b""
        self.text = self.content.decode()
        self.headers = headers or {}

    def json(self):
        if not self.text:
            raise ValueError("no body")
        return json.loads(self.text)


def _upstream(
    monkeypatch,
    *,
    www_authenticate: str | None,
    documents: dict[str, dict] | None = None,
) -> list[str]:
    """Fake both hops at once and record every URL the metadata probe fetched.

    ``mcp_client`` and ``oauth_probe`` share one ``httpx`` module object, so a
    single patch has to serve the MCP handshake (``post``) and the metadata
    reads (``get``) — patching them separately would silently overwrite one.
    """

    import app.mcp_client as mcp_client_module

    assert mcp_client_module.httpx is oauth_probe.httpx
    documents = documents or {}
    requested: list[str] = []

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, endpoint, *, json, headers):
            return _Response(
                None,
                status_code=401,
                headers={"www-authenticate": www_authenticate} if www_authenticate else {},
            )

        async def get(self, url, *, headers):
            requested.append(url)
            document = documents.get(url)
            if document is None:
                return _Response({"error": "not found"}, status_code=404)
            return _Response(document)

    monkeypatch.setattr(mcp_client_module.httpx, "AsyncClient", lambda **kwargs: Client())
    return requested


def _discover(url: str = "https://provider.example/mcp"):
    with TestClient(app) as client:
        return client.post(
            "/v1/tools/discover",
            headers=_headers(),
            json={"installed_item_id": str(uuid4()), "custom_config": {"url": url}},
        )


def test_oauth_challenge_names_the_provider(monkeypatch):
    """A 401 carrying a challenge becomes a named, non-retryable cause."""

    _upstream(
        monkeypatch,
        www_authenticate=(
            'Bearer realm="OAuth", '
            'resource_metadata="https://provider.example/.well-known/oauth-protected-resource/mcp"'
        ),
        documents={
            "https://provider.example/.well-known/oauth-protected-resource/mcp": {
                "resource": "https://provider.example/mcp",
                "resource_name": "Example API MCP Server",
                "authorization_servers": ["https://provider.example"],
            }
        },
    )

    response = _discover()

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "mcp_oauth_required"
    assert error["provider"] == "Example API MCP Server"
    assert error["retryable"] is False


def test_plain_401_stays_an_auth_failure(monkeypatch):
    """Without a challenge a wrong bearer must still read as a wrong bearer."""

    requested = _upstream(monkeypatch, www_authenticate=None)

    response = _discover()

    error = response.json()["error"]
    assert error["code"] == "mcp_auth_failed"
    assert error["provider"] is None
    assert requested == [], "no metadata probe without an OAuth challenge"


def test_non_bearer_challenge_is_not_oauth(monkeypatch):
    _upstream(monkeypatch, www_authenticate='Basic realm="internal"')

    assert _discover().json()["error"]["code"] == "mcp_auth_failed"


def test_metadata_url_into_private_space_is_not_followed(monkeypatch):
    """The advertised metadata URL crosses the same SSRF perimeter."""

    requested = _upstream(
        monkeypatch,
        www_authenticate='Bearer resource_metadata="http://127.0.0.1:9/.well-known/oauth-protected-resource"',
    )

    response = _discover()
    error = response.json()["error"]

    assert not any("127.0.0.1" in url for url in requested)
    # The finding survives the blocked lookup; only the label degrades.
    assert error["code"] == "mcp_oauth_required"
    assert error["provider"] == "provider.example"


def test_unreachable_metadata_degrades_to_the_hostname(monkeypatch):
    _upstream(monkeypatch, www_authenticate="Bearer")

    error = _discover().json()["error"]

    assert error["code"] == "mcp_oauth_required"
    assert error["provider"] == "provider.example"


def test_issuer_names_the_provider_when_resource_name_is_absent(monkeypatch):
    _upstream(
        monkeypatch,
        www_authenticate="Bearer",
        documents={
            "https://provider.example/.well-known/oauth-protected-resource/mcp": {
                "authorization_servers": ["https://login.provider.example"],
            },
            "https://login.provider.example/.well-known/oauth-authorization-server": {
                "issuer": "https://login.provider.example",
                "registration_endpoint": "https://login.provider.example/register",
            },
        },
    )

    assert _discover().json()["error"]["provider"] == "login.provider.example"


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "Basic realm=x",
        "Bearer realm=\x00evil",
        "Bearer " + "x" * 5000,
    ],
)
def test_parse_challenge_rejects_what_is_not_a_bearer_challenge(header):
    assert oauth_probe.parse_challenge(header) is None


def test_parse_challenge_reads_quoted_parameters():
    challenge = oauth_probe.parse_challenge(
        'Bearer realm="OAuth", resource_metadata="https://p.example/.well-known/x", scope="dns"'
    )

    assert challenge is not None
    assert challenge.resource_metadata == "https://p.example/.well-known/x"
    assert challenge.params["scope"] == "dns"


def test_label_never_carries_a_url_or_an_unbounded_string():
    assert oauth_probe._safe_label("https://evil.example/pwn") is None
    assert oauth_probe._safe_label("  Cloudflare  ") == "Cloudflare"
    assert oauth_probe._safe_label("x" * 200).endswith("…")
    assert len(oauth_probe._safe_label("x" * 200)) <= oauth_probe.MAX_LABEL_CHARS + 1
