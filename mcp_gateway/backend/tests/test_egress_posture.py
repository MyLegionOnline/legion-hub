"""internal design notes Stage 9 — honest egress posture, Gateway side.

Two things are being pinned here:

* the first-party VPS transport is routed like mcp-vision (its bearer token
  lives in Gateway configuration, never in the Market catalog record), and
* the posture endpoint reports the operator's *declared* mode next to what
  the service itself reports, and never invents the second one.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config
from app import main as gateway_main


def _admin_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.GATEWAY_ADMIN_TOKEN}"}


def _service_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.GATEWAY_TOKEN}"}


def test_catalog_vps_uses_its_private_bearer(monkeypatch):
    monkeypatch.setattr(config, "MCP_VPS_AUTH_TOKEN", "vps-mcp-token")
    endpoint, token = gateway_main._endpoint_config(
        {"slug": "legion-vps-mcp", "internal_endpoint": "http://vps-mcp:3000/mcp"},
        None,
    )
    assert endpoint == "http://vps-mcp:3000/mcp"
    assert token == "vps-mcp-token"
    assert gateway_main._catalog_token({"slug": "legion-vps-mcp"}) == "vps-mcp-token"


def _patch_health(monkeypatch, *, body: Any = None, status_code: int = 200, boom: bool = False) -> None:
    class _Response:
        def __init__(self) -> None:
            self.status_code = status_code

        def json(self) -> Any:
            if body is None:
                raise ValueError("no body")
            return body

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url: str) -> Any:
            assert url == "http://vps-mcp:3000/health"
            if boom:
                raise httpx.ConnectError("no route")
            return _Response()

    monkeypatch.setattr(gateway_main.httpx, "AsyncClient", _Client)

    async def _no_catalog(_slug: str) -> dict:
        raise gateway_main.HTTPException(status_code=502, detail="market down")

    monkeypatch.setattr(gateway_main.market, "resolve_by_slug", _no_catalog)


@pytest.mark.parametrize("path, headers", [
    ("/v1/egress-posture", {}),
    ("/v1/admin/egress-posture", {}),
])
def test_posture_requires_a_token(path, headers):
    with TestClient(gateway_main.app) as client:
        assert client.get(path, headers=headers).status_code == 401


def test_posture_reports_broker_when_service_confirms_it(monkeypatch):
    monkeypatch.setattr(config, "MCP_EGRESS_MODE", "broker")
    _patch_health(monkeypatch, body={"version": "0.1.0", "egress": {"mode": "broker"}})
    with TestClient(gateway_main.app) as client:
        body = client.get("/v1/egress-posture", headers=_service_headers()).json()
    assert body["declared_mode"] == "broker"
    assert body["verdict"] == "enforced"
    assert body["service"] == {"reachable": True, "version": "0.1.0", "egress_mode": "broker"}


def test_off_in_env_is_off_in_the_panel(monkeypatch):
    # Acceptance criterion of Stage 9: there is no "silent" weakening.
    monkeypatch.setattr(config, "MCP_EGRESS_MODE", "off")
    _patch_health(monkeypatch, body={"version": "0.1.0", "egress": {"mode": "direct"}})
    with TestClient(gateway_main.app) as client:
        body = client.get("/v1/admin/egress-posture", headers=_admin_headers()).json()
    assert body["declared_mode"] == "off"
    assert body["verdict"] == "not_enforced"


def test_declared_broker_but_direct_service_is_a_mismatch(monkeypatch):
    monkeypatch.setattr(config, "MCP_EGRESS_MODE", "broker")
    _patch_health(monkeypatch, body={"version": "0.1.0", "egress": {"mode": "direct"}})
    with TestClient(gateway_main.app) as client:
        body = client.get("/v1/egress-posture", headers=_service_headers()).json()
    assert body["verdict"] == "mismatch"


def test_host_firewall_is_reported_as_an_unverifiable_operator_claim(monkeypatch):
    monkeypatch.setattr(config, "MCP_EGRESS_MODE", "host_firewall")
    _patch_health(monkeypatch, body={"version": "0.1.0", "egress": {"mode": "direct"}})
    with TestClient(gateway_main.app) as client:
        body = client.get("/v1/egress-posture", headers=_service_headers()).json()
    assert body["verdict"] == "claimed_by_operator"


def test_unreachable_service_is_unverified_not_assumed(monkeypatch):
    monkeypatch.setattr(config, "MCP_EGRESS_MODE", "broker")
    _patch_health(monkeypatch, boom=True)
    with TestClient(gateway_main.app) as client:
        body = client.get("/v1/egress-posture", headers=_service_headers()).json()
    assert body["verdict"] == "unverified"
    assert body["service"] == {"reachable": False, "reason": "service_unreachable"}


def test_service_without_the_egress_block_is_unknown_not_direct(monkeypatch):
    monkeypatch.setattr(config, "MCP_EGRESS_MODE", "broker")
    _patch_health(monkeypatch, body={"version": "0.0.9"})
    with TestClient(gateway_main.app) as client:
        body = client.get("/v1/egress-posture", headers=_service_headers()).json()
    assert body["service"]["egress_mode"] == "unknown"
    assert body["verdict"] == "unverified"


def test_typo_in_the_mode_is_misconfigured_not_a_posture(monkeypatch):
    monkeypatch.setattr(config, "MCP_EGRESS_MODE", "brokerr")
    _patch_health(monkeypatch, body={"version": "0.1.0", "egress": {"mode": "broker"}})
    with TestClient(gateway_main.app) as client:
        body = client.get("/v1/egress-posture", headers=_service_headers()).json()
    assert body["declared_mode_valid"] is False
    assert body["verdict"] == "misconfigured"
