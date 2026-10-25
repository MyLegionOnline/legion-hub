"""GET /api/v1/features: the core says "no billing", a plugin can say otherwise."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.v1 import features
from app.plugins import reset_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(features.router, prefix="/api/v1")
    return TestClient(app)


def test_core_without_plugins_has_no_billing_and_no_market(monkeypatch):
    monkeypatch.delenv("LEGION_SOURCE_URL", raising=False)
    monkeypatch.delenv("MARKET_ENABLED", raising=False)
    res = _client().get("/api/v1/features")
    assert res.status_code == 200
    assert res.json() == {"billing": False, "market": False, "source_url": features.DEFAULT_SOURCE_URL}


def test_market_flag_follows_the_switch(monkeypatch):
    monkeypatch.setenv("MARKET_ENABLED", "true")
    assert _client().get("/api/v1/features").json()["market"] is True
    monkeypatch.setenv("MARKET_ENABLED", "false")
    assert _client().get("/api/v1/features").json()["market"] is False


def test_source_link_follows_the_operator_and_ignores_unsafe_values(monkeypatch):
    monkeypatch.setenv("LEGION_SOURCE_URL", "https://git.example.org/me/my-legion-fork")
    assert _client().get("/api/v1/features").json()["source_url"] == "https://git.example.org/me/my-legion-fork"
    for bad in ("javascript:alert(1)", "not a url", "ftp://x.example/y", "   "):
        monkeypatch.setenv("LEGION_SOURCE_URL", bad)
        assert _client().get("/api/v1/features").json()["source_url"] == features.DEFAULT_SOURCE_URL


def test_plugin_flags_are_added_and_override_core_defaults():
    reset_registry().features.update(billing=True, pricing_url="https://market.example/pricing")
    body = _client().get("/api/v1/features").json()
    assert body["billing"] is True
    assert body["pricing_url"] == "https://market.example/pricing"


def test_route_is_public_and_registered_in_the_app():
    # No token and no lifespan: the login page asks before anyone is signed in.
    from app.main import app

    res = TestClient(app).get("/api/v1/features")
    assert res.status_code == 200
    assert res.json()["billing"] in (True, False)
