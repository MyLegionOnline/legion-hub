"""OpenRouter App Attribution headers: only openrouter.ai gets them, env can override."""

from __future__ import annotations

import pytest

from app.core.llm_providers import (
    openai_base_url,
    openai_client_extras,
    openrouter_attribution_headers,
)
from app.core.reasoning import _probe_request


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OPENROUTER_APP_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)


def test_openrouter_gets_default_app_identity():
    headers = openrouter_attribution_headers("https://openrouter.ai/api/v1")
    assert headers["HTTP-Referer"] == "https://mylegion.online"
    assert headers["X-OpenRouter-Title"] == "Legion Hub"
    assert headers["X-Title"] == "Legion Hub"


def test_other_hosts_get_nothing():
    for url in (
        "https://api.openai.com/v1",
        "https://api.anthropic.com/v1",
        "http://host.docker.internal:11434/v1",
        "https://evil-openrouter.ai.example.com/v1",
        "https://notopenrouter.ai/v1",
        "",
        None,
    ):
        assert openrouter_attribution_headers(url) == {}


def test_env_overrides_identity(monkeypatch):
    monkeypatch.setenv("OPENROUTER_APP_URL", "https://my.example.org")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "My Fork")
    headers = openrouter_attribution_headers("https://openrouter.ai/api/v1")
    assert headers["HTTP-Referer"] == "https://my.example.org"
    assert headers["X-OpenRouter-Title"] == "My Fork"


def test_empty_url_turns_attribution_off(monkeypatch):
    monkeypatch.setenv("OPENROUTER_APP_URL", "")
    assert openrouter_attribution_headers("https://openrouter.ai/api/v1") == {}


def test_empty_title_keeps_referer_only(monkeypatch):
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "")
    headers = openrouter_attribution_headers("https://openrouter.ai/api/v1")
    assert headers == {"HTTP-Referer": "https://mylegion.online"}


def test_openai_client_extras_shape():
    extras = openai_client_extras(openai_base_url("openrouter", None))
    assert extras["default_headers"]["HTTP-Referer"] == "https://mylegion.online"
    assert openai_client_extras(openai_base_url("groq", None)) == {}


def test_probe_request_carries_attribution_only_on_openrouter():
    _, headers, _ = _probe_request(
        provider="openrouter", model="m", wire="chat",
        api_key="k", base_url=None, fields={},
    )
    assert headers["HTTP-Referer"] == "https://mylegion.online"
    _, headers, _ = _probe_request(
        provider="groq", model="m", wire="chat",
        api_key="k", base_url=None, fields={},
    )
    assert "HTTP-Referer" not in headers
