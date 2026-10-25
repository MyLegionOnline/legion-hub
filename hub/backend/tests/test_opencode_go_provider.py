"""OpenCode Go provider: wire resolution, endpoints and catalog filtering.

The gateway picks its wire per model, so these tests pin the two things that
break silently: that a MiniMax/Qwen model is driven over ``/v1/messages`` with a
bearer token, and that the Responses-only models never reach a model picker.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from app.api.v1.providers import (
    _PROBE_CACHE,
    _opencode_go_models,
    _parse_models,
    _validate_opencode_go,
)
from app.core.llm_providers import (
    OPENCODE_GO,
    OPENCODE_GO_BASE_URL,
    WIRE_CHAT,
    WIRE_MESSAGES,
    WIRE_RESPONSES,
    is_unsupported_wire,
    messages_headers,
    messages_text_and_thinking,
    messages_url,
    openai_base_url,
    provider_wire,
    uses_anthropic_env_key,
    uses_messages_wire,
)
from app.core.media_refs import ResolvedRef, build_vision_blocks
from app.schemas.agent import AgentUpdate


# --------------------------------------------------------------------- wires


def test_chat_wire_models_use_chat_completions():
    for model in ("glm-5.3", "kimi-k3", "deepseek-v4-pro", "longcat-2.0", "hy3"):
        assert provider_wire(OPENCODE_GO, model) == WIRE_CHAT
        assert not uses_messages_wire(OPENCODE_GO, model)


def test_messages_wire_models_use_anthropic_shape():
    for model in ("minimax-m3", "minimax-m2.5", "qwen3.8-max", "qwen3.6-plus"):
        assert provider_wire(OPENCODE_GO, model) == WIRE_MESSAGES
        assert uses_messages_wire(OPENCODE_GO, model)


def test_responses_only_models_are_marked_unsupported():
    for model in ("grok-4.6", "gpt-5.6-luna", "muse-spark-1.2-contributor"):
        assert provider_wire(OPENCODE_GO, model) == WIRE_RESPONSES
        assert is_unsupported_wire(OPENCODE_GO, model)


def test_unlisted_model_falls_back_to_its_family_wire():
    # Served by the gateway before the published catalog listed it.
    assert provider_wire(OPENCODE_GO, "kimi-k2.5") == WIRE_CHAT
    assert provider_wire(OPENCODE_GO, "minimax-m4") == WIRE_MESSAGES
    assert provider_wire(OPENCODE_GO, "qwen4-max") == WIRE_MESSAGES


def test_other_providers_keep_their_previous_wire():
    assert uses_messages_wire("anthropic", "claude-sonnet-4-6")
    for provider in ("openrouter", "openai", "groq", "together", "ollama", "deepseek"):
        assert not uses_messages_wire(provider, "whatever")
    # An unset provider must not be guessed into Anthropic.
    assert not uses_messages_wire("", "whatever")


# ----------------------------------------------------------------- endpoints


def test_base_urls_are_unchanged_for_existing_providers():
    assert openai_base_url("openrouter", None) == "https://openrouter.ai/api/v1"
    assert openai_base_url("groq", None) == "https://api.groq.com/openai/v1"
    assert openai_base_url("together", None) == "https://api.together.xyz/v1"
    assert openai_base_url("deepseek", None) == "https://api.deepseek.com/v1"
    assert openai_base_url("ollama", None) == "http://host.docker.internal:11434/v1"
    assert openai_base_url("openai", None) == "https://api.openai.com/v1"


def test_opencode_go_base_url_and_override():
    assert openai_base_url(OPENCODE_GO, None) == OPENCODE_GO_BASE_URL
    assert openai_base_url(OPENCODE_GO, "https://proxy.local/v1/") == "https://proxy.local/v1"


def test_messages_url_per_provider():
    assert messages_url("anthropic", None) == "https://api.anthropic.com/v1/messages"
    assert messages_url(OPENCODE_GO, None) == f"{OPENCODE_GO_BASE_URL}/messages"
    assert messages_url(OPENCODE_GO, "https://proxy.local/v1") == "https://proxy.local/v1/messages"


def test_messages_headers_auth_scheme():
    anthropic = messages_headers("anthropic", "sk-ant-123")
    assert anthropic["x-api-key"] == "sk-ant-123"
    assert "Authorization" not in anthropic
    assert anthropic["anthropic-version"] == "2023-06-01"

    # The OpenCode gateway authenticates /messages with x-api-key, not bearer.
    # Measured 27.08.2026: a bearer token on /zen/go/v1/messages returns 401.
    gateway = messages_headers(OPENCODE_GO, "oc-123")
    assert gateway["x-api-key"] == "oc-123"
    assert "Authorization" not in gateway
    assert gateway["anthropic-version"] == "2023-06-01"


def test_messages_headers_falls_back_to_empty_key():
    assert messages_headers(OPENCODE_GO, None)["x-api-key"] == ""
    assert messages_headers("anthropic", "")["x-api-key"] == ""


def test_anthropic_env_key_is_not_reused_for_a_gateway():
    assert uses_anthropic_env_key("anthropic")
    assert not uses_anthropic_env_key(OPENCODE_GO)
    assert not uses_anthropic_env_key("openrouter")


# ------------------------------------------------------------------- catalog


def _catalog(*ids):
    return {"object": "list", "data": [{"id": model_id} for model_id in ids]}


def test_catalog_drops_responses_only_models_and_names_the_rest():
    models = _opencode_go_models(_catalog("kimi-k3", "grok-4.6", "minimax-m3", "gpt-5.6-luna"))
    assert [m.id for m in models] == ["kimi-k3", "minimax-m3"]
    assert {m.name for m in models} == {"Kimi K3", "MiniMax M3"}


def test_catalog_keeps_unlisted_models_under_their_own_id():
    # minimax-m4 is served by the gateway but absent from OPENCODE_GO_CATALOG,
    # so its display name falls back to the model id itself.
    models = _opencode_go_models(_catalog("minimax-m4"))
    assert [(m.id, m.name) for m in models] == [("minimax-m4", "minimax-m4")]


def test_parse_models_routes_opencode_go_to_its_own_parser():
    models = _parse_models(OPENCODE_GO, _catalog("glm-5.3", "grok-4.6"))
    assert [m.id for m in models] == ["glm-5.3"]


# ---------------------------------------------------------------- validation

LIVE_CATALOG = _catalog(
    "deepseek-v4-flash", "deepseek-v4-pro", "glm-5.3-flash", "glm-5.3",
    "kimi-k3", "minimax-m3", "grok-4.6",
)
# grok-4.6 is Responses-only: dropped from the catalog, never probed.
PROBED = {"deepseek-v4-flash", "deepseek-v4-pro", "glm-5.3-flash", "glm-5.3", "kimi-k3", "minimax-m3"}


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.headers = {"content-type": "application/json"} if payload is not None else {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _ok(_model_id):
    return _FakeResponse(200, {"choices": [{"message": {"content": "pong"}}]})


class _FakeClient:
    """Stands in for httpx.AsyncClient inside _validate_opencode_go.

    Probes run concurrently, so the answer is chosen per model, not by order.
    """

    def __init__(self, respond=_ok):
        self._respond = respond
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, headers=None, timeout=None):
        return _FakeResponse(200, LIVE_CATALOG)

    async def post(self, url, headers=None, json=None):
        self.posts.append((url, headers or {}, json or {}))
        return self._respond(json["model"])

    def probed(self):
        return {body["model"] for _url, _headers, body in self.posts}


def _patch_client(monkeypatch, client):
    import app.api.v1.providers as providers_module

    monkeypatch.setattr(
        providers_module.httpx, "AsyncClient", lambda *a, **kw: client, raising=True
    )
    return client


@pytest.fixture(autouse=True)
def _empty_probe_cache():
    # The 24h probe cache is module-level; one test's outcomes must not
    # answer the next test's probes.
    _PROBE_CACHE.clear()
    yield
    _PROBE_CACHE.clear()


@pytest.mark.asyncio
async def test_every_chat_and_messages_model_is_probed_once(monkeypatch):
    client = _patch_client(monkeypatch, _FakeClient())

    result = await _validate_opencode_go("oc-good", None)

    assert result.valid is True
    assert result.authenticated is True
    assert result.warning is None
    assert client.probed() == PROBED
    assert len(client.posts) == len(PROBED)
    assert all(m.available for m in result.models)


@pytest.mark.asyncio
async def test_probe_uses_each_models_own_wire(monkeypatch):
    client = _patch_client(monkeypatch, _FakeClient())

    await _validate_opencode_go("oc-good", None)

    by_model = {body["model"]: (url, headers) for url, headers, body in client.posts}
    chat_url, chat_headers = by_model["glm-5.3-flash"]
    assert chat_url.endswith("/chat/completions")
    assert chat_headers["Authorization"] == "Bearer oc-good"

    msg_url, msg_headers = by_model["minimax-m3"]
    assert msg_url.endswith("/messages")
    assert msg_headers["x-api-key"] == "oc-good"
    assert "Authorization" not in msg_headers


@pytest.mark.asyncio
async def test_401_reports_the_gateway_message_as_a_key_problem(monkeypatch):
    auth_error = {"type": "error", "error": {"type": "AuthError", "message": "Invalid API key."}}
    _patch_client(monkeypatch, _FakeClient(lambda _m: _FakeResponse(401, auth_error)))

    result = await _validate_opencode_go("oc-bad", None)

    assert result.valid is False
    assert "Invalid API key." in result.error
    assert result.models == []


@pytest.mark.asyncio
async def test_401_is_never_cached_as_a_model_outcome(monkeypatch):
    _patch_client(monkeypatch, _FakeClient(lambda _m: _FakeResponse(401, {})))

    await _validate_opencode_go("oc-bad", None)

    assert _PROBE_CACHE == {}


@pytest.mark.asyncio
async def test_403_keeps_the_key_and_the_catalog_and_marks_models_unavailable(monkeypatch):
    forbidden = {"error": {"message": "Model not available on your plan"}}
    _patch_client(monkeypatch, _FakeClient(lambda _m: _FakeResponse(403, forbidden)))

    result = await _validate_opencode_go("oc-good", None)

    # A 403 is not a dead key: the gateway answers 401 for those.
    assert result.valid is True
    assert result.authenticated is True
    assert result.error is None
    assert result.warning
    assert {m.id for m in result.models} == PROBED
    assert all(not m.available and m.unavailable_reason == "plan" for m in result.models)


@pytest.mark.asyncio
async def test_one_refused_model_does_not_spoil_the_rest(monkeypatch):
    def respond(model_id):
        if model_id == "deepseek-v4-flash":
            return _FakeResponse(403, {"error": {"message": "no"}})
        return _ok(model_id)

    _patch_client(monkeypatch, _FakeClient(respond))

    result = await _validate_opencode_go("oc-good", None)

    assert result.valid is True
    assert result.warning is None
    assert {m.id for m in result.models if not m.available} == {"deepseek-v4-flash"}


@pytest.mark.asyncio
async def test_region_lock_carries_the_opt_in_link_from_the_error(monkeypatch):
    region = {
        "error": {
            "type": "RegionError",
            "message": "Enable China-hosted models at https://opencode.ai/workspace/wrk_1/settings.",
        }
    }

    def respond(model_id):
        return _FakeResponse(403, region) if model_id == "deepseek-v4-flash" else _ok(model_id)

    _patch_client(monkeypatch, _FakeClient(respond))

    result = await _validate_opencode_go("oc-good", None)

    locked = next(m for m in result.models if m.id == "deepseek-v4-flash")
    assert locked.available is False
    assert locked.unavailable_reason == "region_opt_in"
    assert locked.opt_in_url == "https://opencode.ai/workspace/wrk_1/settings"


@pytest.mark.asyncio
async def test_400_marks_a_listed_but_unserved_model_unsupported(monkeypatch):
    def respond(model_id):
        if model_id == "kimi-k3":
            return _FakeResponse(400, {"error": {"message": "Unsupported model"}})
        return _ok(model_id)

    _patch_client(monkeypatch, _FakeClient(respond))

    result = await _validate_opencode_go("oc-good", None)

    kimi = next(m for m in result.models if m.id == "kimi-k3")
    assert kimi.available is False
    assert kimi.unavailable_reason == "unsupported"


@pytest.mark.asyncio
async def test_transport_error_leaves_the_model_selectable_and_uncached(monkeypatch):
    def respond(model_id):
        if model_id == "glm-5.3":
            raise httpx.ReadTimeout("slow")
        return _ok(model_id)

    _patch_client(monkeypatch, _FakeClient(respond))

    result = await _validate_opencode_go("oc-good", None)

    assert next(m for m in result.models if m.id == "glm-5.3").available is True
    assert not any(key[2] == "glm-5.3" for key in _PROBE_CACHE)


@pytest.mark.asyncio
async def test_outcomes_are_cached_per_key_and_refresh_bypasses_the_cache(monkeypatch):
    client = _patch_client(monkeypatch, _FakeClient())

    await _validate_opencode_go("oc-good", None)
    first = len(client.posts)

    await _validate_opencode_go("oc-good", None)
    assert len(client.posts) == first  # served from the 24h cache

    await _validate_opencode_go("oc-other", None)
    assert len(client.posts) == 2 * first  # another key may have another plan

    await _validate_opencode_go("oc-good", None, refresh_probe=True)
    assert len(client.posts) == 3 * first


# ------------------------------------------------------ thinking-aware parsing


def test_messages_text_and_thinking_plain_text_reply():
    data = {"content": [{"type": "text", "text": "hello"}]}
    text, thinking = messages_text_and_thinking(data)
    assert text == "hello"
    assert thinking == ""


def test_messages_text_and_thinking_splits_thinking_from_text():
    # OpenCode Go models reason by default: content[0] is a thinking block,
    # so the blind content[0]["text"] index used to raise KeyError here.
    data = {
        "content": [
            {"type": "thinking", "thinking": "let me consider..."},
            {"type": "text", "text": "the answer is 42"},
        ]
    }
    text, thinking = messages_text_and_thinking(data)
    assert text == "the answer is 42"
    assert thinking == "let me consider..."


def test_messages_text_and_thinking_handles_empty_content():
    text, thinking = messages_text_and_thinking({})
    assert text == ""
    assert thinking == ""

    text, thinking = messages_text_and_thinking({"content": []})
    assert text == ""
    assert thinking == ""


def test_messages_text_and_thinking_concatenates_multiple_text_blocks():
    data = {
        "content": [
            {"type": "thinking", "thinking": "step 1"},
            {"type": "text", "text": "part A "},
            {"type": "text", "text": "part B"},
        ]
    }
    text, thinking = messages_text_and_thinking(data)
    assert text == "part A part B"
    assert thinking == "step 1"


# -------------------------------------------------------------- vision blocks


def _png_ref():
    return ResolvedRef(
        path="a.png",
        mime="image/png",
        data=b"\x89PNG\r\n" + b"x" * 8,
        width=4,
        height=4,
        role="style",
    )


def test_vision_blocks_follow_the_model_wire_not_just_the_provider():
    ref = _png_ref()
    chat_blocks = build_vision_blocks(OPENCODE_GO, [ref], "glm-5.3")
    assert chat_blocks[0]["type"] == "image_url"

    messages_blocks = build_vision_blocks(OPENCODE_GO, [ref], "minimax-m3")
    assert messages_blocks[0]["type"] == "image"
    assert messages_blocks[0]["source"]["type"] == "base64"


def test_vision_blocks_default_to_the_provider_wire_without_a_model():
    ref = _png_ref()
    assert build_vision_blocks("anthropic", [ref])[0]["type"] == "image"
    assert build_vision_blocks("openrouter", [ref])[0]["type"] == "image_url"


# ------------------------------------------------------------ agent config


def test_agent_config_rejects_a_responses_only_model():
    with pytest.raises(ValidationError):
        AgentUpdate(llm_config={"provider": OPENCODE_GO, "model": "grok-4.6"})


def test_agent_config_rejects_a_responses_only_vision_model():
    with pytest.raises(ValidationError):
        AgentUpdate(llm_config={
            "provider": OPENCODE_GO,
            "model": "kimi-k3",
            "vision": {"enabled": True, "provider": OPENCODE_GO, "model": "gpt-5.6-luna"},
        })


def test_agent_config_accepts_supported_opencode_go_models():
    for model in ("kimi-k3", "minimax-m3", "glm-5.3"):
        agent = AgentUpdate(llm_config={"provider": OPENCODE_GO, "model": model})
        assert agent.llm_config["model"] == model
