"""SIWC protocol/security tests. All OpenAI requests are mocked; no user quota is spent."""
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from app.core.chatgpt_credentials import PROVIDER, parse_reference, runtime_reference
from app.core.chatgpt_oauth import ChatGPTError, SCOPES, saved_tokens
from app.core.chatgpt_responses import ChatGPTClient, chat_completion, consume_sse, llm_client, responses_payload

def test_runtime_reference_is_bound_and_not_forgeable():
    owner, secret = uuid4(), uuid4()
    reference = runtime_reference(owner, secret)
    assert parse_reference(reference.reveal()) == (owner, secret)
    with pytest.raises(ValueError):
        parse_reference(reference.reveal().replace(str(owner), str(uuid4())))
    assert "legion-chatgpt" not in str(reference)
    with pytest.raises(ChatGPTError, match="another provider"):
        llm_client("openai", api_key=reference.reveal())
    with pytest.raises(ValueError):
        ChatGPTClient(api_key="sk-some-service-key")

def test_only_direct_permission_with_offline_session_enables_inference():
    result = {"access_token": "access", "refresh_token": "refresh", "expires_in": 3600, "scope": SCOPES}
    saved = saved_tokens(result, client_id="oaiapp_test", subject="user", email="test@example.com")
    assert saved["client_id"] == "oaiapp_test"
    assert saved["expires_at"] > time.time()
    with pytest.raises(ChatGPTError, match="not granted"):
        saved_tokens({**result, "scope": "openid profile"}, client_id="oaiapp_test", subject="user", email=None)
    with pytest.raises(ChatGPTError, match="renewable"):
        saved_tokens({**result, "refresh_token": None}, client_id="oaiapp_test", subject="user", email=None)

def test_responses_request_preserves_history_tools_images_and_omits_preview_forbidden_fields():
    body = responses_payload({
        "model": "account-model",
        "messages": [{"role": "system", "content": "Be helpful"},
                     {"role": "user", "content": [{"type": "text", "text": "look"},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]},
                     {"role": "assistant", "content": None, "tool_calls": [
                       {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}]},
                     {"role": "tool", "tool_call_id": "call_1", "content": "file contents"}],
        "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "read_file"}},
        "temperature": 0.3, "max_tokens": 800, "metadata": {"s": "x"},
        "extra_body": {"session_id": "s", "prompt_cache_retention": "24h"},
    })
    assert body["input"][0]["role"] == "developer"
    assert body["input"][1]["content"][1]["type"] == "input_image"
    assert body["input"][2]["call_id"] == body["input"][3]["call_id"] == "call_1"
    assert body["tools"][0]["type"] == "namespace"
    assert body["tool_choice"] == "required"
    assert body["tools"][0]["tools"][0]["name"] == "read_file"
    assert body["store"] is False and body["stream"] is True
    assert not set(body) & {"temperature", "metadata", "max_tokens", "max_output_tokens", "previous_response_id", "extra_body"}

def completed(output=None):
    return {"type": "response.completed", "response": {"id": "resp_1", "output": output or [
        {"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}}}

@pytest.mark.asyncio
async def test_stream_requires_terminal_completion_and_maps_usage():
    response = httpx.Response(200, text="data: " + json.dumps(completed()) + "\n\n")
    result = await consume_sse(response, "account-model")
    assert result.choices[0].message.content == "done"
    assert result.usage.prompt_tokens == 12
    assert result.usage.completion_tokens == 3
    with pytest.raises(ChatGPTError, match="without response.completed"):
        await consume_sse(httpx.Response(200, text='data: {"type":"response.output_text.delta","delta":"partial"}\n\n'), "m")

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["response.failed", "response.incomplete", "error"])
async def test_stream_failure_after_text_is_not_success(kind):
    event = {"type": kind, "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}}
    with pytest.raises(ChatGPTError, match="usage limit"):
        await consume_sse(httpx.Response(200, text="data: " + json.dumps(event) + "\n\n"), "m")

def test_tools_return_in_existing_agent_loop_contract():
    response = completed([{"type": "function_call", "call_id": "call_1", "namespace": "legion",
                           "name": "close_session", "arguments": '{"ok":true}'}])["response"]
    result = chat_completion(response, "model")
    assert result.choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.tool_calls[0].function.name == "close_session"

def test_helper_rejects_insecure_remote_origin():
    from app.core.chatgpt_helper import validate_hub
    assert validate_hub("http://localhost:3000/") == "http://localhost:3000"
    assert validate_hub("https://hub.example.com") == "https://hub.example.com"
    for value in ("http://remote.example.com", "https://evil@hub.example.com", "https://hub.example.com/path", "file:///x"):
        with pytest.raises(ValueError):
            validate_hub(value)

@pytest.mark.asyncio
async def test_other_owner_cannot_inspect_or_modify_connection():
    from app.api.v1.chatgpt_auth import owned
    db = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(user_id=uuid4(), provider=PROVIDER)))
    with pytest.raises(HTTPException) as caught:
        await owned(db, uuid4(), uuid4())
    assert caught.value.status_code == 404

@pytest.mark.asyncio
async def test_helper_transaction_replay_expiry_and_state(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from app.api.v1 import chatgpt_auth as api
    row = SimpleNamespace(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    result = SimpleNamespace(scalar_one_or_none=lambda: row)
    db = SimpleNamespace(execute=AsyncMock(return_value=result))
    with pytest.raises(HTTPException) as caught:
        await api.pending(db, "x" * 43)
    assert caught.value.status_code == 410
    row.expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    row.redirect_uri = "http://127.0.0.1:12345/auth/callback"
    monkeypatch.setattr(api, "pending", AsyncMock(return_value=(row, {"state": "expected"})))
    with pytest.raises(HTTPException) as caught:
        await api.helper_complete(api.Complete(ticket="x" * 43, state="wrong", code="code"), db=db)
    assert caught.value.status_code == 400

@pytest.mark.asyncio
async def test_client_uses_only_official_responses_endpoint(monkeypatch):
    from app.core import chatgpt_responses as transport
    requests = []
    async def handler(request):
        requests.append(request)
        return httpx.Response(200, text="data: " + json.dumps(completed()) + "\n\n")
    original_client = httpx.AsyncClient
    monkeypatch.setattr(transport, "access_token", AsyncMock(return_value="oauth-access"))
    monkeypatch.setattr(transport.httpx, "AsyncClient",
                        lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    client = ChatGPTClient(api_key=runtime_reference(uuid4(), uuid4()).reveal(), base_url="https://evil.example")
    await client.chat.completions.create(model="model", messages=[{"role": "user", "content": "hello"}])
    assert str(requests[0].url) == "https://api.openai.com/v1/responses"
    assert requests[0].headers["authorization"] == "Bearer oauth-access"

@pytest.mark.asyncio
async def test_identity_rejects_wrong_nonce_audience_and_signature(monkeypatch):
    from app.core import chatgpt_oauth as oauth
    async def handler(request):
        if request.url.path.endswith("openid-configuration"):
            return httpx.Response(200, json={"issuer": oauth.ISSUER, "jwks_uri": oauth.ISSUER + "/jwks"})
        return httpx.Response(200, json={"keys": []})
    original_client = httpx.AsyncClient
    monkeypatch.setattr(oauth.httpx, "AsyncClient", lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    monkeypatch.setattr(oauth.jwt, "decode", lambda *a, **k: {"sub": "u", "nonce": "wrong"})
    with pytest.raises(ChatGPTError, match="verify"):
        await oauth.verify_identity("id-token", "oaiapp_test", "expected")
    def reject(*args, **kwargs):
        assert kwargs["audience"] == "oaiapp_test"
        assert kwargs["issuer"] == oauth.ISSUER
        raise ValueError("invalid signature")
    monkeypatch.setattr(oauth.jwt, "decode", reject)
    with pytest.raises(ChatGPTError):
        await oauth.verify_identity("bad-signature", "oaiapp_test", "expected")

@pytest.mark.asyncio
async def test_helper_source_is_packaged_with_backend():
    from app.api.v1.chatgpt_auth import helper_source
    response = await helper_source(user=SimpleNamespace(id=uuid4()))
    assert b"LEGION_SETUP = None" in response.body
    assert b'class NoRedirect' in response.body

@pytest.mark.asyncio
async def test_admission_failure_preserves_code_shape_and_request_id_without_retrying_quota(monkeypatch):
    from app.core import chatgpt_responses as transport
    calls = []
    async def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"x-request-id": "req-test"},
                              json={"error": {"code": "subscription_sharing_usage_limit_exceeded", "param": None}})
    original_client = httpx.AsyncClient
    monkeypatch.setattr(transport, "access_token", AsyncMock(return_value="access"))
    monkeypatch.setattr(transport.httpx, "AsyncClient",
                        lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    client = ChatGPTClient(api_key=runtime_reference(uuid4(), uuid4()).reveal())
    with pytest.raises(ChatGPTError) as caught:
        await client.create(model="m", messages=[])
    assert caught.value.status_code == 429
    assert caught.value.code == "subscription_sharing_usage_limit_exceeded"
    assert caught.value.request_id == "req-test"
    assert caught.value.response_body["error"]["param"] is None
    assert len(calls) == 1

@pytest.mark.asyncio
async def test_transient_admission_retries_are_bounded(monkeypatch):
    from app.core import chatgpt_responses as transport
    calls = []
    async def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"detail": "direct routing unavailable"})
    original_client = httpx.AsyncClient
    monkeypatch.setattr(transport, "access_token", AsyncMock(return_value="access"))
    monkeypatch.setattr(transport.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(transport.httpx, "AsyncClient",
                        lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
    client = ChatGPTClient(api_key=runtime_reference(uuid4(), uuid4()).reveal())
    with pytest.raises(ChatGPTError) as caught:
        await client.create(model="m", messages=[])
    assert len(calls) == 3
    assert caught.value.response_body == {"detail": "direct routing unavailable"}

def test_subscription_provider_is_paused_at_runtime_and_cannot_be_created():
    from pydantic import ValidationError
    from app.api.v1.provider_connections import ConnectionCreate
    reference = runtime_reference(uuid4(), uuid4()).reveal()
    with pytest.raises(ChatGPTError, match="temporarily unavailable"):
        llm_client(PROVIDER, api_key=reference)
    with pytest.raises(ValidationError):
        ConnectionCreate(provider=PROVIDER)
    assert ConnectionCreate(provider="openai").provider == "openai"
