"""Adapt the Hub's chat/tool contract to SIWC's stateless Responses SSE protocol."""
from __future__ import annotations
import asyncio
import json
import time
from types import SimpleNamespace
import httpx
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from app.core.chatgpt_credentials import PREFIX, PROVIDER, parse_reference
from app.core.chatgpt_oauth import ChatGPTError, RESOURCE, access_token

def responses_payload(kwargs: dict) -> dict:
    items = []
    for message in kwargs.get("messages", []):
        role = message["role"]
        content = message.get("content")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": content or ""})
            continue
        if role in {"system", "developer"}:
            items.append({"role": "developer", "content": content or ""})
        elif content:
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "output_text" if role == "assistant" else "input_text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        image = part["image_url"]
                        parts.append({"type": "input_image", "image_url": image["url"], "detail": image.get("detail", "auto")})
                    else:
                        raise ChatGPTError("This content type is not supported by the ChatGPT plan connection")
                content = parts
            items.append({"role": role, "content": content})
        for call in message.get("tool_calls", []):
            items.append({"type": "function_call", "call_id": call["id"], "name": call["function"]["name"],
                          "namespace": "legion", "arguments": call["function"]["arguments"]})
    body = {"model": kwargs["model"], "input": items, "store": False, "stream": True}
    tools = kwargs.get("tools") or []
    if tools:
        body["tools"] = [{"type": "namespace", "name": "legion", "description": "Legion agent tools",
                          "tools": [{"type": "function", **tool["function"], "strict": False} for tool in tools]}]
        choice = kwargs.get("tool_choice", "auto")
        if isinstance(choice, dict):
            # ToolChoiceFunction has no namespace field. Limit this request's
            # namespace to the forced function, then use the supported required mode.
            name = choice["function"]["name"]
            selected = [tool for tool in body["tools"][0]["tools"] if tool["name"] == name]
            if not selected:
                raise ChatGPTError("The requested agent tool is not available")
            body["tools"][0]["tools"] = selected
            choice = "required"
        body["tool_choice"] = choice
    extra = kwargs.get("extra_body") or {}
    effort = kwargs.get("reasoning_effort") or extra.get("reasoning_effort")
    if effort:
        body["reasoning"] = {"effort": effort}
    format_spec = kwargs.get("response_format")
    if format_spec:
        if format_spec["type"] == "json_schema":
            body["text"] = {"format": {"type": "json_schema", **format_spec["json_schema"]}}
        else:
            body["text"] = {"format": format_spec}
    return body

def chat_completion(response: dict, model: str) -> ChatCompletion:
    output = response.get("output") or []
    content = "".join(part.get("text", "") if part.get("type") == "output_text" else part.get("refusal", "")
                      for item in output if item.get("type") == "message"
                      for part in item.get("content", []) if part.get("type") in {"output_text", "refusal"})
    calls = []
    for item in output:
        if item.get("type") == "function_call":
            if item.get("namespace") not in (None, "legion"):
                raise ChatGPTError("OpenAI returned an unknown tool namespace")
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
    usage = response.get("usage") or {}
    message = {"role": "assistant", "content": content or None}
    if calls:
        message["tool_calls"] = calls
    return ChatCompletion.model_validate({
        "id": response["id"], "object": "chat.completion", "created": int(time.time()), "model": model,
        "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}],
        "usage": {"prompt_tokens": usage.get("input_tokens", 0), "completion_tokens": usage.get("output_tokens", 0),
                  "total_tokens": usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
                  "prompt_tokens_details": usage.get("input_tokens_details"),
                  "completion_tokens_details": usage.get("output_tokens_details")}})

def stream_error(event: dict) -> ChatGPTError:
    error = event.get("response", {}).get("error") or event.get("error") or (event if event.get("code") else {})
    code = error.get("code") if isinstance(error, dict) else None
    messages = {
        "subscription_sharing_usage_limit_exceeded": "ChatGPT plan usage limit reached. Review ChatGPT Settings → Usage.",
        "subscription_sharing_user_not_eligible": "This ChatGPT account or workspace is not eligible for plan usage.",
        "subscription_sharing_usage_unavailable": "ChatGPT usage is temporarily unavailable. Try again later.",
    }
    return ChatGPTError(messages.get(code, f"ChatGPT response did not complete ({code or event.get('type', 'interrupted')})"),
                        code=code, response_body=event,
                        retryable=code in {"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"})

async def consume_sse(response: httpx.Response, model: str) -> ChatCompletion:
    data = []
    completed = None
    async def consume():
        nonlocal completed
        if not data:
            return
        event = json.loads("\n".join(data))
        if event.get("type") in {"error", "response.failed", "response.incomplete"}:
            raise stream_error(event)
        if event.get("type") == "response.completed":
            completed = chat_completion(event["response"], model)
    async for line in response.aiter_lines():
        if line == "":
            await consume()
            data = []
        elif line.startswith("data:"):
            fragment = line[5:].lstrip()
            if fragment != "[DONE]":
                data.append(fragment)
    await consume()
    if completed is None:
        raise ChatGPTError("ChatGPT stream ended without response.completed. The response is incomplete.")
    return completed

class ChatGPTClient:
    def __init__(self, *, api_key: str, timeout=120, **_):
        parse_reference(api_key)  # Never accept a service key or an unbound bearer.
        self.reference = api_key
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        for attempt in range(3):
            try:
                return await self._create_once(**kwargs)
            except ChatGPTError as exc:
                if not exc.retryable or attempt == 2:
                    raise
            except httpx.TransportError:
                if attempt == 2:
                    raise ChatGPTError("ChatGPT connection is temporarily unavailable. Try again later.") from None
            await asyncio.sleep(2 ** attempt)

    async def _create_once(self, **kwargs):
        bearer = await access_token(self.reference)
        body = responses_payload(kwargs)
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            async with client.stream("POST", f"{RESOURCE}/responses",
                                     headers={"Authorization": f"Bearer {bearer}", "Content-Type": "application/json"},
                                     json=body) as response:
                if response.status_code != 200:
                    await response.aread()
                    request_id = response.headers.get("x-request-id", "")
                    try:
                        body = response.json()
                    except ValueError:
                        body = {"detail": response.text}
                    error = body.get("error") if isinstance(body, dict) else None
                    code = error.get("code") if isinstance(error, dict) else None
                    message = f"ChatGPT request failed (HTTP {response.status_code}). Request: {request_id}"
                    if response.status_code in (401, 403):
                        message = f"ChatGPT access denied (HTTP {response.status_code}). Check account permissions. Request: {request_id}"
                    if response.status_code == 429:
                        message = "ChatGPT usage limit reached. Review ChatGPT Settings → Usage."
                    raise ChatGPTError(message, status=response.status_code, code=code,
                                       request_id=request_id, response_body=body,
                                       retryable=response.status_code in {500, 502, 503, 504})
                return await consume_sse(response, kwargs["model"])

def llm_client(provider: str, *, chat_client_factory=AsyncOpenAI, **kwargs):
    if provider == PROVIDER:
        # Paused: internal design notes.
        # return ChatGPTClient(**kwargs)
        raise ChatGPTError("ChatGPT subscription connections are temporarily unavailable")
    if str(kwargs.get("api_key", "")).startswith(PREFIX):
        raise ChatGPTError("A ChatGPT connection cannot be used as an API key for another provider")
    return chat_client_factory(**kwargs)
