"""Provider wire-protocol resolution shared by every LLM call site.

Each provider speaks one of three wire protocols:

* ``chat``      — OpenAI ``/chat/completions`` (the ``AsyncOpenAI`` client)
* ``messages``  — Anthropic ``/v1/messages`` (hand-rolled ``httpx`` calls)
* ``responses`` — OpenAI ``/responses`` (ChatGPT subscription adapter)

Until now the mapping provider -> base URL and provider -> wire was duplicated
inline at every call site (``proxy_session``, ``user_sessions``, ``chat_master``,
``plan_session``, ``router``, ``vision_dispatch``).  That worked while the wire
was decided by the provider alone.  OpenCode Go is a gateway fronting several
vendors that picks the wire **per model** — MiniMax and Qwen answer on
``/v1/messages`` while GLM, Kimi and DeepSeek answer on ``/v1/chat/completions``
— so the decision no longer fits into ``if provider == "anthropic"``.

For every pre-existing provider these helpers return exactly the values that
were hardcoded before; only ``opencode_go`` is new behaviour.
"""

from __future__ import annotations

import os
from typing import Literal
from urllib.parse import urlparse

Wire = Literal["chat", "messages", "responses"]

WIRE_CHAT: Wire = "chat"
WIRE_MESSAGES: Wire = "messages"
WIRE_RESPONSES: Wire = "responses"

OPENCODE_GO = "opencode_go"
OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Default base URLs for OpenAI-compatible providers.  A provider missing here
# falls back to the OpenAI default, which is what the inline chains these
# helpers replace did by leaving ``base_url`` unset on the client.
_OPENAI_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://host.docker.internal:11434/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    OPENCODE_GO: OPENCODE_GO_BASE_URL,
}

# Providers whose wire is Anthropic's /v1/messages for every model.
_MESSAGES_PROVIDERS = {"anthropic"}


class OpencodeGoModel:
    """One entry of the OpenCode Go catalog."""

    __slots__ = ("id", "name", "wire")

    def __init__(self, model_id: str, name: str, wire: Wire) -> None:
        self.id = model_id
        self.name = name
        self.wire = wire


# Catalog published at https://opencode.ai/zen/go — the endpoint column of that
# table is the wire.  Availability is confirmed live against /v1/models during
# provider validation; this table only supplies display names and wires.
OPENCODE_GO_CATALOG: tuple[OpencodeGoModel, ...] = (
    OpencodeGoModel("grok-4.6", "Grok 4.6", WIRE_RESPONSES),
    OpencodeGoModel("grok-4.5", "Grok 4.5", WIRE_RESPONSES),
    OpencodeGoModel("gpt-5.6-luna", "GPT 5.6 Luna", WIRE_RESPONSES),
    OpencodeGoModel("muse-spark-1.2-contributor", "Muse Spark 1.2 Contributor", WIRE_RESPONSES),
    OpencodeGoModel("glm-5.3-flash", "GLM-5.3-Flash", WIRE_CHAT),
    OpencodeGoModel("glm-5.3", "GLM-5.3", WIRE_CHAT),
    OpencodeGoModel("glm-5.2", "GLM-5.2", WIRE_CHAT),
    OpencodeGoModel("glm-5.1", "GLM-5.1", WIRE_CHAT),
    OpencodeGoModel("glm-5", "GLM-5", WIRE_CHAT),
    OpencodeGoModel("kimi-k3", "Kimi K3", WIRE_CHAT),
    OpencodeGoModel("kimi-k2.7-code", "Kimi K2.7 Code", WIRE_CHAT),
    OpencodeGoModel("kimi-k2.6", "Kimi K2.6", WIRE_CHAT),
    OpencodeGoModel("kimi-k2.5", "Kimi K2.5", WIRE_CHAT),
    OpencodeGoModel("longcat-2.0", "LongCat-2.0", WIRE_CHAT),
    OpencodeGoModel("deepseek-v4-pro", "DeepSeek V4 Pro", WIRE_CHAT),
    OpencodeGoModel("deepseek-v4-flash", "DeepSeek V4 Flash", WIRE_CHAT),
    OpencodeGoModel("deepseek-v4-flash-vision-exp", "DeepSeek V4 Flash Vision Exp", WIRE_CHAT),
    OpencodeGoModel("mimo-v2.5", "MiMo-V2.5", WIRE_CHAT),
    OpencodeGoModel("mimo-v2.5-pro", "MiMo-V2.5-Pro", WIRE_CHAT),
    OpencodeGoModel("mimo-v2-pro", "MiMo-V2-Pro", WIRE_CHAT),
    OpencodeGoModel("hy3", "Hy3", WIRE_CHAT),
    OpencodeGoModel("hy3-preview", "Hy3 Preview", WIRE_CHAT),
    OpencodeGoModel("minimax-m3", "MiniMax M3", WIRE_MESSAGES),
    OpencodeGoModel("minimax-m2.7", "MiniMax M2.7", WIRE_MESSAGES),
    OpencodeGoModel("minimax-m2.5", "MiniMax M2.5", WIRE_MESSAGES),
    OpencodeGoModel("qwen3.8-max", "Qwen3.8 Max", WIRE_MESSAGES),
    OpencodeGoModel("qwen3.7-max", "Qwen3.7 Max", WIRE_MESSAGES),
    OpencodeGoModel("qwen3.7-plus", "Qwen3.7 Plus", WIRE_MESSAGES),
    OpencodeGoModel("qwen3.6-plus", "Qwen3.6 Plus", WIRE_MESSAGES),
    OpencodeGoModel("qwen3.5-plus", "Qwen3.5 Plus", WIRE_MESSAGES),
)

_OPENCODE_GO_BY_ID = {entry.id: entry for entry in OPENCODE_GO_CATALOG}

# The gateway ships point releases between catalog updates (it served kimi-k2.5
# before the published table listed it).  Family prefixes keep an unlisted model
# on the right wire instead of defaulting it to chat and 404-ing at first use.
_OPENCODE_GO_FAMILIES: tuple[tuple[str, Wire], ...] = (
    ("minimax-", WIRE_MESSAGES),
    ("qwen", WIRE_MESSAGES),
    ("grok", WIRE_RESPONSES),
    ("gpt-", WIRE_RESPONSES),
    ("muse-", WIRE_RESPONSES),
)


def normalize_provider(provider: str | None) -> str:
    """Lowercase a provider id.

    An empty id stays empty rather than defaulting to Anthropic: every call site
    already applies its own default, and guessing here would silently point a
    misconfigured agent at api.anthropic.com.
    """
    return (provider or "").lower()


def opencode_go_wire(model: str | None) -> Wire:
    """Wire of one OpenCode Go model (catalog first, then family prefix)."""
    model_id = (model or "").strip().lower()
    entry = _OPENCODE_GO_BY_ID.get(model_id)
    if entry is not None:
        return entry.wire
    for prefix, wire in _OPENCODE_GO_FAMILIES:
        if model_id.startswith(prefix):
            return wire
    return WIRE_CHAT


def provider_wire(provider: str | None, model: str | None = None) -> Wire:
    """Wire protocol a (provider, model) pair speaks."""
    name = normalize_provider(provider)
    if name == "openai_chatgpt":
        return WIRE_RESPONSES
    if name in _MESSAGES_PROVIDERS:
        return WIRE_MESSAGES
    if name == OPENCODE_GO:
        return opencode_go_wire(model)
    return WIRE_CHAT


def uses_messages_wire(provider: str | None, model: str | None = None) -> bool:
    """True when the call must use Anthropic's /v1/messages request shape."""
    return provider_wire(provider, model) == WIRE_MESSAGES


def is_unsupported_wire(provider: str | None, model: str | None = None) -> bool:
    """True for Responses models without an enabled Hub transport."""
    return normalize_provider(provider) != "openai_chatgpt" and provider_wire(provider, model) == WIRE_RESPONSES


def openai_base_url(provider: str | None, configured: str | None = None) -> str:
    """Base URL for an OpenAI-compatible call; a configured URL always wins."""
    if configured:
        return configured.rstrip("/")
    return _OPENAI_BASE_URLS.get(normalize_provider(provider), OPENAI_DEFAULT_BASE_URL)


OPENROUTER_APP_URL_DEFAULT = "https://mylegion.online"
OPENROUTER_APP_TITLE_DEFAULT = "Legion Hub"


def openrouter_attribution_headers(url: str | None) -> dict[str, str]:
    """OpenRouter App Attribution headers for a call to ``url``; ``{}`` elsewhere.

    OpenRouter files traffic under the app named by ``HTTP-Referer`` — without
    it every Hub call shows up as "unknown" in logs and rankings.  The headers
    go only to openrouter.ai: other providers have no use for them.

    Self-hosted installs report under the same app by default.  Override with
    ``OPENROUTER_APP_URL`` / ``OPENROUTER_APP_TITLE``; an empty value turns the
    headers off.  Read per call so a changed environment needs no restart.
    """
    host = (urlparse(url or "").hostname or "").lower()
    if host != "openrouter.ai" and not host.endswith(".openrouter.ai"):
        return {}
    app_url = os.environ.get("OPENROUTER_APP_URL", OPENROUTER_APP_URL_DEFAULT).strip()
    if not app_url:
        return {}
    headers = {"HTTP-Referer": app_url}
    title = os.environ.get("OPENROUTER_APP_TITLE", OPENROUTER_APP_TITLE_DEFAULT).strip()
    if title:
        headers["X-OpenRouter-Title"] = title
        headers["X-Title"] = title  # legacy name, still honoured
    return headers


def openai_client_extras(base_url: str | None) -> dict[str, dict[str, str]]:
    """``AsyncOpenAI`` kwargs carrying the attribution headers, ``{}`` when none apply."""
    headers = openrouter_attribution_headers(base_url)
    return {"default_headers": headers} if headers else {}


def messages_url(provider: str | None, configured: str | None = None) -> str:
    """URL of the /messages endpoint for a messages-wire provider."""
    if configured:
        return configured.rstrip("/") + "/messages"
    if normalize_provider(provider) == OPENCODE_GO:
        return f"{OPENCODE_GO_BASE_URL}/messages"
    return f"{ANTHROPIC_BASE_URL}/messages"


def messages_headers(provider: str | None, api_key: str | None) -> dict[str, str]:
    """Auth headers for a /messages call.

    Both Anthropic and the OpenCode gateway authenticate this endpoint with
    ``x-api-key``.  OpenCode's *other* endpoints (``/chat/completions``,
    ``/responses``) take ``Authorization: Bearer`` instead — auth on that
    gateway is per-endpoint, not per-provider.  Measured 27.08.2026: a bearer
    token on ``/zen/go/v1/messages`` returns 401 ``AuthError: Missing API key``.
    """
    return {
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
        "x-api-key": api_key or "",
    }


def messages_text_and_thinking(data: dict) -> tuple[str, str]:
    """Split an Anthropic-shaped reply into (visible text, thinking text).

    With thinking enabled ``content[0]`` is a ``thinking`` block, so indexing
    content[0]["text"] raises KeyError — three call sites did exactly that
    before this helper existed.  OpenCode Go models reason by default, which is
    what made those sites fail on the first thinking reply.
    """
    blocks = data.get("content") or []
    text = "".join(
        b.get("text", "") for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )
    thinking = "".join(
        b.get("thinking", "") for b in blocks
        if isinstance(b, dict) and b.get("type") == "thinking"
    )
    return text, thinking


def uses_anthropic_env_key(provider: str | None) -> bool:
    """Whether ANTHROPIC_API_KEY is a legitimate fallback for this provider.

    Only Anthropic itself: a gateway key is never interchangeable with a
    first-party one, and silently sending the wrong key is worse than failing.
    """
    return normalize_provider(provider) == "anthropic"
