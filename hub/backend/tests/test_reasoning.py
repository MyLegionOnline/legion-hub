"""Reasoning capability layer (plan stage 3 / R3): request fields, extraction,
probe and its cache.

The two facts this file pins hardest, both measured live on the OpenCode Go
gateway (plan 0.6): a per-model off-switch either works or is silently
ignored (`deepseek-v4-flash-vision-exp` vs `glm-5.3` on the same gateway), and
effort=None must keep outgoing request bodies byte-identical to pre-feature
behaviour (decision 0.10 p.4).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio

from app.core.llm_providers import OPENCODE_GO, OPENCODE_GO_BASE_URL, WIRE_CHAT, WIRE_MESSAGES, WIRE_RESPONSES
from app.core.reasoning import (
    ReasoningProbeError,
    ReasoningSupport,
    probe_reasoning_support,
    reasoning_from_chat_choice,
    reasoning_request_fields,
)

# ---------------------------------------------------------- request fields


def test_off_switch_on_messages_wire_is_the_anthropic_shape():
    # Acceptance criterion 4.4, verbatim.
    assert reasoning_request_fields(
        OPENCODE_GO, "qwen3.7-plus", WIRE_MESSAGES, "off"
    ) == {"thinking": {"type": "disabled"}}


def test_effort_none_means_provider_default_for_any_arguments():
    # Acceptance criterion 4.4: an agent configured before this feature keeps
    # behaving exactly as it did — nothing is added to the request body.
    for wire in (WIRE_CHAT, WIRE_MESSAGES, WIRE_RESPONSES):
        for provider in (OPENCODE_GO, "anthropic", "openrouter", None):
            assert reasoning_request_fields(provider, "whatever", wire, None) == {}


def test_messages_budgets_map_to_the_plan_table():
    assert reasoning_request_fields(OPENCODE_GO, "qwen3.7-plus", WIRE_MESSAGES, "low") == {
        "thinking": {"type": "enabled", "budget_tokens": 1024}
    }
    assert reasoning_request_fields(OPENCODE_GO, "qwen3.7-plus", WIRE_MESSAGES, "medium") == {
        "thinking": {"type": "enabled", "budget_tokens": 4096}
    }
    assert reasoning_request_fields(OPENCODE_GO, "qwen3.7-plus", WIRE_MESSAGES, "high") == {
        "thinking": {"type": "enabled", "budget_tokens": 16384}
    }


def test_budget_is_clamped_under_max_tokens():
    # 4.1: budget must be strictly below max_tokens — 2048 - 512 = 1536.
    fields = reasoning_request_fields(
        OPENCODE_GO, "qwen3.7-plus", WIRE_MESSAGES, "high", max_tokens=2048
    )
    assert fields == {"thinking": {"type": "enabled", "budget_tokens": 1536}}


def test_budget_below_the_floor_is_not_requested_at_all():
    # 1500 - 512 = 988 < 1024: requesting thinking would 400 the whole call,
    # so the call proceeds at the provider default instead.
    assert reasoning_request_fields(
        OPENCODE_GO, "qwen3.7-plus", WIRE_MESSAGES, "low", max_tokens=1500
    ) == {}


def test_off_effort_never_touches_max_tokens():
    # 5.3: at effort="off" nothing must be done to max_tokens — the disabled
    # switch is safe at any cap size.
    assert reasoning_request_fields(
        OPENCODE_GO, "qwen3.7-plus", WIRE_MESSAGES, "off", max_tokens=64
    ) == {"thinking": {"type": "disabled"}}


def test_chat_wire_uses_reasoning_effort():
    assert reasoning_request_fields(OPENCODE_GO, "glm-5.3", WIRE_CHAT, "off") == {
        "reasoning_effort": "none"
    }
    assert reasoning_request_fields(OPENCODE_GO, "glm-5.3", WIRE_CHAT, "high") == {
        "reasoning_effort": "high"
    }


def test_responses_wire_gets_no_guess():
    # No Responses transport exists (internal design notes); a chat-shaped
    # knob onto that wire would be the hardcoding this module prevents.
    assert reasoning_request_fields(OPENCODE_GO, "grok-4.6", WIRE_RESPONSES, "off") == {}
    assert reasoning_request_fields(OPENCODE_GO, "grok-4.6", WIRE_RESPONSES, "medium") == {}


# ------------------------------------------------------- reply extraction


def test_reasoning_from_chat_choice_reads_sdk_objects():
    message = SimpleNamespace(reasoning_content="pondering…")
    usage = SimpleNamespace(
        completion_tokens_details=SimpleNamespace(reasoning_tokens=762)
    )
    assert reasoning_from_chat_choice(message, usage) == ("pondering…", 762)


def test_reasoning_from_chat_choice_reads_openrouter_style_reasoning():
    message = SimpleNamespace(reasoning="openrouter thinking")
    assert reasoning_from_chat_choice(message, None) == ("openrouter thinking", 0)


def test_reasoning_from_chat_choice_reads_plain_dicts():
    # The probe parses raw JSON; call sites parse the SDK — one helper, both.
    message = {"reasoning_content": "hmm"}
    usage = {"completion_tokens_details": {"reasoning_tokens": 42}}
    assert reasoning_from_chat_choice(message, usage) == ("hmm", 42)


def test_reasoning_from_chat_choice_tolerates_missing_everything():
    assert reasoning_from_chat_choice({}, {}) == ("", 0)
    assert reasoning_from_chat_choice(None, None) == ("", 0)
    assert reasoning_from_chat_choice({}, {"completion_tokens_details": None}) == ("", 0)


def test_reasoning_from_chat_choice_tolerates_garbage_token_counts():
    assert reasoning_from_chat_choice({}, {"completion_tokens_details": {"reasoning_tokens": "many"}}) == ("", 0)


# --------------------------------------------------------------- the probe


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    @property
    def text(self) -> str:
        return str(self._payload)

    def json(self) -> Any:
        return self._payload


def _chat_payload(reasoning_tokens: int, content: str = "ok") -> dict:
    message: dict[str, Any] = {"content": content}
    if reasoning_tokens:
        message["reasoning_content"] = "internal deliberation"
    return {
        "choices": [{"message": message}],
        "usage": {
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}
        },
    }


def _messages_payload(thinking: bool, text: str = "ok") -> dict:
    blocks: list[dict[str, Any]] = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": "deliberating"})
    blocks.append({"type": "text", "text": text})
    return {"content": blocks, "usage": {"input_tokens": 10, "output_tokens": 5}}


class _FakeClient:
    """Stands in for httpx.AsyncClient inside the probe."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.posts: list[tuple[str, dict, dict]] = []
        self.closed = False

    async def post(self, url: str, headers: dict | None = None, json: dict | None = None):
        self.posts.append((url, headers or {}, json or {}))
        return self._responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_probe_reports_controllable_for_deepseek_style_chat_model():
    # Acceptance 4.4 (mocked half; live half in test_reasoning_live.py):
    # baseline reasons, off-switch zeroes it → controllable with a switch.
    client = _FakeClient([
        _FakeResponse(200, _chat_payload(862)),
        _FakeResponse(200, _chat_payload(0)),
    ])

    support = await probe_reasoning_support(
        provider=OPENCODE_GO, model="deepseek-v4-flash-vision-exp",
        api_key="oc-1", client=client,
    )

    assert support == ReasoningSupport(
        supported=True, controllable=True,
        off_switch={"reasoning_effort": "none"},
    )
    # A borrowed client is never closed by the probe.
    assert client.closed is False
    assert len(client.posts) == 2

    url, headers, baseline_body = client.posts[0]
    assert url == f"{OPENCODE_GO_BASE_URL}/chat/completions"
    assert headers["Authorization"] == "Bearer oc-1"
    # 0.7: the chat wire sends no max_tokens — the probe must not either.
    assert "max_tokens" not in baseline_body
    assert baseline_body["model"] == "deepseek-v4-flash-vision-exp"

    _url, _headers, off_body = client.posts[1]
    assert off_body["reasoning_effort"] == "none"


@pytest.mark.asyncio
async def test_probe_reports_uncontrollable_for_glm_style_chat_model():
    # The always-reasoning class (plan 0.6): baseline reasons, the off call
    # still reasons → controllable=False, off_switch=None.  glm-5.3 was the
    # measured example on 27.08; the gateway started honouring its switch by
    # 31.08 and glm-5.2 (the current ChatMaster model) is today's live member
    # of this class — the probe decides per model, never per family.
    client = _FakeClient([
        _FakeResponse(200, _chat_payload(762)),
        _FakeResponse(200, _chat_payload(781)),
    ])

    support = await probe_reasoning_support(
        provider=OPENCODE_GO, model="glm-5.2", api_key="oc-1", client=client,
    )

    assert support == ReasoningSupport(
        supported=True, controllable=False, off_switch=None,
    )


@pytest.mark.asyncio
async def test_probe_authenticates_messages_wire_with_x_api_key():
    # The stage-3 probe relies on stage 0 (R0): /messages authenticates with
    # x-api-key, so probing messages-wire models must use messages_headers.
    client = _FakeClient([
        _FakeResponse(200, _messages_payload(thinking=True)),
        _FakeResponse(200, _messages_payload(thinking=False)),
    ])

    support = await probe_reasoning_support(
        provider=OPENCODE_GO, model="qwen3.7-plus", api_key="oc-1", client=client,
    )

    assert support == ReasoningSupport(
        supported=True, controllable=True,
        off_switch={"thinking": {"type": "disabled"}},
    )

    url, headers, baseline_body = client.posts[0]
    assert url == f"{OPENCODE_GO_BASE_URL}/messages"
    assert headers["x-api-key"] == "oc-1"
    assert "Authorization" not in headers
    # Anthropic-shaped calls need max_tokens; 2048 leaves room for the enabled
    # probe budget (1024) under the clamp formula.
    assert baseline_body["max_tokens"] == 2048

    _url, _headers, off_body = client.posts[1]
    assert off_body["thinking"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_probe_detects_opt_in_reasoners_with_a_third_call():
    # minimax-m3 stays silent by default (measured 31.08.2026) — a two-call
    # probe would call it unsupported.  The third, thinking-on call decides.
    client = _FakeClient([
        _FakeResponse(200, _messages_payload(thinking=False)),
        _FakeResponse(200, _messages_payload(thinking=False)),
        _FakeResponse(200, _messages_payload(thinking=True)),
    ])

    support = await probe_reasoning_support(
        provider=OPENCODE_GO, model="minimax-m3", api_key="oc-1", client=client,
    )

    assert support == ReasoningSupport(
        supported=True, controllable=True,
        off_switch={"thinking": {"type": "disabled"}},
    )
    _url, _headers, on_body = client.posts[2]
    assert on_body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert on_body["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_probe_reports_unsupported_for_a_model_that_never_thinks():
    client = _FakeClient([
        _FakeResponse(200, _messages_payload(thinking=False)),
        _FakeResponse(200, _messages_payload(thinking=False)),
        _FakeResponse(200, _messages_payload(thinking=False)),
    ])

    support = await probe_reasoning_support(
        provider=OPENCODE_GO, model="minimax-m3", api_key="oc-1", client=client,
    )

    assert support == ReasoningSupport(
        supported=False, controllable=False, off_switch=None,
    )


@pytest.mark.asyncio
async def test_probe_treats_a_rejected_off_switch_as_uncontrollable():
    # A model whose endpoint 400s on the off parameter is exactly as
    # uncontrollable as one that ignores it — measured, not assumed.
    client = _FakeClient([
        _FakeResponse(200, _chat_payload(500)),
        _FakeResponse(400, {"error": {"message": "unknown parameter"}}),
    ])

    support = await probe_reasoning_support(
        provider=OPENCODE_GO, model="some-model", api_key="oc-1", client=client,
    )

    assert support == ReasoningSupport(
        supported=True, controllable=False, off_switch=None,
    )


@pytest.mark.asyncio
async def test_probe_raises_when_the_model_cannot_be_called_at_all():
    # A dead baseline (403 RegionError etc.) means no measurable capability —
    # the caller must show the error, never cache a guess as an answer.
    client = _FakeClient([
        _FakeResponse(403, {"error": {"type": "RegionError", "message": "opt in"}}),
    ])

    with pytest.raises(ReasoningProbeError) as excinfo:
        await probe_reasoning_support(
            provider=OPENCODE_GO, model="deepseek-v4-flash",
            api_key="oc-1", client=client,
        )
    assert excinfo.value.status_code == 403
    assert len(client.posts) == 1


@pytest.mark.asyncio
async def test_probe_skips_responses_wire_without_any_network():
    support = await probe_reasoning_support(
        provider=OPENCODE_GO, model="grok-4.6", api_key="oc-1",
        client=_FakeClient([]),  # any post would pop from an empty list and fail
    )

    assert support == ReasoningSupport(False, False, None)


@pytest.mark.asyncio
async def test_probe_closes_a_client_it_created_but_not_a_borrowed_one(monkeypatch):
    import app.core.reasoning as reasoning_mod

    client = _FakeClient([_FakeResponse(200, _chat_payload(0))])
    # Baseline shows no reasoning → an own-client probe also runs off+on calls.
    client._responses.extend([
        _FakeResponse(200, _chat_payload(0)),
        _FakeResponse(200, _chat_payload(0)),
    ])
    monkeypatch.setattr(
        reasoning_mod.httpx, "AsyncClient", lambda *a, **kw: client, raising=True
    )

    await probe_reasoning_support(
        provider=OPENCODE_GO, model="some-model", api_key="oc-1",
    )

    assert client.closed is True


# ------------------------------------------------------- cache (PostgreSQL)

DB_URL = os.environ.get("DATABASE_URL", "")
pytestmark_db = pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")

# The cache tests run against whatever PostgreSQL DATABASE_URL points at —
# in the deployment container that is the production `somemaru`.  They use a
# provider id that no real provider can ever have, so they never overwrite a
# live probe result (e.g. opencode_go/glm-5.3) with test values.
_TEST_PROVIDER = "capability-cache-test"


@pytestmark_db
@pytest.mark.asyncio
@pytest.mark.db
class TestCapabilityCache:
    @pytest_asyncio.fixture
    async def db_session(self):
        from sqlalchemy import delete
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker
        from sqlmodel import SQLModel

        import app.models  # noqa: F401  (populate metadata)
        from app.models.model_capability import ModelCapability

        engine = create_async_engine(DB_URL, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with factory() as session:
            yield session
            # Leave no rows behind: the table is shared with live probe data.
            await session.execute(
                delete(ModelCapability).where(
                    ModelCapability.provider == _TEST_PROVIDER
                )
            )
            await session.commit()

        await engine.dispose()

    async def test_save_then_read_roundtrip(self, db_session):
        from app.core.reasoning import (
            get_cached_reasoning_support,
            save_reasoning_support,
        )

        support = ReasoningSupport(True, True, {"thinking": {"type": "disabled"}})
        await save_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-a", support=support
        )

        cached = await get_cached_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-a"
        )
        assert cached == support

    async def test_provider_is_normalized_before_the_cache_lookup(self, db_session):
        from app.core.reasoning import save_reasoning_support

        await save_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-a",
            support=ReasoningSupport(True, True, None),
        )

        from app.core.reasoning import get_cached_reasoning_support

        cached = await get_cached_reasoning_support(
            db_session, provider="Capability-Cache-Test", model="model-a"
        )
        assert cached is not None

    async def test_expired_row_is_ignored(self, db_session):
        from app.core.reasoning import save_reasoning_support
        from app.models.model_capability import ModelCapability

        await save_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-a",
            support=ReasoningSupport(True, True, None),
        )
        row = await db_session.get(ModelCapability, (_TEST_PROVIDER, "model-a"))
        row.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        await db_session.commit()

        from app.core.reasoning import get_cached_reasoning_support

        assert await get_cached_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-a"
        ) is None

    async def test_save_upserts_an_existing_row(self, db_session):
        from app.core.reasoning import (
            get_cached_reasoning_support,
            save_reasoning_support,
        )

        await save_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-a",
            support=ReasoningSupport(True, True, {"thinking": {"type": "disabled"}}),
        )
        await save_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-a",
            support=ReasoningSupport(True, False, None),
        )

        cached = await get_cached_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-a"
        )
        assert cached == ReasoningSupport(True, False, None)

    async def test_reasoning_support_is_cache_first(self, db_session):
        # A fresh cache row must make reasoning_support() return without any
        # outgoing HTTP call — the probe never runs in a read path.
        from app.core.reasoning import reasoning_support, save_reasoning_support

        await save_reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-b",
            support=ReasoningSupport(True, False, None),
        )

        class _Boom:
            async def post(self, *a, **kw):
                raise AssertionError("cache hit must not reach the network")

            async def aclose(self):
                pass

        support = await reasoning_support(
            db_session, provider=_TEST_PROVIDER, model="model-b",
            api_key="oc-1", client=_Boom(),
        )
        assert support == ReasoningSupport(True, False, None)
