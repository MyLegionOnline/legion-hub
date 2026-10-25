"""Agent-side reasoning config: validation and the router's forced off-switch.

Covers plan §5.2 (the shape an agent may save), §6 (the router asks its model
to stop thinking and records whether it obeyed) and the §0.7 regression that
OR1 re-confirmed on OpenRouter: a reasoning model under a small cap can spend
the whole cap on thought and return an empty message.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core import reasoning as reasoning_mod
from app.core.reasoning import OPENROUTER, ReasoningSupport, remember_reasoning_support
from app.schemas.agent import AgentBase


def _agent(llm_config: dict) -> AgentBase:
    return AgentBase(nickname="tester", llm_config=llm_config)


# ------------------------------------------------------------- §5.2 validation


def test_absent_reasoning_key_is_the_default_and_stays_absent():
    agent = _agent({"provider": OPENROUTER, "model": "deepseek/deepseek-v4-flash"})
    assert "reasoning" not in agent.llm_config


@pytest.mark.parametrize("effort", ["off", "low", "medium", "high"])
def test_every_documented_level_is_accepted(effort):
    agent = _agent({"reasoning": {"effort": effort}})
    assert agent.llm_config["reasoning"]["effort"] == effort


def test_show_in_chat_is_accepted_as_a_boolean():
    agent = _agent({"reasoning": {"effort": "low", "show_in_chat": False}})
    assert agent.llm_config["reasoning"]["show_in_chat"] is False


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="reasoning.effort"):
        _agent({"reasoning": {"effort": "extreme"}})


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        _agent({"reasoning": {"effort": "low", "budget": 4096}})


def test_non_boolean_show_in_chat_is_rejected():
    with pytest.raises(ValueError, match="show_in_chat"):
        _agent({"reasoning": {"effort": "low", "show_in_chat": "yes"}})


def test_null_block_is_rejected_rather_than_read_as_the_default():
    # "Provider default" is spelled by the key being ABSENT.  Accepting null
    # here would let the UI write a block that means nothing, and the next
    # reader would have to guess which spelling it was looking at.
    with pytest.raises(ValueError, match="omit the key"):
        _agent({"reasoning": None})


def test_non_object_block_is_rejected():
    with pytest.raises(ValueError, match="reasoning must be an object"):
        _agent({"reasoning": "high"})


def test_an_empty_block_is_accepted_and_means_provider_default():
    # Not a hole: `{}` carries no effort, and resolve_effort reads it as None.
    agent = _agent({"reasoning": {}})
    assert reasoning_mod.resolve_effort(agent.llm_config) is None


# --------------------------------------------------- §6 router forced off


class _Recorder:
    """Captures the outgoing chat request instead of making one."""

    def __init__(self, content: str = '{"route": "chatmaster_p2", "confidence": 0.9, "reasoning": "x"}'):
        self.kwargs = None
        self.content = content

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=self.content, reasoning="thought", reasoning_content=None,
            ))],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        )


@pytest.fixture(autouse=True)
def _clean_capability_snapshot():
    reasoning_mod._CAPABILITY_SNAPSHOT.clear()
    reasoning_mod._OFF_SWITCH_DENIED.clear()
    yield
    reasoning_mod._CAPABILITY_SNAPSHOT.clear()
    reasoning_mod._OFF_SWITCH_DENIED.clear()


async def _call_router(llm_config: dict, recorder: _Recorder, metrics: dict | None = None):
    from app.core import router as router_mod

    client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
    with patch.object(router_mod, "AsyncOpenAI", lambda **_kw: client):
        return await router_mod._call_router_llm(
            "hi",
            llm_config,
            secret_value=SimpleNamespace(reveal=lambda: "k"),
            metrics=metrics,
        )


@pytest.mark.asyncio
async def test_router_forces_the_off_switch_on_a_controllable_model():
    remember_reasoning_support(
        OPENROUTER, "deepseek/deepseek-v4-flash",
        ReasoningSupport(supported=True, controllable=True, off_switch=None),
    )
    recorder = _Recorder()
    await _call_router(
        {"provider": OPENROUTER, "model": "deepseek/deepseek-v4-flash"}, recorder
    )
    assert recorder.kwargs["extra_body"] == {"reasoning": {"enabled": False}}


@pytest.mark.asyncio
async def test_router_does_not_send_a_switch_a_mandatory_model_would_reject():
    # §4A.5 q1: this request would come back 400 on every user message, i.e. a
    # working-but-slow router replaced by a broken one.
    remember_reasoning_support(
        OPENROUTER, "z-ai/glm-5.3",
        ReasoningSupport(supported=True, controllable=False, off_switch=None),
    )
    recorder = _Recorder()
    await _call_router({"provider": OPENROUTER, "model": "z-ai/glm-5.3"}, recorder)
    assert "extra_body" not in recorder.kwargs


@pytest.mark.asyncio
async def test_router_keeps_openrouter_session_logging_while_forcing_off():
    from app.core import router as router_mod

    remember_reasoning_support(
        OPENROUTER, "deepseek/deepseek-v4-flash",
        ReasoningSupport(supported=True, controllable=True, off_switch=None),
    )
    recorder = _Recorder()
    client = SimpleNamespace(chat=SimpleNamespace(completions=recorder))
    with patch.object(router_mod, "AsyncOpenAI", lambda **_kw: client):
        await router_mod._call_router_llm(
            "hi",
            {"provider": OPENROUTER, "model": "deepseek/deepseek-v4-flash",
             "session_logging": True},
            or_session_id="router-1",
            secret_value=SimpleNamespace(reveal=lambda: "k"),
        )
    assert recorder.kwargs["extra_body"] == {
        "session_id": "router-1",
        "reasoning": {"enabled": False},
    }


@pytest.mark.asyncio
async def test_router_records_the_reasoning_tokens_it_actually_paid_for():
    # §6.1: the switch is best effort, so the honest number is what makes
    # "is the router still thinking?" answerable.
    metrics: dict = {}
    await _call_router(
        {"provider": OPENROUTER, "model": "z-ai/glm-5.3"}, _Recorder(), metrics
    )
    assert metrics["reasoning_tokens"] == 3


@pytest.mark.asyncio
async def test_router_metrics_are_optional_so_existing_mocks_keep_working():
    text, tin, tout, model = await _call_router(
        {"provider": OPENROUTER, "model": "z-ai/glm-5.3"}, _Recorder()
    )
    assert json.loads(text)["route"] == "chatmaster_p2"
    assert (tin, tout, model) == (10, 5, "z-ai/glm-5.3")


# ------------------------------------------- §0.7 / §4A.5 q7 empty-reply trap


def test_router_reports_an_empty_reply_as_such_not_as_bad_json():
    """A cap eaten by thinking must not be reported as a malformed reply.

    Measured on OpenRouter (§4A.5 q7): deepseek-v4-flash with max_tokens=200
    returns reasoning_tokens=200 and content_chars=0.  `json.loads("")` would
    log 'Expecting value: line 1 column 1', sending whoever reads
    ROUTE_LLM_FAILURE hunting for a broken payload that was never sent.
    """
    from app.core.router import _parse_router_reply

    with pytest.raises(ValueError) as excinfo:
        _parse_router_reply("", reasoning_tokens=200)
    message = str(excinfo.value)
    assert "empty message" in message
    assert "200 tokens on reasoning" in message
    assert "Expecting value" not in message


def test_router_still_reports_genuinely_malformed_json_as_json():
    from app.core.router import _parse_router_reply

    with pytest.raises(json.JSONDecodeError):
        _parse_router_reply("not valid json {{")


def test_router_reply_parser_unwraps_a_fenced_reply():
    from app.core.router import _parse_router_reply

    decision = _parse_router_reply(
        '```json\n{"route": "chatmaster_p2", "confidence": 0.9, "reasoning": "x"}\n```'
    )
    assert decision.route == "chatmaster_p2"
