"""Track 1 of internal design notes — OpenRouter reasoning.

Everything asserted here was measured live on 05.09.2026 and written down in
plan §4A.5; the tests exist so the measurement cannot rot back into a guess.

The single most load-bearing fact: OpenRouter does NOT behave like the gateway
the rest of the plan was measured on.  OpenCode Go ignores an off-switch a
model cannot honour; OpenRouter answers
``400 "Reasoning is mandatory for this endpoint and cannot be disabled."``
That is why the switch is gated (§4B.5) instead of sent optimistically.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.llm_providers import OPENCODE_GO, WIRE_CHAT, WIRE_MESSAGES
from app.core import reasoning as reasoning_mod
from app.core.reasoning import (
    OPENROUTER,
    ReasoningSupport,
    apply_reasoning_body,
    apply_reasoning_chat,
    gated_reasoning_fields,
    known_reasoning_support,
    off_switch_allowed,
    openrouter_catalog_support,
    remember_reasoning_support,
    resolve_effort,
    show_reasoning_in_chat,
)


@pytest.fixture(autouse=True)
def _clean_capability_snapshot():
    """Each test starts from an unmeasured world and leaves one behind."""
    reasoning_mod._CAPABILITY_SNAPSHOT.clear()
    reasoning_mod._OFF_SWITCH_DENIED.clear()
    yield
    reasoning_mod._CAPABILITY_SNAPSHOT.clear()
    reasoning_mod._OFF_SWITCH_DENIED.clear()


# ------------------------------------------------- §4B.1 request field shape


@pytest.mark.parametrize(
    "effort,expected",
    [
        ("off", {"reasoning": {"enabled": False}}),
        ("low", {"reasoning": {"effort": "low"}}),
        ("medium", {"reasoning": {"effort": "medium"}}),
        ("high", {"reasoning": {"effort": "high"}}),
    ],
)
def test_openrouter_chat_fields_use_the_unified_reasoning_block(effort, expected):
    # §4A.5 q1: the three off forms are one flag after the gateway normalises
    # them; the unified block is declared by 302 catalog models against 159 for
    # reasoning_effort, so it is the one the Hub sends.
    assert reasoning_mod.reasoning_request_fields(
        OPENROUTER, "deepseek/deepseek-v4-flash", WIRE_CHAT, effort
    ) == expected


def test_openrouter_effort_none_still_sends_nothing():
    # Decision 0.10 p.4 survives the new branch: an agent with no reasoning
    # block gets a byte-identical request.
    assert reasoning_mod.reasoning_request_fields(
        OPENROUTER, "deepseek/deepseek-v4-flash", WIRE_CHAT, None
    ) == {}


def test_unmeasured_providers_keep_the_old_chat_branch():
    # openai/groq/together/deepseek/ollama were never measured; the stage must
    # not change their behaviour by a byte.
    for provider in ("openai", "groq", "together", "deepseek", "ollama"):
        assert reasoning_mod.reasoning_request_fields(
            provider, "some-model", WIRE_CHAT, "off"
        ) == {"reasoning_effort": "none"}


def test_opencode_go_chat_branch_is_untouched():
    assert reasoning_mod.reasoning_request_fields(
        OPENCODE_GO, "glm-5.3", WIRE_CHAT, "high"
    ) == {"reasoning_effort": "high"}


# ------------------------------------------------- §4B.2 catalog capabilities


def test_catalog_mandatory_true_means_not_controllable():
    # z-ai/glm-5.3, google/gemini-3.8-flash, openai/gpt-6-astra and
    # anthropic/claude-fable-5.1 all declare mandatory:true and all answered
    # 400 to every off-switch (§4A.5).
    support = openrouter_catalog_support(
        {"id": "z-ai/glm-5.3", "reasoning": {"mandatory": True, "default_enabled": True}}
    )
    assert support == ReasoningSupport(supported=True, controllable=False, off_switch=None)


def test_catalog_mandatory_false_carries_the_measured_off_switch():
    support = openrouter_catalog_support(
        {"id": "deepseek/deepseek-v4-flash", "reasoning": {"mandatory": False}}
    )
    assert support.supported is True
    assert support.controllable is True
    assert support.off_switch == {"reasoning": {"enabled": False}}


def test_catalog_without_a_reasoning_block_means_no_reasoning():
    support = openrouter_catalog_support({"id": "writer/palmyra-x5"})
    assert support == ReasoningSupport(supported=False, controllable=False, off_switch=None)


def test_catalog_parameter_without_an_object_is_unmeasured_not_uncontrollable():
    support = openrouter_catalog_support(
        {"id": "x/y", "supported_parameters": ["reasoning", "tools"]}
    )
    assert support.supported is True
    assert support.controllable is False


# ------------------------------------------------------- §4B.5 off-switch gate


def test_off_switch_is_dropped_for_an_unmeasured_openrouter_model():
    # No snapshot entry: sending the switch risks a 400 on every single turn,
    # so the call goes out at the provider default instead — today's behaviour.
    assert off_switch_allowed(OPENROUTER, "z-ai/glm-5.3") is False
    assert gated_reasoning_fields(OPENROUTER, "z-ai/glm-5.3", WIRE_CHAT, "off") == {}


def test_off_switch_is_dropped_for_a_mandatory_openrouter_model():
    remember_reasoning_support(
        OPENROUTER, "z-ai/glm-5.3",
        ReasoningSupport(supported=True, controllable=False, off_switch=None),
    )
    assert off_switch_allowed(OPENROUTER, "z-ai/glm-5.3") is False
    assert gated_reasoning_fields(OPENROUTER, "z-ai/glm-5.3", WIRE_CHAT, "off") == {}


def test_off_switch_is_sent_for_a_controllable_openrouter_model():
    remember_reasoning_support(
        OPENROUTER, "deepseek/deepseek-v4-flash",
        ReasoningSupport(
            supported=True, controllable=True,
            off_switch={"reasoning": {"enabled": False}},
        ),
    )
    assert gated_reasoning_fields(
        OPENROUTER, "deepseek/deepseek-v4-flash", WIRE_CHAT, "off"
    ) == {"reasoning": {"enabled": False}}


def test_levels_are_never_gated_and_never_touch_the_snapshot():
    # Only "off" can 400.  A level is accepted by every reasoning model and
    # ignored by every non-reasoning one (§4A.5 q5).
    assert gated_reasoning_fields(
        OPENROUTER, "never/measured", WIRE_CHAT, "high"
    ) == {"reasoning": {"effort": "high"}}
    assert known_reasoning_support(OPENROUTER, "never/measured") is None


def test_other_gateways_are_not_gated():
    # OpenCode Go ignores a switch it cannot honour (§0.6), so withholding it
    # there would remove a control that costs nothing and sometimes works.
    assert off_switch_allowed(OPENCODE_GO, "glm-5.2") is True
    assert gated_reasoning_fields(
        OPENCODE_GO, "glm-5.2", WIRE_CHAT, "off"
    ) == {"reasoning_effort": "none"}


# ------------------------------------------- §5.1–5.3 config and plumbing


def test_resolve_effort_reads_the_agent_block():
    assert resolve_effort({"reasoning": {"effort": "medium"}}) == "medium"


def test_resolve_effort_without_a_block_is_provider_default():
    assert resolve_effort({}) is None
    assert resolve_effort(None) is None
    assert resolve_effort({"reasoning": {"show_in_chat": False}}) is None


def test_resolve_effort_ignores_a_junk_level():
    assert resolve_effort({"reasoning": {"effort": "extreme"}}) is None


def test_override_none_wins_over_the_agent_setting():
    # Class B of §5.3.2: the Hub's own utility calls never inherit the agent's
    # thinking budget.
    assert resolve_effort({"reasoning": {"effort": "high"}}, None) is None


def test_override_off_wins_over_the_agent_setting():
    # Class C: the router forces the switch on top of whatever is configured.
    assert resolve_effort({"reasoning": {"effort": "high"}}, "off") == "off"


def test_show_in_chat_defaults_to_on():
    assert show_reasoning_in_chat({}) is True
    assert show_reasoning_in_chat({"reasoning": {"effort": "low"}}) is True
    assert show_reasoning_in_chat({"reasoning": {"show_in_chat": False}}) is False


def test_chat_helper_merges_with_openrouter_session_logging():
    # §5.3.3: assigning extra_body instead of merging would silently switch
    # OpenRouter session logging off — for the very provider this stage exists
    # for, and invisibly, since the symptom is missing log rows a week later.
    create_extra = {"extra_body": {"session_id": "router-42"}}
    apply_reasoning_chat(
        create_extra, {"reasoning": {"effort": "high"}},
        OPENROUTER, "deepseek/deepseek-v4-flash",
    )
    assert create_extra["extra_body"] == {
        "session_id": "router-42",
        "reasoning": {"effort": "high"},
    }


def test_chat_helper_leaves_an_unconfigured_agent_byte_identical():
    create_extra = {"extra_body": {"session_id": "router-42"}}
    apply_reasoning_chat(create_extra, {}, OPENROUTER, "deepseek/deepseek-v4-flash")
    assert create_extra == {"extra_body": {"session_id": "router-42"}}


def test_chat_helper_creates_no_empty_extra_body():
    # Without session logging there is no extra_body today, and an agent with
    # no reasoning must not grow one.
    create_extra: dict = {}
    apply_reasoning_chat(create_extra, {}, OPENROUTER, "deepseek/deepseek-v4-flash")
    assert create_extra == {}


def test_body_helper_clamps_the_budget_under_the_call_cap():
    body = {"model": "qwen3.7-plus", "max_tokens": 4096}
    apply_reasoning_body(
        body, {"reasoning": {"effort": "high"}}, OPENCODE_GO, "qwen3.7-plus"
    )
    assert body["thinking"]["budget_tokens"] < body["max_tokens"]


def test_body_helper_skips_reasoning_when_the_cap_leaves_no_room():
    body = {"model": "qwen3.7-plus", "max_tokens": 512}
    apply_reasoning_body(
        body, {"reasoning": {"effort": "high"}}, OPENCODE_GO, "qwen3.7-plus"
    )
    assert "thinking" not in body


def test_body_helper_forced_off_is_gated_the_same_way():
    body = {"model": "z-ai/glm-5.3", "max_tokens": 512}
    apply_reasoning_body(
        body, {}, OPENROUTER, "z-ai/glm-5.3", override="off"
    )
    assert body == {"model": "z-ai/glm-5.3", "max_tokens": 512}


def test_body_helper_forced_off_is_sent_when_the_model_allows_it():
    remember_reasoning_support(
        OPENROUTER, "deepseek/deepseek-v4-flash",
        ReasoningSupport(supported=True, controllable=True, off_switch=None),
    )
    body = {"model": "deepseek/deepseek-v4-flash", "max_tokens": 512}
    apply_reasoning_body(
        body, {}, OPENROUTER, "deepseek/deepseek-v4-flash", override="off"
    )
    # messages wire keeps the Anthropic shape regardless of provider
    assert body["thinking"] == {"type": "disabled"}


# ------------------------------------------------------- reply-side reading


def test_reasoning_tokens_are_read_from_an_openrouter_reply():
    message = SimpleNamespace(reasoning="thinking out loud", reasoning_content=None)
    usage = SimpleNamespace(
        completion_tokens_details=SimpleNamespace(reasoning_tokens=42)
    )
    text, tokens = reasoning_mod.reasoning_from_chat_choice(message, usage)
    assert (text, tokens) == ("thinking out loud", 42)


def test_a_closed_reasoner_reports_tokens_without_text():
    # §4A.5 q3: gemini-3.8-flash spends reasoning tokens and returns no text.
    # Stage 6 must count the tokens and emit no empty chat block.
    message = {"reasoning": "", "reasoning_details": [{"type": "encrypted"}]}
    usage = {"completion_tokens_details": {"reasoning_tokens": 85}}
    text, tokens = reasoning_mod.reasoning_from_chat_choice(message, usage)
    assert text == ""
    assert tokens == 85


def test_messages_reasoning_tokens_tolerates_both_usage_shapes():
    assert reasoning_mod.messages_reasoning_tokens(
        {"usage": {"completion_tokens_details": {"reasoning_tokens": 7}}}
    ) == 7
    assert reasoning_mod.messages_reasoning_tokens(
        {"usage": {"output_tokens_details": {"reasoning_tokens": 9}}}
    ) == 9
    assert reasoning_mod.messages_reasoning_tokens({"usage": {}}) == 0
    assert reasoning_mod.messages_reasoning_tokens({}) == 0
