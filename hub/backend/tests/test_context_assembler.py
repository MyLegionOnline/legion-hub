from __future__ import annotations

import importlib
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import app.core.context_policy as policy_mod
from app.core.context_assembler import ContextAssembler
from app.core.context_policy import (
    ContextPolicy,
    apply_policy_overrides,
)
from app.core.token_counter import TokenCounter


def _policy(**overrides) -> ContextPolicy:
    values = {
        "enabled": True,
        "input_cap": 16_000,
        "active_tool_result_cap": 6_000,
        "keep_raw_results": 3,
        "output_reserve": 4_000,
        "model_context_limit": 200_000,
        "checkpoint_cap": 600,
        "eviction_mode": "on_demand",
    }
    values.update(overrides)
    return ContextPolicy(**values)


def _anthropic_turn(call_id: str, tool: str, content: str, **args):
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": tool,
                    "input": args,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content,
                }
            ],
        },
    ]


def _openai_turn(call_id: str, tool: str, content: str, **args):
    import json

    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool,
                        "arguments": json.dumps(args),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": content,
        },
    ]


def test_token_counter_is_utf8_aware_and_cached():
    counter = TokenCounter()
    latin = counter.count("a" * 100)
    non_ascii = counter.count("é" * 100)

    assert non_ascii > latin
    assert counter.count("é" * 100) == non_ascii
    assert len(counter._cache) == 2


def test_policy_overrides_are_room_then_agent_then_llm():
    base = _policy(enabled=False)
    resolved = apply_policy_overrides(
        base,
        {"context_gateway": {"enabled": True, "input_cap": 12_000}},
        {"context_gateway_enabled": False, "context_tool_result_cap": 900},
        {"context_gateway": {"agent": {"enabled": True, "keep_raw_results": 1}}},
        role="agent",
    )

    assert resolved.enabled is True
    assert resolved.input_cap == 12_000
    assert resolved.active_tool_result_cap == 900
    assert resolved.keep_raw_results == 1


def test_anthropic_eviction_preserves_tool_use_result_pairing():
    messages = [{"role": "user", "content": "Fix the bug"}]
    messages += _anthropic_turn("old", "read_file", "x" * 40_000, path="app/a.py")
    messages += _anthropic_turn("new", "read_file", "latest", path="app/b.py")

    assembly = ContextAssembler(
        _policy(
            eviction_mode="eager", keep_raw_results=1, active_tool_result_cap=1_000
        )
    ).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    )

    old_result = assembly.messages[2]["content"][0]
    new_result = assembly.messages[4]["content"][0]
    assert old_result["tool_use_id"] == "old"
    assert "context receipt" in old_result["content"]
    assert "x" * 100 not in old_result["content"]
    assert new_result["tool_use_id"] == "new"
    assert new_result["content"] == "latest"
    assert assembly.evicted_results == 1
    assert assembly.active_tool_tokens <= 1_000


def test_openai_eviction_preserves_tool_call_id_and_budget():
    messages = [{"role": "user", "content": "Inspect"}]
    messages += _openai_turn("one", "grep", "a" * 1_000, pattern="alpha")
    messages += _openai_turn("two", "read_file", "b" * 1_000, path="b.py")
    messages += _openai_turn("three", "bash", "exit_code: 0\n" + "c" * 1_000, cmd="pytest")

    assembly = ContextAssembler(
        _policy(
            eviction_mode="eager", keep_raw_results=3, active_tool_result_cap=400
        )
    ).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    )

    tool_messages = [m for m in assembly.messages if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["one", "two", "three"]
    assert assembly.active_tool_tokens <= 400
    assert sum("context receipt" in m["content"] for m in tool_messages) >= 2
    assert "bash:ok" in assembly.checkpoint["checks"]


def test_prompt_is_compacted_under_cap_without_splitting_tool_pairs():
    messages = [{"role": "user", "content": "Original task"}]
    for index in range(8):
        messages += _openai_turn(
            f"call-{index}",
            "read_file",
            str(index) * 900,
            path=f"app/{index}.py",
        )
        messages.append(
            {"role": "assistant", "content": f"Decision {index} " + ("z" * 500)}
        )
    messages.append({"role": "user", "content": "Continue"})

    assembly = ContextAssembler(
        _policy(
            input_cap=1_500,
            active_tool_result_cap=300,
            keep_raw_results=1,
            output_reserve=256,
            eviction_mode="eager",
        )
    ).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    )

    assert assembly.allowed
    assert assembly.estimated_input_tokens <= 1_500
    assert assembly.dropped_messages > 0
    call_ids = {
        call["id"]
        for message in assembly.messages
        for call in message.get("tool_calls") or []
    }
    result_ids = {
        message["tool_call_id"]
        for message in assembly.messages
        if message.get("role") == "tool"
    }
    assert call_ids == result_ids


def test_on_demand_keeps_every_tool_result_while_the_prompt_fits():
    """A session well inside its window must not lose anything it read."""
    messages = [{"role": "user", "content": "Study the repo"}]
    for index in range(6):
        messages += _openai_turn(
            f"call-{index}", "read_file", str(index) * 900, path=f"app/{index}.go"
        )

    assembly = ContextAssembler(_policy()).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    )

    contents = [m["content"] for m in assembly.messages if m.get("role") == "tool"]
    assert assembly.evicted_results == 0
    assert assembly.evicted_tokens == 0
    assert not any("context receipt" in text for text in contents)
    assert all(str(index) * 900 in contents[index] for index in range(6))


def test_on_demand_evicts_oldest_first_only_down_to_the_budget():
    messages = [{"role": "user", "content": "Study the repo"}]
    for index in range(6):
        messages += _openai_turn(
            f"call-{index}", "read_file", str(index) * 4_000, path=f"app/{index}.go"
        )

    assembly = ContextAssembler(
        _policy(input_cap=4_000, output_reserve=256)
    ).build(
        system_prompt="system",
        tool_schemas=[],
        messages=messages,
        session_remaining_tokens=100_000,
    )

    contents = [m["content"] for m in assembly.messages if m.get("role") == "tool"]
    evicted = [i for i, text in enumerate(contents) if "context receipt" in text]
    assert assembly.allowed
    assert evicted, "an over-budget prompt must still evict"
    # Oldest first, contiguously, and the newest result always survives.
    assert evicted == list(range(len(evicted)))
    assert "context receipt" not in contents[-1]


def test_legacy_preflight_rejects_before_session_output_reserve_is_spent(monkeypatch):
    monkeypatch.setenv("CONTEXT_LIMIT_MODE", "cumulative")
    importlib.reload(policy_mod)
    try:
        assembly = ContextAssembler(_policy(output_reserve=4_000)).build(
            system_prompt="system",
            tool_schemas=[],
            messages=[{"role": "user", "content": "hello"}],
            session_remaining_tokens=4_000,
        )

        assert not assembly.allowed
        assert assembly.rejected_reason == "session_budget"
    finally:
        monkeypatch.delenv("CONTEXT_LIMIT_MODE", raising=False)
        importlib.reload(policy_mod)


@pytest.mark.asyncio
async def test_legacy_proxy_preflight_blocks_normal_and_forced_close_http_requests(monkeypatch):
    from app.core.proxy_session import _call_llm_with_tools

    monkeypatch.setenv("CONTEXT_LIMIT_MODE", "cumulative")
    importlib.reload(policy_mod)
    try:
        db = AsyncMock()
        with patch("app.core.proxy_session.httpx.AsyncClient") as http_client:
            result = await _call_llm_with_tools(
                {"provider": "anthropic", "api_key": "test"},
                "system",
                [{"role": "user", "content": "hello"}],
                room_id=uuid4(),
                session_id=uuid4(),
                task_id=None,
                db=db,
                budget_used=96_000,
                context_role="agent",
                on_exhaust="force_close",
            )

        assert result[:2] == (None, None)
        http_client.assert_not_called()
    finally:
        monkeypatch.delenv("CONTEXT_LIMIT_MODE", raising=False)
        importlib.reload(policy_mod)
