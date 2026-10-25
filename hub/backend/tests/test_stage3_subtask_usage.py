from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.context_policy import ContextPolicy
from app.core.tools import REGISTRY


def _policy() -> ContextPolicy:
    return ContextPolicy(
        enabled=True,
        input_cap=100_000,
        active_tool_result_cap=10_000,
        keep_raw_results=3,
        output_reserve=1_000,
        model_context_limit=200_000,
        checkpoint_cap=600,
        context_window_limit=100_000,
    )


def _openai_response(*, tool_calls=None, content=None, prompt_tokens=0, completion_tokens=0):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(
        finish_reason="tool_calls" if tool_calls else "stop",
        message=message,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


@pytest.mark.asyncio
async def test_proxy_rolls_up_run_subtask_usage_into_parent_totals():
    from app.core import proxy_session

    subtask = REGISTRY.get("run_subtask")
    assert subtask is not None
    subtask_result = json.dumps({
        "step_title": "Inspect module",
        "status": "completed",
        "summary": "Done",
        "file_writes": [],
        "tokens_used": 18,
        "tokens_in": 11,
        "tokens_out": 7,
        "tool_calls_used": 4,
    })
    tool_call = SimpleNamespace(
        id="run-subtask-1",
        function=SimpleNamespace(
            name="run_subtask",
            arguments=json.dumps({
                "step_title": "Inspect module",
                "step_instructions": "Inspect the module and report back.",
            }),
        ),
    )
    create = AsyncMock(side_effect=[
        _openai_response(
            tool_calls=[tool_call],
            prompt_tokens=100,
            completion_tokens=10,
        ),
        _openai_response(
            content="Parent continued.",
            prompt_tokens=200,
            completion_tokens=20,
        ),
    ])
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    usage_stats: dict[str, int] = {}

    with (
        patch.object(proxy_session, "resolve_context_policy", AsyncMock(return_value=_policy())),
        patch.object(proxy_session, "AsyncOpenAI", return_value=client),
        patch.object(proxy_session, "persist_context_assembly", AsyncMock()),
        patch.object(proxy_session, "persist_provider_cache_usage", AsyncMock()),
        patch.object(proxy_session, "record_context_measurement", AsyncMock()),
        patch.object(proxy_session, "emit_status", AsyncMock()),
        patch.object(proxy_session, "_emit_tool_call_event", AsyncMock()),
        patch.object(subtask, "execute", AsyncMock(return_value=subtask_result)),
    ):
        result = await proxy_session._call_llm_with_tools(
            {"provider": "openai", "api_key": "test", "model": "test-model"},
            "system",
            [{"role": "user", "content": "Start."}],
            room_id=uuid4(),
            session_id=uuid4(),
            task_id=None,
            db=MagicMock(),
            workspace_oai_tools=[subtask.oai_schema()],
            max_tool_calls=10,
            on_exhaust="raise",
            context_role="agent",
            usage_stats=usage_stats,
        )

    assert result[0] == "Parent continued."
    assert result[2:4] == (311, 37)
    assert usage_stats == {
        "tokens_in": 311,
        "tokens_out": 37,
        "tool_calls_used": 5,
        # reasoning_and_router_model_plan stage 6 (§7.3): published alongside
        # the token totals, never folded into them.  A model that did not
        # think reports 0 — the key is always present so the caller does not
        # have to distinguish "no reasoning" from "field missing".
        "reasoning_tokens": 0,
    }


def test_extract_subtask_usage_supports_legacy_aggregate_reports():
    from app.core.proxy_session import _extract_subtask_usage

    assert _extract_subtask_usage('{"tokens_used": 18, "tool_calls_used": 2}') == (18, 0, 2)
    assert _extract_subtask_usage("not-json") == (0, 0, 0)
