from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.context_baseline import (
    ToolCallStats,
    build_context_measurement,
    prompt_components,
    record_context_measurement,
)
from app.core.context_replay import (
    build_anonymized_trajectory,
    replay_corpus,
    replay_trajectory,
)


def test_prompt_components_separate_tool_results_from_conversation():
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x" * 1000}],
        },
    ]

    blocks = prompt_components("system", [{"name": "read_file"}], messages)

    assert set(blocks) == {"system", "schemas", "conversation", "tool_results"}
    assert blocks["tool_results"]["chars"] > 1000
    assert blocks["conversation"]["chars"] < blocks["tool_results"]["chars"]
    assert all(value["estimated_tokens"] >= 1 for value in blocks.values())


def test_tool_call_stats_tracks_exact_repeats_and_partial_overlap():
    tracker = ToolCallStats()
    delta = tracker.observe([
        {"name": "read_file", "input": {"path": "src/a.py", "offset": 1, "limit": 100}},
        {"name": "read_file", "input": {"path": "src/a.py", "offset": 51, "limit": 50}},
        {"name": "read_file", "input": {"path": "src/a.py", "offset": 51, "limit": 50}},
        {"name": "grep", "input": {"pattern": "secret"}},
    ])

    assert delta["tool_calls"] == 4
    assert delta["repeated_reads"] == 1
    assert delta["read_requested_lines"] == 200
    assert delta["read_overlap_lines"] == 100
    assert delta["read_overlap_ratio"] == 0.5


@pytest.mark.asyncio
async def test_record_context_measurement_persists_sanitized_trace():
    row = SimpleNamespace(context_snapshot={"messages": []})
    db = AsyncMock()
    db.get = AsyncMock(return_value=row)
    db.add = MagicMock()
    measurement = build_context_measurement(
        provider="anthropic",
        model="model",
        role="agent",
        step=0,
        system_prompt="system",
        tools=[],
        messages=[{"role": "user", "content": "private text"}],
        actual_input_tokens=123,
        actual_output_tokens=9,
        response_calls=[],
        tracker=ToolCallStats(),
    )

    with patch("app.core.metrics.record_context_request") as metric:
        await record_context_measurement(measurement, session_id=uuid4(), db=db)

    assert row.context_snapshot["context_baseline"] == [measurement]
    assert "private text" not in json.dumps(measurement)
    metric.assert_called_once_with(measurement)


def test_anonymized_trajectory_has_baseline_and_no_raw_text():
    snapshot = {
        "messages": [
            {"role": "user", "content": "TOP SECRET TASK"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use", "id": "r1", "name": "read_file",
                    "input": {"path": "private/customer.py", "offset": 1, "limit": 100},
                }],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "r1", "content": "CUSTOMER SECRET"}],
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use", "id": "r2", "name": "read_file",
                    "input": {"path": "private/customer.py", "offset": 1, "limit": 100},
                }],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "r2", "content": "CUSTOMER SECRET"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "b1", "name": "bash", "input": {"cmd": "pytest secret"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "b1", "content": "exit_code: 0\nSECRET LOG"}],
            },
            {"role": "assistant", "content": "done"},
        ]
    }

    trajectory = build_anonymized_trajectory(
        session_id="raw-session-id",
        snapshot=snapshot,
        close_reason="task_done",
        task_state="completed",
        task_title="Edit code",
    )
    rendered = json.dumps(trajectory)

    assert trajectory["baseline"]["tool_calls"] == 3
    assert trajectory["baseline"]["repeated_reads"] == 1
    assert trajectory["verification_result"] == "passed"
    assert trajectory["success"] is True
    for secret in (
        "TOP SECRET TASK", "CUSTOMER SECRET", "private/customer.py",
        "pytest secret", "SECRET LOG", "raw-session-id",
    ):
        assert secret not in rendered


def test_replay_current_vs_proposed_and_corpus_baseline():
    trajectory = {
        "session_hash": "abc",
        "trajectory_type": "code_edit",
        "success": True,
        "verification_result": "passed",
        "baseline": {
            "tool_calls": 5,
            "repeated_reads": 1,
            "read_requested_lines": 200,
            "read_overlap_lines": 50,
        },
        "requests": [{
            "system_tokens": 50,
            "schema_tokens": 25,
            "conversation_tokens": 25,
            "tool_results": [{"tool": "read_file", "tokens": 100} for _ in range(5)],
            "actual_input_tokens": 0,
        }],
    }

    replay = replay_trajectory(trajectory, keep_raw_results=3, receipt_tokens=32)
    report = replay_corpus([trajectory], keep_raw_results=3, receipt_tokens=32)

    assert replay["current_total_input_tokens"] == 600
    assert replay["proposed_total_input_tokens"] == 464
    assert replay["saved_input_tokens"] == 136
    assert report["baseline"]["total_input_tokens"] == 600
    assert report["baseline"]["repeated_reads"] == 1
    assert report["baseline"]["read_overlap_ratio"] == 0.25
