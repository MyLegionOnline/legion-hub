"""Regression coverage for the durable subtask chat-observability contract."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.subtask_activity import activity_metadata, completion_metadata, new_subtask_activity
from app.core.tools import REGISTRY, ToolContext
from app.core.tools import _run_subtask as run_subtask_module


def test_activity_metadata_correlates_parent_and_child_calls():
    activity = new_subtask_activity("run_subtask", {"step_title": "Check the API", "max_tool_calls": 12})

    assert activity is not None
    parent = activity_metadata(activity, phase="started")
    child = activity_metadata(activity, phase="activity")

    assert parent["id"] == child["id"] == activity["id"]
    assert parent["title"] == child["title"] == "Check the API"
    assert parent["phase"] == "started"
    assert child["phase"] == "activity"
    assert parent["max_tool_calls"] == 12


def test_completion_metadata_keeps_only_compact_subtask_report_fields():
    activity = new_subtask_activity("run_subtask", {"step_title": "Build the interface"})
    payload = completion_metadata(activity, json.dumps({
        "status": "completed",
        "summary": "Built and verified.",
        "file_writes": ["src/Chat.jsx"],
        "tokens_in": 11,
        "tokens_out": 7,
        "tool_calls_used": 3,
        "unrelated": "must not reach chat metadata",
    }))

    assert payload == {
        "id": activity["id"],
        "title": "Build the interface",
        "phase": "finished",
        "status": "completed",
        "summary": "Built and verified.",
        "file_writes": ["src/Chat.jsx"],
        "tokens_in": 11,
        "tokens_out": 7,
        "tool_calls_used": 3,
    }


@pytest.mark.asyncio
async def test_run_subtask_persists_one_completion_event_for_correlated_call():
    tool = REGISTRY.get("run_subtask")
    assert tool is not None
    session_id = uuid4()
    activity = new_subtask_activity("run_subtask", {"step_title": "Isolated step"})
    ctx = ToolContext(
        room_id=uuid4(),
        session_id=session_id,
        task_id=uuid4(),
        db=MagicMock(),
        subtask_activity=activity,
    )
    report = json.dumps({"status": "completed", "step_title": "Isolated step", "summary": "Done"})

    run_subtask_module._SUBTASK_CALLS.pop(str(session_id), None)
    with (
        patch.object(tool, "_execute_isolated", AsyncMock(return_value=report)),
        patch("app.core.tools._run_subtask.emit_event", new_callable=AsyncMock) as emit,
    ):
        result = await tool.execute(ctx, {
            "step_title": "Isolated step",
            "step_instructions": "Do only this step.",
        })
    run_subtask_module._SUBTASK_CALLS.pop(str(session_id), None)

    assert result == report
    emit.assert_awaited_once()
    kwargs = emit.await_args.kwargs
    assert kwargs["event_type"] == "AGENT_SUBTASK"
    assert kwargs["metadata"]["session_id"] == str(session_id)
    assert kwargs["metadata"]["subtask"]["id"] == activity["id"]
    assert kwargs["metadata"]["subtask"]["status"] == "completed"
