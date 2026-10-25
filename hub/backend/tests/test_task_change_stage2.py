"""Stage 2 tests for the ChatMaster task-change proposal flow."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.user_sessions import (
    PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC,
    PROPOSE_TASK_CHANGE_TOOL_OPENAI,
    _parse_and_validate_task_change,
    _workspace_tool_names_for_session,
)
from app.models.task import TaskState
from app.schemas.task_change import TaskChangeProposal


def _result_with_scalars(values):
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(values)
    result.all.return_value = list(values)
    return result


def _proposal(*, source_id, agent_id, updated_at):
    return TaskChangeProposal.model_validate(
        {
            "rationale": "Split a wide blocked task into verifiable parts.",
            "sources": [
                {
                    "task_id": str(source_id),
                    "expected_updated_at": updated_at.isoformat(),
                    "expected_state": "blocked",
                }
            ],
            "create_tasks": [
                {
                    "ref": "seo-json-ld",
                    "title": "Add JSON-LD",
                    "description": "Acceptance criteria: valid WebSite markup and a test.",
                    "assigned_agent_id": str(agent_id),
                    "depends_on": [],
                }
            ],
        }
    )


def test_propose_task_change_provider_tools_have_one_structured_contract():
    assert PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC["name"] == "propose_task_change"
    assert PROPOSE_TASK_CHANGE_TOOL_OPENAI["function"]["name"] == "propose_task_change"
    assert (
        PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC["input_schema"]
        is PROPOSE_TASK_CHANGE_TOOL_OPENAI["function"]["parameters"]
    )
    assert "sources" in PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC["input_schema"]["required"]
    assert "create_tasks" in PROPOSE_TASK_CHANGE_TOOL_ANTHROPIC["input_schema"]["required"]


def test_user_cm_p1_workspace_boundary_removes_direct_task_mutations():
    registry = SimpleNamespace(
        tools_for=lambda role: [
            SimpleNamespace(name="read_file"),
            SimpleNamespace(name="task_create"),
            SimpleNamespace(name="task_update"),
            SimpleNamespace(name="task_list"),
        ]
    )

    assert _workspace_tool_names_for_session(
        SimpleNamespace(kind="user_cm_p1"), "cm", registry
    ) == ["read_file", "task_list"]
    assert "task_create" in _workspace_tool_names_for_session(
        SimpleNamespace(kind="user_cm_p2"), "cm", registry
    )


@pytest.mark.asyncio
async def test_valid_proposal_is_live_validated_without_task_mutation():
    room_id = uuid4()
    goal_id = uuid4()
    source_id = uuid4()
    agent_id = uuid4()
    updated_at = datetime.now(timezone.utc)
    source = SimpleNamespace(
        id=source_id,
        room_id=room_id,
        goal_id=goal_id,
        state=TaskState.BLOCKED,
        updated_at=updated_at,
    )
    proposal = _proposal(source_id=source_id, agent_id=agent_id, updated_at=updated_at)
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[
            _result_with_scalars([source]),
            _result_with_scalars([agent_id]),
            _result_with_scalars([]),
        ]
    )
    session = SimpleNamespace(
        id=uuid4(), room_id=room_id, goal_id=goal_id, kind="user_cm_p1"
    )

    normalized, error, tasks = await _parse_and_validate_task_change(
        session, proposal.model_dump(mode="json"), db
    )

    assert error is None
    assert normalized == proposal
    assert tasks == [source]
    assert source.state is TaskState.BLOCKED
    db.add.assert_not_called()
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_proposal_returns_compact_model_error_without_mutation():
    room_id = uuid4()
    goal_id = uuid4()
    source_id = uuid4()
    agent_id = uuid4()
    updated_at = datetime.now(timezone.utc)
    proposal = _proposal(source_id=source_id, agent_id=agent_id, updated_at=updated_at)
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[
            _result_with_scalars(
                [
                    SimpleNamespace(
                        id=source_id,
                        room_id=room_id,
                        goal_id=goal_id,
                        state=TaskState.BLOCKED,
                        updated_at=updated_at,
                    )
                ]
            ),
            _result_with_scalars([]),
            _result_with_scalars([]),
        ]
    )
    session = SimpleNamespace(
        id=uuid4(), room_id=room_id, goal_id=goal_id, kind="user_cm_p1"
    )

    normalized, error, _ = await _parse_and_validate_task_change(
        session, proposal.model_dump(mode="json"), db
    )

    assert normalized is None
    assert error is not None
    assert error.startswith("Error: propose_task_change was rejected:")
    assert "No tasks were changed" in error
    db.add.assert_not_called()
    db.commit.assert_not_called()
