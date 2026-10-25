"""Tests for Stage 5: task_list, task_update, task_create, plan_read.

Coverage:
  - role enforcement: task_create not in agent tools, only in cm tools
  - task_list returns tasks for the room
  - task_update: valid transition, invalid transition, not found
  - task_update: emits TASK_UPDATED event
  - task_create: creates task, emits TASK_CREATED event (cm only)
  - plan_read: returns plans with steps; no goals; no plans
"""
from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uuid():
    return uuid.uuid4()


def _make_ctx(room_id=None, session_id=None, task_id=None):
    from app.core.tools.registry import ToolContext

    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    return ToolContext(
        room_id=room_id or _uuid(),
        session_id=session_id or _uuid(),
        task_id=task_id,
        db=db,
    )


# ---------------------------------------------------------------------------
# Role enforcement
# ---------------------------------------------------------------------------


def test_task_create_not_available_to_agent():
    from app.core.tools import REGISTRY

    agent_names = {t.name for t in REGISTRY.tools_for("agent")}
    assert "task_create" not in agent_names


def test_task_create_available_to_cm():
    from app.core.tools import REGISTRY

    cm_names = {t.name for t in REGISTRY.tools_for("cm")}
    assert "task_create" in cm_names


def test_task_list_available_to_both():
    from app.core.tools import REGISTRY

    assert "task_list" in {t.name for t in REGISTRY.tools_for("agent")}
    assert "task_list" in {t.name for t in REGISTRY.tools_for("cm")}


def test_task_update_available_to_both():
    from app.core.tools import REGISTRY

    assert "task_update" in {t.name for t in REGISTRY.tools_for("agent")}
    assert "task_update" in {t.name for t in REGISTRY.tools_for("cm")}


def test_plan_read_available_to_both():
    from app.core.tools import REGISTRY

    assert "plan_read" in {t.name for t in REGISTRY.tools_for("agent")}
    assert "plan_read" in {t.name for t in REGISTRY.tools_for("cm")}


# ---------------------------------------------------------------------------
# task_list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_list_returns_tasks():
    from app.core.tools._task_list import TaskListTool
    from app.models.task import TaskState

    tool = TaskListTool()
    room_id = _uuid()
    ctx = _make_ctx(room_id=room_id)

    task1_id = _uuid()
    task2_id = _uuid()
    agent_id = _uuid()

    task1 = MagicMock()
    task1.id = task1_id
    task1.title = "Implement feature"
    task1.state = TaskState.IN_PROGRESS
    task1.assigned_agent_id = agent_id
    task1.room_id = room_id
    task1.created_at = MagicMock()

    task2 = MagicMock()
    task2.id = task2_id
    task2.title = "Write tests"
    task2.state = TaskState.ASSIGNED
    task2.assigned_agent_id = None
    task2.room_id = room_id
    task2.created_at = MagicMock()

    # First execute() call is for Task query; second for TaskDependency
    task_result = MagicMock()
    task_result.scalars.return_value.all.return_value = [task1, task2]

    dep_result = MagicMock()
    dep_result.scalars.return_value.all.return_value = []

    ctx.db.execute = AsyncMock(side_effect=[task_result, dep_result])

    result = await tool.execute(ctx, {})
    data = json.loads(result)
    assert len(data) == 2
    assert data[0]["title"] == "Implement feature"
    assert data[0]["state"] == "in_progress"
    assert data[0]["assigned_agent_id"] == str(agent_id)
    assert data[1]["depends_on"] == []


@pytest.mark.asyncio
async def test_task_list_empty_room():
    from app.core.tools._task_list import TaskListTool

    tool = TaskListTool()
    ctx = _make_ctx()

    task_result = MagicMock()
    task_result.scalars.return_value.all.return_value = []
    ctx.db.execute = AsyncMock(return_value=task_result)

    result = await tool.execute(ctx, {})
    assert "No tasks" in result


# ---------------------------------------------------------------------------
# task_update
# ---------------------------------------------------------------------------


def _make_task_mock(task_id, room_id, state):
    from app.models.task import DispatchStatus, TaskState

    t = MagicMock()
    t.id = task_id
    t.room_id = room_id
    t.title = "My Task"
    t.state = TaskState(state)
    t.dispatch_status = DispatchStatus.READY
    t.assigned_agent_id = _uuid()
    t.goal_id = None
    t.updated_at = None
    return t


@pytest.mark.asyncio
async def test_task_update_valid_transition():
    from app.core.tools._task_update import TaskUpdateTool

    tool = TaskUpdateTool()
    room_id = _uuid()
    task_id = _uuid()
    ctx = _make_ctx(room_id=room_id)

    task = _make_task_mock(task_id, room_id, "assigned")

    # Advisory lock call (resource_lock) does an execute; then the task SELECT
    lock_result = MagicMock()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task

    ctx.db.execute = AsyncMock(side_effect=[lock_result, task_result])

    with (
        patch("app.core.events.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=MagicMock(goal_id=None, plan_id=None)),
        patch("app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock,
              return_value="started"),
    ):
        result = await tool.execute(ctx, {"task_id": str(task_id), "state": "in_progress"})

    data = json.loads(result)
    assert data["updated"] is True
    assert data["state"] == "in_progress"


@pytest.mark.asyncio
async def test_task_update_starts_ready_direct_task_after_preflight():
    from app.core.tools._task_update import TaskUpdateTool
    from app.models.task import DispatchStatus, TaskState

    tool = TaskUpdateTool()
    room_id = _uuid()
    task_id = _uuid()
    ctx = _make_ctx(room_id=room_id)
    task = _make_task_mock(task_id, room_id, "assigned")
    task.dispatch_status = DispatchStatus.READY

    lock_result = MagicMock()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task
    ctx.db.execute = AsyncMock(side_effect=[lock_result, task_result])

    with (
        patch("app.core.events.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=MagicMock(goal_id=None, plan_id=None)) as mock_preflight,
        patch("app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock,
              return_value="started") as mock_dispatch,
    ):
        result = await tool.execute(ctx, {"task_id": str(task_id), "state": "in_progress"})

    assert json.loads(result)["state"] == "in_progress"
    # target_state, not the task's current state: the preflight has to judge
    # the state the task is moving into (see tests/test_task_back_to_work.py).
    mock_preflight.assert_awaited_once_with(
        task, ctx.db, target_state=TaskState.IN_PROGRESS,
    )
    mock_dispatch.assert_awaited_once_with(
        task, room_id, ctx.db, source="task_update_tool",
    )


@pytest.mark.asyncio
async def test_task_update_keeps_task_assigned_when_dispatch_preflight_fails():
    from app.core.proxy_session import TaskDispatchPreflightError
    from app.core.tools._task_update import TaskUpdateTool
    from app.models.task import DispatchStatus, TaskState

    tool = TaskUpdateTool()
    room_id = _uuid()
    task_id = _uuid()
    ctx = _make_ctx(room_id=room_id)
    task = _make_task_mock(task_id, room_id, "assigned")
    task.dispatch_status = DispatchStatus.READY

    lock_result = MagicMock()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task
    ctx.db.execute = AsyncMock(side_effect=[lock_result, task_result])

    with patch(
        "app.core.proxy_session.prepare_task_dispatch",
        new_callable=AsyncMock,
        side_effect=TaskDispatchPreflightError("no_chat_master", "coordinator not configured"),
    ):
        result = await tool.execute(ctx, {"task_id": str(task_id), "state": "in_progress"})

    assert "not moved to in_progress" in result
    assert task.state == TaskState.ASSIGNED
    ctx.db.add.assert_not_called()


@pytest.mark.asyncio
async def test_task_update_does_not_treat_stale_dispatched_flag_as_running():
    from app.core.tools._task_update import TaskUpdateTool
    from app.models.task import DispatchStatus, TaskState

    tool = TaskUpdateTool()
    room_id = _uuid()
    task_id = _uuid()
    ctx = _make_ctx(room_id=room_id)
    task = _make_task_mock(task_id, room_id, "assigned")
    task.dispatch_status = DispatchStatus.DISPATCHED

    lock_result = MagicMock()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task
    ctx.db.execute = AsyncMock(side_effect=[lock_result, task_result])

    result = await tool.execute(ctx, {"task_id": str(task_id), "state": "in_progress"})

    assert "already claimed" in result
    assert task.state == TaskState.ASSIGNED
    ctx.db.add.assert_not_called()


@pytest.mark.asyncio
async def test_task_update_assigned_to_waiting_review():
    """A continuation task starts ASSIGNED with work already done — it must be
    able to go straight to waiting_review without a spurious in_progress hop."""
    from app.core.tools._task_update import TaskUpdateTool

    tool = TaskUpdateTool()
    room_id = _uuid()
    task_id = _uuid()
    ctx = _make_ctx(room_id=room_id)

    task = _make_task_mock(task_id, room_id, "assigned")

    lock_result = MagicMock()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task

    ctx.db.execute = AsyncMock(side_effect=[lock_result, task_result])

    with patch("app.core.events.emit_event", new_callable=AsyncMock):
        result = await tool.execute(ctx, {"task_id": str(task_id), "state": "waiting_review"})

    data = json.loads(result)
    assert data["updated"] is True
    assert data["state"] == "waiting_review"


@pytest.mark.asyncio
async def test_task_update_invalid_transition():
    from app.core.tools._task_update import TaskUpdateTool

    tool = TaskUpdateTool()
    room_id = _uuid()
    task_id = _uuid()
    ctx = _make_ctx(room_id=room_id)

    task = _make_task_mock(task_id, room_id, "assigned")

    lock_result = MagicMock()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task

    ctx.db.execute = AsyncMock(side_effect=[lock_result, task_result])

    result = await tool.execute(ctx, {"task_id": str(task_id), "state": "completed"})
    assert "Error" in result
    assert "transition" in result


@pytest.mark.asyncio
async def test_task_update_not_found():
    from app.core.tools._task_update import TaskUpdateTool

    tool = TaskUpdateTool()
    ctx = _make_ctx()

    lock_result = MagicMock()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = None

    ctx.db.execute = AsyncMock(side_effect=[lock_result, task_result])

    result = await tool.execute(ctx, {"task_id": str(_uuid()), "state": "in_progress"})
    assert "not found" in result


@pytest.mark.asyncio
async def test_task_update_invalid_uuid():
    from app.core.tools._task_update import TaskUpdateTool

    tool = TaskUpdateTool()
    ctx = _make_ctx()
    result = await tool.execute(ctx, {"task_id": "not-a-uuid", "state": "in_progress"})
    assert "invalid task_id" in result


@pytest.mark.asyncio
async def test_task_update_emits_event():
    from app.core.tools._task_update import TaskUpdateTool

    tool = TaskUpdateTool()
    room_id = _uuid()
    task_id = _uuid()
    ctx = _make_ctx(room_id=room_id)

    task = _make_task_mock(task_id, room_id, "assigned")

    lock_result = MagicMock()
    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task

    ctx.db.execute = AsyncMock(side_effect=[lock_result, task_result])

    emitted: list[dict] = []

    async def _capture(**kwargs):
        emitted.append(kwargs)

    with (
        patch("app.core.events.emit_event", side_effect=_capture),
        patch("app.core.proxy_session.prepare_task_dispatch", new_callable=AsyncMock,
              return_value=MagicMock(goal_id=None, plan_id=None)),
        patch("app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock,
              return_value="started"),
    ):
        await tool.execute(ctx, {"task_id": str(task_id), "state": "in_progress"})

    assert len(emitted) == 1
    assert emitted[0]["event_type"] == "TASK_UPDATED"
    assert emitted[0]["metadata"]["changes"]["state"] == "in_progress"


@pytest.mark.asyncio
async def test_task_update_no_fields_error():
    from app.core.tools._task_update import TaskUpdateTool

    tool = TaskUpdateTool()
    ctx = _make_ctx()
    result = await tool.execute(ctx, {"task_id": str(_uuid())})
    assert "Error" in result


# ---------------------------------------------------------------------------
# task_create (CM only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_create_emits_event():
    from app.core.tools._task_create import TaskCreateTool

    tool = TaskCreateTool()
    assert "cm" in tool.roles
    assert "agent" not in tool.roles

    room_id = _uuid()
    ctx = _make_ctx(room_id=room_id)

    created_task = MagicMock()
    created_task.id = _uuid()
    created_task.title = "New task"
    created_task.state = MagicMock(value="assigned")
    created_task.assigned_agent_id = None

    lock_result = MagicMock()
    # Nothing found by any lookup — in particular no goal already formed by this
    # dialogue, which would refuse a goal-less task (core/dialogue_scope).
    lock_result.scalars.return_value.first.return_value = None
    ctx.db.execute = AsyncMock(return_value=lock_result)

    emitted: list[dict] = []

    async def _capture(**kwargs):
        emitted.append(kwargs)

    with patch("app.core.task_deps.create_tasks_with_deps", new_callable=AsyncMock) as mock_deps:
        mock_deps.return_value = [created_task]
        with patch("app.core.events.emit_event", side_effect=_capture):
            result = await tool.execute(ctx, {"title": "New task", "description": "Do the thing."})

    data = json.loads(result)
    assert data["created"] is True
    assert data["title"] == "New task"
    assert len(emitted) == 1
    assert emitted[0]["event_type"] == "TASK_CREATED"


@pytest.mark.asyncio
async def test_task_create_schedules_direct_task_without_goal_or_plan():
    from app.core.tools._task_create import TaskCreateTool
    from app.models.task import DispatchStatus, TaskState

    tool = TaskCreateTool()
    room_id = _uuid()
    ctx = _make_ctx(room_id=room_id)
    created_task = MagicMock()
    created_task.id = _uuid()
    created_task.title = "Direct task"
    created_task.state = TaskState.ASSIGNED
    created_task.assigned_agent_id = _uuid()
    created_task.dispatch_status = DispatchStatus.READY
    created_task.goal_id = None
    empty_result = MagicMock()
    empty_result.scalars.return_value.first.return_value = None
    ctx.db.execute = AsyncMock(return_value=empty_result)

    with (
        patch("app.core.task_deps.create_tasks_with_deps", new_callable=AsyncMock,
              return_value=[created_task]),
        patch("app.core.events.emit_event", new_callable=AsyncMock),
        patch("app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock,
              return_value="started") as mock_dispatch,
    ):
        result = await tool.execute(
            ctx,
            {"title": "Direct task", "assigned_agent_id": str(created_task.assigned_agent_id)},
        )

    assert json.loads(result)["dispatch"] == "scheduled"
    mock_dispatch.assert_awaited_once_with(
        created_task, room_id, ctx.db, source="task_create_tool",
    )


@pytest.mark.asyncio
async def test_task_create_refuses_a_goal_less_task_beside_its_own_goal():
    """One dialogue, one kind of work (core/dialogue_scope).

    A dialogue that already formed a goal must not also hand out a quick task
    with no goal: the two lines of work start from different bases and their
    branches diverge (incident 2026-09-18).
    """
    from app.core.tools._task_create import TaskCreateTool

    tool = TaskCreateTool()
    ctx = _make_ctx()

    goal = MagicMock()
    goal.id = _uuid()
    goal.title = "Fix the compilation of the kanban tests"
    result_row = MagicMock()
    result_row.scalars.return_value.first.return_value = goal
    ctx.db.execute = AsyncMock(return_value=result_row)

    with patch("app.core.task_deps.create_tasks_with_deps", new_callable=AsyncMock) as deps:
        result = await tool.execute(ctx, {"title": "Quick fix"})

    assert result.startswith("Error:")
    assert goal.title in result
    deps.assert_not_awaited()


@pytest.mark.asyncio
async def test_task_create_missing_title():
    from app.core.tools._task_create import TaskCreateTool

    tool = TaskCreateTool()
    ctx = _make_ctx()
    result = await tool.execute(ctx, {"title": ""})
    assert "Error" in result and "title" in result


@pytest.mark.asyncio
async def test_task_create_invalid_goal_id():
    from app.core.tools._task_create import TaskCreateTool

    tool = TaskCreateTool()
    ctx = _make_ctx()
    result = await tool.execute(ctx, {"title": "Task", "goal_id": "bad-uuid"})
    assert "Error" in result and "goal_id" in result


# ---------------------------------------------------------------------------
# plan_read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_read_no_goals():
    from app.core.tools._plan_read import PlanReadTool

    tool = PlanReadTool()
    ctx = _make_ctx()

    goals_result = MagicMock()
    goals_result.scalars.return_value.all.return_value = []
    ctx.db.execute = AsyncMock(return_value=goals_result)

    result = await tool.execute(ctx, {})
    assert "No goals" in result


@pytest.mark.asyncio
async def test_plan_read_returns_plan_with_steps():
    from app.core.tools._plan_read import PlanReadTool

    tool = PlanReadTool()
    room_id = _uuid()
    goal_id = _uuid()
    plan_id = _uuid()
    step_id = _uuid()
    agent_id = _uuid()
    ctx = _make_ctx(room_id=room_id)

    goal = MagicMock()
    goal.id = goal_id
    goal.title = "Build feature X"
    goal.room_id = room_id
    goal.created_at = MagicMock()

    plan = MagicMock()
    plan.id = plan_id
    plan.goal_id = goal_id
    plan.status = MagicMock(value="approved")

    step = MagicMock()
    step.id = step_id
    step.title = "Implement API"
    step.status = MagicMock(value="in_progress")
    step.assigned_agent_id = agent_id
    step.position = 0

    goals_result = MagicMock()
    goals_result.scalars.return_value.all.return_value = [goal]

    plan_result = MagicMock()
    plan_result.scalars.return_value.first.return_value = plan

    steps_result = MagicMock()
    steps_result.scalars.return_value.all.return_value = [step]

    ctx.db.execute = AsyncMock(side_effect=[goals_result, plan_result, steps_result])

    result = await tool.execute(ctx, {})
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["goal_title"] == "Build feature X"
    assert data[0]["plan_status"] == "approved"
    assert len(data[0]["steps"]) == 1
    assert data[0]["steps"][0]["title"] == "Implement API"
    assert data[0]["steps"][0]["status"] == "in_progress"
    assert data[0]["steps"][0]["assigned_agent_id"] == str(agent_id)


@pytest.mark.asyncio
async def test_plan_read_goal_without_plan():
    from app.core.tools._plan_read import PlanReadTool

    tool = PlanReadTool()
    ctx = _make_ctx()

    goal = MagicMock()
    goal.id = _uuid()
    goal.title = "Goal with no plan yet"
    goal.created_at = MagicMock()

    goals_result = MagicMock()
    goals_result.scalars.return_value.all.return_value = [goal]

    plan_result = MagicMock()
    plan_result.scalars.return_value.first.return_value = None  # no plan

    ctx.db.execute = AsyncMock(side_effect=[goals_result, plan_result])

    result = await tool.execute(ctx, {})
    assert "No plans found" in result
