"""Unit tests for memory_layer (Step 4).

All tests are pure unit tests — DB helpers are mocked so no PostgreSQL is needed.
Validates §3.5 presence/absence rules for all 6 session kinds.
"""
from __future__ import annotations

import types
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from app.core.memory_layer import build_memory_layer, build_system_prompt

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_ROOM_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()
_GOAL_ID = uuid.uuid4()
_TASK_ID = uuid.uuid4()

_ROOM_MEMORY = "# Project Summary\n\nSome content."
_REPO_SNAPSHOT = [{"path": "/README.md", "version": 1}, {"path": "/src/app.py", "version": 3}]
_AGENTS_LIST = [
    {"id": str(_AGENT_ID), "nickname": "agent-alpha", "short_description": "Does backend work"}
]
_GOAL = {
    "id": str(_GOAL_ID),
    "title": "Build landing page",
    "description": "Next.js landing page",
    "status": "active",
}
_PLAN = {
    "id": str(uuid.uuid4()),
    "status": "approved",
    "steps": [
        {
            "position": 0,
            "title": "Design",
            "description": "Create mockups",
            "status": "pending",
            "assigned_agent_id": None,
            "is_parallel": False,
        }
    ],
}
_TASK_DATA = {
    "id": str(_TASK_ID),
    "title": "Implement hero section",
    "description": "Build the top section of the landing page",
    "state": "in_progress",
    "assigned_agent_id": str(_AGENT_ID),
    "goal_id": str(_GOAL_ID),
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-02T00:00:00",
    "state_history": [],
    "file_links": [],
}
_LIST_OF_GOALS = [_GOAL]
_LIST_TASKS = [
    {
        "id": str(_TASK_ID),
        "title": "Implement hero section",
        "state": "in_progress",
        "assigned_agent_id": str(_AGENT_ID),
        "goal_id": str(_GOAL_ID),
        "updated_at": "2024-01-02T00:00:00",
    }
]
_AGENT_MEMORY = "# Agent Memory\n\nI worked on the landing page last session."
_AGENT_SKILLS = "## Skill: Art Vision\nSee art."
_BOARD_SNAPSHOT = {
    "goals": [],
    "without_goal": {"columns": {}, "collapsed": {}, "total": 0},
    "blocked_summary": [],
    "counters": {"total": 0, "active": 0, "blocked": 0, "completed": 0},
    "scan_truncated": False,
}

MODULE = "app.core.memory_layer"

_ALL_KEYS = {
    "room_memory",
    "repo_snapshot",
    "agents_list",
    "goal",
    "list_of_goals",
    "plan_of_goal",
    "task_data",
    "list_tasks_last10",
    "board_snapshot",
    "agent_memory",
}


def _mock_helpers() -> dict:
    return {
        "_board_snapshot": AsyncMock(return_value=_BOARD_SNAPSHOT),
        "_room_memory": AsyncMock(return_value=_ROOM_MEMORY),
        "_repo_snapshot": AsyncMock(return_value=_REPO_SNAPSHOT),
        "_agents_list_short": AsyncMock(return_value=_AGENTS_LIST),
        "_goal": AsyncMock(return_value=_GOAL),
        "_list_of_goals": AsyncMock(return_value=_LIST_OF_GOALS),
        "_plan_of_goal": AsyncMock(return_value=_PLAN),
        "_task_data": AsyncMock(return_value=_TASK_DATA),
        "_list_tasks_last10": AsyncMock(return_value=_LIST_TASKS),
        "_agent_memory": AsyncMock(return_value=_AGENT_MEMORY),
        "_agent_skills": AsyncMock(return_value=_AGENT_SKILLS),
    }


def _session(**attrs):
    defaults = {"room_id": _ROOM_ID}
    defaults.update(attrs)
    return types.SimpleNamespace(**defaults)


async def _build(session_ns) -> dict:
    db = AsyncMock()
    with patch.multiple(MODULE, **_mock_helpers()):
        return await build_memory_layer(session_ns, db)


# ---------------------------------------------------------------------------
# §3.5 presence/absence tests
# ---------------------------------------------------------------------------


class TestUserAgentP1:
    """UA-p1: task-focused agent session."""

    _REQUIRED = {
        "room_memory", "repo_snapshot", "agents_list", "goal", "plan_of_goal",
        "task_data", "agent_memory", "skills",
    }
    _FORBIDDEN = {"list_of_goals", "list_tasks_last10"}

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        s = _session(kind="user_agent_p1", goal_id=_GOAL_ID, task_id=_TASK_ID, agent_id=_AGENT_ID)
        layer = await _build(s)
        assert self._REQUIRED <= set(layer.keys())

    @pytest.mark.asyncio
    async def test_forbidden_keys_absent(self):
        s = _session(kind="user_agent_p1", goal_id=_GOAL_ID, task_id=_TASK_ID, agent_id=_AGENT_ID)
        layer = await _build(s)
        assert not self._FORBIDDEN & set(layer.keys())

    @pytest.mark.asyncio
    async def test_exact_keys(self):
        s = _session(kind="user_agent_p1", goal_id=_GOAL_ID, task_id=_TASK_ID, agent_id=_AGENT_ID)
        layer = await _build(s)
        assert set(layer.keys()) == self._REQUIRED


class TestUserAgentP2:
    """UA-p2: open agent conversation, not tied to a task."""

    _REQUIRED = {"room_memory", "repo_snapshot", "agents_list", "list_of_goals", "agent_memory", "skills"}
    _FORBIDDEN = {"goal", "plan_of_goal", "task_data", "list_tasks_last10"}

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        s = _session(kind="user_agent_p2", agent_id=_AGENT_ID)
        layer = await _build(s)
        assert self._REQUIRED <= set(layer.keys())

    @pytest.mark.asyncio
    async def test_forbidden_keys_absent(self):
        s = _session(kind="user_agent_p2", agent_id=_AGENT_ID)
        layer = await _build(s)
        assert not self._FORBIDDEN & set(layer.keys())

    @pytest.mark.asyncio
    async def test_exact_keys(self):
        s = _session(kind="user_agent_p2", agent_id=_AGENT_ID)
        layer = await _build(s)
        assert set(layer.keys()) == self._REQUIRED


class TestUserCmP1:
    """UCM-p1: ChatMaster session tied to an existing Goal."""

    # Stage 3.2: board_snapshot replaces the flat list_tasks_last10 (defect M2).
    _REQUIRED = {"room_memory", "repo_snapshot", "agents_list", "goal", "plan_of_goal", "board_snapshot", "agent_memory"}
    _FORBIDDEN = {"list_of_goals", "task_data", "list_tasks_last10"}

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        s = _session(kind="user_cm_p1", goal_id=_GOAL_ID, chat_master_agent_id=_AGENT_ID)
        layer = await _build(s)
        assert self._REQUIRED <= set(layer.keys())

    @pytest.mark.asyncio
    async def test_forbidden_keys_absent(self):
        s = _session(kind="user_cm_p1", goal_id=_GOAL_ID, chat_master_agent_id=_AGENT_ID)
        layer = await _build(s)
        assert not self._FORBIDDEN & set(layer.keys())

    @pytest.mark.asyncio
    async def test_exact_keys(self):
        s = _session(kind="user_cm_p1", goal_id=_GOAL_ID, chat_master_agent_id=_AGENT_ID)
        layer = await _build(s)
        assert set(layer.keys()) == self._REQUIRED


class TestUserCmP2:
    """UCM-p2: ChatMaster session for a new topic."""

    # Stage 3.2: a p2 session used to see no project state at all (defect M1) —
    # it now gets the goals and the board, which is what replaces replaying the
    # previous conversation.
    _REQUIRED = {
        "room_memory", "repo_snapshot", "agents_list", "agent_memory",
        "list_of_goals", "board_snapshot",
    }
    _FORBIDDEN = {"goal", "plan_of_goal", "task_data", "list_tasks_last10"}

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        s = _session(kind="user_cm_p2", chat_master_agent_id=_AGENT_ID)
        layer = await _build(s)
        assert self._REQUIRED <= set(layer.keys())

    @pytest.mark.asyncio
    async def test_forbidden_keys_absent(self):
        s = _session(kind="user_cm_p2", chat_master_agent_id=_AGENT_ID)
        layer = await _build(s)
        assert not self._FORBIDDEN & set(layer.keys())

    @pytest.mark.asyncio
    async def test_exact_keys(self):
        s = _session(kind="user_cm_p2", chat_master_agent_id=_AGENT_ID)
        layer = await _build(s)
        assert set(layer.keys()) == self._REQUIRED


class TestCmSelfPlan:
    """cm_self_plan: ChatMaster decomposing a plan into tasks (§4)."""

    _REQUIRED = {"room_memory", "repo_snapshot", "agents_list"}
    _FORBIDDEN = {"goal", "list_of_goals", "plan_of_goal", "task_data", "list_tasks_last10", "agent_memory"}

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        s = _session(kind="cm_self_plan", chat_master_agent_id=_AGENT_ID, goal_id=_GOAL_ID, plan_id=uuid.uuid4())
        layer = await _build(s)
        assert self._REQUIRED <= set(layer.keys())

    @pytest.mark.asyncio
    async def test_forbidden_keys_absent(self):
        s = _session(kind="cm_self_plan", chat_master_agent_id=_AGENT_ID, goal_id=_GOAL_ID, plan_id=uuid.uuid4())
        layer = await _build(s)
        assert not self._FORBIDDEN & set(layer.keys())

    @pytest.mark.asyncio
    async def test_exact_keys(self):
        s = _session(kind="cm_self_plan", chat_master_agent_id=_AGENT_ID, goal_id=_GOAL_ID, plan_id=uuid.uuid4())
        layer = await _build(s)
        assert set(layer.keys()) == self._REQUIRED


class TestAgentSubtask:
    """agent_subtask: run_subtask's own row, minimal layer like cm_self_plan (§5.4).

    Not actually exercised by _run_subtask.py — the isolated child loop still
    builds its system prompt from the PARENT session — this only guarantees
    build_memory_layer never raises if this kind's own row is ever passed in.
    """

    _REQUIRED = {"room_memory", "repo_snapshot", "agents_list"}
    _FORBIDDEN = {"goal", "list_of_goals", "plan_of_goal", "task_data", "list_tasks_last10", "agent_memory"}

    @pytest.mark.asyncio
    async def test_required_keys_present(self):
        s = _session(kind="agent_subtask", agent_id=_AGENT_ID, parent_session_id=uuid.uuid4())
        layer = await _build(s)
        assert self._REQUIRED <= set(layer.keys())

    @pytest.mark.asyncio
    async def test_exact_keys(self):
        s = _session(kind="agent_subtask", agent_id=_AGENT_ID, parent_session_id=uuid.uuid4())
        layer = await _build(s)
        assert set(layer.keys()) == self._REQUIRED


class TestCmAgentProxy:
    """cm_agent_proxy: both sides get full memory layer (§5.1, §5.2)."""

    _REQUIRED = {"room_memory", "repo_snapshot", "agents_list", "goal", "plan_of_goal", "task_data", "agent_memory", "skills"}
    _FORBIDDEN = {"list_of_goals", "list_tasks_last10"}

    @pytest.mark.asyncio
    async def test_agent_side_keys(self):
        """The executing agent gets its own task — never the whole board (3.2)."""
        s = _session(
            kind="cm_agent_proxy",
            side="agent",
            goal_id=_GOAL_ID,
            task_id=_TASK_ID,
            agent_id=_AGENT_ID,
        )
        layer = await _build(s)
        assert set(layer.keys()) == self._REQUIRED

    @pytest.mark.asyncio
    async def test_chat_master_side_keys(self):
        s = _session(
            kind="cm_agent_proxy",
            side="chat_master",
            goal_id=_GOAL_ID,
            task_id=_TASK_ID,
            agent_id=_AGENT_ID,
            peer_session_id=uuid.uuid4(),
        )
        layer = await _build(s)
        # The coordinator side also sees the board: it decides what else is in
        # flight, the agent side does not (3.2).
        assert set(layer.keys()) == self._REQUIRED | {"board_snapshot"}

    @pytest.mark.asyncio
    async def test_forbidden_keys_absent(self):
        s = _session(kind="cm_agent_proxy", side="agent", goal_id=_GOAL_ID, task_id=_TASK_ID, agent_id=_AGENT_ID)
        layer = await _build(s)
        assert not self._FORBIDDEN & set(layer.keys())


# ---------------------------------------------------------------------------
# Unknown kind raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownKind:
    @pytest.mark.asyncio
    async def test_raises_value_error(self):
        s = _session(kind="nonexistent_kind")
        with pytest.raises(ValueError, match="Unknown session kind"):
            await _build(s)


# ---------------------------------------------------------------------------
# Task Data structure validation
# ---------------------------------------------------------------------------


class TestTaskDataFormat:
    """Verify Task Data dict has all required fields (§7.5, task_data.md)."""

    _REQUIRED_FIELDS = {
        "id", "title", "description", "state",
        "assigned_agent_id", "goal_id",
        "created_at", "updated_at",
        "state_history", "file_links",
    }

    @pytest.mark.asyncio
    async def test_task_data_has_all_fields(self):
        s = _session(kind="user_agent_p1", goal_id=_GOAL_ID, task_id=_TASK_ID, agent_id=_AGENT_ID)
        layer = await _build(s)
        assert set(layer["task_data"].keys()) >= self._REQUIRED_FIELDS

    @pytest.mark.asyncio
    async def test_state_history_is_list(self):
        s = _session(kind="user_agent_p1", goal_id=_GOAL_ID, task_id=_TASK_ID, agent_id=_AGENT_ID)
        layer = await _build(s)
        assert isinstance(layer["task_data"]["state_history"], list)

    @pytest.mark.asyncio
    async def test_file_links_is_list(self):
        s = _session(kind="user_agent_p1", goal_id=_GOAL_ID, task_id=_TASK_ID, agent_id=_AGENT_ID)
        layer = await _build(s)
        assert isinstance(layer["task_data"]["file_links"], list)


# ---------------------------------------------------------------------------
# build_system_prompt smoke tests
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Smoke-test that build_system_prompt returns non-empty strings for each kind."""

    @pytest.mark.parametrize("kind", [
        "user_agent_p1",
        "user_agent_p2",
        "user_cm_p1",
        "user_cm_p2",
        "cm_self_plan",
        "cm_agent_proxy",
    ])
    def test_prompt_contains_kind_header(self, kind):
        layer = {
            "room_memory": _ROOM_MEMORY,
            "repo_snapshot": _REPO_SNAPSHOT,
            "agents_list": _AGENTS_LIST,
        }
        prompt = build_system_prompt(layer, kind)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "## Room Memory" in prompt
        assert "## Repository Snapshot" in prompt
        assert "## Agents in Room" in prompt

    def test_prompt_includes_goal_section(self):
        layer = {
            "room_memory": _ROOM_MEMORY,
            "repo_snapshot": _REPO_SNAPSHOT,
            "agents_list": _AGENTS_LIST,
            "goal": _GOAL,
            "plan_of_goal": _PLAN,
            "task_data": _TASK_DATA,
            "agent_memory": _AGENT_MEMORY,
        }
        prompt = build_system_prompt(layer, "user_agent_p1")
        assert "## Goal" in prompt
        assert "Build landing page" in prompt
        assert "## Plan of Goal" in prompt
        assert "## Task Data" in prompt
        assert "## Agent Memory" in prompt

    def test_task_agent_prompt_starts_with_efficient_execution_rules(self):
        """Every newly dispatched task agent gets the anti-loop workflow first."""
        prompt = build_system_prompt({}, "cm_agent_proxy_agent")

        assert "TASK EXECUTION EFFICIENCY — MANDATORY:" in prompt
        assert "targeted read → one consolidated patch → one verification" in prompt
        assert prompt.index("TASK EXECUTION EFFICIENCY") < prompt.index("WORKSPACE TOOLS")

    def test_prompt_excludes_missing_sections(self):
        layer = {
            "room_memory": _ROOM_MEMORY,
            "repo_snapshot": _REPO_SNAPSHOT,
            "agents_list": _AGENTS_LIST,
        }
        prompt = build_system_prompt(layer, "cm_self_plan")
        assert "## Goal" not in prompt
        assert "## Task Data" not in prompt
        assert "## Agent Memory" not in prompt

    def test_prompt_list_of_goals_section(self):
        layer = {
            "room_memory": _ROOM_MEMORY,
            "repo_snapshot": _REPO_SNAPSHOT,
            "agents_list": _AGENTS_LIST,
            "list_of_goals": _LIST_OF_GOALS,
            "agent_memory": _AGENT_MEMORY,
        }
        prompt = build_system_prompt(layer, "user_agent_p2")
        assert "## Goals" in prompt
        assert "Build landing page" in prompt

    def test_prompt_list_tasks_section(self):
        layer = {
            "room_memory": _ROOM_MEMORY,
            "repo_snapshot": _REPO_SNAPSHOT,
            "agents_list": _AGENTS_LIST,
            "goal": _GOAL,
            "plan_of_goal": _PLAN,
            "list_tasks_last10": _LIST_TASKS,
            "agent_memory": _AGENT_MEMORY,
        }
        prompt = build_system_prompt(layer, "user_cm_p1")
        assert "## Tasks (last 10)" in prompt
        assert "Implement hero section" in prompt


# ---------------------------------------------------------------------------
# Stage 6 — user_cm_p2 "REFERENCED IMAGES" prompt section
# ---------------------------------------------------------------------------


class TestUserCmP2ReferencedImages:
    """Verify the REFERENCED IMAGES block is present and correct in user_cm_p2."""

    def _cm_p2_prompt(self) -> str:
        from app.core.memory_layer import _KIND_HEADERS  # type: ignore[attr-defined]
        return _KIND_HEADERS["user_cm_p2"]

    def test_referenced_images_section_present(self):
        prompt = self._cm_p2_prompt()
        assert "REFERENCED IMAGES" in prompt

    def test_describe_image_tool_named(self):
        prompt = self._cm_p2_prompt()
        assert "mcp__mcp-vision__describe_image" in prompt

    def test_extract_design_spec_tool_named(self):
        prompt = self._cm_p2_prompt()
        assert "mcp__mcp-vision__extract_design_spec" in prompt

    def test_observation_read_recall_path(self):
        """The prompt must tell the model to use observation_read for compacted turns."""
        prompt = self._cm_p2_prompt()
        assert "observation_read" in prompt

    def test_call_once_not_twice(self):
        """The prompt must forbid calling the vision tool again for the same image."""
        prompt = self._cm_p2_prompt()
        assert "ONCE" in prompt or "once" in prompt

    def test_goal_description_embed_instruction(self):
        """goal_description placeholder must mention embedding visual facts."""
        prompt = self._cm_p2_prompt()
        assert "layout agent" in prompt

    def test_never_ask_user_to_describe(self):
        prompt = self._cm_p2_prompt()
        assert "Never ask the user to describe" in prompt

    def test_referenced_images_before_your_workflow(self):
        """The REFERENCED IMAGES block must appear before YOUR WORKFLOW."""
        prompt = self._cm_p2_prompt()
        idx_images = prompt.index("REFERENCED IMAGES")
        idx_workflow = prompt.index("YOUR WORKFLOW")
        assert idx_images < idx_workflow

    def test_file_reading_limit_unchanged(self):
        """The 5-file-read limit must still be present."""
        prompt = self._cm_p2_prompt()
        assert "at most 5 file reads" in prompt
