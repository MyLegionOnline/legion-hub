"""Unit tests for Stage 0: skills injected into cm_agent_proxy system prompts.

No DB required — AsyncSession is mocked. Tests verify:
  - Skills section appears when agent has AgentSkill assignments
  - Skills section absent when nothing is assigned / agent not found
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.memory_layer import build_system_prompt
from app.models.market import UserInstalledItem


def _make_db_for_memory_layer(agent, skills):
    """Return an AsyncMock db for AgentSkill-based skill resolution.

    Agent lookup, then the AgentSkill lookup (one assignment per skill), then
    the Skill bodies.
    """
    db = AsyncMock()

    agent_result = MagicMock()
    agent_result.scalars.return_value.first.return_value = agent

    explicit_assignment_result = MagicMock()
    explicit_assignment_result.scalars.return_value.all.return_value = [
        UserInstalledItem(user_id=agent.owner_id, type="skill", name=s.name, skill_id=s.id)
        for s in skills
    ]

    skill_result = MagicMock()
    skill_result.scalars.return_value.all.return_value = skills

    # Agent lookup → AgentSkill lookup → Skill lookup.
    db.execute = AsyncMock(
        side_effect=[agent_result, explicit_assignment_result, skill_result]
    )
    return db


@pytest.mark.asyncio
async def test_skills_section_present_in_cm_agent_proxy():
    from app.models.agent import Agent
    from app.models.skill import Skill
    from app.core.memory_layer import _agent_skills

    skill = Skill(name="art-vision", display_name="Art Vision", description="See art.", skill_md="Paint boldly.")
    agent = Agent(
        owner_id=uuid.uuid4(),
        token=uuid.uuid4().hex,
        nickname="canvas",
        agent_uid="canvas",
        llm_config={},
    )
    db = _make_db_for_memory_layer(agent, [skill])

    result = await _agent_skills(agent.id, db)

    assert "Art Vision" in result
    assert "Paint boldly." in result


@pytest.mark.asyncio
async def test_skills_section_absent_when_nothing_assigned():
    from app.models.agent import Agent
    from app.core.memory_layer import _agent_skills

    agent = Agent(
        owner_id=uuid.uuid4(),
        token=uuid.uuid4().hex,
        nickname="coder",
        agent_uid="coder",
        llm_config={},
    )
    agent_result = MagicMock()
    agent_result.scalars.return_value.first.return_value = agent
    db = AsyncMock()
    db.execute = AsyncMock(return_value=agent_result)

    result = await _agent_skills(agent.id, db)

    assert result == ""


@pytest.mark.asyncio
async def test_skills_not_rendered_when_empty():
    """build_system_prompt must omit the Skills section when skills is empty string."""
    memory = {
        "room_memory": "ctx",
        "repo_snapshot": [],
        "agents_list": [],
        "goal": None,
        "plan_of_goal": None,
        "task_data": None,
        "agent_memory": "",
        "skills": "",
    }
    prompt = build_system_prompt(memory, "cm_agent_proxy_agent")
    assert "## Skills" not in prompt


@pytest.mark.asyncio
async def test_skills_rendered_when_present():
    """build_system_prompt must include the Skills section when skills is non-empty."""
    memory = {
        "room_memory": "ctx",
        "repo_snapshot": [],
        "agents_list": [],
        "goal": None,
        "plan_of_goal": None,
        "task_data": None,
        "agent_memory": "",
        "skills": "## Skill: Art Vision\nSee art.\n\nPaint boldly.",
    }
    prompt = build_system_prompt(memory, "cm_agent_proxy_agent")
    assert "## Skills" in prompt
    assert "Paint boldly." in prompt
