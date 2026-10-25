"""internal design notes Stage 11 — prompt block + room_config tool.

Pure unit tests: DB calls are monkeypatched so no PostgreSQL is needed.
"""
from __future__ import annotations

import types
import uuid
from unittest.mock import ANY, AsyncMock, patch

import pytest

from app.core.memory_layer import (
    _render_room_config,
    _room_config_block,
    build_memory_layer,
    build_system_prompt,
)

_ROOM_ID = uuid.uuid4()


# --- _render_room_config -----------------------------------------------------


def test_render_room_config_variables_only():
    out = _render_room_config({"variables": {"APP_ENV": "staging"}, "secrets": []})
    assert "## Room variables and secrets" in out
    assert "APP_ENV=staging" in out
    assert "Secrets" not in out


def test_render_room_config_secrets_only():
    out = _render_room_config(
        {
            "variables": {},
            "secrets": [{"env_name": "DEPLOY_TOKEN", "scope": ["deploy"], "purpose": "room_env"}],
        }
    )
    assert "DEPLOY_TOKEN (scope: deploy)" in out
    assert 'bash(cmd="…", secrets=["NAME"])' in out
    assert "room_config" in out  # fallback-tool mention


def test_render_room_config_secret_no_scope_reads_explicit_only():
    out = _render_room_config(
        {"variables": {}, "secrets": [{"env_name": "X", "scope": [], "purpose": "room_env"}]}
    )
    assert "X (scope: explicit request only)" in out


def test_render_room_config_cm_flavor_no_bash_mention():
    """ChatMaster has no bash tool — the block must not tell it to call one."""
    out = _render_room_config(
        {
            "variables": {"APP_ENV": "staging"},
            "secrets": [{"env_name": "DEPLOY_TOKEN", "scope": ["deploy"], "purpose": "room_env"}],
            "for_agent": False,
        }
    )
    assert "APP_ENV=staging" in out
    assert "DEPLOY_TOKEN (scope: deploy)" in out
    assert 'bash(cmd="…"' not in out  # not told to call bash directly — it has no such tool
    assert "env_name" in out  # told to quote it in a task description instead
    assert "room_config" in out


def test_render_room_config_cm_flavor_variables_only_still_gets_footer():
    """Even without secrets, ChatMaster gets the reference-by-name instruction —
    unlike the agent flavor, which stays silent when there is nothing to call
    bash(secrets=[...]) for."""
    out = _render_room_config({"variables": {"APP_ENV": "staging"}, "secrets": [], "for_agent": False})
    assert "env_name" in out
    assert 'bash(cmd="…"' not in out  # not told to call bash directly — it has no such tool


def test_render_room_config_agent_flavor_variables_only_has_no_footer():
    out = _render_room_config({"variables": {"APP_ENV": "staging"}, "secrets": [], "for_agent": True})
    assert "Usage" not in out


def test_render_room_config_names_the_repository():
    """internal design notes Stage 14 — the Room's repository is a fact an agent
    cannot get from the workspace: the checkout has no git remote, so an empty
    `git remote -v` there used to read as "this Room has no repository" and
    sent a deploy looking for a file transport instead.
    """
    out = _render_room_config({
        "variables": {},
        "secrets": [],
        "repo": {"url": "https://github.com/acme/site.git", "branch": "main", "provider": "github"},
        "for_agent": True,
    })
    assert "https://github.com/acme/site.git" in out
    assert "branch main" in out
    assert "git remote -v" in out


# --- _room_config_block -------------------------------------------------------


@pytest.mark.asyncio
async def test_room_config_block_none_when_empty():
    with patch("app.core.room_config.effective_variables", AsyncMock(return_value={})), patch(
        "app.core.secrets.room_secrets.list_room_secret_names", AsyncMock(return_value=[])
    ):
        out = await _room_config_block(_ROOM_ID, AsyncMock())
    assert out is None


@pytest.mark.asyncio
async def test_room_config_block_present_with_data():
    with patch(
        "app.core.room_config.effective_variables", AsyncMock(return_value={"APP_ENV": "staging"})
    ), patch("app.core.secrets.room_secrets.list_room_secret_names", AsyncMock(return_value=[])):
        out = await _room_config_block(_ROOM_ID, AsyncMock())
    assert out == {"variables": {"APP_ENV": "staging"}, "secrets": [], "repo": None, "for_agent": True}


@pytest.mark.asyncio
async def test_room_config_block_for_agent_false_is_threaded_through():
    with patch(
        "app.core.room_config.effective_variables", AsyncMock(return_value={"APP_ENV": "staging"})
    ), patch("app.core.secrets.room_secrets.list_room_secret_names", AsyncMock(return_value=[])):
        out = await _room_config_block(_ROOM_ID, AsyncMock(), for_agent=False)
    assert out == {"variables": {"APP_ENV": "staging"}, "secrets": [], "repo": None, "for_agent": False}


@pytest.mark.asyncio
async def test_room_config_block_best_effort_on_error():
    with patch(
        "app.core.room_config.effective_variables", AsyncMock(side_effect=RuntimeError("db down"))
    ):
        out = await _room_config_block(_ROOM_ID, AsyncMock())
    assert out is None


# --- build_memory_layer wiring ------------------------------------------------


def _session(**attrs):
    defaults = {"room_id": _ROOM_ID}
    defaults.update(attrs)
    return types.SimpleNamespace(**defaults)


_BASE_MOCKS = {
    "_room_memory": AsyncMock(return_value=""),
    "_repo_snapshot": AsyncMock(return_value=[]),
    "_agents_list_short": AsyncMock(return_value=[]),
    "_goal": AsyncMock(return_value=None),
    "_list_of_goals": AsyncMock(return_value=[]),
    "_plan_of_goal": AsyncMock(return_value=None),
    "_task_data": AsyncMock(return_value=None),
    "_agent_memory": AsyncMock(return_value=""),
    "_agent_skills": AsyncMock(return_value=""),
    "_task_handoff": AsyncMock(return_value=None),
}


@pytest.mark.asyncio
async def test_room_config_present_for_user_agent_p1():
    mocks = {**_BASE_MOCKS, "_room_config_block": AsyncMock(return_value={"variables": {"A": "1"}, "secrets": []})}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="user_agent_p1", goal_id=uuid.uuid4(), task_id=uuid.uuid4(), agent_id=uuid.uuid4())
        layer = await build_memory_layer(s, AsyncMock())
    assert "room_config" in layer


@pytest.mark.asyncio
async def test_room_config_present_for_user_agent_p2():
    mocks = {**_BASE_MOCKS, "_room_config_block": AsyncMock(return_value={"variables": {"A": "1"}, "secrets": []})}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="user_agent_p2", agent_id=uuid.uuid4())
        layer = await build_memory_layer(s, AsyncMock())
    assert "room_config" in layer


@pytest.mark.asyncio
async def test_room_config_absent_when_room_has_nothing():
    mocks = {**_BASE_MOCKS, "_room_config_block": AsyncMock(return_value=None)}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="user_agent_p2", agent_id=uuid.uuid4())
        layer = await build_memory_layer(s, AsyncMock())
    assert "room_config" not in layer


@pytest.mark.asyncio
async def test_room_config_present_for_cm_agent_proxy_agent_side():
    mocks = {**_BASE_MOCKS, "_room_config_block": AsyncMock(return_value={"variables": {"A": "1"}, "secrets": []})}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="cm_agent_proxy", goal_id=uuid.uuid4(), task_id=uuid.uuid4(), agent_id=uuid.uuid4(), side="agent")
        layer = await build_memory_layer(s, AsyncMock())
    assert "room_config" in layer


@pytest.mark.asyncio
async def test_room_config_present_for_cm_agent_proxy_chat_master_side_cm_flavor():
    """ChatMaster has no bash tool, but still gets the CM-flavored block
    (for_agent=False) — it reviews the agent's work and may need to reference
    an exact env_name when requesting changes (plan §3.7)."""
    block = AsyncMock(return_value={"variables": {"A": "1"}, "secrets": [], "for_agent": False})
    mocks = {
        **_BASE_MOCKS,
        "_room_config_block": block,
        "_board_snapshot": AsyncMock(return_value=None),
    }
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="cm_agent_proxy", goal_id=uuid.uuid4(), task_id=uuid.uuid4(), agent_id=uuid.uuid4(), side="chat_master")
        layer = await build_memory_layer(s, AsyncMock())
    assert layer["room_config"]["for_agent"] is False
    block.assert_called_once_with(_ROOM_ID, ANY, for_agent=False)


@pytest.mark.asyncio
async def test_room_config_agent_side_uses_agent_flavor():
    block = AsyncMock(return_value={"variables": {"A": "1"}, "secrets": []})
    mocks = {**_BASE_MOCKS, "_room_config_block": block}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="cm_agent_proxy", goal_id=uuid.uuid4(), task_id=uuid.uuid4(), agent_id=uuid.uuid4(), side="agent")
        await build_memory_layer(s, AsyncMock())
    block.assert_called_once()
    assert block.call_args.kwargs == {}  # default for_agent=True, not passed explicitly


@pytest.mark.asyncio
async def test_room_config_present_for_user_cm_p1_cm_flavor():
    block = AsyncMock(return_value={"variables": {"A": "1"}, "secrets": [], "for_agent": False})
    mocks = {**_BASE_MOCKS, "_room_config_block": block, "_board_snapshot": AsyncMock(return_value=None), "_list_tasks_last10": AsyncMock(return_value=[])}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="user_cm_p1", goal_id=uuid.uuid4(), chat_master_agent_id=uuid.uuid4())
        layer = await build_memory_layer(s, AsyncMock())
    assert layer["room_config"]["for_agent"] is False
    block.assert_called_once_with(_ROOM_ID, ANY, for_agent=False)


@pytest.mark.asyncio
async def test_room_config_present_for_user_cm_p2_cm_flavor():
    block = AsyncMock(return_value={"variables": {"A": "1"}, "secrets": [], "for_agent": False})
    mocks = {**_BASE_MOCKS, "_room_config_block": block, "_board_snapshot": AsyncMock(return_value=None)}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="user_cm_p2", chat_master_agent_id=uuid.uuid4())
        layer = await build_memory_layer(s, AsyncMock())
    assert layer["room_config"]["for_agent"] is False


@pytest.mark.asyncio
async def test_room_config_present_for_cm_self_plan_cm_flavor():
    block = AsyncMock(return_value={"variables": {"A": "1"}, "secrets": [], "for_agent": False})
    mocks = {**_BASE_MOCKS, "_room_config_block": block}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="cm_self_plan")
        layer = await build_memory_layer(s, AsyncMock())
    assert layer["room_config"]["for_agent"] is False


@pytest.mark.asyncio
async def test_room_config_absent_for_cm_self_plan_when_room_has_nothing():
    mocks = {**_BASE_MOCKS, "_room_config_block": AsyncMock(return_value=None)}
    with patch.multiple("app.core.memory_layer", **mocks):
        s = _session(kind="cm_self_plan")
        layer = await build_memory_layer(s, AsyncMock())
    assert "room_config" not in layer


def test_build_system_prompt_renders_room_config():
    layer = {"room_config": {"variables": {"APP_ENV": "staging"}, "secrets": []}}
    prompt = build_system_prompt(layer, "user_agent_p1")
    assert "APP_ENV=staging" in prompt


# --- room_config tool ---------------------------------------------------------


@pytest.mark.asyncio
async def test_room_config_tool_registered():
    from app.core.tools import REGISTRY

    tool = REGISTRY.get("room_config")
    assert tool is not None
    assert tool.roles == frozenset({"agent", "cm"})
    assert tool.requires_container is False
    assert tool.mutating is False


@pytest.mark.asyncio
async def test_room_config_tool_has_cm_specific_description():
    from app.core.tools import REGISTRY

    tool = REGISTRY.get("room_config")
    cm_desc = tool.ant_schema("cm")["description"]
    agent_desc = tool.ant_schema("agent")["description"]
    assert cm_desc != agent_desc
    assert "bash(" not in cm_desc
    assert "bash(" in agent_desc


@pytest.mark.asyncio
async def test_room_config_tool_execute_lists_names_not_values():
    from app.core.tools import REGISTRY
    from app.core.tools.registry import ToolContext

    tool = REGISTRY.get("room_config")
    ctx = ToolContext(room_id=_ROOM_ID, session_id=uuid.uuid4(), task_id=None, db=AsyncMock())
    with patch(
        "app.core.room_config.effective_variables", AsyncMock(return_value={"APP_ENV": "staging"})
    ), patch(
        "app.core.secrets.room_secrets.list_room_secret_names",
        AsyncMock(return_value=[{"env_name": "DEPLOY_TOKEN", "scope": ["deploy"], "purpose": "room_env"}]),
    ):
        out = await tool.execute(ctx, {})
    assert "APP_ENV=staging" in out
    assert "DEPLOY_TOKEN (scope: deploy)" in out


@pytest.mark.asyncio
async def test_room_config_tool_empty_room():
    from app.core.tools import REGISTRY
    from app.core.tools.registry import ToolContext

    tool = REGISTRY.get("room_config")
    ctx = ToolContext(room_id=_ROOM_ID, session_id=uuid.uuid4(), task_id=None, db=AsyncMock())
    with patch("app.core.room_config.effective_variables", AsyncMock(return_value={})), patch(
        "app.core.secrets.room_secrets.list_room_secret_names", AsyncMock(return_value=[])
    ):
        out = await tool.execute(ctx, {})
    assert "no variables or secrets" in out


@pytest.mark.asyncio
async def test_room_config_tool_rejects_non_string_environment():
    from app.core.tools import REGISTRY
    from app.core.tools.registry import ToolContext

    tool = REGISTRY.get("room_config")
    ctx = ToolContext(room_id=_ROOM_ID, session_id=uuid.uuid4(), task_id=None, db=AsyncMock())
    out = await tool.execute(ctx, {"environment": 123})
    assert out.startswith("Error:")
