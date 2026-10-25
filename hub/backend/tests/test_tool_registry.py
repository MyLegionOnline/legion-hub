"""
Tests for Stage 0: Tool Registry + Stage 1: new tools are registered.

Verifies:
 1. Registry produces correct schemas for all registered tools.
 2. is_registered() routes correctly.
 3. Tool role assignments match spec.
 4. execute() dispatches to the correct tool implementation.
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.tools import REGISTRY, ToolContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimum set of tools that must always be registered for each role.
# Use subset checks (<=) so adding new tools doesn't require updating this snapshot.
_AGENT_TOOL_NAMES = {
    "bash", "bash_output", "bash_kill",
    "read_file", "write_file", "edit_file", "glob", "grep", "git_diff", "find_file",
    "preview_fetch",
    "workspace_service",
    # web_fetch and web_search are conditional (WEB_ACCESS_ENABLED=true + API key)
}
_CM_TOOL_NAMES = {
    "read_file", "git_diff", "grep", "glob",
    # Stage 5: task management and plan inspection
    "task_list", "task_update", "task_create", "plan_read",
}


def _schema_by_name(schemas: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((s for s in schemas if s.get("name") == name), None)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_all_agent_tools_registered():
    assert _AGENT_TOOL_NAMES <= {t.name for t in REGISTRY.tools_for("agent")}


def test_cm_tools_registered():
    assert _CM_TOOL_NAMES <= {t.name for t in REGISTRY.tools_for("cm")}


def test_bash_is_agent_only():
    bash = REGISTRY.get("bash")
    assert bash is not None
    assert "agent" in bash.roles
    assert "cm" not in bash.roles


def test_write_file_is_agent_only():
    wf = REGISTRY.get("write_file")
    assert wf is not None
    assert "agent" in wf.roles
    assert "cm" not in wf.roles


def test_edit_file_is_agent_only():
    ef = REGISTRY.get("edit_file")
    assert ef is not None
    assert "agent" in ef.roles
    assert "cm" not in ef.roles


def test_grep_available_to_agent_and_cm():
    """grep is cheaper than read_file — CM must be able to search instead of
    reading whole files (token budget on large projects)."""
    g = REGISTRY.get("grep")
    assert g is not None
    assert "agent" in g.roles
    assert "cm" in g.roles


def test_glob_available_to_agent_and_cm():
    g = REGISTRY.get("glob")
    assert g is not None
    assert "agent" in g.roles
    assert "cm" in g.roles


def test_find_file_is_agent_only():
    ff = REGISTRY.get("find_file")
    assert ff is not None
    assert "agent" in ff.roles
    assert "cm" not in ff.roles


def test_list_files_not_registered():
    """list_files was renamed to glob in Stage 1."""
    assert not REGISTRY.is_registered("list_files")


def test_read_file_and_git_diff_available_to_cm():
    for name in ("read_file", "git_diff"):
        tool = REGISTRY.get(name)
        assert tool is not None, f"{name} not found"
        assert "cm" in tool.roles, f"{name} should be available to cm"


def test_request_file_read_registered_but_not_in_schemas():
    """request_file_read is in the registry for Stage 4 programmatic use,
    but must NOT appear in any LLM-facing schema until pause/resume is implemented."""
    assert REGISTRY.is_registered("request_file_read")
    agent_names = {s["name"] for s in REGISTRY.ant_schemas_for("agent")}
    cm_names = {s["name"] for s in REGISTRY.ant_schemas_for("cm")}
    assert "request_file_read" not in agent_names
    assert "request_file_read" not in cm_names


def test_close_session_not_registered():
    assert not REGISTRY.is_registered("close_session")


def test_unknown_tool_not_registered():
    assert not REGISTRY.is_registered("nonexistent_tool_xyz")


# ---------------------------------------------------------------------------
# Schema snapshot: Anthropic format
# ---------------------------------------------------------------------------


def test_ant_agent_schemas_have_correct_names():
    schemas = REGISTRY.ant_schemas_for("agent")
    names = {s["name"] for s in schemas}
    assert _AGENT_TOOL_NAMES <= names, f"Missing tools: {_AGENT_TOOL_NAMES - names}"


def test_ant_cm_schemas_have_correct_names():
    schemas = REGISTRY.ant_schemas_for("cm")
    names = {s["name"] for s in schemas}
    assert _CM_TOOL_NAMES <= names, f"Missing CM tools: {_CM_TOOL_NAMES - names}"


def test_ant_schema_structure():
    """Each Anthropic schema has name, description, input_schema."""
    for schema in REGISTRY.ant_schemas_for("agent"):
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        assert isinstance(schema["input_schema"], dict)


def test_ant_bash_schema():
    schema = _schema_by_name(REGISTRY.ant_schemas_for("agent"), "bash")
    assert schema is not None
    inp = schema["input_schema"]
    assert "cmd" in inp["properties"]
    assert "cmd" in inp["required"]
    assert "timeout_sec" in inp["properties"]


def test_ant_read_file_schema():
    schema = _schema_by_name(REGISTRY.ant_schemas_for("agent"), "read_file")
    assert schema is not None
    props = schema["input_schema"]["properties"]
    assert "path" in props
    assert "offset" in props
    assert "limit" in props
    assert "path" in schema["input_schema"]["required"]


def test_ant_write_file_schema():
    schema = _schema_by_name(REGISTRY.ant_schemas_for("agent"), "write_file")
    assert schema is not None
    props = schema["input_schema"]["properties"]
    assert "path" in props
    assert "content" in props


def test_ant_edit_file_schema():
    schema = _schema_by_name(REGISTRY.ant_schemas_for("agent"), "edit_file")
    assert schema is not None
    props = schema["input_schema"]["properties"]
    assert "path" in props
    assert "old_string" in props
    assert "new_string" in props
    assert "replace_all" in props
    req = schema["input_schema"]["required"]
    assert "path" in req
    assert "old_string" in req
    assert "new_string" in req


def test_ant_glob_schema():
    schema = _schema_by_name(REGISTRY.ant_schemas_for("agent"), "glob")
    assert schema is not None
    assert "pattern" in schema["input_schema"]["properties"]


def test_ant_grep_schema():
    schema = _schema_by_name(REGISTRY.ant_schemas_for("agent"), "grep")
    assert schema is not None
    props = schema["input_schema"]["properties"]
    assert "pattern" in props
    assert "glob" in props
    assert "output_mode" in props
    assert "max_results" in props
    assert "pattern" in schema["input_schema"]["required"]


def test_ant_git_diff_schema():
    schema = _schema_by_name(REGISTRY.ant_schemas_for("agent"), "git_diff")
    assert schema is not None
    assert "path" in schema["input_schema"]["properties"]


def test_ant_find_file_schema():
    schema = _schema_by_name(REGISTRY.ant_schemas_for("agent"), "find_file")
    assert schema is not None
    props = schema["input_schema"]["properties"]
    assert "name" in props
    assert "dir" in props
    assert "name" in schema["input_schema"]["required"]


def test_git_diff_has_role_specific_descriptions():
    """CM sees a slightly different git_diff description than agent."""
    agent_schemas = REGISTRY.ant_schemas_for("agent")
    cm_schemas = REGISTRY.ant_schemas_for("cm")
    agent_desc = _schema_by_name(agent_schemas, "git_diff")["description"]
    cm_desc = _schema_by_name(cm_schemas, "git_diff")["description"]
    assert agent_desc != cm_desc
    assert "agent" in cm_desc.lower()


# ---------------------------------------------------------------------------
# Schema snapshot: OpenAI format
# ---------------------------------------------------------------------------


def test_oai_agent_schemas_have_correct_names():
    schemas = REGISTRY.oai_schemas_for("agent")
    names = {s["function"]["name"] for s in schemas}
    assert _AGENT_TOOL_NAMES <= names, f"Missing tools: {_AGENT_TOOL_NAMES - names}"


def test_oai_schema_structure():
    for schema in REGISTRY.oai_schemas_for("agent"):
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn


# ---------------------------------------------------------------------------
# Execute dispatch
# ---------------------------------------------------------------------------


def _make_ctx() -> ToolContext:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    return ToolContext(
        room_id=uuid4(),
        session_id=uuid4(),
        task_id=uuid4(),
        db=db,
    )


@pytest.mark.asyncio
async def test_read_file_execute_returns_content():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_file")
    assert tool is not None

    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as mock_read:
        mock_read.return_value = "hello world\n"
        result = await tool.execute(ctx, {"path": "/workspace/foo.py"})

    assert "hello world" in result
    assert "1\t" in result  # line numbers present
    mock_read.assert_awaited_once_with(str(ctx.room_id), "/workspace/foo.py")


@pytest.mark.asyncio
async def test_read_file_missing_path():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_file")
    result = await tool.execute(ctx, {"path": ""})
    assert "Error" in result


@pytest.mark.asyncio
async def test_read_file_not_found():
    ctx = _make_ctx()
    tool = REGISTRY.get("read_file")
    with patch("app.core.workspaced_client.read_workspace_file", new_callable=AsyncMock) as m:
        m.return_value = None
        result = await tool.execute(ctx, {"path": "/workspace/missing.py"})
    assert "not found" in result.lower() or "File not found" in result


@pytest.mark.asyncio
async def test_write_file_execute():
    ctx = _make_ctx()
    tool = REGISTRY.get("write_file")
    assert tool is not None

    with patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock) as mock_write:
        result = await tool.execute(ctx, {"path": "/workspace/out.py", "content": "x = 1\n"})

    assert "Written" in result
    assert "out.py" in result
    mock_write.assert_awaited_once()


@pytest.mark.asyncio
async def test_glob_execute():
    ctx = _make_ctx()
    tool = REGISTRY.get("glob")
    assert tool is not None

    fake_tree = [
        {"path": "/workspace/src/main.py", "mtime": 1000.0, "size": 100},
        {"path": "/workspace/README.md", "mtime": 900.0, "size": 50},
    ]
    with patch("app.core.workspaced_client.get_workspace_tree", new_callable=AsyncMock) as mock_tree:
        mock_tree.return_value = fake_tree
        result = await tool.execute(ctx, {"pattern": "**/*.py"})

    assert "src/main.py" in result
    assert "README.md" not in result


@pytest.mark.asyncio
async def test_git_diff_execute():
    """No path → --stat summary of all changed files."""
    ctx = _make_ctx()
    tool = REGISTRY.get("git_diff")
    assert tool is not None

    with patch("app.core.workspaced_client.git_diff_workspace", new_callable=AsyncMock) as mock_diff:
        mock_diff.return_value = {"diff": " foo.py | 2 +-\n 1 file changed\n"}
        result = await tool.execute(ctx, {})

    assert "foo.py" in result
    assert "Changed files vs main" in result
    mock_diff.assert_awaited_once_with(
        str(ctx.room_id), from_ref="main", stat=True, merge_base=True
    )


@pytest.mark.asyncio
async def test_git_diff_no_changes():
    ctx = _make_ctx()
    tool = REGISTRY.get("git_diff")
    with patch("app.core.workspaced_client.git_diff_workspace", new_callable=AsyncMock) as mock_diff:
        mock_diff.return_value = {"diff": ""}
        result = await tool.execute(ctx, {})
    assert "No changes" in result


@pytest.mark.asyncio
async def test_git_diff_with_path_returns_full_diff():
    ctx = _make_ctx()
    tool = REGISTRY.get("git_diff")
    with patch("app.core.workspaced_client.git_diff_workspace", new_callable=AsyncMock) as mock_diff:
        mock_diff.return_value = {"diff": "--- a/apps/web/x.tsx\n+++ b/apps/web/x.tsx\n@@ -1 +1 @@\n-a\n+b\n"}
        result = await tool.execute(ctx, {"path": "apps/web"})
    assert "x.tsx" in result
    mock_diff.assert_awaited_once_with(
        str(ctx.room_id), from_ref="main", path="apps/web", merge_base=True
    )


@pytest.mark.asyncio
async def test_git_diff_path_falls_back_to_glob():
    """Repo in a workspace subfolder: exact path misses, glob pathspec finds it."""
    ctx = _make_ctx()
    tool = REGISTRY.get("git_diff")

    async def fake_diff(room_id, from_ref=None, path=None, **kw):
        if path == "apps/web/src":
            return {"diff": ""}
        if path == "*apps/web/src*":
            return {"diff": "--- a/workspace/apps/web/src/page.tsx\n+++ b/workspace/apps/web/src/page.tsx\n"}
        return {"diff": ""}

    with patch("app.core.workspaced_client.git_diff_workspace", side_effect=fake_diff):
        result = await tool.execute(ctx, {"path": "apps/web/src"})

    assert "workspace/apps/web/src/page.tsx" in result
    assert "No exact match" in result


@pytest.mark.asyncio
async def test_git_diff_path_no_match_hints_full_listing():
    ctx = _make_ctx()
    tool = REGISTRY.get("git_diff")
    with patch("app.core.workspaced_client.git_diff_workspace", new_callable=AsyncMock) as mock_diff:
        mock_diff.return_value = {"diff": ""}
        result = await tool.execute(ctx, {"path": "apps/web/src"})
    assert "without `path`" in result
    assert "ALL changed files" in result


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


def test_bash_requires_container():
    assert REGISTRY.get("bash").requires_container is True


def test_read_file_not_requires_container():
    assert REGISTRY.get("read_file").requires_container is False


def test_edit_file_not_requires_container():
    assert REGISTRY.get("edit_file").requires_container is False


def test_grep_not_requires_container():
    assert REGISTRY.get("grep").requires_container is False


def test_write_file_is_mutating():
    assert REGISTRY.get("write_file").mutating is True


def test_edit_file_is_mutating():
    assert REGISTRY.get("edit_file").mutating is True


def test_grep_not_mutating():
    assert REGISTRY.get("grep").mutating is False


def test_git_diff_not_mutating():
    assert REGISTRY.get("git_diff").mutating is False
