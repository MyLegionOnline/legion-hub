"""Size budget for the agent's tool schemas (internal design notes S1).

Every LLM step of an agent session carries these schemas, so they must not
regrow silently.  A budget is the Anthropic form serialized as JSON, in
characters, with ~10% headroom over the trimmed size.  When a tool genuinely
needs more words, raise its budget here in the same change — on purpose.
"""
from __future__ import annotations

import json

import pytest

from app.core.proxy_session import close_tools_for_role
from app.core.tools import REGISTRY

# name -> max chars of json.dumps(tool.ant_schema("agent"), ensure_ascii=False)
AGENT_TOOL_BUDGET: dict[str, int] = {
    "workspace_service": 3100,
    "bash": 1950,
    "screenshot": 1550,
    "ask_user": 1150,
    "release_tag": 950,
    "plan_dialogue": 860,
    "edit_file": 800,
    "task_update": 800,
    "publish_for_deploy": 770,
    "grep": 770,
    "run_subtask": 750,
    "room_config": 660,
    "read_file": 630,
    "bash_output": 620,
    "find_file": 620,
    "merge_draft": 620,
    "preview_fetch": 620,
    "observation_read": 620,
    "reply_to_user": 610,
    "todo_write": 610,
    "glob": 570,
    "write_file": 560,
    "bash_wait": 540,
    "get_outline": 540,
    "read_files": 510,
    "plan_read": 430,
    "repo_map": 430,
    "git_diff": 420,
    "bash_kill": 410,
    "task_list": 290,
    "todo_read": 200,
}
# Any agent tool not listed above (e.g. web tools behind a deployment flag).
DEFAULT_TOOL_BUDGET = 1000
# Sum over the tools listed above; the per-tool headroom must not add up.
AGENT_TOTAL_BUDGET = 22_500
# close_session as offered to the proxy agent side (only its allowed fields).
AGENT_CLOSE_BUDGET = 1400


def _size(schema: dict) -> int:
    return len(json.dumps(schema, ensure_ascii=False))


def test_every_agent_tool_fits_its_budget():
    over = {}
    for tool in REGISTRY.tools_for("agent"):
        size = _size(tool.ant_schema("agent"))
        budget = AGENT_TOOL_BUDGET.get(tool.name, DEFAULT_TOOL_BUDGET)
        if size > budget:
            over[tool.name] = (size, budget)
    assert not over, f"tool schemas over budget (size, budget): {over}"


def test_agent_tools_fit_the_total_budget():
    total = sum(
        _size(tool.ant_schema("agent"))
        for tool in REGISTRY.tools_for("agent")
        if tool.name in AGENT_TOOL_BUDGET
    )
    assert total <= AGENT_TOTAL_BUDGET, f"agent tool schemas total {total} > {AGENT_TOTAL_BUDGET}"


@pytest.mark.parametrize("role", ["agent", "agent_subtask"])
def test_agent_close_session_offers_only_its_fields(role):
    ant, oai = close_tools_for_role(role)
    assert ant["name"] == "close_session"
    assert oai["function"]["name"] == "close_session"
    assert set(ant["input_schema"]["properties"]) == {
        "agent_memory_patch",
        "tasks_update",
        "file_writes",
        "summary",
    }
    assert _size(ant) <= AGENT_CLOSE_BUDGET


def test_chat_master_close_session_keeps_its_verdict():
    ant, _ = close_tools_for_role("chat_master")
    assert "verdict" in ant["input_schema"]["properties"]
    assert "file_writes" not in ant["input_schema"]["properties"]


def test_unknown_role_gets_the_full_close_schema():
    from app.core.session_close import ANTHROPIC_CLOSE_TOOL, OPENAI_CLOSE_TOOL

    assert close_tools_for_role("unknown") == (ANTHROPIC_CLOSE_TOOL, OPENAI_CLOSE_TOOL)
