"""Tool-call arguments a model gets slightly wrong must not cost it a turn.

Prod, 2026-11-14: an agent on DeepSeek v4.1 flash named bash's argument
``command`` on every other call and got "cmd is required" back; malformed
JSON was silently run as ``{}`` with the same misleading error.
"""
from __future__ import annotations

import pytest

from app.core.session_close import (
    ToolArgumentsParseError,
    malformed_arguments_result,
    parse_tool_arguments,
)
from app.core.tools.registry import canonicalize_tool_input


def test_bash_command_alias_becomes_cmd() -> None:
    args = canonicalize_tool_input("bash", {"command": "ls -la", "timeout_sec": 30})
    assert args == {"cmd": "ls -la", "timeout_sec": 30}


def test_bash_cmd_wins_over_command() -> None:
    args = canonicalize_tool_input("bash", {"cmd": "pwd", "command": "ls"})
    assert args["cmd"] == "pwd"


def test_command_alias_only_for_bash() -> None:
    args = canonicalize_tool_input("workspace_service", {"command": "npm run dev"})
    assert args == {"command": "npm run dev"}


def test_malformed_arguments_result_names_the_real_cause() -> None:
    raw = '{"cmd": "find . -name x -exec rm {} \\;"}'
    with pytest.raises(ToolArgumentsParseError) as info:
        parse_tool_arguments(raw)
    text = malformed_arguments_result("bash", info.value)
    assert text.startswith("Error: the arguments of this bash call are not valid JSON")
    assert "not executed" in text
    assert "cmd is required" not in text
