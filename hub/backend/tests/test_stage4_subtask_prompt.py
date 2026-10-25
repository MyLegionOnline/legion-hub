"""Regression tests for Stage 4: todo/run_subtask prompt integration."""
from __future__ import annotations

from app.core.memory_layer import _KIND_HEADERS


def test_agent_prompt_explains_when_to_use_subtasks() -> None:
    prompt = _KIND_HEADERS["cm_agent_proxy_agent"]

    assert "TODO AND SUBTASKS" in prompt
    assert "todo_write" in prompt
    assert "todo_read" in prompt
    assert "run_subtask" in prompt
    assert "3+ files/components" in prompt
    assert "separate" in prompt
    assert "visual or runtime verification" in prompt


def test_agent_prompt_discourages_over_splitting() -> None:
    prompt = _KIND_HEADERS["cm_agent_proxy_agent"]

    assert "short, self-contained task" in prompt
    assert "fixed" in prompt
    assert "many trivial subtasks" in prompt
    assert "local failure" in prompt
