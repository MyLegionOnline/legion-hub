"""A stop fence tripped mid-turn must preserve the in-flight tool-loop
transcript instead of discarding it (the reason Resume used to restart the
agent from almost nothing after a workspace/user pause)."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.proxy_session import (
    RunStopped,
    _persist_stopped_progress,
    _seal_dangling_tool_calls,
)


def _session() -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.context_snapshot = None
    s.message_count = 0
    s.turn_count = 0
    s.tokens_in = 0
    s.tokens_out = 0
    return s


def test_seal_dangling_anthropic_tool_use_gets_error_result():
    working = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}},
            {"type": "tool_use", "id": "toolu_2", "name": "read_file", "input": {}},
        ]},
    ]
    sealed = _seal_dangling_tool_calls(working)
    assert sealed[-1]["role"] == "user"
    results = sealed[-1]["content"]
    assert {r["tool_use_id"] for r in results} == {"toolu_1", "toolu_2"}
    assert all(r["is_error"] for r in results)


def test_seal_dangling_openai_tool_calls_get_tool_messages():
    working = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "bash"}},
        ]},
    ]
    sealed = _seal_dangling_tool_calls(working)
    assert sealed[-1]["role"] == "tool"
    assert sealed[-1]["tool_call_id"] == "call_1"


def test_seal_noop_when_last_turn_already_resolved():
    working = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "b", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
    ]
    assert _seal_dangling_tool_calls(working) is working


@pytest.mark.asyncio
async def test_persist_stopped_progress_saves_working_history():
    session = _session()
    db = AsyncMock()
    db.add = MagicMock()

    exc = RunStopped(uuid.uuid4(), "run-1")
    exc.working = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "read three files, edited one"},
    ]
    exc.tokens_in, exc.tokens_out, exc.model = 1000, 200, "claude-x"

    saved = await _persist_stopped_progress(session, exc, db)

    assert saved is True
    assert session.context_snapshot["messages"] == exc.working
    assert session.message_count == 2
    db.flush.assert_awaited()


@pytest.mark.asyncio
async def test_persist_stopped_progress_noop_without_working():
    session = _session()
    db = AsyncMock()
    db.add = MagicMock()

    saved = await _persist_stopped_progress(session, RunStopped(None, None), db)

    assert saved is False
    assert session.context_snapshot is None
