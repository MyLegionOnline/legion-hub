"""internal design notes Stage 6 — chat_master really reaches the
shared Gate 1, not a private copy of it.

Stage 3 pulled Gate 1's sequence (checks -> risk -> merge/review/blocked) out
of ``chat_master.approve_and_start`` into ``room_git_flow.finalize_task_gate1``
so the dialogue path (``session_work.py``) could reuse the exact same merge
policy. ``test_session_work_finalize.py`` already exercises
``finalize_task_gate1`` itself in isolation from the dialogue side; this file
is the other half of that regression: it drives
``chat_master.approve_and_start`` end to end (with only the LLM call, S3 log
flush and project/agent memory updates mocked out) and checks that the real
shared function is what fires — the same ``TASK_AUTO_MERGED`` /
``TASK_STATE_CHANGED`` events chat_master used to emit inline before the
refactor.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import chat_master, room_git_flow
from app.models.agent import Agent
from app.models.room import Room, RoomState
from app.models.task import Task, TaskState


class _FakeResult:
    def __init__(self, items):
        self._items = list(items)

    def scalars(self):
        return self

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None


class _FakeDB:
    """No real Postgres involved — every collaborator below is mocked, so the
    only thing this needs to do is remember what got ``add``-ed and answer
    ``monitor_completion``'s ``select(Task)`` with it."""

    def __init__(self):
        self.added: list[object] = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def refresh(self, _obj):
        pass

    async def execute(self, _stmt):
        return _FakeResult([o for o in self.added if isinstance(o, Task)])


class _FakeSessionLogger:
    def __init__(self, *_args, **_kwargs):
        pass

    async def log_error(self, *_args, **_kwargs):
        pass

    async def flush_to_s3(self, _archive):
        return "logs/fake.ndjson", 42


def _room():
    return Room(id=uuid4(), name="test", creator_id=uuid4(), state=RoomState.EXECUTION, git_config={})


def _agent(nickname="Frontender"):
    return Agent(
        id=uuid4(), owner_id=uuid4(), token=str(uuid4()), nickname=nickname,
        agent_uid=str(uuid4()), llm_config={"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
    )


def _plan(nickname="Frontender"):
    return {"tasks": [{"title": "fix the padding", "description": "fix the padding", "agent_nickname": nickname}]}


def _patch_collaborators(monkeypatch, *, ci_passed=True, risk="trivial", merge_result=None):
    """Mock everything approve_and_start touches other than the Gate 1
    sequence itself: the agent turn, log storage/S3, and memory updates."""
    monkeypatch.setattr(chat_master, "build_agent_context", AsyncMock(return_value={}))
    monkeypatch.setattr(
        chat_master, "run_agent_loop", AsyncMock(return_value=("Done, fixed the padding.", 100, 50))
    )
    monkeypatch.setattr(chat_master, "update_project_memory", AsyncMock())
    monkeypatch.setattr(chat_master, "update_agent_memory", AsyncMock())

    import app.core.log_storage as log_storage
    monkeypatch.setattr(log_storage, "get_log_buffer", lambda: None)
    monkeypatch.setattr(log_storage, "get_log_archive", lambda: None)

    import app.core.session_logger as session_logger_module
    monkeypatch.setattr(session_logger_module, "SessionLogger", _FakeSessionLogger)

    monkeypatch.setattr(room_git_flow, "ensure_task_branch", AsyncMock())

    # The Gate 1 sequence itself: real finalize_task_gate1, fake checks/merge —
    # exactly what test_session_work_finalize.py does for the dialogue side.
    monkeypatch.setattr(
        room_git_flow, "run_task_checks",
        AsyncMock(return_value={"passed": ci_passed, "ran": True, "results": []}),
    )
    monkeypatch.setattr(room_git_flow, "classify_task_risk", AsyncMock(return_value=risk))
    monkeypatch.setattr(
        room_git_flow, "merge_task_branch",
        AsyncMock(return_value=merge_result or {
            "ok": True, "sha": "deadbeef", "target": "dev", "branch": "task/x",
        }),
    )
    monkeypatch.setattr(room_git_flow, "maybe_open_promotion", AsyncMock())

    events = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", events)
    return events


def _event_types(events) -> list[str]:
    return [c.kwargs["event_type"] for c in events.await_args_list]


@pytest.mark.asyncio
async def test_trivial_task_is_auto_merged_through_the_shared_gate(monkeypatch):
    events = _patch_collaborators(monkeypatch, ci_passed=True, risk="trivial")
    room, agent, db = _room(), _agent(), _FakeDB()

    await chat_master.approve_and_start(room, _plan(), [agent], {}, db)

    types = _event_types(events)
    assert "TASK_AUTO_MERGED" in types
    # the reply is still broadcast as a chat MESSAGE after the gate resolves —
    # that ordering is chat_master's own, finalize_task_gate1 does not decide it
    assert types.index("TASK_AUTO_MERGED") < types.index("MESSAGE")

    task = next(o for o in db.added if isinstance(o, Task))
    assert task.state == TaskState.COMPLETED
    assert task.merge_commit == "deadbeef"


@pytest.mark.asyncio
async def test_red_checks_block_the_task_through_the_shared_gate(monkeypatch):
    events = _patch_collaborators(monkeypatch, ci_passed=False)
    room, agent, db = _room(), _agent(), _FakeDB()

    await chat_master.approve_and_start(room, _plan(), [agent], {}, db)

    types = _event_types(events)
    assert "TASK_STATE_CHANGED" in types
    assert "TASK_AUTO_MERGED" not in types
    # a blocked task still gets its reply broadcast — only the merge is withheld
    assert "MESSAGE" in types

    task = next(o for o in db.added if isinstance(o, Task))
    assert task.state == TaskState.BLOCKED
    assert task.merge_commit is None


@pytest.mark.asyncio
async def test_diverged_base_holds_task_in_review_not_auto_merged(monkeypatch):
    """A trivial task whose base moved under it (the pre-merge refresh conflicts)
    must not be reported auto-merged — it waits for a human."""
    events = _patch_collaborators(
        monkeypatch, ci_passed=True, risk="trivial",
        merge_result={"ok": False, "conflict": True, "target": "dev", "branch": "task/x"},
    )
    room, agent, db = _room(), _agent(), _FakeDB()

    await chat_master.approve_and_start(room, _plan(), [agent], {}, db)

    types = _event_types(events)
    assert "TASK_AUTO_MERGED" not in types
    assert "TASK_WAITING_REVIEW" in types

    task = next(o for o in db.added if isinstance(o, Task))
    assert task.state == TaskState.WAITING_REVIEW
    assert task.merge_commit is None
