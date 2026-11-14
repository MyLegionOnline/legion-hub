"""Texts an agent or a person reads name the room's working branch, not `dev`.

internal design notes Stage 2. A room may develop on
`core` or `transport`; hints, gate messages and ChatMaster's divergence prompt
must not send anyone to a `dev` that does not exist there.
"""
from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import deploy_publish, divergence_review, room_git_flow
from app.core.tools._write_file import main_protected_error
from app.models.task import DispatchStatus, TaskState

_DEV = re.compile(r"\bdev\b")


class _DB:
    def __init__(self, room=None, fail=False):
        self._room = room
        self._fail = fail

    def add(self, _obj):
        pass

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def get(self, _model, _pk):
        return SimpleNamespace(nickname="Frontender")

    async def execute(self, _stmt):
        if self._fail:
            raise RuntimeError("db down")
        room = self._room
        return SimpleNamespace(scalar_one_or_none=lambda: room)


def test_main_protected_hint_names_the_working_branch():
    text = main_protected_error("a.txt", None, "core")
    assert "`core`" in text
    assert not _DEV.search(text)


def test_main_protected_hint_for_a_task_names_its_branch():
    task_id = uuid4()
    text = main_protected_error("a.txt", task_id)
    assert f"task/{task_id}" in text


@pytest.mark.asyncio
async def test_room_working_branch_reads_the_room():
    room = SimpleNamespace(git_config={"working_branch": "transport"})
    assert await room_git_flow.room_working_branch(uuid4(), _DB(room)) == "transport"


@pytest.mark.asyncio
async def test_room_working_branch_never_raises():
    assert await room_git_flow.room_working_branch(uuid4(), _DB(fail=True)) == "dev"


@pytest.mark.asyncio
async def test_auto_merge_event_names_the_merge_target(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "run_task_checks",
        AsyncMock(return_value={"passed": True, "ran": True, "results": []}),
    )
    monkeypatch.setattr(room_git_flow, "classify_task_risk", AsyncMock(return_value="trivial"))
    monkeypatch.setattr(
        room_git_flow, "merge_task_branch",
        AsyncMock(return_value={"ok": True, "sha": "beef", "target": "core", "branch": "task/x"}),
    )
    monkeypatch.setattr(room_git_flow, "maybe_open_promotion", AsyncMock())
    events = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", events)
    task = SimpleNamespace(
        id=uuid4(), title="fix the padding", state=TaskState.IN_PROGRESS, branch_name="task/x",
        ci_status=None, risk=None, merge_commit=None, origin="direct_session",
        dispatch_status=DispatchStatus.DISPATCHED, updated_at=None, goal_id=None,
    )
    room = SimpleNamespace(id=uuid4(), git_config={"working_branch": "core"})

    await room_git_flow.finalize_task_gate1(room, task, _DB(room))

    merged = [c.kwargs for c in events.await_args_list if c.kwargs["event_type"] == "TASK_AUTO_MERGED"]
    assert merged and "into core" in merged[0]["content"]


def test_divergence_prompts_do_not_assume_dev():
    assert not _DEV.search(divergence_review._SYSTEM_PROMPT)
    for prop in divergence_review._ANALYSIS_SCHEMA["properties"].values():
        assert not _DEV.search(prop.get("description", ""))


def test_divergence_fallbacks_name_the_real_base():
    degraded = divergence_review._degraded_analysis("timeout", "task/1", "core")
    fallback = divergence_review._fallback_analysis(
        {"base": "core", "branch": "task/1", "branch_commits": [], "overlap_files": []}
    )
    for card in (degraded, fallback):
        assert "core" in card["rework_consequence"]
        assert not _DEV.search(card["rework_consequence"])
    assert not _DEV.search(degraded["rework_brief"])


@pytest.mark.asyncio
async def test_main_bypass_texts_do_not_name_dev(monkeypatch):
    seen = {}

    async def _ask(_db, **kwargs):
        seen.update(kwargs)
        return False

    monkeypatch.setattr(deploy_publish, "_ask_person", _ask)
    await deploy_publish.confirm_main_bypass_publish(
        None, room_id=uuid4(), session_id=uuid4(),
        commits=[{"sha": "a" * 40, "short_sha": "aaaaaaaa", "message": "x", "author": "a", "timestamp": ""}],
    )
    assert not _DEV.search(" ".join(seen["lines"]))
    assert not _DEV.search(seen["deny"])
