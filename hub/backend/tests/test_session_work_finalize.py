"""Stage 3 of internal design notes — Gate 1 at the end of a turn.

The dialogue's work is recorded as a task and then routed exactly like a
dispatched one: quick checks, risk class, and either an auto-merge into the
working branch or a review card. ``finalize_task_gate1`` is the shared
implementation (``chat_master`` runs the same code), and
``finalize_session_work`` is the dialogue's entry into it.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow, session_work
from app.models.task import DispatchStatus, TaskState


class _FakeDB:
    def __init__(self, scalar_result=None):
        self.added: list[object] = []
        self._scalar_result = scalar_result

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def get(self, _model, _pk):
        return SimpleNamespace(nickname="Frontender")

    async def execute(self, _stmt):
        result = self._scalar_result
        return SimpleNamespace(scalar_one_or_none=lambda: result)


def _room(**git_config):
    return SimpleNamespace(id=uuid4(), git_config=git_config)


def _task(**kwargs):
    base = {
        "id": uuid4(),
        "title": "fix the padding",
        "state": TaskState.IN_PROGRESS,
        "branch_name": "task/x",
        "ci_status": None,
        "risk": None,
        "merge_commit": None,
        "origin": "direct_session",
        "dispatch_status": DispatchStatus.DISPATCHED,
        "updated_at": None,
        "goal_id": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _patch_gate(monkeypatch, *, passed=True, risk="trivial", merge=None):
    checks = AsyncMock(return_value={"passed": passed, "ran": True, "results": []})
    monkeypatch.setattr(room_git_flow, "run_task_checks", checks)
    monkeypatch.setattr(room_git_flow, "classify_task_risk", AsyncMock(return_value=risk))
    merge = merge or AsyncMock(return_value={"ok": True, "sha": "deadbeef", "target": "dev", "branch": "task/x"})
    monkeypatch.setattr(room_git_flow, "merge_task_branch", merge)
    monkeypatch.setattr(room_git_flow, "maybe_open_promotion", AsyncMock())
    events = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", events)
    return checks, merge, events


def _event_types(events) -> list[str]:
    return [c.kwargs["event_type"] for c in events.await_args_list]


# ── finalize_task_gate1 ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_green_trivial_is_auto_merged(monkeypatch):
    _, merge, events = _patch_gate(monkeypatch, risk="trivial")
    task = _task()

    result = await room_git_flow.finalize_task_gate1(_room(), task, _FakeDB())

    assert result["merged"] is True
    assert result["merge_commit"] == "deadbeef"
    assert task.state == TaskState.COMPLETED
    merge.assert_awaited_once()
    assert "TASK_AUTO_MERGED" in _event_types(events)


@pytest.mark.asyncio
async def test_green_normal_waits_for_a_human(monkeypatch):
    _, merge, events = _patch_gate(monkeypatch, risk="normal")
    task = _task()

    result = await room_git_flow.finalize_task_gate1(_room(), task, _FakeDB())

    assert result["merged"] is False
    assert task.state == TaskState.WAITING_REVIEW
    merge.assert_not_awaited()          # `dev` untouched until the human clicks
    assert "TASK_WAITING_REVIEW" in _event_types(events)


@pytest.mark.asyncio
async def test_red_checks_block_the_task(monkeypatch):
    _, merge, events = _patch_gate(monkeypatch, passed=False)
    task = _task()

    result = await room_git_flow.finalize_task_gate1(_room(), task, _FakeDB())

    assert result["ci_passed"] is False
    assert task.state == TaskState.BLOCKED
    merge.assert_not_awaited()
    assert "TASK_STATE_CHANGED" in _event_types(events)
    assert "TASK_AUTO_MERGED" not in _event_types(events)


@pytest.mark.asyncio
async def test_risky_is_never_auto_merged_even_with_normal_too(monkeypatch):
    _, merge, _ = _patch_gate(monkeypatch, risk="risky")

    result = await room_git_flow.finalize_task_gate1(
        _room(), _task(), _FakeDB(), auto_merge_normal=True
    )

    assert result["merged"] is False
    merge.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_too_folds_a_normal_diff_in(monkeypatch):
    """The dialogue path's one difference: the user was in the conversation and
    already saw the work."""
    _, merge, _ = _patch_gate(monkeypatch, risk="normal")

    result = await room_git_flow.finalize_task_gate1(
        _room(), _task(), _FakeDB(), auto_merge_normal=True
    )

    assert result["merged"] is True
    merge.assert_awaited_once()


@pytest.mark.asyncio
async def test_room_can_disable_auto_merge(monkeypatch):
    _, merge, _ = _patch_gate(monkeypatch, risk="trivial")

    result = await room_git_flow.finalize_task_gate1(
        _room(auto_merge_trivial=False), _task(), _FakeDB()
    )

    assert result["merged"] is False
    merge.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_outage_is_treated_as_green(monkeypatch):
    """Fail-open, as before: a transient workspace problem must not wedge a
    task in BLOCKED."""
    _patch_gate(monkeypatch)
    monkeypatch.setattr(room_git_flow, "run_task_checks", AsyncMock(side_effect=RuntimeError("down")))

    result = await room_git_flow.finalize_task_gate1(_room(), _task(), _FakeDB())

    assert result["ci_passed"] is True


@pytest.mark.asyncio
async def test_failed_risk_classification_holds_for_review(monkeypatch):
    _, merge, _ = _patch_gate(monkeypatch)
    monkeypatch.setattr(
        room_git_flow, "classify_task_risk", AsyncMock(side_effect=RuntimeError("diff failed"))
    )

    result = await room_git_flow.finalize_task_gate1(_room(), _task(), _FakeDB())

    assert result["risk"] == "normal"
    merge.assert_not_awaited()


@pytest.mark.asyncio
async def test_skip_notice_can_be_silenced(monkeypatch):
    checks, _, _ = _patch_gate(monkeypatch)

    await room_git_flow.finalize_task_gate1(_room(), _task(), _FakeDB(), emit_skipped=False)

    assert checks.await_args.kwargs["emit_skipped"] is False


# ── finalize_session_work ───────────────────────────────────────────────────

def _session(**kwargs):
    base = {
        "id": uuid4(),
        "room_id": uuid4(),
        "kind": "user_agent_p2",
        "agent_id": uuid4(),
        "goal_id": None,
        "context_snapshot": {session_work.SNAPSHOT_WORK_TOUCHED: True},
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def _patch_finalize(monkeypatch, *, room=None, task=None):
    monkeypatch.setattr(
        room_git_flow, "_load_room", AsyncMock(return_value=room if room is not None else _room())
    )
    adopt = AsyncMock(return_value=task)
    monkeypatch.setattr(session_work, "adopt_session_branch_as_task", adopt)
    gate = AsyncMock(return_value={"merged": True, "state": TaskState.COMPLETED})
    monkeypatch.setattr(room_git_flow, "finalize_task_gate1", gate)
    return adopt, gate


@pytest.mark.asyncio
async def test_reading_turn_finalizes_nothing(monkeypatch):
    """No writing tool ran, so there is nothing to record — and no check run,
    which is what keeps an ordinary question fast."""
    adopt, gate = _patch_finalize(monkeypatch, task=_task())
    session = _session(context_snapshot={})

    assert await session_work.finalize_session_work(session, _FakeDB(), reason="turn_end") is None
    adopt.assert_not_awaited()
    gate.assert_not_awaited()


@pytest.mark.asyncio
async def test_writing_turn_runs_the_gate(monkeypatch):
    task = _task()
    adopt, gate = _patch_finalize(monkeypatch, task=task)
    session = _session()

    result = await session_work.finalize_session_work(
        session, _FakeDB(), reason="turn_end",
        user_request="fix the padding", agent_reply="done",
    )

    assert result["merged"] is True
    gate.assert_awaited_once()
    assert gate.await_args.kwargs["emit_skipped"] is False
    assert gate.await_args.kwargs["auto_merge_normal"] is False
    # the claim taken while the dialogue owned the tree is released
    assert task.dispatch_status == DispatchStatus.READY
    # and the burst is closed, so the next reading turn does not re-run it
    assert session_work.SNAPSHOT_WORK_TOUCHED not in session.context_snapshot


@pytest.mark.asyncio
async def test_nothing_committed_closes_the_burst_quietly(monkeypatch):
    _, gate = _patch_finalize(monkeypatch, task=None)
    session = _session()

    assert await session_work.finalize_session_work(session, _FakeDB(), reason="turn_end") is None
    gate.assert_not_awaited()
    assert session_work.SNAPSHOT_WORK_TOUCHED not in session.context_snapshot


@pytest.mark.asyncio
async def test_normal_too_is_passed_through(monkeypatch):
    _, gate = _patch_finalize(
        monkeypatch, room=_room(direct_session_auto_merge="normal_too"), task=_task()
    )

    await session_work.finalize_session_work(_session(), _FakeDB(), reason="turn_end")

    assert gate.await_args.kwargs["auto_merge_normal"] is True


@pytest.mark.asyncio
async def test_opted_out_room_runs_no_gate(monkeypatch):
    adopt, gate = _patch_finalize(
        monkeypatch, room=_room(direct_session_mode="off"), task=_task()
    )
    session = _session()

    assert await session_work.finalize_session_work(session, _FakeDB(), reason="turn_end") is None
    adopt.assert_not_awaited()
    gate.assert_not_awaited()
    assert session_work.SNAPSHOT_WORK_TOUCHED not in session.context_snapshot


@pytest.mark.asyncio
async def test_branch_only_merges_without_checks(monkeypatch):
    """A room whose suite is too slow to run after every writing turn still
    gets the branch and the task — only the verdict is skipped."""
    task = _task()
    _, gate = _patch_finalize(
        monkeypatch, room=_room(direct_session_mode="branch_only"), task=task
    )
    merge = AsyncMock(return_value={"ok": True, "sha": "cafe", "target": "dev", "branch": "task/x"})
    monkeypatch.setattr(room_git_flow, "merge_task_branch", merge)
    monkeypatch.setattr(room_git_flow, "maybe_open_promotion", AsyncMock())
    events = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", events)

    result = await session_work.finalize_session_work(_session(), _FakeDB(), reason="turn_end")

    gate.assert_not_awaited()
    assert result["merged"] is True
    assert task.state == TaskState.COMPLETED
    assert task.merge_commit == "cafe"
    assert "TASK_MERGED" in _event_types(events)


@pytest.mark.asyncio
async def test_branch_only_conflict_holds_for_review(monkeypatch):
    task = _task()
    _patch_finalize(monkeypatch, room=_room(direct_session_mode="branch_only"), task=task)
    monkeypatch.setattr(
        room_git_flow, "merge_task_branch",
        AsyncMock(return_value={"ok": False, "conflict": True}),
    )
    monkeypatch.setattr("app.core.events.emit_event", AsyncMock())

    result = await session_work.finalize_session_work(_session(), _FakeDB(), reason="turn_end")

    assert result["merged"] is False
    assert task.state == TaskState.WAITING_REVIEW


@pytest.mark.asyncio
async def test_claim_is_released_even_when_the_gate_explodes(monkeypatch):
    task = _task()
    _, gate = _patch_finalize(monkeypatch, task=task)
    gate.side_effect = RuntimeError("gate blew up")
    session = _session()

    with pytest.raises(RuntimeError):
        await session_work.finalize_session_work(session, _FakeDB(), reason="turn_end")

    assert task.dispatch_status == DispatchStatus.READY
    assert session_work.SNAPSHOT_WORK_TOUCHED not in session.context_snapshot


@pytest.mark.asyncio
async def test_chatmaster_session_is_not_finalized(monkeypatch):
    adopt, gate = _patch_finalize(monkeypatch, task=_task())

    assert await session_work.finalize_session_work(
        _session(kind="user_cm_p2"), _FakeDB(), reason="turn_end"
    ) is None
    adopt.assert_not_awaited()
    gate.assert_not_awaited()
