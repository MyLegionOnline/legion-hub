"""Core side of the work gates (internal design notes stage 3, module 3).

The cloud implementation (hours, bought packs) is tested with the cloud plugin.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import room_storage, work_gates
from app.core.work_gates import HourAllowance, WorkRefusal
from app.plugins import get_registry, reset_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


class _Gate:
    def __init__(self, *, work=None, opens=None, extra=None):
        self._work, self._opens, self._extra = work, opens, extra
        self.calls = 0

    async def work_refusal(self, room_id, db):
        self.calls += 1
        return self._work

    async def open_refusal(self, room_id, db):
        return self._opens

    async def extra_hours(self, user_id, window_start, db):
        return self._extra


class _Boom:
    async def work_refusal(self, room_id, db):
        raise RuntimeError("db is down")


REFUSAL = WorkRefusal("out of hours", "HOURS_EXHAUSTED", "hours_exhausted")


@pytest.mark.asyncio
async def test_without_plugins_nothing_refuses():
    room = uuid4()
    assert await work_gates.work_refusal(room, MagicMock()) is None
    assert await work_gates.open_refusal(room, MagicMock()) is None
    assert await work_gates.extra_hours(uuid4(), datetime.now(timezone.utc), MagicMock()) is None


@pytest.mark.asyncio
async def test_the_first_gate_with_an_answer_wins():
    first, second, third = _Gate(), _Gate(work=REFUSAL), _Gate(work=WorkRefusal("x", "E", "r"))
    get_registry().work_gates.extend([first, second, third])
    assert await work_gates.work_refusal(uuid4(), MagicMock()) is REFUSAL
    assert (first.calls, second.calls, third.calls) == (1, 1, 0)


@pytest.mark.asyncio
async def test_a_gate_may_implement_only_some_questions():
    get_registry().work_gates.append(object())
    get_registry().work_gates.append(_Gate(opens=REFUSAL))
    assert await work_gates.open_refusal(uuid4(), MagicMock()) is REFUSAL
    assert await work_gates.work_refusal(uuid4(), MagicMock()) is None


@pytest.mark.asyncio
async def test_a_failing_gate_is_skipped_not_fatal():
    get_registry().work_gates.extend([_Boom(), _Gate(work=REFUSAL)])
    assert await work_gates.work_refusal(uuid4(), MagicMock()) is REFUSAL


def test_hour_allowance_extra_is_charged_plus_remaining():
    assert HourAllowance(charged=2.0, remaining=13.0).extra == 15.0
    assert HourAllowance().extra == 0.0


@pytest.mark.asyncio
async def test_refuse_room_work_posts_the_gates_notice(monkeypatch):
    from app.core import events

    get_registry().work_gates.append(_Gate(work=REFUSAL))
    storage = AsyncMock(return_value=None)
    monkeypatch.setattr(room_storage, "room_storage_refusal", storage)
    emit = AsyncMock()
    monkeypatch.setattr(events, "emit_event", emit)
    room = uuid4()
    assert await work_gates.refuse_room_work(room, MagicMock())
    assert emit.await_args.args[:3] == (room, "HOURS_EXHAUSTED", "out of hours")
    assert emit.await_args.kwargs["metadata"] == {"reason": "hours_exhausted"}
    storage.assert_not_awaited()


@pytest.mark.asyncio
async def test_refuse_room_work_without_plugins_still_enforces_the_files_quota(monkeypatch):
    from app.core import events

    monkeypatch.setattr(room_storage, "room_storage_refusal", AsyncMock(return_value="full"))
    emit = AsyncMock()
    monkeypatch.setattr(events, "emit_event", emit)
    room = uuid4()
    assert await work_gates.refuse_room_work(room, MagicMock())
    assert emit.await_args.args[:3] == (room, room_storage.STORAGE_FULL_EVENT, "full")

    monkeypatch.setattr(room_storage, "room_storage_refusal", AsyncMock(return_value=None))
    emit.reset_mock()
    assert not await work_gates.refuse_room_work(room, MagicMock())
    emit.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconciler_steps_run_in_order_and_a_failure_does_not_stop_the_rest():
    ran = []

    async def first(db):
        ran.append("first")

    async def broken(db):
        ran.append("broken")
        raise RuntimeError("x")

    async def last(db):
        ran.append("last")

    get_registry().reconciler_steps.extend([first, broken, last])
    db = MagicMock()
    db.rollback = AsyncMock()
    await work_gates.run_reconciler_steps(db)
    assert ran == ["first", "broken", "last"]
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_hours_limit_without_a_plugin_is_the_plain_config_limit(monkeypatch):
    import app.core.workspace_quota as wq

    monkeypatch.setattr(wq, "monthly_hours_consumed", AsyncMock(return_value=40.0))
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(wq, "get_hours_window", AsyncMock(return_value=(now, now)))
    assert await wq.hours_limit_exhausted(uuid4(), MagicMock(), 40.0)
    assert not await wq.hours_limit_exhausted(uuid4(), MagicMock(), 41.0)

    get_registry().work_gates.append(_Gate(extra=HourAllowance(remaining=5.0)))
    assert not await wq.hours_limit_exhausted(uuid4(), MagicMock(), 40.0)


# ---------- room lifecycle questions (module 4) ----------------------------------------


class _Lifecycle:
    def __init__(self, *, read_only=None, restore=None, start=None, files=None, payload=None):
        self._a = (read_only, restore, start, files, payload)

    async def read_only(self, room_id, db):
        return self._a[0]

    async def restore_on_open(self, room_id, db):
        return self._a[1]

    async def start_refusal(self, room_id, db):
        return self._a[2]

    async def files_refusal(self, room_id, db):
        return self._a[3]

    async def lifecycle_payload(self, user_id, db):
        return self._a[4]


@pytest.mark.asyncio
async def test_without_plugins_no_room_is_read_only_archived_or_frozen():
    room = uuid4()
    assert await work_gates.room_read_only(room, MagicMock()) is None
    assert await work_gates.restore_on_open(room, MagicMock()) is None
    assert await work_gates.start_refusal(room, MagicMock()) is None
    assert await work_gates.lifecycle_payload(uuid4(), MagicMock()) is None
    await work_gates.require_room_writable(room, MagicMock())
    await work_gates.require_room_files(room, MagicMock())


@pytest.mark.asyncio
async def test_lifecycle_questions_are_answered_by_the_gates():
    reason = {"reason": "plan_expired", "message": "read only"}
    refusal = work_gates.StartRefusal("room_archived", "wait", status_code=409)
    get_registry().work_gates.append(
        _Lifecycle(read_only=reason, restore=reason, start=refusal, payload={"phase": "active"})
    )
    room = uuid4()
    assert await work_gates.room_read_only(room, MagicMock()) == reason
    assert await work_gates.restore_on_open(room, MagicMock()) == reason
    assert await work_gates.start_refusal(room, MagicMock()) == refusal
    assert await work_gates.lifecycle_payload(uuid4(), MagicMock()) == {"phase": "active"}


@pytest.mark.asyncio
async def test_file_api_gates_answer_409_with_the_gates_reason():
    from fastapi import HTTPException

    reason = {"reason": "room_archived", "message": "in the archive"}
    get_registry().work_gates.append(_Lifecycle(read_only=reason, files=reason))
    for gate in (work_gates.require_room_writable, work_gates.require_room_files):
        with pytest.raises(HTTPException) as err:
            await gate(uuid4(), MagicMock())
        assert err.value.status_code == 409
        assert err.value.detail == {"error_code": "room_archived", "message": "in the archive"}
