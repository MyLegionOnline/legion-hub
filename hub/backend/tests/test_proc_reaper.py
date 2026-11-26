"""Processes of finished agents are stopped (internal design notes, E2)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import proc_reaper
from app.core.proc_reaper import NO_RUN, finished_owners, owner_tag_for, parse_owner


def test_scratch_dir_is_per_task_not_per_run():
    """A resumed task keeps its $TMPDIR; workspaced removes it with the run-less tag."""
    task_id, sid = uuid4(), uuid4()
    first = SimpleNamespace(task_id=task_id, run_id="r1", session_id=sid)
    second = SimpleNamespace(task_id=task_id, run_id="r2", session_id=uuid4())
    assert proc_reaper.tmp_dir_for(first) == proc_reaper.tmp_dir_for(second)
    assert proc_reaper.tmp_dir_for(first) == f"/tmp/legion/task-{task_id}"
    chat = SimpleNamespace(task_id=None, session_id=sid)
    assert proc_reaper.tmp_dir_for(chat) == f"/tmp/legion/session-{sid}"


def test_owner_tag_for_task_and_direct_chat():
    task_id, sid = uuid4(), uuid4()
    ctx = SimpleNamespace(task_id=task_id, run_id="r1", session_id=sid)
    assert owner_tag_for(ctx) == f"task:{task_id}:r1"
    ctx.run_id = None
    assert owner_tag_for(ctx) == f"task:{task_id}:{NO_RUN}"
    assert owner_tag_for(SimpleNamespace(task_id=None, session_id=sid)) == f"session:{sid}"


def test_parse_owner_rejects_foreign_tags():
    t = uuid4()
    assert parse_owner(f"task:{t}:abc") == ("task", t, "abc")
    assert parse_owner(f"session:{t}") == ("session", t, None)
    for bad in ("", "task:nope:1", f"room:{t}", f"task:{t}", "x"):
        assert parse_owner(bad) is None


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalars(self):
        return SimpleNamespace(all=lambda: self._rows)


def _db(task_rows, open_session_ids):
    db = MagicMock()
    results = []
    if task_rows is not None:
        results.append(_Rows(task_rows))
    if open_session_ids is not None:
        results.append(_Rows(open_session_ids))
    db.execute = AsyncMock(side_effect=results)
    return db


@pytest.mark.asyncio
async def test_finished_owners_by_run_runtime_and_session():
    live, retried, paused, gone, adhoc_live, adhoc_done = (uuid4() for _ in range(6))
    open_sid, closed_sid = uuid4(), uuid4()
    tags = [
        f"task:{live}:run-1",          # current run, running -> alive
        f"task:{retried}:old-run",     # Retry moved run_id on -> dead
        f"task:{paused}:run-3",        # current run but paused -> dead
        f"task:{gone}:run-4",          # task deleted -> dead
        f"task:{adhoc_live}:{NO_RUN}",  # no run, task in review -> kept
        f"task:{adhoc_done}:{NO_RUN}",  # no run, task completed -> dead
        f"session:{open_sid}",         # open direct chat -> alive
        f"session:{closed_sid}",       # closed direct chat -> dead
        "something:else",              # foreign -> never
    ]
    task_rows = [
        (live, "run-1", "running", "in_progress"),
        (retried, "new-run", "running", "in_progress"),
        (paused, "run-3", "paused", "paused"),
        (adhoc_live, None, "stopped", "waiting_review"),
        (adhoc_done, None, "stopped", "completed"),
    ]
    dead = await finished_owners(tags, _db(task_rows, [open_sid]))
    assert dead == sorted([
        f"task:{retried}:old-run",
        f"task:{paused}:run-3",
        f"task:{gone}:run-4",
        f"task:{adhoc_done}:{NO_RUN}",
        f"session:{closed_sid}",
    ])


@pytest.mark.asyncio
async def test_stop_requested_run_is_still_alive():
    t = uuid4()
    dead = await finished_owners([f"task:{t}:r"], _db([(t, "r", "stop_requested", "pausing")], None))
    assert dead == []


@pytest.mark.asyncio
async def test_reap_room_stops_dead_owners_and_marks_bg_rows(monkeypatch):
    from app.core import workspaced_client as wsd

    t = uuid4()
    dead_tag = f"task:{t}:old"
    monkeypatch.setattr(wsd, "list_proc_owners", AsyncMock(return_value={
        "owners": {dead_tag: [{"pid": 412, "cmd": "next dev"}]},
    }))
    reap = AsyncMock(return_value={"pids": [412, 418]})
    monkeypatch.setattr(wsd, "reap_proc_owners", reap)
    monkeypatch.setattr(proc_reaper, "finished_owners", AsyncMock(return_value=[dead_tag]))

    ws_id = uuid4()
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[
        SimpleNamespace(scalar_one_or_none=lambda: ws_id),
        SimpleNamespace(rowcount=1),
    ])
    db.commit = AsyncMock()

    pids = await proc_reaper.reap_room(uuid4(), db)
    assert pids == [412, 418]
    assert reap.await_args.args[1] == [dead_tag]
    update_sql = str(db.execute.await_args_list[1].args[0])
    assert "workspace_bg_proc" in update_sql and "status" in update_sql
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_reap_room_leaves_live_owners_alone(monkeypatch):
    from app.core import workspaced_client as wsd

    monkeypatch.setattr(wsd, "list_proc_owners", AsyncMock(return_value={
        "owners": {f"session:{uuid4()}": [{"pid": 9, "cmd": "sleep"}]},
    }))
    reap = AsyncMock()
    monkeypatch.setattr(wsd, "reap_proc_owners", reap)
    monkeypatch.setattr(proc_reaper, "finished_owners", AsyncMock(return_value=[]))

    assert await proc_reaper.reap_room(uuid4(), MagicMock()) == []
    reap.assert_not_awaited()


@pytest.mark.asyncio
async def test_reap_room_survives_workspaced_errors(monkeypatch):
    from app.core import workspaced_client as wsd

    monkeypatch.setattr(wsd, "list_proc_owners", AsyncMock(side_effect=RuntimeError("down")))
    assert await proc_reaper.reap_room(uuid4(), MagicMock()) == []
