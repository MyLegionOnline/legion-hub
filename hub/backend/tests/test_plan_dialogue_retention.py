"""internal design notes Stage 8 — retention sweep.

No Redis, no PostgreSQL: workspaced client and the DB session are faked.
"""
from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import pytest

from app.core import plan_dialogue, plan_dialogue_retention as ret


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return self

    def all(self):
        return list(self._rows)


class _FakeDB:
    """Return preset row lists in call order; RoomGoal query first, Task second."""

    def __init__(self, results):
        self._results = list(results)

    async def execute(self, _stmt):
        return _FakeResult(self._results.pop(0) if self._results else [])


@pytest.fixture
def fake_wsd(monkeypatch):
    state = {"files": [], "deleted": []}

    async def _list(room_id):
        return list(state["files"])

    async def _delete(room_id, path, commit_message=None):
        state["deleted"].append(path)
        return True

    import app.core.workspaced_client as real_wsd

    monkeypatch.setattr(real_wsd, "list_dialogue_archive_files", _list)
    monkeypatch.setattr(real_wsd, "delete_workspace_file", _delete)
    return state


@pytest.fixture
def no_redis(monkeypatch):
    monkeypatch.setattr(plan_dialogue, "_get_redis", lambda: None)


def _f(path, *, age_days=0.0):
    return {"path": path, "mtime": time.time() - age_days * 86400, "size": 10}


@pytest.mark.asyncio
async def test_sweep_deletes_old_drafts_only(fake_wsd, no_redis):
    fresh, stale = uuid.uuid4(), uuid.uuid4()
    fake_wsd["files"] = [
        _f(f".legion/drafts/{fresh}.md", age_days=2),
        _f(f".legion/drafts/{stale}.md", age_days=ret.DRAFT_MAX_AGE_DAYS + 1),
    ]
    db = _FakeDB([[], []])  # no goals, no tasks
    n = await ret.sweep_room(uuid.uuid4(), db)
    assert n == 1
    assert fake_wsd["deleted"] == [f".legion/drafts/{stale}.md"]


@pytest.mark.asyncio
async def test_sweep_deletes_merged_draft_even_if_fresh(fake_wsd, monkeypatch):
    merged = uuid.uuid4()
    fake_wsd["files"] = [_f(f".legion/drafts/{merged}.md", age_days=1)]

    async def _is_merged(sid):
        return sid == str(merged)

    monkeypatch.setattr(ret, "_is_merged", _is_merged)
    db = _FakeDB([[], []])
    n = await ret.sweep_room(uuid.uuid4(), db)
    assert n == 1
    assert fake_wsd["deleted"] == [f".legion/drafts/{merged}.md"]


@pytest.mark.asyncio
async def test_sweep_deletes_orphaned_goal_and_task_archives(fake_wsd, no_redis):
    live_goal, dead_goal = uuid.uuid4(), uuid.uuid4()
    live_task, dead_task = uuid.uuid4(), uuid.uuid4()
    fake_wsd["files"] = [
        _f(f".legion/plans/goal-{live_goal}.md"),
        _f(f".legion/plans/goal-{dead_goal}.md"),
        _f(f".legion/plans/goal-{dead_goal}-attachments/01-nginx.conf"),
        _f(f".legion/plans/task-{live_task}.md"),
        _f(f".legion/plans/task-{dead_task}.md"),
    ]
    db = _FakeDB([[live_goal], [live_task]])
    n = await ret.sweep_room(uuid.uuid4(), db)
    assert n == 3
    assert set(fake_wsd["deleted"]) == {
        f".legion/plans/goal-{dead_goal}.md",
        f".legion/plans/goal-{dead_goal}-attachments/01-nginx.conf",
        f".legion/plans/task-{dead_task}.md",
    }


@pytest.mark.asyncio
async def test_sweep_keeps_everything_live(fake_wsd, no_redis):
    g, t = uuid.uuid4(), uuid.uuid4()
    fake_wsd["files"] = [
        _f(f".legion/plans/goal-{g}.md"),
        _f(f".legion/plans/task-{t}.md"),
        _f(f".legion/drafts/{uuid.uuid4()}.md", age_days=1),
    ]
    db = _FakeDB([[g], [t]])
    assert await ret.sweep_room(uuid.uuid4(), db) == 0
    assert fake_wsd["deleted"] == []


@pytest.mark.asyncio
async def test_sweep_room_swallows_listing_failure(monkeypatch):
    async def _boom(room_id):
        raise RuntimeError("workspaced down")

    import app.core.workspaced_client as real_wsd

    monkeypatch.setattr(real_wsd, "list_dialogue_archive_files", _boom)
    assert await ret.sweep_room(uuid.uuid4(), _FakeDB([])) == 0


@pytest.mark.asyncio
async def test_sweep_all_iterates_distinct_rooms(fake_wsd, no_redis, monkeypatch):
    room_a, room_b = uuid.uuid4(), uuid.uuid4()
    seen: list = []

    async def _sweep_room(room_id, db):
        seen.append(room_id)
        return 0

    monkeypatch.setattr(ret, "sweep_room", _sweep_room)
    db = _FakeDB([[room_a, room_b], [room_b]])  # goal rooms, then task rooms
    await ret.sweep_all(db)
    assert set(seen) == {room_a, room_b}
