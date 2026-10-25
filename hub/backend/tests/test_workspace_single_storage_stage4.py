"""
Stage 4 of internal design notes — agent file_writes go to
the workspace volume.

Stage 6 removed the legacy branch; these tests pin the single behaviour:

  * ``_do_apply`` block 8 skips DB rows entirely and records
    ``{path, created}`` (no ``file_id``) in the outcome;
  * ``apply_close_actions`` writes agent files to the volume via
    ``workspaced_client.write_workspace_file`` (with auto-commit) post-DB-tx;
  * ``_task_scoped_file_changes`` reads the volume snapshot (mtime-based).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core import session_close as _sc
from app.core.session_close import FileWrite, SessionCloseActions


# ── helpers ──────────────────────────────────────────────────────────────────

def _session(room_id=None, agent_id=None):
    s = MagicMock()
    s.id = uuid4()
    s.room_id = room_id or uuid4()
    s.agent_id = agent_id or uuid4()
    s.kind = "proxy_agent"
    s.plan_id = None
    s.pending_actions = None
    return s


def _make_db(executes: dict | None = None):
    async def _execute(stmt, *a, **kw):
        sql_text = str(stmt)
        for needle, value in (executes or {}).items():
            if needle in sql_text:
                return value
        return MagicMock()

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=_execute)
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.add = MagicMock()
    db.refresh = AsyncMock()
    return db


# ── _do_apply block 8 ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_do_apply_skips_db_rows():
    """Block 8 must NOT create DB rows — the volume is the primary truth,
    writes go post-commit."""
    room_id = uuid4()
    session = _session(room_id=room_id)
    actions = SessionCloseActions(file_writes=[
        FileWrite(path="src/main.py", content="print('hi')"),
    ])
    db = _make_db()

    with patch.object(_sc, "resource_lock", new_callable=AsyncMock):
        outcome = await _sc._do_apply(session, actions, db)

    assert len(outcome["file_changes"]) == 1
    fc = outcome["file_changes"][0]
    assert fc["path"] == "/src/main.py"
    assert fc["created"] is True
    assert "file_id" not in fc


@pytest.mark.asyncio
async def test_do_apply_multiple_files_no_file_id():
    session = _session()
    actions = SessionCloseActions(file_writes=[
        FileWrite(path="a.py", content="a = 1"),
        FileWrite(path="b.py", content="b = 2"),
    ])
    db = _make_db()

    with patch.object(_sc, "resource_lock", new_callable=AsyncMock):
        outcome = await _sc._do_apply(session, actions, db)

    assert len(outcome["file_changes"]) == 2
    for fc in outcome["file_changes"]:
        assert "file_id" not in fc
        assert "path" in fc
        assert fc["created"] is True


@pytest.mark.asyncio
async def test_do_apply_no_storage_put():
    """storage.put must not be called."""
    session = _session()
    actions = SessionCloseActions(file_writes=[
        FileWrite(path="x.py", content="x"),
    ])
    db = _make_db()

    with patch.object(_sc, "resource_lock", new_callable=AsyncMock), \
         patch("app.core.storage.get_storage_backend") as gs:
        storage_mock = MagicMock()
        storage_mock.put = AsyncMock()
        gs.return_value = storage_mock
        await _sc._do_apply(session, actions, db)

    storage_mock.put.assert_not_called()


# ── apply_close_actions — post-commit volume write ──────────────────────────

@pytest.mark.asyncio
async def test_apply_close_actions_writes_to_volume():
    """apply_close_actions must call wsd.write_workspace_file for each
    file_write, with an agent-sourced commit message."""
    room_id = uuid4()
    agent_id = uuid4()
    session = _session(room_id=room_id, agent_id=agent_id)
    actions = SessionCloseActions(file_writes=[
        FileWrite(path="src/main.py", content="print('hi')"),
        FileWrite(path="src/util.py", content="x = 1"),
    ])

    outcome = {
        "file_changes": [
            {"path": "/src/main.py", "created": True},
            {"path": "/src/util.py", "created": True},
        ],
        "touched_goal_ids": set(),
        "created_tasks": [],
        "reopened_goals": {},
        "terminal_task_ids": set(),
        "failed_task_ids": set(),
    }

    with patch.object(_sc, "_do_apply", new_callable=AsyncMock, return_value=outcome), \
         patch.object(_sc, "filter_by_kind", return_value=(actions, None)), \
         patch("app.core.events.emit_event", new_callable=AsyncMock), \
         patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock) as mw:
        await _sc.apply_close_actions(session, actions, AsyncMock())

    assert mw.await_count == 2
    first_call = mw.call_args_list[0]
    assert first_call.args[0] == str(room_id)
    assert first_call.args[1] == "/src/main.py"
    assert first_call.args[2] == "print('hi')"
    assert f"agent:{agent_id}" in first_call.kwargs["commit_message"]
    assert "/src/main.py" in first_call.kwargs["commit_message"]


@pytest.mark.asyncio
async def test_apply_close_actions_volume_write_failure_logged_not_raised():
    """If wsd.write_workspace_file raises, the error must be logged but not
    propagated — the DB transaction already succeeded and other side effects
    (events, task dispatch) must still run."""
    session = _session()
    actions = SessionCloseActions(file_writes=[
        FileWrite(path="x.py", content="x"),
    ])

    outcome = {
        "file_changes": [{"path": "/x.py", "created": True}],
        "touched_goal_ids": set(),
        "created_tasks": [],
        "reopened_goals": {},
        "terminal_task_ids": set(),
        "failed_task_ids": set(),
    }

    with patch.object(_sc, "_do_apply", new_callable=AsyncMock, return_value=outcome), \
         patch.object(_sc, "filter_by_kind", return_value=(actions, None)), \
         patch("app.core.events.emit_event", new_callable=AsyncMock) as me, \
         patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock,
               side_effect=RuntimeError("network down")):
        await _sc.apply_close_actions(session, actions, AsyncMock())

    event_types = [c.kwargs["event_type"] for c in me.call_args_list]
    assert "FILE_CHANGED" in event_types


# ── FILE_CHANGED events — no file_id ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_file_changed_event_metadata_has_no_file_id():
    session = _session()
    actions = SessionCloseActions(file_writes=[
        FileWrite(path="a.py", content="a"),
    ])

    outcome = {
        "file_changes": [{"path": "/a.py", "created": True}],
        "touched_goal_ids": set(),
        "created_tasks": [],
        "reopened_goals": {},
        "terminal_task_ids": set(),
        "failed_task_ids": set(),
    }

    with patch.object(_sc, "_do_apply", new_callable=AsyncMock, return_value=outcome), \
         patch.object(_sc, "filter_by_kind", return_value=(actions, None)), \
         patch("app.core.events.emit_event", new_callable=AsyncMock) as me, \
         patch("app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock):
        await _sc.apply_close_actions(session, actions, AsyncMock())

    fc_event = next(
        c for c in me.call_args_list
        if c.kwargs["event_type"] == "FILE_CHANGED"
    )
    md = fc_event.kwargs["metadata"]
    assert md["path"] == "/a.py"
    assert md["created"] is True
    assert "file_id" not in md


# ── _task_scoped_file_changes — uses volume snapshot ────────────────────────

@pytest.mark.asyncio
async def test_task_scoped_file_changes_uses_snapshot():
    """_task_scoped_file_changes reads the volume snapshot and filters by
    mtime >= task start."""
    from app.core import proxy_session as _ps

    task_id = uuid4()
    room_id = uuid4()
    now_ts = datetime.now(timezone.utc).timestamp()
    old_ts = now_ts - 7200
    recent_ts = now_ts - 60

    started_at = datetime.now(timezone.utc) - timedelta(minutes=30)

    db = AsyncMock()

    async def _execute(stmt, *a, **kw):
        sql_text = str(stmt)
        if "min(" in sql_text and "started_at" in sql_text:
            r = MagicMock()
            r.scalar.return_value = started_at
            return r
        return MagicMock()

    db.execute = AsyncMock(side_effect=_execute)

    snapshot = {
        "files": [
            {"path": "src/new.py", "mtime": recent_ts},
            {"path": "src/old.py", "mtime": old_ts},
            {"path": "src/recent2.py", "mtime": recent_ts - 10},
            {"path": "node_modules/pkg/index.js", "mtime": recent_ts},
        ],
        "generation": {"revision": 42},
    }

    with patch("app.core.workspaced_client.get_workspace_snapshot",
               new_callable=AsyncMock, return_value=snapshot):
        changes = await _ps._task_scoped_file_changes(task_id, room_id, db)

    assert changes is not None
    paths_in_changes = [c for c in changes if "old.py" in c]
    assert len(paths_in_changes) == 0
    new_paths = [c for c in changes if "new.py" in c]
    assert len(new_paths) == 1
    recent2_paths = [c for c in changes if "recent2.py" in c]
    assert len(recent2_paths) == 1
    nm_paths = [c for c in changes if "node_modules" in c]
    assert len(nm_paths) == 0
    new_idx = changes.index(new_paths[0])
    recent2_idx = changes.index(recent2_paths[0])
    assert new_idx < recent2_idx


@pytest.mark.asyncio
async def test_task_scoped_file_changes_no_files():
    from app.core import proxy_session as _ps

    task_id = uuid4()
    room_id = uuid4()
    started_at = datetime.now(timezone.utc) - timedelta(minutes=5)

    db = AsyncMock()

    async def _execute(stmt, *a, **kw):
        r = MagicMock()
        r.scalar.return_value = started_at
        return r

    db.execute = AsyncMock(side_effect=_execute)

    with patch("app.core.workspaced_client.get_workspace_snapshot",
               new_callable=AsyncMock, return_value={"files": [], "generation": {}}):
        changes = await _ps._task_scoped_file_changes(task_id, room_id, db)

    assert changes == []


@pytest.mark.asyncio
async def test_task_scoped_file_changes_returns_none_when_no_sessions():
    """When no proxy sessions exist for the task, return None so the caller
    falls back to the pair-start snapshot comparison."""
    from app.core import proxy_session as _ps

    task_id = uuid4()
    room_id = uuid4()
    db = AsyncMock()

    async def _execute(stmt, *a, **kw):
        r = MagicMock()
        r.scalar.return_value = None
        return r

    db.execute = AsyncMock(side_effect=_execute)

    changes = await _ps._task_scoped_file_changes(task_id, room_id, db)
    assert changes is None
