"""Dialogue context reaches the executor and survives a restart.

Three regressions around the plan-dialogue archive (internal design notes):

* a task ChatMaster creates without a Goal (``task_create`` / ``user_cm_p2``
  close-flow) had no "Discussion context" — only goal-bound work was archived;
* a backend restart closed every open user session, so the next message found
  nothing to ``continue_session`` and no draft to ``merge_draft``;
* ``merge_draft`` wrote the carried turns into ``context_snapshot`` but the
  running turn saved its own history over them, and the tool result did not
  carry them either — the model never actually saw the merged dialogue.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import plan_dialogue

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, list[str]] = {}
        self.kv: dict[str, str] = {}

    async def rpush(self, key, value):
        self.store.setdefault(key, []).append(value)

    async def lpush(self, key, value):
        self.store.setdefault(key, []).insert(0, value)

    async def lrange(self, key, start, end):
        items = self.store.get(key, [])
        return list(items[start:] if end == -1 else items[start : end + 1])

    async def ltrim(self, key, start, end):
        pass

    async def expire(self, key, ttl):
        pass

    async def get(self, key):
        return self.kv.get(key)

    async def set(self, key, value, **_kw):
        self.kv[key] = value

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture
def fake_redis(monkeypatch):
    r = _FakeRedis()
    monkeypatch.setattr(plan_dialogue, "_get_redis", lambda: r)
    return r


class _FakeDB:
    def __init__(self, rows: dict) -> None:
        self._rows = rows

    async def get(self, model, ident):
        return self._rows.get((model.__name__, str(ident)))


# --- merge_draft ------------------------------------------------------------


@pytest.mark.asyncio
async def test_merged_draft_survives_the_turns_own_history_save(fake_redis):
    from app.core.user_sessions import _adopt_merged_draft_prefix, _get_history

    draft_sid, cur_sid = uuid.uuid4(), uuid.uuid4()
    await plan_dialogue.journal_append(draft_sid, "user", "install nginx in front of the api")
    await plan_dialogue.journal_append(draft_sid, "assistant", "what rate limit?")
    cur = SimpleNamespace(
        id=cur_sid, kind="user_cm_p2",
        context_snapshot={"messages": [{"role": "user", "content": "100 r/s"}]},
        message_count=1, turn_count=1,
    )
    db = _FakeDB({("Session", str(draft_sid)): SimpleNamespace(id=draft_sid, context_snapshot={})})

    # The turn loaded its history before the tool loop ran merge_draft.
    history = _get_history(cur)
    assert await plan_dialogue.merge_draft_into_session(cur, draft_sid, db) is True
    history.append({"role": "assistant", "content": "accepted, 100 r/s"})

    _adopt_merged_draft_prefix(cur, history)

    saved = cur.context_snapshot["messages"]
    assert saved[0]["content"].startswith(plan_dialogue._MERGE_MARKER_PREFIX)
    assert "install nginx" in saved[1]["content"]
    assert saved[-2]["content"] == "100 r/s"
    assert saved[-1]["content"] == "accepted, 100 r/s"
    assert plan_dialogue.PENDING_MERGED_PREFIX_KEY not in cur.context_snapshot

    # A second adopt (e.g. another save path) is a no-op.
    before = list(saved)
    _adopt_merged_draft_prefix(cur, history)
    assert cur.context_snapshot["messages"] == before


@pytest.mark.asyncio
async def test_merge_draft_tool_returns_the_transcript(fake_redis):
    from app.core.tools._merge_draft import MergeDraftTool

    draft_sid, cur_sid = uuid.uuid4(), uuid.uuid4()
    await plan_dialogue.journal_append(draft_sid, "user", "landing page for a coffee shop, palette #3b2f2f")
    cur = SimpleNamespace(id=cur_sid, kind="user_cm_p2", context_snapshot={"messages": []})
    db = _FakeDB({
        ("Session", str(draft_sid)): SimpleNamespace(id=draft_sid, context_snapshot={}),
        ("Session", str(cur_sid)): cur,
    })
    ctx = SimpleNamespace(session_id=cur_sid, room_id=uuid.uuid4(), task_id=None, db=db)

    out = await MergeDraftTool().execute(ctx, {"draft_session_id": str(draft_sid)})

    assert "palette #3b2f2f" in out


# --- task_create archives the dialogue ---------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,archived", [("user_cm_p2", True), ("cm_agent_proxy", False)])
async def test_task_create_archives_only_user_dialogues(monkeypatch, kind, archived):
    from app.core.tools import _task_create

    session = SimpleNamespace(id=uuid.uuid4(), kind=kind, user_id=uuid.uuid4())
    room_id = uuid.uuid4()
    db = AsyncMock()

    async def _get(model, ident):
        return session if model.__name__ == "Session" else None

    db.get = _get
    archive = AsyncMock(return_value=".legion/plans/task-x.md")
    monkeypatch.setattr(plan_dialogue, "archive_for_task", archive)
    ctx = SimpleNamespace(session_id=session.id, room_id=room_id, db=db)
    task = SimpleNamespace(id=uuid.uuid4(), title="Landing page")

    await _task_create._archive_dialogue_for_task(ctx, task, None)

    assert archive.await_count == (1 if archived else 0)
    if archived:
        args, kwargs = archive.await_args
        assert args[:3] == (room_id, task.id, session)
        assert kwargs["goal_id"] is None and kwargs["task_title"] == "Landing page"


def test_close_flow_archives_before_dispatch():
    """The executor's prompt links the archive only if it exists at start."""
    import inspect

    from app.core import session_close

    src = inspect.getsource(session_close.apply_close_actions)
    archive_at = src.index("plan_dialogue.archive_for_task(")
    dispatch_at = src.index("request_task_dispatch(")
    assert archive_at < dispatch_at
    assert 'kind in ("user_cm_p1", "user_cm_p2")' in src


# --- restart keeps young user sessions ---------------------------------------


def _db_capturing(results):
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    stmts = []

    async def _execute(stmt, *a, **kw):
        stmts.append(stmt)
        return results.pop(0) if results else MagicMock()

    db.execute = _execute
    return db, stmts


def _rows(items):
    r = MagicMock()
    r.scalars.return_value.all.return_value = items
    return r


@pytest.mark.asyncio
async def test_restart_closes_only_stale_user_sessions(monkeypatch):
    from app.core.watchdog import cleanup_orphaned_sessions

    monkeypatch.delenv("USER_SESSION_SURVIVE_RESTART", raising=False)
    db, stmts = _db_capturing([_rows([]), _rows([])])

    closed = await cleanup_orphaned_sessions(db, now=NOW)

    assert closed == 0
    user_sql = str(stmts[1]).split("WHERE", 1)[1]
    assert "last_activity_at" in user_sql and "started_at" in user_sql


@pytest.mark.asyncio
async def test_restart_kill_switch_closes_all_user_sessions(monkeypatch):
    from app.core.watchdog import cleanup_orphaned_sessions

    monkeypatch.setenv("USER_SESSION_SURVIVE_RESTART", "false")
    db, stmts = _db_capturing([_rows([]), _rows([])])

    await cleanup_orphaned_sessions(db, now=NOW)

    assert "last_activity_at" not in str(stmts[1]).split("WHERE", 1)[1]


# --- watchdog close leaves a draft --------------------------------------------


def _first(value):
    r = MagicMock()
    r.first.return_value = value
    return r


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_row,goal_row,saved",
    [(None, None, True), (("t",), None, False), (None, ("g",), False)],
)
async def test_watchdog_close_saves_draft_only_for_unfinished_dialogue(
    task_row, goal_row, saved
):
    from app.core import watchdog

    session = SimpleNamespace(id=uuid.uuid4(), kind="user_cm_p2")
    db, _ = _db_capturing([_first(task_row), _first(goal_row)])

    with patch(
        "app.core.user_sessions._maybe_save_dialogue_draft", new_callable=AsyncMock
    ) as save:
        await watchdog._save_unfinished_dialogue_draft(session, "idle", db)

    assert save.await_count == (1 if saved else 0)
