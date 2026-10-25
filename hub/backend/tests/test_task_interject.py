"""internal design notes Stage 2 — a user's message into a running task.

- refusals for tasks that cannot take a message;
- the note joins the trailing user message (two ``user`` turns in a row are
  rejected by the Messages API) without mutating the caller's history;
- the proxy tool loop delivers queued messages before the next model request
  and leaves the note for the agent history;
- ``reply_to_user`` is offered despite a tools_allowed list and refuses when
  nothing is waiting for an answer.

Pure/mocked — no database required.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.core import task_interject as ti
from tests.test_db_release import _END_TURN, _TOOL_USE, TxDb, _ant_client, _dirty_emit


def _task(**kw):
    base = {"runtime_state": "running", "awaiting_user_kind": None}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "task, session, code",
    [
        (_task(awaiting_user_kind="question"), object(), "awaiting_user"),
        (_task(runtime_state="paused"), object(), "paused"),
        (_task(runtime_state="idle"), object(), "not_running"),
        (_task(), None, "no_session"),
    ],
)
def test_refusals(task, session, code):
    with pytest.raises(ti.InterjectRefused) as err:
        ti.check_can_interject(task, session)
    assert err.value.code == code


def test_running_task_with_agent_session_is_accepted():
    ti.check_can_interject(_task(), object())


def test_note_quotes_every_message():
    note = ti.format_note(["check the status", "and one more\nquestion"])
    assert note.startswith("[MESSAGE FROM THE USER DURING THE TASK")
    assert "> check the status" in note
    assert "> and one more\n> question" in note
    assert "reply_to_user" in note


def test_inject_anthropic_joins_trailing_user_message_without_mutating_history():
    history = [{"role": "user", "content": "task text"}]
    working = list(history)
    ti.inject_anthropic(working, "NOTE")
    assert len(working) == 1
    assert working[0]["content"] == [
        {"type": "text", "text": "task text"},
        {"type": "text", "text": "NOTE"},
    ]
    assert history[0]["content"] == "task text"


def test_inject_anthropic_after_tool_results():
    results = [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]
    working = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": results},
    ]
    ti.inject_anthropic(working, "NOTE")
    assert working[-1]["content"] == [*results, {"type": "text", "text": "NOTE"}]
    assert results == [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]


def test_inject_anthropic_after_assistant_appends():
    working = [{"role": "assistant", "content": "x"}]
    ti.inject_anthropic(working, "NOTE")
    assert working[-1] == {"role": "user", "content": "NOTE"}


def test_inject_openai_appends_user_message():
    working = [{"role": "tool", "tool_call_id": "c", "content": "ok"}]
    ti.inject_openai(working, "NOTE")
    assert working[-1] == {"role": "user", "content": "NOTE"}


def test_persist_in_history_folds_into_last_user_message():
    first = {"role": "user", "content": "answer from CM"}
    history = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "q"}, first]
    ti.persist_in_history(history, ["NOTE"])
    assert history[-1]["content"] == "answer from CM\n\nNOTE"
    assert first["content"] == "answer from CM"
    ti.persist_in_history(history, None)
    assert history[-1]["content"] == "answer from CM\n\nNOTE"


def test_reply_tool_is_offered_despite_allowlist():
    from app.core.tools.registry import REGISTRY
    import app.core.tools  # noqa: F401 — registers the tools

    names = [t.name for t in REGISTRY.tools_for("agent", ["read_file"])]
    assert "reply_to_user" in names
    assert "write_file" not in names
    assert "reply_to_user" not in [t.name for t in REGISTRY.tools_for("cm", None)]


@pytest.mark.asyncio
async def test_reply_tool_refuses_without_waiting_message():
    from app.core.tools._reply_to_user import ReplyToUserTool
    from app.core.tools.registry import ToolContext

    ctx = ToolContext(room_id=uuid.uuid4(), session_id=uuid.uuid4(), task_id=uuid.uuid4(), db=None)
    with patch("app.core.task_interject.record_reply", new=AsyncMock(return_value=None)):
        out = await ReplyToUserTool().execute(ctx, {"text": "working"})
    assert out.startswith("Error: there is no user message")

    with patch("app.core.task_interject.record_reply", new=AsyncMock(return_value=object())):
        out = await ReplyToUserTool().execute(ctx, {"text": "working"})
    assert out.startswith("Delivered")


class _NestedDb:
    """begin_nested + execute + commit; ``fail`` makes the query raise."""

    def __init__(self, rows, fail=False):
        self.rows, self.fail = rows, fail
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.added = []

    def begin_nested(self):
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def cm():
            yield

        return cm()

    async def execute(self, *a, **kw):
        if self.fail:
            raise RuntimeError("db down")
        from unittest.mock import MagicMock

        res = MagicMock()
        res.scalars.return_value.all.return_value = self.rows
        return res

    def add(self, obj):
        self.added.append(obj)


@pytest.mark.asyncio
async def test_deliver_pending_marks_rows_and_commits():
    from app.models.task_interjection import TaskInterjection

    task_id, sid = uuid.uuid4(), uuid.uuid4()
    rows = [TaskInterjection(room_id=uuid.uuid4(), task_id=task_id, text="status?")]
    db = _NestedDb(rows)
    with patch("app.core.task_interject._broadcast_state", new=AsyncMock()) as bc:
        texts = await ti.deliver_pending(task_id, sid, db)
    assert texts == ["status?"]
    assert rows[0].delivered_at is not None and rows[0].delivered_session_id == sid
    db.commit.assert_awaited_once()
    bc.assert_awaited_once()


@pytest.mark.asyncio
async def test_stamp_messages_keeps_the_mark_in_bubble_metadata():
    """Stage 5.3: the "delivered to the agent" mark survives a history reload."""
    from datetime import datetime, timezone

    from app.models.message import Message
    from app.models.task_interjection import TaskInterjection

    msg = Message(room_id=uuid.uuid4(), content="status?", event_metadata={"kind": ti.KIND_MESSAGE})
    original = msg.event_metadata
    row = TaskInterjection(room_id=msg.room_id, task_id=uuid.uuid4(), text="status?", message_id=msg.id)
    db = _NestedDb([msg])
    when = datetime(2026, 10, 1, 11, 48, tzinfo=timezone.utc)
    await ti._stamp_messages([row], "delivered_at", when, db)
    assert msg.event_metadata == {"kind": ti.KIND_MESSAGE, "delivered_at": when.isoformat()}
    assert original == {"kind": ti.KIND_MESSAGE}  # replaced, not mutated
    assert db.added == [msg]

    db = _NestedDb([], fail=True)  # no message ids → no query at all
    await ti._stamp_messages([TaskInterjection(room_id=uuid.uuid4(), task_id=uuid.uuid4(), text="x")],
                             "delivered_at", when, db)


@pytest.mark.asyncio
async def test_deliver_pending_failure_keeps_the_loop_transaction():
    db = _NestedDb([], fail=True)
    assert await ti.deliver_pending(uuid.uuid4(), uuid.uuid4(), db) == []
    db.rollback.assert_not_awaited()
    db.commit.assert_not_awaited()
    assert await ti.deliver_pending(None, uuid.uuid4(), db) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("context_role, delivered", [("agent", True), ("chat_master", False)])
async def test_tool_loop_delivers_before_next_request(context_role, delivered):
    from app.core.proxy_session import _call_llm_with_tools

    db = TxDb()
    ctx = _ant_client(db, [_TOOL_USE, _END_TURN])
    queue = [["check the status"]]

    async def fake_deliver(task_id, session_id, db):
        return queue.pop(0) if queue else []

    history = [{"role": "user", "content": "hi"}]
    signals: dict = {}
    with (
        patch("app.core.proxy_session.httpx.AsyncClient", return_value=ctx),
        patch("app.core.proxy_session.emit_event", side_effect=_dirty_emit(db)),
        patch("app.core.proxy_session.emit_status", side_effect=_dirty_emit(db)),
        patch("app.core.task_interject.deliver_pending", side_effect=fake_deliver),
    ):
        result = await _call_llm_with_tools(
            {"provider": "anthropic", "model": "m", "api_key": "k"},
            "SYS",
            history,
            room_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            db=db,
            workspace_ant_tools=[{"name": "no_such_tool", "input_schema": {"type": "object"}}],
            context_role=context_role,
            loop_signals=signals,
        )
    assert result[0] == "done"
    client = ctx.__aenter__.return_value
    first_body = client.post.call_args_list[0].kwargs["json"]
    first_text = str(first_body["messages"][0]["content"])
    assert ("check the status" in first_text) is delivered
    roles = [m["role"] for m in first_body["messages"]]
    assert all(not (a == b == "user") for a, b in zip(roles, roles[1:]))
    assert history == [{"role": "user", "content": "hi"}]
    assert bool(signals.get("interjections")) is delivered
