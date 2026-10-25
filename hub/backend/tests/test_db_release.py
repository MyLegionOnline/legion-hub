"""Stage 1.2 of internal design notes (§8.2): no transaction across a model wait.

The 0.4 ladder measured N_max = pool_size + max_overflow rooms: every running
proxy session held one connection ``idle in transaction`` for its whole task.
These tests pin the fix and its instruments:

  1.2.0  pool wait timing (private SQLAlchemy API — fails loudly on a rename),
         connection ownership by asyncio task, hub_llm_wait_open_tx_total;
  1.2.1  _llm_request_with_retry commits before every attempt;
  1.2.2  proxy tool loop (both wires, forced answer / forced close), the
         ChatMaster side and the user-session call reach the model with no
         open transaction;
  1.2.4  a pair closed by the watchdog while the model was answering gets
         neither text nor tokens;
  1.2.5  task is flushed before session;
  locks  release_db never commits away a transaction-scoped lock.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import uuid
import warnings
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.db.release import XACT_LOCK_INFO_KEY, mark_xact_lock, release_db


# ---------------------------------------------------------------------------
# A session double that knows whether it has an open transaction
# ---------------------------------------------------------------------------


class TxDb:
    """Minimal AsyncSession stand-in with real transaction state.

    Any query opens a transaction, commit/rollback closes it — exactly the
    property the fix is about.  ``at_model_call`` records the state at the
    instant a (faked) provider request starts.
    """

    def __init__(self) -> None:
        self.tx = False
        self.info: dict = {}
        self.commits = 0
        self.at_model_call: list[bool] = []
        result = MagicMock()
        result.scalars.return_value.first.return_value = None
        result.scalars.return_value.all.return_value = []
        result.first.return_value = None
        result.all.return_value = []
        result.scalar_one_or_none.return_value = None
        self._result = result
        self.add = MagicMock(side_effect=self._touch)

    def _touch(self, *args, **kwargs) -> None:
        self.tx = True

    def in_transaction(self) -> bool:
        return self.tx

    def model_called(self) -> None:
        self.at_model_call.append(self.tx)

    async def execute(self, *args, **kwargs):
        self.tx = True
        return self._result

    async def scalar(self, *args, **kwargs):
        self.tx = True
        return None

    async def get(self, *args, **kwargs):
        self.tx = True
        return None

    async def flush(self, *args, **kwargs) -> None:
        self.tx = True

    async def refresh(self, *args, **kwargs) -> None:
        self.tx = True

    async def commit(self) -> None:
        self.tx = False
        self.commits += 1
        self.info.pop(XACT_LOCK_INFO_KEY, None)

    async def rollback(self) -> None:
        self.tx = False
        self.info.pop(XACT_LOCK_INFO_KEY, None)


# ---------------------------------------------------------------------------
# release_db
# ---------------------------------------------------------------------------


class TestReleaseDb:
    @pytest.mark.asyncio
    async def test_commits_an_open_transaction(self):
        db = TxDb()
        db.tx = True
        assert await release_db(db, reason="t") is True
        assert db.tx is False and db.commits == 1

    @pytest.mark.asyncio
    async def test_no_transaction_no_commit(self):
        db = TxDb()
        assert await release_db(db, reason="t") is False
        assert db.commits == 0

    @pytest.mark.asyncio
    async def test_kill_switch_restores_pre_fix_behaviour(self, monkeypatch):
        """DB_RELEASE_BEFORE_AWAIT=off — the A-local run and the fast rollback."""
        import app.db.release as release

        monkeypatch.setattr(release, "_ENABLED", False)
        db = TxDb()
        db.tx = True
        assert await release.release_db(db, reason="t") is False
        assert db.tx is True and db.commits == 0

    @pytest.mark.asyncio
    async def test_none_is_a_noop(self):
        assert await release_db(None, reason="t") is False

    @pytest.mark.asyncio
    async def test_keeps_a_transaction_that_holds_a_lock(self):
        """Committing would drop pg_advisory_xact_lock / FOR UPDATE silently."""
        db = TxDb()
        db.tx = True
        mark_xact_lock(db)
        assert await release_db(db, reason="t") is False
        assert db.tx is True and db.commits == 0

    @pytest.mark.asyncio
    async def test_test_doubles_are_left_alone(self):
        """MagicMock / AsyncMock sessions of older tests must not be committed
        (a MagicMock commit is not awaitable) nor leak un-awaited coroutines."""
        magic = MagicMock()
        assert await release_db(magic, reason="t") is False
        magic.commit.assert_not_called()
        amock = AsyncMock()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert await release_db(amock, reason="t") is False
        amock.commit.assert_not_awaited()


class TestLockFlagDetection:
    """The flag is raised by the ORM session itself, not by each call site."""

    def _session(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        engine = create_engine("sqlite://")
        return Session(engine)

    def test_for_update_flags_the_transaction_until_it_ends(self):
        from sqlalchemy import literal, select

        s = self._session()
        s.execute(select(literal(1)).with_for_update())
        assert s.info.get(XACT_LOCK_INFO_KEY) is True
        s.commit()
        assert XACT_LOCK_INFO_KEY not in s.info

    def test_advisory_xact_lock_flags_the_transaction(self):
        from sqlalchemy import text

        s = self._session()
        # SQLite has no such function; the flag is raised before execution,
        # which is exactly the point — and rollback clears it.
        with pytest.raises(Exception):
            s.execute(text("SELECT pg_advisory_xact_lock(1)"))
        assert s.info.get(XACT_LOCK_INFO_KEY) is True
        s.rollback()
        assert XACT_LOCK_INFO_KEY not in s.info

    def test_plain_select_does_not_flag(self):
        from sqlalchemy import literal, select

        s = self._session()
        s.execute(select(literal(1)))
        assert XACT_LOCK_INFO_KEY not in s.info
        s.close()

    @pytest.mark.asyncio
    async def test_resource_lock_marks_the_session(self):
        from app.core.session_close import resource_lock

        db = TxDb()
        await resource_lock(db, uuid.uuid4())
        assert db.info.get(XACT_LOCK_INFO_KEY) is True
        assert await release_db(db, reason="t") is False


# ---------------------------------------------------------------------------
# 1.2.0 instruments
# ---------------------------------------------------------------------------


class TestPoolInstrumentation:
    def test_private_do_get_still_exists(self):
        """``_do_get`` is private SQLAlchemy API.  If a version bump renames it,
        the override silently stops timing — fail here instead."""
        from sqlalchemy.pool import AsyncAdaptedQueuePool, QueuePool

        from app.db.pool_instrumentation import (
            TimedAsyncAdaptedQueuePool,
            _TimedGetMixin,
        )

        assert "_do_get" in QueuePool.__dict__
        assert issubclass(AsyncAdaptedQueuePool, QueuePool)
        assert TimedAsyncAdaptedQueuePool._do_get is _TimedGetMixin._do_get

    def test_engine_uses_the_timed_pool(self):
        from app.db.pool_instrumentation import TimedAsyncAdaptedQueuePool
        from app.db.session import engine

        assert isinstance(engine.sync_engine.pool, TimedAsyncAdaptedQueuePool)

    def test_checkout_is_timed(self):
        from app.db.pool_instrumentation import TimedQueuePool

        pool = TimedQueuePool(lambda: sqlite3.connect(":memory:"), pool_size=1, max_overflow=0)
        hist = MagicMock()
        with patch("app.core.metrics.db_pool_wait_seconds", hist):
            conn = pool.connect()
            conn.close()
        assert hist.observe.call_count == 1
        assert hist.observe.call_args.args[0] >= 0

    @pytest.mark.asyncio
    async def test_ownership_follows_the_asyncio_task(self):
        from app.db.pool_instrumentation import (
            TimedQueuePool,
            current_task_holds_connection,
            install_connection_tracking,
        )

        pool = TimedQueuePool(lambda: sqlite3.connect(":memory:"), pool_size=2, max_overflow=0)
        install_connection_tracking(pool)

        assert current_task_holds_connection() is False
        conn = pool.connect()
        assert current_task_holds_connection() is True

        async def other_task() -> bool:
            return current_task_holds_connection()

        assert await asyncio.create_task(other_task()) is False
        conn.close()
        assert current_task_holds_connection() is False

    @pytest.mark.asyncio
    async def test_open_tx_counter_counts_only_held_connections(self):
        from app.core import step_profile
        from app.db.pool_instrumentation import TimedQueuePool, install_connection_tracking

        counter = MagicMock()
        pool = TimedQueuePool(lambda: sqlite3.connect(":memory:"), pool_size=1, max_overflow=0)
        install_connection_tracking(pool)
        with patch.object(step_profile, "llm_wait_open_tx_total", counter):
            step_profile.note_llm_call("agent")
            counter.labels.assert_not_called()

            conn = pool.connect()
            step_profile.step_tick(step_profile.SIDE_CHAT_MASTER)
            step_profile.note_llm_call("agent")  # the open step's side wins
            conn.close()
            step_profile.step_end()
        counter.labels.assert_called_once_with(side="chat_master")
        counter.labels.return_value.inc.assert_called_once()


# ---------------------------------------------------------------------------
# 1.2.1 _llm_request_with_retry
# ---------------------------------------------------------------------------


class TestRetryHelperReleases:
    @pytest.mark.asyncio
    async def test_commits_before_the_request(self):
        from app.core.proxy_session import _llm_request_with_retry

        db = TxDb()
        db.tx = True

        async def request():
            db.model_called()
            return {"ok": True}

        out = await _llm_request_with_retry(request, session_id="s", label="t", db=db)
        assert out == {"ok": True}
        assert db.at_model_call == [False]

    @pytest.mark.asyncio
    async def test_commits_before_every_retry(self):
        from app.core import proxy_session

        db = TxDb()
        db.tx = True
        calls = 0

        async def request():
            nonlocal calls
            calls += 1
            db.model_called()
            if calls == 1:
                db.tx = True  # something re-opened it before the failure
                raise httpx.ConnectError("drop")
            return "ok"

        with patch.object(proxy_session, "_LLM_RETRY_BACKOFF_S", (0.0, 0.0, 0.0)):
            out = await proxy_session._llm_request_with_retry(
                request, session_id="s", label="t", db=db
            )
        assert out == "ok"
        assert db.at_model_call == [False, False]

    @pytest.mark.asyncio
    async def test_without_db_behaves_as_before(self):
        from app.core.proxy_session import _llm_request_with_retry

        async def request():
            return 1

        assert await _llm_request_with_retry(request, session_id="s", label="t") == 1


# ---------------------------------------------------------------------------
# 1.2.2 the proxy tool loop, both wires, all exits
# ---------------------------------------------------------------------------


def _ant_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = payload
    return resp


def _ant_client(db: TxDb, payloads: list[dict]) -> MagicMock:
    responses = iter(payloads)

    async def post(*args, **kwargs):
        db.model_called()
        return _ant_response(next(responses))

    client = MagicMock()
    client.post = AsyncMock(side_effect=post)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _oai_resp(content: str | None, tool_calls=None) -> MagicMock:
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.model_extra = {}
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "tool_calls" if tool_calls else "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    resp.usage.completion_tokens_details = None
    resp.usage.prompt_tokens_details = None
    return resp


def _oai_tool_call(name: str, args: str = "{}") -> MagicMock:
    tc = MagicMock()
    tc.id = f"call-{uuid.uuid4().hex[:6]}"
    tc.type = "function"
    tc.function.name = name
    tc.function.arguments = args
    return tc


_TOOL_USE = {
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "stop_reason": "tool_use",
    "content": [{"type": "tool_use", "id": "t1", "name": "no_such_tool", "input": {}}],
}
_END_TURN = {
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "done"}],
}
_CLOSE = {
    "usage": {"input_tokens": 10, "output_tokens": 5},
    "stop_reason": "tool_use",
    "content": [{"type": "tool_use", "id": "c1", "name": "close_session", "input": {"summary": "s"}}],
}


def _dirty_emit(db: TxDb):
    """emit_event/emit_status stand-ins that touch the session like the real ones."""

    async def emit(*args, **kwargs):
        db.tx = True

    return emit


async def _proxy_loop(db: TxDb, llm_config: dict, **kwargs):
    from app.core.proxy_session import _call_llm_with_tools

    return await _call_llm_with_tools(
        llm_config,
        "SYS",
        [{"role": "user", "content": "hi"}],
        room_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        task_id=None,
        db=db,
        workspace_ant_tools=[{"name": "no_such_tool", "input_schema": {"type": "object"}}],
        workspace_oai_tools=[
            {"type": "function", "function": {"name": "no_such_tool", "parameters": {"type": "object"}}}
        ],
        **kwargs,
    )


class TestProxyToolLoopReleases:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("context_role", ["agent", "chat_master"])
    async def test_anthropic_loop_every_request_without_tx(self, context_role):
        db = TxDb()
        db.tx = True
        ctx = _ant_client(db, [_TOOL_USE, _END_TURN])
        with (
            patch("app.core.proxy_session.httpx.AsyncClient", return_value=ctx),
            patch("app.core.proxy_session.emit_event", side_effect=_dirty_emit(db)),
            patch("app.core.proxy_session.emit_status", side_effect=_dirty_emit(db)),
        ):
            result = await _proxy_loop(
                db, {"provider": "anthropic", "model": "m", "api_key": "k"},
                context_role=context_role,
            )
        assert result[0] == "done"
        assert db.at_model_call == [False, False]

    @pytest.mark.asyncio
    async def test_anthropic_forced_close_without_tx(self):
        db = TxDb()
        db.tx = True
        ctx = _ant_client(db, [_TOOL_USE, _CLOSE])
        with (
            patch("app.core.proxy_session.httpx.AsyncClient", return_value=ctx),
            patch("app.core.proxy_session.emit_event", side_effect=_dirty_emit(db)),
            patch("app.core.proxy_session.emit_status", side_effect=_dirty_emit(db)),
        ):
            await _proxy_loop(
                db, {"provider": "anthropic", "model": "m", "api_key": "k"},
                max_tool_calls=1, on_exhaust="force_close",
            )
        assert len(db.at_model_call) == 2
        assert db.at_model_call == [False, False]

    @pytest.mark.asyncio
    async def test_anthropic_forced_answer_without_tx(self):
        db = TxDb()
        db.tx = True
        ctx = _ant_client(db, [_TOOL_USE, _END_TURN])
        with (
            patch("app.core.proxy_session.httpx.AsyncClient", return_value=ctx),
            patch("app.core.proxy_session.emit_event", side_effect=_dirty_emit(db)),
            patch("app.core.proxy_session.emit_status", side_effect=_dirty_emit(db)),
        ):
            await _proxy_loop(
                db, {"provider": "anthropic", "model": "m", "api_key": "k"},
                max_tool_calls=1, on_exhaust="answer",
            )
        assert db.at_model_call == [False, False]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_exhaust", ["answer", "force_close"])
    async def test_openai_loop_and_forced_exit_without_tx(self, on_exhaust):
        db = TxDb()
        db.tx = True
        replies = iter([
            _oai_resp(None, [_oai_tool_call("no_such_tool")]),
            _oai_resp("done")
            if on_exhaust == "answer"
            else _oai_resp(None, [_oai_tool_call("close_session", '{"summary": "s"}')]),
        ])

        async def create(**kwargs):
            db.model_called()
            return next(replies)

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=create)
        with (
            patch("app.core.proxy_session.AsyncOpenAI", return_value=client),
            patch("app.core.proxy_session.emit_event", side_effect=_dirty_emit(db)),
            patch("app.core.proxy_session.emit_status", side_effect=_dirty_emit(db)),
        ):
            await _proxy_loop(
                db, {"provider": "openai", "model": "m", "api_key": "k"},
                max_tool_calls=1, on_exhaust=on_exhaust,
            )
        assert db.at_model_call == [False, False]


class TestUserSessionReleases:
    @pytest.mark.asyncio
    async def test_single_turn_call_commits_the_callers_session(self):
        from app.core.user_sessions import _call_llm_with_tools

        db = TxDb()
        db.tx = True

        async def create(**kwargs):
            db.model_called()
            return _oai_resp("reply")

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=create)
        with patch("app.core.user_sessions.AsyncOpenAI", return_value=client):
            result = await _call_llm_with_tools(
                {"provider": "openai", "model": "m", "api_key": "k"},
                "SYS",
                [{"role": "user", "content": "hi"}],
                matrix_key="user_cm_p2",
                db=db,
            )
        assert result.text == "reply"
        assert db.at_model_call == [False]


# ---------------------------------------------------------------------------
# 1.2.4 closed while the model was answering
# ---------------------------------------------------------------------------


def _proxy_session(side: str) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_agent_proxy"
    s.side = side
    s.tokens_in = 0
    s.tokens_out = 0
    s.reasoning_tokens = 0
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.context_snapshot = None
    s.agent_id = uuid.uuid4()
    s.task_id = uuid.uuid4()
    s.goal_id = uuid.uuid4()
    s.peer_session_id = None
    s.started_at = datetime.now(timezone.utc)
    s.summary = None
    s.pending_actions = None
    s.last_activity_at = None
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = 0
    return s


def _proxy_task() -> MagicMock:
    from app.models.task import DispatchStatus, TaskState

    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "t"
    t.description = "d"
    t.state = TaskState.IN_PROGRESS
    t.assigned_agent_id = uuid.uuid4()
    t.goal_id = uuid.uuid4()
    t.created_at = datetime.now(timezone.utc)
    t.updated_at = datetime.now(timezone.utc)
    t.dispatch_status = DispatchStatus.READY
    t.stall_streak = 0
    return t


class TestClosedDuringModelWait:
    """The watchdog closes the pair from another DB session while the model is
    answering; the loop must re-read, discard the reply and stop cleanly."""

    async def _run(self, close_on_call: int):
        from app.core.proxy_session import _run_proxy_loop

        agent_session, cm_session = _proxy_session("agent"), _proxy_session("chat_master")
        task = _proxy_task()
        db = TxDb()
        closed = {"v": False}
        # The real _pair_closed_externally reads `.first()` of a fresh select.
        db._result.first.side_effect = lambda: (agent_session.id,) if closed["v"] else None

        replies = [
            ("Agent reply", None, 50, 100, "m", 50, 50),
            (None, {"summary": "ok"}, 40, 80, "m", 40, 40),
        ]
        calls = {"n": 0}

        async def model(coro, *args, **kwargs):
            coro.close()
            calls["n"] += 1
            if calls["n"] == close_on_call:
                closed["v"] = True  # the watchdog closed the pair meanwhile
            return replies[calls["n"] - 1]

        broker = AsyncMock()
        with (
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock,
                  return_value={"provider": "anthropic", "model": "m", "api_key": "x"}),
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock),
            patch("app.core.proxy_session.build_system_prompt", return_value="sys"),
            patch("app.core.workspace_tools.workspace_tools_available",
                  new_callable=AsyncMock, return_value=False),
            patch("app.core.proxy_session._call_with_heartbeat", side_effect=model),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session.emit_status", new_callable=AsyncMock),
            patch("app.core.proxy_session.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker", return_value=broker),
            patch("app.core.proxy_session._close_side", new_callable=AsyncMock,
                  return_value=MagicMock(summary="done", files_changed=[], output=None)) as close_side,
            patch("app.core.proxy_session._emit_closed", new_callable=AsyncMock),
            patch("app.core.proxy_session._render_close_note", return_value="[done]"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock) as apply_actions,
        ):
            out = await _run_proxy_loop(
                agent_session=agent_session,
                cm_session=cm_session,
                task=task,
                task_data_text="do",
                room_id=uuid.uuid4(),
                db=db,
            )
        return out, agent_session, cm_session, broker, close_side, apply_actions

    @pytest.mark.asyncio
    async def test_agent_reply_is_discarded(self):
        out, agent_session, _cm, broker, close_side, _apply = await self._run(close_on_call=1)
        assert out is True
        assert agent_session.tokens_in == 0 and agent_session.tokens_out == 0
        broker.publish.assert_not_awaited()
        close_side.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chat_master_verdict_is_discarded(self):
        out, _agent, cm_session, _broker, close_side, apply_actions = await self._run(close_on_call=2)
        assert out is True
        assert cm_session.tokens_in == 0 and cm_session.tokens_out == 0
        close_side.assert_not_awaited()
        apply_actions.assert_not_awaited()


# ---------------------------------------------------------------------------
# 1.2.5 lock order task -> session
# ---------------------------------------------------------------------------


class TestFlushTaskFirst:
    @pytest.mark.asyncio
    async def test_task_is_flushed_on_its_own_first(self):
        from app.core.proxy_session import _flush_task_first

        db = MagicMock()
        db.flush = AsyncMock()
        task = object()
        await _flush_task_first(db, task)
        assert [c.args for c in db.flush.await_args_list] == [([task],), ()]


DB_URL = os.environ.get("DATABASE_URL", "")


@pytest.mark.db
@pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["flush_task_first", "plain_commit"])
async def test_update_task_precedes_update_session_on_postgres(mode):
    """Real unit of work, real SQL order.  Everything happens inside one
    transaction that is rolled back — nothing is left in the database.

    ``flush_task_first`` — the proxy loop's explicit helper.
    ``plain_commit`` — any other code path: the session-wide ``before_commit``
    hook must order it too (A3: ``checkpoint_workspace`` deadlocked against
    the proxy loop by committing session before task)."""
    from sqlalchemy import event, text
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from app.core.proxy_session import _flush_task_first
    from app.models.agent import Agent
    from app.models.room import Room
    from app.models.session import CmAgentProxySession
    from app.models.task import Task
    from app.models.user import User

    engine = create_async_engine(DB_URL)
    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(" ".join(statement.split()[:2]))

    user_id, room_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    try:
        async with engine.connect() as conn:
            trans = await conn.begin()
            db = AsyncSession(
                bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint"
            )
            # ORM rows, not raw SQL: model defaults fill whatever columns a
            # fresh create_all schema (load stack) or a migrated one (dev) needs.
            db.add(User(id=user_id, email=f"{user_id}@order.test", hashed_password="x"))
            await db.flush()
            db.add(Room(id=room_id, name="order-test", creator_id=user_id))
            db.add(Agent(id=agent_id, nickname="a", agent_uid=f"order-{agent_id}",
                         agent_type="internal", owner_id=user_id, token=f"tok-{agent_id}"))
            await db.flush()
            task = Task(room_id=room_id, title="order")
            db.add(task)
            await db.flush()
            session = CmAgentProxySession(
                id=uuid.uuid4(), room_id=room_id, kind="cm_agent_proxy",
                side="agent", task_id=task.id, agent_id=agent_id,
            )
            db.add(session)
            await db.flush()

            statements.clear()
            session.last_activity_at = datetime.now(timezone.utc)
            task.updated_at = datetime.utcnow()
            db.add(session)
            db.add(task)
            if mode == "flush_task_first":
                await _flush_task_first(db, task)
            else:
                await db.commit()

            updates = [s for s in statements if s.startswith("UPDATE")]
            assert updates, statements
            assert updates[0] == "UPDATE task", updates
            assert "UPDATE session" in updates
            await trans.rollback()
            await db.close()
    finally:
        await engine.dispose()


@pytest.mark.db
@pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")
@pytest.mark.asyncio
async def test_autoflushed_session_does_not_deadlock_a_closer_on_postgres():
    """The repeated A3 deadlock, replayed on two real connections.

    The proxy loop's transaction dirties ``session.context_snapshot`` (what
    ``record_context_measurement`` does after every model reply), a SELECT
    autoflushes it, then the turn writes ``task``.  Meanwhile a closer (a
    workspace pause) has locked the task and closes that session.  Without the
    ``before_flush`` task lock the autoflush takes the session row first and
    Postgres reports a deadlock.  The rows are committed (two connections must
    see them) and deleted at the end.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

    from app.models.agent import Agent
    from app.models.room import Room
    from app.models.session import CmAgentProxySession
    from app.models.session.base import Session as SessionRow
    from app.models.task import Task
    from app.models.user import User

    engine = create_async_engine(DB_URL)
    user_id, room_id, agent_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    task_id, session_id = uuid.uuid4(), uuid.uuid4()
    try:
        async with AsyncSession(engine, expire_on_commit=False) as setup:
            setup.add(User(id=user_id, email=f"{user_id}@order.test", hashed_password="x"))
            await setup.flush()
            setup.add(Room(id=room_id, name="order-test", creator_id=user_id))
            setup.add(Agent(id=agent_id, nickname="a", agent_uid=f"order-{agent_id}",
                            agent_type="internal", owner_id=user_id, token=f"tok-{agent_id}"))
            await setup.flush()
            setup.add(Task(id=task_id, room_id=room_id, title="order"))
            await setup.flush()
            setup.add(CmAgentProxySession(
                id=session_id, room_id=room_id, kind="cm_agent_proxy",
                side="chat_master", task_id=task_id, agent_id=agent_id,
            ))
            await setup.commit()

        closer = AsyncSession(engine, expire_on_commit=False)
        loop_db = AsyncSession(engine, expire_on_commit=False)
        try:
            # Closer: task row first, as every close path does.
            c_task = await closer.get(Task, task_id)
            c_task.updated_at = datetime.utcnow()
            await closer.flush()

            l_session = await loop_db.get(SessionRow, session_id)
            l_task = await loop_db.get(Task, task_id)
            await loop_db.commit()  # release_db before the model call

            async def proxy_turn() -> None:
                l_session.context_snapshot = {"context_trace": [{"turn": 1}]}
                loop_db.add(l_session)
                # _pair_closed_externally: a SELECT that autoflushes.
                await loop_db.execute(select(SessionRow.id).where(SessionRow.id == session_id))
                l_task.updated_at = datetime.utcnow()
                loop_db.add(l_task)
                await loop_db.flush()
                await loop_db.commit()

            turn = asyncio.create_task(proxy_turn())
            await asyncio.sleep(0.5)  # let the autoflush reach Postgres

            c_session = await closer.get(SessionRow, session_id)
            c_session.closed_at = datetime.now(timezone.utc)
            c_session.close_reason = "workspace_paused"
            await closer.flush()
            await closer.commit()

            await asyncio.wait_for(turn, timeout=15)
        finally:
            await closer.close()
            await loop_db.close()
    finally:
        async with AsyncSession(engine) as cleanup:
            for model, row_id in (
                (SessionRow, session_id), (Task, task_id), (Agent, agent_id),
                (Room, room_id), (User, user_id),
            ):
                row = await cleanup.get(model, row_id)
                if row is not None:
                    await cleanup.delete(row)
                    await cleanup.flush()
            await cleanup.commit()
        await engine.dispose()
