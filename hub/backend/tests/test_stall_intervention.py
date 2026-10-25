"""Mid-task dialogue must survive the ChatMaster's tool budget, and a proxy
generation must not be able to fence itself out of existence.

Two failures observed together on a live room (task "Check: cycling through
images…"), both of which made the loop guard look broken:

  1. The stall intervention asks the CM to inspect the workspace before naming a
     way forward. With a 5-call budget and on_exhaust="force_close" the CM ran
     out mid-inspection and was *forced* to call close_session — its answer
     arrived as a CHANGES_REQUESTED verdict, tearing the pair down, spending a
     review attempt and restarting the agent with every loop-guard counter
     reset. Covered here: the answer-turn budget/mode wiring and the mechanical
     gate that converts such a close into an answer.

  2. The outgoing run wrote runtime_state="stopped" without a run_id fence,
     racing the rework it had just scheduled. The replacement generation kept
     run_id=<new> with runtime_state="stopped" and died at turn 0, leaving the
     task IN_PROGRESS/DISPATCHED with two open sessions and no coroutine.
     Covered here: the claim writes the runtime state itself, the redispatch
     happens only after finalization, and a lost fence blocks the task instead
     of hanging.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.proxy_session import (
    _CHANGES_SENTINEL,
    _CM_ANSWER_MAX_TOOL_CALLS,
    _CM_CLOSE_CONVERTED_NOTE,
    _CM_MAX_TOOL_CALLS,
    _run_fence_self_orphaned,
    handle_task_created,
)
from app.models.task import DispatchStatus, TaskState


# ---------------------------------------------------------------------------
# Doubles (same shape as tests/test_request_changes.py)
# ---------------------------------------------------------------------------


def _make_task(assigned_agent_id: uuid.UUID, goal_id: uuid.UUID) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.room_id = uuid.uuid4()
    t.title = "Layout check"
    t.description = "Open the pages and check."
    t.state = TaskState.IN_PROGRESS
    t.assigned_agent_id = assigned_agent_id
    t.goal_id = goal_id
    t.created_at = datetime.now(timezone.utc)
    t.updated_at = datetime.now(timezone.utc)
    t.dispatch_status = DispatchStatus.READY
    t.rework_count = 0
    t.stall_streak = 0
    return t


def _make_session(side: str) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = "cm_agent_proxy"
    s.side = side
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.is_open = True
    s.context_snapshot = None
    s.agent_id = uuid.uuid4()
    s.task_id = uuid.uuid4()
    s.goal_id = uuid.uuid4()
    s.plan_id = uuid.uuid4()
    s.peer_session_id = None
    s.started_at = datetime.now(timezone.utc)
    s.summary = None
    s.pending_actions = None
    s.max_context_window_tokens = 0
    s.last_context_window_tokens = 0
    s.last_activity_at = None
    return s


def _make_room(room_id: uuid.UUID, master_agent_id: uuid.UUID) -> MagicMock:
    room = MagicMock()
    room.id = room_id
    room.coordinator_config = {"master_agent_id": str(master_agent_id)}
    return room


def _empty_result() -> MagicMock:
    r = MagicMock()
    r.scalars.return_value.first.return_value = None
    r.scalars.return_value.all.return_value = []
    r.first.return_value = None
    r.all.return_value = []
    return r


def _db_results(*results):
    yield from results
    while True:
        yield _empty_result()


def _setup_db(task: MagicMock, room: MagicMock) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()

    task_result = MagicMock()
    task_result.scalars.return_value.first.return_value = task
    claim_result = MagicMock()
    claim_result.rowcount = 1
    agent_result = MagicMock()
    agent_result.scalars.return_value.first.return_value = MagicMock(llm_config={})
    room_result = MagicMock()
    room_result.scalars.return_value.first.return_value = room

    db.execute = AsyncMock(
        side_effect=_db_results(task_result, claim_result, agent_result, room_result)
    )
    return db


def _capture_emit():
    captured: list[dict] = []

    async def _emit(room_id: Any, event_type: str, **kwargs: Any) -> MagicMock:
        captured.append({"event_type": event_type, **kwargs})
        return MagicMock()

    return captured, _emit


def _history(session: MagicMock) -> list[dict[str, Any]]:
    return (session.context_snapshot or {}).get("messages", [])


class _Ctx:
    """One dispatched pair, driven by a scripted provider boundary."""

    def __init__(self, llm_returns: list[Any], pair_closed: list[bool] | None = None):
        self.room_id = uuid.uuid4()
        self.goal_id = uuid.uuid4()
        self.plan_id = uuid.uuid4()
        self.agent_id = uuid.uuid4()
        self.cm_agent_id = uuid.uuid4()
        self.task = _make_task(self.agent_id, self.goal_id)
        self.task_id = self.task.id
        self.room = _make_room(self.room_id, self.cm_agent_id)
        self.agent_session = _make_session("agent")
        self.agent_session.agent_id = self.agent_id
        self.agent_session.room_id = self.room_id
        self.cm_session = _make_session("chat_master")
        self.cm_session.agent_id = self.cm_agent_id
        self.cm_session.room_id = self.room_id
        self.db = _setup_db(self.task, self.room)
        self.llm_returns = llm_returns
        self.pair_closed = pair_closed
        self.llm_calls: list[dict[str, Any]] = []
        self.events: list[dict] = []
        self.order: list[str] = []

    async def run(self, **kwargs: Any) -> None:
        captured, emit_side_effect = _capture_emit()
        self.events = captured
        returns = iter(self.llm_returns)

        async def spy_llm(llm_config, system_prompt, history, **kw):
            self.llm_calls.append({"history": list(history), **kw})
            return next(returns)

        pair_closed_patch = (
            patch(
                "app.core.proxy_session._pair_closed_externally",
                new_callable=AsyncMock,
                side_effect=self.pair_closed,
            )
            if self.pair_closed is not None
            else patch(
                "app.core.proxy_session._pair_closed_externally",
                new_callable=AsyncMock,
                return_value=False,
            )
        )

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            pair_closed_patch,
            patch("app.core.proxy_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session.build_system_prompt", return_value="SYS"),
            patch("app.core.proxy_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            patch("app.core.proxy_session._get_agent_llm_config", new_callable=AsyncMock, return_value={}),
            patch("app.core.proxy_session._build_review_verification",
                  new_callable=AsyncMock, return_value="[SYSTEM VERIFICATION] ok"),
            patch("app.core.proxy_session._call_llm_with_tools", side_effect=spy_llm),
            patch("app.core.proxy_session.track_llm_usage"),
            patch("app.core.proxy_session.apply_close_actions", new_callable=AsyncMock),
            patch("app.core.proxy_session.ProxyBroker") as MockBroker,
            patch("app.core.proxy_session.emit_event", side_effect=emit_side_effect),
            patch("app.core.proxy_session.request_task_dispatch",
                  new_callable=AsyncMock, return_value="started") as mock_dispatch,
            patch("app.core.task_runtime.finalize_run", new_callable=AsyncMock) as mock_finalize,
        ):
            repo_instance = AsyncMock()
            repo_instance.create = AsyncMock(
                side_effect=[self.agent_session, self.cm_session]
            )
            MockRepo.return_value = repo_instance
            MockBroker.return_value = AsyncMock()
            self.mock_dispatch = mock_dispatch
            self.mock_finalize = mock_finalize

            # Record the order of the two: a redispatch that lands before this
            # run finalizes is the race that wedged the replacement pair.
            async def _finalize_rec(*a: Any, **kw: Any) -> None:
                self.order.append("finalize")

            async def _dispatch_rec(*a: Any, **kw: Any) -> str:
                self.order.append("dispatch")
                return "started"

            mock_finalize.side_effect = _finalize_rec
            mock_dispatch.side_effect = _dispatch_rec

            await handle_task_created(
                self.task_id, self.room_id, self.goal_id, self.plan_id, self.db, **kwargs
            )

    def event_types(self) -> list[str]:
        return [e["event_type"] for e in self.events]


_MODEL = "claude-haiku-4-5-20251001"


def _agent_question(text: str = "Cannot find the files. What should I do?"):
    return (text, None, 100, 80, _MODEL, 100, 100)


def _agent_submit(summary: str = "Done."):
    return (None, {"summary": summary}, 100, 80, _MODEL, 100, 100)


def _cm_close(summary: str):
    return (None, {"summary": summary}, 120, 90, _MODEL, 120, 120)


def _cm_text(text: str):
    return (text, None, 120, 90, _MODEL, 120, 120)


# ---------------------------------------------------------------------------
# 1. The mechanical gate
# ---------------------------------------------------------------------------


class TestCloseSessionGate:
    @pytest.mark.asyncio
    async def test_close_on_a_question_becomes_an_answer(self):
        """A CM close while the agent is mid-task is delivered as an answer."""
        guidance = (
            f"{_CHANGES_SENTINEL} — start the server and open the pages id=1..4."
        )
        ctx = _Ctx(
            [_agent_question(), _cm_close(guidance)],
            # turn 0 head, after agent, after CM, then stop before turn 1
            pair_closed=[False, False, False, True],
        )
        await ctx.run()

        # No verdict was applied: the review budget is untouched and the task
        # never went back to the board for rework.
        assert ctx.task.rework_count == 0
        assert "TASK_CHANGES_REQUESTED" not in ctx.event_types()
        assert "TASK_REJECTED" not in ctx.event_types()
        assert ctx.task.state != TaskState.REJECTED
        ctx.mock_dispatch.assert_not_awaited()

        # Both sides stay open — the agent keeps working on the same task.
        assert ctx.agent_session.is_open
        assert ctx.cm_session.is_open

        # The CM's text reached the agent, marked as guidance and not a verdict.
        last_user = [m for m in _history(ctx.agent_session) if m["role"] == "user"][-1]
        assert guidance in last_user["content"]
        assert _CM_CLOSE_CONVERTED_NOTE.strip()[:40] in last_user["content"]

    @pytest.mark.asyncio
    async def test_close_after_a_submission_is_still_a_verdict(self):
        """The gate must not disarm review: a submitted task can be rejected."""
        ctx = _Ctx([_agent_submit(), _cm_close("Bad. TASK_REJECTED")])
        await ctx.run()

        assert "TASK_REJECTED" in ctx.event_types()
        assert ctx.task.state == TaskState.REJECTED


# ---------------------------------------------------------------------------
# 2. Answer-turn budget and exhaustion mode
# ---------------------------------------------------------------------------


class TestAnswerTurnBudget:
    @pytest.mark.asyncio
    async def test_answer_turn_gets_the_bigger_budget_and_cannot_be_force_closed(self):
        ctx = _Ctx(
            [_agent_question(), _cm_text("Open the page id=1 and look at the HTML.")],
            pair_closed=[False, False, False, True],
        )
        await ctx.run()

        cm_call = ctx.llm_calls[1]
        assert cm_call["context_role"] == "chat_master"
        assert cm_call["max_tool_calls"] == _CM_ANSWER_MAX_TOOL_CALLS
        assert _CM_ANSWER_MAX_TOOL_CALLS > _CM_MAX_TOOL_CALLS
        # Exhaustion must end in text, never in a forced close_session.
        assert cm_call["on_exhaust"] == "answer"

    @pytest.mark.asyncio
    async def test_review_turn_keeps_the_review_budget(self):
        ctx = _Ctx([_agent_submit(), _cm_close("Good.")])
        await ctx.run()

        cm_call = ctx.llm_calls[1]
        assert cm_call["max_tool_calls"] == _CM_MAX_TOOL_CALLS
        assert cm_call["on_exhaust"] == "force_close"


# ---------------------------------------------------------------------------
# 3. Runtime fence
# ---------------------------------------------------------------------------


class TestRuntimeFence:
    @pytest.mark.asyncio
    async def test_claim_writes_the_runtime_state_itself(self):
        """run_id and runtime_state are part of the atomic claim, not a later flush."""
        ctx = _Ctx([_agent_submit(), _cm_close("Good.")])
        await ctx.run()

        claim_stmt = str(ctx.db.execute.await_args_list[1].args[0])
        assert "UPDATE task SET" in claim_stmt
        assert "run_id" in claim_stmt
        assert "runtime_state" in claim_stmt
        assert "stop_reason" in claim_stmt

    @pytest.mark.asyncio
    async def test_rework_is_dispatched_only_after_finalization(self):
        """The successor starts after this generation released its fence.

        The reverse order is the live bug: the outgoing run stamped
        runtime_state="stopped" on a row the replacement already owned.
        """
        ctx = _Ctx([_agent_submit(), _cm_close(f"{_CHANGES_SENTINEL}: fix the tests")])
        await ctx.run()

        assert ctx.order == ["finalize", "dispatch"]
        assert ctx.mock_finalize.await_args.args[0] == ctx.task.id
        assert ctx.mock_dispatch.await_args.kwargs["rework_feedback"]
        assert ctx.mock_dispatch.await_args.kwargs["resume"] is True

    @pytest.mark.asyncio
    async def test_orphaned_fence_blocks_the_task_instead_of_hanging(self):
        """A pair that never spoke, with its own run still current, is not a supersede."""
        ctx = _Ctx([], pair_closed=[True])

        with (
            patch("app.core.proxy_session._run_fence_self_orphaned",
                  new_callable=AsyncMock, return_value=True),
            patch("app.core.watchdog.block_stalled_task", new_callable=AsyncMock) as mock_block,
        ):
            await ctx.run()

        mock_block.assert_awaited_once()
        assert mock_block.await_args.args[1] == "runtime_fence_lost"
        # The model was never called and both sides were closed, not left open.
        assert ctx.llm_calls == []
        assert ctx.agent_session.close_reason == "error"
        assert ctx.cm_session.close_reason == "error"


class TestRunFenceSelfOrphaned:
    """Only 'my own run, no longer running' counts as a lost fence."""

    @staticmethod
    def _db(row):
        db = AsyncMock()
        result = MagicMock()
        result.first.return_value = row
        db.execute = AsyncMock(return_value=result)
        return db

    @pytest.mark.asyncio
    async def test_own_run_stopped_is_orphaned(self):
        db = self._db(("run-1", "stopped"))
        assert await _run_fence_self_orphaned(uuid.uuid4(), "run-1", db) is True

    @pytest.mark.asyncio
    async def test_running_is_not_orphaned(self):
        db = self._db(("run-1", "running"))
        assert await _run_fence_self_orphaned(uuid.uuid4(), "run-1", db) is False

    @pytest.mark.asyncio
    async def test_user_stop_is_not_orphaned(self):
        for state in ("stop_requested", "paused"):
            db = self._db(("run-1", state))
            assert await _run_fence_self_orphaned(uuid.uuid4(), "run-1", db) is False

    @pytest.mark.asyncio
    async def test_superseded_by_another_run_is_not_orphaned(self):
        db = self._db(("run-2", "running"))
        assert await _run_fence_self_orphaned(uuid.uuid4(), "run-1", db) is False

    @pytest.mark.asyncio
    async def test_missing_row_is_not_orphaned(self):
        db = self._db(None)
        assert await _run_fence_self_orphaned(uuid.uuid4(), "run-1", db) is False


# ---------------------------------------------------------------------------
# 4. Durable context on rework
# ---------------------------------------------------------------------------


class TestReworkCarriesDurableContext:
    @pytest.mark.asyncio
    async def test_read_manifest_is_seeded_into_a_reworked_session(self):
        """Without it the respawned session re-reads what it already read."""
        ctx = _Ctx([_agent_submit(), _cm_close("Good.")])

        with (
            patch("app.core.proxy_session.load_paused_history", new_callable=AsyncMock,
                  return_value=[{"role": "user", "content": "TASK DATA"}]),
            patch("app.core.proxy_session._durable_context_notes", new_callable=AsyncMock,
                  return_value="[ALREADY READ] blog/models.py"),
        ):
            await ctx.run(resume=True, rework_feedback="fix the layout")

        seeded = ctx.llm_calls[0]["history"]
        text = "\n".join(
            m["content"] for m in seeded
            if m["role"] == "user" and isinstance(m["content"], str)
        )
        assert "fix the layout" in text
        assert "[ALREADY READ] blog/models.py" in text
