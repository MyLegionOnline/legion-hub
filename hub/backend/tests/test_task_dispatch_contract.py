"""Regression tests for internal design notes, part 3, stage 6.

Every earlier stage of part 3 (single dispatch owner, idempotent claim,
runtime-aware Start) is exercised indirectly elsewhere in the suite, but
always with `request_task_dispatch`/`prepare_task_dispatch` mocked away by
the caller under test — the actual claim/idempotency mechanics these stages
introduced had no direct coverage. This file targets exactly that gap:

- `request_task_dispatch`'s own "already active" gate (dispatch_status OR an
  open session) — the fix for "TASK_UNBLOCKED + concurrent manual Start
  create two proxy pairs".
- `handle_task_created`'s atomic claim: the losing side of a concurrent
  READY→DISPATCHED race must create nothing.
- `_emit_dispatch_failed` resetting dispatch_status back to READY (and
  demoting a stuck IN_PROGRESS to BLOCKED) so a failed claim is retryable
  instead of stranded as "assigned + dispatched" forever — and that a task
  recovered this way is genuinely re-dispatchable again.
- The idempotent manual `Start` branch in `change_task_state` added in
  stage 4 (in_progress → in_progress no-op, gated on `run_id` to avoid ever
  touching the legacy `approve_and_start` pipeline).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.tasks import change_task_state
from app.core.proxy_session import (
    TaskDispatchContext,
    TaskDispatchPreflightError,
    _emit_dispatch_failed,
    handle_task_created,
    request_task_dispatch,
)
from app.models.task import DispatchStatus, TaskState
from app.schemas.task import TaskStateChange

NOW = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_session_result(found: bool) -> MagicMock:
    """Result for request_task_dispatch's open-CmAgentProxySession lookup."""
    r = MagicMock()
    r.first.return_value = (uuid.uuid4(),) if found else None
    return r


def _make_task(**overrides) -> SimpleNamespace:
    base = dict(
        id=uuid.uuid4(),
        room_id=uuid.uuid4(),
        goal_id=None,
        title="Deploy service",
        description="",
        state=TaskState.ASSIGNED,
        dispatch_status=DispatchStatus.READY,
        retry_count=0,
        assigned_agent_id=uuid.uuid4(),
        created_by_agent_id=None,
        created_at=NOW,
        updated_at=NOW,
        run_id=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _result(all_rows=None, first_row=None, row=None) -> MagicMock:
    m = MagicMock()
    m.scalars.return_value.all.return_value = all_rows or []
    m.scalars.return_value.first.return_value = first_row
    m.first.return_value = row
    return m


def _make_db(results: list) -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock(side_effect=results)
    return db


def _endpoint_db(task) -> AsyncMock:
    """db.execute sequence for change_task_state's happy path: membership,
    _get_task, then _enrich_task's agent/sessions/[dispatch_started if
    run_id is set]/last-dispatch-failure/last-interruption."""
    agent = MagicMock()
    agent.nickname = "worker"
    results = [
        _result(first_row=object()),   # _verify_membership
        _result(first_row=task),       # _get_task
    ]
    if task.state == TaskState.CONTINUED:
        # Stopping a handed-off task walks its continuation chain first; one
        # empty batch ends the walk.
        results.append(_result(all_rows=[]))
    results += [
        _result(first_row=agent),      # _enrich_task: agent
        _result(all_rows=[]),          # _enrich_task: sessions
        _result(row=(1,)),             # _enrich_task: assignee still in the room
    ]
    if task.run_id is not None:
        results.append(_result(row=None))  # _compute_task_runtime: dispatch_started lookup
    results.append(_result(row=None))      # _compute_task_runtime: last dispatch failure
    results.append(_result(row=None))      # _compute_task_runtime: last interruption
    return _make_db(results)


# ---------------------------------------------------------------------------
# request_task_dispatch: the idempotency gate itself
# ---------------------------------------------------------------------------


class TestRequestTaskDispatchIdempotency:
    @pytest.mark.asyncio
    async def test_already_active_via_dispatch_status_skips_dispatch(self):
        """A task already claimed (DISPATCHED) must not be re-dispatched — the
        exact race the plan calls out: TASK_UNBLOCKED auto-dispatch racing a
        concurrent manual Start, both reaching request_task_dispatch."""
        task = _make_task(dispatch_status=DispatchStatus.DISPATCHED)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_open_session_result(False))

        with (
            patch("app.core.proxy_session.prepare_task_dispatch", new_callable=AsyncMock) as mock_prepare,
            patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch_event,
        ):
            outcome = await request_task_dispatch(task, task.room_id, db, source="manual_start")

        assert outcome == "already_active"
        mock_prepare.assert_not_awaited()
        mock_dispatch_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_already_active_via_open_session_even_if_dispatch_status_ready(self):
        """dispatch_status can already read READY while a session is still open
        (e.g. right after Retry reset it) — the open-session lookup, not
        dispatch_status alone, is what actually blocks a second pair."""
        task = _make_task(dispatch_status=DispatchStatus.READY)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_open_session_result(True))

        with (
            patch("app.core.proxy_session.prepare_task_dispatch", new_callable=AsyncMock) as mock_prepare,
            patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch_event,
        ):
            outcome = await request_task_dispatch(task, task.room_id, db, source="task_unblocked")

        assert outcome == "already_active"
        mock_prepare.assert_not_awaited()
        mock_dispatch_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_inactive_ready_task_is_dispatched_exactly_once(self):
        task = _make_task(dispatch_status=DispatchStatus.READY)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_open_session_result(False))
        ctx = TaskDispatchContext(None, None, MagicMock(), uuid.uuid4(), False)

        with (
            patch("app.core.proxy_session.prepare_task_dispatch", new_callable=AsyncMock,
                  return_value=ctx) as mock_prepare,
            patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch_event,
        ):
            outcome = await request_task_dispatch(task, task.room_id, db, source="task_unblocked")

        assert outcome == "started"
        mock_prepare.assert_awaited_once()
        mock_dispatch_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_preflight_failure_propagates_without_dispatching(self):
        """Missing agent / ChatMaster / a non-dispatchable task.state must
        surface as an error the caller turns into 422 or a log line — never
        silently dispatch anyway."""
        task = _make_task(dispatch_status=DispatchStatus.READY)
        db = AsyncMock()
        db.execute = AsyncMock(return_value=_open_session_result(False))

        with (
            patch("app.core.proxy_session.prepare_task_dispatch", new_callable=AsyncMock,
                  side_effect=TaskDispatchPreflightError("no_assigned_agent", "no executor is assigned")),
            patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch_event,
        ):
            with pytest.raises(TaskDispatchPreflightError):
                await request_task_dispatch(task, task.room_id, db, source="manual_start")

        mock_dispatch_event.assert_not_called()


# ---------------------------------------------------------------------------
# handle_task_created: the atomic claim has exactly one winner
# ---------------------------------------------------------------------------


class TestHandleTaskCreatedClaimHasOneWinner:
    @pytest.mark.asyncio
    async def test_losing_claim_creates_no_session_pair(self):
        """Two dispatchers racing the same READY task: only the transaction
        whose conditional UPDATE actually matches a row may proceed. The loser
        must roll back and create nothing — the sole guard behind "one and
        only one dispatcher claim, one proxy pair" from the target contract."""
        task_id, room_id, goal_id, plan_id = (uuid.uuid4() for _ in range(4))
        task = _make_task(id=task_id, dispatch_status=DispatchStatus.READY)

        task_result = MagicMock()
        task_result.scalars.return_value.first.return_value = task
        lost_claim = MagicMock()
        lost_claim.rowcount = 0

        db = AsyncMock()
        db.rollback = AsyncMock()
        db.execute = AsyncMock(side_effect=[task_result, lost_claim])

        with patch("app.core.proxy_session.SessionRepository") as MockRepo:
            await handle_task_created(task_id, room_id, goal_id, plan_id, db)

        db.rollback.assert_awaited_once()
        MockRepo.return_value.create.assert_not_called()
        # The loser's in-memory task must not read as claimed — only the
        # winning transaction's own UPDATE actually flipped it to DISPATCHED.
        assert task.dispatch_status == DispatchStatus.READY


# ---------------------------------------------------------------------------
# _emit_dispatch_failed: a failed claim must be retryable, not stranded
# ---------------------------------------------------------------------------


class TestEmitDispatchFailedRecovery:
    @pytest.mark.asyncio
    async def test_resets_dispatch_status_and_demotes_stuck_in_progress(self):
        """A claimed-but-never-executed task (missing ChatMaster, invalid
        config, room deleted mid-flight) must not be left as
        in_progress + dispatched forever — that combination is
        indistinguishable from a genuinely running session and would make
        request_task_dispatch's idempotency check treat it as already
        active."""
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED)
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock()

        with patch("app.core.proxy_session.emit_event", new_callable=AsyncMock) as mock_emit:
            await _emit_dispatch_failed(task, task.room_id, "no_chat_master", db, run_id="run-1")

        assert task.dispatch_status == DispatchStatus.READY
        assert task.state == TaskState.BLOCKED
        db.commit.assert_awaited_once()
        mock_emit.assert_awaited_once()
        kwargs = mock_emit.await_args.kwargs
        assert kwargs["event_type"] == "TASK_DISPATCH_FAILED"
        assert kwargs["metadata"]["reason"] == "no_chat_master"

    @pytest.mark.asyncio
    async def test_leaves_non_in_progress_state_untouched(self):
        """An assigned (not yet in_progress) task hitting a preflight failure
        should not be force-promoted to blocked — only a stuck in_progress
        needs demoting."""
        task = _make_task(state=TaskState.ASSIGNED, dispatch_status=DispatchStatus.DISPATCHED)
        db = AsyncMock()
        db.add = MagicMock()
        db.commit = AsyncMock()

        with patch("app.core.proxy_session.emit_event", new_callable=AsyncMock):
            await _emit_dispatch_failed(task, task.room_id, "room_not_found", db, run_id="run-1")

        assert task.dispatch_status == DispatchStatus.READY
        assert task.state == TaskState.ASSIGNED

    @pytest.mark.asyncio
    async def test_task_recovers_to_redispatchable_after_failure(self):
        """End-to-end of the stage-2 bug fix: after _emit_dispatch_failed
        resets dispatch_status, request_task_dispatch's own idempotency check
        must see the task as inactive again, not permanently 'active'."""
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED)
        fail_db = AsyncMock()
        fail_db.add = MagicMock()
        fail_db.commit = AsyncMock()
        with patch("app.core.proxy_session.emit_event", new_callable=AsyncMock):
            await _emit_dispatch_failed(task, task.room_id, "no_chat_master", fail_db, run_id="run-1")
        assert task.dispatch_status == DispatchStatus.READY

        retry_db = AsyncMock()
        retry_db.execute = AsyncMock(return_value=_open_session_result(False))
        ctx = TaskDispatchContext(None, None, MagicMock(), uuid.uuid4(), False)
        with (
            patch("app.core.proxy_session.prepare_task_dispatch", new_callable=AsyncMock, return_value=ctx),
            patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch_event,
        ):
            outcome = await request_task_dispatch(task, task.room_id, retry_db, source="retry")

        assert outcome == "started"
        mock_dispatch_event.assert_called_once()


# ---------------------------------------------------------------------------
# change_task_state: idempotent manual Start (stage 4)
# ---------------------------------------------------------------------------


class TestIdempotentManualStart:
    @pytest.mark.asyncio
    async def test_already_running_modern_task_is_a_dispatcher_no_op(self):
        """A repeated Start PATCH for a task the modern dispatcher already
        claimed (run_id set) must not 422 and must not attempt a second
        execution — it routes through request_task_dispatch's own
        already-active guard instead."""
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.DISPATCHED,
                           run_id="run-abc")
        db = _endpoint_db(task)

        with patch("app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock,
                    return_value="already_active") as mock_dispatch:
            out = await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.IN_PROGRESS), db, MagicMock(),
            )

        mock_dispatch.assert_awaited_once_with(task, task.room_id, db, source="manual_start_idempotent")
        assert task.state == TaskState.IN_PROGRESS
        assert out.state == TaskState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_legacy_task_without_run_id_is_not_redispatched(self):
        """A task stuck at in_progress via the legacy approve_and_start
        pipeline has no run_id — the modern dispatcher must never be invoked
        for it, or a second execution would start alongside the still-running
        legacy loop."""
        task = _make_task(state=TaskState.IN_PROGRESS, dispatch_status=DispatchStatus.READY,
                           run_id=None)
        db = _endpoint_db(task)

        with patch("app.api.v1.tasks.request_task_dispatch", new_callable=AsyncMock) as mock_dispatch:
            out = await change_task_state(
                task.room_id, task.id, TaskStateChange(state=TaskState.IN_PROGRESS), db, MagicMock(),
            )

        mock_dispatch.assert_not_awaited()
        assert out.state == TaskState.IN_PROGRESS


# ---------------------------------------------------------------------------
# _compute_task_runtime: last_interruption (internal design notes, part 4,
# stage 2) — why the last execution attempt *stopped*, as opposed to
# dispatch_reason above (why an attempt never *started*).
# ---------------------------------------------------------------------------


class TestLastInterruption:
    @pytest.mark.asyncio
    async def test_surfaces_reason_and_fields_from_the_event(self):
        from app.api.v1.tasks import _compute_task_runtime

        task = _make_task(state=TaskState.BLOCKED, dispatch_status=DispatchStatus.READY, run_id=None)
        meta = {
            "reason": "repeated_tool_error",
            "recommended_action": "resume_with_correction",
            "tool_calls_used": 32,
            "tool_calls_limit": 200,
            "trigger_tool": "workspace_service",
            "repeat_count": 3,
            "safe_error": "Error: invalid cwd",
        }
        db = _make_db([
            _result(row=None),         # last dispatch failure
            _result(row=(NOW, meta)),  # last interruption
        ])

        runtime = await _compute_task_runtime(task, [], db)

        interruption = runtime["last_interruption"]
        assert interruption["reason"] == "repeated_tool_error"
        assert interruption["trigger_tool"] == "workspace_service"
        assert interruption["repeat_count"] == 3
        assert interruption["safe_error"] == "Error: invalid cwd"
        assert interruption["recommended_action"] == "resume_with_correction"
        assert interruption["superseded"] is False
        # BLOCKED wins the runtime_status race regardless of dispatch_status —
        # the same "needs_retry" signal the panels already key Retry off.
        assert runtime["runtime_status"] == "needs_retry"

    @pytest.mark.asyncio
    async def test_superseded_when_a_newer_dispatch_started_after_it(self):
        """An interruption older than the current run's own dispatch_started_at
        is history, not the current blocker — same staleness rule already used
        for dispatch_reason (see the query right above this one)."""
        from datetime import timedelta

        from app.api.v1.tasks import _compute_task_runtime

        task = _make_task(state=TaskState.ASSIGNED, dispatch_status=DispatchStatus.DISPATCHED, run_id="run-2")
        interruption_ts = NOW
        newer_dispatch_ts = NOW + timedelta(minutes=5)
        db = _make_db([
            _result(row=(newer_dispatch_ts, {"run_id": "run-2"})),        # dispatch_started lookup
            _result(row=None),                                            # last dispatch failure
            _result(row=(interruption_ts, {"reason": "token_budget"})),   # last interruption
        ])

        runtime = await _compute_task_runtime(task, [], db)

        assert runtime["last_interruption"]["reason"] == "token_budget"
        assert runtime["last_interruption"]["superseded"] is True

    @pytest.mark.asyncio
    async def test_no_interruption_event_returns_none(self):
        from app.api.v1.tasks import _compute_task_runtime

        task = _make_task(state=TaskState.COMPLETED, run_id=None)
        db = _make_db([
            _result(row=None),  # last dispatch failure
            _result(row=None),  # last interruption
        ])

        runtime = await _compute_task_runtime(task, [], db)
        assert runtime["last_interruption"] is None
