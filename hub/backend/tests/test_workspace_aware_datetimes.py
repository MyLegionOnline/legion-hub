"""Regression tests: workspace timestamps must be written timezone-aware UTC.

Workspace/WorkspaceUsage columns are TIMESTAMPTZ (models declare
DateTime(timezone=True) with aware-UTC defaults), and asyncpg rejects
naive datetimes for such columns with
"can't subtract offset-naive and offset-aware datetimes".
SQLite-based tests don't catch this, so these tests assert the invariant
directly on the objects the writers produce.
"""

from __future__ import annotations

import uuid
from datetime import timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.models.workspace import Workspace, WorkspaceUsage


def _mock_db(**overrides):
    db = MagicMock()
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.execute = AsyncMock()
    for k, v in overrides.items():
        setattr(db, k, v)
    return db


def _result_with(scalar=None, scalars_all=None):
    res = MagicMock()
    res.scalar_one_or_none.return_value = scalar
    res.scalars.return_value.all.return_value = scalars_all or []
    res.all.return_value = []
    return res


def _reconciler_execute_side_effect(workspace, usage):
    """Return realistic results for the reconciler's current query flow.

    The first two queries are the objects under test.  Process recovery,
    lifecycle maintenance, and room cleanup issue additional queries; those
    are empty in these datetime-focused tests and must not exhaust an
    order-sensitive AsyncMock side effect list.
    """
    initial_results = iter([
        _result_with(scalars_all=[workspace]),
        _result_with(scalar=usage),
    ])

    async def execute(_statement):
        try:
            return next(initial_results)
        except StopIteration:
            return _result_with()

    return execute


def _assert_aware_utc(dt):
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0


def test_model_defaults_are_aware_utc():
    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    _assert_aware_utc(ws.created_at)
    _assert_aware_utc(usage.started_at)


@pytest.fixture
def idle_room(monkeypatch):
    """A room without live work.

    _close_workspace_usage now first asks room_has_live_work and, when
    there is live work, leaves the interval alone (billing outlives the container). The tests
    below check how time is recorded, so the policy is fixed explicitly. The same
    goes for the reconciler: a vanished container closes the interval only when there is no work.
    """
    from unittest.mock import AsyncMock

    monkeypatch.setattr(
        "app.core.workspace_lifecycle.room_has_live_work", AsyncMock(return_value=False)
    )


@pytest.mark.asyncio
async def test_close_usage_writes_aware_stopped_at(idle_room):
    """The ordinary stop path, which owns almost every closed interval.

    This used to exercise ``api/v1/workspace._close_usage`` — a leftover that
    no caller had reached since the lifecycle moved into
    ``core/workspace_lifecycle``. Testing it kept the real closer untested and
    hid the fact that it never fed ``workspace_session_hours_total``.
    """
    from app.core.workspace_lifecycle import _close_workspace_usage

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)

    db = _mock_db()
    db.execute.return_value = _result_with(scalars_all=[usage])

    await _close_workspace_usage(ws, db, stop_reason="user")

    assert usage.stop_reason == "user"
    _assert_aware_utc(usage.stopped_at)


@pytest.mark.asyncio
async def test_close_usage_records_session_hours(idle_room):
    """Closing an interval must move the counter the dashboards read.

    Regression for the gap measured 2026-09-23: 808.7 billed hours in
    ``workspace_usage`` against a ``workspace_session_hours_total`` of ~0,
    because only the reconciler ever fed the counter.
    """
    from datetime import datetime, timedelta

    from app.core import metrics
    from app.core.workspace_lifecycle import _close_workspace_usage

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
    # Used up to the stop, so the whole interval is billed (an interval with
    # no evidence of use is closed back-dated — see test_billing_interval_unit).
    ws.last_activity_at = datetime.now(timezone.utc)

    # The counter object itself is stubbed in this suite, so assert on the
    # recorded value rather than on the Prometheus internals.
    recorded: list[float] = []
    original = metrics.record_workspace_session_hours
    metrics.record_workspace_session_hours = recorded.append
    try:
        db = _mock_db()
        db.execute.return_value = _result_with(scalars_all=[usage])
        await _close_workspace_usage(ws, db, stop_reason="user")
    finally:
        metrics.record_workspace_session_hours = original

    assert recorded == [pytest.approx(2.0, abs=0.01)]


@pytest.mark.asyncio
async def test_close_usage_accepts_naive_started_at(idle_room):
    """Rows written before the column became aware must not raise here."""
    from datetime import datetime, timedelta

    from app.core.workspace_lifecycle import _close_workspace_usage

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4())
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)
    usage.started_at = datetime.utcnow() - timedelta(hours=1)  # naive, legacy row

    db = _mock_db()
    db.execute.return_value = _result_with(scalars_all=[usage])

    await _close_workspace_usage(ws, db, stop_reason="idle")

    assert usage.stop_reason == "idle"
    _assert_aware_utc(usage.stopped_at)


@pytest.mark.asyncio
async def test_reconciler_writes_aware_stopped_at(monkeypatch, idle_room):
    from app.core import workspace_reconciler
    from app.core import workspaced_client

    monkeypatch.setattr(
        workspaced_client,
        "get_workspace_status",
        AsyncMock(side_effect=httpx.RequestError("workspaced down")),
    )
    monkeypatch.setattr(workspaced_client, "list_workspace_ids", AsyncMock(return_value=[]))

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4(), status="running")
    usage = WorkspaceUsage(id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024)

    db = _mock_db()
    db.execute.side_effect = _reconciler_execute_side_effect(ws, usage)

    await workspace_reconciler._reconcile_tick(db)

    assert ws.status == "stopped"
    assert usage.stop_reason == "reconciler"
    _assert_aware_utc(usage.stopped_at)


@pytest.mark.asyncio
async def test_reconciler_tolerates_legacy_naive_started_at(monkeypatch, idle_room):
    """Rows written before the TIMESTAMPTZ migration may still surface naive
    (e.g. via SQLite); duration math must not crash on them."""
    from datetime import datetime

    from app.core import workspace_reconciler
    from app.core import workspaced_client

    monkeypatch.setattr(
        workspaced_client,
        "get_workspace_status",
        AsyncMock(side_effect=httpx.RequestError("workspaced down")),
    )
    monkeypatch.setattr(workspaced_client, "list_workspace_ids", AsyncMock(return_value=[]))

    ws = Workspace(id=uuid.uuid4(), room_id=uuid.uuid4(), status="running")
    usage = WorkspaceUsage(
        id=uuid.uuid4(), workspace_id=ws.id, mem_limit_mb=1024,
        started_at=datetime.utcnow(),  # naive legacy value
    )
    assert usage.started_at.tzinfo is None

    db = _mock_db()
    db.execute.side_effect = _reconciler_execute_side_effect(ws, usage)

    await workspace_reconciler._reconcile_tick(db)

    _assert_aware_utc(usage.stopped_at)
