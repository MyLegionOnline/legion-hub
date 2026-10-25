"""A vanished container resets its services and says so (browser services, stage 5)."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core import metrics
from app.core.workspace_reconciler import _reset_orphaned_processes
from app.models.workspace import WorkspaceServiceEvent


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Db:
    """Answers the two queries of the reset: background procs, then managed ones."""

    def __init__(self, bg_rows, managed_rows):
        self._answers = [_Rows(bg_rows), _Rows(managed_rows)]
        self.added = []

    async def execute(self, _statement):
        return self._answers.pop(0)

    def add(self, obj):
        self.added.append(obj)


def _service(**overrides):
    values = dict(
        id=uuid4(), workspace_id=uuid4(), slug="web", argv=["npm", "run", "dev"],
        restart_policy="manual", resume_policy="never", status="running",
        runtime_state="ready", pid=42, pgid=42, updated_at=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def failures(monkeypatch):
    """The reasons passed to the failure counter (metrics are mocked under test)."""
    seen: list[str] = []
    monkeypatch.setattr(metrics, "record_workspace_service_failure", seen.append)
    return seen


@pytest.mark.asyncio
async def test_orphaned_service_is_stopped_recorded_and_counted(failures):
    service = _service()
    db = _Db([], [service])

    reset = await _reset_orphaned_processes(db, service.workspace_id)

    assert reset == [service]
    assert (service.runtime_state, service.status, service.pid, service.pgid) == ("stopped", "stopped", None, None)
    events = [obj for obj in db.added if isinstance(obj, WorkspaceServiceEvent)]
    assert [(e.event_type, e.source) for e in events] == [("workspace.service.stopped", "reconciler")]
    assert failures == ["orphan_process"]


@pytest.mark.asyncio
async def test_legacy_run_is_reset_without_a_service_event(failures):
    legacy = _service(argv=[], resume_policy="restart")
    db = _Db([], [legacy])

    reset = await _reset_orphaned_processes(db, legacy.workspace_id)

    assert reset == []
    assert legacy.status == "paused" and legacy.pid is None
    assert not [obj for obj in db.added if isinstance(obj, WorkspaceServiceEvent)]
    assert failures == []
