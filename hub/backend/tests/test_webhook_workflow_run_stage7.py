"""Stage 7 of internal design notes — external CI status from GitHub.

  * ``room_git_flow.apply_external_ci_status``: maps a ``workflow_run``
    conclusion onto the open promotion's ``ci_status`` and emits ``CI_STATUS``;
    a no-op when no open promotion targets the branch.
  * ``app.api.v1.git._handle_ci_status_webhook``: unwraps the ``workflow_run`` /
    ``check_suite`` node, ignores a run that has not completed or carries no
    ``head_branch``, and otherwise defers to ``apply_external_ci_status``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow


class _Scalars:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows


class _ExecResult:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return _Scalars(self._rows)


class _FakeDB:
    def __init__(self, promo):
        self._promo = promo
        self.added: list = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def execute(self, _stmt):
        return _ExecResult([self._promo] if self._promo is not None else [])


def _promo(**kw):
    base = dict(
        id=uuid4(), room_id=uuid4(), from_branch="dev", to_branch="main",
        state=room_git_flow.PROMOTION_WAITING, ci_status=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _emit(monkeypatch):
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)
    return emit


# ── apply_external_ci_status ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_marks_promotion_passed(_emit):
    promo = _promo()
    out = await room_git_flow.apply_external_ci_status(
        str(uuid4()), "dev", "success", _FakeDB(promo)
    )
    assert out["matched"] is True and out["ci_status"] == "passed"
    assert promo.ci_status == "passed"
    assert _emit.await_args.kwargs["event_type"] == "CI_STATUS"


@pytest.mark.asyncio
async def test_failure_marks_promotion_failed(_emit):
    promo = _promo()
    out = await room_git_flow.apply_external_ci_status(
        str(uuid4()), "dev", "failure", _FakeDB(promo)
    )
    assert out["ci_status"] == "failed"
    assert promo.ci_status == "failed"


@pytest.mark.asyncio
async def test_no_open_promotion_is_noop(_emit):
    out = await room_git_flow.apply_external_ci_status(
        str(uuid4()), "dev", "success", _FakeDB(None)
    )
    assert out == {"matched": False}
    _emit.assert_not_awaited()


class _SeqDB:
    """execute() returns a different result each call, in order."""

    def __init__(self, *result_rows):
        self._results = list(result_rows)
        self.added: list = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def execute(self, _stmt):
        rows = self._results.pop(0) if self._results else []
        return _ExecResult(rows)


@pytest.mark.asyncio
async def test_goal_branch_ci_status_stamps_review_ci_status(_emit):
    """internal design notes Stage 4 — a webhook for a pushed goal
    branch stamps the open Goal gate's ``review_ci_status``."""
    goal = SimpleNamespace(
        id=uuid4(), title="Ship it",
        branch_name="goal/abc-ship", review_state=room_git_flow.GOAL_REVIEW_WAITING,
        review_ci_status=None,
    )
    db = _SeqDB([], [goal])  # no promotion, then the goal
    out = await room_git_flow.apply_external_ci_status(
        str(uuid4()), "goal/abc-ship", "failure", db
    )
    assert out["matched"] is True and out["ci_status"] == "failed"
    assert out["goal_id"] == str(goal.id)
    assert goal.review_ci_status == "failed"
    assert _emit.await_args.kwargs["event_type"] == "CI_STATUS"
    assert _emit.await_args.kwargs["metadata"]["goal_id"] == str(goal.id)


@pytest.mark.asyncio
async def test_goal_branch_no_open_gate_is_noop(_emit):
    db = _SeqDB([], [])  # no promotion, no matching goal
    out = await room_git_flow.apply_external_ci_status(
        str(uuid4()), "goal/abc-ship", "success", db
    )
    assert out == {"matched": False}
    _emit.assert_not_awaited()


# ── _handle_ci_status_webhook ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_unwraps_workflow_run(monkeypatch):
    from app.api.v1 import git as git_api

    applied = AsyncMock(return_value={"matched": True, "ci_status": "passed"})
    monkeypatch.setattr(room_git_flow, "apply_external_ci_status", applied)

    room = SimpleNamespace(id=uuid4())
    payload = {"workflow_run": {"status": "completed", "conclusion": "success", "head_branch": "dev"}}
    out = await git_api._handle_ci_status_webhook(room, "workflow_run", payload, _FakeDB(None))

    assert out["ok"] is True and out["ci_status"] == "passed"
    assert applied.await_args.args[1] == "dev"
    assert applied.await_args.args[2] == "success"


@pytest.mark.asyncio
async def test_webhook_ignores_in_progress_run(monkeypatch):
    from app.api.v1 import git as git_api

    applied = AsyncMock()
    monkeypatch.setattr(room_git_flow, "apply_external_ci_status", applied)

    payload = {"workflow_run": {"status": "in_progress", "head_branch": "dev"}}
    out = await git_api._handle_ci_status_webhook(SimpleNamespace(id=uuid4()), "workflow_run", payload, _FakeDB(None))

    assert out["ok"] is True and "ignored" in out
    applied.assert_not_awaited()


@pytest.mark.asyncio
async def test_webhook_ignores_missing_head_branch(monkeypatch):
    from app.api.v1 import git as git_api

    applied = AsyncMock()
    monkeypatch.setattr(room_git_flow, "apply_external_ci_status", applied)

    payload = {"check_suite": {"status": "completed", "conclusion": "success"}}
    out = await git_api._handle_ci_status_webhook(SimpleNamespace(id=uuid4()), "check_suite", payload, _FakeDB(None))

    assert out["ok"] is True and "ignored" in out
    applied.assert_not_awaited()
