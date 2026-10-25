"""Stage 5 of internal design notes — whole goal → ``dev`` review gate.

Covers ``room_git_flow``:
  * ``open_goal_review``:
      - a goal with no branch → ``GoalReviewError``;
      - a bundle-diff transport error → ``GoalReviewError``;
      - green full checks → ``review_state="waiting_review"``,
        ``review_ci_status="passed"`` + ``GOAL_REVIEW_OPENED`` /
        ``CHECKPOINT_REQUEST`` events;
      - red full checks → ``review_ci_status="failed"``;
      - no checks declared → ``review_ci_status="passed"`` (Gate 1 contract);
      - a checks transport error → ``review_ci_status`` left ``None``;
  * ``resolve_goal_review``:
      - ``reject`` → ``review_state="rejected"`` + ``GOAL_REVIEW_REJECTED``, no merge;
      - ``approve`` blocked when CI is not green and not ``force``;
      - ``approve`` + ``force`` over a red CI → clean merge (no ``-X theirs``),
        ``review_state="approved"``, ``GOAL_PROMOTED_TO_DEV``, goal finalized;
      - ``approve`` conflict → ``conflict=True`` + ``GIT_CONFLICT``, ``dev`` untouched;
      - resolving an already-resolved goal → ``ok=False``;
  * ``maybe_open_goal_review`` only opens for a branched goal not already
    in / past the gate.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow


class _FakeDB:
    def __init__(self):
        self.added: list[object] = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def refresh(self, obj):
        pass


def _goal(**kw):
    base = dict(
        id=uuid4(),
        title="G",
        branch_name=f"goal/{uuid4()}-x",
        review_state=None,
        review_ci_status=None,
        review_branch_pushed_at=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _stub_common(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={})),
    )
    monkeypatch.setattr(room_git_flow, "_goal_task_titles", AsyncMock(return_value=["A", "B"]))
    monkeypatch.setattr(room_git_flow, "_stamp_goal_task_merges", AsyncMock())
    monkeypatch.setattr(
        "app.core.goal_progress.finalize_goal_completion", AsyncMock(return_value=True)
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)
    return emit


def _stub_diff(monkeypatch, stat: str):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_diff_workspace",
        AsyncMock(return_value={"diff": stat}),
    )


def _stub_checks(monkeypatch, **ret):
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(return_value=ret),
    )


def _stub_merge(monkeypatch, **ret):
    m = AsyncMock(return_value=ret)
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", m)
    return m


def _event_types(emit) -> list[str]:
    return [c.kwargs["event_type"] for c in emit.await_args_list]


# ── open_goal_review ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_open_goal_review_no_branch_raises(monkeypatch):
    with pytest.raises(room_git_flow.GoalReviewError):
        await room_git_flow.open_goal_review(str(uuid4()), _goal(branch_name=None), _FakeDB())


@pytest.mark.asyncio
async def test_open_goal_review_diff_error_raises(monkeypatch):
    monkeypatch.setattr(
        "app.core.workspaced_client.git_diff_workspace",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )
    with pytest.raises(room_git_flow.GoalReviewError):
        await room_git_flow.open_goal_review(str(uuid4()), _goal(), _FakeDB())


@pytest.mark.asyncio
async def test_open_goal_review_green(monkeypatch, _stub_common):
    _stub_diff(monkeypatch, " a.py | 4 ++--\n 1 file changed, 2 insertions(+), 2 deletions(-)")
    _stub_checks(monkeypatch, ran=True, passed=True, results=[{"name": "pytest", "exit_code": 0}])

    goal = _goal()
    out = await room_git_flow.open_goal_review(str(uuid4()), goal, _FakeDB())

    assert out is goal
    assert goal.review_state == "waiting_review"
    assert goal.review_ci_status == "passed"
    types = _event_types(_stub_common)
    assert "GOAL_REVIEW_OPENED" in types
    assert "CHECKPOINT_REQUEST" in types


@pytest.mark.asyncio
async def test_open_goal_review_red(monkeypatch):
    _stub_diff(monkeypatch, " a.py | 4 ++--\n 1 file changed")
    _stub_checks(monkeypatch, ran=True, passed=False, results=[{"name": "pytest", "exit_code": 1}])

    goal = _goal()
    await room_git_flow.open_goal_review(str(uuid4()), goal, _FakeDB())
    assert goal.review_ci_status == "failed"
    assert goal.review_state == "waiting_review"


@pytest.mark.asyncio
async def test_open_goal_review_no_checks_is_green(monkeypatch):
    _stub_diff(monkeypatch, " a.py | 4 ++--\n 1 file changed")
    _stub_checks(monkeypatch, ran=False, passed=True, results=[])

    goal = _goal()
    await room_git_flow.open_goal_review(str(uuid4()), goal, _FakeDB())
    assert goal.review_ci_status == "passed"


@pytest.mark.asyncio
async def test_open_goal_review_checks_transport_error_leaves_ci_none(monkeypatch):
    _stub_diff(monkeypatch, " a.py | 4 ++--\n 1 file changed")
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )
    goal = _goal()
    await room_git_flow.open_goal_review(str(uuid4()), goal, _FakeDB())
    assert goal.review_ci_status is None
    assert goal.review_state == "waiting_review"


# ── resolve_goal_review ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_reject_no_merge(monkeypatch, _stub_common):
    merge = _stub_merge(monkeypatch, ok=True)
    goal = _goal(review_state="waiting_review")

    out = await room_git_flow.resolve_goal_review(
        str(uuid4()), goal, _FakeDB(), decision="reject", note="not yet"
    )
    assert out["ok"] is True and out["state"] == "rejected"
    assert goal.review_state == "rejected"
    merge.assert_not_awaited()
    assert "GOAL_REVIEW_REJECTED" in _event_types(_stub_common)


@pytest.mark.asyncio
async def test_resolve_approve_blocked_when_ci_not_passed(monkeypatch):
    merge = _stub_merge(monkeypatch, ok=True)
    goal = _goal(review_state="waiting_review", review_ci_status="failed")

    out = await room_git_flow.resolve_goal_review(
        str(uuid4()), goal, _FakeDB(), decision="approve"
    )
    assert out["ok"] is False and out["blocked"] is True
    merge.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_approve_force_over_red_ci(monkeypatch, _stub_common):
    merge = _stub_merge(monkeypatch, ok=True, conflict=False, sha="devsha")
    from app.core import goal_progress

    goal = _goal(review_state="waiting_review", review_ci_status="failed")
    out = await room_git_flow.resolve_goal_review(
        str(uuid4()), goal, _FakeDB(), decision="approve", force=True
    )
    assert out["ok"] is True and out["state"] == "approved"
    assert out["merge_commit"] == "devsha"
    assert merge.await_args.kwargs["theirs"] is False
    assert merge.await_args.kwargs["target"] == "dev"
    room_git_flow._stamp_goal_task_merges.assert_awaited_once()
    goal_progress.finalize_goal_completion.assert_awaited_once()
    assert "GOAL_PROMOTED_TO_DEV" in _event_types(_stub_common)


@pytest.mark.asyncio
async def test_resolve_approve_conflict(monkeypatch, _stub_common):
    _stub_merge(monkeypatch, ok=False, conflict=True)
    goal = _goal(review_state="waiting_review", review_ci_status="passed")

    out = await room_git_flow.resolve_goal_review(
        str(uuid4()), goal, _FakeDB(), decision="approve"
    )
    assert out["conflict"] is True
    assert goal.review_state == "waiting_review"  # dev untouched, gate still open
    assert "GIT_CONFLICT" in _event_types(_stub_common)


@pytest.mark.asyncio
async def test_resolve_already_resolved_is_noop(monkeypatch):
    out = await room_git_flow.resolve_goal_review(
        str(uuid4()), _goal(review_state="approved"), _FakeDB(), decision="approve"
    )
    assert out["ok"] is False and out["state"] == "approved"


# ── maybe_open_goal_review ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_open_goal_review_no_branch_returns_none(monkeypatch):
    opened = AsyncMock()
    monkeypatch.setattr(room_git_flow, "open_goal_review", opened)
    out = await room_git_flow.maybe_open_goal_review(_goal(branch_name=None), str(uuid4()), _FakeDB())
    assert out is None
    opened.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_open_goal_review_holds_when_pending(monkeypatch):
    opened = AsyncMock()
    monkeypatch.setattr(room_git_flow, "open_goal_review", opened)
    goal = _goal(review_state="waiting_review")
    assert await room_git_flow.maybe_open_goal_review(goal, str(uuid4()), _FakeDB()) is goal
    opened.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_open_goal_review_approved_returns_none(monkeypatch):
    opened = AsyncMock()
    monkeypatch.setattr(room_git_flow, "open_goal_review", opened)
    goal = _goal(review_state="approved")
    assert await room_git_flow.maybe_open_goal_review(goal, str(uuid4()), _FakeDB()) is None
    opened.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_open_goal_review_reopens_after_reject(monkeypatch):
    sentinel = object()
    opened = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(room_git_flow, "open_goal_review", opened)
    goal = _goal(review_state="rejected")
    assert await room_git_flow.maybe_open_goal_review(goal, str(uuid4()), _FakeDB()) is sentinel
    opened.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_open_goal_review_swallows_errors(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "open_goal_review",
        AsyncMock(side_effect=room_git_flow.GoalReviewError("nope")),
    )
    assert await room_git_flow.maybe_open_goal_review(_goal(), str(uuid4()), _FakeDB()) is None


@pytest.mark.asyncio
async def test_maybe_open_goal_review_resets_stuck_testing_state(monkeypatch):
    """internal design notes Stage 4 — a ``GoalReviewError`` raised
    *after* ``open_goal_review`` flipped ``review_state`` to "testing" must not
    leave the fail-open completion carrying a misleading "testing"."""
    async def _raise_but_dirty(room_id, goal, db):
        goal.review_state = room_git_flow.GOAL_REVIEW_TESTING
        raise room_git_flow.GoalReviewError("diff blew up")

    monkeypatch.setattr(room_git_flow, "open_goal_review", _raise_but_dirty)
    goal = _goal()
    out = await room_git_flow.maybe_open_goal_review(goal, str(uuid4()), _FakeDB())
    assert out is None
    assert goal.review_state is None


# ── publish_goal_branch (Stage 4) ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_publish_goal_branch_pushes_and_stamps(monkeypatch, _stub_common):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={"repo_url": "https://x/y"})),
    )
    push = AsyncMock(return_value={"pushed": 1, "errors": []})
    monkeypatch.setattr("app.core.git_sync.push_workspace_to_git", push)

    goal = _goal(review_state="waiting_review")
    out = await room_git_flow.publish_goal_branch(
        str(uuid4()), goal, _FakeDB(), access_token="tok"
    )

    assert out["ok"] is True
    assert push.await_args.kwargs["branch"] == goal.branch_name
    assert push.await_args.kwargs["mode"] == "push"
    assert goal.review_branch_pushed_at is not None
    assert goal.review_state == "waiting_review"  # publishing never touches the gate
    assert "GOAL_BRANCH_PUBLISHED" in _event_types(_stub_common)


@pytest.mark.asyncio
async def test_publish_goal_branch_non_ff_leaves_state(monkeypatch, _stub_common):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={"repo_url": "https://x/y"})),
    )
    monkeypatch.setattr(
        "app.core.git_sync.push_workspace_to_git",
        AsyncMock(return_value={"pushed": 0, "errors": ["! [rejected] non-fast-forward"]}),
    )
    goal = _goal(review_state="waiting_review")
    out = await room_git_flow.publish_goal_branch(str(uuid4()), goal, _FakeDB())

    assert out["ok"] is False
    assert out["hint"] == "sync_then_promote"
    assert goal.review_branch_pushed_at is None
    assert "GOAL_BRANCH_PUBLISHED" not in _event_types(_stub_common)


@pytest.mark.asyncio
async def test_publish_goal_branch_rejected_when_gate_not_open(monkeypatch):
    push = AsyncMock()
    monkeypatch.setattr("app.core.git_sync.push_workspace_to_git", push)
    out = await room_git_flow.publish_goal_branch(
        str(uuid4()), _goal(review_state="approved"), _FakeDB()
    )
    assert out["ok"] is False
    push.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_goal_branch_rejected_without_repo(monkeypatch, _stub_common):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={})),
    )
    push = AsyncMock()
    monkeypatch.setattr("app.core.git_sync.push_workspace_to_git", push)
    out = await room_git_flow.publish_goal_branch(
        str(uuid4()), _goal(review_state="waiting_review"), _FakeDB()
    )
    assert out["ok"] is False
    push.assert_not_awaited()
