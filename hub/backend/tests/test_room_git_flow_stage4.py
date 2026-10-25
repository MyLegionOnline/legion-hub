"""Stage 4 of internal design notes — Gate 2 (``dev → main`` promotions).

Covers ``room_git_flow``:
  * ``open_promotion``:
      - empty ``main..dev`` diff → ``PromotionEmpty``;
      - a promotion already awaiting review → ``PromotionAlreadyOpen``;
      - green full checks → ``Promotion(state="waiting_review", ci_status="passed")``
        + ``PROMOTION_OPENED`` / ``CHECKPOINT_REQUEST`` events;
      - red full checks → ``ci_status="failed"``;
      - no checks declared → ``ci_status="passed"`` (Gate 1 contract);
  * ``resolve_promotion``:
      - ``reject`` → ``state="rejected"`` + ``PROMOTION_REJECTED``, no merge;
      - ``approve`` blocked when ``ci_status != "passed"`` and not ``force``;
      - ``approve`` + ``force`` over a red CI → clean merge, ``PROMOTED_TO_MAIN``,
        HEAD restored to the working branch, ``merge_commit`` recorded;
      - ``approve`` conflict → ``state="conflict"`` + ``GIT_CONFLICT``, no move;
      - resolving an already-resolved promotion → ``ok=False``;
  * ``maybe_open_promotion`` only fires under ``promote_policy="on_each_task"``.
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


def _promo(**kw):
    base = dict(
        id=uuid4(),
        room_id=uuid4(),
        from_branch="dev",
        to_branch="main",
        task_ids=["t1", "t2"],
        diff_stat="x | 2 +-\n 1 file changed",
        ci_status="passed",
        ci_results=[],
        state="waiting_review",
        note=None,
        merge_commit=None,
        resolved_at=None,
        resolved_by=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _stub_common(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={})),
    )
    monkeypatch.setattr(room_git_flow, "_open_promotion_row", AsyncMock(return_value=None))
    monkeypatch.setattr(room_git_flow, "_promotable_task_ids", AsyncMock(return_value=["t1", "t2"]))
    monkeypatch.setattr(room_git_flow, "_task_titles", AsyncMock(return_value=["A", "B"]))
    # Stage 9 deploy hook is exercised in its own suite.
    monkeypatch.setattr(room_git_flow, "_collect_promoted_commits", AsyncMock(return_value=[]))
    monkeypatch.setattr(room_git_flow, "dispatch_deploy_hook", AsyncMock(return_value={}))
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


def _event_types(emit) -> list[str]:
    return [c.kwargs["event_type"] for c in emit.await_args_list]


# ── open_promotion ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_open_promotion_empty_diff_raises(monkeypatch):
    _stub_diff(monkeypatch, "   \n  ")
    with pytest.raises(room_git_flow.PromotionEmpty):
        await room_git_flow.open_promotion(str(uuid4()), _FakeDB())


@pytest.mark.asyncio
async def test_open_promotion_already_open_raises(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_open_promotion_row", AsyncMock(return_value=_promo()))
    with pytest.raises(room_git_flow.PromotionAlreadyOpen):
        await room_git_flow.open_promotion(str(uuid4()), _FakeDB())


@pytest.mark.asyncio
async def test_open_promotion_green_full_checks(monkeypatch, _stub_common):
    _stub_diff(monkeypatch, " a.py | 4 ++--\n 1 file changed, 2 insertions(+), 2 deletions(-)")
    _stub_checks(monkeypatch, ran=True, passed=True, results=[{"name": "pytest", "exit_code": 0}])

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())

    assert promo.state == "waiting_review"
    assert promo.ci_status == "passed"
    assert promo.task_ids == ["t1", "t2"]
    types = _event_types(_stub_common)
    assert "PROMOTION_OPENED" in types
    assert "CHECKPOINT_REQUEST" in types


@pytest.mark.asyncio
async def test_open_promotion_red_full_checks(monkeypatch):
    _stub_diff(monkeypatch, " a.py | 4 ++--\n 1 file changed")
    _stub_checks(monkeypatch, ran=True, passed=False, results=[{"name": "pytest", "exit_code": 1}])

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())
    assert promo.ci_status == "failed"


@pytest.mark.asyncio
async def test_open_promotion_no_checks_is_green(monkeypatch):
    _stub_diff(monkeypatch, " a.py | 4 ++--\n 1 file changed")
    _stub_checks(monkeypatch, ran=False, passed=True, results=[])

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())
    assert promo.ci_status == "passed"


@pytest.mark.asyncio
async def test_open_promotion_check_transport_error_leaves_ci_none(monkeypatch):
    _stub_diff(monkeypatch, " a.py | 4 ++--\n 1 file changed")
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )
    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())
    assert promo.ci_status is None


# ── resolve_promotion ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolve_reject_no_merge(monkeypatch, _stub_common):
    merge = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", merge)

    out = await room_git_flow.resolve_promotion(
        str(uuid4()), _promo(), _FakeDB(), decision="reject", note="not yet"
    )
    assert out["ok"] is True and out["state"] == "rejected"
    merge.assert_not_awaited()
    assert "PROMOTION_REJECTED" in _event_types(_stub_common)


@pytest.mark.asyncio
async def test_resolve_approve_blocked_when_ci_not_passed(monkeypatch):
    merge = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", merge)

    out = await room_git_flow.resolve_promotion(
        str(uuid4()), _promo(ci_status="failed"), _FakeDB(), decision="approve"
    )
    assert out["ok"] is False and out["blocked"] is True
    merge.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_approve_force_over_red_ci(monkeypatch, _stub_common):
    merge = AsyncMock(return_value={"ok": True, "conflict": False, "sha": "mainsha"})
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", merge)
    restore = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", restore)

    promo = _promo(ci_status="failed")
    out = await room_git_flow.resolve_promotion(
        str(uuid4()), promo, _FakeDB(), decision="approve", force=True
    )
    assert out["ok"] is True and out["state"] == "approved"
    assert promo.merge_commit == "mainsha"
    # clean merge, no -X theirs
    assert merge.await_args.kwargs["theirs"] is False
    assert merge.await_args.kwargs["target"] == "main"
    # HEAD put back on the working branch
    restore.assert_awaited_once()
    assert restore.await_args.args[1] == "dev"
    assert "PROMOTED_TO_MAIN" in _event_types(_stub_common)


@pytest.mark.asyncio
async def test_resolve_approve_conflict(monkeypatch, _stub_common):
    monkeypatch.setattr(
        "app.core.workspaced_client.merge_workspace_branch",
        AsyncMock(return_value={"ok": False, "conflict": True}),
    )
    promo = _promo()
    out = await room_git_flow.resolve_promotion(
        str(uuid4()), promo, _FakeDB(), decision="approve"
    )
    assert out["conflict"] is True and promo.state == "conflict"
    assert promo.merge_commit is None
    assert "GIT_CONFLICT" in _event_types(_stub_common)


@pytest.mark.asyncio
async def test_resolve_already_resolved_is_noop(monkeypatch):
    out = await room_git_flow.resolve_promotion(
        str(uuid4()), _promo(state="approved"), _FakeDB(), decision="approve"
    )
    assert out["ok"] is False and out["state"] == "approved"


# ── maybe_open_promotion ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_open_promotion_skips_when_manual(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={"promote_policy": "manual"})),
    )
    opened = AsyncMock()
    monkeypatch.setattr(room_git_flow, "open_promotion", opened)

    assert await room_git_flow.maybe_open_promotion(str(uuid4()), _FakeDB()) is None
    opened.assert_not_awaited()


@pytest.mark.asyncio
async def test_maybe_open_promotion_fires_on_each_task(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={"promote_policy": "on_each_task"})),
    )
    sentinel = object()
    opened = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(room_git_flow, "open_promotion", opened)

    assert await room_git_flow.maybe_open_promotion(str(uuid4()), _FakeDB()) is sentinel
    opened.assert_awaited_once()


@pytest.mark.asyncio
async def test_maybe_open_promotion_swallows_empty(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={"promote_policy": "on_each_task"})),
    )
    monkeypatch.setattr(
        room_git_flow, "open_promotion",
        AsyncMock(side_effect=room_git_flow.PromotionEmpty("nothing")),
    )
    assert await room_git_flow.maybe_open_promotion(str(uuid4()), _FakeDB()) is None
