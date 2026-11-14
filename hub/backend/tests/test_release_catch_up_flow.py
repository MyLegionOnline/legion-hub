"""Several rooms publishing into one remote ``main``.

internal design notes Stage 5:
  * ``catch_up_release_branch`` fast-forwards the room's ``main`` through
    workspaced before a publication; no remote / a failure never blocks;
  * ``open_promotion`` refuses with ``PromotionNeedsPull`` (and a
    ``PROMOTION_NEEDS_PULL`` event) when the room's ``main`` diverged from the
    remote or the working branch conflicts with the fresh ``main``, and takes
    the bundle diff from the merge base;
  * ``resolve_promotion`` (local merge) runs the same check before merging and
    leaves the gate open on ``needs_pull``; a PR-backed one skips it;
  * the Git panel state carries what the last fetched remote ``main`` has.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import git as git_api
from app.core import room_git_flow

_REPO = {"repo_url": "https://github.com/acme/app.git", "working_branch": "core"}


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


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room", AsyncMock(return_value=SimpleNamespace(git_config=dict(_REPO)))
    )
    monkeypatch.setattr(room_git_flow, "_open_promotion_row", AsyncMock(return_value=None))
    monkeypatch.setattr(room_git_flow, "_promotable_task_ids", AsyncMock(return_value=["t1"]))
    monkeypatch.setattr(room_git_flow, "_task_titles", AsyncMock(return_value=["A"]))
    monkeypatch.setattr(room_git_flow, "_maybe_seed_shared_gitignore", AsyncMock())
    monkeypatch.setattr(room_git_flow, "_collect_promoted_commits", AsyncMock(return_value=[]))
    monkeypatch.setattr(room_git_flow, "dispatch_deploy_hook", AsyncMock(return_value={}))
    monkeypatch.setattr("app.db.release.release_db", AsyncMock())
    monkeypatch.setattr(
        "app.core.git_forge.resolve_forge_token", AsyncMock(return_value="legacy-token")
    )
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(return_value={"ran": False}),
    )
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)
    return emit


def _stub_wsd(monkeypatch, *, catch_up=None, report=None, diff=" core.py | 1 +\n"):
    catch = AsyncMock(return_value=catch_up or {"ok": True, "state": "up_to_date"})
    divergence = AsyncMock(return_value=report or {"ok": True, "behind": 0, "conflict": False})
    diff_mock = AsyncMock(return_value={"diff": diff})
    merge = AsyncMock(return_value={"ok": True, "sha": "abc123"})
    monkeypatch.setattr("app.core.workspaced_client.git_catch_up_release", catch)
    monkeypatch.setattr("app.core.workspaced_client.git_divergence_workspace", divergence)
    monkeypatch.setattr("app.core.workspaced_client.git_diff_workspace", diff_mock)
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", merge)
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", AsyncMock())
    return SimpleNamespace(catch=catch, divergence=divergence, diff=diff_mock, merge=merge)


def _types(emit) -> list[str]:
    return [c.kwargs["event_type"] for c in emit.await_args_list]


# ── catch_up_release_branch ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_catch_up_without_a_remote_does_nothing(monkeypatch):
    w = _stub_wsd(monkeypatch)

    result = await room_git_flow.catch_up_release_branch(
        "r", SimpleNamespace(git_config={}), _FakeDB()
    )

    assert result["state"] == "no_remote"
    w.catch.assert_not_awaited()


@pytest.mark.asyncio
async def test_catch_up_passes_clone_url_and_token(monkeypatch):
    w = _stub_wsd(monkeypatch, catch_up={"ok": True, "state": "fast_forwarded", "remote_sha": "f" * 40})
    room = SimpleNamespace(git_config={"repo_url": "https://github.com/acme/app/tree/main"})

    result = await room_git_flow.catch_up_release_branch("r", room, _FakeDB(), "explicit")

    assert result["state"] == "fast_forwarded"
    w.catch.assert_awaited_once_with("r", "https://github.com/acme/app", "explicit")


@pytest.mark.asyncio
async def test_catch_up_falls_back_to_the_room_token(monkeypatch):
    w = _stub_wsd(monkeypatch)

    await room_git_flow.catch_up_release_branch("r", SimpleNamespace(git_config=dict(_REPO)), _FakeDB())

    assert w.catch.await_args.args[2] == "legacy-token"


@pytest.mark.asyncio
async def test_catch_up_failure_never_raises(monkeypatch):
    _stub_wsd(monkeypatch)
    monkeypatch.setattr(
        "app.core.workspaced_client.git_catch_up_release",
        AsyncMock(side_effect=RuntimeError("workspaced down")),
    )

    result = await room_git_flow.catch_up_release_branch(
        "r", SimpleNamespace(git_config=dict(_REPO)), _FakeDB()
    )

    assert result["ok"] is False and result["state"] == "unavailable"


# ── open_promotion ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_open_promotion_catches_up_and_diffs_from_the_merge_base(monkeypatch, _stubs):
    w = _stub_wsd(monkeypatch, catch_up={"ok": True, "state": "fast_forwarded"})

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB(), access_token="tok")

    assert promo.from_branch == "core" and promo.state == "waiting_review"
    assert w.catch.await_args.args[2] == "tok"
    assert w.divergence.await_args.kwargs == {"branch": "core", "base": "main"}
    assert w.diff.await_args.kwargs["merge_base"] is True
    opened = next(c for c in _stubs.await_args_list if c.kwargs["event_type"] == "PROMOTION_OPENED")
    assert opened.kwargs["metadata"]["release_catch_up"] == "fast_forwarded"


@pytest.mark.asyncio
async def test_open_promotion_refuses_a_diverged_main(monkeypatch, _stubs):
    w = _stub_wsd(monkeypatch, catch_up={"ok": True, "state": "diverged"})

    with pytest.raises(room_git_flow.PromotionNeedsPull) as exc:
        await room_git_flow.open_promotion(str(uuid4()), _FakeDB())

    assert exc.value.reason == "main_diverged"
    assert "Pull main" in str(exc.value)
    assert _types(_stubs) == ["PROMOTION_NEEDS_PULL"]
    w.diff.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_promotion_refuses_a_conflict_with_the_new_main(monkeypatch, _stubs):
    _stub_wsd(
        monkeypatch,
        catch_up={"ok": True, "state": "fast_forwarded"},
        report={"ok": True, "behind": 2, "conflict": True, "conflict_files": ["shared.py"]},
    )

    with pytest.raises(room_git_flow.PromotionNeedsPull) as exc:
        await room_git_flow.open_promotion(str(uuid4()), _FakeDB())

    assert exc.value.reason == "conflict"
    assert exc.value.conflict_files == ["shared.py"]
    meta = _stubs.await_args_list[-1].kwargs["metadata"]
    assert meta == {
        "reason": "conflict", "conflict_files": ["shared.py"],
        "working_branch": "core", "release_branch": "main",
    }


@pytest.mark.asyncio
async def test_open_promotion_goes_ahead_when_the_new_main_merges_cleanly(monkeypatch):
    _stub_wsd(monkeypatch, report={"ok": True, "behind": 3, "conflict": False})

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())

    assert promo.state == "waiting_review"


@pytest.mark.asyncio
async def test_open_promotion_without_a_report_goes_ahead(monkeypatch):
    """workspaced could not preview the merge: the approve-time merge still guards."""
    _stub_wsd(monkeypatch, report={"ok": False, "error": "workspaced unreachable"})

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())

    assert promo.state == "waiting_review"


# ── resolve_promotion ───────────────────────────────────────────────────────

def _promo(**kw):
    base = dict(
        id=uuid4(), room_id=uuid4(), from_branch="core", to_branch="main",
        task_ids=["t1"], diff_stat="", ci_status="passed", ci_results=[],
        state="waiting_review", note=None, merge_commit=None,
        resolved_at=None, resolved_by=None, pr_url=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_approve_leaves_the_gate_open_when_a_pull_is_needed(monkeypatch, _stubs):
    w = _stub_wsd(
        monkeypatch,
        report={"ok": True, "behind": 1, "conflict": True, "conflict_files": ["a.py"]},
    )
    promo = _promo()

    result = await room_git_flow.resolve_promotion(
        str(uuid4()), promo, _FakeDB(), decision="approve", git_token="tok"
    )

    assert result["ok"] is False and result["needs_pull"] is True
    assert result["reason"] == "conflict" and result["conflict_files"] == ["a.py"]
    assert promo.state == "waiting_review"
    w.merge.assert_not_awaited()
    assert w.catch.await_args.args[2] == "tok"


@pytest.mark.asyncio
async def test_approve_merges_after_catching_up(monkeypatch, _stubs):
    w = _stub_wsd(monkeypatch, catch_up={"ok": True, "state": "fast_forwarded"})
    promo = _promo()

    result = await room_git_flow.resolve_promotion(str(uuid4()), promo, _FakeDB(), decision="approve")

    assert result["ok"] is True and promo.state == "approved"
    w.catch.assert_awaited_once()
    w.merge.assert_awaited_once()


@pytest.mark.asyncio
async def test_pr_backed_approve_skips_the_local_check(monkeypatch):
    w = _stub_wsd(monkeypatch)
    merge_pr = AsyncMock(return_value={"ok": True, "sha": "pr-sha", "pr_number": 7})
    monkeypatch.setattr(room_git_flow, "_merge_promotion_pr", merge_pr)
    promo = _promo(pr_url="https://github.com/acme/app/pull/7")

    result = await room_git_flow.resolve_promotion(str(uuid4()), promo, _FakeDB(), decision="approve")

    assert result["ok"] is True
    w.catch.assert_not_awaited()
    w.divergence.assert_not_awaited()


# ── Git panel state ─────────────────────────────────────────────────────────

def _db_returning(*rows_by_call):
    idx = [0]

    async def execute(_stmt):
        rows = rows_by_call[idx[0]] if idx[0] < len(rows_by_call) else []
        idx[0] += 1
        scalars = MagicMock()
        scalars.all.return_value = rows
        scalars.first.return_value = rows[0] if rows else None
        result = MagicMock()
        result.scalars.return_value = scalars
        return result

    return execute


@pytest.mark.asyncio
async def test_git_state_reports_the_remote_main_ahead():
    room = MagicMock()
    room.id = uuid4()
    room.git_config = dict(_REPO)
    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([], [], [], [], [], [])
    status = AsyncMock(return_value={"ok": True, "main_behind": 1, "working_behind": 4})

    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=[]), \
         patch.object(git_api.wsd, "git_ahead_behind_remote_workspace", new_callable=AsyncMock, return_value=None), \
         patch.object(git_api.wsd, "git_release_remote_status", status):
        state = await git_api.build_git_state(room.id, db)

    assert state.flow.release_remote_ahead == 4
    assert state.flow.release_local_behind == 1
    status.assert_awaited_once_with(str(room.id), "core")


@pytest.mark.asyncio
async def test_git_state_without_a_remote_skips_the_status():
    room = MagicMock()
    room.id = uuid4()
    room.git_config = {}
    db = AsyncMock()
    db.get = AsyncMock(return_value=room)
    db.execute = _db_returning([], [], [], [], [], [])
    status = AsyncMock()

    with patch.object(git_api.wsd, "list_workspace_branches", new_callable=AsyncMock, return_value=[]), \
         patch.object(git_api.wsd, "git_release_remote_status", status):
        state = await git_api.build_git_state(room.id, db)

    assert state.flow.release_remote_ahead is None
    status.assert_not_awaited()
