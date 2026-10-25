"""Stage 7 of internal design notes — PR-backed Gate 2 (variant A).

  * ``open_promotion`` under ``ci_provider="github_actions"``:
      - the in-container full suite is skipped → ``ci_status`` stays ``None``
        (pending, filled later by the ``workflow_run`` webhook);
      - a successful ``dev`` push opens a ``dev → main`` PR → ``pr_url`` set.
  * ``_open_promotion_pr`` delegates to ``git_forge.create_pull_request``.
  * ``resolve_promotion`` approve on a PR-backed promotion merges through the
    forge API (``_merge_promotion_pr``) instead of a local ``merge``.
  * ``_pr_number`` parses ``/pull/`` and ``/merge_requests/`` urls.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core import room_git_flow


class _FakeDB:
    def __init__(self):
        self.added: list = []

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
        id=uuid4(), room_id=uuid4(), from_branch="dev", to_branch="main",
        task_ids=["t1"], diff_stat="x | 1 +", ci_status="passed", ci_results=[],
        state="waiting_review", note=None, merge_commit=None, pr_url=None,
        resolved_at=None, resolved_by=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _stub_common(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(
            git_config={"ci_provider": "github_actions", "repo_url": "https://github.com/a/b"},
            id=uuid4(), creator_id=uuid4(),
        )),
    )
    monkeypatch.setattr(room_git_flow, "_open_promotion_row", AsyncMock(return_value=None))
    monkeypatch.setattr(room_git_flow, "_promotable_task_ids", AsyncMock(return_value=["t1"]))
    monkeypatch.setattr(room_git_flow, "_task_titles", AsyncMock(return_value=["A"]))
    monkeypatch.setattr(room_git_flow, "_collect_promoted_commits", AsyncMock(return_value=[]))
    monkeypatch.setattr(room_git_flow, "dispatch_deploy_hook", AsyncMock(return_value={}))
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)
    monkeypatch.setattr(
        "app.core.workspaced_client.git_diff_workspace",
        AsyncMock(return_value={"diff": " a.py | 2 +-\n 1 file changed"}),
    )
    return emit


def test_pr_number():
    assert room_git_flow._pr_number("https://github.com/a/b/pull/42") == 42
    assert room_git_flow._pr_number("https://gitlab.com/a/b/-/merge_requests/9") == 9
    assert room_git_flow._pr_number(None) is None
    assert room_git_flow._pr_number("https://github.com/a/b/pull/nope") is None


@pytest.mark.asyncio
async def test_open_promotion_github_actions_opens_pr(monkeypatch, _stub_common):
    monkeypatch.setattr(room_git_flow, "_push_branch_best_effort", AsyncMock(return_value=True))
    monkeypatch.setattr(
        room_git_flow, "_open_promotion_pr",
        AsyncMock(return_value="https://github.com/a/b/pull/5"),
    )
    checks = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.run_workspace_checks", checks)

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())

    assert promo.ci_status is None          # pending until the webhook
    assert promo.dev_pushed is True
    assert promo.pr_url == "https://github.com/a/b/pull/5"
    checks.assert_not_awaited()             # no in-container suite for variant A


@pytest.mark.asyncio
async def test_open_promotion_no_pr_when_push_failed(monkeypatch, _stub_common):
    monkeypatch.setattr(room_git_flow, "_push_branch_best_effort", AsyncMock(return_value=False))
    pr = AsyncMock()
    monkeypatch.setattr(room_git_flow, "_open_promotion_pr", pr)

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())
    assert promo.pr_url is None
    pr.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_promotion_pr_delegates_to_forge(monkeypatch):
    created = AsyncMock(return_value={"ok": True, "url": "https://github.com/a/b/pull/8"})
    monkeypatch.setattr("app.core.git_forge.create_pull_request", created)
    monkeypatch.setattr("app.core.git_forge.resolve_forge_token", AsyncMock(return_value="tok"))
    monkeypatch.setattr("app.core.git_forge.remote_branch_exists", AsyncMock(return_value=True))

    room = SimpleNamespace(git_config={"repo_url": "https://github.com/a/b"}, id=uuid4())
    cfg = room_git_flow.get_flow_config(SimpleNamespace(git_config={}))
    url = await room_git_flow._open_promotion_pr(room, _FakeDB(), cfg, 2)

    assert url == "https://github.com/a/b/pull/8"
    assert created.await_args.kwargs["head"] == cfg.working_branch
    assert created.await_args.kwargs["base"] == cfg.release_branch


@pytest.mark.asyncio
async def test_resolve_approve_merges_pr_not_local(monkeypatch, _stub_common):
    merged = AsyncMock(return_value={"ok": True, "sha": "prsha"})
    monkeypatch.setattr("app.core.git_forge.merge_pull_request", merged)
    monkeypatch.setattr("app.core.git_forge.resolve_forge_token", AsyncMock(return_value="tok"))
    monkeypatch.setattr(
        "app.core.git_sync.fetch_workspace_from_git", AsyncMock(return_value={"synced": 1}),
    )
    monkeypatch.setattr("app.core.workspaced_client.create_workspace_branch", AsyncMock())
    local_merge = AsyncMock()
    monkeypatch.setattr("app.core.workspaced_client.merge_workspace_branch", local_merge)

    promo = _promo(pr_url="https://github.com/a/b/pull/12")
    out = await room_git_flow.resolve_promotion(
        str(uuid4()), promo, _FakeDB(), decision="approve", git_token="tok",
    )

    assert out["ok"] is True and out["state"] == "approved"
    assert promo.merge_commit == "prsha"
    local_merge.assert_not_awaited()
    merged.assert_awaited_once()
    assert merged.await_args.args[2] == 12


@pytest.mark.asyncio
async def test_resolve_approve_pr_merge_failure_surfaces(monkeypatch, _stub_common):
    monkeypatch.setattr(
        "app.core.git_forge.merge_pull_request",
        AsyncMock(return_value={"ok": False, "error": "not mergeable"}),
    )
    monkeypatch.setattr("app.core.git_forge.resolve_forge_token", AsyncMock(return_value="tok"))

    promo = _promo(pr_url="https://github.com/a/b/pull/12")
    out = await room_git_flow.resolve_promotion(
        str(uuid4()), promo, _FakeDB(), decision="approve",
    )
    assert out["ok"] is False and "not mergeable" in out["error"]
    assert promo.state == "waiting_review"
