"""internal design notes Stage 2 — publication guards.

Covers:
  * ``POST /rooms/{id}/git/push`` refuses an empty branch (only ``.gitignore``
    or nothing) with 409, tied to the branch actually being pushed — so a
    ``dev`` push from a room whose ``main`` is still empty is allowed;
  * a non-fast-forward rejection comes back with ``hint="sync_then_promote"``
    (but not under ``mode=force``);
  * ``git_forge.remote_branch_exists`` is conservative (False on an
    unparseable / non-GitHub URL);
  * ``room_git_flow._open_promotion_pr`` skips the PR + emits
    ``PROMOTION_PR_SKIPPED`` when the remote has no ``main`` and
    ``seed_main_on_first_promotion`` is off, and seeds ``main`` once when it
    is on.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import git as git_api
from app.core import git_forge, room_git_flow


def _room(branch="main"):
    return MagicMock(git_config={"repo_url": "https://github.com/o/r", "branch": branch})


def _user():
    return MagicMock(id=uuid4(), email="u@e")


def _request():
    return MagicMock(client=MagicMock(host="127.0.0.1"))


# ── §4.1 empty branch is not pushed ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_push_empty_main_blocked_with_409():
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch(
             "app.core.workspaced_client.git_ls_tree_workspace",
             new_callable=AsyncMock,
             return_value=[{"path": ".gitignore", "size": 5}],
         ), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as push:
        with pytest.raises(HTTPException) as ei:
            await git_api.git_push(
                room_id=uuid4(), request=_request(), body=None,
                db=AsyncMock(), current_user=_user(),
            )

    assert ei.value.status_code == 409
    assert "main" in ei.value.detail
    push.assert_not_awaited()


@pytest.mark.asyncio
async def test_push_dev_allowed_when_dev_has_content():
    ls = AsyncMock(return_value=[{"path": "app/main.py", "size": 120}])
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch("app.core.workspaced_client.git_ls_tree_workspace", ls), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=None), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as push:
        push.return_value = {"pushed": 1, "errors": []}
        resp = await git_api.git_push(
            room_id=uuid4(), request=_request(),
            body=git_api.GitPushBody(branch="dev"),
            db=AsyncMock(), current_user=_user(),
        )

    assert resp["ok"] is True
    assert ls.await_args.kwargs["ref"] == "dev"
    assert push.await_args.kwargs["branch"] == "dev"


@pytest.mark.asyncio
async def test_push_not_blocked_when_ls_tree_unavailable():
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch(
             "app.core.workspaced_client.git_ls_tree_workspace",
             new_callable=AsyncMock, side_effect=RuntimeError("workspaced down"),
         ), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=None), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as push:
        push.return_value = {"pushed": 1, "errors": []}
        resp = await git_api.git_push(
            room_id=uuid4(), request=_request(), body=None,
            db=AsyncMock(), current_user=_user(),
        )
    assert resp["ok"] is True


# ── §4.3 non-fast-forward hint ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_push_non_ff_returns_sync_then_promote_hint():
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch(
             "app.core.workspaced_client.git_ls_tree_workspace",
             new_callable=AsyncMock, return_value=[{"path": "app.py", "size": 10}],
         ), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=None), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as push:
        push.return_value = {
            "pushed": 0,
            "errors": ["! [rejected]        main -> main (non-fast-forward)"],
        }
        resp = await git_api.git_push(
            room_id=uuid4(), request=_request(), body=None,
            db=AsyncMock(), current_user=_user(),
        )

    assert resp["ok"] is False
    assert resp["hint"] == "sync_then_promote"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        # GitHub-side failure: Pulling cannot help, so no "remote moved" hint.
        "! [remote rejected] main -> main (Internal Server Error)",
        "! [remote rejected] main -> main (protected branch hook declined)",
        "remote: error: GH013: Repository rule violations found",
    ],
)
async def test_push_server_side_rejection_gets_no_pull_hint(error):
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch(
             "app.core.workspaced_client.git_ls_tree_workspace",
             new_callable=AsyncMock, return_value=[{"path": "app.py", "size": 10}],
         ), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=None), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as push:
        push.return_value = {"pushed": 0, "errors": [error]}
        resp = await git_api.git_push(
            room_id=uuid4(), request=_request(), body=None,
            db=AsyncMock(), current_user=_user(),
        )

    assert resp["ok"] is False
    assert resp.get("hint") is None
    assert resp["errors"] == [error]  # the UI shows the real reason instead


@pytest.mark.parametrize(
    "error",
    [
        "! [rejected] main -> main (non-fast-forward)",
        "! [rejected] main -> main (fetch first)",
        "hint: Updates were rejected because the tip of your current branch is behind",
    ],
)
def test_real_divergence_is_still_classified_as_non_fast_forward(error):
    from app.core.room_git_flow import is_non_fast_forward

    assert is_non_fast_forward([error]) is True


@pytest.mark.asyncio
async def test_push_force_mode_gets_no_hint():
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch(
             "app.core.workspaced_client.git_ls_tree_workspace",
             new_callable=AsyncMock, return_value=[{"path": "app.py", "size": 10}],
         ), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=None), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as push:
        push.return_value = {"pushed": 0, "errors": ["! [rejected] (non-fast-forward)"]}
        resp = await git_api.git_push(
            room_id=uuid4(), request=_request(), mode="force", body=None,
            db=AsyncMock(), current_user=_user(),
        )

    assert resp.get("hint") is None


# ── §4.2 remote_branch_exists ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_remote_branch_exists_false_on_unparseable_url():
    assert await git_forge.remote_branch_exists({"repo_url": "nope"}, "tok", "main") is False


@pytest.mark.asyncio
async def test_remote_branch_exists_false_on_non_github():
    cfg = {"repo_url": "https://gitlab.com/o/r", "provider": "gitlab"}
    assert await git_forge.remote_branch_exists(cfg, "tok", "main") is False


# ── §4.2 open_promotion PR guard ──────────────────────────────────────────

class _FakeDB:
    def add(self, obj):
        pass

    async def flush(self):
        pass


@pytest.mark.asyncio
async def test_open_promotion_pr_skips_and_hints_when_remote_main_missing(monkeypatch):
    monkeypatch.setattr("app.core.git_forge.resolve_forge_token", AsyncMock(return_value="tok"))
    monkeypatch.setattr("app.core.git_forge.remote_branch_exists", AsyncMock(return_value=False))
    created = AsyncMock()
    monkeypatch.setattr("app.core.git_forge.create_pull_request", created)
    emit = AsyncMock()
    monkeypatch.setattr("app.core.events.emit_event", emit)

    room = SimpleNamespace(git_config={"repo_url": "https://github.com/a/b"}, id=uuid4())
    cfg = room_git_flow.get_flow_config(SimpleNamespace(git_config={}))
    url = await room_git_flow._open_promotion_pr(room, _FakeDB(), cfg, 1)

    assert url is None
    created.assert_not_awaited()
    assert emit.await_args.kwargs["event_type"] == "PROMOTION_PR_SKIPPED"
    assert emit.await_args.kwargs["metadata"]["hint"] == "push_main_once"


@pytest.mark.asyncio
async def test_open_promotion_pr_seeds_main_when_flag_set(monkeypatch):
    monkeypatch.setattr("app.core.git_forge.resolve_forge_token", AsyncMock(return_value="tok"))
    monkeypatch.setattr("app.core.git_forge.remote_branch_exists", AsyncMock(return_value=False))
    monkeypatch.setattr(
        "app.core.git_forge.create_pull_request",
        AsyncMock(return_value={"ok": True, "url": "https://github.com/a/b/pull/3"}),
    )
    push = AsyncMock(return_value=True)
    monkeypatch.setattr(room_git_flow, "_push_branch_best_effort", push)

    room = SimpleNamespace(
        git_config={"repo_url": "https://github.com/a/b", "seed_main_on_first_promotion": True},
        id=uuid4(),
    )
    cfg = room_git_flow.get_flow_config(SimpleNamespace(git_config={}))
    url = await room_git_flow._open_promotion_pr(room, _FakeDB(), cfg, 1)

    assert url == "https://github.com/a/b/pull/3"
    push.assert_awaited_once()
    assert push.await_args.args[2] == cfg.release_branch
