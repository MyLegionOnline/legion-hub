"""Stage 6 of internal design notes — push an arbitrary branch to the remote.

Covers:
  * ``git_sync.push_workspace_to_git`` picks the explicit ``branch`` argument
    over ``git_config["branch"]`` (and falls back to it when omitted);
  * ``POST /rooms/{id}/git/push`` forwards an optional body ``{"branch": ...}``
    to ``push_workspace_to_git`` and defaults to ``None`` (configured branch);
  * ``room_git_flow.open_promotion`` pushes the working branch for an external
    CI only under ``ci_provider == "github_actions"`` and records the outcome
    in ``Promotion.dev_pushed``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import git as git_api
from app.core import git_sync, room_git_flow


# ── push_workspace_to_git: branch selection ─────────────────────────────────

@pytest.mark.asyncio
async def test_push_workspace_uses_explicit_branch_over_config():
    with patch("app.core.workspaced_client.git_push_workspace", new_callable=AsyncMock) as mp:
        mp.return_value = {"ok": True}
        await git_sync.push_workspace_to_git(
            room_id=uuid4(),
            git_config={"repo_url": "https://github.com/o/r", "branch": "main"},
            branch="dev",
        )
    assert mp.await_args.kwargs["branch"] == "dev"


@pytest.mark.asyncio
async def test_push_workspace_falls_back_to_config_branch():
    with patch("app.core.workspaced_client.git_push_workspace", new_callable=AsyncMock) as mp:
        mp.return_value = {"ok": True}
        await git_sync.push_workspace_to_git(
            room_id=uuid4(),
            git_config={"repo_url": "https://github.com/o/r", "branch": "main"},
        )
    assert mp.await_args.kwargs["branch"] == "main"


# ── POST /git/push: optional {branch} body ──────────────────────────────────

def _room():
    return MagicMock(git_config={"repo_url": "https://github.com/o/r", "branch": "main"})


def _user():
    return MagicMock(id=uuid4(), email="u@e")


def _request():
    return MagicMock(client=MagicMock(host="127.0.0.1"))


@pytest.mark.asyncio
async def test_push_endpoint_forwards_branch_body():
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "_branch_has_content", new_callable=AsyncMock, return_value=True), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as mp:
        mp.return_value = {"pushed": 1, "errors": []}
        await git_api.git_push(
            room_id=uuid4(),
            request=_request(),
            body=git_api.GitPushBody(branch="dev"),
            db=AsyncMock(),
            current_user=_user(),
        )

    assert mp.await_args.kwargs["branch"] == "dev"


@pytest.mark.asyncio
async def test_push_endpoint_defaults_branch_to_none():
    token_value = MagicMock()
    token_value.reveal.return_value = "ghp"

    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "_branch_has_content", new_callable=AsyncMock, return_value=True), \
         patch.object(git_api, "push_workspace_to_git", new_callable=AsyncMock) as mp:
        mp.return_value = {"pushed": 1, "errors": []}
        await git_api.git_push(
            room_id=uuid4(), request=_request(), db=AsyncMock(), current_user=_user()
        )

    assert mp.await_args.kwargs["branch"] is None


# ── open_promotion: push the working branch for external CI ──────────────────

class _FakeDB:
    def add(self, obj):  # noqa: D401
        pass

    async def flush(self):
        pass


@pytest.fixture(autouse=True)
def _stub_promotion(monkeypatch):
    monkeypatch.setattr(room_git_flow, "_open_promotion_row", AsyncMock(return_value=None))
    monkeypatch.setattr(room_git_flow, "_promotable_task_ids", AsyncMock(return_value=["t1"]))
    monkeypatch.setattr(room_git_flow, "_task_titles", AsyncMock(return_value=["A"]))
    monkeypatch.setattr(
        "app.core.workspaced_client.git_diff_workspace",
        AsyncMock(return_value={"diff": " a.py | 2 +-\n 1 file changed"}),
    )
    monkeypatch.setattr(
        "app.core.workspaced_client.run_workspace_checks",
        AsyncMock(return_value={"ran": False, "passed": True, "results": []}),
    )
    monkeypatch.setattr("app.core.events.emit_event", AsyncMock())


@pytest.mark.asyncio
async def test_open_promotion_pushes_dev_under_github_actions(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={
            "ci_provider": "github_actions",
            "repo_url": "https://github.com/o/r",
        })),
    )
    push = AsyncMock(return_value={"pushed": 1, "errors": []})
    monkeypatch.setattr("app.core.git_sync.push_workspace_to_git", push)

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())

    push.assert_awaited_once()
    assert push.await_args.kwargs["branch"] == "dev"
    assert push.await_args.kwargs["mode"] == "push"
    assert promo.dev_pushed is True


@pytest.mark.asyncio
async def test_open_promotion_marks_dev_pushed_false_on_push_error(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={
            "ci_provider": "github_actions",
            "repo_url": "https://github.com/o/r",
        })),
    )
    monkeypatch.setattr(
        "app.core.git_sync.push_workspace_to_git",
        AsyncMock(return_value={"pushed": 0, "errors": ["rejected"]}),
    )

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())
    assert promo.dev_pushed is False


@pytest.mark.asyncio
async def test_open_promotion_does_not_push_under_workspace_provider(monkeypatch):
    monkeypatch.setattr(
        room_git_flow, "_load_room",
        AsyncMock(return_value=SimpleNamespace(git_config={})),  # ci_provider → "workspace"
    )
    push = AsyncMock()
    monkeypatch.setattr("app.core.git_sync.push_workspace_to_git", push)

    promo = await room_git_flow.open_promotion(str(uuid4()), _FakeDB())

    push.assert_not_awaited()
    assert promo.dev_pushed is False
