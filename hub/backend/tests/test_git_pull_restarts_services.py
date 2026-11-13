"""A Pull that moves the checked-out commit restarts running services.

A declared service (e.g. a Flask app without the reloader) keeps serving the
code it loaded at launch, and the supervisor's Start is a no-op on a live
process — so before this a Pull showed "Changes pulled" while the Browser tab
kept serving the pre-pull version.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.api.v1 import git as git_api
from app.api.v1 import workspace as ws_api


def _room():
    return MagicMock(
        git_config={"repo_url": "https://github.com/o/r", "branch": "main", "provider": "github"},
        creator_id=uuid4(),
    )


async def _sync(fetch_result, restart_mock):
    token_value = MagicMock()
    token_value.reveal.return_value = "t"
    with patch.object(git_api, "_verify_membership", new_callable=AsyncMock), \
         patch.object(git_api, "_get_room_with_git", new_callable=AsyncMock, return_value=_room()), \
         patch.object(git_api, "_resolve_git_access_token", new_callable=AsyncMock, return_value=token_value), \
         patch.object(git_api, "apply_userfiles_policy", new_callable=AsyncMock), \
         patch.object(git_api, "fetch_workspace_from_git", new_callable=AsyncMock, return_value=fetch_result), \
         patch.object(ws_api, "restart_running_services", restart_mock):
        return await git_api.git_sync(
            room_id=uuid4(), request=MagicMock(), db=AsyncMock(), current_user=MagicMock(id=uuid4()),
        )


@pytest.mark.asyncio
async def test_pull_with_new_commits_restarts_running_services():
    restart = AsyncMock(return_value=["tracker"])
    result = await _sync({"synced": 1, "errors": [], "head_changed": True}, restart)
    restart.assert_awaited_once()
    assert result.ok is True
    assert result.services_restarted == ["tracker"]


@pytest.mark.asyncio
async def test_noop_pull_leaves_services_alone():
    restart = AsyncMock(return_value=["tracker"])
    result = await _sync({"synced": 1, "errors": [], "head_changed": False}, restart)
    restart.assert_not_awaited()
    assert result.services_restarted == []


@pytest.mark.asyncio
async def test_restart_failure_does_not_fail_the_pull():
    restart = AsyncMock(side_effect=RuntimeError("supervisor down"))
    result = await _sync({"synced": 1, "errors": [], "head_changed": True}, restart)
    assert result.ok is True
    assert result.services_restarted == []


@pytest.mark.asyncio
async def test_restart_running_services_restarts_only_live_ones():
    def svc(slug, state, deps=()):
        return SimpleNamespace(slug=slug, runtime_state=state, depends_on=list(deps))

    services = [svc("web", "ready", ["api"]), svc("api", "ready"), svc("old", "stopped"), svc("bad", "error")]
    registry = MagicMock()
    registry.list = AsyncMock(return_value=services)
    workspace = SimpleNamespace(id=uuid4(), status="running")
    start = AsyncMock()
    with patch.object(ws_api, "services_enabled", return_value=True), \
         patch.object(ws_api, "_get_or_create_workspace", new_callable=AsyncMock, return_value=workspace), \
         patch.object(ws_api, "WorkspaceServiceRegistry", return_value=registry), \
         patch.object(ws_api, "_reconcile_service_runtime", new_callable=AsyncMock), \
         patch.object(ws_api, "_start_declared_service", start):
        restarted = await ws_api.restart_running_services(uuid4(), AsyncMock(), MagicMock(), source="t")

    # Dependency first; stopped/error services are not brought up by a Pull.
    assert restarted == ["api", "web"]
    assert all(call.kwargs["restart"] is True for call in start.await_args_list)


@pytest.mark.asyncio
async def test_restart_running_services_skips_stopped_workspace():
    workspace = SimpleNamespace(id=uuid4(), status="stopped")
    start = AsyncMock()
    with patch.object(ws_api, "services_enabled", return_value=True), \
         patch.object(ws_api, "_get_or_create_workspace", new_callable=AsyncMock, return_value=workspace), \
         patch.object(ws_api, "_start_declared_service", start):
        assert await ws_api.restart_running_services(uuid4(), AsyncMock(), MagicMock(), source="t") == []
    start.assert_not_awaited()
