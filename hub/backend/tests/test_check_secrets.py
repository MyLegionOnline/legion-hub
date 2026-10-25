"""Gate 1/2 CI checks: Room secrets injection + output masking
(internal design notes Stage 6).

``_resolve_checks_env`` reads ``checks[].secrets`` names from
``.legion/run.json`` (via workspaced) and resolves them under the Room
owner — checks have no human actor, same reasoning as bash's system path.
``_mask_check_run`` then redacts those values out of the checks-run result
before it reaches the CI log archive or a chat message.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.room_git_flow import _mask_check_run, _resolve_checks_env
from app.core.secrets import room_secrets


def _db_returning(obj):
    db = AsyncMock()
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=obj)
    db.execute = AsyncMock(return_value=r)
    return db


@pytest.mark.asyncio
async def test_no_declared_secrets_returns_only_room_variables():
    room_id = uuid4()
    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={"APP_ENV": "staging"},
        ),
        patch(
            "app.core.workspaced_client.get_workspace_runs",
            new_callable=AsyncMock,
            return_value={"runs": [], "checks": [{"name": "lint", "command": "ruff check .", "quick": True}]},
        ),
    ):
        env, secrets = await _resolve_checks_env(room_id, _db_returning(None), "quick")

    assert env == {"APP_ENV": "staging"}
    assert secrets == {}


@pytest.mark.asyncio
async def test_quick_scope_skips_non_quick_checks():
    room_id = uuid4()
    room = MagicMock()
    room.creator_id = uuid4()
    checks = [
        {"name": "pytest", "command": "pytest -q", "quick": True, "secrets": ["DEPLOY_TOKEN"]},
        {"name": "build", "command": "docker build .", "quick": False, "secrets": ["DOCKER_PASSWORD"]},
    ]
    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.workspaced_client.get_workspace_runs",
            new_callable=AsyncMock,
            return_value={"runs": [], "checks": checks},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_env",
            new_callable=AsyncMock,
            return_value={"DEPLOY_TOKEN": "supersecretvalue"},
        ) as mock_resolve,
    ):
        env, secrets = await _resolve_checks_env(room_id, _db_returning(room), "quick")

    assert mock_resolve.await_args.kwargs["names"] == ["DEPLOY_TOKEN"]
    assert mock_resolve.await_args.kwargs["actor_user_id"] == room.creator_id
    # Explicit-name request from the check's own declaration bypasses scope.
    assert mock_resolve.await_args.kwargs["scope"] is None
    assert env == {"DEPLOY_TOKEN": "supersecretvalue"}
    assert secrets == {"DEPLOY_TOKEN": "supersecretvalue"}


@pytest.mark.asyncio
async def test_full_scope_includes_non_quick_checks():
    room_id = uuid4()
    room = MagicMock()
    room.creator_id = uuid4()
    checks = [
        {"name": "pytest", "command": "pytest -q", "quick": True, "secrets": ["DEPLOY_TOKEN"]},
        {"name": "build", "command": "docker build .", "quick": False, "secrets": ["DOCKER_PASSWORD"]},
    ]
    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.workspaced_client.get_workspace_runs",
            new_callable=AsyncMock,
            return_value={"runs": [], "checks": checks},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_env",
            new_callable=AsyncMock,
            return_value={"DEPLOY_TOKEN": "tok", "DOCKER_PASSWORD": "pw"},
        ) as mock_resolve,
    ):
        await _resolve_checks_env(room_id, _db_returning(room), "full")

    assert set(mock_resolve.await_args.kwargs["names"]) == {"DEPLOY_TOKEN", "DOCKER_PASSWORD"}


@pytest.mark.asyncio
async def test_workspaced_unreachable_fails_open():
    room_id = uuid4()
    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={"APP_ENV": "staging"},
        ),
        patch(
            "app.core.workspaced_client.get_workspace_runs",
            new_callable=AsyncMock,
            side_effect=RuntimeError("workspaced down"),
        ),
    ):
        env, secrets = await _resolve_checks_env(room_id, _db_returning(None), "quick")

    assert env == {"APP_ENV": "staging"}
    assert secrets == {}


@pytest.mark.asyncio
async def test_secret_resolution_failure_fails_open():
    room_id = uuid4()
    room = MagicMock()
    room.creator_id = uuid4()
    checks = [{"name": "pytest", "command": "pytest -q", "quick": True, "secrets": ["DEPLOY_TOKEN"]}]
    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.workspaced_client.get_workspace_runs",
            new_callable=AsyncMock,
            return_value={"runs": [], "checks": checks},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_env",
            new_callable=AsyncMock,
            side_effect=room_secrets.RoomSecretNotAvailable("DEPLOY_TOKEN"),
        ),
    ):
        env, secrets = await _resolve_checks_env(room_id, _db_returning(room), "quick")

    assert env == {} and secrets == {}


def test_mask_check_run_redacts_tail_and_error():
    run = {
        "passed": False,
        "ran": True,
        "results": [
            {"name": "pytest", "command": "pytest -q", "exit_code": 1, "tail": "token=supersecretvalue failed", "timed_out": False},
        ],
        "error": "deploy failed with supersecretvalue",
    }
    out = _mask_check_run(run, {"DEPLOY_TOKEN": "supersecretvalue"})

    assert "supersecretvalue" not in out["results"][0]["tail"]
    assert "***MASKED:DEPLOY_TOKEN***" in out["results"][0]["tail"]
    assert "supersecretvalue" not in out["error"]


def test_mask_check_run_is_noop_without_secrets():
    run = {"results": [{"tail": "hello world"}], "error": None}
    out = _mask_check_run(run, {})
    assert out["results"][0]["tail"] == "hello world"
