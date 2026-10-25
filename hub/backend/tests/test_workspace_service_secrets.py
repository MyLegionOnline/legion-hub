"""workspace_service tool: Room secrets injection into service launch env
(internal design notes Stage 6).

Mirrors test_bash_secrets.py's style: Room variables always ride along,
``service.secrets`` names resolve under the Room owner (no human actor at
service-start time), and resolved values stay maskable in `logs`/`request`
output until the service stops.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.core.secrets import room_masking, room_secrets
from app.core.tools import REGISTRY, ToolContext


def _make_ctx(db):
    return ToolContext(room_id=uuid4(), session_id=uuid4(), task_id=uuid4(), db=db)


def _db_returning(obj):
    db = AsyncMock()
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=obj)
    db.execute = AsyncMock(return_value=r)
    return db


@pytest.mark.asyncio
async def test_no_secrets_returns_only_room_variables():
    tool = REGISTRY.get("workspace_service")
    ctx = _make_ctx(_db_returning(None))
    service = MagicMock()
    service.secrets = []

    with patch(
        "app.core.room_config.effective_variables",
        new_callable=AsyncMock,
        return_value={"APP_ENV": "staging"},
    ):
        room_vars, resolved = await tool._resolve_service_secrets(ctx, service)

    assert room_vars == {"APP_ENV": "staging"}
    assert resolved == {}


@pytest.mark.asyncio
async def test_declared_secret_resolves_under_room_creator():
    tool = REGISTRY.get("workspace_service")
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(_db_returning(room))
    service = MagicMock()
    service.secrets = ["DEPLOY_TOKEN"]

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={"APP_ENV": "staging"},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_env",
            new_callable=AsyncMock,
            return_value={"DEPLOY_TOKEN": "supersecretvalue"},
        ) as mock_resolve,
    ):
        room_vars, resolved = await tool._resolve_service_secrets(ctx, service)

    assert room_vars == {"APP_ENV": "staging"}
    assert resolved == {"DEPLOY_TOKEN": "supersecretvalue"}
    assert mock_resolve.await_args.kwargs["actor_user_id"] == room.creator_id
    assert mock_resolve.await_args.kwargs["names"] == ["DEPLOY_TOKEN"]
    # Explicit-name request from the declaration bypasses the scope allow-list.
    assert mock_resolve.await_args.kwargs["scope"] is None


@pytest.mark.asyncio
async def test_missing_secret_raises_tool_error():
    tool = REGISTRY.get("workspace_service")
    room = MagicMock()
    room.creator_id = uuid4()
    ctx = _make_ctx(_db_returning(room))
    service = MagicMock()
    service.secrets = ["DEPLOY_TOKEN"]

    from app.core.tools._workspace_service import WorkspaceServiceToolError

    with (
        patch(
            "app.core.room_config.effective_variables",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "app.core.secrets.room_secrets.resolve_room_env",
            new_callable=AsyncMock,
            side_effect=room_secrets.RoomSecretNotAvailable("DEPLOY_TOKEN"),
        ),
    ):
        with pytest.raises(WorkspaceServiceToolError):
            await tool._resolve_service_secrets(ctx, service)


@pytest.mark.asyncio
async def test_logs_action_masks_remembered_service_secrets():
    tool = REGISTRY.get("workspace_service")
    service_id = uuid4()
    service = MagicMock()
    service.id = service_id
    service.slug = "deploy-api"
    service.runtime_state = "ready"
    service.last_error = None
    service.log_cursor = 0

    room_masking.remember_bg_secrets(f"service:{service_id}", {"DEPLOY_TOKEN": "supersecretvalue"})
    try:
        registry = MagicMock()
        with patch(
            "app.core.workspaced_client.get_workspace_service_logs",
            new_callable=AsyncMock,
            return_value={"content": "using token supersecretvalue", "cursor": 5, "rotated": False},
        ):
            result = await tool._logs(
                _make_ctx(AsyncMock()), MagicMock(), registry, service, {"cursor": 0}
            )
    finally:
        room_masking.forget_bg_secrets(f"service:{service_id}")

    assert "supersecretvalue" not in result
    assert "***MASKED:DEPLOY_TOKEN***" in result


@pytest.mark.asyncio
async def test_stop_forgets_remembered_secrets():
    tool = REGISTRY.get("workspace_service")
    service_id = uuid4()
    service = MagicMock()
    service.id = service_id
    service.pid = 123
    service.pgid = 123
    service.runtime_state = "ready"

    room_masking.remember_bg_secrets(f"service:{service_id}", {"DEPLOY_TOKEN": "supersecretvalue"})

    registry = AsyncMock()
    wsd = AsyncMock()
    ctx = _make_ctx(AsyncMock())
    await tool._stop(ctx, service, registry, wsd)

    assert room_masking.bg_secrets(f"service:{service_id}") == {}
