"""Unit tests for workspace_lifecycle._restart_for_image_mismatch.

internal design notes Stage 5 §7.1, "Note on changing the image":
a running workspace whose task-assigned agents now need the office image
(e.g. an executable skill was assigned after the container started on base)
is paused, stopped and picked up fresh by the normal open/start flow.

No DB required — db.execute and the collaborator functions are mocked, per
[[feedback_pytest_windows_env]] / [[feedback_no_tests_in_prod]].
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core import workspace_lifecycle as wl
from app.models.workspace import Workspace

ROOM_ID = uuid.uuid4()


def _db_returning_workspace(ws) -> AsyncMock:
    result = MagicMock()
    result.scalar_one_or_none.return_value = ws
    db = AsyncMock()
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_noop_when_workspace_not_running():
    ws = Workspace(room_id=ROOM_ID, status="stopped", image="legion-workspace:base")
    db = _db_returning_workspace(ws)

    with patch("app.core.workspace_image.resolve_workspace_image") as resolve_mock, \
         patch.object(wl, "pause_workspace_tasks") as pause_mock, \
         patch.object(wl, "stop_workspace") as stop_mock:
        await wl._restart_for_image_mismatch(ROOM_ID, db)

    resolve_mock.assert_not_called()
    pause_mock.assert_not_called()
    stop_mock.assert_not_called()


@pytest.mark.asyncio
async def test_noop_when_image_already_matches():
    ws = Workspace(room_id=ROOM_ID, status="running", image="legion-workspace:office")
    db = _db_returning_workspace(ws)

    with patch(
        "app.core.workspace_image.resolve_workspace_image",
        new=AsyncMock(return_value="legion-workspace:office"),
    ), patch.object(wl, "pause_workspace_tasks") as pause_mock, \
       patch.object(wl, "stop_workspace") as stop_mock:
        await wl._restart_for_image_mismatch(ROOM_ID, db)

    pause_mock.assert_not_called()
    stop_mock.assert_not_called()


@pytest.mark.asyncio
async def test_pauses_and_stops_when_office_skill_needs_a_different_image():
    ws = Workspace(room_id=ROOM_ID, status="running", image="legion-workspace:base")
    db = _db_returning_workspace(ws)

    with patch(
        "app.core.workspace_image.resolve_workspace_image",
        new=AsyncMock(return_value="legion-workspace:office"),
    ), patch.object(wl, "pause_workspace_tasks", new=AsyncMock()) as pause_mock, \
       patch.object(wl, "stop_workspace", new=AsyncMock()) as stop_mock:
        await wl._restart_for_image_mismatch(ROOM_ID, db)

    pause_mock.assert_awaited_once_with(ROOM_ID, db, pause_reason="image_switch")
    stop_mock.assert_awaited_once_with(ROOM_ID, db, reason="image_switch", force=True)


@pytest.mark.asyncio
async def test_unknown_image_is_stamped_not_restarted():
    """A workspace row with no recorded image (legacy / reconciled) must be
    stamped, never force-restarted — otherwise every room open bounces a
    healthy container."""
    ws = Workspace(room_id=ROOM_ID, status="running", image=None)
    db = _db_returning_workspace(ws)
    db.add = MagicMock()  # SQLAlchemy's add is sync

    with patch(
        "app.core.workspace_image.resolve_workspace_image",
        new=AsyncMock(return_value="legion-workspace:base"),
    ), patch.object(wl, "pause_workspace_tasks", new=AsyncMock()) as pause_mock, \
       patch.object(wl, "stop_workspace", new=AsyncMock()) as stop_mock:
        await wl._restart_for_image_mismatch(ROOM_ID, db)

    assert ws.image == "legion-workspace:base"
    db.commit.assert_awaited()
    pause_mock.assert_not_called()
    stop_mock.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_failure_does_not_restart():
    ws = Workspace(room_id=ROOM_ID, status="running", image="legion-workspace:base")
    db = _db_returning_workspace(ws)

    with patch(
        "app.core.workspace_image.resolve_workspace_image",
        new=AsyncMock(side_effect=RuntimeError("db down")),
    ), patch.object(wl, "pause_workspace_tasks") as pause_mock, \
       patch.object(wl, "stop_workspace") as stop_mock:
        await wl._restart_for_image_mismatch(ROOM_ID, db)

    pause_mock.assert_not_called()
    stop_mock.assert_not_called()


@pytest.mark.asyncio
async def test_stop_failure_is_swallowed():
    ws = Workspace(room_id=ROOM_ID, status="running", image="legion-workspace:base")
    db = _db_returning_workspace(ws)

    with patch(
        "app.core.workspace_image.resolve_workspace_image",
        new=AsyncMock(return_value="legion-workspace:office"),
    ), patch.object(wl, "pause_workspace_tasks", new=AsyncMock()), \
       patch.object(wl, "stop_workspace", new=AsyncMock(side_effect=RuntimeError("workspaced down"))):
        await wl._restart_for_image_mismatch(ROOM_ID, db)  # must not raise
