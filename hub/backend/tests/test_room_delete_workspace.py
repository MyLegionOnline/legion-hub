"""Room deletion must destroy the external workspace before DB cleanup."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.api.v1.rooms import delete_room
from app.models.room import Room


class _Result:
    def __init__(self, room=None, scalar=None):
        self._room = room
        self._scalar = scalar

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def first(self):
        return self._room

    def all(self):
        return []


@pytest.mark.asyncio
async def test_delete_room_destroys_workspace_before_removing_room_data():
    room_id = uuid.uuid4()
    user = MagicMock(id=uuid.uuid4())
    room = Room(id=room_id, creator_id=user.id, name="Disposable project")

    db = MagicMock()
    db.execute = AsyncMock(side_effect=[_Result(room)] + [_Result() for _ in range(40)])
    db.commit = AsyncMock()
    db.delete = AsyncMock()
    db.add = MagicMock()

    with patch(
        "app.core.workspaced_client.destroy_workspace",
        new=AsyncMock(return_value={"status": "destroyed"}),
    ) as destroy:
        await delete_room(room_id, db, user)

    destroy.assert_awaited_once_with(str(room_id))
    assert room.deletion_started_at is not None
    assert db.commit.await_count >= 2
    db.delete.assert_awaited_once_with(room)


@pytest.mark.asyncio
async def test_delete_room_takes_its_archive_along():
    """internal design notes §5.5 (T11): an archived room's archive goes too."""
    from datetime import datetime, timezone

    room_id = uuid.uuid4()
    user = MagicMock(id=uuid.uuid4())
    room = Room(id=room_id, creator_id=user.id, name="Archived project")

    db = MagicMock()
    db.execute = AsyncMock(side_effect=[_Result(room), _Result(scalar=datetime.now(timezone.utc))]
                           + [_Result() for _ in range(40)])
    db.commit = AsyncMock()
    db.delete = AsyncMock()
    db.add = MagicMock()

    with patch(
        "app.core.workspaced_client.destroy_workspace",
        new=AsyncMock(return_value={"status": "destroyed"}),
    ), patch(
        "app.core.workspaced_client.delete_workspace_archive",
        new=AsyncMock(return_value={"deleted": True}),
    ) as drop_archive:
        await delete_room(room_id, db, user)

    drop_archive.assert_awaited_once_with(str(room_id))
    db.delete.assert_awaited_once_with(room)


@pytest.mark.asyncio
async def test_delete_room_keeps_marker_when_workspaced_is_unavailable():
    room_id = uuid.uuid4()
    user = MagicMock(id=uuid.uuid4())
    room = Room(id=room_id, creator_id=user.id, name="Retry project")

    db = MagicMock()
    db.execute = AsyncMock(return_value=_Result(room))
    db.commit = AsyncMock()
    db.delete = AsyncMock()
    db.add = MagicMock()

    with patch(
        "app.core.workspaced_client.destroy_workspace",
        new=AsyncMock(side_effect=httpx.RequestError("daemon unavailable")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await delete_room(room_id, db, user)

    assert exc_info.value.status_code == 503
    assert room.deletion_started_at is not None
    db.delete.assert_not_awaited()
