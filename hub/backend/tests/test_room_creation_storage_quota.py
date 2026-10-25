"""T9: a cloud room is not visible before its XFS project is prepared."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException


def _room_input():
    return SimpleNamespace(
        name="Quota test room",
        type="group",
        room_type="group",
        has_coordinator=False,
        coordinator_config={},
        user_ids=[],
        agent_ids=[],
    )


def _db():
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_new_cloud_room_syncs_storage_project_before_commit(monkeypatch):
    from app.api.v1 import rooms
    from app.core import storage_projects

    owner = uuid4()
    db = _db()
    sync = AsyncMock(return_value=True)
    monkeypatch.setattr(storage_projects, "hard_quota_sync_enabled", lambda: True)
    monkeypatch.setattr(storage_projects, "sync_owner_project", sync)

    await rooms.create_room(
        _room_input(), db=db, current_user=SimpleNamespace(id=owner)
    )

    sync.assert_awaited_once_with(owner, db)
    db.commit.assert_awaited_once()
    db.rollback.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_cloud_room_rolls_back_when_storage_quota_cannot_be_prepared(monkeypatch):
    from app.api.v1 import rooms
    from app.core import storage_projects

    db = _db()
    monkeypatch.setattr(storage_projects, "hard_quota_sync_enabled", lambda: True)
    monkeypatch.setattr(
        storage_projects, "sync_owner_project", AsyncMock(side_effect=OSError("xfs down"))
    )

    with pytest.raises(HTTPException) as exc:
        await rooms.create_room(
            _room_input(), db=db, current_user=SimpleNamespace(id=uuid4())
        )

    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "storage_quota_unavailable"
    db.rollback.assert_awaited_once()
    db.commit.assert_not_awaited()
