"""A new room needs a free slot under the plan's ``max_rooms``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException


def _room_input():
    return SimpleNamespace(
        name="Limit test room",
        type="group",
        room_type="group",
        has_coordinator=False,
        coordinator_config={},
        auto_retry_enabled=True,
        auto_retry_max=3,
        user_ids=[],
        agent_ids=[],
    )


def _db(owned: int):
    db = AsyncMock()
    db.add = MagicMock()
    db.execute = AsyncMock(return_value=MagicMock(scalar_one=MagicMock(return_value=owned)))
    return db


def _limits(monkeypatch, max_rooms):
    from app.core import storage_projects, workspace_quota

    monkeypatch.setattr(storage_projects, "hard_quota_sync_enabled", lambda: False)
    monkeypatch.setattr(
        workspace_quota, "get_effective_limits",
        AsyncMock(return_value={"max_rooms": max_rooms, "plan_tier": "hobby"}),
    )


@pytest.mark.asyncio
async def test_room_over_the_plan_limit_is_refused(monkeypatch):
    from app.api.v1 import rooms

    _limits(monkeypatch, 3)
    db = _db(owned=3)

    with pytest.raises(HTTPException) as exc:
        await rooms.create_room(_room_input(), db=db, current_user=SimpleNamespace(id=uuid4()))

    assert exc.value.status_code == 403
    assert exc.value.detail["code"] == rooms.ROOM_LIMIT_REACHED
    assert exc.value.detail["max_rooms"] == 3
    assert "up to 3 rooms" in exc.value.detail["message"]
    db.add.assert_not_called()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_room_under_the_plan_limit_is_created(monkeypatch):
    from app.api.v1 import rooms

    _limits(monkeypatch, 3)
    db = _db(owned=2)

    await rooms.create_room(_room_input(), db=db, current_user=SimpleNamespace(id=uuid4()))

    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_room_cap_skips_the_count(monkeypatch):
    from app.api.v1 import rooms

    _limits(monkeypatch, None)
    db = _db(owned=100)

    await rooms.create_room(_room_input(), db=db, current_user=SimpleNamespace(id=uuid4()))

    db.commit.assert_awaited_once()
