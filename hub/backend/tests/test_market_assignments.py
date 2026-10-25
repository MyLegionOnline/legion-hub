"""Security boundaries for room MCP and agent Skill assignments."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.v1 import agents, rooms
from app.models.market import UserInstalledItem


def _scalar_first(value):
    result = MagicMock()
    result.scalars.return_value.first.return_value = value
    return result


@pytest.mark.asyncio
async def test_cannot_assign_mcp_to_a_room_the_user_does_not_belong_to():
    user = SimpleNamespace(id=uuid4())
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(None))

    with pytest.raises(HTTPException) as exc:
        await rooms.assign_room_mcp_server(uuid4(), uuid4(), db, user)

    assert exc.value.status_code == 403
    db.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_cannot_assign_another_users_mcp_to_an_owned_room():
    user = SimpleNamespace(id=uuid4())
    foreign_item = UserInstalledItem(user_id=uuid4(), type="mcp", name="Foreign MCP")
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(SimpleNamespace()))
    db.get = AsyncMock(return_value=foreign_item)

    with pytest.raises(HTTPException) as exc:
        await rooms.assign_room_mcp_server(uuid4(), foreign_item.id, db, user)

    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_cannot_assign_skill_to_another_users_agent():
    user = SimpleNamespace(id=uuid4())
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(None))

    with pytest.raises(HTTPException) as exc:
        await agents.assign_agent_skill(uuid4(), uuid4(), db, user)

    assert exc.value.status_code == 403
    db.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_cannot_assign_another_users_skill_to_an_owned_agent():
    user = SimpleNamespace(id=uuid4())
    foreign_item = UserInstalledItem(
        user_id=uuid4(), type="skill", name="Foreign skill", skill_id=uuid4()
    )
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(SimpleNamespace()))
    db.get = AsyncMock(return_value=foreign_item)

    with pytest.raises(HTTPException) as exc:
        await agents.assign_agent_skill(uuid4(), foreign_item.id, db, user)

    assert exc.value.status_code == 403
