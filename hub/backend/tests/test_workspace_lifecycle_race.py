"""Regression test: two near-simultaneous POST /rooms/{id}/open calls used to
race _get_or_create_workspace's SELECT-then-INSERT and blow up with
asyncpg.UniqueViolationError on ix_workspace_room_id (real incident — see
internal design notes session notes). The room's workspace never
came up at all afterwards. No DB required: AsyncSession is mocked.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.core.workspace_lifecycle import _get_or_create_workspace


def _scalar_first(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
async def test_returns_existing_workspace_without_inserting():
    ws = MagicMock()
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(ws))

    result = await _get_or_create_workspace(uuid.uuid4(), db)

    assert result is ws
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_creates_workspace_when_none_exists():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalar_first(None))

    @asynccontextmanager
    async def _begin_nested():
        yield

    db.begin_nested = _begin_nested
    db.add = MagicMock()
    db.flush = AsyncMock()

    room_id = uuid.uuid4()
    result = await _get_or_create_workspace(room_id, db)

    assert result.room_id == room_id
    db.add.assert_called_once()
    db.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_losing_a_create_race_reads_back_the_winners_row_instead_of_raising():
    """Concurrent request A already committed a Workspace row for this room
    between our SELECT and our INSERT — the flush inside the savepoint raises
    IntegrityError, and we must recover by reading A's row, not propagate a
    500 (the actual bug: POST /open returned 500 and the workspace was never
    started)."""
    winners_row = MagicMock()
    db = AsyncMock()
    db.execute = AsyncMock(
        side_effect=[_scalar_first(None), _scalar_first(winners_row)]
    )

    @asynccontextmanager
    async def _begin_nested():
        yield
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    db.begin_nested = _begin_nested
    db.add = MagicMock()
    db.flush = AsyncMock()

    result = await _get_or_create_workspace(uuid.uuid4(), db)

    assert result is winners_row
    assert db.execute.await_count == 2


@pytest.mark.asyncio
async def test_reraises_if_the_row_still_cannot_be_found_after_losing_the_race():
    """If the retry SELECT still finds nothing, the IntegrityError was for
    some other reason — don't swallow it."""
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[_scalar_first(None), _scalar_first(None)])

    @asynccontextmanager
    async def _begin_nested():
        yield
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    db.begin_nested = _begin_nested
    db.add = MagicMock()
    db.flush = AsyncMock()

    with pytest.raises(IntegrityError):
        await _get_or_create_workspace(uuid.uuid4(), db)


def _losing_race_db(winners_row):
    db = AsyncMock()
    db.execute = AsyncMock(
        side_effect=[_scalar_first(None), _scalar_first(winners_row)]
    )

    @asynccontextmanager
    async def _begin_nested():
        yield
        raise IntegrityError("INSERT", {}, Exception("duplicate key"))

    db.begin_nested = _begin_nested
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


@pytest.mark.asyncio
async def test_workspace_api_helper_returns_the_winners_row_on_a_lost_race():
    """GET /rooms/{id}/workspace and friends used their own select-then-insert
    with db.commit(); two of them firing as a new room opened made the loser
    fail with "duplicate key ... ix_workspace_room_id" (2026-10-04)."""
    from app.api.v1 import workspace as workspace_api

    winners_row = MagicMock()
    db = _losing_race_db(winners_row)

    result = await workspace_api._get_or_create_workspace(uuid.uuid4(), db)

    assert result is winners_row
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_workspace_service_tool_returns_the_winners_row_on_a_lost_race():
    from types import SimpleNamespace

    from app.core.tools._workspace_service import WorkspaceServiceTool

    winners_row = MagicMock()
    ctx = SimpleNamespace(room_id=uuid.uuid4(), db=_losing_race_db(winners_row))

    result = await WorkspaceServiceTool._workspace(MagicMock(), ctx)

    assert result is winners_row
