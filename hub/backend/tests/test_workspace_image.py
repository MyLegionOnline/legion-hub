"""Unit tests for core.workspace_image.resolve_workspace_image.

internal design notes Stage 5 §7.1. No DB required — db.execute is
mocked per [[feedback_pytest_windows_env]] / [[feedback_no_tests_in_prod]].
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.workspace_image import (
    WORKSPACE_IMAGE_BASE,
    WORKSPACE_IMAGE_OFFICE,
    resolve_workspace_image,
)

ROOM_ID = uuid.uuid4()


def _agent_ids_result(ids: list[uuid.UUID]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.all.return_value = ids
    return result


def _office_check_result(found: bool) -> MagicMock:
    result = MagicMock()
    result.first.return_value = (uuid.uuid4(),) if found else None
    return result


def _db(side_effects: list) -> AsyncMock:
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=side_effects)
    return db


@pytest.mark.asyncio
async def test_no_task_assigned_agents_returns_base():
    db = _db([_agent_ids_result([])])

    image = await resolve_workspace_image(ROOM_ID, db)

    assert image == WORKSPACE_IMAGE_BASE
    assert db.execute.await_count == 1  # second query never fires


@pytest.mark.asyncio
async def test_assigned_agent_without_executable_skill_returns_base():
    agent_id = uuid.uuid4()
    db = _db([_agent_ids_result([agent_id]), _office_check_result(False)])

    image = await resolve_workspace_image(ROOM_ID, db)

    assert image == WORKSPACE_IMAGE_BASE


@pytest.mark.asyncio
async def test_assigned_agent_with_executable_skill_returns_office():
    agent_id = uuid.uuid4()
    db = _db([_agent_ids_result([agent_id]), _office_check_result(True)])

    image = await resolve_workspace_image(ROOM_ID, db)

    assert image == WORKSPACE_IMAGE_OFFICE


@pytest.mark.asyncio
async def test_query_failure_falls_back_to_base_without_raising():
    db = AsyncMock()
    db.execute = AsyncMock(side_effect=RuntimeError("db down"))

    image = await resolve_workspace_image(ROOM_ID, db)

    assert image == WORKSPACE_IMAGE_BASE
