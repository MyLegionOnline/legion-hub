"""Backfill of pre-existing uploads into UserInstalledItem
(internal design notes stage 4)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.skill_backfill import _backfill_uploaded_items
from app.models.market import UserInstalledItem
from app.models.skill import Skill


def _scalars_all(values):
    result = MagicMock()
    result.scalars.return_value.all.return_value = values
    return result


def _scalar_one_or_none(value):
    result = MagicMock()
    result.scalar_one_or_none.return_value = value
    return result


@pytest.mark.asyncio
async def test_backfill_creates_installed_item_for_skill_missing_one():
    owner_id = uuid4()
    skill = Skill(
        id=uuid4(), owner_id=owner_id, name="mine", display_name="Mine",
        source="upload", version="1.0.0",
    )

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _scalars_all([skill]),        # select owned upload skills
        _scalar_one_or_none(None),    # no installed item yet
    ])
    db.add = MagicMock()

    created = await _backfill_uploaded_items(db)

    assert created == 1
    added = db.add.call_args.args[0]
    assert isinstance(added, UserInstalledItem)
    assert added.user_id == owner_id
    assert added.skill_id == skill.id
    assert added.is_custom is True
    assert added.name == "Mine"
    db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_skips_skill_that_already_has_installed_item():
    owner_id = uuid4()
    skill = Skill(id=uuid4(), owner_id=owner_id, name="mine", source="upload", version="1.0.0")

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[
        _scalars_all([skill]),
        _scalar_one_or_none(uuid4()),  # already has an installed item
    ])
    db.add = MagicMock()

    created = await _backfill_uploaded_items(db)

    assert created == 0
    db.add.assert_not_called()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_is_noop_with_no_uploaded_skills():
    db = AsyncMock()
    db.execute = AsyncMock(return_value=_scalars_all([]))
    db.add = MagicMock()

    created = await _backfill_uploaded_items(db)

    assert created == 0
    db.add.assert_not_called()
    db.commit.assert_not_awaited()
