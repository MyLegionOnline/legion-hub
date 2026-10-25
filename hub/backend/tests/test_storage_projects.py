"""T9 owner-to-XFS-project mapping, without a real PostgreSQL server."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core import storage_projects as sp


def _result(*, one_or_none=None, one=None, rows=None):
    result = MagicMock()
    result.scalar_one_or_none.return_value = one_or_none
    result.scalar_one.return_value = one
    result.scalars.return_value.all.return_value = rows or []
    return result


def _db(*results):
    db = AsyncMock()
    # The production allocation lock is PostgreSQL-only.  This mock validates
    # the mapping semantics without pretending SQLite has that primitive.
    # ``AsyncSession.get_bind`` is synchronous, so a plain MagicMock is needed.
    db.get_bind = MagicMock(return_value=SimpleNamespace(dialect=SimpleNamespace(name="sqlite")))
    db.execute = AsyncMock(side_effect=list(results))
    db.add = MagicMock()
    db.flush = AsyncMock()
    return db


def test_finite_positive_mb_rejects_unlimited_and_fractional_values():
    assert sp.finite_positive_mb(3072) == 3072
    assert sp.finite_positive_mb(None) is None
    assert sp.finite_positive_mb(0) is None
    assert sp.finite_positive_mb(1.5) is None
    assert sp.finite_positive_mb(True) is None


@pytest.mark.asyncio
async def test_project_for_owner_reuses_the_stable_id_and_all_live_rooms():
    owner, first_room, second_room = uuid4(), uuid4(), uuid4()
    quota = SimpleNamespace(xfs_project_id=10042)
    db = _db(
        _result(one_or_none=quota),
        _result(rows=[first_room, second_room]),
    )

    project = await sp.project_for_owner(
        owner,
        db,
        limits={"room_storage_quota_mb": 3072},
    )

    assert project == sp.StorageProject(10042, 3072, [str(first_room), str(second_room)])
    db.add.assert_not_called()
    db.flush.assert_not_awaited()


@pytest.mark.asyncio
async def test_project_for_owner_skips_the_database_for_unlimited_storage():
    db = _db()

    assert await sp.project_for_owner(
        uuid4(), db, limits={"room_storage_quota_mb": None}
    ) is None
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_project_id_is_persisted_once():
    owner = uuid4()
    quota = SimpleNamespace(xfs_project_id=None)
    db = _db(
        _result(one_or_none=quota),
        _result(one=10042),
    )

    assert await sp.ensure_xfs_project_id(owner, db) == 10042
    assert quota.xfs_project_id == 10042
    db.add.assert_called_once_with(quota)
    db.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_sends_a_large_room_list_in_batches(monkeypatch):
    rooms = [str(uuid4()) for _ in range(sp.ROOM_IDS_PER_CALL * 2 + 3)]
    project = sp.StorageProject(10042, 3072, rooms)
    monkeypatch.setattr(sp, "project_for_owner", AsyncMock(return_value=project))
    calls = []

    async def configure(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("app.core.workspaced_client.configure_storage_project", configure)

    assert await sp.sync_owner_project(uuid4(), AsyncMock()) is True
    assert [len(c["room_ids"]) for c in calls] == [sp.ROOM_IDS_PER_CALL, sp.ROOM_IDS_PER_CALL, 3]
    assert {c["quota_mb"] for c in calls} == {3072}


@pytest.mark.asyncio
async def test_reconcile_continues_after_one_owner_fails(monkeypatch):
    owners = [uuid4(), uuid4(), uuid4()]

    class _Session:
        def __init__(self):
            self.execute = AsyncMock(return_value=_result(rows=owners))
            self.commit = AsyncMock()
            self.rollback = AsyncMock()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(sp, "async_sessionmaker", lambda *a, **k: _Session)

    async def sync(owner_id, db):
        if owner_id == owners[1]:
            raise RuntimeError("workspaced timeout")
        return True

    monkeypatch.setattr(sp, "sync_owner_project", sync)

    assert await sp.reconcile_storage_projects(object()) == (2, 1)
