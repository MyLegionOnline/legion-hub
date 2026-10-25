"""Stale proxy-loop guard: _pair_closed_externally.

Retry / manual Start / watchdog close proxy sessions from other DB sessions;
a loop whose sessions were closed under it must detect this via a fresh DB
read and stop without publishing or double-closing.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.proxy_session import _pair_closed_externally


def _session(is_open: bool) -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.is_open = is_open
    return s


def _db(row) -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.first.return_value = row
    db.execute = AsyncMock(return_value=result)
    return db


@pytest.mark.asyncio
async def test_returns_true_when_db_shows_open_side_closed():
    agent, cm = _session(True), _session(True)
    db = _db(row=(agent.id,))
    assert await _pair_closed_externally(agent, cm, db) is True


@pytest.mark.asyncio
async def test_returns_false_when_db_agrees_sessions_open():
    agent, cm = _session(True), _session(True)
    db = _db(row=None)
    assert await _pair_closed_externally(agent, cm, db) is False


@pytest.mark.asyncio
async def test_skips_db_query_when_both_sides_already_closed_in_memory():
    agent, cm = _session(False), _session(False)
    db = _db(row=None)
    assert await _pair_closed_externally(agent, cm, db) is False
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_checks_only_sides_still_open_in_memory():
    # Agent side already closed by the loop itself (legitimate) — only the CM
    # side should be part of the freshness check.
    agent, cm = _session(False), _session(True)
    db = _db(row=None)
    assert await _pair_closed_externally(agent, cm, db) is False
    db.execute.assert_awaited_once()
