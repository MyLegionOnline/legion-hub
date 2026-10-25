"""Unit tests for internal design notes stage 10:

A group room must have a ChatMaster assigned at creation time, and the
ChatMaster must be one of the room's agents.  No DB required — pure Pydantic
validation on RoomCreate.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.room import RoomCreate
from app.models.room import RoomType


def _agent_ids(n: int) -> list:
    return [uuid4() for _ in range(n)]


def _group(**cfg) -> RoomCreate:
    return RoomCreate(
        name="ops",
        room_type=RoomType.group,
        agent_ids=cfg.pop("agent_ids", _agent_ids(2)),
        coordinator_config=cfg,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_group_room_with_chatmaster_accepted():
    a, b = _agent_ids(2)
    room = _group(agent_ids=[a, b], master_agent_id=str(a))
    assert room.coordinator_config["master_agent_id"] == str(a)


def test_group_room_accepts_legacy_chat_master_agent_id_key():
    a, b = _agent_ids(2)
    room = _group(agent_ids=[a, b], chat_master_agent_id=str(a))
    # No exception: the legacy key is recognised.


def test_group_room_accepts_legacy_agent_id_key():
    a, b = _agent_ids(2)
    room = _group(agent_ids=[a, b], agent_id=str(a))


# ---------------------------------------------------------------------------
# Rejection
# ---------------------------------------------------------------------------

def test_group_room_without_chatmaster_rejected():
    with pytest.raises(ValidationError) as exc:
        _group()
    assert "ChatMaster" in str(exc.value)


def test_group_room_with_empty_coordinator_config_rejected():
    with pytest.raises(ValidationError):
        _group(agent_ids=_agent_ids(2))  # coordinator_config defaults to {}


def test_group_room_with_chatmaster_not_in_agents_rejected():
    a, b = _agent_ids(2)
    outsider = uuid4()
    with pytest.raises(ValidationError) as exc:
        _group(agent_ids=[a, b], master_agent_id=str(outsider))
    assert "room's agents" in str(exc.value)


# ---------------------------------------------------------------------------
# direct rooms are unaffected
# ---------------------------------------------------------------------------

def test_direct_room_still_requires_exactly_one_agent():
    with pytest.raises(ValidationError):
        RoomCreate(name="dm", room_type=RoomType.direct, agent_ids=_agent_ids(2))
    # One agent, no coordinator_config — still fine.
    RoomCreate(name="dm", room_type=RoomType.direct, agent_ids=_agent_ids(1))


def test_direct_room_ignores_missing_chatmaster():
    # A direct room is its own coordinator; no ChatMaster key needed.
    RoomCreate(name="dm", room_type=RoomType.direct, agent_ids=_agent_ids(1), coordinator_config={})
