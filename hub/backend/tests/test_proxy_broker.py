"""Unit tests for ProxyBroker (Step 5).

Tests cover:
  - TestPublishSuccess   — happy path: message persisted, peer queue populated, WS broadcast called
  - TestPublishTurn      — turn counter increments monotonically
  - TestPublishValidation — rejects missing peer; closed peer = persist without queue push
  - TestQueueManagement  — get_or_create_queue, clear_queue

All tests are pure unit tests (no DB); the SQLAlchemy session is mocked.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.core.proxy_broker import ProxyBroker, ProxyBrokerError
from app.models.message import PROXY_MESSAGE
from app.schemas.proxy import ProxyMessageCreate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOM_ID = uuid.uuid4()
_TASK_ID = uuid.uuid4()
_SESSION_ID = uuid.uuid4()
_PEER_SESSION_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()


def _make_proxy_msg(**overrides) -> ProxyMessageCreate:
    base = dict(
        room_id=_ROOM_ID,
        sender_type="agent",
        sender_id=_AGENT_ID,
        content="hello from agent",
        task_id=_TASK_ID,
        session_id=_SESSION_ID,
        peer_session_id=_PEER_SESSION_ID,
    )
    base.update(overrides)
    return ProxyMessageCreate(**base)


def _open_session(session_id=None):
    s = MagicMock()
    s.id = session_id or _PEER_SESSION_ID
    s.is_open = True
    s.closed_at = None
    return s


def _closed_session():
    s = MagicMock()
    s.id = _PEER_SESSION_ID
    s.is_open = False
    s.closed_at = datetime.now(timezone.utc)
    return s


def _make_execute_result_scalars(peer_session):
    """Sync result object for session queries (peer lookup)."""
    scalars_mock = MagicMock()
    scalars_mock.first.return_value = peer_session
    result = MagicMock()
    result.scalars = MagicMock(return_value=scalars_mock)
    return result


def _make_execute_result_count(count: int):
    """Sync result object for count() queries."""
    result = MagicMock()
    result.scalar.return_value = count
    return result


def _build_db(peer_session=None, message_count: int = 0, sender_agent=None):
    """Return a mock AsyncSession with pre-configured execute results."""
    db = AsyncMock()
    calls = []

    async def fake_execute(stmt):
        idx = len(calls)
        calls.append(stmt)
        if idx == 0:
            # First call: peer session lookup
            return _make_execute_result_scalars(peer_session)
        elif idx == 1:
            # Second call: sender agent nickname lookup
            return _make_execute_result_scalars(sender_agent)
        else:
            # Third call: count query
            return _make_execute_result_count(message_count)

    db.execute.side_effect = fake_execute
    db.flush = AsyncMock()

    async def fake_refresh(obj):
        pass  # timestamp is set by Message.__init__ via default_factory

    db.refresh.side_effect = fake_refresh
    return db


# ---------------------------------------------------------------------------
# TestPublishSuccess
# ---------------------------------------------------------------------------

class TestPublishSuccess:
    @pytest.mark.asyncio
    async def test_message_added_to_db(self):
        db = _build_db(peer_session=_open_session())
        broker = ProxyBroker()
        proxy_msg = _make_proxy_msg()

        with patch("app.core.proxy_broker.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await broker.publish(proxy_msg, db)

        db.add.assert_called_once()
        added = db.add.call_args[0][0]
        assert added.event_type == PROXY_MESSAGE
        assert added.content == "hello from agent"
        assert added.task_id == _TASK_ID

    @pytest.mark.asyncio
    async def test_event_metadata_fields(self):
        db = _build_db(peer_session=_open_session())
        broker = ProxyBroker()
        proxy_msg = _make_proxy_msg()

        with patch("app.core.proxy_broker.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await broker.publish(proxy_msg, db)

        added = db.add.call_args[0][0]
        meta = added.event_metadata
        assert meta["session_id"] == str(_SESSION_ID)
        assert meta["peer_session_id"] == str(_PEER_SESSION_ID)
        assert meta["kind"] == "cm_agent_proxy"
        assert meta["sender_type"] == "agent"
        assert meta["sender_id"] == str(_AGENT_ID)
        assert "turn" in meta

    @pytest.mark.asyncio
    async def test_peer_queue_receives_message(self):
        # Clear any stale queue from previous tests
        ProxyBroker.clear_queue(_PEER_SESSION_ID)

        db = _build_db(peer_session=_open_session())
        broker = ProxyBroker()
        proxy_msg = _make_proxy_msg()

        with patch("app.core.proxy_broker.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await broker.publish(proxy_msg, db)

        queue = ProxyBroker.get_or_create_queue(_PEER_SESSION_ID)
        assert not queue.empty()
        item = queue.get_nowait()
        assert item["content"] == "hello from agent"
        assert item["session_id"] == str(_SESSION_ID)

    @pytest.mark.asyncio
    async def test_websocket_broadcast_called(self):
        db = _build_db(peer_session=_open_session())
        broker = ProxyBroker()
        proxy_msg = _make_proxy_msg()

        with patch("app.core.proxy_broker.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await broker.publish(proxy_msg, db)

        mock_manager.broadcast.assert_awaited_once()
        broadcast_payload = mock_manager.broadcast.call_args[0][0]
        assert broadcast_payload["event_type"] == PROXY_MESSAGE
        assert broadcast_payload["room_id"] == str(_ROOM_ID)

    @pytest.mark.asyncio
    async def test_chat_master_sender_type(self):
        """sender_type='chat_master' is stored correctly in metadata."""
        db = _build_db(peer_session=_open_session())
        broker = ProxyBroker()
        proxy_msg = _make_proxy_msg(sender_type="chat_master")

        with patch("app.core.proxy_broker.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await broker.publish(proxy_msg, db)

        added = db.add.call_args[0][0]
        assert added.event_metadata["sender_type"] == "chat_master"


# ---------------------------------------------------------------------------
# TestPublishTurn
# ---------------------------------------------------------------------------

class TestPublishTurn:
    @pytest.mark.asyncio
    async def test_turn_starts_at_zero(self):
        db = _build_db(peer_session=_open_session(), message_count=0)
        broker = ProxyBroker()

        with patch("app.core.proxy_broker.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await broker.publish(_make_proxy_msg(), db)

        added = db.add.call_args[0][0]
        assert added.event_metadata["turn"] == 0

    @pytest.mark.asyncio
    async def test_turn_increments_with_existing_messages(self):
        db = _build_db(peer_session=_open_session(), message_count=5)
        broker = ProxyBroker()

        with patch("app.core.proxy_broker.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await broker.publish(_make_proxy_msg(), db)

        added = db.add.call_args[0][0]
        assert added.event_metadata["turn"] == 5


# ---------------------------------------------------------------------------
# TestPublishValidation
# ---------------------------------------------------------------------------

class TestPublishValidation:
    @pytest.mark.asyncio
    async def test_raises_when_peer_not_found(self):
        db = _build_db(peer_session=None)
        broker = ProxyBroker()

        with pytest.raises(ProxyBrokerError, match="not found"):
            with patch("app.core.proxy_broker.manager"):
                await broker.publish(_make_proxy_msg(), db)

    @pytest.mark.asyncio
    async def test_closed_peer_persists_without_queue_push(self):
        """A closed peer is not an error: the message is recorded for room
        visibility, but nothing is pushed to the peer's input queue."""
        ProxyBroker.clear_queue(_PEER_SESSION_ID)
        db = _build_db(peer_session=_closed_session())
        broker = ProxyBroker()

        with patch("app.core.proxy_broker.manager") as mock_manager:
            mock_manager.broadcast = AsyncMock()
            await broker.publish(_make_proxy_msg(), db)

        db.add.assert_called_once()
        mock_manager.broadcast.assert_awaited_once()
        assert _PEER_SESSION_ID not in ProxyBroker._queues

    @pytest.mark.asyncio
    async def test_no_db_write_on_validation_failure(self):
        db = _build_db(peer_session=None)
        broker = ProxyBroker()

        with pytest.raises(ProxyBrokerError):
            with patch("app.core.proxy_broker.manager"):
                await broker.publish(_make_proxy_msg(), db)

        db.add.assert_not_called()


# ---------------------------------------------------------------------------
# TestQueueManagement
# ---------------------------------------------------------------------------

class TestQueueManagement:
    def test_get_or_create_returns_queue(self):
        sid = uuid.uuid4()
        ProxyBroker.clear_queue(sid)
        q = ProxyBroker.get_or_create_queue(sid)
        assert isinstance(q, asyncio.Queue)

    def test_get_or_create_returns_same_instance(self):
        sid = uuid.uuid4()
        ProxyBroker.clear_queue(sid)
        q1 = ProxyBroker.get_or_create_queue(sid)
        q2 = ProxyBroker.get_or_create_queue(sid)
        assert q1 is q2

    def test_clear_queue_removes_instance(self):
        sid = uuid.uuid4()
        q1 = ProxyBroker.get_or_create_queue(sid)
        ProxyBroker.clear_queue(sid)
        q2 = ProxyBroker.get_or_create_queue(sid)
        assert q1 is not q2

    def test_clear_nonexistent_queue_is_noop(self):
        ProxyBroker.clear_queue(uuid.uuid4())  # no exception
