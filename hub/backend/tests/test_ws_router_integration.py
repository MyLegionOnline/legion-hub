"""
E2E integration tests for WebSocket router integration (Step 10 / §1.1, §7.9, §7.10).

Tests:
  TestDispatchGroupRoute
    - two @mentions → multi_agent_split → 2 separate sessions
    - split > 3 parts → forced to multi_agent_ask_detail
    - single_agent_p1/p2 → one session started
    - chatmaster_p1/p2 → ChatMaster session started with cm_agent_id
    - chatmaster route without coordinator → legacy chat_master fallback
    - multi_agent_ask_detail → ChatMaster clarification session

  TestBgDirectAgent
    - direct rooms (_bg_direct_agent) build a RouteDecision without the router
      LLM call and go through the same start_user_session/process_user_turn
      flow as group rooms (internal design notes §9.3 follow-up — the old
      handle_mentions()/get_llm_completion() path had no tool-calling at all,
      so MCP/workspace tools attached to a direct room were silently unusable)
    - an open session in the room is resumed instead of starting a new one
    - an active task assigned to the agent routes single_agent_p1 with the
      task/goal id; otherwise single_agent_p2
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.api.v1.chat import (
    _bg_direct_agent,
    _dispatch_group_route,
    _get_cm_agent_id,
    _handle_ask_detail,
    _handle_chatmaster_session,
    _handle_direct_agent_session,
    _handle_split,
)
from app.core.router import RouteDecision, SplitPart
from app.models.room import Room, RoomType


@pytest.fixture(autouse=True)
def _owner_has_hours(monkeypatch):
    """The fake DBs below script every query; the work gates (T8, own tests
    in the cloud plugin) would take one of them."""
    from app.core import work_gates

    monkeypatch.setattr(work_gates, "refuse_room_work", AsyncMock(return_value=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_room(
    coordinator_agent_id: uuid.UUID | None = None,
    room_type: RoomType = RoomType.group,
) -> MagicMock:
    room = MagicMock(spec=Room)
    room.id = uuid.uuid4()
    room.room_type = room_type
    room.coordinator_config = (
        {"agent_id": str(coordinator_agent_id)} if coordinator_agent_id else {}
    )
    return room


def _make_user() -> MagicMock:
    user = MagicMock()
    user.id = uuid.uuid4()
    return user


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    return db


def _make_fake_session(kind: str = "user_agent_p2") -> MagicMock:
    s = MagicMock()
    s.id = uuid.uuid4()
    s.room_id = uuid.uuid4()
    s.kind = kind
    s.is_open = True
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.context_snapshot = None
    s.agent_id = uuid.uuid4()
    s.chat_master_agent_id = None
    s.pending_actions = None
    s.started_at = datetime.now(timezone.utc)
    s.last_context_window_tokens = 0
    return s


def _split_decision(agent_ids: list[uuid.UUID], texts: list[str]) -> RouteDecision:
    return RouteDecision(
        route="multi_agent_split",
        mentioned_agent_ids=agent_ids,
        can_split=True,
        split_messages=[
            SplitPart(agent_id=aid, text=txt)
            for aid, txt in zip(agent_ids, texts)
        ],
        confidence=0.9,
        reasoning="split test",
    )


# ---------------------------------------------------------------------------
# _get_cm_agent_id
# ---------------------------------------------------------------------------


class TestGetCmAgentId:
    def test_returns_uuid_from_agent_id_key(self):
        agent_id = uuid.uuid4()
        room = _make_room(coordinator_agent_id=agent_id)
        result = _get_cm_agent_id(room)
        assert result == agent_id

    def test_returns_none_when_no_coordinator(self):
        room = _make_room(coordinator_agent_id=None)
        result = _get_cm_agent_id(room)
        assert result is None

    def test_returns_none_on_invalid_uuid(self):
        room = MagicMock(spec=Room)
        room.coordinator_config = {"agent_id": "not-a-uuid"}
        result = _get_cm_agent_id(room)
        assert result is None


# ---------------------------------------------------------------------------
# _dispatch_group_route — main dispatch table
# ---------------------------------------------------------------------------


def _patch_session_repo():
    """Patch SessionRepository so find_open_user_session returns None (no in-flight session)."""
    mock_repo = MagicMock()
    mock_repo.find_open_user_session = AsyncMock(return_value=None)
    return patch("app.repositories.session.SessionRepository", return_value=mock_repo)


def _open_session_ctx(session_id: uuid.UUID, kind: str = "user_agent_p2"):
    """The stage-2 view of an open session the router may continue."""
    from app.core.router import OpenSessionContext

    return OpenSessionContext(session_id=session_id, kind=kind, turns=4)


class TestDispatchGroupRoute:

    @pytest.mark.asyncio
    async def test_two_mentions_split_creates_two_sessions(self):
        """E2E: message with 2 @mentions → multi_agent_split → 2 separate sessions."""
        agent_id_1 = uuid.uuid4()
        agent_id_2 = uuid.uuid4()
        room = _make_room()
        user = _make_user()
        db = _make_db()

        decision = _split_decision(
            [agent_id_1, agent_id_2],
            ["fix the login bug", "update the docs"],
        )

        sessions_started: list[tuple[uuid.UUID, str]] = []

        async def _fake_start(r: Any, u: Any, dec: RouteDecision, db: Any, **kw: Any):
            sessions_started.append((dec.mentioned_agent_ids[0], dec.route))
            return _make_fake_session()

        with (
            _patch_session_repo(),
            patch("app.api.v1.chat.route_user_message", new_callable=AsyncMock, return_value=decision),
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _dispatch_group_route(room, user, "@alice fix login @bob update docs", db)

        assert len(sessions_started) == 2
        ids_seen = {s[0] for s in sessions_started}
        assert agent_id_1 in ids_seen
        assert agent_id_2 in ids_seen
        assert all(route == "single_agent_p2" for _, route in sessions_started)

    @pytest.mark.asyncio
    async def test_split_more_than_3_forces_ask_detail(self):
        """split_messages with 4 parts → forced to multi_agent_ask_detail."""
        agent_ids = [uuid.uuid4() for _ in range(4)]
        room = _make_room(coordinator_agent_id=uuid.uuid4())
        user = _make_user()
        db = _make_db()

        decision = _split_decision(agent_ids, [f"task {i}" for i in range(4)])

        cm_sessions: list[str] = []

        async def _fake_start(r: Any, u: Any, dec: RouteDecision, db: Any, **kw: Any):
            cm_sessions.append(dec.route)
            return _make_fake_session("user_cm_p2")

        with (
            _patch_session_repo(),
            patch("app.api.v1.chat.route_user_message", new_callable=AsyncMock, return_value=decision),
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _dispatch_group_route(room, user, "do many things", db)

        assert len(cm_sessions) == 1
        assert cm_sessions[0] == "chatmaster_p2"

    @pytest.mark.asyncio
    async def test_single_agent_p1_starts_one_session(self):
        agent_id = uuid.uuid4()
        room = _make_room()
        user = _make_user()
        db = _make_db()
        task_id = uuid.uuid4()

        decision = RouteDecision(
            route="single_agent_p1",
            mentioned_agent_ids=[agent_id],
            matched_task_id=task_id,
            confidence=0.9,
            reasoning="task match",
        )

        started: list[RouteDecision] = []

        async def _fake_start(r: Any, u: Any, dec: RouteDecision, db: Any, **kw: Any):
            started.append(dec)
            return _make_fake_session("user_agent_p1")

        with (
            _patch_session_repo(),
            patch("app.api.v1.chat.route_user_message", new_callable=AsyncMock, return_value=decision),
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _dispatch_group_route(room, user, "@dev fix it", db)

        assert len(started) == 1
        assert started[0].route == "single_agent_p1"
        assert started[0].matched_task_id == task_id

    @pytest.mark.asyncio
    async def test_single_agent_route_without_mention_falls_back_to_chatmaster(self):
        """Model picks single_agent_* but the deterministic @-parse resolved no
        room agent — route to ChatMaster instead of IndexError-ing on
        mentioned_agent_ids[0] (which lost the turn silently in production)."""
        cm_id = uuid.uuid4()
        room = _make_room(coordinator_agent_id=cm_id)
        user = _make_user()
        db = _make_db()

        decision = RouteDecision(
            route="single_agent_p2",
            mentioned_agent_ids=[],
            confidence=0.9,
            reasoning="model said single agent, parse found none",
        )

        started: list[RouteDecision] = []

        async def _fake_start(r: Any, u: Any, dec: RouteDecision, db: Any, **kw: Any):
            started.append(dec)
            return _make_fake_session("user_cm_p2")

        with (
            _patch_session_repo(),
            patch("app.api.v1.chat.route_user_message", new_callable=AsyncMock, return_value=decision),
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _dispatch_group_route(room, user, "@ghost do the thing", db)

        assert len(started) == 1
        assert started[0].route == "chatmaster_p2"

    @pytest.mark.asyncio
    async def test_chatmaster_p1_passes_cm_agent_id(self):
        cm_id = uuid.uuid4()
        room = _make_room(coordinator_agent_id=cm_id)
        user = _make_user()
        db = _make_db()

        decision = RouteDecision(
            route="chatmaster_p1",
            matched_goal_id=uuid.uuid4(),
            confidence=0.85,
            reasoning="goal match",
        )

        received_cm_ids: list[uuid.UUID | None] = []

        async def _fake_start(r: Any, u: Any, dec: RouteDecision, db: Any, **kw: Any):
            received_cm_ids.append(kw.get("cm_agent_id"))
            return _make_fake_session("user_cm_p1")

        with (
            _patch_session_repo(),
            patch("app.api.v1.chat.route_user_message", new_callable=AsyncMock, return_value=decision),
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _dispatch_group_route(room, user, "update the plan", db)

        assert len(received_cm_ids) == 1
        assert received_cm_ids[0] == cm_id

    @pytest.mark.asyncio
    async def test_chatmaster_route_without_coordinator_falls_back(self):
        """If room has no coordinator, chatmaster routes fall back to legacy chat_master."""
        room = _make_room(coordinator_agent_id=None)
        user = _make_user()
        db = _make_db()

        decision = RouteDecision(
            route="chatmaster_p2",
            confidence=0.8,
            reasoning="no goals",
        )

        with (
            _patch_session_repo(),
            patch("app.api.v1.chat.route_user_message", new_callable=AsyncMock, return_value=decision),
            patch("app.api.v1.chat.start_user_session", new_callable=AsyncMock) as mock_start,
            patch("app.api.v1.chat.chat_master") as mock_cm,
        ):
            mock_cm.chat_master_process = AsyncMock()
            await _dispatch_group_route(room, user, "hey team", db)

        mock_start.assert_not_called()
        mock_cm.chat_master_process.assert_called_once_with(room.id, "hey team", db)

    @pytest.mark.asyncio
    async def test_ask_detail_routes_to_chatmaster_p2(self):
        cm_id = uuid.uuid4()
        room = _make_room(coordinator_agent_id=cm_id)
        user = _make_user()
        db = _make_db()

        decision = RouteDecision(
            route="multi_agent_ask_detail",
            can_split=False,
            confidence=0.7,
            reasoning="cannot split",
        )

        started: list[str] = []

        async def _fake_start(r: Any, u: Any, dec: RouteDecision, db: Any, **kw: Any):
            started.append(dec.route)
            return _make_fake_session("user_cm_p2")

        with (
            _patch_session_repo(),
            patch("app.api.v1.chat.route_user_message", new_callable=AsyncMock, return_value=decision),
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _dispatch_group_route(room, user, "@alice @bob do something", db)

        assert started == ["chatmaster_p2"]

    @pytest.mark.asyncio
    async def test_process_user_turn_called_with_correct_content(self):
        """process_user_turn must receive the original message text."""
        agent_id = uuid.uuid4()
        room = _make_room()
        user = _make_user()
        db = _make_db()
        content = "@dev please help me refactor this"

        decision = RouteDecision(
            route="single_agent_p2",
            mentioned_agent_ids=[agent_id],
            confidence=0.88,
            reasoning="mention only",
        )

        turn_contents: list[str] = []

        async def _fake_turn(session: Any, msg: str, db: Any) -> bool:
            turn_contents.append(msg)
            return True

        with (
            _patch_session_repo(),
            patch("app.api.v1.chat.route_user_message", new_callable=AsyncMock, return_value=decision),
            patch("app.api.v1.chat.start_user_session", new_callable=AsyncMock, return_value=_make_fake_session()),
            patch("app.api.v1.chat.process_user_turn", side_effect=_fake_turn),
        ):
            await _dispatch_group_route(room, user, content, db)

        assert turn_contents == [content]


# ---------------------------------------------------------------------------
# _bg_direct_agent — direct (1:1) rooms must use the same session-flow as
# group rooms, not the legacy no-tools handle_mentions() path.
# ---------------------------------------------------------------------------


class _FakeSessionCM:
    """Stands in for `async with _make_session()() as session: ...`."""

    def __init__(self, db: Any) -> None:
        self._db = db

    async def __aenter__(self) -> Any:
        return self._db

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _patch_make_session(db: Any):
    factory = MagicMock(return_value=_FakeSessionCM(db))
    return patch("app.api.v1.chat._make_session", return_value=factory)


def _scalar_result(obj: Any) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.first.return_value = obj
    return result


class TestBgDirectAgent:
    @pytest.mark.asyncio
    async def test_no_active_task_starts_single_agent_p2_via_session_flow(self):
        room = _make_room(room_type=RoomType.direct)
        user = _make_user()
        agent = MagicMock()
        agent.id = uuid.uuid4()
        db = _make_db()
        # Order of queries in _bg_direct_agent: Room, User, Agent, Task.
        db.execute = AsyncMock(side_effect=[
            _scalar_result(room),
            _scalar_result(user),
            _scalar_result(agent),
            _scalar_result(None),  # no active task
        ])

        started: list[RouteDecision] = []

        async def _fake_start(r: Any, u: Any, dec: RouteDecision, db: Any, **kw: Any):
            started.append(dec)
            return _make_fake_session("user_agent_p2")

        with (
            _patch_make_session(db),
            _patch_session_repo(),
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _bg_direct_agent("hi", room.id, user.id)

        assert len(started) == 1
        assert started[0].route == "single_agent_p2"
        assert started[0].mentioned_agent_ids == [agent.id]
        db.commit.assert_awaited()

    @pytest.mark.asyncio
    async def test_active_task_starts_single_agent_p1_with_task_and_goal(self):
        room = _make_room(room_type=RoomType.direct)
        user = _make_user()
        agent = MagicMock()
        agent.id = uuid.uuid4()
        task = MagicMock()
        task.id = uuid.uuid4()
        task.goal_id = uuid.uuid4()
        db = _make_db()
        db.execute = AsyncMock(side_effect=[
            _scalar_result(room),
            _scalar_result(user),
            _scalar_result(agent),
            _scalar_result(task),
        ])

        started: list[RouteDecision] = []

        async def _fake_start(r: Any, u: Any, dec: RouteDecision, db: Any, **kw: Any):
            started.append(dec)
            return _make_fake_session("user_agent_p1")

        with (
            _patch_make_session(db),
            _patch_session_repo(),
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _bg_direct_agent("continue", room.id, user.id)

        assert len(started) == 1
        assert started[0].route == "single_agent_p1"
        assert started[0].matched_task_id == task.id
        assert started[0].matched_goal_id == task.goal_id

    @pytest.mark.asyncio
    async def test_young_open_session_is_resumed_not_restarted(self):
        """A young session with the same task binding continues (stage 2.1)."""
        room = _make_room(room_type=RoomType.direct)
        user = _make_user()
        agent = MagicMock()
        agent.id = uuid.uuid4()
        db = _make_db()
        db.execute = AsyncMock(side_effect=[
            _scalar_result(room),
            _scalar_result(user),
            _scalar_result(agent),
            _scalar_result(None),   # no active task → p2 session stays valid
        ])
        existing = _make_fake_session("user_agent_p2")
        existing.task_id = None

        mock_repo = MagicMock()
        mock_repo.find_open_user_session = AsyncMock(return_value=existing)
        turns: list[Any] = []

        async def _fake_turn(session: Any, content: str, db: Any, **kw: Any):
            turns.append(session)
            return True

        with (
            _patch_make_session(db),
            patch("app.repositories.session.SessionRepository", return_value=mock_repo),
            patch(
                "app.core.user_sessions.open_session_context",
                new=AsyncMock(return_value=_open_session_ctx(existing.id)),
            ),
            patch(
                "app.core.user_sessions.close_user_session_gracefully",
                new=AsyncMock(),
            ) as mock_close,
            patch("app.api.v1.chat.start_user_session", new_callable=AsyncMock) as mock_start,
            patch("app.api.v1.chat.process_user_turn", side_effect=_fake_turn),
        ):
            await _bg_direct_agent("one more question", room.id, user.id)

        mock_start.assert_not_called()
        mock_close.assert_not_awaited()
        assert turns == [existing]

    @pytest.mark.asyncio
    async def test_an_over_limit_open_session_is_closed_and_replaced(self):
        """Defect C1/C2 in a direct room: the open row is no longer unconditional."""
        room = _make_room(room_type=RoomType.direct)
        user = _make_user()
        agent = MagicMock()
        agent.id = uuid.uuid4()
        db = _make_db()
        db.execute = AsyncMock(side_effect=[
            _scalar_result(room),
            _scalar_result(user),
            _scalar_result(agent),
            _scalar_result(None),
        ])
        existing = _make_fake_session("user_agent_p2")
        existing.task_id = None

        mock_repo = MagicMock()
        mock_repo.find_open_user_session = AsyncMock(return_value=existing)

        with (
            _patch_make_session(db),
            patch("app.repositories.session.SessionRepository", return_value=mock_repo),
            patch(
                "app.core.user_sessions.open_session_context",
                new=AsyncMock(return_value=None),   # too old / too long to continue
            ),
            patch(
                "app.core.user_sessions.close_user_session_gracefully",
                new=AsyncMock(),
            ) as mock_close,
            patch(
                "app.api.v1.chat.start_user_session",
                new=AsyncMock(return_value=_make_fake_session("user_agent_p2")),
            ) as mock_start,
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _bg_direct_agent("one more question", room.id, user.id)

        mock_close.assert_awaited_once()
        assert mock_close.await_args.args[0] is existing
        mock_start.assert_awaited_once()


# ---------------------------------------------------------------------------
# _handle_split edge cases
# ---------------------------------------------------------------------------


class TestHandleSplit:
    @pytest.mark.asyncio
    async def test_empty_split_creates_no_sessions(self):
        room = _make_room()
        user = _make_user()
        db = _make_db()

        decision = RouteDecision(
            route="multi_agent_split",
            split_messages=[],
            can_split=True,
            confidence=0.9,
            reasoning="empty split",
        )

        with (
            patch("app.api.v1.chat.start_user_session", new_callable=AsyncMock) as mock_start,
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock),
        ):
            await _handle_split(room, user, "msg", decision, db)

        mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_exactly_three_parts_creates_three_sessions(self):
        agent_ids = [uuid.uuid4() for _ in range(3)]
        room = _make_room()
        user = _make_user()
        db = _make_db()
        decision = _split_decision(agent_ids, ["a", "b", "c"])

        calls: list[int] = []

        async def _fake_start(*args: Any, **kw: Any):
            calls.append(1)
            return _make_fake_session()

        with (
            patch("app.api.v1.chat.start_user_session", side_effect=_fake_start),
            patch("app.api.v1.chat.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            await _handle_split(room, user, "msg", decision, db)

        assert len(calls) == 3
