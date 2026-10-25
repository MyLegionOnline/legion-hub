"""
Unit tests for the LLM-router (Step 3 / §1, §8.1).

Test groups:
  - TestParseMentions          — deterministic @mention extractor
  - TestConfidenceFallback     — _apply_confidence_fallback for all 6 routes
  - TestRouteUserMessage       — route_user_message with mocked LLM + DB
      6 positive cases (one per route)
      invalid JSON from LLM  → chatmaster_p2 + ROUTE_LLM_FAILURE emitted
      timeout               → chatmaster_p2 + ROUTE_LLM_FAILURE emitted
      low confidence        → forced to p2 variant

Run:
    pytest tests/test_router.py
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import ValidationError

from app.core.router import (
    CONFIDENCE_THRESHOLD,
    RouteDecision,
    SplitPart,
    _ROUTER_TIMEOUT_S,
    _apply_confidence_fallback,
    _call_router_llm,
    parse_mentions,
    route_user_message,
)
from app.core.secrets.crypto import SecretValue
from app.models.room import Room, RoomType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_room() -> Room:
    return Room(
        id=uuid.uuid4(),
        creator_id=uuid.uuid4(),
        name="test-room",
    )


def _mock_db() -> AsyncMock:
    """Return a DB mock where all execute() calls return empty result sets."""
    db = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute.return_value = result
    return db


def _llm_payload(**kwargs: Any) -> str:
    """Build a JSON string suitable as a mock LLM response."""
    base: dict[str, Any] = {
        "mentioned_agent_ids": [],
        "can_split": None,
        "split_messages": None,
        "matched_task_id": None,
        "matched_goal_id": None,
        "confidence": 0.9,
        "reasoning": "test reasoning",
    }
    base.update(kwargs)
    return json.dumps(base)


def _llm_return(raw: str) -> tuple[str, int, int, str]:
    return raw, 10, 5, "claude-haiku-4-5-20251001"


def _emitted_event_types(mock_emit: AsyncMock) -> list[str]:
    return [c.kwargs["event_type"] for c in mock_emit.call_args_list]


# ---------------------------------------------------------------------------
# 1. parse_mentions
# ---------------------------------------------------------------------------


class TestParseMentions:
    def test_empty_text(self):
        assert parse_mentions("hello world") == []

    def test_single_mention(self):
        assert parse_mentions("hey @Alice fix the bug") == ["alice"]

    def test_multiple_mentions(self):
        result = parse_mentions("@Alice and @BOB please review")
        assert result == ["alice", "bob"]

    def test_mention_with_hyphen(self):
        assert parse_mentions("@backend-dev do it") == ["backend-dev"]

    def test_no_at_sign(self):
        assert parse_mentions("plain text here") == []

    def test_duplicate_mentions_preserved(self):
        result = parse_mentions("@alice @alice again")
        assert result == ["alice", "alice"]

    def test_mention_at_start(self):
        assert parse_mentions("@dev check this") == ["dev"]


# ---------------------------------------------------------------------------
# 2. Confidence fallback
# ---------------------------------------------------------------------------


class TestConfidenceFallback:
    def _route(self, route: str, confidence: float, **extra: Any) -> RouteDecision:
        return RouteDecision(route=route, confidence=confidence, reasoning="x", **extra)

    def test_single_agent_p1_falls_back_to_p2(self):
        d = self._route(
            "single_agent_p1", 0.4, matched_task_id=uuid.uuid4()
        )
        out = _apply_confidence_fallback(d)
        assert out.route == "single_agent_p2"
        assert out.matched_task_id is None

    def test_chatmaster_p1_falls_back_to_p2(self):
        d = self._route("chatmaster_p1", 0.3, matched_goal_id=uuid.uuid4())
        out = _apply_confidence_fallback(d)
        assert out.route == "chatmaster_p2"
        assert out.matched_goal_id is None

    def test_multi_agent_split_falls_back_to_ask_detail(self):
        d = self._route("multi_agent_split", 0.2, can_split=True, split_messages=[])
        out = _apply_confidence_fallback(d)
        assert out.route == "multi_agent_ask_detail"
        assert out.can_split is None
        assert out.split_messages is None

    def test_already_p2_routes_unchanged(self):
        for route in ("single_agent_p2", "chatmaster_p2", "multi_agent_ask_detail"):
            d = self._route(route, 0.1)
            out = _apply_confidence_fallback(d)
            assert out.route == route

    def test_above_threshold_no_change(self):
        d = self._route("single_agent_p1", 0.9, matched_task_id=uuid.uuid4())
        out = _apply_confidence_fallback(d)
        assert out.route == "single_agent_p1"

    def test_exactly_at_threshold_no_change(self):
        d = self._route("chatmaster_p1", CONFIDENCE_THRESHOLD, matched_goal_id=uuid.uuid4())
        out = _apply_confidence_fallback(d)
        assert out.route == "chatmaster_p1"

    def test_just_below_threshold_falls_back(self):
        d = self._route("chatmaster_p1", CONFIDENCE_THRESHOLD - 0.01, matched_goal_id=uuid.uuid4())
        out = _apply_confidence_fallback(d)
        assert out.route == "chatmaster_p2"


# ---------------------------------------------------------------------------
# 3. route_user_message  (positive cases — one per route)
# ---------------------------------------------------------------------------

_AGENT_ID = str(uuid.uuid4())
_TASK_ID = str(uuid.uuid4())
_GOAL_ID = str(uuid.uuid4())

_POSITIVE_CASES = [
    {
        "label": "multi_agent_split",
        "payload": dict(
            route="multi_agent_split",
            can_split=True,
            split_messages=[{"agent_id": _AGENT_ID, "text": "do x"}],
            confidence=0.95,
        ),
        "expected_route": "multi_agent_split",
    },
    {
        "label": "multi_agent_ask_detail",
        "payload": dict(
            route="multi_agent_ask_detail",
            can_split=False,
            confidence=0.88,
        ),
        "expected_route": "multi_agent_ask_detail",
    },
    {
        "label": "single_agent_p1",
        "payload": dict(
            route="single_agent_p1",
            matched_task_id=_TASK_ID,
            confidence=0.92,
        ),
        "expected_route": "single_agent_p1",
    },
    {
        "label": "single_agent_p2",
        "payload": dict(
            route="single_agent_p2",
            confidence=0.87,
        ),
        "expected_route": "single_agent_p2",
    },
    {
        "label": "chatmaster_p1",
        "payload": dict(
            route="chatmaster_p1",
            matched_goal_id=_GOAL_ID,
            confidence=0.83,
        ),
        "expected_route": "chatmaster_p1",
    },
    {
        "label": "chatmaster_p2",
        "payload": dict(
            route="chatmaster_p2",
            confidence=0.78,
        ),
        "expected_route": "chatmaster_p2",
    },
]


@pytest.mark.parametrize("case", _POSITIVE_CASES, ids=[c["label"] for c in _POSITIVE_CASES])
@pytest.mark.asyncio
async def test_route_positive(case: dict) -> None:
    room = _make_room()
    db = _mock_db()
    raw = _llm_payload(**case["payload"])

    with (
        patch("app.core.router._call_router_llm", return_value=_llm_return(raw)),
        patch("app.core.router.emit_event", new=AsyncMock()) as mock_emit,
    ):
        decision = await route_user_message(room, "some message", db)

    assert decision.route == case["expected_route"]
    assert isinstance(decision.confidence, float)
    assert "ROUTE_DECIDED" in _emitted_event_types(mock_emit)
    assert "ROUTE_LLM_FAILURE" not in _emitted_event_types(mock_emit)


# ---------------------------------------------------------------------------
# 4. Error cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_json_falls_back_to_chatmaster_p2() -> None:
    room = _make_room()
    db = _mock_db()

    with (
        patch("app.core.router._call_router_llm", return_value=("not valid json {{", 5, 3, "claude-haiku-4-5-20251001")),
        patch("app.core.router.emit_event", new=AsyncMock()) as mock_emit,
    ):
        decision = await route_user_message(room, "hello", db)

    assert decision.route == "chatmaster_p2"
    assert decision.confidence == 0.0
    types = _emitted_event_types(mock_emit)
    assert "ROUTE_LLM_FAILURE" in types
    assert "ROUTE_DECIDED" in types


@pytest.mark.asyncio
async def test_timeout_falls_back_to_chatmaster_p2() -> None:
    room = _make_room()
    db = _mock_db()

    async def _raise_timeout(*_a: Any, **_kw: Any) -> None:
        raise asyncio.TimeoutError("timed out")

    with (
        patch("app.core.router._call_router_llm", side_effect=_raise_timeout),
        patch("app.core.router.emit_event", new=AsyncMock()) as mock_emit,
    ):
        decision = await route_user_message(room, "hello", db)

    assert decision.route == "chatmaster_p2"
    types = _emitted_event_types(mock_emit)
    assert "ROUTE_LLM_FAILURE" in types
    assert "ROUTE_DECIDED" in types


@pytest.mark.asyncio
async def test_low_confidence_forces_p2_variant() -> None:
    """single_agent_p1 with confidence=0.4 must be rewritten to single_agent_p2."""
    room = _make_room()
    db = _mock_db()
    raw = _llm_payload(route="single_agent_p1", matched_task_id=_TASK_ID, confidence=0.4)

    with (
        patch("app.core.router._call_router_llm", return_value=_llm_return(raw)),
        patch("app.core.router.emit_event", new=AsyncMock()),
    ):
        decision = await route_user_message(room, "@agent do something", db)

    assert decision.route == "single_agent_p2"
    assert decision.matched_task_id is None


@pytest.mark.asyncio
async def test_route_decided_metadata_fields() -> None:
    """ROUTE_DECIDED event must carry route, confidence, reasoning, latency_ms, cost_usd."""
    room = _make_room()
    db = _mock_db()
    raw = _llm_payload(route="chatmaster_p2", confidence=0.8, reasoning="no goals match")

    with (
        patch("app.core.router._call_router_llm", return_value=_llm_return(raw)),
        patch("app.core.router.emit_event", new=AsyncMock()) as mock_emit,
    ):
        await route_user_message(room, "free text", db)

    decided_call = next(
        c for c in mock_emit.call_args_list if c.kwargs["event_type"] == "ROUTE_DECIDED"
    )
    meta = decided_call.kwargs["metadata"]
    assert "route" in meta
    assert "confidence" in meta
    assert "reasoning" in meta
    assert "latency_ms" in meta
    assert "cost_usd" in meta


# ---------------------------------------------------------------------------
# 5. Goal status filter (_gather_context must exclude COMPLETED goals)
# ---------------------------------------------------------------------------


class TestGatherContextGoalFilter:
    """Stage-1 fix: only ACTIVE goals reach the router LLM."""

    @pytest.mark.asyncio
    async def test_active_goal_appears_in_context(self) -> None:
        from app.core.router import _gather_context
        from app.models.goal import GoalStatus, RoomGoal

        room = _make_room()
        active_goal = RoomGoal(
            id=uuid.uuid4(),
            room_id=room.id,
            title="Build feature",
            status=GoalStatus.ACTIVE,
        )

        db = AsyncMock()
        agents_res = MagicMock()
        agents_res.scalars.return_value.all.return_value = []
        waiting_res = MagicMock()
        waiting_res.scalars.return_value.all.return_value = []
        goals_res = MagicMock()
        goals_res.scalars.return_value.all.return_value = [active_goal]
        db.execute.side_effect = [agents_res, waiting_res, goals_res]

        _, goals_ctx, _, _ = await _gather_context(room, "some text", db)

        assert len(goals_ctx) == 1
        assert goals_ctx[0]["title"] == "Build feature"

    @pytest.mark.asyncio
    async def test_completed_goal_not_in_context(self) -> None:
        """When DB returns no goals (because filter excluded COMPLETED), goals_ctx is empty."""
        from app.core.router import _gather_context

        room = _make_room()
        db = AsyncMock()
        agents_res = MagicMock()
        agents_res.scalars.return_value.all.return_value = []
        waiting_res = MagicMock()
        waiting_res.scalars.return_value.all.return_value = []
        goals_res = MagicMock()
        goals_res.scalars.return_value.all.return_value = []  # DB filtered out COMPLETED
        db.execute.side_effect = [agents_res, waiting_res, goals_res]

        _, goals_ctx, _, _ = await _gather_context(room, "some text", db)

        assert goals_ctx == []

    @pytest.mark.asyncio
    async def test_goals_query_includes_active_status_filter(self) -> None:
        """The SQLAlchemy query sent to the DB must filter status = 'active'."""
        from app.core.router import _gather_context

        room = _make_room()
        db = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        db.execute.return_value = result

        await _gather_context(room, "some text", db)

        # Last execute call is the goals query
        goals_call = db.execute.call_args_list[-1]
        query = goals_call[0][0]
        compiled = str(query.compile(compile_kwargs={"literal_binds": True}))
        assert "active" in compiled.lower(), (
            f"Goals query must filter by ACTIVE status; compiled SQL: {compiled}"
        )


@pytest.mark.asyncio
async def test_no_active_goals_routes_to_chatmaster_p2() -> None:
    """With empty active goals list, router should choose chatmaster_p2."""
    room = _make_room()
    db = _mock_db()  # returns [] for all queries → no active goals
    raw = _llm_payload(route="chatmaster_p2", confidence=0.85, reasoning="no active goals")

    with (
        patch("app.core.router._call_router_llm", return_value=_llm_return(raw)),
        patch("app.core.router.emit_event", new=AsyncMock()),
    ):
        decision = await route_user_message(room, "add a new feature", db)

    assert decision.route == "chatmaster_p2"


# ---------------------------------------------------------------------------
# 6. (was 5) mentioned_agent_ids from deterministic parser
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mentioned_agent_ids_from_deterministic_parser() -> None:
    """mentioned_agent_ids in the decision always comes from the @-parser, not the LLM."""
    room = _make_room()
    db = _mock_db()
    # LLM returns a different agent id — should be overridden
    raw = _llm_payload(
        route="single_agent_p2",
        mentioned_agent_ids=[str(uuid.uuid4())],
        confidence=0.85,
    )

    with (
        patch("app.core.router._call_router_llm", return_value=_llm_return(raw)),
        patch("app.core.router.emit_event", new=AsyncMock()),
    ):
        decision = await route_user_message(room, "no mentions here", db)

    # No @mentions in message → empty list, regardless of what LLM returned
    assert decision.mentioned_agent_ids == []


# ---------------------------------------------------------------------------
# Stage 2 (R2) — router timeout 30→90 s, max_retries=0
# ---------------------------------------------------------------------------


def test_router_timeout_is_90_seconds() -> None:
    """A 60 s response (measured median under load + headroom) must fit the window."""
    assert _ROUTER_TIMEOUT_S == 90.0


@pytest.mark.asyncio
async def test_chat_wire_passes_timeout_and_zero_retries() -> None:
    """The OpenAI client must be built with timeout=90 and max_retries=0.

    max_retries=0 is what makes the 90 s timeout meaningful: without it the
    SDK retries three times and a failure costs 270 s before fallback.
    """
    secret = SecretValue("sk-test")

    with patch("app.core.router.AsyncOpenAI") as mock_ctor:
        client = MagicMock()
        create = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content='{"route":"chatmaster_p2"}'))]
        resp.usage = MagicMock(prompt_tokens=3, completion_tokens=2)
        create.return_value = resp
        client.chat.completions.create = create
        mock_ctor.return_value = client

        await _call_router_llm(
            "hello",
            {"provider": "openai", "model": "gpt-4o-mini"},
            secret_value=secret,
        )

    kwargs = mock_ctor.call_args.kwargs
    assert kwargs["timeout"] == 90.0
    assert kwargs["max_retries"] == 0


@pytest.mark.asyncio
async def test_chat_wire_no_retry_on_timeout() -> None:
    """A never-responding model yields exactly one outgoing attempt.

    Verifies the acceptance criterion: counter on the mock == 1, and the
    timeout propagates so route_user_message emits ROUTE_LLM_FAILURE.
    """
    secret = SecretValue("sk-test")

    async def _boom(*_a: Any, **_kw: Any) -> None:
        raise asyncio.TimeoutError("Request timed out.")

    with patch("app.core.router.AsyncOpenAI") as mock_ctor:
        client = MagicMock()
        create = AsyncMock(side_effect=_boom)
        client.chat.completions.create = create
        mock_ctor.return_value = client

        with pytest.raises(asyncio.TimeoutError):
            await _call_router_llm(
                "hello",
                {"provider": "openai", "model": "gpt-4o-mini"},
                secret_value=secret,
            )

    assert create.await_count == 1


@pytest.mark.asyncio
async def test_messages_wire_passes_90s_timeout() -> None:
    """The httpx client on the messages wire must use the 90 s timeout."""
    secret = SecretValue("sk-test")

    captured: dict[str, Any] = {}

    class _FakeAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["timeout"] = kwargs.get("timeout")

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def post(self, *_a: Any, **_kw: Any) -> Any:
            m = MagicMock()
            m.json.return_value = {
                "content": [{"type": "text", "text": '{"route":"chatmaster_p2"}'}],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
            return m

    with patch("app.core.router.httpx.AsyncClient", _FakeAsyncClient):
        await _call_router_llm(
            "hello",
            {"provider": "opencode_go", "model": "qwen3.7-plus"},
            secret_value=secret,
        )

    assert captured["timeout"] == 90.0


@pytest.mark.asyncio
async def test_slow_mock_within_window_succeeds() -> None:
    """A mock that 'responds on the 60th second' (here: instantly) succeeds.

    The point of the 90 s window is that a 60 s response no longer trips the
    timeout. We simulate the slow-but-in-window path without sleeping 60 s:
    the mock returns immediately, and the timeout boundary is asserted above.
    """
    secret = SecretValue("sk-test")

    with patch("app.core.router.AsyncOpenAI") as mock_ctor:
        client = MagicMock()
        create = AsyncMock()
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content='{"route":"chatmaster_p2"}'))]
        resp.usage = MagicMock(prompt_tokens=3, completion_tokens=2)
        create.return_value = resp
        client.chat.completions.create = create
        mock_ctor.return_value = client

        text, _in_tok, _out_tok, model = await _call_router_llm(
            "hello",
            {"provider": "openai", "model": "gpt-4o-mini"},
            secret_value=secret,
        )

    assert "chatmaster_p2" in text
    assert create.await_count == 1


# ---------------------------------------------------------------------------
# Stage 1 (R1) — router inherits the ChatMaster's model
# ---------------------------------------------------------------------------


def _make_agent(
    *,
    nickname: str = "agent",
    llm_config: dict[str, Any] | None = None,
    agent_id: uuid.UUID | None = None,
) -> "Agent":
    from app.models.agent import Agent

    return Agent(
        id=agent_id or uuid.uuid4(),
        owner_id=uuid.uuid4(),
        nickname=nickname,
        agent_uid=nickname,
        token=f"tok-{nickname}",
        llm_config=llm_config or {"provider": "opencode_go", "model": "glm-5.3"},
    )


class TestResolveRouterAgent:
    """``_resolve_router_agent`` is the single point that decides which agent's
    model the router speaks with — always the ChatMaster in a group room, the
    sole agent in a direct room, None for a misconfigured group room."""

    def test_direct_room_returns_its_only_agent(self) -> None:
        from app.core.router import _resolve_router_agent

        room = _make_room()
        room.room_type = RoomType.direct
        sole = _make_agent(nickname="sole")

        assert _resolve_router_agent(room, [sole]) is sole

    def test_direct_room_without_agents_returns_none(self) -> None:
        from app.core.router import _resolve_router_agent

        room = _make_room()
        room.room_type = RoomType.direct

        assert _resolve_router_agent(room, []) is None

    def test_group_room_returns_chatmaster_not_first_agent(self) -> None:
        """The router must inherit the ChatMaster's model, not the first agent
        with a configured provider — the old behaviour picked a random row
        (plan §0.2)."""
        from app.core.router import _resolve_router_agent

        master = _make_agent(nickname="master")
        other = _make_agent(
            nickname="other",
            llm_config={"provider": "opencode_go", "model": "deepseek-v4-flash"},
        )
        room = _make_room()
        room.room_type = RoomType.group
        room.coordinator_config = {"master_agent_id": str(master.id)}

        # `other` is first in the list — the old loop would have picked it.
        assert _resolve_router_agent(room, [other, master]) is master

    def test_group_room_without_master_returns_none(self) -> None:
        from app.core.router import _resolve_router_agent

        room = _make_room()
        room.room_type = RoomType.group
        room.coordinator_config = {}
        agents = [_make_agent(nickname="a"), _make_agent(nickname="b")]

        assert _resolve_router_agent(room, agents) is None

    def test_group_room_accepts_all_three_key_names(self) -> None:
        """The triple ``agent_id`` / ``chat_master_agent_id`` /
        ``master_agent_id`` is the contract shared with chat_master.py and
        plan_session.py via ``app.core.room_coordinator``."""
        from app.core.router import _resolve_router_agent

        master = _make_agent(nickname="master")
        room = _make_room()
        room.room_type = RoomType.group
        room.coordinator_config = {"agent_id": str(master.id)}

        assert _resolve_router_agent(room, [master]) is master

        room.coordinator_config = {"chat_master_agent_id": str(master.id)}
        assert _resolve_router_agent(room, [master]) is master


@pytest.mark.asyncio
async def test_group_room_without_master_emits_chatmaster_missing() -> None:
    """A group room with no ChatMaster surfaces a visible warning and falls
    back to the deterministic rule — no exception (plan §2.2 / §2.3)."""
    room = _make_room()  # room_type defaults to group, coordinator_config empty
    db = _mock_db()
    raw = _llm_payload(route="chatmaster_p2", confidence=0.7, reasoning="fallback")

    with (
        patch("app.core.router._call_router_llm", return_value=_llm_return(raw)),
        patch("app.core.router.emit_event", new=AsyncMock()) as mock_emit,
    ):
        decision = await route_user_message(room, "hello", db)

    assert decision.route == "chatmaster_p2"
    types = _emitted_event_types(mock_emit)
    assert "ROOM_CHATMASTER_MISSING" in types
    assert "ROUTE_DECIDED" in types


@pytest.mark.asyncio
async def test_group_room_with_master_does_not_emit_chatmaster_missing() -> None:
    """When the ChatMaster is configured and present in the room, no
    ROOM_CHATMASTER_MISSING event fires and the router uses the master's
    model."""
    from app.models.agent import Agent

    master = _make_agent(
        nickname="master",
        llm_config={"provider": "opencode_go", "model": "glm-5.3"},
    )
    room = _make_room()
    room.room_type = RoomType.group
    room.coordinator_config = {"master_agent_id": str(master.id)}

    db = AsyncMock()
    agents_res = MagicMock()
    agents_res.scalars.return_value.all.return_value = [master]
    waiting_res = MagicMock()
    waiting_res.scalars.return_value.all.return_value = []
    goals_res = MagicMock()
    goals_res.scalars.return_value.all.return_value = []
    db.execute.side_effect = [agents_res, waiting_res, goals_res]

    raw = _llm_payload(route="chatmaster_p2", confidence=0.8, reasoning="ok")

    with (
        patch("app.core.router._call_router_llm", return_value=_llm_return(raw)),
        patch("app.core.router.emit_event", new=AsyncMock()) as mock_emit,
        patch("app.core.router.resolve_agent_secret", new=AsyncMock(return_value=SecretValue("k"))),
    ):
        decision = await route_user_message(room, "hello", db)

    assert decision.route == "chatmaster_p2"
    types = _emitted_event_types(mock_emit)
    assert "ROOM_CHATMASTER_MISSING" not in types
    assert "ROUTE_DECIDED" in types


@pytest.mark.asyncio
async def test_direct_room_uses_its_only_agent_no_missing_event() -> None:
    """A direct room's sole agent is its coordinator — no missing-master
    warning, the router speaks that agent's model (plan §2.3)."""
    sole = _make_agent(
        nickname="sole",
        llm_config={"provider": "opencode_go", "model": "glm-5.2"},
    )
    room = _make_room()
    room.room_type = RoomType.direct

    db = AsyncMock()
    agents_res = MagicMock()
    agents_res.scalars.return_value.all.return_value = [sole]
    waiting_res = MagicMock()
    waiting_res.scalars.return_value.all.return_value = []
    goals_res = MagicMock()
    goals_res.scalars.return_value.all.return_value = []
    db.execute.side_effect = [agents_res, waiting_res, goals_res]

    raw = _llm_payload(route="chatmaster_p2", confidence=0.8, reasoning="ok")

    with (
        patch("app.core.router._call_router_llm", return_value=_llm_return(raw)),
        patch("app.core.router.emit_event", new=AsyncMock()) as mock_emit,
        patch("app.core.router.resolve_agent_secret", new=AsyncMock(return_value=SecretValue("k"))),
    ):
        decision = await route_user_message(room, "hello", db)

    assert decision.route == "chatmaster_p2"
    types = _emitted_event_types(mock_emit)
    assert "ROOM_CHATMASTER_MISSING" not in types


def test_room_coordinator_get_cm_agent_id_reads_triple_keys() -> None:
    """The shared helper reads the three historical key names and tolerates
    garbage (returns None instead of raising)."""
    from app.core.room_coordinator import get_cm_agent_id

    room = _make_room()
    room.coordinator_config = {}
    assert get_cm_agent_id(room) is None

    aid = uuid.uuid4()
    room.coordinator_config = {"master_agent_id": str(aid)}
    assert get_cm_agent_id(room) == aid

    room.coordinator_config = {"chat_master_agent_id": str(aid)}
    assert get_cm_agent_id(room) == aid

    room.coordinator_config = {"agent_id": str(aid)}
    assert get_cm_agent_id(room) == aid

    room.coordinator_config = {"master_agent_id": "not-a-uuid"}
    assert get_cm_agent_id(room) is None



class TestNullMentionsTolerated:
    """`"mentioned_agent_ids": null` must not cost us the whole decision.

    Observed in production 2026-09-04 (deepseek-v4-pro): the reply was valid
    JSON, but the mentions list came back as `null` — every other optional
    field in the prompt is spelled `| null` — and the ValidationError sent the
    turn down the fallback route.
    """

    def test_null_mentions_parse_as_empty_list(self) -> None:
        from app.core.router import _parse_router_reply

        decision = _parse_router_reply(
            json.dumps(
                {
                    "route": "chatmaster_p2",
                    "mentioned_agent_ids": None,
                    "can_split": None,
                    "split_messages": None,
                    "matched_task_id": None,
                    "matched_goal_id": None,
                    "risk_hint": None,
                    "confidence": 0.9,
                    "reasoning": "no mentions",
                }
            )
        )
        assert decision.route == "chatmaster_p2"
        assert decision.mentioned_agent_ids == []

    def test_real_mentions_still_parse(self) -> None:
        from app.core.router import _parse_router_reply

        aid = uuid.uuid4()
        decision = _parse_router_reply(
            json.dumps(
                {
                    "route": "single_agent_p2",
                    "mentioned_agent_ids": [str(aid)],
                    "confidence": 0.8,
                    "reasoning": "one @mention",
                }
            )
        )
        assert decision.mentioned_agent_ids == [aid]

    def test_garbage_mentions_still_rejected(self) -> None:
        from app.core.router import _parse_router_reply

        with pytest.raises(ValidationError):
            _parse_router_reply(
                json.dumps(
                    {
                        "route": "chatmaster_p2",
                        "mentioned_agent_ids": "none",
                        "confidence": 0.8,
                        "reasoning": "x",
                    }
                )
            )
