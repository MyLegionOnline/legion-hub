"""
Regression suite — all 6 router routes and all 6 session kinds.

This file is a "smoke test": one file, all the critical paths.
On any regression at least one test here will fail.

Router routes (§8.1):
  multi_agent_split, multi_agent_ask_detail,
  single_agent_p1, single_agent_p2,
  chatmaster_p1, chatmaster_p2

Session kinds (§8.4):
  user_agent_p1, user_agent_p2,
  user_cm_p1, user_cm_p2,
  cm_self_plan, cm_agent_proxy

Run:
    pytest tests/test_regression_all_routes_and_kinds.py -v
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.core.router import RouteDecision, SplitPart, _apply_confidence_fallback
from app.schemas.session import (
    CmAgentProxySessionCreate,
    CmSelfPlanSessionCreate,
    UserAgentSessionCreate,
    UserCmSessionCreate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ROOM_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()
_AGENT_ID_A = uuid.uuid4()
_AGENT_ID_B = uuid.uuid4()
_TASK_ID = uuid.uuid4()
_GOAL_ID = uuid.uuid4()
_PLAN_ID = uuid.uuid4()


def _make_room(coordinator_agent_id: uuid.UUID | None = None) -> MagicMock:
    room = MagicMock()
    room.id = _ROOM_ID
    room.coordinator_config = {"agent_id": str(coordinator_agent_id)} if coordinator_agent_id else {}
    return room


def _make_db() -> AsyncMock:
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = []
    db.execute.return_value = result
    return db


# ---------------------------------------------------------------------------
# 1. Router — all 6 routes
# ---------------------------------------------------------------------------


ROUTE_CASES = [
    (
        "multi_agent_split",
        RouteDecision(
            route="multi_agent_split",
            mentioned_agent_ids=[_AGENT_ID_A, _AGENT_ID_B],
            can_split=True,
            split_messages=[
                SplitPart(agent_id=_AGENT_ID_A, text="Fix the login bug"),
                SplitPart(agent_id=_AGENT_ID_B, text="Update the docs"),
            ],
            confidence=0.9,
            reasoning="two agents, clear split",
        ),
    ),
    (
        "multi_agent_ask_detail",
        RouteDecision(
            route="multi_agent_ask_detail",
            mentioned_agent_ids=[_AGENT_ID_A, _AGENT_ID_B],
            can_split=False,
            confidence=0.75,
            reasoning="cannot split cleanly",
        ),
    ),
    (
        "single_agent_p1",
        RouteDecision(
            route="single_agent_p1",
            mentioned_agent_ids=[_AGENT_ID_A],
            matched_task_id=_TASK_ID,
            confidence=0.88,
            reasoning="matches existing task",
        ),
    ),
    (
        "single_agent_p2",
        RouteDecision(
            route="single_agent_p2",
            mentioned_agent_ids=[_AGENT_ID_A],
            confidence=0.82,
            reasoning="no matching task",
        ),
    ),
    (
        "chatmaster_p1",
        RouteDecision(
            route="chatmaster_p1",
            matched_goal_id=_GOAL_ID,
            confidence=0.91,
            reasoning="matches Goal X",
        ),
    ),
    (
        "chatmaster_p2",
        RouteDecision(
            route="chatmaster_p2",
            confidence=0.79,
            reasoning="no matching goal",
        ),
    ),
]


class TestAllRouterRoutes:
    """Smoke tests: route_user_message returns correct route for each of 6 cases."""

    @pytest.mark.parametrize("expected_route,decision", ROUTE_CASES)
    @pytest.mark.asyncio
    async def test_route_returns_correct_route(
        self, expected_route: str, decision: RouteDecision
    ):
        from app.core.router import route_user_message

        room = _make_room()
        db = _make_db()

        # _call_router_llm returns (raw_json, in_tokens, out_tokens, model)
        llm_return = (decision.model_dump_json(), 10, 5, "router-test-model")

        with (
            patch("app.core.router._call_router_llm", new_callable=AsyncMock, return_value=llm_return),
            patch("app.core.router.emit_event", new_callable=AsyncMock),
        ):
            result = await route_user_message(room, "test message", db)

        assert result.route == expected_route

    @pytest.mark.parametrize("expected_route,decision", ROUTE_CASES)
    def test_route_decision_schema_valid(
        self, expected_route: str, decision: RouteDecision
    ):
        """All 6 RouteDecision variants round-trip through the schema."""
        dumped = decision.model_dump()
        restored = RouteDecision(**dumped)
        assert restored.route == expected_route
        assert 0.0 <= restored.confidence <= 1.0

    @pytest.mark.parametrize("expected_route,decision", ROUTE_CASES)
    def test_confidence_fallback_does_not_raise(
        self, expected_route: str, decision: RouteDecision
    ):
        """_apply_confidence_fallback must handle all 6 routes without exception."""
        low_conf = decision.model_copy(update={"confidence": 0.4})
        result = _apply_confidence_fallback(low_conf)
        # Low confidence must push to a p2 variant or ask_detail
        assert result.route in {
            "single_agent_p2",
            "chatmaster_p2",
            "multi_agent_ask_detail",
            # multi_agent_split stays if can_split is True and confidence was already low
            "multi_agent_split",
        }


# ---------------------------------------------------------------------------
# 2. Session kinds — all 6 kinds
# ---------------------------------------------------------------------------


class TestAllSessionKindsSchemas:
    """Pydantic schema validation for all 6 kinds."""

    def test_user_agent_p1_requires_task_and_goal(self):
        with pytest.raises(ValidationError):
            UserAgentSessionCreate(kind="user_agent_p1", room_id=_ROOM_ID, user_id=_USER_ID)

    def test_user_agent_p1_valid(self):
        schema = UserAgentSessionCreate(
            kind="user_agent_p1",
            room_id=_ROOM_ID,
            user_id=_USER_ID,
            agent_id=_AGENT_ID_A,
            task_id=_TASK_ID,
            goal_id=_GOAL_ID,
        )
        assert schema.kind == "user_agent_p1"

    def test_user_agent_p2_no_task_required(self):
        schema = UserAgentSessionCreate(
            kind="user_agent_p2",
            room_id=_ROOM_ID,
            user_id=_USER_ID,
            agent_id=_AGENT_ID_A,
        )
        assert schema.kind == "user_agent_p2"
        assert schema.task_id is None

    def test_user_cm_p1_requires_goal(self):
        with pytest.raises(ValidationError):
            UserCmSessionCreate(
                kind="user_cm_p1",
                room_id=_ROOM_ID,
                user_id=_USER_ID,
                chat_master_agent_id=_AGENT_ID_A,
            )

    def test_user_cm_p1_valid(self):
        schema = UserCmSessionCreate(
            kind="user_cm_p1",
            room_id=_ROOM_ID,
            user_id=_USER_ID,
            chat_master_agent_id=_AGENT_ID_A,
            goal_id=_GOAL_ID,
        )
        assert schema.kind == "user_cm_p1"

    def test_user_cm_p2_valid_no_goal(self):
        schema = UserCmSessionCreate(
            kind="user_cm_p2",
            room_id=_ROOM_ID,
            user_id=_USER_ID,
            chat_master_agent_id=_AGENT_ID_A,
        )
        assert schema.kind == "user_cm_p2"
        assert schema.goal_id is None

    def test_cm_self_plan_requires_plan_id_and_goal_id(self):
        with pytest.raises(ValidationError):
            CmSelfPlanSessionCreate(
                kind="cm_self_plan",
                room_id=_ROOM_ID,
                chat_master_agent_id=_AGENT_ID_A,
            )

    def test_cm_self_plan_valid(self):
        schema = CmSelfPlanSessionCreate(
            kind="cm_self_plan",
            room_id=_ROOM_ID,
            chat_master_agent_id=_AGENT_ID_A,
            goal_id=_GOAL_ID,
            plan_id=_PLAN_ID,
        )
        assert schema.kind == "cm_self_plan"

    def test_cm_agent_proxy_requires_side_task_and_agent(self):
        with pytest.raises(ValidationError):
            CmAgentProxySessionCreate(
                kind="cm_agent_proxy",
                room_id=_ROOM_ID,
            )

    def test_cm_agent_proxy_valid(self):
        schema = CmAgentProxySessionCreate(
            kind="cm_agent_proxy",
            room_id=_ROOM_ID,
            side="agent",
            task_id=_TASK_ID,
            goal_id=_GOAL_ID,
            plan_id=_PLAN_ID,
            agent_id=_AGENT_ID_A,
            peer_session_id=uuid.uuid4(),
        )
        assert schema.kind == "cm_agent_proxy"
        assert schema.side == "agent"

    def test_cm_agent_proxy_allows_direct_task_without_goal_or_plan(self):
        schema = CmAgentProxySessionCreate(
            kind="cm_agent_proxy",
            room_id=_ROOM_ID,
            side="agent",
            task_id=_TASK_ID,
            agent_id=_AGENT_ID_A,
        )
        assert schema.goal_id is None
        assert schema.plan_id is None


# ---------------------------------------------------------------------------
# 3. Session close actions — matrix kind → allowed fields
# ---------------------------------------------------------------------------


# List-typed SessionCloseActions fields (empty value is [] rather than None)
_CLOSE_LIST_FIELDS = {"goals_create", "tasks_create", "tasks_update", "tasks_cancel", "file_writes"}

SESSION_CLOSE_MATRIX = [
    # (kind, side, field, allowed) — side only matters for cm_agent_proxy
    ("user_agent_p1", None, "agent_memory_patch", True),
    ("user_agent_p1", None, "room_memory_patch", False),
    ("user_agent_p1", None, "plan_update", False),
    ("user_agent_p1", None, "goals_create", False),
    ("user_agent_p1", None, "tasks_create", False),
    ("user_agent_p1", None, "file_writes", False),
    ("user_agent_p2", None, "agent_memory_patch", True),
    ("user_agent_p2", None, "room_memory_patch", False),
    ("user_cm_p1", None, "agent_memory_patch", True),
    ("user_cm_p1", None, "room_memory_patch", True),
    ("user_cm_p1", None, "plan_update", True),
    ("user_cm_p1", None, "goals_create", False),
    ("user_cm_p1", None, "tasks_create", True),
    ("user_cm_p1", None, "tasks_cancel", True),
    ("user_cm_p2", None, "agent_memory_patch", True),
    ("user_cm_p2", None, "room_memory_patch", True),
    ("user_cm_p2", None, "goals_create", True),
    ("user_cm_p2", None, "plan_update", True),
    ("user_cm_p2", None, "tasks_create", False),
    ("cm_self_plan", None, "tasks_create", True),
    ("cm_self_plan", None, "agent_memory_patch", False),
    ("cm_self_plan", None, "room_memory_patch", False),
    ("cm_self_plan", None, "goals_create", False),
    ("cm_agent_proxy", "agent", "agent_memory_patch", True),
    ("cm_agent_proxy", "agent", "tasks_update", True),
    ("cm_agent_proxy", "agent", "file_writes", True),
    ("cm_agent_proxy", "agent", "room_memory_patch", False),
    ("cm_agent_proxy", "agent", "plan_update", False),
    ("cm_agent_proxy", "chat_master", "agent_memory_patch", True),
    ("cm_agent_proxy", "chat_master", "room_memory_patch", True),
    ("cm_agent_proxy", "chat_master", "plan_update", True),
    ("cm_agent_proxy", "chat_master", "tasks_update", False),
    ("cm_agent_proxy", "chat_master", "file_writes", False),
]


class TestSessionCloseActionsMatrix:
    """filter_by_kind strips disallowed fields for each session kind."""

    @pytest.mark.parametrize("kind,side,field,allowed", SESSION_CLOSE_MATRIX)
    def test_filter_by_kind(self, kind: str, side: str | None, field: str, allowed: bool):
        from app.core.session_close import (
            FileWrite,
            GoalCreate,
            MemoryPatch,
            PlanUpdate,
            SessionCloseActions,
            TaskCreate,
            TaskUpdate,
            filter_by_kind,
        )

        actions = SessionCloseActions(
            room_memory_patch=MemoryPatch(content="room"),
            agent_memory_patch=MemoryPatch(content="agent"),
            goals_create=[GoalCreate(title="G", description="D")],
            plan_update=PlanUpdate(plan_id=_PLAN_ID),
            tasks_create=[TaskCreate(title="T", description="D")],
            tasks_update=[TaskUpdate(task_id=_TASK_ID, state="done")],
            tasks_cancel=[_TASK_ID],
            file_writes=[FileWrite(path="a.txt", content="x")],
            summary="done",
        )

        session = MagicMock()
        session.id = uuid.uuid4()
        session.kind = kind
        session.side = side

        filtered, _ = filter_by_kind(actions, session)

        val = getattr(filtered, field)
        if field in _CLOSE_LIST_FIELDS:
            is_present = len(val) > 0
        else:
            is_present = val is not None

        if allowed:
            assert is_present, f"{field} should be allowed for {kind} (side={side})"
        else:
            assert not is_present, f"{field} should NOT be allowed for {kind} (side={side})"


# ---------------------------------------------------------------------------
# 4. Token budget — limit 100k per session
# ---------------------------------------------------------------------------


def _make_budget_session(tokens_in: int, tokens_out: int, is_open: bool = True) -> MagicMock:
    session = MagicMock()
    session.id = uuid.uuid4()
    session.room_id = _ROOM_ID
    session.kind = "user_agent_p2"
    session.is_open = is_open
    session.tokens_in = tokens_in
    session.tokens_out = tokens_out
    return session


class TestTokenBudgetRegression:
    """check_token_budget must fire exactly at the 100k threshold.

    The estimate for the next call is max(1, len(next_input) // 4) + 1000
    (fixed conservative output estimate) — see estimate_next_call_tokens.
    """

    @pytest.mark.asyncio
    async def test_budget_ok_below_limit(self):
        from app.core.token_budget import check_token_budget

        # 40k + 30k + (1000 input + 1000 output) = 72_000 < 100_000
        session = _make_budget_session(tokens_in=40_000, tokens_out=30_000)
        db = _make_db()

        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as emit:
            result = await check_token_budget(session, "x" * 4_000, db)

        assert result is True  # OK, can proceed
        db.add.assert_not_called()
        emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_budget_exceeded_over_limit(self):
        from app.core.token_budget import check_token_budget

        # 50k + 49k + (1 + 1000) = 100_001 > 100_000 → close + event
        session = _make_budget_session(tokens_in=50_000, tokens_out=49_000)
        db = _make_db()

        with (
            patch("app.core.token_budget.uses_cumulative_limit", return_value=True),
            patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as emit,
        ):
            result = await check_token_budget(session, "", db)

        assert result is False
        assert session.close_reason == "token_limit"
        assert session.closed_at is not None
        db.add.assert_called_once_with(session)
        emit.assert_awaited_once()
        assert emit.await_args.kwargs["event_type"] == "SESSION_TOKEN_LIMIT_REACHED"

    @pytest.mark.asyncio
    async def test_budget_exactly_at_limit(self):
        from app.core.token_budget import check_token_budget

        # 50k + 48_999 + (1 + 1000) = exactly 100_000 — limit is inclusive
        session = _make_budget_session(tokens_in=50_000, tokens_out=48_999)
        db = _make_db()

        with (
            patch("app.core.token_budget.uses_cumulative_limit", return_value=True),
            patch("app.core.token_budget.emit_event", new_callable=AsyncMock),
        ):
            result = await check_token_budget(session, "", db)

        assert result is False

    @pytest.mark.asyncio
    async def test_budget_closed_session_returns_false(self):
        from app.core.token_budget import check_token_budget

        session = _make_budget_session(tokens_in=0, tokens_out=0, is_open=False)
        db = _make_db()

        with patch("app.core.token_budget.emit_event", new_callable=AsyncMock) as emit:
            result = await check_token_budget(session, "hi", db)

        assert result is False
        emit.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Proxy broker — publish and peer validation
# ---------------------------------------------------------------------------


def _proxy_msg(peer_session_id: uuid.UUID | None = None) -> Any:
    from app.schemas.proxy import ProxyMessageCreate

    return ProxyMessageCreate(
        room_id=_ROOM_ID,
        sender_type="agent",
        sender_id=_AGENT_ID_A,
        content="code here",
        task_id=_TASK_ID,
        session_id=uuid.uuid4(),
        peer_session_id=peer_session_id or uuid.uuid4(),
    )


def _exec_result(first: Any = None, scalar: Any = None) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value.first.return_value = first
    result.scalar.return_value = scalar
    return result


class TestProxyBrokerRegression:
    """ProxyBroker.publish must write to room history and copy to peer inbox.

    A missing peer session raises ProxyBrokerError; a closed peer is
    tolerated (persist + broadcast for room visibility, queue skipped).
    """

    @pytest.mark.asyncio
    async def test_publish_rejected_when_peer_missing(self):
        from app.core.proxy_broker import ProxyBroker, ProxyBrokerError

        broker = ProxyBroker()
        db = _make_db()
        db.execute = AsyncMock(return_value=_exec_result(first=None))  # peer lookup → not found

        with pytest.raises(ProxyBrokerError):
            await broker.publish(_proxy_msg(), db)

        db.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_publish_tolerates_closed_peer(self):
        from app.core.proxy_broker import ProxyBroker

        broker = ProxyBroker()
        peer_session_id = uuid.uuid4()
        msg = _proxy_msg(peer_session_id=peer_session_id)

        closed_peer = MagicMock()
        closed_peer.is_open = False

        db = _make_db()
        db.refresh = AsyncMock()
        db.execute = AsyncMock(
            side_effect=[
                _exec_result(first=closed_peer),  # peer session lookup
                _exec_result(first=None),         # sender agent (nickname) lookup
                _exec_result(scalar=3),           # PROXY_MESSAGE count → turn
            ]
        )

        ProxyBroker.clear_queue(peer_session_id)
        with patch("app.core.proxy_broker.manager.broadcast", new_callable=AsyncMock) as broadcast:
            persisted = await broker.publish(msg, db)

        # (a) persisted to room history
        db.add.assert_called_once()
        assert persisted.event_type == "PROXY_MESSAGE"
        assert persisted.event_metadata["turn"] == 3
        assert persisted.event_metadata["peer_session_id"] == str(peer_session_id)
        # (b) closed peer → its input queue must NOT be created
        assert peer_session_id not in ProxyBroker._queues
        # (c) still broadcast to the room
        broadcast.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Memory layer — presence of fields for each kind
# ---------------------------------------------------------------------------


MEMORY_LAYER_FIELDS = {
    "user_agent_p1": {"room_memory", "repo_snapshot", "agents_list", "goal", "plan_of_goal", "task_data", "agent_memory", "skills"},
    "user_agent_p2": {"room_memory", "repo_snapshot", "agents_list", "list_of_goals", "agent_memory", "skills"},
    # Stage 3.2 of internal design notes: the coordinator
    # kinds see the board instead of (p1) / in addition to (p2) what they had.
    "user_cm_p1": {"room_memory", "repo_snapshot", "agents_list", "goal", "plan_of_goal", "board_snapshot", "agent_memory"},
    "user_cm_p2": {"room_memory", "repo_snapshot", "agents_list", "agent_memory", "list_of_goals", "board_snapshot"},
    "cm_self_plan": {"room_memory", "agents_list", "repo_snapshot"},
    "cm_agent_proxy": {"room_memory", "repo_snapshot", "agents_list", "goal", "plan_of_goal", "task_data", "agent_memory", "skills"},
}

MEMORY_ABSENT_FIELDS = {
    "user_agent_p1": {"list_of_goals", "list_tasks_last10"},
    "user_agent_p2": {"goal", "plan_of_goal", "task_data", "list_tasks_last10"},
    "user_cm_p1": {"list_of_goals", "task_data", "list_tasks_last10"},
    "user_cm_p2": {"goal", "plan_of_goal", "task_data", "list_tasks_last10"},
    "cm_self_plan": {"goal", "plan_of_goal", "task_data", "list_of_goals", "list_tasks_last10", "agent_memory"},
}


class TestMemoryLayerRegression:
    """build_memory_layer must include/exclude correct fields per §3.5."""

    @pytest.mark.parametrize("kind", list(MEMORY_LAYER_FIELDS.keys()))
    @pytest.mark.asyncio
    async def test_memory_layer_includes_required_fields(self, kind: str):
        from app.core.memory_layer import build_memory_layer

        session = MagicMock()
        session.kind = kind
        session.room_id = _ROOM_ID
        session.goal_id = _GOAL_ID if "goal" in MEMORY_LAYER_FIELDS.get(kind, set()) else None
        session.task_id = _TASK_ID if "task_data" in MEMORY_LAYER_FIELDS.get(kind, set()) else None
        session.agent_id = _AGENT_ID_A
        session.chat_master_agent_id = _AGENT_ID_B

        with (
            patch("app.core.memory_layer._room_memory", new_callable=AsyncMock, return_value="room mem"),
            patch("app.core.memory_layer._repo_snapshot", new_callable=AsyncMock, return_value=[]),
            patch("app.core.memory_layer._agents_list_short", new_callable=AsyncMock, return_value=[]),
            patch("app.core.memory_layer._goal", new_callable=AsyncMock, return_value="goal data"),
            patch("app.core.memory_layer._list_of_goals", new_callable=AsyncMock, return_value=[]),
            patch("app.core.memory_layer._plan_of_goal", new_callable=AsyncMock, return_value="plan"),
            patch("app.core.memory_layer._task_data", new_callable=AsyncMock, return_value="task data"),
            patch("app.core.memory_layer._list_tasks_last10", new_callable=AsyncMock, return_value=[]),
            patch("app.core.memory_layer._agent_memory", new_callable=AsyncMock, return_value="agent mem"),
            patch("app.core.memory_layer._agent_skills", new_callable=AsyncMock, return_value="skill content"),
        ):
            db = _make_db()
            layer = await build_memory_layer(session, db)

        expected_keys = MEMORY_LAYER_FIELDS[kind]
        for field in expected_keys:
            assert field in layer, f"Expected '{field}' in memory layer for kind={kind}"

    @pytest.mark.parametrize("kind,absent_fields", [
        (k, v) for k, v in MEMORY_ABSENT_FIELDS.items()
    ])
    @pytest.mark.asyncio
    async def test_memory_layer_excludes_forbidden_fields(
        self, kind: str, absent_fields: set
    ):
        from app.core.memory_layer import build_memory_layer

        session = MagicMock()
        session.kind = kind
        session.room_id = _ROOM_ID
        session.goal_id = None
        session.task_id = None
        session.agent_id = _AGENT_ID_A
        session.chat_master_agent_id = _AGENT_ID_B

        with (
            patch("app.core.memory_layer._room_memory", new_callable=AsyncMock, return_value="room mem"),
            patch("app.core.memory_layer._repo_snapshot", new_callable=AsyncMock, return_value=[]),
            patch("app.core.memory_layer._agents_list_short", new_callable=AsyncMock, return_value=[]),
            patch("app.core.memory_layer._goal", new_callable=AsyncMock, return_value="goal data"),
            patch("app.core.memory_layer._list_of_goals", new_callable=AsyncMock, return_value=[]),
            patch("app.core.memory_layer._plan_of_goal", new_callable=AsyncMock, return_value="plan"),
            patch("app.core.memory_layer._task_data", new_callable=AsyncMock, return_value="task data"),
            patch("app.core.memory_layer._list_tasks_last10", new_callable=AsyncMock, return_value=[]),
            patch("app.core.memory_layer._agent_memory", new_callable=AsyncMock, return_value="agent mem"),
            patch("app.core.memory_layer._agent_skills", new_callable=AsyncMock, return_value="skill content"),
        ):
            db = _make_db()
            layer = await build_memory_layer(session, db)

        for field in absent_fields:
            assert field not in layer or layer.get(field) is None, \
                f"Field '{field}' should be absent for kind={kind}"
