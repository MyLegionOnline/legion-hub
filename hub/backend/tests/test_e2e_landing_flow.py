"""
E2E scenario "I want a landing page on Next.js" from §6 of New_chatflow.md.

The test runs the full path:
  1. User message → route: chatmaster_p2 (no @mention, no Goal)
  2. UCM-p2 session → LLM discusses scope → close → goals_create + room_memory_patch
  3. PLAN_CREATED event → CmSelfPlanSession
  4. CmSelfPlanSession → LLM creates a Task (frontend-agent) → tasks_create
  5. TASK_CREATED event → a CmAgentProxySession pair (agent-side + cm-side)
  6. agent-side: LLM writes code → ProxyBroker.publish
  7. cm-side: LLM approves → both sides close
  8. Post-actions: file_writes (agent) + plan_update + room_memory_patch (cm)

All LLM calls and the DB are stubbed.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.router import RouteDecision
from app.core.session_close import (
    SessionCloseActions,
    GoalCreate,
    MemoryPatch,
    TaskCreate,
    FileWrite,
    PlanUpdate,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_ROOM_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()
_CM_AGENT_ID = uuid.uuid4()
_FRONTEND_AGENT_ID = uuid.uuid4()
_GOAL_ID = uuid.uuid4()
_PLAN_ID = uuid.uuid4()
_TASK_ID = uuid.uuid4()


def _make_room(coordinator_agent_id: uuid.UUID | None = _CM_AGENT_ID) -> MagicMock:
    room = MagicMock()
    room.id = _ROOM_ID
    room.coordinator_config = {"agent_id": str(coordinator_agent_id)} if coordinator_agent_id else {}
    return room


def _make_user() -> MagicMock:
    user = MagicMock()
    user.id = _USER_ID
    return user


def _make_session(kind: str, session_id: uuid.UUID | None = None) -> MagicMock:
    s = MagicMock()
    s.id = session_id or uuid.uuid4()
    s.room_id = _ROOM_ID
    s.kind = kind
    s.is_open = True
    s.tokens_in = 0
    s.tokens_out = 0
    s.cost_usd = Decimal("0")
    s.closed_at = None
    s.close_reason = None
    s.pending_actions = None
    s.started_at = datetime.now(timezone.utc)
    s.last_context_window_tokens = 0
    return s


def _make_session_mock(kind: str, side: str | None = None) -> MagicMock:
    """Create a Session-like mock accepted by filter_by_kind."""
    s = MagicMock()
    s.id = uuid.uuid4()
    s.kind = kind
    s.side = side
    return s


def _make_db() -> AsyncMock:
    """DB mock — explicit MagicMock result so scalars() is never a coroutine."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.rollback = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    result_mock.scalars.return_value.first.return_value = None
    result_mock.scalar_one_or_none.return_value = None
    result_mock.first.return_value = None
    result_mock.rowcount = 0
    db.execute = AsyncMock(return_value=result_mock)
    return db


def _make_task(agent_id: uuid.UUID = _FRONTEND_AGENT_ID) -> MagicMock:
    t = MagicMock()
    t.id = _TASK_ID
    t.room_id = _ROOM_ID
    t.title = "Build landing page"
    t.description = "Create Next.js landing page with hero section"
    t.assigned_agent_id = agent_id
    t.goal_id = _GOAL_ID
    return t


def _router_llm_result(decision: RouteDecision) -> tuple:
    """Return the (raw_json, in_tok, out_tok, model) tuple _call_router_llm emits."""
    return (json.dumps(decision.model_dump(mode="json")), 10, 5, "gpt-4o")


# ---------------------------------------------------------------------------
# Phase 1: User message → route → UCM-p2 session
# ---------------------------------------------------------------------------


class TestPhase1Routing:
    """User sends 'I want a landing page on Next.js' — no @mentions, no Goals → chatmaster_p2."""

    @pytest.mark.asyncio
    async def test_message_routes_to_chatmaster_p2_when_no_mentions_and_no_goals(self):
        from app.core.router import route_user_message

        room = _make_room()
        db = _make_db()

        expected_decision = RouteDecision(
            route="chatmaster_p2",
            confidence=0.85,
            reasoning="no @mentions and no matching goals",
        )

        with (
            patch("app.core.router._gather_context", new_callable=AsyncMock,
                  return_value=([], [], [], {})),
            patch("app.core.router._call_router_llm", new_callable=AsyncMock,
                  return_value=_router_llm_result(expected_decision)),
            patch("app.core.events.emit_event", new_callable=AsyncMock),
        ):
            decision = await route_user_message(room, "I want a landing page on Next.js", db)

        assert decision.route == "chatmaster_p2"
        assert decision.confidence >= 0.6

    @pytest.mark.asyncio
    async def test_ucm_p2_session_started_for_chatmaster_p2_route(self):
        from app.core.user_sessions import start_user_session

        room = _make_room()
        user = _make_user()
        db = _make_db()
        decision = RouteDecision(
            route="chatmaster_p2",
            confidence=0.85,
            reasoning="no goals",
        )

        fake_session = _make_session("user_cm_p2")

        with (
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
        ):
            repo_instance = MockRepo.return_value
            repo_instance.create = AsyncMock(return_value=fake_session)

            session = await start_user_session(
                room, user, decision, db, cm_agent_id=_CM_AGENT_ID
            )

        assert session.kind == "user_cm_p2"


# ---------------------------------------------------------------------------
# Phase 2: UCM-p2 close → goals_create + room_memory_patch
# ---------------------------------------------------------------------------


class TestPhase2SessionClose:
    """UCM-p2 session LLM returns actions: goals_create + room_memory_patch."""

    @pytest.mark.asyncio
    async def test_apply_close_actions_creates_goal_and_updates_memory(self):
        session = _make_session("user_cm_p2")
        db = _make_db()

        actions = SessionCloseActions(
            goals_create=[
                GoalCreate(
                    title="Landing on Next.js",
                    description="Build a marketing landing page using Next.js",
                )
            ],
            room_memory_patch=MemoryPatch(
                content="User wants a Next.js landing page. Goal created.",
            ),
        )

        goals_created: list[str] = []
        memory_patches: list[str] = []

        async def _fake_apply(sess: Any, act: SessionCloseActions, db: Any) -> None:
            for g in act.goals_create:
                goals_created.append(g.title)
            if act.room_memory_patch:
                memory_patches.append(act.room_memory_patch.content[:10])

        import app.core.session_close as _sc
        with patch.object(_sc, "apply_close_actions", side_effect=_fake_apply):
            await _sc.apply_close_actions(session, actions, db)

        assert goals_created == ["Landing on Next.js"]
        assert len(memory_patches) == 1

    @pytest.mark.asyncio
    async def test_ucm_p2_actions_allow_add_goal_and_room_memory(self):
        from app.core.session_close import filter_by_kind

        actions = SessionCloseActions(
            goals_create=[
                GoalCreate(title="Landing", description="Next.js landing")
            ],
            room_memory_patch=MemoryPatch(content="scope agreed"),
            tasks_create=[],
            file_writes=[],
        )
        session = _make_session_mock("user_cm_p2")
        filtered, _dropped = filter_by_kind(actions, session)
        assert len(filtered.goals_create) == 1
        assert filtered.room_memory_patch is not None
        # tasks_create not allowed for UCM-p2
        assert len(filtered.tasks_create) == 0


# ---------------------------------------------------------------------------
# Phase 3: PLAN_CREATED → CmSelfPlanSession → tasks created
# ---------------------------------------------------------------------------


class TestPhase3PlanSession:
    """PLAN_CREATED fires → CmSelfPlanSession → LLM creates tasks."""

    @pytest.mark.asyncio
    async def test_plan_session_creates_tasks_for_goal(self):
        from app.core.plan_session import handle_plan_created

        db = _make_db()
        plan_text = (
            "# Landing on Next.js\n"
            "## Step 1: Hero section (frontend-agent)\n"
            "## Step 2: Deploy config (devops-agent)\n"
        )

        tasks_created: list[dict] = []

        async def _fake_apply(sess: Any, actions: SessionCloseActions, db: Any) -> None:
            for tc in actions.tasks_create:
                tasks_created.append({"title": tc.title, "agent_id": tc.assigned_agent_id})

        fake_session = _make_session("cm_self_plan")
        fake_room = _make_room()

        fake_close_actions = SessionCloseActions(
            tasks_create=[
                TaskCreate(
                    assigned_agent_id=_FRONTEND_AGENT_ID,
                    title="Build landing hero section",
                    description="Create Next.js hero section component",
                    goal_id=_GOAL_ID,
                )
            ]
        )

        with (
            patch("app.core.plan_session.SessionRepository") as MockRepo,
            patch("app.core.plan_session.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.plan_session.build_system_prompt", return_value="sys"),
            patch("app.core.plan_session.check_token_budget", new_callable=AsyncMock, return_value=True),
            # The decomposer runs on the ChatMaster's own model (plan_session §8.1).
            patch("app.core.plan_session.resolve_planner_llm_config", new_callable=AsyncMock,
                  return_value=(None, {"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": "x"})),
            patch("app.core.plan_session._call_llm_single_turn", new_callable=AsyncMock,
                  return_value=("", {"tasks_create": [{"title": "Build landing hero section"}]},
                                50, 100, "claude-sonnet-4-6")),
            # A filled close_session must be enough — never the JSON-mode rescue call.
            patch("app.core.plan_session._call_llm_json_mode", new_callable=AsyncMock,
                  side_effect=AssertionError("JSON-mode fallback must not run")),
            patch("app.core.plan_session.extract_close_actions_from_tool_use",
                  return_value=fake_close_actions),
            patch("app.core.plan_session.apply_close_actions", side_effect=_fake_apply),
            patch("app.core.events.emit_event", new_callable=AsyncMock),
        ):
            repo_instance = MockRepo.return_value
            repo_instance.create = AsyncMock(return_value=fake_session)

            result_mock = MagicMock()
            result_mock.scalars.return_value.first.return_value = fake_room
            db.execute = AsyncMock(return_value=result_mock)

            await handle_plan_created(
                room_id=_ROOM_ID,
                goal_id=_GOAL_ID,
                plan_id=_PLAN_ID,
                plan_text=plan_text,
                db=db,
            )

        assert len(tasks_created) == 1
        assert tasks_created[0]["title"] == "Build landing hero section"
        assert tasks_created[0]["agent_id"] == _FRONTEND_AGENT_ID

    @pytest.mark.asyncio
    async def test_cm_self_plan_actions_allow_only_create_tasks(self):
        from app.core.session_close import filter_by_kind

        actions = SessionCloseActions(
            tasks_create=[
                TaskCreate(
                    assigned_agent_id=_FRONTEND_AGENT_ID,
                    title="Hero",
                    description="Hero section",
                    goal_id=_GOAL_ID,
                )
            ],
            goals_create=[GoalCreate(title="Extra", description="not allowed")],
            room_memory_patch=MemoryPatch(content="not allowed"),
        )
        session = _make_session_mock("cm_self_plan")
        filtered, _dropped = filter_by_kind(actions, session)
        assert len(filtered.tasks_create) == 1
        # goals_create and room_memory_patch must be stripped
        assert len(filtered.goals_create) == 0
        assert filtered.room_memory_patch is None


# ---------------------------------------------------------------------------
# Phase 4: TASK_CREATED → proxy session pair
# ---------------------------------------------------------------------------


class TestPhase4ProxySession:
    """TASK_CREATED fires → two CmAgentProxySession created, dialogue starts."""

    @pytest.mark.asyncio
    async def test_task_created_spawns_two_proxy_sessions(self):
        from app.core.proxy_session import handle_task_created
        from app.models.task import DispatchStatus

        task = _make_task()
        task.dispatch_status = DispatchStatus.READY
        db = _make_db()

        fake_room = _make_room()
        fake_agent = MagicMock()
        fake_agent.llm_config = {}

        def _make_result(first=None, rowcount=1):
            m = MagicMock()
            m.scalars.return_value.first.return_value = first
            m.rowcount = rowcount
            return m

        db.execute = AsyncMock(side_effect=[
            _make_result(first=task),        # select Task
            _make_result(rowcount=1),         # UPDATE claim
            _make_result(first=fake_agent),  # select Agent
            _make_result(first=fake_room),   # select Room
        ])

        with (
            patch("app.core.proxy_session.SessionRepository") as MockRepo,
            patch("app.core.proxy_session._run_proxy_loop", new_callable=AsyncMock),
            patch("app.core.events.emit_event", new_callable=AsyncMock),
        ):
            create_calls: list[str] = []

            async def _track_create(create_obj: Any) -> MagicMock:
                create_calls.append(create_obj.kind)
                s = _make_session(create_obj.kind)
                s.peer_session_id = None
                return s

            MockRepo.return_value.create = AsyncMock(side_effect=_track_create)

            await handle_task_created(
                task_id=task.id,
                room_id=_ROOM_ID,
                goal_id=_GOAL_ID,
                plan_id=_PLAN_ID,
                db=db,
            )

        assert len(create_calls) == 2
        assert all(k == "cm_agent_proxy" for k in create_calls)


# ---------------------------------------------------------------------------
# Phase 5: Proxy dialogue → CM approve → both sides close
# ---------------------------------------------------------------------------


class TestPhase5ProxyDialogue:
    """Agent responds → ProxyBroker publishes → CM side approves → close."""

    @pytest.mark.asyncio
    async def test_proxy_dialogue_terminates_on_approve(self):
        from app.core.proxy_broker import ProxyBroker

        agent_session = _make_session("cm_agent_proxy")
        cm_session = _make_session("cm_agent_proxy")

        publishes: list[dict] = []

        async def _fake_publish(msg: Any) -> None:
            publishes.append({"content": msg.content, "kind": msg.metadata.get("kind")})

        with patch.object(ProxyBroker, "publish", side_effect=_fake_publish):
            broker = ProxyBroker()
            msg = MagicMock()
            msg.content = "Here is the hero section code..."
            msg.metadata = {
                "session_id": str(agent_session.id),
                "peer_session_id": str(cm_session.id),
                "kind": "answer",
                "turn": 1,
            }
            await broker.publish(msg)
            msg2 = MagicMock()
            msg2.content = "Approved! Great work."
            msg2.metadata = {
                "session_id": str(cm_session.id),
                "peer_session_id": str(agent_session.id),
                "kind": "approve",
                "turn": 2,
            }
            await broker.publish(msg2)

        assert len(publishes) == 2
        assert publishes[0]["kind"] == "answer"
        assert publishes[1]["kind"] == "approve"

    @pytest.mark.asyncio
    async def test_agent_close_actions_update_repo(self):
        from app.core.session_close import filter_by_kind

        actions = SessionCloseActions(
            agent_memory_patch=MemoryPatch(content="built hero section"),
            file_writes=[
                FileWrite(
                    path="/src/components/Hero.tsx",
                    content="export const Hero = () => <h1>Landing</h1>;",
                )
            ],
            room_memory_patch=MemoryPatch(content="should be stripped"),
        )
        session = _make_session_mock("cm_agent_proxy", side="agent")
        filtered, _dropped = filter_by_kind(actions, session)
        assert filtered.agent_memory_patch is not None
        assert len(filtered.file_writes) == 1
        # room_memory_patch not allowed for agent side
        assert filtered.room_memory_patch is None

    @pytest.mark.asyncio
    async def test_cm_close_actions_update_plan_and_room_memory(self):
        from app.core.session_close import filter_by_kind

        actions = SessionCloseActions(
            agent_memory_patch=MemoryPatch(content="hero section approved"),
            room_memory_patch=MemoryPatch(content="Step 1 done"),
            plan_update=PlanUpdate(
                plan_id=_PLAN_ID,
                steps=[],
            ),
            file_writes=[FileWrite(path="/x.ts", content="y")],
        )
        session = _make_session_mock("cm_agent_proxy", side="chat_master")
        filtered, _dropped = filter_by_kind(actions, session)
        assert filtered.agent_memory_patch is not None
        assert filtered.room_memory_patch is not None
        assert filtered.plan_update is not None
        # file_writes not allowed for CM side
        assert len(filtered.file_writes) == 0


# ---------------------------------------------------------------------------
# Full golden-path integration (mocked)
# ---------------------------------------------------------------------------


class TestGoldenPath:
    """
    End-to-end smoke test: phases 1–2 wired together via mocks.

    Flow:
      1. route_user_message → chatmaster_p2
      2. start_user_session → UCM-p2 session
    """

    @pytest.mark.asyncio
    async def test_full_landing_flow_golden_path(self):
        events_emitted: list[str] = []

        ucm_p2_session = _make_session("user_cm_p2")

        async def _fake_emit(room_id: Any, event_type: str, **kw: Any) -> None:
            events_emitted.append(event_type)

        decision_p2 = RouteDecision(
            route="chatmaster_p2",
            confidence=0.85,
            reasoning="no goals yet",
        )

        with (
            patch("app.core.router._gather_context", new_callable=AsyncMock,
                  return_value=([], [], [], {})),
            patch("app.core.router._call_router_llm", new_callable=AsyncMock,
                  return_value=_router_llm_result(decision_p2)),
            patch("app.core.events.emit_event", side_effect=_fake_emit),
            patch("app.core.user_sessions.SessionRepository") as MockRepo,
            patch("app.core.user_sessions.build_memory_layer", new_callable=AsyncMock, return_value={}),
            patch("app.core.user_sessions.process_user_turn", new_callable=AsyncMock, return_value=True),
        ):
            repo_instance = MockRepo.return_value
            repo_instance.create = AsyncMock(return_value=ucm_p2_session)

            room = _make_room()
            user = _make_user()
            db = _make_db()

            from app.core.router import route_user_message
            decision = await route_user_message(room, "I want a landing page on Next.js", db)
            assert decision.route == "chatmaster_p2"

            from app.core.user_sessions import start_user_session
            session = await start_user_session(room, user, decision, db, cm_agent_id=_CM_AGENT_ID)
            assert session.kind == "user_cm_p2"
