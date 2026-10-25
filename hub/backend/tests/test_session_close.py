"""
Unit + integration tests for session_close module (Step 2).

Unit tests (no DB):
  - TestKindFiltering          — filter_by_kind for all 7 matrix entries
  - TestToolUseExtraction      — extract_close_actions_from_tool_use
  - TestJsonEnvelopeExtraction — extract_close_actions_from_json_envelope incl. retry/repair

Integration tests (require PostgreSQL via DATABASE_URL env var):
  - TestAdvisoryLockDB         — advisory lock blocks concurrent transactions
  - TestApplyCloseActionsDB    — full apply round-trip (room_memory, tasks, plan, etc.)

Run DB tests:
    DATABASE_URL=postgresql+asyncpg://... pytest tests/test_session_close.py -m db
"""

from __future__ import annotations

import asyncio
import os
import types
import json
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pydantic import ValidationError

from app.core.session_close import (
    SESSION_CLOSE_ACTIONS_INVALID,
    ANTHROPIC_CLOSE_TOOL,
    OPENAI_CLOSE_TOOL,
    FileWrite,
    GoalCreate,
    MemoryPatch,
    PlanStepChange,
    PlanUpdate,
    SessionCloseActions,
    TaskCreate,
    TaskUpdate,
    _advisory_key,
    _KIND_ALLOWED,
    _matrix_key,
    _TOOL_INPUT_SCHEMA,
    close_tool_for_kind,
    ToolArgumentsParseError,
    apply_close_actions,
    parse_tool_arguments,
    extract_close_actions_from_json_envelope,
    extract_close_actions_from_tool_use,
    filter_by_kind,
    resource_lock,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_ROOM_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()
_TASK_ID = uuid.uuid4()
_GOAL_ID = uuid.uuid4()
_PLAN_ID = uuid.uuid4()
_STEP_ID = uuid.uuid4()


def _make_session(kind: str, *, side: str = "agent") -> object:
    """Build a minimal duck-typed Session object for unit tests.

    For cm_agent_proxy we use MagicMock(spec=...) so that
    isinstance(obj, CmAgentProxySession) returns True, which is what
    _matrix_key() relies on.  For all other kinds _matrix_key() just
    reads session.kind, so a plain SimpleNamespace is sufficient.
    """
    common = dict(
        id=uuid.uuid4(),
        room_id=_ROOM_ID,
        kind=kind,
        agent_id=_AGENT_ID,
        chat_master_agent_id=_AGENT_ID,
        goal_id=_GOAL_ID,
        plan_id=_PLAN_ID,
        context_snapshot=None,
        summary=None,
        pending_actions=None,
        side=side,
    )
    if kind == "cm_agent_proxy":
        from app.models.session import CmAgentProxySession

        obj = MagicMock(spec=CmAgentProxySession)
        for k, v in common.items():
            setattr(obj, k, v)
    else:
        obj = types.SimpleNamespace(**common)
    return obj


def _full_actions(**kwargs) -> SessionCloseActions:
    """Build a SessionCloseActions with every field populated."""
    return SessionCloseActions(
        room_memory_patch=MemoryPatch(content="room content"),
        agent_memory_patch=MemoryPatch(content="agent content"),
        goals_create=[GoalCreate(title="New goal")],
        plan_update=PlanUpdate(
            plan_id=_PLAN_ID,
            status="approved",
            steps=[PlanStepChange(step_id=_STEP_ID, status="completed")],
        ),
        tasks_create=[TaskCreate(title="New task")],
        tasks_update=[TaskUpdate(task_id=_TASK_ID, state="in_progress")],
        tasks_cancel=[_TASK_ID],
        file_writes=[FileWrite(path="README.md", content="# Hello")],
        summary="All done",
        **kwargs,
    )


class TestTaskUpdateStateNormalization:
    """LLMs send enum reprs / upper-case names instead of enum values."""

    def test_enum_repr_normalized(self):
        upd = TaskUpdate(task_id=_TASK_ID, state="TaskState.COMPLETED")
        assert upd.state == "completed"

    def test_uppercase_normalized(self):
        upd = TaskUpdate(task_id=_TASK_ID, state="IN_PROGRESS")
        assert upd.state == "in_progress"

    def test_plain_value_untouched(self):
        upd = TaskUpdate(task_id=_TASK_ID, state="completed")
        assert upd.state == "completed"

    def test_none_untouched(self):
        upd = TaskUpdate(task_id=_TASK_ID)
        assert upd.state is None


class TestTaskCreateCheckpoint:
    """Generation checkpoint plan, stage 2 — checkpoint fields on TaskCreate."""

    def test_checkpoint_fields_accepted(self):
        spec = TaskCreate(
            title="Approve design variant",
            checkpoint_kind="select_variant",
            checkpoint_payload={
                "prompt": "Pick a variant",
                "variants": [
                    {"id": "v1", "label": "Variant A", "paths": ["/project/images/a.png"]},
                    {"id": "v2", "label": "Variant B", "paths": ["/project/images/b.png"]},
                ],
            },
        )
        assert spec.checkpoint_kind == "select_variant"
        assert spec.assigned_agent_id is None
        assert len(spec.checkpoint_payload["variants"]) == 2

    def test_checkpoint_kind_invalid_value_rejected(self):
        with pytest.raises(ValidationError):
            TaskCreate(title="x", checkpoint_kind="bogus")

    def test_checkpoint_drops_stray_assigned_agent_id(self):
        """A checkpoint task is a human-decision gate — a stray agent id must
        be silently dropped rather than produce an agent-bound task that the
        dispatcher would skip via _NON_DISPATCHABLE_STATES."""
        agent_id = uuid.uuid4()
        spec = TaskCreate(
            title="Confirm before deploy",
            assigned_agent_id=agent_id,
            checkpoint_kind="confirm",
            checkpoint_payload={"prompt": "Ready to deploy?"},
        )
        assert spec.checkpoint_kind == "confirm"
        assert spec.assigned_agent_id is None

    def test_non_checkpoint_task_keeps_assigned_agent_id(self):
        agent_id = uuid.uuid4()
        spec = TaskCreate(title="Build API", assigned_agent_id=agent_id)
        assert spec.checkpoint_kind is None
        assert spec.assigned_agent_id == agent_id


# ---------------------------------------------------------------------------
# 1. Kind filtering
# ---------------------------------------------------------------------------


class TestKindFiltering:
    """filter_by_kind must keep only fields allowed for each session kind."""

    def _kept_fields(self, kind: str, side: str = "agent") -> set[str]:
        session = _make_session(kind, side=side)
        filtered, _ = filter_by_kind(_full_actions(), session)
        d = filtered.model_dump()
        non_empty = set()
        for k, v in d.items():
            if v is not None and v != []:
                non_empty.add(k)
        return non_empty

    def test_user_agent_p1_only_agent_memory_and_summary(self):
        kept = self._kept_fields("user_agent_p1")
        assert kept == {"agent_memory_patch", "summary"}

    def test_user_agent_p2_only_agent_memory_and_summary(self):
        kept = self._kept_fields("user_agent_p2")
        assert kept == {"agent_memory_patch", "summary"}

    def test_user_cm_p1_allowed_fields(self):
        kept = self._kept_fields("user_cm_p1")
        assert kept == {
            "agent_memory_patch",
            "room_memory_patch",
            "plan_update",
            "tasks_create",
            "tasks_cancel",
            "summary",
        }
        # goals_create and tasks_update must be dropped
        assert "goals_create" not in kept
        assert "tasks_update" not in kept

    def test_user_cm_p2_allowed_fields(self):
        kept = self._kept_fields("user_cm_p2")
        assert kept == {
            "agent_memory_patch",
            "room_memory_patch",
            "goals_create",
            "plan_update",
            "summary",
        }
        assert "tasks_create" not in kept

    def test_cm_self_plan_only_tasks_create(self):
        kept = self._kept_fields("cm_self_plan")
        assert kept == {"tasks_create", "summary"}

    def test_cm_agent_proxy_agent_side(self):
        kept = self._kept_fields("cm_agent_proxy", side="agent")
        assert kept == {"agent_memory_patch", "tasks_update", "file_writes", "summary"}
        assert "room_memory_patch" not in kept
        assert "plan_update" not in kept

    def test_cm_agent_proxy_chat_master_side(self):
        kept = self._kept_fields("cm_agent_proxy", side="chat_master")
        assert kept == {"agent_memory_patch", "room_memory_patch", "plan_update", "summary"}
        assert "file_writes" not in kept
        assert "tasks_update" not in kept

    def test_no_fields_dropped_when_all_empty(self):
        """Empty actions should pass through filter unchanged (nothing to drop)."""
        empty = SessionCloseActions()
        session = _make_session("user_agent_p1")
        result, dropped = filter_by_kind(empty, session)
        assert result == empty
        assert dropped == []

    def test_filter_logs_dropped_fields(self, caplog):
        import logging

        session = _make_session("user_agent_p1")
        actions = SessionCloseActions(room_memory_patch=MemoryPatch(content="x"))
        with caplog.at_level(logging.WARNING, logger="app.core.session_close"):
            _, dropped = filter_by_kind(actions, session)
        assert "room_memory_patch" in caplog.text
        assert "room_memory_patch" in dropped

    def test_filter_by_kind_returns_dropped_field_names(self):
        """filter_by_kind includes disallowed non-empty fields in the returned list."""
        session = _make_session("user_cm_p1")
        actions = SessionCloseActions(
            file_writes=[FileWrite(path="x.txt", content="hi")],
            tasks_create=[TaskCreate(title="allowed task")],
        )
        filtered, dropped = filter_by_kind(actions, session)
        assert "file_writes" in dropped
        assert "tasks_create" not in dropped
        assert filtered.file_writes == []
        assert len(filtered.tasks_create) == 1

    def test_filter_by_kind_empty_dropped_when_all_allowed(self):
        """No dropped list items when all provided fields are permitted."""
        session = _make_session("cm_agent_proxy", side="agent")
        actions = SessionCloseActions(
            file_writes=[FileWrite(path="x.txt", content="hi")],
            summary="done",
        )
        _, dropped = filter_by_kind(actions, session)
        assert dropped == []

    def test_matrix_key_splits_proxy_by_side(self):
        agent_session = _make_session("cm_agent_proxy", side="agent")
        cm_session = _make_session("cm_agent_proxy", side="chat_master")
        assert _matrix_key(agent_session) == "cm_agent_proxy_agent"
        assert _matrix_key(cm_session) == "cm_agent_proxy_chat_master"


# ---------------------------------------------------------------------------
# 2. Tool-use adapter (Anthropic / OpenAI)
# ---------------------------------------------------------------------------


class TestToolUseExtraction:
    def test_parses_full_payload(self):
        payload = {
            "agent_memory_patch": {"content": "updated memory"},
            "tasks_create": [{"title": "Fix bug", "description": "desc"}],
            "summary": "Session done",
        }
        actions = extract_close_actions_from_tool_use(payload)
        assert actions.agent_memory_patch is not None
        assert actions.agent_memory_patch.content == "updated memory"
        assert len(actions.tasks_create) == 1
        assert actions.tasks_create[0].title == "Fix bug"
        assert actions.summary == "Session done"

    def test_parses_empty_payload(self):
        actions = extract_close_actions_from_tool_use({})
        assert actions.agent_memory_patch is None
        assert actions.tasks_create == []

    def test_raises_on_invalid_payload(self):
        with pytest.raises(ValidationError):
            extract_close_actions_from_tool_use(
                {"tasks_create": [{"not_a_valid_field": True}]}
            )

    def test_parses_plan_update_with_steps(self):
        plan_id = str(uuid.uuid4())
        step_id = str(uuid.uuid4())
        payload = {
            "plan_update": {
                "plan_id": plan_id,
                "status": "approved",
                "steps": [{"step_id": step_id, "status": "completed"}],
            }
        }
        actions = extract_close_actions_from_tool_use(payload)
        assert actions.plan_update is not None
        assert str(actions.plan_update.plan_id) == plan_id
        assert actions.plan_update.steps[0].status == "completed"

    def test_null_file_write_encoding_defaults_to_utf8(self):
        # LLMs send explicit null for optional fields (seen in prod: session ba36a2fa)
        actions = extract_close_actions_from_tool_use(
            {"file_writes": [{"path": "index.html", "content": "<html>", "encoding": None}]}
        )
        assert actions.file_writes[0].encoding == "utf-8"

    def test_invalid_field_is_pruned_without_losing_valid_ones(self):
        # One malformed field must not discard goals_create (prod incident:
        # bad file_writes entry silently dropped the goal → no plan, no tasks)
        payload = {
            "goals_create": [{"title": "Shoe catalog", "description": "index.html"}],
            "file_writes": [{"path": "index.html"}],  # missing required 'content'
            "summary": "done",
        }
        actions = extract_close_actions_from_tool_use(payload)
        assert len(actions.goals_create) == 1
        assert actions.goals_create[0].title == "Shoe catalog"
        assert actions.file_writes == []
        assert actions.summary == "done"

    def test_invalid_aliased_field_is_pruned(self):
        payload = {
            "goals": [{"title": "G"}],
            "files": [{"path": "a.txt"}],  # alias for file_writes, missing 'content'
        }
        actions = extract_close_actions_from_tool_use(payload)
        assert len(actions.goals_create) == 1
        assert actions.file_writes == []

    def test_plan_steps_with_fake_ids_are_dropped(self):
        plan_id = str(uuid.uuid4())
        step_id = str(uuid.uuid4())
        payload = {
            "plan_update": {
                "plan_id": plan_id,
                "steps": [
                    {"step_id": "step-1", "status": "completed"},
                    {"step_id": step_id, "status": "completed"},
                ],
            }
        }
        actions = extract_close_actions_from_tool_use(payload)
        assert actions.plan_update is not None
        assert len(actions.plan_update.steps) == 1
        assert str(actions.plan_update.steps[0].step_id) == step_id

    def test_anthropic_tool_schema_structure(self):
        assert ANTHROPIC_CLOSE_TOOL["name"] == "close_session"
        assert "input_schema" in ANTHROPIC_CLOSE_TOOL
        assert ANTHROPIC_CLOSE_TOOL["input_schema"]["type"] == "object"

    def test_openai_tool_schema_structure(self):
        assert OPENAI_CLOSE_TOOL["type"] == "function"
        assert OPENAI_CLOSE_TOOL["function"]["name"] == "close_session"
        assert "parameters" in OPENAI_CLOSE_TOOL["function"]


# ---------------------------------------------------------------------------
# 3. JSON-envelope adapter (deepseek / groq / ollama)
# ---------------------------------------------------------------------------


class TestJsonEnvelopeExtraction:
    @pytest.mark.asyncio
    async def test_parses_valid_json(self):
        raw = '{"summary": "done", "tasks_create": [{"title": "T1"}]}'
        result = await extract_close_actions_from_json_envelope(raw)
        assert isinstance(result, SessionCloseActions)
        assert result.summary == "done"
        assert result.tasks_create[0].title == "T1"

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        raw = "```json\n{\"summary\": \"ok\"}\n```"
        result = await extract_close_actions_from_json_envelope(raw)
        assert isinstance(result, SessionCloseActions)
        assert result.summary == "ok"

    @pytest.mark.asyncio
    async def test_strips_plain_fences(self):
        raw = "```\n{\"summary\": \"plain\"}\n```"
        result = await extract_close_actions_from_json_envelope(raw)
        assert isinstance(result, SessionCloseActions)
        assert result.summary == "plain"

    @pytest.mark.asyncio
    async def test_returns_sentinel_on_invalid_no_repair_fn(self):
        result = await extract_close_actions_from_json_envelope("not json at all")
        assert result == SESSION_CLOSE_ACTIONS_INVALID

    @pytest.mark.asyncio
    async def test_repair_fn_called_on_first_failure(self):
        good_json = '{"summary": "repaired"}'
        repair = AsyncMock(return_value=good_json)

        result = await extract_close_actions_from_json_envelope("bad json", repair_fn=repair)
        assert isinstance(result, SessionCloseActions)
        assert result.summary == "repaired"
        repair.assert_called_once()

    @pytest.mark.asyncio
    async def test_repair_fn_called_twice_on_two_failures(self):
        repair = AsyncMock(side_effect=["still bad", "still bad"])

        result = await extract_close_actions_from_json_envelope("bad json", repair_fn=repair)
        assert result == SESSION_CLOSE_ACTIONS_INVALID
        assert repair.call_count == 2

    @pytest.mark.asyncio
    async def test_repair_fn_succeeds_on_second_attempt(self):
        good_json = '{"summary": "second repair"}'
        repair = AsyncMock(side_effect=["still bad json", good_json])

        result = await extract_close_actions_from_json_envelope("bad json", repair_fn=repair)
        assert isinstance(result, SessionCloseActions)
        assert result.summary == "second repair"
        assert repair.call_count == 2

    @pytest.mark.asyncio
    async def test_repair_fn_exception_returns_sentinel(self):
        repair = AsyncMock(side_effect=RuntimeError("LLM down"))

        result = await extract_close_actions_from_json_envelope("bad json", repair_fn=repair)
        assert result == SESSION_CLOSE_ACTIONS_INVALID

    @pytest.mark.asyncio
    async def test_empty_json_object_is_valid(self):
        result = await extract_close_actions_from_json_envelope("{}")
        assert isinstance(result, SessionCloseActions)
        assert result.tasks_create == []


# ---------------------------------------------------------------------------
# 4. Advisory lock key derivation (pure unit)
# ---------------------------------------------------------------------------


class TestAdvisoryKey:
    def test_deterministic(self):
        uid = uuid.uuid4()
        assert _advisory_key(uid) == _advisory_key(uid)

    def test_different_uuids_produce_different_keys(self):
        a, b = uuid.uuid4(), uuid.uuid4()
        assert _advisory_key(a) != _advisory_key(b)

    def test_key_is_signed_int64(self):
        for _ in range(20):
            key = _advisory_key(uuid.uuid4())
            assert -(1 << 63) <= key < (1 << 63)


class TestParseToolArguments:
    def test_valid_json(self):
        assert parse_tool_arguments('{"summary": "ok"}') == {"summary": "ok"}

    def test_empty_defaults_to_empty_object(self):
        assert parse_tool_arguments("") == {}
        assert parse_tool_arguments(None) == {}

    def test_markdown_fences_tolerated(self):
        raw = '```json\n{"summary": "ok"}\n```'
        assert parse_tool_arguments(raw) == {"summary": "ok"}

    def test_malformed_raises_specific_error(self):
        with pytest.raises(ToolArgumentsParseError):
            parse_tool_arguments('{"tasks_create": [{"title": "a" "desc": "b"}]}')


# ---------------------------------------------------------------------------
# 5. close_tool_for_kind — per-kind schema generation
# ---------------------------------------------------------------------------


class TestCloseToolForKind:
    """close_tool_for_kind must expose only allowed fields for each session kind."""

    def _props_anthropic(self, matrix_key: str) -> set[str]:
        tool = close_tool_for_kind(matrix_key, "anthropic")
        return set(tool["input_schema"]["properties"].keys())

    def _props_openai(self, matrix_key: str) -> set[str]:
        tool = close_tool_for_kind(matrix_key, "openai")
        return set(tool["function"]["parameters"]["properties"].keys())

    # user_cm_p1 must NOT expose file_writes; must expose tasks_create
    def test_user_cm_p1_no_file_writes_anthropic(self):
        props = self._props_anthropic("user_cm_p1")
        assert "file_writes" not in props
        assert "tasks_create" in props

    def test_user_cm_p1_no_file_writes_openai(self):
        props = self._props_openai("user_cm_p1")
        assert "file_writes" not in props
        assert "tasks_create" in props

    # cm_agent_proxy_agent must expose file_writes
    def test_cm_agent_proxy_agent_has_file_writes_anthropic(self):
        props = self._props_anthropic("cm_agent_proxy_agent")
        assert "file_writes" in props

    def test_cm_agent_proxy_agent_has_file_writes_openai(self):
        props = self._props_openai("cm_agent_proxy_agent")
        assert "file_writes" in props

    # user_cm_p2 must NOT expose file_writes; must expose goals_create
    def test_user_cm_p2_no_file_writes(self):
        props = self._props_anthropic("user_cm_p2")
        assert "file_writes" not in props
        assert "goals_create" in props

    # Full schemas are unchanged (backward compat for proxy sessions and tests)
    def test_anthropic_full_schema_unchanged(self):
        full_props = set(_TOOL_INPUT_SCHEMA["properties"].keys())
        assert set(ANTHROPIC_CLOSE_TOOL["input_schema"]["properties"].keys()) == full_props

    def test_openai_full_schema_unchanged(self):
        full_props = set(_TOOL_INPUT_SCHEMA["properties"].keys())
        assert (
            set(OPENAI_CLOSE_TOOL["function"]["parameters"]["properties"].keys()) == full_props
        )

    # Returned tool dict has correct structure
    def test_anthropic_format_structure(self):
        tool = close_tool_for_kind("user_cm_p1", "anthropic")
        assert tool["name"] == "close_session"
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"

    def test_openai_format_structure(self):
        tool = close_tool_for_kind("user_cm_p1", "openai")
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "close_session"
        assert "parameters" in tool["function"]

    # Unknown matrix_key falls back to full schema
    def test_unknown_key_falls_back_to_full_anthropic(self):
        tool = close_tool_for_kind("nonexistent_kind", "anthropic")
        assert tool == ANTHROPIC_CLOSE_TOOL

    def test_unknown_key_falls_back_to_full_openai(self):
        tool = close_tool_for_kind("nonexistent_kind", "openai")
        assert tool == OPENAI_CLOSE_TOOL

    # Exact allowed fields match the _KIND_ALLOWED matrix
    def test_props_match_kind_allowed_matrix(self):
        for mk, allowed in _KIND_ALLOWED.items():
            props = self._props_anthropic(mk)
            assert props == allowed, f"Mismatch for {mk}: got {props}, expected {allowed}"

    def test_agent_subtask_only_allows_memory_and_summary(self):
        """Stage 5.4: a run_subtask child's own row never receives file_writes/tasks_*."""
        assert _KIND_ALLOWED["agent_subtask"] == frozenset({"agent_memory_patch", "summary"})


# ---------------------------------------------------------------------------
# DB integration tests
# ---------------------------------------------------------------------------

DB_URL = os.environ.get("DATABASE_URL", "")
pytestmark_db = pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator:
    """Async SQLAlchemy session connected to the test database."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import SQLModel

    # Ensure all model metadata is registered
    import app.models.session  # noqa: F401
    import app.models.goal  # noqa: F401
    import app.models.goal_plan  # noqa: F401
    import app.models.task  # noqa: F401
    import app.models.project_memory  # noqa: F401
    import app.models.agent_memory  # noqa: F401
    import app.models.message  # noqa: F401

    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytest_asyncio.fixture
async def db_engine():
    """Raw async engine for spawning independent sessions in concurrency tests."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import SQLModel

    import app.models.session  # noqa: F401
    import app.models.goal  # noqa: F401
    import app.models.goal_plan  # noqa: F401
    import app.models.task  # noqa: F401
    import app.models.project_memory  # noqa: F401
    import app.models.agent_memory  # noqa: F401
    import app.models.message  # noqa: F401

    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    await engine.dispose()


async def _insert_fixtures(db, room_id, user_id, agent_id):
    """Insert minimal room/user/agent rows needed for FK constraints."""
    from sqlalchemy import text

    await db.execute(
        text(
            "INSERT INTO \"user\" (id, email, hashed_password, is_active, two_factor_enabled) "
            "VALUES (:id, :email, 'x', true, false) ON CONFLICT DO NOTHING"
        ),
        {"id": str(user_id), "email": f"{user_id}@test.com"},
    )
    await db.execute(
        text(
            "INSERT INTO room (id, name, type, room_type, has_coordinator, creator_id, state, coordinator_config) "
            "VALUES (:id, 'test', 'group', 'group', false, :cid, 'PLANNING', CAST(:coord AS JSON)) ON CONFLICT DO NOTHING"
        ),
        # The agent is also the room's ChatMaster: a task is dispatched only in a room that has one.
        {"id": str(room_id), "cid": str(user_id), "coord": json.dumps({"agent_id": str(agent_id)})},
    )
    await db.execute(
        text(
            "INSERT INTO agent (id, nickname, agent_uid, agent_type, status, version, "
            "update_policy, short_description, owner_id, token) "
            "VALUES (:id, 'a', :uid, 'internal', 'offline', '1', 'manual', '', :oid, :tok) "
            "ON CONFLICT DO NOTHING"
        ),
        {
            "id": str(agent_id),
            "uid": f"test-{agent_id}",
            "oid": str(user_id),
            "tok": f"tok-{agent_id}",
        },
    )
    # A task is dispatched only to an agent that is a member of the room.
    await db.execute(
        text("INSERT INTO roomagentlink (room_id, agent_id) VALUES (:rid, :aid) ON CONFLICT DO NOTHING"),
        {"rid": str(room_id), "aid": str(agent_id)},
    )
    await db.execute(
        text(
            "INSERT INTO projectmemory (id, room_id, content, token_count, updated_at) "
            "VALUES (:id, :rid, 'old content', 0, now()) ON CONFLICT DO NOTHING"
        ),
        {"id": str(uuid.uuid4()), "rid": str(room_id)},
    )
    await db.flush()


@pytest.mark.db
@pytestmark_db
@pytest.mark.asyncio
class TestAdvisoryLockDB:
    """Advisory lock must block a second concurrent transaction."""

    async def test_two_sessions_same_resource_serialized(self, db_engine):
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        factory = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
        resource_id = uuid.uuid4()
        results: list[int] = []

        async def hold_lock(order: int, hold_seconds: float):
            async with factory() as db:
                async with db.begin():
                    await resource_lock(db, resource_id)
                    results.append(order)
                    await asyncio.sleep(hold_seconds)

        # First task holds the lock for 0.3 s; second should wait and then acquire
        t1 = asyncio.create_task(hold_lock(1, 0.3))
        await asyncio.sleep(0.05)  # ensure t1 acquires first
        t2 = asyncio.create_task(hold_lock(2, 0.0))

        await asyncio.gather(t1, t2)

        # t1 must have acquired before t2
        assert results == [1, 2]


@pytest.mark.db
@pytestmark_db
@pytest.mark.asyncio
class TestApplyCloseActionsDB:
    """End-to-end apply_close_actions against a real PostgreSQL database."""

    async def _setup(self, db):
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        step_id = uuid.uuid4()
        task_id = uuid.uuid4()

        await _insert_fixtures(db, room_id, user_id, agent_id)

        from sqlalchemy import text

        await db.execute(
            text(
                "INSERT INTO room_goal (id, room_id, title, description, status, position, "
                "created_at, updated_at) VALUES (:id, :rid, 'g', '', 'ACTIVE', 0, now(), now())"
            ),
            {"id": str(goal_id), "rid": str(room_id)},
        )
        await db.execute(
            text(
                "INSERT INTO goal_plan (id, goal_id, status, created_at, updated_at) "
                "VALUES (:id, :gid, 'DRAFT', now(), now())"
            ),
            {"id": str(plan_id), "gid": str(goal_id)},
        )
        await db.execute(
            text(
                "INSERT INTO goal_plan_step (id, plan_id, title, description, status, position, "
                "is_parallel, created_at, updated_at) "
                "VALUES (:id, :pid, 'Step 1', '', 'PENDING', 0, false, now(), now())"
            ),
            {"id": str(step_id), "pid": str(plan_id)},
        )
        await db.execute(
            text(
                "INSERT INTO task (id, room_id, title, description, state, goal_id, "
                "created_at, updated_at) "
                "VALUES (:id, :rid, 'Old task', '', 'DRAFT', :gid, now(), now())"
            ),
            {"id": str(task_id), "rid": str(room_id), "gid": str(goal_id)},
        )
        await db.flush()

        return dict(
            room_id=room_id,
            user_id=user_id,
            agent_id=agent_id,
            goal_id=goal_id,
            plan_id=plan_id,
            step_id=step_id,
            task_id=task_id,
        )

    async def _make_db_session(
        self, kind: str, *, fixtures: dict, db, side: str = "agent"
    ) -> object:
        """Create and persist a real ORM session row for the given kind.

        apply_close_actions calls db.add(session), so the object must be a
        mapped instance, not a duck-typed stand-in.
        """
        from app.models.session import (
            CmAgentProxySession,
            CmSelfPlanSession,
            UserCmSession,
        )

        common = dict(id=uuid.uuid4(), room_id=fixtures["room_id"], kind=kind)
        if kind in ("user_cm_p1", "user_cm_p2"):
            obj = UserCmSession(
                **common,
                user_id=fixtures["user_id"],
                chat_master_agent_id=fixtures["agent_id"],
                goal_id=fixtures["goal_id"],
            )
        elif kind == "cm_self_plan":
            obj = CmSelfPlanSession(
                **common,
                chat_master_agent_id=fixtures["agent_id"],
                goal_id=fixtures["goal_id"],
                plan_id=fixtures["plan_id"],
            )
        elif kind == "cm_agent_proxy":
            obj = CmAgentProxySession(
                **common,
                side=side,
                task_id=fixtures["task_id"],
                goal_id=fixtures["goal_id"],
                plan_id=fixtures["plan_id"],
                agent_id=fixtures["agent_id"],
            )
        else:
            raise ValueError(f"unsupported session kind for tests: {kind}")
        db.add(obj)
        await db.flush()
        return obj

    async def test_room_memory_patch(self, db_session):
        from sqlalchemy import select
        from app.models.project_memory import ProjectMemory

        fixtures = await self._setup(db_session)
        session = await self._make_db_session("user_cm_p1", fixtures=fixtures, db=db_session)

        actions = SessionCloseActions(
            room_memory_patch=MemoryPatch(content="new room memory"),
            summary="done",
        )
        await apply_close_actions(session, actions, db_session)

        pm = (
            await db_session.execute(
                select(ProjectMemory).where(ProjectMemory.room_id == fixtures["room_id"])
            )
        ).scalars().first()
        assert pm is not None
        assert pm.content == "new room memory"

    async def test_tasks_create(self, db_session):
        from sqlalchemy import select
        from app.models.task import Task

        fixtures = await self._setup(db_session)
        session = await self._make_db_session("cm_self_plan", fixtures=fixtures, db=db_session)

        actions = SessionCloseActions(
            tasks_create=[
                TaskCreate(title="Task A"),
                TaskCreate(title="Task B", description="desc"),
            ],
            summary="plan decomposed",
        )
        await apply_close_actions(session, actions, db_session)

        tasks = (
            await db_session.execute(
                select(Task).where(
                    Task.room_id == fixtures["room_id"],
                    Task.title.in_(["Task A", "Task B"]),
                )
            )
        ).scalars().all()
        assert len(tasks) == 2

    async def test_plan_update_step_status(self, db_session):
        from sqlalchemy import select
        from app.models.goal_plan import GoalPlanStep, PlanStepStatus

        fixtures = await self._setup(db_session)
        session = await self._make_db_session("user_cm_p1", fixtures=fixtures, db=db_session)

        actions = SessionCloseActions(
            plan_update=PlanUpdate(
                plan_id=fixtures["plan_id"],
                steps=[PlanStepChange(step_id=fixtures["step_id"], status="completed")],
            )
        )
        await apply_close_actions(session, actions, db_session)

        step = (
            await db_session.execute(
                select(GoalPlanStep).where(GoalPlanStep.id == fixtures["step_id"])
            )
        ).scalars().first()
        assert step is not None
        assert step.status == PlanStepStatus.COMPLETED

    async def test_tasks_cancel(self, db_session):
        from sqlalchemy import select
        from app.models.task import Task, TaskState

        fixtures = await self._setup(db_session)
        session = await self._make_db_session("user_cm_p1", fixtures=fixtures, db=db_session)

        actions = SessionCloseActions(tasks_cancel=[fixtures["task_id"]])
        await apply_close_actions(session, actions, db_session)

        task = (
            await db_session.execute(select(Task).where(Task.id == fixtures["task_id"]))
        ).scalars().first()
        assert task is not None
        assert task.state == TaskState.CANCELLED

    async def test_tasks_create_invalid_goal_id_falls_back_to_session_goal(self, db_session):
        """LLM-hallucinated goal_id (e.g. the nil UUID) must not break task creation."""
        from sqlalchemy import select
        from app.models.task import Task

        fixtures = await self._setup(db_session)
        session = await self._make_db_session("cm_self_plan", fixtures=fixtures, db=db_session)

        actions = SessionCloseActions(
            tasks_create=[
                TaskCreate(title="Nil goal task", goal_id=uuid.UUID(int=0)),
                TaskCreate(title="Unknown goal task", goal_id=uuid.uuid4()),
            ],
            summary="decomposed",
        )
        await apply_close_actions(session, actions, db_session)

        tasks = (
            await db_session.execute(
                select(Task).where(
                    Task.room_id == fixtures["room_id"],
                    Task.title.in_(["Nil goal task", "Unknown goal task"]),
                )
            )
        ).scalars().all()
        assert len(tasks) == 2
        assert all(t.goal_id == fixtures["goal_id"] for t in tasks)

    async def test_disallowed_fields_stripped_before_apply(self, db_session):
        """cm_self_plan must not create goals even if actions.goals_create is set."""
        from sqlalchemy import select
        from app.models.goal import RoomGoal

        fixtures = await self._setup(db_session)
        session = await self._make_db_session("cm_self_plan", fixtures=fixtures, db=db_session)

        actions = SessionCloseActions(
            goals_create=[GoalCreate(title="Should be stripped")],
            tasks_create=[TaskCreate(title="Allowed task")],
        )
        await apply_close_actions(session, actions, db_session)

        goals = (
            await db_session.execute(
                select(RoomGoal).where(
                    RoomGoal.room_id == fixtures["room_id"],
                    RoomGoal.title == "Should be stripped",
                )
            )
        ).scalars().all()
        assert goals == []

    async def test_file_writes_written_to_volume(self, db_session):
        # internal design notes stage 4/6: agent file_writes
        # go to the workspace volume post-commit with an agent-attributed commit.
        # (Detailed coverage: test_workspace_single_storage_stage4.py.)
        fixtures = await self._setup(db_session)
        proxy_session = await self._make_db_session(
            "cm_agent_proxy", fixtures=fixtures, db=db_session
        )

        actions = SessionCloseActions(
            file_writes=[FileWrite(path="src/main.py", content="print('hi')")],
        )

        with patch(
            "app.core.workspaced_client.write_workspace_file", new_callable=AsyncMock
        ) as write:
            await apply_close_actions(proxy_session, actions, db_session)

        write.assert_awaited_once()
        args, kwargs = write.call_args
        assert args[0] == str(proxy_session.room_id)
        assert args[1].lstrip("/") == "src/main.py"
        assert args[2] == "print('hi')"
        assert kwargs["commit_message"].endswith("write /src/main.py")

    async def test_idempotent_apply_same_actions_twice(self, db_session):
        """Applying the same memory patch twice should end with the second content."""
        from sqlalchemy import select
        from app.models.project_memory import ProjectMemory

        fixtures = await self._setup(db_session)
        session = await self._make_db_session("user_cm_p1", fixtures=fixtures, db=db_session)

        for content in ["first write", "second write"]:
            actions = SessionCloseActions(
                room_memory_patch=MemoryPatch(content=content)
            )
            await apply_close_actions(session, actions, db_session)

        pm = (
            await db_session.execute(
                select(ProjectMemory).where(ProjectMemory.room_id == fixtures["room_id"])
            )
        ).scalars().first()
        assert pm.content == "second write"


# ---------------------------------------------------------------------------
# Stage 4 unit tests — TASK_CREATED events and dispatch (mocked _do_apply)
# ---------------------------------------------------------------------------


class TestApplyCloseActionsDispatch:
    """Stage 4: apply_close_actions must emit TASK_CREATED and dispatch proxy sessions."""

    def _make_outcome(self, *, tasks=None, reopened_goals=None):
        return {
            "file_changes": [],
            "touched_goal_ids": set(),
            "created_tasks": tasks or [],
            "reopened_goals": reopened_goals or {},
        }

    @staticmethod
    def _dispatch_task(task_id, room_id, title, agent_id, goal_id=None):
        from app.models.task import DispatchStatus

        return types.SimpleNamespace(
            id=task_id,
            room_id=room_id,
            title=title,
            assigned_agent_id=agent_id,
            goal_id=goal_id,
            dispatch_status=DispatchStatus.READY,
        )

    @staticmethod
    def _dispatch_context(goal_id=None, plan_id=None):
        from app.core.proxy_session import TaskDispatchContext

        return TaskDispatchContext(goal_id, plan_id, MagicMock(), uuid.uuid4(), False)

    @pytest.mark.asyncio
    async def test_task_created_event_emitted(self):
        """TASK_CREATED must be emitted for every created task."""
        task_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        outcome = self._make_outcome(tasks=[{
            "task_id": task_id,
            "title": "Write tests",
            "goal_id": goal_id,
            "assigned_agent_id": None,
            "plan_id": None,
        }])
        session = _make_session("user_cm_p1")
        emitted: list[str] = []

        async def fake_emit(room_id, event_type, content, db, **kw):
            emitted.append(event_type)

        actions = SessionCloseActions()
        with (
            patch("app.core.session_close._do_apply", new_callable=AsyncMock, return_value=outcome),
            patch("app.core.session_close.filter_by_kind", return_value=(actions, [])),
            patch("app.core.events.emit_event", side_effect=fake_emit),
        ):
            db = AsyncMock()
            await apply_close_actions(session, actions, db)

        assert "TASK_CREATED" in emitted

    @pytest.mark.asyncio
    async def test_dispatch_called_for_task_with_agent_and_plan(self):
        """dispatch_task_event must be called when task has assigned_agent_id and plan_id."""
        task_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        outcome = self._make_outcome(tasks=[{
            "task_id": task_id,
            "title": "Do work",
            "goal_id": goal_id,
            "assigned_agent_id": agent_id,
            "plan_id": plan_id,
        }])
        session = _make_session("user_cm_p1")
        actions = SessionCloseActions()

        with (
            patch("app.core.session_close._do_apply", new_callable=AsyncMock, return_value=outcome),
            patch("app.core.session_close.filter_by_kind", return_value=(actions, [])),
            patch("app.core.events.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock,
                  return_value="started") as mock_dispatch,
        ):
            db = AsyncMock()
            dispatch_task = self._dispatch_task(
                task_id, session.room_id, "Do work", agent_id, goal_id
            )
            db.get = AsyncMock(return_value=dispatch_task)
            await apply_close_actions(session, actions, db)

        mock_dispatch.assert_awaited_once_with(
            dispatch_task, session.room_id, db, source="session_close_created_task",
        )

    @pytest.mark.asyncio
    async def test_dispatch_not_called_without_agent(self):
        """Tasks without assigned_agent_id must NOT trigger dispatch."""
        outcome = self._make_outcome(tasks=[{
            "task_id": uuid.uuid4(),
            "title": "Unassigned task",
            "goal_id": uuid.uuid4(),
            "assigned_agent_id": None,
            "plan_id": None,
        }])
        session = _make_session("user_cm_p1")
        actions = SessionCloseActions()

        with (
            patch("app.core.session_close._do_apply", new_callable=AsyncMock, return_value=outcome),
            patch("app.core.session_close.filter_by_kind", return_value=(actions, [])),
            patch("app.core.events.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch,
        ):
            db = AsyncMock()
            await apply_close_actions(session, actions, db)

        mock_dispatch.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_direct_task_without_goal_or_plan(self):
        """A single-executor task does not need a goal/plan decomposition."""
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        outcome = self._make_outcome(tasks=[{
            "task_id": task_id,
            "title": "Direct task",
            "goal_id": None,
            "assigned_agent_id": agent_id,
            "plan_id": None,
        }])
        session = _make_session("user_cm_p1")
        actions = SessionCloseActions()

        with (
            patch("app.core.session_close._do_apply", new_callable=AsyncMock, return_value=outcome),
            patch("app.core.session_close.filter_by_kind", return_value=(actions, [])),
            patch("app.core.events.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.request_task_dispatch", new_callable=AsyncMock,
                  return_value="started") as mock_dispatch,
        ):
            db = AsyncMock()
            dispatch_task = self._dispatch_task(
                task_id, session.room_id, "Direct task", agent_id
            )
            db.get = AsyncMock(return_value=dispatch_task)
            await apply_close_actions(session, actions, db)

        mock_dispatch.assert_awaited_once_with(
            dispatch_task, session.room_id, db, source="session_close_created_task",
        )

    @pytest.mark.asyncio
    async def test_goal_reopened_event_emitted(self):
        """GOAL_REOPENED must be emitted for any goal that was revived."""
        goal_id = uuid.uuid4()
        outcome = self._make_outcome(
            reopened_goals={goal_id: "My completed goal"},
            tasks=[{
                "task_id": uuid.uuid4(),
                "title": "Follow-up",
                "goal_id": goal_id,
                "assigned_agent_id": None,
                "plan_id": None,
            }],
        )
        session = _make_session("user_cm_p1")
        actions = SessionCloseActions()
        emitted: list[str] = []

        async def fake_emit(room_id, event_type, content, db, **kw):
            emitted.append(event_type)

        with (
            patch("app.core.session_close._do_apply", new_callable=AsyncMock, return_value=outcome),
            patch("app.core.session_close.filter_by_kind", return_value=(actions, [])),
            patch("app.core.events.emit_event", side_effect=fake_emit),
        ):
            db = AsyncMock()
            await apply_close_actions(session, actions, db)

        assert "GOAL_REOPENED" in emitted
        assert "TASK_CREATED" in emitted
        # GOAL_REOPENED must come before TASK_CREATED
        assert emitted.index("GOAL_REOPENED") < emitted.index("TASK_CREATED")

    @pytest.mark.asyncio
    async def test_multiple_tasks_same_agent_all_dispatched(self):
        """Each task with an agent must be dispatched independently."""
        plan_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        task_ids = [uuid.uuid4(), uuid.uuid4()]
        outcome = self._make_outcome(tasks=[
            {"task_id": tid, "title": f"Task {i}", "goal_id": goal_id,
             "assigned_agent_id": agent_id, "plan_id": plan_id}
            for i, tid in enumerate(task_ids)
        ])
        session = _make_session("cm_self_plan")
        actions = SessionCloseActions()
        dispatched: list[uuid.UUID] = []

        async def fake_dispatch(task, room_id, db, **kwargs):
            dispatched.append(task.id)
            return "started"

        with (
            patch("app.core.session_close._do_apply", new_callable=AsyncMock, return_value=outcome),
            patch("app.core.session_close.filter_by_kind", return_value=(actions, [])),
            patch("app.core.events.emit_event", new_callable=AsyncMock),
            patch("app.core.proxy_session.request_task_dispatch", side_effect=fake_dispatch),
        ):
            db = AsyncMock()
            db.get = AsyncMock(side_effect=[
                self._dispatch_task(task_id, session.room_id, f"Task {index}", agent_id, goal_id)
                for index, task_id in enumerate(task_ids)
            ])
            await apply_close_actions(session, actions, db)

        assert sorted(dispatched) == sorted(task_ids)


# ---------------------------------------------------------------------------
# Stage 4 DB integration tests — goal reopen and plan dispatch look-up
# ---------------------------------------------------------------------------


@pytest.mark.db
@pytestmark_db
@pytest.mark.asyncio
class TestApplyCloseActionsStage4DB:
    """Stage 4 DB tests: goal reopening and TASK_CREATED emission."""

    async def _setup(self, db):
        """Same fixture helper as TestApplyCloseActionsDB."""
        room_id = uuid.uuid4()
        user_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        step_id = uuid.uuid4()

        await _insert_fixtures(db, room_id, user_id, agent_id)

        from sqlalchemy import text

        await db.execute(
            text(
                "INSERT INTO room_goal (id, room_id, title, description, status, position, "
                "created_at, updated_at) VALUES (:id, :rid, 'g', '', 'ACTIVE', 0, now(), now())"
            ),
            {"id": str(goal_id), "rid": str(room_id)},
        )
        await db.execute(
            text(
                "INSERT INTO goal_plan (id, goal_id, status, created_at, updated_at) "
                "VALUES (:id, :gid, 'DRAFT', now(), now())"
            ),
            {"id": str(plan_id), "gid": str(goal_id)},
        )
        await db.execute(
            text(
                "INSERT INTO goal_plan_step (id, plan_id, title, description, status, position, "
                "is_parallel, created_at, updated_at) "
                "VALUES (:id, :pid, 'Step 1', '', 'PENDING', 0, false, now(), now())"
            ),
            {"id": str(step_id), "pid": str(plan_id)},
        )
        await db.flush()

        return dict(
            room_id=room_id, user_id=user_id, agent_id=agent_id,
            goal_id=goal_id, plan_id=plan_id, step_id=step_id,
        )

    async def _make_session(self, kind, *, fixtures, db, side="agent"):
        from app.models.session import CmSelfPlanSession, UserCmSession

        common = dict(id=uuid.uuid4(), room_id=fixtures["room_id"], kind=kind)
        if kind in ("user_cm_p1", "user_cm_p2"):
            obj = UserCmSession(
                **common,
                user_id=fixtures["user_id"],
                chat_master_agent_id=fixtures["agent_id"],
                goal_id=fixtures["goal_id"],
            )
        elif kind == "cm_self_plan":
            obj = CmSelfPlanSession(
                **common,
                chat_master_agent_id=fixtures["agent_id"],
                goal_id=fixtures["goal_id"],
                plan_id=fixtures["plan_id"],
            )
        else:
            raise ValueError(kind)
        db.add(obj)
        await db.flush()
        return obj

    async def test_task_created_in_active_goal_emits_task_created(self, db_session):
        """user_cm_p1 tasks_create → TASK_CREATED emitted (no dispatch, no agent)."""
        from sqlalchemy import select, text
        from app.models.message import Message

        fixtures = await self._setup(db_session)
        session = await self._make_session("user_cm_p1", fixtures=fixtures, db=db_session)

        actions = SessionCloseActions(
            tasks_create=[TaskCreate(title="Unassigned task", goal_id=fixtures["goal_id"])],
        )

        with patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch:
            await apply_close_actions(session, actions, db_session)

        mock_dispatch.assert_not_called()

        msgs = (
            await db_session.execute(
                select(Message).where(
                    Message.room_id == fixtures["room_id"],
                    Message.event_type == "TASK_CREATED",
                )
            )
        ).scalars().all()
        assert len(msgs) == 1
        assert msgs[0].event_metadata["title"] == "Unassigned task"

    async def test_task_with_agent_dispatches_proxy(self, db_session):
        """cm_self_plan with assigned_agent_id → dispatch_task_event called."""
        fixtures = await self._setup(db_session)
        session = await self._make_session("cm_self_plan", fixtures=fixtures, db=db_session)

        actions = SessionCloseActions(
            tasks_create=[
                TaskCreate(
                    title="Agent task",
                    goal_id=fixtures["goal_id"],
                    assigned_agent_id=fixtures["agent_id"],
                )
            ],
        )

        with patch("app.core.proxy_session.dispatch_task_event") as mock_dispatch:
            await apply_close_actions(session, actions, db_session)

        mock_dispatch.assert_called_once()
        call_kwargs = mock_dispatch.call_args.kwargs
        assert call_kwargs["goal_id"] == fixtures["goal_id"]
        assert call_kwargs["plan_id"] == fixtures["plan_id"]

    async def test_completed_goal_reopened_on_new_task(self, db_session):
        """Creating a task for a COMPLETED goal reopens it to ACTIVE + emits GOAL_REOPENED."""
        from sqlalchemy import select, text
        from app.models.goal import GoalStatus, RoomGoal
        from app.models.message import Message

        fixtures = await self._setup(db_session)

        # Mark the goal as completed
        await db_session.execute(
            text(
                "UPDATE room_goal SET status='COMPLETED', completed_at=now() WHERE id=:id"
            ),
            {"id": str(fixtures["goal_id"])},
        )
        await db_session.flush()

        session = await self._make_session("user_cm_p1", fixtures=fixtures, db=db_session)
        actions = SessionCloseActions(
            tasks_create=[TaskCreate(title="New work", goal_id=fixtures["goal_id"])],
        )

        with patch("app.core.proxy_session.dispatch_task_event"):
            await apply_close_actions(session, actions, db_session)

        # Goal must be ACTIVE again
        goal = (
            await db_session.execute(
                select(RoomGoal).where(RoomGoal.id == fixtures["goal_id"])
            )
        ).scalars().first()
        assert goal is not None
        assert goal.status == GoalStatus.ACTIVE
        assert goal.completed_at is None

        # GOAL_REOPENED event must have been emitted
        msgs = (
            await db_session.execute(
                select(Message).where(
                    Message.room_id == fixtures["room_id"],
                    Message.event_type == "GOAL_REOPENED",
                )
            )
        ).scalars().all()
        assert len(msgs) == 1
        assert str(fixtures["goal_id"]) in msgs[0].event_metadata.get("goal_id", "")
