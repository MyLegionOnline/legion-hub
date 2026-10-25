"""Unit tests for Session hierarchy (Step 1).

Tests cover:
- Pydantic schema validation (model_validator, required fields per kind)
- Session model class hierarchy (polymorphic identities)
- Polymorphic list_room via SessionRepository (requires a live PostgreSQL DB)

DB tests use the DATABASE_URL env var and are skipped when it is unset.
Run DB tests with:
    DATABASE_URL=postgresql+asyncpg://... pytest tests/test_session_models.py -m db
"""

from __future__ import annotations

import os
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from pydantic import ValidationError

from app.models.session import (
    AgentSubtaskSession,
    CmAgentProxySession,
    CmSelfPlanSession,
    Session,
    UserAgentSession,
    UserAgentSessionP2,
    UserCmSession,
    UserCmSessionP2,
)
from app.schemas.session import (
    AgentSubtaskSessionCreate,
    CmAgentProxySessionCreate,
    CmSelfPlanSessionCreate,
    UserAgentSessionCreate,
    UserCmSessionCreate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROOM_ID = uuid.uuid4()
_USER_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()
_TASK_ID = uuid.uuid4()
_GOAL_ID = uuid.uuid4()
_PLAN_ID = uuid.uuid4()


# ---------------------------------------------------------------------------
# 1. Pydantic schema validation
# ---------------------------------------------------------------------------


class TestUserAgentSessionCreate:
    def test_p1_requires_task_and_goal(self):
        with pytest.raises(ValidationError, match="task_id and goal_id"):
            UserAgentSessionCreate(
                kind="user_agent_p1",
                room_id=_ROOM_ID,
                user_id=_USER_ID,
                agent_id=_AGENT_ID,
                # task_id and goal_id missing
            )

    def test_p1_valid(self):
        s = UserAgentSessionCreate(
            kind="user_agent_p1",
            room_id=_ROOM_ID,
            user_id=_USER_ID,
            agent_id=_AGENT_ID,
            task_id=_TASK_ID,
            goal_id=_GOAL_ID,
        )
        assert s.kind == "user_agent_p1"
        assert s.task_id == _TASK_ID

    def test_p2_no_task_required(self):
        s = UserAgentSessionCreate(
            kind="user_agent_p2",
            room_id=_ROOM_ID,
            user_id=_USER_ID,
            agent_id=_AGENT_ID,
        )
        assert s.kind == "user_agent_p2"
        assert s.task_id is None
        assert s.goal_id is None


class TestUserCmSessionCreate:
    def test_p1_requires_goal(self):
        with pytest.raises(ValidationError, match="goal_id"):
            UserCmSessionCreate(
                kind="user_cm_p1",
                room_id=_ROOM_ID,
                user_id=_USER_ID,
                chat_master_agent_id=_AGENT_ID,
                # goal_id missing
            )

    def test_p1_valid(self):
        s = UserCmSessionCreate(
            kind="user_cm_p1",
            room_id=_ROOM_ID,
            user_id=_USER_ID,
            chat_master_agent_id=_AGENT_ID,
            goal_id=_GOAL_ID,
        )
        assert s.goal_id == _GOAL_ID

    def test_p2_no_goal_required(self):
        s = UserCmSessionCreate(
            kind="user_cm_p2",
            room_id=_ROOM_ID,
            user_id=_USER_ID,
            chat_master_agent_id=_AGENT_ID,
        )
        assert s.goal_id is None


class TestCmSelfPlanSessionCreate:
    def test_requires_all_fields(self):
        with pytest.raises(ValidationError):
            CmSelfPlanSessionCreate(
                room_id=_ROOM_ID,
                chat_master_agent_id=_AGENT_ID,
                goal_id=_GOAL_ID,
                # plan_id missing
            )

    def test_valid(self):
        s = CmSelfPlanSessionCreate(
            room_id=_ROOM_ID,
            chat_master_agent_id=_AGENT_ID,
            goal_id=_GOAL_ID,
            plan_id=_PLAN_ID,
        )
        assert s.kind == "cm_self_plan"
        assert s.plan_id == _PLAN_ID


class TestCmAgentProxySessionCreate:
    def test_valid_direct_task_without_goal_or_plan(self):
        s = CmAgentProxySessionCreate(
            room_id=_ROOM_ID,
            side="agent",
            task_id=_TASK_ID,
            agent_id=_AGENT_ID,
        )
        assert s.goal_id is None
        assert s.plan_id is None

    def test_valid_agent_side(self):
        s = CmAgentProxySessionCreate(
            room_id=_ROOM_ID,
            side="agent",
            task_id=_TASK_ID,
            goal_id=_GOAL_ID,
            plan_id=_PLAN_ID,
            agent_id=_AGENT_ID,
        )
        assert s.side == "agent"
        assert s.peer_session_id is None

    def test_valid_chat_master_side(self):
        peer = uuid.uuid4()
        s = CmAgentProxySessionCreate(
            room_id=_ROOM_ID,
            side="chat_master",
            task_id=_TASK_ID,
            goal_id=_GOAL_ID,
            plan_id=_PLAN_ID,
            agent_id=_AGENT_ID,
            peer_session_id=peer,
        )
        assert s.peer_session_id == peer


class TestAgentSubtaskSessionCreate:
    """Stage 5.4 of internal design notes."""

    def test_valid(self):
        parent = uuid.uuid4()
        s = AgentSubtaskSessionCreate(
            room_id=_ROOM_ID,
            agent_id=_AGENT_ID,
            task_id=_TASK_ID,
            parent_session_id=parent,
            step_title="Implement parser",
        )
        assert s.kind == "agent_subtask"
        assert s.parent_session_id == parent
        assert s.task_id == _TASK_ID

    def test_task_id_optional(self):
        s = AgentSubtaskSessionCreate(
            room_id=_ROOM_ID,
            agent_id=_AGENT_ID,
            parent_session_id=uuid.uuid4(),
        )
        assert s.task_id is None
        assert s.step_title == ""

    def test_parent_session_id_required(self):
        with pytest.raises(ValidationError):
            AgentSubtaskSessionCreate(
                room_id=_ROOM_ID,
                agent_id=_AGENT_ID,
            )


# ---------------------------------------------------------------------------
# 2. ORM class hierarchy (no DB needed)
# ---------------------------------------------------------------------------


class TestSessionClassHierarchy:
    def test_all_subtypes_inherit_from_session(self):
        assert issubclass(UserAgentSession, Session)
        assert issubclass(UserAgentSessionP2, Session)
        assert issubclass(UserCmSession, Session)
        assert issubclass(UserCmSessionP2, Session)
        assert issubclass(CmSelfPlanSession, Session)
        assert issubclass(CmAgentProxySession, Session)
        assert issubclass(AgentSubtaskSession, Session)

    def test_p2_inherits_from_p1_class(self):
        assert issubclass(UserAgentSessionP2, UserAgentSession)
        assert issubclass(UserCmSessionP2, UserCmSession)

    def test_polymorphic_identities(self):
        mapper_ua = UserAgentSession.__mapper__
        mapper_ua_p2 = UserAgentSessionP2.__mapper__
        mapper_ucm = UserCmSession.__mapper__
        mapper_ucm_p2 = UserCmSessionP2.__mapper__
        mapper_csp = CmSelfPlanSession.__mapper__
        mapper_cap = CmAgentProxySession.__mapper__
        mapper_ats = AgentSubtaskSession.__mapper__

        assert mapper_ua.polymorphic_identity == "user_agent_p1"
        assert mapper_ua_p2.polymorphic_identity == "user_agent_p2"
        assert mapper_ucm.polymorphic_identity == "user_cm_p1"
        assert mapper_ucm_p2.polymorphic_identity == "user_cm_p2"
        assert mapper_csp.polymorphic_identity == "cm_self_plan"
        assert mapper_cap.polymorphic_identity == "cm_agent_proxy"
        assert mapper_ats.polymorphic_identity == "agent_subtask"

    def test_session_has_polymorphic_on_kind(self):
        polymorphic_on = Session.__mapper__.polymorphic_on
        assert polymorphic_on is not None
        assert polymorphic_on.key == "kind"

    def test_session_has_is_open_property(self):
        import types
        from datetime import datetime, timezone

        # Use SimpleNamespace so SQLAlchemy descriptors don't interfere
        s = types.SimpleNamespace(closed_at=None)
        assert Session.is_open.fget(s) is True
        s.closed_at = datetime.now(timezone.utc)
        assert Session.is_open.fget(s) is False

    def test_session_total_tokens(self):
        import types

        s = types.SimpleNamespace(tokens_in=30_000, tokens_out=70_000)
        assert Session.total_tokens.fget(s) == 100_000


# ---------------------------------------------------------------------------
# 3. DB integration tests (require PostgreSQL)
# ---------------------------------------------------------------------------

DB_URL = os.environ.get("DATABASE_URL", "")
pytestmark_db = pytest.mark.skipif(not DB_URL, reason="DATABASE_URL not set")


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator:
    """Async SQLAlchemy session connected to the test database."""
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from sqlmodel import SQLModel

    # Import all models so metadata is populated
    import app.models.session  # noqa: F401

    engine = create_async_engine(DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


@pytestmark_db
@pytest.mark.asyncio
@pytest.mark.db
class TestSessionRepositoryDB:
    """Integration tests against a live PostgreSQL database."""

    async def _make_room_user_agent(self, db):
        """Helper: insert minimal fixtures and return (room_id, user_id, agent_id)."""
        import uuid as _uuid
        from sqlalchemy import text

        room_id = _uuid.uuid4()
        user_id = _uuid.uuid4()
        agent_id = _uuid.uuid4()

        await db.execute(
            text(
                "INSERT INTO \"user\" (id, email, hashed_password) VALUES (:id, :email, 'x')"
            ),
            {"id": str(user_id), "email": f"{user_id}@test.com"},
        )
        await db.execute(
            text("INSERT INTO room (id, name, creator_id) VALUES (:id, 'test', :cid)"),
            {"id": str(room_id), "cid": str(user_id)},
        )
        await db.execute(
            text(
                "INSERT INTO agent (id, name, owner_id, system_prompt, short_description)"
                " VALUES (:id, 'a', :oid, '', '')"
            ),
            {"id": str(agent_id), "oid": str(user_id)},
        )
        await db.flush()
        return room_id, user_id, agent_id

    async def test_create_user_agent_p1(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        task_id = uuid.uuid4()
        goal_id = uuid.uuid4()

        await db_session.execute(
            __import__("sqlalchemy").text(
                "INSERT INTO task (id, room_id, title) VALUES (:id, :rid, 't')"
            ),
            {"id": str(task_id), "rid": str(room_id)},
        )
        await db_session.execute(
            __import__("sqlalchemy").text(
                "INSERT INTO room_goal (id, room_id, title) VALUES (:id, :rid, 'g')"
            ),
            {"id": str(goal_id), "rid": str(room_id)},
        )
        await db_session.flush()

        repo = SessionRepository(db_session)
        payload = UserAgentSessionCreate(
            kind="user_agent_p1",
            room_id=room_id,
            user_id=user_id,
            agent_id=agent_id,
            task_id=task_id,
            goal_id=goal_id,
        )
        session = await repo.create(payload)

        assert isinstance(session, UserAgentSession)
        assert session.kind == "user_agent_p1"
        assert session.task_id == task_id
        assert session.is_open

    async def test_create_user_agent_p2(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        repo = SessionRepository(db_session)
        payload = UserAgentSessionCreate(
            kind="user_agent_p2",
            room_id=room_id,
            user_id=user_id,
            agent_id=agent_id,
        )
        session = await repo.create(payload)
        assert isinstance(session, UserAgentSessionP2)
        assert session.kind == "user_agent_p2"
        assert session.task_id is None

    async def test_create_user_cm_p1(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        goal_id = uuid.uuid4()
        await db_session.execute(
            __import__("sqlalchemy").text(
                "INSERT INTO room_goal (id, room_id, title) VALUES (:id, :rid, 'g')"
            ),
            {"id": str(goal_id), "rid": str(room_id)},
        )
        await db_session.flush()

        repo = SessionRepository(db_session)
        payload = UserCmSessionCreate(
            kind="user_cm_p1",
            room_id=room_id,
            user_id=user_id,
            chat_master_agent_id=agent_id,
            goal_id=goal_id,
        )
        session = await repo.create(payload)
        assert isinstance(session, UserCmSession)
        assert session.kind == "user_cm_p1"

    async def test_create_user_cm_p2(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        repo = SessionRepository(db_session)
        payload = UserCmSessionCreate(
            kind="user_cm_p2",
            room_id=room_id,
            user_id=user_id,
            chat_master_agent_id=agent_id,
        )
        session = await repo.create(payload)
        assert isinstance(session, UserCmSessionP2)

    async def test_create_cm_self_plan(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        sa = __import__("sqlalchemy")
        await db_session.execute(
            sa.text("INSERT INTO room_goal (id, room_id, title) VALUES (:id, :rid, 'g')"),
            {"id": str(goal_id), "rid": str(room_id)},
        )
        await db_session.execute(
            sa.text("INSERT INTO goal_plan (id, goal_id) VALUES (:id, :gid)"),
            {"id": str(plan_id), "gid": str(goal_id)},
        )
        await db_session.flush()

        repo = SessionRepository(db_session)
        payload = CmSelfPlanSessionCreate(
            room_id=room_id,
            chat_master_agent_id=agent_id,
            goal_id=goal_id,
            plan_id=plan_id,
        )
        session = await repo.create(payload)
        assert isinstance(session, CmSelfPlanSession)
        assert session.kind == "cm_self_plan"

    async def test_create_cm_agent_proxy_pair(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        sa = __import__("sqlalchemy")
        await db_session.execute(
            sa.text("INSERT INTO room_goal (id, room_id, title) VALUES (:id, :rid, 'g')"),
            {"id": str(goal_id), "rid": str(room_id)},
        )
        await db_session.execute(
            sa.text("INSERT INTO goal_plan (id, goal_id) VALUES (:id, :gid)"),
            {"id": str(plan_id), "gid": str(goal_id)},
        )
        await db_session.execute(
            sa.text("INSERT INTO task (id, room_id, title) VALUES (:id, :rid, 't')"),
            {"id": str(task_id), "rid": str(room_id)},
        )
        await db_session.flush()

        repo = SessionRepository(db_session)

        agent_side = await repo.create(
            CmAgentProxySessionCreate(
                room_id=room_id,
                side="agent",
                task_id=task_id,
                goal_id=goal_id,
                plan_id=plan_id,
                agent_id=agent_id,
            )
        )
        cm_side = await repo.create(
            CmAgentProxySessionCreate(
                room_id=room_id,
                side="chat_master",
                task_id=task_id,
                goal_id=goal_id,
                plan_id=plan_id,
                agent_id=agent_id,
                peer_session_id=agent_side.id,
            )
        )

        assert isinstance(agent_side, CmAgentProxySession)
        assert isinstance(cm_side, CmAgentProxySession)
        assert cm_side.peer_session_id == agent_side.id

    async def test_polymorphic_list_room(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        repo = SessionRepository(db_session)

        await repo.create(
            UserAgentSessionCreate(
                kind="user_agent_p2",
                room_id=room_id,
                user_id=user_id,
                agent_id=agent_id,
            )
        )
        await repo.create(
            UserCmSessionCreate(
                kind="user_cm_p2",
                room_id=room_id,
                user_id=user_id,
                chat_master_agent_id=agent_id,
            )
        )

        sessions = await repo.list_room(room_id)
        assert len(sessions) == 2
        kinds = {s.kind for s in sessions}
        assert kinds == {"user_agent_p2", "user_cm_p2"}

        # Verify subtype fields are loaded (not Session base only)
        ua_sessions = [s for s in sessions if isinstance(s, UserAgentSessionP2)]
        assert len(ua_sessions) == 1
        assert ua_sessions[0].agent_id == agent_id

    async def test_close_session(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        repo = SessionRepository(db_session)

        created = await repo.create(
            UserAgentSessionCreate(
                kind="user_agent_p2",
                room_id=room_id,
                user_id=user_id,
                agent_id=agent_id,
            )
        )
        assert created.is_open

        closed = await repo.close(created.id, close_reason="task_done", summary="done")
        assert closed is not None
        assert not closed.is_open
        assert closed.close_reason == "task_done"
        assert closed.summary == "done"

    async def test_find_open_peer(self, db_session):
        from app.repositories.session import SessionRepository

        room_id, user_id, agent_id = await self._make_room_user_agent(db_session)
        goal_id = uuid.uuid4()
        plan_id = uuid.uuid4()
        task_id = uuid.uuid4()
        sa = __import__("sqlalchemy")
        await db_session.execute(
            sa.text("INSERT INTO room_goal (id, room_id, title) VALUES (:id, :rid, 'g')"),
            {"id": str(goal_id), "rid": str(room_id)},
        )
        await db_session.execute(
            sa.text("INSERT INTO goal_plan (id, goal_id) VALUES (:id, :gid)"),
            {"id": str(plan_id), "gid": str(goal_id)},
        )
        await db_session.execute(
            sa.text("INSERT INTO task (id, room_id, title) VALUES (:id, :rid, 't')"),
            {"id": str(task_id), "rid": str(room_id)},
        )
        await db_session.flush()

        repo = SessionRepository(db_session)
        proxy_payload = CmAgentProxySessionCreate(
            room_id=room_id,
            side="agent",
            task_id=task_id,
            goal_id=goal_id,
            plan_id=plan_id,
            agent_id=agent_id,
        )
        agent_side = await repo.create(proxy_payload)
        cm_side = await repo.create(
            CmAgentProxySessionCreate(
                room_id=room_id,
                side="chat_master",
                task_id=task_id,
                goal_id=goal_id,
                plan_id=plan_id,
                agent_id=agent_id,
                peer_session_id=agent_side.id,
            )
        )

        # cm_side's peer is agent_side (open)
        peer = await repo.find_open_peer(cm_side.id)
        assert peer is not None
        assert peer.id == agent_side.id

        # after closing agent_side, peer should not be found
        await repo.close(agent_side.id)
        peer_after = await repo.find_open_peer(cm_side.id)
        assert peer_after is None
