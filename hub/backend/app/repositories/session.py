"""SessionRepository — unified CRUD for all session kinds."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectin_polymorphic

from app.models.session import (
    SESSION_SUBTYPES,
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

logger = logging.getLogger(__name__)


class SessionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # create
    # ------------------------------------------------------------------

    async def create(
        self,
        payload: (
            UserAgentSessionCreate
            | UserCmSessionCreate
            | CmSelfPlanSessionCreate
            | CmAgentProxySessionCreate
            | AgentSubtaskSessionCreate
        ),
    ) -> Session:
        """Create a new session of the appropriate subtype.

        SQLAlchemy JTI handles both INSERTs (session + subtype table) in one
        flush. Pydantic validation guarantees required-per-kind fields exist.
        """
        kind = payload.kind
        db = self._db

        if kind == "user_agent_p1":
            assert isinstance(payload, UserAgentSessionCreate)
            obj: Session = UserAgentSession(
                room_id=payload.room_id,
                kind=kind,
                user_id=payload.user_id,
                agent_id=payload.agent_id,
                task_id=payload.task_id,
                goal_id=payload.goal_id,
            )

        elif kind == "user_agent_p2":
            assert isinstance(payload, UserAgentSessionCreate)
            obj = UserAgentSessionP2(
                room_id=payload.room_id,
                kind=kind,
                user_id=payload.user_id,
                agent_id=payload.agent_id,
                task_id=payload.task_id,
                goal_id=payload.goal_id,
            )

        elif kind == "user_cm_p1":
            assert isinstance(payload, UserCmSessionCreate)
            obj = UserCmSession(
                room_id=payload.room_id,
                kind=kind,
                user_id=payload.user_id,
                chat_master_agent_id=payload.chat_master_agent_id,
                goal_id=payload.goal_id,
            )

        elif kind == "user_cm_p2":
            assert isinstance(payload, UserCmSessionCreate)
            obj = UserCmSessionP2(
                room_id=payload.room_id,
                kind=kind,
                user_id=payload.user_id,
                chat_master_agent_id=payload.chat_master_agent_id,
                goal_id=payload.goal_id,
            )

        elif kind == "cm_self_plan":
            assert isinstance(payload, CmSelfPlanSessionCreate)
            obj = CmSelfPlanSession(
                room_id=payload.room_id,
                kind=kind,
                chat_master_agent_id=payload.chat_master_agent_id,
                goal_id=payload.goal_id,
                plan_id=payload.plan_id,
            )

        elif kind == "agent_subtask":
            assert isinstance(payload, AgentSubtaskSessionCreate)
            obj = AgentSubtaskSession(
                room_id=payload.room_id,
                kind=kind,
                agent_id=payload.agent_id,
                task_id=payload.task_id,
                parent_session_id=payload.parent_session_id,
                step_title=payload.step_title,
            )

        else:  # cm_agent_proxy
            assert isinstance(payload, CmAgentProxySessionCreate)
            obj = CmAgentProxySession(
                room_id=payload.room_id,
                kind=kind,
                side=payload.side,
                task_id=payload.task_id,
                goal_id=payload.goal_id,
                plan_id=payload.plan_id,
                agent_id=payload.agent_id,
                peer_session_id=payload.peer_session_id,
            )

        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    async def close(
        self,
        session_id: UUID,
        close_reason: str = "task_done",
        summary: Optional[str] = None,
        pending_actions: Optional[dict] = None,
    ) -> Optional[Session]:
        """Mark a session as closed. Returns None if not found."""
        session = await self._get_base(session_id)
        if session is None:
            return None
        session.closed_at = datetime.now(timezone.utc)
        session.close_reason = close_reason
        if summary is not None:
            session.summary = summary
        if pending_actions is not None:
            session.pending_actions = pending_actions
        await self._db.flush()
        return session

    # ------------------------------------------------------------------
    # list_room
    # ------------------------------------------------------------------

    async def list_room(
        self,
        room_id: UUID,
        kinds: Optional[List[str]] = None,
        open_only: bool = False,
    ) -> List[Session]:
        """Return all sessions for a room with subtype data loaded eagerly."""
        stmt = (
            select(Session)
            .where(Session.room_id == room_id)
            .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
            .order_by(Session.started_at.desc())
        )
        if kinds:
            stmt = stmt.where(Session.kind.in_(kinds))
        if open_only:
            stmt = stmt.where(Session.closed_at.is_(None))

        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # find_open_peer
    # ------------------------------------------------------------------

    async def find_open_peer(self, session_id: UUID) -> Optional[Session]:
        """Return the open peer session for a cm_agent_proxy pair."""
        base = await self._get_base(session_id)
        if base is None or base.peer_session_id is None:
            return None

        result = await self._db.execute(
            select(Session)
            .where(
                Session.id == base.peer_session_id,
                Session.closed_at.is_(None),
            )
            .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
        )
        return result.scalars().first()

    # ------------------------------------------------------------------
    # find_open_user_session
    # ------------------------------------------------------------------

    async def find_open_user_session(
        self,
        room_id: UUID,
        user_id: UUID,
    ) -> Optional[Session]:
        """Find the most recent open user-facing session for a user in a room.

        Checks both user_agent_session and user_cm_session tables.
        Returns None if no open session exists.
        """
        user_kinds = ("user_agent_p1", "user_agent_p2", "user_cm_p1", "user_cm_p2")
        stmt = (
            select(Session)
            .where(
                Session.room_id == room_id,
                Session.kind.in_(user_kinds),
                Session.closed_at.is_(None),
            )
            .options(selectin_polymorphic(Session, SESSION_SUBTYPES))
            .order_by(Session.started_at.desc())
            .limit(1)
        )
        result = await self._db.execute(stmt)
        session = result.scalars().first()
        if session is None:
            return None

        session_user_id: Optional[UUID] = getattr(session, "user_id", None)
        if session_user_id != user_id:
            return None
        return session

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def _get_base(self, session_id: UUID) -> Optional[Session]:
        result = await self._db.execute(
            select(Session).where(Session.id == session_id)
        )
        return result.scalars().first()
