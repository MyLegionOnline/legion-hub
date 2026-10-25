"""ProxyBroker — publishes PROXY_MESSAGE events for cm_agent_proxy sessions.

Responsibilities (spec §5.3):
  (a) Record the message in the room's history (Message table).
  (b) Push a copy into the peer session's async input queue so the peer's
      execution loop can consume it.
  (c) Broadcast the event to all WebSocket clients in the room.

Validation:
  - peer_session_id must exist and be open.
  - turn is computed server-side as the count of existing PROXY_MESSAGE
    records for the task, which guarantees monotonic growth.
"""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, Dict
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent import Agent
from app.models.message import Message, PROXY_MESSAGE
from app.models.session.base import Session
from app.schemas.proxy import ProxyMessageCreate
from app.websockets.manager import manager

logger = logging.getLogger(__name__)


class ProxyBrokerError(ValueError):
    """Raised when a PROXY_MESSAGE cannot be published."""


class ProxyBroker:
    """Stateless service; peer session queues are class-level singletons."""

    _queues: ClassVar[Dict[UUID, asyncio.Queue]] = {}

    # ------------------------------------------------------------------
    # Queue management (consumed by session execution loops in Steps 7/9)
    # ------------------------------------------------------------------

    @classmethod
    def get_or_create_queue(cls, session_id: UUID) -> asyncio.Queue:
        if session_id not in cls._queues:
            cls._queues[session_id] = asyncio.Queue()
        return cls._queues[session_id]

    @classmethod
    def clear_queue(cls, session_id: UUID) -> None:
        cls._queues.pop(session_id, None)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    async def publish(
        self,
        proxy_msg: ProxyMessageCreate,
        db: AsyncSession,
    ) -> Message:
        """Publish a proxy message and return the persisted Message record."""

        # --- validation: peer session must exist ---
        peer_result = await db.execute(
            select(Session).where(Session.id == proxy_msg.peer_session_id)
        )
        peer_session = peer_result.scalars().first()
        if peer_session is None:
            raise ProxyBrokerError(
                f"Peer session {proxy_msg.peer_session_id} not found"
            )
        # A closed peer is fine for visibility purposes (e.g. CM verdict after
        # the agent already closed) — persist + broadcast, just skip its queue.
        peer_open = peer_session.is_open

        # --- resolve sender nickname for the room chat ---
        nickname: str | None = None
        agent_row = (
            await db.execute(select(Agent).where(Agent.id == proxy_msg.sender_id))
        ).scalars().first()
        if agent_row is not None:
            nickname = agent_row.nickname

        # --- turn = count of existing PROXY_MESSAGEs for this task ---
        count_result = await db.execute(
            select(func.count()).select_from(Message).where(
                Message.event_type == PROXY_MESSAGE,
                Message.task_id == proxy_msg.task_id,
            )
        )
        turn: int = count_result.scalar() or 0

        # --- (a) persist to room history ---
        msg = Message(
            room_id=proxy_msg.room_id,
            event_type=PROXY_MESSAGE,
            content=proxy_msg.content,
            task_id=proxy_msg.task_id,
            sender_agent_id=proxy_msg.sender_id,
            event_metadata={
                "session_id": str(proxy_msg.session_id),
                "peer_session_id": str(proxy_msg.peer_session_id),
                "turn": turn,
                "kind": proxy_msg.kind,
                "sender_type": proxy_msg.sender_type,
                "sender_id": str(proxy_msg.sender_id),
                "nickname": nickname,
            },
        )
        db.add(msg)
        await db.flush()
        await db.refresh(msg)

        # --- (b) copy to peer session's input queue (open peers only) ---
        if peer_open:
            peer_queue = self.__class__.get_or_create_queue(proxy_msg.peer_session_id)
            await peer_queue.put(
                {
                    "content": proxy_msg.content,
                    "sender_type": proxy_msg.sender_type,
                    "sender_id": str(proxy_msg.sender_id),
                    "task_id": str(proxy_msg.task_id),
                    "session_id": str(proxy_msg.session_id),
                    "turn": turn,
                }
            )
        else:
            logger.info(
                "PROXY_MESSAGE to closed peer %s — recorded for room visibility only",
                proxy_msg.peer_session_id,
            )

        # --- (c) broadcast via WebSocket ---
        await manager.broadcast(
            {
                "type": "message",
                "event_type": PROXY_MESSAGE,
                "id": str(msg.id),
                "room_id": str(proxy_msg.room_id),
                "sender_id": str(proxy_msg.sender_id),
                "sender_type": proxy_msg.sender_type,
                "content": proxy_msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "status": msg.status,
                "task_id": str(proxy_msg.task_id),
                "metadata": {
                    "session_id": str(proxy_msg.session_id),
                    "peer_session_id": str(proxy_msg.peer_session_id),
                    "turn": turn,
                    "kind": proxy_msg.kind,
                    "nickname": nickname,
                },
            },
            proxy_msg.room_id,
        )

        logger.info(
            "PROXY_MESSAGE published: task=%s session=%s→%s turn=%d",
            proxy_msg.task_id,
            proxy_msg.session_id,
            proxy_msg.peer_session_id,
            turn,
        )

        return msg
