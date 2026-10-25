"""
Emit system events into a room's message stream and broadcast over WebSocket.
Import this from tasks.py, memory.py, etc. to avoid circular deps with chat.py.

Two channels:
  * emit_event  — persistent: writes a Message row, then broadcasts (history).
  * emit_status — ephemeral: WS broadcast only, no DB write (live agent status).
"""
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import llm_call_state
from app.models.message import Message
from app.websockets.manager import manager

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    """Ephemeral agent activity statuses broadcast via emit_status."""

    LLM_REQUEST_SENT = "llm_request_sent"
    LLM_RESPONSE_RECEIVED = "llm_response_received"
    LLM_HEARTBEAT = "llm_heartbeat"
    LLM_ERROR = "llm_error"
    LLM_EMPTY_RESPONSE = "llm_empty_response"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_FINISHED = "tool_call_finished"
    IMAGE_GENERATION_PROGRESS = "image_generation_progress"
    WAITING_PEER = "waiting_peer"
    SESSION_STALLED = "session_stalled"
    DISPATCHING = "dispatching"
    # A direct-chat turn was picked up; context/workspace/MCP prep is running
    # and no model request has gone out yet.
    TURN_ACCEPTED = "turn_accepted"


async def emit_event(
    room_id: UUID,
    event_type: str,
    content: str,
    db: AsyncSession,
    sender_agent_id: Optional[UUID] = None,
    task_id: Optional[UUID] = None,
    metadata: Optional[Dict[str, Any]] = None,
    sender_type: Optional[str] = None,
) -> Message:
    msg = Message(
        room_id=room_id,
        event_type=event_type,
        content=content,
        task_id=task_id,
        sender_agent_id=sender_agent_id,
        event_metadata=metadata or {},
    )
    db.add(msg)
    await db.commit()

    resolved_sender_type = sender_type or ("agent" if sender_agent_id else "system")

    await manager.broadcast(
        {
            "type": "message",
            "event_type": event_type,
            "id": str(msg.id),
            "room_id": str(room_id),
            "sender_id": str(sender_agent_id) if sender_agent_id else None,
            "sender_type": resolved_sender_type,
            "nickname": (metadata or {}).get("nickname"),
            "content": content,
            "timestamp": msg.timestamp.isoformat(),
            "status": msg.status,
            "task_id": str(task_id) if task_id else None,
            "metadata": metadata or {},
        },
        room_id,
    )
    return msg


# Reasoning text is a chat nicety, not a transcript: 8000 characters is far
# more than anyone expands, and the tail costs a row in every history load.
REASONING_CONTENT_LIMIT = 8000


async def emit_reasoning_event(
    *,
    room_id: UUID,
    session_id: UUID,
    agent_id: Optional[UUID],
    actor_label: Optional[str],
    task_id: Optional[UUID],
    thinking: str,
    reasoning_tokens: int,
    duration_ms: int,
    db: AsyncSession,
) -> None:
    """Persist the agent's thinking as a collapsible chat block (plan §7.1).

    Swallows its own failures for the same reason ``_emit_tool_call_event``
    does: a chat-visibility nicety must never break a turn.

    An empty ``thinking`` emits nothing, and that is a real case rather than a
    defensive check — measured §4A.5 q3: closed reasoners (Gemini, Claude via
    OpenRouter) and the explicit ``reasoning:{exclude:true}`` mode spend
    reasoning tokens while returning no text at all.  The tokens are still
    counted on the session; only the empty block is skipped.
    """
    if not (thinking or "").strip():
        return
    try:
        await emit_event(
            room_id=room_id,
            event_type="AGENT_REASONING",
            content=thinking[:REASONING_CONTENT_LIMIT],
            db=db,
            sender_agent_id=agent_id,
            task_id=task_id,
            metadata={
                "session_id": str(session_id),
                "nickname": actor_label,
                "agent_id": str(agent_id) if agent_id else None,
                "reasoning_tokens": reasoning_tokens,
                "duration_ms": duration_ms,
                "truncated": len(thinking) > REASONING_CONTENT_LIMIT,
            },
        )
    except Exception as exc:
        logger.warning("emit AGENT_REASONING failed (session=%s): %s", session_id, exc)


async def emit_interim_text_event(
    *,
    room_id: UUID,
    session_id: UUID,
    agent_id: Optional[UUID],
    actor_label: Optional[str],
    task_id: Optional[UUID],
    text: Optional[str],
    db: AsyncSession,
) -> None:
    """Persist the text a model wrote alongside its tool calls.

    A tool-loop step that calls a tool may also say something ("Timing has to
    match the real transients — checking."). The loops kept that text only in
    their working history, so the chat showed the reasoning and the tool
    calls but never the agent's own words.  It is stored as its own
    AGENT_INTERIM_TEXT event rather than MESSAGE: the text is already in the
    turn's working history, and nothing that rebuilds a dialogue from MESSAGE
    rows may see it twice.

    Swallows its own failures, like ``emit_reasoning_event``.
    """
    from app.core.session_close import strip_tool_call_markup

    cleaned, _leaked = strip_tool_call_markup(text or "")
    if not cleaned:
        return
    sender_type = "agent" if agent_id else "master"
    try:
        await emit_event(
            room_id=room_id,
            event_type="AGENT_INTERIM_TEXT",
            content=cleaned,
            db=db,
            sender_agent_id=agent_id,
            task_id=task_id,
            sender_type=sender_type,
            metadata={
                "session_id": str(session_id),
                # Tool-call labels carry "@nick"; a bubble's sender is the bare name.
                "nickname": (actor_label or "").lstrip("@") or None,
                "agent_id": str(agent_id) if agent_id else None,
                "sender_type": sender_type,
            },
        )
    except Exception as exc:
        logger.warning("emit AGENT_INTERIM_TEXT failed (session=%s): %s", session_id, exc)


async def emit_status(
    room_id: UUID,
    status: "AgentStatus | str",
    *,
    agent_id: Optional[UUID] = None,
    session_id: Optional[UUID] = None,
    task_id: Optional[UUID] = None,
    detail: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Broadcast an ephemeral agent status over WebSocket. No DB write.

    Statuses can fire dozens of times per minute (heartbeats, tool calls),
    so they must never be persisted and must never break the caller —
    all failures are swallowed and logged.
    """
    try:
        # internal design notes Stage 0: every tool call of the proxy loop
        # already reports its start/end here, so this is the one place that
        # tells the heartbeat "running read_file" apart from "waiting on the
        # model" (see app/core/llm_call_state.py).
        if status == AgentStatus.TOOL_CALL_STARTED:
            llm_call_state.begin_tool(session_id, detail)
        elif status == AgentStatus.TOOL_CALL_FINISHED:
            llm_call_state.end_tool(session_id)
        payload = {
            "type": "agent_status",
            "status": status.value if isinstance(status, AgentStatus) else str(status),
            "room_id": str(room_id),
            "agent_id": str(agent_id) if agent_id else None,
            "session_id": str(session_id) if session_id else None,
            "task_id": str(task_id) if task_id else None,
            "detail": detail,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            payload.update(extra)
        await manager.broadcast(payload, room_id)
    except Exception as exc:
        logger.warning("emit_status failed (room=%s, status=%s): %s", room_id, status, exc)
