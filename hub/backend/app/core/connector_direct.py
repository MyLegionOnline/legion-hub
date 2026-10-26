"""Direct room with an agent that runs on a connector.

A direct room with such an agent is a console to the owner's machine: there
is no ChatMaster and no Hub-side LLM loop. Every message the owner types
becomes one task (``Task.origin = direct_connector``) dispatched to the
connector through the usual ``request_task_dispatch`` → ``handle_task_created``
→ ``_run_connector_loop`` path. The connector's answer comes back as the
agent's chat message (``deliver_result``) and the task is completed without a
review. Group rooms are untouched: there an external agent still gets its
tasks from ChatMaster.

While a message is being worked on, the next one is either the answer to the
agent's pending question or a "the agent is busy" notice — never a second
task running in parallel on the same machine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import ask_user_registry
from app.core.events import emit_event
from app.models.agent import Agent
from app.models.message import Message
from app.models.room import Room, RoomType
from app.models.session.cm_agent_proxy import CmAgentProxySession
from app.models.task import (
    TASK_ORIGIN_DIRECT_CONNECTOR,
    DispatchStatus,
    Task,
    TaskState,
)

logger = logging.getLogger(__name__)

AGENT_BUSY = "AGENT_BUSY"

# How many earlier chat messages travel with each new message, so the remote
# agent sees the conversation and not one isolated line.
_CONTEXT_MESSAGES = 8
_CONTEXT_MESSAGE_CHARS = 2000
_TITLE_CHARS = 80


def is_direct_connector(room: Optional[Room], agent: Optional[Agent]) -> bool:
    """A direct room whose single agent is executed by a connector."""
    return (
        room is not None
        and agent is not None
        and room.room_type == RoomType.direct
        and agent.executor == "connector"
        and agent.connector_id is not None
    )


def _title(content: str) -> str:
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    if not first_line:
        return "Message"
    if len(first_line) <= _TITLE_CHARS:
        return first_line
    return first_line[: _TITLE_CHARS - 1].rstrip() + "…"


async def _running_task(room_id: UUID, agent_id: UUID, db: AsyncSession) -> Optional[Task]:
    return (
        await db.execute(
            select(Task)
            .where(
                Task.room_id == room_id,
                Task.assigned_agent_id == agent_id,
                Task.dispatch_status == DispatchStatus.DISPATCHED,
                # A finished task keeps dispatch_status=DISPATCHED.
                Task.state.in_([TaskState.ASSIGNED, TaskState.IN_PROGRESS]),
            )
            .order_by(Task.created_at.desc())
            .limit(1)
        )
    ).scalars().first()


async def _pending_question_id(task_id: UUID, db: AsyncSession) -> Optional[str]:
    session_ids = (
        await db.execute(
            select(CmAgentProxySession.id).where(
                CmAgentProxySession.task_id == task_id,
                CmAgentProxySession.side == "agent",
                CmAgentProxySession.closed_at.is_(None),
            )
        )
    ).scalars().all()
    for session_id in session_ids:
        question_id = ask_user_registry.pending_question_for_session(session_id)
        if question_id is not None:
            return question_id
    return None


async def handle_direct_message(
    content: str, room: Room, agent: Agent, db: AsyncSession
) -> str:
    """Route one owner message. Returns what happened: answered / busy / dispatched / failed."""
    running = await _running_task(room.id, agent.id, db)
    if running is not None:
        question_id = await _pending_question_id(running.id, db)
        if question_id is not None and ask_user_registry.resolve(
            question_id, {"selected_options": [], "free_text": content}
        ):
            await emit_event(
                room_id=room.id,
                event_type="AGENT_QUESTION_ANSWERED",
                content="Answered in the chat",
                db=db,
                task_id=running.id,
                metadata={"question_id": question_id, "task_id": str(running.id)},
            )
            return "answered"
        await emit_event(
            room_id=room.id,
            event_type=AGENT_BUSY,
            content=(
                f"@{agent.nickname} is still working on the previous message. "
                "Wait for the reply or press Stop."
            ),
            db=db,
            task_id=running.id,
            metadata={"task_id": str(running.id), "agent_id": str(agent.id)},
        )
        return "busy"

    task = Task(
        room_id=room.id,
        title=_title(content),
        description=content,
        assigned_agent_id=agent.id,
        state=TaskState.ASSIGNED,
        origin=TASK_ORIGIN_DIRECT_CONNECTOR,
    )
    db.add(task)
    await db.commit()

    from app.core.proxy_session import TaskDispatchPreflightError, request_task_dispatch

    try:
        await request_task_dispatch(task, room.id, db, source="direct_connector")
    except TaskDispatchPreflightError as exc:
        logger.info("connector_direct: message task %s not started: %s", task.id, exc.code)
        await emit_event(
            room_id=room.id,
            event_type="TASK_DISPATCH_FAILED",
            content=f"The message was not sent to @{agent.nickname} ({exc.code}).",
            db=db,
            task_id=task.id,
            metadata={"task_id": str(task.id), "reason": exc.code},
        )
        return "failed"
    return "dispatched"


async def conversation_context(task: Task, db: AsyncSession) -> str:
    """The chat lines before this message, oldest first, for the dispatch text.

    The connector reads only ``task_data_text``; this is how the remote agent
    learns what was said before. Event rows (dispatch pills, statuses) are
    left out, and so is the message that created ``task`` itself.
    """
    rows = (
        await db.execute(
            select(Message)
            .where(
                Message.room_id == task.room_id,
                Message.event_type == "MESSAGE",
                Message.timestamp <= task.created_at,
            )
            .order_by(Message.timestamp.desc())
            .limit(_CONTEXT_MESSAGES + 1)
        )
    ).scalars().all()
    lines: list[str] = []
    skipped_own = False
    for row in rows:
        if not skipped_own and row.sender_user_id is not None and row.content == task.description:
            skipped_own = True
            continue
        text = (row.content or "").strip()
        if not text:
            continue
        if len(text) > _CONTEXT_MESSAGE_CHARS:
            text = text[:_CONTEXT_MESSAGE_CHARS] + "…"
        speaker = "User" if row.sender_user_id is not None else "You"
        lines.append(f"{speaker}: {text}")
    lines = lines[:_CONTEXT_MESSAGES]
    lines.reverse()
    return "\n\n".join(lines)


async def deliver_result(
    db: AsyncSession,
    *,
    agent_session: CmAgentProxySession,
    task: Task,
    room_id: UUID,
    summary: str,
    success: Any,
) -> None:
    """Post the connector's answer as the agent's message and complete the task."""
    from app.core.proxy_session import _close_side, _emit_closed

    if agent_session.is_open:
        actions = await _close_side(agent_session, {"summary": summary}, "task_done", db)
        await _emit_closed(agent_session, "task_done", actions, db)

    agent = (
        await db.execute(select(Agent).where(Agent.id == agent_session.agent_id))
    ).scalars().first()
    await emit_event(
        room_id=room_id,
        event_type="MESSAGE",
        content=summary,
        db=db,
        sender_agent_id=agent_session.agent_id,
        task_id=task.id,
        metadata={
            "nickname": agent.nickname if agent else None,
            "task_id": str(task.id),
            "session_id": str(agent_session.id),
            "direct_connector": True,
            "success": success,
        },
    )

    task.state = TaskState.COMPLETED
    db.add(task)
    await db.flush()
    await emit_event(
        room_id=room_id,
        event_type="TASK_STATE_CHANGED",
        content=f"Task \"{task.title}\": in_progress → completed",
        db=db,
        task_id=task.id,
        metadata={
            "task_id": str(task.id),
            "task_title": task.title,
            "prev_state": TaskState.IN_PROGRESS.value,
            "task_state": TaskState.COMPLETED.value,
        },
    )
    await db.commit()
