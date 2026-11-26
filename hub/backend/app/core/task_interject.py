"""A user's message into a running task (internal design notes Stage 2).

Flow:

1. ``create`` — ``POST /rooms/{room}/tasks/{task}/interject``: the message is
   saved as a chat bubble tied to the task and queued as a
   ``TaskInterjection`` row.
2. ``deliver_pending`` — the agent's tool loop, at the top of every
   iteration (before the next model request), claims the queued rows and
   ``inject``\\ s them into the request as a marked note.  Nothing can be
   delivered sooner: while a request is in flight the agent reads nothing.
3. ``persist_in_history`` — after the turn, the proxy loop copies the note
   into the agent's history: the tool loop's working list is discarded, and
   without this the next turn would not know the user ever wrote.
4. ``record_reply`` — the ``reply_to_user`` tool answers the oldest delivered,
   unanswered message and posts the answer as the agent's chat bubble.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.task_interjection import TaskInterjection

logger = logging.getLogger(__name__)

MAX_TEXT = 4000
# Chat metadata ``kind`` values — the frontend keys the "delivered" mark on these.
KIND_MESSAGE = "task_interjection"
KIND_REPLY = "task_interjection_reply"

_NOTE_HEAD = (
    "[MESSAGE FROM THE USER DURING THE TASK — the user wrote this while you were "
    "working. It is not a new task and not a verdict. Answer it briefly with the "
    "reply_to_user tool (your turn does not end), then continue the task. If it "
    "asks for the status, say what you are doing and what is left. If it changes "
    "the scope of the task, follow it only when it is unambiguous; otherwise ask "
    "{cm} in your next message.]"
)


class InterjectRefused(Exception):
    """The task cannot take a message right now; ``code`` is for the UI."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 1. create
# ---------------------------------------------------------------------------


async def _open_agent_session(task_id: UUID, db: AsyncSession) -> Any:
    from app.models.session.cm_agent_proxy import CmAgentProxySession

    return (
        await db.execute(
            select(CmAgentProxySession)
            .where(
                CmAgentProxySession.task_id == task_id,
                CmAgentProxySession.side == "agent",
                CmAgentProxySession.closed_at.is_(None),
            )
            .order_by(CmAgentProxySession.started_at.desc())
            .limit(1)
        )
    ).scalars().first()


def check_can_interject(task: Any, agent_session: Any) -> None:
    """Raise ``InterjectRefused`` unless the task is running with an agent on it."""
    if getattr(task, "awaiting_user_kind", None):
        raise InterjectRefused(
            "awaiting_user",
            "The task is waiting for your answer — reply in the chat, the agent will get the reply immediately.",
        )
    state = getattr(task, "runtime_state", None) or "idle"
    if state == "paused":
        raise InterjectRefused("paused", "The task is paused — resume it to message the agent.")
    if state != "running":
        raise InterjectRefused("not_running", "The task is not running right now — the agent will not receive the message.")
    if agent_session is None:
        raise InterjectRefused("no_session", "The task has no open executor session.")


async def create(
    task: Any,
    *,
    user: Any,
    text: str,
    db: AsyncSession,
    existing_message_id: UUID | None = None,
) -> TaskInterjection:
    """Queue ``text`` into the running ``task``.

    ``existing_message_id`` (Stage 4, the router's ``interject_task``): the
    user's chat message is already saved and shown — it is tagged with the
    task instead of posting a second bubble.
    """
    from app.models.message import Message
    from app.websockets.manager import manager

    text = (text or "").strip()
    if not text:
        raise InterjectRefused("empty", "Empty message.")
    text = text[:MAX_TEXT]
    check_can_interject(task, await _open_agent_session(task.id, db))

    row = TaskInterjection(
        room_id=task.room_id,
        task_id=task.id,
        author_user_id=getattr(user, "id", None),
        text=text,
    )
    metadata = {"kind": KIND_MESSAGE, "interjection_id": str(row.id)}

    existing = (
        await db.get(Message, existing_message_id) if existing_message_id is not None else None
    )
    if existing is not None:
        existing.task_id = task.id
        existing.event_metadata = {**(existing.event_metadata or {}), **metadata}
        row.message_id = existing.id
        db.add(existing)
        db.add(row)
        await db.commit()
        return row

    msg = Message(
        room_id=task.room_id,
        content=text,
        sender_user_id=getattr(user, "id", None),
        task_id=task.id,
        event_metadata=metadata,
    )
    row.message_id = msg.id
    db.add(msg)
    db.add(row)
    await db.commit()

    try:
        await manager.broadcast(
            {
                "type": "message",
                "event_type": "MESSAGE",
                "id": str(msg.id),
                "room_id": str(task.room_id),
                "sender_id": str(user.id) if getattr(user, "id", None) else None,
                "sender_type": "user",
                "content": text,
                "timestamp": msg.timestamp.isoformat(),
                "status": msg.status,
                "task_id": str(task.id),
                "metadata": metadata,
            },
            task.room_id,
        )
    except Exception as exc:  # the row is queued either way
        logger.warning("task_interject: broadcast failed (task=%s): %s", task.id, exc)
    return row


# ---------------------------------------------------------------------------
# 2. deliver
# ---------------------------------------------------------------------------


def format_note(texts: list[str], cm_label: str = "ChatMaster") -> str:
    body = "\n\n".join(f"> {t}".replace("\n", "\n> ") for t in texts)
    return f"{_NOTE_HEAD.format(cm=cm_label)}\n\n{body}"


async def deliver_pending(
    task_id: UUID | None,
    session_id: UUID,
    db: AsyncSession,
) -> list[str]:
    """Claim the queued messages of ``task_id``; returns their texts.

    ``SKIP LOCKED`` so two loops of one task (a stale generation still
    finishing) never both deliver the same row.  Never raises: a queue
    failure must not break the agent's turn.
    """
    if task_id is None:
        return []
    try:
        # Plain read first: almost every turn has nothing queued, and a
        # ``FOR UPDATE`` (even on zero rows) flags the loop's transaction as
        # holding a lock, so ``release_db`` would keep its connection across
        # the model wait — with several tasks running that drained the pool
        # (internal design notes, E4).
        waiting = (
            await db.execute(
                select(TaskInterjection.id)
                .where(
                    TaskInterjection.task_id == task_id,
                    TaskInterjection.delivered_at.is_(None),
                )
                .limit(1)
            )
        ).scalars().first()
        if waiting is None:
            return []
        # A savepoint, not a rollback on failure: this is the tool loop's own
        # session, and rolling it back would drop whatever the loop has not
        # committed yet.
        async with db.begin_nested():
            rows = (
                await db.execute(
                    select(TaskInterjection)
                    .where(
                        TaskInterjection.task_id == task_id,
                        TaskInterjection.delivered_at.is_(None),
                    )
                    .order_by(TaskInterjection.created_at.asc())
                    .with_for_update(skip_locked=True)
                )
            ).scalars().all()
            now = _now()
            for row in rows:
                row.delivered_at = now
                row.delivered_session_id = session_id
                db.add(row)
            await _stamp_messages(rows, "delivered_at", now, db)
        if not rows:
            return []
        await db.commit()
    except Exception as exc:
        logger.warning("task_interject: delivery failed (task=%s): %s", task_id, exc)
        return []

    await _broadcast_state(rows, "delivered")
    return [row.text for row in rows]


async def _stamp_messages(
    rows: list[TaskInterjection], field: str, when: datetime, db: AsyncSession,
) -> None:
    """Mirror a delivery/answer time into the user's chat bubble metadata.

    The WS ``task_interjection`` frame marks the bubble live; this keeps the
    mark ("delivered to the agent 11:48") after the chat history is reloaded.
    """
    from app.models.message import Message

    ids = [row.message_id for row in rows if row.message_id is not None]
    if not ids:
        return
    msgs = (await db.execute(select(Message).where(Message.id.in_(ids)))).scalars().all()
    for msg in msgs:
        # A new dict, not an in-place change: the column is plain JSON.
        msg.event_metadata = {**(msg.event_metadata or {}), field: when.isoformat()}
        db.add(msg)


async def _broadcast_state(rows: list[TaskInterjection], state: str) -> None:
    from app.websockets.manager import manager

    for row in rows:
        try:
            await manager.broadcast(
                {
                    "type": "task_interjection",
                    "state": state,
                    "interjection_id": str(row.id),
                    "message_id": str(row.message_id) if row.message_id else None,
                    "task_id": str(row.task_id),
                    "ts": _now().isoformat(),
                },
                row.room_id,
            )
        except Exception as exc:
            logger.debug("task_interject: state broadcast failed: %s", exc)


def _with_text(content: Any, note: str) -> Any:
    if isinstance(content, list):
        return [*content, {"type": "text", "text": note}]
    if isinstance(content, str) and content:
        return [{"type": "text", "text": content}, {"type": "text", "text": note}]
    return [{"type": "text", "text": note}]


def inject_anthropic(working: list[dict[str, Any]], note: str) -> None:
    """Add ``note`` to the user message the next request ends with.

    Two ``user`` messages in a row are rejected by the Messages API, so the
    note joins the trailing user message (task text, the CM's answer or the
    tool results).  The message dict is replaced, not mutated: the first
    iteration's ``working`` shares its dicts with the caller's history.
    """
    if working and working[-1].get("role") == "user":
        last = working[-1]
        working[-1] = {**last, "content": _with_text(last.get("content"), note)}
    else:
        working.append({"role": "user", "content": note})


def inject_openai(working: list[dict[str, Any]], note: str) -> None:
    """Chat Completions takes a user message right after ``tool`` results."""
    working.append({"role": "user", "content": note})


# Once answered, a note must stop asking for an answer: left as is, it stays
# the newest user message of the turn and a weak model re-answers it and redoes
# work it already did (Loki's friends, 2026-11-24: ~25 steps, three replies).
_ANSWERED_HEAD = (
    "[MESSAGE FROM THE USER DURING THE TASK — ALREADY ANSWERED with reply_to_user. "
    "Do not answer it again. Everything you did after it is newer than it: do not "
    "redo that work. Keep any instruction it gives and continue the task.]"
)
_REPLY_OK = "Delivered to the user."


def _is_note(text: Any) -> bool:
    return isinstance(text, str) and text.startswith(_NOTE_HEAD.split(" — ", 1)[0] + " — the user")


def _answered(note: str) -> str:
    _head, sep, body = note.partition("]\n\n")
    return f"{_ANSWERED_HEAD}\n\n{body}" if sep else _ANSWERED_HEAD


def _tool_result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(b.get("text") or "") for b in content if isinstance(b, dict))
    return ""


def _replies_in(msg: dict[str, Any]) -> int:
    """Successful reply_to_user results carried by ``msg`` (either API shape)."""
    content = msg.get("content")
    if msg.get("role") == "tool":
        return int(isinstance(content, str) and _REPLY_OK in content)
    if msg.get("role") == "user" and isinstance(content, list):
        return sum(
            1
            for b in content
            if isinstance(b, dict) and b.get("type") == "tool_result" and _REPLY_OK in _tool_result_text(b)
        )
    return 0


def settle_answered(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Rewrite the notes a later reply_to_user answered; returns {old: new}.

    Mirrors ``record_reply``: each successful reply answers the oldest open
    note.  A reply in the same message as a note came before that note was
    delivered.  Message dicts are replaced, not mutated (shared with history).
    """
    pending: list[tuple[int, int | None]] = []  # (message index, block index or None)
    answered: list[tuple[int, int | None]] = []
    for idx, msg in enumerate(messages):
        for _ in range(_replies_in(msg)):
            if pending:
                answered.append(pending.pop(0))
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if _is_note(content):
            pending.append((idx, None))
        elif isinstance(content, list):
            for bidx, block in enumerate(content):
                if isinstance(block, dict) and block.get("type") == "text" and _is_note(block.get("text")):
                    pending.append((idx, bidx))
    changed: dict[str, str] = {}
    for idx, bidx in answered:
        msg = messages[idx]
        if bidx is None:
            old = msg["content"]
            changed[old] = _answered(old)
            messages[idx] = {**msg, "content": changed[old]}
        else:
            blocks = list(msg["content"])
            old = blocks[bidx]["text"]
            changed[old] = _answered(old)
            blocks[bidx] = {**blocks[bidx], "text": changed[old]}
            messages[idx] = {**msg, "content": blocks}
    return changed


async def settle_for_history(
    notes: list[str] | None, task_id: UUID | None, db: AsyncSession
) -> list[str] | None:
    """Mark this turn's notes answered when nothing delivered is still open.

    The history keeps the notes but not the turn's reply_to_user results, so
    without this a later turn would read them as still waiting for an answer.
    """
    if not notes or task_id is None:
        return notes
    try:
        open_id = (
            await db.execute(
                select(TaskInterjection.id)
                .where(
                    TaskInterjection.task_id == task_id,
                    TaskInterjection.delivered_at.is_not(None),
                    TaskInterjection.answered_at.is_(None),
                )
                .limit(1)
            )
        ).scalars().first()
    except Exception as exc:  # noqa: BLE001 — keep the notes as they are
        logger.debug("task_interject: open check failed (task=%s): %s", task_id, exc)
        return notes
    if open_id is not None:
        return notes
    return [_answered(n) if _is_note(n) else n for n in notes]


# ---------------------------------------------------------------------------
# 3. persist
# ---------------------------------------------------------------------------


def persist_in_history(history: list[dict[str, Any]], notes: list[str] | None) -> None:
    """Fold this turn's delivered notes into the user message that opened it."""
    if not notes:
        return
    note = "\n\n".join(notes)
    for idx in range(len(history) - 1, -1, -1):
        if history[idx].get("role") == "user":
            msg = history[idx]
            content = msg.get("content")
            if isinstance(content, str):
                merged: Any = f"{content}\n\n{note}" if content else note
            else:
                merged = _with_text(content, note)
            history[idx] = {**msg, "content": merged}
            return
    history.append({"role": "user", "content": note})


# ---------------------------------------------------------------------------
# 4. reply
# ---------------------------------------------------------------------------


async def record_reply(
    *,
    room_id: UUID,
    task_id: UUID | None,
    agent_id: UUID | None,
    actor_label: str | None,
    text: str,
    db: AsyncSession,
) -> TaskInterjection | None:
    """Answer the oldest delivered, unanswered message; ``None`` if there is none."""
    from app.core.events import emit_event

    if task_id is None:
        return None
    row = (
        await db.execute(
            select(TaskInterjection)
            .where(
                TaskInterjection.task_id == task_id,
                TaskInterjection.delivered_at.is_not(None),
                TaskInterjection.answered_at.is_(None),
            )
            .order_by(TaskInterjection.created_at.asc())
            .limit(1)
        )
    ).scalars().first()
    if row is None:
        return None
    row.answered_at = _now()
    row.reply_text = text[:MAX_TEXT]
    db.add(row)
    await _stamp_messages([row], "answered_at", row.answered_at, db)
    await db.commit()
    await emit_event(
        room_id=room_id,
        event_type="MESSAGE",
        content=row.reply_text,
        db=db,
        sender_agent_id=agent_id,
        task_id=task_id,
        metadata={
            "kind": KIND_REPLY,
            "interjection_id": str(row.id),
            "reply_to_message_id": str(row.message_id) if row.message_id else None,
            "nickname": actor_label,
        },
    )
    await _broadcast_state([row], "answered")
    return row
